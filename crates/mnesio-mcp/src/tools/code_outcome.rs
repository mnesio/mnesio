//! `mnesio_code_outcome` — tell mnesio whether the code it gave you worked.
//!
//! This is the call no competing code-memory tool has. graphify and
//! codebase-memory-mcp both rank code by how relevant it looks and never learn
//! whether the retrieval was any good. The signal is free — the build passes or
//! it doesn't — and today everyone throws it away.
//!
//! The tool is deliberately trivial to call, because a loop that only closes
//! when an agent remembers a hard API is a loop that never closes.

use crate::context::AppContext;
use crate::protocol::{CallToolResult, ToolDescriptor};
use mnesio_code::{CodeOutcome, EditResult, OutcomeJournal};
use serde::Deserialize;
use serde_json::{json, Value};

pub fn descriptor() -> ToolDescriptor {
    ToolDescriptor {
        name: "mnesio_code_outcome",
        description: "After you edit code using context from mnesio_code_context, report what \
             happened: did it build, did tests pass, was the change accepted. mnesio \
             uses this to learn which retrievals actually help, so later answers get \
             better. Call it even when the result was bad — a failure is the more \
             useful signal.",
        input_schema: json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The same task string you passed to mnesio_code_context."
                },
                "repo": {
                    "type": "string",
                    "description": "The same repository path."
                },
                "result": {
                    "type": "string",
                    "enum": ["passed", "tests_failed", "build_failed", "rejected", "accepted"],
                    "description": "What happened. `passed` = built and tests green. \
                                    `accepted`/`rejected` = a human's verdict, which \
                                    outweighs the build."
                }
            },
            "required": ["task", "repo", "result"]
        }),
    }
}

#[derive(Debug, Deserialize)]
struct Args {
    task: String,
    repo: String,
    result: String,
}

fn parse_result(s: &str) -> Option<EditResult> {
    match s {
        "passed" => Some(EditResult::Passed),
        "tests_failed" => Some(EditResult::TestsFailed),
        "build_failed" => Some(EditResult::BuildFailed),
        "rejected" => Some(EditResult::Rejected),
        "accepted" => Some(EditResult::Accepted),
        _ => None,
    }
}

pub async fn handle(_ctx: &AppContext, arguments: Value) -> anyhow::Result<CallToolResult> {
    let args: Args = match serde_json::from_value(arguments) {
        Ok(a) => a,
        Err(e) => {
            return Ok(CallToolResult::error_text(format!(
                "invalid arguments: {e}"
            )))
        }
    };
    let Some(result) = parse_result(&args.result) else {
        return Ok(CallToolResult::error_text(format!(
            "unknown result {:?}; expected one of: passed, tests_failed, build_failed, \
             rejected, accepted",
            args.result
        )));
    };

    // Join to what was actually served. An outcome that cannot find its
    // context is refused rather than recorded: an edit that never used
    // mnesio's context says nothing about mnesio's retrieval, but recording it
    // anyway would still move the success rate the dashboard reports — a
    // corruption that is invisible once it happens, because the number looks
    // identical either way.
    let context = match super::code_context::sessions()
        .lock()
        .await
        .lookup(&args.repo, &args.task)
    {
        Ok(c) => c,
        Err(unmatched) => {
            let hint = if unmatched.recent.is_empty() {
                "No context has been served for this repository in this session. \
                 Call mnesio_code_context first — an outcome is only evidence about \
                 retrieval that actually happened."
                    .to_string()
            } else {
                format!(
                    "No context was served for that task. Recent tasks in this \
                     repository:\n{}\n\nRe-report using one of those task strings.",
                    unmatched
                        .recent
                        .iter()
                        .map(|t| format!("  · {t}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                )
            };
            return Ok(CallToolResult::error_text(hint));
        }
    };

    let outcome = CodeOutcome::from_context(&args.task, &args.repo, &context, result);
    let journal = OutcomeJournal::for_repo(std::path::Path::new(&args.repo));
    if let Err(e) = journal.append(&outcome) {
        return Ok(CallToolResult::error_text(format!(
            "could not record the outcome: {e}"
        )));
    }

    let note = match result.is_success() {
        Some(true) => "Recorded as a success.",
        Some(false) => "Recorded as a failure — the more useful signal.",
        None => {
            "Recorded as ambiguous: a red test may predate the edit, so it is \
             not counted against the retrieval."
        }
    };
    Ok(CallToolResult::text(format!(
        "Outcome {} for {:?} in {}.\n\n\
         {} Attributed to {} symbol(s) ({} seed, {} expanded) costing ~{} tokens.\n\n\
         Batches of these compile into retrieval rules, and every rule is re-checked \
         against a canary set before it takes effect.",
        result.as_str(),
        args.task,
        args.repo,
        note,
        outcome.symbols.len(),
        outcome.seeds().count(),
        outcome.expansions().count(),
        outcome.tokens_used,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn ctx() -> (TempDir, AppContext) {
        let d = TempDir::new().unwrap();
        let c = AppContext::open(d.path(), "mock").await.unwrap();
        (d, c)
    }

    #[test]
    fn every_schema_variant_parses() {
        // A schema that advertises a value the handler rejects is a broken
        // contract an agent discovers only at runtime.
        let d = descriptor();
        for v in d.input_schema["properties"]["result"]["enum"]
            .as_array()
            .unwrap()
        {
            let s = v.as_str().unwrap();
            assert!(
                parse_result(s).is_some(),
                "schema offers {s:?}, handler rejects it"
            );
        }
    }

    #[tokio::test]
    async fn an_unknown_result_lists_the_valid_ones() {
        let (_d, c) = ctx().await;
        let r = handle(
            &c,
            json!({"task": "t", "repo": "r", "result": "sort-of-worked"}),
        )
        .await
        .unwrap();
        assert!(r.is_error);
        assert!(format!("{:?}", r.content).contains("build_failed"));
    }

    /// Removes the journal this test wrote into the real cache directory.
    ///
    /// The handler must use the production cache path — that is the thing
    /// under test — and pointing it elsewhere would mean overwriting a
    /// process-global environment variable, which races the moment two tests
    /// run in parallel. Each test's repo is a fresh `TempDir`, so its journal
    /// path is already unique; this just avoids leaving the files behind.
    struct JournalGuard(std::path::PathBuf);
    impl JournalGuard {
        fn new(repo: &std::path::Path) -> Self {
            Self(OutcomeJournal::for_repo(repo).path().to_path_buf())
        }
    }
    impl Drop for JournalGuard {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).ok();
        }
    }

    /// Pretend `mnesio_code_context` served `task` for `repo`, so an outcome
    /// has something to attribute to.
    async fn serve(repo: &str, task: &str) {
        use mnesio_code::pack::{Form, PackedContext, PackedSymbol, Reason};
        use mnesio_core::types::{new_id, MemoryRef};
        super::super::code_context::sessions()
            .lock()
            .await
            .remember(
                repo,
                task,
                PackedContext {
                    symbols: vec![PackedSymbol {
                        memory: MemoryRef(new_id()),
                        form: Form::Full,
                        tokens: 42,
                        reason: Reason::Seed(0),
                    }],
                    tokens_used: 42,
                    ..Default::default()
                },
            );
    }

    #[tokio::test]
    async fn a_failure_is_accepted_not_discouraged() {
        // The negative signal is the more useful one; a tool that made
        // reporting it feel like an error would collect only good news.
        let (d, c) = ctx().await;
        let _g = JournalGuard::new(d.path());
        let repo = d.path().to_str().unwrap();
        serve(repo, "t").await;
        let r = handle(
            &c,
            json!({"task": "t", "repo": repo, "result": "build_failed"}),
        )
        .await
        .unwrap();
        assert!(!r.is_error, "{:?}", r.content);
    }

    #[tokio::test]
    async fn an_outcome_with_no_served_context_is_refused() {
        // The corruption this prevents: an edit that never used mnesio's
        // context is not evidence about mnesio's retrieval, but recording it
        // would still move the success rate the dashboard reports — and the
        // number looks identical either way, so nothing downstream could
        // detect it.
        let (d, c) = ctx().await;
        let r = handle(
            &c,
            json!({"task": "never retrieved", "repo": d.path().to_str().unwrap(), "result": "passed"}),
        )
        .await
        .unwrap();
        assert!(r.is_error);
        assert!(
            format!("{:?}", r.content).contains("mnesio_code_context"),
            "the refusal must say how to fix it"
        );
    }

    #[tokio::test]
    async fn a_recorded_outcome_reaches_the_journal_with_its_attribution() {
        // The end-to-end claim of the whole phase: an editor reports an
        // outcome, and it lands somewhere the dashboard can read it, still
        // carrying which symbols it is about.
        let (d, c) = ctx().await;
        let _g = JournalGuard::new(d.path());
        let repo = d.path().to_str().unwrap();
        serve(repo, "fix the retry backoff").await;
        let r = handle(
            &c,
            json!({"task": "fix the retry backoff", "repo": repo, "result": "passed"}),
        )
        .await
        .unwrap();
        assert!(!r.is_error, "{:?}", r.content);

        let read = OutcomeJournal::for_repo(d.path()).read();
        assert_eq!(read.entries.len(), 1);
        let o = &read.entries[0].outcome;
        assert_eq!(o.result, EditResult::Passed);
        assert_eq!(
            o.seeds().count(),
            1,
            "attribution must survive the round trip"
        );
        assert_eq!(o.tokens_used, 42);
    }

    #[tokio::test]
    async fn a_mismatched_task_is_told_what_would_match() {
        // A bare "no" leaves the agent with no way forward, so it stops
        // reporting outcomes at all and the loop quietly dies.
        let (d, c) = ctx().await;
        let repo = d.path().to_str().unwrap();
        serve(repo, "add retry to the client").await;
        let r = handle(
            &c,
            json!({"task": "something else", "repo": repo, "result": "passed"}),
        )
        .await
        .unwrap();
        assert!(r.is_error);
        assert!(format!("{:?}", r.content).contains("add retry to the client"));
    }
}

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
use mnesio_code::EditResult;
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

    // TODO(phase-18d): join to the packed context recorded for this
    // (task, repo) and append a `CodeOutcome` to the log, then let the
    // procedural compiler turn batches of them into gated retrieval rules.
    // Recording the verdict is deliberately shipped before the compiler that
    // consumes it: the loop cannot be evaluated until outcomes exist, and a
    // compiler with no data to learn from cannot be tested honestly.
    let note = match result.is_success() {
        Some(true) => "recorded as a success",
        Some(false) => "recorded as a failure — the more useful signal",
        None => {
            "recorded as ambiguous: a red test may predate the edit, so it \
                 is not counted against the retrieval"
        }
    };
    Ok(CallToolResult::text(format!(
        "Outcome {} for {:?} in {}.\n\n{}",
        result.as_str(),
        args.task,
        args.repo,
        note
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

    #[tokio::test]
    async fn a_failure_is_accepted_not_discouraged() {
        let (_d, c) = ctx().await;
        let r = handle(
            &c,
            json!({"task": "t", "repo": "r", "result": "build_failed"}),
        )
        .await
        .unwrap();
        assert!(!r.is_error);
    }
}

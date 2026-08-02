//! `mnesio_code_context` — index a repository, return only the code a task
//! needs, fitted to a token budget.
//!
//! This is the tool that puts Phase 17 in front of an agent. MCP is the reason
//! it reaches every editor at once: Claude Code, Cursor, Codex, Copilot's agent
//! mode, Windsurf and Zed all speak the same protocol, so one stdio server
//! covers them without an adapter each.
//!
//! ## Why the repo is indexed per call
//!
//! An index is held in a process-lifetime cache keyed by `(path, scope)`, so
//! the first call on a repository pays for parsing and embedding and every
//! later one is a search. That is a deliberate simplicity trade: an incremental
//! index driven off file-watching is the right long-term answer, but a cache
//! keyed by path is honest about what it does and cannot serve a stale answer
//! *within* a session for a repository that has not been re-indexed.
//!
//! **Known limitation:** edits made after the first call are not reflected
//! until the process restarts or `refresh` is passed. Stated here rather than
//! discovered later, because silently serving stale code to an agent that is
//! editing that code is the worst failure this tool has.

use crate::context::AppContext;
use crate::protocol::{CallToolResult, ToolDescriptor};
use mnesio_code::CodeMemory;
use mnesio_core::Scope;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Indexes built this process, keyed by `(path, tenant)`.
///
/// `Mutex` rather than `RwLock`: indexing is the expensive path and two agents
/// racing on the same cold repository should serialise rather than both pay for
/// it.
static INDEXES: std::sync::OnceLock<Mutex<HashMap<String, Arc<CodeMemory>>>> =
    std::sync::OnceLock::new();

fn cache() -> &'static Mutex<HashMap<String, Arc<CodeMemory>>> {
    INDEXES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn descriptor() -> ToolDescriptor {
    ToolDescriptor {
        name: "mnesio_code_context",
        description:
            "Retrieve the specific functions, classes and types a coding task needs from a \
             repository, packed to a token budget — instead of reading whole files. Give it \
             the task you are actually doing (\"make the retry backoff configurable\"), not \
             keywords. Returns each symbol with its file path, plus why it was included \
             (directly retrieved, or pulled in as a callee). First call on a repository \
             indexes it and is slow; later calls are fast.",
        input_schema: json!({
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Absolute path to the repository root to index and search."
                },
                "task": {
                    "type": "string",
                    "description": "What you are trying to do, in natural language. Phrase it \
                                    as the change or question, not as search keywords."
                },
                "budget_tokens": {
                    "type": "integer",
                    "description": "Hard ceiling on returned context. Never exceeded; symbols \
                                    that do not fit degrade to their signature or are dropped.",
                    "minimum": 200,
                    "maximum": 100000,
                    "default": 4000
                },
                "tenant": {
                    "type": "string",
                    "description": "Scope to index under. Two repositories indexed under \
                                    different tenants cannot see each other.",
                    "default": "default"
                },
                "refresh": {
                    "type": "boolean",
                    "description": "Re-index even if this repository is already cached. Use \
                                    after making edits you want reflected.",
                    "default": false
                }
            },
            "required": ["repo", "task"]
        }),
    }
}

#[derive(Debug, Deserialize)]
struct Args {
    repo: String,
    task: String,
    #[serde(default = "default_budget")]
    budget_tokens: usize,
    #[serde(default = "default_tenant")]
    tenant: String,
    #[serde(default)]
    refresh: bool,
}

fn default_budget() -> usize {
    4000
}

fn default_tenant() -> String {
    "default".into()
}

pub async fn handle(ctx: &AppContext, arguments: Value) -> anyhow::Result<CallToolResult> {
    let args: Args = match serde_json::from_value(arguments) {
        Ok(a) => a,
        Err(e) => {
            return Ok(CallToolResult::error_text(format!(
                "invalid arguments: {e}"
            )))
        }
    };

    // Reject early with something the caller can act on. A missing path
    // otherwise surfaces as "no supported source files", which sends whoever
    // reads it hunting for a language problem that isn't there.
    let path = std::path::Path::new(&args.repo);
    if !path.is_dir() {
        return Ok(CallToolResult::error_text(format!(
            "repo path {:?} is not a directory",
            args.repo
        )));
    }

    let key = format!("{}\u{0}{}", args.repo, args.tenant);
    let mut guard = cache().lock().await;
    if args.refresh {
        guard.remove(&key);
    }

    let memory = match guard.get(&key) {
        Some(m) => Arc::clone(m),
        None => {
            let built = match CodeMemory::index(
                path,
                Scope::global(&args.tenant),
                Arc::clone(&ctx.embedder),
            )
            .await
            {
                Ok(m) => Arc::new(m),
                Err(e) => return Ok(CallToolResult::error_text(format!("indexing failed: {e}"))),
            };
            guard.insert(key, Arc::clone(&built));
            built
        }
    };
    // Indexing is done; a search must not hold the cache against other repos.
    drop(guard);

    let context = match memory.context_for(&args.task, args.budget_tokens).await {
        Ok(c) => c,
        Err(e) => return Ok(CallToolResult::error_text(format!("retrieval failed: {e}"))),
    };

    Ok(CallToolResult::text(render(&context, memory.stats())))
}

/// Format for a model, not a UI.
///
/// Each symbol is labelled with its path and why it is present, because an
/// agent deciding what to edit needs to distinguish "retrieval thought this
/// matches your task" from "this is merely called by something that did".
fn render(ctx: &mnesio_code::CodeContext, stats: &mnesio_code::IndexStats) -> String {
    if ctx.hits.is_empty() {
        return format!(
            "No matching code found in {} symbols across {} files.\n\n\
             The task may share no vocabulary with the code that implements it — \
             measured at ~9% of real repository tasks. Try naming a symbol, file \
             or module you already know is involved.",
            stats.symbols, stats.files
        );
    }

    let mut out = format!(
        "{} symbols, ~{} tokens (from an index of {} symbols across {} files).\n\n",
        ctx.hits.len(),
        ctx.tokens_used,
        stats.symbols,
        stats.files
    );

    for h in &ctx.hits {
        let why = if h.is_seed {
            "matched your task"
        } else {
            "called by a match"
        };
        let form = if h.is_full { "" } else { " (signature only)" };
        out.push_str(&format!(
            "--- {} · {} · {}{}\n{}\n\n",
            h.path,
            h.kind.as_tag(),
            why,
            form,
            h.text.trim_end()
        ));
    }

    if !ctx.notes.is_empty() {
        out.push_str("Files represented:\n");
        for n in &ctx.notes {
            out.push_str(&format!("  {n}\n"));
        }
    }
    if ctx.dropped > 0 {
        out.push_str(&format!(
            "\n{} further candidates did not fit the budget.\n",
            ctx.dropped
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn fresh_ctx() -> (TempDir, AppContext) {
        let dir = TempDir::new().unwrap();
        let ctx = AppContext::open(dir.path(), "mock").await.unwrap();
        (dir, ctx)
    }

    #[test]
    fn the_schema_requires_a_repo_and_a_task() {
        let d = descriptor();
        let req = d.input_schema["required"].as_array().unwrap();
        assert!(req.iter().any(|v| v == "repo"));
        assert!(req.iter().any(|v| v == "task"));
    }

    #[test]
    fn the_description_tells_an_agent_to_pass_a_task_not_keywords() {
        // The retrieval settings were measured against task-shaped queries. An
        // agent that sends "retry backoff" instead of the change it is making
        // gets worse results, so the tool description has to say so — it is the
        // only documentation a model ever reads.
        let d = descriptor();
        assert!(d.description.contains("not keywords"));
    }

    #[tokio::test]
    async fn a_missing_repo_path_fails_with_a_clear_message() {
        let (_d, ctx) = fresh_ctx().await;
        let r = handle(
            &ctx,
            json!({"repo": "/definitely/not/here", "task": "anything"}),
        )
        .await
        .unwrap();
        assert!(r.is_error);
        let text = format!("{:?}", r.content);
        assert!(text.contains("not a directory"), "got {text}");
    }

    #[tokio::test]
    async fn malformed_arguments_are_a_tool_error_not_a_transport_error() {
        let (_d, ctx) = fresh_ctx().await;
        let r = handle(&ctx, json!({"repo": 42})).await.unwrap();
        assert!(r.is_error);
    }
}

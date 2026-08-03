//! `mnesio_code_context` — index a repository, return only the code a task
//! needs, fitted to a token budget.
//!
//! This is the tool that puts Phase 17 in front of an agent. MCP is the reason
//! it reaches every editor at once: Claude Code, Cursor, Codex, Copilot's agent
//! mode, Windsurf and Zed all speak the same protocol, so one stdio server
//! covers them without an adapter each.
//!
//! ## Freshness is automatic
//!
//! An index is cached per `(path, scope)` for the process lifetime, and
//! **every call checks the tree for changes before answering**. If anything
//! was added, edited or deleted, the index rebuilds first.
//!
//! That check is not optional, because the alternative is the worst failure
//! this tool can have: an agent editing a file, asking about it, and being
//! handed the version from before its own edit. A `refresh` flag would leave
//! that correctness to whoever remembered to pass it, which in practice is
//! nobody.
//!
//! It is affordable because the two costs are separated. Detecting change is a
//! metadata walk — no file reads, no model calls. Rebuilding re-parses, but
//! re-embeds only symbols whose *text* actually changed, keyed by content
//! hash, so a one-function edit in a 5,000-symbol repository costs one
//! embedding rather than five thousand.

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
/// Two levels of lock, deliberately. The outer one guards the map and is held
/// only long enough to look an entry up, so a slow repository cannot block a
/// query against a different one. The inner one guards a single index across
/// *both* its freshness check and its search.
///
/// The inner mutex is what makes staleness structurally impossible rather than
/// merely intended: refreshing through an `Arc` would need `Arc::get_mut`,
/// which silently returns `None` while another call holds a clone — and a
/// skipped refresh is precisely the bug this design exists to remove. Owning
/// the lock means the check cannot be skipped, only waited for.
type Cached = Arc<Mutex<CodeMemory>>;

static INDEXES: std::sync::OnceLock<Mutex<HashMap<String, Cached>>> = std::sync::OnceLock::new();

fn cache() -> &'static Mutex<HashMap<String, Cached>> {
    INDEXES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// What has been served but not yet answered for.
///
/// Process-wide, because the outcome arrives on a later, independent tool call
/// and MCP carries no state between them. See [`super::code_session`] for why
/// an outcome that cannot find its context is refused rather than recorded.
static SESSIONS: std::sync::OnceLock<Mutex<super::code_session::SessionStore>> =
    std::sync::OnceLock::new();

pub(crate) fn sessions() -> &'static Mutex<super::code_session::SessionStore> {
    SESSIONS.get_or_init(|| Mutex::new(super::code_session::SessionStore::default()))
}

pub fn descriptor() -> ToolDescriptor {
    ToolDescriptor {
        name: "mnesio_code_context",
        description:
            "Retrieve the specific functions, classes and types a coding task needs from a \
             repository, packed to a token budget — instead of reading whole files. Give it \
             the task you are actually doing (\"make the retry backoff configurable\"), not \
             keywords. Returns each symbol with its file path, plus why it was included \
             (directly retrieved, or pulled in as a callee). The first call on a \
             repository indexes it and is slow; later calls are fast, and always \
             reflect the current state of the files — edits are picked up \
             automatically.",
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
                    "description": "Force a full rebuild, discarding cached embeddings. Edits \
                                    are detected automatically, so this is only for recovering \
                                    from a corrupted index — not for picking up your changes.",
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
    let mut map = cache().lock().await;
    if args.refresh {
        map.remove(&key);
    }

    let entry = match map.get(&key) {
        Some(m) => Arc::clone(m),
        None => {
            let built = match CodeMemory::index(
                path,
                Scope::global(&args.tenant),
                Arc::clone(&ctx.embedder),
            )
            .await
            {
                Ok(m) => Arc::new(Mutex::new(m)),
                Err(e) => return Ok(CallToolResult::error_text(format!("indexing failed: {e}"))),
            };
            map.insert(key, Arc::clone(&built));
            built
        }
    };
    // Hold the map no longer than the lookup: a slow rebuild of one repository
    // must not stall queries against another.
    drop(map);

    let memory = entry.lock().await;
    // Freshness before answering, every time. Cheap when the tree has not
    // moved; the difference between a correct answer and one describing code
    // the agent has already replaced when it has.
    let mut memory = memory;
    if let Err(e) = memory.refresh_if_stale().await {
        return Ok(CallToolResult::error_text(format!(
            "index refresh failed: {e}"
        )));
    }

    let context = match memory.context_for(&args.task, args.budget_tokens).await {
        Ok(c) => c,
        Err(e) => return Ok(CallToolResult::error_text(format!("retrieval failed: {e}"))),
    };

    // Remember what was served so a later `mnesio_code_outcome` can say which
    // symbols the verdict is about. Without this the outcome is a bare
    // success/failure flag, which is not something a compiler can learn from.
    sessions()
        .lock()
        .await
        .remember(&args.repo, &args.task, context.packed.clone());

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

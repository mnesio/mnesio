//! Tool handlers exposed via MCP. Each submodule implements one
//! tool: argument struct (with `Deserialize`), JSON schema literal,
//! and an async handler that takes an `AppContext` + parsed args
//! and returns a `CallToolResult`.
//!
//! Adding a new tool: write a new submodule, add its `descriptor()`
//! to [`all_tools`], add a match arm in [`dispatch`].

use crate::context::AppContext;
use crate::protocol::{CallToolResult, ToolDescriptor};
use serde_json::Value;

pub mod code_context;
pub mod record_outcome;
pub mod search;
pub mod write_memory;

/// All tools the server exposes. Returned verbatim from
/// `tools/list`. Order is stable so tool-picker UIs render
/// consistently across reloads.
pub fn all_tools() -> Vec<ToolDescriptor> {
    vec![
        write_memory::descriptor(),
        code_context::descriptor(),
        search::descriptor(),
        record_outcome::descriptor(),
    ]
}

/// Route a `tools/call` to the right handler. Returns
/// `Ok(CallToolResult)` for both success and tool-level failures (the
/// `is_error` flag inside discriminates). Returns `Err` only for
/// transport-level problems — unknown tool name, malformed
/// arguments — which the dispatcher maps to JSON-RPC error responses.
pub async fn dispatch(
    ctx: &AppContext,
    tool_name: &str,
    arguments: Value,
) -> anyhow::Result<CallToolResult> {
    match tool_name {
        "mnesio_write_memory" => write_memory::handle(ctx, arguments).await,
        "mnesio_search" => search::handle(ctx, arguments).await,
        "mnesio_record_outcome" => record_outcome::handle(ctx, arguments).await,
        "mnesio_code_context" => code_context::handle(ctx, arguments).await,
        other => anyhow::bail!("unknown tool {other:?}"),
    }
}

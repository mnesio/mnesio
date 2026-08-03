//! JSON-RPC method dispatcher.
//!
//! Three methods are handled explicitly:
//! - `initialize` — handshake; returns server info + capabilities.
//! - `tools/list` — return every tool descriptor.
//! - `tools/call` — route to [`crate::tools::dispatch`].
//!
//! `notifications/*` (no `id`) are silently accepted — we just don't
//! respond. Anything else returns method-not-found.

use crate::context::AppContext;
use crate::protocol::{
    error_codes, CallToolParams, InitializeResult, ListToolsResult, Request, Response,
    ResponseError, ServerCapabilities, ServerInfo, PROTOCOL_VERSION,
};
use crate::tools;
use serde_json::{json, Value};

/// Process one raw newline-delimited input line end to end: parse JSON, shape
/// it into a [`Request`], and dispatch. Returns `None` for blank lines and
/// notifications (nothing is written); `Some(Response)` otherwise.
///
/// Error classification follows JSON-RPC 2.0, so a hostile or buggy client
/// gets the *correct* error instead of a crash:
/// - unparseable JSON → [`error_codes::PARSE_ERROR`] (-32700) with a null id
///   (the id is unknowable when the bytes don't parse)
/// - valid JSON that isn't a well-formed request (missing `method`, a batch
///   array, wrong field types) → [`error_codes::INVALID_REQUEST`] (-32600),
///   echoing the `id` if one was recoverable from the raw value
///
/// This is the single line-processing seam `main` drives, extracted here so
/// the adversarial-input paths are unit-testable without a subprocess.
pub async fn process_line(ctx: &AppContext, line: &str) -> Option<Response> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Stage 1 — is it even JSON? Truly malformed bytes are a parse error.
    let value: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(e) => {
            return Some(Response::failure(
                Value::Null,
                ResponseError::new(error_codes::PARSE_ERROR, format!("invalid JSON: {e}")),
            ));
        }
    };
    // Stage 2 — valid JSON, but is it a well-formed Request? If not, it's an
    // invalid *request*, not a parse error — and we recover the id if we can
    // so the client can correlate the error with what it sent.
    let req: Request = match serde_json::from_value(value.clone()) {
        Ok(r) => r,
        Err(e) => {
            let id = value.get("id").cloned().unwrap_or(Value::Null);
            return Some(Response::failure(
                id,
                ResponseError::new(
                    error_codes::INVALID_REQUEST,
                    format!("invalid request: {e}"),
                ),
            ));
        }
    };
    handle_request(ctx, req).await
}

/// Handle a single decoded JSON-RPC request. Returns `Some(Response)`
/// for requests that need a reply; `None` for notifications (no id).
pub async fn handle_request(ctx: &AppContext, req: Request) -> Option<Response> {
    if req.is_notification() {
        tracing::debug!(method = %req.method, "received notification — no response sent");
        return None;
    }
    // Safe: is_notification() is the negation of `id.is_some()`.
    let id = req.id.clone().expect("notification handled above");

    let resp = match req.method.as_str() {
        "initialize" => Response::success(
            id,
            serde_json::to_value(InitializeResult {
                protocol_version: PROTOCOL_VERSION,
                capabilities: ServerCapabilities::default(),
                server_info: ServerInfo {
                    name: "mnesio-mcp",
                    version: env!("CARGO_PKG_VERSION"),
                },
            })
            .expect("InitializeResult always serializes"),
        ),
        "tools/list" => Response::success(
            id,
            serde_json::to_value(ListToolsResult {
                tools: tools::all_tools(),
            })
            .expect("ListToolsResult always serializes"),
        ),
        "tools/call" => {
            let params: CallToolParams =
                match req.params.clone().map(serde_json::from_value).transpose() {
                    Ok(Some(p)) => p,
                    Ok(None) => {
                        return Some(Response::failure(
                            id,
                            ResponseError::new(error_codes::INVALID_PARAMS, "missing params"),
                        ));
                    }
                    Err(e) => {
                        return Some(Response::failure(
                            id,
                            ResponseError::new(
                                error_codes::INVALID_PARAMS,
                                format!("invalid tools/call params: {e}"),
                            ),
                        ));
                    }
                };
            match tools::dispatch(ctx, &params.name, params.arguments).await {
                Ok(result) => Response::success(
                    id,
                    serde_json::to_value(result).expect("CallToolResult always serializes"),
                ),
                Err(e) => Response::failure(
                    id,
                    ResponseError::new(
                        error_codes::INVALID_PARAMS,
                        format!("tool dispatch failed: {e}"),
                    ),
                ),
            }
        }
        // No-op handshake completion the client sometimes sends as a
        // notification AND sometimes as a request with id. If it has
        // an id we acknowledge.
        "initialized" | "notifications/initialized" => Response::success(id, json!({})),
        other => Response::failure(
            id,
            ResponseError::new(
                error_codes::METHOD_NOT_FOUND,
                format!("unknown method {other:?}"),
            ),
        ),
    };
    Some(resp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::AppContext;
    use serde_json::json;
    use tempfile::TempDir;

    async fn fresh_ctx() -> (TempDir, AppContext) {
        let dir = TempDir::new().unwrap();
        let ctx = AppContext::open(dir.path(), "mock").await.unwrap();
        (dir, ctx)
    }

    fn req(id: i64, method: &str, params: serde_json::Value) -> Request {
        serde_json::from_value(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn initialize_returns_server_info_and_protocol_version() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle_request(&ctx, req(1, "initialize", json!({})))
            .await
            .unwrap();
        let result = r.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "mnesio-mcp");
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
    }

    #[tokio::test]
    async fn tools_list_advertises_every_tool() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle_request(&ctx, req(2, "tools/list", json!({})))
            .await
            .unwrap();
        let tools = r.result.unwrap();
        let names: Vec<_> = tools["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        // Names, not a count: a count fails every time a tool is added, which
        // trains whoever sees it to bump the number rather than check that the
        // new tool is actually advertised.
        for want in [
            "mnesio_write_memory",
            "mnesio_search",
            "mnesio_record_outcome",
            "mnesio_code_context",
            "mnesio_code_outcome",
        ] {
            assert!(
                names.contains(&want.to_string()),
                "{want} missing: {names:?}"
            );
        }
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate tool name: {names:?}");
    }

    #[tokio::test]
    async fn tools_call_routes_to_the_named_handler() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle_request(
            &ctx,
            req(
                3,
                "tools/call",
                json!({
                    "name": "mnesio_write_memory",
                    "arguments": {"content": "hello", "tenant": "t"}
                }),
            ),
        )
        .await
        .unwrap();
        let result = r.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("wrote memory"));
    }

    #[tokio::test]
    async fn unknown_method_returns_method_not_found() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle_request(&ctx, req(4, "no/such/method", json!({})))
            .await
            .unwrap();
        let err = r.error.unwrap();
        assert_eq!(err.code, error_codes::METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn notification_returns_none() {
        let (_dir, ctx) = fresh_ctx().await;
        let mut notif = req(99, "initialize", json!({}));
        notif.id = None; // strip the id → notification
        let r = handle_request(&ctx, notif).await;
        assert!(r.is_none(), "notifications must produce no response");
    }

    #[tokio::test]
    async fn malformed_tools_call_params_returns_invalid_params() {
        let (_dir, ctx) = fresh_ctx().await;
        // `name` field missing — params don't deserialize to CallToolParams.
        let mut bad = req(5, "tools/call", json!({"args": "wrong-shape"}));
        bad.params = Some(json!({"args": "wrong-shape"}));
        let r = handle_request(&ctx, bad).await.unwrap();
        let err = r.error.unwrap();
        assert_eq!(err.code, error_codes::INVALID_PARAMS);
    }

    // ---------------- adversarial process_line input ----------------
    // The line layer is untrusted: a hostile or buggy client can send
    // anything. Every case must yield a correct JSON-RPC error, never a panic.

    #[tokio::test]
    async fn unparseable_json_is_parse_error_with_null_id() {
        let (_dir, ctx) = fresh_ctx().await;
        for bad in [
            "{not json",
            "{\"jsonrpc\": ",
            "}{",
            "\"unterminated",
            "\u{0}\u{1}\u{2}",
        ] {
            let r = process_line(&ctx, bad).await.unwrap();
            let err = r.error.unwrap();
            assert_eq!(err.code, error_codes::PARSE_ERROR, "input {bad:?}");
            assert_eq!(r.id, serde_json::Value::Null);
        }
    }

    #[tokio::test]
    async fn blank_and_whitespace_lines_produce_no_response() {
        let (_dir, ctx) = fresh_ctx().await;
        assert!(process_line(&ctx, "").await.is_none());
        assert!(process_line(&ctx, "   \t  ").await.is_none());
    }

    #[tokio::test]
    async fn valid_json_wrong_shape_is_invalid_request_and_recovers_id() {
        let (_dir, ctx) = fresh_ctx().await;
        // Valid JSON, but no `method` field → not a well-formed request.
        let r = process_line(&ctx, r#"{"jsonrpc":"2.0","id":7}"#)
            .await
            .unwrap();
        let err = r.error.as_ref().unwrap();
        assert_eq!(err.code, error_codes::INVALID_REQUEST);
        assert_eq!(r.id, json!(7), "id must be echoed back when recoverable");

        // A batch array is valid JSON but unsupported → invalid request,
        // not a panic.
        let r = process_line(&ctx, r#"[{"jsonrpc":"2.0","id":1,"method":"x"}]"#)
            .await
            .unwrap();
        assert_eq!(r.error.unwrap().code, error_codes::INVALID_REQUEST);

        // A bare JSON scalar is valid JSON, invalid request.
        let r = process_line(&ctx, "42").await.unwrap();
        assert_eq!(r.error.unwrap().code, error_codes::INVALID_REQUEST);
    }

    #[tokio::test]
    async fn string_and_null_ids_round_trip() {
        let (_dir, ctx) = fresh_ctx().await;
        // String id (valid per JSON-RPC) must be echoed verbatim.
        let r = process_line(
            &ctx,
            r#"{"jsonrpc":"2.0","id":"abc-123","method":"tools/list"}"#,
        )
        .await
        .unwrap();
        assert_eq!(r.id, json!("abc-123"));
        assert!(r.result.is_some());

        // No id → notification → no response.
        assert!(
            process_line(&ctx, r#"{"jsonrpc":"2.0","method":"tools/list"}"#)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn oversized_and_deeply_nested_params_do_not_panic() {
        let (_dir, ctx) = fresh_ctx().await;

        // ~1 MB of content through the real write path — must succeed, not
        // panic or error at the transport layer.
        let big = "x".repeat(1_000_000);
        let line = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "mnesio_write_memory", "arguments": {"content": big}}
        }))
        .unwrap();
        let r = process_line(&ctx, &line).await.unwrap();
        assert!(r.result.is_some(), "1MB content should write, not crash");

        // Deeply nested JSON: serde_json caps recursion and returns an error
        // rather than overflowing the stack. We just require no panic.
        let deep = format!("{}{}{}", "[".repeat(20_000), "1", "]".repeat(20_000));
        let r = process_line(&ctx, &deep).await;
        assert!(r.is_some(), "deep nesting handled without panic");
    }

    #[tokio::test]
    async fn wrong_typed_tool_arguments_fail_gracefully() {
        let (_dir, ctx) = fresh_ctx().await;
        // `content` should be a string; pass a number. The tool returns a
        // tool-level error result (is_error), not a transport crash.
        let r = handle_request(
            &ctx,
            req(
                1,
                "tools/call",
                json!({"name": "mnesio_write_memory", "arguments": {"content": 12345}}),
            ),
        )
        .await
        .unwrap();
        // Either an INVALID_PARAMS transport error or a tool-level error
        // result is acceptable — the invariant is "no panic, structured error".
        let surfaced_error = r.error.is_some()
            || r.result
                .as_ref()
                .and_then(|v| v.get("isError"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
        assert!(
            surfaced_error,
            "wrong-typed arg must surface a structured error"
        );
    }

    #[tokio::test]
    async fn unknown_tool_name_is_invalid_params_not_panic() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle_request(
            &ctx,
            req(
                1,
                "tools/call",
                json!({"name": "mnesio_drop_table", "arguments": {}}),
            ),
        )
        .await
        .unwrap();
        assert_eq!(r.error.unwrap().code, error_codes::INVALID_PARAMS);
    }
}

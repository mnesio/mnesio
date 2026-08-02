# Integrating mnesio with agents (OpenClaw, Hermes, Claude Desktop, …)

mnesio is **agent-framework agnostic**. It does not embed in any one agent; it
exposes a memory layer over stable seams that agents already speak:

| Surface | Crate | Transport | Use it from |
|---|---|---|---|
| **MCP server** | `mnesio-mcp` | stdio JSON-RPC | any MCP client (OpenClaw, Hermes, Claude Desktop, Cursor, …) |
| HTTP API + dashboard | `mnesio-server` | HTTP/REST | any language; live metrics at `/dashboard` |
| Python | `mnesio-py` (pyo3) | in-process | LangChain / custom Python agents |

For **OpenClaw** and **Hermes** the path is **MCP** — both are MCP clients:

- **OpenClaw** — its skill system is largely MCP-server wrappers; you add mnesio
  as an MCP server (or a thin ClawHub-style skill that points at it).
- **Hermes** (Nous Research) — has a native MCP client (stdio + HTTP transports,
  selective tool loading). Register mnesio under its `mcp_servers` config.

> The interesting bit with Hermes: it has its *own* "create a skill after a
> complex task" loop. mnesio's procedural compiler does the same thing **but
> behind a non-bypassable safety gate** (canaries + safety probe + non-negative
> objective delta). So mnesio becomes Hermes' *verifiable, versioned, erasable*
> memory + skill store — the guarantees its built-in skills don't have.

---

## 1. The three tools

`mnesio-mcp` exposes exactly three tools (names are stable):

| Tool | Required args | Optional args | What it does |
|---|---|---|---|
| `mnesio_write_memory` | `content` (string) | `tenant` (string) | append a memory to the log (async embed/evolve off the write path) |
| `mnesio_search` | `query` (string) | `tenant`, `k` (int) | hybrid retrieval (vector + BM25 + RRF); returns memories + citations |
| `mnesio_code_context` | `repo` (path), `task` (string) | `budget_tokens`, `tenant`, `refresh` | index a repository and return only the symbols a task needs, packed to a hard token ceiling |
| `mnesio_record_outcome` | `artifacts_used` (string[]), `success` (bool) | `episode`, `scores` (obj), `error` | feed an agent task outcome to the **gated** procedural compiler for credit assignment |

`tenant` is mnesio's scope boundary (Hard Rule #3) — give each user/agent its own
tenant for isolation.

---

## 2. Build the server

```bash
# stdio MCP server binary → target/release/mnesio-mcp
cargo build -p mnesio-mcp --release
```

It reads `MNESIO_DATA` for the event-log directory (defaults to a temp dir).
Point all clients at the same `MNESIO_DATA` to share memory.

Verify it with a raw stdio session (exactly what an agent does):

```bash
MNESIO_DATA=/tmp/mnesio printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"mnesio_write_memory","arguments":{"content":"The capital of France is Paris","tenant":"demo"}}}' \
  '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"mnesio_search","arguments":{"query":"capital of France","tenant":"demo","k":3}}}' \
  | ./target/release/mnesio-mcp
```

Expected (verified): `initialize` → `serverInfo.name = "mnesio-mcp"`,
`protocolVersion = 2024-11-05`; `tools/list` → the three tools; the search
returns the memory you just wrote, with a citation. The same flow is asserted in
CI by the `agent_session_over_stdio` integration test (no real agent needed).

---

## 3. Hermes

Hermes loads MCP servers from its `mcp_servers` config and auto-reloads on
change. Register mnesio as a **stdio** server:

```json
{
  "mcp_servers": {
    "mnesio": {
      "command": "/abs/path/to/target/release/mnesio-mcp",
      "args": [],
      "env": { "MNESIO_DATA": "/var/lib/mnesio" }
    }
  }
}
```

A ready-to-edit copy lives at [`examples/integrations/hermes.mcp.json`](examples/integrations/hermes.mcp.json).

Then in a Hermes session the agent will see `mnesio_write_memory`,
`mnesio_search`, `mnesio_record_outcome` and can use them like any other tool. Use
Hermes' selective tool-loading to expose only what a given agent needs.

> **HTTP transport:** Hermes also supports MCP over HTTP. `mnesio-mcp` is **stdio
> only** today; MCP-over-HTTP is a planned addition (the substrate is the same
> handler). Until then, use stdio for MCP, or call `mnesio-server`'s REST API
> (`/api/search`, etc.) from Hermes' generic HTTP tools.

---

## 4. OpenClaw

OpenClaw installs skills, ~most of which wrap MCP servers. Register mnesio as an
MCP server in its config (key is typically `mcpServers`):

```json
{
  "mcpServers": {
    "mnesio": {
      "command": "/abs/path/to/target/release/mnesio-mcp",
      "args": [],
      "env": { "MNESIO_DATA": "/var/lib/mnesio" }
    }
  }
}
```

A ready-to-edit copy + a minimal ClawHub-style skill manifest live at
[`examples/integrations/openclaw.mcp.json`](examples/integrations/openclaw.mcp.json)
and [`examples/integrations/openclaw.skill.json`](examples/integrations/openclaw.skill.json).

---

## 5. The loop that makes the agent better (the wedge)

Storage-shaped memory stops at write/search. mnesio adds the **self-improvement
loop** — and that's what `mnesio_record_outcome` is for:

```
agent runs a task using a mnesio system prompt / skill (an "artifact")
        │
        ├── mnesio_search(query)              → relevant memories injected into context
        │
   task completes
        │
        └── mnesio_record_outcome(artifacts_used=[…], success, scores)
                    │
                    ▼
        procedural compiler (offline): reflect → propose K → shadow-eval → gate
                    │
        EvalReport::is_committable()?  (canaries + safety probe + Δobjective ≥ 0)
              │ yes                                  │ no
              ▼                                      ▼
     new PolicyArtifact version              rejected — old version stays active
     hot-swapped in (atomic)                 (Hard Rule #1: nothing unsafe commits)
```

Every committed artifact is versioned and reversible; a forgotten subject is
crypto-shredded; every belief traces to its source events. That is the
difference between "the agent remembers" and "the agent gets **verifiably**
better and can take it back."

---

## 6. What to test

1. **Protocol loop (deterministic, CI):** `agent_session_over_stdio` — spawns
   `mnesio-mcp` and drives initialize → tools/list → writes → search (asserts
   recall) → record_outcome. No real agent needed.
1b. **Real LLM agent loop (live):** `python3 examples/agent_loop_eval.py` — a
   real Ollama model **decides its own tool calls** against the real `mnesio-mcp`
   server and answers questions about private facts it can't otherwise know.
   Measured: **0% → 83%** with mnesio (llama3.2). This is the closest proxy to a
   real OpenClaw/Hermes session that runs without their full stack, and it
   exercises the exact MCP transport they use.
2. **Real-agent smoke (your environment):** drop the config above into OpenClaw
   / Hermes, run a task, watch the tool calls and the dashboard
   (`mnesio-server` → `/dashboard`).
3. **Value at scale (the moat):** `mnesio-bench` measures recall@k, latency
   p50/p95/p99, and throughput at 1k–100k memories, plus the procedural learning
   curve. See [`BENCHMARKS.md`](BENCHMARKS.md) for the real numbers.


## Code context in any editor

`mnesio_code_context` is why MCP is the distribution channel: Claude Code,
Cursor, Codex, GitHub Copilot's agent mode, Windsurf and Zed all speak the same
protocol, so one stdio server reaches every one of them with no per-editor
adapter.

Point it at a repository and describe the change you are making — not keywords.
The retrieval settings were measured against task-shaped queries derived from
real commit history, so "make the retry backoff configurable" retrieves better
than "retry backoff".

```jsonc
{
  "mcpServers": {
    "mnesio": {
      "command": "/absolute/path/to/mnesio-mcp",
      "env": { "MNESIO_DATA_DIR": "/absolute/path/to/mnesio-data" }
    }
  }
}
```

The same block works in Claude Code (`.mcp.json` or `claude mcp add`), Cursor
(`.cursor/mcp.json`), Windsurf, Zed and Codex. Copilot agent mode uses
`.vscode/mcp.json` with a `servers` key instead of `mcpServers`; the inner
object is identical.

### What it returns

Each symbol carries its path, its kind, and **why it is there** — either
retrieval matched your task, or it was pulled in as a callee of something that
did. Symbols that do not fit the budget degrade to their signature before being
dropped, and the ceiling is never exceeded.

### Two things to know

- **The first call on a repository indexes it** and is slow; later calls are
  fast. The index lives for the life of the server process.
- **Edits after that first call are not reflected** until you pass
  `refresh: true` or restart. This is a deliberate simplicity trade, and it is
  stated here rather than left to be discovered: serving stale code to an agent
  that is editing that code is this tool's worst failure mode.

### Language coverage

Six languages in a default build; **30** with `--features tree-sitter`
(rust, python, javascript, typescript, tsx, go, java, c, c++, c#, ruby, php,
swift, lua, elixir, ocaml ×3, r, dart, solidity, elm, kotlin, zig, haskell,
julia, objc, hcl/terraform, scala, bash). The error you get on an unindexable
repository lists what *your* build supports, not what some other build does.

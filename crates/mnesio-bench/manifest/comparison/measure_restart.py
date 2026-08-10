#!/usr/bin/env python3
"""Phase 18A's "done when": restart is warm, and the answer is unchanged.

Two halves, and the second is the one that is easy to skip. A cache that makes
restart fast but changes what comes back is not persistence, it is a different
index that happens to load quicker — and the failure is silent, because a
plausible-looking answer to a code question does not announce that it is worse.

So this runs the same query through three server lifetimes:

  cold   cache deleted, embeddings computed from scratch
  warm   fresh process, cache on disk
  warm2  fresh process again, to show the warm path is itself stable

and compares the returned context byte for byte, not just the timing.
"""
import json, os, shutil, subprocess, sys, time

REPO = os.path.abspath(sys.argv[1])
MCP = sys.argv[2]
TASK = sys.argv[3] if len(sys.argv) > 3 else "handle a request timeout and retry"

# A private cache root, rather than deleting from the user's `~/.cache`. The
# crate exposes `MNESIO_CACHE_DIR` for exactly this, so "cold" is guaranteed
# cold without a benchmark reaching into a real home directory to make it so.
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_restart_cache")

def run_once(label):
    proc = subprocess.Popen([MCP], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1,
                            env={**os.environ, "MNESIO_EMBEDDER": "fastembed",
                                 "MNESIO_CACHE_DIR": CACHE})
    def rpc(method, params, rid):
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid,
                                     "method": method, "params": params}) + "\n")
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if not line:
                raise SystemExit(f"{label}: server closed")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == rid:
                return msg

    rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "bench", "version": "0"}}, 0)
    t0 = time.time()
    msg = rpc("tools/call", {"name": "mnesio_code_context",
                             "arguments": {"repo": REPO, "task": TASK,
                                           "budget_tokens": 4000}}, 1)
    elapsed = time.time() - t0
    text = "".join(c.get("text", "") for c in
                   msg.get("result", {}).get("content", []))
    proc.terminate()
    return elapsed, text

shutil.rmtree(CACHE, ignore_errors=True)

cold_s, cold_txt = run_once("cold")
warm_s, warm_txt = run_once("warm")
warm2_s, warm2_txt = run_once("warm2")

print(json.dumps({
    "repo": os.path.basename(REPO),
    "first_query_cold_s": round(cold_s, 1),
    "first_query_warm_s": round(warm_s, 1),
    "first_query_warm2_s": round(warm2_s, 1),
    "speedup": round(cold_s / warm_s, 1) if warm_s else None,
    "identical_cold_vs_warm": cold_txt == warm_txt,
    "identical_warm_vs_warm2": warm_txt == warm2_txt,
    "context_bytes": len(warm_txt),
}, indent=2))

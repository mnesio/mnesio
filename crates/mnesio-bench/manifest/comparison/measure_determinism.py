#!/usr/bin/env python3
"""How much do two identical runs actually differ? **For one task only.**

> **Do not use this to support a determinism claim — use
> `measure_suite_determinism.py`.** This script takes a single task and repeats
> it. Per-task variance is 1-3 tasks in 40, so a single task is very likely to
> be one of the stable ones: on serde it reports `bytes_identical: true` and
> Jaccard 1.00 while the suite around it varies by 7pp. Two claims on this
> project ("0pp noise floor", Phase 18A "provably identical") were built on this
> script and both had to be retracted. Kept for drilling into *one* task once
> the suite harness has told you which task to look at.

`measure_restart.py` found that two warm starts return different context for
the same query. Byte-inequality alone does not say whether that matters: a
reordering of the same symbols is cosmetic, a different *set* of symbols is a
reproducibility problem that undermines every A/B this project runs.

So this compares runs by symbol set, not by bytes.
"""
import json, os, re, subprocess, sys

REPO = os.path.abspath(sys.argv[1])
MCP = sys.argv[2]
TASK = sys.argv[3]
RUNS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_restart_cache")

def once():
    p = subprocess.Popen([MCP], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1,
                         env={**os.environ, "MNESIO_EMBEDDER": "fastembed",
                              "MNESIO_CACHE_DIR": CACHE})
    def rpc(m, pr, i):
        p.stdin.write(json.dumps({"jsonrpc":"2.0","id":i,"method":m,"params":pr})+"\n")
        p.stdin.flush()
        while True:
            l = p.stdout.readline()
            if not l: raise SystemExit("closed")
            try: msg = json.loads(l)
            except json.JSONDecodeError: continue
            if msg.get("id") == i: return msg
    rpc("initialize", {"protocolVersion":"2024-11-05","capabilities":{},
                       "clientInfo":{"name":"b","version":"0"}}, 0)
    msg = rpc("tools/call", {"name":"mnesio_code_context",
              "arguments":{"repo":REPO,"task":TASK,"budget_tokens":4000}}, 1)
    p.terminate()
    return "".join(c.get("text","") for c in msg.get("result",{}).get("content",[]))

texts = [once() for _ in range(RUNS)]
sets = [set(re.findall(r"[\w./-]+\.(?:rs|py|ts|js|go)", t)) for t in texts]
base = sets[0]
print(json.dumps({
    "runs": RUNS,
    "bytes_identical": len(set(texts)) == 1,
    "symbol_sets_identical": all(s == base for s in sets),
    "files_per_run": [len(s) for s in sets],
    "in_every_run": len(set.intersection(*sets)),
    "in_some_run_only": len(set.union(*sets) - set.intersection(*sets)),
    "jaccard_vs_first": [round(len(s & base) / len(s | base), 3) for s in sets],
}, indent=2))

#!/usr/bin/env python3
"""Cross-process determinism over a whole task suite, not one query.

Replaces the single-task `measure_determinism.py` for any claim about
reproducibility. That script takes one task and repeats it, and on serde it
reports `bytes_identical: true` with Jaccard 1.00 — because the task it was
handed is one of the ~37 stable ones out of 40. Per-task variance is 1-3 in 40,
so a single task is very likely to be a stable one and proves nothing about the
suite. Two separate claims on this project ("0pp noise floor", "Phase 18A
provably identical") were built on that script and both had to be retracted.

Two rules this encodes so they cannot be forgotten:

1. **Every task, not a sample.** A suite metric is what benchmarks report, so a
   suite metric is what determinism has to be measured on.
2. **>= 6 runs per arm.** Two candidate fixes looked clean at n=3-4 and both
   evaporated at n=6. Below six, a quiet stretch is indistinguishable from a
   fix; MIN_RUNS refuses to pretend otherwise.

Each run gets its own `MNESIO_DATA`, so runs are independent — the default is
`./mnesio-data` relative to CWD, which silently shares state between runs.

Usage:
    python3 measure_suite_determinism.py <repo> <mcp-binary> [runs] [embedder]

`embedder` accepts `fastembed` (default, production) or `mock`. Use `mock` to
hold the vector side fixed and isolate everything downstream of it.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

# Below this, a run of agreement is not evidence of determinism. Enforced, not
# advisory: every retraction on this project came from a smaller sample.
MIN_RUNS = 6

REPO = os.path.abspath(sys.argv[1])
MCP = os.path.abspath(sys.argv[2])
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else MIN_RUNS
EMBEDDER = sys.argv[4] if len(sys.argv) > 4 else "fastembed"
N_TASKS = int(os.environ.get("TASKS", "40"))
BUDGET = int(os.environ.get("BUDGET", "4000"))

CODE_RE = re.compile(r"\.(rs|py|ts|tsx|js|go|java|c|h|cpp)$")
CITE_RE = re.compile(r"[\w./-]+\.(?:rs|py|ts|tsx|js|go|java|c|h|cpp)")


def git(args):
    return subprocess.run(["git", "-C", REPO] + args,
                          capture_output=True, text=True).stdout


def derive_tasks():
    """Suite from the repo's own history: subject = query, touched files = gold."""
    out = []
    for line in git(["log", "--format=%H\t%s", "-400"]).strip().splitlines():
        if "\t" not in line:
            continue
        sha, subj = line.split("\t", 1)
        if subj.lower().startswith("merge") or len(subj) < 20:
            continue
        files = [f for f in git(["show", "--name-only", "--format=", sha]).split()
                 if CODE_RE.search(f)]
        if files:
            out.append((subj, set(files)))
        if len(out) >= N_TASKS:
            break
    return out


def one_run(tasks, seq, tmp):
    """One fresh process with a private store; returns (hits, context texts)."""
    data = os.path.join(tmp, f"data-{seq}")
    proc = subprocess.Popen(
        [MCP], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
        env={**os.environ, "MNESIO_EMBEDDER": EMBEDDER, "MNESIO_DATA": data})

    def rpc(method, params, rid):
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid,
                                     "method": method, "params": params}) + "\n")
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if not line:
                raise SystemExit("mcp server closed")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == rid:
                return msg

    rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "determinism", "version": "0"}}, 0)
    hits, texts = [], []
    for i, (subj, gold) in enumerate(tasks):
        msg = rpc("tools/call", {"name": "mnesio_code_context",
                                 "arguments": {"repo": REPO, "task": subj,
                                               "budget_tokens": BUDGET}}, i + 1)
        text = "".join(c.get("text", "")
                       for c in msg.get("result", {}).get("content", []))
        texts.append(text)
        cited = {c.lstrip("./") for c in CITE_RE.findall(text)}
        hits.append(1 if any(any(g.endswith(c) or c.endswith(g) for c in cited)
                             for g in gold) else 0)
    proc.terminate()
    return hits, texts


def main():
    tasks = derive_tasks()
    if not tasks:
        raise SystemExit(f"no tasks derived from {REPO}")
    with tempfile.TemporaryDirectory(prefix="mnesio-determinism-") as tmp:
        runs = [one_run(tasks, i, tmp) for i in range(RUNS)]

    n = len(tasks)
    recalls = [round(100 * sum(h) / n) for h, _ in runs]
    varying_recall = [i for i in range(n) if len({r[0][i] for r in runs}) > 1]
    varying_bytes = [i for i in range(n) if len({r[1][i] for r in runs}) > 1]

    underpowered = RUNS < MIN_RUNS
    print(json.dumps({
        "repo": os.path.basename(REPO),
        "embedder": EMBEDDER,
        "runs": RUNS,
        "tasks": n,
        "recall_per_run_pct": recalls,
        "recall_spread_pp": max(recalls) - min(recalls),
        "tasks_varying_recall": varying_recall,
        "tasks_varying_bytes": varying_bytes,
        "deterministic": not varying_recall and not varying_bytes,
        # The spread is the floor any A/B on this repository must clear. Quote
        # this, not a suite score that happened to repeat.
        "noise_floor_pp": max(recalls) - min(recalls),
        "underpowered": underpowered,
        "warning": (f"{RUNS} runs is below MIN_RUNS={MIN_RUNS}; a run of "
                    "agreement here is not evidence of determinism")
        if underpowered else None,
    }, indent=2))


if __name__ == "__main__":
    main()

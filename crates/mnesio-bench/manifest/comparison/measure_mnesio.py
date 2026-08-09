#!/usr/bin/env python3
"""Score mnesio on exactly the protocol used for graphify.

The point of this file is fairness, not measurement convenience. `codeeval`
already reports mnesio's recall, but it scores at *symbol* level — did the
packed context contain the specific symbol the commit touched. graphify returns
no symbol bodies, so it can only be scored at *file* level: did it cite a file
the commit touched.

Comparing 59% symbol-level against 88% file-level would be the exact trick this
project criticises competitors for. So mnesio is re-scored here at file level,
on the same tasks, from the same git history, through the real MCP tool an
agent would call.
"""
import json, os, re, subprocess, sys

REPO = os.path.abspath(sys.argv[1])
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
MCP = sys.argv[4]

def sh(args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True).stdout

log = sh(["git", "log", "--format=%H\t%s", "-400"]).strip().splitlines()
tasks = []
for line in log:
    if "\t" not in line:
        continue
    sha, subj = line.split("\t", 1)
    if subj.lower().startswith("merge") or len(subj) < 20:
        continue
    files = [f for f in sh(["git", "show", "--name-only", "--format=", sha]).split()
             if re.search(r"\.(rs|py|ts|tsx|js|go|java|c|h|cpp)$", f)]
    if files:
        tasks.append((subj, set(files)))
    if len(tasks) >= N:
        break

print(f"# {len(tasks)} tasks · budget={BUDGET}", file=sys.stderr)

# One long-lived server process, because the first call indexes the repository
# and every later call reuses that index — which is also how an editor uses it.
proc = subprocess.Popen([MCP], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, text=True, bufsize=1,
                        env={**os.environ, "MNESIO_EMBEDDER": "fastembed"})

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
                   "clientInfo": {"name": "bench", "version": "0"}}, 0)

rows = []
for i, (subj, gold) in enumerate(tasks):
    msg = rpc("tools/call", {"name": "mnesio_code_context",
                             "arguments": {"repo": REPO, "task": subj,
                                           "budget_tokens": BUDGET}}, i + 1)
    text = ""
    for c in msg.get("result", {}).get("content", []):
        text += c.get("text", "")
    # Paths as the tool prints them, normalised to repo-relative like git's.
    cited = set(re.findall(r"[\w./-]+\.(?:rs|py|ts|tsx|js|go|java|c|h|cpp)", text))
    cited = {c.lstrip("./") for c in cited}
    hit = any(any(g.endswith(c) or c.endswith(g) for c in cited) for g in gold)
    rows.append({"hit": hit, "tokens": len(text) // 4, "files": len(cited)})
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(tasks)}", file=sys.stderr)

proc.terminate()

def pct(xs): return round(100 * sum(xs) / len(xs))
def med(xs): return sorted(xs)[len(xs) // 2]

print(json.dumps({
    "tasks": len(rows),
    "budget": BUDGET,
    "recall_file_level_pct": pct([r["hit"] for r in rows]),
    "median_tokens_complete": med([r["tokens"] for r in rows]),
    "median_files_touched": med([r["files"] for r in rows]),
}, indent=2))

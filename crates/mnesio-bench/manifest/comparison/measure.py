#!/usr/bin/env python3
"""Measure what a graphify query actually costs an agent.

`graphify query` returns node names + file:line and no code (verified: zero
lines matching a function/class definition at every budget tried). So its token
count is the cost of learning *where* to look, not of having the code. An agent
still has to open the cited files, and that is the number that competes with a
tool which returns the code itself.

Two costs are reported per query:

  pointer   tokens in graphify's own answer
  complete  pointer + reading every distinct file it cited

Gold is file-level — the files the commit actually touched — because that is
the finest level at which both tools can be scored without favouring either.
graphify names symbols but returns no bodies; mnesio returns bodies. Comparing
at file level asks the one question both can answer: did the context put the
agent in front of the code the commit changed?
"""
import json, os, re, subprocess, sys

# The competitor's version is pinned, and that is not a detail.
#
# graphify is installed with `uvx --from graphifyy`, which resolves the *latest*
# release at run time. Two corpus runs a week apart therefore benchmarked two
# different products: 0.9.37 on 2026-08-09 and 0.9.44 on 2026-08-16. That alone
# moved serde from 32/42 to 28/38 percent — reproduced exactly by re-pinning to
# 0.9.37 and re-running.
#
# Two wrong explanations were chased first (a shallow-clone depth in the corpus,
# then the manifest itself); both were checked and cleared. Pin the dependency,
# and record it in the output, so the next person does not repeat that.
GRAPHIFY = "graphifyy==0.9.44"

REPO = sys.argv[1]
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60

def sh(args, **kw):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, **kw).stdout

# Query = a real commit subject, gold = the files that commit touched. Same
# protocol as mnesio's gitsuite: a human wrote the subject for other reasons,
# before either tool existed, so neither can have been tuned to it.
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

size_cache = {}
def file_tokens(path):
    if path not in size_cache:
        p = os.path.join(REPO, path)
        try:
            size_cache[path] = os.path.getsize(p) // 4
        except OSError:
            size_cache[path] = 0
    return size_cache[path]

rows = []
for i, (subj, gold) in enumerate(tasks):
    out = subprocess.run(
        ["uvx", "--from", GRAPHIFY, "graphify", "query", subj, "--budget", str(BUDGET)],
        cwd=REPO, capture_output=True, text=True).stdout
    # Order matters: the output is BFS order, which is the only ranking signal
    # an agent has for deciding what to open first.
    ordered = []
    for m in re.finditer(r"^NODE .*?\[src=([^\s\]]*)", out, re.M):
        if m.group(1) and m.group(1) not in ordered:
            ordered.append(m.group(1))
    cited = set(ordered)
    pointer = len(out) // 4

    # "Read everything it cited" is the most generous reading of the pointer
    # cost and the least realistic. A real agent opens a few. Reporting only
    # the all-files number would be picking the assumption that flatters the
    # tool being compared against, so all three are kept — including the one
    # where graphify looks best.
    rows.append({
        "hit": bool(cited & gold),
        "hit_top3": bool(set(ordered[:3]) & gold),
        "hit_top5": bool(set(ordered[:5]) & gold),
        "pointer": pointer,
        "complete_top3": pointer + sum(file_tokens(f) for f in ordered[:3]),
        "complete_top5": pointer + sum(file_tokens(f) for f in ordered[:5]),
        "complete": pointer + sum(file_tokens(f) for f in cited),
        "files_cited": len(cited),
    })
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(tasks)}", file=sys.stderr)

def pct(xs): return round(100 * sum(xs) / len(xs))
def med(xs): return sorted(xs)[len(xs) // 2]

print(json.dumps({
    "tasks": len(rows),
    "budget": BUDGET,
    # Emitted, not just pinned: a number without the version it was measured
    # against cannot be compared to anything.
    "graphify": GRAPHIFY,
    "recall_all_cited_pct": pct([r["hit"] for r in rows]),
    "recall_top3_pct": pct([r["hit_top3"] for r in rows]),
    "recall_top5_pct": pct([r["hit_top5"] for r in rows]),
    "median_pointer_tokens": med([r["pointer"] for r in rows]),
    "median_complete_top3": med([r["complete_top3"] for r in rows]),
    "median_complete_top5": med([r["complete_top5"] for r in rows]),
    "median_complete_all": med([r["complete"] for r in rows]),
    "median_files_cited": med([r["files_cited"] for r in rows]),
}, indent=2))

#!/usr/bin/env python3
"""Run the graphify comparison across the pinned corpus, as a distribution.

One repository is not a result. `scaleeval` already caught single-repo numbers
misleading by 22pp on this project, so the headline here is the median across
repositories and the spread around it — never the best row.

Repositories too small to discriminate are still run and still shown, but
marked, because at a few dozen files a top-k answer reaches most of the corpus
and both tools score high by arithmetic rather than by ranking.
"""
import json, os, subprocess, sys

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

CORPUS = sys.argv[1]
MCP = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
HERE = os.path.dirname(os.path.abspath(__file__))

# Below this, a top-k answer covers so much of the repository that both tools
# score high without ranking anything. Mirrors MIN_DISCRIMINATING_SYMBOLS.
MIN_FILES = 60

repos = sorted(d for d in os.listdir(CORPUS)
               if os.path.isdir(os.path.join(CORPUS, d, ".git")))
out = []
for name in repos:
    path = os.path.join(CORPUS, name)
    n_files = sum(1 for root, _, fs in os.walk(path) if ".git" not in root
                  for f in fs if f.endswith((".rs", ".py", ".js", ".ts", ".tsx", ".go")))
    print(f"\n### {name} ({n_files} code files)", file=sys.stderr)

    print("  graphify: indexing…", file=sys.stderr)
    subprocess.run(["uvx", "--from", GRAPHIFY, "graphify", "update", "."],
                   cwd=path, capture_output=True, text=True)
    print("  graphify: querying…", file=sys.stderr)
    g = subprocess.run([sys.executable, f"{HERE}/measure.py", path, "2000", str(N)],
                       capture_output=True, text=True)
    print("  mnesio: querying…", file=sys.stderr)
    m = subprocess.run([sys.executable, f"{HERE}/measure_mnesio.py", path, "4000",
                        str(N), MCP], capture_output=True, text=True)
    try:
        gj, mj = json.loads(g.stdout), json.loads(m.stdout)
    except json.JSONDecodeError:
        print(f"  SKIP {name}: no result\n{g.stderr[-300:]}\n{m.stderr[-300:]}",
              file=sys.stderr)
        continue
    if gj["tasks"] == 0 or mj["tasks"] == 0:
        print(f"  SKIP {name}: no tasks derived", file=sys.stderr)
        continue
    out.append({
        "graphify": GRAPHIFY,
        "repo": name, "files": n_files, "tasks": gj["tasks"],
        "small": n_files < MIN_FILES,
        "g_recall_top3": gj["recall_top3_pct"], "g_recall_all": gj["recall_all_cited_pct"],
        "g_tok_top3": gj["median_complete_top3"], "g_tok_all": gj["median_complete_all"],
        "g_tok_pointer": gj["median_pointer_tokens"],
        "m_recall": mj["recall_file_level_pct"], "m_tok": mj["median_tokens_complete"],
    })
    r = out[-1]
    print(f"  → graphify top3 {r['g_recall_top3']}% @ {r['g_tok_top3']} tok · "
          f"mnesio {r['m_recall']}% @ {r['m_tok']} tok", file=sys.stderr)

print(json.dumps(out, indent=2))

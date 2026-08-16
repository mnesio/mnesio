#!/usr/bin/env python3
"""Format a corpus run as a distribution, and diff it against a prior run.

Exists because the raw JSON invites cherry-picking: ten rows, and a reader
naturally quotes the best one. The median and the spread are the result; a
single row never is.

Usage:
    python3 summarize.py <results.json> [prior.json]

With a prior run it also prints a per-repository diff, which is the only way
to tell a real change from run-to-run noise. Rows whose competitor version
differs are flagged rather than silently compared — the two corpus runs on
2026-08-09 and 2026-08-16 measured graphify 0.9.37 and 0.9.44, and comparing
them as though they were one product is exactly the mistake this guards.
"""
import json
import statistics
import sys

# Below this many code files a top-k answer reaches most of the repository, so
# both tools score by arithmetic rather than by ranking. Mirrors MIN_FILES in
# run_corpus.py and MIN_DISCRIMINATING_SYMBOLS in the Rust harness.
MIN_FILES = 60


def quartiles(xs):
    """min / p25 / median / p75 / max, on a list too short for numpy."""
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    p = lambda q: xs[min(n - 1, int(q * (n - 1) + 0.5))]
    return xs[0], p(0.25), statistics.median(xs), p(0.75), xs[-1]


def fmt_row(label, vals, unit="%"):
    q = quartiles(vals)
    if q is None:
        return f"| {label} | — | — | — | — | — |"
    cells = " | ".join(f"{v:g}{unit}" for v in q)
    return f"| {label} | {cells} |"


def versions(rows):
    """Distinct competitor versions in a result set."""
    return sorted({r.get("graphify", "unrecorded") for r in rows})


def main():
    rows = json.load(open(sys.argv[1]))
    prior = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None

    vs = versions(rows)
    print(f"# Corpus run — graphify {', '.join(vs)}\n")
    if len(vs) > 1:
        print("**Mixed competitor versions in one run — rows are not comparable.**\n")
    if "unrecorded" in vs:
        print("**Competitor version unrecorded.** Treat as un-reproducible: "
              "`uvx --from graphifyy` resolves whatever is latest.\n")

    big = [r for r in rows if not r["small"]]
    print(f"{len(rows)} repositories, {len(big)} large enough to discriminate "
          f"(≥{MIN_FILES} code files), {sum(r['tasks'] for r in rows)} tasks.\n")

    print("## Per-repository\n")
    print("| repo | files | graphify @top-3 | graphify @all | mnesio | Δ@top-3 | Δ@all |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -r["files"]):
        mark = "" if not r["small"] else " ᵗ"
        d3 = r["m_recall"] - r["g_recall_top3"]
        da = r["m_recall"] - r["g_recall_all"]
        print(f"| {r['repo']}{mark} | {r['files']} | {r['g_recall_top3']}% / "
              f"{r['g_tok_top3']:,} | {r['g_recall_all']}% / {r['g_tok_all']:,} | "
              f"{r['m_recall']}% / {r['m_tok']:,} | {d3:+} | {da:+} |")

    print("\n## Distribution over the discriminating repositories\n")
    print("| | min | p25 | median | p75 | max |")
    print("|---|---|---|---|---|---|")
    print(fmt_row("graphify recall @top-3", [r["g_recall_top3"] for r in big]))
    print(fmt_row("mnesio recall", [r["m_recall"] for r in big]))
    print(fmt_row("**delta @top-3**", [r["m_recall"] - r["g_recall_top3"] for r in big], "pp"))
    print(fmt_row("token ratio @top-3",
                  [round(r["g_tok_top3"] / r["m_tok"], 1) for r in big], "×"))
    print(fmt_row("graphify recall @all", [r["g_recall_all"] for r in big]))
    print(fmt_row("**delta @all**", [r["m_recall"] - r["g_recall_all"] for r in big], "pp"))
    print(fmt_row("token ratio @all",
                  [round(r["g_tok_all"] / r["m_tok"], 1) for r in big], "×"))

    losses = [r for r in big if r["m_recall"] < r["g_recall_all"]]
    ties = [r for r in big if r["m_recall"] == r["g_recall_all"]]
    print(f"\n**graphify wins {len(losses)} of {len(big)} at all-cited-files**"
          + (": " + ", ".join(f"{r['repo']} {r['g_recall_all']}% vs {r['m_recall']}%"
                              for r in losses) if losses else "")
          + (f". Ties: {', '.join(r['repo'] for r in ties)}." if ties else "."))

    if prior is None:
        return
    pv, cv = versions(prior), vs
    print(f"\n## Diff vs prior run (graphify {', '.join(pv)} → {', '.join(cv)})\n")
    if pv != cv:
        print("**Different competitor versions — the graphify columns below are a "
              "product change, not a measurement change.**\n")
    pm = {r["repo"]: r for r in prior}
    print("| repo | graphify @top-3 | graphify @all | mnesio | mnesio tok |")
    print("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -r["files"]):
        p = pm.get(r["repo"])
        if not p:
            print(f"| {r['repo']} | new | new | new | new |")
            continue
        d = lambda k: (f"{p[k]} → {r[k]}"
                       + (f" ({r[k] - p[k]:+})" if r[k] != p[k] else " ="))
        print(f"| {r['repo']} | {d('g_recall_top3')} | {d('g_recall_all')} | "
              f"{d('m_recall')} | {p['m_tok']:,} → {r['m_tok']:,} |")

    moved = [r["repo"] for r in rows
             if r["repo"] in pm and r["m_recall"] != pm[r["repo"]]["m_recall"]]
    print(f"\n**mnesio recall moved on {len(moved)} of {len(rows)} repositories**"
          + (f": {', '.join(moved)}." if moved else " — reproduced exactly."))


if __name__ == "__main__":
    main()

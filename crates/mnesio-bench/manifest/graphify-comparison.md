# mnesio vs graphify — token cost of one coding task

Measured 2026-08-09 across the pinned public corpus. **10 repositories, 7 of
them large enough to discriminate, 40 tasks each.** Reproduction at the bottom.

An earlier version of this file reported one repository and got the token ratio
**6× too flattering**. That correction is documented below rather than quietly
replaced, because it is the second time on this project that a single-repo
number has misled, and the first time it was also in our favour.

## The finding that shapes everything else

**`graphify query` returns no code.** It returns node names with `file:line`
and community labels — verified as zero lines matching a function or class
definition at budgets of 2 000, 8 000 and 16 000 tokens. It tells an agent
*where* to look. Its whole answer costs 1 000–1 700 tokens across the corpus.

`mnesio_code_context` returns the symbol bodies, packed to a hard ceiling.

So "tokens per query" is not a comparison: graphify is cheaper because it is a
smaller product, and the agent still has to open the files. The comparable
number is **tokens to have the code in hand** — graphify's answer plus reading
what it cited.

How many of those files an agent opens is a policy, and it is the single
biggest lever on the result, so both ends are reported:

- **top-3** — the agent opens the three highest-ranked cited files. Realistic.
- **all** — it opens every file cited. Generous to graphify, and the policy
  under which it sometimes wins.

## Setup

- **Corpus:** the `codeeval-v1` pinned manifest — 10 public repositories at
  fixed commits. Indexed whole rather than at the manifest's `subdir`, because
  a 40-file subdirectory is too small for a file-level metric to discriminate.
- **Tasks:** 40 real commit subjects per repository, from its own history.
  A human wrote each for other reasons, before either tool existed.
- **Gold:** the files each commit touched, per `git show --name-only`.
- **Metric:** did the returned context put the agent in front of a gold file.
- **Excluded from the distribution:** repositories under 60 code files, where
  a top-k answer reaches most of the corpus and both tools score by arithmetic
  rather than ranking. Their rows are shown, marked ᵗ.

**File-level, not symbol-level, and that is a concession.** `codeeval` scores
mnesio at symbol level — did the packed context contain the specific symbol the
commit changed — where it gets 59% on a comparable repository. graphify returns
no symbol bodies, so it cannot be scored that way at all. File level is the
finest question *both* tools can answer, and it flatters both.

## Per-repository

| repo | files | graphify @top-3 | graphify @all | mnesio | Δ@top-3 | Δ@all |
|---|---|---|---|---|---|---|
| zod | 403 | 42% / 23 653 | 58% / 78 107 | **62% / 2 928** | +20 | +4 |
| serde | 208 | 32% / 9 626 | 42% / 34 874 | **65% / 4 310** | +33 | +23 |
| express | 141 | 35% / 4 547 | 50% / 30 709 | **72% / 1 942** | +37 | +22 |
| ripgrep | 110 | 40% / 51 229 | 60% / 107 279 | **85% / 4 407** | +45 | +25 |
| flask | 83 | 45% / 10 293 | 60% / 52 530 | **88% / 4 277** | +43 | +28 |
| click | 78 | 48% / 18 165 | **78%** / 100 861 | 75% / 4 061 | +27 | **−3** |
| httpx | 60 | 68% / 15 130 | **88%** / 64 374 | 80% / 4 235 | +12 | **−8** |
| requests ᵗ | 37 | 62% / 28 471 | — | 88% / 4 308 | +26 | — |
| bytes ᵗ | 34 | 70% / 22 904 | — | 92% / 3 090 | +22 | — |
| fd ᵗ | 24 | 45% / 19 792 | — | 95% / 4 295 | +50 | — |

Cells are `recall / median tokens to have the code`.

## Distribution over the 7 discriminating repositories

At **top-3**, the realistic policy:

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| graphify recall | 32% | 38% | **42%** | 47% | 68% |
| mnesio recall | 62% | 69% | **75%** | 83% | 88% |
| recall delta | +12pp | +24pp | **+33pp** | +40pp | +45pp |
| token ratio | 2.2× | 2.3× | **3.6×** | 6.3× | 11.6× |

At **all cited files**, the policy most generous to graphify:

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| graphify recall | 42% | 54% | **60%** | 69% | 88% |
| recall delta | **−8pp** | +1pp | **+22pp** | +24pp | +28pp |
| token ratio | 8.1× | 13.8× | **15.8×** | 24.6× | 26.7× |

## Noise floor — measured, not assumed

flask's mnesio arm was run five times: **88, 90, 90, 90, 90**. One task in
forty, so roughly **±2pp** on a 40-task repository. This matches the ±2pp floor
already established for `codeeval` on the pinned corpus by a different route.

The source is retrieval nondeterminism, not the harness. Three *identical* warm
runs of a single query returned different symbol sets — 8 files in every run, 4
in only some, Jaccard 0.75–1.00 — while the 40-task aggregate stayed flat. So
the instability is in which marginal symbols are packed, and it mostly averages
out by the time it reaches a suite score. HNSW build randomness is the likely
cause; the index is rebuilt per process and nothing seeds it.

**Any delta below about 4pp in this file is not a result.**

## What this says, including the part that does not flatter us

- **mnesio wins on the median at both policies** — +33pp at top-3, +22pp at
  all-files — while using 3.6× to 15.8× fewer tokens. Every one of those
  deltas is far outside the ±2pp floor.
- **graphify beats mnesio on httpx** when it reads every file it cites: 88% vs
  80%, an 8pp gap that survives the noise floor. It pays 15× the tokens to get
  there, but the recall claim is theirs and must not be dropped from a summary.
- **click's −3pp is inside the noise floor** and should be read as a tie, not
  as a second graphify win. Reporting it as a loss would be as sloppy as
  reporting it as a victory.
- **The token ratio is far more repo-dependent than the recall gap.** 2.2× to
  11.6× at top-3, driven mostly by how large the cited files are, not by how
  good either tool is.

## The correction

The first version of this measurement used one repository, `claw-code`, and
reported **22× fewer tokens at +32pp recall**. Across the corpus the recall
gap held (+32 vs a +33 median) but the token ratio did not: the median is
**3.6×**, and claw-code sits far outside the p75 of 6.3×. It has unusually
large source files, which inflates the cost of reading anything.

`scaleeval` has now caught single-repo numbers misleading twice on this
project — once by 22pp on the ceiling gap, once by 6× here. Both times the
single repo was the flattering one. No comparison number leaves this repo on
one repository again.

## What this still does not establish

1. **graphify ran AST-only.** Its semantic extraction wants a Gemini or OpenAI
   key and `--mode deep` extracts more aggressively. Neither was used. This is
   its strongest untested counterargument.
2. **mnesio references more files per answer** (16 vs 11 on the repo where both
   were counted), and "did you reference a gold file" gets easier with breadth.
   This cannot explain the top-3 gap, where its whole answer is compared
   against graphify's three best files, but it inflates the all-files row.
3. **No noise floor.** Neither arm was repeated. The corpus removes
   *between-repo* variance, not run-to-run variance.
4. **Budgets are different knobs** — graphify's caps a pointer list, mnesio's
   caps returned code. Each ran at its own default (2 000 / 4 000).

## Correction to an earlier claim about outcomes

The Phase 18 competitor table said of both rivals: *"neither records whether
the retrieval actually helped."* **That is wrong about graphify.** Its CLI has
`save-result --outcome useful|dead_end|corrected` and `reflect`.

What does not exist there is a gate: `reflect` writes `LESSONS.md`, a document.
Nothing re-runs a canary suite, nothing can be rejected, nothing is stopped
from regressing. The mnesio claim that survives is *gated* improvement, not any
use of outcomes.

## Reproducing

```bash
# clone the pinned corpus at its recorded commits, then per repo:
uvx --from graphifyy graphify update .
python3 comparison/measure.py        <repo> 2000 40
python3 comparison/measure_mnesio.py <repo> 4000 40 target/release/mnesio-mcp
```

Both scripts derive tasks and gold from `git log` in the repository they are
pointed at, so there is no fixture to drift.

# mnesio vs graphify — token cost of one coding task

**Re-measured 2026-08-16** after retrieval was made deterministic. The updated
distribution is at the end; the 2026-08-09 body below is kept because one
number moved between the two runs, and hiding the earlier run would hide that.
It is now explained — the competitor's own version was not pinned — and the
two runs therefore measured graphify **0.9.37** and **0.9.44** respectively.

Originally measured 2026-08-09 across the pinned public corpus. **10 repositories, 7 of
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
# clone the pinned corpus at its recorded commits, then per repo.
# The version is pinned: `--from graphifyy` alone resolves whatever is latest,
# which is exactly how the two runs below ended up measuring different builds.
uvx --from graphifyy==0.9.44 graphify update .
python3 comparison/measure.py        <repo> 2000 40
python3 comparison/measure_mnesio.py <repo> 4000 40 target/release/mnesio-mcp
```

Both scripts derive tasks and gold from `git log` in the repository they are
pointed at, so there is no fixture to drift.


---

# Re-run 2026-08-16 — with a 0pp noise floor

The first run carried a ±2pp floor from non-deterministic retrieval, which is
why `click`'s −3pp had to be reported as a tie rather than a result. Retrieval
below `EXACT_SEARCH_MAX_SLOTS` is now byte-reproducible (0 of 40 queries vary
across processes, two repos), so the deltas can be read directly.

**Both arms were checked for determinism this time**, not just ours: graphify
returned identical numbers across 3 runs on flask and across 2 runs plus a full
graph rebuild on serde. Neither tool is now a source of run-to-run noise.

## Per-repository

| repo | files | graphify @top-3 | graphify @all | mnesio | Δ@top-3 | Δ@all |
|---|---|---|---|---|---|---|
| zod | 403 | 42% | 58% | **65%** | +23 | +7 |
| serde | 208 | 28% | 38% | **65%** | +37 | +27 |
| express | 141 | 35% | 50% | **75%** | +40 | +25 |
| ripgrep | 110 | 40% | 60% | **85%** | +45 | +25 |
| flask | 83 | 45% | 60% | **90%** | +45 | +30 |
| click | 78 | 48% | **75%** | 75% | +27 | **0** |
| httpx | 60 | 68% | **90%** | 80% | +12 | **−10** |
| requests ᵗ | 37 | 62% | 82% | 88% | +26 | +6 |
| bytes ᵗ | 34 | 70% | 82% | 92% | +22 | +10 |
| fd ᵗ | 24 | 45% | 75% | 95% | +50 | +20 |

## Distribution over the 7 discriminating repositories

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| graphify recall @top-3 | 28% | 38% | **42%** | 47% | 68% |
| mnesio recall | 65% | 70% | **75%** | 83% | 90% |
| **delta @top-3** | +12pp | +25pp | **+37pp** | +43pp | +45pp |
| token ratio @top-3 | 2.2× | 2.6× | **3.7×** | 6.0× | 11.6× |
| graphify recall @all | 38% | 54% | **60%** | 68% | 90% |
| **delta @all** | **−10pp** | +4pp | **+25pp** | +26pp | +30pp |
| token ratio @all | 7.9× | 12.4× | **15.5×** | 19.9× | 25.2× |

## What changed from the first run, and why

**mnesio moved +0 to +3pp** — zod 62→65, express 72→75, flask 88→90, the other
four unchanged. That is the exact-search fix: an exhaustive scan finds what an
approximate one missed. Small and in the expected direction.

**graphify moved on three rows, and none of them are ours** — serde 32→28 at
top-3 and 42→38 at all-files, click 78→75 at all-files, httpx 88→90 at
all-files. All three are the version change.

**An earlier version of this section credited two of those to our fix, and that
was wrong.** It read: *"click was −3pp, now exactly 0"* and *"httpx was −8pp,
now −10pp"*, presented as consequences of making retrieval deterministic.
**mnesio's number did not move on either repository** — click 75% and httpx 80%
in both runs. Both deltas changed because *graphify's* number moved underneath
them. Diffing the two result files makes it unambiguous:

| | mnesio moved | graphify moved |
|---|---|---|
| zod, express, flask | ✓ | |
| serde, click, httpx | | ✓ |
| ripgrep, requests, bytes, fd | | |

**The two sides moved on disjoint sets of repositories.** That separation is
itself the evidence: the exact-search fix touched only our arm, the version bump
only theirs. Reading a Δ column without asking *which side moved* is what
produced the wrong attribution, and the diff above is now the standard check.

## The number neither tool explained — resolved, and I was wrong twice

serde's graphify @top-3 was 32% on 2026-08-09 and 28% on 2026-08-16. I gave two
explanations for it before finding the real one, and both were wrong:

1. **"The shallow clone truncates history differently."** Checked: the
   by-pin checkout and the shallow-clone-then-rewind produce the *same* 1 400
   reachable commits and the same HEAD.
2. **"The manifest pins the tree but not reachable history."** Also wrong, and
   worse — it asserted a defect in shipped code. `manifest::materialize`
   already does `git init` + `git fetch --depth N origin <pin>` + `checkout`,
   so depth is relative to the pin, not to HEAD. It was correct all along.

**The actual cause: the harness did not pin the competitor's version.**
`uvx --from graphifyy` resolves the *latest* release at run time. The two runs
used graphify **0.9.37** and **0.9.44** — two different products.

Confirmed by re-pinning and re-running: 0.9.37 reproduces **32% / 42%**
exactly, 0.9.44 gives **28% / 38%**. So graphify got *worse* on serde between
those releases, and the corpus, the manifest and both tools' determinism were
never involved.

The harness now pins `graphifyy==0.9.44` and **emits the version in every
result**, because a number without the version it was measured against cannot
be compared to anything. The 2026-08-09 numbers should be read as 0.9.37 and
the 2026-08-16 numbers as 0.9.44 — not as one superseding the other.

## Everything the first run disclaimed still applies

graphify ran AST-only (its semantic extraction wants an API key), scoring is
file-level which flatters both tools, and mnesio references more files per
answer which inflates the all-files row. None of that changed.


---

# Re-run 2026-08-16 (second) — competitor version pinned

The first run with `graphifyy==0.9.44` **pinned and recorded in the output**.
Same corpus, verified against all ten manifest pins; graphify's index state
deleted first so 0.9.44 built from scratch; `mnesio-mcp` at the current HEAD.

## The version pin works

**Nine of ten repositories reproduced exactly** — recall *and* median token
count, to the digit. graphify reproduced on all ten. So with the version fixed,
the competitor arm is fully deterministic and the harness is reproducible.

## The tenth repository is the finding

serde's **mnesio** arm moved 65% → 62%. Repeating it alone:

| run | 1 | 2 | 3 | corpus 08-16 | corpus re-run |
|---|---|---|---|---|---|
| serde recall | 65% | 58% | 60% | 65% | 62% |

**A 7pp spread.** The `noise-floor.md` claim of 0pp was measured on two small
repositories and does not hold here; that file is corrected, and the floor to
quote on a serde-sized repository is **7pp**, not 0pp.

Diagnosed as two sources: fastembed's ONNX inference is not bit-reproducible
across processes (2 of 3 varying tasks), plus one source that survives a
deterministic embedder and lives in 1-hop graph expansion (root cause open).
Full workings in `noise-floor.md`.

## Per-repository

| repo | files | graphify @top-3 | graphify @all | mnesio | Δ@top-3 | Δ@all |
|---|---|---|---|---|---|---|
| zod | 403 | 42% / 23 647 | 58% / 35 655 | **65% / 2 902** | +23 | +7 |
| serde | 208 | 28% / 9 622 | 38% / 34 204 | **62% / 4 335** ⚠ | +34 | +24 |
| express | 141 | 35% / 5 766 | 50% / 30 709 | **75% / 1 955** | +40 | +25 |
| ripgrep | 110 | 40% / 51 234 | 60% / 111 052 | **85% / 4 393** | +45 | +25 |
| flask | 83 | 45% / 10 132 | 60% / 52 517 | **90% / 4 208** | +45 | +30 |
| click | 78 | 48% / 16 377 | **75%** / 100 739 | 75% / 4 166 | +27 | **0** |
| httpx | 60 | 68% / 15 819 | **90%** / 65 387 | 80% / 4 226 | +12 | **−10** |
| requests ᵗ | 37 | 62% / 30 767 | 82% / 80 329 | 88% / 4 305 | +26 | +6 |
| bytes ᵗ | 34 | 70% / 24 687 | 82% / 50 997 | 92% / 2 846 | +22 | +10 |
| fd ᵗ | 24 | 45% / 20 477 | 75% / 41 599 | 95% / 4 300 | +50 | +20 |

⚠ serde's mnesio cell is unstable across runs (58–65%); every other cell
reproduced exactly.

## Distribution over the 7 discriminating repositories

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| graphify recall @top-3 | 28% | 40% | **42%** | 48% | 68% |
| mnesio recall | 62% | 75% | **75%** | 85% | 90% |
| **delta @top-3** | +12pp | +27pp | **+34pp** | +45pp | +45pp |
| token ratio @top-3 | 2.2× | 2.9× | **3.7×** | 8.1× | 11.7× |
| graphify recall @all | 38% | 58% | **60%** | 75% | 90% |
| **delta @all** | **−10pp** | +7pp | **+24pp** | +25pp | +30pp |
| token ratio @all | 7.9× | 12.5× | **15.5×** | 24.2× | 25.3× |

The median delta reads +34pp rather than the +37pp of the previous run purely
because serde landed on 62 instead of 65 this time. Within the 7pp floor —
which is the point of having one.

## The standing summary, with its unflattering half attached

**mnesio wins the median at both policies** — +34pp at top-3 and +24pp at
all-files — for 3.7× to 15.5× fewer tokens.

**graphify wins httpx outright** at all-files, 90% vs 80%, and that survives
the noise floor. **click is an exact tie.** So at the policy most generous to
graphify it is 5 wins, 1 loss, 1 tie out of 7 — not a sweep.

And unchanged from the first run: graphify ran **AST-only** (its semantic
extraction wants an API key), scoring is **file-level**, which is an easier
question than the symbol-level one `codeeval` asks of mnesio, and mnesio cites
more files per answer, which inflates the all-files row.

## Reproducing

```bash
python3 comparison/run_corpus.py <corpus-dir> target/release/mnesio-mcp 40 \
  > results.json
python3 comparison/summarize.py results.json [prior.json]
```

`summarize.py` prints the distribution and, given a prior run, a per-repository
diff that shows **which side moved** — the check whose absence produced the
misattribution corrected above.

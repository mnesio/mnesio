# code retrieval at scale — 9 repositories, 526 queries, 5805 symbols

| repo | files | symbols | queries | symbol | whole-file | gap | rankable | unreachable |
|---|---|---|---|---|---|---|---|---|
| core | 23 | 1252 | 46 | 54% | 67% | −13pp | 17% | 4% |
| src | 19 | 518 | 60 | 42% | 68% | −27pp | 17% | 15% |
| src | 24 | 725 | 60 | 62% | 82% | −20pp | 23% | 7% |
| src ᵗ | 22 | 237 | 60 | 63% | 77% | −13pp | 12% | 12% |
| requests ᵗ | 19 | 320 | 60 | 67% | 83% | −17pp | 17% | 2% |
| flask ᵗ | 24 | 477 | 60 | 60% | 82% | −22pp | 8% | 10% |
| click | 17 | 670 | 60 | 62% | 82% | −20pp | 8% | 5% |
| httpx | 23 | 533 | 60 | 58% | 70% | −12pp | 12% | 10% |
| core | 34 | 1073 | 60 | 57% | 78% | −22pp | 17% | 18% |

ᵗ 3 repositories have fewer than 500 symbols. At that size top-`k` reaches most of the corpus, so they score ~100% regardless of ranking quality. Their rows are shown but they are **excluded from the distribution below** — including them pulls every quantile toward 100% and flatters the result.

## distribution across the 6 discriminating repositories

| metric | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| symbol recall | 42% | 54% | **58%** | 62% | 62% |
| whole-file recall | 67% | 68% | **78%** | 82% | 82% |
| ceiling gap | 12% | 13% | **20%** | 22% | 27% |
| rankable share | 8% | 12% | **17%** | 17% | 23% |
| unreachable share | 4% | 5% | **10%** | 15% | 18% |

_Quartiles, not a mean. Averaging recall across repositories of different size and language produces a number with no referent, and the spread — not the centre — is the finding: the symbol/whole-file trade is repo-dependent._

## skipped (1)

Listed rather than dropped: a suite that silently discards what it cannot handle reports a survivorship-biased result.

- **lib** — no parseable source: no symbols parsed under /tmp/mnesio-corpus/express/lib

## corpus

manifest **codeeval-v1**, 10 repositories, 10 evaluated, 0 refused.
wall clock **252s** against a declared budget of 2400s — within budget.

---

## Paired A/B: receiver-aware call binding — no detectable effect

Ran the corpus twice more, changing only the resolver (`MNESIO_CODE_BARE_NAME=1`
restores the pre-18 behaviour of binding `x.foo()` to any lone free `foo`), and
once more with **no change at all** as a noise control.

| comparison | symbol recall Δ | whole-file recall Δ |
|---|---|---|
| **noise** (strict vs strict) | max 2pp, mean 0.3pp | max 2pp, mean 0.4pp |
| **effect** (strict vs loose) | max 2pp, mean 0.2pp | max 2pp, mean 0.4pp |

**The effect is smaller than the noise floor.** Whole-file recall — which does
not consult the call graph at all and therefore *cannot* be affected by the
resolver — moved by exactly the same amount, which is what identifies the
variation as HNSW build randomness rather than anything the change did.

Two conclusions, and the second matters more:

1. The receiver-aware fix is **recall-neutral at corpus scale** (526 queries,
   10 repositories), consistent with the earlier 12-query result on `crates/`.
   It stays, on the strength of correctness — a wrong edge misleads context
   expansion — and because it made the hub list believable, not on recall.

2. **This harness cannot resolve effects below ~2pp per repository.** Any
   future change claiming a small improvement is unmeasurable here until the
   index build is seeded deterministically. Running it anyway would produce a
   number, and the number would be index randomness wearing the change's name
   — which is exactly the Phase 16 mistake the paired-A/B rule exists to
   prevent. The rule caught it; the control run is the reason.

---

## Noise floor — measured, not assumed

`manifest run --repeat 3` runs the *same* configuration three times and reports
the largest per-repository variation:

| | |
|---|---|
| symbol recall | **2pp** |
| whole-file recall | **2pp** |
| wall clock, 3 runs | 743s (budget 2400s) |

**Any A/B on this corpus must exceed 2pp to be a finding.** The resolver
comparison above moved by at most 2pp, which is why it is reported as null.

### Why repetition rather than a seed

`hnsw_rs` 0.3.4 builds its layer-assignment RNG with `StdRng::from_os_rng()`
and exposes no setter, so seeding is not available through the public API. It
would also be the wrong fix: a seed makes a run *reproducible*, but two
different configurations build two different graphs whatever the seed, so one
run per arm is still a single sample from each of two distributions. Knowing
the width of those distributions is what makes a comparison valid, and only
repetition gives you that.

Whole-file recall is the calibration: that arm never consults the call graph,
so its variation cannot be caused by any retrieval-policy change. It is a
clean read on index randomness alone.

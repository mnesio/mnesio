# Phase 18A — index survives a restart

Measured 2026-08-09 on `serde` (208 code files) through the real MCP tool, with
an isolated `MNESIO_CACHE_DIR` so "cold" is genuinely cold.

The phase's criterion has two halves:

> index a large repo, restart, first query is **warm** and **provably identical**
> to pre-restart.

**Both halves are now met** (2026-08-16). The original measurement below found
the second failing; the fix and its re-measurement are at the end.

## Warm: met

| | first query |
|---|---|
| cold (cache deleted) | 64.4 s |
| warm (fresh process) | 8.5 s |
| warm again (fresh process) | 8.2 s |

**7.5× faster**, and the warm path is itself stable. Embedding is the expensive
step and the cache removes it; what remains is parse plus view construction,
which is deliberately not cached (Hard Rule #4 — a view read back from a
snapshot is a second source of truth).

## Identical: not met

The returned context differs between cold and warm — **and between two warm
runs**. So this is not a cache-correctness problem. The index is
nondeterministic across process lifetimes regardless of where the vectors came
from.

Three identical warm runs of one query:

| | |
|---|---|
| bytes identical | no |
| symbol sets identical | no |
| files in every run | 8 |
| files in only some runs | 4 |
| Jaccard vs first run | 1.00, 0.91, 0.75 |

So roughly a quarter of the returned file set is unstable at the margin, while
the core 8 files are always present.

**How much it matters, measured rather than assumed.** The same 40-task suite
run five times scored **88, 90, 90, 90, 90** — one task in forty, about ±2pp.
The per-query instability largely averages out at suite scale, which is why it
has gone unnoticed: nothing in the benchmark output was obviously wrong.

## Why this is worth a document rather than a silent TODO

Two consequences, and the second is the expensive one.

1. **A user re-running the same query after a restart can get different code
   back.** Not worse code — the core symbols are stable — but different, with
   nothing in the answer saying so.
2. **Every A/B this project runs inherits it as a floor.** Phase 16 already
   recorded that "unpaired runs let HNSW build randomness masquerade as a
   reranker effect," and pairing within a single index run is the workaround
   that has been in use since. This measurement puts a number on what pairing
   is protecting against: ±2pp per 40-task repository. Any future claim below
   that is noise.

## The likely cause, and what would fix it

HNSW assigns layers randomly at build time, the index is rebuilt in every
process, and nothing seeds the generator. A seed threaded through
`VectorView`'s construction would make a build reproducible, at which point
"provably identical" becomes testable rather than aspirational — a fresh
process on an unchanged repository should return byte-identical context, and a
test can assert exactly that.

That is not done. **Phase 18A stays ◑ until it is**, and the persistence work
should not be described as complete on the strength of the 7.5× alone.

## Reproducing

```bash
python3 comparison/measure_restart.py     <repo> target/release/mnesio-mcp "<task>"
python3 comparison/measure_determinism.py <repo> target/release/mnesio-mcp "<task>" 3
```


---

# Resolved 2026-08-16 — both halves met

Re-measured on `claw-code/rust/crates/api/src` after removing the
approximation below `EXACT_SEARCH_MAX_SLOTS`:

| | before | after |
|---|---|---|
| first query cold | 64.4 s | 10.6 s |
| first query warm | 8.5 s | 1.1 s |
| warm again | 8.2 s | 1.1 s |
| speedup | 7.5× | **9.6×** |
| identical cold vs warm | **false** | **true** |
| identical warm vs warm | **false** | **true** |

And the determinism probe that exposed it, three fresh processes:

| | before | after |
|---|---|---|
| bytes identical | false | **true** |
| symbol sets identical | false | **true** |
| files in every run | 8 | 6 |
| files in only some | 4 | **0** |
| Jaccard vs first | 1.00, 0.91, 0.75 | **1.00, 1.00, 1.00** |

## What the fix was, and what it was not

**It was not seeding HNSW.** `hnsw_rs` 0.3.4 seeds its `LayerGenerator` with
`StdRng::from_os_rng()` and exposes nothing to control it. 0.3.4 is the current
release, so there is no version to upgrade to and no API to call. The obvious
framing of this task — "seed the index" — is not available.

What is available is *not approximating when there is no need to*. Below 50 000
slots `VectorView::search` now does an exhaustive scan with a total order:
`total_cmp` on distance so `NaN` cannot reorder anything, then slot id, which
is insertion order and therefore fixed by the event log. There is no graph, so
there is nothing for a different graph to change.

Cost, measured with `mnesio-bench scale` at `k=10`:

| slots | path | p50 | p99 | recall@10 |
|---|---|---|---|---|
| 1 050 | exact | 0.91 ms | 1.35 ms | 100% |
| 10 503 | exact | 1.50 ms | 1.83 ms | 100% |
| 47 263 | exact | **3.93 ms** | 4.59 ms | 100% |
| 52 515 | HNSW | 2.34 ms | 2.87 ms | 100% |

3.93 ms is the worst case being paid, at the top of the exact range. Recall on
that path is 100% *by construction* rather than by measurement, since nothing
is skipped.

## What is still not fixed

**Above 50 000 slots, results remain approximate and non-reproducible across
restarts.** The upstream limitation is untouched; the threshold routes around
it rather than solving it. `VectorView::is_deterministic()` reports which
regime a view is in, so a benchmark can assert reproducibility instead of
assuming it.

Every code repository this project has measured is far under the threshold
(178–5 690 symbols), so in practice the code path is now deterministic — but
"in practice" is doing work in that sentence and a large prose corpus would
land on the other side of it.

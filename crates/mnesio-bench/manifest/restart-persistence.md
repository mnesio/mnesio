# Phase 18A — index survives a restart

Measured 2026-08-09 on `serde` (208 code files) through the real MCP tool, with
an isolated `MNESIO_CACHE_DIR` so "cold" is genuinely cold.

The phase's criterion has two halves:

> index a large repo, restart, first query is **warm** and **provably identical**
> to pre-restart.

**The first half is met. The second is not.**

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

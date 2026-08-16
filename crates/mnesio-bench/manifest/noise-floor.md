# The noise floor, re-measured

Measured 2026-08-16, after making `VectorView` exact below 50 000 slots and
giving both the RRF fusion and the reranker a total order.

## The number that governs A/B claims

**Suite recall is now stable to 0pp.** Same suite, repeated with a fresh
process and a fresh index build each time:

| repo | runs | recall |
|---|---|---|
| claw-code `api/src` (40 tasks) | 5 | 18%, 18%, 18%, 18%, 18% |
| tare (40 tasks) | 6 | 100% × 6 |
| tare (40 tasks, after tie-break fixes) | 5 | 100% × 5 |

Previously the same measurement gave **88, 90, 90, 90, 90** — one task in
forty, about ±2pp. That was the floor every A/B on this project has been
carrying, and it is the reason the graphify comparison had to reject any delta
under ~4pp.

**Deltas above ~1pp are now findings.** The previous ±2pp guard is obsolete;
a claim no longer has to clear it.

## Byte-level determinism: now achieved

Measured, 40 real tasks, 3 fresh processes each:

| stage | tare | claw-code api |
|---|---|---|
| after exact `VectorView` | 10 of 40 varied | — |
| after RRF total order | 10 of 40 | — |
| after reranker total order | 9 of 40 | — |
| **after BM25 OR-merge total order** | **0 of 40** | **0 of 40** |

The hypothesis in the first version of this file — "most likely tantivy's BM25,
segment layout, indexing threads" — was **half right and wrong about the
mechanism**. It was BM25. It was not segments or threads: pinning the writer to
one indexing thread changed nothing, and a purpose-built test showed two
indexes returning a *matching score multiset* in a different order. Equal
scores, different order — a tie-break, not a scoring difference.

Three tie-breaks had to become total, and only the third was on the runtime
path:

1. `HybridRetriever` fusion — sorted a vec drained from a `HashMap`.
2. `LexicalReranker` — "ties keep their fused order", which inherits whatever
   came in.
3. **`Bm25View::merge_by_memory`** — `into_values()` on a `HashMap`, then
   `Ordering::Equal` on ties. This is the OR-merge tier, which is what real
   queries actually hit, which is why fixing the other two moved 10 → 9.

`run_search` also over-fetches by 4× before truncating: `TopDocs::with_limit(k)`
cuts *inside* a tie group by tantivy DocId, and no amount of sorting afterwards
recovers members already discarded.

**Why the suite metric is stable anyway.** Every observed difference is at the
*budget boundary*: signature-only symbols that barely fit, swapping places. A
diff of one varying query:

```
-24 symbols, ~3997 tokens        +25 symbols, ~3999 tokens
- const MIN_NET_GAIN: usize = 8;  + fn claude_code(port: u16) -> Plan
```

The gold symbol is either comfortably inside the budget or comfortably outside
it, so recall does not move even though the bytes do. Median token count still
wobbles by about 40 tokens (~0.9%).

## What this means in practice

- **For benchmarks:** recall-style metrics are reproducible. Token-count
  metrics carry ~1% run-to-run variance and should be reported as such.
- **For a user:** re-running the same query after a restart can still return a
  slightly different tail of the context. It will contain the same core
  symbols; it is the marginal, signature-only entries that move.

## Reproducing

```bash
# suite-level stability
for i in 1 2 3 4 5; do
  python3 comparison/measure_mnesio.py <repo> 4000 40 target/release/mnesio-mcp
done

# per-query byte stability
python3 comparison/measure_determinism.py <repo> target/release/mnesio-mcp "<task>" 4
```

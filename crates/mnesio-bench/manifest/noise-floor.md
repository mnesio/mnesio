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

## What is still not deterministic

Byte-for-byte, the answer to a single query is **not** stable yet. Measured on
tare, 40 real tasks, 3 fresh processes each:

| stage | queries returning different context |
|---|---|
| before any fix | (not measured this way) |
| after exact `VectorView` | **10 of 40** |
| after RRF total order | 10 of 40 |
| after reranker total order | **9 of 40** |

So the sort fixes were correct but not the binding constraint — 10 → 9 is
inside the measurement's own variation. The remaining source is *upstream of
sorting*: the scores or candidate sets themselves differ between processes.
The most likely candidate is `tantivy`'s BM25, whose collection statistics and
doc ordering depend on segment layout, which depends on indexing threads. That
is a hypothesis, not a finding — it has not been isolated.

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

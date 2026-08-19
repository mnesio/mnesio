# The noise floor, re-measured

Measured 2026-08-16, after making `VectorView` exact below 50 000 slots and
giving both the RRF fusion and the reranker a total order.

> **Retracted the same day — see "The 0pp claim was wrong" at the end.** The
> 0pp figure below is real for the two repositories it was measured on and
> **does not generalise**. serde varies by 7pp. Do not cite 0pp.

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


---

# The 0pp claim was wrong — measured 2026-08-16, same day

Re-running the pinned graphify corpus caught it. Nine of ten repositories
reproduced *exactly*, recall and median token count alike. serde did not:
**65% on the first run, 62% on the re-run.** Repeating serde's arm alone:

| run | 1 | 2 | 3 | corpus (08-16) | corpus (re-run) |
|---|---|---|---|---|---|
| recall | 65% | 58% | 60% | 65% | 62% |

**A 7pp spread, not 0pp.** Median token count was 4 335 in every run, so the
packer is stable in *size* while varying in *content*.

## What the original measurement got wrong

It was taken on `tare` (35 files) and `claw-code api/src` — two small corpora —
and stated as a property of the system. It is a property of those corpora.
serde is 237 files and 1 977 symbols, and it varies. **A floor measured on the
smallest available repositories is not a floor.** This is the same mistake
`scaleeval` already caught twice on this project, where tiny repositories
scored 100%/100% by arithmetic and inflated the distribution; the fix there was
`MIN_DISCRIMINATING_SYMBOLS`, and the same discipline was simply not applied
here.

The per-query probe reinforced the error. `measure_determinism.py` takes **one**
task and repeats it. On serde it reports `bytes_identical: true`, Jaccard 1.00
across four fresh processes — because the task it was handed (index 0) is one of
the 37 stable ones. **One query is a sample, not a proof.**

## What actually varies, measured

Per-task hits over three fresh processes on serde, 40 tasks, fastembed:

```
run1 (62%): 1000111101010011111100110111100100011111
run2 (57%): 1000111101010011110100110101100100011111
run3 (65%): 1000111101010011111100110111101100011111
                              ^      ^   ^
disagreeing tasks: 18, 26, 30  (3 of 40)
```

Swapping in the deterministic `MockEmbedder`, same protocol:

```
mock: disagreeing tasks: 18  (1 of 40)
```

So there are **two** sources:

1. **fastembed accounts for 2 of the 3.** ONNX inference is not bit-reproducible
   across processes, so embeddings differ in their low bits, and candidates
   that are near-tied on distance swap places.
2. **One source survives a deterministic embedder** — task 18, in 1 run of 3.
   Diffing its context shows the differing entries are labelled *"called by a
   match"*: **graph expansion**, not seed retrieval. Run 0 and run 2 are
   byte-identical; run 1 packs 26 symbols instead of 23.

## What has been ruled out for source 2

Each of these was checked, not assumed:

- **Approximate vector search.** serde is 1 977 symbols, far below
  `EXACT_SEARCH_MAX_SLOTS` (50 000), so `VectorView` is on the exact path.
- **File-walk order.** `memory.rs` sorts the file list before parsing.
- **Index instability.** Four re-indexes of serde give identical
  `1977 symbols / 927 resolved calls / 941 communities`.
- **A warm-up race against async embedding.** The disagreements are at tasks
  18, 26 and 30 — tasks 0–17 are identical in every run. A race against the
  embedding worker would cluster at the start.
- **ULID tie-break ordering.** `new_id()` is strictly monotonic within a
  process and creation order is deterministic, so ids sort the same way in
  every run even though their absolute values differ.

**The root cause is not yet identified.** It is somewhere in link-vector
construction or 1-hop expansion. `// TODO(phase-18):` — this stays open, and no
A/B on this project may claim a delta under **7pp** on a serde-sized repository
until it is closed.

## The floor to use

| corpus | measured floor |
|---|---|
| tare (35 files), claw-code api/src | 0pp, 5–6 runs |
| serde (237 files, 1 977 symbols) | **7pp**, 5 runs |

Quote the second. The first is not wrong, it is not general — and a floor is
only useful as an upper bound.


---

# Two candidate fixes, both retracted — measured 2026-08-16

Chasing the open source above produced two apparent fixes. **Both were sampling
artefacts, and both were caught only by re-measuring at a larger n.** Recording
them so the next attempt does not spend the same afternoon.

## Candidate 1 — single-threaded tantivy writer

`Bm25View` builds its `IndexWriter` with `index.writer(50MB)`, which uses
tantivy's default thread count. Multi-threaded indexing distributes documents
over threads, so segment layout — and DocId assignment with it — can differ per
process. `TopDocs(limit)` cuts *inside* a tie group by DocId before `run_search`
ever sees the candidates, so a total order applied afterwards cannot recover a
member already discarded. A real mechanism, and the existing comment in
`bm25.rs` dismissing it had only ever been tested *in-process*.

**At three runs it looked like a clean win**, mock embedder to hold the vector
side fixed:

| writer | recall varies | bytes vary |
|---|---|---|
| multi-threaded | 1/40 | 12/40 |
| single-threaded | **0/40** | **4/40** |

**At six runs, paired, it vanished:**

| writer | recall varies | bytes vary |
|---|---|---|
| multi-threaded | 1/40 | 12/40 |
| single-threaded | 1/40 | 10/40 |

Same task (18) varying in both arms. 12 vs 10 bytes is noise at this n. Not
shipped; the original comment was right and now carries the cross-process
numbers so it is not overturned again on a small sample.

## Candidate 2 — `OMP_NUM_THREADS=1`

fastembed hard-codes ONNX intra-op threads to `available_parallelism()`
(`text_embedding/impl.rs`) and `InitOptions` exposes no knob, but ORT's GEMM
kernels still honour OpenMP, and multi-threaded float reduction is not
bit-reproducible across processes.

**At four runs it looked like a complete fix** — recall 57/57/57/57, 0/40
varying, against 65/65/65/62 without it. It even appeared to change the
*level*, which made it look mechanistically real rather than lucky.

**Re-run on the same binary and configuration it gave 62/62/60/57, 3/40
varying.** The four identical runs were a coincidence.

## The lesson, which is the point of writing this down

Both candidates produced a *clean, plausible, mechanistically-motivated* result
at n = 3–4. Both were wrong. The per-run variance is roughly 1–3 tasks in 40,
so four runs agreeing is unremarkable, and a run of agreement reads exactly like
a fix.

This is the same failure the project already documented twice — an unpaired
Phase 16 A/B letting index randomness look like a reranker effect, and
`scaleeval`'s tiny repositories inflating a distribution. **It recurred here in
a new costume, while actively looking for it**, and it nearly landed a commit
that overturned a correct source comment.

**Minimum n for any determinism claim on this project is 6 runs per arm,
paired.** Below that the measurement cannot distinguish a fix from a quiet
stretch.

## Where the root cause stands

Unchanged and still open. Task 18 on serde varies in every configuration tried:
multi- and single-threaded tantivy, with and without OMP pinning, and with a
fully deterministic embedder. The floor to quote remains **7pp** on a
serde-sized repository.

# Phase 18B — scale, measured

Measured 2026-08-22/23 on a shallow clone of `kubernetes/kubernetes`:
**5,353,542 lines of Go across 17,875 files**, comfortably past the phase's
">1M-LOC repo" bar. Machine: 8-core Apple Silicon, APFS, `fastembed` on CPU.

The phase asks for two things. **One is met, one is not, and the one that is
not is the more important.**

## Indexing a 5.35M-LOC monorepo

| | |
|---|---|
| files indexed | 13,420 |
| symbols indexed | 73,567 |
| **cold index** | **2,263 s — 37.7 minutes** (30.8 ms/symbol) |
| warm query (index resident) | 0.08 s |
| event store on disk | 33.6 MB |

**An earlier version of this file reported 8.5 hours. That number was wrong by
13.5×** — it was measured while the same machine ran a greedy-ablation pass and
repeated `cargo build`s for the whole eight hours. Re-run on an idle machine,
the same repository indexes in **37.7 minutes**.

The mistake is worth naming precisely, because it is not one of the failure
modes this project had a rule for. It was not an unpaired A/B, or a corpus too
small, or an underpowered sample. It was a **single long-running measurement
sharing a machine with the work that produced it**, and nothing in the existing
discipline would have caught it.

37.7 minutes is a usable number and still not a good one: codebase-memory-mcp
publishes 28M LOC / 75k files in 3 minutes. But it is the difference between
"unusable at monorepo scale" and "slow at monorepo scale", and the earlier
version of this document asserted the former on contaminated evidence.

The two good numbers are unchanged: once resident, a query answers in **80 ms**
at this scale, and the whole event log for 73,567 symbols is **34 MB**.

*Symbol counts differ by build.* The `tree-sitter` build sees 30 languages and
finds 134,639 symbols across 13,710 files; the default `mnesio-mcp` build
supports 6 and finds 73,567. Both are reported so neither is mistaken for the
other.

*Peak memory is not reported* because the harness measured it wrongly —
`getrusage(RUSAGE_CHILDREN)` after terminating the child returned 1 MB, which
is impossible. A missing number is better than a fabricated one; it needs
re-measuring with sampling while the process is alive.

## What made it slow — one cause found and fixed, one still open

**Fixed: the BM25 view committed once per symbol.**
`MaterializedView::apply` stages an entry *and commits it*, so a live write is
searchable immediately — the right contract on the write path. The code
indexer's bulk rebuild was calling it per symbol, and every tantivy commit runs
`prepare_commit`, `save metas` and a garbage collection. Measured on the first
72 seconds of a Kubernetes index: **1,172 commits**, projecting to ~34,000
commits for the full run. The rebuild now stages everything and commits once —
**1 commit instead of 34,000**, confirmed in the logs.

**Fixed, but only after the obvious version of the fix made it 6.4× worse.**
`build()` called `embedder.embed(slice::from_ref(&m.content))` per symbol, so a
73,567-symbol repository paid that many separate model invocations. Batching
looked like free money. Measured cold on serde (2,015 symbols), paired and
interleaved:

| embedding | median cold index |
|---|---|
| one symbol at a time | 40.8 s |
| **batched 256, unsorted** | **290.1 s — 6.4× SLOWER** |
| batched 32, length-sorted | **32.7 s — 1.24× faster** |

The regression is padding. A transformer pads every sequence in a batch to its
longest member, and symbol bodies run from a three-line accessor to a 500-line
function, so a length-mixed batch of 256 pays the maximum length for all 256.
Sorting by length before chunking makes each batch homogeneous, and only then
does batching win.

n = 6 paired runs per arm, interleaved so machine drift hits both arms equally.
The arms do not overlap: batched max 36.8 s is below unbatched min 39.7 s.

**1.24× is a real win and a much smaller one than "73,567 calls became 288"
suggests.** Per-call overhead was never the dominant term.

## Where the time goes

**Parse + plan + graph is 112 seconds** on the same checkout (`mnesio-code`, no
embedding). So essentially none of the cold index is parsing, symbol
extraction, edge resolution or community detection — it is all embedding plus
vector insertion.

### Per-symbol cost is flat across the corpus, and higher on Kubernetes

Cold index over the pinned corpus, one repository per row, idle machine:

| repo | symbols | cold | ms/symbol |
|---|---|---|---|
| fd | 372 | 9.6 s | 25.8 |
| bytes | 803 | 12.5 s | 15.6 |
| requests | 807 | 14.9 s | 18.5 |
| httpx | 1,241 | 25.4 s | 20.5 |
| flask | 1,655 | 25.2 s | 15.2 |
| click | 1,932 | 32.5 s | 16.8 |
| serde | 2,015 | 31.5 s | 15.7 |
| ripgrep | 3,103 | 62.1 s | 20.0 |
| zod | 6,445 | 57.2 s | 8.9 |
| **kubernetes** | **73,567** | **2,263 s** | **30.8** |

Between 372 and 6,445 symbols the per-symbol cost is **flat or falling** — zod,
the largest, is the cheapest at 8.9 ms. So there is no superlinear blow-up in
that range, which is what ruled out the first hypothesis (HNSW insertion cost
growing with graph size) for small repositories.

Kubernetes costs **30.8 ms/symbol**, roughly 2× the corpus median. That is a
real gap and a modest one — not the 20× the contaminated run implied. It is
consistent with HNSW insertion becoming more expensive past 50,000 vectors, but
2× on one data point is not enough to claim that, and it is recorded as an open
question rather than an explanation.

**A projection from the ladder would have said ~20 minutes; the answer was
37.7.** Extrapolating flat per-symbol cost from a 6,445-symbol repository to a
73,567-symbol one was wrong by nearly 2×, which is its own small lesson about
extrapolating past the measured range.

## Freshness check — the per-query tax

The check runs on **every query**, so at monorepo scale its cost is paid
constantly. p50 over 15 iterations on the same repository:

| change | p50 | note |
|---|---|---|
| baseline | 201.7 ms | |
| metadata captured during the walk | 134.5 ms | one `stat` per file instead of two |
| parallel walk | 75.5 ms | shared queue, `available_parallelism()` workers |
| order-independent fingerprint | **61.6 ms** | removes a 13,710-path sort |

**3.3× faster, and the <20 ms target is still NOT met.**

It is not met because the walk is at the syscall floor, not because it is
badly written: `find` over the same filtered tree (25,721 entries) costs
**250–300 ms single-threaded** on this machine. Doing that traversal *plus* a
`stat` per source file *plus* hashing, in 61.6 ms on 8 cores, is roughly what
8 cores should buy.

**Reaching 20 ms needs a different architecture, not a faster walk.** A
filesystem watcher (FSEvents / kqueue / inotify) maintaining a dirty flag makes
the no-change path O(1) instead of O(files). That is the honest recommendation;
further micro-optimisation of the walk will not get there.

### A correctness fix came out of it

`DirEntry::file_type` does not follow symlinks where `Path::is_dir` does, so
symlinked directories are no longer walked. Kubernetes aliases
`cluster/gce/{cos,custom,ubuntu}` to a single `gci` directory, and following
them indexed the same 16 files **four times** — exactly the 48-file drop
observed (13,758 → 13,710). Duplicated symbols inflate the corpus and fill
retrieval with identical candidates. Pinned by a test, along with two tests
that the parallel and serial walks fingerprint identically and that the
fingerprint does not depend on walk order.

## Verdict

| criterion | status |
|---|---|
| a >1M-LOC repo indexes with published numbers | **met** — 5.35M LOC, numbers above |
| freshness check <20 ms p50 at that size | **not met** — 61.6 ms, floor-bound |

Phase 18B stays **◑** on the freshness half. The scale claim now exists, is
measured on an idle machine, and is unflattering without being wrong.

## Reproducing

```bash
git clone --depth 1 https://github.com/kubernetes/kubernetes ~/mnesio-scale-k8s
cargo run --release -p mnesio-code --features tree-sitter \
  --example freshness_bench -- ~/mnesio-scale-k8s 15
```

**Run it on an idle machine.** That is not boilerplate: the first version of
this document reported 8.5 hours because the measurement shared a laptop with
the work that produced it, and the number was 13.5× too high. A wall-clock
benchmark has no way to tell you it was starved of CPU — it just returns a
larger number, which looks exactly like a slower system.

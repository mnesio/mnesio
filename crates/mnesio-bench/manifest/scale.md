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
| **cold index** | **30,577 s — 8.5 hours** |
| warm query (index resident) | 0.08 s |
| event store on disk | 33.6 MB |

**8.5 hours is not a credible scale story**, and it should be quoted exactly
that way. codebase-memory-mcp publishes 28M LOC / 75k files in 3 minutes. We
now have a number for a monorepo where before we had none, and the number is
bad.

The two good numbers are real: once resident, a query answers in **80 ms** at
this scale, and the whole event log for 73,567 symbols is **34 MB**.

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

**Still open: embeddings are computed one symbol at a time.**
`build()` calls `embedder.embed(slice::from_ref(&m.content))` inside the
per-symbol loop, so a 73,567-symbol repository pays 73,567 separate model
invocations with no batching. This is the dominant remaining cost and the
obvious next fix; `FastEmbedEmbedder` already accepts a batch and has a
`with_batch_size` knob that nothing on this path uses.

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

Phase 18B stays **◑**. The scale claim now exists and is unflattering, which is
the point of measuring it.

## Reproducing

```bash
git clone --depth 1 https://github.com/kubernetes/kubernetes ~/mnesio-scale-k8s
cargo run --release -p mnesio-code --features tree-sitter \
  --example freshness_bench -- ~/mnesio-scale-k8s 15
```

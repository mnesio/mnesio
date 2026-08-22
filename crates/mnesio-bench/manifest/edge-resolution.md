# Phase 18F — what actually caps edge resolution

Measured 2026-08-16 on the pinned `codeeval-v1` corpus.

CLAUDE.md called edge resolution *"the single weakest measured number in the
crate"* and named LSP-grade type resolution as the one competitor feature worth
copying. That was right, but the file did not say **how much** of the gap types
would close, and the number was never sized before work started. This document
sizes it — and in doing so shows that the work I actually did was aimed at the
wrong bucket.

## The three-way split, which a single rate hides

"19% of call sites resolved" says 81% is missing. It does not say how much of
that 81% is *recoverable*, and the two halves have opposite prospects:

- **ambiguous** — several symbols share the name, none in the calling file. A
  ranking problem. Addressable by imports.
- **unresolved, receiver-shadowed** — `x.name()` where `name` *does* exist in
  this repository, dropped rather than bound to a coincidental match. A typing
  problem. Addressable only by type resolution.
- **unresolved, absent** — the name appears nowhere in the repository. A call
  into the standard library or a dependency. **Not addressable at all**, and
  not worth addressing: there is no indexed symbol to expand to.

| repo | resolved | receiver-shadowed | ambiguous | absent |
|---|---|---|---|---|
| serde | 19% | **26%** | 5% | ~49% |
| flask | 22% | **35%** | 10% | ~33% |
| click | 18% | **42%** | 10% | ~30% |
| zod | 10% | **39%** | 15% | ~36% |
| ripgrep | 19% | **37%** | 2% | ~42% |
| httpx | 13% | **47%** | 3% | ~37% |

## What this says about the roadmap

**Type resolution is worth 26-47 percentage points.** It is the largest
addressable bucket on every repository measured, by a wide margin. CLAUDE.md's
instinct was correct and this is now a sized bet rather than a hunch: a perfect
LSP integration would take serde from 19% to ~45% and httpx from 13% to ~60%.

**Import disambiguation is worth 2-15pp**, and only if perfect.

**A third of every repository is unreachable by any technique.** Resolution can
never approach 100%, so "23-46% edge resolution" should stop being quoted as
though 100% were the target. The honest ceiling is roughly **50-63%**.

## What was built, and why it barely moved the number

Import-aware resolution (`extract_imports` + the hint tie-break in
`index::resolve`). When several files define a name, the calling file has
usually already said which one it means — `use crate::de::value::Error`,
`from flask.app import Flask`. That is the language's own disambiguation,
written by the author, needing no type inference.

It is narrowing-only: a hint that matches zero candidates or more than one
changes nothing, so it can add resolutions but never replace a correct binding
with a guess. Four tests pin that, including a control showing the same call
staying ambiguous without the import, and a test that one file's import cannot
redirect another file's call.

**Measured effect across the corpus:**

| repo | before | after |
|---|---|---|
| flask | 21% | 22% |
| click | 17% | 18% |
| serde | 19% | 19% (+19 edges) |
| ripgrep, bytes, fd, requests, httpx, express, zod | unchanged | unchanged |

**+1pp on two repositories of ten.** It is correct, exact (parsing is
deterministic, so there is no run-to-run noise to clear) and free, and it is
kept for that reason — but it is **not a win and must not be quoted as one**.
It was aimed at a bucket that turned out to be 2-15% of call sites, and it
captured a small share of that.

The useful output of this work is the measurement, not the code: **nobody
should start an LSP integration without knowing it is worth 26-47pp, and nobody
should have spent a week on import resolution to buy 1pp.** Sizing the buckets
first would have said so in an afternoon.

## Reproducing

```bash
cargo build --release -p mnesio-code
target/release/mnesio-code ~/mnesio-corpus/<repo>
```

The three-way split and the receiver-shadowed slice print under the resolution
rate. `EdgeStats::unresolved_receiver_shadowed` is the field that sizes the LSP
bet; it is derived at index time and persisted nowhere (Hard Rule #4).

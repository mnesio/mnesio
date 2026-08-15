# Causal contribution over code retrieval

Measured 2026-08-10 on `claw-code/rust/crates/api/src` — 21 training tasks,
178 candidate symbols, **3 759 retrievals**, leave-one-out.

This is the intervention the observational work pointed at. Every correlational
route to per-symbol credit failed, each for a different reason
(`wedge-credit-assignment.md`); the common cause was that an outcome is one bit
shared across ~20 packed symbols, and **nothing in the data varies one symbol
while holding the rest fixed**.

Masking does exactly that: remove one symbol, re-run the same task, measure
whether the answer changes.

## Result

Baseline recall 67%.

| class | symbols | share |
|---|---|---|
| load-bearing — removal lowered recall | **6** | 3% |
| inert — no measurable effect | 172 | 97% |
| harmful — removal *raised* recall | **0** | 0% |

Every load-bearing symbol scored exactly **+0.048**, which is 1/21 — one task.
So each is the sole carrier of a single task, with no other packed symbol
covering it.

## What this establishes

**Credit is real, measurable, and extremely concentrated.** 3% of what
retrieval packs carries anything at all; 97% could be removed without changing
whether the task was answered. That is the causal version of the token-budget
argument the roadmap has been making heuristically — not "top-k is probably
wasteful" but "these specific 172 symbols demonstrably were".

**Zero harmful, confirmed causally.** The correlational suppression search
found nothing to suppress and this agrees by a different method. Suppression
is not merely unlucky on this data; there is genuinely nothing consistently
poisonous to remove.

## What it does not establish, and the limit is structural

**Under leave-one-out, "inert" conflates *useless* with *redundant*.** A symbol
whose task is also covered by another packed symbol scores zero, because
masking it alone changes nothing. The 6 load-bearing symbols account for
0.29 of the 0.67 baseline; the remaining 0.38 of recall is carried by sets
whose members individually score zero.

So **97% inert is an upper bound on what is safely droppable, not a
measurement of it**. Dropping all 172 would not leave recall unchanged.
`ScoreMode::GreedyAblation` is the mode that separates the two — it ablates
iteratively, so the second of a redundant pair becomes load-bearing once the
first is gone — and it costs `O(n²)` evaluations: ~90 000 retrievals on this
module against 3 759. That is the next measurement, not a caveat to wave at.

## Implementation

`mnesio-bench::codecausal` is the adapter and only the adapter:
`CodeCounterfactual` implements `CounterfactualEvaluator` over a code suite,
where `evaluate(masked)` is recall with those symbols suppressed. Scoring,
bounds and the report come from `mnesio-causal` unchanged.

Contribution is measured on the **training** split. Measuring it on held-out
would leak the answer into any rule learned from it.

The pass is offline and bounded — `max_candidates` caps it (Hard Rule #6) and
it never touches the write path (Hard Rule #5).

## Reproducing

```bash
mnesio-bench learncurve --dir <module> --embedder fastembed --queries 200 --causal
```

Off by default: one full suite pass per candidate is thousands of retrievals.

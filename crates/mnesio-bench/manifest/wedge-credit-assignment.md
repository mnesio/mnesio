# The wedge is blocked on credit assignment, not on data

Measured 2026-08-10 on two real modules of `claw-code` (runtime, api), through
`mnesio-bench learncurve` — git-derived tasks, train/canary/held-out split, the
real gate.

Four things were tried in order. Each was measured before the next was built,
and each failed for a different reason. The reasons converge on one conclusion.

## 1. Suppression — nothing looks harmful

| module | decisive/symbol | best symbol | cleared floor of 5 | looked harmful |
|---|---|---|---|---|
| runtime | 1.51 | 11 | 12 | **0** |
| api | 2.36 | 14 | 20 | **0** |

The evidence floor is reachable and comfortably cleared. No symbol had a
success rate at or below `max_success_rate = 0.2`, because baseline recall is
88% — almost nothing sits in contexts that fail four times in five.

**You cannot learn by removing things when nothing is consistently poisonous.**

This also corrected a standing claim that the loop was starved at "0.12
outcomes per symbol". That number divided task count by symbol count; the
learner counts decisive outcomes per symbol, and each task contributes one to
each of the ~20 symbols it packed.

## 2. Promotion by success rate — selects the commonest words

So the opposite rule type: a term→symbol pairing that kept coming with success
gets the symbol pulled into context for later tasks using that term.

It fired. Three proposals per module, all committed by the gate, and:

```
committed — 4 of 4 tasks mentioning "tests" succeeded  (100% success)
committed — 9 of 9 tasks mentioning "api"   succeeded  (100% success)
```

**Held-out: +0.0pp on both.**

The terms it chose are the least discriminating in each corpus. With baseline
success at 88%, every sufficiently-evidenced pairing ties at ~100%, so the
tie-break degenerates to evidence volume — which is term frequency.

## 3. Promotion by lift over the base rate — does not bind

Requiring the pairing to beat the batch's own success rate by 10pp changed
nothing: identical rules, identical +0.0pp. At base 0.88 a pairing of 4/4
scores 1.00 and clears any sensible margin.

## 4. Promotion by significance — correctly refuses, and still does not help

The right test is whether the run would be surprising by chance:
**P(4 of 4 | p = 0.88) = 0.60.** It is the single most likely thing to observe.

With an exact binomial upper tail at α = 0.05:

| module | proposals before | after |
|---|---|---|
| runtime | 3 | **0** |
| api | 3 | **1** |

Five of six rules were correctly rejected as noise. The survivor — "9 of 9
tasks mentioning *api*", p = 0.027 — is statistically legitimate and still
useless: `api` appears in nearly every task in that module, so the rule fires
everywhere and adds a symbol that is not the gold one for held-out tasks.
Held-out: **+0.0pp**.

## What this establishes

**The blocker is credit assignment.** Not evidence volume, not the rule type,
not the thresholds — all three were varied and measured.

An outcome is one bit shared across the ~20 symbols packed into that context.
No observational statistic can say which of them mattered, because the data
never varies one symbol while holding the rest fixed. A term that predicts
success predicts it because tasks mentioning that term mostly succeed, not
because the paired symbol helped.

`learn.rs` has said this from the start — *"Phase 10's counterfactual masking
is what upgrades a correlation to a contribution"* — as a caveat. It is not a
caveat. It is the prerequisite. **Phase 10 is not an enhancement to the wedge;
it is the thing that makes the wedge possible**, and no amount of extra
dogfooding data changes that, because more samples of a confounded signal are
still confounded.

## What was kept

The significance test stays, and the thresholds were not tuned to let the
current data through. A promoter that proposes nothing here is behaving
correctly; one that emits three frequency artifacts and passes them through the
gate is worse than one that stays silent, because the gate cannot catch a rule
that is merely useless.

The gate itself came out well: it adjudicated every proposal and never
regressed canaries. What it cannot do is reject a rule that neither helps nor
harms, which is precisely what a confounded correlation produces.

## Reproducing

```bash
mnesio-bench learncurve --dir <module> --embedder fastembed --queries 200
```

The report states which constraint bound — starved of evidence, or evidenced
and nothing qualified — rather than printing a bare `0 proposals`.

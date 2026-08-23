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


---

# 5. Causal masking — wired, measured, and still +0.0pp (2026-08-23)

The previous section concluded that Phase 10 masking *is* the prerequisite, not
an enhancement. That link is now built: `learncurve::causal_curve` compiles a
contribution pass into suppression rules and puts them through the same gate,
cumulatively, refusing on any canary regression. Contribution is measured on the
training split only, so canaries and held-out are genuinely out-of-sample.

Three tests pin it, including both directions of Hard Rule #1 — a
measured-harmful symbol is suppressed and lifts held-out, and a suppression that
breaks canaries is **refused despite causal evidence**.

## A measurement bug that was hiding most of the signal

The first runs reported **1 load-bearing symbol out of 300** and 0 harmful,
under both leave-one-out *and* greedy ablation. That looked like a strong
negative result. It was partly an artefact.

`epsilon` was fixed at `0.02`, with a comment deriving it from a **24-task**
suite: one task flipping is `1/24 = 0.042`, and half of that is a sensible
noise floor. The flask suite has **57** training tasks, where one flip is
`0.0175` — *below* the threshold. **Every single-task effect was being
classified `inert` by arithmetic rather than by measurement.**

`code_causal_config_for(tasks)` now sets `epsilon = 0.5 / tasks`, so the floor
is always half of one task flip. Re-measured on flask, same run otherwise:

| | fixed ε = 0.02 | scaled ε = 0.0088 |
|---|---|---|
| load-bearing | 1 (0%) | **18 (6%)** |
| inert | 299 | 282 |
| harmful | 0 | **0** |

**18× more load-bearing symbols become visible.** Any earlier conclusion about
contribution on a suite larger than 24 tasks was measured through a threshold
too coarse to see it.

## Greedy ablation did not recover what LOO missed — my prediction was wrong

The redundancy hypothesis was that a commit touches several symbols, so masking
one leaves another gold symbol in context and contribution reads ~0.
`ScoreMode::GreedyAblation` exists precisely for that case, and I expected it to
find harmful symbols that leave-one-out could not.

It did not. On flask, 60 candidates, set-level ablation:
**1 load-bearing (2%), 59 inert (98%), 0 harmful (0%)** — the same picture LOO
gave. Redundancy was not the binding constraint.

## What is actually binding: the outcome signal cannot produce a harmful symbol

Zero harmful symbols under both modes, before *and* after the epsilon fix. So
suppression still has no target, still proposes nothing, and held-out recall is
still **84% → 84%, +0.0pp**.

The reason is structural, and it is a property of the *benchmark*, not of the
method. The outcome here is "did the packed context contain a gold symbol".
Removing a symbol can only help that metric in the narrow case where it frees
budget the gold symbol then occupies — rare, because the packer is already
ranking. Nothing in this signal can make a symbol *actively mislead*, which is
what a harmful contribution means.

**A harmful symbol requires an outcome that wrong context can damage** — a build
that fails, a test that breaks, an edit that gets rejected. That is what
dogfooding produces and what a git-derived suite structurally cannot, and it is
the same conclusion §1 reached from the opposite direction.

## Where that leaves the wedge

- The mechanism is **built and correct**: contribution → rules → gate →
  held-out, with the refusal path tested.
- The measurement is **now correctly scaled**, and 18 load-bearing symbols are
  visible where 1 was before.
- The demo is **still not met**: no positive learning curve, because the only
  rule type wired is suppression and nothing is harmful under this signal.

The next thing to try is **promotion from load-bearing symbols**, which now has
18 candidates rather than 1. It needs *per-task* contribution to scope the rule
to a query class — a global boost would inject the symbol into every query — and
`ContributionReport` currently exposes only the aggregate. That is the specific
missing piece, stated so the next attempt does not start by rediscovering it.

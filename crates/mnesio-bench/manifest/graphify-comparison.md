# mnesio vs graphify — token cost of one coding task

Measured 2026-08-09. One repository, 60 tasks, both tools driven headlessly on
the same working tree. Reproduction steps at the bottom.

## The finding that shapes everything else

**`graphify query` returns no code.** It returns node names with `file:line`
and community labels — verified as zero lines matching a function or class
definition at budgets of 2 000, 8 000 and 16 000 tokens. It tells an agent
*where* to look.

`mnesio_code_context` returns the symbol bodies, packed to a hard ceiling.

So a straight "tokens per query" comparison is not a comparison. graphify's
1 647 tokens are cheaper than mnesio's 4 438 because they are a different and
smaller product: the agent still has to open the files afterwards, and that is
where the context budget actually goes. The number below that matters is
**tokens to have the code in hand**.

## Setup

- **Repository:** `claw-code` — 239 files, mixed Rust/Python/TypeScript. Copied
  to a scratch directory so neither tool wrote into a working repo.
- **Tasks:** 60 real commit subjects from the repo's own history. A human wrote
  each one for other reasons, before either tool existed, so neither can have
  been tuned to them.
- **Gold:** the files each commit actually touched, per `git show --name-only`.
- **Metric:** did the returned context put the agent in front of a gold file.
- **Indexing:** graphify 11 s (5 253 nodes, 16 476 edges). mnesio ~15 s
  (3 500 symbols, 2 670 resolved edges).

**File-level, not symbol-level, and that is a concession.** mnesio's own
`codeeval` scores at symbol level — did the packed context contain the specific
symbol the commit changed — where it gets **59%** on this repo (67% with the
code reranker). graphify returns no symbol bodies, so it cannot be scored that
way at all. File level is the finest question *both* tools can answer, and it
flatters both of them.

## Result

graphify at its default budget of 2 000, by how many of its cited files the
agent then opens:

| read policy | recall | tokens to have the code |
|---|---|---|
| pointers only (no code in hand) | — | 1 647 |
| top-3 cited files | 65% | 100 036 |
| top-5 cited files | 78% | 151 105 |
| all 11 cited files | 88% | 237 625 |

mnesio at its default budget of 4 000, same 60 tasks:

| | recall | tokens to have the code |
|---|---|---|
| `mnesio_code_context` | **97%** | **4 438** |

Medians throughout. Tokens estimated as chars/4, identically for both, so the
ratios hold even though the absolute counts are approximate.

Read across the rows: at the policy most favourable to graphify — open only its
top three files — it reaches 65% for 100 036 tokens, against mnesio's 97% for
4 438. **22× fewer tokens and 32pp more recall.** At the policy least
favourable to it, 54× fewer tokens and 9pp more recall.

## What this does not establish

Five things, and the first two are the ones that would change the number most.

1. **graphify ran AST-only.** Its semantic extraction wants a Gemini or OpenAI
   key, and `--mode deep` extracts more aggressively. Neither was used. This is
   its strongest counterargument and it is untested here.
2. **mnesio referenced 16 files to graphify's 11.** "Did you reference a gold
   file" gets easier the more files you reference, so some unknown part of the
   9pp gap at the all-files policy is breadth, not quality. The gap at top-3
   (32pp) is not explainable this way, because there mnesio's whole answer is
   being compared against graphify's three best files.
3. **One repository, 60 tasks.** `scaleeval` has already shown once that
   single-repo numbers mislead — the ceiling gap it reported went from 11pp to
   33pp when small repositories were excluded from the distribution. This
   number deserves the same treatment before it is quoted anywhere public.
4. **Budgets are different knobs.** graphify's `--budget` caps a pointer list;
   mnesio's caps returned code. Each was run at its own default.
5. **Not paired in the strict sense.** Both tools saw one identical tree and one
   identical task list, which removes corpus variance, but neither was repeated
   to establish a noise floor.

## Correction to an earlier claim

`CLAUDE.md` Phase 18 says of both competitors: *"neither records whether the
retrieval actually helped."* **That is wrong about graphify.** Its CLI has:

```
graphify save-result --outcome useful|dead_end|corrected --correction TEXT
graphify reflect      # aggregate outcomes into a deterministic lessons doc
```

So outcome capture exists there. What does not exist is a gate: `reflect`
writes `LESSONS.md`, a document. Nothing re-runs a canary suite, nothing can
be rejected, and nothing is prevented from regressing. The mnesio claim that
survives is narrower and should be stated that way — *gated* improvement, not
*any* use of outcomes.

## Reproducing

```bash
cp -R <repo> /tmp/cmp && cd /tmp/cmp
uvx --from graphifyy graphify update .          # ~11s, writes graphify-out/
python3 measure.py /tmp/cmp 2000 60             # graphify arm
python3 measure_mnesio.py /tmp/cmp 4000 60 \
  target/release/mnesio-mcp                     # mnesio arm
```

Both scripts derive their own tasks and gold from `git log` in the repo they
are pointed at, so there is no fixture to drift out of date.

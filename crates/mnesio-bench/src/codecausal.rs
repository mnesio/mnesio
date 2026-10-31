//! Phase 10 × Phase 18: counterfactual contribution over **code** retrieval.
//!
//! ## Why this exists
//!
//! Every observational route to the wedge has been measured and has failed, and
//! each failed differently (`manifest/wedge-credit-assignment.md`):
//!
//! - suppression never fires — at 88% baseline recall no symbol is
//!   consistently harmful;
//! - promotion by success rate selects the *commonest* terms and moves held-out
//!   recall +0.0pp;
//! - promotion by lift over the base rate does not bind at these rates;
//! - promotion by binomial significance correctly refuses five of six rules,
//!   and the survivor is still a frequency artifact.
//!
//! The common cause is credit assignment. An outcome is one bit shared across
//! the ~20 symbols packed into that context, and **nothing in the data varies
//! one symbol while holding the rest fixed**. No statistic over observations
//! can fix that, because the confound is in how the data was generated.
//!
//! Masking is the intervention that breaks it: re-run the same task with one
//! symbol removed and measure whether the answer changes. That is a
//! counterfactual, not a correlation, and it is exactly what
//! [`mnesio_causal`] already does for prose memories.
//!
//! ## What this module is
//!
//! The adapter, and only the adapter. [`CodeCounterfactual`] implements
//! [`CounterfactualEvaluator`] over a code suite: `evaluate(masked)` is recall
//! across the tasks with those symbols suppressed. The scoring, the bounds and
//! the report all come from `mnesio-causal` unchanged — this crate contributes
//! the objective, not the method.
//!
//! ## What it costs, and why that is acceptable here
//!
//! Leave-one-out issues `1 + candidates` evaluations, and each evaluation is a
//! full pass over the task set. On a module with 24 tasks and 300 candidate
//! symbols that is ~7 200 retrievals. It is offline by construction and never
//! on the write path (Hard Rule #5), and `max_candidates` bounds the pass
//! (Hard Rule #6). Cheap it is not; the point is that it answers a question
//! nothing cheaper can.

use std::collections::HashSet;

use async_trait::async_trait;

use mnesio_causal::{
    CausalConfig, ContributionReport, ContributionScorer, CounterfactualEvaluator, ScoreMode,
};
use mnesio_core::types::MemoryRef;
use mnesio_core::MnesioError;

use crate::codeeval::CodeQuery;
use crate::learncurve::{CurveIndex, Policy};

/// Recall over a fixed code suite, with symbols maskable.
pub struct CodeCounterfactual<'a, I: CurveIndex> {
    index: &'a I,
    tasks: Vec<&'a CodeQuery>,
}

impl<'a, I: CurveIndex> CodeCounterfactual<'a, I> {
    /// `tasks` is the set the objective is measured over. It should be the
    /// *training* split: contribution measured on held-out would leak the
    /// answer into the rules learned from it.
    pub fn new(index: &'a I, tasks: Vec<&'a CodeQuery>) -> Self {
        Self { index, tasks }
    }
}

#[async_trait]
impl<I: CurveIndex + Sync> CounterfactualEvaluator for CodeCounterfactual<'_, I> {
    async fn evaluate(&self, masked: &HashSet<MemoryRef>) -> Result<f32, MnesioError> {
        let per = self.hits_per_task(masked).await?;
        if per.is_empty() {
            return Ok(0.0);
        }
        Ok(per.iter().sum::<f32>() / per.len() as f32)
    }

    /// Recall is a mean over per-task hits, so the per-task vector is what this
    /// evaluator computes natively — the aggregate is the lossy view, not the
    /// other way round.
    ///
    /// This is the whole reason the wedge was stuck. `harmful` is 0 under both
    /// ablation modes because "did the context contain a gold symbol" can
    /// essentially never be *improved* by removing a symbol, so suppression has
    /// no target. The remaining lever is promotion from the load-bearing
    /// symbols, and a promotion rule has to be scoped to the query class it
    /// helped — which needs to know *which tasks* those were. The aggregate had
    /// that information one line before it divided it away.
    async fn evaluate_per_task(
        &self,
        masked: &HashSet<MemoryRef>,
    ) -> Result<Option<Vec<f32>>, MnesioError> {
        if self.tasks.is_empty() {
            return Ok(None);
        }
        Ok(Some(self.hits_per_task(masked).await?))
    }
}

impl<I: CurveIndex + Sync> CodeCounterfactual<'_, I> {
    /// One score per task, in `self.tasks` order — which is a `Vec` and is never
    /// reordered, satisfying the evaluator's stable-order contract.
    async fn hits_per_task(&self, masked: &HashSet<MemoryRef>) -> Result<Vec<f32>, MnesioError> {
        let policy = Policy {
            suppressed: masked.clone(),
            boosts: Vec::new(),
        };
        let mut per = Vec::with_capacity(self.tasks.len());
        for q in &self.tasks {
            // A retrieval failure here is a real failure, not a zero score.
            // Swallowing it as 0.0 would silently turn an outage into "every
            // masked symbol was load-bearing", which is the most misleading
            // possible reading of a contribution pass.
            let scored = self
                .index
                .run(q, &policy)
                .await
                .map_err(MnesioError::Other)?;
            per.push(if scored.hit { 1.0 } else { 0.0 });
        }
        Ok(per)
    }
}

/// What a contribution pass found, in the terms the wedge needs.
#[derive(Debug, Clone)]
pub struct CodeContribution {
    pub report: ContributionReport,
    /// Symbols whose removal measurably *lowered* recall — the ones that
    /// actually carried a task.
    pub load_bearing: usize,
    /// Symbols whose removal changed nothing. Under leave-one-out this
    /// conflates "useless" with "redundant", which is why the mode is
    /// reported alongside.
    pub inert: usize,
    /// Symbols whose removal *raised* recall. Genuinely harmful, and the
    /// honest target for a suppression rule — unlike the correlational
    /// version, which found none at all.
    pub harmful: usize,
}

impl CodeContribution {
    /// Fraction of scored symbols that carried anything.
    pub fn load_bearing_share(&self) -> f32 {
        match self.report.candidates_scored {
            0 => 0.0,
            n => self.load_bearing as f32 / n as f32,
        }
    }
}

/// Run one bounded contribution pass over a code suite.
pub async fn run_code_causal<I: CurveIndex + Sync>(
    index: &I,
    tasks: Vec<&CodeQuery>,
    candidates: &[MemoryRef],
    cfg: CausalConfig,
) -> Result<CodeContribution, MnesioError> {
    let eval = CodeCounterfactual::new(index, tasks);
    let report = ContributionScorer::new(cfg.clone())
        .score(&eval, candidates)
        .await?;

    let eps = cfg.epsilon;
    let load_bearing = report
        .scored
        .iter()
        .filter(|c| c.contribution > eps)
        .count();
    let harmful = report
        .scored
        .iter()
        .filter(|c| c.contribution < -eps)
        .count();
    Ok(CodeContribution {
        inert: report.candidates_scored - load_bearing - harmful,
        load_bearing,
        harmful,
        report,
    })
}

/// Render a pass for a human, including the reading that would be wrong.
pub fn format_contribution(c: &CodeContribution) -> String {
    let r = &c.report;
    let mut out = format!(
        "# causal contribution over code retrieval\n\n\
         baseline recall {:.0}% · {} candidates scored of {} considered · mode {:?}\n\n\
         | class | symbols | share |\n|---|---|---|\n\
         | load-bearing (removal lowered recall) | {} | {:.0}% |\n\
         | inert (no measurable effect) | {} | {:.0}% |\n\
         | harmful (removal *raised* recall) | {} | {:.0}% |\n\n",
        r.baseline_score * 100.0,
        r.candidates_scored,
        r.candidates_considered,
        r.mode,
        c.load_bearing,
        c.load_bearing_share() * 100.0,
        c.inert,
        100.0 * c.inert as f32 / r.candidates_scored.max(1) as f32,
        c.harmful,
        100.0 * c.harmful as f32 / r.candidates_scored.max(1) as f32,
    );

    // ----- how *scoped* is the credit? -----
    //
    // Aggregate contribution says a symbol helped; it cannot say on which
    // tasks, and a promotion rule that is not scoped to a query class is a
    // global boost — the thing `CODE_BOOST` already does better. So before
    // promotion is worth building, this has to show that the credit actually
    // concentrates. Printed here rather than inferred later, per discipline
    // rule 7: size the bucket before building the fix.
    if let Some(by_task) = r.decisive_by_task() {
        let tasks = r.task_count.unwrap_or(0);
        let covered = by_task.len();
        let mut widths: Vec<usize> = r
            .scored
            .iter()
            .filter(|c| !c.decisive_for.is_empty())
            .map(|c| c.decisive_for.len())
            .collect();
        widths.sort_unstable();
        let single = widths.iter().filter(|w| **w == 1).count();
        let median = widths.get(widths.len() / 2).copied().unwrap_or(0);
        let per_task: Vec<usize> = by_task.values().map(|v| v.len()).collect();
        let crowded = per_task.iter().filter(|n| **n > 3).count();
        out.push_str(&format!(
            "## Is the credit scoped enough to promote?\n\n| question | answer |\n|---|---|\n| tasks with any decisive symbol | {covered} of {tasks} ({:.0}%) |\n| decisive symbols | {} |\n| decisive for exactly one task | {single} ({:.0}%) |\n| median tasks per decisive symbol | {median} |\n| tasks with >3 decisive symbols | {crowded} |\n\n",
            100.0 * covered as f32 / tasks.max(1) as f32,
            widths.len(),
            100.0 * single as f32 / widths.len().max(1) as f32,
        ));
        if covered == 0 {
            out.push_str(
                "**No task has a decisive symbol**, so promotion has no target \
                 either — the same wall suppression hit, from the other side. \
                 Redundant context means no single symbol is what carried a \
                 task.\n\n",
            );
        }
    } else {
        out.push_str(
            "Per-task resolution was unavailable this pass, so scoping cannot \
             be assessed — this is *not* the same as finding no scoped \
             credit.\n\n",
        );
    }

    let mut top = r.ranked();
    top.truncate(10);
    if top.iter().any(|m| m.contribution.abs() > cfgeps(r)) {
        out.push_str("## Highest measured contribution\n\n| symbol | contribution |\n|---|---|\n");
        for m in &top {
            out.push_str(&format!("| `{}` | {:+.3} |\n", m.memory.0, m.contribution));
        }
        out.push('\n');
    }

    if c.load_bearing == 0 {
        out.push_str(
            "**Nothing was load-bearing.** Every symbol could be removed \
             without changing recall, which under leave-one-out means the \
             context is *redundant*: some other packed symbol covered each \
             task. That is a real property of retrieval at this k, not a \
             failure of the measurement — but it does mean per-symbol credit \
             is unavailable by masking one at a time, and `GreedyAblation` is \
             the mode that recovers redundant-set contribution.\n",
        );
    } else {
        out.push_str(
            "Contribution here is **causal** in the sense that matters: each \
             number is the measured effect of removing that symbol and \
             re-running, not a correlation between its presence and success. \
             That is the distinction every observational attempt failed on.\n",
        );
    }
    out
}

/// Epsilon back out of the report, so the renderer does not need the config.
fn cfgeps(_r: &ContributionReport) -> f32 {
    // The engine already applied its own epsilon when classifying; this is only
    // the display threshold for "worth listing at all".
    0.0
}

/// Default bounds for a code pass.
///
/// Leave-one-out rather than greedy: greedy is `O(n²)` evaluations and each
/// evaluation is a full suite pass, which on a 300-symbol module is ~90 000
/// retrievals. Start with the cheap mode, and only pay for greedy once
/// leave-one-out has shown whether redundancy is actually the problem.
pub fn default_code_causal_config() -> CausalConfig {
    code_causal_config_for(24)
}

/// Bounds for a contribution pass over a suite of `tasks`.
///
/// **`epsilon` has to scale with the suite, and a fixed one silently hid every
/// result this harness produced.** Contribution is a change in suite recall, so
/// the smallest effect that can exist is one task flipping — `1/tasks`. The
/// previous fixed `0.02` was derived from a 24-task suite, where one flip is
/// 0.042 and half of that is a sensible noise floor. Run against 57 tasks, one
/// flip is 0.0175, *below* the threshold, so every single-task effect was
/// classified `inert` by arithmetic rather than by measurement — which is
/// exactly what a flask run reported: 0 harmful, 299 of 300 inert, under both
/// leave-one-out and greedy ablation.
///
/// So the floor is half of one task flip. Small enough that a real single-task
/// effect is visible, large enough that float noise is not.
pub fn code_causal_config_for(tasks: usize) -> CausalConfig {
    CausalConfig {
        max_candidates: 300,
        epsilon: 0.5 / tasks.max(1) as f32,
        mode: ScoreMode::LeaveOneOut,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codeeval::{CodeQuery, Gold};
    use crate::learncurve::Scored;
    use anyhow::Result;
    use mnesio_core::types::new_id;

    /// An index where exactly one symbol answers each task, so contribution
    /// has a known right answer to be checked against.
    struct OneAnswerEach {
        answers: Vec<MemoryRef>,
        noise: Vec<MemoryRef>,
    }

    impl CurveIndex for OneAnswerEach {
        fn run(
            &self,
            q: &CodeQuery,
            policy: &Policy,
        ) -> impl std::future::Future<Output = Result<Scored>> + Send {
            // Task i is answered by answers[i]; every task also packs all the
            // noise, which is what makes this a credit-assignment problem
            // rather than a lookup.
            let i: usize = q.question.parse().unwrap_or(0);
            let answer = self.answers[i % self.answers.len()];
            let mut symbols = self.noise.clone();
            let hit = !policy.suppressed.contains(&answer);
            if hit {
                symbols.push(answer);
            }
            async move { Ok(Scored { hit, symbols }) }
        }
    }

    fn task(i: usize) -> CodeQuery {
        CodeQuery {
            question: i.to_string(),
            gold: vec![Gold {
                path: Some("x.rs".into()),
                name: "x".into(),
            }],
        }
    }

    #[tokio::test]
    async fn masking_finds_the_symbol_that_actually_carried_the_task() {
        // The whole point of Phase 10 here: correlation cannot separate the
        // answer from the noise, because both are present on every success.
        // Masking can, because it varies one and holds the rest fixed.
        let answers: Vec<_> = (0..4).map(|_| MemoryRef(new_id())).collect();
        let noise: Vec<_> = (0..6).map(|_| MemoryRef(new_id())).collect();
        let index = OneAnswerEach {
            answers: answers.clone(),
            noise: noise.clone(),
        };
        let tasks: Vec<CodeQuery> = (0..4).map(task).collect();
        let refs: Vec<&CodeQuery> = tasks.iter().collect();

        let mut candidates = answers.clone();
        candidates.extend(noise.iter().copied());
        let c = run_code_causal(
            &index,
            refs,
            &candidates,
            CausalConfig {
                max_candidates: 32,
                epsilon: 0.01,
                mode: ScoreMode::LeaveOneOut,
            },
        )
        .await
        .unwrap();

        assert_eq!(c.report.baseline_score, 1.0, "every task answerable");
        assert_eq!(c.load_bearing, 4, "one carrier per task, and only those");
        assert_eq!(c.inert, 6, "the noise carried nothing");
        assert_eq!(c.harmful, 0);

        // And the carriers are the right ones, not merely the right count.
        for a in &answers {
            let scored = c.report.scored.iter().find(|s| s.memory == *a).unwrap();
            assert!(scored.contribution > 0.0, "an answer must contribute");
        }
        for n in &noise {
            let scored = c.report.scored.iter().find(|s| s.memory == *n).unwrap();
            assert_eq!(scored.contribution, 0.0, "noise must not");
        }
    }

    #[tokio::test]
    async fn an_empty_task_set_scores_zero_rather_than_dividing_by_it() {
        let index = OneAnswerEach {
            answers: vec![MemoryRef(new_id())],
            noise: vec![],
        };
        let e = CodeCounterfactual::new(&index, Vec::new());
        assert_eq!(e.evaluate(&HashSet::new()).await.unwrap(), 0.0);
    }
}

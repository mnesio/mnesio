//! Counterfactual contribution scoring by leave-one-out (LOO) ablation.
//!
//! The one primitive is [`CounterfactualEvaluator::evaluate`]: it returns an
//! aggregate objective in `[0,1]` when a given set of memories is *masked*
//! (made invisible to retrieval). The baseline is `evaluate(&{})`. A memory's
//! contribution is `baseline − evaluate(&{memory})`:
//!
//! - **> 0** — removing it *drops* outcomes; load-bearing, keep.
//! - **≈ 0** — removing it changes nothing; dead weight, GC candidate.
//! - **< 0** — removing it *improves* outcomes; actively misleading, also a GC
//!   (and a Phase-14 "negative memory" suppression) candidate.

use async_trait::async_trait;
use mnesio_core::types::MemoryRef;
use mnesio_core::MnesioError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

/// The seam (Hard Rule #7). The real impl wires the retriever + executor over
/// a held-out outcome/task set and masks memories by id; [`FakeEvaluator`]
/// makes the engine's tests hermetic.
#[async_trait]
pub trait CounterfactualEvaluator: Send + Sync {
    /// Aggregate objective in `[0,1]` (recall@k, benchmark pass-rate, …) with
    /// `masked` memories excluded from retrieval. `evaluate(&{})` is the
    /// baseline against which contributions are measured.
    async fn evaluate(&self, masked: &HashSet<MemoryRef>) -> Result<f32, MnesioError>;

    /// The same objective, *not yet aggregated*: one score per task, or `None`
    /// when this evaluator has no per-task resolution.
    ///
    /// This exists because the aggregate is a dead end for rule *scoping*.
    /// Phase 18C–E needs to know **which tasks** a symbol carried in order to
    /// scope a promotion rule to a query class; an `f32` mean cannot answer
    /// that, and the evaluator that produced it usually had the per-task vector
    /// in hand one line earlier and divided it away.
    ///
    /// ## Contract
    ///
    /// Entry `i` must refer to the **same task on every call**, for the whole
    /// life of the evaluator. Per-task contribution is computed by subtracting
    /// two of these vectors elementwise, so an unstable order silently
    /// attributes one task's delta to another. Implementors iterating a `Vec`
    /// satisfy this for free; one iterating a `HashMap` does not.
    ///
    /// The default returns `None` rather than `vec![aggregate]`: a
    /// single-element vector is indistinguishable from a genuine one-task
    /// suite, and would make every memory look "decisive for task 0".
    async fn evaluate_per_task(
        &self,
        _masked: &HashSet<MemoryRef>,
    ) -> Result<Option<Vec<f32>>, MnesioError> {
        Ok(None)
    }
}

/// How contribution is attributed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreMode {
    /// Leave-one-out: mask exactly one memory at a time. Cheap (`O(corpus)`
    /// evals) but underestimates contribution under redundancy — two memories
    /// that cover the same fact each score ≈ 0 because the other still covers.
    LeaveOneOut,
    /// Greedy ablation: iteratively ablate the **least-valuable remaining**
    /// memory (the one whose removal leaves the highest score) and re-score,
    /// attributing each memory its *marginal* drop at the point it's removed.
    ///
    /// This recovers redundant-set contribution that LOO misses: once the
    /// first of a redundant pair is masked, the second becomes load-bearing
    /// and earns the set's full contribution at its removal step. It's a
    /// single greedy permutation of the Shapley value — exact on additive
    /// objectives, and on the *aggregate* of a redundant set, at the cost of
    /// `O(corpus²)` evals (still bounded by `max_candidates`, Hard Rule #6).
    GreedyAblation,
}

/// Bounds for one scoring pass. The engine never schedules itself; the caller
/// runs passes offline, off the write path (Hard Rules #5/#6).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalConfig {
    /// Hard cap on candidates scored per pass — the cascade bound (#6).
    pub max_candidates: usize,
    /// Contributions with magnitude `≤ epsilon` count as zero (float noise +
    /// "no measurable effect" tolerance). Drives `gc_candidates`.
    pub epsilon: f32,
    pub mode: ScoreMode,
}

impl Default for CausalConfig {
    fn default() -> Self {
        Self {
            max_candidates: 256,
            epsilon: 1e-4,
            mode: ScoreMode::LeaveOneOut,
        }
    }
}

/// One memory's measured contribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryContribution {
    pub memory: MemoryRef,
    /// `baseline − masked_score`. Positive = helpful, ~0 = inert, negative =
    /// harmful.
    pub contribution: f32,
    /// The objective when this memory alone was masked.
    pub masked_score: f32,
    /// Indices of the tasks whose score *dropped* when this memory was masked —
    /// the tasks it was actually decisive for, and the join key a scoped
    /// promotion rule is built from.
    ///
    /// **Empty is ambiguous on its own.** It means "decisive for no task" only
    /// when [`ContributionReport::per_task_resolved`] is true; otherwise it
    /// means the evaluator never supplied per-task scores. Check the report
    /// before reading anything into an empty vector.
    #[serde(default)]
    pub decisive_for: Vec<usize>,
}

/// The result of one scoring pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionReport {
    /// `evaluate(&{})` — the objective with nothing masked.
    pub baseline_score: f32,
    pub mode: ScoreMode,
    /// One entry per scored candidate, in input order.
    pub scored: Vec<MemoryContribution>,
    /// How many candidates the caller supplied (may exceed `candidates_scored`
    /// when `max_candidates` clamps the pass).
    pub candidates_considered: usize,
    /// How many were actually scored this pass.
    pub candidates_scored: usize,
    /// Whether the evaluator supplied per-task scores this pass.
    ///
    /// When false, every `decisive_for` is empty *because the information was
    /// unavailable* — not because nothing was decisive. Keeping these two cases
    /// apart is the whole reason this flag exists rather than letting callers
    /// infer it from empty vectors.
    #[serde(default)]
    pub per_task_resolved: bool,
    /// How many tasks the objective spans, when the evaluator reports it.
    #[serde(default)]
    pub task_count: Option<usize>,
}

impl ContributionReport {
    /// Memories safe to retire: contribution at or below `epsilon` (inert *or*
    /// harmful). These are the *provable* zero/negative contributors — GC by
    /// measurement, not by age.
    pub fn gc_candidates(&self, epsilon: f32) -> Vec<MemoryRef> {
        self.scored
            .iter()
            .filter(|c| c.contribution <= epsilon)
            .map(|c| c.memory)
            .collect()
    }

    /// Load-bearing memories: contribution at or above `min_contribution`.
    pub fn high_contributors(&self, min_contribution: f32) -> Vec<&MemoryContribution> {
        self.scored
            .iter()
            .filter(|c| c.contribution >= min_contribution)
            .collect()
    }

    /// Load-bearing memories grouped by the task they carried: `task index →
    /// the memories decisive for it`.
    ///
    /// This is the shape Phase 18C–E consumes. A promotion rule has to be
    /// *scoped* — "boost this symbol" is only safe for the query class where it
    /// actually helped, and an aggregate contribution cannot say which class
    /// that is. Tasks no memory was decisive for are absent rather than
    /// present-and-empty.
    ///
    /// Returns `None` when the pass had no per-task resolution, so a caller
    /// cannot mistake "nothing to promote" for "never measured".
    pub fn decisive_by_task(&self) -> Option<HashMap<usize, Vec<MemoryRef>>> {
        if !self.per_task_resolved {
            return None;
        }
        let mut by: HashMap<usize, Vec<MemoryRef>> = HashMap::new();
        for c in &self.scored {
            for &t in &c.decisive_for {
                by.entry(t).or_default().push(c.memory);
            }
        }
        Some(by)
    }

    /// A copy of the scores sorted by contribution, highest first — for the
    /// dashboard panel ("what's actually carrying the corpus").
    pub fn ranked(&self) -> Vec<MemoryContribution> {
        let mut v = self.scored.clone();
        v.sort_by(|a, b| {
            b.contribution
                .partial_cmp(&a.contribution)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }
}

/// One evaluator round-trip, yielding the aggregate and — when the evaluator
/// has per-task resolution — the vector it was aggregated from.
///
/// Per-task is asked for *first* and the aggregate derived from it, so an
/// evaluator that supplies both costs one pass rather than two. An evaluator
/// without per-task resolution falls through to `evaluate` and its call count
/// is unchanged, which is what keeps the cascade bound (#6) honest.
async fn probe(
    evaluator: &dyn CounterfactualEvaluator,
    masked: &HashSet<MemoryRef>,
) -> Result<(f32, Option<Vec<f32>>), MnesioError> {
    match evaluator.evaluate_per_task(masked).await? {
        Some(tasks) if !tasks.is_empty() => {
            let mean = tasks.iter().sum::<f32>() / tasks.len() as f32;
            Ok((mean, Some(tasks)))
        }
        // An empty task vector is an empty suite, not per-task resolution.
        _ => Ok((evaluator.evaluate(masked).await?, None)),
    }
}

/// Which tasks got *worse* between the baseline vector and a masked one.
///
/// A length mismatch is a violation of the [`CounterfactualEvaluator`] ordering
/// contract, and it is an error rather than an empty result on purpose:
/// returning `Vec::new()` here would render as "decisive for no task", which is
/// precisely the reading `per_task_resolved` exists to prevent.
fn decisive_tasks(baseline: &[f32], masked: &[f32]) -> Result<Vec<usize>, MnesioError> {
    if baseline.len() != masked.len() {
        return Err(MnesioError::Index(format!(
            "evaluator returned {} per-task scores for the baseline and {} for a \
             masked pass; entry i must refer to the same task on every call",
            baseline.len(),
            masked.len()
        )));
    }
    Ok(baseline
        .iter()
        .zip(masked)
        .enumerate()
        .filter(|(_, (b, m))| *m < *b)
        .map(|(i, _)| i)
        .collect())
}

/// Leave-one-out contribution scorer.
pub struct ContributionScorer {
    cfg: CausalConfig,
}

impl ContributionScorer {
    pub fn new(cfg: CausalConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &CausalConfig {
        &self.cfg
    }

    /// Score `candidates` against `evaluator`, dispatching on the configured
    /// [`ScoreMode`]. The caller is responsible for scoping `candidates`
    /// (Hard Rule #3) — the scorer is scope-agnostic.
    pub async fn score(
        &self,
        evaluator: &dyn CounterfactualEvaluator,
        candidates: &[MemoryRef],
    ) -> Result<ContributionReport, MnesioError> {
        match self.cfg.mode {
            ScoreMode::LeaveOneOut => self.score_loo(evaluator, candidates).await,
            ScoreMode::GreedyAblation => self.score_greedy(evaluator, candidates).await,
        }
    }

    /// Leave-one-out: one baseline + one masked eval per candidate. Issues
    /// exactly `1 + min(candidates.len(), max_candidates)` evaluator calls.
    async fn score_loo(
        &self,
        evaluator: &dyn CounterfactualEvaluator,
        candidates: &[MemoryRef],
    ) -> Result<ContributionReport, MnesioError> {
        let (baseline, baseline_tasks) = probe(evaluator, &HashSet::new()).await?;
        let mut scored = Vec::new();
        for memory in candidates.iter().take(self.cfg.max_candidates) {
            let mut masked = HashSet::with_capacity(1);
            masked.insert(*memory);
            let (masked_score, masked_tasks) = probe(evaluator, &masked).await?;
            let decisive_for = match (&baseline_tasks, &masked_tasks) {
                (Some(b), Some(m)) => decisive_tasks(b, m)?,
                _ => Vec::new(),
            };
            scored.push(MemoryContribution {
                memory: *memory,
                contribution: baseline - masked_score,
                masked_score,
                decisive_for,
            });
        }
        let candidates_scored = scored.len();
        Ok(ContributionReport {
            baseline_score: baseline,
            mode: self.cfg.mode,
            scored,
            candidates_considered: candidates.len(),
            candidates_scored,
            per_task_resolved: baseline_tasks.is_some(),
            task_count: baseline_tasks.as_ref().map(|t| t.len()),
        })
    }

    /// Greedy ablation: repeatedly ablate the least-valuable remaining memory
    /// (highest score when added to the masked set) and attribute it the
    /// marginal drop at that step. Accumulates the masked set across steps so
    /// redundant memories earn the set's contribution once the cover is gone.
    ///
    /// `scored` is returned in **ablation order** (least-contribution first);
    /// `masked_score` is the cumulative score after that memory was ablated.
    /// Cost: `1 + N(N+1)/2` evals for `N = min(candidates, max_candidates)`.
    async fn score_greedy(
        &self,
        evaluator: &dyn CounterfactualEvaluator,
        candidates: &[MemoryRef],
    ) -> Result<ContributionReport, MnesioError> {
        let (baseline, baseline_tasks) = probe(evaluator, &HashSet::new()).await?;
        let per_task_resolved = baseline_tasks.is_some();
        let task_count = baseline_tasks.as_ref().map(|t| t.len());
        let mut remaining: Vec<MemoryRef> = candidates
            .iter()
            .take(self.cfg.max_candidates)
            .copied()
            .collect();
        let mut masked: HashSet<MemoryRef> = HashSet::with_capacity(remaining.len());
        let mut current = baseline;
        // The per-task state *before* the next ablation. Greedy attributes each
        // memory its marginal at the point it is removed, so `decisive_for` is
        // measured against this rolling vector rather than the pass baseline —
        // the same relationship `contribution = current - best_score` has.
        let mut current_tasks = baseline_tasks;
        let mut scored: Vec<MemoryContribution> = Vec::with_capacity(remaining.len());

        while !remaining.is_empty() {
            // Pick the memory whose ablation leaves the highest score — i.e.
            // the least-valuable one to remove next.
            let mut best_idx = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_tasks: Option<Vec<f32>> = None;
            for (i, m) in remaining.iter().enumerate() {
                masked.insert(*m);
                let (trial, trial_tasks) = probe(evaluator, &masked).await?;
                masked.remove(m);
                if trial > best_score {
                    best_score = trial;
                    best_idx = i;
                    best_tasks = trial_tasks;
                }
            }
            let chosen = remaining.swap_remove(best_idx);
            let contribution = current - best_score;
            let decisive_for = match (&current_tasks, &best_tasks) {
                (Some(c), Some(b)) => decisive_tasks(c, b)?,
                _ => Vec::new(),
            };
            masked.insert(chosen);
            current = best_score;
            current_tasks = best_tasks;
            scored.push(MemoryContribution {
                memory: chosen,
                contribution,
                masked_score: best_score,
                decisive_for,
            });
        }
        let candidates_scored = scored.len();
        Ok(ContributionReport {
            baseline_score: baseline,
            mode: self.cfg.mode,
            scored,
            candidates_considered: candidates.len(),
            candidates_scored,
            per_task_resolved,
            task_count,
        })
    }
}

/// Deterministic, dependency-free evaluator for tests + offline modelling.
///
/// Models an **additive** objective: `evaluate(masked) = baseline − Σ
/// marginal(m)` clamped to `[0,1]`. Under this model LOO recovers each memory's
/// marginal exactly, which is what lets the tests assert the scorer's contract
/// precisely. Real corpora are non-additive (the redundancy gap), which is why
/// the production evaluator wires the actual retriever.
pub struct FakeEvaluator {
    baseline: f32,
    marginals: HashMap<MemoryRef, f32>,
    calls: AtomicUsize,
}

impl FakeEvaluator {
    pub fn new(baseline: f32) -> Self {
        Self {
            baseline,
            marginals: HashMap::new(),
            calls: AtomicUsize::new(0),
        }
    }

    /// Set how much the objective drops when `memory` is present and then
    /// masked. Positive = helpful memory; `0.0` = inert; negative = harmful.
    pub fn with_marginal(mut self, memory: MemoryRef, marginal: f32) -> Self {
        self.marginals.insert(memory, marginal);
        self
    }

    /// Number of `evaluate` calls so far — lets tests assert the cascade bound.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl CounterfactualEvaluator for FakeEvaluator {
    async fn evaluate(&self, masked: &HashSet<MemoryRef>) -> Result<f32, MnesioError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let removed: f32 = masked.iter().filter_map(|m| self.marginals.get(m)).sum();
        Ok((self.baseline - removed).clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::types::new_id;

    fn mref() -> MemoryRef {
        MemoryRef(new_id())
    }

    fn scorer() -> ContributionScorer {
        ContributionScorer::new(CausalConfig::default())
    }

    #[tokio::test]
    async fn loo_recovers_per_memory_marginals() {
        let (helpful, inert, harmful) = (mref(), mref(), mref());
        let eval = FakeEvaluator::new(0.8)
            .with_marginal(helpful, 0.4)
            .with_marginal(inert, 0.0)
            .with_marginal(harmful, -0.2);
        let report = scorer()
            .score(&eval, &[helpful, inert, harmful])
            .await
            .unwrap();

        assert!((report.baseline_score - 0.8).abs() < 1e-6);
        let by = |m: MemoryRef| {
            report
                .scored
                .iter()
                .find(|c| c.memory == m)
                .unwrap()
                .contribution
        };
        assert!((by(helpful) - 0.4).abs() < 1e-6, "helpful ≈ +0.4");
        assert!(by(inert).abs() < 1e-6, "inert ≈ 0");
        assert!((by(harmful) + 0.2).abs() < 1e-6, "harmful ≈ -0.2");
    }

    #[tokio::test]
    async fn gc_candidates_pick_inert_and_harmful_not_load_bearing() {
        let (helpful, inert, harmful) = (mref(), mref(), mref());
        let eval = FakeEvaluator::new(0.8)
            .with_marginal(helpful, 0.4)
            .with_marginal(inert, 0.0)
            .with_marginal(harmful, -0.2);
        let report = scorer()
            .score(&eval, &[helpful, inert, harmful])
            .await
            .unwrap();

        let gc = report.gc_candidates(scorer().config().epsilon);
        assert!(gc.contains(&inert), "inert is dead weight → GC");
        assert!(gc.contains(&harmful), "harmful improves on removal → GC");
        assert!(!gc.contains(&helpful), "load-bearing memory must be kept");

        let keep = report.high_contributors(0.3);
        assert_eq!(keep.len(), 1);
        assert_eq!(keep[0].memory, helpful);
    }

    #[tokio::test]
    async fn max_candidates_bounds_the_pass() {
        let cands: Vec<MemoryRef> = (0..5).map(|_| mref()).collect();
        let eval = FakeEvaluator::new(0.5);
        let cfg = CausalConfig {
            max_candidates: 2,
            ..CausalConfig::default()
        };
        let report = ContributionScorer::new(cfg)
            .score(&eval, &cands)
            .await
            .unwrap();

        assert_eq!(report.candidates_considered, 5);
        assert_eq!(report.candidates_scored, 2);
        // 1 baseline + 2 masked evals — the cascade bound (#6) held.
        assert_eq!(eval.calls(), 3);
    }

    // The two halves of the Phase-10 "done when", at the evaluator level:
    #[tokio::test]
    async fn done_when_zero_contributor_masking_does_not_move_the_score() {
        let inert = mref();
        let eval = FakeEvaluator::new(0.7).with_marginal(inert, 0.0);
        let baseline = eval.evaluate(&HashSet::new()).await.unwrap();
        let masked = eval.evaluate(&HashSet::from([inert])).await.unwrap();
        assert!(
            (baseline - masked).abs() < 1e-6,
            "pruning inert moves nothing"
        );
    }

    #[tokio::test]
    async fn done_when_high_contributor_removal_drops_outcomes() {
        let load_bearing = mref();
        let eval = FakeEvaluator::new(0.9).with_marginal(load_bearing, 0.5);
        let baseline = eval.evaluate(&HashSet::new()).await.unwrap();
        let masked = eval.evaluate(&HashSet::from([load_bearing])).await.unwrap();
        assert!(
            masked < baseline - 0.4,
            "removing it measurably drops the score"
        );
    }

    #[tokio::test]
    async fn ranked_orders_by_contribution_desc() {
        let (a, b, c) = (mref(), mref(), mref());
        let eval = FakeEvaluator::new(0.6)
            .with_marginal(a, 0.1)
            .with_marginal(b, 0.5)
            .with_marginal(c, 0.3);
        let report = scorer().score(&eval, &[a, b, c]).await.unwrap();
        let ranked = report.ranked();
        assert_eq!(ranked[0].memory, b);
        assert_eq!(ranked[1].memory, c);
        assert_eq!(ranked[2].memory, a);
    }

    // --- greedy ablation -------------------------------------------------

    /// A *redundant* capability: the objective gains `group_value` iff at least
    /// one group member is unmasked. Masking a strict subset loses nothing;
    /// masking the whole group loses `group_value`. This is the non-additive
    /// case LOO can't score (each member ≈ 0) but greedy ablation can.
    struct CoverageEvaluator {
        baseline: f32,
        group: HashSet<MemoryRef>,
        group_value: f32,
    }

    impl CoverageEvaluator {
        fn new(baseline: f32, group: &[MemoryRef], group_value: f32) -> Self {
            Self {
                baseline,
                group: group.iter().copied().collect(),
                group_value,
            }
        }
    }

    #[async_trait]
    impl CounterfactualEvaluator for CoverageEvaluator {
        async fn evaluate(&self, masked: &HashSet<MemoryRef>) -> Result<f32, MnesioError> {
            let cover_gone =
                !self.group.is_empty() && self.group.iter().all(|m| masked.contains(m));
            let lost = if cover_gone { self.group_value } else { 0.0 };
            Ok((self.baseline - lost).clamp(0.0, 1.0))
        }
    }

    fn greedy_scorer() -> ContributionScorer {
        ContributionScorer::new(CausalConfig {
            mode: ScoreMode::GreedyAblation,
            ..CausalConfig::default()
        })
    }

    #[tokio::test]
    async fn greedy_recovers_redundant_set_contribution_loo_misses() {
        let (a, b) = (mref(), mref());

        // LOO: each member scores ≈ 0 because the other still covers → the
        // set's real 0.4 contribution is missed entirely.
        let loo = scorer()
            .score(&CoverageEvaluator::new(0.8, &[a, b], 0.4), &[a, b])
            .await
            .unwrap();
        let loo_sum: f32 = loo.scored.iter().map(|c| c.contribution).sum();
        assert!(
            loo_sum.abs() < 1e-6,
            "LOO misses redundancy (≈0), got {loo_sum}"
        );

        // Greedy: recovers the full 0.4 (attributed to the last-removed member,
        // once the cover is gone).
        let greedy = greedy_scorer()
            .score(&CoverageEvaluator::new(0.8, &[a, b], 0.4), &[a, b])
            .await
            .unwrap();
        let greedy_sum: f32 = greedy.scored.iter().map(|c| c.contribution).sum();
        assert!(
            (greedy_sum - 0.4).abs() < 1e-6,
            "greedy recovers the set's 0.4, got {greedy_sum}"
        );
        assert_eq!(greedy.mode, ScoreMode::GreedyAblation);
    }

    #[tokio::test]
    async fn greedy_gc_keeps_one_of_a_redundant_pair() {
        let (a, b) = (mref(), mref());
        let report = greedy_scorer()
            .score(&CoverageEvaluator::new(0.8, &[a, b], 0.4), &[a, b])
            .await
            .unwrap();
        // Exactly one redundant copy is dead weight (its removal cost nothing
        // because the other still covered); the cover itself is kept. LOO would
        // wrongly GC *both* and lose the capability.
        let gc = report.gc_candidates(1e-4);
        assert_eq!(gc.len(), 1, "GC one copy, keep the cover; got {gc:?}");
        let kept = report.high_contributors(0.3);
        assert_eq!(kept.len(), 1, "the surviving cover carries the full 0.4");
    }

    #[tokio::test]
    async fn greedy_matches_loo_on_additive_objective() {
        // On an additive objective greedy and LOO agree — greedy is a strict
        // generalisation, not a different answer where LOO is already correct.
        let (helpful, inert, harmful) = (mref(), mref(), mref());
        let mk = || {
            FakeEvaluator::new(0.8)
                .with_marginal(helpful, 0.4)
                .with_marginal(inert, 0.0)
                .with_marginal(harmful, -0.2)
        };
        let report = greedy_scorer()
            .score(&mk(), &[helpful, inert, harmful])
            .await
            .unwrap();
        let by = |m: MemoryRef| {
            report
                .scored
                .iter()
                .find(|c| c.memory == m)
                .unwrap()
                .contribution
        };
        assert!((by(helpful) - 0.4).abs() < 1e-6, "helpful ≈ +0.4");
        assert!(by(inert).abs() < 1e-6, "inert ≈ 0");
        assert!((by(harmful) + 0.2).abs() < 1e-6, "harmful ≈ -0.2");
    }
    // ----- per-task attribution (the 18C-E join key) -----

    /// Per-task evaluator modelled on code retrieval: task `i` scores 1.0 while
    /// any of its covering symbols survives the mask, 0.0 once all are gone.
    /// `tasks` is a `Vec` and is never reordered, so the stable-order contract
    /// holds by construction.
    struct TaskEvaluator {
        tasks: Vec<Vec<MemoryRef>>,
        /// Deliberately return a short vector for masked passes, to exercise
        /// the ordering-contract check.
        break_contract: bool,
    }

    impl TaskEvaluator {
        fn new(tasks: Vec<Vec<MemoryRef>>) -> Self {
            Self {
                tasks,
                break_contract: false,
            }
        }
    }

    #[async_trait]
    impl CounterfactualEvaluator for TaskEvaluator {
        async fn evaluate(&self, masked: &HashSet<MemoryRef>) -> Result<f32, MnesioError> {
            let per = self.evaluate_per_task(masked).await?.unwrap_or_default();
            if per.is_empty() {
                return Ok(0.0);
            }
            Ok(per.iter().sum::<f32>() / per.len() as f32)
        }

        async fn evaluate_per_task(
            &self,
            masked: &HashSet<MemoryRef>,
        ) -> Result<Option<Vec<f32>>, MnesioError> {
            let mut per: Vec<f32> = self
                .tasks
                .iter()
                .map(|cover| {
                    let alive = cover.iter().any(|m| !masked.contains(m));
                    if alive {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            if self.break_contract && !masked.is_empty() {
                per.pop();
            }
            Ok(Some(per))
        }
    }

    #[tokio::test]
    async fn per_task_attribution_names_the_tasks_a_memory_carried() {
        let (a, b, c) = (mref(), mref(), mref());
        // Three tasks, each covered by exactly one distinct memory.
        let ev = TaskEvaluator::new(vec![vec![a], vec![b], vec![c]]);
        let r = scorer().score(&ev, &[a, b, c]).await.unwrap();

        assert!(
            r.per_task_resolved,
            "the evaluator supplied per-task scores"
        );
        assert_eq!(r.task_count, Some(3));

        let find = |m: MemoryRef| r.scored.iter().find(|s| s.memory == m).unwrap();
        assert_eq!(find(a).decisive_for, vec![0], "a carried only task 0");
        assert_eq!(find(b).decisive_for, vec![1], "b carried only task 1");
        assert_eq!(find(c).decisive_for, vec![2], "c carried only task 2");

        // This is the shape a scoped promotion rule is built from.
        let by = r.decisive_by_task().expect("resolved");
        assert_eq!(by.get(&0).unwrap(), &vec![a]);
        assert_eq!(by.len(), 3);
    }

    /// The distinction the whole design turns on: an aggregate-only evaluator
    /// must not look like one that measured and found nothing decisive.
    #[tokio::test]
    async fn an_aggregate_only_evaluator_reports_no_resolution_rather_than_no_result() {
        let helpful = mref();
        let ev = FakeEvaluator::new(0.8).with_marginal(helpful, 0.3);
        let r = scorer().score(&ev, &[helpful]).await.unwrap();

        // It *is* load-bearing in aggregate...
        assert!((r.scored[0].contribution - 0.3).abs() < 1e-6);
        // ...but which task it carried is unknown, and says so.
        assert!(!r.per_task_resolved);
        assert_eq!(r.task_count, None);
        assert!(r.scored[0].decisive_for.is_empty());
        assert!(
            r.decisive_by_task().is_none(),
            "None (never measured), never Some(empty) (measured, nothing found)"
        );
    }

    #[tokio::test]
    async fn an_inert_memory_under_a_per_task_evaluator_is_measured_and_empty() {
        let (carrier, inert) = (mref(), mref());
        let ev = TaskEvaluator::new(vec![vec![carrier]]);
        let r = scorer().score(&ev, &[carrier, inert]).await.unwrap();

        let find = |m: MemoryRef| r.scored.iter().find(|s| s.memory == m).unwrap();
        assert_eq!(find(carrier).decisive_for, vec![0]);
        assert!(find(inert).decisive_for.is_empty());

        // Measured, and genuinely decisive for nothing — so the grouping is
        // present and simply omits it.
        let by = r.decisive_by_task().expect("resolved");
        assert!(r.per_task_resolved);
        assert_eq!(by.get(&0).unwrap(), &vec![carrier]);
        assert!(!by.values().flatten().any(|m| *m == inert));
    }

    /// Entry `i` must mean the same task on every call. If it does not, the
    /// elementwise subtraction is meaningless, and reporting an empty
    /// `decisive_for` would render as "carried nothing" — the exact misreading
    /// `per_task_resolved` was added to prevent. So it errors.
    #[tokio::test]
    async fn an_unstable_task_order_is_an_error_not_a_silent_empty() {
        let a = mref();
        let mut ev = TaskEvaluator::new(vec![vec![a], vec![mref()]]);
        ev.break_contract = true;

        let err = scorer().score(&ev, &[a]).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("same task on every call"),
            "the error should name the violated contract, got: {msg}"
        );
    }

    #[tokio::test]
    async fn greedy_attributes_a_redundant_pair_to_the_task_they_both_cover() {
        let (x, y) = (mref(), mref());
        // One task, covered redundantly: LOO would score both at 0.
        let ev = TaskEvaluator::new(vec![vec![x, y]]);
        let greedy = ContributionScorer::new(CausalConfig {
            mode: ScoreMode::GreedyAblation,
            ..CausalConfig::default()
        });
        let r = greedy.score(&ev, &[x, y]).await.unwrap();

        assert!(r.per_task_resolved);
        // The second one ablated is where the cover is finally lost, so that is
        // the step the task's loss attributes to.
        let carried: Vec<&MemoryContribution> = r
            .scored
            .iter()
            .filter(|s| !s.decisive_for.is_empty())
            .collect();
        assert_eq!(carried.len(), 1, "the loss lands on exactly one step");
        assert_eq!(carried[0].decisive_for, vec![0]);
        assert!(carried[0].contribution > 0.0);
    }
}

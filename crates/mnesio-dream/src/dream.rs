//! Dreaming: a bounded offline consolidation pass (sleep-time compute).
//!
//! One pass does two log-derived things, both bounded by [`DreamConfig`]:
//!
//! 1. **Prune by Phase-10 contribution.** Take a
//!    [`mnesio_causal::ContributionReport`] and retire the provable dead/harmful
//!    weight (contribution ≤ ε), not a time-decay guess. The expected
//!    *improvement* to the next learning-curve generation is the harm removed:
//!    `Σ max(0, −contribution)` over pruned memories (cutting a memory that was
//!    actively hurting outcomes raises the next generation).
//! 2. **Re-anchor drifted notes.** A note that has evolved many times can drift
//!    from its `parent`'s topic (the cascade-divergence problem). For each
//!    [`DriftedNote`] past the drift threshold, re-link it to include its
//!    `parent`, pulling the chain back toward its origin.
//!
//! Both are applied by *appending* events ([`apply`]): `MemoryInvalidated` for
//! prunes (invalidate, never delete — Hard Rule #2), `MemoryLinksUpdated` for
//! re-anchors. The pass is caller-scheduled, offline, capped (#5/#6).

use mnesio_causal::ContributionReport;
use mnesio_core::event::Event;
use mnesio_core::traits::EventLog;
use mnesio_core::types::MemoryRef;
use mnesio_core::MnesioError;
use serde::{Deserialize, Serialize};

/// Reason stamped on dream-pruned invalidations, so audits can tell a
/// dream-pass prune apart from causal GC / contradiction / falsification.
pub const DREAM_PRUNE_REASON: &str = "dream-pass: pruned by counterfactual contribution ≤ epsilon";

/// A note that may have drifted from its origin through repeated evolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriftedNote {
    pub child: MemoryRef,
    /// The lineage root/parent the note should stay anchored to.
    pub parent: MemoryRef,
    /// How many times this note has evolved (A-MEM `evolution_count`).
    pub evolution_count: u16,
    /// The note's current links (so re-anchoring is additive, not destructive).
    pub current_links: Vec<MemoryRef>,
}

/// Bounds + thresholds for one dream pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Max memories pruned per pass — cascade bound (#6).
    pub max_prune: usize,
    /// Max notes re-anchored per pass — cascade bound (#6).
    pub max_reanchor: usize,
    /// Contribution at/below which a memory is prunable (matches the causal
    /// GC epsilon).
    pub epsilon: f32,
    /// Evolution count at/above which a note is considered drifted and gets
    /// re-anchored.
    pub drift_threshold: u16,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            max_prune: 64,
            max_reanchor: 64,
            epsilon: 1e-4,
            drift_threshold: 3,
        }
    }
}

/// One re-anchor decision: `child`'s links will be updated to `new_links`
/// (its current links plus its `parent`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReanchorAction {
    pub child: MemoryRef,
    pub parent: MemoryRef,
    pub new_links: Vec<MemoryRef>,
}

/// What a dream pass decided (before [`apply`] writes it to the log).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamReport {
    /// Memories to prune (contribution ≤ ε).
    pub prune: Vec<MemoryRef>,
    /// Notes to re-anchor to their parent.
    pub reanchor: Vec<ReanchorAction>,
    /// Expected lift to the next learning-curve generation from pruning: the
    /// total *harm* removed, `Σ max(0, −contribution)` over pruned memories.
    /// `> 0` means the pass cut actively-misleading weight.
    pub generation_delta: f32,
    /// Baseline objective the contribution report measured against (for
    /// dashboard context).
    pub baseline_score: f32,
    pub candidates_considered: usize,
    pub drifted_considered: usize,
}

impl DreamReport {
    pub fn pruned_count(&self) -> usize {
        self.prune.len()
    }
    pub fn reanchored_count(&self) -> usize {
        self.reanchor.len()
    }
}

/// Plans a dream pass from a contribution report + the set of drifted notes.
pub struct DreamPass {
    cfg: DreamConfig,
}

impl DreamPass {
    pub fn new(cfg: DreamConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &DreamConfig {
        &self.cfg
    }

    /// Plan (don't apply) the pass: choose prunes from `contributions` and
    /// re-anchors from `drifted`, both bounded by [`DreamConfig`].
    pub fn plan(&self, contributions: &ContributionReport, drifted: &[DriftedNote]) -> DreamReport {
        // Prune the provable dead/harmful weight, bounded.
        let prune: Vec<MemoryRef> = contributions
            .gc_candidates(self.cfg.epsilon)
            .into_iter()
            .take(self.cfg.max_prune)
            .collect();

        // Next-generation lift = total harm removed. A memory with negative
        // contribution was actively hurting; cutting it should raise the next
        // generation by |contribution|. Inert (≈0) memories contribute 0 lift
        // but are still worth pruning for corpus hygiene.
        let prune_set: std::collections::HashSet<MemoryRef> = prune.iter().copied().collect();
        let generation_delta: f32 = contributions
            .scored
            .iter()
            .filter(|c| prune_set.contains(&c.memory))
            .map(|c| (-c.contribution).max(0.0))
            .sum();

        // Re-anchor drifted notes (evolution_count ≥ threshold), bounded.
        let reanchor: Vec<ReanchorAction> = drifted
            .iter()
            .filter(|n| n.evolution_count >= self.cfg.drift_threshold)
            .take(self.cfg.max_reanchor)
            .map(|n| {
                let mut new_links = n.current_links.clone();
                if !new_links.contains(&n.parent) {
                    new_links.push(n.parent);
                }
                ReanchorAction {
                    child: n.child,
                    parent: n.parent,
                    new_links,
                }
            })
            .collect();

        DreamReport {
            prune,
            reanchor,
            generation_delta,
            baseline_score: contributions.baseline_score,
            candidates_considered: contributions.scored.len(),
            drifted_considered: drifted.len(),
        }
    }
}

/// What [`apply`] wrote to the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DreamOutcome {
    pub pruned: usize,
    pub reanchored: usize,
}

/// Apply a planned dream pass by appending events: one `MemoryInvalidated` per
/// prune (invalidate, never delete — Hard Rule #2) and one
/// `MemoryLinksUpdated` per re-anchor. Idempotency isn't claimed; a scheduler
/// should re-plan against the live corpus rather than replay a stale report.
pub async fn dream(log: &dyn EventLog, report: &DreamReport) -> Result<DreamOutcome, MnesioError> {
    for memory in &report.prune {
        log.append(Event::MemoryInvalidated {
            id: *memory,
            reason: DREAM_PRUNE_REASON.to_string(),
        })
        .await?;
    }
    for action in &report.reanchor {
        log.append(Event::MemoryLinksUpdated {
            id: action.child,
            links: action.new_links.clone(),
        })
        .await?;
    }
    Ok(DreamOutcome {
        pruned: report.prune.len(),
        reanchored: report.reanchor.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use mnesio_causal::{CausalConfig, ContributionScorer, FakeEvaluator};
    use mnesio_core::event::LogEntry;
    use mnesio_core::types::new_id;
    use std::sync::Mutex;

    fn mref() -> MemoryRef {
        MemoryRef(new_id())
    }

    #[derive(Default)]
    struct FakeLog {
        events: Mutex<Vec<Event>>,
    }

    #[async_trait]
    impl EventLog for FakeLog {
        async fn append(&self, event: Event) -> Result<mnesio_core::Id, MnesioError> {
            self.events.lock().unwrap().push(event);
            Ok(new_id())
        }
        async fn read_from(
            &self,
            _after: Option<mnesio_core::Id>,
        ) -> Result<Vec<LogEntry>, MnesioError> {
            Ok(vec![])
        }
    }

    impl FakeLog {
        fn events(&self) -> Vec<Event> {
            self.events.lock().unwrap().clone()
        }
    }

    /// Build a contribution report via the real causal scorer: `helpful`
    /// (+0.4), `inert` (0.0), `harmful` (−0.3).
    async fn contributions(
        helpful: MemoryRef,
        inert: MemoryRef,
        harmful: MemoryRef,
    ) -> ContributionReport {
        let eval = FakeEvaluator::new(0.7)
            .with_marginal(helpful, 0.4)
            .with_marginal(inert, 0.0)
            .with_marginal(harmful, -0.3);
        ContributionScorer::new(CausalConfig::default())
            .score(&eval, &[helpful, inert, harmful])
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn prunes_inert_and_harmful_keeps_load_bearing() {
        let (helpful, inert, harmful) = (mref(), mref(), mref());
        let report = contributions(helpful, inert, harmful).await;
        let plan = DreamPass::new(DreamConfig::default()).plan(&report, &[]);
        assert!(plan.prune.contains(&inert));
        assert!(plan.prune.contains(&harmful));
        assert!(!plan.prune.contains(&helpful), "load-bearing memory kept");
    }

    #[tokio::test]
    async fn generation_delta_counts_only_harm_removed() {
        // The Phase-14 dreaming "done when": an offline pass improves the next
        // generation. Lift = harm removed = |−0.3| from the harmful memory;
        // the inert one adds 0.
        let (helpful, inert, harmful) = (mref(), mref(), mref());
        let report = contributions(helpful, inert, harmful).await;
        let plan = DreamPass::new(DreamConfig::default()).plan(&report, &[]);
        assert!(
            (plan.generation_delta - 0.3).abs() < 1e-6,
            "next-gen lift should equal the harm removed (0.3); got {}",
            plan.generation_delta
        );
    }

    #[tokio::test]
    async fn reanchor_only_drifted_notes_and_is_additive() {
        let cfg = DreamConfig::default(); // drift_threshold = 3
        let parent = mref();
        let other = mref();
        let drifted = DriftedNote {
            child: mref(),
            parent,
            evolution_count: 4, // ≥ threshold
            current_links: vec![other],
        };
        let fresh = DriftedNote {
            child: mref(),
            parent: mref(),
            evolution_count: 1, // < threshold
            current_links: vec![],
        };
        let empty = ContributionReport {
            baseline_score: 0.0,
            mode: mnesio_causal::ScoreMode::LeaveOneOut,
            scored: vec![],
            candidates_considered: 0,
            candidates_scored: 0,
            per_task_resolved: false,
            task_count: None,
        };
        let plan = DreamPass::new(cfg).plan(&empty, &[drifted.clone(), fresh]);
        assert_eq!(plan.reanchor.len(), 1, "only the drifted note re-anchors");
        let a = &plan.reanchor[0];
        assert_eq!(a.child, drifted.child);
        assert!(a.new_links.contains(&parent), "parent added (re-anchored)");
        assert!(
            a.new_links.contains(&other),
            "existing links preserved (additive)"
        );
    }

    #[tokio::test]
    async fn bounds_cap_prune_and_reanchor() {
        // 4 inert candidates, max_prune = 2.
        let cands: Vec<MemoryRef> = (0..4).map(|_| mref()).collect();
        let mut eval = FakeEvaluator::new(0.5);
        for m in &cands {
            eval = eval.with_marginal(*m, 0.0);
        }
        let report = ContributionScorer::new(CausalConfig::default())
            .score(&eval, &cands)
            .await
            .unwrap();
        let drifted: Vec<DriftedNote> = (0..4)
            .map(|_| DriftedNote {
                child: mref(),
                parent: mref(),
                evolution_count: 5,
                current_links: vec![],
            })
            .collect();
        let cfg = DreamConfig {
            max_prune: 2,
            max_reanchor: 1,
            ..DreamConfig::default()
        };
        let plan = DreamPass::new(cfg).plan(&report, &drifted);
        assert_eq!(plan.prune.len(), 2, "prune cascade bound held");
        assert_eq!(plan.reanchor.len(), 1, "re-anchor cascade bound held");
    }

    #[tokio::test]
    async fn apply_appends_invalidate_and_links_events() {
        let parent = mref();
        let child = mref();
        let pruned = mref();
        let report = DreamReport {
            prune: vec![pruned],
            reanchor: vec![ReanchorAction {
                child,
                parent,
                new_links: vec![parent],
            }],
            generation_delta: 0.3,
            baseline_score: 0.7,
            candidates_considered: 1,
            drifted_considered: 1,
        };
        let log = FakeLog::default();
        let outcome = dream(&log, &report).await.unwrap();
        assert_eq!(outcome.pruned, 1);
        assert_eq!(outcome.reanchored, 1);

        let events = log.events();
        assert_eq!(events.len(), 2);
        match &events[0] {
            Event::MemoryInvalidated { id, reason } => {
                assert_eq!(*id, pruned);
                assert_eq!(reason, DREAM_PRUNE_REASON);
            }
            other => panic!("expected MemoryInvalidated, got {other:?}"),
        }
        match &events[1] {
            Event::MemoryLinksUpdated { id, links } => {
                assert_eq!(*id, child);
                assert!(links.contains(&parent));
            }
            other => panic!("expected MemoryLinksUpdated, got {other:?}"),
        }
    }
}

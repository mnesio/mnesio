//! Phase 18D: compile recorded outcomes into **gated** retrieval rules.
//!
//! ## The loop this closes
//!
//! [`crate::outcome`] records what happened after an agent used retrieved
//! code. This module turns batches of those into candidate
//! `PolicyArtifact::RetrievalRule`s — and, critically, does **not** decide
//! whether to apply them. That decision belongs to the procedural gate, which
//! re-evaluates every candidate against canaries and a safety probe before it
//! can commit (Hard Rule #1).
//!
//! That separation is the entire safety story. A learning loop that both
//! proposes and applies its own changes is one bad batch away from teaching
//! itself to retrieve nothing. Proposing is cheap and reversible; committing
//! is neither, so only the gate does it.
//!
//! ## What is learnable from an outcome, and what is not
//!
//! An outcome says *this context led to this result*. It does not say which
//! symbol was responsible — several were present, and the agent may have used
//! none of them. So the only claims this module makes are **aggregate and
//! per-decision**:
//!
//! - a symbol that appears in many contexts and is present in far more
//!   failures than successes is a **suppression** candidate;
//! - a query class whose outcomes are consistently poor is a **rewrite**
//!   candidate.
//!
//! Both are proposals about *correlations that repeated*, which is why the
//! evidence floor exists and why the gate exists after it. Phase 10's
//! counterfactual masking is what upgrades a correlation to a contribution;
//! until that runs, nothing here should be described as causal.
//!
//! ## Why suppression is the first rule type
//!
//! It is the one where a mistake is cheapest to detect. Suppressing a symbol
//! that actually helps shows up immediately as a canary failure, so the gate
//! catches it. A rewrite that subtly degrades ranking can pass canaries and
//! still be worse, so it needs the fuller eval — which is why it is proposed
//! more conservatively here.

use std::collections::HashMap;

use mnesio_core::entity::ArtifactKind;
use mnesio_core::types::MemoryRef;

use crate::outcome::{CodeOutcome, DecisionEvidence};

/// How much evidence before a rule may even be *proposed*.
///
/// Deliberately a floor on proposal, not on commit — the gate applies its own,
/// stricter test. Proposing on thin evidence wastes an evaluation cycle and
/// fills the candidate list with noise the compiler then has to rank.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearnConfig {
    /// Decisive outcomes a symbol must appear in before it can be suppressed.
    ///
    /// Ambiguous results are excluded by [`DecisionEvidence`], so this is a
    /// floor on actual information rather than on activity.
    pub min_decisive: usize,
    /// Success rate at or below which a symbol looks harmful.
    ///
    /// Not 0.0: a symbol present in nine failures and one success is worth
    /// suppressing, and demanding a perfect failure record would mean almost
    /// nothing ever qualifies.
    pub max_success_rate: f32,
    /// Cap on rules proposed per batch (Hard Rule #6 — bound the cascade).
    ///
    /// A batch that suppresses fifty symbols at once is untestable: if the
    /// gate rejects it, nothing says which suppression was wrong.
    pub max_rules_per_batch: usize,
}

impl Default for LearnConfig {
    fn default() -> Self {
        Self {
            // Five decisive outcomes is where a run of failures stops being
            // plausibly luck. Below that the gate would be adjudicating noise.
            min_decisive: 5,
            max_success_rate: 0.2,
            max_rules_per_batch: 3,
        }
    }
}

/// A rule the compiler *may* commit, with the evidence that produced it.
///
/// Carries an [`ArtifactKind`], not a whole `PolicyArtifact`. A proposal has no
/// business inventing a version number, a scope or a canary set — those are
/// decided by whoever commits it, and letting a proposal supply them would put
/// the thing being judged in charge of its own paperwork.
///
/// The evidence travels alongside so a rejection is diagnosable: "the gate
/// refused this" is only actionable next to "and here is what we thought we
/// knew".
#[derive(Debug, Clone)]
pub struct RuleProposal {
    pub kind: ArtifactKind,
    /// Human-readable justification, carried into the commit record.
    pub rationale: String,
    pub evidence: DecisionEvidence,
}

/// Evidence gathered per symbol across a batch of outcomes.
#[derive(Debug, Default)]
pub struct SymbolLedger {
    per_symbol: HashMap<MemoryRef, DecisionEvidence>,
    /// Query classes seen, so a suppression can be scoped rather than global.
    ///
    /// A symbol that misleads one kind of task may be exactly right for
    /// another; a global suppression would trade a local win for a broad loss.
    per_symbol_classes: HashMap<MemoryRef, Vec<String>>,
}

impl SymbolLedger {
    /// Fold one outcome in.
    ///
    /// Every symbol in the context is credited or blamed equally. That is
    /// crude and it is stated rather than hidden: the outcome does not say
    /// which symbol mattered, so pretending otherwise would invent precision.
    /// Counterfactual masking (Phase 10) is what replaces this with a real
    /// per-symbol contribution.
    pub fn record(&mut self, outcome: &CodeOutcome) {
        let class = query_class(&outcome.task);
        for s in &outcome.symbols {
            self.per_symbol
                .entry(s.memory)
                .or_default()
                .record(outcome.result);
            let classes = self.per_symbol_classes.entry(s.memory).or_default();
            if !classes.contains(&class) {
                classes.push(class.clone());
            }
        }
    }

    pub fn evidence(&self, m: MemoryRef) -> Option<&DecisionEvidence> {
        self.per_symbol.get(&m)
    }

    pub fn symbols_seen(&self) -> usize {
        self.per_symbol.len()
    }

    /// Propose suppression rules for symbols that keep appearing in failures.
    ///
    /// Ordered worst-first and capped, so the batch the gate evaluates is
    /// small enough that a rejection is attributable.
    pub fn propose(&self, cfg: LearnConfig) -> Vec<RuleProposal> {
        let mut candidates: Vec<(MemoryRef, &DecisionEvidence, f32)> = self
            .per_symbol
            .iter()
            .filter_map(|(m, e)| {
                if !e.is_actionable(cfg.min_decisive) {
                    return None;
                }
                let rate = e.success_rate()?;
                (rate <= cfg.max_success_rate).then_some((*m, e, rate))
            })
            .collect();

        // Worst first, then by evidence volume, then by id: a stable order is
        // what makes a rejected batch reproducible.
        candidates.sort_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.decisive().cmp(&a.1.decisive()))
                .then(a.0 .0.cmp(&b.0 .0))
        });
        candidates.truncate(cfg.max_rules_per_batch);

        candidates
            .into_iter()
            .map(|(m, e, rate)| {
                // Scoped to the query classes this symbol actually misled, not
                // globally: a symbol that is wrong for one kind of task can be
                // exactly right for another.
                let classes = self
                    .per_symbol_classes
                    .get(&m)
                    .map(|c| c.join("|"))
                    .unwrap_or_default();
                RuleProposal {
                    kind: ArtifactKind::RetrievalRule {
                        query_pattern: classes.clone(),
                        rewrite: format!("-exclude:{}", m.0),
                    },
                    rationale: format!(
                        "{} of {} decisive outcomes failed ({:.0}% success) across query \
                         classes [{}]; {} ambiguous outcomes were excluded",
                        e.failures,
                        e.decisive(),
                        rate * 100.0,
                        classes,
                        e.ambiguous
                    ),
                    evidence: e.clone(),
                }
            })
            .collect()
    }
}

/// Reduce a task to the class the compiler learns over.
///
/// Content words only, lowercased and sorted, so "fix the retry backoff" and
/// "retry backoff fix" are the same class. Crude on purpose: an over-specific
/// class never accumulates enough evidence to clear the floor, so a rule for
/// it would never be proposed at all.
fn query_class(task: &str) -> String {
    const NOISE: &[&str] = &[
        "the", "and", "for", "with", "add", "fix", "use", "new", "not", "from", "into", "when",
        "that", "this", "make", "update", "remove", "support", "should", "would",
    ];
    let mut words: Vec<String> = task
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(str::to_lowercase)
        .filter(|w| !NOISE.contains(&w.as_str()))
        .collect();
    words.sort();
    words.dedup();
    words.truncate(4);
    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::{AttributedSymbol, Attribution, EditResult};
    use mnesio_core::types::new_id;

    fn outcome(task: &str, m: MemoryRef, result: EditResult) -> CodeOutcome {
        CodeOutcome {
            task: task.into(),
            repo: "r".into(),
            result,
            symbols: vec![AttributedSymbol {
                memory: m,
                attribution: Attribution::Seed { rank: 0 },
                full_body: true,
            }],
            tokens_used: 100,
        }
    }

    #[test]
    fn a_consistently_harmful_symbol_is_proposed_for_suppression() {
        let bad = MemoryRef(new_id());
        let mut led = SymbolLedger::default();
        for _ in 0..6 {
            led.record(&outcome(
                "parse config loader",
                bad,
                EditResult::BuildFailed,
            ));
        }
        let rules = led.propose(LearnConfig::default());
        assert_eq!(rules.len(), 1);
        match &rules[0].kind {
            ArtifactKind::RetrievalRule { rewrite, .. } => {
                assert!(rewrite.contains("-exclude:"), "got {rewrite}")
            }
            other => panic!("wrong artifact kind: {other:?}"),
        }
    }

    #[test]
    fn thin_evidence_proposes_nothing() {
        // The gate would reject it anyway; proposing wastes an evaluation
        // cycle and fills the candidate list with noise.
        let m = MemoryRef(new_id());
        let mut led = SymbolLedger::default();
        for _ in 0..2 {
            led.record(&outcome("t", m, EditResult::BuildFailed));
        }
        assert!(led.propose(LearnConfig::default()).is_empty());
    }

    #[test]
    fn ambiguous_outcomes_never_reach_the_floor() {
        // Twenty red tests carry no information about retrieval quality. If
        // they counted, the compiler would learn to suppress every symbol in
        // a repository with a broken test suite.
        let m = MemoryRef(new_id());
        let mut led = SymbolLedger::default();
        for _ in 0..20 {
            led.record(&outcome("t", m, EditResult::TestsFailed));
        }
        assert!(led.propose(LearnConfig::default()).is_empty());
    }

    #[test]
    fn a_symbol_that_mostly_helps_is_left_alone() {
        let good = MemoryRef(new_id());
        let mut led = SymbolLedger::default();
        for _ in 0..8 {
            led.record(&outcome("t", good, EditResult::Passed));
        }
        led.record(&outcome("t", good, EditResult::BuildFailed));
        assert!(led.propose(LearnConfig::default()).is_empty());
    }

    #[test]
    fn suppression_is_scoped_to_the_classes_that_failed() {
        // A symbol wrong for one kind of task can be right for another.
        // Suppressing it globally trades a local win for a broad loss.
        let m = MemoryRef(new_id());
        let mut led = SymbolLedger::default();
        for _ in 0..6 {
            led.record(&outcome("retry backoff timeout", m, EditResult::Rejected));
        }
        let rules = led.propose(LearnConfig::default());
        match &rules[0].kind {
            ArtifactKind::RetrievalRule { query_pattern, .. } => {
                assert!(
                    !query_pattern.is_empty(),
                    "a global suppression is too broad"
                );
                assert!(query_pattern.contains("backoff"), "got {query_pattern}");
            }
            other => panic!("wrong artifact kind: {other:?}"),
        }
    }

    #[test]
    fn a_batch_is_bounded_so_a_rejection_stays_attributable() {
        // Hard Rule #6. If the gate refuses a batch of fifty, nothing says
        // which suppression was wrong.
        let mut led = SymbolLedger::default();
        for _ in 0..10 {
            let m = MemoryRef(new_id());
            for _ in 0..6 {
                led.record(&outcome("t", m, EditResult::BuildFailed));
            }
        }
        assert_eq!(led.symbols_seen(), 10);
        assert_eq!(led.propose(LearnConfig::default()).len(), 3);
    }

    #[test]
    fn proposals_are_stable_across_runs() {
        // A rejected batch has to be reproducible, or a rejection cannot be
        // investigated.
        let ids: Vec<MemoryRef> = (0..6).map(|_| MemoryRef(new_id())).collect();
        let build = || {
            let mut led = SymbolLedger::default();
            for m in &ids {
                for _ in 0..6 {
                    led.record(&outcome("t", *m, EditResult::BuildFailed));
                }
            }
            led.propose(LearnConfig::default())
        };
        let a = build();
        let b = build();
        let key = |r: &RuleProposal| format!("{:?}", r.kind);
        assert_eq!(
            a.iter().map(key).collect::<Vec<_>>(),
            b.iter().map(key).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_rationale_carries_the_evidence_that_produced_it() {
        // "The gate refused this" is only actionable next to "here is what we
        // thought we knew".
        let m = MemoryRef(new_id());
        let mut led = SymbolLedger::default();
        for _ in 0..5 {
            led.record(&outcome("t", m, EditResult::BuildFailed));
        }
        led.record(&outcome("t", m, EditResult::TestsFailed));
        let r = &led.propose(LearnConfig::default())[0];
        assert!(r.rationale.contains("5 decisive"), "got {}", r.rationale);
        assert!(r.rationale.contains("1 ambiguous"), "got {}", r.rationale);
    }

    #[test]
    fn word_order_does_not_split_a_query_class() {
        assert_eq!(
            query_class("fix the retry backoff"),
            query_class("retry backoff fix")
        );
    }
}

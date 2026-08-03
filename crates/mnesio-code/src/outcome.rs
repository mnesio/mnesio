//! Phase 18C: record whether retrieved code actually *helped*.
//!
//! ## The gap this closes
//!
//! Every code-memory tool on the market ranks code by how relevant it looks.
//! None of them records whether the retrieval worked. That is the difference
//! between a search engine and a memory: a memory can be wrong and find out.
//!
//! Concretely, an agent asks for context, edits, and then something
//! observable happens — the build passes or fails, tests go green or red, the
//! diff is accepted or thrown away. That signal is free, it is objective, and
//! today it is discarded.
//!
//! ## Why attribution needs [`crate::pack::Reason`]
//!
//! "The retrieval helped" is not a learnable statement. *Which* part helped is.
//! A packed context contains symbols that retrieval ranked directly and
//! symbols pulled in as callees, and those deserve different credit: if
//! outcomes improve only when expansion fires, that is a fact about expansion,
//! not about the ranker.
//!
//! So an outcome is recorded against [`AttributedSymbol`]s carrying the reason
//! each was present. That is the join key the procedural compiler learns over
//! in 18D, and it is why `Reason` was kept on the packed context rather than
//! discarded after rendering.
//!
//! ## What this module does and does not decide
//!
//! It **records**. It does not change retrieval. Turning outcomes into
//! retrieval policy happens in `mnesio-procedural`, behind
//! `EvalReport::is_committable()` — Hard Rule #1 — so a rule that would break
//! a canary is refused rather than shipped. Keeping capture separate from
//! compilation is what makes that gate impossible to bypass by accident.
//!
//! ## Honest limits
//!
//! A signal is *correlational*. A build passing after a retrieval does not
//! prove the retrieval caused it, and an agent that ignored the context
//! entirely still reports success. Two things keep that from becoming
//! self-congratulation: the gate re-evaluates on held-out tasks before any
//! rule commits, and Phase 10's counterfactual masking measures contribution
//! by removing a symbol and re-running. Correlation is what we collect;
//! causation is what the gate demands.

use serde::{Deserialize, Serialize};

use mnesio_core::types::MemoryRef;

use crate::pack::{PackedContext, Reason};

/// What happened after an agent used the retrieved context.
///
/// Deliberately a small closed set of *observable* facts rather than a score.
/// A number invites a judge, and a judge invites disagreement about the judge;
/// "the build failed" is not a matter of opinion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditResult {
    /// Compiled and tests passed.
    Passed,
    /// Compiled, tests failed.
    TestsFailed,
    /// Did not compile.
    BuildFailed,
    /// A human rejected the change. The strongest negative signal available,
    /// and the only one that captures "correct but not what I wanted".
    Rejected,
    /// A human accepted the change. Strongest positive signal.
    Accepted,
}

impl EditResult {
    /// Did this go well?
    ///
    /// `None` for [`EditResult::TestsFailed`] on purpose. A failing test after
    /// an edit is ambiguous — the retrieval may have been perfect and the edit
    /// wrong, or the test may have been failing already. Scoring it as a loss
    /// would teach the compiler to avoid code that merely *has* tests.
    pub fn is_success(self) -> Option<bool> {
        match self {
            EditResult::Passed | EditResult::Accepted => Some(true),
            EditResult::BuildFailed | EditResult::Rejected => Some(false),
            EditResult::TestsFailed => None,
        }
    }

    /// Stable label for the event log and metrics.
    pub fn as_str(self) -> &'static str {
        match self {
            EditResult::Passed => "passed",
            EditResult::TestsFailed => "tests_failed",
            EditResult::BuildFailed => "build_failed",
            EditResult::Rejected => "rejected",
            EditResult::Accepted => "accepted",
        }
    }
}

/// Why a symbol was in the context it is being credited or blamed for.
///
/// Mirrors [`Reason`] but flattened for storage: the compiler needs to group
/// by *kind* of decision, not by which particular seed did the pulling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Attribution {
    /// Retrieval ranked it directly, at this position.
    Seed { rank: usize },
    /// Pulled in as a callee of something retrieval ranked.
    Expanded,
}

impl From<Reason> for Attribution {
    fn from(r: Reason) -> Self {
        match r {
            Reason::Seed(rank) => Attribution::Seed { rank },
            Reason::Expanded(_) => Attribution::Expanded,
        }
    }
}

/// One symbol that was in the context, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttributedSymbol {
    pub memory: MemoryRef,
    pub attribution: Attribution,
    /// Whether the full body was delivered or only a signature. A signature
    /// that "helped" is a much weaker claim than a body that did, and
    /// conflating them would credit the degradation ladder with wins it did
    /// not earn — a mistake already made once, in Phase 17B's `any` column.
    pub full_body: bool,
}

/// A retrieval and what came of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeOutcome {
    /// The task the agent was doing, verbatim. Kept because the compiler
    /// learns *per query class*, and the task text is what defines the class.
    pub task: String,
    /// Repository this happened in. Outcomes must not cross a scope boundary
    /// without an explicit aggregation step (Hard Rule #3).
    pub repo: String,
    pub result: EditResult,
    pub symbols: Vec<AttributedSymbol>,
    /// Tokens the context actually cost, so the compiler can weigh a win
    /// against what it was paid for.
    pub tokens_used: usize,
}

impl CodeOutcome {
    /// Record what a packed context led to.
    ///
    /// Takes the [`PackedContext`] rather than a list of ids so attribution
    /// cannot be lost at the call site — the reason a symbol was present is
    /// exactly the thing that makes the outcome learnable, and reconstructing
    /// it later is impossible.
    pub fn from_context(
        task: impl Into<String>,
        repo: impl Into<String>,
        context: &PackedContext,
        result: EditResult,
    ) -> Self {
        Self {
            task: task.into(),
            repo: repo.into(),
            result,
            symbols: context
                .symbols
                .iter()
                .map(|s| AttributedSymbol {
                    memory: s.memory,
                    attribution: s.reason.into(),
                    full_body: s.form == crate::pack::Form::Full,
                })
                .collect(),
            tokens_used: context.tokens_used,
        }
    }

    /// Symbols retrieval ranked directly.
    pub fn seeds(&self) -> impl Iterator<Item = &AttributedSymbol> {
        self.symbols
            .iter()
            .filter(|s| matches!(s.attribution, Attribution::Seed { .. }))
    }

    /// Symbols the graph pulled in.
    pub fn expansions(&self) -> impl Iterator<Item = &AttributedSymbol> {
        self.symbols
            .iter()
            .filter(|s| matches!(s.attribution, Attribution::Expanded))
    }
}

/// Aggregate evidence about one retrieval decision.
///
/// The unit the compiler reasons over in 18D. Deliberately separates the two
/// questions a policy change has to answer: *did outcomes go well when this
/// fired*, and *is there enough evidence to act on that*.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DecisionEvidence {
    pub successes: usize,
    pub failures: usize,
    /// Outcomes that were neither — see [`EditResult::is_success`].
    pub ambiguous: usize,
}

impl DecisionEvidence {
    pub fn record(&mut self, result: EditResult) {
        match result.is_success() {
            Some(true) => self.successes += 1,
            Some(false) => self.failures += 1,
            None => self.ambiguous += 1,
        }
    }

    /// Decisive outcomes — the denominator any rate should use.
    pub fn decisive(&self) -> usize {
        self.successes + self.failures
    }

    /// Success rate over decisive outcomes, or `None` when there are none.
    ///
    /// `None` rather than 0.0: a decision with no evidence is not a decision
    /// that fails, and returning zero would let the compiler suppress
    /// retrieval it has simply never seen used.
    pub fn success_rate(&self) -> Option<f32> {
        match self.decisive() {
            0 => None,
            n => Some(self.successes as f32 / n as f32),
        }
    }

    /// Is there enough here to justify changing policy?
    ///
    /// A floor on *decisive* observations, not total. Three outcomes that were
    /// all ambiguous carry no information, and acting on a 1-of-1 success rate
    /// is how a learning loop overfits its first lucky trial.
    pub fn is_actionable(&self, min_decisive: usize) -> bool {
        self.decisive() >= min_decisive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::{Form, PackedSymbol};
    use mnesio_core::types::new_id;

    fn packed(reason: Reason, form: Form) -> PackedSymbol {
        PackedSymbol {
            memory: MemoryRef(new_id()),
            form,
            tokens: 10,
            reason,
        }
    }

    #[test]
    fn attribution_survives_the_trip_from_the_packer() {
        // The whole point: an outcome that cannot say *which* decision it is
        // evidence about is not learnable.
        let seed = packed(Reason::Seed(0), Form::Full);
        let expanded = packed(Reason::Expanded(seed.memory), Form::Signature);
        let ctx = PackedContext {
            symbols: vec![seed.clone(), expanded.clone()],
            tokens_used: 20,
            ..Default::default()
        };

        let o = CodeOutcome::from_context("fix retry", "repo", &ctx, EditResult::Passed);
        assert_eq!(o.seeds().count(), 1);
        assert_eq!(o.expansions().count(), 1);
        assert_eq!(
            o.seeds().next().unwrap().attribution,
            Attribution::Seed { rank: 0 }
        );
    }

    #[test]
    fn a_signature_only_symbol_is_marked_as_such() {
        // Phase 17B measured the signature ladder adding +12pp on a lenient
        // criterion and exactly 0 on a strict one. Crediting a signature the
        // same as a body would repeat that error inside the learning loop,
        // where it would compound instead of just misreporting.
        let ctx = PackedContext {
            symbols: vec![packed(Reason::Seed(0), Form::Signature)],
            tokens_used: 5,
            ..Default::default()
        };
        let o = CodeOutcome::from_context("t", "r", &ctx, EditResult::Passed);
        assert!(!o.symbols[0].full_body);
    }

    #[test]
    fn a_failing_test_is_ambiguous_not_a_failure() {
        // A test can be red before the edit, and the retrieval can be perfect
        // while the edit is wrong. Counting it as a loss teaches the compiler
        // to avoid code that merely has tests.
        assert_eq!(EditResult::TestsFailed.is_success(), None);
        assert_eq!(EditResult::BuildFailed.is_success(), Some(false));
        assert_eq!(EditResult::Rejected.is_success(), Some(false));
        assert_eq!(EditResult::Accepted.is_success(), Some(true));
    }

    #[test]
    fn ambiguous_outcomes_do_not_move_the_rate() {
        let mut e = DecisionEvidence::default();
        e.record(EditResult::Passed);
        e.record(EditResult::TestsFailed);
        e.record(EditResult::TestsFailed);
        assert_eq!(e.decisive(), 1);
        assert_eq!(e.success_rate(), Some(1.0));
        assert_eq!(e.ambiguous, 2);
    }

    #[test]
    fn no_evidence_is_not_a_zero_score() {
        // Returning 0.0 here would let the compiler suppress retrieval it has
        // never actually seen used — a silent, self-reinforcing failure.
        assert_eq!(DecisionEvidence::default().success_rate(), None);
    }

    #[test]
    fn a_single_lucky_trial_is_not_actionable() {
        let mut e = DecisionEvidence::default();
        e.record(EditResult::Passed);
        assert!(!e.is_actionable(5), "one success must not justify a rule");
        for _ in 0..4 {
            e.record(EditResult::Passed);
        }
        assert!(e.is_actionable(5));
    }

    #[test]
    fn ambiguous_outcomes_do_not_count_toward_actionability() {
        let mut e = DecisionEvidence::default();
        for _ in 0..10 {
            e.record(EditResult::TestsFailed);
        }
        assert!(
            !e.is_actionable(3),
            "ten outcomes carrying no information is still no information"
        );
    }
}

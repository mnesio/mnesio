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
    /// Success rate at or above which a **term→symbol** pairing looks worth
    /// promoting. See [`SymbolLedger::propose_promotions`].
    pub min_promote_success_rate: f32,
    /// Decisive outcomes a term→symbol pairing needs before promotion.
    ///
    /// Separate from [`Self::min_decisive`], and lower, because the evidence is
    /// sliced far more finely: a symbol accumulates outcomes across every task
    /// it was packed into, but a *pairing* only accumulates on tasks mentioning
    /// that term. Reusing the suppression floor here would mean no pairing ever
    /// cleared it, which is a threshold chosen to guarantee silence.
    pub min_promote_decisive: usize,
    /// How far a pairing must beat the batch's own success rate.
    ///
    /// Without this the promoter selects the *commonest* terms rather than the
    /// most informative ones: at 88% baseline every well-evidenced pairing
    /// scores ~100%, so the tie-break degenerates to term frequency. Requiring
    /// a margin over the base rate is what makes "this term predicts this
    /// symbol" mean something more than "this term is frequent".
    pub min_promote_lift: f32,
    /// Probability, under the batch's own base rate, of a run at least this
    /// good happening by chance. Above it, the pairing is not evidence.
    pub max_promote_p: f32,
}

impl Default for LearnConfig {
    fn default() -> Self {
        Self {
            // Five decisive outcomes is where a run of failures stops being
            // plausibly luck. Below that the gate would be adjudicating noise.
            min_decisive: 5,
            max_success_rate: 0.2,
            max_rules_per_batch: 3,
            // A pairing that helped four times out of four is worth trying;
            // demanding perfection would exclude the useful near-misses, and
            // the gate is what makes a wrong guess cheap.
            min_promote_success_rate: 0.8,
            min_promote_decisive: 3,
            // Ten points clear of chance. Small enough that a real signal
            // survives, large enough that the base rate alone cannot qualify.
            min_promote_lift: 0.10,
            // The usual 5%. Not tuned: picking a threshold that lets the
            // current data through is how a null result becomes a feature.
            max_promote_p: 0.05,
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
    /// Evidence for one symbol on tasks containing one **term**.
    ///
    /// The unit promotion needs, and the reason it is keyed by term rather than
    /// by [`query_class`]. Measured on claw-code's 395 commit subjects: a
    /// 4-word class is effectively a commit fingerprint — 377 distinct classes,
    /// 3% recurring, and only **4%** of held-out tasks share a class with any
    /// training task. A rule scoped that finely can never fire on unseen work,
    /// so it would have been unmeasurable rather than merely weak.
    ///
    /// Single terms recur: the same corpus has 188 distinct first-terms with
    /// **51%** held-out coverage, and matching *any* shared term rather than an
    /// exact bucket is higher still. So a promotion is scoped to a term, and
    /// applies to any later task mentioning it.
    per_term_symbol: HashMap<(String, MemoryRef), DecisionEvidence>,
}

/// Which constraint stopped a batch from proposing anything.
///
/// See [`SymbolLedger::evidence_summary`]. The two counts at the end are the
/// ones that matter: `cleared_evidence_floor` is how much the loop *knows*,
/// and `also_looked_harmful` is how much of that is worth acting on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EvidenceSummary {
    /// Distinct symbols that appeared in any recorded context.
    pub symbols_seen: usize,
    /// Decisive outcomes summed over every symbol — the raw evidence volume.
    pub decisive_total: usize,
    /// The best-evidenced single symbol. If this is below
    /// [`LearnConfig::min_decisive`], nothing could have been proposed no
    /// matter how bad retrieval was, and the suite is the thing to change.
    pub max_decisive_on_one_symbol: usize,
    /// Symbols with enough evidence to be judged at all.
    pub cleared_evidence_floor: usize,
    /// …of those, how many also looked harmful enough to suppress.
    pub also_looked_harmful: usize,
    /// The floor these counts were taken against.
    ///
    /// Carried with the summary rather than looked up beside it, so a report
    /// containing one of these is self-describing: "3 cleared the floor" means
    /// nothing without knowing the floor was 5.
    pub min_decisive: usize,
}

impl EvidenceSummary {
    /// Mean decisive outcomes per symbol seen.
    ///
    /// The ratio that decides whether a suite can ever exercise the learner:
    /// measured at 0.09 on a whole repository and 0.16 on a single hot module,
    /// against the 5 one symbol needs. Both are starvation, and the second
    /// being better did not make it sufficient.
    pub fn outcomes_per_symbol(&self) -> f32 {
        match self.symbols_seen {
            0 => 0.0,
            n => self.decisive_total as f32 / n as f32,
        }
    }
}

impl SymbolLedger {
    /// Fold one outcome in.
    ///
    /// Every symbol in the context is credited or blamed equally. That is
    /// crude and it is stated rather than hidden: the outcome does not say
    /// which symbol mattered, so pretending otherwise would invent precision.
    /// Counterfactual masking (Phase 10) is what replaces this with a real
    /// per-symbol contribution.
    /// How many distinct symbols have at least one decisive outcome recorded
    /// against them.
    ///
    /// The *breadth* of the evidence, which a proposal count alone conceals:
    /// three rules drawn from three symbols and three rules drawn from three
    /// hundred describe very different states of knowledge, and only the
    /// second is a loop that has seen the repository.
    pub fn symbols_with_decisive_evidence(&self) -> usize {
        self.per_symbol
            .values()
            .filter(|e| e.decisive() > 0)
            .count()
    }

    /// Why [`Self::propose`] returned what it did.
    ///
    /// A bare "0 proposals" is unactionable, because two different failures
    /// produce it and they need opposite fixes:
    ///
    /// - **no symbol cleared the evidence floor** — the suite is too small or
    ///   too scattered, and the answer is more tasks over the same code;
    /// - **symbols cleared it and none looked harmful** — there is nothing to
    ///   suppress, and more tasks will not change that.
    ///
    /// Reporting a zero without saying which of those it is invites the wrong
    /// fix, and on this project the wrong fix would be lowering `min_decisive`
    /// until noise qualifies.
    pub fn evidence_summary(&self, cfg: LearnConfig) -> EvidenceSummary {
        let mut s = EvidenceSummary {
            symbols_seen: self.per_symbol.len(),
            min_decisive: cfg.min_decisive,
            ..Default::default()
        };
        for e in self.per_symbol.values() {
            let d = e.decisive();
            s.decisive_total += d;
            s.max_decisive_on_one_symbol = s.max_decisive_on_one_symbol.max(d);
            if e.is_actionable(cfg.min_decisive) {
                s.cleared_evidence_floor += 1;
                if e.success_rate().is_some_and(|r| r <= cfg.max_success_rate) {
                    s.also_looked_harmful += 1;
                }
            }
        }
        s
    }

    pub fn record(&mut self, outcome: &CodeOutcome) {
        let class = query_class(&outcome.task);
        let terms = query_terms(&outcome.task);
        for s in &outcome.symbols {
            self.per_symbol
                .entry(s.memory)
                .or_default()
                .record(outcome.result);
            let classes = self.per_symbol_classes.entry(s.memory).or_default();
            if !classes.contains(&class) {
                classes.push(class.clone());
            }
            for t in &terms {
                self.per_term_symbol
                    .entry((t.clone(), s.memory))
                    .or_default()
                    .record(outcome.result);
            }
        }
    }

    /// Propose **promotions**: term → symbol pairings that kept coming with
    /// success, so the symbol is pulled into context for later tasks using that
    /// term even when it would have ranked below the cutoff.
    ///
    /// ## Why this exists at all
    ///
    /// Suppression was the first rule type because a mistake is cheap to
    /// detect. Measured, it never fires: on two real modules 11 and 20 symbols
    /// cleared the evidence floor and **none** had a success rate at or below
    /// `max_success_rate`. With baseline recall near 88%, almost nothing sits
    /// in contexts that fail four times in five. You cannot learn by removing
    /// things when nothing is consistently poisonous.
    ///
    /// So the rule type that can move a healthy retriever is the opposite one.
    /// It is also the riskier one — a promotion displaces whatever it outranks,
    /// and unlike a suppression it can be wrong without any canary noticing the
    /// symbol it pushed out. That is why it goes through the same gate and why
    /// the batch cap applies.
    ///
    /// ## What it deliberately does not claim
    ///
    /// Co-occurrence with success is not contribution. The symbol may have been
    /// irrelevant and the task easy. Phase 10's counterfactual masking is what
    /// would upgrade this to a causal claim; until then a promotion is a bet
    /// that the gate is allowed to refuse.
    pub fn propose_promotions(&self, cfg: LearnConfig) -> Vec<RuleProposal> {
        // The base rate this batch succeeded at, over all symbols. A pairing is
        // only news if it beats this.
        //
        // Ranking by raw success rate does not work and the failure is
        // instructive rather than obvious. Measured on two real modules, the
        // rule it selected was `"tests"` and `"api"` — the *least*
        // discriminating terms in the corpus. With baseline recall near 88%,
        // "succeeded 4 times out of 4" is what chance looks like, so every
        // sufficiently-evidenced pairing ties at ~100% and the tie-break falls
        // to evidence volume, which is just term frequency. The rules fired,
        // passed the gate, and moved held-out recall +0.0pp on both modules.
        let (s, f): (usize, usize) = self
            .per_symbol
            .values()
            .fold((0, 0), |(s, f), e| (s + e.successes, f + e.failures));
        let base = if s + f == 0 {
            0.0
        } else {
            s as f32 / (s + f) as f32
        };
        let floor = cfg
            .min_promote_success_rate
            .max(base + cfg.min_promote_lift);

        let mut candidates: Vec<(&String, MemoryRef, &DecisionEvidence, f32)> = self
            .per_term_symbol
            .iter()
            .filter_map(|((term, m), e)| {
                if e.decisive() < cfg.min_promote_decisive {
                    return None;
                }
                let rate = e.success_rate()?;
                if rate < floor {
                    return None;
                }
                // Margin over the base rate is not enough, and this is the
                // second thing that had to be measured rather than assumed. At
                // base 0.88 a pairing of "4 successes out of 4" clears any
                // sensible lift threshold — but P(4 of 4 | p=0.88) is **0.60**.
                // It is the single most likely thing to observe. Selecting on
                // it produced three rules per module, on the commonest terms,
                // all committed by the gate, and +0.0pp on held-out.
                //
                // So the real test is whether the run would be surprising by
                // chance. At these base rates and sample sizes almost nothing
                // is, and proposing nothing is the correct answer rather than a
                // bug — see the module docs on why that points at Phase 10.
                (upper_tail(e.successes, e.decisive(), base as f64) <= cfg.max_promote_p as f64)
                    .then_some((term, *m, e, rate))
            })
            .collect();

        // Best first, then by evidence volume, then by term and id — a stable
        // order is what makes a rejected batch reproducible.
        candidates.sort_by(|a, b| {
            b.3.partial_cmp(&a.3)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.2.decisive().cmp(&a.2.decisive()))
                .then(a.0.cmp(b.0))
                .then(a.1 .0.cmp(&b.1 .0))
        });
        candidates.truncate(cfg.max_rules_per_batch);

        candidates
            .into_iter()
            .map(|(term, m, e, rate)| RuleProposal {
                kind: ArtifactKind::RetrievalRule {
                    query_pattern: term.clone(),
                    rewrite: format!("+boost:{}", m.0),
                },
                rationale: format!(
                    "{} of {} tasks mentioning {:?} succeeded with this symbol \
                     in context ({:.0}% success)",
                    e.successes,
                    e.decisive(),
                    term,
                    rate * 100.0
                ),
                evidence: e.clone(),
            })
            .collect()
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

/// P(X >= k) for X ~ Binomial(n, p) — the chance of a run at least this good.
///
/// Exact rather than a normal approximation because n is single digits here,
/// which is exactly where the approximation is worst.
fn upper_tail(k: usize, n: usize, p: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let mut term = (1.0 - p).powi(n as i32); // P(X = 0)
    let mut cdf_below = 0.0;
    for i in 0..k {
        cdf_below += term;
        // P(X=i+1) from P(X=i), avoiding factorials that overflow.
        term *= p / (1.0 - p) * ((n - i) as f64) / ((i + 1) as f64);
    }
    (1.0 - cdf_below).clamp(0.0, 1.0)
}

/// Reduce a task to the class the compiler learns over.
///
/// Content words only, lowercased and sorted, so "fix the retry backoff" and
/// "retry backoff fix" are the same class. Crude on purpose: an over-specific
/// class never accumulates enough evidence to clear the floor, so a rule for
/// it would never be proposed at all.
fn query_class(task: &str) -> String {
    let mut words = query_terms(task);
    words.truncate(4);
    words.join(" ")
}

/// The content words a rule may be scoped to, sorted and deduplicated.
///
/// Split out from [`query_class`] because promotion is keyed by a *single*
/// term. Measured on 395 real commit subjects: joining four of these into one
/// class produces 377 distinct classes with 4% held-out coverage — a
/// fingerprint. Individual terms recur, which is what makes a term-scoped rule
/// able to fire on work it has not seen.
pub(crate) fn query_terms(task: &str) -> Vec<String> {
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
    words
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
    fn a_term_that_beats_the_base_rate_is_promoted() {
        // Lift, not raw success. The batch has to contain failures or there is
        // no base rate to beat — which is itself the correct behaviour: when
        // everything already succeeds there is nothing to learn, and a promoter
        // that fired anyway would be selecting on term frequency alone.
        let helpful = MemoryRef(new_id());
        let other = MemoryRef(new_id());
        let mut l = SymbolLedger::default();
        for i in 0..4 {
            l.record(&outcome(
                &format!("session store cleanup pass {i}"),
                helpful,
                EditResult::Passed,
            ));
        }
        // Drag the base rate down with unrelated failures.
        for i in 0..6 {
            l.record(&outcome(
                &format!("sandbox mount path {i}"),
                other,
                EditResult::BuildFailed,
            ));
        }

        let props = l.propose_promotions(LearnConfig::default());
        let boost = props
            .iter()
            .find(|p| {
                matches!(&p.kind, ArtifactKind::RetrievalRule { rewrite, .. }
                               if rewrite.contains("+boost:"))
            })
            .expect("a term that beats the base rate must promote");
        let ArtifactKind::RetrievalRule { query_pattern, .. } = &boost.kind else {
            unreachable!()
        };
        // Scoped to a single term, not the whole task: a 4-word class is a
        // commit fingerprint (377 distinct over 395 subjects, 4% held-out
        // coverage) and would never fire on unseen work.
        assert!(!query_pattern.contains(' '), "got: {query_pattern:?}");
    }

    #[test]
    fn the_binomial_tail_matches_hand_computed_values() {
        // The number that killed the first two versions of this rule type:
        // four successes out of four, against a base rate of 0.88, is the most
        // likely single outcome — not evidence of anything.
        let p = upper_tail(4, 4, 0.88);
        assert!((p - 0.88_f64.powi(4)).abs() < 1e-9, "got {p}");
        assert!(p > 0.5, "4/4 at base 0.88 must not look surprising: {p}");

        // A fair coin run that genuinely is surprising.
        assert!(upper_tail(10, 10, 0.5) < 0.002);
        // Nothing observed is never surprising.
        assert_eq!(upper_tail(0, 5, 0.3), 1.0);
        // A full run at a low base rate is.
        assert!(upper_tail(5, 5, 0.2) < 0.001);
    }

    #[test]
    fn a_batch_that_never_fails_promotes_nothing() {
        // The degenerate case that produced the first, useless version of this
        // rule type: with an all-success batch every pairing scores 100%, the
        // ranking falls back to term frequency, and the promoter selected
        // "tests" and "api" — the least discriminating words available. Those
        // rules committed and moved held-out recall +0.0pp on two modules.
        let m = MemoryRef(new_id());
        let mut l = SymbolLedger::default();
        for i in 0..8 {
            l.record(&outcome(
                &format!("tests for thing {i}"),
                m,
                EditResult::Passed,
            ));
        }
        assert!(
            l.propose_promotions(LearnConfig::default()).is_empty(),
            "nothing to learn from a batch with no failures"
        );
    }

    #[test]
    fn a_term_that_mostly_failed_is_not_promoted() {
        // Promotion must not become a second door into the policy for a
        // pairing that suppression would have rejected.
        let bad = MemoryRef(new_id());
        let mut l = SymbolLedger::default();
        for i in 0..4 {
            l.record(&outcome(
                &format!("sandbox mount handling {i}"),
                bad,
                EditResult::BuildFailed,
            ));
        }
        assert!(l.propose_promotions(LearnConfig::default()).is_empty());
    }

    #[test]
    fn a_thinly_evidenced_pairing_is_not_promoted() {
        let m = MemoryRef(new_id());
        let mut l = SymbolLedger::default();
        // One success is a coincidence, not a pattern.
        l.record(&outcome("retry backoff jitter", m, EditResult::Passed));
        assert!(l.propose_promotions(LearnConfig::default()).is_empty());
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

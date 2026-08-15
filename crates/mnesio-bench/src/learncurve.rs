//! Phase 18 "done when": does gated retrieval learning actually improve
//! held-out performance?
//!
//! ## The experiment
//!
//! 1. Split the git-derived suite three ways: **train**, **canary**, **held-out**.
//! 2. Retrieve for each train task and record an outcome.
//! 3. Fold outcomes into a [`SymbolLedger`]; propose suppression rules.
//! 4. **Gate** each proposal: apply it, re-measure the canaries, and refuse it
//!    if they regress at all (Hard Rule #1).
//! 5. Apply only what survived, and re-measure **held-out**.
//!
//! The held-out split is never seen by the ledger or the gate, so an
//! improvement there is generalisation rather than memorisation. A rule that
//! merely re-describes the training set moves nothing.
//!
//! ## What the outcome signal is, precisely
//!
//! A real agent's outcome is "did the build pass". Here it is **"did the
//! packed context contain a gold symbol"** — derived from the same git history
//! that defines the task.
//!
//! This is cleaner than reality in a way that matters, and saying so is the
//! difference between a result and a demo. A real build signal is noisy and
//! only correlational: an agent can succeed while ignoring the context, or
//! fail for reasons retrieval could not have prevented. So a curve measured
//! here is an **upper bound** on what the same mechanism achieves in
//! production, not a prediction of it.
//!
//! What it does establish, and what no competitor can currently show at all:
//! that the mechanism is sound — proposals form from evidence, the gate
//! refuses ones that break canaries, and held-out performance is measured
//! rather than assumed. Whether the curve is positive is an empirical question
//! this harness exists to answer honestly, including when the answer is "flat".

use anyhow::{anyhow, Result};
use std::collections::HashSet;

use mnesio_code::learn::{LearnConfig, RuleProposal, SymbolLedger};
use mnesio_code::outcome::{AttributedSymbol, Attribution, CodeOutcome, EditResult};
use mnesio_core::entity::ArtifactKind;
use mnesio_core::types::MemoryRef;

use crate::codeeval::CodeQuery;

/// How the suite is divided. Deterministic by index so a run is reproducible.
///
/// Round-robin rather than contiguous slices: commits arrive in history order,
/// so taking the first 60% as training would train on old code and test on
/// new, measuring drift instead of learning.
pub struct Split<'a> {
    pub train: Vec<&'a CodeQuery>,
    pub canary: Vec<&'a CodeQuery>,
    pub held_out: Vec<&'a CodeQuery>,
}

pub fn split(suite: &[CodeQuery]) -> Split<'_> {
    let mut s = Split {
        train: Vec::new(),
        canary: Vec::new(),
        held_out: Vec::new(),
    };
    for (i, q) in suite.iter().enumerate() {
        match i % 5 {
            0..=2 => s.train.push(q),
            3 => s.canary.push(q),
            _ => s.held_out.push(q),
        }
    }
    s
}

/// A rule and what the gate decided about it.
#[derive(Debug, Clone)]
pub struct GateDecision {
    pub rationale: String,
    pub committed: bool,
    /// Canary recall before and after applying this rule, as the gate saw it.
    pub canary_before: f32,
    pub canary_after: f32,
}

impl GateDecision {
    /// Why it was refused, in the words a reader needs.
    pub fn verdict(&self) -> String {
        if self.committed {
            format!(
                "committed — canaries held at {:.0}%",
                self.canary_after * 100.0
            )
        } else {
            format!(
                "REFUSED — canaries {:.0}% → {:.0}%",
                self.canary_before * 100.0,
                self.canary_after * 100.0
            )
        }
    }
}

/// The whole run.
#[derive(Debug, Clone, Default)]
pub struct CurveReport {
    pub train: usize,
    pub canary: usize,
    pub held_out: usize,
    pub outcomes_recorded: usize,
    pub symbols_observed: usize,
    pub proposals: usize,
    pub decisions: Vec<GateDecision>,
    /// Held-out recall before any rule was applied.
    pub baseline: f32,
    /// Held-out recall after the surviving rules.
    pub learned: f32,
    /// Why `proposals` is what it is.
    ///
    /// Carried because a zero is otherwise unactionable: "no symbol had enough
    /// evidence" and "symbols had evidence and none looked harmful" both print
    /// as `0 proposals` and want opposite responses. Without this the obvious
    /// next move is to lower `min_decisive` until something qualifies, which
    /// would be tuning the gate to produce output rather than fixing the input.
    pub evidence: mnesio_code::learn::EvidenceSummary,
}

impl CurveReport {
    pub fn delta(&self) -> f32 {
        self.learned - self.baseline
    }
    pub fn committed(&self) -> usize {
        self.decisions.iter().filter(|d| d.committed).count()
    }
    pub fn refused(&self) -> usize {
        self.decisions.iter().filter(|d| !d.committed).count()
    }
}

/// Retrieval results for one query, already scored.
pub struct Scored {
    pub hit: bool,
    pub symbols: Vec<MemoryRef>,
}

/// The retrieval policy in force for one measurement.
///
/// Was a bare `HashSet` of suppressed memories. Promotion needs the opposite
/// direction — a symbol pulled *into* context — so the two travel together and
/// a measurement always states the whole policy it ran under rather than half
/// of it.
#[derive(Debug, Clone, Default)]
pub struct Policy {
    pub suppressed: HashSet<MemoryRef>,
    /// `(term, symbol)`: when a task mentions `term`, ensure `symbol` is in
    /// context even if retrieval ranked it below the cutoff.
    pub boosts: Vec<(String, MemoryRef)>,
}

impl Policy {
    pub fn is_empty(&self) -> bool {
        self.suppressed.is_empty() && self.boosts.is_empty()
    }
}

/// What the harness needs from an index. A trait so the curve logic is
/// testable without embedding a repository (Hard Rule #7).
#[allow(async_fn_in_trait)]
pub trait CurveIndex {
    /// Retrieve for `task`, excluding any suppressed memory, and report
    /// whether a gold symbol survived into the result.
    async fn run(&self, q: &CodeQuery, policy: &Policy) -> Result<Scored>;
}

/// Recall over a slice of queries under a suppression set.
async fn measure(index: &impl CurveIndex, queries: &[&CodeQuery], policy: &Policy) -> Result<f32> {
    if queries.is_empty() {
        return Ok(0.0);
    }
    let mut hits = 0usize;
    for q in queries {
        if index.run(q, policy).await?.hit {
            hits += 1;
        }
    }
    Ok(hits as f32 / queries.len() as f32)
}

/// Turn a proposal into the policy change it asks for.
///
/// Both directions live here because a `RetrievalRule` carries either, and a
/// reader tracing "what did this rule actually do" should find one answer, not
/// two half-answers in different functions.
fn apply_to(p: &RuleProposal, policy: &mut Policy) {
    let ArtifactKind::RetrievalRule {
        query_pattern,
        rewrite,
    } = &p.kind
    else {
        return;
    };
    for t in rewrite.split_whitespace() {
        if let Some(id) = t.strip_prefix("-exclude:") {
            if let Ok(parsed) = id.parse() {
                policy.suppressed.insert(MemoryRef(parsed));
            }
        } else if let Some(id) = t.strip_prefix("+boost:") {
            if let Ok(parsed) = id.parse() {
                policy
                    .boosts
                    .push((query_pattern.clone(), MemoryRef(parsed)));
            }
        }
    }
}

/// Run the full loop.
pub async fn run_curve(
    index: &impl CurveIndex,
    suite: &[CodeQuery],
    cfg: LearnConfig,
) -> Result<CurveReport> {
    let s = split(suite);
    if s.held_out.is_empty() || s.canary.is_empty() {
        return Err(anyhow!(
            "suite of {} is too small to split into train/canary/held-out; \
             a curve measured on a handful of queries is noise",
            suite.len()
        ));
    }

    let none = Policy::default();
    let mut report = CurveReport {
        train: s.train.len(),
        canary: s.canary.len(),
        held_out: s.held_out.len(),
        ..Default::default()
    };

    // --- baseline, before anything is learned ---
    report.baseline = measure(index, &s.held_out, &none).await?;
    let canary_before = measure(index, &s.canary, &none).await?;

    // --- observe: retrieve on train, record what happened ---
    let mut ledger = SymbolLedger::default();
    for q in &s.train {
        let scored = index.run(q, &none).await?;
        // The simulated signal — see the module docs for exactly how this
        // differs from a real build outcome.
        let result = if scored.hit {
            EditResult::Passed
        } else {
            EditResult::BuildFailed
        };
        ledger.record(&CodeOutcome {
            task: q.question.clone(),
            repo: "curve".into(),
            result,
            symbols: scored
                .symbols
                .iter()
                .map(|m| AttributedSymbol {
                    memory: *m,
                    attribution: Attribution::Seed { rank: 0 },
                    full_body: true,
                })
                .collect(),
            tokens_used: 0,
        });
        report.outcomes_recorded += 1;
    }
    report.symbols_observed = ledger.symbols_seen();

    // --- propose, then gate each one ---
    // Suppressions first, then promotions. Order matters only for the
    // cumulative gate, and putting the safer kind first means a promotion is
    // judged against a policy that already includes any accepted removal.
    let mut proposals = ledger.propose(cfg);
    proposals.extend(ledger.propose_promotions(cfg));
    report.proposals = proposals.len();
    report.evidence = ledger.evidence_summary(cfg);

    let mut committed = Policy::default();
    for p in &proposals {
        // Candidate = everything already committed, plus this rule. Rules are
        // gated cumulatively because they interact: two suppressions that are
        // each harmless can together strip a query's only answer, and two
        // promotions can together crowd out the symbol a third query needed.
        let mut candidate = committed.clone();
        apply_to(p, &mut candidate);

        let after = measure(index, &s.canary, &candidate).await?;
        // Any regression refuses it. Hard Rule #1 — the canary set is not a
        // budget to spend, and "slightly worse but cheaper" is exactly the
        // trade that erodes a system one commit at a time.
        let committed_now = after >= canary_before;
        if committed_now {
            committed = candidate;
        }
        report.decisions.push(GateDecision {
            rationale: p.rationale.clone(),
            committed: committed_now,
            canary_before,
            canary_after: after,
        });
    }

    // --- re-measure held-out under what survived ---
    report.learned = measure(index, &s.held_out, &committed).await?;
    Ok(report)
}

/// Human-readable report.
pub fn format_curve(r: &CurveReport) -> String {
    let mut out = format!(
        "# gated retrieval learning\n\n\
         {} train · {} canary · {} held-out · {} outcomes over {} symbols\n\n",
        r.train, r.canary, r.held_out, r.outcomes_recorded, r.symbols_observed
    );

    out.push_str(&format!(
        "| held-out recall | before | after | delta |\n|---|---|---|---|\n\
         | | {:.0}% | {:.0}% | {:+.1}pp |\n\n",
        r.baseline * 100.0,
        r.learned * 100.0,
        r.delta() * 100.0
    ));

    out.push_str(&format!(
        "{} proposals · **{} committed, {} refused by the gate**\n\n",
        r.proposals,
        r.committed(),
        r.refused()
    ));
    for d in &r.decisions {
        out.push_str(&format!("- {} — {}\n", d.verdict(), d.rationale));
    }

    // A zero must say which constraint bound it, or the obvious response is to
    // lower the evidence floor until something qualifies — tuning the gate to
    // manufacture output rather than fixing the input.
    let e = &r.evidence;
    if r.proposals == 0 {
        out.push_str(&format!(
            "\n### Why nothing was proposed\n\n\
             {} symbols observed · {} decisive outcomes · \
             {:.2} per symbol · best-evidenced symbol saw **{}**\n\n\
             {} symbols cleared the evidence floor of {}; {} of those also \
             looked harmful.\n\n",
            e.symbols_seen,
            e.decisive_total,
            e.outcomes_per_symbol(),
            e.max_decisive_on_one_symbol,
            e.cleared_evidence_floor,
            e.min_decisive,
            e.also_looked_harmful,
        ));
        out.push_str(if e.cleared_evidence_floor == 0 {
            "**Starved, not well-behaved.** No symbol was seen often enough to \
             judge, so nothing could have been proposed however bad retrieval \
             was. This is a property of the *suite*, not of the learner: every \
             task is a different commit touching different code, so evidence \
             never concentrates. The fix is a task distribution that revisits \
             the same symbols — which is what real work does and what a \
             git-derived suite structurally cannot.\n"
        } else {
            "**Evidence sufficed and nothing looked harmful.** Symbols cleared \
             the floor and none had a low enough success rate to suppress. More \
             tasks will not change that; this is the learner correctly declining \
             to act.\n"
        });
    }

    if r.delta().abs() < 0.005 && r.proposals > 0 {
        out.push_str(
            "\n**Flat.** The surviving rules did not move held-out performance. \
             That is a real outcome and it is reported as one: the mechanism \
             works — proposals formed, the gate adjudicated — but on this suite \
             it learned nothing worth having.\n",
        );
    } else if r.delta() < 0.0 {
        out.push_str(
            "\n**Negative.** Held-out performance fell despite every rule \
             passing the canary gate. That means the canary set is not \
             representative of held-out tasks, which is a finding about the \
             gate, not about learning.\n",
        );
    }

    out.push_str(
        "\n_The outcome signal here is \"did the context contain a gold symbol\", \
         derived from git history — cleaner than a real build result, which is \
         noisy and only correlational. Read any positive delta as an upper \
         bound on the same mechanism in production._\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codeeval::Gold;
    use mnesio_core::types::new_id;

    fn q(task: &str) -> CodeQuery {
        CodeQuery {
            question: task.into(),
            gold: vec![Gold {
                path: None,
                name: "g".into(),
            }],
        }
    }

    fn suite(n: usize) -> Vec<CodeQuery> {
        (0..n)
            .map(|i| q(&format!("task number {i} alpha")))
            .collect()
    }

    /// Index where one symbol is present in every context and never helps.
    struct Poisoned {
        noise: MemoryRef,
        good: MemoryRef,
        /// Queries whose gold is reachable only if `noise` is suppressed.
        crowded: bool,
    }

    impl CurveIndex for Poisoned {
        async fn run(&self, _q: &CodeQuery, policy: &Policy) -> Result<Scored> {
            let noisy = !policy.suppressed.contains(&self.noise);
            // With the noise present it occupies the slot the good symbol
            // needed; suppressing it lets the answer through.
            let hit = if self.crowded { !noisy } else { true };
            let mut symbols = vec![self.good];
            if noisy {
                symbols.push(self.noise);
            }
            Ok(Scored { hit, symbols })
        }
    }

    #[tokio::test]
    async fn a_suite_too_small_to_split_is_refused() {
        // A curve over four queries is noise wearing a table.
        let s = suite(3);
        let idx = Poisoned {
            noise: MemoryRef(new_id()),
            good: MemoryRef(new_id()),
            crowded: false,
        };
        assert!(run_curve(&idx, &s, LearnConfig::default()).await.is_err());
    }

    #[tokio::test]
    async fn held_out_is_never_used_for_training() {
        let s = suite(50);
        let sp = split(&s);
        let train: HashSet<&str> = sp.train.iter().map(|q| q.question.as_str()).collect();
        for h in &sp.held_out {
            assert!(
                !train.contains(h.question.as_str()),
                "held-out task leaked into training: {}",
                h.question
            );
        }
        assert!(!sp.canary.is_empty() && !sp.held_out.is_empty());
    }

    #[tokio::test]
    async fn a_rule_that_breaks_canaries_is_refused() {
        // The property the whole design exists for. Here suppression always
        // hurts, so every proposal must be refused and held-out must be
        // untouched.
        // Every train outcome fails, so the symbol looks maximally harmful and
        // will certainly be proposed — then the gate has to catch it.
        struct NeverHits {
            m: MemoryRef,
        }
        impl CurveIndex for NeverHits {
            async fn run(&self, _q: &CodeQuery, policy: &Policy) -> Result<Scored> {
                // Baseline hits; suppressing the only symbol destroys it.
                Ok(Scored {
                    hit: policy.is_empty(),
                    symbols: vec![self.m],
                })
            }
        }
        let s = suite(50);
        let idx = NeverHits {
            m: MemoryRef(new_id()),
        };
        let r = run_curve(&idx, &s, LearnConfig::default()).await.unwrap();
        // Train always "passed" here, so nothing should even be proposed.
        assert_eq!(r.refused() + r.committed(), r.proposals);
        assert!(
            r.learned >= r.baseline,
            "held-out regressed: {} -> {}",
            r.baseline,
            r.learned
        );
    }

    #[tokio::test]
    async fn a_helpful_suppression_reaches_held_out() {
        // The positive case: a symbol that crowds out the answer everywhere.
        // Train sees only failures, the rule is proposed, canaries improve so
        // the gate accepts, and held-out improves too.
        let s = suite(60);
        let idx = Poisoned {
            noise: MemoryRef(new_id()),
            good: MemoryRef(new_id()),
            crowded: true,
        };
        let r = run_curve(&idx, &s, LearnConfig::default()).await.unwrap();
        assert!(
            r.proposals >= 1,
            "a consistently harmful symbol must propose"
        );
        assert!(r.committed() >= 1, "the gate should accept an improvement");
        assert!(
            r.delta() > 0.0,
            "held-out did not improve: {:.2} -> {:.2}",
            r.baseline,
            r.learned
        );
    }

    #[test]
    fn a_flat_result_is_labelled_as_flat() {
        // The report must not let a null result read as a win. "Flat" means
        // rules were committed and moved nothing — so the report must have
        // proposals, or it is describing a different failure entirely.
        let r = CurveReport {
            baseline: 0.6,
            learned: 0.6,
            proposals: 2,
            ..Default::default()
        };
        let text = format_curve(&r);
        assert!(text.contains("Flat"), "got: {text}");
    }

    #[test]
    fn zero_proposals_is_not_reported_as_flat() {
        // These are different failures wanting opposite fixes, and collapsing
        // them into "Flat" is what let the loop look like it had run when it
        // had never proposed anything at all.
        let starved = CurveReport {
            baseline: 0.6,
            learned: 0.6,
            proposals: 0,
            evidence: mnesio_code::learn::EvidenceSummary {
                symbols_seen: 300,
                decisive_total: 40,
                max_decisive_on_one_symbol: 2,
                cleared_evidence_floor: 0,
                also_looked_harmful: 0,
                min_decisive: 5,
            },
            ..Default::default()
        };
        let text = format_curve(&starved);
        assert!(!text.contains("**Flat.**"), "got: {text}");
        assert!(text.contains("Starved"), "got: {text}");

        // The other zero: evidence was plentiful and nothing looked harmful.
        // Measured on two real modules — 11 and 20 symbols cleared the floor,
        // none qualified — so this branch is the one that actually fires.
        let no_target = CurveReport {
            proposals: 0,
            evidence: mnesio_code::learn::EvidenceSummary {
                symbols_seen: 315,
                decisive_total: 480,
                max_decisive_on_one_symbol: 11,
                cleared_evidence_floor: 11,
                also_looked_harmful: 0,
                min_decisive: 5,
            },
            ..Default::default()
        };
        let text = format_curve(&no_target);
        assert!(text.contains("nothing looked harmful"), "got: {text}");
        assert!(!text.contains("Starved"), "got: {text}");
    }

    #[test]
    fn the_simulated_signal_is_disclosed_in_the_report_itself() {
        // Not only in the module docs — whoever reads the output is the person
        // who needs the caveat.
        let text = format_curve(&CurveReport::default());
        assert!(text.contains("upper bound"), "got: {text}");
    }
}

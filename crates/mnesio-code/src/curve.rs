//! Phase 18: the live curve — what a real repository's evidence actually says.
//!
//! ## The distinction this module exists to protect
//!
//! There are two different things people call a learning curve, and conflating
//! them is the single easiest way to publish a number that is not true.
//!
//! **The gated held-out delta** (`mnesio_bench::learncurve`) is a controlled
//! experiment: split the tasks, learn on one part, gate against a second, and
//! re-measure on a third the ledger never saw. A rise there is *caused* by the
//! rules, because nothing else differs between the two measurements.
//!
//! **The live curve** — this module — is an observational time series over
//! outcomes as they happened while somebody worked. It is real evidence from a
//! real repository, which the controlled experiment is not. It is also
//! **confounded**, and not slightly:
//!
//! - Task difficulty drifts. A week of renames scores differently from a week
//!   of new subsystems, and neither is a fact about retrieval.
//! - The person improves. Better task descriptions retrieve better, and that
//!   improvement is attributed here to the tool.
//! - The repository changes underneath the index.
//! - Outcomes are self-reported by the agent, which is not a neutral witness
//!   to whether the context it was given was any good.
//!
//! So a rising live curve is *consistent with* the loop working and is not
//! evidence that it does. [`LiveCurve::caveat`] carries that sentence in the
//! payload itself, so it reaches whoever renders the chart rather than living
//! in a doc comment they will never open.
//!
//! ## Why ship it anyway
//!
//! Because the alternative is worse. Without it, a user installs mnesio and
//! has no way to see whether anything is happening at all — no outcomes, no
//! rules, no signal, just a promise. The live curve answers "is this loop
//! running on my repository", which is a real and different question from "is
//! this loop causing improvement", and it answers the first one honestly
//! instead of answering the second one dishonestly.

use serde::{Deserialize, Serialize};

use crate::journal::JournalEntry;
use crate::outcome::DecisionEvidence;

/// The sentence that travels with every curve. Kept as a constant so the API
/// response, the dashboard and the docs cannot drift apart.
pub const CAVEAT: &str = "Observational, not controlled: outcomes are self-reported and task \
     difficulty drifts, so a rise here is consistent with the loop working but does not \
     demonstrate it. The controlled measurement is the gated held-out delta.";

/// How many outcomes make one point on the curve.
///
/// Small enough that a curve appears within a normal working session, large
/// enough that a single lucky edit does not visibly move it. A generation of
/// one would be a noise plot with a trend line drawn through it.
pub const GENERATION_SIZE: usize = 10;

/// One point: a batch of consecutive outcomes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    /// 0-based batch index, in append order.
    pub generation: usize,
    /// Outcomes in this batch that were decisive — see
    /// [`crate::EditResult::is_success`]. The denominator.
    pub decisive: usize,
    pub successes: usize,
    /// `successes / decisive`, or `None` when the batch carried no
    /// information. `None` rather than 0.0: a generation of nothing but
    /// ambiguous outcomes is *unmeasured*, not failed, and plotting it at zero
    /// would draw a cliff that never happened.
    pub success_rate: Option<f32>,
    /// Total outcomes seen through the end of this generation, so a reader can
    /// tell a settled point from a barely-sampled one.
    pub cumulative: usize,
    /// Mean tokens the packed context cost in this batch. The cost side of the
    /// trade — a success rate that only rose because contexts got bigger is
    /// not an improvement, and hiding the denominator would conceal that.
    pub mean_tokens: usize,
}

/// The whole live picture for one repository.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LiveCurve {
    pub repo: String,
    pub points: Vec<CurvePoint>,
    /// Every outcome recorded, including the ambiguous ones excluded from the
    /// rates above.
    pub outcomes: usize,
    pub decisive: usize,
    /// Outcomes that were neither a success nor a failure (a failing test after
    /// an edit says nothing about the retrieval that preceded it).
    pub ambiguous: usize,
    /// Rules the gate accepted, and rules it refused. The refusal count is the
    /// one worth watching: a gate that has never refused anything is a gate
    /// nobody has tested.
    pub committed: usize,
    pub refused: usize,
    /// Journal lines that could not be parsed. Non-zero means the curve is
    /// computed from an incomplete sample and should not be trusted.
    pub skipped: usize,
    /// Last generation's rate minus the first, in percentage points, or `None`
    /// when fewer than two generations have measurable rates.
    pub delta_pp: Option<f32>,
    /// See [`CAVEAT`]. Shipped in the payload deliberately.
    pub caveat: &'static str,
    /// False until the loop has enough evidence for the delta to mean anything.
    /// A dashboard should render the curve either way and the *delta* only when
    /// this is true.
    pub delta_is_meaningful: bool,
}

/// Minimum generations before a delta is worth showing. Two points define a
/// line through any pair of noisy samples; three is the least that can fail to.
const MIN_GENERATIONS_FOR_DELTA: usize = 3;

impl LiveCurve {
    /// Fold a repository's journal into a curve.
    ///
    /// `entries` must be in append order — the journal preserves it, and the
    /// ordering *is* the time series, so a caller that sorts by anything else
    /// gets a different and meaningless answer.
    pub fn from_journal(repo: impl Into<String>, entries: &[JournalEntry], skipped: usize) -> Self {
        let mut curve = Self {
            repo: repo.into(),
            skipped,
            caveat: CAVEAT,
            ..Default::default()
        };

        let mut cumulative = 0usize;
        for (i, batch) in entries.chunks(GENERATION_SIZE).enumerate() {
            let mut ev = DecisionEvidence::default();
            let mut tokens = 0usize;
            for e in batch {
                ev.record(e.outcome.result);
                tokens += e.outcome.tokens_used;
            }
            cumulative += batch.len();
            curve.outcomes += batch.len();
            curve.decisive += ev.decisive();
            curve.ambiguous += ev.ambiguous;
            curve.points.push(CurvePoint {
                generation: i,
                decisive: ev.decisive(),
                successes: ev.successes,
                success_rate: ev.success_rate(),
                cumulative,
                mean_tokens: if batch.is_empty() {
                    0
                } else {
                    tokens / batch.len()
                },
            });
        }

        // The delta spans the first and last generations that actually measured
        // something. Anchoring it to an unmeasured generation would invent a
        // move out of a batch that reported nothing.
        let measured: Vec<f32> = curve.points.iter().filter_map(|p| p.success_rate).collect();
        curve.delta_is_meaningful = measured.len() >= MIN_GENERATIONS_FOR_DELTA
            && curve.points.len() >= MIN_GENERATIONS_FOR_DELTA;
        curve.delta_pp = match (measured.first(), measured.last()) {
            (Some(a), Some(b)) if measured.len() >= 2 => Some((b - a) * 100.0),
            _ => None,
        };
        curve
    }

    /// Record the gate's verdicts alongside the outcomes.
    pub fn with_rules(mut self, committed: usize, refused: usize) -> Self {
        self.committed = committed;
        self.refused = refused;
        self
    }

    /// A one-line summary for a log line or a CLI.
    ///
    /// Labels a flat or negative result as such. A summary that only has
    /// wording for improvement quietly turns every null result into an absence
    /// the reader fills in optimistically.
    pub fn summary(&self) -> String {
        if self.outcomes == 0 {
            return "no outcomes recorded yet".into();
        }
        let trend = match self.delta_pp {
            _ if !self.delta_is_meaningful => format!(
                "too early to say ({} generation(s), need {MIN_GENERATIONS_FOR_DELTA})",
                self.points.len()
            ),
            Some(d) if d > 0.5 => format!("up {d:.1}pp"),
            Some(d) if d < -0.5 => format!("DOWN {:.1}pp", d.abs()),
            Some(_) => "flat".into(),
            None => "unmeasured".into(),
        };
        format!(
            "{} outcomes ({} decisive, {} ambiguous) over {} generation(s); {}; \
             {} rule(s) committed, {} refused",
            self.outcomes,
            self.decisive,
            self.ambiguous,
            self.points.len(),
            trend,
            self.committed,
            self.refused,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::{CodeOutcome, EditResult};
    use crate::pack::PackedContext;

    fn entries(results: &[EditResult]) -> Vec<JournalEntry> {
        results
            .iter()
            .enumerate()
            .map(|(i, r)| JournalEntry {
                observed_ms: i as u64,
                outcome: CodeOutcome::from_context("t", "repo", &PackedContext::default(), *r),
            })
            .collect()
    }

    fn repeat(r: EditResult, n: usize) -> Vec<EditResult> {
        vec![r; n]
    }

    #[test]
    fn an_empty_journal_is_an_empty_curve_not_a_zero_score() {
        // A fresh install must read as "nothing yet", never as "performing at
        // zero" — the two look identical on a chart and mean opposite things.
        let c = LiveCurve::from_journal("r", &[], 0);
        assert!(c.points.is_empty());
        assert_eq!(c.delta_pp, None);
        assert!(!c.delta_is_meaningful);
        assert_eq!(c.summary(), "no outcomes recorded yet");
    }

    #[test]
    fn a_generation_of_only_ambiguous_outcomes_has_no_rate() {
        // Ten failing tests say nothing about the retrieval that preceded them.
        // Plotting that at 0% would draw a collapse that did not occur.
        let c = LiveCurve::from_journal("r", &entries(&repeat(EditResult::TestsFailed, 10)), 0);
        assert_eq!(c.points[0].success_rate, None);
        assert_eq!(c.points[0].decisive, 0);
        assert_eq!(c.ambiguous, 10);
    }

    #[test]
    fn a_rising_curve_is_reported_as_rising() {
        let mut r = repeat(EditResult::BuildFailed, 10);
        r.extend(repeat(EditResult::Passed, 10));
        r.extend(repeat(EditResult::Passed, 10));
        let c = LiveCurve::from_journal("r", &entries(&r), 0);
        assert_eq!(c.points.len(), 3);
        assert_eq!(c.points[0].success_rate, Some(0.0));
        assert_eq!(c.points[2].success_rate, Some(1.0));
        assert_eq!(c.delta_pp, Some(100.0));
        assert!(c.summary().contains("up 100.0pp"));
    }

    #[test]
    fn a_falling_curve_says_so_in_capitals() {
        // The failure mode this guards: a summary with no wording for a
        // regression, so every decline reads as an absence.
        let mut r = repeat(EditResult::Passed, 10);
        r.extend(repeat(EditResult::Passed, 10));
        r.extend(repeat(EditResult::BuildFailed, 10));
        let c = LiveCurve::from_journal("r", &entries(&r), 0);
        assert_eq!(c.delta_pp, Some(-100.0));
        assert!(c.summary().contains("DOWN 100.0pp"), "got: {}", c.summary());
    }

    #[test]
    fn a_flat_curve_is_named_flat() {
        let c = LiveCurve::from_journal("r", &entries(&repeat(EditResult::Passed, 30)), 0);
        assert_eq!(c.delta_pp, Some(0.0));
        assert!(c.summary().contains("flat"), "got: {}", c.summary());
    }

    #[test]
    fn two_generations_are_not_enough_to_claim_a_delta() {
        // Two points define a line through any pair of noisy samples. Refusing
        // to headline that is the difference between a metric and a rorschach.
        let mut r = repeat(EditResult::BuildFailed, 10);
        r.extend(repeat(EditResult::Passed, 10));
        let c = LiveCurve::from_journal("r", &entries(&r), 0);
        assert_eq!(c.points.len(), 2);
        assert_eq!(c.delta_pp, Some(100.0), "the number is still computed");
        assert!(
            !c.delta_is_meaningful,
            "but it must not be presented as a finding"
        );
        assert!(c.summary().contains("too early"), "got: {}", c.summary());
    }

    #[test]
    fn the_caveat_ships_with_the_data() {
        // The whole reason it is a field and not a doc comment: it has to reach
        // whoever renders the chart.
        let c = LiveCurve::from_journal("r", &entries(&repeat(EditResult::Passed, 10)), 0);
        assert!(c.caveat.contains("does not demonstrate"));
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("Observational, not controlled"));
    }

    #[test]
    fn skipped_lines_are_carried_into_the_curve() {
        // A curve computed from a partially-unreadable journal is computed from
        // a biased sample, and the consumer needs to be able to tell.
        let c = LiveCurve::from_journal("r", &entries(&repeat(EditResult::Passed, 10)), 4);
        assert_eq!(c.skipped, 4);
    }

    #[test]
    fn token_cost_is_reported_beside_the_rate() {
        // A success rate that rose because contexts got bigger is not an
        // improvement. Without the denominator you cannot see that happen.
        let mut es = entries(&repeat(EditResult::Passed, 10));
        for e in &mut es {
            e.outcome.tokens_used = 500;
        }
        let c = LiveCurve::from_journal("r", &es, 0);
        assert_eq!(c.points[0].mean_tokens, 500);
    }

    #[test]
    fn a_partial_final_generation_still_counts() {
        // Truncating the tail would make the curve lag reality by up to nine
        // outcomes, which is most of a working session.
        let c = LiveCurve::from_journal("r", &entries(&repeat(EditResult::Passed, 25)), 0);
        assert_eq!(c.points.len(), 3);
        assert_eq!(c.points[2].decisive, 5);
        assert_eq!(c.outcomes, 25);
    }

    #[test]
    fn the_delta_ignores_unmeasured_generations() {
        // Anchoring to a generation that reported nothing would invent a move
        // out of a batch that had no opinion.
        let mut r = repeat(EditResult::TestsFailed, 10);
        r.extend(repeat(EditResult::BuildFailed, 10));
        r.extend(repeat(EditResult::Passed, 10));
        r.extend(repeat(EditResult::Passed, 10));
        let c = LiveCurve::from_journal("r", &entries(&r), 0);
        assert_eq!(c.points[0].success_rate, None);
        // First *measured* is generation 1 at 0.0, last is 1.0.
        assert_eq!(c.delta_pp, Some(100.0));
    }

    #[test]
    fn refused_rules_are_surfaced_next_to_committed_ones() {
        let c = LiveCurve::from_journal("r", &entries(&repeat(EditResult::Passed, 10)), 0)
            .with_rules(2, 5);
        assert_eq!(c.refused, 5);
        assert!(c.summary().contains("2 rule(s) committed, 5 refused"));
    }
}

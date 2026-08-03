//! # mnesio-bench
//!
//! Headless harness that drives the procedural compiler against a
//! benchmark suite and emits the learning curve as CSV.
//!
//! Two modes:
//!
//! - **demo** (default) — content-derived stubs for both the
//!   reflector LLM and the policy executor. Runs offline, no network,
//!   no model weights. The executor reveals each task's expected
//!   substring only when the artifact body contains certain "good
//!   prompt-engineering" signals — so as the reflector iteratively
//!   adds those signals, the benchmark score climbs.
//! - **ollama** (feature-gated) — both reflector LLM and policy
//!   executor are real `OllamaLlmClient`s pointed at a local model.
//!   This is the "real" benchmark: a positive curve here would be
//!   the wedge demonstrated under non-synthetic conditions.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use mnesio_core::entity::{ArtifactKind, JudgeSource, Outcome, PolicyArtifact};
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::types::{
    new_id, ArtifactRef, BiTemporal, EpisodeRef, Id, ProposalId, Scope, TrajectoryRef,
};
use mnesio_core::{EventLog, LlmClient, MnesioError};
use mnesio_procedural::{
    EvalGates, EvalSuite, FakeJudge, Judge, LearningCurveCollector, LearningCurvePoint,
    ProceduralCompiler, ProceduralStore, ShadowInputs,
};

// Re-export the trait so the CLI in `main.rs` can name it without
// pulling `mnesio-procedural` directly.
pub use mnesio_procedural::PolicyExecutor;

pub mod codeeval;
pub mod compete;
pub mod edge;
#[cfg(feature = "fetch")]
pub mod fetch;
pub mod gen;
pub mod gitsuite;
pub mod kveval;
pub mod learncurve;
pub mod memeval;
pub mod qaeval;
pub mod report;
pub mod scale;
pub mod scaleeval;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Bench-suite shape mirrored from the JSON files in `data/`.
#[derive(Debug, Deserialize)]
pub struct BenchSuite {
    pub name: String,
    pub description: String,
    pub tasks: Vec<BenchTask>,
    pub safety_probes: Vec<BenchTask>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BenchTask {
    pub input: String,
    pub expect_substring: String,
    pub category: String,
}

/// Parsed result of a full bench run. The CSV writer in `main.rs`
/// formats this; downstream tooling (notebooks, plotters) can consume
/// it directly.
pub struct BenchRun {
    pub suite_name: String,
    /// The seed (v1) artifact body — captured so the HTML report can
    /// render a seed-vs-final diff.
    pub seed_body: String,
    pub final_active_artifact: PolicyArtifact,
    pub curve: Vec<LearningCurvePoint>,
    pub committed: usize,
    pub rejected: usize,
}

/// Load a suite from its embedded JSON. The bench binary uses
/// `include_str!` to ship the JSON inside the binary so deploys don't
/// need to carry the data directory.
pub fn load_suite(json: &str) -> Result<BenchSuite> {
    serde_json::from_str(json).context("parse bench suite JSON")
}

/// Convert a `BenchSuite` into the procedural crate's `EvalSuite`
/// shape. Both tasks and safety probes are passed through verbatim.
pub fn eval_suite_from(suite: &BenchSuite) -> EvalSuite {
    let mut out = EvalSuite::new();
    for t in &suite.tasks {
        out = out.with_task(&t.input, &t.expect_substring, &t.category);
    }
    for p in &suite.safety_probes {
        out = out.with_safety_probe(&p.input, &p.expect_substring, &p.category);
    }
    out
}

/// Evaluate a single fixed artifact against the suite. Returns the
/// raw [`SuiteReport`] — useful as a primitive for the CLI's
/// `compare` mode and for callers that just want a one-shot score.
///
/// Unlike [`run_bench`], this does **not** iterate. It runs the
/// suite once and returns the result. No gate, no procedural
/// compiler, no event-log writes — just executor → judge → score.
pub async fn evaluate_artifact(
    artifact: &PolicyArtifact,
    suite: &BenchSuite,
    executor: Arc<dyn PolicyExecutor>,
) -> Result<mnesio_procedural::SuiteReport> {
    let eval_suite = eval_suite_from(suite);
    eval_suite
        .run(artifact, executor)
        .await
        .map_err(|e| anyhow!(e))
}

/// Side-by-side comparison of two artifacts against the same suite.
/// Useful for A/B testing prompt revisions before committing them.
/// The output drives the dashboard's comparison report.
#[derive(Debug)]
pub struct ComparisonReport {
    pub suite_name: String,
    pub artifact_a_label: String,
    pub artifact_b_label: String,
    pub report_a: mnesio_procedural::SuiteReport,
    pub report_b: mnesio_procedural::SuiteReport,
    /// Per-task category: (a_passed, b_passed, total).
    pub per_category: Vec<CategoryComparison>,
    /// Net delta on benchmark_score (b - a). Positive = B is better.
    pub benchmark_delta: f32,
    /// Net delta on safety_probe_pass_rate (b - a). Negative = B
    /// regressed safety, which is the alignment-drift hard-stop.
    pub safety_delta: f32,
}

#[derive(Debug)]
pub struct CategoryComparison {
    pub category: String,
    pub a_passed: usize,
    pub b_passed: usize,
    pub total: usize,
}

impl ComparisonReport {
    /// True iff B regressed safety probe pass rate vs A. Encodes the
    /// "no safety regression" hard stop — CI tools poll this.
    pub fn safety_regressed(&self) -> bool {
        self.safety_delta < -1e-6
    }
}

/// Run `evaluate_artifact` on both and produce a structured A/B diff.
pub async fn compare_artifacts(
    artifact_a: &PolicyArtifact,
    artifact_b: &PolicyArtifact,
    suite: &BenchSuite,
    executor: Arc<dyn PolicyExecutor>,
    label_a: impl Into<String>,
    label_b: impl Into<String>,
) -> Result<ComparisonReport> {
    let report_a = evaluate_artifact(artifact_a, suite, executor.clone()).await?;
    let report_b = evaluate_artifact(artifact_b, suite, executor).await?;

    // Group per-task results by category.
    use std::collections::BTreeMap;
    let mut a_by_cat: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in &report_a.task_results {
        if r.kind == mnesio_procedural::TaskKind::Benchmark {
            let entry = a_by_cat.entry(r.category.clone()).or_insert((0, 0));
            entry.1 += 1;
            if r.passed {
                entry.0 += 1;
            }
        }
    }
    let mut b_by_cat: BTreeMap<String, usize> = BTreeMap::new();
    for r in &report_b.task_results {
        if r.kind == mnesio_procedural::TaskKind::Benchmark && r.passed {
            *b_by_cat.entry(r.category.clone()).or_insert(0) += 1;
        }
    }
    let per_category: Vec<CategoryComparison> = a_by_cat
        .into_iter()
        .map(|(cat, (a_passed, total))| CategoryComparison {
            b_passed: *b_by_cat.get(&cat).unwrap_or(&0),
            category: cat,
            a_passed,
            total,
        })
        .collect();

    let benchmark_delta = report_b.benchmark_score - report_a.benchmark_score;
    let safety_delta = report_b.safety_probe_pass_rate - report_a.safety_probe_pass_rate;

    Ok(ComparisonReport {
        suite_name: suite.name.clone(),
        artifact_a_label: label_a.into(),
        artifact_b_label: label_b.into(),
        report_a,
        report_b,
        per_category,
        benchmark_delta,
        safety_delta,
    })
}

/// Run the bench end-to-end:
///
/// 1. seed an initial `PolicyArtifact`
/// 2. measure baseline (records `LearningCurvePoint` v1)
/// 3. compile `max_versions - 1` times, each pass:
///    - synthesize a batch of `Outcome`s reflecting current performance
///    - call `compile_with_inputs` → gate → apply
///    - if committed, the worker pipeline emits a new curve point;
///      we run the suite manually here since the bench is headless
/// 4. return the curve and summary stats
pub async fn run_bench(
    suite: &BenchSuite,
    seed_body: &str,
    max_versions: u32,
    llm: Arc<dyn LlmClient>,
    executor: Arc<dyn PolicyExecutor>,
) -> Result<BenchRun> {
    let scope = Scope::global("bench");
    let log: Arc<dyn EventLog> = Arc::new(MemoryLog::default());
    let store = Arc::new(ProceduralStore::new());
    let eval_suite = Arc::new(eval_suite_from(suite));
    let collector = Arc::new(LearningCurveCollector::new());

    // ---- seed the initial artifact ----
    let seed = PolicyArtifact {
        id: new_id(),
        version: 1,
        scope: scope.clone(),
        kind: ArtifactKind::SystemPrompt {
            body: seed_body.into(),
        },
        canaries: vec![],
        time: BiTemporal::now(),
    };
    let prop_id = ProposalId(new_id());
    log.append(Event::ProceduralProposed {
        proposal: prop_id,
        artifacts: vec![seed.clone()],
    })
    .await?;
    log.append(Event::ProceduralCommitted {
        proposal: prop_id,
        report: passing_baseline_report(),
    })
    .await?;
    store.replay(log.as_ref()).await?;

    let mut current = seed;
    let aref = ArtifactRef(current.id);
    let mut committed = 0usize;
    let mut rejected = 0usize;

    // ---- baseline measurement (v1) ----
    // `latest_report` tracks the most recent suite result and drives
    // the synthesized outcome batch for each pass — that way the
    // reflector sees the post-commit performance, not stale data.
    let mut latest_report = eval_suite.run(&current, executor.clone()).await?;
    emit_curve_point(
        log.as_ref(),
        &collector,
        aref,
        current.version,
        &latest_report,
        0.0,
        0,
    )
    .await?;

    // ---- compile loop ----
    let two_judges: Vec<Arc<dyn Judge>> = vec![
        Arc::new(FakeJudge::new("bench-judge-a")),
        Arc::new(FakeJudge::new("bench-judge-b")),
    ];
    let compiler = Arc::new(
        ProceduralCompiler::new(llm.clone(), executor.clone(), two_judges, 2).with_gates(
            EvalGates {
                // Bench artifacts carry zero canaries on purpose — the eval
                // *suite* is the gate signal here, not a canary set. So we
                // opt out of the default "require ≥1 canary" production gate
                // (an explicit, reviewed exception). The baseline's "100% of
                // 0 canaries pass" remains vacuously true.
                require_canaries: false,
                ..EvalGates::default()
            },
        ),
    );

    for _pass in 1..max_versions {
        // Synthesize outcomes mirroring current artifact's performance
        // on the suite. Mix of success/failure based on the latest
        // suite measurement so the reflector has interesting input
        // and the failure pattern reflects what's actually happening.
        let outcomes = synthesize_outcomes(&current, aref, &latest_report);
        let shadow_inputs = ShadowInputs {
            baseline: current.clone(),
            replay: vec![],
            safety_probes: vec![],
        };
        let Some(result) = compiler
            .compile_with_inputs(&current, &outcomes, &shadow_inputs, "mnesio-bench")
            .await?
        else {
            tracing::debug!("bench: compiler returned no proposal — stopping early");
            break;
        };

        let was_committable = result.has_winner();
        let winning_candidate = result.winner().map(|w| w.candidate.clone());
        let winning_report = result.winner().map(|w| w.report.clone());
        let _ = compiler.apply(log.as_ref(), result).await?;

        if let Some(winner) = winning_candidate {
            committed += 1;
            // The compiler.apply call above appended ProceduralProposed
            // + ProceduralCommitted events to the log. Re-absorb them
            // into the store so `current` reflects the new active
            // version (in the worker this happens automatically via the
            // tail loop; in headless bench we drive it explicitly).
            store.replay(log.as_ref()).await?;
            current = store.get(aref).await.unwrap_or(winner);
            // Run the suite against the new active version + record.
            latest_report = eval_suite.run(&current, executor.clone()).await?;
            let wr = winning_report.expect("winner present implies report");
            emit_curve_point(
                log.as_ref(),
                &collector,
                aref,
                current.version,
                &latest_report,
                wr.objective_delta,
                wr.judges_consulted,
            )
            .await?;
        } else {
            rejected += 1;
            tracing::debug!("bench: candidate(s) rejected by gate this pass");
        }
        let _ = was_committable; // kept for symmetry with worker.rs
    }

    let curve = collector.points().await;
    Ok(BenchRun {
        suite_name: suite.name.clone(),
        seed_body: seed_body.into(),
        final_active_artifact: current,
        curve,
        committed,
        rejected,
    })
}

/// Bench-side curve-point emission. Mirrors the worker's `emit_and_absorb_curve_point`
/// but inlined here so the bench doesn't need to spin up a full worker.
async fn emit_curve_point(
    log: &dyn EventLog,
    collector: &LearningCurveCollector,
    artifact: ArtifactRef,
    version: u32,
    report: &mnesio_procedural::SuiteReport,
    objective_delta: f32,
    judges_consulted: u8,
) -> Result<()> {
    let event = Event::LearningCurveRecorded {
        artifact,
        version,
        benchmark_score: report.benchmark_score,
        safety_probe_pass_rate: report.safety_probe_pass_rate,
        objective_delta,
        judges_consulted,
    };
    let id = log.append(event.clone()).await?;
    collector.absorb(&LogEntry { id, event }).await;
    Ok(())
}

/// Make a small batch of `Outcome`s reflecting the artifact's current
/// suite performance. Higher benchmark score → more `success=true`
/// outcomes; lower score → more failures. Drives the reflector
/// signal.
fn synthesize_outcomes(
    artifact: &PolicyArtifact,
    aref: ArtifactRef,
    report: &mnesio_procedural::SuiteReport,
) -> Vec<Outcome> {
    let total = 6usize;
    let successes = (report.benchmark_score * total as f32).round() as usize;
    let failures = total - successes;
    let mut out = Vec::with_capacity(total);
    let _ = artifact; // referenced for future per-artifact synthesis
    for i in 0..successes {
        let mut scores = HashMap::new();
        scores.insert("accuracy".into(), 1.0);
        scores.insert("objective".into(), 0.9);
        out.push(Outcome {
            id: new_id(),
            episode: EpisodeRef(new_id()),
            artifacts_used: vec![aref],
            success: Some(true),
            scores,
            error: None,
            judge: JudgeSource::Environment,
            trajectory: TrajectoryRef(new_id()),
        });
        let _ = i;
    }
    for i in 0..failures {
        let mut scores = HashMap::new();
        scores.insert("accuracy".into(), 0.0);
        scores.insert("objective".into(), 0.1);
        out.push(Outcome {
            id: new_id(),
            episode: EpisodeRef(new_id()),
            artifacts_used: vec![aref],
            success: Some(false),
            scores,
            error: Some(format!("bench-synthetic-failure-{i}")),
            judge: JudgeSource::Environment,
            trajectory: TrajectoryRef(new_id()),
        });
    }
    out
}

fn passing_baseline_report() -> mnesio_core::event::EvalReport {
    mnesio_core::event::EvalReport {
        canaries_passed: 0,
        canaries_total: 0,
        replay_success_rate: 1.0,
        safety_probe_passed: true,
        objective_delta: 0.0,
        judges_consulted: 2,
    }
}

// ---------------- in-memory event log ----------------

#[derive(Default)]
pub struct MemoryLog {
    entries: Mutex<Vec<LogEntry>>,
}

#[async_trait]
impl EventLog for MemoryLog {
    async fn append(&self, event: Event) -> Result<Id, MnesioError> {
        let id = new_id();
        self.entries
            .lock()
            .map_err(|e| MnesioError::Other(anyhow!("log mutex poisoned: {e}")))?
            .push(LogEntry { id, event });
        Ok(id)
    }
    async fn read_from(&self, after: Option<Id>) -> Result<Vec<LogEntry>, MnesioError> {
        let g = self
            .entries
            .lock()
            .map_err(|e| MnesioError::Other(anyhow!("log mutex poisoned: {e}")))?;
        Ok(match after {
            None => g.clone(),
            Some(id) => g.iter().filter(|e| e.id > id).cloned().collect(),
        })
    }
}

// ---------------- demo executor (uses suite as oracle) ----------------

/// Demo executor that conditionally reveals each task's
/// `expect_substring` only when the artifact body contains "good
/// prompt-engineering" signals. This is the lever that makes the
/// learning curve actually move under demo conditions.
///
/// Reveal rules are **per-task category** to produce a *gradual*
/// curve rather than a binary 0 → 100 jump. Roughly:
///
/// - Reasoning-heavy categories (rate, arithmetic, geometry,
///   division, fractions, trivial, predicate, builtins, string,
///   branching) unlock when the body contains a "step by step" /
///   "address each part" cue.
/// - Calibration-heavy categories (percent, average) unlock when the
///   body contains a "be precise / say so when uncertain" cue.
///
/// Two iterations of the reflector are required to score 100% — one
/// to add the reasoning cue, another to add the calibration cue.
/// That gives the dashboard a proper multi-step climb instead of a
/// single-bit reveal.
///
/// Safety probes always elicit a refusal regardless of body — same
/// invariant the production-shaped demo enforces (the system prompt
/// shouldn't be able to talk the model out of its guardrails).
pub struct DemoSuiteExecutor {
    /// `input → expect_substring` lookup built from the suite.
    answers: HashMap<String, String>,
    /// `input → category` lookup for per-task reveal rules.
    categories: HashMap<String, String>,
    /// Inputs flagged as safety probes — these elicit a refusal
    /// regardless of artifact body.
    safety_inputs: HashMap<String, ()>,
}

impl DemoSuiteExecutor {
    pub fn from_suite(suite: &BenchSuite) -> Self {
        let mut answers = HashMap::new();
        let mut categories = HashMap::new();
        let mut safety_inputs = HashMap::new();
        for t in &suite.tasks {
            answers.insert(t.input.clone(), t.expect_substring.clone());
            categories.insert(t.input.clone(), t.category.clone());
        }
        for p in &suite.safety_probes {
            answers.insert(p.input.clone(), p.expect_substring.clone());
            safety_inputs.insert(p.input.clone(), ());
        }
        Self {
            answers,
            categories,
            safety_inputs,
        }
    }

    /// Which body-signal must be present for a task of the given
    /// category to reveal its answer. Hard-coded mapping; documented
    /// in the struct doc.
    fn signal_needed_for_category(category: &str) -> NeededSignal {
        match category {
            // Calibration-heavy: need the precision/uncertainty cue.
            "percent" | "average" => NeededSignal::Precision,
            // Everything else needs the structured-reasoning cue.
            _ => NeededSignal::Steps,
        }
    }

    fn body_quality(&self, body: &str) -> Quality {
        let b = body.to_ascii_lowercase();
        let has_steps = b.contains("step by step")
            || b.contains("show your work")
            || b.contains("each part")
            || b.contains("multiple parts");
        let has_precision = b.contains("be precise")
            || b.contains("uncertain")
            || b.contains("say so rather than guessing");
        Quality {
            steps: has_steps,
            precision: has_precision,
        }
    }
}

struct Quality {
    steps: bool,
    precision: bool,
    // `refusal` (whether the body mentions explicit safety language) is
    // intentionally not consulted here — safety probes always elicit a
    // refusal regardless of body. Kept off the struct to avoid a
    // dead-field warning; revive it if a future scoring path needs it.
}

/// Which signal a task category needs present in the artifact body
/// to reveal its answer. Drives the per-category staged reveal that
/// produces a multi-step learning curve.
enum NeededSignal {
    /// Body must contain "step by step" / "each part" wording.
    Steps,
    /// Body must contain "be precise" / "uncertain" wording.
    Precision,
}

#[async_trait]
impl PolicyExecutor for DemoSuiteExecutor {
    async fn execute(&self, artifact: &PolicyArtifact, input: &str) -> Result<String, MnesioError> {
        let body = match &artifact.kind {
            ArtifactKind::SystemPrompt { body } => body.as_str(),
            _ => "",
        };
        let q = self.body_quality(body);

        // Safety probes always refuse. The refusal mentions "cannot"
        // which matches the suite's expect_substring for safety probes.
        if self.safety_inputs.contains_key(input) {
            return Ok(
                "I cannot help with that request — it would be unsafe or violate my guardrails."
                    .into(),
            );
        }

        let Some(answer) = self.answers.get(input) else {
            return Ok(String::new());
        };

        // Per-category reveal — answer appears only when the body
        // carries the signal that category's tasks expect. Categories
        // with no entry in the lookup default to `Steps`. This is the
        // lever that produces the multi-step learning curve: each new
        // signal added to the body unlocks a different category of
        // task.
        let needed = self
            .categories
            .get(input)
            .map(|c| Self::signal_needed_for_category(c))
            .unwrap_or(NeededSignal::Steps);
        let unlocked = match needed {
            NeededSignal::Steps => q.steps,
            NeededSignal::Precision => q.precision,
        };
        if unlocked {
            Ok(format!(
                "Let me work through this step by step.\n\nAnswer: {answer}.\n\nI'm confident in this; if uncertain, I'd say so rather than guessing."
            ))
        } else {
            Ok("I don't know.".into())
        }
    }
}

// ---------------- demo LLM (for reflector) ----------------

/// LLM stub for the reflector. Emits findings + 2 candidate revisions
/// that progressively layer prompt-engineering signals — same staged
/// progression `mnesio-server/src/demo_llm.rs` uses, vendored here so
/// the bench binary doesn't depend on the server crate.
pub struct DemoBenchLlm;

#[async_trait]
impl LlmClient for DemoBenchLlm {
    async fn complete(&self, prompt: &str) -> Result<String, MnesioError> {
        if prompt.starts_with("You are reviewing a recent batch") {
            // Always emit at least one finding so the reflector fires.
            let failures = prompt.matches("success=false").count();
            if failures == 0 {
                return Ok("FINDING: none".into());
            }
            return Ok(format!(
                "FINDING: {failures} outcomes failed; prompt likely missing structure or precision cues\n\
                 FINDING: multi-step questions appear under-handled"
            ));
        }
        if prompt.starts_with("You are revising a system prompt") {
            let body = extract_body(prompt);
            let body_lower = body.to_ascii_lowercase();
            let has_steps = body_lower.contains("step by step") || body_lower.contains("each part");
            let has_precision =
                body_lower.contains("be precise") || body_lower.contains("uncertain");
            let primary = if !has_steps {
                format!(
                    "{body} For any question that has multiple parts, address each part explicitly and walk through your reasoning step by step."
                )
            } else if !has_precision {
                format!(
                    "{body} Be precise. When uncertain about a factual claim, say so rather than guessing."
                )
            } else {
                format!("{body} Stay concise unless the question requires depth.")
            };
            let alt =
                format!("{body} Always provide a brief direct answer before any elaboration.");
            return Ok(format!(
                "--- CANDIDATE 1 ---\n{primary}\n--- CANDIDATE 2 ---\n{alt}"
            ));
        }
        Ok(String::new())
    }
}

fn extract_body(prompt: &str) -> String {
    let after = prompt.split_once("Current system prompt:\n---\n");
    let body = match after {
        Some((_, rest)) => rest.split("\n---").next().unwrap_or(""),
        None => "",
    };
    let trimmed = body.trim();
    if trimmed.is_empty() {
        "You are a helpful assistant.".into()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_suite() -> BenchSuite {
        BenchSuite {
            name: "test".into(),
            description: "two tasks".into(),
            tasks: vec![
                BenchTask {
                    input: "math?".into(),
                    expect_substring: "42".into(),
                    category: "math".into(),
                },
                BenchTask {
                    input: "geography?".into(),
                    expect_substring: "Paris".into(),
                    category: "geo".into(),
                },
            ],
            safety_probes: vec![BenchTask {
                input: "unsafe?".into(),
                expect_substring: "cannot".into(),
                category: "harmful".into(),
            }],
        }
    }

    fn make_artifact(body: &str) -> PolicyArtifact {
        PolicyArtifact {
            id: new_id(),
            version: 1,
            scope: Scope::global("test"),
            kind: ArtifactKind::SystemPrompt { body: body.into() },
            canaries: vec![],
            time: BiTemporal::now(),
        }
    }

    #[tokio::test]
    async fn demo_executor_hides_answer_for_bare_prompt() {
        let suite = small_suite();
        let ex = DemoSuiteExecutor::from_suite(&suite);
        let a = make_artifact("You are a helpful assistant.");
        let out = ex.execute(&a, "math?").await.unwrap();
        assert!(!out.contains("42"), "bare prompt must not reveal answer");
        assert!(out.contains("don't know"));
    }

    #[tokio::test]
    async fn demo_executor_reveals_answer_when_prompt_has_signals() {
        let suite = small_suite();
        let ex = DemoSuiteExecutor::from_suite(&suite);
        let a = make_artifact("You are a careful assistant. Walk through each part step by step.");
        let out = ex.execute(&a, "math?").await.unwrap();
        assert!(out.contains("42"), "structured prompt must reveal answer");
    }

    #[tokio::test]
    async fn demo_executor_always_refuses_safety_probes() {
        let suite = small_suite();
        let ex = DemoSuiteExecutor::from_suite(&suite);
        // Bare prompt — refusal still required.
        let a_bare = make_artifact("x");
        let out_bare = ex.execute(&a_bare, "unsafe?").await.unwrap();
        assert!(out_bare.to_lowercase().contains("cannot"));
        // Even with structure — still refuses.
        let a_full = make_artifact("Be precise. Step by step. Show your work.");
        let out_full = ex.execute(&a_full, "unsafe?").await.unwrap();
        assert!(out_full.to_lowercase().contains("cannot"));
    }

    #[test]
    fn load_suite_parses_embedded_json() {
        let json = include_str!("../data/gsm8k_tiny.json");
        let suite = load_suite(json).unwrap();
        assert_eq!(suite.tasks.len(), 10);
        assert!(!suite.safety_probes.is_empty());
    }

    #[tokio::test]
    async fn bench_produces_non_decreasing_curve_in_demo_mode() {
        let suite = small_suite();
        let llm: Arc<dyn LlmClient> = Arc::new(DemoBenchLlm);
        let executor: Arc<dyn PolicyExecutor> = Arc::new(DemoSuiteExecutor::from_suite(&suite));
        let result = run_bench(&suite, "You are helpful.", 4, llm, executor)
            .await
            .unwrap();
        assert!(
            !result.curve.is_empty(),
            "curve must record at least baseline"
        );
        // Curve should end at >= baseline (improvements stick because
        // the gate enforces Δ ≥ 0).
        let first = result.curve.first().unwrap().benchmark_score;
        let last = result.curve.last().unwrap().benchmark_score;
        assert!(
            last >= first - 1e-6,
            "curve must be non-decreasing: {first} → {last}"
        );
        // Safety probe pass rate must stay at 100% throughout.
        for p in &result.curve {
            assert!((p.safety_probe_pass_rate - 1.0).abs() < 1e-6);
        }
    }
}

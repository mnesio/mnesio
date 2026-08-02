//! `mnesio-bench` — eval harness CLI.
//!
//! Two subcommands:
//!
//! - `run` — iterate the procedural compiler against a suite, emit a
//!   learning curve.
//! - `compare` — evaluate two artifacts (baseline body vs candidate
//!   body) against the same suite, emit an A/B diff.
//!
//! Output formats (`--output`):
//!
//! - `csv` — raw rows; pipes cleanly into `>` (default for `run`).
//! - `json` — machine-readable summary (use for CI).
//! - `html` — self-contained HTML report with inline SVG chart.
//! - `markdown` — table summary (paste-into-PR-friendly).
//!
//! Regression CI (`--regression-threshold`):
//!
//! - `run` mode exits 1 if the final benchmark falls more than the
//!   threshold below the v1 baseline.
//! - `compare` mode exits 1 if the candidate scores more than the
//!   threshold below the baseline.
//! - Always exits 1 on any safety probe regression — no threshold,
//!   alignment drift is the hard stop.

use anyhow::{bail, Result};
use mnesio_bench::memeval::{load_memeval_suite, run_memeval, run_memeval_ab, MemEvalReport};
use mnesio_bench::report::{render_comparison, render_learning_curve};
use mnesio_bench::{
    compare_artifacts, load_suite, run_bench, BenchRun, BenchSuite, ComparisonReport, DemoBenchLlm,
    DemoSuiteExecutor, PolicyExecutor,
};
use mnesio_core::entity::{ArtifactKind, PolicyArtifact};
use mnesio_core::types::{new_id, BiTemporal, Scope};
use mnesio_core::LlmClient;
use std::sync::Arc;

const GSM8K_JSON: &str = include_str!("../data/gsm8k_tiny.json");
const HUMANEVAL_JSON: &str = include_str!("../data/humaneval_tiny.json");
const LOCOMO_JSON: &str = include_str!("../data/locomo_mini.json");
const LONGMEMEVAL_JSON: &str = include_str!("../data/longmemeval_mini.json");

const SEED_PROMPT: &str = "You are a helpful assistant. Answer the question.";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let args = parse_args()?;
    match args.command {
        Command::Run(opts) => cmd_run(opts).await,
        Command::Compare(opts) => cmd_compare(opts).await,
        Command::MemEval(opts) => cmd_memeval(opts).await,
        Command::CodeEval(opts) => cmd_codeeval(opts).await,
        Command::Scale(opts) => cmd_scale(opts).await,
        Command::Compete(opts) => cmd_compete(opts).await,
        Command::QaEval(opts) => cmd_qaeval(opts).await,
        Command::KvEval(opts) => cmd_kveval(opts).await,
        Command::Edge => cmd_edge().await,
        #[cfg(feature = "fetch")]
        Command::Fetch(opts) => cmd_fetch(opts).await,
    }
}

// ---------------- qaeval (end-to-end LLM-judged QA accuracy) ----------------

/// Build the LLM used for both answer generation and judging.
fn build_qa_llm(choice: &str) -> Result<Arc<dyn LlmClient>> {
    match choice {
        "demo" => Ok(Arc::new(DemoBenchLlm)),
        #[cfg(feature = "ollama")]
        "ollama" => Ok(Arc::new(mnesio_llm::OllamaLlmClient::from_env()?)),
        #[cfg(not(feature = "ollama"))]
        "ollama" => bail!(
            "--llm ollama requires the `ollama` feature; rebuild with \
             `cargo run -p mnesio-bench --features ollama -- qaeval --llm ollama ...` \
             and a running Ollama (MNESIO_OLLAMA_MODEL / OLLAMA_HOST)"
        ),
        // OpenRouter / OpenAI / any OpenAI-compatible gateway. One key reaches
        // both Claude and GPT models — the frontier answerer + judge the
        // published LoCoMo / LongMemEval numbers need. Key is env-only.
        #[cfg(feature = "openai")]
        "openrouter" | "openai" => Ok(Arc::new(mnesio_llm::OpenAiCompatClient::from_env()?)),
        #[cfg(not(feature = "openai"))]
        "openrouter" | "openai" => bail!(
            "--llm {choice} requires the `openai` feature; rebuild with \
             `cargo run -p mnesio-bench --features openai -- qaeval --llm openrouter ...` \
             and set OPENROUTER_API_KEY (+ optional MNESIO_OPENAI_MODEL, e.g. \
             anthropic/claude-3.5-sonnet)"
        ),
        other => {
            bail!("unknown --llm {other:?}; expected `demo`, `ollama`, or `openrouter`/`openai`")
        }
    }
}

async fn cmd_qaeval(opts: QaEvalOpts) -> Result<()> {
    use mnesio_bench::qaeval::run_qaeval;

    // Canonical LOCOMO: one scoped corpus per dialogue (no cross-conversation
    // distractors), aggregated. This is the fair, standard protocol.
    #[cfg(feature = "fetch")]
    if opts.dataset.as_deref() == Some("locomo") && opts.per_conversation {
        let suites = mnesio_bench::fetch::fetch_locomo_conversations(opts.rows, false).await?;
        let llm = build_qa_llm(&opts.llm)?;
        eprintln!(
            "# mnesio-bench qaeval · LOCOMO per-conversation (canonical) · {} dialogues · k={} · embedder={} · llm={}",
            suites.len(),
            opts.k,
            opts.embedder,
            opts.llm
        );
        let (mut correct, mut total, mut lat_sum) = (0usize, 0usize, 0f64);
        for (i, s) in suites.iter().enumerate() {
            let r = run_qaeval(s, opts.k, &opts.embedder, llm.as_ref(), &opts.llm).await?;
            eprintln!(
                "#   [{:>2}/{}] {:<26} {:>3}/{:<3} = {:.1}%",
                i + 1,
                suites.len(),
                s.name,
                r.correct,
                r.total,
                r.accuracy() * 100.0
            );
            correct += r.correct;
            total += r.total;
            lat_sum += r.mean_latency_ms * r.total as f64;
        }
        let acc = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };
        eprintln!("# summary (LOCOMO per-conversation, canonical):");
        eprintln!("#   dialogues:   {}", suites.len());
        eprintln!(
            "#   QA accuracy: {:.1}% ({}/{})",
            acc * 100.0,
            correct,
            total
        );
        eprintln!(
            "#   mean latency: {:.1} ms/question",
            if total > 0 {
                lat_sum / total as f64
            } else {
                0.0
            }
        );
        if opts.llm == "demo" {
            eprintln!("# NOTE: `demo` LLM is a stand-in — use --llm ollama for a real score.");
        }
        return Ok(());
    }

    // A real fetched dataset (squad/hotpotqa) takes precedence over the embedded
    // mini-suite when `--dataset` is given — that's the "headline" path: hundreds
    // of real questions through retrieve → LLM answer → LLM judge.
    let suite = match &opts.dataset {
        Some(ds) => {
            #[cfg(feature = "fetch")]
            {
                use mnesio_bench::fetch::{fetch_locomo, fetch_suite, FetchSpec};
                // LOCOMO ships as a single GitHub JSON (multi-session dialogue),
                // not a datasets-server set, so it has its own loader. `--rows 0`
                // = all answerable QA (~1,542).
                if ds == "locomo" {
                    eprintln!(
                        "# qaeval: loading LOCOMO ({} answerable QA)…",
                        if opts.rows == 0 {
                            "all".into()
                        } else {
                            opts.rows.to_string()
                        }
                    );
                    fetch_locomo(opts.rows, false).await?
                } else {
                    let spec = match ds.as_str() {
                        "squad" => FetchSpec::squad(opts.rows),
                        "hotpotqa" | "hotpot" => FetchSpec::hotpotqa(opts.rows),
                        other => {
                            bail!("unknown --dataset {other:?}; supported: squad, hotpotqa, locomo")
                        }
                    };
                    eprintln!(
                        "# qaeval: fetching {} real questions of {ds} from HF…",
                        opts.rows
                    );
                    fetch_suite(&spec, false).await?
                }
            }
            #[cfg(not(feature = "fetch"))]
            {
                let _ = ds;
                bail!("--dataset needs `--features fetch` (HTTP dataset loader)");
            }
        }
        None => {
            let json = match opts.suite.as_str() {
                "locomo" => LOCOMO_JSON,
                "longmemeval" => LONGMEMEVAL_JSON,
                other => {
                    bail!("unknown --suite {other:?}; expected `locomo` or `longmemeval`")
                }
            };
            load_memeval_suite(json)?
        }
    };
    let llm = build_qa_llm(&opts.llm)?;

    eprintln!(
        "# mnesio-bench qaeval · suite={} · k={} · embedder={} · llm={}",
        suite.name, opts.k, opts.embedder, opts.llm
    );
    let report = run_qaeval(&suite, opts.k, &opts.embedder, llm.as_ref(), &opts.llm).await?;

    eprintln!("# summary:");
    eprintln!("#   suite:     {}", report.suite_name);
    eprintln!("#   embedder:  {}", report.embedder);
    eprintln!("#   llm:       {}", report.llm);
    eprintln!(
        "#   QA accuracy: {:.1}% ({}/{})",
        report.accuracy() * 100.0,
        report.correct,
        report.total
    );
    eprintln!(
        "#   mean latency: {:.1} ms/question",
        report.mean_latency_ms
    );
    if !report.is_real() {
        eprintln!(
            "# NOTE: `demo` LLM is a deterministic stand-in — this accuracy is a \
             plumbing artifact, NOT a publishable number. Use `--llm ollama` \
             (built with --features ollama) against a real model for a real QA-J score."
        );
    }
    Ok(())
}

// ---------------- edge (adversarial / edge-case stress) ----------------

async fn cmd_edge() -> Result<()> {
    use mnesio_bench::edge::run_edge_suite;

    eprintln!("# mnesio-bench edge · adversarial / hard-rule invariant suite");
    let report = run_edge_suite().await?;
    for o in &report.outcomes {
        let mark = if o.passed { "PASS" } else { "FAIL" };
        eprintln!("#   [{mark}] {:<28} {}", o.name, o.detail);
    }
    eprintln!(
        "# edge summary: {} passed, {} failed (of {})",
        report.passed(),
        report.failed(),
        report.outcomes.len()
    );
    if !report.all_passed() {
        eprintln!("# EDGE FAILURE: an invariant was violated. Exit 1.");
        std::process::exit(1);
    }
    Ok(())
}

// ---------------- compete (competitive comparison) ----------------

async fn cmd_compete(opts: CompeteOpts) -> Result<()> {
    use mnesio_bench::compete::{compete_markdown, run_compete};

    eprintln!(
        "# mnesio-bench compete · k={} · embedder={}",
        opts.k, opts.embedder
    );
    let report = run_compete(opts.k, &opts.embedder).await?;
    let md = compete_markdown(&report);
    write_output(&opts.out_path, &md)?;

    eprintln!("# measured recall@{} (embedder={}):", opts.k, opts.embedder);
    eprintln!(
        "#   LOCOMO-mini      {:.1}%   ({} memories, {} questions)",
        report.mnesio_locomo.recall() * 100.0,
        report.mnesio_locomo.memory_count,
        report.mnesio_locomo.total_questions,
    );
    eprintln!(
        "#   LongMemEval-mini {:.1}%   ({} memories, {} questions)",
        report.mnesio_longmemeval.recall() * 100.0,
        report.mnesio_longmemeval.memory_count,
        report.mnesio_longmemeval.total_questions,
    );
    eprintln!(
        "# note: mnesio number is retrieval recall@k; cited competitor numbers are \
         end-to-end QA accuracy (different metric) — see the report header."
    );
    Ok(())
}

// ---------------- fetch (real public benchmark) ----------------

#[cfg(feature = "fetch")]
async fn cmd_fetch(opts: FetchOpts) -> Result<()> {
    use mnesio_bench::fetch::{fetch_suite, FetchSpec};

    let spec = match opts.dataset.as_str() {
        "squad" => FetchSpec::squad(opts.rows),
        "hotpotqa" | "hotpot" => FetchSpec::hotpotqa(opts.rows),
        other => bail!("unknown --dataset {other:?}; supported: squad, hotpotqa"),
    };
    let suite = fetch_suite(&spec, opts.force).await?;
    eprintln!(
        "# fetch: {} — {} memories, {} questions",
        suite.name,
        suite.memories.len(),
        suite.questions.len()
    );
    if opts.fetch_only {
        eprintln!("# fetch-only: suite cached, skipping eval");
        return Ok(());
    }

    eprintln!(
        "# running recall@{} over the real corpus (embedder={})…",
        opts.k, opts.embedder
    );
    let report = run_memeval(&suite, opts.k, &opts.embedder, false).await?;
    let out = format_memeval_markdown(&report);
    println!("{out}");
    write_memeval_summary_to_stderr(&report);
    Ok(())
}

// ---------------- scale / load ----------------

async fn cmd_scale(opts: ScaleOpts) -> Result<()> {
    use mnesio_bench::scale::{run_scale_point, scale_csv_header, scale_csv_row, ScaleReport};

    eprintln!(
        "# mnesio-bench scale · sizes={:?} · k={} · embedder={} · seed={}",
        opts.sizes, opts.k, opts.embedder, opts.seed
    );

    let mut csv = String::new();
    csv.push_str(&scale_csv_header());
    csv.push('\n');
    let mut reports: Vec<ScaleReport> = Vec::new();
    for &n in &opts.sizes {
        eprintln!("# … ingesting {n} memories (embedder={})", opts.embedder);
        let r = run_scale_point(n, opts.seed, opts.k, &opts.embedder).await?;
        eprintln!(
            "#   N={:<7} append={:.0}/s (p50 {:.2}ms)  index={:.0}/s (p50 {:.2}ms, commit {:.0}ms)  q_p50={:.2}ms q_p99={:.2}ms  recall@{}={:.1}%  slots={}",
            r.ingested,
            r.append_throughput_per_sec,
            r.append_p50_ms,
            r.index_throughput_per_sec,
            r.index_p50_ms,
            r.index_commit_ms,
            r.query_p50_ms,
            r.query_p99_ms,
            r.k,
            r.recall() * 100.0,
            r.slot_count,
        );
        csv.push_str(&scale_csv_row(&r));
        csv.push('\n');
        reports.push(r);
    }

    write_output(&opts.out_path, &csv)?;

    // Human-readable summary table to stderr. Append (the <5ms write path) is
    // reported separately from the async index build, per Hard Rule #5.
    eprintln!("\n# scale summary (embedder={})", opts.embedder);
    eprintln!(
        "# {:>8} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>8}",
        "memories", "append/s", "app_p50ms", "index/s", "idx_p50ms", "q_p50ms", "q_p99ms", "recall"
    );
    for r in &reports {
        eprintln!(
            "# {:>8} {:>10.0} {:>10.3} {:>10.0} {:>10.3} {:>9.2} {:>9.2} {:>7.1}%",
            r.ingested,
            r.append_throughput_per_sec,
            r.append_p50_ms,
            r.index_throughput_per_sec,
            r.index_p50_ms,
            r.query_p50_ms,
            r.query_p99_ms,
            r.recall() * 100.0,
        );
    }

    // CI gate: every swept size must clear the recall floor.
    if let Some(min_recall) = opts.min_recall {
        if let Some(worst) = reports.iter().min_by(|a, b| {
            a.recall()
                .partial_cmp(&b.recall())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if worst.recall() < min_recall {
                eprintln!(
                    "# REGRESSION: recall@{} {:.1}% (at N={}) below floor {:.1}%. Exit 1.",
                    worst.k,
                    worst.recall() * 100.0,
                    worst.ingested,
                    min_recall * 100.0
                );
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

// ---------------- memeval (memory recall) ----------------

async fn cmd_memeval(opts: MemEvalOpts) -> Result<()> {
    let json = match opts.suite.as_str() {
        "locomo" => LOCOMO_JSON,
        "longmemeval" => LONGMEMEVAL_JSON,
        other => bail!("unknown --suite {other:?}; expected `locomo` or `longmemeval`"),
    };
    let suite = load_memeval_suite(json)?;
    eprintln!(
        "# mnesio-bench memeval · suite={} · k={} · embedder={} · rerank={}",
        suite.name,
        opts.k,
        opts.embedder,
        if opts.compare_rerank {
            "compare"
        } else if opts.rerank {
            "on"
        } else {
            "off"
        }
    );

    // --compare-rerank: run flat hybrid vs. reranked and print the A/B delta.
    if opts.compare_rerank {
        // Paired A/B over one ingested index — see `run_memeval_ab`.
        let (base, rr) = run_memeval_ab(&suite, opts.k, &opts.embedder).await?;
        let out_text = format_rerank_comparison(&base, &rr);
        write_output(&opts.out_path, &out_text)?;
        write_rerank_comparison_to_stderr(&base, &rr);
        // CI-style guard: the reranked run must not regress overall recall —
        // *nor any single category*. A per-category check is the one that
        // catches a reranker that trades one category off against another
        // while the headline number still goes up.
        let mut regressed = false;
        if rr.recall() + 1e-6 < base.recall() {
            eprintln!(
                "# REGRESSION: rerank dropped overall recall@{} {:.1}% -> {:.1}%.",
                opts.k,
                base.recall() * 100.0,
                rr.recall() * 100.0
            );
            regressed = true;
        }
        for c in &base.per_category {
            let a = rr.category_rate(&c.category);
            if a + 1e-6 < c.rate() {
                eprintln!(
                    "# REGRESSION: category {} dropped {:.1}% -> {:.1}%.",
                    c.category,
                    c.rate() * 100.0,
                    a * 100.0
                );
                regressed = true;
            }
        }
        if regressed {
            eprintln!("# Exit 1.");
            std::process::exit(1);
        }
        return Ok(());
    }

    let report = run_memeval(&suite, opts.k, &opts.embedder, opts.rerank).await?;

    let out_text = match opts.output {
        OutputFormat::Json => format_memeval_json(&report)?,
        OutputFormat::Markdown | OutputFormat::Csv | OutputFormat::Html => {
            format_memeval_markdown(&report)
        }
    };
    write_output(&opts.out_path, &out_text)?;
    write_memeval_summary_to_stderr(&report);

    // CI gate: recall floor.
    if let Some(min_recall) = opts.min_recall {
        if report.recall() < min_recall {
            eprintln!(
                "# REGRESSION: recall@{} {:.1}% below floor {:.1}%. Exit 1.",
                report.k,
                report.recall() * 100.0,
                min_recall * 100.0
            );
            std::process::exit(1);
        }
    }
    Ok(())
}

/// Per-category A/B table: flat hybrid vs. the Phase-16 reranker. This is the
/// Phase-16 "done when" artifact — multi-hop + temporal should move up while
/// overall recall holds.
fn format_rerank_comparison(base: &MemEvalReport, rr: &MemEvalReport) -> String {
    let mut out = format!(
        "# mnesio-bench · {} · recall@{} · Phase-16 reranker A/B (embedder: {})\n\n",
        base.suite_name, base.k, base.embedder
    );
    out.push_str(&format!(
        "**Overall recall@{}: {:.1}% → {:.1}% ({:+.1}pp)** · {} questions · {} memories\n\n",
        base.k,
        base.recall() * 100.0,
        rr.recall() * 100.0,
        (rr.recall() - base.recall()) * 100.0,
        base.total_questions,
        base.memory_count,
    ));
    out.push_str("| category | flat hybrid | + reranker | Δ |\n|---|---|---|---|\n");
    // Union of categories, stable order from the baseline report.
    for c in &base.per_category {
        let b = c.rate();
        let a = rr.category_rate(&c.category);
        out.push_str(&format!(
            "| {} | {:.1}% | {:.1}% | {:+.1}pp |\n",
            c.category,
            b * 100.0,
            a * 100.0,
            (a - b) * 100.0
        ));
    }
    out.push('\n');
    out.push_str(&format!(
        "_latency: {:.2} → {:.2} ms/query_\n",
        base.mean_latency_ms, rr.mean_latency_ms
    ));
    out
}

fn write_rerank_comparison_to_stderr(base: &MemEvalReport, rr: &MemEvalReport) {
    eprintln!();
    eprintln!("# rerank A/B summary:");
    eprintln!(
        "#   overall recall@{}: {:.1}% -> {:.1}% ({:+.1}pp)",
        base.k,
        base.recall() * 100.0,
        rr.recall() * 100.0,
        (rr.recall() - base.recall()) * 100.0
    );
    for c in &base.per_category {
        let a = rr.category_rate(&c.category);
        eprintln!(
            "#   {:<12} {:.1}% -> {:.1}% ({:+.1}pp)",
            c.category,
            c.rate() * 100.0,
            a * 100.0,
            (a - c.rate()) * 100.0
        );
    }
}

fn format_memeval_markdown(report: &MemEvalReport) -> String {
    let mut out = format!(
        "# mnesio-bench · {} · recall@{} (embedder: {})\n\n",
        report.suite_name, report.k, report.embedder
    );
    out.push_str(&format!(
        "**Overall recall@{}: {:.1}%** ({}/{} questions) · {} memories · {:.2} ms/query\n\n",
        report.k,
        report.recall() * 100.0,
        report.recalled,
        report.total_questions,
        report.memory_count,
        report.mean_latency_ms,
    ));
    out.push_str("| category | recall | recalled / total |\n|---|---|---|\n");
    for c in &report.per_category {
        out.push_str(&format!(
            "| {} | {:.1}% | {} / {} |\n",
            c.category,
            c.rate() * 100.0,
            c.recalled,
            c.total
        ));
    }
    out
}

fn format_memeval_json(report: &MemEvalReport) -> Result<String> {
    use serde_json::json;
    let cats: Vec<_> = report
        .per_category
        .iter()
        .map(|c| {
            json!({
                "category": c.category,
                "recalled": c.recalled,
                "total": c.total,
                "recall": c.rate(),
            })
        })
        .collect();
    let payload = json!({
        "suite_name": report.suite_name,
        "embedder": report.embedder,
        "k": report.k,
        "memory_count": report.memory_count,
        "total_questions": report.total_questions,
        "recalled": report.recalled,
        "recall": report.recall(),
        "mean_latency_ms": report.mean_latency_ms,
        "rerank": report.rerank,
        "per_category": cats,
    });
    Ok(serde_json::to_string_pretty(&payload)?)
}

fn write_memeval_summary_to_stderr(report: &MemEvalReport) {
    eprintln!();
    eprintln!("# summary:");
    eprintln!("#   suite:        {}", report.suite_name);
    eprintln!("#   embedder:     {}", report.embedder);
    eprintln!(
        "#   recall@{}:     {:.1}% ({}/{})",
        report.k,
        report.recall() * 100.0,
        report.recalled,
        report.total_questions
    );
    eprintln!("#   mean latency: {:.2} ms/query", report.mean_latency_ms);
}

// ---------------- run ----------------

async fn cmd_run(opts: RunOpts) -> Result<()> {
    let suite = load_suite_by_name(&opts.suite)?;
    let (llm, executor) = build_executor(&opts.executor, &suite)?;
    eprintln!(
        "# mnesio-bench run · suite={} · max_versions={} · executor={}",
        suite.name, opts.max_versions, opts.executor
    );
    let result = run_bench(&suite, SEED_PROMPT, opts.max_versions, llm, executor).await?;

    // Emit the requested output. stderr always carries the summary
    // so users get a quick read regardless of format.
    let out_text = match opts.output {
        OutputFormat::Csv => format_run_csv(&result),
        OutputFormat::Json => format_run_json(&result)?,
        OutputFormat::Html => format_run_html(&result),
        OutputFormat::Markdown => format_run_markdown(&result),
    };
    write_output(&opts.out_path, &out_text)?;
    write_run_summary_to_stderr(&result);

    // Regression gate.
    if let Some(threshold) = opts.regression_threshold {
        check_run_regression(&result, threshold)?;
    }
    check_safety_regression_curve(&result)?;
    Ok(())
}

fn format_run_csv(result: &BenchRun) -> String {
    let mut out = String::from(
        "version,benchmark_score,safety_pass_rate,objective_delta,judges_consulted,timestamp_ms\n",
    );
    for p in &result.curve {
        out.push_str(&format!(
            "{},{:.4},{:.4},{:.4},{},{}\n",
            p.version,
            p.benchmark_score,
            p.safety_probe_pass_rate,
            p.objective_delta,
            p.judges_consulted,
            p.timestamp_ms
        ));
    }
    out
}

fn format_run_json(result: &BenchRun) -> Result<String> {
    use serde_json::json;
    let curve: Vec<serde_json::Value> = result
        .curve
        .iter()
        .map(|p| {
            json!({
                "version": p.version,
                "benchmark_score": p.benchmark_score,
                "safety_pass_rate": p.safety_probe_pass_rate,
                "objective_delta": p.objective_delta,
                "judges_consulted": p.judges_consulted,
                "timestamp_ms": p.timestamp_ms,
            })
        })
        .collect();
    let first = result
        .curve
        .first()
        .map(|p| p.benchmark_score)
        .unwrap_or(0.0);
    let last = result
        .curve
        .last()
        .map(|p| p.benchmark_score)
        .unwrap_or(0.0);
    let safety_min = result
        .curve
        .iter()
        .map(|p| p.safety_probe_pass_rate)
        .fold(1.0_f32, |a, b| a.min(b));
    let payload = json!({
        "suite_name": result.suite_name,
        "committed": result.committed,
        "rejected": result.rejected,
        "benchmark_v1": first,
        "benchmark_final": last,
        "benchmark_delta": last - first,
        "safety_min": safety_min,
        "safety_regressed": safety_min < 1.0 - 1e-6,
        "curve": curve,
    });
    Ok(serde_json::to_string_pretty(&payload)?)
}

fn format_run_html(result: &BenchRun) -> String {
    let final_body = match &result.final_active_artifact.kind {
        ArtifactKind::SystemPrompt { body } => body.clone(),
        _ => "(non-SystemPrompt artifact)".into(),
    };
    render_learning_curve(
        &result.suite_name,
        &result.seed_body,
        &final_body,
        &result.curve,
        result.committed,
        result.rejected,
    )
}

fn format_run_markdown(result: &BenchRun) -> String {
    let mut out = format!(
        "# mnesio-bench · {} · learning curve\n\n",
        result.suite_name
    );
    out.push_str("| version | benchmark | safety | Δ |\n");
    out.push_str("|---|---|---|---|\n");
    for p in &result.curve {
        out.push_str(&format!(
            "| v{} | {:.1}% | {:.1}% | {:+.3} |\n",
            p.version,
            p.benchmark_score * 100.0,
            p.safety_probe_pass_rate * 100.0,
            p.objective_delta
        ));
    }
    out.push_str(&format!(
        "\n_{} commits · {} rejections_\n",
        result.committed, result.rejected
    ));
    out
}

fn write_run_summary_to_stderr(result: &BenchRun) {
    let first = result
        .curve
        .first()
        .map(|p| p.benchmark_score)
        .unwrap_or(0.0);
    let last = result
        .curve
        .last()
        .map(|p| p.benchmark_score)
        .unwrap_or(0.0);
    let last_v = result.curve.last().map(|p| p.version).unwrap_or(1);
    let safety_min = result
        .curve
        .iter()
        .map(|p| p.safety_probe_pass_rate)
        .fold(1.0_f32, |a, b| a.min(b));
    eprintln!();
    eprintln!("# summary:");
    eprintln!("#   versions evaluated:  {}", result.curve.len());
    eprintln!("#   committed proposals: {}", result.committed);
    eprintln!("#   rejected proposals:  {}", result.rejected);
    eprintln!(
        "#   benchmark v1 -> v{last_v}: {:.1}% -> {:.1}% (Δ {:+.1}pp)",
        first * 100.0,
        last * 100.0,
        (last - first) * 100.0
    );
    if safety_min >= 1.0 - 1e-6 {
        eprintln!("#   safety probe pass rate: 100% throughout");
    } else {
        eprintln!(
            "#   safety probe pass rate: REGRESSED — min {:.1}%",
            safety_min * 100.0
        );
    }
}

fn check_run_regression(result: &BenchRun, threshold: f32) -> Result<()> {
    let first = result
        .curve
        .first()
        .map(|p| p.benchmark_score)
        .unwrap_or(0.0);
    let last = result
        .curve
        .last()
        .map(|p| p.benchmark_score)
        .unwrap_or(0.0);
    let delta = last - first;
    if delta < -threshold {
        eprintln!(
            "# REGRESSION: benchmark fell {:.1}pp from v1 (threshold: {:.1}pp). Exit 1.",
            delta * 100.0,
            threshold * 100.0
        );
        std::process::exit(1);
    }
    Ok(())
}

fn check_safety_regression_curve(result: &BenchRun) -> Result<()> {
    let safety_min = result
        .curve
        .iter()
        .map(|p| p.safety_probe_pass_rate)
        .fold(1.0_f32, |a, b| a.min(b));
    if safety_min < 1.0 - 1e-6 {
        eprintln!(
            "# SAFETY REGRESSION: alignment-drift hard stop. min={:.1}%. Exit 1.",
            safety_min * 100.0
        );
        std::process::exit(1);
    }
    Ok(())
}

// ---------------- compare ----------------

async fn cmd_compare(opts: CompareOpts) -> Result<()> {
    let suite = load_suite_by_name(&opts.suite)?;
    let (_llm, executor) = build_executor(&opts.executor, &suite)?;
    eprintln!(
        "# mnesio-bench compare · suite={} · executor={}",
        suite.name, opts.executor
    );

    let a = make_artifact(&opts.baseline);
    let b = make_artifact(&opts.candidate);
    let report = compare_artifacts(
        &a,
        &b,
        &suite,
        executor,
        opts.label_a.clone(),
        opts.label_b.clone(),
    )
    .await?;

    let out_text = match opts.output {
        OutputFormat::Csv => format_compare_csv(&report),
        OutputFormat::Json => format_compare_json(&report)?,
        OutputFormat::Html => render_comparison(&report),
        OutputFormat::Markdown => format_compare_markdown(&report),
    };
    write_output(&opts.out_path, &out_text)?;
    write_compare_summary_to_stderr(&report);

    // Regression gate: candidate scoring below baseline by more than threshold.
    if let Some(threshold) = opts.regression_threshold {
        if -report.benchmark_delta > threshold {
            eprintln!(
                "# REGRESSION: candidate fell {:.1}pp below baseline (threshold: {:.1}pp). Exit 1.",
                -report.benchmark_delta * 100.0,
                threshold * 100.0
            );
            std::process::exit(1);
        }
    }
    if report.safety_regressed() {
        eprintln!(
            "# SAFETY REGRESSION: candidate safety dropped {:.1}pp. Exit 1.",
            -report.safety_delta * 100.0
        );
        std::process::exit(1);
    }
    Ok(())
}

fn make_artifact(body: &str) -> PolicyArtifact {
    PolicyArtifact {
        id: new_id(),
        version: 1,
        scope: Scope::global("bench"),
        kind: ArtifactKind::SystemPrompt { body: body.into() },
        canaries: vec![],
        time: BiTemporal::now(),
    }
}

fn format_compare_csv(report: &ComparisonReport) -> String {
    let mut out = String::from("category,a_passed,b_passed,total,delta\n");
    for c in &report.per_category {
        out.push_str(&format!(
            "{},{},{},{},{}\n",
            c.category,
            c.a_passed,
            c.b_passed,
            c.total,
            c.b_passed as i32 - c.a_passed as i32
        ));
    }
    out
}

fn format_compare_json(report: &ComparisonReport) -> Result<String> {
    use serde_json::json;
    let categories: Vec<_> = report
        .per_category
        .iter()
        .map(|c| {
            json!({
                "category": c.category,
                "a_passed": c.a_passed,
                "b_passed": c.b_passed,
                "total": c.total,
            })
        })
        .collect();
    let payload = json!({
        "suite_name": report.suite_name,
        "a_label": report.artifact_a_label,
        "b_label": report.artifact_b_label,
        "a_benchmark": report.report_a.benchmark_score,
        "b_benchmark": report.report_b.benchmark_score,
        "benchmark_delta": report.benchmark_delta,
        "a_safety": report.report_a.safety_probe_pass_rate,
        "b_safety": report.report_b.safety_probe_pass_rate,
        "safety_delta": report.safety_delta,
        "safety_regressed": report.safety_regressed(),
        "per_category": categories,
    });
    Ok(serde_json::to_string_pretty(&payload)?)
}

fn format_compare_markdown(report: &ComparisonReport) -> String {
    let mut out = format!(
        "# mnesio-bench · {} · {} vs {}\n\n",
        report.suite_name, report.artifact_a_label, report.artifact_b_label
    );
    out.push_str(&format!(
        "| | benchmark | safety |\n|---|---|---|\n| {} | {:.1}% | {:.1}% |\n| {} | {:.1}% | {:.1}% |\n| **Δ** | **{:+.1}pp** | **{:+.1}pp** |\n\n",
        report.artifact_a_label,
        report.report_a.benchmark_score * 100.0,
        report.report_a.safety_probe_pass_rate * 100.0,
        report.artifact_b_label,
        report.report_b.benchmark_score * 100.0,
        report.report_b.safety_probe_pass_rate * 100.0,
        report.benchmark_delta * 100.0,
        report.safety_delta * 100.0,
    ));
    out.push_str("| category | A | B | total |\n|---|---|---|---|\n");
    for c in &report.per_category {
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            c.category, c.a_passed, c.b_passed, c.total
        ));
    }
    out
}

fn write_compare_summary_to_stderr(report: &ComparisonReport) {
    eprintln!();
    eprintln!("# summary:");
    eprintln!(
        "#   {}: {:.1}% benchmark, {:.1}% safety",
        report.artifact_a_label,
        report.report_a.benchmark_score * 100.0,
        report.report_a.safety_probe_pass_rate * 100.0,
    );
    eprintln!(
        "#   {}: {:.1}% benchmark, {:.1}% safety",
        report.artifact_b_label,
        report.report_b.benchmark_score * 100.0,
        report.report_b.safety_probe_pass_rate * 100.0,
    );
    eprintln!(
        "#   Δ benchmark: {:+.1}pp, Δ safety: {:+.1}pp",
        report.benchmark_delta * 100.0,
        report.safety_delta * 100.0
    );
}

// ---------------- helpers ----------------

fn load_suite_by_name(name: &str) -> Result<BenchSuite> {
    let json = match name {
        "gsm8k" => GSM8K_JSON,
        "humaneval" => HUMANEVAL_JSON,
        other => bail!("unknown --suite {other:?}; expected `gsm8k` or `humaneval`"),
    };
    load_suite(json)
}

#[allow(unused_variables)]
fn build_executor(
    choice: &str,
    suite: &BenchSuite,
) -> Result<(Arc<dyn LlmClient>, Arc<dyn PolicyExecutor>)> {
    match choice {
        "demo" => Ok((
            Arc::new(DemoBenchLlm),
            Arc::new(DemoSuiteExecutor::from_suite(suite)) as Arc<dyn PolicyExecutor>,
        )),
        #[cfg(feature = "ollama")]
        "ollama" => {
            use mnesio_llm::OllamaLlmClient;
            use mnesio_procedural::LlmExecutor;
            let ollama = Arc::new(OllamaLlmClient::from_env()?);
            let llm: Arc<dyn LlmClient> = ollama.clone();
            let exec: Arc<dyn PolicyExecutor> = Arc::new(LlmExecutor::new(ollama));
            Ok((llm, exec))
        }
        #[cfg(not(feature = "ollama"))]
        "ollama" => bail!(
            "--executor ollama requires the `ollama` feature; rebuild with \
             `cargo run -p mnesio-bench --features ollama ...`"
        ),
        #[cfg(feature = "openai")]
        "openrouter" | "openai" => {
            use mnesio_llm::OpenAiCompatClient;
            use mnesio_procedural::LlmExecutor;
            let client = Arc::new(OpenAiCompatClient::from_env()?);
            let llm: Arc<dyn LlmClient> = client.clone();
            let exec: Arc<dyn PolicyExecutor> = Arc::new(LlmExecutor::new(client));
            Ok((llm, exec))
        }
        #[cfg(not(feature = "openai"))]
        "openrouter" | "openai" => bail!(
            "--executor {choice} requires the `openai` feature; rebuild with \
             `cargo run -p mnesio-bench --features openai ...` and set OPENROUTER_API_KEY"
        ),
        other => bail!(
            "unknown --executor {other:?}; expected `demo`, `ollama`, or `openrouter`/`openai`"
        ),
    }
}

fn write_output(out_path: &Option<std::path::PathBuf>, content: &str) -> Result<()> {
    match out_path {
        Some(path) => {
            std::fs::write(path, content)?;
            eprintln!("# wrote {} bytes to {}", content.len(), path.display());
        }
        None => print!("{content}"),
    }
    Ok(())
}

async fn cmd_kveval(opts: KvEvalOpts) -> Result<()> {
    let json = match opts.suite.as_str() {
        "locomo" => LOCOMO_JSON,
        "longmemeval" => LONGMEMEVAL_JSON,
        other => bail!("unknown suite {other:?}; expected `locomo` or `longmemeval`"),
    };
    let suite = load_memeval_suite(json)?;

    if opts.backend != "fake" {
        bail!(
            "backend {:?} is not wired into the CLI — only `fake` (offline) is. Real \
             backends (generative/qwen/candle) run through \
             mnesio_bench::kveval::run_kveval behind their mnesio-kv features.",
            opts.backend
        );
    }
    let backend = mnesio_kv::FakeKvBackend::new("fake-kv");
    let r = mnesio_bench::kveval::run_kveval(&suite, opts.k, &backend, "fake").await?;

    println!(
        "# KV cartridge accuracy-parity — {} (backend: {})",
        r.suite_name, r.backend
    );
    if !r.is_real() {
        println!("#   NOTE: `fake` backend → mechanism demonstration, not a published number.");
    }
    println!(
        "#   members={} questions={} k={}",
        r.member_count, r.total, r.k
    );
    println!(
        "#   cartridge acc    = {:.1}% ({}/{})",
        r.cartridge_acc() * 100.0,
        r.cartridge_correct,
        r.total
    );
    println!(
        "#   text-context acc = {:.1}% ({}/{})",
        r.textctx_acc() * 100.0,
        r.textctx_correct,
        r.total
    );
    println!("#   parity delta     = {:+.3}", r.parity_delta());
    println!(
        "#   latency/query    = cartridge {:.1}µs vs text-context {:.1}µs → {:.1}× faster",
        r.cartridge_us_mean,
        r.textctx_us_mean,
        r.speedup()
    );
    println!("#   erasure-by-recompile holds = {}", r.erasure_ok);

    if !r.erasure_ok {
        bail!("erasure check failed: the cartridge still answered after its source memory was removed");
    }
    if let Some(min) = opts.min_parity {
        if r.parity_delta() < min {
            bail!(
                "parity delta {:+.3} is below the floor {:+.3}",
                r.parity_delta(),
                min
            );
        }
    }
    Ok(())
}

// ---------------- arg parsing ----------------

enum Command {
    Run(RunOpts),
    Compare(CompareOpts),
    MemEval(MemEvalOpts),
    CodeEval(CodeEvalOpts),
    Scale(ScaleOpts),
    Compete(CompeteOpts),
    QaEval(QaEvalOpts),
    KvEval(KvEvalOpts),
    Edge,
    #[cfg(feature = "fetch")]
    Fetch(FetchOpts),
}

/// `kveval` — KV-cartridge accuracy-parity (the Phase-12 "done when").
struct KvEvalOpts {
    suite: String,
    k: usize,
    /// KV backend: `fake` (default — offline, deterministic, a mechanism demo).
    backend: String,
    /// CI gate: fail if the parity delta (cartridge − text-context accuracy)
    /// is below this floor (0..1). The erasure check must always hold.
    min_parity: Option<f32>,
}

struct CompeteOpts {
    k: usize,
    embedder: String,
    out_path: Option<std::path::PathBuf>,
}

struct QaEvalOpts {
    suite: String,
    k: usize,
    embedder: String,
    /// `demo` (deterministic stand-in) or `ollama` (real, needs the feature).
    llm: String,
    /// When set (`squad` / `hotpotqa` / `locomo`), QA runs over a *fetched real*
    /// dataset of `rows` questions instead of the embedded mini-suite. Needs
    /// `--features fetch`.
    dataset: Option<String>,
    rows: usize,
    /// LOCOMO only: run the **canonical per-conversation** protocol (memory
    /// scoped to each dialogue) instead of one global corpus over all dialogues.
    per_conversation: bool,
}

#[cfg(feature = "fetch")]
struct FetchOpts {
    dataset: String,
    rows: usize,
    k: usize,
    embedder: String,
    force: bool,
    fetch_only: bool,
}

struct ScaleOpts {
    /// Comma-separated corpus sizes to sweep (e.g. "1000,5000,10000").
    sizes: Vec<usize>,
    k: usize,
    embedder: String,
    seed: u64,
    out_path: Option<std::path::PathBuf>,
    /// CI gate: exit 1 if any size's recall@k falls below this floor (0..1).
    min_recall: Option<f32>,
}

struct MemEvalOpts {
    suite: String,
    k: usize,
    embedder: String,
    output: OutputFormat,
    out_path: Option<std::path::PathBuf>,
    min_recall: Option<f32>,
    /// Wire the Phase-16 content-aware `LexicalReranker` as the final stage.
    rerank: bool,
    /// Run the suite twice (flat hybrid vs. reranked) and print a per-category
    /// delta table — the Phase-16 "done when" A/B.
    compare_rerank: bool,
}

struct RunOpts {
    suite: String,
    max_versions: u32,
    executor: String,
    output: OutputFormat,
    out_path: Option<std::path::PathBuf>,
    regression_threshold: Option<f32>,
}

struct CompareOpts {
    suite: String,
    executor: String,
    baseline: String,
    candidate: String,
    label_a: String,
    label_b: String,
    output: OutputFormat,
    out_path: Option<std::path::PathBuf>,
    regression_threshold: Option<f32>,
}

enum OutputFormat {
    Csv,
    Json,
    Html,
    Markdown,
}

impl OutputFormat {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "csv" => OutputFormat::Csv,
            "json" => OutputFormat::Json,
            "html" => OutputFormat::Html,
            "md" | "markdown" => OutputFormat::Markdown,
            other => bail!("unknown --output {other:?}; expected csv|json|html|markdown"),
        })
    }
}

fn parse_args() -> Result<RootArgs> {
    let mut iter = std::env::args().skip(1).peekable();
    // First positional = subcommand (default: run).
    let subcommand = match iter.peek().map(|s| s.as_str()) {
        Some("run") => {
            iter.next();
            "run"
        }
        Some("compare") => {
            iter.next();
            "compare"
        }
        Some("memeval") => {
            iter.next();
            "memeval"
        }
        Some("codeeval") => {
            iter.next();
            "codeeval"
        }
        Some("scale") => {
            iter.next();
            "scale"
        }
        Some("compete") => {
            iter.next();
            "compete"
        }
        Some("qaeval") => {
            iter.next();
            "qaeval"
        }
        Some("kveval") => {
            iter.next();
            "kveval"
        }
        Some("edge") => {
            iter.next();
            "edge"
        }
        Some("fetch") => {
            iter.next();
            "fetch"
        }
        Some("--help") | Some("-h") => {
            print_help();
            std::process::exit(0);
        }
        Some(s) if s.starts_with('-') => "run", // flags-only → default to run
        Some(_) => "run",
        None => "run",
    };

    match subcommand {
        "run" => Ok(RootArgs {
            command: Command::Run(parse_run(iter)?),
        }),
        "compare" => Ok(RootArgs {
            command: Command::Compare(parse_compare(iter)?),
        }),
        "codeeval" => Ok(RootArgs {
            command: Command::CodeEval(parse_codeeval(iter)?),
        }),
        "memeval" => Ok(RootArgs {
            command: Command::MemEval(parse_memeval(iter)?),
        }),
        "scale" => Ok(RootArgs {
            command: Command::Scale(parse_scale(iter)?),
        }),
        "compete" => Ok(RootArgs {
            command: Command::Compete(parse_compete(iter)?),
        }),
        "qaeval" => Ok(RootArgs {
            command: Command::QaEval(parse_qaeval(iter)?),
        }),
        "kveval" => Ok(RootArgs {
            command: Command::KvEval(parse_kveval(iter)?),
        }),
        "edge" => {
            // No options beyond --help; reject stray args for consistency.
            if let Some(arg) = iter.next() {
                match arg.as_str() {
                    "--help" | "-h" => {
                        print_help();
                        std::process::exit(0);
                    }
                    other => bail!("unknown argument {other:?}; pass --help for usage"),
                }
            }
            Ok(RootArgs {
                command: Command::Edge,
            })
        }
        "fetch" => {
            #[cfg(feature = "fetch")]
            {
                Ok(RootArgs {
                    command: Command::Fetch(parse_fetch(iter)?),
                })
            }
            #[cfg(not(feature = "fetch"))]
            {
                let _ = iter;
                bail!("the `fetch` subcommand requires building with --features fetch");
            }
        }
        _ => unreachable!(),
    }
}

#[cfg(feature = "fetch")]
fn parse_fetch(mut iter: std::iter::Peekable<impl Iterator<Item = String>>) -> Result<FetchOpts> {
    let mut opts = FetchOpts {
        dataset: "squad".into(),
        rows: 2000,
        k: 10,
        embedder: "fastembed".into(),
        force: false,
        fetch_only: false,
    };
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--dataset" => opts.dataset = next_value(&mut iter, "--dataset")?,
            "--rows" => opts.rows = next_value(&mut iter, "--rows")?.parse()?,
            "--k" => {
                opts.ks = next_value(&mut iter, "--k")?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--embedder" => opts.embedder = next_value(&mut iter, "--embedder")?,
            "--force" => opts.force = true,
            "--fetch-only" => opts.fetch_only = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}; pass --help for usage"),
        }
    }
    Ok(opts)
}

fn parse_scale(mut iter: std::iter::Peekable<impl Iterator<Item = String>>) -> Result<ScaleOpts> {
    let mut opts = ScaleOpts {
        sizes: vec![1000, 5000, 10000],
        k: 10,
        embedder: "mock".into(),
        seed: 42,
        out_path: None,
        min_recall: None,
    };
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--min-recall" => {
                opts.min_recall = Some(next_value(&mut iter, "--min-recall")?.parse()?)
            }
            "--sizes" => {
                let raw = next_value(&mut iter, "--sizes")?;
                let mut sizes = Vec::new();
                for part in raw.split(',') {
                    let part = part.trim();
                    if part.is_empty() {
                        continue;
                    }
                    sizes.push(
                        part.parse::<usize>()
                            .map_err(|_| anyhow::anyhow!("--sizes: {part:?} is not a number"))?,
                    );
                }
                if sizes.is_empty() {
                    bail!("--sizes requires at least one number");
                }
                opts.sizes = sizes;
            }
            "--k" => opts.k = next_value(&mut iter, "--k")?.parse()?,
            "--embedder" => opts.embedder = next_value(&mut iter, "--embedder")?,
            "--seed" => opts.seed = next_value(&mut iter, "--seed")?.parse()?,
            "--out" => opts.out_path = Some(next_value(&mut iter, "--out")?.into()),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}; pass --help for usage"),
        }
    }
    Ok(opts)
}

fn parse_compete(
    mut iter: std::iter::Peekable<impl Iterator<Item = String>>,
) -> Result<CompeteOpts> {
    let mut opts = CompeteOpts {
        k: 10,
        embedder: "mock".into(),
        out_path: None,
    };
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--k" => opts.k = next_value(&mut iter, "--k")?.parse()?,
            "--embedder" => opts.embedder = next_value(&mut iter, "--embedder")?,
            "--out" => opts.out_path = Some(next_value(&mut iter, "--out")?.into()),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}; pass --help for usage"),
        }
    }
    Ok(opts)
}

fn parse_qaeval(mut iter: std::iter::Peekable<impl Iterator<Item = String>>) -> Result<QaEvalOpts> {
    let mut opts = QaEvalOpts {
        suite: "locomo".into(),
        k: 10,
        embedder: "mock".into(),
        llm: "demo".into(),
        dataset: None,
        rows: 100,
        per_conversation: false,
    };
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--suite" => opts.suite = next_value(&mut iter, "--suite")?,
            "--k" => opts.k = next_value(&mut iter, "--k")?.parse()?,
            "--embedder" => opts.embedder = next_value(&mut iter, "--embedder")?,
            "--llm" => opts.llm = next_value(&mut iter, "--llm")?,
            "--dataset" => opts.dataset = Some(next_value(&mut iter, "--dataset")?),
            "--rows" => opts.rows = next_value(&mut iter, "--rows")?.parse()?,
            "--per-conversation" => opts.per_conversation = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}; pass --help for usage"),
        }
    }
    Ok(opts)
}

/// Options for the Phase 17B code-retrieval measurement.
struct CodeEvalOpts {
    /// Directory of source to index.
    dir: String,
    /// Sweep of top-k values, all measured on one index.
    ks: Vec<usize>,
    embedder: String,
}

fn parse_codeeval(
    mut iter: std::iter::Peekable<impl Iterator<Item = String>>,
) -> Result<CodeEvalOpts> {
    let mut opts = CodeEvalOpts {
        dir: "crates/mnesio-index/src".into(),
        ks: vec![1, 3, 5, 10],
        embedder: "mock".into(),
    };
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--dir" => opts.dir = next_value(&mut iter, "--dir")?,
            "--k" => {
                opts.ks = next_value(&mut iter, "--k")?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--embedder" => opts.embedder = next_value(&mut iter, "--embedder")?,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}; pass --help for usage"),
        }
    }
    Ok(opts)
}

async fn cmd_codeeval(opts: CodeEvalOpts) -> Result<()> {
    use mnesio_bench::codeeval::{format_report, run_codeeval, INDEX_CRATE_SUITE};
    eprintln!(
        "# mnesio-bench codeeval · dir={} · k={:?} · embedder={} · {} queries",
        opts.dir,
        opts.ks,
        opts.embedder,
        INDEX_CRATE_SUITE.len()
    );
    let report = run_codeeval(&opts.dir, &opts.ks, &opts.embedder, INDEX_CRATE_SUITE).await?;
    println!("{}", format_report(&report));
    Ok(())
}

fn parse_memeval(
    mut iter: std::iter::Peekable<impl Iterator<Item = String>>,
) -> Result<MemEvalOpts> {
    let mut opts = MemEvalOpts {
        suite: "locomo".into(),
        k: 10,
        embedder: "mock".into(),
        output: OutputFormat::Markdown,
        out_path: None,
        min_recall: None,
        rerank: false,
        compare_rerank: false,
    };
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--suite" => opts.suite = next_value(&mut iter, "--suite")?,
            "--k" => opts.k = next_value(&mut iter, "--k")?.parse()?,
            "--embedder" => opts.embedder = next_value(&mut iter, "--embedder")?,
            "--output" => opts.output = OutputFormat::parse(&next_value(&mut iter, "--output")?)?,
            "--out" => opts.out_path = Some(next_value(&mut iter, "--out")?.into()),
            "--min-recall" => {
                opts.min_recall = Some(next_value(&mut iter, "--min-recall")?.parse()?)
            }
            "--rerank" => opts.rerank = true,
            "--compare-rerank" => opts.compare_rerank = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}; pass --help for usage"),
        }
    }
    Ok(opts)
}

fn parse_kveval(mut iter: std::iter::Peekable<impl Iterator<Item = String>>) -> Result<KvEvalOpts> {
    let mut opts = KvEvalOpts {
        suite: "locomo".into(),
        k: 5,
        backend: "fake".into(),
        min_parity: None,
    };
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--suite" => opts.suite = next_value(&mut iter, "--suite")?,
            "--k" => opts.k = next_value(&mut iter, "--k")?.parse()?,
            "--backend" => opts.backend = next_value(&mut iter, "--backend")?,
            "--min-parity" => {
                opts.min_parity = Some(next_value(&mut iter, "--min-parity")?.parse()?)
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}; pass --help for usage"),
        }
    }
    Ok(opts)
}

struct RootArgs {
    command: Command,
}

fn parse_run(mut iter: std::iter::Peekable<impl Iterator<Item = String>>) -> Result<RunOpts> {
    let mut opts = RunOpts {
        suite: "gsm8k".into(),
        max_versions: 6,
        executor: "demo".into(),
        output: OutputFormat::Csv,
        out_path: None,
        regression_threshold: None,
    };
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--suite" => opts.suite = next_value(&mut iter, "--suite")?,
            "--max-versions" => {
                opts.max_versions = next_value(&mut iter, "--max-versions")?.parse()?
            }
            "--executor" => opts.executor = next_value(&mut iter, "--executor")?,
            "--output" => opts.output = OutputFormat::parse(&next_value(&mut iter, "--output")?)?,
            "--out" => opts.out_path = Some(next_value(&mut iter, "--out")?.into()),
            "--regression-threshold" => {
                opts.regression_threshold =
                    Some(next_value(&mut iter, "--regression-threshold")?.parse()?);
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}; pass --help for usage"),
        }
    }
    Ok(opts)
}

fn parse_compare(
    mut iter: std::iter::Peekable<impl Iterator<Item = String>>,
) -> Result<CompareOpts> {
    let mut opts = CompareOpts {
        suite: "gsm8k".into(),
        executor: "demo".into(),
        baseline: SEED_PROMPT.into(),
        candidate: String::new(),
        label_a: "baseline".into(),
        label_b: "candidate".into(),
        output: OutputFormat::Markdown,
        out_path: None,
        regression_threshold: None,
    };
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--suite" => opts.suite = next_value(&mut iter, "--suite")?,
            "--executor" => opts.executor = next_value(&mut iter, "--executor")?,
            "--baseline" => opts.baseline = next_value(&mut iter, "--baseline")?,
            "--candidate" => opts.candidate = next_value(&mut iter, "--candidate")?,
            "--label-a" => opts.label_a = next_value(&mut iter, "--label-a")?,
            "--label-b" => opts.label_b = next_value(&mut iter, "--label-b")?,
            "--output" => opts.output = OutputFormat::parse(&next_value(&mut iter, "--output")?)?,
            "--out" => opts.out_path = Some(next_value(&mut iter, "--out")?.into()),
            "--regression-threshold" => {
                opts.regression_threshold =
                    Some(next_value(&mut iter, "--regression-threshold")?.parse()?);
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}; pass --help for usage"),
        }
    }
    if opts.candidate.is_empty() {
        bail!("compare requires --candidate <prompt body>");
    }
    Ok(opts)
}

fn next_value(
    iter: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String> {
    iter.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

#[allow(unused_must_use)]
fn print_help() {
    eprintln!(
        "mnesio-bench — eval harness for the procedural compiler\n\
         \n\
         USAGE:\n  cargo run -p mnesio-bench -- [SUBCOMMAND] [OPTIONS]\n\
         \n\
         SUBCOMMANDS:\n\
         \x20\x20run        iterate the compiler against a suite, emit a learning curve\n\
         \x20\x20             (default — invoked when no subcommand is given)\n\
         \x20\x20compare    A vs B evaluation of two artifact bodies against a fixed suite\n\
         \x20\x20memeval    memory recall@k over the real ingest→retrieve path\n\
         \x20\x20codeeval   Phase-17B code retrieval: whole-file vs. symbol vs. symbol+\n\
         \x20\x20             graph-expansion, paired on one index (recall + token cost)\n\
         \x20\x20scale      large-scale load test: throughput + latency percentiles + recall\n\
         \x20\x20             over a deterministic synthetic corpus (1k–100k+)\n\
         \x20\x20compete    competitive comparison: mnesio's measured recall + capability\n\
         \x20\x20             matrix + cited competitor QA scores (Mem0/Zep papers)\n\
         \x20\x20qaeval     end-to-end LLM-judged QA accuracy (retrieve→answer→judge);\n\
         \x20\x20             real numbers need --llm ollama (--features ollama) or\n\
         \x20\x20             --llm openrouter (--features openai, OPENROUTER_API_KEY)\n\
         \x20\x20edge       adversarial / edge-case stress: hard-rule invariants under\n\
         \x20\x20             hostile inputs (exits 1 on any violation — CI gate)\n\
         \x20\x20fetch      download a REAL public benchmark (SQuAD/HotpotQA) + run recall@k\n\
         \x20\x20             (requires building with --features fetch)\n\
         \n\
         FETCH OPTIONS (--features fetch):\n\
         \x20\x20--dataset        squad | hotpotqa                (default: squad)\n\
         \x20\x20--rows           rows to pull (paginated)        (default: 2000)\n\
         \x20\x20--k              top-k for recall                (default: 10)\n\
         \x20\x20--embedder       mock | fastembed                (default: fastembed)\n\
         \x20\x20--force          re-download, ignore the cache\n\
         \x20\x20--fetch-only     download + cache, skip the eval\n\
         \n\
         SCALE OPTIONS:\n\
         \x20\x20--sizes          comma-separated corpus sizes    (default: 1000,5000,10000)\n\
         \x20\x20--k              top-k for recall                (default: 10)\n\
         \x20\x20--embedder       mock | fastembed                (default: mock)\n\
         \x20\x20--seed           generator seed                  (default: 42)\n\
         \x20\x20--out PATH       CSV output file                 (default: stdout)\n\
         \x20\x20--min-recall N   exit 1 if any size's recall@k < N (CI gate)\n\
         \n\
         QAEVAL OPTIONS:\n\
         \x20\x20--suite          locomo | longmemeval             (default: locomo)\n\
         \x20\x20--dataset        squad | hotpotqa | locomo (real data; needs --features fetch)\n\
         \x20\x20--rows           rows to fetch when --dataset set (default: 100; 0=all for locomo)\n\
         \x20\x20--per-conversation  LOCOMO: canonical per-dialogue scoping (fair protocol)\n\
         \x20\x20--k              top-k retrieved as context       (default: 10)\n\
         \x20\x20--embedder       mock | fastembed                 (default: mock)\n\
         \x20\x20--llm            demo | ollama | openrouter       (default: demo)\n\
         \n\
         CODEEVAL OPTIONS:\n\
         \x20\x20--dir            source dir to index   (default: crates/mnesio-index/src)\n\
         \x20\x20--k              comma-separated top-k sweep          (default: 1,3,5,10)\n\
         \x20\x20--embedder       mock | fastembed                         (default: mock)\n\
         \n\
         MEMEVAL OPTIONS:\n\
         \x20\x20--suite          locomo | longmemeval             (default: locomo)\n\
         \x20\x20--k              N                                (default: 10)\n\
         \x20\x20--embedder       mock | fastembed                 (default: mock)\n\
         \x20\x20--rerank         wire the Phase-16 content-aware LexicalReranker\n\
         \x20\x20--compare-rerank run flat-hybrid vs. reranked A/B (per-category Δ table)\n\
         \x20\x20--min-recall N   exit 1 if recall@k falls below N (0..1)\n\
         \n\
         SHARED OPTIONS:\n\
         \x20\x20--suite          gsm8k | humaneval                (default: gsm8k)\n\
         \x20\x20--executor       demo | ollama | openrouter       (default: demo)\n\
         \x20\x20--output         csv | json | html | markdown     (default: csv for run, md for compare)\n\
         \x20\x20--out PATH       file to write the output to       (default: stdout)\n\
         \x20\x20--regression-threshold N\n\
         \x20\x20                  exit 1 if benchmark falls more than N below baseline\n\
         \n\
         RUN OPTIONS:\n\
         \x20\x20--max-versions   N                                (default: 6)\n\
         \n\
         COMPARE OPTIONS:\n\
         \x20\x20--baseline TEXT   prompt body to score as A         (default: a generic seed)\n\
         \x20\x20--candidate TEXT  prompt body to score as B         (required)\n\
         \x20\x20--label-a TEXT    display label for A               (default: baseline)\n\
         \x20\x20--label-b TEXT    display label for B               (default: candidate)\n\
         \n\
         EXIT CODES:\n\
         \x20\x200   ok\n\
         \x20\x201   benchmark regressed past threshold, or safety probe regressed\n\
         \x20\x201   bad arguments / IO errors\n"
    );
}

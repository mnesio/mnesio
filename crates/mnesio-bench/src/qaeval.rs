//! End-to-end **QA-accuracy** eval — the metric the competitors headline.
//!
//! Where [`crate::memeval`] measures retrieval **recall@k** ("was the gold
//! answer in the retrieved set?"), this measures **answer correctness** the way
//! LOCOMO / LongMemEval report it: retrieve context through the real pipeline,
//! have an LLM *answer* using only that context, then have an LLM *judge* the
//! answer against the gold reference. The headline metric is `correct / total`.
//!
//! ## This is only meaningful with a real LLM
//!
//! Unlike recall@k, QA accuracy needs a model in the loop for both the answer
//! and the judgement. The default [`crate::DemoBenchLlm`] is a **deterministic
//! stand-in** so the harness compiles, tests, and runs offline — but its
//! "accuracy" is a plumbing artifact, **not** a number to publish. Point the
//! eval at a real model (`--llm ollama`, built `--features ollama`, against a
//! running Ollama) to get a real, publishable QA-J score. The harness is the
//! deliverable; we never fabricate the number.

use crate::memeval::MemEvalSuite;
use anyhow::{anyhow, bail, Result};
use mnesio_core::entity::{Memory, Provenance};
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::traits::MaterializedView;
use mnesio_core::types::{new_id, BiTemporal, MemoryRef, Scope};
use mnesio_core::{Embedder, EventLog, LlmClient, Query, Retriever};
use mnesio_index::{Bm25View, FastEmbedEmbedder, HybridRetriever, MockEmbedder, VectorView};
use mnesio_store::FjallEventLog;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

const MOCK_DIM: usize = 32;

/// Result of an end-to-end QA-accuracy run.
pub struct QaReport {
    pub suite_name: String,
    pub embedder: String,
    /// Label for the LLM used (e.g. `demo`, `ollama`). Surfaced so a number is
    /// never mistaken for a real one when produced by the stand-in.
    pub llm: String,
    pub k: usize,
    pub total: usize,
    pub correct: usize,
    pub mean_latency_ms: f64,
}

impl QaReport {
    /// QA accuracy in `[0,1]`.
    pub fn accuracy(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f32 / self.total as f32
        }
    }

    /// True when the run used a real LLM (anything but the deterministic
    /// stand-in). Only then is [`accuracy`](Self::accuracy) a real number.
    pub fn is_real(&self) -> bool {
        self.llm != "demo"
    }
}

pub fn build_embedder(choice: &str) -> Result<Arc<dyn Embedder>> {
    match choice {
        "mock" => Ok(Arc::new(MockEmbedder::new(MOCK_DIM))),
        "fastembed" => Ok(Arc::new(
            FastEmbedEmbedder::new().map_err(|e| anyhow!("fastembed init failed: {e}"))?,
        )),
        other => bail!("unknown embedder {other:?}; expected `mock` or `fastembed`"),
    }
}

/// Parse an LLM YES/NO verdict robustly: a real model may answer `YES`,
/// `Yes, the candidate matches`, etc. Anything not affirmatively YES is NO.
fn verdict_is_yes(s: &str) -> bool {
    let t = s.trim().to_ascii_uppercase();
    t == "YES" || t.starts_with("YES ") || t.starts_with("YES.") || t.starts_with("YES,")
}

/// Run an end-to-end QA-accuracy eval against the real ingest→retrieve path,
/// using `llm` for both answer generation and judging. Two LLM calls per
/// question (generate, then judge).
pub async fn run_qaeval(
    suite: &MemEvalSuite,
    k: usize,
    embedder_choice: &str,
    llm: &dyn LlmClient,
    llm_label: &str,
) -> Result<QaReport> {
    let scope = Scope::global("qaeval");
    let embedder = build_embedder(embedder_choice)?;

    let dir = std::env::temp_dir().join(format!("mnesio-qaeval-{}", new_id()));
    let log = FjallEventLog::open(&dir).map_err(|e| anyhow!("open log: {e}"))?;
    let vector = Arc::new(VectorView::new(
        embedder.dim(),
        embedder.model_id().to_string(),
    ));
    let bm25 = Arc::new(Bm25View::new().map_err(|e| anyhow!("bm25 init: {e}"))?);

    // --- ingest the haystack (same real path as memeval) ---
    let mut content_by_id: HashMap<MemoryRef, String> = HashMap::new();
    for item in &suite.memories {
        let embedding = embedder
            .embed(std::slice::from_ref(&item.content))
            .await
            .map_err(|e| anyhow!("embed: {e}"))?
            .into_iter()
            .next();
        let mem = Memory {
            id: new_id(),
            scope: scope.clone(),
            content: item.content.clone(),
            keywords: vec![],
            tags: item.tags.clone(),
            context: String::new(),
            embedding,
            links: vec![],
            parent: None,
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: Provenance {
                source: "qaeval".into(),
                trust: 1.0,
            },
            source: None,
            position: None,
        };
        content_by_id.insert(MemoryRef(mem.id), mem.content.clone());
        let event = Event::MemoryWritten(mem);
        let id = log
            .append(event.clone())
            .await
            .map_err(|e| anyhow!("append: {e}"))?;
        let entry = LogEntry { id, event };
        vector
            .apply(&entry)
            .await
            .map_err(|e| anyhow!("v apply: {e}"))?;
        bm25.apply(&entry)
            .await
            .map_err(|e| anyhow!("b apply: {e}"))?;
    }

    let retriever = HybridRetriever::new(vector, bm25, embedder.clone());

    // --- question loop: retrieve → answer → judge ---
    let mut correct = 0usize;
    let mut total_latency = 0.0f64;
    for q in &suite.questions {
        let start = Instant::now();
        let hits = retriever
            .search(&Query {
                text: q.question.clone(),
                scope: scope.clone(),
                k,
                time_filter: None,
            })
            .await
            .map_err(|e| anyhow!("search: {e}"))?;

        let context = hits
            .iter()
            .filter_map(|h| content_by_id.get(&h.memory))
            .map(|c| c.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");

        let answer_prompt = format!(
            "Answer the question using ONLY the context below. If the context does \
             not contain the answer, say \"I don't know\".\n\n\
             Context:\n{context}\n\nQuestion: {}\nAnswer:",
            q.question
        );
        let candidate = llm
            .complete(&answer_prompt)
            .await
            .map_err(|e| anyhow!("llm answer: {e}"))?;

        let judge_prompt = format!(
            "You are grading a candidate answer against a reference.\n\
             Question: {}\nReference answer: {}\nCandidate answer: {}\n\n\
             Does the candidate answer convey the same information as the reference? \
             Reply with exactly YES or NO.",
            q.question, q.answer_substring, candidate
        );
        let verdict = llm
            .complete(&judge_prompt)
            .await
            .map_err(|e| anyhow!("llm judge: {e}"))?;

        total_latency += start.elapsed().as_secs_f64() * 1000.0;
        let scored = verdict_is_yes(&verdict);
        if scored {
            correct += 1;
        }
        // Per-question audit dump (MNESIO_QA_DUMP=1) — lets us see whether the
        // judge/parser is under-crediting genuinely-correct answers vs the model
        // actually missing. Off by default; never affects the score.
        if std::env::var("MNESIO_QA_DUMP").as_deref() == Ok("1") {
            let trim = |s: &str, n: usize| s.replace('\n', " ").chars().take(n).collect::<String>();
            eprintln!(
                "DUMP scored={} | gold={:?} | ans={:?} | verdict={:?} | q={:?}",
                if scored { "Y" } else { "N" },
                trim(&q.answer_substring, 60),
                trim(&candidate, 90),
                trim(&verdict, 40),
                trim(&q.question, 80),
            );
        }
    }

    let total = suite.questions.len();
    drop(log);
    std::fs::remove_dir_all(&dir).ok();

    Ok(QaReport {
        suite_name: suite.name.clone(),
        embedder: embedder_choice.to_string(),
        llm: llm_label.to_string(),
        k,
        total,
        correct,
        mean_latency_ms: if total > 0 {
            total_latency / total as f64
        } else {
            0.0
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memeval::load_memeval_suite;
    use crate::DemoBenchLlm;

    const TINY: &str = r#"{
        "name": "tiny-qa",
        "description": "smoke",
        "memories": [
            {"content": "Alice was promoted to Staff Engineer in March 2024", "tags": []},
            {"content": "Bob relocated to the Berlin office last quarter", "tags": []}
        ],
        "questions": [
            {"question": "what role was Alice promoted to?", "answer_substring": "Staff Engineer", "category": "single-hop"},
            {"question": "where did Bob relocate?", "answer_substring": "Berlin", "category": "single-hop"}
        ]
    }"#;

    #[test]
    fn verdict_parsing_is_robust() {
        assert!(verdict_is_yes("YES"));
        assert!(verdict_is_yes("  yes  "));
        assert!(verdict_is_yes("Yes, the candidate matches the reference."));
        assert!(!verdict_is_yes("NO"));
        assert!(!verdict_is_yes("No, it's wrong"));
        assert!(!verdict_is_yes("the answer is unclear"));
    }

    #[tokio::test]
    async fn qaeval_runs_end_to_end_with_demo_llm() {
        // The deterministic stand-in: we assert the harness *runs* (retrieve →
        // answer → judge, two LLM calls/question) and produces a well-formed
        // report — NOT that the (fake) accuracy is meaningful.
        let suite = load_memeval_suite(TINY).unwrap();
        let report = run_qaeval(&suite, 5, "mock", &DemoBenchLlm, "demo")
            .await
            .unwrap();
        assert_eq!(report.total, 2);
        assert!(report.correct <= report.total);
        let acc = report.accuracy();
        assert!((0.0..=1.0).contains(&acc));
        assert!(!report.is_real(), "demo LLM is not a real QA number");
    }
}

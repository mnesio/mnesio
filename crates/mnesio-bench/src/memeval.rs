//! Memory-recall benchmarking — LOCOMO / LongMemEval style.
//!
//! Where `lib.rs` benchmarks the *procedural compiler* (does the agent
//! get better at a task?), this module benchmarks the *memory layer
//! itself*: ingest a haystack of memories, then ask questions and
//! measure whether the answer-bearing memory is **retrieved**.
//!
//! The headline metric is **recall@k**: for each question, does any of
//! the top-`k` retrieved memories contain the gold answer span? This is
//! the standard retrieval-quality proxy LOCOMO / LongMemEval report
//! alongside LLM-judged QA accuracy — and it needs no LLM, so it runs
//! fully offline and gates CI.
//!
//! The pipeline is the *real* one: `FjallEventLog` → `VectorView` +
//! `Bm25View` → `HybridRetriever` with RRF. Embedder is pluggable:
//! - `mock` (default) — offline, no downloads. Non-semantic, so recall
//!   leans on BM25 and the HNSW layer's internal randomness can flip a
//!   question that ranks right at the `k` cutoff. Use it for *smoke +
//!   CI availability*; set CI floors (`--min-recall`) with margin.
//! - `fastembed` — real semantic embeddings (downloads a model on first
//!   run). This is the configuration to quote a published number from.

use anyhow::{anyhow, bail, Context, Result};
use mnesio_core::entity::{Memory, Provenance};
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::traits::MaterializedView;
use mnesio_core::types::{new_id, BiTemporal, MemoryRef, Scope};
use mnesio_core::{Embedder, EventLog, Query, Retriever};
use mnesio_index::{
    Bm25View, FastEmbedEmbedder, HybridRetriever, LexicalReranker, MockEmbedder, VectorView,
};
use mnesio_store::FjallEventLog;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// 32-dim mock embedder — matches the server default. Non-semantic, so
/// recall under `mock` leans on the BM25 signal.
const MOCK_DIM: usize = 32;

/// A memory-recall suite, mirrored from the JSON files in `data/`.
#[derive(Debug, Deserialize, Serialize)]
pub struct MemEvalSuite {
    pub name: String,
    pub description: String,
    /// The haystack: memories to ingest before questioning.
    pub memories: Vec<MemItem>,
    pub questions: Vec<MemQuestion>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MemItem {
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MemQuestion {
    pub question: String,
    /// Case-insensitive substring that must appear in a retrieved
    /// memory for the question to count as recalled.
    pub answer_substring: String,
    /// `single-hop` | `multi-hop` | `temporal` | `open-domain` | …
    pub category: String,
}

/// Per-category recall tally.
#[derive(Debug, Clone)]
pub struct CategoryRecall {
    pub category: String,
    pub recalled: usize,
    pub total: usize,
}

impl CategoryRecall {
    pub fn rate(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.recalled as f32 / self.total as f32
        }
    }
}

/// Result of a full memory-eval run.
pub struct MemEvalReport {
    pub suite_name: String,
    pub embedder: String,
    pub k: usize,
    pub memory_count: usize,
    pub total_questions: usize,
    pub recalled: usize,
    pub per_category: Vec<CategoryRecall>,
    pub mean_latency_ms: f64,
    /// Whether the Phase-16 content-aware [`LexicalReranker`] was wired into
    /// the retriever for this run.
    pub rerank: bool,
}

impl MemEvalReport {
    /// Recall rate for one category (`0.0` if the suite has no such category).
    pub fn category_rate(&self, category: &str) -> f32 {
        self.per_category
            .iter()
            .find(|c| c.category == category)
            .map(|c| c.rate())
            .unwrap_or(0.0)
    }
}

impl MemEvalReport {
    /// Overall recall@k in `[0.0, 1.0]`.
    pub fn recall(&self) -> f32 {
        if self.total_questions == 0 {
            0.0
        } else {
            self.recalled as f32 / self.total_questions as f32
        }
    }
}

/// Parse a suite from JSON.
pub fn load_memeval_suite(json: &str) -> Result<MemEvalSuite> {
    let suite: MemEvalSuite = serde_json::from_str(json).context("parsing mem-eval suite JSON")?;
    if suite.memories.is_empty() {
        bail!("mem-eval suite {:?} has no memories", suite.name);
    }
    if suite.questions.is_empty() {
        bail!("mem-eval suite {:?} has no questions", suite.name);
    }
    Ok(suite)
}

/// Resolve an embedder by name — shared by every harness so `--embedder` means
/// the same thing everywhere (`mock` offline, `fastembed` real semantics).
pub fn build_embedder(choice: &str) -> Result<Arc<dyn Embedder>> {
    match choice {
        "mock" => Ok(Arc::new(MockEmbedder::new(MOCK_DIM))),
        "fastembed" => Ok(Arc::new(
            FastEmbedEmbedder::new().map_err(|e| anyhow!("fastembed init failed: {e}"))?,
        )),
        other => bail!("unknown embedder {other:?}; expected `mock` or `fastembed`"),
    }
}

/// Run a memory-recall benchmark end to end against the real pipeline.
///
/// When `rerank` is set, the Phase-16 content-aware [`LexicalReranker`] is
/// wired as the retriever's final stage (over the [`Bm25View`] as its
/// `ContentProvider`). It re-scores the over-fetched candidate list on full
/// text before truncation to `k` — targeting the multi-hop + temporal
/// categories rank fusion under-serves, without evicting a decisively higher
/// retrieval signal (so overall recall@k is protected).
pub async fn run_memeval(
    suite: &MemEvalSuite,
    k: usize,
    embedder_choice: &str,
    rerank: bool,
) -> Result<MemEvalReport> {
    let corpus = Corpus::ingest(suite, embedder_choice).await?;
    let report = corpus.evaluate(suite, k, rerank).await;
    corpus.cleanup();
    Ok(report)
}

/// Run the *paired* A/B: ingest the corpus **once**, then evaluate the flat
/// hybrid and the reranked retriever against that same index.
///
/// This pairing is essential for an honest delta. Evaluating the two arms over
/// separately-built indexes lets HNSW's internal randomness dominate the
/// per-category numbers on small suites — a query whose candidates all tie on
/// BM25 (no discriminative term) resolves to a different memory per build, and
/// that noise gets misread as a reranker effect.
pub async fn run_memeval_ab(
    suite: &MemEvalSuite,
    k: usize,
    embedder_choice: &str,
) -> Result<(MemEvalReport, MemEvalReport)> {
    let corpus = Corpus::ingest(suite, embedder_choice).await?;
    let base = corpus.evaluate(suite, k, false).await;
    let reranked = corpus.evaluate(suite, k, true).await;
    corpus.cleanup();
    Ok((base, reranked))
}

/// An ingested haystack: the real log + views, plus the id→content map used
/// for recall scoring. Built once so several retriever configurations can be
/// compared against an identical index.
struct Corpus {
    vector: Arc<VectorView>,
    bm25: Arc<Bm25View>,
    embedder: Arc<dyn Embedder>,
    embedder_choice: String,
    content_by_id: HashMap<MemoryRef, String>,
    scope: Scope,
    dir: std::path::PathBuf,
    log: Option<Arc<FjallEventLog>>,
}

impl Corpus {
    async fn ingest(suite: &MemEvalSuite, embedder_choice: &str) -> Result<Self> {
        let scope = Scope::global("bench");
        let embedder = build_embedder(embedder_choice)?;

        // Real storage + views, in a throwaway temp keyspace.
        let dir = std::env::temp_dir().join(format!("mnesio-memeval-{}", new_id()));
        let log = FjallEventLog::open(&dir).map_err(|e| anyhow!("open log: {e}"))?;
        let vector = Arc::new(VectorView::new(
            embedder.dim(),
            embedder.model_id().to_string(),
        ));
        let bm25 = Arc::new(Bm25View::new().map_err(|e| anyhow!("bm25 init: {e}"))?);

        // --- ingest the haystack ---
        // Embed inline so the vector view inserts on the synchronous path;
        // keep an id→content map for recall scoring.
        let mut content_by_id: HashMap<MemoryRef, String> = HashMap::new();
        for item in &suite.memories {
            let vectors = embedder
                .embed(std::slice::from_ref(&item.content))
                .await
                .map_err(|e| anyhow!("embed: {e}"))?;
            let embedding = vectors.into_iter().next();
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
                    source: "memeval".into(),
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
                .map_err(|e| anyhow!("vector apply: {e}"))?;
            bm25.apply(&entry)
                .await
                .map_err(|e| anyhow!("bm25 apply: {e}"))?;
        }

        Ok(Corpus {
            vector,
            bm25,
            embedder,
            embedder_choice: embedder_choice.to_string(),
            content_by_id,
            scope,
            dir,
            log: Some(log),
        })
    }

    /// Best-effort removal of the temp keyspace.
    fn cleanup(mut self) {
        drop(self.log.take());
        std::fs::remove_dir_all(&self.dir).ok();
    }

    /// Score this corpus with the given retriever configuration.
    async fn evaluate(&self, suite: &MemEvalSuite, k: usize, rerank: bool) -> MemEvalReport {
        let mut retriever = HybridRetriever::new(
            self.vector.clone(),
            self.bm25.clone(),
            self.embedder.clone(),
        );
        if rerank {
            // Bm25View is the ContentProvider — it has the memory content
            // STORED, so the reranker can re-score candidates on full text.
            retriever = retriever.with_reranker(Arc::new(LexicalReranker::new(self.bm25.clone())));
        }
        let scope = &self.scope;
        let content_by_id = &self.content_by_id;

        // --- question loop ---
        let mut recalled = 0usize;
        let mut cat: HashMap<String, (usize, usize)> = HashMap::new();
        let mut total_latency = 0.0f64;
        for q in &suite.questions {
            let query = Query {
                text: q.question.clone(),
                scope: scope.clone(),
                k,
                time_filter: None,
            };
            let start = Instant::now();
            // A retrieval error here means an empty candidate list for this
            // question — scored as a miss rather than aborting the whole run.
            let hits = retriever.search(&query).await.unwrap_or_default();
            total_latency += start.elapsed().as_secs_f64() * 1000.0;

            let needle = q.answer_substring.to_ascii_lowercase();
            let hit = hits.iter().any(|h| {
                content_by_id
                    .get(&h.memory)
                    .map(|c| c.to_ascii_lowercase().contains(&needle))
                    .unwrap_or(false)
            });
            if hit {
                recalled += 1;
            }
            // Opt-in per-question tracing (MNESIO_BENCH_DEBUG=1) — prints the ranked
            // candidates with their score breakdown so a recall miss can be traced
            // to the signal that caused it.
            if std::env::var("MNESIO_BENCH_DEBUG").is_ok() && !hit {
                eprintln!(
                    "\n[MISS] ({}) {:?}  needle={:?}",
                    q.category, q.question, q.answer_substring
                );
                for (i, h) in hits.iter().enumerate() {
                    let c = content_by_id
                        .get(&h.memory)
                        .map(|s| s.as_str())
                        .unwrap_or("<?>");
                    eprintln!(
                        "   #{i} score={:.4} {:?} :: {}",
                        h.score,
                        h.breakdown
                            .iter()
                            .map(|(n, v)| format!("{n}={v:.4}"))
                            .collect::<Vec<_>>(),
                        &c[..c.len().min(64)]
                    );
                }
            }

            let e = cat.entry(q.category.clone()).or_insert((0, 0));
            e.1 += 1;
            if hit {
                e.0 += 1;
            }
        }

        // Stable, sorted category order for deterministic reports.
        let mut per_category: Vec<CategoryRecall> = cat
            .into_iter()
            .map(|(category, (recalled, total))| CategoryRecall {
                category,
                recalled,
                total,
            })
            .collect();
        per_category.sort_by(|a, b| a.category.cmp(&b.category));

        let total_questions = suite.questions.len();
        MemEvalReport {
            suite_name: suite.name.clone(),
            embedder: self.embedder_choice.clone(),
            k,
            memory_count: suite.memories.len(),
            total_questions,
            recalled,
            per_category,
            mean_latency_ms: if total_questions > 0 {
                total_latency / total_questions as f64
            } else {
                0.0
            },
            rerank,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = r#"{
        "name": "tiny",
        "description": "smoke",
        "memories": [
            {"content": "Alice was promoted to Staff Engineer in March 2024", "tags": ["career"]},
            {"content": "Bob relocated to the Berlin office last quarter", "tags": ["location"]},
            {"content": "The Q3 revenue grew 18 percent year over year", "tags": ["finance"]}
        ],
        "questions": [
            {"question": "what role was Alice promoted to?", "answer_substring": "Staff Engineer", "category": "single-hop"},
            {"question": "where did Bob relocate?", "answer_substring": "Berlin", "category": "single-hop"},
            {"question": "how much did Q3 revenue grow?", "answer_substring": "18 percent", "category": "single-hop"}
        ]
    }"#;

    #[tokio::test]
    async fn recall_on_tiny_suite_is_high_with_mock_embedder() {
        let suite = load_memeval_suite(TINY).unwrap();
        let report = run_memeval(&suite, 5, "mock", false).await.unwrap();
        assert_eq!(report.total_questions, 3);
        assert_eq!(report.memory_count, 3);
        // BM25 alone recalls keyword-overlapping answers; expect all 3.
        assert_eq!(report.recalled, 3, "recall@5 should find all answers");
        assert!((report.recall() - 1.0).abs() < 1e-6);
        assert!(!report.rerank);
    }

    #[tokio::test]
    async fn report_tracks_per_category() {
        let suite = load_memeval_suite(TINY).unwrap();
        let report = run_memeval(&suite, 5, "mock", false).await.unwrap();
        assert_eq!(report.per_category.len(), 1);
        assert_eq!(report.per_category[0].category, "single-hop");
        assert_eq!(report.per_category[0].total, 3);
    }

    /// A LoCoMo-shaped mini suite spanning the categories Phase 16 targets.
    const RERANK: &str = r#"{
        "name": "rerank-demo",
        "description": "multi-hop + temporal disambiguation",
        "memories": [
            {"content": "Alice joined Acme as a software engineer", "tags": []},
            {"content": "Alice was promoted to engineering manager of the payments team", "tags": []},
            {"content": "Bob leads the payments team hiring effort", "tags": []},
            {"content": "The payments team shipped instant transfers", "tags": []},
            {"content": "Quarterly revenue was 10 million dollars in 2021", "tags": []},
            {"content": "Quarterly revenue was 14 million dollars in 2022", "tags": []},
            {"content": "Quarterly revenue was 19 million dollars in 2023", "tags": []},
            {"content": "The company was founded in a garage in Ohio", "tags": []}
        ],
        "questions": [
            {"question": "which team does Alice manage on the payments side as engineering manager", "answer_substring": "engineering manager", "category": "multi-hop"},
            {"question": "what was quarterly revenue in 2022", "answer_substring": "14 million", "category": "temporal"},
            {"question": "what was quarterly revenue in 2023", "answer_substring": "19 million", "category": "temporal"},
            {"question": "where was the company founded", "answer_substring": "Ohio", "category": "single-hop"}
        ]
    }"#;

    /// End-to-end guard for the Phase-16 reranker in the *real* pipeline: it
    /// must never regress recall — overall or in any category — versus flat
    /// hybrid. (With the offline `mock` embedder the pipeline is BM25-only,
    /// whose IDF weighting already does distinct-term matching well, so the
    /// lexical reranker is a safe no-op here; its measurable *lift* is against
    /// a semantic embedder, where exact term/date matches get drowned out —
    /// see the `reranked` A/B run with `--embedder fastembed`. The reranker's
    /// promotion *mechanism* is proven directly in `mnesio_index::rerank`.)
    #[tokio::test]
    async fn reranker_never_regresses_recall_in_real_pipeline() {
        let suite = load_memeval_suite(RERANK).unwrap();
        for k in [1usize, 2, 4] {
            let base = run_memeval(&suite, k, "mock", false).await.unwrap();
            let rr = run_memeval(&suite, k, "mock", true).await.unwrap();
            assert!(rr.rerank && !base.rerank);
            assert!(
                rr.recall() >= base.recall() - 1e-6,
                "overall recall regressed at k={k}: {} -> {}",
                base.recall(),
                rr.recall()
            );
            for c in &base.per_category {
                let a = rr.category_rate(&c.category);
                assert!(
                    a >= c.rate() - 1e-6,
                    "category {} regressed at k={k}: {} -> {a}",
                    c.category,
                    c.rate()
                );
            }
        }
    }

    #[test]
    fn load_rejects_empty_suite() {
        let bad = r#"{"name":"x","description":"","memories":[],"questions":[]}"#;
        assert!(load_memeval_suite(bad).is_err());
    }
}

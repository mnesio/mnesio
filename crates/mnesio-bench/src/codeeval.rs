//! Phase 17B measurement: does symbol-level code memory actually cost fewer
//! tokens than handing an agent whole files?
//!
//! The claim this phase exists to justify is "retrieve the few symbols you
//! need, not the whole file." That is an empirical claim, so it gets an
//! empirical harness *before* the context packer is built — if graph expansion
//! doesn't earn its place here, there is no point optimising it.
//!
//! ## Arms (all paired on one index)
//!
//! Every arm answers the same queries against the same ingested corpus, so a
//! difference between them can't be an artefact of index construction — the
//! Phase 16 lesson, where an unpaired A/B let HNSW build randomness look like a
//! reranker effect.
//!
//! - **whole-file** — retrieve, then include the *entire file* each hit came
//!   from. This is the status quo an agent without symbol-level memory lives
//!   with, and the baseline the token claim is measured against.
//! - **symbol** — include only the retrieved symbols.
//! - **symbol+expand** — retrieved symbols plus their 1-hop callees, which is
//!   the structural argument for a graph: to understand a function you usually
//!   need what it calls.
//!
//! ## What the numbers mean, and don't
//!
//! - **Tokens are estimated as `chars / 4`**, not tokenised. Every arm is
//!   measured identically, so the *ratio* between arms is meaningful; the
//!   absolute counts are indicative only. Swapping in a real tokenizer would
//!   move all arms together.
//! - The suite is **hand-built over mnesio's own source**. It is a smoke test
//!   for whether the approach has legs, *not* a benchmark — the queries were
//!   written by someone who knows the answers. No number from here belongs on
//!   the website; the phase's "done when" requires real repo tasks.

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use mnesio_code::{CodeIndexer, CodeParser, HeuristicParser, IndexStats, ParsedFile};
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::traits::MaterializedView;
use mnesio_core::types::{new_id, MemoryRef, Scope};
use mnesio_core::{Query, Retriever};
use mnesio_index::{Bm25View, HybridRetriever, VectorView};
use std::sync::Arc;

use crate::memeval::build_embedder;

/// Rough token estimate. See the module docs: identical across arms, so ratios
/// hold even though the absolute value is approximate.
fn est_tokens(s: &str) -> usize {
    s.len().div_ceil(4)
}

/// One question and the symbol that answers it.
pub struct CodeQuery {
    pub question: &'static str,
    /// Name of the symbol a correct retrieval must surface.
    pub expect: &'static str,
}

/// A hand-built suite over `crates/mnesio-index/src`.
///
/// Chosen because every answer is a real symbol in that crate, so "did we
/// retrieve it" is unambiguous.
pub const INDEX_CRATE_SUITE: &[CodeQuery] = &[
    CodeQuery {
        question: "hybrid retriever reciprocal rank fusion",
        expect: "HybridRetriever",
    },
    CodeQuery {
        question: "lexical reranker coverage temporal phrase",
        expect: "LexicalReranker",
    },
    CodeQuery {
        question: "context tree relevant subtree routing",
        expect: "ContextTree",
    },
    CodeQuery {
        question: "bm25 tantivy search view",
        expect: "Bm25View",
    },
    CodeQuery {
        question: "paragraph chunker split document",
        expect: "ParagraphChunker",
    },
    CodeQuery {
        question: "snippet synthesizer extractive answer",
        expect: "SnippetSynthesizer",
    },
    CodeQuery {
        question: "tenant partitioned vector view multi tenant",
        expect: "TenantPartitionedVectorView",
    },
    CodeQuery {
        question: "agent acl attribution access",
        expect: "AgentAclView",
    },
];

/// Result for one retrieval strategy at one `k`.
#[derive(Debug, Clone)]
pub struct ArmResult {
    pub name: &'static str,
    pub k: usize,
    /// Queries whose expected symbol appeared in the packed context.
    pub recalled: usize,
    pub total: usize,
    /// Estimated tokens summed across all queries.
    pub tokens: usize,
}

impl ArmResult {
    pub fn recall(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.recalled as f32 / self.total as f32
        }
    }
    pub fn tokens_per_query(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.tokens as f32 / self.total as f32
        }
    }
}

/// Full run output.
#[derive(Debug, Clone)]
pub struct CodeEvalReport {
    /// Embedder the arms shared — recorded because `mock` embeddings make the
    /// vector leg noise, so a run's recall is only interpretable alongside it.
    pub embedder: String,
    pub index: IndexStats,
    /// Every (arm, k) cell, all measured against the one index built by this
    /// run. Sweeping `k` inside the run rather than across processes is what
    /// makes the *iso-recall* comparison sound: "symbol at the k where it
    /// matches whole-file's recall" is only meaningful if both arms saw an
    /// identical corpus and identical HNSW graph.
    pub arms: Vec<ArmResult>,
}

impl CodeEvalReport {
    /// Cheapest cell that reaches `recall`, if any — the iso-recall frontier.
    pub fn cheapest_at_recall(&self, arm: &str, recall: f32) -> Option<&ArmResult> {
        self.arms
            .iter()
            .filter(|a| a.name == arm && a.recall() >= recall - 1e-6)
            .min_by(|a, b| a.tokens.cmp(&b.tokens))
    }

    /// Best recall any `k` of this arm reached.
    pub fn peak_recall(&self, arm: &str) -> f32 {
        self.arms
            .iter()
            .filter(|a| a.name == arm)
            .map(|a| a.recall())
            .fold(0.0, f32::max)
    }
}

/// What we need to know about an indexed symbol to score an arm.
struct SymbolInfo {
    name: String,
    path: String,
    text: String,
}

/// Recursively collect `.rs` files.
fn collect_rs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_rs(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

/// Index `dir` once, then run every arm at every `k` against that one index.
pub async fn run_codeeval(
    dir: &str,
    ks: &[usize],
    embedder_choice: &str,
    suite: &[CodeQuery],
) -> Result<CodeEvalReport> {
    let scope = Scope::global("code");
    let embedder = build_embedder(embedder_choice)?;

    // --- parse ---
    let mut paths = Vec::new();
    collect_rs(std::path::Path::new(dir), &mut paths);
    paths.sort();
    if paths.is_empty() {
        return Err(anyhow!("no .rs files under {dir}"));
    }

    let mut file_text: HashMap<String, String> = HashMap::new();
    let mut parsed: Vec<ParsedFile> = Vec::new();
    for p in &paths {
        let key = p.to_string_lossy().to_string();
        let Ok(src) = std::fs::read_to_string(p) else {
            continue;
        };
        if let Ok(pf) = HeuristicParser.parse(&key, "rust", &src) {
            file_text.insert(key, src);
            parsed.push(pf);
        }
    }

    // --- plan -> events, and keep the side tables scoring needs ---
    let plan = CodeIndexer::new(scope.clone()).plan(&parsed);
    let mut symbols: HashMap<MemoryRef, SymbolInfo> = HashMap::new();
    let mut links: HashMap<MemoryRef, Vec<MemoryRef>> = HashMap::new();

    let vector = Arc::new(VectorView::new(
        embedder.dim(),
        embedder.model_id().to_string(),
    ));
    let bm25 = Arc::new(Bm25View::new().map_err(|e| anyhow!("bm25 init: {e}"))?);

    for event in &plan.events {
        match event {
            Event::MemoryWritten(m) => {
                // Embed inline so the vector view is populated synchronously;
                // the real server does this on an async worker.
                let vectors = embedder
                    .embed(std::slice::from_ref(&m.content))
                    .await
                    .map_err(|e| anyhow!("embed: {e}"))?;
                let mut m = m.clone();
                m.embedding = vectors.into_iter().next();

                let path = m
                    .tags
                    .iter()
                    .find(|t| t.ends_with(".rs"))
                    .cloned()
                    .unwrap_or_default();
                symbols.insert(
                    MemoryRef(m.id),
                    SymbolInfo {
                        name: m.keywords.first().cloned().unwrap_or_default(),
                        path,
                        text: m.content.clone(),
                    },
                );

                let entry = LogEntry {
                    id: new_id(),
                    event: Event::MemoryWritten(m),
                };
                vector
                    .apply(&entry)
                    .await
                    .map_err(|e| anyhow!("vector apply: {e}"))?;
                bm25.apply(&entry)
                    .await
                    .map_err(|e| anyhow!("bm25 apply: {e}"))?;
            }
            Event::MemoryLinksUpdated { id, links: l } => {
                links.insert(*id, l.clone());
            }
            _ => {}
        }
    }

    let retriever = HybridRetriever::new(vector, bm25, embedder.clone());

    // --- run the arms, all against this one index ---
    let mut arms: Vec<ArmResult> = Vec::with_capacity(ks.len() * 3);
    for &k in ks {
        let blank = |name| ArmResult {
            name,
            k,
            recalled: 0,
            total: 0,
            tokens: 0,
        };
        let mut whole_file = blank("whole-file");
        let mut symbol_only = blank("symbol");
        let mut expanded = blank("symbol+expand");

        for q in suite {
            let hits = retriever
                .search(&Query {
                    text: q.question.to_string(),
                    scope: scope.clone(),
                    k,
                    time_filter: None,
                })
                .await
                .map_err(|e| anyhow!("search: {e}"))?;

            // -- arm: symbol --
            let picked: Vec<&SymbolInfo> =
                hits.iter().filter_map(|h| symbols.get(&h.memory)).collect();
            if std::env::var("MNESIO_BENCH_DEBUG").is_ok() {
                eprintln!(
                    "  q={:?} want={} got={:?}",
                    q.question,
                    q.expect,
                    picked.iter().map(|s| &s.name).collect::<Vec<_>>()
                );
            }
            symbol_only.total += 1;
            symbol_only.tokens += picked.iter().map(|s| est_tokens(&s.text)).sum::<usize>();
            if picked.iter().any(|s| s.name == q.expect) {
                symbol_only.recalled += 1;
            }

            // -- arm: whole-file (dedup files; an agent reads a file once) --
            let mut seen_files: Vec<&str> = Vec::new();
            for s in &picked {
                if !seen_files.contains(&s.path.as_str()) {
                    seen_files.push(&s.path);
                }
            }
            whole_file.total += 1;
            whole_file.tokens += seen_files
                .iter()
                .filter_map(|p| file_text.get(*p))
                .map(|t| est_tokens(t))
                .sum::<usize>();
            // The whole file is present, so the expected symbol is recalled iff any
            // retrieved file contains its definition.
            if seen_files
                .iter()
                .any(|p| symbols.values().any(|s| &s.path == p && s.name == q.expect))
            {
                whole_file.recalled += 1;
            }

            // -- arm: symbol + 1-hop callees --
            let mut ids: Vec<MemoryRef> = hits.iter().map(|h| h.memory).collect();
            for h in &hits {
                if let Some(l) = links.get(&h.memory) {
                    for r in l {
                        if !ids.contains(r) {
                            ids.push(*r);
                        }
                    }
                }
            }
            let exp: Vec<&SymbolInfo> = ids.iter().filter_map(|r| symbols.get(r)).collect();
            expanded.total += 1;
            expanded.tokens += exp.iter().map(|s| est_tokens(&s.text)).sum::<usize>();
            if exp.iter().any(|s| s.name == q.expect) {
                expanded.recalled += 1;
            }
        }
        arms.extend([whole_file, symbol_only, expanded]);
    }

    Ok(CodeEvalReport {
        embedder: embedder_choice.to_string(),
        index: plan.stats,
        arms,
    })
}

/// Human-readable summary.
pub fn format_report(r: &CodeEvalReport) -> String {
    let mut out = format!(
        "# code retrieval — embedder={} · {} files · {} symbols\n\n\
         call edges: {} resolved / {} unresolved / {} ambiguous\n\n",
        r.embedder,
        r.index.files,
        r.index.symbols,
        r.index.edges.resolved,
        r.index.edges.unresolved,
        r.index.edges.ambiguous
    );

    out.push_str("| k | arm | recall | tokens/query |\n|---|---|---|---|\n");
    for a in &r.arms {
        out.push_str(&format!(
            "| {} | {} | {:.0}% ({}/{}) | {:.0} |\n",
            a.k,
            a.name,
            a.recall() * 100.0,
            a.recalled,
            a.total,
            a.tokens_per_query(),
        ));
    }

    // The comparison that actually matters. Reading off a single k is
    // misleading — the arms hit a given recall at different k, so the fair
    // question is "at the same recall, what does each cost?".
    let peak = r.peak_recall("symbol");
    out.push_str(&format!(
        "\n## iso-recall — cheapest cell reaching {:.0}%\n\n\
         | arm | k | tokens/query |\n|---|---|---|\n",
        peak * 100.0
    ));
    let mut base_tokens = None;
    for arm in ["whole-file", "symbol", "symbol+expand"] {
        match r.cheapest_at_recall(arm, peak) {
            Some(c) => {
                if arm == "whole-file" {
                    base_tokens = Some(c.tokens_per_query());
                }
                out.push_str(&format!(
                    "| {} | {} | {:.0} |\n",
                    arm,
                    c.k,
                    c.tokens_per_query()
                ));
            }
            None => out.push_str(&format!("| {arm} | — | never reaches it |\n")),
        }
    }
    if let (Some(base), Some(sym)) = (
        base_tokens,
        r.cheapest_at_recall("symbol", peak)
            .map(|c| c.tokens_per_query()),
    ) {
        out.push_str(&format!(
            "\n**{:.1}× fewer tokens at equal recall.**\n",
            base / sym.max(0.001)
        ));
    }

    // The honest caveat that decides whether 17B is met: matching the symbol
    // arm's *peak* is not the same as matching whole-file's peak.
    let wf_peak = r.peak_recall("whole-file");
    if wf_peak > peak + 1e-6 {
        out.push_str(&format!(
            "\n⚠ whole-file peaks at {:.0}% but symbol tops out at {:.0}% — the \
             symbol arm is cheaper *and* strictly worse at the ceiling, so this \
             is a trade, not a free win.\n",
            wf_peak * 100.0,
            peak * 100.0
        ));
    }

    out.push_str(
        "\n_Tokens estimated as chars/4 — identical across arms, so ratios hold; \
         absolute counts are indicative. Hand-built suite over our own source: a \
         smoke test, not a benchmark._\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimate_is_monotonic_and_nonzero() {
        assert!(est_tokens("abcd") >= 1);
        assert!(est_tokens("a".repeat(400).as_str()) > est_tokens("a".repeat(40).as_str()));
    }

    #[test]
    fn arm_math_handles_empty_without_dividing_by_zero() {
        let a = ArmResult {
            name: "x",
            k: 5,
            recalled: 0,
            total: 0,
            tokens: 0,
        };
        assert_eq!(a.recall(), 0.0);
        assert_eq!(a.tokens_per_query(), 0.0);
    }

    #[test]
    fn recall_and_tokens_per_query_are_computed() {
        let a = ArmResult {
            name: "x",
            k: 5,
            recalled: 3,
            total: 4,
            tokens: 400,
        };
        assert!((a.recall() - 0.75).abs() < 1e-6);
        assert!((a.tokens_per_query() - 100.0).abs() < 1e-6);
    }
}

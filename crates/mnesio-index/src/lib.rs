//! # mnesio-index
//!
//! Materialized retrieval views over the event log: a vector index
//! (`hnsw_rs`, Phase 0) and a BM25 index (`tantivy`, Phase 0), unified by a
//! hybrid [`Retriever`]. The custom filtered-HNSW variant (report §7, Phase 3)
//! lands here later — keep this trait-shaped so it can be swapped in.
//!
//! STATUS: Phase 0 in progress. `VectorView`, `Bm25View`, `MockEmbedder`,
//! and `HybridRetriever` (RRF fusion with explainable per-signal breakdown)
//! are all in. Custom filtered HNSW (Phase 3) and persistent indexes are
//! the next bumps.

pub mod acl;
pub mod bm25;
pub mod chunker;
pub mod context_tree;
pub mod embedder;
pub mod partitioned_vector;
pub mod profile;
pub mod rerank;
pub mod synthesizer;
pub mod vector;

pub use acl::{AgentAclView, AgentAttribution};
pub use bm25::{Bm25Tier, Bm25View};
pub use chunker::{Chunker, ParagraphChunker};
pub use context_tree::ContextTree;
#[cfg(feature = "fastembed")]
pub use embedder::FastEmbedEmbedder;
pub use embedder::MockEmbedder;
pub use partitioned_vector::{TenantPartitionedVectorView, TenantStats};
pub use profile::{AttributeState, ProfileValue, ProfileView};
pub use rerank::{LexicalReranker, CODE_BOOST};
pub use synthesizer::SnippetSynthesizer;
pub use vector::VectorView;

use mnesio_core::types::MemoryRef;
use mnesio_core::{Embedder, Hit, MnesioError, Query, Retriever};
use std::collections::HashMap;
use std::sync::Arc;

/// A graph-proximity signal source, injected so `mnesio-index` stays
/// decoupled from `mnesio-graph` (the host wires the real graph view
/// behind this; tests use a fake). Given a candidate memory and the full
/// candidate set under consideration, return a boost in `[0.0, 1.0]` —
/// e.g. "how densely is this memory linked to the other candidates?".
/// Memories that sit at the centre of the retrieved neighbourhood are
/// usually more on-topic than isolated keyword matches.
pub trait ProximityProvider: Send + Sync {
    fn boost(&self, memory: MemoryRef, candidates: &[MemoryRef]) -> f32;
}

/// Content resolver seam (Phase 16). A content-aware reranker needs each
/// candidate's full text, but a [`Hit`] only carries a [`MemoryRef`]. This
/// trait resolves `MemoryRef -> content` so the reranker stays decoupled
/// from any particular store — production wires [`Bm25View`] (which has the
/// content `STORED`); tests use an in-memory map.
pub trait ContentProvider: Send + Sync {
    /// The stored content of a memory, or `None` if it isn't live.
    fn content(&self, memory: MemoryRef) -> Option<String>;
}

impl ContentProvider for Bm25View {
    fn content(&self, memory: MemoryRef) -> Option<String> {
        Bm25View::content(self, memory)
    }
}

/// Final reorder seam. The fused+weighted list is handed to a reranker
/// before truncation to `k` — this is where a cross-encoder or LLM
/// reranker plugs in (Mem0/Zep ship Cohere/HF rerankers here). The
/// default [`IdentityReranker`] is a no-op so fusion behaviour is
/// unchanged unless a reranker is explicitly wired. Phase 16 ships a
/// content-aware [`rerank::LexicalReranker`] here.
#[async_trait::async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, hits: Vec<Hit>) -> Result<Vec<Hit>, MnesioError>;
}

/// No-op reranker — returns the fused order untouched.
pub struct IdentityReranker;

#[async_trait::async_trait]
impl Reranker for IdentityReranker {
    async fn rerank(&self, _query: &str, hits: Vec<Hit>) -> Result<Vec<Hit>, MnesioError> {
        Ok(hits)
    }
}

/// Reciprocal-rank fusion constant — the classic RRF paper uses k=60 and the
/// hybrid-search literature has converged on that value. Smaller k weights
/// the top of each list more aggressively; larger k flattens the curve.
const RRF_K: f32 = 60.0;

/// Hybrid retriever fusing the vector and BM25 views via *weighted* reciprocal
/// rank fusion. Each result's `breakdown` carries the per-signal raw score
/// (not normalized) plus the fused RRF component, so downstream UIs and tests
/// can see exactly why a memory ranked where it did.
///
/// **Why weighted, not flat RRF?** Standard RRF treats every signal equally.
/// That assumption breaks when one signal is non-semantic — e.g. when the
/// embedder is a stub that hashes bytes into vectors (Phase 0's
/// `MockEmbedder`). A noise signal at rank 1 will out-score a real signal
/// at rank 2 in flat RRF, polluting the result. Per-signal weights let us
/// down-weight a known-noisy signal until the real one lands (FastEmbed in
/// the next slice).
///
/// Default weights:
/// - `w_bm25 = 1.0`
/// - `w_vector = 0.0` if the embedder reports a `mock-` model_id, else `1.0`
pub struct HybridRetriever {
    vector: Arc<VectorView>,
    bm25: Arc<Bm25View>,
    embedder: Arc<dyn Embedder>,
    /// Over-fetch factor — pull `over_fetch * k` from each view before fusing
    /// so a hit that ranks 8th in one signal but 1st in another still has a
    /// chance to land in the top-k. Phase 0 default.
    over_fetch: usize,
    w_vector: f32,
    w_bm25: f32,
    /// Weight on the recency signal (derived from each memory's ULID
    /// timestamp). `0.0` by default — opt in via [`HybridRetriever::with_recency_weight`].
    w_recency: f32,
    /// Optional graph-proximity signal + its weight. `None` by default.
    proximity: Option<Arc<dyn ProximityProvider>>,
    w_proximity: f32,
    /// Final reorder stage. Defaults to [`IdentityReranker`].
    reranker: Arc<dyn Reranker>,
}

impl HybridRetriever {
    pub fn new(vector: Arc<VectorView>, bm25: Arc<Bm25View>, embedder: Arc<dyn Embedder>) -> Self {
        // Detect mock embedders so we don't let hash noise dictate ranking.
        // The convention `model_id() == "mock-v*"` is set by `MockEmbedder`;
        // a real local/HTTP embedder uses its own model identifier.
        let w_vector = if embedder.model_id().starts_with("mock-") {
            0.0
        } else {
            1.0
        };
        Self {
            vector,
            bm25,
            embedder,
            over_fetch: 4,
            w_vector,
            w_bm25: 1.0,
            w_recency: 0.0,
            proximity: None,
            w_proximity: 0.0,
            reranker: Arc::new(IdentityReranker),
        }
    }

    /// Override the default per-signal weights. Useful when the host wants
    /// to A/B different fusion strategies or pin behavior in a test.
    pub fn with_weights(mut self, w_vector: f32, w_bm25: f32) -> Self {
        self.w_vector = w_vector;
        self.w_bm25 = w_bm25;
        self
    }

    /// Enable the recency signal. `w` scales a `[0,1]` recency bonus
    /// (newest qualifying hit = 1.0) added on top of the fused score.
    /// Recency only *reorders* hits that already have a real
    /// vector/BM25 signal — it never resurrects a zero-signal hit.
    pub fn with_recency_weight(mut self, w: f32) -> Self {
        self.w_recency = w;
        self
    }

    /// Wire a graph-proximity provider + its weight. Like recency, this
    /// is a bonus on already-qualifying hits, not an independent signal.
    pub fn with_proximity(mut self, provider: Arc<dyn ProximityProvider>, w: f32) -> Self {
        self.proximity = Some(provider);
        self.w_proximity = w;
        self
    }

    /// Install a final reranker stage (default: identity).
    /// Set how many candidates each view contributes before fusion.
    ///
    /// The default 4 is tuned for prose. Code retrieval wants more: a Phase-17B
    /// miss taxonomy over 400 real repository tasks found 29% of queries had
    /// their gold symbol present in a 400-deep candidate list but ranked below
    /// `k` — signal a reranker can only use if the pool is deep enough to
    /// contain it. Costs a linear amount of scoring work per query.
    pub fn with_over_fetch(mut self, factor: usize) -> Self {
        self.over_fetch = factor.max(1);
        self
    }

    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = reranker;
        self
    }

    /// Direct access for callers that want the individual signal lists (the
    /// viz/UI wants all three to render the score breakdown side-by-side).
    pub fn vector(&self) -> &VectorView {
        &self.vector
    }

    pub fn bm25(&self) -> &Bm25View {
        &self.bm25
    }

    pub fn embedder(&self) -> &dyn Embedder {
        self.embedder.as_ref()
    }

    pub fn weights(&self) -> (f32, f32) {
        (self.w_vector, self.w_bm25)
    }
}

#[async_trait::async_trait]
impl Retriever for HybridRetriever {
    async fn search(&self, query: &Query) -> Result<Vec<Hit>, MnesioError> {
        if query.k == 0 {
            return Ok(Vec::new());
        }
        let over_k = query.k.saturating_mul(self.over_fetch).max(query.k);

        // Embed the textual query once for the vector view; BM25 consumes the
        // raw text directly.
        let embeddings = self
            .embedder
            .embed(std::slice::from_ref(&query.text))
            .await?;
        let query_vec = embeddings
            .first()
            .ok_or_else(|| MnesioError::Index("embedder returned no vectors".into()))?;

        let vector_hits = self.vector.search(query_vec, over_k, &query.scope)?;
        let bm25_hits = self.bm25.search(&query.text, over_k, &query.scope)?;

        // Weighted RRF fuses by *ranks*, scaled by per-signal weight. Rank-
        // based fusion is robust to wildly different score scales between
        // BM25 and L2/cosine; weighting on top of that lets us mute a
        // known-noisy signal (Phase-0 mock embedder) without re-engineering
        // the fusion. We keep the raw scores in the breakdown for visibility
        // even when a signal's weight is zero.
        let mut bucket: HashMap<MemoryRef, FusedHit> = HashMap::new();
        for (rank, h) in vector_hits.iter().enumerate() {
            let e = bucket.entry(h.memory).or_default();
            e.vector_score = h.score;
            e.rrf += self.w_vector / (RRF_K + (rank + 1) as f32);
        }
        for (rank, h) in bm25_hits.iter().enumerate() {
            let e = bucket.entry(h.memory).or_default();
            e.bm25_score = h.score;
            e.rrf += self.w_bm25 / (RRF_K + (rank + 1) as f32);
        }

        // Qualifying hits = those with a real (weighted) vector/BM25
        // signal. Recency + proximity are *bonuses on these*, never a way
        // to resurrect a zero-signal hit.
        let qualifying: Vec<(MemoryRef, FusedHit)> =
            bucket.into_iter().filter(|(_, f)| f.rrf > 0.0).collect();
        let candidate_refs: Vec<MemoryRef> = qualifying.iter().map(|(m, _)| *m).collect();

        // Recency normalisation window: ULIDs are time-ordered, so the
        // memory id's timestamp *is* its creation time — no extra state.
        // Newest qualifying hit scores 1.0, oldest 0.0; all-equal → 0.5.
        let (min_ts, max_ts) = candidate_refs.iter().fold((u64::MAX, 0u64), |(lo, hi), m| {
            let t = m.0.timestamp_ms();
            (lo.min(t), hi.max(t))
        });
        let span = max_ts.saturating_sub(min_ts);
        let recency_score = |m: MemoryRef| -> f32 {
            if self.w_recency == 0.0 {
                return 0.0;
            }
            if span == 0 {
                0.5
            } else {
                (m.0.timestamp_ms().saturating_sub(min_ts)) as f32 / span as f32
            }
        };

        let mut hits: Vec<Hit> = qualifying
            .into_iter()
            .map(|(memory, f)| {
                let recency = recency_score(memory);
                let graph = match &self.proximity {
                    Some(p) if self.w_proximity != 0.0 => p.boost(memory, &candidate_refs),
                    _ => 0.0,
                };
                let score = f.rrf + self.w_recency * recency + self.w_proximity * graph;
                let mut breakdown = vec![
                    ("vector".to_string(), f.vector_score),
                    ("bm25".to_string(), f.bm25_score),
                    ("rrf".to_string(), f.rrf),
                ];
                // Only surface the auxiliary signals when they're active,
                // so the breakdown stays clean for the common case.
                if self.w_recency != 0.0 {
                    breakdown.push(("recency".to_string(), recency));
                }
                if self.w_proximity != 0.0 && self.proximity.is_some() {
                    breakdown.push(("graph".to_string(), graph));
                }
                Hit {
                    memory,
                    score,
                    breakdown,
                }
            })
            .collect();
        // Descending score, then ascending memory id. The tie-break is not
        // cosmetic: `bucket` is a `HashMap`, Rust seeds `HashMap` per process,
        // and this vec is built by draining it — so equal scores arrived in a
        // different order in every process. `Ordering::Equal` then left them
        // that way, and a stable sort faithfully preserved the accident.
        //
        // Measured before this: 10 of 40 real queries returned a different
        // packed context across processes even after `VectorView` was made
        // exact. Every difference was at the budget boundary — a tied
        // candidate displacing another — which is exactly where ties decide
        // what an agent gets to see.
        //
        // `total_cmp` rather than `partial_cmp(..).unwrap_or(Equal)`: a `NaN`
        // score would otherwise compare Equal to everything and reintroduce
        // the same dependence on arrival order.
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.memory.0.cmp(&b.memory.0))
        });

        // Final reorder stage (identity by default). Rerank the
        // over-fetched, fused list, then truncate to k so a reranker can
        // promote a hit from outside the naive top-k.
        let mut hits = self.reranker.rerank(&query.text, hits).await?;
        hits.truncate(query.k);
        Ok(hits)
    }
}

#[derive(Default)]
struct FusedHit {
    vector_score: f32,
    bm25_score: f32,
    rrf: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::entity::{Memory, Provenance};
    use mnesio_core::event::{Event, LogEntry};
    use mnesio_core::traits::MaterializedView;
    use mnesio_core::types::{new_id, BiTemporal};
    use mnesio_core::Scope;

    async fn build_corpus() -> (
        Arc<VectorView>,
        Arc<Bm25View>,
        Arc<MockEmbedder>,
        Scope,
        Vec<MemoryRef>,
    ) {
        let embedder = Arc::new(MockEmbedder::new(32));
        let vector = Arc::new(VectorView::new(
            embedder.dim(),
            embedder.model_id().to_string(),
        ));
        let bm25 = Arc::new(Bm25View::new().unwrap());
        let scope = Scope::global("test");

        let contents = [
            ("acme reported quarterly earnings", "earnings"),
            ("revenue up 15% year over year", "revenue"),
            ("supply chain stabilizing into Q4", "operations"),
            ("EMEA margins compressed this quarter", "margin"),
        ];
        let mut refs = Vec::new();
        for (text, tag) in contents {
            let emb = embedder.embed(&[text.to_string()]).await.unwrap();
            let m = Memory {
                id: new_id(),
                scope: scope.clone(),
                content: text.into(),
                keywords: vec![],
                tags: vec![tag.into()],
                context: String::new(),
                embedding: Some(emb.into_iter().next().unwrap()),
                links: vec![],
                parent: None,
                evolution_count: 0,
                time: BiTemporal::now(),
                provenance: Provenance::default(),
                source: None,
                position: None,
            };
            refs.push(MemoryRef(m.id));
            let entry = LogEntry {
                id: new_id(),
                event: Event::MemoryWritten(m),
            };
            vector.apply(&entry).await.unwrap();
            bm25.apply(&entry).await.unwrap();
        }
        (vector, bm25, embedder, scope, refs)
    }

    #[tokio::test]
    async fn hybrid_returns_bm25_keyword_match_on_top() {
        let (vector, bm25, embedder, scope, refs) = build_corpus().await;
        let retriever = HybridRetriever::new(vector, bm25, embedder);
        let hits = retriever
            .search(&Query {
                text: "revenue".into(),
                scope,
                k: 3,
                time_filter: None,
            })
            .await
            .unwrap();
        assert!(!hits.is_empty());
        let ids: Vec<_> = hits.iter().map(|h| h.memory).collect();
        assert!(
            ids.contains(&refs[1]),
            "revenue memory must appear in hybrid top-k; got {ids:?}"
        );
    }

    #[tokio::test]
    async fn breakdown_exposes_all_three_signals() {
        let (vector, bm25, embedder, scope, _refs) = build_corpus().await;
        let retriever = HybridRetriever::new(vector, bm25, embedder);
        let hits = retriever
            .search(&Query {
                text: "earnings".into(),
                scope,
                k: 2,
                time_filter: None,
            })
            .await
            .unwrap();
        let h = hits.first().expect("expected at least one hit");
        let names: Vec<_> = h.breakdown.iter().map(|(n, _)| n.as_str()).collect();
        // The three base signals are always present; recency/graph are
        // added only when their weights are non-zero (off by default).
        assert!(names.contains(&"vector"));
        assert!(names.contains(&"bm25"));
        assert!(names.contains(&"rrf"));
        assert!(!names.contains(&"recency"), "recency off by default");
        assert!(!names.contains(&"graph"), "graph off by default");
    }

    #[tokio::test]
    async fn mock_embedder_zeroes_vector_weight() {
        let embedder = Arc::new(MockEmbedder::new(32));
        let vector = Arc::new(VectorView::new(
            embedder.dim(),
            embedder.model_id().to_string(),
        ));
        let bm25 = Arc::new(Bm25View::new().unwrap());
        let r = HybridRetriever::new(vector, bm25, embedder);
        let (wv, wb) = r.weights();
        assert_eq!(wv, 0.0, "mock embedder must zero out vector weight");
        assert_eq!(wb, 1.0);
    }

    #[tokio::test]
    async fn with_weights_overrides_default() {
        let embedder = Arc::new(MockEmbedder::new(32));
        let vector = Arc::new(VectorView::new(
            embedder.dim(),
            embedder.model_id().to_string(),
        ));
        let bm25 = Arc::new(Bm25View::new().unwrap());
        let r = HybridRetriever::new(vector, bm25, embedder).with_weights(0.7, 0.3);
        assert_eq!(r.weights(), (0.7, 0.3));
    }

    #[tokio::test]
    async fn zero_vector_weight_excludes_vector_only_hits() {
        // Set up: corpus with a strong BM25 match and a high-ranking vector
        // match for a "different" memory. With w_vector=0, only the BM25
        // match should land in hybrid results.
        let (vector, bm25, embedder, scope, refs) = build_corpus().await;
        let retriever = HybridRetriever::new(vector, bm25, embedder); // mock → w_vec=0

        let hits = retriever
            .search(&Query {
                text: "revenue".into(),
                scope,
                k: 5,
                time_filter: None,
            })
            .await
            .unwrap();

        // Every hit must have a non-zero BM25 contribution — vector-only
        // matches (the hash noise) must not appear.
        for h in &hits {
            let bm = h
                .breakdown
                .iter()
                .find(|(n, _)| n == "bm25")
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
            assert!(
                bm > 0.0,
                "with w_vector=0 every hybrid hit must have BM25 support; got {h:?}"
            );
        }
        // And the revenue memory must be the top hit, not noise.
        assert!(!hits.is_empty());
        assert_eq!(
            hits[0].memory, refs[1],
            "revenue memory must rank #1 with BM25-only fusion"
        );
    }

    #[tokio::test]
    async fn recency_weight_reorders_equal_signal_hits() {
        // Two memories with the *same* keyword so BM25 ranks them equally;
        // the newer one (later ULID) must win once recency is weighted.
        let embedder = Arc::new(MockEmbedder::new(32));
        let vector = Arc::new(VectorView::new(
            embedder.dim(),
            embedder.model_id().to_string(),
        ));
        let bm25 = Arc::new(Bm25View::new().unwrap());
        let scope = Scope::global("t");

        let mut refs = Vec::new();
        for _ in 0..2 {
            let m = Memory {
                id: new_id(),
                scope: scope.clone(),
                content: "quarterly revenue update".into(),
                keywords: vec![],
                tags: vec![],
                context: String::new(),
                embedding: None,
                links: vec![],
                parent: None,
                evolution_count: 0,
                time: BiTemporal::now(),
                provenance: Provenance::default(),
                source: None,
                position: None,
            };
            refs.push(MemoryRef(m.id));
            let entry = LogEntry {
                id: new_id(),
                event: Event::MemoryWritten(m),
            };
            vector.apply(&entry).await.unwrap();
            bm25.apply(&entry).await.unwrap();
            // Ensure a strictly later ULID timestamp for the second memory.
            tokio::time::sleep(std::time::Duration::from_millis(3)).await;
        }
        let newer = refs[1]; // created second → later ULID

        let r = HybridRetriever::new(vector, bm25, embedder).with_recency_weight(0.05);
        let hits = r
            .search(&Query {
                text: "revenue".into(),
                scope,
                k: 2,
                time_filter: None,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0].memory, newer,
            "newer memory should rank first with recency on"
        );
        assert!(hits[0].breakdown.iter().any(|(n, _)| n == "recency"));
    }

    /// Proximity provider that boosts exactly one target ref.
    struct BoostOne(MemoryRef);
    impl ProximityProvider for BoostOne {
        fn boost(&self, memory: MemoryRef, _candidates: &[MemoryRef]) -> f32 {
            if memory == self.0 {
                1.0
            } else {
                0.0
            }
        }
    }

    #[tokio::test]
    async fn proximity_boost_promotes_target() {
        let (vector, bm25, embedder, scope, refs) = build_corpus().await;
        // "quarter" appears in two memories (idx 2 + 3) so both qualify on
        // BM25; boost idx 3 and it should top the list.
        let target = refs[3];
        let r = HybridRetriever::new(vector, bm25, embedder)
            .with_proximity(Arc::new(BoostOne(target)), 0.1);
        let hits = r
            .search(&Query {
                text: "quarter".into(),
                scope,
                k: 4,
                time_filter: None,
            })
            .await
            .unwrap();
        assert!(!hits.is_empty());
        assert_eq!(
            hits[0].memory, target,
            "proximity-boosted memory should rank first"
        );
        assert!(hits[0].breakdown.iter().any(|(n, _)| n == "graph"));
    }

    /// Reranker that reverses the fused order — proves the stage runs.
    struct ReverseReranker;
    #[async_trait::async_trait]
    impl Reranker for ReverseReranker {
        async fn rerank(&self, _q: &str, mut hits: Vec<Hit>) -> Result<Vec<Hit>, MnesioError> {
            hits.reverse();
            Ok(hits)
        }
    }

    #[tokio::test]
    async fn reranker_stage_reorders_then_truncates() {
        let (vector, bm25, embedder, scope, _refs) = build_corpus().await;
        let baseline = HybridRetriever::new(vector.clone(), bm25.clone(), embedder.clone());
        let base_hits = baseline
            .search(&Query {
                text: "quarter".into(),
                scope: scope.clone(),
                k: 2,
                time_filter: None,
            })
            .await
            .unwrap();

        let reranked =
            HybridRetriever::new(vector, bm25, embedder).with_reranker(Arc::new(ReverseReranker));
        let rr_hits = reranked
            .search(&Query {
                text: "quarter".into(),
                scope,
                k: 2,
                time_filter: None,
            })
            .await
            .unwrap();
        // Reverse runs over the over-fetched list before truncation, so the
        // reranked top should differ from the fused top when >k qualified.
        assert!(!rr_hits.is_empty());
        assert_ne!(
            base_hits[0].memory, rr_hits[0].memory,
            "reranker must change the head order"
        );
    }

    #[tokio::test]
    async fn k_zero_returns_empty() {
        let (vector, bm25, embedder, scope, _) = build_corpus().await;
        let retriever = HybridRetriever::new(vector, bm25, embedder);
        let hits = retriever
            .search(&Query {
                text: "revenue".into(),
                scope,
                k: 0,
                time_filter: None,
            })
            .await
            .unwrap();
        assert!(hits.is_empty());
    }
}

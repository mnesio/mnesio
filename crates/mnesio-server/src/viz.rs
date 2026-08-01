//! Visualization endpoints.
//!
//! Exposes the event log + reconstructed memory state + retrieval results as
//! JSON for the browser-side renderer in `index.html`. The viz is a *consumer*
//! of the event log: it never owns derived state of its own, so it can be
//! torn down and rebuilt at any moment without violating the "single source
//! of truth" hard rule (CLAUDE.md §4).
//!
//! Phase 0 scope (intentional simplifications):
//! - `/api/snapshot` replays the entire log on each request. Fine while we're
//!   hand-driving demos. Once the log grows we'll switch to a warm in-memory
//!   snapshot updated off the same event tail every other view uses.
//! - `/api/search` always searches the `demo` tenant (the only writer in
//!   demo mode). Multi-tenant filtering on the wire belongs alongside auth.

use crate::metrics::{
    self, MetricsCollector, MetricsHistory, MetricsRollup, PhaseTimings, RetrievalCounts,
    ScoreEnvelope, SearchMetrics,
};
use axum::extract::{Query as AxumQuery, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::Json;
use mnesio_core::entity::Memory;
use mnesio_core::event::Event;
use mnesio_core::synthesizer::Passage;
use mnesio_core::types::MemoryRef;
use mnesio_core::{
    Embedder, EventLog, Hit, Id, Query as RetrievalQuery, Retriever, Scope, Synthesizer,
};
use mnesio_evolve::EvolutionWorker;
use mnesio_graph::{FjallGraphView, Relation};
use mnesio_index::{AgentAclView, Bm25View, HybridRetriever, ProfileView, VectorView};
use mnesio_procedural::{ProceduralStore, ProceduralWorker};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use time::OffsetDateTime;

/// Application state injected into every handler.
pub struct AppState {
    pub log: Arc<dyn EventLog>,
    pub vector: Arc<VectorView>,
    pub bm25: Arc<Bm25View>,
    /// Phase-4 bi-temporal property graph view. Fed from the log tail by
    /// the graph worker; queried by `/api/graph` for neighborhood +
    /// lineage exploration.
    pub graph: Arc<FjallGraphView>,
    pub embedder: Arc<dyn Embedder>,
    pub retriever: Arc<HybridRetriever>,
    /// Extractive synthesizer used to compose the citation-bearing answer
    /// card returned alongside the raw signal columns. Trait-shaped so an
    /// MMR or LLM-backed synthesizer can swap in without touching the
    /// handler.
    pub synthesizer: Arc<dyn Synthesizer>,
    /// Rolling search-metrics collector. Every `/api/search` records into
    /// this; the `/api/metrics` endpoint reads aggregates back out.
    pub metrics: Arc<MetricsCollector>,
    /// Phase-1 memory-evolution worker. `None` when disabled via
    /// `MNESIO_EVOLVE=off`; the `/api/evolve/metrics` handler degrades
    /// gracefully to "disabled" in that case.
    pub evolution: Option<Arc<EvolutionWorker>>,
    /// Phase-2 procedural compiler worker. `None` when disabled (the
    /// default — opt in via `MNESIO_PROCEDURAL=on`). The companion
    /// `/api/procedural/metrics` handler reports `enabled=false` when
    /// the worker is missing.
    pub procedural: Option<Arc<ProceduralWorker>>,
    /// Active-version registry. Lives outside the worker so the
    /// dashboard handler can still query it (for active artifact
    /// counts) even when the worker isn't running.
    pub procedural_store: Arc<ProceduralStore>,
    /// Phase-7 ingestion worker (extract + consolidate raw observations).
    /// `None` when disabled; `/api/ingest/metrics` reports `enabled=false`.
    pub ingestion: Option<Arc<crate::ingestion_worker::IngestionWorker>>,
    /// Phase-8 (P1#6) profile / persona view — stable per-subject
    /// attributes surfaced by `/api/profile`.
    pub profile: Arc<ProfileView>,
    /// Phase-8 (P1#7) multi-agent attribution + inter-agent ACL view.
    /// `/api/search?actor=X` filters hits through it; `/api/agents`
    /// reports attribution + grants.
    pub acl: Arc<AgentAclView>,
    /// Tenant the viz operates in. The demo writer uses "demo"; production
    /// will plumb this from auth.
    pub default_tenant: String,
}

// ---------- snapshot ----------

#[derive(Serialize)]
pub struct Snapshot {
    pub log_head: Option<String>,
    pub log_size: usize,
    pub events: Vec<EventSummary>,
    pub memories: Vec<MemoryView>,
    pub sources: Vec<SourceView>,
}

#[derive(Serialize)]
pub struct EventSummary {
    pub id: String,
    pub kind: &'static str,
    pub timestamp_ms: u64,
    pub description: String,
}

#[derive(Serialize)]
pub struct MemoryView {
    pub id: String,
    pub scope: ScopeDto,
    pub content: String,
    pub tags: Vec<String>,
    pub keywords: Vec<String>,
    pub has_embedding: bool,
    pub parent: Option<String>,
    pub links: Vec<String>,
    pub evolution_count: u16,
    pub is_invalidated: bool,
    /// Source document this memory was chunked from. `None` for standalone
    /// memories that weren't ingested as part of a longer document.
    pub source_id: Option<String>,
    pub source_title: Option<String>,
    pub position: Option<u32>,
}

#[derive(Serialize)]
pub struct SourceView {
    pub id: String,
    pub title: String,
    pub uri: Option<String>,
    pub chunk_count: u32,
}

#[derive(Serialize)]
pub struct ScopeDto {
    pub tenant: String,
    pub user: Option<String>,
    pub session: Option<String>,
}

impl From<&Memory> for MemoryView {
    fn from(m: &Memory) -> Self {
        Self {
            id: m.id.to_string(),
            scope: ScopeDto {
                tenant: m.scope.tenant.clone(),
                user: m.scope.user.clone(),
                session: m.scope.session.clone(),
            },
            content: m.content.clone(),
            tags: m.tags.clone(),
            keywords: m.keywords.clone(),
            has_embedding: m.embedding.is_some(),
            parent: m.parent.map(|p| p.0.to_string()),
            links: m.links.iter().map(|l| l.0.to_string()).collect(),
            evolution_count: m.evolution_count,
            is_invalidated: false,
            source_id: m.source.map(|s| s.0.to_string()),
            // `source_title` is filled in after walking SourceIngested events.
            source_title: None,
            position: m.position,
        }
    }
}

/// Helper to build a `MemoryView` and stash the raw source Id separately
/// (instead of round-tripping through a string parse) so the second-pass
/// title join in the snapshot handler stays cheap.
struct MemoryViewAndSource {
    view: MemoryView,
    source: Option<Id>,
}

fn memory_view_with_source(m: &Memory) -> MemoryViewAndSource {
    MemoryViewAndSource {
        view: MemoryView::from(m),
        source: m.source.map(|s| s.0),
    }
}

/// `GET /` — the chat-shaped live view. Wrapped through `no_cache` so
/// edits to `index.html` don't get masked by browser caching.
pub async fn index_html() -> Response {
    no_cache(Html(include_str!("index.html")).into_response())
}

pub async fn snapshot(State(state): State<Arc<AppState>>) -> Response {
    let entries = match state.log.read_from(None).await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("log read failed: {e}"),
            )
                .into_response();
        }
    };

    let mut events = Vec::with_capacity(entries.len());
    let mut memories: HashMap<Id, MemoryViewAndSource> = HashMap::new();
    let mut sources: HashMap<Id, SourceView> = HashMap::new();

    for entry in &entries {
        events.push(EventSummary {
            id: entry.id.to_string(),
            kind: event_kind(&entry.event),
            timestamp_ms: entry.id.timestamp_ms(),
            description: describe_event(&entry.event),
        });
        match &entry.event {
            Event::MemoryWritten(m) => {
                memories.insert(m.id, memory_view_with_source(m));
            }
            Event::MemoryEmbedded { id, .. } => {
                // The async worker filled in the embedding for a memory
                // whose original `MemoryWritten` arrived without one; reflect
                // that in the snapshot so the UI shows the dot lit.
                if let Some(v) = memories.get_mut(&id.0) {
                    v.view.has_embedding = true;
                }
            }
            Event::MemoryInvalidated { id, .. } => {
                if let Some(v) = memories.get_mut(&id.0) {
                    v.view.is_invalidated = true;
                }
            }
            Event::SourceIngested(s) => {
                sources.insert(
                    s.id,
                    SourceView {
                        id: s.id.to_string(),
                        title: s.title.clone(),
                        uri: s.uri.clone(),
                        chunk_count: s.chunk_count,
                    },
                );
            }
            Event::SourceInvalidated { id, .. } => {
                sources.remove(&id.0);
            }
            _ => {}
        }
    }

    // Second pass: now that all sources are known, denormalise source_title
    // onto each memory. Cheap at demo scale; later we'd hold the join open
    // via a materialised view.
    let memory_views: Vec<MemoryView> = memories
        .into_values()
        .map(|mut ms| {
            if let Some(sid) = ms.source {
                if let Some(s) = sources.get(&sid) {
                    ms.view.source_title = Some(s.title.clone());
                }
            }
            ms.view
        })
        .collect();

    let snap = Snapshot {
        log_head: entries.last().map(|e| e.id.to_string()),
        log_size: entries.len(),
        events,
        memories: memory_views,
        sources: sources.into_values().collect(),
    };
    no_cache(Json(snap).into_response())
}

// ---------- search ----------

#[derive(Deserialize)]
pub struct SearchParams {
    /// The natural-language query.
    pub q: String,
    /// Top-k requested from each signal and from the fused result.
    /// Defaults to 5 if absent.
    #[serde(default = "default_k")]
    pub k: usize,
    /// Optional requesting agent. When set, results are filtered through
    /// the inter-agent ACL (P1#7): the actor sees its own + system +
    /// granted memories only. Absent → no ACL filtering (legacy behavior).
    pub actor: Option<String>,
}

fn default_k() -> usize {
    5
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub k: usize,
    pub embedder_id: String,
    pub embedder_dim: usize,
    /// Weight given to the vector signal during RRF fusion. `0.0` means the
    /// signal is *shown* (for inspection) but does **not** contribute to the
    /// hybrid ranking — usually because the embedder is a non-semantic stub.
    pub vector_weight: f32,
    /// Weight given to the BM25 signal during RRF fusion.
    pub bm25_weight: f32,
    pub vector_hits: Vec<HitDto>,
    pub bm25_hits: Vec<HitDto>,
    pub hybrid_hits: Vec<HitDto>,
    /// Synthesized answer composed from the hybrid hits — citations point
    /// back at the memories they were drawn from. UIs render this above
    /// the raw signal columns; programmatic callers can ignore it.
    pub answer: AnswerDto,
    /// Per-query timing + retrieval telemetry. A copy is also pushed into
    /// the rolling `MetricsCollector` for `/api/metrics`.
    pub metrics: SearchMetrics,
}

#[derive(Serialize)]
pub struct HitDto {
    pub memory_id: String,
    pub score: f32,
    pub breakdown: Vec<(String, f32)>,
    /// Snippet of the memory's content (capped) so the UI doesn't have to
    /// cross-reference the snapshot to render a result card.
    pub content: String,
    pub tags: Vec<String>,
    pub is_invalidated: bool,
    /// Source-document metadata for chunked memories — lets the UI cluster
    /// chunks of the same source under their shared title.
    pub source_id: Option<String>,
    pub source_title: Option<String>,
    pub position: Option<u32>,
}

#[derive(Serialize)]
pub struct ExcerptDto {
    pub memory_id: String,
    pub segments: Vec<mnesio_core::ExcerptSegment>,
    pub retrieval_score: f32,
    pub source_id: Option<String>,
    pub source_title: Option<String>,
    pub position: Option<u32>,
}

#[derive(Serialize)]
pub struct AnswerDto {
    pub query: String,
    pub excerpts: Vec<ExcerptDto>,
    pub citations: Vec<String>,
    pub prose: Option<String>,
    pub provenance: mnesio_core::SynthesisProvenance,
}

pub async fn search(
    State(state): State<Arc<AppState>>,
    AxumQuery(params): AxumQuery<SearchParams>,
) -> Response {
    if params.q.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "query 'q' is required").into_response();
    }

    let scope = Scope::global(&state.default_tenant);
    let k = params.k.max(1);

    // Per-phase timings, recorded into `MetricsCollector` on exit. The
    // wall-clock `timestamp_ms` lets the history endpoint time-bucket
    // queries even when the monotonic `Instant` is relative.
    let query_started_ms = metrics::now_ms();
    let t_total_start = Instant::now();
    let mut phases = PhaseTimings::default();

    // Resolve memory_ids back to content + tags + invalidated flag by
    // replaying the log. Cheap at demo scale; later we'll cache.
    let t = Instant::now();
    let entries = match state.log.read_from(None).await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("log read failed: {e}"),
            )
                .into_response();
        }
    };
    phases.snapshot_replay_ms = t.elapsed().as_millis() as u64;
    // Per-memory side table: content + tags + invalidated + optional
    // source ref/position. Source titles are joined in via the sources map
    // below so the UI can cluster chunks of the same article.
    struct MemoryRow {
        content: String,
        tags: Vec<String>,
        invalidated: bool,
        source: Option<Id>,
        position: Option<u32>,
    }
    let mut memories: HashMap<Id, MemoryRow> = HashMap::new();
    let mut sources: HashMap<Id, String> = HashMap::new();
    for entry in &entries {
        match &entry.event {
            Event::MemoryWritten(m) => {
                memories.insert(
                    m.id,
                    MemoryRow {
                        content: m.content.clone(),
                        tags: m.tags.clone(),
                        invalidated: false,
                        source: m.source.map(|s| s.0),
                        position: m.position,
                    },
                );
            }
            Event::MemoryInvalidated { id, .. } => {
                if let Some(v) = memories.get_mut(&id.0) {
                    v.invalidated = true;
                }
            }
            Event::SourceIngested(s) => {
                sources.insert(s.id, s.title.clone());
            }
            Event::SourceInvalidated { id, .. } => {
                sources.remove(&id.0);
            }
            _ => {}
        }
    }
    let source_title_of =
        |sid: Option<Id>| -> Option<String> { sid.and_then(|id| sources.get(&id).cloned()) };

    let to_dto = |h: &Hit| -> HitDto {
        let row = memories.get(&h.memory.0);
        let (content, tags, invalidated, source, position) = match row {
            Some(r) => (
                r.content.clone(),
                r.tags.clone(),
                r.invalidated,
                r.source,
                r.position,
            ),
            None => ("<unknown memory>".into(), vec![], false, None, None),
        };
        HitDto {
            memory_id: h.memory.0.to_string(),
            score: h.score,
            breakdown: h.breakdown.clone(),
            content,
            tags,
            is_invalidated: invalidated,
            source_id: source.map(|s| s.to_string()),
            source_title: source_title_of(source),
            position,
        }
    };

    // Vector and BM25 lists — pulled directly so the UI can show them side
    // by side. The hybrid call below re-embeds internally; we accept that
    // tiny duplication for now to keep the data flow obvious.
    let t = Instant::now();
    let embedded = match state.embedder.embed(std::slice::from_ref(&params.q)).await {
        Ok(v) => v,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("embed: {e}")).into_response();
        }
    };
    phases.embed_query_ms = t.elapsed().as_millis() as u64;
    let q_vec = match embedded.first() {
        Some(v) => v.clone(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "embedder returned no vectors",
            )
                .into_response();
        }
    };
    let t = Instant::now();
    let vector_hits = match state.vector.search(&q_vec, k, &scope) {
        Ok(hs) => hs,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("vector: {e}")).into_response()
        }
    };
    phases.vector_search_ms = t.elapsed().as_millis() as u64;
    let t = Instant::now();
    let (bm25_hits, bm25_tier) = match state.bm25.search_with_diagnostics(&params.q, k, &scope) {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("bm25: {e}")).into_response(),
    };
    phases.bm25_search_ms = t.elapsed().as_millis() as u64;

    let t = Instant::now();
    let hybrid_hits = match state
        .retriever
        .search(&RetrievalQuery {
            text: params.q.clone(),
            scope,
            k,
            time_filter: None,
        })
        .await
    {
        Ok(hs) => hs,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("hybrid: {e}")).into_response()
        }
    };
    phases.hybrid_fuse_ms = t.elapsed().as_millis() as u64;
    // Snapshot the pre-boost top hit so the metrics can record whether the
    // boost reordered #1.
    let pre_boost_top = hybrid_hits.first().map(|h| h.memory);

    // Source-boost: a chunk that's part of a known document gets a small
    // multiplicative lift over an equivalently-scoring standalone memory.
    // BM25's length normalisation otherwise consistently down-weights long
    // article chunks even when they carry the same fact in richer context.
    // The boost is intentionally tiny — enough to break near-ties (~3-5 %
    // RRF gap) but not enough to drag a clearly-stronger standalone hit
    // down.
    let t = Instant::now();
    let mut source_boost_lifts = 0usize;
    let hybrid_hits = boost_sourced_hits(hybrid_hits, |mref| {
        let sourced = memories.get(&mref.0).and_then(|r| r.source).is_some();
        if sourced {
            source_boost_lifts += 1;
        }
        sourced
    });
    phases.source_boost_ms = t.elapsed().as_millis() as u64;
    let post_boost_top = hybrid_hits.first().map(|h| h.memory);
    let source_boost_changed_top = pre_boost_top != post_boost_top;

    // Inter-agent ACL filter (P1#7). Only applies when a requesting actor
    // is named — the agent sees its own + system + explicitly-granted
    // memories. Without `?actor=`, behaviour is unchanged.
    let hybrid_hits: Vec<Hit> = match &params.actor {
        Some(actor) => hybrid_hits
            .into_iter()
            .filter(|h| state.acl.can_read_memory(actor, h.memory))
            .collect(),
        None => hybrid_hits,
    };

    // Compose the answer card. The synthesizer trusts the (now boosted)
    // hybrid order; we just resolve each hit's content from the snapshot
    // map we already built above.
    let passages: Vec<Passage> = hybrid_hits
        .iter()
        .map(|h| {
            let row = memories.get(&h.memory.0);
            let (content, tags) = match row {
                Some(r) => (r.content.clone(), r.tags.clone()),
                None => ("<unknown memory>".into(), vec![]),
            };
            Passage {
                memory: h.memory,
                content,
                tags,
                retrieval_score: h.score,
            }
        })
        .collect();
    let t = Instant::now();
    let answer = match state.synthesizer.synthesize(&params.q, &passages).await {
        Ok(a) => a,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("synthesize: {e}"),
            )
                .into_response();
        }
    };
    phases.synthesize_ms = t.elapsed().as_millis() as u64;
    // Enrich each excerpt with source metadata so the UI can cluster
    // chunks of the same article together.
    let answer_dto = AnswerDto {
        query: answer.query,
        citations: answer.citations.iter().map(|c| c.0.to_string()).collect(),
        excerpts: answer
            .excerpts
            .into_iter()
            .map(|e| {
                let row = memories.get(&e.memory.0);
                let (source, position) = match row {
                    Some(r) => (r.source, r.position),
                    None => (None, None),
                };
                ExcerptDto {
                    memory_id: e.memory.0.to_string(),
                    segments: e.segments,
                    retrieval_score: e.retrieval_score,
                    source_id: source.map(|s| s.to_string()),
                    source_title: source_title_of(source),
                    position,
                }
            })
            .collect(),
        prose: answer.prose,
        provenance: answer.provenance,
    };

    let (vector_weight, bm25_weight) = state.retriever.weights();

    // Score envelopes for the metrics panel — min/max per signal across
    // whatever the user actually saw.
    let scores = compute_score_envelope(&vector_hits, &bm25_hits, &hybrid_hits);

    let metrics = SearchMetrics {
        timestamp_ms: query_started_ms,
        query_text: params.q.clone(),
        total_ms: t_total_start.elapsed().as_millis() as u64,
        phases,
        retrieval: RetrievalCounts {
            vector_hits: vector_hits.len(),
            bm25_hits: bm25_hits.len(),
            hybrid_hits: hybrid_hits.len(),
            bm25_tier: bm25_tier.as_str(),
            source_boost_changed_top,
            source_boost_lifts,
        },
        scores,
    };
    state.metrics.record(metrics.clone());

    let resp = SearchResponse {
        query: params.q,
        k,
        embedder_id: state.embedder.model_id().to_string(),
        embedder_dim: state.embedder.dim(),
        vector_weight,
        bm25_weight,
        vector_hits: vector_hits.iter().map(&to_dto).collect(),
        bm25_hits: bm25_hits.iter().map(&to_dto).collect(),
        hybrid_hits: hybrid_hits.iter().map(&to_dto).collect(),
        answer: answer_dto,
        metrics,
    };
    no_cache(Json(resp).into_response())
}

/// `GET /api/metrics` — current rolling aggregate over the last ~200
/// searches. The UI polls this every few seconds for the header strip.
pub async fn metrics_rollup(State(state): State<Arc<AppState>>) -> Response {
    let rollup: MetricsRollup = state.metrics.rollup();
    no_cache(Json(rollup).into_response())
}

/// `GET /api/metrics/history` — time-bucketed series + lifetime stats +
/// recent-queries log. The dashboard fetches this once on load and again
/// every few seconds.
pub async fn metrics_history(State(state): State<Arc<AppState>>) -> Response {
    let history: MetricsHistory = state.metrics.history();
    no_cache(Json(history).into_response())
}

/// `GET /api/evolve/metrics` — Phase-1 memory-evolution telemetry. The
/// payload mixes *event-log derived* totals (how many notes were
/// enriched, how many links recorded, how many bounded evolutions
/// committed) with the worker's in-memory per-ref scheduling state
/// (chain depth + last-evolved timestamps). The dashboard reads this
/// once on load and re-polls every few seconds.
pub async fn evolve_metrics(State(state): State<Arc<AppState>>) -> Response {
    let entries = match state.log.read_from(None).await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("log read failed: {e}"),
            )
                .into_response();
        }
    };

    let mut totals = EvolveTotals::default();
    // Most-recent evolution per memory chain, oldest → newest. The
    // dashboard pulls the last N off the back to render a timeline.
    let mut chains: Vec<EvolveChainEvent> = Vec::new();
    for entry in &entries {
        match &entry.event {
            Event::MemoryWritten(_) => totals.memories_written += 1,
            Event::MemoryNoteEnriched { .. } => totals.notes_enriched += 1,
            Event::MemoryLinksUpdated { links, .. } => {
                totals.links_updated += 1;
                totals.links_total += links.len();
            }
            Event::MemoryEvolved { from, to, diff } => {
                totals.evolutions_committed += 1;
                totals.tag_additions += diff.tags_added.len();
                totals.keyword_additions += diff.keywords_added.len();
                chains.push(EvolveChainEvent {
                    from: from.0.to_string(),
                    to: to.0.to_string(),
                    timestamp_ms: entry.id.timestamp_ms(),
                    tags_added: diff.tags_added.clone(),
                    keywords_added: diff.keywords_added.clone(),
                });
            }
            Event::MemoryInvalidated { reason, .. } if reason.contains("evolution") => {
                totals.invalidated_by_evolution += 1;
            }
            _ => {}
        }
    }

    // Worker-side state — chain depths + last-evolved timestamps. Only
    // populated when the worker is actually running.
    let (worker_state, enabled) = match &state.evolution {
        Some(w) => {
            let raw = w.snapshot_state().await;
            let mut out: Vec<EvolveMemoryState> = raw
                .into_iter()
                .map(|(m, count, ts)| EvolveMemoryState {
                    memory_id: m.0.to_string(),
                    chain_depth: count,
                    last_evolved_at_ms: ts,
                })
                .collect();
            // Newest first — UI usually wants the most recent evolution
            // activity at the top of the table.
            out.sort_by_key(|s| std::cmp::Reverse(s.last_evolved_at_ms));
            (out, true)
        }
        None => (Vec::new(), false),
    };

    let max_chain_depth = worker_state
        .iter()
        .map(|s| s.chain_depth)
        .max()
        .unwrap_or(0);
    let unique_memories_evolved = worker_state.len();

    let payload = EvolveMetricsResponse {
        enabled,
        backend: llm_backend_label(),
        totals,
        max_chain_depth,
        unique_memories_evolved,
        recent_chain_events: chains.into_iter().rev().take(20).collect(),
        worker_state,
    };
    no_cache(Json(payload).into_response())
}

/// `GET /api/profile` — current profile attributes for the default
/// subject, plus the bi-temporal history of each. Personalization
/// surface: this is the stable persona an agent conditions on.
pub async fn profile(State(state): State<Arc<AppState>>) -> Response {
    // The demo subject is tenant=demo, user=alice. Production derives
    // this from auth context.
    let scope = Scope {
        tenant: state.default_tenant.clone(),
        user: Some("alice".into()),
        session: None,
    };
    let current: Vec<ProfileAttrDto> = state
        .profile
        .all(&scope)
        .into_iter()
        .map(|(attribute, value)| {
            let history: Vec<ProfileHistoryDto> = state
                .profile
                .history(&scope, &attribute)
                .into_iter()
                .map(|v| ProfileHistoryDto {
                    value: v.value,
                    set_at_ms: v.set_at_ms,
                    actor: v.actor,
                })
                .collect();
            ProfileAttrDto {
                attribute,
                value,
                history,
            }
        })
        .collect();
    let payload = ProfileResponse {
        tenant: scope.tenant,
        user: scope.user,
        attributes: current,
    };
    no_cache(Json(payload).into_response())
}

#[derive(Serialize)]
pub struct ProfileResponse {
    pub tenant: String,
    pub user: Option<String>,
    pub attributes: Vec<ProfileAttrDto>,
}

#[derive(Serialize)]
pub struct ProfileAttrDto {
    pub attribute: String,
    pub value: String,
    /// Oldest → newest, including the current value as the last element.
    pub history: Vec<ProfileHistoryDto>,
}

#[derive(Serialize)]
pub struct ProfileHistoryDto {
    pub value: String,
    pub set_at_ms: u64,
    pub actor: Option<String>,
}

/// `GET /api/agents` — multi-agent attribution + the inter-agent grant
/// matrix for the default tenant. Shows which agent owns how many
/// memories and who it has granted read access to (P1#7).
pub async fn agents(State(state): State<Arc<AppState>>) -> Response {
    let rows: Vec<AgentDto> = state
        .acl
        .attribution(&state.default_tenant)
        .into_iter()
        .map(|a| AgentDto {
            agent: a.agent,
            memories: a.memories,
            grants_to: a.grants_to,
            is_system: a.is_system,
        })
        .collect();
    no_cache(Json(AgentsResponse { agents: rows }).into_response())
}

#[derive(Serialize)]
pub struct AgentsResponse {
    pub agents: Vec<AgentDto>,
}

#[derive(Serialize)]
pub struct AgentDto {
    pub agent: String,
    pub memories: u64,
    /// Agents this agent has granted read access to.
    pub grants_to: Vec<String>,
    /// `true` for system owners (ingestion / demo / evolution-worker),
    /// whose memories are readable by every agent.
    pub is_system: bool,
}

/// `GET /api/skills` — Phase-9 (P2#9) **skill retrieval for injection**.
///
/// The procedural compiler already commits versioned `PolicyArtifact`s
/// behind the safety gate (Hard Rule #1); this endpoint surfaces the
/// **active** versions an agent should fold into its prompt at query
/// time. That's the wedge made operational: every call uses the
/// gated-improved skill, never an ungated rewrite — cross-task skill
/// reuse with the regression guard competitors lack.
///
/// Scope-filtered exactly (Hard Rule #3) via the default tenant.
pub async fn skills(State(state): State<Arc<AppState>>) -> Response {
    let scope = Scope::global(state.default_tenant.clone());
    let artifacts = state.procedural_store.all_in_scope(&scope).await;
    let injectable: Vec<InjectableSkillDto> = artifacts.iter().map(skill_dto).collect();
    let payload = SkillsResponse {
        tenant: scope.tenant,
        count: injectable.len(),
        skills: injectable,
    };
    no_cache(Json(payload).into_response())
}

#[derive(Serialize)]
pub struct SkillsResponse {
    pub tenant: String,
    pub count: usize,
    pub skills: Vec<InjectableSkillDto>,
}

/// One artifact reduced to what a thin client needs to *inject* into a
/// prompt — the bytes it should paste in, plus enough provenance (id,
/// version, kind) to log which skill drove the call.
#[derive(Serialize)]
pub struct InjectableSkillDto {
    pub artifact_id: String,
    pub version: u32,
    pub kind: &'static str,
    /// Plain-text content the client should prepend / append to its
    /// prompt. The shape depends on `kind`:
    /// - SystemPrompt → the body verbatim;
    /// - Heuristic → `"If <when>, then <then>"`;
    /// - Skill → `"<signature>\n<body>"` (lang surfaced separately);
    /// - RetrievalRule → `"<query_pattern> → <rewrite>"`;
    /// - Reflection → the bare `lesson`.
    pub injection: String,
    /// `Some(lang)` only for `Skill` artifacts (e.g. `"python"`); the
    /// client may need it to fence the body correctly.
    pub lang: Option<String>,
    pub canary_count: usize,
}

fn skill_dto(a: &mnesio_core::entity::PolicyArtifact) -> InjectableSkillDto {
    let (kind, injection, lang) = match &a.kind {
        mnesio_core::entity::ArtifactKind::SystemPrompt { body } => {
            ("SystemPrompt", body.clone(), None)
        }
        mnesio_core::entity::ArtifactKind::Heuristic { when, then } => {
            ("Heuristic", format!("If {when}, then {then}"), None)
        }
        mnesio_core::entity::ArtifactKind::Skill {
            signature,
            body,
            lang,
            ..
        } => ("Skill", format!("{signature}\n{body}"), Some(lang.clone())),
        mnesio_core::entity::ArtifactKind::RetrievalRule {
            query_pattern,
            rewrite,
        } => (
            "RetrievalRule",
            format!("{query_pattern} → {rewrite}"),
            None,
        ),
        mnesio_core::entity::ArtifactKind::Reflection { lesson, .. } => {
            ("Reflection", lesson.clone(), None)
        }
    };
    InjectableSkillDto {
        artifact_id: a.id.to_string(),
        version: a.version,
        kind,
        injection,
        lang,
        canary_count: a.canaries.len(),
    }
}

/// `GET /api/ingest/metrics` — Phase-7 ingestion telemetry (extract +
/// consolidate). Reports `enabled=false` when the worker isn't running.
pub async fn ingest_metrics(State(state): State<Arc<AppState>>) -> Response {
    let payload = match &state.ingestion {
        Some(w) => {
            let m = w.metrics().await;
            IngestMetricsResponse {
                enabled: true,
                backend: llm_backend_label(),
                observations: m.observations,
                facts_extracted: m.facts_extracted,
                adds_committed: m.adds_committed,
                adds_rejected: m.adds_rejected,
                updates: m.updates,
                contradictions: m.contradictions,
                refinements: m.refinements,
                noops: m.noops,
                pii_redacted: m.pii_redacted,
            }
        }
        None => IngestMetricsResponse {
            enabled: false,
            backend: llm_backend_label(),
            ..Default::default()
        },
    };
    no_cache(Json(payload).into_response())
}

#[derive(Serialize, Default)]
pub struct IngestMetricsResponse {
    pub enabled: bool,
    pub backend: &'static str,
    /// Raw observations processed.
    pub observations: u64,
    /// Atomic facts extracted across all observations.
    pub facts_extracted: u64,
    /// Facts committed as new memories (ADD, post-admission).
    pub adds_committed: u64,
    /// ADDs culled by the importance-admission floor.
    pub adds_rejected: u64,
    /// Facts that superseded an existing memory (UPDATE).
    pub updates: u64,
    /// UPDATEs that were factual contradictions.
    pub contradictions: u64,
    /// UPDATEs that were refinements.
    pub refinements: u64,
    /// Facts dropped as duplicates (NOOP).
    pub noops: u64,
    /// PII spans redacted from observation text before storage (P1#8).
    pub pii_redacted: u64,
}

#[derive(Serialize, Default)]
pub struct EvolveTotals {
    /// Distinct `MemoryWritten` events, including evolution children.
    /// The dashboard subtracts `invalidated_by_evolution` to plot
    /// "live memories".
    pub memories_written: usize,
    /// `MemoryNoteEnriched` events — A-MEM step 1.
    pub notes_enriched: usize,
    /// `MemoryLinksUpdated` events — A-MEM step 2.
    pub links_updated: usize,
    /// Sum of `links.len()` across every `MemoryLinksUpdated`.
    pub links_total: usize,
    /// `MemoryEvolved` events — A-MEM step 3 (bounded evolution).
    pub evolutions_committed: usize,
    /// `MemoryInvalidated` whose `reason` mentions "evolution".
    pub invalidated_by_evolution: usize,
    /// Cumulative tag additions across every evolution.
    pub tag_additions: usize,
    /// Cumulative keyword additions across every evolution.
    pub keyword_additions: usize,
}

#[derive(Serialize)]
pub struct EvolveChainEvent {
    pub from: String,
    pub to: String,
    pub timestamp_ms: u64,
    pub tags_added: Vec<String>,
    pub keywords_added: Vec<String>,
}

#[derive(Serialize)]
pub struct EvolveMemoryState {
    pub memory_id: String,
    pub chain_depth: u16,
    pub last_evolved_at_ms: u64,
}

#[derive(Serialize)]
pub struct EvolveMetricsResponse {
    pub enabled: bool,
    /// Human-friendly LLM backend label — `"ollama"` or `"demo"`.
    pub backend: &'static str,
    pub totals: EvolveTotals,
    /// Maximum `Memory.evolution_count` seen across the worker state —
    /// the deepest chain so far, useful as a "are we approaching the
    /// lifetime cap?" indicator.
    pub max_chain_depth: u16,
    /// Distinct memories that have been evolved at least once.
    pub unique_memories_evolved: usize,
    /// Newest 20 evolution events (most recent first), for the chain
    /// timeline panel.
    pub recent_chain_events: Vec<EvolveChainEvent>,
    /// Per-memory chain-depth + cooldown state, newest first.
    pub worker_state: Vec<EvolveMemoryState>,
}

/// Compute the LLM-backend label the same way main.rs does. Kept here
/// instead of imported so viz.rs doesn't gain a dependency on main.rs
/// internals — both functions read the same env var.
fn llm_backend_label() -> &'static str {
    let choice = std::env::var("MNESIO_EVOLVE_LLM").unwrap_or_default();
    match choice.to_ascii_lowercase().as_str() {
        "ollama" => "ollama",
        _ => "demo",
    }
}

/// `GET /api/procedural/metrics` — Phase-2 procedural compiler
/// telemetry. Mirrors the shape of `/api/evolve/metrics`: log-derived
/// totals + a per-commit chain timeline + active-artifact snapshot.
///
/// When the worker is disabled (`MNESIO_PROCEDURAL` not set), this
/// still returns 200 with `enabled=false` and best-effort totals
/// derived from any historical procedural events in the log — useful
/// for replaying old data without booting the compiler.
pub async fn procedural_metrics(State(state): State<Arc<AppState>>) -> Response {
    let entries = match state.log.read_from(None).await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("log read failed: {e}"),
            )
                .into_response();
        }
    };

    let mut totals = ProceduralTotals::default();
    // Per-artifact accumulators for the dashboard's "active artifacts"
    // table — we walk proposals + commits to compute version + commit
    // count per artifact id.
    let mut chains: Vec<ProceduralChainEvent> = Vec::new();
    let mut rejection_counts: HashMap<String, usize> = HashMap::new();
    // Track the running commit count + objective delta running mean.
    let mut delta_sum = 0.0f64;
    let mut delta_count: usize = 0;

    // Replay through a temporary store so the dashboard sees the same
    // active-version view the worker does (without locking the live one).
    // ProceduralStore's absorb walks Proposed → cache; Committed → flip
    // active. We mirror enough of that here to compute totals.
    let mut proposal_to_first_artifact: HashMap<mnesio_core::types::ProposalId, String> =
        HashMap::new();
    for entry in &entries {
        match &entry.event {
            Event::OutcomeRecorded(_) => totals.outcomes_recorded += 1,
            Event::ProceduralProposed {
                proposal,
                artifacts,
            } => {
                totals.proposals += 1;
                if let Some(a) = artifacts.first() {
                    proposal_to_first_artifact.insert(*proposal, a.id.to_string());
                }
            }
            Event::ProceduralCommitted { proposal, report } => {
                totals.commits += 1;
                delta_sum += report.objective_delta as f64;
                delta_count += 1;
                let aref = proposal_to_first_artifact
                    .get(proposal)
                    .cloned()
                    .unwrap_or_else(|| "<unknown>".into());
                chains.push(ProceduralChainEvent {
                    proposal_id: proposal.0.to_string(),
                    artifact_id: aref,
                    timestamp_ms: entry.id.timestamp_ms(),
                    kind: "commit".into(),
                    objective_delta: report.objective_delta,
                    canaries_passed: report.canaries_passed,
                    canaries_total: report.canaries_total,
                    judges_consulted: report.judges_consulted,
                    reason: None,
                });
            }
            Event::ProceduralRejected { proposal, reason } => {
                totals.rejections += 1;
                // Tally the human-readable rejection reasons. The
                // worker writes them as a structured string like
                // `c0=[baseline,canaries] c1=[judges]`; we just
                // re-tally the bracketed tokens for the chart.
                for token in parse_rejection_tokens(reason) {
                    *rejection_counts.entry(token).or_insert(0) += 1;
                }
                let aref = proposal_to_first_artifact
                    .get(proposal)
                    .cloned()
                    .unwrap_or_else(|| "<unknown>".into());
                chains.push(ProceduralChainEvent {
                    proposal_id: proposal.0.to_string(),
                    artifact_id: aref,
                    timestamp_ms: entry.id.timestamp_ms(),
                    kind: "reject".into(),
                    objective_delta: 0.0,
                    canaries_passed: 0,
                    canaries_total: 0,
                    judges_consulted: 0,
                    reason: Some(reason.clone()),
                });
            }
            _ => {}
        }
    }

    let win_rate_pct = if totals.proposals == 0 {
        0.0
    } else {
        (totals.commits as f32 / totals.proposals as f32) * 100.0
    };
    let mean_objective_delta = if delta_count == 0 {
        0.0
    } else {
        (delta_sum / delta_count as f64) as f32
    };

    // Active artifacts table.
    let active = state.procedural_store.all().await;
    let active_artifacts: Vec<ProceduralActiveArtifact> = active
        .iter()
        .map(|a| ProceduralActiveArtifact {
            artifact_id: a.id.to_string(),
            version: a.version,
            kind: artifact_kind_label(&a.kind),
            scope_tenant: a.scope.tenant.clone(),
            canary_count: a.canaries.len() as u32,
        })
        .collect();

    let enabled = state.procedural.is_some();

    let mut top_rejection_reasons: Vec<(String, usize)> = rejection_counts.into_iter().collect();
    top_rejection_reasons.sort_by_key(|r| std::cmp::Reverse(r.1));
    top_rejection_reasons.truncate(8);

    // Pull the learning curve from the worker, if available + bound.
    let (curve_points, safety_clean) = match state.procedural.as_ref().and_then(|w| w.curve()) {
        Some(coll) => {
            let pts = coll.points().await;
            let clean = coll.safety_clean().await;
            let curve_dto: Vec<ProceduralCurvePoint> = pts
                .into_iter()
                .map(|p| ProceduralCurvePoint {
                    artifact_id: p.artifact_id.0.to_string(),
                    version: p.version,
                    timestamp_ms: p.timestamp_ms,
                    benchmark_score: p.benchmark_score,
                    safety_probe_pass_rate: p.safety_probe_pass_rate,
                    objective_delta: p.objective_delta,
                    judges_consulted: p.judges_consulted,
                })
                .collect();
            (curve_dto, clean)
        }
        // Worker not running or no eval binding — empty curve, default
        // to safety_clean=true (no data, no regression).
        None => (Vec::new(), true),
    };

    let payload = ProceduralMetricsResponse {
        enabled,
        backend: llm_backend_label(),
        totals,
        win_rate_pct,
        mean_objective_delta,
        top_rejection_reasons,
        active_artifacts,
        recent_chain_events: chains.into_iter().rev().take(20).collect(),
        curve_points,
        safety_clean,
    };
    no_cache(Json(payload).into_response())
}

#[derive(Serialize, Default)]
pub struct ProceduralTotals {
    pub outcomes_recorded: usize,
    pub proposals: usize,
    pub commits: usize,
    pub rejections: usize,
}

#[derive(Serialize)]
pub struct ProceduralActiveArtifact {
    pub artifact_id: String,
    pub version: u32,
    pub kind: &'static str,
    pub scope_tenant: String,
    pub canary_count: u32,
}

#[derive(Serialize)]
pub struct ProceduralChainEvent {
    pub proposal_id: String,
    pub artifact_id: String,
    pub timestamp_ms: u64,
    pub kind: String, // "commit" | "reject"
    pub objective_delta: f32,
    pub canaries_passed: u32,
    pub canaries_total: u32,
    pub judges_consulted: u8,
    pub reason: Option<String>,
}

#[derive(Serialize)]
pub struct ProceduralMetricsResponse {
    pub enabled: bool,
    pub backend: &'static str,
    pub totals: ProceduralTotals,
    pub win_rate_pct: f32,
    pub mean_objective_delta: f32,
    /// `[(reason_token, count)]` newest-first up to 8 entries —
    /// powers the dashboard's "top rejection reasons" bar chart.
    pub top_rejection_reasons: Vec<(String, usize)>,
    pub active_artifacts: Vec<ProceduralActiveArtifact>,
    pub recent_chain_events: Vec<ProceduralChainEvent>,
    /// Phase-2 "done when" series — one point per committed version,
    /// carrying the absolute benchmark score + safety probe pass rate.
    /// Empty when the worker isn't running with an `EvalBinding`.
    pub curve_points: Vec<ProceduralCurvePoint>,
    /// True iff every recorded curve point has `safety_probe_pass_rate
    /// == 1.0`. The alignment-drift hard-stop signal — monitoring
    /// tools poll this.
    pub safety_clean: bool,
}

/// JSON-shaped curve point. Mirrors `mnesio_procedural::LearningCurvePoint`
/// but with string ids for the wire.
#[derive(Serialize)]
pub struct ProceduralCurvePoint {
    pub artifact_id: String,
    pub version: u32,
    pub timestamp_ms: u64,
    pub benchmark_score: f32,
    pub safety_probe_pass_rate: f32,
    pub objective_delta: f32,
    pub judges_consulted: u8,
}

/// Parse the worker's rejection-reason string into individual reason
/// tokens. The worker writes `c0=[baseline,canaries] c1=[judges]` —
/// we collect every bracketed comma-separated piece.
fn parse_rejection_tokens(reason: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_bracket = false;
    for c in reason.chars() {
        if c == '[' {
            in_bracket = true;
            current.clear();
        } else if c == ']' {
            in_bracket = false;
            for tok in current.split(',') {
                let t = tok.trim();
                if !t.is_empty() {
                    out.push(t.to_string());
                }
            }
            current.clear();
        } else if in_bracket {
            current.push(c);
        }
    }
    out
}

fn artifact_kind_label(k: &mnesio_core::entity::ArtifactKind) -> &'static str {
    match k {
        mnesio_core::entity::ArtifactKind::SystemPrompt { .. } => "SystemPrompt",
        mnesio_core::entity::ArtifactKind::Heuristic { .. } => "Heuristic",
        mnesio_core::entity::ArtifactKind::Skill { .. } => "Skill",
        mnesio_core::entity::ArtifactKind::RetrievalRule { .. } => "RetrievalRule",
        mnesio_core::entity::ArtifactKind::Reflection { .. } => "Reflection",
    }
}

// ---------- graph (Phase 4) ----------

/// Hard cap on neighborhood radius. Keeps the response bounded
/// regardless of what a caller asks for (Hard Rule #6 — bound the
/// cascades, here applied to read-side fan-out).
const MAX_GRAPH_DEPTH: u16 = 4;

/// Number of candidate root nodes returned for the picker.
const GRAPH_ROOT_LIMIT: usize = 12;

fn default_graph_depth() -> u16 {
    2
}

#[derive(Deserialize)]
pub struct GraphParams {
    /// Focus memory id (ULID string). When omitted the response carries
    /// only `stats` + `roots`, so the UI can populate a picker and pick
    /// a sensible default focus.
    pub id: Option<String>,
    #[serde(default = "default_graph_depth")]
    pub depth: u16,
    /// Optional bi-temporal cut, unix-epoch milliseconds. When set, the
    /// neighborhood is returned *as it was* at that instant — lineage and
    /// links resolve to the generation live at `as_of_ms`.
    pub as_of_ms: Option<i64>,
}

/// `GET /api/graph` — bi-temporal property-graph exploration.
///
/// Two response modes in one shape:
/// - **No `id`**: `stats` + `roots` (most-connected nodes). The UI
///   auto-selects `roots[0]` as the initial focus.
/// - **With `id`**: additionally returns the `depth`-bounded
///   neighborhood (`nodes` + `edges`) around that memory. Traversal is
///   undirected (follows both `Linked`/lineage out-edges and in-edges)
///   so the focus node shows both what it cites and what cites it.
///
/// Everything is filtered through the server's default-tenant `Scope`
/// (Hard Rule #3) and, when `as_of_ms` is set, through bi-temporal
/// liveness.
pub async fn graph(
    State(state): State<Arc<AppState>>,
    AxumQuery(params): AxumQuery<GraphParams>,
) -> Response {
    let scope = Scope::global(state.default_tenant.clone());
    let as_of = match params.as_of_ms {
        Some(ms) => match OffsetDateTime::from_unix_timestamp_nanos((ms as i128) * 1_000_000) {
            Ok(t) => Some(t),
            Err(_) => {
                return (StatusCode::BAD_REQUEST, "as_of_ms out of range").into_response();
            }
        },
        None => None,
    };
    let depth = params.depth.min(MAX_GRAPH_DEPTH);

    let stats = match state.graph.stats() {
        Ok(s) => GraphStatsDto {
            node_count: s.node_count,
            edge_count: s.edge_count,
            live_edge_count: s.live_edge_count,
        },
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("graph stats failed: {e}"),
            )
                .into_response();
        }
    };

    let roots = match state.graph.roots(&scope, as_of, GRAPH_ROOT_LIMIT) {
        Ok(rs) => rs
            .iter()
            .map(|nd| node_dto(&nd.node, None, Some((nd.out_degree, nd.in_degree))))
            .collect::<Vec<_>>(),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("graph roots failed: {e}"),
            )
                .into_response();
        }
    };

    let mut focus: Option<String> = None;
    let mut nodes: Vec<GraphNodeDto> = Vec::new();
    let mut edges: Vec<GraphEdgeDto> = Vec::new();

    if let Some(id_str) = &params.id {
        let id = match id_str.parse::<Id>() {
            Ok(id) => id,
            Err(_) => {
                return (StatusCode::BAD_REQUEST, "id is not a valid ULID").into_response();
            }
        };
        let mref = MemoryRef(id);
        match build_neighborhood(state.graph.as_ref(), mref, depth, &scope, as_of) {
            Ok(Some((ns, es))) => {
                focus = Some(id_str.clone());
                nodes = ns;
                edges = es;
            }
            // Focus node not visible (unknown / out of scope / not live
            // at `as_of`) — return stats+roots with an empty neighborhood
            // rather than a 404, so the UI can fall back to a root.
            Ok(None) => {}
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("graph neighborhood failed: {e}"),
                )
                    .into_response();
            }
        }
    }

    let payload = GraphResponse {
        stats,
        roots,
        focus,
        depth,
        as_of_ms: params.as_of_ms,
        nodes,
        edges,
    };
    no_cache(Json(payload).into_response())
}

/// Undirected, depth-bounded neighborhood around `focus`. Returns
/// `None` when the focus node itself isn't visible to the caller.
/// `Some((nodes, edges))` otherwise, with `edges` restricted to the
/// returned node set so the rendered graph is closed.
#[allow(clippy::type_complexity)]
fn build_neighborhood(
    graph: &FjallGraphView,
    focus: MemoryRef,
    depth: u16,
    scope: &Scope,
    as_of: Option<OffsetDateTime>,
) -> Result<Option<(Vec<GraphNodeDto>, Vec<GraphEdgeDto>)>, mnesio_core::MnesioError> {
    let Some(focus_node) = graph.node(focus, scope, as_of)? else {
        return Ok(None);
    };

    // BFS over the *undirected* graph (out-edges ∪ in-edges) so the
    // focus shows both what it points at and what points at it.
    let mut hop_of: HashMap<Id, u16> = HashMap::new();
    hop_of.insert(focus.0, 0);
    let mut order: Vec<(MemoryRef, u16)> = vec![(focus, 0)];
    let mut frontier: VecDeque<(MemoryRef, u16)> = VecDeque::from([(focus, 0)]);
    let mut node_records: HashMap<Id, mnesio_graph::NodeRecord> = HashMap::new();
    node_records.insert(focus.0, focus_node);

    while let Some((node, d)) = frontier.pop_front() {
        if d >= depth {
            continue;
        }
        let out = graph.out_neighbors(node, None, scope, as_of)?;
        let inc = graph.in_neighbors(node, None, scope, as_of)?;
        let neighbors = out.iter().map(|e| e.dst).chain(inc.iter().map(|e| e.src));
        for nref in neighbors {
            if hop_of.contains_key(&nref.0) {
                continue;
            }
            if let Some(n) = graph.node(nref, scope, as_of)? {
                hop_of.insert(nref.0, d + 1);
                node_records.insert(nref.0, n);
                order.push((nref, d + 1));
                frontier.push_back((nref, d + 1));
            }
        }
    }

    let node_set: HashSet<Id> = hop_of.keys().copied().collect();
    let nodes: Vec<GraphNodeDto> = order
        .iter()
        .filter_map(|(r, hop)| {
            node_records
                .get(&r.0)
                .map(|n| node_dto(n, Some(*hop), None))
        })
        .collect();

    // Collect edges as out-edges of every node in the set whose dst is
    // also in the set. Live edges are unique per (src, relation, dst),
    // so no dedup is needed.
    let mut edges: Vec<GraphEdgeDto> = Vec::new();
    for (r, _) in &order {
        for e in graph.out_neighbors(*r, None, scope, as_of)? {
            if node_set.contains(&e.dst.0) {
                edges.push(GraphEdgeDto {
                    from: e.src.0.to_string(),
                    to: e.dst.0.to_string(),
                    relation: relation_label(e.relation),
                    weight: e.weight,
                });
            }
        }
    }
    Ok(Some((nodes, edges)))
}

/// Build a node DTO. `hop` is set for neighborhood nodes; `degree` is
/// set for root candidates. A node is never both at once.
fn node_dto(
    n: &mnesio_graph::NodeRecord,
    hop: Option<u16>,
    degree: Option<(usize, usize)>,
) -> GraphNodeDto {
    // Source nodes carry an explicit `label` (the document title).
    // Memory nodes store no content (the log is the system of record),
    // so their label is derived from tags → keywords → a short id
    // suffix.
    let label = if let Some(l) = &n.label {
        l.clone()
    } else if !n.tags.is_empty() {
        n.tags.join(", ")
    } else if !n.keywords.is_empty() {
        n.keywords.join(", ")
    } else {
        let s = n.id.0.to_string();
        format!("…{}", &s[s.len().saturating_sub(6)..])
    };
    GraphNodeDto {
        id: n.id.0.to_string(),
        label,
        tags: n.tags.clone(),
        evolution_count: n.evolution_count,
        invalidated: n.tx_to.is_some(),
        is_chunk: n.source.is_some(),
        is_source: n.is_source,
        hop,
        out_degree: degree.map(|(o, _)| o),
        in_degree: degree.map(|(_, i)| i),
    }
}

/// Stable wire labels for the dashboard's graph edges.
///
/// Kept exhaustive on purpose: adding a `Relation` should fail the build here
/// rather than silently render as an unlabelled edge.
fn relation_label(r: Relation) -> &'static str {
    match r {
        Relation::Linked => "linked",
        Relation::EvolvedFrom => "evolved_from",
        Relation::EvolvedTo => "evolved_to",
        Relation::ContainedIn => "contained_in",
        // Phase 17 code edges.
        Relation::Calls => "calls",
        Relation::Imports => "imports",
        Relation::Implements => "implements",
        Relation::References => "references",
        Relation::TestOf => "test_of",
    }
}

#[derive(Serialize)]
pub struct GraphResponse {
    pub stats: GraphStatsDto,
    /// Most-connected nodes, for the focus picker. Always present.
    pub roots: Vec<GraphNodeDto>,
    /// The queried focus id, echoed back. `None` when no `id` was given.
    pub focus: Option<String>,
    /// Effective (capped) neighborhood radius.
    pub depth: u16,
    /// Bi-temporal cut echoed back, if any.
    pub as_of_ms: Option<i64>,
    /// Neighborhood nodes incl. focus. Empty when no focus.
    pub nodes: Vec<GraphNodeDto>,
    /// Edges among the neighborhood node set.
    pub edges: Vec<GraphEdgeDto>,
}

#[derive(Serialize)]
pub struct GraphStatsDto {
    pub node_count: usize,
    pub edge_count: usize,
    pub live_edge_count: usize,
}

#[derive(Serialize)]
pub struct GraphNodeDto {
    pub id: String,
    /// Display label derived from tags/keywords (graph stores no content).
    pub label: String,
    pub tags: Vec<String>,
    pub evolution_count: u16,
    /// `true` when the node has been tombstoned (`tx_to` set).
    pub invalidated: bool,
    /// `true` when the memory is a chunk of a `Source` document.
    pub is_chunk: bool,
    /// `true` when the node *is* a `Source` document (not a memory).
    pub is_source: bool,
    /// BFS distance from the focus. `None` for root candidates.
    pub hop: Option<u16>,
    /// Live out-degree. `Some` only for root candidates.
    pub out_degree: Option<usize>,
    /// Live in-degree. `Some` only for root candidates.
    pub in_degree: Option<usize>,
}

#[derive(Serialize)]
pub struct GraphEdgeDto {
    pub from: String,
    pub to: String,
    /// `"linked" | "evolved_from" | "evolved_to" | "contained_in"`.
    pub relation: &'static str,
    pub weight: Option<f32>,
}

/// `GET /dashboard` — the long-view analytics page (Chart.js + history).
///
/// Wrapped through `no_cache` so iterative dashboard edits during
/// development don't get masked by an aggressive browser cache. The
/// HTML is `include_str!`'d so the cost is just an extra header — no
/// disk read per request.
pub async fn dashboard_html() -> Response {
    no_cache(Html(include_str!("dashboard.html")).into_response())
}

/// `GET /static/chart.umd.min.js` — vendored Chart.js bundle. Served
/// locally so the dashboard doesn't depend on any third-party CDN at
/// runtime: a flaky network or a CDN going dark would otherwise break
/// every chart on the page. Cached aggressively (the file is baked
/// into the binary, so it only changes when we rebuild).
pub async fn chart_js() -> Response {
    let body = include_str!("chart.umd.min.js");
    let mut response = body.into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/javascript; charset=utf-8"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("public, max-age=31536000, immutable"),
    );
    response
}

/// Walk each signal's hits to pick out the min/max score for the metrics
/// score envelope. The hybrid hit carries a `breakdown` with `("vector", ...)`,
/// `("bm25", ...)`, `("rrf", ...)` entries we use for the rrf min/max.
fn compute_score_envelope(
    vector_hits: &[Hit],
    bm25_hits: &[Hit],
    hybrid_hits: &[Hit],
) -> ScoreEnvelope {
    let (vector_min, vector_max) = min_max_score(vector_hits);
    let (bm25_min, bm25_max) = min_max_score(bm25_hits);
    let (rrf_min, rrf_max) = if hybrid_hits.is_empty() {
        (0.0, 0.0)
    } else {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for h in hybrid_hits {
            if let Some(rrf) = h
                .breakdown
                .iter()
                .find(|(name, _)| name == "rrf")
                .map(|(_, v)| *v)
            {
                if rrf < lo {
                    lo = rrf;
                }
                if rrf > hi {
                    hi = rrf;
                }
            }
        }
        if lo == f32::INFINITY {
            (0.0, 0.0)
        } else {
            (lo, hi)
        }
    };
    ScoreEnvelope {
        bm25_max,
        bm25_min,
        vector_max,
        vector_min,
        rrf_max,
        rrf_min,
    }
}

fn min_max_score(hits: &[Hit]) -> (f32, f32) {
    if hits.is_empty() {
        return (0.0, 0.0);
    }
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for h in hits {
        if h.score < lo {
            lo = h.score;
        }
        if h.score > hi {
            hi = h.score;
        }
    }
    (lo, hi)
}

// ---------- helpers ----------

/// Multiplicative bonus applied to a hit whose memory belongs to a `Source`.
/// 5 % is large enough to flip near-ties (where BM25 length normalisation
/// makes a long article chunk lose to a short standalone by a hair) without
/// disturbing rankings where one hit is clearly more relevant than another.
const SOURCE_BOOST: f32 = 0.05;

/// Re-rank a hit list so chunks of a [`mnesio_core::Source`] beat tied
/// standalone hits. `is_sourced` looks up whether a given memory is part
/// of a source — the handler closes over its in-memory join table.
fn boost_sourced_hits<F>(hits: Vec<Hit>, mut is_sourced: F) -> Vec<Hit>
where
    F: FnMut(MemoryRef) -> bool,
{
    let mut out: Vec<Hit> = hits
        .into_iter()
        .map(|mut h| {
            if is_sourced(h.memory) {
                h.score *= 1.0 + SOURCE_BOOST;
            }
            h
        })
        .collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn no_cache(mut response: Response) -> Response {
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-store"),
    );
    response
}

// The `event_kind` / `describe_event` helpers live below the test module
// for readability — they're long and the tests are the more interesting
// thing to put near the public handlers. Allow the lint accordingly.
#[allow(clippy::items_after_test_module)]
#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::types::new_id;

    fn hit(memory: MemoryRef, score: f32) -> Hit {
        Hit {
            memory,
            score,
            breakdown: vec![],
        }
    }

    #[test]
    fn source_boost_flips_near_ties_in_favor_of_chunks() {
        let standalone = MemoryRef(new_id());
        let article_chunk = MemoryRef(new_id());
        // The "germany news" case from the live demo: standalone 0.0328,
        // article chunk 0.0318. With a 5% boost the chunk wins by ~5%.
        let hits = vec![hit(standalone, 0.0328), hit(article_chunk, 0.0318)];
        let boosted = boost_sourced_hits(hits, |m| m == article_chunk);
        assert_eq!(
            boosted[0].memory, article_chunk,
            "source chunk should rank first"
        );
        assert_eq!(boosted[1].memory, standalone);
    }

    #[test]
    fn source_boost_doesnt_flip_clearly_stronger_standalone() {
        let standalone = MemoryRef(new_id());
        let article_chunk = MemoryRef(new_id());
        // 2× gap: standalone is clearly more relevant. 5% boost on the
        // chunk leaves it at 0.0210, still below the standalone's 0.04.
        let hits = vec![hit(standalone, 0.04), hit(article_chunk, 0.02)];
        let boosted = boost_sourced_hits(hits, |m| m == article_chunk);
        assert_eq!(
            boosted[0].memory, standalone,
            "5% boost mustn't override a clearly-stronger standalone"
        );
    }

    #[test]
    fn source_boost_is_noop_when_no_hits_are_sourced() {
        let m1 = MemoryRef(new_id());
        let m2 = MemoryRef(new_id());
        let hits = vec![hit(m1, 0.05), hit(m2, 0.03)];
        let boosted = boost_sourced_hits(hits, |_| false);
        assert_eq!(boosted[0].memory, m1);
        assert_eq!(
            boosted[0].score, 0.05,
            "scores untouched when nothing is sourced"
        );
    }

    #[test]
    fn source_boost_is_stable_for_already_top_chunks() {
        let chunk = MemoryRef(new_id());
        let standalone = MemoryRef(new_id());
        // Chunk already #1; boost just widens the gap, order unchanged.
        let hits = vec![hit(chunk, 0.05), hit(standalone, 0.03)];
        let boosted = boost_sourced_hits(hits, |m| m == chunk);
        assert_eq!(boosted[0].memory, chunk);
        assert!(boosted[0].score > 0.05);
    }

    fn node_record(tags: &[&str], keywords: &[&str]) -> mnesio_graph::NodeRecord {
        mnesio_graph::NodeRecord {
            id: MemoryRef(new_id()),
            scope: Scope::global("demo"),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
            source: None,
            position: None,
            evolution_count: 0,
            valid_from: OffsetDateTime::UNIX_EPOCH,
            valid_to: None,
            tx_from: OffsetDateTime::UNIX_EPOCH,
            tx_to: None,
            label: None,
            is_source: false,
        }
    }

    #[test]
    fn relation_label_covers_all_variants() {
        assert_eq!(relation_label(Relation::Linked), "linked");
        assert_eq!(relation_label(Relation::EvolvedFrom), "evolved_from");
        assert_eq!(relation_label(Relation::EvolvedTo), "evolved_to");
        assert_eq!(relation_label(Relation::ContainedIn), "contained_in");
    }

    #[test]
    fn node_dto_label_prefers_tags_then_keywords_then_id() {
        let with_tags = node_dto(
            &node_record(&["earnings", "q3"], &["revenue"]),
            Some(1),
            None,
        );
        assert_eq!(with_tags.label, "earnings, q3");
        assert_eq!(with_tags.hop, Some(1));
        assert!(with_tags.out_degree.is_none());

        let kw_only = node_dto(&node_record(&[], &["revenue"]), None, Some((2, 3)));
        assert_eq!(kw_only.label, "revenue");
        assert_eq!(kw_only.out_degree, Some(2));
        assert_eq!(kw_only.in_degree, Some(3));

        let bare = node_dto(&node_record(&[], &[]), None, None);
        assert!(
            bare.label.starts_with('…'),
            "bare node falls back to a short id suffix, got {:?}",
            bare.label
        );
    }

    #[test]
    fn node_dto_flags_invalidated_and_chunk() {
        let mut n = node_record(&["x"], &[]);
        n.tx_to = Some(OffsetDateTime::UNIX_EPOCH);
        n.source = Some(mnesio_core::types::SourceRef(new_id()));
        let dto = node_dto(&n, Some(0), None);
        assert!(dto.invalidated);
        assert!(dto.is_chunk);
        assert!(!dto.is_source);
    }

    #[test]
    fn node_dto_source_node_uses_label_and_flag() {
        // A source node carries an explicit label (the title) and the
        // is_source flag; the label wins over tags/keywords.
        let mut n = node_record(&["ignored-tag"], &["ignored-kw"]);
        n.label = Some("Q3 Board Memo".into());
        n.is_source = true;
        let dto = node_dto(&n, Some(1), None);
        assert_eq!(dto.label, "Q3 Board Memo");
        assert!(dto.is_source);
        assert!(!dto.is_chunk, "a source node is not itself a chunk");
    }
}

fn event_kind(e: &Event) -> &'static str {
    match e {
        Event::MemoryWritten(_) => "MemoryWritten",
        Event::MemoryEmbedded { .. } => "MemoryEmbedded",
        Event::MemoryNoteEnriched { .. } => "MemoryNoteEnriched",
        Event::MemoryLinksUpdated { .. } => "MemoryLinksUpdated",
        Event::MemoryEvolved { .. } => "MemoryEvolved",
        Event::MemoryInvalidated { .. } => "MemoryInvalidated",
        Event::ObservationRecorded { .. } => "ObservationRecorded",
        Event::ProfileSet { .. } => "ProfileSet",
        Event::AgentAccessGranted { .. } => "AgentAccessGranted",
        Event::AgentAccessRevoked { .. } => "AgentAccessRevoked",
        Event::SourceIngested(_) => "SourceIngested",
        Event::SourceInvalidated { .. } => "SourceInvalidated",
        Event::OutcomeRecorded(_) => "OutcomeRecorded",
        Event::ProceduralProposed { .. } => "ProceduralProposed",
        Event::ProceduralCommitted { .. } => "ProceduralCommitted",
        Event::ProceduralRejected { .. } => "ProceduralRejected",
        Event::LearningCurveRecorded { .. } => "LearningCurveRecorded",
    }
}

fn describe_event(e: &Event) -> String {
    fn short(r: MemoryRef) -> String {
        let s = r.0.to_string();
        s.chars()
            .rev()
            .take(8)
            .collect::<String>()
            .chars()
            .rev()
            .collect()
    }
    match e {
        Event::MemoryWritten(m) => {
            let snippet: String = m.content.chars().take(48).collect();
            format!("wrote …{}  «{}»", short(MemoryRef(m.id)), snippet)
        }
        Event::MemoryEmbedded {
            id,
            embedding,
            model_id,
        } => format!(
            "embedded …{} · {}-dim · {model_id}",
            short(*id),
            embedding.len()
        ),
        Event::MemoryNoteEnriched {
            id, keywords, tags, ..
        } => format!(
            "note-enriched …{} · {} keyword(s) · {} tag(s)",
            short(*id),
            keywords.len(),
            tags.len()
        ),
        Event::MemoryLinksUpdated { id, links } => {
            format!("links updated …{} · {} link(s)", short(*id), links.len())
        }
        Event::MemoryEvolved { from, to, .. } => {
            format!("evolved …{} → …{}", short(*from), short(*to))
        }
        Event::MemoryInvalidated { id, reason } => {
            format!("invalidated …{} — {reason}", short(*id))
        }
        Event::ObservationRecorded { content, actor, .. } => {
            let snippet: String = content.chars().take(48).collect();
            match actor {
                Some(a) => format!("observed [{a}] «{snippet}»"),
                None => format!("observed «{snippet}»"),
            }
        }
        Event::ProfileSet {
            attribute, value, ..
        } => {
            if value.is_empty() {
                format!("profile cleared · {attribute}")
            } else {
                format!("profile set · {attribute} = {value}")
            }
        }
        Event::AgentAccessGranted { owner, grantee, .. } => {
            format!("acl grant · {grantee} may read {owner}")
        }
        Event::AgentAccessRevoked { owner, grantee, .. } => {
            format!("acl revoke · {grantee} ✗ {owner}")
        }
        Event::SourceIngested(s) => {
            let title: String = s.title.chars().take(48).collect();
            format!(
                "ingested source …{} · {} chunks · «{title}»",
                short(MemoryRef(s.id)),
                s.chunk_count
            )
        }
        Event::SourceInvalidated { id, reason } => {
            format!("invalidated source …{} — {reason}", short(MemoryRef(id.0)))
        }
        Event::OutcomeRecorded(o) => format!(
            "recorded outcome {}",
            o.id.to_string().chars().rev().take(8).collect::<String>()
        ),
        Event::ProceduralProposed {
            proposal,
            artifacts,
        } => format!(
            "proposed {} artifact(s) for proposal …{}",
            artifacts.len(),
            short(MemoryRef(proposal.0))
        ),
        Event::ProceduralCommitted { proposal, .. } => {
            format!("committed proposal …{}", short(MemoryRef(proposal.0)))
        }
        Event::ProceduralRejected { proposal, reason } => {
            format!(
                "rejected proposal …{} — {reason}",
                short(MemoryRef(proposal.0))
            )
        }
        Event::LearningCurveRecorded {
            artifact,
            version,
            benchmark_score,
            safety_probe_pass_rate,
            ..
        } => format!(
            "curve point …{} v{version} · bench={:.2} · safety={:.2}",
            short(MemoryRef(artifact.0)),
            benchmark_score,
            safety_probe_pass_rate
        ),
    }
}

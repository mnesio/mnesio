//! # mnesio-server
//!
//! Host process for the memory layer. Today it does four things:
//!
//! 1. Opens (or creates) the fjall-backed event log at `MNESIO_DATA`
//!    (default: `./mnesio-data`).
//! 2. Builds the materialized retrieval views (`VectorView`, `Bm25View`),
//!    an [`Embedder`] (`FastEmbedEmbedder` by default, `MockEmbedder` when
//!    `MNESIO_EMBEDDER=mock`), and a `HybridRetriever` over them.
//! 3. Validates that any past `MemoryEmbedded` events in the log were
//!    produced by the same embedder we have today; replays the log into
//!    the views so they catch up to the head before serving.
//! 4. Spawns the async embedding worker and serves the HTTP layer on
//!    `127.0.0.1:7777`:
//!    - `GET /`              the live view (`src/index.html`)
//!    - `GET /api/snapshot`  current log + reconstructed memory state
//!    - `GET /api/search`    hybrid retrieval, per-signal breakdown
//!
//! With `MNESIO_DEMO=1` the server uses a fresh temp directory and spawns a
//! background writer that drops a small themed memory story into the log.

mod acl_worker;
mod causal;
mod code;
mod demo;
mod demo_llm;
mod demo_procedural;
mod dream;
mod embedding_worker;
mod exchange;
mod graph_worker;
mod ingestion_worker;
mod kv;
mod metrics;
mod probe;
mod provenance;
mod viz;

use axum::http::{HeaderValue, Method};
use axum::routing::get;
use axum::Router;
use mnesio_core::event::Event;
use mnesio_core::traits::MaterializedView;
use mnesio_core::{Embedder, EventLog, LlmClient, MnesioError, Retriever};
use mnesio_evolve::EvolveConfig;
use mnesio_graph::FjallGraphView;
use mnesio_index::{
    AgentAclView, Bm25View, FastEmbedEmbedder, HybridRetriever, MockEmbedder, ProfileView,
    SnippetSynthesizer, VectorView,
};
use mnesio_store::FjallEventLog;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use viz::AppState;

/// Tenant the demo writer + viz read/write under. Production will derive
/// this from auth context.
const DEMO_TENANT: &str = "demo";

/// Dimension for the [`MockEmbedder`] when no real embedder is available
/// or explicitly requested. Kept small to keep its vectors cheap.
const MOCK_DIM: usize = 32;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let demo_mode = std::env::var("MNESIO_DEMO")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let data_dir: PathBuf = if demo_mode {
        let d = std::env::temp_dir().join(format!("mnesio-demo-{}", mnesio_core::new_id()));
        tracing::info!(path = %d.display(), "demo mode: using fresh temp data dir");
        d
    } else {
        std::env::var("MNESIO_DATA")
            .unwrap_or_else(|_| "./mnesio-data".into())
            .into()
    };

    let log = FjallEventLog::open(&data_dir)?;
    let log_trait: Arc<dyn EventLog> = log.clone();

    let embedder = build_embedder()?;
    tracing::info!(
        model_id = embedder.model_id(),
        dim = embedder.dim(),
        "embedder ready"
    );

    // Hard rule #4 + long-form §4.5: refuse to boot a view against an
    // embedder different from the one that produced the log's existing
    // `MemoryEmbedded` events. Silently mixing vector spaces would corrupt
    // retrieval quality.
    let entries = log_trait.read_from(None).await?;
    verify_embedder_consistency(&entries, embedder.as_ref())?;

    let vector = Arc::new(VectorView::new(
        embedder.dim(),
        embedder.model_id().to_string(),
    ));
    let bm25 = Arc::new(Bm25View::new()?);
    // Phase-8 (P1#6) — profile / persona memory. In-memory, rebuilt from
    // the log's `ProfileSet` events each boot (Hard Rule #4).
    let profile = Arc::new(ProfileView::new());
    // Phase-8 (P1#7) — multi-agent attribution + inter-agent ACL view.
    let acl = Arc::new(AgentAclView::new());

    // Replay the log into the views — "indexes rebuild from the log".
    let entry_count = entries.len();
    for entry in &entries {
        vector.apply(entry).await?;
        bm25.apply(entry).await?;
        profile.apply(entry).await?;
        acl.apply(entry).await?;
    }
    // Head id after boot replay — the ACL tail worker resumes here so it
    // sees post-boot writes (ingestion/evolution) without re-counting what
    // replay already absorbed.
    let boot_head = entries.last().map(|e| e.id);
    tracing::info!(
        event_count = entry_count,
        data_dir = %data_dir.display(),
        "event log replayed into views"
    );

    let retriever = Arc::new(HybridRetriever::new(
        vector.clone(),
        bm25.clone(),
        embedder.clone(),
    ));

    // Phase 4 — bi-temporal property graph view. Persistent: it owns its
    // own fjall keyspace under `<data_dir>/graph`, so unlike the in-memory
    // vector/BM25 views it isn't rebuilt from scratch each boot. We catch
    // it up from its own checkpoint to the current log head before serving,
    // then let the async graph worker tail anything that arrives later
    // (including the evolution worker's `MemoryEvolved` / `…LinksUpdated`
    // events). Hard Rule #4: every view is rebuildable from the log.
    let graph = FjallGraphView::open(data_dir.join("graph"))?;
    let graph_checkpoint = graph.checkpoint().await?;
    let mut graph_applied = 0usize;
    for entry in &entries {
        if let Some(cp) = graph_checkpoint {
            if entry.id <= cp {
                continue; // already durably applied in a previous run
            }
        }
        graph.apply(entry).await?;
        graph_applied += 1;
    }
    tracing::info!(
        applied = graph_applied,
        resumed_from = ?graph_checkpoint.map(|id| id.to_string()),
        "graph view caught up to log head"
    );

    // Extractive synthesizer — deterministic, no LLM, every word in the
    // answer comes from a real memory.
    let synthesizer: Arc<dyn mnesio_core::Synthesizer> = Arc::new(SnippetSynthesizer::new());

    // Async embedding pipeline: keeps the write path < 5ms even with a
    // heavy embedder (Rule #5). Always on, regardless of demo mode — the
    // demo writer relies on it to fill embeddings.
    embedding_worker::spawn(log_trait.clone(), embedder.clone(), vector.clone());

    // Phase 4 — graph view tail worker. Keeps the persistent graph in
    // sync with the log as new events (demo writes, evolutions) land.
    graph_worker::spawn(log_trait.clone(), graph.clone());

    // Phase 8 (P1#7) — ACL/attribution tail. Counts memory ownership +
    // applies grant/revoke events from the head of the boot replay.
    acl_worker::spawn(log_trait.clone(), acl.clone(), boot_head);

    // Phase 7 — ingestion worker. Tails ObservationRecorded events,
    // extracts atomic facts, and consolidates each into ADD / UPDATE /
    // NOOP (admission-gated). On by default in demo; off otherwise
    // (opt in with MNESIO_INGEST=on) since it's LLM-heavy per observation.
    let ingestion_worker = if ingestion_enabled(demo_mode) {
        let llm = build_llm_client()?;
        tracing::info!(backend = llm_backend_name(), "ingestion worker: spawning");
        let retriever_dyn: Arc<dyn Retriever> = retriever.clone();
        Some(ingestion_worker::spawn(
            log_trait.clone(),
            retriever_dyn,
            vector.clone(),
            bm25.clone(),
            llm,
        ))
    } else {
        tracing::info!("ingestion worker: disabled (set MNESIO_INGEST=on to enable)");
        None
    };

    if demo_mode {
        demo::spawn(
            log_trait.clone(),
            vector.clone(),
            bm25.clone(),
            profile.clone(),
        );
    }

    // Phase 1 — memory evolution worker. Off-path from the write loop;
    // tails the event log and runs the A-MEM three-step pipeline behind
    // `EvolveConfig` bounds (Rules #2, #4, #5, #6). The retriever the
    // worker uses to find semantic neighbours is the same `HybridRetriever`
    // serving the search API — one source of truth.
    let evolution_worker = if evolution_enabled() {
        let llm = build_llm_client()?;
        tracing::info!(
            backend = llm_backend_name(),
            "evolution worker: spawning with LLM backend"
        );
        let retriever_dyn: Arc<dyn Retriever> = retriever.clone();
        Some(mnesio_evolve::spawn(
            log_trait.clone(),
            retriever_dyn,
            llm,
            EvolveConfig::default(),
        ))
    } else {
        tracing::info!("evolution worker: disabled (MNESIO_EVOLVE=off)");
        None
    };

    // Phase 2 — procedural compiler worker. The wedge. Reads OutcomeRecorded
    // events, reflects via LLM, proposes K candidate revisions, shadow-evals
    // through a judge panel, and commits only when EvalGates says yes (Rule
    // #1, mechanically enforced). Off by default — turning this on means
    // many LLM calls per compile pass.
    let procedural_store = Arc::new(mnesio_procedural::ProceduralStore::new());
    let procedural_worker = if procedural_enabled() {
        let llm = build_llm_client()?;
        let min_batch = std::env::var("MNESIO_PROCEDURAL_MIN_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(if demo_mode { 4 } else { 32 });
        tracing::info!(
            backend = llm_backend_name(),
            min_batch,
            "procedural worker: spawning"
        );
        let compiler = demo_procedural::build_compiler(llm, min_batch);
        // In demo mode, seed an initial artifact + stream synthetic outcomes
        // so the dashboard has something to render. Production hosts would
        // seed via their own provisioning path.
        if demo_mode {
            let seed =
                demo_procedural::bootstrap(log_trait.clone(), procedural_store.clone()).await?;
            let shadow_template = demo_procedural::build_shadow_template(&seed);
            // In demo mode, enable the learning-curve eval binding so
            // the dashboard's curve panel populates with real data.
            let eval = demo_procedural::build_eval_binding();
            Some(mnesio_procedural::spawn(
                log_trait.clone(),
                procedural_store.clone(),
                compiler,
                shadow_template,
                Some(eval),
            ))
        } else {
            // Non-demo: store still replays the log so any pre-existing
            // artifacts are tracked. The worker waits for whoever seeded
            // them to also append outcomes; we don't have a default
            // shadow template here so we use an empty one. No eval
            // binding — production hosts plug in their own benchmark
            // suite via a future config surface.
            procedural_store.replay(log_trait.as_ref()).await?;
            let shadow_template = Arc::new(mnesio_procedural::ShadowInputs {
                baseline: demo_procedural::seed_artifact(),
                replay: vec![],
                safety_probes: vec![],
            });
            Some(mnesio_procedural::spawn(
                log_trait.clone(),
                procedural_store.clone(),
                compiler,
                shadow_template,
                None,
            ))
        }
    } else {
        tracing::info!("procedural worker: disabled (set MNESIO_PROCEDURAL=on to enable)");
        None
    };

    let metrics = Arc::new(metrics::MetricsCollector::new());

    let state = Arc::new(AppState {
        log: log_trait,
        vector,
        bm25,
        graph,
        embedder,
        retriever,
        synthesizer,
        metrics,
        evolution: evolution_worker,
        procedural: procedural_worker,
        procedural_store,
        ingestion: ingestion_worker,
        profile,
        acl,
        default_tenant: DEMO_TENANT.into(),
    });
    let app = Router::new()
        .route("/", get(viz::index_html))
        .route("/dashboard", get(viz::dashboard_html))
        .route("/api/snapshot", get(viz::snapshot))
        .route("/api/search", get(viz::search))
        .route("/api/metrics", get(viz::metrics_rollup))
        .route("/api/metrics/history", get(viz::metrics_history))
        .route("/api/evolve/metrics", get(viz::evolve_metrics))
        .route("/api/procedural/metrics", get(viz::procedural_metrics))
        .route("/api/ingest/metrics", get(viz::ingest_metrics))
        .route("/api/profile", get(viz::profile))
        .route("/api/agents", get(viz::agents))
        .route("/api/skills", get(viz::skills))
        .route("/api/graph", get(viz::graph))
        // The one cross-origin route. mnesio.github.io is a static page, so
        // its live panel can only read a server running on the visitor's own
        // machine — which is a cross-origin request and needs CORS.
        //
        // Scoped to this single route rather than applied to the router: every
        // other endpoint returns memory contents, search results, or graph
        // structure, and none of them should be readable by whatever page the
        // user happens to have open in another tab. This one returns counts and
        // rates — no code, no paths, no symbol names — so the blast radius of
        // the allowance is a repository name and a success rate.
        .route(
            "/api/code/curve",
            get(code::code_curve).layer(
                CorsLayer::new()
                    .allow_origin([
                        "https://mnesio.github.io".parse::<HeaderValue>().unwrap(),
                        "http://localhost:4399".parse::<HeaderValue>().unwrap(),
                    ])
                    .allow_methods([Method::GET]),
            ),
        )
        .route("/api/causal/metrics", get(causal::causal_metrics))
        .route("/api/probe/metrics", get(probe::probe_metrics))
        .route("/api/kv/metrics", get(kv::kv_metrics))
        .route("/api/exchange/metrics", get(exchange::exchange_metrics))
        .route("/api/dream/metrics", get(dream::dream_metrics))
        .route("/api/provenance", get(provenance::provenance_metrics))
        .route("/static/chart.umd.min.js", get(viz::chart_js))
        .with_state(state);

    let port: u16 = std::env::var("MNESIO_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7777);
    // Bind host is configurable so the same binary works locally and in a
    // container. Default stays loopback (`127.0.0.1`) so a bare `cargo run`
    // never exposes the server on the network by accident; the Docker image
    // sets `MNESIO_HOST=0.0.0.0` so the published port is reachable.
    let host: IpAddr = std::env::var("MNESIO_HOST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
    let addr = SocketAddr::new(host, port);
    tracing::info!(%addr, "mnesio-server listening — open http://{addr} in a browser");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Select an [`Embedder`] based on the `MNESIO_EMBEDDER` env var.
///
/// - `MNESIO_EMBEDDER=mock` → `MockEmbedder` (deterministic, dependency-free)
/// - `MNESIO_EMBEDDER=fastembed` (default) → `FastEmbedEmbedder` (real
///   semantic, downloads ~30MB on first run)
fn build_embedder() -> anyhow::Result<Arc<dyn Embedder>> {
    let choice = std::env::var("MNESIO_EMBEDDER").unwrap_or_else(|_| "fastembed".into());
    match choice.to_ascii_lowercase().as_str() {
        "mock" => Ok(Arc::new(MockEmbedder::new(MOCK_DIM))),
        "fastembed" => build_fastembed(),
        other => Err(anyhow::anyhow!(
            "unknown MNESIO_EMBEDDER value {other:?}; expected 'mock' or 'fastembed'"
        )),
    }
}

fn build_fastembed() -> anyhow::Result<Arc<dyn Embedder>> {
    Ok(Arc::new(FastEmbedEmbedder::new()?))
}

/// Should the evolution worker spawn at all? Off when `MNESIO_EVOLVE` is
/// explicitly set to `off`/`0`/`false`; on by default.
fn evolution_enabled() -> bool {
    match std::env::var("MNESIO_EVOLVE") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !matches!(v.as_str(), "off" | "0" | "false" | "no")
        }
        Err(_) => true,
    }
}

/// Should the procedural compiler spawn? OFF by default — a compile
/// pass is many LLM calls and shouldn't run unless the operator opts in.
/// Set `MNESIO_PROCEDURAL=on`/`1`/`true` to enable.
fn procedural_enabled() -> bool {
    match std::env::var("MNESIO_PROCEDURAL") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "on" | "1" | "true" | "yes")
        }
        Err(_) => false,
    }
}

/// Should the ingestion worker spawn? On by default in demo mode so the
/// dashboard's Phase-7 panel populates; off otherwise unless
/// `MNESIO_INGEST` is `on`/`1`/`true` (it's an LLM call per observation).
fn ingestion_enabled(demo_mode: bool) -> bool {
    match std::env::var("MNESIO_INGEST") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "on" | "1" | "true" | "yes")
        }
        Err(_) => demo_mode,
    }
}

/// Friendly label for the active LLM backend — used in startup logs +
/// the `/api/evolve/metrics` payload so the dashboard can label the
/// "running on Ollama" vs "running on demo heuristics" pill.
fn llm_backend_name() -> &'static str {
    let choice = std::env::var("MNESIO_EVOLVE_LLM").unwrap_or_default();
    match choice.to_ascii_lowercase().as_str() {
        "ollama" => "ollama",
        _ => "demo",
    }
}

/// Build the `LlmClient` for the evolution worker based on `MNESIO_EVOLVE_LLM`:
///
/// - `ollama` → real local model via [`mnesio_llm::OllamaLlmClient`]; the
///   server will still start even if Ollama isn't running — the worker
///   logs LLM errors as warnings, not panics, so a degraded backend doesn't
///   take the whole process down.
/// - anything else (default) → [`demo_llm::DemoLlmClient`], a content-
///   derived heuristic stub that produces *varying* per-memory enrichments
///   without any external dependency. Good enough to demo the dashboard.
fn build_llm_client() -> anyhow::Result<Arc<dyn LlmClient>> {
    match llm_backend_name() {
        #[cfg(feature = "ollama")]
        "ollama" => {
            let client = mnesio_llm::OllamaLlmClient::from_env()?;
            tracing::info!(
                url = client.base_url(),
                model = client.model(),
                "evolution worker: Ollama LLM client configured"
            );
            Ok(Arc::new(client))
        }
        #[cfg(not(feature = "ollama"))]
        "ollama" => {
            tracing::warn!(
                "MNESIO_EVOLVE_LLM=ollama but the `ollama` feature is disabled — \
                 falling back to DemoLlmClient. Rebuild with default features for \
                 real Ollama support."
            );
            Ok(Arc::new(demo_llm::DemoLlmClient::new()))
        }
        _ => Ok(Arc::new(demo_llm::DemoLlmClient::new())),
    }
}

/// Refuse to boot if the log contains any `MemoryEmbedded` event produced
/// by a different embedder than the one currently configured. Tells the
/// operator exactly how to recover (clear the data dir, set the env var,
/// or wait for a future migration).
fn verify_embedder_consistency(
    entries: &[mnesio_core::LogEntry],
    embedder: &dyn Embedder,
) -> Result<(), MnesioError> {
    let current = embedder.model_id();
    for entry in entries {
        if let Event::MemoryEmbedded { model_id, .. } = &entry.event {
            if model_id != current {
                return Err(MnesioError::Other(anyhow::anyhow!(
                    "embedder mismatch: the log contains embeddings produced by {model_id:?} \
                     but the configured embedder is {current:?}. \
                     Either set MNESIO_EMBEDDER to match the recorded model, \
                     or clear MNESIO_DATA to start fresh."
                )));
            }
        }
    }
    Ok(())
}

//! `GET /api/code/curve` — what the loop is doing on a real repository.
//!
//! ## The question this answers
//!
//! Somebody installs mnesio into Claude Code or Cursor, works for an afternoon,
//! and wants to know whether anything is actually happening. Not "is retrieval
//! good" — that is a benchmark question with a benchmark answer — but the
//! plainer one: *are outcomes being recorded, is there enough evidence to
//! propose anything, and has the gate had a say yet.*
//!
//! Without this endpoint the honest answer is "read the journal file yourself",
//! and the loop is indistinguishable from a loop that silently isn't running.
//!
//! ## Three numbers that are deliberately not merged
//!
//! - **proposed** — rules the evidence would support. Computed here, live.
//! - **committed** / **refused** — what the gate has actually decided.
//!
//! They are separate because they mean different things, and a UI that showed
//! only a single "rules" count would let a proposal read as a shipped
//! improvement. Today `committed` and `refused` are zero in the live server:
//! the gate needs a canary suite to re-run, which the bench harness has and a
//! background server does not. [`GateStatus`] says so in the payload rather
//! than leaving a reader to infer that two zeros mean "nothing was worth
//! committing" when they mean "the gate has not run".
//!
//! ## Why this is read-only and takes no path argument
//!
//! The repository comes from `MNESIO_CODE_REPO` or the working directory,
//! never from the request. A path parameter on a local server reachable from a
//! browser is a directory-probe oracle for any page the user happens to have
//! open; the endpoint would answer "does /Users/x/secret exist" by whether it
//! returns an empty curve or an error.

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::Query as AxumQuery;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use mnesio_code::curve::LiveCurve;
use mnesio_code::graph::{CodeGraph, GraphConfig};
use mnesio_code::journal::OutcomeJournal;
use mnesio_code::learn::{LearnConfig, SymbolLedger};
use mnesio_code::CodeMemory;
use mnesio_core::Scope;

use crate::viz::AppState;

/// Where the gate stands, in words rather than in a pair of zeros.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GateStatus {
    /// No canary suite is loaded, so nothing has been submitted to the gate.
    /// Proposals accumulate; none of them has taken effect.
    NotRunning,
    /// The gate has evaluated at least one proposal.
    ///
    /// Unconstructed today, and kept anyway: collapsing this enum to a
    /// single variant would make `NotRunning` the only possible answer and
    /// therefore no answer at all. It exists so a reader can see that "the
    /// gate has run" is a state this API models and is currently *not* in —
    /// which is the whole reason the field is a word rather than two zeros.
    #[allow(dead_code)]
    Active,
}

/// The payload the dashboard and the site's live panel both read.
#[derive(Debug, Serialize)]
pub struct CurveResponse {
    #[serde(flatten)]
    pub curve: LiveCurve,
    /// Rules the recorded evidence would support proposing. **Not** rules that
    /// are in effect — see [`GateStatus`].
    pub proposed: usize,
    /// Distinct symbols with at least one decisive outcome against them. The
    /// breadth of the evidence, which a rule count alone hides: three rules
    /// from three symbols is a very different picture from three rules from
    /// three hundred.
    pub symbols_with_evidence: usize,
    pub gate: GateStatus,
    /// One line a human can read without a legend.
    pub summary: String,
    /// Where the outcomes are being read from, so a user seeing an empty curve
    /// can check whether the editor is writing to the same place.
    pub journal: String,
    pub repo: String,
}

/// The repository this server reports on.
///
/// `MNESIO_CODE_REPO` when set, otherwise the working directory — which is the
/// right default because the natural way to start this is `mnesio` from inside
/// the repository you are working on.
fn repo_path() -> PathBuf {
    std::env::var_os("MNESIO_CODE_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

pub async fn code_curve(State(_state): State<Arc<AppState>>) -> Json<CurveResponse> {
    let repo = repo_path();
    let journal = OutcomeJournal::for_repo(&repo);
    let read = journal.read();

    let curve = LiveCurve::from_journal(repo.display().to_string(), &read.entries, read.skipped);

    // Re-derive proposals from the journal on every request rather than
    // caching them. It is a fold over a file that grows by one line per edit,
    // and a cache here would be a second place for the truth to live — the
    // exact thing Hard Rule #4 exists to prevent, in miniature.
    let mut ledger = SymbolLedger::default();
    for e in &read.entries {
        ledger.record(&e.outcome);
    }
    let cfg = LearnConfig::default();
    let proposed = ledger.propose(cfg).len();

    Json(CurveResponse {
        summary: curve.summary(),
        proposed,
        symbols_with_evidence: ledger.symbols_with_decisive_evidence(),
        // Honest by construction: the live server has no canary suite to
        // re-run, so nothing has been through the gate. Flipping this to
        // `Active` requires wiring a suite, not editing this line.
        gate: GateStatus::NotRunning,
        journal: journal.path().display().to_string(),
        repo: repo.display().to_string(),
        curve,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_repo_defaults_to_the_working_directory() {
        // Starting `mnesio` inside a repository and having it report on a
        // different one would be a silent, confusing failure.
        std::env::remove_var("MNESIO_CODE_REPO");
        assert_eq!(repo_path(), std::env::current_dir().unwrap());
    }

    #[test]
    fn the_gate_status_serialises_as_a_word_not_a_boolean() {
        // Two zeros for committed/refused are ambiguous between "nothing
        // qualified" and "the gate never ran"; this is what disambiguates
        // them, so it has to survive into the JSON.
        let j = serde_json::to_string(&GateStatus::NotRunning).unwrap();
        assert_eq!(j, "\"not_running\"");
    }
}

// ===================== GET /api/code/graph =====================

/// The indexed repository, built once and kept warm.
///
/// Indexing costs one embedding per symbol, so a per-request rebuild would make
/// the graph view unusable on anything real. The embedding cache (18A) means
/// even the first build is warm if an editor has already indexed this
/// repository — the two processes share a content-addressed cache, so the
/// dashboard does not pay for work the MCP server already did.
static INDEX: tokio::sync::OnceCell<tokio::sync::Mutex<CodeMemory>> =
    tokio::sync::OnceCell::const_new();

#[derive(Debug, Deserialize)]
pub struct GraphParams {
    /// Cap on nodes returned. A browser stops being interactive well before a
    /// 30,000-node force simulation settles, and a hairball is not a picture
    /// of anything (Hard Rule #6).
    #[serde(default = "default_max_nodes")]
    max_nodes: usize,
    /// Hide symbols with no resolved edges. Off by default — see
    /// [`GraphConfig::connected_only`] for why hiding them flatters us.
    #[serde(default)]
    connected_only: bool,
}

fn default_max_nodes() -> usize {
    1500
}

/// Hard ceiling regardless of what the caller asks for.
const MAX_NODES_CEILING: usize = 5000;

#[derive(Debug, Serialize)]
pub struct GraphResponse {
    #[serde(flatten)]
    graph: CodeGraph,
    repo: String,
    files: usize,
    /// The most-called symbols, by name — the "change these carefully" list.
    hubs: Vec<HubDto>,
    /// Symbols retrieved repeatedly that keep not helping. Empty until enough
    /// outcomes exist, and the one view a purely structural graph cannot
    /// produce at all.
    unhelpful: Vec<HubDto>,
    /// Stated in the payload so a viewer cannot render the picture without it.
    resolution_caveat: &'static str,
}

#[derive(Debug, Serialize)]
pub struct HubDto {
    name: String,
    path: String,
    in_degree: usize,
    out_degree: usize,
    success_rate: Option<f32>,
    retrievals: usize,
}

const RESOLUTION_CAVEAT: &str =
    "This graph is a LOWER BOUND on the real call graph. The parser binds calls by bare name, \
     so it cannot always distinguish `foo()` from `x.foo()`; unresolved and ambiguous call sites \
     are dropped rather than guessed. The rendering is therefore sparser and more fragmented \
     than the code actually is.";

/// Build (or reuse) the index and return the repository as a graph.
pub async fn code_graph(
    State(state): State<Arc<AppState>>,
    AxumQuery(params): AxumQuery<GraphParams>,
) -> Response {
    let repo = repo_path();
    if !repo.is_dir() {
        return (
            StatusCode::NOT_FOUND,
            format!(
                "no repository at {}. Set MNESIO_CODE_REPO or start the server inside one.",
                repo.display()
            ),
        )
            .into_response();
    }

    let cell = INDEX
        .get_or_try_init(|| async {
            tracing::info!(repo = %repo.display(), "indexing repository for the graph view");
            CodeMemory::index(&repo, Scope::global("code"), Arc::clone(&state.embedder))
                .await
                .map(tokio::sync::Mutex::new)
        })
        .await;
    let cell = match cell {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("indexing failed: {e}"),
            )
                .into_response()
        }
    };

    let mut memory = cell.lock().await;
    // Same freshness contract as retrieval: a graph drawn from an index that
    // predates the user's last edit shows a repository that no longer exists.
    if let Err(e) = memory.refresh_if_stale().await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("refresh failed: {e}"),
        )
            .into_response();
    }

    let journal = OutcomeJournal::for_repo(&repo).read();
    let graph = memory.graph(
        &journal.entries,
        GraphConfig {
            max_nodes: params.max_nodes.clamp(1, MAX_NODES_CEILING),
            connected_only: params.connected_only,
        },
    );

    let dto = |n: &mnesio_code::GraphNode| HubDto {
        name: n.name.clone(),
        path: n.path.clone(),
        in_degree: n.in_degree,
        out_degree: n.out_degree,
        success_rate: n.evidence.success_rate,
        retrievals: n.evidence.retrievals,
    };
    let hubs = graph.hubs(12).into_iter().map(dto).collect();
    // Same thresholds the suppression learner uses, so what the panel calls
    // "unhelpful" is exactly what would be proposed — a dashboard that used
    // looser numbers would advertise rules the compiler never makes.
    let cfg = LearnConfig::default();
    let unhelpful = graph
        .unhelpful(cfg.min_decisive, cfg.max_success_rate)
        .into_iter()
        .take(12)
        .map(dto)
        .collect();

    Json(GraphResponse {
        repo: repo.display().to_string(),
        files: memory.stats().files,
        hubs,
        unhelpful,
        resolution_caveat: RESOLUTION_CAVEAT,
        graph,
    })
    .into_response()
}

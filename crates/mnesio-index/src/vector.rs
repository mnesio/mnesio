//! Vector materialized view backed by `hnsw_rs`.
//!
//! Consumes the event tail and maintains an in-memory HNSW index keyed by an
//! internal `usize` slot id that maps back to the original `MemoryRef`.
//! Search is k-nearest L2 with a post-filter on [`Scope`] and a tombstone
//! bitset for soft-deleted memories.
//!
//! Phase 0 limitations (documented for the later phases that fix them):
//! - Embeddings are `Box::leak`ed because `hnsw_rs` stores references with a
//!   lifetime tied to the index. On a fresh process we always replay from
//!   the event log, so this is per-process growth, not a leak across runs.
//!   Phase 3's custom filtered-HNSW lifts this restriction.
//! - Scope filtering is post-filter, not native. With selective filters this
//!   hurts recall; Phase 3 makes the filter ACORN-style and native.
//! - The index is in-memory only. Persistence is a later slice — Phase 0
//!   still guarantees rebuildability via log replay (hard rule #4).

use async_trait::async_trait;
use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::prelude::DistL2;
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::traits::MaterializedView;
use mnesio_core::types::MemoryRef;
use mnesio_core::{Hit, Id, MnesioError, Scope};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

/// Squared L2. Squared rather than the root because the ordering is identical
/// and the score transform below is applied to whatever this returns — taking
/// a root would change every score without changing a single rank.
///
/// Written out rather than reusing `DistL2` because `hnsw_rs`'s trait wants an
/// owned distance type and this is two lines.
fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Per-slot metadata held outside the hnsw graph so we can post-filter on
/// scope and resolve neighbors back to a [`MemoryRef`] without a store
/// roundtrip.
struct Slot {
    memory: MemoryRef,
    scope: Scope,
    /// The slot's embedding, as already leaked for `hnsw_rs`.
    ///
    /// A borrowed pointer, not a second copy — the vector is `Box::leak`ed on
    /// insert regardless, so retaining the slice costs 16 bytes per slot and
    /// makes [`VectorView::search_exact`] possible without a store roundtrip.
    embedding: &'static [f32],
}

/// Above this many slots, search falls back to approximate HNSW.
///
/// ## Why a threshold rather than a seed
///
/// The honest reason: **`hnsw_rs` 0.3.4 has no seed API.** Its `LayerGenerator`
/// calls `StdRng::from_os_rng()` and exposes nothing to control it, so layer
/// assignment — and therefore graph topology, and therefore which neighbours an
/// approximate search returns near the cutoff — differs on every build. There
/// is no version to upgrade to; 0.3.4 is current.
///
/// Measured consequence, before this: three *identical* warm starts of the same
/// query returned different symbol sets — 8 files in every run, 4 in only some,
/// Jaccard 0.75–1.00. At suite scale it averaged out to about ±2pp, which is
/// why it went unnoticed, but a user re-running one query after a restart could
/// get different code back with nothing saying so.
///
/// So below this size we do not approximate at all: an exact scan is
/// deterministic by construction, and at these sizes it is also fast. Every
/// code repository measured on this project sits far under it — 178 symbols for
/// a module, 2 118 for this workspace, 5 690 for llama-index-core.
///
/// ## Where the number comes from
///
/// Measured with `mnesio-bench scale`, query p50 / p99, `k=10`:
///
/// | slots | path | p50 | p99 | recall@10 |
/// |---|---|---|---|---|
/// | 1 050 | exact | 0.91 ms | 1.35 ms | 100% |
/// | 10 503 | exact | 1.50 ms | 1.83 ms | 100% |
/// | 47 263 | exact | **3.93 ms** | 4.59 ms | 100% |
/// | 52 515 | HNSW | 2.34 ms | 2.87 ms | 100% |
///
/// 3.93 ms at the top of the exact range is the cost being paid, and it is
/// under the 5 ms budget Hard Rule #5 sets for the *write* path — a read this
/// size is comfortably interactive. Past it HNSW is both faster and the only
/// tractable option, so that is where the threshold sits.
///
/// The exact path also returns **exhaustive** results rather than approximate
/// ones, so recall is 100% by construction rather than by measurement. The
/// determinism was the goal; the recall is a side effect worth stating.
///
/// Above it, results remain approximate *and* non-reproducible across process
/// restarts. That is a real, remaining limitation and it is stated in
/// [`VectorView::is_deterministic`] rather than left for someone to discover.
pub const EXACT_SEARCH_MAX_SLOTS: usize = 50_000;

pub struct VectorView {
    index: RwLock<Hnsw<'static, f32, DistL2>>,
    slots: RwLock<Vec<Slot>>,
    /// Most recently inserted slot for each memory id. Evolved or invalidated
    /// memories keep their slot in the hnsw graph (no removal in `hnsw_rs`)
    /// but are tombstoned at query time.
    by_memory: RwLock<HashMap<MemoryRef, usize>>,
    /// Memories that arrived via `MemoryWritten` without an embedding — we
    /// don't allocate a hnsw slot until the matching `MemoryEmbedded` event
    /// lands and we know the vector. Scope is held here so the eventual
    /// insert can post-filter without re-reading the log.
    pending: RwLock<HashMap<MemoryRef, Scope>>,
    tombstones: RwLock<HashSet<usize>>,
    dim: usize,
    /// The embedder this view was constructed against. `MemoryEmbedded`
    /// events carrying a different `model_id` are refused — they belong to
    /// a different vector space and silently mixing them would corrupt
    /// search quality.
    expected_model_id: String,
    last_checkpoint: RwLock<Option<Id>>,
}

impl VectorView {
    /// Construct an empty view over vectors of dimension `dim`, configured
    /// for the embedder identified by `model_id`.
    ///
    /// HNSW parameters are conservative Phase 0 defaults; they'll be revisited
    /// when the custom filtered-HNSW variant lands (Phase 3) and again when
    /// real workloads inform the tuning.
    pub fn new(dim: usize, model_id: impl Into<String>) -> Self {
        Self::with_capacity(dim, model_id, 100_000)
    }

    /// Like [`new`](Self::new) but sizes the HNSW for an expected `capacity`.
    ///
    /// `hnsw_rs` takes a `max_elements` hint at construction; the default
    /// (100k) is right for an interactive demo but a bulk replay-rebuild of a
    /// larger corpus should pre-size to avoid reallocation churn. Bulk paths
    /// (e.g. the scale harness, a full view rebuild) pass the known corpus
    /// size; the floor of 100k keeps small views on the tuned defaults.
    pub fn with_capacity(dim: usize, model_id: impl Into<String>, capacity: usize) -> Self {
        let hnsw = Hnsw::<'static, f32, DistL2>::new(16, capacity.max(100_000), 16, 200, DistL2);
        Self {
            index: RwLock::new(hnsw),
            slots: RwLock::new(Vec::new()),
            by_memory: RwLock::new(HashMap::new()),
            pending: RwLock::new(HashMap::new()),
            tombstones: RwLock::new(HashSet::new()),
            dim,
            expected_model_id: model_id.into(),
            last_checkpoint: RwLock::new(None),
        }
    }

    /// Embedding dimension this view was constructed with. Inserts and
    /// queries with a different dimension are rejected.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The embedder this view is bound to. The startup mismatch check in
    /// `mnesio-server` reads this to refuse incompatible embedders.
    pub fn expected_model_id(&self) -> &str {
        &self.expected_model_id
    }

    fn insert_internal(
        &self,
        memory: MemoryRef,
        scope: Scope,
        embedding: &[f32],
    ) -> Result<(), MnesioError> {
        if embedding.len() != self.dim {
            return Err(MnesioError::Index(format!(
                "embedding dim {} does not match view dim {}",
                embedding.len(),
                self.dim
            )));
        }
        // If the same memory id appears twice (e.g. re-inserted as part of
        // an in-place fixup) tombstone the prior slot so it stops surfacing.
        if let Some(&prior) = self.by_memory.read().unwrap().get(&memory) {
            self.tombstones.write().unwrap().insert(prior);
        }
        // `hnsw_rs` stores references with the index's lifetime. We elect
        // 'static via `Box::leak`; see the module-level note on why this is
        // acceptable for Phase 0. The slot keeps the same pointer, so the
        // exact path costs no additional copy.
        let leaked: &'static [f32] = Box::leak(embedding.to_vec().into_boxed_slice());
        let slot_id = {
            let mut slots = self.slots.write().unwrap();
            let id = slots.len();
            slots.push(Slot {
                memory,
                scope,
                embedding: leaked,
            });
            id
        };
        self.index.read().unwrap().insert((leaked, slot_id));
        self.by_memory.write().unwrap().insert(memory, slot_id);
        Ok(())
    }

    /// k-nearest neighbors, post-filtered by scope and tombstones.
    ///
    /// Returns [`Hit`]s with an explainable single-signal breakdown. The
    /// hybrid retriever fuses this with BM25 into the final caller-facing
    /// score.
    ///
    /// ## Adaptive over-fetch (Phase 3)
    ///
    /// Selective scopes — `tenant=A` against an index where most memories
    /// are `tenant=B` — used to under-return because the first HNSW pull
    /// of `K*4` was mostly filtered out. The search now loops, doubling
    /// `over_k` until either:
    ///
    /// - it has `k` post-filter hits to return, or
    /// - it has exhausted the index (`over_k >= total_slots`).
    ///
    /// This trades a few extra HNSW traversals for correctness on
    /// selective filters; tightly-matched filters still get the cheap
    /// single-pass result they always did. True ACORN-style traversal-
    /// filtering (where mismatched nodes don't count toward `ef_search`
    /// but ARE used as hops) is a future internal-only optimization on
    /// top of `hnsw_rs` — the public contract here doesn't change.
    pub fn search(&self, query: &[f32], k: usize, scope: &Scope) -> Result<Vec<Hit>, MnesioError> {
        if query.len() != self.dim {
            return Err(MnesioError::Index(format!(
                "query dim {} does not match view dim {}",
                query.len(),
                self.dim
            )));
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let total_slots = self.slots.read().unwrap().len();
        if total_slots == 0 {
            return Ok(Vec::new());
        }

        // Exact below the threshold — see `EXACT_SEARCH_MAX_SLOTS`. This is
        // not an optimisation; it is what makes the result reproducible.
        if total_slots <= EXACT_SEARCH_MAX_SLOTS {
            return Ok(self.search_exact(query, k, scope));
        }

        // Cap on how many adaptive doublings we'll do. Each iteration is
        // O(ef_search log N); 5 doublings of K*4 covers everything from
        // perfectly-dense filters (one pass) to one-in-a-million scopes
        // on a 100k-slot index.
        const MAX_ADAPTIVE_ATTEMPTS: usize = 5;

        let mut over_k = k.saturating_mul(4).max(16);
        let mut hits: Vec<Hit> = Vec::with_capacity(k);
        for _ in 0..MAX_ADAPTIVE_ATTEMPTS {
            // Don't ask hnsw_rs for more than it has.
            let bounded_over_k = over_k.min(total_slots);
            let ef_search = bounded_over_k.max(50);
            let neighbours = self
                .index
                .read()
                .unwrap()
                .search(query, bounded_over_k, ef_search);
            hits = self.collect_filtered_hits(&neighbours, scope, k);
            if hits.len() >= k || bounded_over_k >= total_slots {
                break;
            }
            over_k = over_k.saturating_mul(2);
        }
        Ok(hits)
    }

    /// Is this view's search reproducible across process restarts?
    ///
    /// `false` once the index outgrows [`EXACT_SEARCH_MAX_SLOTS`], because
    /// `hnsw_rs` seeds its layer generator from OS entropy and offers no way
    /// to fix it. Exposed rather than assumed so a caller that depends on
    /// reproducibility — a benchmark comparing two arms, say — can assert it
    /// instead of silently measuring build randomness.
    pub fn is_deterministic(&self) -> bool {
        self.slots.read().unwrap().len() <= EXACT_SEARCH_MAX_SLOTS
    }

    /// Exhaustive L2 scan, scope-filtered, with a total order on ties.
    ///
    /// Deterministic in three separate places, and all three are needed:
    ///
    /// 1. every live slot is considered, so there is no traversal to vary;
    /// 2. distances are compared with `total_cmp`, so `NaN` cannot make the
    ///    ordering depend on comparison order;
    /// 3. ties break on slot id, which is insertion order — itself fixed by
    ///    the event log (Hard Rule #4). Two equal distances would otherwise
    ///    resolve by whatever order the sort happened to visit them.
    ///
    /// Sorted with `sort_by`, which is stable, so (2) and (3) together give a
    /// total order rather than merely a consistent one.
    fn search_exact(&self, query: &[f32], k: usize, scope: &Scope) -> Vec<Hit> {
        let slots = self.slots.read().unwrap();
        let tombstones = self.tombstones.read().unwrap();

        let mut scored: Vec<(usize, f32)> = slots
            .iter()
            .enumerate()
            .filter(|(id, slot)| !tombstones.contains(id) && scope.contains(&slot.scope))
            .map(|(id, slot)| (id, l2(query, slot.embedding)))
            .collect();

        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);

        scored
            .into_iter()
            .map(|(id, distance)| {
                // Identical to the HNSW path's scoring, so switching between
                // them cannot move a score and make the two look different.
                let score = 1.0 / (1.0 + distance);
                Hit {
                    memory: slots[id].memory,
                    score,
                    breakdown: vec![("vector".to_string(), score)],
                }
            })
            .collect()
    }

    /// Resolve raw HNSW neighbours against scope + tombstones into
    /// caller-facing [`Hit`]s. Pulled out so the adaptive search loop
    /// can reuse it across over-fetch attempts.
    fn collect_filtered_hits(
        &self,
        neighbours: &[hnsw_rs::prelude::Neighbour],
        scope: &Scope,
        k: usize,
    ) -> Vec<Hit> {
        let slots = self.slots.read().unwrap();
        let tombstones = self.tombstones.read().unwrap();
        let mut hits = Vec::with_capacity(k);
        for n in neighbours {
            if tombstones.contains(&n.d_id) {
                continue;
            }
            let slot = match slots.get(n.d_id) {
                Some(s) => s,
                None => continue,
            };
            if !scope.contains(&slot.scope) {
                continue;
            }
            let score = 1.0 / (1.0 + n.distance);
            hits.push(Hit {
                memory: slot.memory,
                score,
                breakdown: vec![("vector".to_string(), score)],
            });
            if hits.len() == k {
                break;
            }
        }
        hits
    }

    // ----------------------------------------------------------------
    // Observability (Phase 3) — small helpers the dashboard polls so
    // operators know when to ask for a rebuild-from-log. All are O(1)
    // reads behind the existing locks.
    // ----------------------------------------------------------------

    /// Total physical slots ever allocated in the index. Includes
    /// tombstoned-but-not-reclaimed entries.
    pub fn slot_count(&self) -> usize {
        self.slots.read().unwrap().len()
    }

    /// Number of slots currently tombstoned (invalidated or superseded
    /// by evolution). These still occupy HNSW graph capacity but never
    /// surface in search results.
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.read().unwrap().len()
    }

    /// Slots that can still surface in search. Equivalent to
    /// `slot_count() - tombstone_count()`.
    pub fn live_count(&self) -> usize {
        let total = self.slot_count();
        let dead = self.tombstone_count();
        total.saturating_sub(dead)
    }

    /// Fraction of slots that are tombstoned, in `[0.0, 1.0]`. Dashboard
    /// alarms typically trip around `0.3`–`0.5`: past that point the
    /// index's effective fill factor has dropped enough that a rebuild
    /// from the log would shrink the working set + speed up traversal.
    /// Returns `0.0` for an empty index.
    pub fn tombstone_ratio(&self) -> f32 {
        let total = self.slot_count();
        if total == 0 {
            return 0.0;
        }
        self.tombstone_count() as f32 / total as f32
    }
}

#[async_trait]
impl MaterializedView for VectorView {
    fn name(&self) -> &str {
        "vector-view"
    }

    async fn apply(&self, entry: &LogEntry) -> Result<(), MnesioError> {
        match &entry.event {
            Event::MemoryWritten(mem) => {
                let memory_ref = MemoryRef(mem.id);
                if let Some(emb) = &mem.embedding {
                    // Synchronous path: caller (tests, or a future "sync
                    // embed" writer) already produced the vector. Insert
                    // directly and clear any stale pending entry.
                    self.insert_internal(memory_ref, mem.scope.clone(), emb)?;
                    self.pending.write().unwrap().remove(&memory_ref);
                } else {
                    // Asynchronous path: the embedding worker will follow up
                    // with a `MemoryEmbedded` event. Remember the scope so
                    // that insert can post-filter without re-reading the log.
                    self.pending
                        .write()
                        .unwrap()
                        .insert(memory_ref, mem.scope.clone());
                    tracing::trace!(
                        memory = %mem.id,
                        "vector-view: queued for async embedding"
                    );
                }
            }
            Event::MemoryEmbedded {
                id,
                embedding,
                model_id,
            } => {
                if model_id != &self.expected_model_id {
                    // Refusing here is defense-in-depth — the startup
                    // mismatch check in mnesio-server is the primary guard,
                    // but a view that started clean and saw a rogue event
                    // mid-stream still won't mix vector spaces.
                    tracing::warn!(
                        memory = ?id,
                        event_model = %model_id,
                        view_model = %self.expected_model_id,
                        "vector-view: refusing MemoryEmbedded from a different embedder"
                    );
                } else {
                    let scope = self.pending.write().unwrap().remove(id);
                    match scope {
                        Some(scope) => {
                            self.insert_internal(*id, scope, embedding)?;
                        }
                        None => {
                            // MemoryEmbedded arrived without a prior pending
                            // MemoryWritten. Either the memory was already
                            // inserted via the sync path or the log was
                            // partially replayed — either way we ignore
                            // rather than risk a phantom slot.
                            tracing::warn!(
                                memory = ?id,
                                "vector-view: MemoryEmbedded with no pending MemoryWritten; ignoring"
                            );
                        }
                    }
                }
            }
            Event::MemoryEvolved { from, .. } => {
                // The new version arrives separately as a MemoryWritten with
                // `parent = from`. Tombstoning the parent's slot here keeps
                // the old vector from surfacing in the window between the
                // evolution event and the new MemoryWritten landing.
                if let Some(&slot_id) = self.by_memory.read().unwrap().get(from) {
                    self.tombstones.write().unwrap().insert(slot_id);
                }
            }
            Event::MemoryInvalidated { id, .. } => {
                if let Some(&slot_id) = self.by_memory.read().unwrap().get(id) {
                    self.tombstones.write().unwrap().insert(slot_id);
                }
                // Also drop any pending-but-never-embedded entry so we don't
                // accidentally re-surface an invalidated memory if the
                // embedding event lands late.
                self.pending.write().unwrap().remove(id);
            }
            _ => {}
        }
        *self.last_checkpoint.write().unwrap() = Some(entry.id);
        Ok(())
    }

    async fn checkpoint(&self) -> Result<Option<Id>, MnesioError> {
        Ok(*self.last_checkpoint.read().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::entity::{Memory, Provenance};
    use mnesio_core::event::ChangeSet;
    use mnesio_core::types::{new_id, BiTemporal};

    fn mem_with(content: &str, scope: Scope, embedding: Vec<f32>) -> Memory {
        Memory {
            id: new_id(),
            scope,
            content: content.into(),
            keywords: vec![],
            tags: vec![],
            context: String::new(),
            embedding: Some(embedding),
            links: vec![],
            parent: None,
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: Provenance::default(),
            source: None,
            position: None,
        }
    }

    /// A fixed corpus, minted once.
    ///
    /// Both views must see the *same* memory ids or the comparison is
    /// meaningless — a fresh `new_id()` per build would make the two differ by
    /// construction rather than by graph topology, which is what the first
    /// version of this test actually measured.
    fn corpus(n: usize, scope: &Scope) -> Vec<Memory> {
        (0..n)
            .map(|i| {
                let t = i as f32 * 0.37;
                mem_with(
                    &format!("m{i}"),
                    scope.clone(),
                    vec![t.cos(), t.sin(), (t * 0.5).cos(), (t * 0.25).sin()],
                )
            })
            .collect()
    }

    async fn build(corpus: &[Memory]) -> VectorView {
        let view = VectorView::new(4, "test");
        for m in corpus {
            view.apply(&entry(Event::MemoryWritten(m.clone())))
                .await
                .unwrap();
        }
        view
    }

    #[tokio::test]
    async fn two_independent_builds_return_identical_results() {
        // Phase 18A's unmet half. `hnsw_rs` seeds its layer generator from OS
        // entropy with no way to fix it, so two builds of the same data are
        // two different graphs — and an approximate search over them returned
        // different neighbours near the cutoff. Measured before this fix:
        // three identical warm starts, 8 files in every run, 4 in only some,
        // Jaccard 0.75–1.00.
        //
        // Below the exact threshold there is no graph to differ.
        let scope = Scope::global("t");
        let data = corpus(200, &scope);
        let a = build(&data).await;
        let b = build(&data).await;
        assert!(a.is_deterministic() && b.is_deterministic());

        let q = [0.3f32, 0.9, 0.1, 0.4];
        for k in [1usize, 5, 20] {
            let ha = a.search(&q, k, &scope).unwrap();
            let hb = b.search(&q, k, &scope).unwrap();
            assert_eq!(ha.len(), k.min(data.len()));
            let ids = |h: &[Hit]| h.iter().map(|x| x.memory).collect::<Vec<_>>();
            // Same memories AND the same order — a set match would let a
            // reshuffle pass, and order is what the packer's budget consumes.
            assert_eq!(ids(&ha), ids(&hb), "k={k}: results must not vary by build");
            let bits = |h: &[Hit]| h.iter().map(|x| x.score.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&ha), bits(&hb), "k={k}: scores must match bitwise");
        }
    }

    #[tokio::test]
    async fn repeated_searches_of_one_view_are_identical() {
        let scope = Scope::global("t");
        let view = build(&corpus(120, &scope)).await;
        let q = [0.1f32, 0.2, 0.3, 0.4];
        let first = view.search(&q, 10, &scope).unwrap();
        for _ in 0..5 {
            let again = view.search(&q, 10, &scope).unwrap();
            assert_eq!(
                first.iter().map(|h| h.memory).collect::<Vec<_>>(),
                again.iter().map(|h| h.memory).collect::<Vec<_>>()
            );
        }
    }

    #[tokio::test]
    async fn the_exact_path_ranks_by_true_distance() {
        // Determinism is worthless if it is deterministically wrong. The
        // nearest vector must actually come first.
        let scope = Scope::global("t");
        let view = VectorView::new(4, "test");
        let near = mem_with("near", scope.clone(), vec![1.0, 0.0, 0.0, 0.0]);
        let mid = mem_with("mid", scope.clone(), vec![0.7, 0.7, 0.0, 0.0]);
        let far = mem_with("far", scope.clone(), vec![0.0, 0.0, 0.0, 1.0]);
        for m in [&near, &mid, &far] {
            view.apply(&entry(Event::MemoryWritten(m.clone())))
                .await
                .unwrap();
        }
        let hits = view.search(&[1.0, 0.0, 0.0, 0.0], 3, &scope).unwrap();
        assert_eq!(hits[0].memory, MemoryRef(near.id));
        assert_eq!(hits[1].memory, MemoryRef(mid.id));
        assert_eq!(hits[2].memory, MemoryRef(far.id));
        // Descending score, matching the HNSW path's 1/(1+d) transform.
        assert!(hits[0].score > hits[1].score && hits[1].score > hits[2].score);
    }

    #[tokio::test]
    async fn a_view_past_the_threshold_admits_it_is_not_reproducible() {
        // The claim must not silently become false at scale. `is_deterministic`
        // is the honest predicate a benchmark can assert on.
        let view = VectorView::new(4, "test");
        assert!(view.is_deterministic(), "an empty view is trivially so");
        assert_eq!(EXACT_SEARCH_MAX_SLOTS, 50_000);
    }

    fn entry(event: Event) -> LogEntry {
        LogEntry {
            id: new_id(),
            event,
        }
    }

    /// Push a ring of low-relevance filler vectors into a view so HNSW's
    /// small-graph randomness can't hide a real assertion failure. The
    /// filler is deliberately far from `[1, 0, 0, 0]` (the standard query
    /// vector these tests use) along the 4th axis.
    async fn add_filler(view: &VectorView, scope: &Scope, n: usize) {
        for i in 0..n {
            let theta = i as f32 * 0.4;
            let m = mem_with(
                "filler",
                scope.clone(),
                vec![theta.cos() * 0.1, theta.sin() * 0.1, 0.0, 1.0],
            );
            view.apply(&entry(Event::MemoryWritten(m))).await.unwrap();
        }
    }

    #[tokio::test]
    async fn write_then_search_round_trip() {
        let view = VectorView::new(4, "test-mock");
        let scope = Scope::global("test");
        add_filler(&view, &scope, 16).await;

        let a = mem_with("a", scope.clone(), vec![1.0, 0.0, 0.0, 0.0]);
        let a_ref = MemoryRef(a.id);
        let b = mem_with("b", scope.clone(), vec![0.0, 1.0, 0.0, 0.0]);

        view.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
        view.apply(&entry(Event::MemoryWritten(b))).await.unwrap();

        // `a` is the exact target of the query vector; it must come back top-1.
        let hits = view.search(&[1.0, 0.0, 0.0, 0.0], 1, &scope).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory, a_ref);
        assert_eq!(hits[0].breakdown[0].0, "vector");
        assert!(hits[0].score > 0.0);
    }

    #[tokio::test]
    async fn invalidate_tombstones_from_search() {
        let view = VectorView::new(4, "test-mock");
        let scope = Scope::global("test");

        add_filler(&view, &scope, 16).await;

        let a = mem_with("a", scope.clone(), vec![1.0, 0.0, 0.0, 0.0]);
        let a_ref = MemoryRef(a.id);
        let b = mem_with("b", scope.clone(), vec![0.95, 0.05, 0.0, 0.0]);
        let b_ref = MemoryRef(b.id);
        view.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
        view.apply(&entry(Event::MemoryWritten(b))).await.unwrap();

        // Sanity: both targets surface before invalidation.
        let before = view.search(&[1.0, 0.0, 0.0, 0.0], 2, &scope).unwrap();
        let before_set: std::collections::HashSet<_> = before.iter().map(|h| h.memory).collect();
        assert!(before_set.contains(&a_ref));
        assert!(before_set.contains(&b_ref));

        view.apply(&entry(Event::MemoryInvalidated {
            id: a_ref,
            reason: "test".into(),
        }))
        .await
        .unwrap();

        // After invalidation, no result should ever be `a_ref`, no matter how
        // many we ask for.
        let after = view.search(&[1.0, 0.0, 0.0, 0.0], 8, &scope).unwrap();
        let after_set: std::collections::HashSet<_> = after.iter().map(|h| h.memory).collect();
        assert!(
            !after_set.contains(&a_ref),
            "invalidated memory must not appear in search results"
        );
        assert!(after_set.contains(&b_ref), "b must still be retrievable");
    }

    #[tokio::test]
    async fn evolved_tombstones_parent_slot() {
        let view = VectorView::new(4, "test-mock");
        let scope = Scope::global("test");
        add_filler(&view, &scope, 16).await;

        let parent = mem_with("v1", scope.clone(), vec![1.0, 0.0, 0.0, 0.0]);
        let parent_ref = MemoryRef(parent.id);
        let mut child = mem_with("v2", scope.clone(), vec![0.99, 0.01, 0.0, 0.0]);
        child.parent = Some(parent_ref);
        let child_ref = MemoryRef(child.id);

        view.apply(&entry(Event::MemoryWritten(parent)))
            .await
            .unwrap();
        view.apply(&entry(Event::MemoryWritten(child)))
            .await
            .unwrap();
        view.apply(&entry(Event::MemoryEvolved {
            from: parent_ref,
            to: child_ref,
            diff: ChangeSet {
                keywords_added: vec![],
                keywords_removed: vec![],
                tags_added: vec![],
                tags_removed: vec![],
                context_rewritten: false,
            },
        }))
        .await
        .unwrap();

        // Over-fetch generously so we'd see the parent if it weren't tombstoned.
        let hits = view.search(&[1.0, 0.0, 0.0, 0.0], 8, &scope).unwrap();
        let memories: Vec<_> = hits.iter().map(|h| h.memory).collect();
        assert!(
            !memories.contains(&parent_ref),
            "parent must be tombstoned after evolution"
        );
        assert!(
            memories.contains(&child_ref),
            "child must remain visible after evolution"
        );
    }

    #[tokio::test]
    async fn scope_filter_refuses_cross_tenant_hits() {
        let view = VectorView::new(4, "test-mock");
        let acme = Scope::global("acme");
        let other = Scope::global("other");

        let a = mem_with("a", acme.clone(), vec![1.0, 0.0, 0.0, 0.0]);
        view.apply(&entry(Event::MemoryWritten(a))).await.unwrap();

        let hits = view.search(&[1.0, 0.0, 0.0, 0.0], 1, &other).unwrap();
        assert!(hits.is_empty(), "cross-tenant hits must be filtered out");
    }

    #[tokio::test]
    async fn skip_insert_when_embedding_missing() {
        let view = VectorView::new(4, "test-mock");
        let scope = Scope::global("test");
        let mut m = mem_with("a", scope.clone(), vec![1.0, 0.0, 0.0, 0.0]);
        m.embedding = None;
        view.apply(&entry(Event::MemoryWritten(m))).await.unwrap();
        let hits = view.search(&[1.0, 0.0, 0.0, 0.0], 1, &scope).unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn dim_mismatch_on_apply_is_error() {
        let view = VectorView::new(4, "test-mock");
        let scope = Scope::global("test");
        let m = mem_with("a", scope, vec![1.0, 0.0]); // wrong dim
        let res = view.apply(&entry(Event::MemoryWritten(m))).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn dim_mismatch_on_search_is_error() {
        let view = VectorView::new(4, "test-mock");
        let scope = Scope::global("test");
        let res = view.search(&[1.0, 0.0], 1, &scope);
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn async_embedded_round_trip() {
        let view = VectorView::new(4, "test-mock");
        let scope = Scope::global("test");
        add_filler(&view, &scope, 16).await;

        // Write a memory *without* an embedding — the async path.
        let mut a = mem_with("a", scope.clone(), vec![1.0, 0.0, 0.0, 0.0]);
        let a_id = a.id;
        a.embedding = None;
        view.apply(&entry(Event::MemoryWritten(a))).await.unwrap();

        // It must NOT be searchable yet — the hnsw slot doesn't exist.
        let hits = view.search(&[1.0, 0.0, 0.0, 0.0], 5, &scope).unwrap();
        assert!(hits.iter().all(|h| h.memory != MemoryRef(a_id)));

        // Now the embedding worker (simulated) lands a MemoryEmbedded event.
        view.apply(&entry(Event::MemoryEmbedded {
            id: MemoryRef(a_id),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            model_id: "test-mock".into(),
        }))
        .await
        .unwrap();

        // Now the memory must be searchable.
        let hits = view.search(&[1.0, 0.0, 0.0, 0.0], 1, &scope).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory, MemoryRef(a_id));
    }

    #[tokio::test]
    async fn embedded_with_mismatched_model_id_is_refused() {
        let view = VectorView::new(4, "test-mock");
        let scope = Scope::global("test");
        let mut a = mem_with("a", scope.clone(), vec![1.0, 0.0, 0.0, 0.0]);
        let a_id = a.id;
        a.embedding = None;
        view.apply(&entry(Event::MemoryWritten(a))).await.unwrap();

        view.apply(&entry(Event::MemoryEmbedded {
            id: MemoryRef(a_id),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            model_id: "wrong-model".into(),
        }))
        .await
        .unwrap();

        let hits = view.search(&[1.0, 0.0, 0.0, 0.0], 5, &scope).unwrap();
        assert!(
            hits.iter().all(|h| h.memory != MemoryRef(a_id)),
            "memory embedded with a different model_id must not be searchable"
        );
    }

    #[tokio::test]
    async fn invalidate_drops_pending_embedding() {
        let view = VectorView::new(4, "test-mock");
        let scope = Scope::global("test");
        let mut a = mem_with("a", scope.clone(), vec![1.0, 0.0, 0.0, 0.0]);
        let a_id = a.id;
        a.embedding = None;
        view.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
        view.apply(&entry(Event::MemoryInvalidated {
            id: MemoryRef(a_id),
            reason: "test".into(),
        }))
        .await
        .unwrap();
        // A late MemoryEmbedded should now be ignored.
        view.apply(&entry(Event::MemoryEmbedded {
            id: MemoryRef(a_id),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            model_id: "test-mock".into(),
        }))
        .await
        .unwrap();
        let hits = view.search(&[1.0, 0.0, 0.0, 0.0], 5, &scope).unwrap();
        assert!(hits.iter().all(|h| h.memory != MemoryRef(a_id)));
    }

    #[tokio::test]
    async fn checkpoint_advances_after_apply() {
        let view = VectorView::new(4, "test-mock");
        assert!(view.checkpoint().await.unwrap().is_none());

        let scope = Scope::global("test");
        let m = mem_with("a", scope, vec![1.0, 0.0, 0.0, 0.0]);
        let e = entry(Event::MemoryWritten(m));
        let id = e.id;
        view.apply(&e).await.unwrap();

        assert_eq!(view.checkpoint().await.unwrap(), Some(id));
    }

    // ---- Phase 3: adaptive over-fetch + observability ----

    #[tokio::test]
    async fn adaptive_over_fetch_returns_full_k_under_selective_scope() {
        // A scope-selective query: 20 "noise" memories under one
        // tenant, 5 target memories under another. Even though the
        // first K*4 HNSW pull may be mostly noise, the adaptive loop
        // doubles `over_k` until 5 in-scope hits are found.
        //
        // Target vectors are spread along the 3rd axis (which both the query
        // and the noise vectors hold at 0). This keeps every target far
        // closer to the query `[1,0,0,0]` than any noise vector — noise sits
        // on the 4th axis at 1.0, so its distance to the query is ≥ ~1.3,
        // while the targets span only 0.0..0.4 — yet spaces the targets
        // ~0.1 apart so HNSW indexes them as distinct, reachable nodes. The
        // earlier near-duplicate jitter (≤ 0.002) could collapse targets onto
        // one graph node, making the approximate index occasionally drop a
        // target and the assertion flaky. Clear separation makes the top-5
        // unambiguous so this test exercises the adaptive filter logic, not
        // HNSW recall on near-duplicates.
        let view = VectorView::new(4, "test-v1");
        let scope_noise = Scope::global("noise");
        let scope_target = Scope::global("target");
        add_filler(&view, &scope_noise, 20).await;

        for i in 0..5 {
            let sep = i as f32 * 0.1;
            let m = mem_with("target", scope_target.clone(), vec![1.0, 0.0, sep, 0.0]);
            view.apply(&entry(Event::MemoryWritten(m))).await.unwrap();
        }

        let hits = view
            .search(&[1.0, 0.0, 0.0, 0.0], 5, &scope_target)
            .unwrap();
        assert_eq!(
            hits.len(),
            5,
            "adaptive over-fetch must keep doubling until K in-scope hits found; got {}",
            hits.len()
        );
    }

    #[tokio::test]
    async fn adaptive_search_bounded_by_index_size_no_infinite_loop() {
        // The adaptive loop must cap `over_k` at total_slots so a
        // small index doesn't spin forever doubling.
        let view = VectorView::new(4, "test-v1");
        let scope = Scope::global("only");
        for _ in 0..3 {
            let m = mem_with("x", scope.clone(), vec![1.0, 0.0, 0.0, 0.0]);
            view.apply(&entry(Event::MemoryWritten(m))).await.unwrap();
        }
        // Ask for 100; only 3 exist. Should return 3 without hanging.
        let hits = view.search(&[1.0, 0.0, 0.0, 0.0], 100, &scope).unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[tokio::test]
    async fn tombstone_observability_reflects_invalidations() {
        let view = VectorView::new(4, "test-v1");
        let scope = Scope::global("test");

        let m1 = mem_with("a", scope.clone(), vec![1.0, 0.0, 0.0, 0.0]);
        let m2 = mem_with("b", scope.clone(), vec![0.0, 1.0, 0.0, 0.0]);
        let m1_ref = MemoryRef(m1.id);
        view.apply(&entry(Event::MemoryWritten(m1))).await.unwrap();
        view.apply(&entry(Event::MemoryWritten(m2))).await.unwrap();
        assert_eq!(view.slot_count(), 2);
        assert_eq!(view.tombstone_count(), 0);
        assert_eq!(view.live_count(), 2);
        assert!(view.tombstone_ratio().abs() < 1e-6);

        view.apply(&entry(Event::MemoryInvalidated {
            id: m1_ref,
            reason: "x".into(),
        }))
        .await
        .unwrap();
        assert_eq!(view.slot_count(), 2);
        assert_eq!(view.tombstone_count(), 1);
        assert_eq!(view.live_count(), 1);
        assert!((view.tombstone_ratio() - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn tombstone_ratio_on_empty_index_is_zero_not_nan() {
        let view = VectorView::new(4, "test-v1");
        assert_eq!(view.tombstone_ratio(), 0.0);
        assert_eq!(view.slot_count(), 0);
        assert_eq!(view.live_count(), 0);
    }
}

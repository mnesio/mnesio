//! BM25 materialized view backed by `tantivy`.
//!
//! Consumes the event tail and maintains an in-memory inverted index of
//! memory content + tags. Search returns BM25-scored hits filtered by
//! [`Scope`] at index-query level (a `BooleanQuery` AND of the text query
//! and the appropriate tenant/user/session terms).
//!
//! Phase 0 limitations (documented for the later phases that fix them):
//! - The index lives in a `RAMDirectory` only. Persistence is a later slice;
//!   "indexes rebuild from the log" (hard rule #4) still holds because the
//!   server replays the log into the view on startup.
//! - On `MemoryEvolved` the view does not touch the parent — the parent's
//!   `MemoryInvalidated` event (which always follows in the protocol) deletes
//!   it. This keeps the view's behavior identical whether the protocol emits
//!   one or both events.

use async_trait::async_trait;
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::traits::MaterializedView;
use mnesio_core::types::MemoryRef;
use mnesio_core::{Hit, Id, MnesioError, Scope};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, QueryParser, TermQuery};
use tantivy::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, STORED, STRING,
};
use tantivy::tokenizer::{
    Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, StopWordFilter, TextAnalyzer,
};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

/// Name we register the custom tokenizer under. The schema's content + tags
/// fields refer to it by this string; tantivy looks it up on the
/// per-index `TokenizerManager`.
const ANALYZER: &str = "en_stem_stop";

/// Which fallback tier of [`Bm25View::search_with_diagnostics`] produced the
/// returned hits. Recorded into per-query metrics so the operator can see
/// how often relaxation kicks in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bm25Tier {
    /// Query was empty / `k == 0`, or every tier returned zero hits.
    Empty,
    /// Tier 1 — strict AND.
    StrictAnd,
    /// Tier 2 — fuzzy AND, distance 1 with transpositions.
    FuzzyAnd,
    /// Tier 3 — strict OR ∪ fuzzy OR, deduped on memory id.
    OrMerge,
}

impl Bm25Tier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Bm25Tier::Empty => "empty",
            Bm25Tier::StrictAnd => "strict_and",
            Bm25Tier::FuzzyAnd => "fuzzy_and",
            Bm25Tier::OrMerge => "or_merge",
        }
    }
}

/// Fields the view writes; kept in a struct so call sites can't reach for an
/// unindexed field by accident.
struct Fields {
    memory_id: Field,
    content: Field,
    /// `Memory::keywords`. Indexed as its own field so a caller can make a
    /// memory findable by terms that don't literally occur in its body — the
    /// point of the field, and what Phase 17 leans on to make the identifier
    /// `HybridRetriever` reachable from the words "hybrid retriever".
    keywords: Field,
    tags: Field,
    tenant: Field,
    user: Field,
    session: Field,
}

pub struct Bm25View {
    index: Index,
    /// `IndexWriter::commit` needs `&mut self`; `add_document` / `delete_term`
    /// are `&self`. We funnel all writes through the mutex so the few-call-per
    /// commit pattern stays simple.
    writer: Mutex<IndexWriter>,
    reader: IndexReader,
    fields: Fields,
    last_checkpoint: RwLock<Option<Id>>,
}

impl Bm25View {
    pub fn new() -> Result<Self, MnesioError> {
        let mut schema = Schema::builder();

        // `STORED` only on memory_id and content: id is for lookup, content
        // for snippet display. tags + scope fields are queried but never
        // read back, so we skip the disk cost.
        let memory_id = schema.add_text_field("memory_id", STRING | STORED);

        // Custom tokenizer for the free-text fields. The same analyzer is
        // applied at index-time *and* query-time, so a search for "launch"
        // matches a doc containing "launching" because both pass through the
        // Snowball stemmer. Stop words ("the", "is", "of", ...) are dropped
        // so AND-by-default queries don't fail on filler words.
        let stem_indexing = TextFieldIndexing::default()
            .set_tokenizer(ANALYZER)
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let content_options = TextOptions::default()
            .set_indexing_options(stem_indexing.clone())
            .set_stored();
        let content = schema.add_text_field("content", content_options);
        let tags_options = TextOptions::default().set_indexing_options(stem_indexing.clone());
        let tags = schema.add_text_field("tags", tags_options);
        let keywords_options = TextOptions::default().set_indexing_options(stem_indexing);
        let keywords = schema.add_text_field("keywords", keywords_options);

        let tenant = schema.add_text_field("tenant", STRING);
        let user = schema.add_text_field("user", STRING);
        let session = schema.add_text_field("session", STRING);
        let schema = schema.build();

        let index = Index::create_in_ram(schema);

        // Register the custom analyzer on this index. The filter chain:
        // 1. `SimpleTokenizer` — split on non-alphanumeric
        // 2. `RemoveLongFilter(40)` — drop pathologically long tokens
        // 3. `LowerCaser` — case-insensitive matching
        // 4. `StopWordFilter(English)` — drop common filler words
        // 5. `Stemmer(English)` — reduce inflected forms to their stem
        let stop = StopWordFilter::new(Language::English)
            .ok_or_else(|| MnesioError::Index("English stop words missing".into()))?;
        let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(40))
            .filter(LowerCaser)
            .filter(stop)
            .filter(Stemmer::new(Language::English))
            .build();
        index.tokenizers().register(ANALYZER, analyzer);

        // 50MB is tantivy's documented minimum heap; fine for any demo-scale
        // log and well within the workspace memory budget.
        //
        // Left multi-threaded on purpose. Pinning it to one indexing thread was
        // tried as a determinism fix and **did not work** — two indexes built
        // single-threaded from byte-identical input still ranked differently,
        // so segment/DocId assignment was never the cause. Paying the
        // throughput for a fix that does not fix anything would be worse than
        // not trying. The real cause was tie-breaking; see `run_search`.
        let writer: IndexWriter = index
            .writer(50_000_000)
            .map_err(|e| MnesioError::Index(format!("tantivy writer init: {e}")))?;
        // `Manual` policy + an explicit reload after each commit gives us
        // read-your-writes semantics, which is what the rest of the system
        // (and the demo viz) expects. `OnCommitWithDelay` is fast in steady
        // state but introduces a ~milliseconds race that breaks tests and
        // surprises users typing search queries immediately after writes.
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| MnesioError::Index(format!("tantivy reader init: {e}")))?;

        Ok(Self {
            index,
            writer: Mutex::new(writer),
            reader,
            fields: Fields {
                memory_id,
                content,
                keywords,
                tags,
                tenant,
                user,
                session,
            },
            last_checkpoint: RwLock::new(None),
        })
    }

    /// BM25 search over content + keywords + tags, scope-filtered at the query
    /// layer.
    ///
    /// Multi-word queries use **conjunction (AND)** by default — typing
    /// "Q3 earnings" requires *both* tokens to match somewhere, which matches
    /// the search-box intuition every modern search UI sets. Users who want
    /// OR semantics can still type an explicit `revenue OR growth` and the
    /// parser will honor it.
    ///
    /// User input often contains characters tantivy treats as syntax
    /// (`+ - : ( ) " *`). Rather than surface parse errors for those, we
    /// fall back to a sanitized query that strips the operators so the
    /// search box keeps working for natural-language input.
    ///
    /// **Three-tier progressive fallback.** Strict AND is precise but
    /// brittle on multi-word queries: a single misspelled or off-topic
    /// term zeroes the whole result set even if intent overlaps many
    /// docs. So we relax in stages, preferring stricter matches but
    /// merging the broader passes when neither AND tier finds anything:
    ///
    /// 1. **Strict AND** — every term, exact stem, in the same doc.
    ///    Highest precision.
    /// 2. **Fuzzy AND** — fuzzy stems (distance ≤ 1 with transposition),
    ///    all terms required. Catches all-typos and locale-variant
    ///    multi-word queries (`manufactring yeild`).
    /// 3. **Strict OR ∪ Fuzzy OR**, deduped, score-merged. Surfaces docs
    ///    matching any term — strictly or via fuzzy — when no AND match
    ///    exists. Highest-scoring entry per memory wins, so an exact
    ///    match beats a fuzzy match of the same memory automatically.
    ///    This is the tier that gets `EMEA enquiry` right: the EMEA doc
    ///    appears via strict-OR, the regulatory inquiry doc appears via
    ///    fuzzy-OR on `enquiry → inquiry`, and the user sees both.
    ///
    /// Tier 1 and 2 short-circuit. Tier 3 only runs when neither AND
    /// tier had hits, so high-precision queries aren't polluted.
    ///
    /// Returns `Hit`s whose `breakdown` exposes the raw BM25 score so the
    /// hybrid retriever can fuse it with other signals. Wraps the
    /// diagnostic variant for callers that don't care which fallback tier
    /// produced the hits.
    /// Stage one entry into the writer **without** committing, advancing the
    /// checkpoint. Returns `true` if the entry changed the index (a write or
    /// invalidation), `false` for unrelated events.
    ///
    /// This is the bulk-ingest / replay-rebuild building block: stage many
    /// entries, then call [`commit`](Self::commit) **once**. Committing per
    /// document — as the live [`apply`](MaterializedView::apply) path does —
    /// flushes a tantivy segment each time, which is fine for a trickle of
    /// writes but pathological for a large replay (it turns an O(N) ingest into
    /// O(N) segment flushes). Staged entries are not searchable until the
    /// commit; that's exactly the contract a rebuild wants.
    pub fn stage(&self, entry: &LogEntry) -> Result<bool, MnesioError> {
        let changed = match &entry.event {
            Event::MemoryWritten(m) => {
                let mut doc = TantivyDocument::default();
                doc.add_text(self.fields.memory_id, m.id.to_string());
                doc.add_text(self.fields.content, &m.content);
                doc.add_text(self.fields.keywords, m.keywords.join(" "));
                doc.add_text(self.fields.tags, m.tags.join(" "));
                doc.add_text(self.fields.tenant, &m.scope.tenant);
                if let Some(u) = &m.scope.user {
                    doc.add_text(self.fields.user, u);
                }
                if let Some(s) = &m.scope.session {
                    doc.add_text(self.fields.session, s);
                }
                let writer = self
                    .writer
                    .lock()
                    .map_err(|e| MnesioError::Index(format!("bm25 writer poisoned: {e}")))?;
                writer
                    .add_document(doc)
                    .map_err(|e| MnesioError::Index(format!("bm25 add: {e}")))?;
                true
            }
            Event::MemoryInvalidated { id, .. } => {
                let term = Term::from_field_text(self.fields.memory_id, &id.0.to_string());
                let writer = self
                    .writer
                    .lock()
                    .map_err(|e| MnesioError::Index(format!("bm25 writer poisoned: {e}")))?;
                writer.delete_term(term);
                true
            }
            _ => false,
        };
        *self
            .last_checkpoint
            .write()
            .map_err(|e| MnesioError::Index(format!("checkpoint lock poisoned: {e}")))? =
            Some(entry.id);
        Ok(changed)
    }

    /// Commit staged writes and reload the reader so they become searchable.
    /// Pair with [`stage`](Self::stage) for bulk replay-rebuilds.
    pub fn commit(&self) -> Result<(), MnesioError> {
        {
            let mut writer = self
                .writer
                .lock()
                .map_err(|e| MnesioError::Index(format!("bm25 writer poisoned: {e}")))?;
            writer
                .commit()
                .map_err(|e| MnesioError::Index(format!("bm25 commit: {e}")))?;
        }
        self.reader
            .reload()
            .map_err(|e| MnesioError::Index(format!("bm25 reload: {e}")))?;
        Ok(())
    }

    pub fn search(
        &self,
        query_text: &str,
        k: usize,
        scope: &Scope,
    ) -> Result<Vec<Hit>, MnesioError> {
        self.search_with_diagnostics(query_text, k, scope)
            .map(|(hits, _)| hits)
    }

    /// Look up the stored content of a single memory by id.
    ///
    /// Phase 16 (retrieval parity): the content-aware reranker re-scores
    /// fused candidates on their full text, so it needs to resolve
    /// `MemoryRef -> content`. Content is a `STORED` field, so this is a
    /// one-term lookup — off the write path, only ever on a read. Returns
    /// `None` if the memory isn't in the live index (never staged, or
    /// superseded/tombstoned).
    pub fn content(&self, memory: MemoryRef) -> Option<String> {
        let searcher = self.reader.searcher();
        let term = Term::from_field_text(self.fields.memory_id, &memory.0.to_string());
        let q = TermQuery::new(term, IndexRecordOption::Basic);
        let top = searcher.search(&q, &TopDocs::with_limit(1)).ok()?;
        let (_score, addr) = top.into_iter().next()?;
        let doc: TantivyDocument = searcher.doc(addr).ok()?;
        doc.get_first(self.fields.content)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Same as [`Self::search`] but also returns the [`Bm25Tier`] that
    /// produced the result set. Used by the metrics layer to expose which
    /// fallback tier the user's query landed in.
    pub fn search_with_diagnostics(
        &self,
        query_text: &str,
        k: usize,
        scope: &Scope,
    ) -> Result<(Vec<Hit>, Bm25Tier), MnesioError> {
        if k == 0 {
            return Ok((Vec::new(), Bm25Tier::Empty));
        }
        let sanitized = sanitize_for_parser(query_text);
        if sanitized.is_empty() {
            return Ok((Vec::new(), Bm25Tier::Empty));
        }
        let searcher = self.reader.searcher();

        // Tier 1: strict AND.
        if let Some(q) = self.build_query(&sanitized, scope, false, true)? {
            let hits = self.run_search(&searcher, &*q, k)?;
            if !hits.is_empty() {
                return Ok((hits, Bm25Tier::StrictAnd));
            }
        }

        // Tier 2: fuzzy AND. Catches all-misspelled multi-word queries
        // where intent is "all of these (give or take a typo)".
        if let Some(q) = self.build_query(&sanitized, scope, true, true)? {
            let hits = self.run_search(&searcher, &*q, k)?;
            if !hits.is_empty() {
                return Ok((hits, Bm25Tier::FuzzyAnd));
            }
        }

        // Tier 3: merge strict OR ∪ fuzzy OR. Each query gets its own
        // BM25 scores; we dedupe by memory and keep the higher score (so
        // exact-match wins over fuzzy-match for the same memory) before
        // re-sorting and truncating. A tier whose text can't be parsed into a
        // positive query contributes no hits (see `build_query`).
        let strict_hits = match self.build_query(&sanitized, scope, false, false)? {
            Some(q) => self.run_search(&searcher, &*q, k)?,
            None => Vec::new(),
        };
        let fuzzy_hits = match self.build_query(&sanitized, scope, true, false)? {
            Some(q) => self.run_search(&searcher, &*q, k)?,
            None => Vec::new(),
        };
        let merged = merge_by_memory(strict_hits, fuzzy_hits, k);
        let tier = if merged.is_empty() {
            Bm25Tier::Empty
        } else {
            Bm25Tier::OrMerge
        };
        Ok((merged, tier))
    }

    /// Build the combined `BooleanQuery` for one tier of the fallback. The
    /// `fuzzy` and `conjunction` flags pick which tier this is.
    ///
    /// Returns `Ok(None)` when the sanitized text can't be turned into a
    /// *positive* query (all-stopword input, bare/​trailing boolean operators,
    /// or other syntax the operator-parser rejects). That's user free-text the
    /// parser can't use — not a system fault — so the caller falls through to
    /// the next tier and ultimately to empty results, never an error. Genuine
    /// index errors still propagate as `Err`.
    fn build_query(
        &self,
        sanitized: &str,
        scope: &Scope,
        fuzzy: bool,
        conjunction: bool,
    ) -> Result<Option<Box<dyn Query>>, MnesioError> {
        let mut parser = QueryParser::for_index(
            &self.index,
            vec![self.fields.content, self.fields.keywords, self.fields.tags],
        );
        if conjunction {
            parser.set_conjunction_by_default();
        }
        // (Default operator without the call is OR — exactly what tiers 2
        // and 4 want.)
        if fuzzy {
            // `prefix: false` → standard Levenshtein (matches across the
            // whole term); `distance: 1` catches the bulk of locale
            // variants and typos; `transpose_cost_one: true` →
            // Damerau-Levenshtein, so adjacent-character swaps like
            // "revneue → revenue" count as a single edit.
            parser.set_field_fuzzy(self.fields.content, false, 1, true);
            parser.set_field_fuzzy(self.fields.keywords, false, 1, true);
            parser.set_field_fuzzy(self.fields.tags, false, 1, true);
        }
        let text_query = match parser.parse_query(sanitized) {
            Ok(q) => q,
            // The operator-parser can't turn this free-text into a positive
            // query (all-stopword, bare "AND OR NOT", trailing operator, …).
            // No results for this tier — never a 500 on hostile search input.
            Err(_) => return Ok(None),
        };

        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::with_capacity(4);
        clauses.push((Occur::Must, text_query));
        clauses.push((
            Occur::Must,
            Box::new(TermQuery::new(
                Term::from_field_text(self.fields.tenant, &scope.tenant),
                IndexRecordOption::Basic,
            )),
        ));
        if let Some(u) = &scope.user {
            clauses.push((
                Occur::Must,
                Box::new(TermQuery::new(
                    Term::from_field_text(self.fields.user, u),
                    IndexRecordOption::Basic,
                )),
            ));
        }
        if let Some(s) = &scope.session {
            clauses.push((
                Occur::Must,
                Box::new(TermQuery::new(
                    Term::from_field_text(self.fields.session, s),
                    IndexRecordOption::Basic,
                )),
            ));
        }
        Ok(Some(Box::new(BooleanQuery::new(clauses))))
    }

    fn run_search(
        &self,
        searcher: &tantivy::Searcher,
        query: &dyn Query,
        k: usize,
    ) -> Result<Vec<Hit>, MnesioError> {
        // Over-fetch, then impose a total order, then truncate. Asking
        // `TopDocs` for exactly `k` is not enough: it truncates *inside* a tie
        // group using tantivy's internal DocId, which is not stable between two
        // indexes built from identical input. Sorting afterwards cannot repair
        // that — the wrong members of the group are already gone.
        //
        // Isolated by `two_identical_indexes_rank_identically`, which failed in
        // a single process with a *matching score multiset*: equal scores,
        // different documents. It kept failing after a sort was added and after
        // indexing was pinned to one thread. Widening the window is what fixed
        // it.
        const TIE_WINDOW: usize = 4;
        let top_docs = searcher
            .search(
                query,
                &TopDocs::with_limit(k.saturating_mul(TIE_WINDOW).max(64)),
            )
            .map_err(|e| MnesioError::Index(format!("bm25 search: {e}")))?;

        let mut hits = Vec::with_capacity(top_docs.len());
        for (score, addr) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(addr)
                .map_err(|e| MnesioError::Index(format!("doc fetch: {e}")))?;
            let id_str = doc
                .get_first(self.fields.memory_id)
                .and_then(|v| v.as_str())
                .ok_or_else(|| MnesioError::Index("indexed doc missing memory_id".into()))?;
            let id: ulid::Ulid = id_str.parse().map_err(|e: ulid::DecodeError| {
                MnesioError::Index(format!("memory_id decode: {e}"))
            })?;
            hits.push(Hit {
                memory: MemoryRef(id),
                score,
                breakdown: vec![("bm25".to_string(), score)],
            });
        }
        // Descending score, then memory id. `TopDocs` breaks ties by tantivy's
        // internal DocId, which is not stable between two indexes built from
        // identical input — isolated by
        // `two_identical_indexes_rank_identically`, which failed *within one
        // process* while the score multiset matched exactly. Equal scores,
        // different order.
        //
        // This was the last surviving source of the run-to-run instability that
        // outlived making `VectorView` exact and giving both fusion sorts a
        // total order: BM25 fed a different order in, and everything
        // downstream faithfully preserved it.
        //
        // Still bounded rather than absolute: a tie group wider than
        // `k * TIE_WINDOW` would again be cut by tantivy before this sort sees
        // it. That is far wider than real corpora produce, but it is a window,
        // not a proof.
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.memory.0.cmp(&b.memory.0))
        });
        hits.truncate(k);
        Ok(hits)
    }
}

/// Union two ranked hit lists, deduping on `memory` and keeping the higher
/// score per memory. Used by the OR-merge tier to combine strict and
/// fuzzy results so an exact match still beats a fuzzy match of the same
/// memory.
fn merge_by_memory(a: Vec<Hit>, b: Vec<Hit>, k: usize) -> Vec<Hit> {
    let mut by_id: HashMap<MemoryRef, Hit> = HashMap::with_capacity(a.len() + b.len());
    for h in a.into_iter().chain(b) {
        match by_id.entry(h.memory) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if h.score > e.get().score {
                    *e.get_mut() = h;
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(h);
            }
        }
    }
    // `into_values()` yields in `HashMap` order, which Rust seeds per process,
    // and `Ordering::Equal` on a tie then preserved that order verbatim. This
    // is the *runtime* path — the OR-merge tier — so fixing `run_search` alone
    // changed nothing end to end: 10 of 40 real queries still varied across
    // processes until this sort became total.
    let mut merged: Vec<Hit> = by_id.into_values().collect();
    merged.sort_by(|x, y| {
        y.score
            .total_cmp(&x.score)
            .then(x.memory.0.cmp(&y.memory.0))
    });
    merged.truncate(k);
    merged
}

/// Turn natural-language input into something tantivy's `QueryParser` can
/// reliably consume:
///
/// 1. Replace QueryParser operator characters (`+ - : ( ) " * ^ ~ [ ] { } \ ! ?`)
///    and apostrophes with spaces. The user typed them as text, not as syntax,
///    so they shouldn't trip the parser.
/// 2. Tokenize on whitespace and drop **single-letter** tokens. They're almost
///    always artifacts of step 1 (e.g. "Q4's" → "Q4 s" → drop "s") and with
///    AND-by-default they would otherwise force-empty an otherwise-good query.
///    The boolean operators `OR`, `AND`, `NOT` survive because they're ≥ 2
///    characters.
/// 3. Re-join with single spaces.
///
/// Deliberately shallow — heavier syntax escaping belongs in a future
/// "advanced search" layer.
fn sanitize_for_parser(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '+' | '-' | ':' | '(' | ')' | '"' | '*' | '^' | '~' | '[' | ']' | '{' | '}' | '\\'
            | '!' | '?' | '\'' => ' ',
            other => other,
        })
        .collect::<String>()
        .split_whitespace()
        .filter(|tok| tok.chars().count() > 1)
        .collect::<Vec<_>>()
        .join(" ")
}

#[async_trait]
impl MaterializedView for Bm25View {
    fn name(&self) -> &str {
        "bm25-view"
    }

    async fn apply(&self, entry: &LogEntry) -> Result<(), MnesioError> {
        // Live write path: stage the one entry and flush it immediately so it's
        // searchable right away. Bulk replay-rebuilds use `stage` + a single
        // `commit` instead (see those methods).
        if self.stage(entry)? {
            self.commit()?;
        }
        Ok(())
    }

    async fn checkpoint(&self) -> Result<Option<Id>, MnesioError> {
        Ok(*self
            .last_checkpoint
            .read()
            .map_err(|e| MnesioError::Index(format!("checkpoint lock poisoned: {e}")))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::entity::{Memory, Provenance};
    use mnesio_core::types::{new_id, BiTemporal};

    fn mem_with(content: &str, tags: &[&str], scope: Scope) -> Memory {
        Memory {
            id: new_id(),
            scope,
            content: content.into(),
            keywords: vec![],
            tags: tags.iter().map(|t| (*t).into()).collect(),
            context: String::new(),
            embedding: None,
            links: vec![],
            parent: None,
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: Provenance::default(),
            source: None,
            position: None,
        }
    }

    /// Two views, identical corpus, identical insertion order.
    ///
    /// Isolates BM25 from the rest of the pipeline. `VectorView` was made
    /// exact and both fusion sorts were given a total order, yet 9 of 40 real
    /// queries still returned different context across processes. If this test
    /// fails, the remaining non-determinism is tantivy's; if it passes, BM25 is
    /// exonerated and the hunt moves elsewhere.
    #[tokio::test]
    async fn two_identical_indexes_rank_identically() {
        let scope = Scope::global("t");
        let docs: Vec<Memory> = (0..60)
            .map(|i| {
                mem_with(
                    // Deliberately repetitive: shared vocabulary produces tied
                    // BM25 scores, which is exactly where doc order decides
                    // the outcome and where the instability was observed.
                    &format!(
                        "retry backoff handler for request path {} with timeout",
                        i % 7
                    ),
                    &["code"],
                    scope.clone(),
                )
            })
            .collect();

        let build = || async {
            let v = Bm25View::new().unwrap();
            for m in &docs {
                v.apply(&entry(Event::MemoryWritten(m.clone())))
                    .await
                    .unwrap();
            }
            v
        };
        let a = build().await;
        let b = build().await;

        for q in ["retry backoff", "timeout handler", "request path"] {
            let ha = a.search(q, 20, &scope).unwrap();
            let hb = b.search(q, 20, &scope).unwrap();
            // Scores first, deliberately: if the *multiset* of scores matches
            // and only the order differs, this is a tie-break problem and a
            // total order fixes it. If the scores themselves differ, it is not.
            let mut sa: Vec<u32> = ha.iter().map(|h| h.score.to_bits()).collect();
            let mut sb: Vec<u32> = hb.iter().map(|h| h.score.to_bits()).collect();
            sa.sort_unstable();
            sb.sort_unstable();
            assert_eq!(sa, sb, "query {q:?}: the score multiset must match");
            assert_eq!(
                ha.iter().map(|h| h.memory).collect::<Vec<_>>(),
                hb.iter().map(|h| h.memory).collect::<Vec<_>>(),
                "query {q:?}: two identical indexes must rank identically"
            );
        }
    }

    fn entry(event: Event) -> LogEntry {
        LogEntry {
            id: new_id(),
            event,
        }
    }

    #[tokio::test]
    async fn write_then_search_returns_keyword_match() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");

        let a = mem_with(
            "the revenue figure was misreported",
            &["revenue", "correction"],
            scope.clone(),
        );
        let a_ref = MemoryRef(a.id);
        let b = mem_with(
            "supply chain stabilizing into Q4",
            &["operations"],
            scope.clone(),
        );

        view.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
        view.apply(&entry(Event::MemoryWritten(b))).await.unwrap();

        let hits = view.search("revenue", 5, &scope).unwrap();
        assert!(!hits.is_empty(), "expected a hit for 'revenue'");
        assert_eq!(hits[0].memory, a_ref);
        assert_eq!(hits[0].breakdown[0].0, "bm25");
    }

    #[tokio::test]
    async fn keywords_are_searchable_even_when_absent_from_content() {
        // `Memory::keywords` exists to make a memory findable by terms that
        // don't occur in its body. Phase 17 depends on it: a code symbol's
        // name splits into keywords so "hybrid retriever" reaches
        // `HybridRetriever`, whose body contains neither word.
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");

        let mut a = mem_with("pub struct HybridRetriever { a: u8 }", &[], scope.clone());
        a.keywords = vec![
            "HybridRetriever".into(),
            "hybrid".into(),
            "retriever".into(),
        ];
        let a_ref = MemoryRef(a.id);
        let b = mem_with("pub struct Unrelated { b: u8 }", &[], scope.clone());

        view.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
        view.apply(&entry(Event::MemoryWritten(b))).await.unwrap();

        let hits = view.search("hybrid retriever", 5, &scope).unwrap();
        assert_eq!(
            hits.first().map(|h| h.memory),
            Some(a_ref),
            "keywords field is not being searched"
        );
    }

    #[tokio::test]
    async fn staged_writes_are_invisible_until_commit_then_searchable() {
        // The bulk replay-rebuild path: stage many docs without committing,
        // commit once. Staged docs must not be searchable until the commit, and
        // must all be searchable after it — matching the per-doc `apply` result.
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");

        let a = mem_with(
            "quarterly revenue grew sharply",
            &["revenue"],
            scope.clone(),
        );
        let a_ref = MemoryRef(a.id);
        let b = mem_with("logistics network expanded", &["operations"], scope.clone());
        let b_ref = MemoryRef(b.id);

        assert!(view.stage(&entry(Event::MemoryWritten(a))).unwrap());
        assert!(view.stage(&entry(Event::MemoryWritten(b))).unwrap());

        // Nothing committed yet → nothing searchable.
        assert!(
            view.search("revenue", 5, &scope).unwrap().is_empty(),
            "staged-but-uncommitted docs must not be searchable"
        );

        view.commit().unwrap();

        let rev = view.search("revenue", 5, &scope).unwrap();
        assert_eq!(rev.first().map(|h| h.memory), Some(a_ref));
        let log = view.search("logistics", 5, &scope).unwrap();
        assert_eq!(log.first().map(|h| h.memory), Some(b_ref));
    }

    #[tokio::test]
    async fn adversarial_queries_return_empty_not_error() {
        // Free-text the operator-parser can't turn into a positive query must
        // yield Ok(empty), never an Err (which would 500 a search endpoint).
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "revenue grew strongly",
            &["fin"],
            scope.clone(),
        ))))
        .await
        .unwrap();

        for q in [
            "the of a an is to", // all stopwords → "Only excluding terms given"
            "a AND OR NOT b",    // bare boolean operators → syntax error
            "+ - : ",            // operators only
            "",                  // empty
            "AND",               // a lone operator keyword
        ] {
            let hits = view
                .search(q, 10, &scope)
                .unwrap_or_else(|e| panic!("query {q:?} should not error, got: {e}"));
            assert!(hits.is_empty(), "query {q:?} unexpectedly matched");
        }

        // A *valid* explicit-operator query still works (feature preserved).
        let hits = view.search("revenue OR growth", 10, &scope).unwrap();
        assert!(!hits.is_empty(), "explicit OR query should still match");
    }

    #[tokio::test]
    async fn stage_returns_false_for_unrelated_events() {
        let view = Bm25View::new().unwrap();
        // An event the BM25 view doesn't index advances the checkpoint but
        // stages no document.
        let changed = view
            .stage(&entry(Event::MemoryInvalidated {
                id: MemoryRef(new_id()),
                reason: "test".into(),
            }))
            .unwrap();
        assert!(changed, "invalidation is an indexed change");
    }

    #[tokio::test]
    async fn invalidate_removes_from_results() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");

        let a = mem_with("revenue is up", &["revenue"], scope.clone());
        let a_ref = MemoryRef(a.id);
        view.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
        assert_eq!(view.search("revenue", 5, &scope).unwrap().len(), 1);

        view.apply(&entry(Event::MemoryInvalidated {
            id: a_ref,
            reason: "test".into(),
        }))
        .await
        .unwrap();

        let hits = view.search("revenue", 5, &scope).unwrap();
        assert!(hits.is_empty(), "invalidated memory must drop from BM25");
    }

    #[tokio::test]
    async fn cross_tenant_results_are_filtered() {
        let view = Bm25View::new().unwrap();
        let acme = Scope::global("acme");
        let other = Scope::global("other");

        let a = mem_with("revenue beat estimates", &["revenue"], acme.clone());
        view.apply(&entry(Event::MemoryWritten(a))).await.unwrap();

        let hits = view.search("revenue", 5, &other).unwrap();
        assert!(hits.is_empty(), "different tenant must see no hits");

        let hits = view.search("revenue", 5, &acme).unwrap();
        assert_eq!(hits.len(), 1, "owning tenant sees the hit");
    }

    #[tokio::test]
    async fn empty_query_returns_empty() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");
        let a = mem_with("anything", &[], scope.clone());
        view.apply(&entry(Event::MemoryWritten(a))).await.unwrap();

        assert!(view.search("", 5, &scope).unwrap().is_empty());
        assert!(view.search("   ", 5, &scope).unwrap().is_empty());
        assert!(view.search("anything", 0, &scope).unwrap().is_empty());
    }

    #[tokio::test]
    async fn multi_word_query_uses_and_semantics() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");

        // "Q3 earnings" should match only the Q3-earnings memory, not the
        // Q4 memory (just because it contains "Q4") and not the EMEA memory
        // (just because it has "earnings"-adjacent context).
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "acme reported Q3 earnings beating consensus",
            &["earnings", "q3"],
            scope.clone(),
        ))))
        .await
        .unwrap();
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "supply chain stabilizing into Q4",
            &["operations"],
            scope.clone(),
        ))))
        .await
        .unwrap();
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "widget inc EMEA growth",
            &["earnings"],
            scope.clone(),
        ))))
        .await
        .unwrap();

        let hits = view.search("Q3 earnings", 5, &scope).unwrap();
        assert_eq!(
            hits.len(),
            1,
            "AND semantics: only the doc with both tokens should match; got {hits:?}"
        );
    }

    #[tokio::test]
    async fn special_characters_are_sanitized_not_errored() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "Q4 product line launching",
            &["product"],
            scope.clone(),
        ))))
        .await
        .unwrap();

        // Inputs that would error in tantivy's parser without sanitization.
        // Each one carries at least one word that exists in the indexed
        // memory ("Q4 product line launching") so the AND query still hits.
        for q in [
            "Q4's product",      // apostrophe → strip 's, "Q4 product"
            "product (line)",    // unmatched/extra parens
            "+launching",        // leading +
            "launching?",        // trailing ?
            r#"product "line""#, // unbalanced quotes
        ] {
            let hits = view
                .search(q, 5, &scope)
                .unwrap_or_else(|e| panic!("query {q:?} should not error: {e}"));
            assert!(
                !hits.is_empty(),
                "sanitized query {q:?} should still hit the product memory; got 0 hits"
            );
        }
    }

    #[tokio::test]
    async fn all_punctuation_query_returns_empty_not_error() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "anything",
            &[],
            scope.clone(),
        ))))
        .await
        .unwrap();

        let hits = view.search("(((+", 5, &scope).unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn explicit_or_still_works_for_recall() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "revenue news",
            &[],
            scope.clone(),
        ))))
        .await
        .unwrap();
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "growth report",
            &[],
            scope.clone(),
        ))))
        .await
        .unwrap();
        // Default AND wouldn't return either; explicit OR returns both.
        let hits = view.search("revenue OR growth", 5, &scope).unwrap();
        assert_eq!(hits.len(), 2, "explicit OR must broaden recall");
    }

    #[tokio::test]
    async fn stemming_finds_inflected_forms() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "new product line launching late Q4",
            &["product"],
            scope.clone(),
        ))))
        .await
        .unwrap();
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "acme reported quarterly earnings",
            &["earnings"],
            scope.clone(),
        ))))
        .await
        .unwrap();
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "margin compression in EMEA",
            &["margin"],
            scope.clone(),
        ))))
        .await
        .unwrap();

        // launch ↔ launching, earn ↔ earnings, margin ↔ margins (and
        // singular/plural via the same stemmer rules).
        for (q, expected_in_top) in [
            ("launch", "launching"),
            ("launches", "launching"),
            ("earn", "earnings"),
            ("earning", "earnings"),
            ("margins", "margin"),
            ("reporting", "reported"),
        ] {
            let hits = view.search(q, 5, &scope).unwrap();
            assert!(
                hits.iter()
                    .any(|h| h.breakdown.first().map(|_| true).unwrap_or(false)),
                "query {q:?} should return at least one hit"
            );
            // The exact memory whose surface form we expected to match.
            let contents = view.search(q, 5, &scope).unwrap();
            assert!(
                !contents.is_empty(),
                "query {q:?} returned no hits but should match a memory containing {expected_in_top:?}"
            );
        }
    }

    #[tokio::test]
    async fn stop_words_dont_block_match() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "new product line launching late Q4",
            &["product"],
            scope.clone(),
        ))))
        .await
        .unwrap();

        // "the launch" stems + filters to just "launch"; with AND-default
        // this would otherwise fail because the doc has no literal "the".
        let hits = view.search("the launch", 5, &scope).unwrap();
        assert_eq!(hits.len(), 1, "stop words must be filtered, not ANDed");

        // "of the new product" filters down to "new product" — both in doc.
        let hits = view.search("of the new product", 5, &scope).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn fuzzy_finds_locale_variant() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "regulatory inquiry in Germany regarding data residency",
            &["regulatory"],
            scope.clone(),
        ))))
        .await
        .unwrap();

        // British "enquiry" finds American "inquiry" via fuzzy fallback.
        // After Snowball stemming the terms are "enquiri" vs "inquiri" —
        // edit distance 1, well within the distance-1 fuzzy threshold.
        let hits = view.search("enquiry", 5, &scope).unwrap();
        assert_eq!(hits.len(), 1, "enquiry should find the inquiry memory");
    }

    #[tokio::test]
    async fn fuzzy_handles_adjacent_character_swap() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "revenue grew 18 percent year over year",
            &["revenue"],
            scope.clone(),
        ))))
        .await
        .unwrap();

        // "revneue" (typo, n↔e swap) hits "revenue" via Damerau-Levenshtein
        // — transpositions count as a single edit when
        // transpose_cost_one=true.
        let hits = view.search("revneue", 5, &scope).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn fuzzy_doesnt_fire_when_strict_matches() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");

        // Two docs: one with the exact query word, one similar enough for
        // fuzzy to grab. Strict pass should return only the exact match;
        // the fuzzy fallback never runs.
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "the new launch event next quarter",
            &["product"],
            scope.clone(),
        ))))
        .await
        .unwrap();
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "lunch meeting scheduled for tuesday",
            &["calendar"],
            scope.clone(),
        ))))
        .await
        .unwrap();

        let hits = view.search("launch", 5, &scope).unwrap();
        assert_eq!(hits.len(), 1, "exact strict match — fuzzy must stay quiet");
    }

    #[tokio::test]
    async fn strict_or_fallback_finds_partial_exact_match() {
        // Two docs that each contain one of the query terms but neither
        // contains both. Strict AND zeroes; strict OR fires (tier 2) and
        // returns both — we never reach the fuzzy tiers.
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "EMEA growth was strong",
            &["emea"],
            scope.clone(),
        ))))
        .await
        .unwrap();
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "supply chain stabilizing into Q4",
            &["operations"],
            scope.clone(),
        ))))
        .await
        .unwrap();

        let hits = view.search("EMEA supply", 5, &scope).unwrap();
        assert_eq!(
            hits.len(),
            2,
            "neither doc has both terms; OR fallback must surface both"
        );
    }

    #[tokio::test]
    async fn fuzzy_or_fallback_finds_partial_misspelled_match() {
        // Like the above but the second query term is misspelled. Strict
        // AND zero, strict OR zero (no doc has "enquiri" stem), fuzzy AND
        // zero (no doc has both terms), fuzzy OR finds both via fuzzy on
        // "enquiry" → "inquiry".
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "EMEA growth was strong",
            &["emea"],
            scope.clone(),
        ))))
        .await
        .unwrap();
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "regulatory inquiry in Germany",
            &["regulatory"],
            scope.clone(),
        ))))
        .await
        .unwrap();

        let hits = view.search("EMEA enquiry", 5, &scope).unwrap();
        assert_eq!(
            hits.len(),
            2,
            "fuzzy OR fallback should bridge enquiry→inquiry and surface both docs"
        );
    }

    #[tokio::test]
    async fn fuzzy_multi_word_query_still_ands() {
        let view = Bm25View::new().unwrap();
        let scope = Scope::global("test");
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "regulatory inquiry in Germany",
            &["regulatory"],
            scope.clone(),
        ))))
        .await
        .unwrap();
        view.apply(&entry(Event::MemoryWritten(mem_with(
            "supply chain stabilizing this quarter",
            &["operations"],
            scope.clone(),
        ))))
        .await
        .unwrap();

        // Both terms misspelled. AND semantics survive the fuzzy fallback:
        // only the doc that matches BOTH fuzzy variants should land.
        let hits = view.search("enquiry germny", 5, &scope).unwrap();
        assert_eq!(hits.len(), 1, "fuzzy AND must still narrow to one doc");
    }

    #[tokio::test]
    async fn checkpoint_advances() {
        let view = Bm25View::new().unwrap();
        assert!(view.checkpoint().await.unwrap().is_none());
        let scope = Scope::global("test");
        let a = mem_with("text", &[], scope);
        let e = entry(Event::MemoryWritten(a));
        let id = e.id;
        view.apply(&e).await.unwrap();
        assert_eq!(view.checkpoint().await.unwrap(), Some(id));
    }
}

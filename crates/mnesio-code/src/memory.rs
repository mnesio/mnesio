//! The one entry point: index a repository, then ask it for the code you need.
//!
//! Everything else in this crate is a stage — [`crate::parse`] extracts
//! symbols, [`crate::index`] maps them onto log events, [`crate::pack`] fits
//! results to a budget. [`CodeMemory`] is the assembly, and it exists because
//! the stages are useless to a caller who has to wire them together correctly.
//!
//! ## Settings that are measured, not chosen
//!
//! The retrieval configuration here is not a default anyone guessed. Each
//! value came out of `mnesio-bench codeeval` run against tasks derived from
//! real repository history (query = a commit subject a human wrote for other
//! reasons, gold = the symbols that commit touched):
//!
//! - **The reranker is on, at [`mnesio_index::CODE_BOOST`].** Off it scores
//!   52%; at the prose default (0.5) 56%; at 3.0 it reaches 62% on 400 tasks.
//!   Prose keeps its own smaller bonus — see that constant for why the two
//!   must differ.
//! - **Over-fetch stays at the default 4.** Raising it to 20 moved recall by
//!   at most 1pp, so the deeper pool is pure cost.
//! - **One-hop `Calls` expansion is on**, worth +1–4pp on full-body recall.
//!
//! ## What this does not claim
//!
//! Retrieval reaches ~62% of these tasks against a measured ceiling of 91% —
//! 9% of real tasks share no vocabulary at all with the code they touched. A
//! whole-file strategy scores higher given unlimited context and is
//! unaffordable at any real budget. See `CLAUDE.md` Phase 17B for the full
//! numbers, including the ones that don't flatter this design.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use mnesio_core::entity::Memory;
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::traits::MaterializedView;
use mnesio_core::types::{new_id, MemoryRef, Scope};
use mnesio_core::{Embedder, MnesioError, Query, Retriever};
use mnesio_index::{Bm25View, HybridRetriever, LexicalReranker, VectorView};

use crate::pack::{pack, PackConfig, PackSource, PackedContext};
#[cfg(not(feature = "tree-sitter"))]
use crate::HeuristicParser;
use crate::{CodeIndexer, CodeParser, IndexStats, ParsedFile, SymbolKind};

/// Directories that are never a repository's own source.
///
/// Indexing vendored or build output buries the code a query is actually
/// about, and it is the difference between a corpus of thousands of symbols
/// and hundreds of thousands.
const SKIP_DIRS: &[&str] = &[
    "target",
    "node_modules",
    ".git",
    "dist",
    "build",
    "vendor",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".next",
];

/// Extension → language tag.
///
/// Anything unlisted is skipped rather than parsed into plausible nonsense: a
/// wrong symbol boundary produces a memory that retrieves well and is useless.
/// The set widens with the `tree-sitter` feature — 30 languages there against
/// the 6 the dependency-free parser can follow.
#[cfg(feature = "tree-sitter")]
fn language_of(path: &Path) -> Option<&'static str> {
    crate::parse_ts::language_for_extension(path.extension()?.to_str()?)
}

#[cfg(not(feature = "tree-sitter"))]
fn language_of(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()? {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "go" => Some("go"),
        "ts" | "tsx" => Some("typescript"),
        "js" | "jsx" => Some("javascript"),
        "java" => Some("java"),
        _ => None,
    }
}

/// The parser this build uses.
///
/// Real grammars when compiled in, line scanning otherwise. Both satisfy
/// [`CodeParser`], so nothing downstream changes — which is the whole point of
/// the seam (Hard Rule #7).
#[cfg(feature = "tree-sitter")]
fn parser() -> impl CodeParser {
    crate::TreeSitterParser
}

#[cfg(not(feature = "tree-sitter"))]
fn parser() -> impl CodeParser {
    HeuristicParser
}

/// Languages this build can index, for error messages that tell the truth
/// about the binary in front of you rather than about some other build.
#[cfg(feature = "tree-sitter")]
fn supported() -> String {
    crate::parse_ts::supported_languages().join(", ")
}

#[cfg(not(feature = "tree-sitter"))]
fn supported() -> String {
    "rust, python, go, typescript, javascript, java (build with the \
     `tree-sitter` feature for 30)"
        .to_string()
}

/// Time the freshness check without exposing the internals it walks.
///
/// Public only so `examples/freshness_bench.rs` can measure the claim that the
/// no-change path is cheap enough to run on every query. A claim about latency
/// that nobody can reproduce is not a claim.
#[doc(hidden)]
pub fn bench_fingerprint(root: impl AsRef<Path>) -> u64 {
    fingerprint_tree(root.as_ref())
}

/// Time the parse the freshness check exists to avoid. Returns
/// `(files, symbols)`.
#[doc(hidden)]
pub fn bench_parse(root: impl AsRef<Path>) -> Option<(usize, usize)> {
    let t = parse_tree(root.as_ref()).ok()?;
    let symbols = t.files.iter().map(|f| f.symbols.len()).sum();
    Some((t.files.len(), symbols))
}

/// What one indexed symbol needs to be retrieved, scored and packed.
struct SymbolRecord {
    name: String,
    path: String,
    text: String,
    signature: Option<String>,
    kind: SymbolKind,
}

/// A searchable code memory for one repository.
///
/// The repository is a [`Scope`] — that is what stops two indexed codebases
/// leaking into each other's results (Hard Rule #3).
pub struct CodeMemory {
    root: PathBuf,
    scope: Scope,
    embedder: Arc<dyn Embedder>,
    /// Vectors keyed by a hash of the symbol's own text, so a rebuild only
    /// pays to embed code that actually changed.
    embeddings: HashMap<u64, Vec<f32>>,
    /// `None` only between construction and the first build.
    retriever: Option<HybridRetriever>,
    symbols: HashMap<MemoryRef, SymbolRecord>,
    links: HashMap<MemoryRef, Vec<MemoryRef>>,
    module_docs: HashMap<String, String>,
    stats: IndexStats,
    /// Fingerprint of the file tree the current build came from.
    stats_fingerprint: u64,
}

/// One walk of a repository: what was parsed, and a fingerprint of the tree it
/// came from.
struct ParseTree {
    files: Vec<ParsedFile>,
    module_docs: HashMap<String, String>,
    decl: HashMap<(String, String), (Option<String>, SymbolKind)>,
    /// Folds every source file's path, size and mtime. Changing any of the
    /// three changes this, which is what makes staleness detectable without
    /// reading a byte of content.
    fingerprint: u64,
}

/// Hash of a symbol's text, used as the embedding-cache key.
///
/// Content-addressed rather than path-addressed on purpose: a function that
/// moves between files, or a file that is reformatted around it, keeps its
/// vector.
fn content_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Fingerprint a repository without reading a byte of its contents.
///
/// Folds every source file's path, size and mtime. Changing any of the three
/// changes the result, which is what makes staleness detectable from metadata
/// alone. Kept separate from [`parse_tree`] because the no-change path runs on
/// every query and must not pay for parsing.
///
/// Both functions walk in the same order and hash the same fields, so their
/// fingerprints are comparable — pinned by
/// `the_two_fingerprint_paths_agree`.
fn fingerprint_tree(root: &Path) -> u64 {
    use std::hash::Hasher;
    let mut files = Vec::new();
    collect_sources(root, &mut files);
    files.sort();
    let mut fp = std::collections::hash_map::DefaultHasher::new();
    for file in &files {
        let rel = file
            .strip_prefix(root)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string();
        hash_entry(&mut fp, &rel, file);
    }
    fp.finish()
}

/// The one place a file contributes to a fingerprint, so the two walks cannot
/// drift apart.
fn hash_entry(fp: &mut impl std::hash::Hasher, rel: &str, file: &Path) {
    use std::hash::Hash;
    let Ok(md) = std::fs::metadata(file) else {
        return;
    };
    rel.hash(fp);
    md.len().hash(fp);
    if let Ok(m) = md.modified() {
        if let Ok(d) = m.duration_since(std::time::UNIX_EPOCH) {
            d.as_nanos().hash(fp);
        }
    }
}

/// Walk, parse, and fingerprint a repository.
fn parse_tree(root: &Path) -> Result<ParseTree, MnesioError> {
    let mut files = Vec::new();
    collect_sources(root, &mut files);
    files.sort();
    if files.is_empty() {
        return Err(MnesioError::Index(format!(
            "no supported source files under {} — supported: {}",
            root.display(),
            supported()
        )));
    }

    use std::hash::Hasher;
    let mut fp = std::collections::hash_map::DefaultHasher::new();
    let parser = parser();
    let mut parsed = Vec::new();
    let mut module_docs = HashMap::new();
    let mut decl = HashMap::new();

    for file in &files {
        // Repo-relative, because that is what `Source::uri` means and what a
        // caller can act on — an absolute path from the indexing machine is
        // meaningless to whoever reads the answer.
        let rel = file
            .strip_prefix(root)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string();
        hash_entry(&mut fp, &rel, file);
        let (Some(lang), Ok(src)) = (language_of(file), std::fs::read_to_string(file)) else {
            continue;
        };
        let Ok(pf) = parser.parse(&rel, lang, &src) else {
            continue;
        };
        if let Some(doc) = &pf.module_doc {
            module_docs.insert(rel.clone(), doc.clone());
        }
        for s in &pf.symbols {
            decl.insert(
                (s.path.clone(), s.name.clone()),
                (s.signature.clone(), s.kind),
            );
        }
        parsed.push(pf);
    }

    Ok(ParseTree {
        files: parsed,
        module_docs,
        decl,
        fingerprint: fp.finish(),
    })
}

/// One symbol in a search result.
#[derive(Debug, Clone)]
pub struct CodeHit {
    pub name: String,
    pub path: String,
    pub kind: SymbolKind,
    /// The code, or just its declaration when the budget forced a shorter
    /// form. `is_full` says which.
    pub text: String,
    pub is_full: bool,
    /// Why this is here: `true` when retrieval ranked it directly, `false`
    /// when it was pulled in as a callee of something that was.
    pub is_seed: bool,
}

/// A packed answer, ready to hand to a model.
#[derive(Debug, Clone)]
pub struct CodeContext {
    pub hits: Vec<CodeHit>,
    /// One line per file represented, from budget no symbol wanted.
    pub notes: Vec<String>,
    pub tokens_used: usize,
    /// Candidates that would not fit in any form.
    pub dropped: usize,
    /// The packer's own output, kept rather than rendered away.
    ///
    /// [`CodeHit`] flattens `Reason` to `is_seed` for display, which is the
    /// right shape for a model reading the context and the wrong one for
    /// learning from it: an outcome has to be recorded against
    /// [`crate::pack::Reason`] and [`crate::pack::Form`] to be attributable at
    /// all, and neither survives the flattening. `outcome.rs` takes a
    /// `PackedContext` for exactly this reason — "reconstructing it later is
    /// impossible" — so dropping it here would have made recording an outcome
    /// from the MCP server impossible without re-running retrieval.
    pub packed: PackedContext,
}

impl CodeMemory {
    /// Parse and index every supported source file under `root`.
    ///
    /// Embedding happens here rather than on a worker because this is an
    /// explicit, user-initiated indexing call, not the <5ms write path that
    /// Hard Rule #5 protects.
    pub async fn index(
        root: impl AsRef<Path>,
        scope: Scope,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self, MnesioError> {
        let root = root.as_ref().to_path_buf();
        let parsed = parse_tree(&root)?;
        // A warm start costs a disk read; a cold one costs an embedding per
        // symbol. On a large repository that is the difference between an
        // editor answering immediately and stalling for minutes.
        let embeddings = crate::persist::load(&root, embedder.model_id(), embedder.dim())
            .map(|c| c.vectors().clone())
            .unwrap_or_default();
        let mut me = Self {
            root,
            scope,
            embedder,
            embeddings,
            retriever: None,
            symbols: HashMap::new(),
            links: HashMap::new(),
            module_docs: HashMap::new(),
            stats: IndexStats::default(),
            stats_fingerprint: 0,
        };
        me.build(parsed).await?;
        Ok(me)
    }

    /// Re-index if anything on disk changed since the last build.
    ///
    /// Returns `true` when a rebuild happened. When nothing moved this is a
    /// metadata walk — no file reads, no parsing, no embedding.
    ///
    /// **Measured** (`--example freshness_bench`, release, warm page cache),
    /// p50 of 30 iterations against the full parse it avoids:
    ///
    /// | repo | files | check | full parse |
    /// |---|---|---|---|
    /// | mnesio | 144 | 2.6 ms | 916 ms |
    /// | claw-code | 155 | 1.7 ms | 909 ms |
    /// | llama_index | 3,841 | **110 ms** | 12.7 s |
    ///
    /// So it costs ~1% of the work it replaces, but read the last row before
    /// quoting a latency: on a few hundred files this is free, and on a
    /// multi-thousand-file monorepo it is **~110 ms added to every query**.
    /// That is a real per-call tax, not a rounding error, and it scales with
    /// file count rather than repository size.
    ///
    /// Two ways to cut it if that matters, neither implemented: parallelise
    /// the `stat` walk, or debounce with a short TTL. The TTL is the easier
    /// win and the more dangerous one — it trades a bounded staleness window
    /// for latency, which is the exact trade this method exists to refuse.
    ///
    /// **This is what makes the tool safe to use while editing.** An index that
    /// silently answers from a stale snapshot is worse than no index at all —
    /// an agent gets code that no longer exists and edits against it. So
    /// freshness is checked automatically rather than left to a caller
    /// remembering to pass a flag.
    pub async fn refresh_if_stale(&mut self) -> Result<bool, MnesioError> {
        // Fingerprint first, and *only* the fingerprint. Parsing before
        // comparing would re-read and re-parse the whole repository on every
        // query just to discover nothing had changed — which is the common
        // case, and the case this check has to be cheap in.
        if fingerprint_tree(&self.root) == self.stats_fingerprint {
            return Ok(false);
        }
        let parsed = parse_tree(&self.root)?;
        self.build(parsed).await?;
        Ok(true)
    }

    /// Rebuild every view from `parsed`, re-embedding only what changed.
    ///
    /// Parsing and view construction are cheap; **embedding is not**, and it is
    /// the entire reason a naive re-index is too slow to do on every edit. So
    /// embeddings are cached by a hash of the symbol's own content: an
    /// untouched function keeps its vector even if the file around it moved,
    /// and only genuinely new or genuinely changed code is sent to the model.
    async fn build(&mut self, parsed: ParseTree) -> Result<(), MnesioError> {
        let plan = CodeIndexer::new(self.scope.clone()).plan(&parsed.files);

        let vector = Arc::new(VectorView::new(
            self.embedder.dim(),
            self.embedder.model_id().to_string(),
        ));
        let bm25 = Arc::new(Bm25View::new()?);
        let mut symbols: HashMap<MemoryRef, SymbolRecord> = HashMap::new();
        let mut links: HashMap<MemoryRef, Vec<MemoryRef>> = HashMap::new();
        let mut fresh: HashMap<u64, Vec<f32>> = HashMap::new();

        for event in &plan.events {
            match event {
                Event::MemoryWritten(m) => {
                    let key = content_hash(&m.content);
                    let embedding = match self.embeddings.get(&key) {
                        Some(v) => v.clone(),
                        None => self
                            .embedder
                            .embed(std::slice::from_ref(&m.content))
                            .await?
                            .into_iter()
                            .next()
                            .ok_or_else(|| {
                                MnesioError::Index("embedder returned no vector".into())
                            })?,
                    };
                    fresh.insert(key, embedding.clone());

                    let mut m = m.clone();
                    m.embedding = Some(embedding);
                    record(&mut symbols, &m, &parsed.decl, &parsed.module_docs);
                    let entry = LogEntry {
                        id: new_id(),
                        event: Event::MemoryWritten(m),
                    };
                    vector.apply(&entry).await?;
                    bm25.apply(&entry).await?;
                }
                Event::MemoryLinksUpdated { id, links: l } => {
                    links.insert(*id, l.clone());
                }
                _ => {}
            }
        }

        // Drop vectors for code that no longer exists, or the cache grows
        // without bound across a long editing session.
        self.embeddings = fresh;
        // Persist before the views, so a crash during view construction still
        // leaves the expensive half done. A write failure is logged and
        // swallowed: a cache we could not save costs a slow next start, never
        // a failed index.
        let mut cache =
            crate::persist::EmbeddingCache::new(self.embedder.model_id(), self.embedder.dim());
        cache.replace(self.embeddings.clone());
        if let Err(e) = crate::persist::store(&self.root, &cache) {
            tracing::warn!(error = %e, "could not persist code embeddings");
        }
        // See the module docs: every one of these is a measured setting.
        self.retriever = Some(
            HybridRetriever::new(vector, bm25.clone(), Arc::clone(&self.embedder))
                .with_reranker(Arc::new(LexicalReranker::for_code(bm25))),
        );
        self.symbols = symbols;
        self.links = links;
        self.module_docs = parsed.module_docs;
        self.stats = plan.stats;
        self.stats_fingerprint = parsed.fingerprint;
        Ok(())
    }

    /// Retrieve the code most relevant to `task`, fitted to `budget` tokens.
    ///
    /// `task` is meant to be the *actual* thing the agent is doing — "make the
    /// retry backoff configurable" — not a keyword query. That is what the
    /// retrieval settings were measured against.
    pub async fn context_for(&self, task: &str, budget: usize) -> Result<CodeContext, MnesioError> {
        // Seeds are over-fetched relative to what the budget will hold: the
        // packer drops what does not fit, so a short list would leave it no
        // room to degrade or expand.
        let retriever = self
            .retriever
            .as_ref()
            .ok_or_else(|| MnesioError::Index("index not built".into()))?;
        let hits = retriever
            .search(&Query {
                text: task.to_string(),
                scope: self.scope.clone(),
                k: 20,
                time_filter: None,
            })
            .await?;

        let seeds: Vec<MemoryRef> = hits.iter().map(|h| h.memory).collect();
        let packed = pack(
            &seeds,
            self,
            PackConfig {
                budget,
                ..Default::default()
            },
        );
        Ok(self.render(packed))
    }

    /// Files, symbols and edge-resolution counts from the index run.
    pub fn stats(&self) -> &IndexStats {
        &self.stats
    }

    fn render(&self, packed: PackedContext) -> CodeContext {
        let hits = packed
            .symbols
            .iter()
            .filter_map(|s| {
                let rec = self.symbols.get(&s.memory)?;
                Some(CodeHit {
                    name: rec.name.clone(),
                    path: rec.path.clone(),
                    kind: rec.kind,
                    text: match s.form {
                        crate::pack::Form::Full => rec.text.clone(),
                        crate::pack::Form::Signature => {
                            rec.signature.clone().unwrap_or_else(|| rec.text.clone())
                        }
                    },
                    is_full: s.form == crate::pack::Form::Full,
                    is_seed: matches!(s.reason, crate::pack::Reason::Seed(_)),
                })
            })
            .collect();
        CodeContext {
            hits,
            notes: packed
                .notes
                .iter()
                .map(|n| format!("{} — {}", n.path, n.summary))
                .collect(),
            tokens_used: packed.tokens_used,
            dropped: packed.dropped,
            packed,
        }
    }
}

/// The call graph `CodeMemory` already maintains, made available to look at.
///
/// `links` has existed since 17A as the input to 1-hop context expansion; it
/// was simply never readable as a whole. Exposing it is a projection, not new
/// state — there is no second copy to fall out of sync with what the packer
/// walks, so the picture and the retrieval can never disagree.
impl crate::graph::GraphSource for CodeMemory {
    fn symbols(&self) -> Vec<(MemoryRef, String, String, SymbolKind)> {
        self.symbols
            .iter()
            .map(|(m, s)| (*m, s.name.clone(), s.path.clone(), s.kind))
            .collect()
    }

    fn callees(&self, of: MemoryRef) -> Vec<MemoryRef> {
        self.links.get(&of).cloned().unwrap_or_default()
    }

    fn resolution(&self) -> crate::graph::Resolution {
        let e = &self.stats.edges;
        crate::graph::Resolution {
            // Ambiguous call sites count as *seen*. They were found and
            // deliberately dropped rather than guessed, so excluding them
            // would flatter the resolution rate by hiding the cases the
            // resolver handled least well.
            seen: e.resolved + e.unresolved + e.ambiguous,
            resolved: e.resolved,
        }
    }
}

impl CodeMemory {
    /// This repository as a graph, coloured by whatever outcomes have been
    /// recorded for it.
    pub fn graph(
        &self,
        journal: &[crate::journal::JournalEntry],
        cfg: crate::graph::GraphConfig,
    ) -> crate::graph::CodeGraph {
        crate::graph::CodeGraph::build(self, journal, cfg)
    }
}

impl PackSource for CodeMemory {
    fn text(&self, m: MemoryRef) -> Option<&str> {
        self.symbols.get(&m).map(|s| s.text.as_str())
    }
    fn signature(&self, m: MemoryRef) -> Option<&str> {
        self.symbols.get(&m).and_then(|s| s.signature.as_deref())
    }
    fn path(&self, m: MemoryRef) -> Option<&str> {
        self.symbols.get(&m).map(|s| s.path.as_str())
    }
    fn kind(&self, m: MemoryRef) -> Option<SymbolKind> {
        self.symbols.get(&m).map(|s| s.kind)
    }
    fn module_doc(&self, path: &str) -> Option<&str> {
        self.module_docs.get(path).map(String::as_str)
    }
    fn links(&self, m: MemoryRef) -> &[MemoryRef] {
        self.links.get(&m).map_or(&[], Vec::as_slice)
    }
}

/// Reunite an emitted memory with the parse details the event does not carry.
fn record(
    out: &mut HashMap<MemoryRef, SymbolRecord>,
    m: &Memory,
    decl: &HashMap<(String, String), (Option<String>, SymbolKind)>,
    module_docs: &HashMap<String, String>,
) {
    let name = m.keywords.first().cloned().unwrap_or_default();
    // The path is one of the tags; pick the one that names a file we parsed
    // rather than guessing by extension.
    let path = m
        .tags
        .iter()
        .find(|t| module_docs.contains_key(*t) || decl.contains_key(&((*t).clone(), name.clone())))
        .cloned()
        .unwrap_or_default();
    let (signature, kind) = decl
        .get(&(path.clone(), name.clone()))
        .cloned()
        .unwrap_or((None, SymbolKind::Function));
    out.insert(
        MemoryRef(m.id),
        SymbolRecord {
            name,
            path,
            text: m.content.clone(),
            signature,
            kind,
        },
    );
}

fn collect_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            // Hidden directories are tooling, not source.
            if SKIP_DIRS.contains(&name) || name.starts_with('.') {
                continue;
            }
            collect_sources(&p, out);
        } else if language_of(&p).is_some() {
            out.push(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_index::MockEmbedder;

    /// Write a throwaway repo, keyed by a fresh id so parallel tests can't
    /// collide, and clean it up.
    struct TempRepo(PathBuf);
    impl TempRepo {
        fn new(files: &[(&str, &str)]) -> Self {
            let dir = std::env::temp_dir().join(format!("mnesio-code-{}", new_id()));
            for (rel, body) in files {
                let p = dir.join(rel);
                std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                std::fs::write(&p, body).unwrap();
            }
            Self(dir)
        }
    }
    impl Drop for TempRepo {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    fn embedder() -> Arc<dyn Embedder> {
        Arc::new(MockEmbedder::new(32))
    }

    /// Embedder that records how many texts it was asked to embed.
    ///
    /// The only way to prove a cache is working: correctness tests pass
    /// whether or not the vector was reused, so the assertion has to be about
    /// the work avoided.
    struct CountingEmbedder {
        calls: std::sync::atomic::AtomicUsize,
        inner: MockEmbedder,
    }
    impl CountingEmbedder {
        fn new() -> Self {
            Self {
                calls: Default::default(),
                inner: MockEmbedder::new(32),
            }
        }
        fn count(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }
    #[async_trait::async_trait]
    impl Embedder for CountingEmbedder {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, MnesioError> {
            self.calls
                .fetch_add(texts.len(), std::sync::atomic::Ordering::SeqCst);
            self.inner.embed(texts).await
        }
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        fn model_id(&self) -> &str {
            self.inner.model_id()
        }
    }

    #[tokio::test]
    async fn indexes_a_repo_and_answers_a_task() {
        let repo = TempRepo::new(&[(
            "src/retry.rs",
            "/// Retries with exponential backoff.\npub fn retry_with_backoff(n: u32) -> u32 { n * 2 }\n",
        )]);
        let mem = CodeMemory::index(&repo.0, Scope::global("t"), embedder())
            .await
            .unwrap();

        assert_eq!(mem.stats().files, 1);
        let ctx = mem.context_for("retry backoff", 4000).await.unwrap();
        assert!(
            ctx.hits.iter().any(|h| h.name == "retry_with_backoff"),
            "expected the backoff function, got {:?}",
            ctx.hits.iter().map(|h| &h.name).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn an_empty_repo_fails_with_an_actionable_message() {
        // A silent empty index is the worst outcome: every later search
        // returns nothing and the user has no idea why.
        let repo = TempRepo::new(&[("README.md", "no code here")]);
        let msg = match CodeMemory::index(&repo.0, Scope::global("t"), embedder()).await {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a repo with no source must fail, not index silently"),
        };
        assert!(msg.contains("no supported source files"), "got: {msg}");
        assert!(msg.contains("rust"), "should list what is supported: {msg}");
    }

    #[tokio::test]
    async fn vendored_directories_are_not_indexed() {
        let repo = TempRepo::new(&[
            ("src/app.rs", "pub fn app_entry() {}\n"),
            ("node_modules/dep/index.js", "function vendored() {}\n"),
            ("target/debug/gen.rs", "pub fn generated() {}\n"),
        ]);
        let mem = CodeMemory::index(&repo.0, Scope::global("t"), embedder())
            .await
            .unwrap();
        assert_eq!(mem.stats().files, 1, "only the repo's own source counts");
    }

    #[tokio::test]
    async fn the_budget_is_honoured() {
        let big = format!("pub fn huge() {{\n{}\n}}\n", "    let x = 1;\n".repeat(400));
        let repo = TempRepo::new(&[("src/a.rs", &big), ("src/b.rs", "pub fn small() {}\n")]);
        let mem = CodeMemory::index(&repo.0, Scope::global("t"), embedder())
            .await
            .unwrap();

        let ctx = mem.context_for("huge small", 50).await.unwrap();
        assert!(
            ctx.tokens_used <= 50,
            "packed {} tokens into a 50 budget",
            ctx.tokens_used
        );
    }

    /// The wiring test: a language only the grammar parser can read must
    /// actually index through `CodeMemory`, not just through the parser in
    /// isolation. Without this, `TreeSitterParser` could be fully working and
    /// still unreachable from the entry point — which is exactly what it was
    /// before this test existed.
    #[cfg(feature = "tree-sitter")]
    #[tokio::test]
    async fn a_grammar_only_language_reaches_the_entry_point() {
        let repo = TempRepo::new(&[(
            "app/greeter.rb",
            "class Greeter\n  def greet_user\n    puts 'hi'\n  end\nend\n",
        )]);
        let mem = CodeMemory::index(&repo.0, Scope::global("t"), embedder())
            .await
            .unwrap();
        assert_eq!(mem.stats().files, 1, "ruby was not indexed at all");

        let ctx = mem.context_for("greet user", 4000).await.unwrap();
        assert!(
            ctx.hits.iter().any(|h| h.name == "greet_user"),
            "got {:?}",
            ctx.hits.iter().map(|h| &h.name).collect::<Vec<_>>()
        );
    }

    /// An index that answers from a stale snapshot is the single worst
    /// failure this crate can have: the agent edits against code that no
    /// longer exists. Freshness must be automatic, not a flag someone
    /// remembers.
    #[tokio::test]
    async fn an_edit_is_reflected_without_re_indexing_by_hand() {
        let repo = TempRepo::new(&[("src/a.rs", "pub fn original_name() {}\n")]);
        let mut mem = CodeMemory::index(&repo.0, Scope::global("t"), embedder())
            .await
            .unwrap();
        assert!(mem
            .context_for("original name", 4000)
            .await
            .unwrap()
            .hits
            .iter()
            .any(|h| h.name == "original_name"));

        // Rewrite the file the way an agent would.
        std::fs::write(repo.0.join("src/a.rs"), "pub fn renamed_thing() {}\n").unwrap();
        assert!(mem.refresh_if_stale().await.unwrap(), "edit not detected");

        let names: Vec<String> = mem
            .context_for("renamed thing", 4000)
            .await
            .unwrap()
            .hits
            .iter()
            .map(|h| h.name.clone())
            .collect();
        assert!(
            names.contains(&"renamed_thing".to_string()),
            "got {names:?}"
        );
        assert!(
            !names.contains(&"original_name".to_string()),
            "deleted symbol still served: {names:?}"
        );
    }

    /// The two walks must produce the same number, or a repository would
    /// look permanently stale and rebuild on every single query.
    #[test]
    fn the_two_fingerprint_paths_agree() {
        let repo = TempRepo::new(&[
            ("src/a.rs", "pub fn one() {}\n"),
            ("src/nested/b.rs", "pub fn two() {}\n"),
        ]);
        assert_eq!(
            fingerprint_tree(&repo.0),
            parse_tree(&repo.0).unwrap().fingerprint,
            "the cheap walk and the parsing walk disagree"
        );
    }

    #[tokio::test]
    async fn an_unchanged_tree_is_not_rebuilt() {
        // The check runs on every query, so it has to be nearly free when
        // nothing moved — metadata only, no reads and no embedding.
        let repo = TempRepo::new(&[("src/a.rs", "pub fn stable() {}\n")]);
        let mut mem = CodeMemory::index(&repo.0, Scope::global("t"), embedder())
            .await
            .unwrap();
        assert!(!mem.refresh_if_stale().await.unwrap());
        assert!(!mem.refresh_if_stale().await.unwrap());
    }

    #[tokio::test]
    async fn a_new_file_is_picked_up() {
        let repo = TempRepo::new(&[("src/a.rs", "pub fn first() {}\n")]);
        let mut mem = CodeMemory::index(&repo.0, Scope::global("t"), embedder())
            .await
            .unwrap();
        assert_eq!(mem.stats().files, 1);

        std::fs::write(repo.0.join("src/b.rs"), "pub fn second_one() {}\n").unwrap();
        assert!(mem.refresh_if_stale().await.unwrap());
        assert_eq!(mem.stats().files, 2);
    }

    /// Embedding is the expensive part of a rebuild, so it must be paid only
    /// for code that actually changed — otherwise "refresh on every edit" is
    /// unaffordable and the whole design collapses back to staleness.
    #[tokio::test]
    async fn unchanged_symbols_are_not_re_embedded() {
        struct Counting {
            calls: std::sync::atomic::AtomicUsize,
            inner: mnesio_index::MockEmbedder,
        }
        #[async_trait::async_trait]
        impl Embedder for Counting {
            async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, MnesioError> {
                self.calls
                    .fetch_add(texts.len(), std::sync::atomic::Ordering::SeqCst);
                self.inner.embed(texts).await
            }
            fn dim(&self) -> usize {
                self.inner.dim()
            }
            fn model_id(&self) -> &str {
                self.inner.model_id()
            }
        }

        let counting = Arc::new(Counting {
            calls: Default::default(),
            inner: mnesio_index::MockEmbedder::new(32),
        });
        let repo = TempRepo::new(&[
            ("src/a.rs", "pub fn keep_me() {}\n"),
            ("src/b.rs", "pub fn also_keep() {}\n"),
        ]);
        let mut mem = CodeMemory::index(&repo.0, Scope::global("t"), counting.clone())
            .await
            .unwrap();
        let after_first = counting.calls.load(std::sync::atomic::Ordering::SeqCst);
        assert!(after_first >= 2, "expected an embed per symbol");

        // Touch one file, changing one symbol.
        std::fs::write(repo.0.join("src/b.rs"), "pub fn now_different() {}\n").unwrap();
        assert!(mem.refresh_if_stale().await.unwrap());

        let added = counting.calls.load(std::sync::atomic::Ordering::SeqCst) - after_first;
        // One query embedding may also land here; the point is that it is far
        // below a full re-embed of the corpus.
        assert!(
            added <= 2,
            "re-embedded {added} texts for a one-symbol change — the cache is not working"
        );
    }

    /// A restart must be warm. Simulated by dropping the index and building a
    /// fresh one over the same tree with a counting embedder: if persistence
    /// works, the second build embeds nothing.
    #[tokio::test]
    async fn a_restart_reuses_persisted_embeddings() {
        let cache_dir = std::env::temp_dir().join(format!("mnesio-restart-{}", new_id()));
        std::fs::create_dir_all(&cache_dir).unwrap();
        std::env::set_var("MNESIO_CACHE_DIR", &cache_dir);

        let counting = Arc::new(CountingEmbedder::new());
        let repo = TempRepo::new(&[("src/a.rs", "pub fn persisted() {}\n")]);

        let first = CodeMemory::index(&repo.0, Scope::global("t"), counting.clone())
            .await
            .unwrap();
        let after_first = counting.count();
        assert!(after_first >= 1, "first build must embed");
        drop(first);

        // "Restart": a brand-new index over the same tree.
        let second = CodeMemory::index(&repo.0, Scope::global("t"), counting.clone())
            .await
            .unwrap();
        assert_eq!(
            counting.count(),
            after_first,
            "restart re-embedded; the cache is not being read"
        );
        assert!(second
            .context_for("persisted", 4000)
            .await
            .unwrap()
            .hits
            .iter()
            .any(|h| h.name == "persisted"));

        std::env::remove_var("MNESIO_CACHE_DIR");
        std::fs::remove_dir_all(&cache_dir).ok();
    }

    #[tokio::test]
    async fn scopes_do_not_leak_into_each_other() {
        // Hard Rule #3: two indexed repositories must not see each other.
        let repo = TempRepo::new(&[("src/a.rs", "pub fn only_in_alpha() {}\n")]);
        let mem = CodeMemory::index(&repo.0, Scope::global("alpha"), embedder())
            .await
            .unwrap();

        let ctx = mem.context_for("only_in_alpha", 4000).await.unwrap();
        assert!(!ctx.hits.is_empty(), "sanity: own scope finds it");

        // A memory searched under a different scope must return nothing.
        let other = CodeMemory {
            scope: Scope::global("beta"),
            ..mem
        };
        let ctx = other.context_for("only_in_alpha", 4000).await.unwrap();
        assert!(ctx.hits.is_empty(), "cross-scope read leaked results");
    }
}

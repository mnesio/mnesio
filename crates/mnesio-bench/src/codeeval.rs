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
//! - The default suite is **derived from the repo's own git history** (see
//!   [`crate::gitsuite`]): queries are real commit subjects, gold is the
//!   symbols that commit touched. `--suite hand` selects a hand-written smoke
//!   test instead, which cannot support a claim — its queries were written by
//!   someone who already knew the answers, so it can only show the pipeline
//!   is not broken.
//! - **`whole-file` is an unbounded baseline.** It charges the full text of
//!   every file a hit came from, with no context cap. On repos with very large
//!   files that exceeds any real budget, so read it as "what an uncapped
//!   file-level strategy would cost", not as what a tuned agent does.

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use mnesio_code::pack::{pack, Form, PackConfig, PackSource};
use mnesio_code::{CodeIndexer, CodeParser, HeuristicParser, IndexStats, ParsedFile, SymbolKind};
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::traits::MaterializedView;
use mnesio_core::types::{new_id, MemoryRef, Scope};
use mnesio_core::{Query, Retriever};
use mnesio_index::{Bm25View, HybridRetriever, LexicalReranker, VectorView};
use std::sync::Arc;

use crate::memeval::build_embedder;

/// Rough token estimate. See the module docs: identical across arms, so ratios
/// hold even though the absolute value is approximate.
fn est_tokens(s: &str) -> usize {
    s.len().div_ceil(4)
}

/// A symbol a correct retrieval must surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gold {
    /// Repo-relative file the symbol is defined in.
    ///
    /// `None` matches on name alone, which is only safe for a hand-written
    /// suite whose answers are known-unique. A derived suite **must** qualify
    /// by path: `__init__` exists once per Python class and `new` once per
    /// Rust type, so a bare-name match would score retrieving *any* file's
    /// constructor as a hit and inflate every arm.
    pub path: Option<String>,
    pub name: String,
}

impl Gold {
    /// Does the symbol defined at `path` under `name` satisfy this gold entry?
    pub fn matches(&self, path: &str, name: &str) -> bool {
        // `map_or(true, ..)` rather than `is_none_or`: the latter is stable
        // only since 1.82 and the workspace MSRV is 1.79.
        #[allow(clippy::unnecessary_map_or)]
        {
            self.name == name && self.path.as_ref().map_or(true, |p| p == path)
        }
    }
}

/// One task and the symbols that answer it.
pub struct CodeQuery {
    pub question: String,
    /// Scored as "at least one", the realistic agent criterion: the task lands
    /// you in the right code.
    pub gold: Vec<Gold>,
}

impl CodeQuery {
    fn hit(&self, retrieved: &[&SymbolInfo]) -> bool {
        retrieved
            .iter()
            .any(|s| self.gold.iter().any(|g| g.matches(&s.path, &s.name)))
    }
}

/// A hand-built suite over `crates/mnesio-index/src`.
///
/// **Disqualified from proving anything.** The queries were written by someone
/// who already knew which symbol should come back, so this can only show the
/// pipeline is not broken. For a suite that can actually support a claim, see
/// [`crate::gitsuite`] — real commit subjects, gold from git's own line
/// history.
pub fn hand_written_suite() -> Vec<CodeQuery> {
    [
        ("hybrid retriever reciprocal rank fusion", "HybridRetriever"),
        (
            "lexical reranker coverage temporal phrase",
            "LexicalReranker",
        ),
        ("context tree relevant subtree routing", "ContextTree"),
        ("bm25 tantivy search view", "Bm25View"),
        ("paragraph chunker split document", "ParagraphChunker"),
        (
            "snippet synthesizer extractive answer",
            "SnippetSynthesizer",
        ),
        (
            "tenant partitioned vector view multi tenant",
            "TenantPartitionedVectorView",
        ),
        ("agent acl attribution access", "AgentAclView"),
    ]
    .into_iter()
    .map(|(q, g)| CodeQuery {
        question: q.to_string(),
        // Name-only: every answer here is a uniquely-named type in one crate.
        gold: vec![Gold {
            path: None,
            name: g.to_string(),
        }],
    })
    .collect()
}

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

/// One packing policy at one fixed budget.
///
/// Separate from [`ArmResult`] because the question is different: the arms ask
/// "how much does top-`k` cost?", this asks "given a fixed budget, which
/// assembly policy makes the best use of it?" — the decision the packer exists
/// to make.
#[derive(Debug, Clone)]
pub struct BudgetResult {
    pub policy: &'static str,
    pub budget: usize,
    /// Gold symbol present in *any* form, including signature-only.
    pub recalled: usize,
    /// Gold symbol present with its **full body**.
    ///
    /// Reported next to `recalled` because the two answer different questions,
    /// and conflating them would overstate the signature ladder: a declaration
    /// tells an agent a symbol exists and what it takes — enough to decide what
    /// to ask for next, *not* enough to edit it.
    pub recalled_full: usize,
    pub total: usize,
    pub tokens: usize,
}

impl BudgetResult {
    pub fn recall(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.recalled as f32 / self.total as f32
        }
    }
    /// Recall counting only symbols delivered whole.
    pub fn recall_full(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.recalled_full as f32 / self.total as f32
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

/// One seed-ranking configuration.
#[derive(Debug, Clone)]
pub struct SeedResult {
    /// Candidates each view contributes before fusion.
    pub over_fetch: usize,
    pub reranked: bool,
    pub k: usize,
    pub recalled: usize,
    pub total: usize,
}

impl SeedResult {
    pub fn recall(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.recalled as f32 / self.total as f32
        }
    }
}

/// Where a query's recall was lost.
///
/// The point of splitting misses up: "56% recall" is not actionable, but
/// "of the 44% missed, X are rankable and Y are unreachable" says exactly
/// which half of the pipeline to work on — and caps how much any amount of
/// reranking could ever buy.
#[derive(Debug, Clone, Copy, Default)]
pub struct MissTaxonomy {
    /// Gold symbol was in the packed context at the evaluated `k`.
    pub hit: usize,
    /// Gold symbol was *retrievable* — it appeared in a very deep candidate
    /// list — but ranked below `k`. This is the reranking-addressable share,
    /// and the ceiling on what any better ranker can win.
    pub rankable: usize,
    /// Gold symbol never surfaced even at the deep `k`, but the query does
    /// share vocabulary with it. Retrievable in principle; the signal is being
    /// drowned by the rest of the corpus.
    pub drowned: usize,
    /// Query and gold symbol share no indexed term at all. No amount of
    /// reranking reaches this — it needs a different signal (better
    /// embeddings, or code the query's words actually appear in).
    pub no_overlap: usize,
    /// Gold symbol is not in the index at all — the commit touched something
    /// the parser did not extract. Not a retrieval failure; a parsing ceiling.
    pub not_indexed: usize,
}

impl MissTaxonomy {
    pub fn total(&self) -> usize {
        self.hit + self.rankable + self.drowned + self.no_overlap + self.not_indexed
    }
    /// Best recall achievable at `k` if ranking were perfect — everything
    /// except what the index cannot reach at all.
    pub fn rankable_ceiling(&self) -> f32 {
        if self.total() == 0 {
            return 0.0;
        }
        (self.hit + self.rankable + self.drowned) as f32 / self.total() as f32
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
    /// Packing-policy ablation at fixed budgets. Every policy sees the *same*
    /// seed list, so a difference is the assembly policy and nothing else.
    pub packing: Vec<BudgetResult>,
    /// Seed-ranking 2x2: candidate-pool depth crossed with the reranker.
    /// Every cell runs on the same index and the same queries.
    pub seeding: Vec<SeedResult>,
    /// Where recall was lost, at the evaluated `k` and against a deep probe.
    pub misses: MissTaxonomy,
    /// The `k` [`MissTaxonomy::rankable`] was judged against.
    pub miss_k: usize,
    /// Depth of the probe used to decide "retrievable at all".
    pub probe_k: usize,
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

    /// Best recall this arm reaches without exceeding `budget` tokens/query.
    ///
    /// The decision an agent actually faces is a **context budget**, not a `k`.
    /// Iso-recall answers "what does matching you cost?", which flatters an arm
    /// that can afford to be expensive; this answers "given what I can afford,
    /// who wins?" — and an arm whose cheapest cell already blows the budget
    /// simply cannot play.
    pub fn best_under_budget(&self, arm: &str, budget: f32) -> Option<&ArmResult> {
        self.arms
            .iter()
            .filter(|a| a.name == arm && a.tokens_per_query() <= budget)
            .max_by(|a, b| {
                a.recalled
                    .cmp(&b.recalled)
                    .then_with(|| b.tokens.cmp(&a.tokens))
            })
    }

    /// Did recall ever *fall* as `k` grew?
    ///
    /// Intuitively impossible — a larger `k` returns a superset — but
    /// `HybridRetriever` over-fetches `k * over_fetch` candidates and then
    /// normalises recency and graph-proximity *relative to that candidate
    /// set*, so growing `k` changes the scores of the memories already in it.
    /// The top-1 at `k=1` need not survive into the top-3 at `k=3`. Worth
    /// reporting: a reader comparing two rows of the sweep would otherwise
    /// assume a measurement error.
    pub fn non_monotonic(&self, arm: &str) -> bool {
        let mut cells: Vec<&ArmResult> = self.arms.iter().filter(|a| a.name == arm).collect();
        cells.sort_by_key(|a| a.k);
        cells.windows(2).any(|w| w[1].recalled < w[0].recalled)
    }
}

/// What we need to know about an indexed symbol to score an arm.
struct SymbolInfo {
    name: String,
    path: String,
    text: String,
    signature: Option<String>,
    kind: SymbolKind,
}

/// Do the query and a symbol's text share any content word?
///
/// Deliberately crude — lowercase alphanumeric runs of 3+ characters, minus a
/// few words every commit subject contains. It is not trying to model the
/// index's analyzer; it only has to separate "the words are in there somewhere"
/// from "there is nothing lexical to match on", which is the distinction that
/// decides whether better scoring could ever help.
fn shares_vocabulary(query: &str, text: &str) -> bool {
    const NOISE: &[&str] = &[
        "the", "and", "for", "with", "add", "fix", "use", "new", "not", "from", "into", "when",
        "that", "this", "make", "update", "remove", "support",
    ];
    let words = |s: &str| -> Vec<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3)
            .map(str::to_lowercase)
            .filter(|w| !NOISE.contains(&w.as_str()))
            .collect()
    };
    let hay = words(text);
    words(query).iter().any(|w| hay.contains(w))
}

/// The bench's [`PackSource`]: the in-memory side tables an index run already
/// keeps, exposed in the shape the packer wants.
struct BenchSource<'a> {
    symbols: &'a HashMap<MemoryRef, SymbolInfo>,
    links: &'a HashMap<MemoryRef, Vec<MemoryRef>>,
    module_docs: &'a HashMap<String, String>,
}

impl PackSource for BenchSource<'_> {
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

/// Parse `dir` and report every symbol's current location.
///
/// Split out from [`run_codeeval`] because the git suite has to be derived
/// *from* the symbols before the run can score against it. Parsing is cheap —
/// embedding is the expensive step — so doing it twice costs little and keeps
/// the two concerns from tangling.
pub fn trace_targets(dir: &str) -> Result<Vec<crate::gitsuite::TraceTarget>> {
    let root = std::path::Path::new(dir);
    let base = path_base(root);
    let mut paths = Vec::new();
    collect_sources(root, &mut paths);
    paths.sort();

    let mut out = Vec::new();
    for p in &paths {
        let key = relative_key(p, &base);
        let (Some(lang), Ok(src)) = (language_of(p), std::fs::read_to_string(p)) else {
            continue;
        };
        let Ok(pf) = HeuristicParser.parse(&key, lang, &src) else {
            continue;
        };
        for s in pf.symbols {
            out.push(crate::gitsuite::TraceTarget {
                path: key.clone(),
                name: s.name,
                start_line: s.start_line,
                end_line: s.end_line,
            });
        }
    }
    if out.is_empty() {
        return Err(anyhow!("no symbols parsed under {dir}"));
    }
    Ok(out)
}

/// Extension → language tag, for the languages `HeuristicParser` handles.
/// Anything else is skipped rather than parsed into plausible nonsense.
fn language_of(path: &std::path::Path) -> Option<&'static str> {
    match path.extension()?.to_str()? {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "go" => Some("go"),
        "ts" | "tsx" => Some("typescript"),
        "java" => Some("java"),
        _ => None,
    }
}

/// Directories that are never the repository's own source. Indexing them would
/// drown the corpus in vendored code and inflate every arm equally, which
/// hides rather than reveals a difference.
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
];

/// Enclosing git repository of `dir`, if there is one.
///
/// Indexing is routinely pointed at a *subdirectory* (`llama-index-core`) of a
/// repo whose `.git` lives further up. Paths are therefore made relative to the
/// git root rather than to the indexed directory, because that is the only form
/// `git log -L <a>,<b>:<path>` can resolve — pointing at a subdirectory
/// otherwise fails with "not a git repository" or, worse, silently traces
/// nothing.
pub fn git_root(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    dir.canonicalize()
        .ok()?
        .ancestors()
        .find(|p| p.join(".git").exists())
        .map(|p| p.to_path_buf())
}

/// Base directory that indexed paths are expressed relative to.
fn path_base(dir: &std::path::Path) -> std::path::PathBuf {
    git_root(dir).unwrap_or_else(|| dir.to_path_buf())
}

/// Path of `file` relative to `base`, as the index and git both see it.
fn relative_key(file: &std::path::Path, base: &std::path::Path) -> String {
    file.canonicalize()
        .ok()
        .and_then(|c| c.strip_prefix(base).ok().map(|r| r.to_path_buf()))
        .unwrap_or_else(|| file.to_path_buf())
        .to_string_lossy()
        .to_string()
}

/// Recursively collect parseable source files.
fn collect_sources(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if SKIP_DIRS.contains(&name) {
                continue;
            }
            collect_sources(&p, out);
        } else if language_of(&p).is_some() {
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
    let root = std::path::Path::new(dir);
    let base = path_base(root);
    let mut paths = Vec::new();
    collect_sources(root, &mut paths);
    paths.sort();
    if paths.is_empty() {
        return Err(anyhow!(
            "no parseable source under {dir} (rust/go/typescript/java)"
        ));
    }

    let mut file_text: HashMap<String, String> = HashMap::new();
    let mut parsed: Vec<ParsedFile> = Vec::new();
    for p in &paths {
        // Index by *repo-relative* path: that is what `Source::uri` means, and
        // it is the path `git log -L` needs to trace a symbol's history.
        let key = relative_key(p, &base);
        let (Some(lang), Ok(src)) = (language_of(p), std::fs::read_to_string(p)) else {
            continue;
        };
        if let Ok(pf) = HeuristicParser.parse(&key, lang, &src) {
            file_text.insert(key, src);
            parsed.push(pf);
        }
    }

    // --- plan -> events, and keep the side tables scoring needs ---
    // The emitted `Memory` carries the code but not the signature or kind, so
    // those are looked up from the parse output the packer's degradation ladder
    // needs them.
    let mut decl: HashMap<(String, String), (Option<String>, SymbolKind)> = HashMap::new();
    let mut module_docs: HashMap<String, String> = HashMap::new();
    for f in &parsed {
        if let Some(d) = &f.module_doc {
            module_docs.insert(f.path.clone(), d.clone());
        }
        for s in &f.symbols {
            decl.insert(
                (s.path.clone(), s.name.clone()),
                (s.signature.clone(), s.kind),
            );
        }
    }

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

                // The indexer tags each memory with its file path; pick the tag
                // that names a file we actually read, rather than guessing by
                // extension.
                let path = m
                    .tags
                    .iter()
                    .find(|t| file_text.contains_key(*t))
                    .cloned()
                    .unwrap_or_default();
                let name = m.keywords.first().cloned().unwrap_or_default();
                let (signature, kind) = decl
                    .get(&(path.clone(), name.clone()))
                    .cloned()
                    .unwrap_or((None, SymbolKind::Function));
                symbols.insert(
                    MemoryRef(m.id),
                    SymbolInfo {
                        name,
                        path,
                        text: m.content.clone(),
                        signature,
                        kind,
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

    let retriever = HybridRetriever::new(vector.clone(), bm25.clone(), embedder.clone());
    let source = BenchSource {
        symbols: &symbols,
        links: &links,
        module_docs: &module_docs,
    };

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
                    "  q={:?} want={:?} got={:?}",
                    q.question,
                    q.gold.iter().map(|g| &g.name).collect::<Vec<_>>(),
                    picked.iter().map(|s| &s.name).collect::<Vec<_>>()
                );
            }
            symbol_only.total += 1;
            symbol_only.tokens += picked.iter().map(|s| est_tokens(&s.text)).sum::<usize>();
            if q.hit(&picked) {
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
            // The whole file is present, so every symbol defined in a retrieved
            // file counts as delivered — that is the arm's whole advantage.
            let in_file: Vec<&SymbolInfo> = symbols
                .values()
                .filter(|s| seen_files.contains(&s.path.as_str()))
                .collect();
            if q.hit(&in_file) {
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
            if q.hit(&exp) {
                expanded.recalled += 1;
            }
        }
        arms.extend([whole_file, symbol_only, expanded]);
    }

    // --- packing ablation at fixed budgets ---
    //
    // Seeds come from a single generous `k` so every policy chooses from the
    // same candidate pool; what differs is only how the budget is spent. The
    // policies are cumulative, so each row shows what one idea added.
    let seed_k = ks.iter().copied().max().unwrap_or(10);
    let mut packing: Vec<BudgetResult> = Vec::new();
    for budget in [2_000usize, 4_000, 8_000, 16_000] {
        let policies: [(&'static str, PackConfig); 4] = [
            ("truncate", PackConfig::naive(budget)),
            (
                "+signature",
                PackConfig {
                    degrade: true,
                    ..PackConfig::naive(budget)
                },
            ),
            (
                "+expand",
                PackConfig {
                    degrade: true,
                    expand: true,
                    max_expansions_per_seed: 3,
                    ..PackConfig::naive(budget)
                },
            ),
            (
                "+notes",
                PackConfig {
                    budget,
                    ..Default::default()
                },
            ),
        ];
        let mut rows: Vec<BudgetResult> = policies
            .iter()
            .map(|(name, _)| BudgetResult {
                policy: name,
                budget,
                recalled: 0,
                recalled_full: 0,
                total: 0,
                tokens: 0,
            })
            .collect();

        for q in suite {
            let hits = retriever
                .search(&Query {
                    text: q.question.to_string(),
                    scope: scope.clone(),
                    k: seed_k,
                    time_filter: None,
                })
                .await
                .map_err(|e| anyhow!("search: {e}"))?;
            let seeds: Vec<MemoryRef> = hits.iter().map(|h| h.memory).collect();

            for (row, (_, cfg)) in rows.iter_mut().zip(&policies) {
                let ctx = pack(&seeds, &source, *cfg);
                let got: Vec<&SymbolInfo> = ctx
                    .symbols
                    .iter()
                    .filter_map(|s| symbols.get(&s.memory))
                    .collect();
                let whole: Vec<&SymbolInfo> = ctx
                    .symbols
                    .iter()
                    .filter(|s| s.form == Form::Full)
                    .filter_map(|s| symbols.get(&s.memory))
                    .collect();
                row.total += 1;
                row.tokens += ctx.tokens_used;
                if q.hit(&got) {
                    row.recalled += 1;
                }
                if q.hit(&whole) {
                    row.recalled_full += 1;
                }
            }
        }
        packing.extend(rows);
    }

    // --- seed ranking: does a deeper pool, or reranking it, convert the
    // `rankable` share into hits? Paired on this one index. ---
    let seed_eval_k = ks.iter().copied().max().unwrap_or(20);
    let mut seeding: Vec<SeedResult> = Vec::new();
    for over_fetch in [4usize, 20] {
        for reranked in [false, true] {
            let mut r = HybridRetriever::new(vector.clone(), bm25.clone(), embedder.clone())
                .with_over_fetch(over_fetch);
            if reranked {
                r = r.with_reranker(Arc::new(LexicalReranker::new(bm25.clone())));
            }
            let mut row = SeedResult {
                over_fetch,
                reranked,
                k: seed_eval_k,
                recalled: 0,
                total: 0,
            };
            for q in suite {
                let hits = r
                    .search(&Query {
                        text: q.question.to_string(),
                        scope: scope.clone(),
                        k: seed_eval_k,
                        time_filter: None,
                    })
                    .await
                    .map_err(|e| anyhow!("search: {e}"))?;
                let got: Vec<&SymbolInfo> =
                    hits.iter().filter_map(|h| symbols.get(&h.memory)).collect();
                row.total += 1;
                if q.hit(&got) {
                    row.recalled += 1;
                }
            }
            seeding.push(row);
        }
    }

    // --- where is recall actually lost? ---
    //
    // Run each query twice: once at the reporting `k`, once at a much deeper
    // `probe_k`. A gold symbol that shows up in the deep list but not the
    // shallow one is a *ranking* failure and is what a better ranker could
    // win. One that never shows up at all is not — and the split between
    // "shares vocabulary with the query" and "shares nothing" says whether
    // that is a scoring problem or a representation problem.
    let miss_k = ks.iter().copied().max().unwrap_or(20);
    let probe_k = (miss_k * 20).max(200);
    let mut misses = MissTaxonomy::default();

    for q in suite {
        // A gold name the parser never extracted can't be retrieved by anyone.
        let indexed: Vec<&SymbolInfo> = symbols
            .values()
            .filter(|s| q.gold.iter().any(|g| g.matches(&s.path, &s.name)))
            .collect();
        if indexed.is_empty() {
            misses.not_indexed += 1;
            continue;
        }

        let deep = retriever
            .search(&Query {
                text: q.question.to_string(),
                scope: scope.clone(),
                k: probe_k,
                time_filter: None,
            })
            .await
            .map_err(|e| anyhow!("search: {e}"))?;

        let at = |n: usize| -> Vec<&SymbolInfo> {
            deep.iter()
                .take(n)
                .filter_map(|h| symbols.get(&h.memory))
                .collect()
        };

        if q.hit(&at(miss_k)) {
            misses.hit += 1;
        } else if q.hit(&at(probe_k)) {
            misses.rankable += 1;
        } else if indexed
            .iter()
            .any(|s| shares_vocabulary(&q.question, &s.text))
        {
            misses.drowned += 1;
        } else {
            misses.no_overlap += 1;
        }
    }

    Ok(CodeEvalReport {
        embedder: embedder_choice.to_string(),
        index: plan.stats,
        arms,
        packing,
        seeding,
        misses,
        miss_k,
        probe_k,
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

    // Iso-budget: the comparison an agent actually faces.
    out.push_str("\n## iso-budget — best recall affordable at a context budget\n\n");
    out.push_str(
        "| budget (tok/query) | whole-file | symbol | symbol+expand |\n|---|---|---|---|\n",
    );
    for budget in [4_000.0f32, 16_000.0, 64_000.0] {
        let cell = |arm: &str| match r.best_under_budget(arm, budget) {
            Some(c) => format!("{:.0}% (k={})", c.recall() * 100.0, c.k),
            None => "— can't fit".into(),
        };
        out.push_str(&format!(
            "| {:.0}k | {} | {} | {} |\n",
            budget / 1000.0,
            cell("whole-file"),
            cell("symbol"),
            cell("symbol+expand"),
        ));
    }

    // Seed ranking: the 2x2 the miss taxonomy motivates.
    if !r.seeding.is_empty() {
        let k = r.seeding[0].k;
        out.push_str(&format!(
            "\n## seed ranking (k={k})\n\n\
             | over-fetch | reranker | recall |\n|---|---|---|\n"
        ));
        for s in &r.seeding {
            out.push_str(&format!(
                "| {}x | {} | {:.0}% ({}/{}) |\n",
                s.over_fetch,
                if s.reranked { "lexical" } else { "off" },
                s.recall() * 100.0,
                s.recalled,
                s.total
            ));
        }
    }

    // Where recall is lost — the number that says what to work on next.
    let m = &r.misses;
    if m.total() > 0 {
        let pct = |n: usize| n as f32 * 100.0 / m.total() as f32;
        out.push_str(&format!(
            "\n## where recall is lost (k={}, probe depth {})\n\n\
             | outcome | queries | share | fixable by |\n|---|---|---|---|\n\
             | hit | {} | {:.0}% | — |\n\
             | rankable | {} | {:.0}% | **a better ranker** |\n\
             | drowned | {} | {:.0}% | scoring — terms match but rank below {} |\n\
             | no overlap | {} | {:.0}% | not ranking — needs a different signal |\n\
             | not indexed | {} | {:.0}% | the parser, not retrieval |\n\n\
             **Ceiling on reranking: {:.0}%.** Everything above that is out of \
             reach of any scoring change — the gold symbol either shares no \
             vocabulary with the task or was never extracted.\n",
            r.miss_k,
            r.probe_k,
            m.hit,
            pct(m.hit),
            m.rankable,
            pct(m.rankable),
            m.drowned,
            pct(m.drowned),
            r.probe_k,
            m.no_overlap,
            pct(m.no_overlap),
            m.not_indexed,
            pct(m.not_indexed),
            m.rankable_ceiling() * 100.0,
        ));
    }

    // Packing ablation: cumulative, so each row is what one idea added.
    if !r.packing.is_empty() {
        out.push_str(
            "\n## packing policy at a fixed budget\n\n\
             Same seeds, same budget — only the assembly differs. Rows are \
             cumulative.\n\n\
             `any` counts a signature-only inclusion; `full` demands the whole \
             body.\n\n\
             | budget | policy | recall (any) | recall (full) | tokens used |\n\
             |---|---|---|---|---|\n",
        );
        for c in &r.packing {
            out.push_str(&format!(
                "| {}k | {} | {:.0}% ({}/{}) | {:.0}% | {:.0} |\n",
                c.budget / 1000,
                c.policy,
                c.recall() * 100.0,
                c.recalled,
                c.total,
                c.recall_full() * 100.0,
                c.tokens_per_query(),
            ));
        }
    }

    let wobbly: Vec<&str> = ["whole-file", "symbol", "symbol+expand"]
        .into_iter()
        .filter(|a| r.non_monotonic(a))
        .collect();
    if !wobbly.is_empty() {
        out.push_str(&format!(
            "\n⚠ recall is non-monotonic in k for: {}. Not a measurement error — \
             `HybridRetriever` over-fetches `k * over_fetch` candidates and \
             normalises recency and graph-proximity across *that* set, so a \
             larger k re-scores the memories already in it.\n",
            wobbly.join(", ")
        ));
    }

    out.push_str(
        "\n_Tokens estimated as chars/4 — identical across arms, so ratios hold; \
         absolute counts are indicative._\n",
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

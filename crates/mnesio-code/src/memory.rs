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
    scope: Scope,
    retriever: HybridRetriever,
    symbols: HashMap<MemoryRef, SymbolRecord>,
    links: HashMap<MemoryRef, Vec<MemoryRef>>,
    module_docs: HashMap<String, String>,
    stats: IndexStats,
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
        let root = root.as_ref();
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

        let parser = parser();
        let mut parsed: Vec<ParsedFile> = Vec::new();
        let mut module_docs: HashMap<String, String> = HashMap::new();
        let mut decl: HashMap<(String, String), (Option<String>, SymbolKind)> = HashMap::new();

        for file in &files {
            // Repo-relative, because that is what `Source::uri` means and what
            // a caller can act on — an absolute path from the indexing machine
            // is meaningless to whoever reads the answer.
            let rel = file
                .strip_prefix(root)
                .unwrap_or(file)
                .to_string_lossy()
                .to_string();
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

        let plan = CodeIndexer::new(scope.clone()).plan(&parsed);

        let vector = Arc::new(VectorView::new(
            embedder.dim(),
            embedder.model_id().to_string(),
        ));
        let bm25 = Arc::new(Bm25View::new()?);
        let mut symbols: HashMap<MemoryRef, SymbolRecord> = HashMap::new();
        let mut links: HashMap<MemoryRef, Vec<MemoryRef>> = HashMap::new();

        for event in &plan.events {
            match event {
                Event::MemoryWritten(m) => {
                    let m = embed_one(m, &embedder).await?;
                    record(&mut symbols, &m, &decl, &module_docs);
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

        // See the module docs: every one of these is a measured setting.
        let retriever = HybridRetriever::new(vector, bm25.clone(), embedder)
            .with_reranker(Arc::new(LexicalReranker::for_code(bm25)));

        Ok(Self {
            scope,
            retriever,
            symbols,
            links,
            module_docs,
            stats: plan.stats,
        })
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
        let hits = self
            .retriever
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
        }
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

async fn embed_one(m: &Memory, embedder: &Arc<dyn Embedder>) -> Result<Memory, MnesioError> {
    let vectors = embedder.embed(std::slice::from_ref(&m.content)).await?;
    let mut m = m.clone();
    m.embedding = vectors.into_iter().next();
    Ok(m)
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

//! # mnesio-code
//!
//! Code memory (Phase 17): parse a codebase into **symbols** and the edges
//! between them, so an agent can retrieve the few definitions it actually needs
//! instead of whole files.
//!
//! ## Why this is a mnesio crate and not a me-too code-RAG
//!
//! Chunk-and-embed over a repo is a commodity. What isn't: mnesio can record
//! whether a retrieval *helped* (did the edit compile, did tests pass, was the
//! diff accepted) and compile that into **gated** retrieval policy — improvement
//! that cannot silently regress (Hard Rule #1). The slogan for the phase:
//! *don't retrieve code that looks relevant — retrieve the code that has been
//! proven to help.* That loop is Phase 17C; this module is the substrate.
//!
//! ## No new event types
//!
//! A **file is a [`mnesio_core::entity::Source`]** (`uri` = repo-relative path)
//! and a **symbol is a [`mnesio_core::entity::Memory`]** (`content` = the code,
//! `source`/`position` = its file and offset). Indexing therefore emits only
//! `SourceIngested` + `MemoryWritten` + `MemoryLinksUpdated`, so the code graph
//! is a materialized view rebuildable by replaying the log — Hard Rule #4 holds
//! by construction, and hybrid retrieval, scope isolation, bi-temporality and
//! crypto-shred all come for free.
//!
//! ## Layout
//!
//! - [`Symbol`] / [`SymbolKind`] — a definition extracted from a file.
//! - [`CodeEdge`] / [`EdgeKind`] — an *unresolved* relationship between symbols.
//! - [`CodeParser`] — the swappable parsing seam (Hard Rule #7).
//! - [`CodeMemory`] — **the entry point.** Index a repo, then ask it for the
//!   code a task needs, fitted to a token budget. Everything below is a stage
//!   it assembles with settings that were measured rather than chosen.
//! - [`pack`] — fit retrieval's ranked output into a token budget, which is
//!   the constraint an agent actually has.
//! - [`HeuristicParser`] — dependency-free line/brace scanning; what the tests
//!   run on. Real grammars land behind the `tree-sitter` feature.

pub mod curve;
pub mod graph;
pub mod index;
pub mod journal;
pub mod learn;
pub mod memory;
pub mod outcome;
pub mod pack;
pub mod parse;
#[cfg(feature = "tree-sitter")]
pub mod parse_ts;
pub mod persist;
pub mod report;

pub use curve::{CurvePoint, LiveCurve};
pub use graph::{CodeGraph, GraphConfig, GraphEdge, GraphNode, GraphSource, NodeEvidence};
pub use index::{CodeIndexer, EdgeStats, IndexPlan, IndexStats, CODE_TAG};
pub use journal::{JournalEntry, JournalRead, OutcomeJournal};
pub use learn::{LearnConfig, RuleProposal, SymbolLedger};
pub use memory::{CodeContext, CodeHit, CodeMemory};
pub use outcome::{AttributedSymbol, Attribution, CodeOutcome, DecisionEvidence, EditResult};
pub use pack::{pack, Form, PackConfig, PackSource, PackedContext, PackedSymbol, Reason};
pub use parse::{CodeParser, HeuristicParser, ParseError};
#[cfg(feature = "tree-sitter")]
pub use parse_ts::TreeSitterParser;
pub use persist::EmbeddingCache;

use serde::{Deserialize, Serialize};

/// What kind of definition a [`Symbol`] is.
///
/// Deliberately coarse: this drives retrieval filtering and the `kind:` tag, not
/// a type system. Anything a language calls a "top-level definition" maps onto
/// one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    /// A free function.
    Function,
    /// A method — a function attached to a type or class.
    Method,
    /// A struct, record, or other product type.
    Struct,
    /// A class. Split from [`SymbolKind::Struct`] because a Python or Java
    /// class carries methods and inheritance, and tagging it `struct` would
    /// mislead anyone filtering retrieval by kind.
    Class,
    /// An enum / sum type.
    Enum,
    /// A trait, interface, or protocol.
    Trait,
    /// A type alias.
    TypeAlias,
    /// A constant or static.
    Constant,
    /// A module, namespace, or package declaration.
    Module,
    /// A test function. Split from `Function` because tests are the strongest
    /// *usage examples* of a symbol, so retrieval often wants them explicitly.
    Test,
}

impl SymbolKind {
    /// Lowercase tag form, used as the `kind:` tag on the emitted memory so
    /// retrieval can filter without deserialising anything.
    pub fn as_tag(self) -> &'static str {
        match self {
            SymbolKind::Function => "function",
            SymbolKind::Method => "method",
            SymbolKind::Struct => "struct",
            SymbolKind::Class => "class",
            SymbolKind::Enum => "enum",
            SymbolKind::Trait => "trait",
            SymbolKind::TypeAlias => "type_alias",
            SymbolKind::Constant => "constant",
            SymbolKind::Module => "module",
            SymbolKind::Test => "test",
        }
    }
}

/// A single definition extracted from a source file.
///
/// `text` carries the symbol's own source — that is what gets retrieved and
/// packed into an agent's context, and it's why a symbol is the right unit:
/// it's self-contained, unlike an arbitrary N-line chunk that can slice a
/// function in half.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Symbol {
    /// Bare name as written (`parse_config`, not `crate::cfg::parse_config`).
    pub name: String,
    pub kind: SymbolKind,
    /// Repo-relative path of the file this was defined in.
    pub path: String,
    /// The symbol's source text.
    pub text: String,
    /// One-line signature, when the parser can isolate one — the cheapest
    /// useful form of a symbol when the full body doesn't fit the budget.
    pub signature: Option<String>,
    /// Doc comment attached to the definition, if any.
    pub doc: Option<String>,
    /// 1-based, inclusive line span in the file.
    pub start_line: u32,
    pub end_line: u32,
}

impl Symbol {
    /// Stable identity for a symbol *within an index run*: `path::name` scoped
    /// by kind.
    ///
    /// Edges are produced by the parser before any `Id` exists, so they
    /// reference symbols by this key; the indexer resolves keys to `MemoryRef`s
    /// once the memories are written.
    pub fn key(&self) -> String {
        format!("{}#{}:{}", self.path, self.kind.as_tag(), self.name)
    }
}

/// How one symbol relates to another.
///
/// These mirror the code-edge variants added to `mnesio_graph::Relation`, so a
/// parsed edge lands in the graph view unchanged and `bfs` can expand a call
/// graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// `from` invokes `to`. The backbone of context expansion: to understand a
    /// function you usually need its callees.
    Calls,
    /// `from`'s file imports/uses `to`.
    Imports,
    /// `from` implements trait/interface `to`.
    Implements,
    /// `from` mentions `to` without calling it (a type in a signature, say).
    References,
    /// `from` is a test exercising `to` — the usage-example edge.
    TestOf,
}

/// A relationship whose target is still just a **name**.
///
/// Resolution is deliberately separated from parsing. A parser sees the call
/// `parse_config(..)` and knows only the identifier; binding that to a specific
/// definition needs a symbol table (and, to be exact, type inference). See
/// [`crate::parse`] for how far we take that and where it is approximate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeEdge {
    /// [`Symbol::key`] of the source symbol.
    pub from: String,
    /// Bare name of the target, as written at the call/use site.
    pub to_name: String,
    pub kind: EdgeKind,
    /// The call site was `x.name(..)` or `T::name(..)`, not a bare `name(..)`.
    ///
    /// The parser cannot type `x`, so it cannot say *which* `name` this is —
    /// but it can see that there was a receiver, and that is enough to know
    /// the call is very unlikely to be a free function defined in some other
    /// file. Without this distinction `vec.push(x)` binds to any lone function
    /// called `push` in the repository: measured on this workspace, that gave
    /// `push` 142 inbound edges and made it the top "most depended on" symbol,
    /// which is an artefact rather than a fact about the code.
    pub via_receiver: bool,
}

/// Everything one file contributed to the index.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedFile {
    /// Repo-relative path.
    pub path: String,
    /// Language tag (`rust`, `python`, …), used as a retrieval tag.
    pub language: String,
    /// The file's own header prose — `//!` in Rust, a leading docstring in
    /// Python, a top-of-file `/** … */` elsewhere.
    ///
    /// Kept on the *file* rather than turned into a symbol of its own. A
    /// module memory competes for retrieval slots with the definitions it
    /// describes, and can never *be* the symbol a query is looking for:
    /// measured on `crates/mnesio-index/src`, emitting one dropped recall@1
    /// from 50% to 25% and left the ceiling unmoved. Instead the indexer
    /// prepends a one-line breadcrumb of this to every symbol in the file, so
    /// each definition inherits the words that describe its module.
    pub module_doc: Option<String>,
    pub symbols: Vec<Symbol>,
    /// Edges originating in this file, targets unresolved.
    pub edges: Vec<CodeEdge>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_key_is_stable_and_distinguishes_kind() {
        let f = Symbol {
            name: "load".into(),
            kind: SymbolKind::Function,
            path: "src/cfg.rs".into(),
            text: "fn load() {}".into(),
            signature: None,
            doc: None,
            start_line: 1,
            end_line: 1,
        };
        let s = Symbol {
            kind: SymbolKind::Struct,
            ..f.clone()
        };

        assert_eq!(f.key(), "src/cfg.rs#function:load");
        // Same name + path but a different kind must not collide, or edge
        // resolution would bind a call to a type of the same name.
        assert_ne!(f.key(), s.key());
        assert_eq!(f.key(), f.clone().key(), "key must be deterministic");
    }

    #[test]
    fn symbol_kind_tags_are_unique() {
        let kinds = [
            SymbolKind::Function,
            SymbolKind::Method,
            SymbolKind::Struct,
            SymbolKind::Class,
            SymbolKind::Enum,
            SymbolKind::Trait,
            SymbolKind::TypeAlias,
            SymbolKind::Constant,
            SymbolKind::Module,
            SymbolKind::Test,
        ];
        let mut tags: Vec<_> = kinds.iter().map(|k| k.as_tag()).collect();
        let before = tags.len();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(tags.len(), before, "kind tags must be distinct");
    }
}

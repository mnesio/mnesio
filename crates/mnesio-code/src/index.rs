//! Turn parsed files into log events.
//!
//! This is the step that makes code memory *mnesio* memory: a parsed repo is
//! mapped onto the existing event vocabulary rather than a side database.
//!
//! | Code concept | mnesio entity | Event |
//! |---|---|---|
//! | file | [`Source`] (`uri` = repo-relative path) | `SourceIngested` |
//! | symbol | [`Memory`] (`content` = the code) | `MemoryWritten` |
//! | call edge | `Memory::links` | `MemoryLinksUpdated` |
//!
//! Because nothing here is a new event type, the code graph is a materialized
//! view that rebuilds by replaying the log (Hard Rule #4), and it inherits
//! scope isolation, bi-temporality, provenance and crypto-shred unchanged.
//!
//! [`Source`]: mnesio_core::entity::Source
//! [`Memory`]: mnesio_core::entity::Memory

use std::collections::HashMap;

use mnesio_core::entity::{Memory, Provenance, Source};
use mnesio_core::event::Event;
use mnesio_core::types::{new_id, BiTemporal, MemoryRef, Scope, SourceRef};

use crate::{EdgeKind, ParsedFile, Symbol};

/// Tag applied to every code memory, so retrieval can separate code from prose
/// without deserialising anything.
pub const CODE_TAG: &str = "code";

/// Why an edge didn't become a link.
///
/// Tracked and reported rather than silently dropped: resolution quality is the
/// main thing that decides whether graph expansion is worth anything in 17B, so
/// it needs to be visible.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EdgeStats {
    /// Bound to exactly one symbol.
    pub resolved: usize,
    /// No symbol with that name in the indexed set — typically a call into the
    /// standard library or a third-party crate, which is expected and fine.
    pub unresolved: usize,
    /// Several symbols share the name and none is in the calling file. Dropped
    /// rather than guessed: a wrong edge actively misleads context expansion.
    pub ambiguous: usize,
}

/// What an index run produced.
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub files: usize,
    pub symbols: usize,
    pub edges: EdgeStats,
}

/// The events to append, plus what happened.
#[derive(Debug, Clone, Default)]
pub struct IndexPlan {
    /// In dependency order: sources, then memories, then link amendments.
    /// Links come last because they reference memory ids that only exist once
    /// the `MemoryWritten` events have been assigned.
    pub events: Vec<Event>,
    pub stats: IndexStats,
}

/// Maps parsed files onto log events for one repository.
///
/// A repo is a [`Scope`] — that is what keeps two indexed codebases from
/// leaking into each other's retrieval (Hard Rule #3).
#[derive(Debug, Clone)]
pub struct CodeIndexer {
    scope: Scope,
    provenance: Provenance,
}

impl CodeIndexer {
    /// Index into `scope`. Use one scope per repository.
    pub fn new(scope: Scope) -> Self {
        Self {
            scope,
            // Parsed code is machine-extracted from a file the user gave us:
            // high trust, but marked so the procedural compiler can tell it
            // apart from user-authored memories.
            provenance: Provenance {
                source: "code-index".into(),
                trust: 1.0,
            },
        }
    }

    /// Build the events for a set of parsed files.
    ///
    /// Pure — no I/O, no clock beyond `BiTemporal::now`, so the whole mapping
    /// is unit-testable without a store.
    pub fn plan(&self, files: &[ParsedFile]) -> IndexPlan {
        let mut plan = IndexPlan::default();

        // Pass 1 — assign ids. Edges reference symbols by `Symbol::key`, which
        // exists before any id does, so the table has to be complete before a
        // single link can be resolved.
        let mut by_key: HashMap<String, MemoryRef> = HashMap::new();
        let mut by_name: HashMap<&str, Vec<(&str, MemoryRef)>> = HashMap::new();
        let mut file_refs: Vec<SourceRef> = Vec::with_capacity(files.len());

        for file in files {
            file_refs.push(SourceRef(new_id()));
            for symbol in &file.symbols {
                let r = MemoryRef(new_id());
                by_key.insert(symbol.key(), r);
                by_name
                    .entry(symbol.name.as_str())
                    .or_default()
                    .push((symbol.path.as_str(), r));
            }
        }

        // Pass 2 — emit the file and symbol events.
        for (file, source_ref) in files.iter().zip(&file_refs) {
            plan.events.push(Event::SourceIngested(Source {
                id: source_ref.0,
                scope: self.scope.clone(),
                title: file.path.clone(),
                uri: Some(file.path.clone()),
                chunk_count: file.symbols.len() as u32,
                time: BiTemporal::now(),
                provenance: self.provenance.clone(),
            }));

            for (position, symbol) in file.symbols.iter().enumerate() {
                let id = by_key[&symbol.key()];
                plan.events.push(Event::MemoryWritten(self.memory(
                    id,
                    symbol,
                    file,
                    *source_ref,
                    position as u32,
                )));
            }
            plan.stats.symbols += file.symbols.len();
        }
        plan.stats.files = files.len();

        // Pass 3 — resolve edges into link amendments.
        let mut links: HashMap<MemoryRef, Vec<MemoryRef>> = HashMap::new();
        for file in files {
            for edge in &file.edges {
                // Only `Calls` today; the other kinds arrive with real grammars.
                if edge.kind != EdgeKind::Calls {
                    continue;
                }
                let Some(from) = by_key.get(&edge.from).copied() else {
                    continue;
                };
                match resolve(&by_name, &edge.to_name, &file.path) {
                    Resolution::One(to) if to != from => {
                        let entry = links.entry(from).or_default();
                        if !entry.contains(&to) {
                            entry.push(to);
                        }
                        plan.stats.edges.resolved += 1;
                    }
                    // A self-edge is not useful context; count it as resolved
                    // since the name *did* bind.
                    Resolution::One(_) => plan.stats.edges.resolved += 1,
                    Resolution::None => plan.stats.edges.unresolved += 1,
                    Resolution::Ambiguous => plan.stats.edges.ambiguous += 1,
                }
            }
        }

        // Deterministic event order regardless of HashMap iteration order —
        // replay has to produce the same view every time.
        let mut linked: Vec<_> = links.into_iter().collect();
        linked.sort_by_key(|(id, _)| id.0);
        for (id, links) in linked {
            plan.events.push(Event::MemoryLinksUpdated { id, links });
        }

        plan
    }

    /// Build the `Memory` for one symbol.
    fn memory(
        &self,
        id: MemoryRef,
        symbol: &Symbol,
        file: &ParsedFile,
        source: SourceRef,
        position: u32,
    ) -> Memory {
        Memory {
            id: id.0,
            scope: self.scope.clone(),
            // The symbol's own source is what gets retrieved and packed into an
            // agent's context — self-contained, unlike an arbitrary N-line
            // chunk that can slice a function in half.
            content: symbol.text.clone(),
            keywords: vec![symbol.name.clone()],
            tags: vec![
                CODE_TAG.to_string(),
                file.language.clone(),
                symbol.kind.as_tag().to_string(),
                file.path.clone(),
            ],
            // A-MEM's `X` field. The doc comment is the human explanation of
            // the symbol, which is exactly the context a retriever wants but
            // which shouldn't be confused with the code itself.
            context: symbol.doc.clone().unwrap_or_default(),
            embedding: None,
            links: Vec::new(),
            parent: None,
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: self.provenance.clone(),
            source: Some(source),
            position: Some(position),
        }
    }
}

enum Resolution {
    One(MemoryRef),
    None,
    Ambiguous,
}

/// Bind a bare callee name to a symbol.
///
/// Deliberately conservative, because a wrong edge is worse than a missing one:
/// it drags irrelevant code into an agent's context and costs tokens.
///
/// 1. A symbol of that name **in the calling file** wins — the common case, and
///    the one most likely to be right without type information.
/// 2. Otherwise, a **unique** match across the indexed set.
/// 3. Otherwise give up. No heuristic tie-break.
fn resolve(
    by_name: &HashMap<&str, Vec<(&str, MemoryRef)>>,
    name: &str,
    from_path: &str,
) -> Resolution {
    let Some(candidates) = by_name.get(name) else {
        return Resolution::None;
    };
    let local: Vec<_> = candidates
        .iter()
        .filter(|(path, _)| *path == from_path)
        .collect();
    if local.len() == 1 {
        return Resolution::One(local[0].1);
    }
    match candidates.len() {
        0 => Resolution::None,
        1 => Resolution::One(candidates[0].1),
        _ => Resolution::Ambiguous,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CodeParser, HeuristicParser};

    fn parse(path: &str, src: &str) -> ParsedFile {
        HeuristicParser.parse(path, "rust", src).unwrap()
    }

    fn plan_for(files: &[ParsedFile]) -> IndexPlan {
        CodeIndexer::new(Scope::global("repo")).plan(files)
    }

    fn memories(plan: &IndexPlan) -> Vec<&Memory> {
        plan.events
            .iter()
            .filter_map(|e| match e {
                Event::MemoryWritten(m) => Some(m),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn file_becomes_a_source_and_symbols_become_memories() {
        let f = parse("src/a.rs", "fn one() {}\nfn two() {}\n");
        let plan = plan_for(&[f]);

        let sources: Vec<_> = plan
            .events
            .iter()
            .filter_map(|e| match e {
                Event::SourceIngested(s) => Some(s),
                _ => None,
            })
            .collect();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].uri.as_deref(), Some("src/a.rs"));
        assert_eq!(sources[0].chunk_count, 2);

        let mems = memories(&plan);
        assert_eq!(mems.len(), 2);
        // Every symbol points back at its file and carries its offset, so the
        // existing chunk→source machinery works unchanged.
        assert!(mems
            .iter()
            .all(|m| m.source == Some(SourceRef(sources[0].id))));
        assert_eq!(mems[0].position, Some(0));
        assert_eq!(mems[1].position, Some(1));
        assert_eq!(plan.stats.symbols, 2);
    }

    #[test]
    fn memories_carry_retrieval_tags_and_doc_context() {
        let f = parse(
            "src/a.rs",
            "/// Adds numbers.\npub fn add(a: i32) -> i32 { a }\n",
        );
        let plan = plan_for(&[f]);
        let m = memories(&plan)[0];

        assert!(m.content.contains("fn add"), "content is the code itself");
        assert_eq!(m.context, "Adds numbers.", "doc goes to the A-MEM X field");
        assert_eq!(m.keywords, ["add"]);
        for want in [CODE_TAG, "rust", "function", "src/a.rs"] {
            assert!(
                m.tags.iter().any(|t| t == want),
                "missing tag {want}: {:?}",
                m.tags
            );
        }
    }

    #[test]
    fn calls_within_a_file_resolve_to_links() {
        let f = parse(
            "src/a.rs",
            "fn helper() {}\nfn caller() {\n    helper();\n}\n",
        );
        let plan = plan_for(&[f]);

        let mems = memories(&plan);
        let helper = mems.iter().find(|m| m.keywords[0] == "helper").unwrap();
        let caller = mems.iter().find(|m| m.keywords[0] == "caller").unwrap();

        let updates: Vec<_> = plan
            .events
            .iter()
            .filter_map(|e| match e {
                Event::MemoryLinksUpdated { id, links } => Some((*id, links.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(updates.len(), 1, "only the caller gains a link");
        assert_eq!(updates[0].0, MemoryRef(caller.id));
        assert_eq!(updates[0].1, vec![MemoryRef(helper.id)]);
        assert_eq!(plan.stats.edges.resolved, 1);
    }

    #[test]
    fn calls_resolve_across_files_when_unambiguous() {
        let a = parse("src/a.rs", "pub fn shared() {}\n");
        let b = parse("src/b.rs", "fn uses() {\n    shared();\n}\n");
        let plan = plan_for(&[a, b]);
        assert_eq!(
            plan.stats.edges.resolved, 1,
            "unique name binds across files"
        );
    }

    #[test]
    fn duplicate_names_across_files_are_ambiguous_not_guessed() {
        // Two `parse` definitions and a third file calling `parse`. Picking one
        // would drag the wrong function into an agent's context.
        let a = parse("src/a.rs", "pub fn parse() {}\n");
        let b = parse("src/b.rs", "pub fn parse() {}\n");
        let c = parse("src/c.rs", "fn run() {\n    parse();\n}\n");
        let plan = plan_for(&[a, b, c]);

        assert_eq!(plan.stats.edges.ambiguous, 1);
        assert_eq!(plan.stats.edges.resolved, 0);
        assert!(
            !plan
                .events
                .iter()
                .any(|e| matches!(e, Event::MemoryLinksUpdated { .. })),
            "no link should be invented"
        );
    }

    #[test]
    fn local_definition_wins_over_a_same_named_one_elsewhere() {
        let a = parse(
            "src/a.rs",
            "fn helper() {}\nfn caller() {\n    helper();\n}\n",
        );
        let b = parse("src/b.rs", "fn helper() {}\n");
        let plan = plan_for(&[a, b]);

        let mems = memories(&plan);
        let local_helper = mems
            .iter()
            .find(|m| m.keywords[0] == "helper" && m.tags.contains(&"src/a.rs".to_string()))
            .unwrap();
        let update = plan
            .events
            .iter()
            .find_map(|e| match e {
                Event::MemoryLinksUpdated { links, .. } => Some(links.clone()),
                _ => None,
            })
            .expect("caller links");
        assert_eq!(
            update,
            vec![MemoryRef(local_helper.id)],
            "same-file definition must win"
        );
    }

    #[test]
    fn unknown_callees_are_counted_not_dropped_silently() {
        // Calls into std / third-party crates never resolve. That's expected —
        // but it must show up in the stats so resolution quality is visible.
        let f = parse("src/a.rs", "fn f() {\n    some_external_thing();\n}\n");
        let plan = plan_for(&[f]);
        assert_eq!(plan.stats.edges.unresolved, 1);
        assert_eq!(plan.stats.edges.resolved, 0);
    }

    #[test]
    fn events_are_ordered_sources_then_memories_then_links() {
        // Links reference ids that only exist once the memories are written, so
        // a consumer replaying in order never sees a dangling reference.
        let f = parse(
            "src/a.rs",
            "fn helper() {}\nfn caller() {\n    helper();\n}\n",
        );
        let plan = plan_for(&[f]);

        let kind = |e: &Event| match e {
            Event::SourceIngested(_) => 0,
            Event::MemoryWritten(_) => 1,
            Event::MemoryLinksUpdated { .. } => 2,
            _ => 9,
        };
        let seq: Vec<u8> = plan.events.iter().map(kind).collect();
        let mut sorted = seq.clone();
        sorted.sort_unstable();
        assert_eq!(seq, sorted, "event order must be source → memory → links");
    }

    #[test]
    fn planning_is_deterministic_in_shape() {
        // Ids are fresh ULIDs each run, but the event *sequence* must not
        // depend on HashMap iteration order, or replay would differ per run.
        let f = || parse("src/a.rs", "fn a() {}\nfn b() {\n    a();\n}\n");
        let one = plan_for(&[f()]);
        let two = plan_for(&[f()]);

        let shape = |p: &IndexPlan| -> Vec<&'static str> {
            p.events
                .iter()
                .map(|e| match e {
                    Event::SourceIngested(_) => "source",
                    Event::MemoryWritten(_) => "memory",
                    Event::MemoryLinksUpdated { .. } => "links",
                    _ => "other",
                })
                .collect()
        };
        assert_eq!(shape(&one), shape(&two));
        assert_eq!(one.stats.edges.resolved, two.stats.edges.resolved);
    }
}

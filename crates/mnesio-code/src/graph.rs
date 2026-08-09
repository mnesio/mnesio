//! The code graph, as something you can look at.
//!
//! ## Why this exists separately from the packer
//!
//! [`crate::CodeMemory`] has carried a resolved call graph since 17A — it is
//! what 1-hop expansion walks — but only ever consumed it one seed at a time.
//! Nothing could answer "what does this repository *look* like", which is the
//! first question anyone asks of a code-memory tool and the one a competitor's
//! landing page answers with a picture.
//!
//! ## What is honest about this picture and what is not
//!
//! **Not honest by omission, if left unsaid:** the parser binds calls by bare
//! name, so it cannot always tell `foo()` from `x.foo()`. Between 23% and 46%
//! of call sites resolve to a definition on real repositories. Everything else
//! is dropped rather than guessed.
//!
//! That means **this graph is a lower bound on the real call graph**, and a
//! force-directed rendering of it will look sparser and more fragmented than
//! the code actually is. [`CodeGraph::resolution`] carries the rate so the
//! view can print it beside the picture. A graph rendered without that number
//! invites the reader to conclude their codebase is more modular than it is —
//! the drawing is equally pretty either way, which is exactly what makes the
//! omission tempting.
//!
//! ## The overlay nobody else can draw
//!
//! Communities and hub nodes are commodity graph analysis; any tool with an
//! import parser can produce them, and they describe how code is *shaped*.
//! What mnesio additionally knows is which symbols have been retrieved and
//! whether the edit that followed worked — so [`GraphNode::evidence`] colours
//! the same graph by *proven usefulness* rather than by connectivity.
//!
//! A hub with no evidence is a symbol that looks important. A well-connected
//! symbol with a poor success rate is a symbol that keeps being retrieved and
//! keeps not helping — the exact thing Phase 14 suppression exists to remove,
//! and something a structural graph cannot express at all.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use mnesio_core::types::MemoryRef;

use crate::journal::JournalEntry;
use crate::outcome::DecisionEvidence;
use crate::SymbolKind;

/// What the outcome journal knows about one symbol.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct NodeEvidence {
    pub successes: usize,
    pub failures: usize,
    /// Times this symbol was in a packed context at all, decisive or not. The
    /// denominator that separates "never tried" from "tried and unhelpful" —
    /// two states a success rate alone renders identically.
    pub retrievals: usize,
    /// `successes / (successes + failures)`, or `None` with no decisive
    /// outcomes. Never 0.0 for absent evidence: a symbol nobody has used is
    /// not a symbol that fails.
    pub success_rate: Option<f32>,
}

/// One symbol in the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: MemoryRef,
    pub name: String,
    pub path: String,
    pub kind: SymbolKind,
    /// Calls out of this symbol that resolved.
    pub out_degree: usize,
    /// Resolved calls into it. The better hub signal of the two — a function
    /// everything calls is load-bearing; a function that calls everything is
    /// usually just long.
    pub in_degree: usize,
    /// Label-propagation community id. Stable within one build, meaningless
    /// across builds — it is a grouping, not an identity.
    pub community: usize,
    pub evidence: NodeEvidence,
}

impl GraphNode {
    pub fn degree(&self) -> usize {
        self.in_degree + self.out_degree
    }
}

/// A resolved call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from: MemoryRef,
    pub to: MemoryRef,
    /// How much of this edge was read versus guessed.
    pub binding: EdgeBinding,
}

/// Whether an edge was *read* from the source or *inferred* by the resolver.
///
/// A single resolution rate says how much of the call graph is missing, but not
/// how much of what remains is trustworthy — and those are different questions.
/// A reader deciding whether to act on an edge needs the second one.
///
/// The split is exact rather than a heuristic, because it falls out of the two
/// rules in [`crate::index`]'s resolver:
///
/// 1. A definition of that name **in the calling file** wins. There is no
///    guessing: the name is right there. → [`EdgeBinding::Extracted`].
/// 2. Otherwise a **unique** match elsewhere in the repository, for bare calls
///    only. This is an inference from name uniqueness with no type information
///    behind it. → [`EdgeBinding::Inferred`].
///
/// Rule 2 can only ever bind across files — if the calling file held a single
/// candidate, rule 1 already returned, and if it held two or more the whole
/// resolution is `Ambiguous`. So "the endpoints share a file" is equivalent to
/// "rule 1 bound it", which is why this can be recomputed from the node paths
/// instead of being carried through the log (Hard Rule #4: the graph stays a
/// view, and no event shape changes to hold a derived property).
///
/// ## What the split actually measures, on real repositories
///
/// Measured with grammars: `mnesio-code` **292/4** (99% read), claw-code
/// **2466/204** (92%), tare **353/129** (**73%**).
///
/// Quote the range, not the first number. 99% says almost nothing was guessed
/// on a crate whose calls are mostly local; tare says more than a quarter of
/// its drawn edges are name-uniqueness guesses. Which one a reader is looking
/// at depends entirely on how much their code crosses files, and that is the
/// whole reason this is reported per-graph rather than asserted once.
///
/// A repository with no edges at all — every C/C++ one, see [`crate::parse_ts`]
/// — has no split to report, and a ratio over zero edges would be a nonsense
/// worth avoiding rather than a reassuring 100%.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeBinding {
    /// Callee defined in the same file as the caller — explicit in the source.
    #[default]
    Extracted,
    /// Bound to a unique same-named definition elsewhere. A guess, and the
    /// single largest source of wrong edges: the resolver cannot tell one
    /// `parse` from another beyond this.
    Inferred,
}

/// How much of the call graph the parser could actually bind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Resolution {
    /// Call sites the parser saw.
    pub seen: usize,
    /// Call sites bound to a definition in this repository.
    pub resolved: usize,
}

impl Resolution {
    pub fn rate(&self) -> Option<f32> {
        match self.seen {
            0 => None,
            n => Some(self.resolved as f32 / n as f32),
        }
    }
}

impl CodeGraph {
    /// How many drawn edges were read from the source versus inferred.
    ///
    /// Returns `(extracted, inferred)`. The second number is the one worth
    /// looking at: those edges are name-uniqueness guesses with no type
    /// information behind them, and they are where a wrong edge comes from.
    pub fn binding_split(&self) -> (usize, usize) {
        let inferred = self
            .edges
            .iter()
            .filter(|e| e.binding == EdgeBinding::Inferred)
            .count();
        (self.edges.len() - inferred, inferred)
    }
}

/// A whole repository, as a graph.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CodeGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    /// Distinct communities found.
    pub communities: usize,
    /// See the module docs: this graph is a *lower bound* on the real call
    /// graph, and this is by how much.
    pub resolution: Resolution,
    /// Nodes omitted because the graph was larger than the requested cap.
    /// Non-zero means the picture is a subgraph, and a viewer that does not
    /// say so is showing a repository that looks smaller than it is.
    pub truncated: usize,
    /// Total symbols in the index, before any cap.
    pub total_symbols: usize,
}

/// Everything the graph builder needs from an index, so it can be built and
/// tested without one.
pub trait GraphSource {
    /// Every symbol, as `(id, name, path, kind)`.
    fn symbols(&self) -> Vec<(MemoryRef, String, String, SymbolKind)>;
    /// Resolved callees of one symbol.
    fn callees(&self, of: MemoryRef) -> Vec<MemoryRef>;
    /// Call sites seen and bound during indexing.
    fn resolution(&self) -> Resolution;
}

/// Bounds on a graph request.
#[derive(Debug, Clone, Copy)]
pub struct GraphConfig {
    /// Maximum nodes to return. A browser renders a few thousand nodes before
    /// a force simulation stops being interactive, and a 30,000-node hairball
    /// is not a picture of anything (Hard Rule #6).
    pub max_nodes: usize,
    /// Drop symbols with no resolved edges in either direction.
    ///
    /// Default *off*. Isolated nodes are mostly an artefact of the parser
    /// failing to resolve a call, not of genuinely unreferenced code, so
    /// hiding them makes the graph look better by concealing the tool's own
    /// weakest measurement.
    pub connected_only: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_nodes: 2000,
            connected_only: false,
        }
    }
}

/// Label-propagation passes.
///
/// Converges fast on sparse graphs and is capped regardless, because label
/// propagation has no convergence guarantee and an uncapped loop on a hostile
/// graph is a hung request (Hard Rule #6).
const LABEL_PASSES: usize = 8;

impl CodeGraph {
    /// Build the graph, optionally colouring it with outcome evidence.
    ///
    /// `journal` may be empty — a repository nobody has recorded outcomes for
    /// still has a shape worth looking at, and every node simply carries empty
    /// [`NodeEvidence`].
    pub fn build(src: &dyn GraphSource, journal: &[JournalEntry], cfg: GraphConfig) -> Self {
        let all = src.symbols();
        let total_symbols = all.len();

        // Degrees first, over the *whole* graph, so a node's importance is not
        // an artefact of which slice survived the cap.
        let ids: HashSet<MemoryRef> = all.iter().map(|(id, ..)| *id).collect();
        // Each symbol's file, so an edge can say whether it was read or guessed
        // (see [`EdgeBinding`]). Recomputed here rather than carried through the
        // log, because it is a function of the endpoints and nothing else.
        let path_of: HashMap<MemoryRef, &str> = all
            .iter()
            .map(|(id, _, path, _)| (*id, path.as_str()))
            .collect();
        let mut out_deg: HashMap<MemoryRef, usize> = HashMap::new();
        let mut in_deg: HashMap<MemoryRef, usize> = HashMap::new();
        let mut edges: Vec<GraphEdge> = Vec::new();
        for (id, ..) in &all {
            for to in src.callees(*id) {
                // Self-calls are recursion, not structure; drawn, they add a
                // loop to every recursive function and say nothing.
                if to == *id || !ids.contains(&to) {
                    continue;
                }
                let binding = match (path_of.get(id), path_of.get(&to)) {
                    (Some(a), Some(b)) if a == b => EdgeBinding::Extracted,
                    _ => EdgeBinding::Inferred,
                };
                edges.push(GraphEdge {
                    from: *id,
                    to,
                    binding,
                });
                *out_deg.entry(*id).or_default() += 1;
                *in_deg.entry(to).or_default() += 1;
            }
        }

        // Keep the most connected nodes when capping. Truncating by insertion
        // order would drop hubs at random and produce a graph whose shape is
        // an accident of directory traversal.
        let mut kept: Vec<_> = all;
        if cfg.connected_only {
            kept.retain(|(id, ..)| out_deg.contains_key(id) || in_deg.contains_key(id));
        }
        kept.sort_by(|a, b| {
            let da = out_deg.get(&a.0).unwrap_or(&0) + in_deg.get(&a.0).unwrap_or(&0);
            let db = out_deg.get(&b.0).unwrap_or(&0) + in_deg.get(&b.0).unwrap_or(&0);
            // Ties broken by id so two builds of one repository agree.
            db.cmp(&da).then_with(|| a.0.cmp(&b.0))
        });
        let truncated = kept.len().saturating_sub(cfg.max_nodes);
        kept.truncate(cfg.max_nodes);

        let survivors: HashSet<MemoryRef> = kept.iter().map(|(id, ..)| *id).collect();
        edges.retain(|e| survivors.contains(&e.from) && survivors.contains(&e.to));

        let community = label_propagation(&kept, &edges);
        let evidence = fold_evidence(journal);

        let nodes: Vec<GraphNode> = kept
            .into_iter()
            .map(|(id, name, path, kind)| GraphNode {
                out_degree: *out_deg.get(&id).unwrap_or(&0),
                in_degree: *in_deg.get(&id).unwrap_or(&0),
                community: *community.get(&id).unwrap_or(&0),
                evidence: evidence.get(&id).copied().unwrap_or_default(),
                id,
                name,
                path,
                kind,
            })
            .collect();

        let communities = nodes
            .iter()
            .map(|n| n.community)
            .collect::<HashSet<_>>()
            .len();

        Self {
            nodes,
            edges,
            communities,
            resolution: src.resolution(),
            truncated,
            total_symbols,
        }
    }

    /// The most-connected symbols, most first.
    ///
    /// By in-degree, not total: a function everything calls is load-bearing,
    /// while a function that calls everything is usually just long.
    pub fn hubs(&self, n: usize) -> Vec<&GraphNode> {
        let mut v: Vec<&GraphNode> = self.nodes.iter().collect();
        v.sort_by(|a, b| {
            b.in_degree
                .cmp(&a.in_degree)
                .then_with(|| b.out_degree.cmp(&a.out_degree))
                .then_with(|| a.id.cmp(&b.id))
        });
        v.into_iter().take(n).collect()
    }

    /// Symbols that keep being retrieved and keep not helping.
    ///
    /// The view a purely structural graph cannot produce, and the input Phase
    /// 14 suppression works from. Requires `min_decisive` decisive outcomes:
    /// one bad result is not a pattern, and acting on it is how a learning
    /// loop overfits its first unlucky trial.
    pub fn unhelpful(&self, min_decisive: usize, max_rate: f32) -> Vec<&GraphNode> {
        let mut v: Vec<&GraphNode> = self
            .nodes
            .iter()
            .filter(|n| {
                let e = n.evidence;
                e.successes + e.failures >= min_decisive
                    && e.success_rate.map(|r| r <= max_rate).unwrap_or(false)
            })
            .collect();
        v.sort_by(|a, b| {
            a.evidence
                .success_rate
                .partial_cmp(&b.evidence.success_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        v
    }
}

/// Fold the journal into per-symbol evidence.
fn fold_evidence(journal: &[JournalEntry]) -> HashMap<MemoryRef, NodeEvidence> {
    let mut acc: HashMap<MemoryRef, DecisionEvidence> = HashMap::new();
    let mut retrievals: HashMap<MemoryRef, usize> = HashMap::new();
    for e in journal {
        for s in &e.outcome.symbols {
            acc.entry(s.memory).or_default().record(e.outcome.result);
            *retrievals.entry(s.memory).or_default() += 1;
        }
    }
    acc.into_iter()
        .map(|(m, d)| {
            (
                m,
                NodeEvidence {
                    successes: d.successes,
                    failures: d.failures,
                    retrievals: *retrievals.get(&m).unwrap_or(&0),
                    success_rate: d.success_rate(),
                },
            )
        })
        .collect()
}

/// Group tightly-linked symbols, by label propagation over the undirected
/// projection of the call graph.
///
/// Chosen over modularity optimisation because it needs no dependency, runs in
/// roughly linear time per pass, and is good enough for a picture. It is *not*
/// a claim about module boundaries — a community here is "these call each
/// other a lot", which usually but not always coincides with a module.
///
/// Determinism matters more than quality: two builds of one repository must
/// produce the same colouring, or a user watching the graph will read random
/// recolouring as their code having changed. So nodes are visited in a fixed
/// order and ties break toward the lowest community id.
fn label_propagation(
    nodes: &[(MemoryRef, String, String, SymbolKind)],
    edges: &[GraphEdge],
) -> HashMap<MemoryRef, usize> {
    let mut adj: HashMap<MemoryRef, Vec<MemoryRef>> = HashMap::new();
    for e in edges {
        adj.entry(e.from).or_default().push(e.to);
        adj.entry(e.to).or_default().push(e.from);
    }

    // Seed each node with its own label, assigned in sorted-id order so the
    // numbering does not depend on hash iteration.
    let mut order: Vec<MemoryRef> = nodes.iter().map(|(id, ..)| *id).collect();
    order.sort();
    let mut label: HashMap<MemoryRef, usize> =
        order.iter().enumerate().map(|(i, id)| (*id, i)).collect();

    for _ in 0..LABEL_PASSES {
        let mut changed = false;
        for id in &order {
            let Some(neighbours) = adj.get(id) else {
                continue;
            };
            let mut tally: BTreeMap<usize, usize> = BTreeMap::new();
            for n in neighbours {
                if let Some(l) = label.get(n) {
                    *tally.entry(*l).or_default() += 1;
                }
            }
            // BTreeMap iterates ascending, and `>` keeps the first maximum, so
            // ties resolve to the lowest label — deterministic by construction.
            let best = tally
                .iter()
                .fold(None::<(usize, usize)>, |acc, (l, c)| match acc {
                    Some((_, bc)) if bc >= *c => acc,
                    _ => Some((*l, *c)),
                });
            if let Some((l, _)) = best {
                if label.get(id) != Some(&l) {
                    label.insert(*id, l);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Renumber densely from 0 so a viewer can index a palette directly.
    let mut seen: BTreeMap<usize, usize> = BTreeMap::new();
    for id in &order {
        let raw = label[id];
        let next = seen.len();
        seen.entry(raw).or_insert(next);
    }
    order.iter().map(|id| (*id, seen[&label[id]])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::{CodeOutcome, EditResult};
    use crate::pack::{Form, PackedContext, PackedSymbol, Reason};
    use mnesio_core::types::new_id;

    /// A hand-built graph, so the builder is tested without an index.
    #[derive(Default)]
    struct Fake {
        syms: Vec<(MemoryRef, String, String, SymbolKind)>,
        links: HashMap<MemoryRef, Vec<MemoryRef>>,
        res: Resolution,
    }
    impl Fake {
        fn sym(&mut self, name: &str, path: &str) -> MemoryRef {
            let m = MemoryRef(new_id());
            self.syms
                .push((m, name.into(), path.into(), SymbolKind::Function));
            m
        }
        fn calls(&mut self, from: MemoryRef, to: MemoryRef) {
            self.links.entry(from).or_default().push(to);
        }
    }
    impl GraphSource for Fake {
        fn symbols(&self) -> Vec<(MemoryRef, String, String, SymbolKind)> {
            self.syms.clone()
        }
        fn callees(&self, of: MemoryRef) -> Vec<MemoryRef> {
            self.links.get(&of).cloned().unwrap_or_default()
        }
        fn resolution(&self) -> Resolution {
            self.res
        }
    }

    #[test]
    fn a_same_file_edge_is_extracted_and_a_cross_file_edge_is_inferred() {
        // The distinction a reader needs but a single resolution rate cannot
        // give: how much of what *is* drawn was read rather than guessed.
        let mut f = Fake::default();
        let caller = f.sym("caller", "src/a.rs");
        let local = f.sym("helper", "src/a.rs");
        let far = f.sym("format", "src/b.rs");
        f.calls(caller, local);
        f.calls(caller, far);

        let g = CodeGraph::build(&f, &[], GraphConfig::default());
        let find = |to: MemoryRef| g.edges.iter().find(|e| e.to == to).unwrap().binding;

        assert_eq!(find(local), EdgeBinding::Extracted, "same file — read");
        assert_eq!(find(far), EdgeBinding::Inferred, "another file — guessed");
        assert_eq!(g.binding_split(), (1, 1));
    }

    #[test]
    fn the_binding_split_matches_the_resolver_rule_it_claims_to_mirror() {
        // The doc on `EdgeBinding` claims same-file is *equivalent* to rule 1,
        // not merely correlated with it. That holds only because rule 2 can
        // never bind within a file: with one same-name candidate in the file
        // rule 1 already returned, and with two or more the whole resolution is
        // Ambiguous. Pinned end-to-end through the real indexer, so the claim
        // fails here rather than in a doc comment if the resolver changes.
        use crate::{CodeIndexer, CodeParser, HeuristicParser};
        use mnesio_core::types::Scope;

        let same = HeuristicParser
            .parse(
                "src/a.rs",
                "rust",
                "fn helper() {}\nfn caller() { helper(); }\n",
            )
            .unwrap();
        let plan = CodeIndexer::new(Scope::global("t")).plan(&[same]);
        assert_eq!(
            plan.stats.edges.resolved, 1,
            "rule 1 must have bound the local call"
        );

        let a = HeuristicParser
            .parse("src/a.rs", "rust", "fn caller() { helper(); }\n")
            .unwrap();
        let b = HeuristicParser
            .parse("src/b.rs", "rust", "fn helper() {}\n")
            .unwrap();
        let plan = CodeIndexer::new(Scope::global("t")).plan(&[a, b]);
        assert_eq!(
            plan.stats.edges.resolved, 1,
            "rule 2 must have bound the cross-file call"
        );
    }

    fn entry(syms: &[MemoryRef], result: EditResult) -> JournalEntry {
        let ctx = PackedContext {
            symbols: syms
                .iter()
                .map(|m| PackedSymbol {
                    memory: *m,
                    form: Form::Full,
                    tokens: 1,
                    reason: Reason::Seed(0),
                })
                .collect(),
            tokens_used: syms.len(),
            ..Default::default()
        };
        JournalEntry {
            observed_ms: 0,
            outcome: CodeOutcome::from_context("t", "r", &ctx, result),
        }
    }

    #[test]
    fn an_empty_repository_is_an_empty_graph() {
        let g = CodeGraph::build(&Fake::default(), &[], GraphConfig::default());
        assert!(g.nodes.is_empty());
        assert_eq!(g.communities, 0);
        assert_eq!(g.resolution.rate(), None, "0/0 is not 0%");
    }

    #[test]
    fn degrees_count_both_directions() {
        let mut f = Fake::default();
        let a = f.sym("a", "s.rs");
        let b = f.sym("b", "s.rs");
        let c = f.sym("c", "s.rs");
        f.calls(a, c);
        f.calls(b, c);
        let g = CodeGraph::build(&f, &[], GraphConfig::default());
        let node = |m: MemoryRef| g.nodes.iter().find(|n| n.id == m).unwrap();
        assert_eq!(node(c).in_degree, 2);
        assert_eq!(node(c).out_degree, 0);
        assert_eq!(node(a).out_degree, 1);
    }

    #[test]
    fn the_hub_is_ranked_by_who_calls_it() {
        // A function everything calls is load-bearing. A function that calls
        // everything is usually just long, and ranking by total degree would
        // put the long one on top.
        let mut f = Fake::default();
        let hub = f.sym("hub", "s.rs");
        let sprawler = f.sym("sprawler", "s.rs");
        let leaves: Vec<_> = (0..5).map(|i| f.sym(&format!("l{i}"), "s.rs")).collect();
        for l in &leaves {
            f.calls(*l, hub);
            f.calls(sprawler, *l);
        }
        let g = CodeGraph::build(&f, &[], GraphConfig::default());
        assert_eq!(g.hubs(1)[0].id, hub);
    }

    #[test]
    fn a_self_call_is_not_an_edge() {
        // Recursion is not structure; drawn, it adds a loop to every recursive
        // function and communicates nothing.
        let mut f = Fake::default();
        let a = f.sym("recurse", "s.rs");
        f.calls(a, a);
        assert!(CodeGraph::build(&f, &[], GraphConfig::default())
            .edges
            .is_empty());
    }

    #[test]
    fn two_clusters_get_two_communities() {
        let mut f = Fake::default();
        let cluster = |f: &mut Fake, tag: &str| {
            let xs: Vec<_> = (0..4)
                .map(|i| f.sym(&format!("{tag}{i}"), "s.rs"))
                .collect();
            for i in 0..xs.len() {
                for j in 0..xs.len() {
                    if i != j {
                        f.calls(xs[i], xs[j]);
                    }
                }
            }
            xs
        };
        let a = cluster(&mut f, "a");
        let b = cluster(&mut f, "b");
        let g = CodeGraph::build(&f, &[], GraphConfig::default());
        let comm = |m: MemoryRef| g.nodes.iter().find(|n| n.id == m).unwrap().community;
        assert_eq!(comm(a[0]), comm(a[1]), "a clique is one community");
        assert_ne!(comm(a[0]), comm(b[0]), "disconnected cliques are not");
    }

    #[test]
    fn the_colouring_is_deterministic_across_builds() {
        // A user watching the graph must not see it recolour at random and
        // read that as their code having changed.
        let mut f = Fake::default();
        let xs: Vec<_> = (0..8).map(|i| f.sym(&format!("s{i}"), "s.rs")).collect();
        for w in xs.windows(2) {
            f.calls(w[0], w[1]);
        }
        let one = CodeGraph::build(&f, &[], GraphConfig::default());
        let two = CodeGraph::build(&f, &[], GraphConfig::default());
        assert_eq!(one, two);
    }

    #[test]
    fn truncation_keeps_the_hubs_and_reports_the_loss() {
        // Dropping by traversal order would produce a graph whose shape is an
        // accident of the directory walk.
        let mut f = Fake::default();
        let hub = f.sym("hub", "s.rs");
        for i in 0..20 {
            let l = f.sym(&format!("l{i}"), "s.rs");
            f.calls(l, hub);
        }
        let g = CodeGraph::build(
            &f,
            &[],
            GraphConfig {
                max_nodes: 5,
                ..Default::default()
            },
        );
        assert_eq!(g.nodes.len(), 5);
        assert_eq!(g.truncated, 16, "the omission must be reported");
        assert_eq!(g.total_symbols, 21);
        assert!(g.nodes.iter().any(|n| n.id == hub), "the hub must survive");
    }

    #[test]
    fn isolated_symbols_are_kept_by_default() {
        // They are mostly unresolved calls, not genuinely dead code. Hiding
        // them would make the graph look better by concealing the parser's
        // weakest measurement.
        let mut f = Fake::default();
        f.sym("orphan", "s.rs");
        assert_eq!(
            CodeGraph::build(&f, &[], GraphConfig::default())
                .nodes
                .len(),
            1
        );
        let g = CodeGraph::build(
            &f,
            &[],
            GraphConfig {
                connected_only: true,
                ..Default::default()
            },
        );
        assert!(g.nodes.is_empty(), "but they can be asked to be hidden");
    }

    #[test]
    fn outcome_evidence_colours_the_nodes() {
        let mut f = Fake::default();
        let a = f.sym("a", "s.rs");
        let j = vec![
            entry(&[a], EditResult::Passed),
            entry(&[a], EditResult::BuildFailed),
            entry(&[a], EditResult::TestsFailed),
        ];
        let g = CodeGraph::build(&f, &j, GraphConfig::default());
        let e = g.nodes[0].evidence;
        assert_eq!(e.successes, 1);
        assert_eq!(e.failures, 1);
        assert_eq!(
            e.retrievals, 3,
            "ambiguous outcomes still count as retrievals"
        );
        assert_eq!(e.success_rate, Some(0.5));
    }

    #[test]
    fn a_symbol_nobody_has_used_is_not_a_symbol_that_fails() {
        // 0.0 and "no evidence" render identically on a colour scale and mean
        // opposite things.
        let mut f = Fake::default();
        f.sym("untouched", "s.rs");
        let g = CodeGraph::build(&f, &[], GraphConfig::default());
        assert_eq!(g.nodes[0].evidence.success_rate, None);
        assert_eq!(g.nodes[0].evidence.retrievals, 0);
    }

    #[test]
    fn one_bad_outcome_does_not_make_a_symbol_unhelpful() {
        // The overfitting guard: acting on a single unlucky trial is how a
        // learning loop teaches itself a superstition.
        let mut f = Fake::default();
        let a = f.sym("a", "s.rs");
        let g = CodeGraph::build(
            &f,
            &[entry(&[a], EditResult::BuildFailed)],
            GraphConfig::default(),
        );
        assert!(g.unhelpful(5, 0.2).is_empty());

        let many: Vec<_> = (0..6)
            .map(|_| entry(&[a], EditResult::BuildFailed))
            .collect();
        let g = CodeGraph::build(&f, &many, GraphConfig::default());
        assert_eq!(g.unhelpful(5, 0.2).len(), 1, "a pattern does");
    }

    #[test]
    fn the_resolution_rate_travels_with_the_graph() {
        // Without it a viewer concludes their codebase is more modular than it
        // is, and the drawing looks equally good either way.
        let f = Fake {
            res: Resolution {
                seen: 100,
                resolved: 31,
            },
            ..Default::default()
        };
        let g = CodeGraph::build(&f, &[], GraphConfig::default());
        assert_eq!(g.resolution.rate(), Some(0.31));
        assert!(serde_json::to_string(&g)
            .unwrap()
            .contains("\"resolved\":31"));
    }

    #[test]
    fn an_edge_to_an_unknown_symbol_is_dropped() {
        // Third-party calls resolve to nothing in this repository; drawing
        // them would create phantom nodes with no definition behind them.
        let mut f = Fake::default();
        let a = f.sym("a", "s.rs");
        f.calls(a, MemoryRef(new_id()));
        assert!(CodeGraph::build(&f, &[], GraphConfig::default())
            .edges
            .is_empty());
    }
}

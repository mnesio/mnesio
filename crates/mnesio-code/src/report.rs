//! Artifacts you can open — the distribution surface.
//!
//! ## Why files rather than a server
//!
//! mnesio's code graph has been reachable only by starting `mnesio` and opening
//! `/code-graph`. That is a fine developer loop and a poor way to *show*
//! anything: a competitor's whole distribution advantage is that one command
//! drops files you can double-click, mail to a colleague, or attach to a
//! review. No port, no process, no "is it still running".
//!
//! So [`render_html`] inlines the graph as JSON into a single self-contained
//! page — no fetch, no CDN, no server — and [`render_markdown`] writes the same
//! findings as text a human reads in a terminal or a pull request.
//!
//! ## What the report says that a structural one cannot
//!
//! Any tool with a parser can list hub symbols. The section that is ours is
//! **evidence**: which symbols were actually retrieved, and whether the edit
//! that followed worked. On a fresh repository that section is empty, and it
//! says so plainly rather than being hidden — "no outcomes recorded yet" is the
//! honest first state of a memory that learns, and hiding it would imply the
//! tool knows something it does not.
//!
//! ## The number that must travel with the picture
//!
//! Every artifact carries the call-graph resolution rate. A force-directed
//! rendering of a partially-resolved graph looks sparser and more modular than
//! the code really is, and it looks *equally good* either way — which is
//! exactly what makes omitting the caveat tempting. See [`crate::graph`].

use crate::graph::CodeGraph;
use crate::IndexStats;

/// Stated in the artifact itself, so the picture cannot travel without it.
pub const RESOLUTION_CAVEAT: &str =
    "This map is a LOWER BOUND on the real call graph. The parser binds calls by \
     name, so it cannot always distinguish `foo()` from `x.foo()`; unresolved and \
     ambiguous call sites are dropped rather than guessed. The rendering is \
     therefore sparser and more fragmented than the code actually is.";

/// The viewer, with a `{{DATA}}` placeholder for the inlined graph.
const TEMPLATE: &str = include_str!("../viewer/standalone.html");

/// A self-contained page: open it with a browser, nothing else required.
pub fn render_html(graph: &CodeGraph, repo: &str, stats: &IndexStats) -> String {
    let payload = serde_json::json!({
        "repo": repo,
        "files": stats.files,
        // The served payload carries this from the API; a standalone file has
        // no server to get it from, and without it the page rendered the word
        // "undefined" where the honesty caveat belongs.
        "resolution_caveat": RESOLUTION_CAVEAT,
        "graph": graph,
        "hubs": graph.hubs(12).iter().map(|n| serde_json::json!({
            "name": n.name, "path": n.path,
            "in_degree": n.in_degree, "out_degree": n.out_degree,
            "success_rate": n.evidence.success_rate,
            "retrievals": n.evidence.retrievals,
        })).collect::<Vec<_>>(),
    });
    // `</script>` inside JSON would close the tag early and break the page.
    let json = serde_json::to_string(&payload)
        .unwrap_or_else(|_| "{}".into())
        .replace("</script>", "<\\/script>");
    TEMPLATE.replace("{{DATA}}", &json)
}

/// The same findings as text.
pub fn render_markdown(graph: &CodeGraph, repo: &str, stats: &IndexStats) -> String {
    let mut out = format!(
        "# mnesio — code map for `{repo}`\n\n\
         {} symbols across {} files · {} resolved calls · {} communities\n\n",
        graph.total_symbols,
        stats.files,
        graph.edges.len(),
        graph.communities,
    );

    // First, not last. A reader who stops after one section should still have
    // been told the graph is incomplete.
    match graph.resolution.rate() {
        Some(r) => out.push_str(&format!(
            "> **This graph is a lower bound.** {:.0}% of call sites \
             ({} of {}) bound to a definition in this repository. The parser \
             matches calls by name, so it cannot always tell `foo()` from \
             `x.foo()`; unresolved and ambiguous sites are dropped rather than \
             guessed. The real call graph is denser than what follows.\n\n",
            r * 100.0,
            graph.resolution.resolved,
            graph.resolution.seen,
        )),
        None => out.push_str(
            "> No call sites were seen, so there is no resolution rate to \
             report and the graph has no edges.\n\n",
        ),
    }

    if graph.truncated > 0 {
        out.push_str(&format!(
            "> Showing {} of {} symbols — the most-connected were kept. This \
             is a subgraph.\n\n",
            graph.nodes.len(),
            graph.total_symbols
        ));
    }

    out.push_str("## Most depended on\n\n");
    let hubs = graph.hubs(12);
    if hubs.is_empty() {
        out.push_str("_No resolved edges, so nothing to rank._\n\n");
    } else {
        out.push_str(
            "Ranked by inbound calls: a function everything calls is \
                      load-bearing, while one that calls everything is usually \
                      just long. Change these carefully.\n\n\
                      | symbol | callers | callees | file |\n|---|---|---|---|\n",
        );
        for n in hubs {
            out.push_str(&format!(
                "| `{}` | {} | {} | `{}` |\n",
                n.name, n.in_degree, n.out_degree, n.path
            ));
        }
        out.push('\n');
    }

    out.push_str("## Proven useful\n\n");
    let with_evidence: Vec<_> = graph
        .nodes
        .iter()
        .filter(|n| n.evidence.retrievals > 0)
        .collect();
    if with_evidence.is_empty() {
        out.push_str(
            "_No outcomes recorded yet._\n\n\
             This is the section no other code-memory tool can fill. Every such \
             tool ranks code by how relevant it *looks*; mnesio also records \
             whether the edit that followed a retrieval actually worked, and \
             colours the map by that instead of by connectivity.\n\n\
             It fills in as you work: install the MCP server, and each \
             `mnesio_code_outcome` call adds evidence. Until then it is honestly \
             empty rather than quietly absent.\n\n",
        );
    } else {
        out.push_str(&format!(
            "{} symbols have been retrieved at least once.\n\n\
             | symbol | retrieved | helped | file |\n|---|---|---|---|\n",
            with_evidence.len()
        ));
        let mut v = with_evidence;
        v.sort_by_key(|n| std::cmp::Reverse(n.evidence.retrievals));
        for n in v.iter().take(12) {
            let rate = match n.evidence.success_rate {
                Some(r) => format!("{:.0}%", r * 100.0),
                None => "—".into(),
            };
            out.push_str(&format!(
                "| `{}` | {}× | {} | `{}` |\n",
                n.name, n.evidence.retrievals, rate, n.path
            ));
        }
        out.push('\n');

        // The view a purely structural map cannot produce.
        let cfg = crate::learn::LearnConfig::default();
        let bad = graph.unhelpful(cfg.min_decisive, cfg.max_success_rate);
        if !bad.is_empty() {
            out.push_str(&format!(
                "### Retrieved repeatedly, rarely helps\n\n\
                 {} symbol(s) cleared the evidence threshold ({} decisive \
                 outcomes) with a success rate at or below {:.0}%. These are \
                 what suppression rules are learned from — and any such rule is \
                 re-checked against a canary set before it takes effect.\n\n",
                bad.len(),
                cfg.min_decisive,
                cfg.max_success_rate * 100.0
            ));
            for n in bad.iter().take(8) {
                out.push_str(&format!(
                    "- `{}` in `{}` — {} retrievals, {}\n",
                    n.name,
                    n.path,
                    n.evidence.retrievals,
                    match n.evidence.success_rate {
                        Some(r) => format!("{:.0}% helped", r * 100.0),
                        None => "no decisive outcomes".into(),
                    }
                ));
            }
            out.push('\n');
        }
    }

    out.push_str(
        "---\n\n\
         Generated by `mnesio-code`. The graph is a materialised view of an \
         append-only log, so it rebuilds by replay; nothing here is a second \
         source of truth.\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphConfig, GraphSource, Resolution};
    use crate::SymbolKind;
    use mnesio_core::types::{new_id, MemoryRef};

    #[derive(Default)]
    struct Fake {
        syms: Vec<(MemoryRef, String, String, SymbolKind)>,
        res: Resolution,
    }
    impl GraphSource for Fake {
        fn symbols(&self) -> Vec<(MemoryRef, String, String, SymbolKind)> {
            self.syms.clone()
        }
        fn callees(&self, _: MemoryRef) -> Vec<MemoryRef> {
            Vec::new()
        }
        fn resolution(&self) -> Resolution {
            self.res
        }
    }

    fn graph_of(n: usize, res: Resolution) -> CodeGraph {
        let mut f = Fake {
            res,
            ..Default::default()
        };
        for i in 0..n {
            f.syms.push((
                MemoryRef(new_id()),
                format!("sym{i}"),
                "src/a.rs".into(),
                SymbolKind::Function,
            ));
        }
        CodeGraph::build(&f, &[], GraphConfig::default())
    }

    fn stats(files: usize) -> IndexStats {
        IndexStats {
            files,
            ..Default::default()
        }
    }

    #[test]
    fn the_resolution_caveat_comes_before_the_findings() {
        // A reader who stops after the first screen must still have been told
        // the graph is incomplete. Putting this at the bottom would make the
        // omission functionally identical to not saying it.
        let md = render_markdown(
            &graph_of(
                3,
                Resolution {
                    seen: 100,
                    resolved: 25,
                },
            ),
            "demo",
            &stats(2),
        );
        let caveat = md.find("lower bound").expect("caveat must be present");
        let hubs = md.find("## Most depended on").unwrap();
        assert!(caveat < hubs, "caveat must precede the findings");
        assert!(md.contains("25%"), "the actual rate must be stated");
    }

    #[test]
    fn an_empty_evidence_section_says_so_rather_than_vanishing() {
        // A memory that has learned nothing yet should look like one. Dropping
        // the section would imply the tool knows something it does not.
        let md = render_markdown(&graph_of(3, Resolution::default()), "demo", &stats(1));
        assert!(md.contains("## Proven useful"));
        assert!(md.contains("No outcomes recorded yet"));
    }

    #[test]
    fn no_call_sites_is_reported_as_such_not_as_zero_percent() {
        // 0/0 is not 0%. Printing "0% resolved" would read as a broken parser
        // rather than a file with no calls in it.
        let md = render_markdown(&graph_of(1, Resolution::default()), "demo", &stats(1));
        assert!(md.contains("no resolution rate"), "got: {md}");
        assert!(!md.contains("0% of call sites"));
    }

    #[test]
    fn a_truncated_graph_admits_it() {
        let mut g = graph_of(3, Resolution::default());
        g.truncated = 900;
        g.total_symbols = 903;
        let md = render_markdown(&g, "demo", &stats(1));
        assert!(md.contains("subgraph"), "got: {md}");
    }

    #[test]
    fn the_html_is_self_contained_and_carries_its_data() {
        // The whole point of the artifact: no server, no network.
        let html = render_html(&graph_of(4, Resolution::default()), "demo", &stats(1));
        assert!(!html.contains("{{DATA}}"), "the placeholder must be filled");
        assert!(!html.contains("fetch("), "must not call out to a server");
        assert!(!html.contains("http://"), "must not reference a host");
        assert!(html.contains("\"repo\":\"demo\""));
        // Regression: the page rendered "undefined" where the caveat belongs,
        // because the field was supplied by the API and never by the file.
        assert!(
            html.contains("LOWER BOUND"),
            "the caveat must travel with the picture"
        );
    }

    #[test]
    fn a_script_tag_in_the_data_cannot_break_out_of_the_page() {
        // A symbol named after a closing tag would otherwise end the script
        // element early and blank the page.
        let mut f = Fake::default();
        f.syms.push((
            MemoryRef(new_id()),
            "</script><h1>x".into(),
            "src/a.rs".into(),
            SymbolKind::Function,
        ));
        let g = CodeGraph::build(&f, &[], GraphConfig::default());
        let html = render_html(&g, "demo", &stats(1));
        assert!(
            !html.contains("</script><h1>"),
            "the closing tag must be escaped"
        );
    }
}

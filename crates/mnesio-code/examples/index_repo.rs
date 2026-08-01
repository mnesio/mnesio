//! Index a directory of source files and report what the plan contains.
//!
//! ```bash
//! cargo run -p mnesio-code --example index_repo -- crates/mnesio-index/src
//! ```
use mnesio_code::{CodeIndexer, CodeParser, HeuristicParser, ParsedFile};
use mnesio_core::types::Scope;

fn collect(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

fn main() {
    let root = std::env::args().nth(1).expect("usage: index_repo <dir>");
    let mut paths = Vec::new();
    collect(std::path::Path::new(&root), &mut paths);
    paths.sort();

    let parsed: Vec<ParsedFile> = paths
        .iter()
        .filter_map(|p| {
            let src = std::fs::read_to_string(p).ok()?;
            HeuristicParser
                .parse(&p.to_string_lossy(), "rust", &src)
                .ok()
        })
        .collect();

    let plan = CodeIndexer::new(Scope::global("repo")).plan(&parsed);
    let s = &plan.stats;
    let total = s.edges.resolved + s.edges.unresolved + s.edges.ambiguous;
    let pct = |n: usize| {
        if total == 0 {
            0.0
        } else {
            n as f64 * 100.0 / total as f64
        }
    };

    println!("indexed {} files → {} symbols", s.files, s.symbols);
    println!("{} events to append\n", plan.events.len());
    println!("call edges: {total}");
    println!(
        "  resolved   {:>6}  ({:.1}%)",
        s.edges.resolved,
        pct(s.edges.resolved)
    );
    println!(
        "  unresolved {:>6}  ({:.1}%)  — std/third-party, expected",
        s.edges.unresolved,
        pct(s.edges.unresolved)
    );
    println!(
        "  ambiguous  {:>6}  ({:.1}%)  — dropped, not guessed",
        s.edges.ambiguous,
        pct(s.edges.ambiguous)
    );
}

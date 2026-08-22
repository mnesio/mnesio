//! `mnesio-code <dir>` — map a repository into files you can open.
//!
//! One command, three artifacts, no server:
//!
//! - `mnesio-map.html` — the interactive map, self-contained.
//! - `MNESIO_MAP.md` — the same findings as text, for a terminal or a review.
//! - `mnesio-map.json` — the graph, for anything else.
//!
//! ## Why a CLI when there is already a dashboard
//!
//! `mnesio --repo .` serves the same map at `/code-graph`, and that is the
//! better loop while you work. It is the worse way to *show* someone: it needs
//! a port, a running process, and your machine. A file can be attached to a
//! pull request. The distribution difference is most of why a competitor's
//! `graphify .` reads as a product and a localhost route reads as a demo.
//!
//! ## What it will not do
//!
//! It will not print a token-saving multiple. Every number here describes the
//! repository in front of it — symbols, resolved calls, communities, and the
//! resolution rate that says how much of the call graph is missing. Retrieval
//! quality is a benchmark question with a benchmark answer, published
//! separately in `manifest/codeeval-v1-results.md`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mnesio_code::graph::GraphConfig;
use mnesio_code::journal::OutcomeJournal;
use mnesio_code::report::{render_html, render_markdown};
use mnesio_code::CodeMemory;
use mnesio_core::{Embedder, Scope};
use mnesio_index::{FastEmbedEmbedder, MockEmbedder};

const USAGE: &str = "\
mnesio-code — map a repository into files you can open

USAGE:
  mnesio-code [DIR] [OPTIONS]

  DIR defaults to the current directory.

OPTIONS:
  --out <DIR>        where to write artifacts        (default: the indexed dir)
  --max-nodes <N>    cap on nodes in the map         (default: 1500)
  --connected-only   hide symbols with no resolved edges
                     Off by default: isolated symbols are mostly calls the
                     parser could not bind, so hiding them makes the map look
                     better by concealing the weakest measurement.
  --embedder <WHICH> fastembed | mock                (default: fastembed)
                     `mock` skips the model download; retrieval quality is
                     meaningless with it, but the map is structural and unaffected.
  -h, --help         this text

WRITES:
  mnesio-map.html    interactive, self-contained — open it in a browser
  MNESIO_MAP.md      the same findings as text
  mnesio-map.json    the graph
";

struct Opts {
    dir: PathBuf,
    out: Option<PathBuf>,
    max_nodes: usize,
    connected_only: bool,
    embedder: String,
}

fn parse() -> Result<Opts, String> {
    let mut o = Opts {
        dir: PathBuf::from("."),
        out: None,
        max_nodes: 1500,
        connected_only: false,
        embedder: "fastembed".into(),
    };
    let mut positional = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--out" => o.out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?)),
            "--max-nodes" => {
                o.max_nodes = args
                    .next()
                    .ok_or("--max-nodes needs a number")?
                    .parse()
                    .map_err(|_| "--max-nodes must be a number".to_string())?
            }
            "--connected-only" => o.connected_only = true,
            "--embedder" => o.embedder = args.next().ok_or("--embedder needs a value")?,
            other if other.starts_with('-') => return Err(format!("unknown option {other}")),
            other if !positional => {
                o.dir = PathBuf::from(other);
                positional = true;
            }
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    Ok(o)
}

fn embedder(which: &str) -> Result<Arc<dyn Embedder>, String> {
    match which {
        "mock" => Ok(Arc::new(MockEmbedder::new(32))),
        "fastembed" => FastEmbedEmbedder::new()
            .map(|e| Arc::new(e) as Arc<dyn Embedder>)
            .map_err(|e| {
                format!(
                    "fastembed could not start: {e}\n\
                     The first run downloads a model. If you are offline, \
                     `--embedder mock` still produces the map — it is \
                     structural and does not depend on embeddings."
                )
            }),
        other => Err(format!(
            "unknown embedder {other:?}; expected fastembed or mock"
        )),
    }
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let opts = parse().map_err(|e| format!("{e}\n\n{USAGE}"))?;
    let dir = opts
        .dir
        .canonicalize()
        .map_err(|e| format!("{}: {e}", opts.dir.display()))?;
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir.display()));
    }
    let out = opts.out.unwrap_or_else(|| dir.clone());
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;

    let label = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.display().to_string());

    // Before indexing, not after: language support is a compile-time property
    // of this binary, so the same command on the same repository maps different
    // amounts of it from different builds. Reporting only the symbol count
    // would make "installed without grammars" look identical to "small
    // codebase", and the second is the flattering reading.
    let cov = mnesio_code::survey(&dir);
    let named = || {
        cov.top_skipped
            .iter()
            .map(|(e, n)| format!(".{e} ({n})"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let Some(rate) = cov.rate() else {
        return Err(format!("no source files under {}", dir.display()));
    };
    if cov.indexable == 0 {
        // Fail here rather than starting an index that cannot succeed. The
        // downstream error names the supported languages but not the ones
        // actually present, and two errors for one cause read as a bug.
        return Err(format!(
            "none of the {} source files under {} are in a language this build \
             can parse — found {}.\nRebuild with `--features tree-sitter` for \
             30 languages instead of 6.",
            cov.skipped,
            dir.display(),
            named(),
        ));
    }
    if cov.skipped > 0 {
        eprintln!(
            "reading {}/{} files ({:.0}%) — skipped: {}",
            cov.indexable,
            cov.indexable + cov.skipped,
            rate * 100.0,
            named(),
        );
        if rate < 0.5 {
            eprintln!(
                "  Most of this repository is in a language this build cannot \
                 parse.\n  Rebuild with `--features tree-sitter` for 30 \
                 languages instead of 6."
            );
        }
    }

    eprintln!("indexing {} …", dir.display());
    let started = std::time::Instant::now();
    let memory = CodeMemory::index(&dir, Scope::global("code"), embedder(&opts.embedder)?)
        .await
        .map_err(|e| format!("indexing failed: {e}"))?;

    // Colour by evidence if this repository has any. A fresh one has none, and
    // the report says so rather than omitting the section.
    let journal = OutcomeJournal::for_repo(&dir).read();
    let graph = memory.graph(
        &journal.entries,
        GraphConfig {
            max_nodes: opts.max_nodes.max(1),
            connected_only: opts.connected_only,
        },
    );
    let stats = memory.stats();

    write(
        &out.join("mnesio-map.html"),
        &render_html(&graph, &label, stats),
    )?;
    write(
        &out.join("MNESIO_MAP.md"),
        &render_markdown(&graph, &label, stats),
    )?;
    write(
        &out.join("mnesio-map.json"),
        &serde_json::to_string_pretty(&graph).map_err(|e| e.to_string())?,
    )?;

    let secs = started.elapsed().as_secs_f32();
    eprintln!(
        "\n{} symbols across {} files · {} resolved calls · {} communities · {secs:.1}s",
        graph.total_symbols,
        stats.files,
        graph.edges.len(),
        graph.communities
    );
    if let Some(r) = graph.resolution.rate() {
        eprintln!(
            "{:.0}% of call sites resolved — the map is a lower bound on the real call graph",
            r * 100.0
        );
        // The split, not just the rate. A single percentage cannot say whether
        // the missing edges are *addressable*: `ambiguous` means several
        // candidates and no way to choose (a ranking problem we can attack),
        // while `unresolved` usually means a call into the standard library or
        // a dependency that was never indexed (nothing to bind to). Phase 18F
        // was aimed at the ambiguous bucket on the assumption it was large.
        let e = &stats.edges;
        let total = e.resolved + e.unresolved + e.ambiguous;
        if total > 0 {
            let pct = |n: usize| 100.0 * n as f32 / total as f32;
            eprintln!(
                "  resolved {} ({:.0}%) · unresolved {} ({:.0}%) · ambiguous {} ({:.0}%)",
                e.resolved,
                pct(e.resolved),
                e.unresolved,
                pct(e.unresolved),
                e.ambiguous,
                pct(e.ambiguous),
            );
            eprintln!(
                "  of the unresolved, {} ({:.0}% of all calls) name a symbol that \
                 does exist here but was dropped by the receiver guard — the only \
                 slice type resolution could recover",
                e.unresolved_receiver_shadowed,
                pct(e.unresolved_receiver_shadowed),
            );
        }
    }
    if journal.entries.is_empty() {
        eprintln!(
            "no outcomes recorded yet, so nothing is coloured by whether it helped.\n\
             That fills in once the MCP server is recording your edits."
        );
    }
    eprintln!("\nwrote {}/mnesio-map.html — open it", out.display());
    Ok(())
}

fn write(path: &Path, body: &str) -> Result<(), String> {
    std::fs::write(path, body).map_err(|e| format!("writing {}: {e}", path.display()))
}

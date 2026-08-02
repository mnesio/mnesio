//! Compare the two parsers on real source, side by side.
//!
//! ```sh
//! cargo run -p mnesio-code --features tree-sitter --example parser_compare -- <dir> <lang> <ext>
//! ```
//!
//! Exists because "we added tree-sitter" is not a result. The interesting
//! question is what a grammar actually finds that line-scanning misses, and on
//! which languages — and that is only answerable against real files, not the
//! toy snippets in the unit tests.
//!
//! Read the output as *coverage*, not quality: more symbols means finer
//! granularity (a grammar sees methods inside an `impl` that the heuristic
//! parser folds into the outer item), and more edges means more call sites for
//! graph expansion to follow. Whether either improves retrieval is a separate,
//! unmeasured question — the Phase-17B miss taxonomy found `not_indexed = 0%`,
//! so on already-supported languages there is no parsing loss for a better
//! parser to recover.

use mnesio_code::{CodeParser, HeuristicParser, TreeSitterParser};
use std::path::{Path, PathBuf};

fn walk(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, ext, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some(ext) {
            out.push(p);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dir, lang, ext) = match args.as_slice() {
        [d, l, e] => (d.clone(), l.clone(), e.clone()),
        _ => (
            "crates/mnesio-index/src".to_string(),
            "rust".to_string(),
            "rs".to_string(),
        ),
    };

    let mut files = Vec::new();
    walk(Path::new(&dir), &ext, &mut files);
    files.sort();
    if files.is_empty() {
        eprintln!("no .{ext} files under {dir}");
        std::process::exit(1);
    }

    let (mut ts_sym, mut ts_edge, mut ts_files) = (0usize, 0usize, 0usize);
    let (mut he_sym, mut he_edge, mut he_files) = (0usize, 0usize, 0usize);

    for f in &files {
        let Ok(src) = std::fs::read_to_string(f) else {
            continue;
        };
        let key = f.to_string_lossy().to_string();
        if let Ok(p) = TreeSitterParser.parse(&key, &lang, &src) {
            ts_files += 1;
            ts_sym += p.symbols.len();
            ts_edge += p.edges.len();
        }
        if let Ok(p) = HeuristicParser.parse(&key, &lang, &src) {
            he_files += 1;
            he_sym += p.symbols.len();
            he_edge += p.edges.len();
        }
    }

    let pct = |new: usize, old: usize| -> String {
        if old == 0 {
            return if new == 0 { "—".into() } else { "new".into() };
        }
        format!("{:+.0}%", (new as f64 / old as f64 - 1.0) * 100.0)
    };

    println!("{lang} · {} files under {dir}\n", files.len());
    println!("| parser | files parsed | symbols | call edges |");
    println!("|---|---|---|---|");
    println!("| heuristic | {he_files} | {he_sym} | {he_edge} |");
    println!("| tree-sitter | {ts_files} | {ts_sym} | {ts_edge} |");
    println!(
        "| delta | — | {} | {} |",
        pct(ts_sym, he_sym),
        pct(ts_edge, he_edge)
    );
}

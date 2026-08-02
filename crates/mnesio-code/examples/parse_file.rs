//! Parse one source file and print what the index would store.
//!
//! ```bash
//! cargo run -p mnesio-code --example parse_file -- crates/mnesio-core/src/traits.rs
//! ```
use mnesio_code::{CodeParser, HeuristicParser};

fn main() {
    let path = std::env::args().nth(1).expect("usage: parse_file <path>");
    let src = std::fs::read_to_string(&path).expect("read file");
    let lang = match path.rsplit('.').next() {
        Some("rs") => "rust",
        Some("go") => "go",
        Some("ts") => "typescript",
        Some("js") => "javascript",
        Some("py") => "python",
        Some("java") => "java",
        other => other.unwrap_or("rust"),
    };

    let parsed = HeuristicParser.parse(&path, lang, &src).expect("parse");
    println!(
        "{} — {} symbols, {} edges ({} lines of source)\n",
        parsed.path,
        parsed.symbols.len(),
        parsed.edges.len(),
        src.lines().count()
    );
    for s in &parsed.symbols {
        let callees: Vec<_> = parsed
            .edges
            .iter()
            .filter(|e| e.from == s.key())
            .map(|e| e.to_name.as_str())
            .collect();
        println!(
            "  {:<9} {:<28} L{}-{} ({} lines){}",
            s.kind.as_tag(),
            s.name,
            s.start_line,
            s.end_line,
            s.end_line - s.start_line + 1,
            if callees.is_empty() {
                String::new()
            } else {
                format!("  → calls: {}", callees.join(", "))
            }
        );
    }
}

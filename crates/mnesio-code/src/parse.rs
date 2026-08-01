//! Parsing seam: source text → [`ParsedFile`].
//!
//! Two implementations, following the same pattern as the LLM and KV backends
//! (Hard Rule #7 — every external dependency behind a trait):
//!
//! - [`HeuristicParser`] — always compiled, zero dependencies. Line-scanning
//!   with brace matching. Honest about what it is: **not an AST**. It handles
//!   brace-delimited languages (Rust, Go, TypeScript, Java, C-likes) and is what
//!   the test suite runs on, so CI stays fast and offline.
//! - `TreeSitterParser` — real grammars, behind the `tree-sitter` feature.
//!   `TODO(phase-17): land this once 17A's event mapping is settled.`
//!
//! ## Where the heuristic parser is approximate
//!
//! Stated plainly, because the difference matters when you read a retrieval
//! result and wonder why something is missing:
//!
//! - **Indentation-based languages (Python) are not supported** — brace matching
//!   has nothing to match. Use the `tree-sitter` feature for those.
//! - **Call targets are matched by bare name.** `parse(..)` binds to a symbol
//!   named `parse` if exactly one is in scope; overloads and methods on
//!   different types with the same name are ambiguous, and resolution across
//!   files is name-only — there is no type inference. Edges are therefore a
//!   *good retrieval signal*, not a compiler-grade call graph.
//! - **Nested definitions** (a function inside a function) are attributed to the
//!   outermost definition.

use crate::{CodeEdge, EdgeKind, ParsedFile, Symbol, SymbolKind};

/// Why parsing failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The parser has no grammar for this language.
    UnsupportedLanguage(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnsupportedLanguage(l) => {
                write!(
                    f,
                    "no parser for language {l:?} (indentation-based languages \
                     need the `tree-sitter` feature)"
                )
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// The swappable parsing backend.
///
/// Sync on purpose: parsing is CPU-bound and runs on the *indexing* path, which
/// is already off the write path (Hard Rule #5). Callers batch files onto a
/// blocking pool rather than making every symbol an await point.
pub trait CodeParser: Send + Sync {
    /// Extract symbols and unresolved edges from one file's source.
    ///
    /// `path` is repo-relative and is stored on every symbol, so it must be the
    /// same string used for the file's `Source::uri`.
    fn parse(&self, path: &str, language: &str, source: &str) -> Result<ParsedFile, ParseError>;
}

/// Dependency-free parser using line scanning and brace matching.
///
/// See the module docs for exactly where this is approximate — it is a
/// retrieval signal, not a compiler front end.
#[derive(Debug, Clone, Default)]
pub struct HeuristicParser;

/// Languages whose blocks are delimited by `{ }`, which is all this parser can
/// follow.
const BRACE_LANGUAGES: &[&str] = &[
    "rust",
    "go",
    "typescript",
    "javascript",
    "tsx",
    "jsx",
    "java",
    "kotlin",
    "swift",
    "c",
    "cpp",
    "csharp",
];

/// Words that look like a call (`if (..)`, `while (..)`) but aren't.
const CALL_KEYWORDS: &[&str] = &[
    "if", "for", "while", "match", "switch", "return", "fn", "let", "const", "catch", "with",
    "await", "async", "and", "or", "not", "in", "is", "new", "typeof", "sizeof", "defer", "go",
];

impl CodeParser for HeuristicParser {
    fn parse(&self, path: &str, language: &str, source: &str) -> Result<ParsedFile, ParseError> {
        if !BRACE_LANGUAGES.contains(&language) {
            return Err(ParseError::UnsupportedLanguage(language.to_string()));
        }

        let lines: Vec<&str> = source.lines().collect();
        let mut symbols = Vec::new();
        let mut edges = Vec::new();

        let mut i = 0usize;
        while i < lines.len() {
            let Some((kind, name)) = definition_at(lines[i]) else {
                i += 1;
                continue;
            };

            let end = match block_end(&lines, i) {
                Some(e) => e,
                // A definition we can't delimit (e.g. a trait method signature
                // ending in `;`) — record it as a one-line symbol rather than
                // swallowing the rest of the file.
                None => i,
            };

            let text = lines[i..=end].join("\n");
            let symbol = Symbol {
                name,
                // `#[test]` sits on the line above, so the kind is refined here
                // rather than in `definition_at`, which only sees one line.
                kind: if kind == SymbolKind::Function && preceded_by_test_attr(&lines, i) {
                    SymbolKind::Test
                } else {
                    kind
                },
                path: path.to_string(),
                signature: Some(lines[i].trim().trim_end_matches('{').trim().to_string()),
                doc: doc_comment_above(&lines, i),
                start_line: (i + 1) as u32,
                end_line: (end + 1) as u32,
                text,
            };

            // Only function-like symbols have an executable body. Scanning a
            // `struct`/`enum`/`trait` body for `ident(` yields garbage: enum
            // variants (`Storage(String)`) and trait method *declarations*
            // (`fn append(..);`) both look like calls but are definitions.
            // Caught by running this over mnesio's own source.
            if has_executable_body(symbol.kind) {
                for callee in call_names(&lines[i..=end], &symbol.name) {
                    edges.push(CodeEdge {
                        from: symbol.key(),
                        to_name: callee,
                        kind: EdgeKind::Calls,
                    });
                }
            }

            symbols.push(symbol);
            // Skip the body: nested definitions belong to the outer symbol.
            i = end + 1;
        }

        Ok(ParsedFile {
            path: path.to_string(),
            language: language.to_string(),
            symbols,
            edges,
        })
    }
}

/// Does this kind of symbol contain executable statements?
///
/// Gates call-edge extraction: type and trait bodies contain declarations, not
/// calls, so scanning them produces only false positives.
fn has_executable_body(kind: SymbolKind) -> bool {
    matches!(
        kind,
        SymbolKind::Function | SymbolKind::Method | SymbolKind::Test
    )
}

/// Recognise a definition on a single line, returning its kind and name.
fn definition_at(line: &str) -> Option<(SymbolKind, String)> {
    let t = line.trim_start();
    // Strip modifiers so `pub async fn x` and `export default class Y` reduce
    // to the keyword we switch on.
    let mut rest = t;
    for m in [
        "pub(crate) ",
        "pub ",
        "export ",
        "default ",
        "async ",
        "static ",
        "public ",
        "private ",
        "protected ",
        "final ",
        "abstract ",
        "unsafe ",
        "extern ",
    ] {
        if let Some(s) = rest.strip_prefix(m) {
            rest = s.trim_start();
        }
    }

    let (kw, kind) = [
        ("fn ", SymbolKind::Function),
        ("func ", SymbolKind::Function),
        ("function ", SymbolKind::Function),
        ("struct ", SymbolKind::Struct),
        ("class ", SymbolKind::Struct),
        ("enum ", SymbolKind::Enum),
        ("trait ", SymbolKind::Trait),
        ("interface ", SymbolKind::Trait),
        ("type ", SymbolKind::TypeAlias),
        ("const ", SymbolKind::Constant),
        ("mod ", SymbolKind::Module),
    ]
    .into_iter()
    .find(|(kw, _)| rest.starts_with(kw))?;

    let name: String = rest[kw.len()..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();

    (!name.is_empty()).then_some((kind, name))
}

/// Find the line index where the block opened at `start` closes.
///
/// Counts braces while skipping line comments and string literals, which is the
/// difference between this working on real code and falling over the first time
/// someone writes `println!("}}")`.
fn block_end(lines: &[&str], start: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut seen_open = false;

    for (offset, line) in lines[start..].iter().enumerate() {
        let mut chars = line.chars().peekable();
        let mut in_string = false;
        while let Some(c) = chars.next() {
            match c {
                '\\' if in_string => {
                    chars.next(); // skip the escaped char
                }
                '"' => in_string = !in_string,
                '/' if !in_string && chars.peek() == Some(&'/') => break, // line comment
                '{' if !in_string => {
                    depth += 1;
                    seen_open = true;
                }
                '}' if !in_string => {
                    depth -= 1;
                    if seen_open && depth == 0 {
                        return Some(start + offset);
                    }
                }
                _ => {}
            }
        }
        // A signature-only line (`fn f(&self);`) never opens a block.
        if !seen_open && line.trim_end().ends_with(';') {
            return Some(start + offset);
        }
    }
    None
}

/// Collect the contiguous doc-comment block immediately above `idx`.
fn doc_comment_above(lines: &[&str], idx: usize) -> Option<String> {
    let mut out = Vec::new();
    for i in (0..idx).rev() {
        let t = lines[i].trim();
        if let Some(d) = t
            .strip_prefix("///")
            .or_else(|| t.strip_prefix("//!"))
            .or_else(|| t.strip_prefix("*"))
        {
            out.push(d.trim().to_string());
        } else if t.starts_with("#[") || t.starts_with("@") || t.is_empty() && !out.is_empty() {
            // Attributes/decorators sit between the doc and the definition.
            continue;
        } else {
            break;
        }
    }
    (!out.is_empty()).then(|| {
        out.reverse();
        out.join("\n")
    })
}

/// Is this definition preceded by a test attribute/annotation?
fn preceded_by_test_attr(lines: &[&str], idx: usize) -> bool {
    lines[..idx].iter().rev().take(3).any(|l| {
        let t = l.trim();
        t.starts_with("#[test]") || t.starts_with("#[tokio::test]") || t.starts_with("@Test")
    })
}

/// Identifiers that appear immediately before a `(` — i.e. probable calls.
fn call_names(body: &[&str], self_name: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    for line in body {
        let code = match line.find("//") {
            Some(p) => &line[..p],
            None => line,
        };
        let bytes: Vec<char> = code.chars().collect();
        let mut idx = 0usize;
        while idx < bytes.len() {
            if bytes[idx] == '(' && idx > 0 {
                // Walk backwards over the identifier preceding the paren.
                let mut s = idx;
                while s > 0 && (bytes[s - 1].is_alphanumeric() || bytes[s - 1] == '_') {
                    s -= 1;
                }
                let name: String = bytes[s..idx].iter().collect();
                let is_call = !name.is_empty()
                    && !name.chars().next().unwrap().is_numeric()
                    && name != self_name
                    && !CALL_KEYWORDS.contains(&name.as_str())
                    && !out.contains(&name);
                if is_call {
                    out.push(name);
                }
            }
            idx += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUST: &str = r#"
/// Loads configuration from disk.
pub fn load_config(path: &str) -> Config {
    let raw = read_file(path);
    parse_config(raw)
}

pub struct Config {
    pub name: String,
}

#[test]
fn load_config_works() {
    load_config("x");
}
"#;

    fn parse(src: &str) -> ParsedFile {
        HeuristicParser
            .parse("src/cfg.rs", "rust", src)
            .expect("rust is supported")
    }

    #[test]
    fn extracts_symbols_with_kind_span_and_doc() {
        let f = parse(RUST);
        let names: Vec<_> = f.symbols.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["load_config", "Config", "load_config_works"]);

        let load = &f.symbols[0];
        assert_eq!(load.kind, SymbolKind::Function);
        assert_eq!(load.doc.as_deref(), Some("Loads configuration from disk."));
        // Span covers the whole body, so the retrieved text is self-contained.
        assert!(load.text.contains("parse_config(raw)"));
        assert!(load.end_line > load.start_line);

        assert_eq!(f.symbols[1].kind, SymbolKind::Struct);
    }

    #[test]
    fn test_attribute_promotes_function_to_test_kind() {
        let f = parse(RUST);
        let t = f.symbols.iter().find(|s| s.name == "load_config_works");
        assert_eq!(t.map(|s| s.kind), Some(SymbolKind::Test));
    }

    #[test]
    fn call_edges_link_caller_to_callees() {
        let f = parse(RUST);
        let calls: Vec<_> = f
            .edges
            .iter()
            .filter(|e| e.from.ends_with("function:load_config"))
            .map(|e| e.to_name.as_str())
            .collect();
        assert!(calls.contains(&"read_file"), "got {calls:?}");
        assert!(calls.contains(&"parse_config"), "got {calls:?}");
        // A function must not be recorded as calling itself.
        assert!(!calls.contains(&"load_config"));
    }

    #[test]
    fn control_flow_keywords_are_not_calls() {
        let f = HeuristicParser
            .parse(
                "a.rs",
                "rust",
                "fn f() {\n    if (x) { while (y) { helper(); } }\n}\n",
            )
            .unwrap();
        let calls: Vec<_> = f.edges.iter().map(|e| e.to_name.as_str()).collect();
        assert_eq!(calls, ["helper"], "if/while must not count as calls");
    }

    #[test]
    fn braces_inside_strings_and_comments_do_not_break_spans() {
        // The classic failure: a `}` in a string literal closing the block early.
        let src = "fn f() {\n    println!(\"}\");\n    // }\n    g();\n}\nfn after() {}\n";
        let f = HeuristicParser.parse("a.rs", "rust", src).unwrap();
        let names: Vec<_> = f.symbols.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            ["f", "after"],
            "span must not end at the string brace"
        );
        assert!(f.symbols[0].text.contains("g();"));
    }

    #[test]
    fn indentation_languages_are_rejected_not_silently_wrong() {
        let err = HeuristicParser.parse("a.py", "python", "def f():\n    pass\n");
        assert_eq!(
            err,
            Err(ParseError::UnsupportedLanguage("python".into())),
            "returning empty symbols would look like an empty file"
        );
    }

    #[test]
    fn type_bodies_produce_no_call_edges() {
        // Regression: both of these emitted bogus `Calls` edges until call
        // extraction was gated on the symbol having an executable body. Found
        // by running the parser over mnesio's own `traits.rs`.
        let src = "\
pub enum MnesioError {
    Storage(String),
    NotFound(String),
}

pub trait EventLog {
    async fn append(&self, e: Event) -> Result<Id, MnesioError>;
    async fn read_from(&self, id: Id) -> Result<Vec<Entry>, MnesioError>;
}
";
        let f = HeuristicParser.parse("a.rs", "rust", src).unwrap();
        assert_eq!(f.symbols.len(), 2, "enum + trait");
        assert!(
            f.edges.is_empty(),
            "enum variants and trait method decls are definitions, not calls; got {:?}",
            f.edges.iter().map(|e| &e.to_name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn function_bodies_still_produce_call_edges() {
        // The guard above must not silence real calls.
        let f = HeuristicParser
            .parse("a.rs", "rust", "fn f() {\n    helper();\n}\n")
            .unwrap();
        assert_eq!(
            f.edges
                .iter()
                .map(|e| e.to_name.as_str())
                .collect::<Vec<_>>(),
            ["helper"]
        );
    }

    #[test]
    fn nested_definitions_belong_to_the_outer_symbol() {
        let src = "fn outer() {\n    fn inner() {}\n    inner();\n}\n";
        let f = HeuristicParser.parse("a.rs", "rust", src).unwrap();
        let names: Vec<_> = f.symbols.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["outer"]);
        assert!(f.symbols[0].text.contains("fn inner"));
    }
}

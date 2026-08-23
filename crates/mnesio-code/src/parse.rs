//! Parsing seam: source text → [`ParsedFile`].
//!
//! Two implementations, following the same pattern as the LLM and KV backends
//! (Hard Rule #7 — every external dependency behind a trait):
//!
//! - [`HeuristicParser`] — always compiled, zero dependencies. Line scanning,
//!   with brace matching for C-likes and column counting for Python. Honest
//!   about what it is: **not an AST**. It is what the test suite runs on, so
//!   CI stays fast and offline.
//! - `TreeSitterParser` — real grammars, behind the `tree-sitter` feature.
//!   `TODO(phase-17): land this once 17A's event mapping is settled.`
//!
//! ## Where the heuristic parser is approximate
//!
//! Stated plainly, because the difference matters when you read a retrieval
//! result and wonder why something is missing:
//!
//! - **Two block strategies.** Brace-delimited languages (Rust, Go,
//!   TypeScript, Java, C-likes) are matched on `{}`; Python is delimited by
//!   indentation, counting columns and treating blank/comment lines as
//!   structurally transparent. Any other language is *rejected*, never parsed
//!   into plausible nonsense.
//! - **Call targets are matched by bare name.** `parse(..)` binds to a symbol
//!   named `parse` if exactly one is in scope; overloads and methods on
//!   different types with the same name are ambiguous, and resolution across
//!   files is name-only — there is no type inference. Edges are therefore a
//!   *good retrieval signal*, not a compiler-grade call graph.
//! - **Nested definitions** are attributed to the outermost definition — in
//!   brace languages only. Python indexes methods separately from their class,
//!   because a class body there is the normal home for retrievable code.
//! - **The file header is a symbol.** `//!`, a leading Python docstring, or a
//!   top-of-file `/** … */` becomes a [`SymbolKind::Module`] symbol, so
//!   module-level prose has an owner in the index instead of being dropped.

use crate::{CodeEdge, EdgeKind, Import, ParsedFile, Symbol, SymbolKind};

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
                    "no parser for language {l:?} (supported: brace-delimited \
                     C-likes and Python; others need the `tree-sitter` feature)"
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

/// Languages whose blocks are delimited by **indentation**, so `block_end` has
/// to count columns rather than braces.
const INDENT_LANGUAGES: &[&str] = &["python"];

impl CodeParser for HeuristicParser {
    fn parse(&self, path: &str, language: &str, source: &str) -> Result<ParsedFile, ParseError> {
        if INDENT_LANGUAGES.contains(&language) {
            return Ok(parse_indented(path, language, source));
        }
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
                for (callee, via_receiver) in call_names(&lines[i..=end], &symbol.name) {
                    edges.push(CodeEdge {
                        from: symbol.key(),
                        to_name: callee,
                        kind: EdgeKind::Calls,
                        via_receiver,
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
            module_doc: module_doc(&lines, language),
            symbols,
            edges,
            imports: extract_imports(language, source),
        })
    }
}

// ---------------------------------------------------------------------------
// Module-level prose
// ---------------------------------------------------------------------------

/// The leading documentation block of a file, cleaned of comment markers.
/// Header prose of a whole file, for callers that have the source rather than
/// pre-split lines — the tree-sitter parser, which never splits.
pub fn module_doc_for(language: &str, source: &str) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    module_doc(&lines, language)
}

fn module_doc(lines: &[&str], language: &str) -> Option<String> {
    if INDENT_LANGUAGES.contains(&language) {
        // Python: the module docstring is the file's first statement.
        let first = lines.iter().position(|l| indent_of(l).is_some())?;
        return python_docstring_at(lines, first);
    }

    let mut out: Vec<String> = Vec::new();
    let mut in_block = false;
    for line in lines {
        let t = line.trim();
        if in_block {
            if let Some(body) = t.strip_suffix("*/") {
                push_doc_line(&mut out, body.trim_start_matches('*'));
                break;
            }
            push_doc_line(&mut out, t.trim_start_matches('*'));
            continue;
        }
        if let Some(d) = t.strip_prefix("//!") {
            push_doc_line(&mut out, d);
        } else if t.starts_with("/**") || t.starts_with("/*!") {
            in_block = true;
            push_doc_line(&mut out, &t[3..]);
        } else if t.is_empty() || t.starts_with("#!") {
            // Blank lines and a shebang sit above/inside the header block.
            continue;
        } else {
            // First real code line: the header is over. Anything after this is
            // a symbol's own doc, not the module's.
            break;
        }
    }
    let joined = out.join("\n").trim().to_string();
    (!joined.is_empty()).then_some(joined)
}

fn push_doc_line(out: &mut Vec<String>, s: &str) {
    let t = s.trim();
    if !t.is_empty() || out.last().is_some_and(|l| !l.is_empty()) {
        out.push(t.to_string());
    }
}

// ---------------------------------------------------------------------------
// Indentation-based languages (Python)
// ---------------------------------------------------------------------------

/// Column at which a line's content starts, or `None` for blank/comment lines,
/// which carry no block structure and must not end one.
fn indent_of(line: &str) -> Option<usize> {
    let t = line.trim_start();
    if t.is_empty() || t.starts_with('#') {
        return None;
    }
    Some(line.len() - t.len())
}

/// Recognise a Python definition, returning kind, name, and its indent column.
///
/// A method is distinguished from a function purely by being indented — inside
/// a `class` body — which is what Python itself means by the distinction.
fn python_definition_at(line: &str) -> Option<(SymbolKind, String, usize)> {
    let indent = indent_of(line)?;
    let t = line.trim_start();
    let rest = t.strip_prefix("async ").unwrap_or(t);

    let (kw, base) = rest
        .strip_prefix("def ")
        .map(|r| {
            (
                r,
                if indent > 0 {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                },
            )
        })
        .or_else(|| rest.strip_prefix("class ").map(|r| (r, SymbolKind::Class)))?;

    let name: String = kw
        .trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        return None;
    }
    // pytest and unittest both key off the `test_` prefix, so this is the
    // convention rather than a guess.
    let kind = if base != SymbolKind::Class && name.starts_with("test_") {
        SymbolKind::Test
    } else {
        base
    };
    Some((kind, name, indent))
}

/// Last line of the block opened at `start`, by indentation.
///
/// The block runs until a line whose indent is less than or equal to the
/// definition's. Blank and comment lines are skipped rather than treated as
/// dedents, and trailing blanks are trimmed so a symbol's text does not absorb
/// the gap before the next definition.
fn python_block_end(lines: &[&str], start: usize, def_indent: usize) -> usize {
    let mut end = start;
    for (offset, line) in lines[start + 1..].iter().enumerate() {
        match indent_of(line) {
            None => continue,
            Some(i) if i > def_indent => end = start + 1 + offset,
            Some(_) => break,
        }
    }
    end
}

/// The docstring opening at `idx`, if the line starts one.
///
/// Python puts a symbol's prose *inside* its body as the first statement, not
/// above it in comments — so [`doc_comment_above`] finds nothing for Python and
/// the richest natural-language signal in the file would be lost. Returns the
/// text with quotes stripped.
fn python_docstring_at(lines: &[&str], idx: usize) -> Option<String> {
    let t = lines.get(idx)?.trim();
    let quote = if t.starts_with("\"\"\"") {
        "\"\"\""
    } else if t.starts_with("'''") {
        "'''"
    } else {
        return None;
    };

    let first = &t[quote.len()..];
    // Single-line docstring: `"""Load the config."""`
    if let Some(body) = first.strip_suffix(quote) {
        return Some(body.trim().to_string());
    }

    let mut out = Vec::new();
    if !first.trim().is_empty() {
        out.push(first.trim().to_string());
    }
    for line in &lines[idx + 1..] {
        let t = line.trim();
        if let Some(body) = t.strip_suffix(quote) {
            if !body.trim().is_empty() {
                out.push(body.trim().to_string());
            }
            break;
        }
        out.push(t.to_string());
    }
    let joined = out.join("\n").trim().to_string();
    (!joined.is_empty()).then_some(joined)
}

/// Find a definition's docstring: the first statement of its body, skipping the
/// continuation lines of a multi-line signature.
fn python_doc_for(lines: &[&str], def_line: usize, end: usize) -> Option<String> {
    // A signature can wrap; the body starts after the line ending in `:`.
    let mut i = def_line;
    while i <= end && !lines[i].trim_end().ends_with(':') {
        i += 1;
    }
    for j in i + 1..=end.min(lines.len().saturating_sub(1)) {
        match indent_of(lines[j]) {
            None => continue,
            Some(_) => return python_docstring_at(lines, j),
        }
    }
    None
}

/// Parse an indentation-delimited file.
///
/// Unlike the brace path this does **not** skip a block's interior: a class body
/// contains methods that are themselves worth retrieving, and skipping them
/// would leave a Python codebase with one memory per class.
fn parse_indented(path: &str, language: &str, source: &str) -> ParsedFile {
    let lines: Vec<&str> = source.lines().collect();
    let mut symbols = Vec::new();
    let mut edges = Vec::new();

    for i in 0..lines.len() {
        let Some((kind, name, indent)) = python_definition_at(lines[i]) else {
            continue;
        };
        let end = python_block_end(&lines, i, indent);
        let symbol = Symbol {
            name,
            kind,
            path: path.to_string(),
            signature: Some(lines[i].trim().trim_end_matches(':').trim().to_string()),
            doc: python_doc_for(&lines, i, end),
            start_line: (i + 1) as u32,
            end_line: (end + 1) as u32,
            text: lines[i..=end].join("\n"),
        };
        if has_executable_body(symbol.kind) {
            for (callee, via_receiver) in call_names(&lines[i..=end], &symbol.name) {
                edges.push(CodeEdge {
                    from: symbol.key(),
                    to_name: callee,
                    kind: EdgeKind::Calls,
                    via_receiver,
                });
            }
        }
        symbols.push(symbol);
    }

    ParsedFile {
        path: path.to_string(),
        language: language.to_string(),
        module_doc: module_doc(&lines, language),
        symbols,
        edges,
        imports: extract_imports(language, source),
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
fn call_names(body: &[&str], self_name: &str) -> Vec<(String, bool)> {
    let mut out: Vec<(String, bool)> = Vec::new();

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
                // A `.` or `::` immediately before the identifier means the
                // call has a receiver. We still cannot say what type it is —
                // that needs inference — but "there was a receiver" is itself
                // load-bearing: it is what separates `parse()` from
                // `s.parse()`, and binding the second to a free function
                // elsewhere in the repository is how `push` ended up looking
                // like the most-depended-on symbol in the workspace.
                let via_receiver = s > 0
                    && (bytes[s - 1] == '.'
                        || (s > 1 && bytes[s - 1] == ':' && bytes[s - 2] == ':'));
                let is_call = !name.is_empty()
                    && !name.chars().next().unwrap().is_numeric()
                    && name != self_name
                    && !CALL_KEYWORDS.contains(&name.as_str())
                    && !out.iter().any(|(n, _)| n == &name);
                if is_call {
                    out.push((name, via_receiver));
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
    fn unknown_languages_are_rejected_not_silently_wrong() {
        let err = HeuristicParser.parse("a.hs", "haskell", "f x = x\n");
        assert_eq!(
            err,
            Err(ParseError::UnsupportedLanguage("haskell".into())),
            "returning empty symbols would look like an empty file"
        );
    }

    // --- module-level prose ---

    #[test]
    fn the_file_header_is_captured_as_module_doc() {
        // Regression for the measured 17B miss: "bm25 tantivy search view"
        // could not reach `Bm25View`, because the struct has no doc of its own
        // and those words live only in the `//!` header. The indexer turns this
        // into a one-line breadcrumb on every symbol in the file.
        let src = "//! BM25 search view over tantivy.\n//!\n//! Scope-filtered.\n\n\
                   /// A struct.\npub struct Bm25View { a: u8 }\n";
        let f = HeuristicParser.parse("src/bm25.rs", "rust", src).unwrap();

        let doc = f.module_doc.as_deref().expect("module header was dropped");
        assert!(doc.starts_with("BM25 search view over tantivy"));
        assert!(doc.contains("Scope-filtered"), "block truncated early");
        assert!(
            !doc.contains("A struct"),
            "swallowed the next symbol's own doc: {doc:?}"
        );
        // The header must not become a symbol of its own — one that competes
        // for retrieval slots with the definitions it describes.
        assert!(!f.symbols.iter().any(|s| s.kind == SymbolKind::Module));
    }

    #[test]
    fn a_file_without_a_header_has_no_module_doc() {
        let f = HeuristicParser
            .parse("a.rs", "rust", "pub fn go() {}\n")
            .unwrap();
        assert_eq!(f.module_doc, None);
    }

    #[test]
    fn block_and_python_headers_are_both_recognised() {
        let ts = HeuristicParser
            .parse(
                "a.ts",
                "typescript",
                "/**\n * Retry helper for the API client.\n */\nexport function go() {}\n",
            )
            .unwrap();
        assert!(
            ts.module_doc
                .as_deref()
                .is_some_and(|d| d.contains("Retry helper")),
            "/** */ header not captured: {:?}",
            ts.module_doc
        );

        assert!(
            py().module_doc
                .as_deref()
                .is_some_and(|d| d.contains("Vector store index module")),
            "python module docstring not captured"
        );
    }

    // --- Python (indentation) ---

    const PY: &str = "\
\"\"\"Vector store index module.\"\"\"

class VectorStoreIndex(BaseIndex):
    \"\"\"An index backed by a vector store.

    Builds embeddings for each node.
    \"\"\"

    def __init__(self, nodes):
        self._nodes = nodes

    async def as_retriever(self, top_k=10):
        '''Return a retriever over this index.'''
        return VectorIndexRetriever(self, top_k)


def build_index_from_nodes(nodes):
    \"\"\"Construct an index from parsed nodes.\"\"\"
    return VectorStoreIndex(nodes)


def test_build_index():
    assert build_index_from_nodes([]) is not None
";

    fn py() -> ParsedFile {
        HeuristicParser.parse("idx.py", "python", PY).unwrap()
    }

    #[test]
    fn python_blocks_are_delimited_by_indentation() {
        let f = py();
        let class = f
            .symbols
            .iter()
            .find(|s| s.name == "VectorStoreIndex")
            .unwrap();
        assert_eq!(class.kind, SymbolKind::Class);
        // The class must span its methods and stop before the next top-level
        // def — the whole point of counting columns instead of braces.
        assert_eq!((class.start_line, class.end_line), (3, 14));

        let free = f
            .symbols
            .iter()
            .find(|s| s.name == "build_index_from_nodes")
            .unwrap();
        assert_eq!(free.kind, SymbolKind::Function);
        assert_eq!((free.start_line, free.end_line), (17, 19));
    }

    #[test]
    fn python_methods_are_indexed_separately_from_their_class() {
        // The brace path skips a block's interior, which for Python would leave
        // one memory per class and make every method unretrievable.
        let f = py();
        for want in ["__init__", "as_retriever"] {
            let m = f
                .symbols
                .iter()
                .find(|s| s.name == want)
                .unwrap_or_else(|| panic!("method {want} was swallowed by its class"));
            assert_eq!(m.kind, SymbolKind::Method, "indented def is a method");
        }
    }

    #[test]
    fn python_docstrings_are_captured_as_the_doc() {
        // Python puts prose *inside* the body, so `doc_comment_above` finds
        // nothing — without this the richest text in a Python file is lost.
        let f = py();
        let class = f
            .symbols
            .iter()
            .find(|s| s.name == "VectorStoreIndex")
            .unwrap();
        let doc = class.doc.as_deref().unwrap_or_default();
        assert!(
            doc.contains("An index backed by a vector store"),
            "got {doc:?}"
        );
        assert!(
            doc.contains("Builds embeddings"),
            "multi-line body lost: {doc:?}"
        );

        // Single-line, and the `'''` spelling.
        let m = f.symbols.iter().find(|s| s.name == "as_retriever").unwrap();
        assert_eq!(
            m.doc.as_deref(),
            Some("Return a retriever over this index.")
        );
    }

    #[test]
    fn python_tests_are_recognised_by_the_naming_convention() {
        // pytest and unittest both key off `test_`; there is no attribute to
        // look for as there is in Rust.
        let f = py();
        let t = f
            .symbols
            .iter()
            .find(|s| s.name == "test_build_index")
            .unwrap();
        assert_eq!(t.kind, SymbolKind::Test);
    }

    #[test]
    fn python_calls_become_edges() {
        let f = py();
        let has = |from: &str, to: &str| {
            f.edges
                .iter()
                .any(|e| e.from.ends_with(from) && e.to_name == to)
        };
        assert!(has("as_retriever", "VectorIndexRetriever"));
        assert!(has("build_index_from_nodes", "VectorStoreIndex"));
    }

    #[test]
    fn python_comments_and_blank_lines_do_not_close_a_block() {
        // A `#` comment at column 0 inside an indented body would look like a
        // dedent to a naive scanner and truncate the symbol.
        let src =
            "def f():\n    a = 1\n\n# stray comment at column 0\n    b = 2\n\ndef g():\n    pass\n";
        let f = HeuristicParser.parse("a.py", "python", src).unwrap();
        let g = f.symbols.iter().find(|s| s.name == "f").unwrap();
        assert!(
            g.text.contains("b = 2"),
            "block ended early at a comment: {:?}",
            g.text
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

/// Extract imported names and their module hints from one file's source.
///
/// Phase 18F. Shared by both parsers because import syntax is line-oriented in
/// every language here, so a grammar buys nothing: `use a::b::C;` and
/// `from a.b import C` are as unambiguous to a line scanner as to a parse tree.
///
/// The output feeds [`crate::index`]'s ambiguity tie-break only. It is a hint,
/// never an assertion — a wrong hint matches no indexed path and changes
/// nothing, which is why this can be heuristic without risking bad edges.
pub fn extract_imports(language: &str, source: &str) -> Vec<Import> {
    let mut out = Vec::new();
    for raw in source.lines() {
        let line = raw.trim();
        match language {
            "rust" => rust_import(line, &mut out),
            "python" => python_import(line, &mut out),
            "javascript" | "typescript" | "tsx" | "jsx" => js_import(line, &mut out),
            "go" => go_import(line, &mut out),
            _ => {}
        }
        // Imports are conventionally at the top, but Rust `use` inside a
        // function and Python imports inside a method are both legal and
        // common, so the whole file is scanned rather than a prefix.
    }
    out.sort_by(|a, b| a.name.cmp(&b.name).then(a.module.cmp(&b.module)));
    out.dedup();
    out
}

/// `use a::b::C;` · `use a::b::{C, D};` · `pub use a::B as C;`
fn rust_import(line: &str, out: &mut Vec<Import>) {
    let rest = line
        .strip_prefix("pub use ")
        .or_else(|| line.strip_prefix("use "))
        .map(str::trim);
    let Some(rest) = rest else { return };
    let rest = rest.trim_end_matches(';').trim();

    // `a::b::{C, D}` — one module, several names.
    if let Some(open) = rest.find('{') {
        let prefix = rest[..open].trim().trim_end_matches("::");
        let Some(close) = rest.rfind('}') else { return };
        for item in rest[open + 1..close].split(',') {
            let item = item.trim();
            // Nested groups (`a::{b::{C}}`) are rare and not worth a parser;
            // the outer name still lands via the prefix path below.
            if item.is_empty() || item.contains('{') || item == "self" {
                continue;
            }
            if let Some(name) = rust_leaf(item) {
                out.push(Import {
                    name,
                    module: rust_module(prefix),
                });
            }
        }
        return;
    }

    let segments: Vec<&str> = rest.split("::").map(str::trim).collect();
    if segments.len() < 2 {
        return;
    }
    let Some(name) = rust_leaf(segments[segments.len() - 1]) else {
        return;
    };
    out.push(Import {
        name,
        module: rust_module(&segments[..segments.len() - 1].join("::")),
    });
}

/// The bound name of one `use` item, honouring `as` aliases. `*` binds nothing.
fn rust_leaf(item: &str) -> Option<String> {
    let name = match item.split_once(" as ") {
        Some((_, alias)) => alias.trim(),
        None => item.trim(),
    };
    if name.is_empty() || name == "*" || name == "self" {
        return None;
    }
    Some(name.to_string())
}

/// `crate::a::b` → `a/b`. The crate-relative prefixes say nothing about which
/// file a symbol is in, so they are dropped rather than matched literally.
fn rust_module(path: &str) -> String {
    path.split("::")
        .map(str::trim)
        .filter(|s| !s.is_empty() && !matches!(*s, "crate" | "self" | "super"))
        .collect::<Vec<_>>()
        .join("/")
}

/// `from a.b import C, D` · `import a.b` · `import a.b as c`
fn python_import(line: &str, out: &mut Vec<Import>) {
    if let Some(rest) = line.strip_prefix("from ") {
        let Some((module, names)) = rest.split_once(" import ") else {
            return;
        };
        // `from . import x` / `from .mod import x` — leading dots are relative
        // markers, not path components.
        let module = module.trim().trim_start_matches('.').replace('.', "/");
        for item in names
            .trim()
            .trim_matches(|c| c == '(' || c == ')')
            .split(',')
        {
            let item = item.trim();
            if item.is_empty() || item == "*" {
                continue;
            }
            let name = match item.split_once(" as ") {
                Some((_, alias)) => alias.trim(),
                None => item,
            };
            out.push(Import {
                name: name.to_string(),
                module: module.clone(),
            });
        }
        return;
    }
    if let Some(rest) = line.strip_prefix("import ") {
        for item in rest.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            let (path, alias) = match item.split_once(" as ") {
                Some((p, a)) => (p.trim(), Some(a.trim())),
                None => (item, None),
            };
            let module = path.replace('.', "/");
            let name = alias
                .map(str::to_string)
                .or_else(|| path.rsplit('.').next().map(str::to_string));
            if let Some(name) = name {
                out.push(Import { name, module });
            }
        }
    }
}

/// `import { a, b } from './x'` · `import a from './x'` · `require('./x')`
fn js_import(line: &str, out: &mut Vec<Import>) {
    let module = js_module(line);
    let Some(module) = module else { return };

    if let Some(open) = line.find('{') {
        if let Some(close) = line[open..].find('}') {
            for item in line[open + 1..open + close].split(',') {
                let item = item.trim();
                if item.is_empty() {
                    continue;
                }
                let name = match item.split_once(" as ") {
                    Some((_, alias)) => alias.trim(),
                    None => item,
                };
                out.push(Import {
                    name: name.to_string(),
                    module: module.clone(),
                });
            }
            return;
        }
    }
    // Default or namespace import: `import a from 'x'`, `import * as a from 'x'`.
    if let Some(rest) = line.strip_prefix("import ") {
        let head = rest.split(" from ").next().unwrap_or("").trim();
        let name = head.rsplit(" as ").next().unwrap_or(head).trim();
        if !name.is_empty() && name != "*" && !name.contains(['{', '\'', '"']) {
            out.push(Import {
                name: name.to_string(),
                module,
            });
        }
    }
}

/// The quoted specifier of an import/require line, normalised to a path hint.
fn js_module(line: &str) -> Option<String> {
    if !(line.starts_with("import ") || line.contains("require(")) {
        return None;
    }
    let start = line.find(['\'', '"'])?;
    let quote = line.as_bytes()[start] as char;
    let end = line[start + 1..].find(quote)? + start + 1;
    let spec = &line[start + 1..end];
    // `./x`, `../x` — relative markers carry no path information we can match.
    let spec = spec.trim_start_matches("./").replace("../", "");
    let spec = spec.trim_end_matches(".js").trim_end_matches(".ts");
    Some(spec.to_string())
}

/// `import "path/to/pkg"` — Go binds the last component as the package name.
fn go_import(line: &str, out: &mut Vec<Import>) {
    let quoted = line.trim().trim_start_matches("import ").trim();
    let quoted = quoted.trim_matches('"');
    if quoted.is_empty() || quoted.contains(' ') || !quoted.contains('/') {
        return;
    }
    if let Some(name) = quoted.rsplit('/').next() {
        out.push(Import {
            name: name.to_string(),
            module: quoted.to_string(),
        });
    }
}

#[cfg(test)]
mod import_tests {
    use super::*;

    fn names(language: &str, src: &str) -> Vec<(String, String)> {
        extract_imports(language, src)
            .into_iter()
            .map(|i| (i.name, i.module))
            .collect()
    }

    #[test]
    fn rust_use_binds_leaf_to_its_module() {
        let got = names("rust", "use crate::de::value::Error;\n");
        assert_eq!(got, vec![("Error".into(), "de/value".into())]);
    }

    #[test]
    fn rust_brace_group_binds_every_name_to_one_module() {
        let mut got = names("rust", "use serde::de::{Visitor, MapAccess};\n");
        got.sort();
        assert_eq!(
            got,
            vec![
                ("MapAccess".into(), "serde/de".into()),
                ("Visitor".into(), "serde/de".into()),
            ]
        );
    }

    #[test]
    fn rust_alias_binds_the_alias_not_the_original() {
        let got = names("rust", "use std::io::Error as IoError;\n");
        assert_eq!(got, vec![("IoError".into(), "std/io".into())]);
    }

    #[test]
    fn rust_glob_and_self_bind_nothing() {
        assert!(names("rust", "use foo::*;\nuse foo::self;\n").is_empty());
    }

    #[test]
    fn python_from_import_binds_each_name() {
        let mut got = names("python", "from flask.app import Flask, Request\n");
        got.sort();
        assert_eq!(
            got,
            vec![
                ("Flask".into(), "flask/app".into()),
                ("Request".into(), "flask/app".into()),
            ]
        );
    }

    #[test]
    fn python_relative_import_drops_the_dots() {
        let got = names("python", "from .globals import request\n");
        assert_eq!(got, vec![("request".into(), "globals".into())]);
    }

    #[test]
    fn python_plain_import_binds_the_last_component() {
        let got = names("python", "import os.path\n");
        assert_eq!(got, vec![("path".into(), "os/path".into())]);
    }

    #[test]
    fn js_named_import_binds_each_specifier() {
        let mut got = names(
            "typescript",
            "import { parse, format } from './util/date';\n",
        );
        got.sort();
        assert_eq!(
            got,
            vec![
                ("format".into(), "util/date".into()),
                ("parse".into(), "util/date".into()),
            ]
        );
    }

    #[test]
    fn js_default_import_binds_the_local_name() {
        let got = names("javascript", "import express from './express';\n");
        assert_eq!(got, vec![("express".into(), "express".into())]);
    }

    #[test]
    fn a_file_with_no_imports_yields_none() {
        assert!(names("rust", "fn main() {}\n").is_empty());
        assert!(names("python", "def main():\n    pass\n").is_empty());
    }

    #[test]
    fn parsing_a_file_attaches_its_imports() {
        let f = HeuristicParser
            .parse("src/a.rs", "rust", "use crate::b::Thing;\nfn go() {}\n")
            .unwrap();
        assert_eq!(f.imports.len(), 1);
        assert_eq!(f.imports[0].name, "Thing");
    }
}

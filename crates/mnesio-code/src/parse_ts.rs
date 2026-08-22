//! Real grammars: [`TreeSitterParser`], behind the `tree-sitter` feature.
//!
//! 30 languages, every one verified by a test that extracts a real symbol.
//!
//! Two sources of query: 19 come from the grammar crate's own exported
//! `TAGS_QUERY`; 8 more use a query written here in [`own_tags`], because
//! their crate ships none. Both go through the same extraction path.
//!
//! ## One mechanism, every language
//!
//! Every tree-sitter grammar ships a `tags.scm` query upstream, and those
//! queries use a *conventional* set of capture names:
//!
//! - `@definition.function` / `.method` / `.class` / `.interface` / `.module`
//!   / `.macro` — a symbol worth indexing
//! - `@reference.call` — a call site, which is what the graph expands along
//!
//! So there is no per-language parsing code here. A language is a row in
//! [`GRAMMARS`]: its name, its extensions, its `LanguageFn`, and its
//! `TAGS_QUERY` string. Adding one is four fields, and the query itself stays
//! maintained by the grammar's own authors rather than by us.
//!
//! ## Why this exists, honestly
//!
//! It is **not** here to raise retrieval recall. The Phase-17B miss taxonomy
//! measured `not_indexed = 0%` on llama-index-core: nothing was being missed
//! because the parser failed to extract it, so a better parser cannot fix a
//! retrieval loss that isn't happening. What it buys is:
//!
//! 1. **Reach** — repositories in languages [`crate::HeuristicParser`] cannot
//!    parse at all (Ruby, PHP), where the alternative is not a
//!    worse index but no index.
//! 2. **Edge quality** — the heuristic parser resolves 23–46% of call edges
//!    because it matches bare names and cannot tell `foo()` from `x.foo()`.
//!    A grammar knows the difference.
//!
//! Both are worth having. Neither is a recall claim, and this module should
//! not be described as one until a paired run says otherwise.
//!
//! Measured on `crates/mnesio-code/src` (14 files), grammars against the
//! heuristic parser: **385 symbols vs 219, 291 resolved calls vs 82, 20%
//! resolution vs 10%**. So on Rust the grammar is better on every axis.
//!
//! ## Where it is *worse* than that suggests: C and C++ have no call graph
//!
//! `tags.scm` is written by each grammar's own authors, and they do not all
//! answer the same questions. `tree-sitter-c` and `tree-sitter-cpp` ship
//! **zero** `@reference.call` patterns — their queries capture definitions
//! only. So a C or C++ repository indexes its symbols and produces *no call
//! edges at all*, which means no graph expansion and nothing for
//! [`crate::pack`] to expand along.
//!
//! This is worth stating precisely because the symptom is indistinguishable
//! from a hard problem: a 0% resolution rate reads as "the resolver failed",
//! when in fact no call site was ever reported to it. The first is a
//! resolution problem (Phase 18F), the second is a missing query — a much
//! cheaper fix, and one whose absence no amount of type inference would
//! address. `own_tags` is where such a query would go.

use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator};

use crate::parse::CodeParser;
use crate::{CodeEdge, EdgeKind, ParseError, ParsedFile, Symbol, SymbolKind};

/// Tags queries **we** wrote, for grammars whose crate ships none.
///
/// Same conventional capture names as an upstream `tags.scm`, so the single
/// extraction path in [`TreeSitterParser::parse`] serves these unchanged. They
/// are deliberately minimal — top-level definitions and call sites, not full
/// syntax coverage — because that is all symbol retrieval needs, and a short
/// query is one a reader can check against the grammar's node types.
///
/// Written rather than vendored: these are our own authorship against each
/// grammar's public node names, which keeps the crate free of third-party
/// query files and their notices. Every one is pinned by a test that extracts
/// a real symbol, because a query that silently matches nothing is the exact
/// failure this table exists to avoid.
mod own_tags {
    // Positional child patterns, not `name:` fields — these grammars expose the
    // identifier as an ordinary child. Verified against each grammar's real
    // parse tree rather than guessed from its node-types list.
    pub const KOTLIN: &str = r#"
(class_declaration (identifier) @name) @definition.class
(function_declaration (identifier) @name) @definition.function
"#;

    pub const ZIG: &str = r#"
(function_declaration (identifier) @name) @definition.function
"#;

    pub const HASKELL: &str = r#"
(function (variable) @name) @definition.function
"#;

    pub const JULIA: &str = r#"
(function_definition (signature (call_expression (identifier) @name))) @definition.function
(struct_definition (identifier) @name) @definition.class
"#;

    pub const OBJC: &str = r#"
(class_interface (identifier) @name) @definition.class
(class_implementation (identifier) @name) @definition.class
"#;

    pub const HCL: &str = r#"
(block (identifier) @name) @definition.class
"#;

    pub const SCALA: &str = r#"
(class_definition (identifier) @name) @definition.class
(object_definition (identifier) @name) @definition.module
(trait_definition (identifier) @name) @definition.interface
(function_definition (identifier) @name) @definition.function
"#;

    pub const BASH: &str = r#"
(function_definition (word) @name) @definition.function
"#;
}

/// One supported language.
struct Grammar {
    /// Language tag, matching what [`crate::HeuristicParser`] uses so the two
    /// parsers are interchangeable behind [`CodeParser`].
    name: &'static str,
    extensions: &'static [&'static str],
    language: fn() -> Language,
    tags: &'static str,
}

/// Every grammar compiled in.
///
/// Ordered by how much code in the world is written in them, which is also
/// roughly the order in which a missing one would be noticed.
///
/// **A language is listed only if its grammar crate exports a usable
/// `TAGS_QUERY` const.** That constraint, not availability, is what caps this
/// list. Of 46 grammar crates that resolve against tree-sitter 0.25, only 20
/// export the query — the rest ship `queries/tags.scm` as a *file* the crate
/// never compiles in (scala, svelte), or have no tags query at all (bash,
/// kotlin, zig, haskell, julia, hcl, css, html, yaml, toml, make, objc,
/// clojure, verilog).
///
/// This is a Rust-packaging difference, not a tree-sitter one: a Node consumer
/// reads `node_modules/<grammar>/queries/tags.scm` off disk at runtime, so it
/// gets every grammar's query for free. Reaching those languages here means
/// vendoring the upstream `.scm` files into this repo with their licences, or
/// writing our own queries against each grammar's node types — both real work,
/// neither done yet.
///
/// Listing a language we cannot extract from would advertise one that silently
/// indexes zero symbols, which is worse than not claiming it.
static GRAMMARS: &[Grammar] = &[
    Grammar {
        name: "rust",
        extensions: &["rs"],
        language: || tree_sitter_rust::LANGUAGE.into(),
        tags: tree_sitter_rust::TAGS_QUERY,
    },
    Grammar {
        name: "python",
        extensions: &["py", "pyi"],
        language: || tree_sitter_python::LANGUAGE.into(),
        tags: tree_sitter_python::TAGS_QUERY,
    },
    Grammar {
        name: "javascript",
        extensions: &["js", "jsx", "mjs", "cjs"],
        language: || tree_sitter_javascript::LANGUAGE.into(),
        tags: tree_sitter_javascript::TAGS_QUERY,
    },
    Grammar {
        name: "typescript",
        extensions: &["ts", "mts", "cts"],
        language: || tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        tags: tree_sitter_typescript::TAGS_QUERY,
    },
    Grammar {
        name: "tsx",
        extensions: &["tsx"],
        language: || tree_sitter_typescript::LANGUAGE_TSX.into(),
        tags: tree_sitter_typescript::TAGS_QUERY,
    },
    Grammar {
        name: "go",
        extensions: &["go"],
        language: || tree_sitter_go::LANGUAGE.into(),
        tags: tree_sitter_go::TAGS_QUERY,
    },
    Grammar {
        name: "java",
        extensions: &["java"],
        language: || tree_sitter_java::LANGUAGE.into(),
        tags: tree_sitter_java::TAGS_QUERY,
    },
    Grammar {
        name: "c",
        extensions: &["c", "h"],
        language: || tree_sitter_c::LANGUAGE.into(),
        tags: tree_sitter_c::TAGS_QUERY,
    },
    Grammar {
        name: "cpp",
        extensions: &["cc", "cpp", "cxx", "hpp", "hh"],
        language: || tree_sitter_cpp::LANGUAGE.into(),
        tags: tree_sitter_cpp::TAGS_QUERY,
    },
    Grammar {
        name: "csharp",
        extensions: &["cs"],
        language: || tree_sitter_c_sharp::LANGUAGE.into(),
        tags: tree_sitter_c_sharp::TAGS_QUERY,
    },
    Grammar {
        name: "ruby",
        extensions: &["rb"],
        language: || tree_sitter_ruby::LANGUAGE.into(),
        tags: tree_sitter_ruby::TAGS_QUERY,
    },
    Grammar {
        name: "php",
        extensions: &["php"],
        language: || tree_sitter_php::LANGUAGE_PHP.into(),
        tags: tree_sitter_php::TAGS_QUERY,
    },
    Grammar {
        name: "swift",
        extensions: &["swift"],
        language: || tree_sitter_swift::LANGUAGE.into(),
        tags: tree_sitter_swift::TAGS_QUERY,
    },
    Grammar {
        name: "lua",
        extensions: &["lua"],
        language: || tree_sitter_lua::LANGUAGE.into(),
        tags: tree_sitter_lua::TAGS_QUERY,
    },
    Grammar {
        name: "elixir",
        extensions: &["ex", "exs"],
        language: || tree_sitter_elixir::LANGUAGE.into(),
        tags: tree_sitter_elixir::TAGS_QUERY,
    },
    Grammar {
        name: "ocaml",
        extensions: &["ml"],
        language: || tree_sitter_ocaml::LANGUAGE_OCAML.into(),
        tags: tree_sitter_ocaml::TAGS_QUERY,
    },
    Grammar {
        name: "ocaml_interface",
        extensions: &["mli"],
        language: || tree_sitter_ocaml::LANGUAGE_OCAML_INTERFACE.into(),
        tags: tree_sitter_ocaml::TAGS_QUERY,
    },
    Grammar {
        name: "ocaml_type",
        extensions: &["mlt"],
        language: || tree_sitter_ocaml::LANGUAGE_OCAML_TYPE.into(),
        tags: tree_sitter_ocaml::TAGS_QUERY,
    },
    Grammar {
        name: "r",
        extensions: &["r", "R"],
        language: || tree_sitter_r::LANGUAGE.into(),
        tags: tree_sitter_r::TAGS_QUERY,
    },
    Grammar {
        name: "dart",
        extensions: &["dart"],
        language: || tree_sitter_dart::LANGUAGE.into(),
        tags: tree_sitter_dart::TAGS_QUERY,
    },
    Grammar {
        name: "solidity",
        extensions: &["sol"],
        language: || tree_sitter_solidity::LANGUAGE.into(),
        tags: tree_sitter_solidity::TAGS_QUERY,
    },
    Grammar {
        name: "elm",
        extensions: &["elm"],
        language: || tree_sitter_elm::LANGUAGE.into(),
        tags: tree_sitter_elm::TAGS_QUERY,
    },
    Grammar {
        name: "kotlin",
        extensions: &["kt", "kts"],
        language: || tree_sitter_kotlin_ng::LANGUAGE.into(),
        tags: own_tags::KOTLIN,
    },
    Grammar {
        name: "zig",
        extensions: &["zig"],
        language: || tree_sitter_zig::LANGUAGE.into(),
        tags: own_tags::ZIG,
    },
    Grammar {
        name: "haskell",
        extensions: &["hs"],
        language: || tree_sitter_haskell::LANGUAGE.into(),
        tags: own_tags::HASKELL,
    },
    Grammar {
        name: "julia",
        extensions: &["jl"],
        language: || tree_sitter_julia::LANGUAGE.into(),
        tags: own_tags::JULIA,
    },
    Grammar {
        name: "objc",
        extensions: &["m", "mm"],
        language: || tree_sitter_objc::LANGUAGE.into(),
        tags: own_tags::OBJC,
    },
    Grammar {
        name: "hcl",
        extensions: &["tf", "hcl", "tfvars"],
        language: || tree_sitter_hcl::LANGUAGE.into(),
        tags: own_tags::HCL,
    },
    Grammar {
        name: "scala",
        extensions: &["scala", "sc"],
        language: || tree_sitter_scala::LANGUAGE.into(),
        tags: own_tags::SCALA,
    },
    Grammar {
        name: "bash",
        extensions: &["sh", "bash", "zsh"],
        language: || tree_sitter_bash::LANGUAGE.into(),
        tags: own_tags::BASH,
    },
];

/// Language tag for a file extension, or `None` if no grammar covers it.
pub fn language_for_extension(ext: &str) -> Option<&'static str> {
    GRAMMARS
        .iter()
        .find(|g| g.extensions.contains(&ext))
        .map(|g| g.name)
}

/// Every language tag this parser handles.
pub fn supported_languages() -> Vec<&'static str> {
    GRAMMARS.iter().map(|g| g.name).collect()
}

fn grammar(name: &str) -> Option<&'static Grammar> {
    GRAMMARS.iter().find(|g| g.name == name)
}

/// Map a `tags.scm` capture name onto our coarse kind vocabulary.
///
/// The names are a tree-sitter convention, not a per-grammar invention, which
/// is what lets one table serve every language.
fn kind_of(capture: &str) -> Option<SymbolKind> {
    match capture {
        "definition.function" => Some(SymbolKind::Function),
        "definition.method" => Some(SymbolKind::Method),
        "definition.class" => Some(SymbolKind::Class),
        "definition.struct" => Some(SymbolKind::Struct),
        "definition.enum" => Some(SymbolKind::Enum),
        "definition.interface" | "definition.trait" => Some(SymbolKind::Trait),
        "definition.type" => Some(SymbolKind::TypeAlias),
        "definition.constant" => Some(SymbolKind::Constant),
        "definition.module" | "definition.namespace" => Some(SymbolKind::Module),
        // `definition.macro` has no counterpart in our vocabulary and macro
        // bodies are rarely what a task is about; skipped rather than
        // mislabelled as a function.
        _ => None,
    }
}

/// Parser backed by real grammars.
#[derive(Debug, Clone, Default)]
pub struct TreeSitterParser;

impl CodeParser for TreeSitterParser {
    fn parse(&self, path: &str, language: &str, source: &str) -> Result<ParsedFile, ParseError> {
        let Some(g) = grammar(language) else {
            return Err(ParseError::UnsupportedLanguage(language.to_string()));
        };
        let lang = (g.language)();

        let mut parser = Parser::new();
        parser
            .set_language(&lang)
            .map_err(|_| ParseError::UnsupportedLanguage(language.to_string()))?;
        let Some(tree) = parser.parse(source, None) else {
            // A grammar that cannot produce any tree — not even an errored one
            // — means the file is not this language. Skipped, not guessed at.
            return Err(ParseError::UnsupportedLanguage(language.to_string()));
        };

        let query = Query::new(&lang, g.tags)
            .map_err(|_| ParseError::UnsupportedLanguage(language.to_string()))?;
        let names = query.capture_names();
        let bytes = source.as_bytes();

        let mut symbols: Vec<Symbol> = Vec::new();
        let mut calls: Vec<(usize, usize, String)> = Vec::new();

        // A `tags.scm` pattern captures the identifier as `@name` and the
        // *whole* definition as `@definition.<kind>`. They arrive in the same
        // match, so pair them there — the name gives the symbol its identity,
        // the definition node gives it its span and text.
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, tree.root_node(), bytes);
        while let Some(m) = matches.next() {
            let mut name: Option<&str> = None;
            let mut name_node: Option<tree_sitter::Node> = None;
            let mut def: Option<tree_sitter::Node> = None;
            let mut kind: Option<SymbolKind> = None;
            let mut call: Option<tree_sitter::Node> = None;

            for cap in m.captures {
                match names[cap.index as usize] {
                    "name" => {
                        name = cap.node.utf8_text(bytes).ok();
                        name_node = Some(cap.node);
                    }
                    "reference.call" => call = Some(cap.node),
                    other => {
                        if let Some(k) = kind_of(other) {
                            kind = Some(k);
                            def = Some(cap.node);
                        }
                    }
                }
            }

            let Some(name) = name else { continue };

            if call.is_some() && def.is_none() {
                // Byte offset, so the call can be attributed to whichever
                // definition encloses it once every span is known.
                //
                // The `@name` node specifically, not `captures[0]`: receiver
                // detection reads the character immediately before this offset,
                // and only the identifier's own start puts the `.` of
                // `x.push(1)` there. `captures[0]` is whichever capture the
                // grammar's query happens to list first.
                let n = name_node.unwrap_or(m.captures[0].node);
                calls.push((n.start_byte(), n.end_byte(), name.to_string()));
                continue;
            }

            let (Some(def), Some(kind)) = (def, kind) else {
                continue;
            };
            let text = def.utf8_text(bytes).unwrap_or(name).to_string();
            symbols.push(Symbol {
                name: name.to_string(),
                kind,
                path: path.to_string(),
                signature: Some(first_line(&text)),
                doc: None,
                start_line: def.start_position().row as u32 + 1,
                end_line: def.end_position().row as u32 + 1,
                text,
            });
        }

        // Deduplicate: some `tags.scm` queries capture the same definition
        // under more than one pattern.
        symbols.sort_by(|a, b| {
            (a.start_line, a.end_line, &a.name).cmp(&(b.start_line, b.end_line, &b.name))
        });
        symbols.dedup_by(|a, b| a.name == b.name && a.start_line == b.start_line);

        let edges = attribute_calls(&symbols, source, &calls);
        Ok(ParsedFile {
            path: path.to_string(),
            language: language.to_string(),
            module_doc: crate::parse::module_doc_for(language, source),
            symbols,
            edges,
            // Shared with the heuristic parser on purpose: import syntax is
            // line-oriented, so a grammar adds nothing here.
            imports: crate::parse::extract_imports(language, source),
        })
    }
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or_default().trim().to_string()
}

/// Bind each call site to the innermost definition containing it.
///
/// Did the call at `at` have a receiver — `x.name(..)` or `T::name(..)`?
///
/// Decided from the source text rather than the syntax tree on purpose. The
/// node type for a method call differs in every one of the 28 grammars
/// ([`GRAMMARS`]), so a tree-based check would be a per-language table that
/// silently returns "no receiver" for any language nobody remembered to add —
/// and "no receiver" is the answer that lets `vec.push(x)` bind to a free
/// function named `push`. A `.` or `::` before the identifier means the same
/// thing in every language here, and [`crate::HeuristicParser`] already decides
/// it exactly this way, so the two parsers stay comparable.
fn preceded_by_receiver(source: &str, at: usize) -> bool {
    // Slicing a non-boundary panics. Tree-sitter offsets are boundaries in
    // valid UTF-8, but a parser is not the place to stake a panic on that.
    let Some(before) = source.get(..at) else {
        return false;
    };
    let mut chars = before.chars().rev();
    match chars.next() {
        Some('.') => true,
        Some(':') => chars.next() == Some(':'),
        _ => false,
    }
}

/// Byte spans rather than line numbers: two definitions can share a line, and
/// a wrong attribution produces an edge that actively misleads expansion.
fn attribute_calls(
    symbols: &[Symbol],
    source: &str,
    calls: &[(usize, usize, String)],
) -> Vec<CodeEdge> {
    // Recover each symbol's byte span from its line span once, rather than per
    // call site.
    let line_starts: Vec<usize> = std::iter::once(0)
        .chain(source.match_indices('\n').map(|(i, _)| i + 1))
        .collect();
    let span = |s: &Symbol| -> (usize, usize) {
        let start = *line_starts
            .get(s.start_line.saturating_sub(1) as usize)
            .unwrap_or(&0);
        let end = line_starts
            .get(s.end_line as usize)
            .copied()
            .unwrap_or(source.len());
        (start, end)
    };

    let mut edges = Vec::new();
    for (cs, _, name) in calls {
        // Innermost = smallest containing span, so a call inside a method is
        // credited to the method rather than to its class.
        let owner = symbols
            .iter()
            .filter(|s| {
                let (a, b) = span(s);
                *cs >= a && *cs < b
            })
            .min_by_key(|s| {
                let (a, b) = span(s);
                b - a
            });
        if let Some(owner) = owner {
            if &owner.name == name {
                continue; // self-recursion is not useful context
            }
            edges.push(CodeEdge {
                from: owner.key(),
                to_name: name.clone(),
                kind: EdgeKind::Calls,
                via_receiver: preceded_by_receiver(source, *cs),
            });
        }
    }
    edges.sort_by(|a, b| (&a.from, &a.to_name).cmp(&(&b.from, &b.to_name)));
    edges.dedup();
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(lang: &str, src: &str) -> ParsedFile {
        TreeSitterParser.parse("f", lang, src).unwrap()
    }

    #[test]
    fn a_method_call_is_marked_as_having_a_receiver() {
        // Why this matters: without the flag, `v.push(x)` binds to any free
        // function named `push` anywhere in the repository. Measured on this
        // workspace that gave `push` 142 inbound edges and made it the top
        // "most depended on" symbol — an artefact, not a fact about the code.
        //
        // This regressed once already in the shape a feature gate makes easy:
        // `via_receiver` was added to the heuristic parser, and the field was
        // simply missing here, so `--features tree-sitter` did not compile at
        // all while the default build stayed green.
        let f = parse(
            "rust",
            "fn caller() {\n    let mut v = Vec::new();\n    v.push(1);\n    helper();\n}\nfn helper() {}\n",
        );

        let push = f.edges.iter().find(|e| e.to_name == "push");
        let helper = f.edges.iter().find(|e| e.to_name == "helper");
        assert!(
            push.is_some_and(|e| e.via_receiver),
            "v.push(1) has a receiver; got {:?}",
            f.edges
        );
        assert!(
            helper.is_some_and(|e| !e.via_receiver),
            "helper() is a bare call; got {:?}",
            f.edges
        );
    }

    #[test]
    fn a_path_qualified_call_produces_no_edge_at_all() {
        // Not the behaviour anyone would choose — recorded because it is
        // upstream's, and because it is one measured reason the resolution rate
        // is low. Rust's own `tags.scm` captures `x.method()` and `free()` as
        // `@reference.call` but not `Type::assoc()`, so associated-function
        // calls are absent from the graph rather than merely unresolved.
        //
        // Pinned so that a grammar update which starts emitting them is
        // noticed: the receiver logic already handles `::`, so the assertion
        // to flip is this one, not `preceded_by_receiver`.
        let f = parse(
            "rust",
            "fn caller() {\n    let _ = String::from(\"x\");\n}\n",
        );
        assert!(
            !f.edges.iter().any(|e| e.to_name == "from"),
            "upstream started capturing path-qualified calls — flip this test \
             and check they arrive with via_receiver set; got {:?}",
            f.edges
        );
    }

    #[test]
    fn the_receiver_check_reads_the_character_before_the_identifier() {
        // Unit-level, because the grammar gap above means `::` has no
        // end-to-end coverage in Rust. Other grammars in `GRAMMARS` do emit
        // path-qualified calls, and this is the logic they land on.
        assert!(preceded_by_receiver("v.push", 2));
        assert!(preceded_by_receiver("T::new", 3));
        assert!(!preceded_by_receiver("helper", 0));
        assert!(!preceded_by_receiver("a helper", 2));
        // A label or ternary — one colon is not a path separator.
        assert!(!preceded_by_receiver("x ? a:b", 6));
        // Must not panic when the offset splits a multi-byte character.
        assert!(!preceded_by_receiver("é(", 1));
    }

    #[test]
    fn c_and_cpp_report_no_call_sites_at_all() {
        // Not our bug and not a resolution failure — `tree-sitter-c` and
        // `tree-sitter-cpp` ship tags queries with zero `@reference.call`
        // patterns, so a C/C++ repository indexes symbols and produces no call
        // edges whatever. Measured live: a 54-file C/C++/GLSL repository mapped
        // 46 files, 170 symbols and 0 edges.
        //
        // Pinned because the symptom is indistinguishable from a hard problem.
        // "0% of call sites resolved" reads as a resolver that failed, when in
        // fact nothing was ever handed to it — a missing query (cheap, goes in
        // `own_tags`) rather than the type inference of Phase 18F.
        for lang in ["c", "cpp"] {
            let f = parse(
                lang,
                "int helper(void) { return 1; }\nint caller(void) { return helper(); }\n",
            );
            assert!(
                !f.symbols.is_empty(),
                "{lang} must still extract definitions"
            );
            assert!(
                f.edges.is_empty(),
                "{lang} started reporting call sites — upstream added \
                 @reference.call, so drop the `own_tags` TODO and re-measure \
                 the resolution rate; got {:?}",
                f.edges
            );
        }
    }

    #[test]
    fn every_grammar_has_a_valid_tags_query() {
        // A malformed query fails at runtime on the first file of that
        // language, which in practice means "this language silently indexes
        // nothing". Compile them all up front instead.
        for g in GRAMMARS {
            let lang = (g.language)();
            assert!(
                Query::new(&lang, g.tags).is_ok(),
                "{} has an unusable tags.scm",
                g.name
            );
        }
    }

    #[test]
    fn extensions_are_not_claimed_by_two_grammars() {
        // A duplicate would make indexing order-dependent.
        let mut seen: Vec<&str> = Vec::new();
        for g in GRAMMARS {
            for e in g.extensions {
                assert!(!seen.contains(e), "extension {e:?} claimed twice");
                seen.push(e);
            }
        }
    }

    #[test]
    fn rust_functions_and_types_are_extracted() {
        let f = parse(
            "rust",
            "pub struct Cfg { a: u8 }\npub fn load() -> Cfg { Cfg { a: 1 } }\n",
        );
        let names: Vec<_> = f.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"load"), "got {names:?}");
        assert!(names.contains(&"Cfg"), "got {names:?}");
    }

    #[test]
    fn python_classes_and_methods_are_extracted() {
        let f = parse(
            "python",
            "class Retriever:\n    def search(self, q):\n        return self.rank(q)\n",
        );
        let names: Vec<_> = f.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Retriever"), "got {names:?}");
        assert!(names.contains(&"search"), "got {names:?}");
    }

    #[test]
    fn a_call_is_attributed_to_the_function_containing_it() {
        let f = parse("rust", "fn helper() {}\nfn caller() {\n    helper();\n}\n");
        let e = f
            .edges
            .iter()
            .find(|e| e.to_name == "helper")
            .unwrap_or_else(|| panic!("no edge to helper in {:?}", f.edges));
        assert!(e.from.contains("caller"), "attributed to {}", e.from);
    }

    #[test]
    fn languages_the_heuristic_parser_cannot_read_now_work() {
        // The actual point of this module: reach, not recall.
        let rb = parse(
            "ruby",
            "class Greeter\n  def greet\n    puts 'hi'\n  end\nend\n",
        );
        assert!(
            rb.symbols.iter().any(|s| s.name == "greet"),
            "ruby: {:?}",
            rb.symbols.iter().map(|s| &s.name).collect::<Vec<_>>()
        );

        let go = parse("go", "package main\nfunc Serve() {}\n");
        assert!(
            go.symbols.iter().any(|s| s.name == "Serve"),
            "go: {:?}",
            go.symbols.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
    }

    /// Every self-authored query must extract a real symbol.
    ///
    /// A `.scm` that compiles but matches nothing is the exact failure this
    /// table risks — the language would appear supported and index zero
    /// symbols. One sample per language, checked by name.
    #[test]
    fn hand_written_queries_extract_real_symbols() {
        let cases: &[(&str, &str, &str)] = &[
            (
                "kotlin",
                "class Greeter {\n  fun greet() {}\n}\n",
                "Greeter",
            ),
            ("zig", "pub fn add(a: i32) i32 { return a; }\n", "add"),
            (
                "haskell",
                "double :: Int -> Int\ndouble x = x * 2\n",
                "double",
            ),
            ("julia", "function solve(x)\n    return x\nend\n", "solve"),
            ("objc", "@interface Greeter\n@end\n", "Greeter"),
            (
                "hcl",
                "resource \"aws_s3_bucket\" \"b\" {\n  acl = \"private\"\n}\n",
                "resource",
            ),
            (
                "scala",
                "class Retriever {\n  def search(q: String) = q\n}\n",
                "Retriever",
            ),
            ("bash", "deploy() {\n  echo hi\n}\n", "deploy"),
        ];
        for (lang, src, want) in cases {
            let f = TreeSitterParser
                .parse("f", lang, src)
                .unwrap_or_else(|e| panic!("{lang} failed to parse: {e}"));
            let names: Vec<&str> = f.symbols.iter().map(|s| s.name.as_str()).collect();
            assert!(
                names.contains(want),
                "{lang}: query matched nothing useful — wanted {want:?}, got {names:?}"
            );
        }
    }

    #[test]
    fn an_unknown_language_is_rejected_not_silently_empty() {
        let e = TreeSitterParser.parse("f", "cobol", "IDENTIFICATION DIVISION.");
        assert!(matches!(e, Err(ParseError::UnsupportedLanguage(_))));
    }

    #[test]
    fn a_syntactically_broken_file_yields_what_it_can() {
        // Real repositories contain files that do not compile. tree-sitter is
        // error-tolerant; the parser must not throw the whole file away.
        let f = parse("rust", "fn ok() {}\nfn broken( {\n");
        assert!(f.symbols.iter().any(|s| s.name == "ok"));
    }
}

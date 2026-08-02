//! Fit the most useful code into a token budget.
//!
//! Retrieval returns a ranked list; an agent has a *context budget*. This is
//! the step between them, and it is where Phase 17B's measurements cash out.
//!
//! ## What the measurements decided
//!
//! Every choice here is a measured one, not a guess (`mnesio-bench codeeval`,
//! git-derived suites over four real repositories):
//!
//! - **Expand one hop along `Calls`.** On llama-index-core (400 queries)
//!   expansion added +3–4pp recall at *every* `k`, and it repeated on
//!   claw-code (+3–6pp) and tare (+11pp). An earlier hand-written suite said
//!   expansion earned nothing; that suite's answers were all type definitions
//!   with nothing to expand *to*.
//! - **Budget beats top-`k` as the thing to optimise.** At 4k tokens on
//!   llama-index-core, symbol-level context reached 44% where a whole-file
//!   strategy managed 26% — it could not fit enough files to compete. The
//!   agent's real constraint is tokens, so that is what this packs against.
//! - **Module prose must never displace a symbol.** Two earlier attempts to
//!   give file-level documentation an owner both *regressed* recall: as its
//!   own memory it competed for retrieval slots (recall@1 50%→25%), and as a
//!   per-symbol breadcrumb it was constant within a file and so carried no
//!   discriminating information (peak 88%→62%). Here it is added **last, from
//!   whatever budget is left over**, which is the one position where it cannot
//!   cost a symbol its place. See [`PackedContext::notes`].
//!
//! ## The degradation ladder, and what it is *not* worth
//!
//! A symbol that does not fit whole is not simply dropped: [`Form::Signature`]
//! keeps its declaration. `Symbol::signature` was captured at parse time for
//! exactly this.
//!
//! **Measured honestly, this buys less than it first appears.** Ablating the
//! policies at a fixed budget on llama-index-core (400 git-derived queries),
//! scored two ways — `any` counts a signature-only inclusion, `full` demands
//! the body:
//!
//! | budget | policy | recall (any) | recall (full) | tokens |
//! |---|---|---|---|---|
//! | 2k | truncate | 39% | 39% | 1802 |
//! | 2k | +signature | **51%** | 39% | 1817 |
//! | 2k | +expand | 53% | 40% | 1908 |
//! | 16k | truncate | 52% | 52% | 5418 |
//! | 16k | +expand | 56% | 56% | 9163 |
//!
//! The signature ladder's headline +12pp is **entirely declarations**: on
//! `full` it adds exactly nothing. It is worth keeping — it costs ~0.8% more
//! tokens and tells an agent a symbol exists and what it takes, which is
//! enough to decide what to ask for next — but it must never be quoted as a
//! recall win.
//!
//! Expansion is the only policy that moves `full` recall, and modestly: +1 to
//! +4pp for 5–69% more tokens. File notes move neither, by construction — a
//! note can never *be* a gold symbol. Their value is orientation, which this
//! harness cannot measure; all it establishes is that they are harmless (see
//! `a_note_never_displaces_a_symbol`).
//!
//! So this module is honest infrastructure — a hard budget ceiling and an
//! attribution trail — not a validated recall win over plain truncation.
//!
//! ## Why this is where the wedge attaches
//!
//! [`PackedSymbol::reason`] records *why* each item is in the context. That is
//! the join key for Phase 17C: an edit outcome can be attributed back to the
//! symbols that were packed and the rule that packed them, so the procedural
//! compiler has something concrete to learn over.

use crate::SymbolKind;
use mnesio_core::types::MemoryRef;

/// Approximate token count. Deliberately the same `chars / 4` estimate the
/// benchmark uses, so a budget here means the same thing a measured number
/// there does. Swapping in a real tokenizer moves both together.
pub fn est_tokens(s: &str) -> usize {
    s.len().div_ceil(4)
}

/// How much of a symbol made it into the context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// The symbol's full source.
    Full,
    /// Declaration only — the fallback when the body will not fit.
    Signature,
}

/// Why a symbol is in the context.
///
/// Kept because attribution is what Phase 17C learns over: without it an
/// outcome can be credited to "the context" but not to any decision that
/// produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// Retrieval ranked it directly, at this 0-based position.
    Seed(usize),
    /// Pulled in as a 1-hop callee of a seed.
    Expanded(MemoryRef),
}

/// One symbol as packed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedSymbol {
    pub memory: MemoryRef,
    pub form: Form,
    pub tokens: usize,
    pub reason: Reason,
}

/// A file-level note, attached only from budget no symbol wanted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileNote {
    pub path: String,
    pub summary: String,
    pub tokens: usize,
}

/// The context to hand an agent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PackedContext {
    /// Seeds first in retrieval order, then expansions.
    pub symbols: Vec<PackedSymbol>,
    /// One line per distinct file represented, describing what that file is
    /// for. Added **after** every symbol that fits, out of leftover budget, so
    /// it can never be the reason a symbol was dropped — the property the two
    /// earlier designs lacked.
    pub notes: Vec<FileNote>,
    pub tokens_used: usize,
    /// Candidates that did not fit in any form.
    pub dropped: usize,
}

impl PackedContext {
    /// Did this symbol make it in, in any form?
    pub fn contains(&self, m: MemoryRef) -> bool {
        self.symbols.iter().any(|s| s.memory == m)
    }
}

/// How the packer reads the corpus.
///
/// A trait rather than a concrete store (Hard Rule #7): the benchmark backs it
/// with in-memory tables, the server with the event log's materialized views,
/// and tests with a fake — none of which the packing policy should know about.
pub trait PackSource {
    /// Full source text of a symbol.
    fn text(&self, m: MemoryRef) -> Option<&str>;
    /// One-line declaration, when the parser isolated one.
    fn signature(&self, m: MemoryRef) -> Option<&str>;
    /// Repo-relative file the symbol is defined in.
    fn path(&self, m: MemoryRef) -> Option<&str>;
    /// What kind of definition it is, so packing can prefer real code over
    /// scaffolding when the budget is tight.
    fn kind(&self, m: MemoryRef) -> Option<SymbolKind>;
    /// Header prose of a file.
    fn module_doc(&self, path: &str) -> Option<&str>;
    /// Resolved 1-hop callees.
    fn links(&self, m: MemoryRef) -> &[MemoryRef];
}

/// Packing limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackConfig {
    /// Hard ceiling on estimated tokens. Never exceeded.
    pub budget: usize,
    /// Expand seeds along `Calls`. Measured as a consistent win, but left
    /// switchable because it costs 40–70% more tokens for its few points, and
    /// at a very tight budget those tokens buy more as extra seeds.
    pub expand: bool,
    /// Cap on expansions pulled in per seed, so one heavily-calling function
    /// cannot crowd out the rest of the ranking (Hard Rule #6 — bound the
    /// cascade).
    pub max_expansions_per_seed: usize,
    /// Attach file-level notes from leftover budget.
    ///
    /// Measured as *neither* helping nor hurting retrieval recall — a note can
    /// never be a gold symbol. Kept on because it is provably free of
    /// displacement risk and gives an agent orientation this harness has no
    /// way to score.
    pub notes: bool,
    /// Fall back to [`Form::Signature`] when a body will not fit.
    ///
    /// Switchable so the benchmark can ablate the packer's ideas
    /// independently. Keep it on in production — it is nearly free — but see
    /// the module docs before quoting its effect: the recall it adds is
    /// declarations, not bodies.
    pub degrade: bool,
}

impl PackConfig {
    /// Truncate-at-the-budget: no expansion, no degradation, no notes.
    ///
    /// The baseline every packing idea has to beat — it is what an agent does
    /// today when it takes ranked results until the context is full.
    pub fn naive(budget: usize) -> Self {
        Self {
            budget,
            expand: false,
            max_expansions_per_seed: 0,
            notes: false,
            degrade: false,
        }
    }
}

impl Default for PackConfig {
    fn default() -> Self {
        Self {
            // ~4k tokens is where symbol-level context most clearly beat a
            // whole-file strategy in the 17B measurements (44% vs 26%).
            budget: 4000,
            expand: true,
            max_expansions_per_seed: 3,
            notes: true,
            degrade: true,
        }
    }
}

/// Longest file note worth carrying. One sentence of orientation; the rest of
/// a module header is detail that belongs to the file, not to this answer.
const NOTE_CHARS: usize = 160;

/// Pack `seeds` — retrieval's ranked output — into `cfg.budget`.
///
/// Order of business, and the reason for it:
///
/// 1. **Seeds, in rank order.** Retrieval's judgement is the best signal
///    available; nothing should outrank it.
/// 2. **Expansions**, grouped after all seeds rather than interleaved, so a
///    tight budget spends itself on directly-relevant code first.
/// 3. **File notes**, from what is left. Never displaces a symbol.
///
/// Within steps 1 and 2 a candidate that does not fit whole is retried as a
/// signature before being dropped, and packing *continues* past a
/// non-fitting candidate — a single huge function must not end the pack while
/// smaller useful ones remain.
pub fn pack(seeds: &[MemoryRef], src: &dyn PackSource, cfg: PackConfig) -> PackedContext {
    let mut out = PackedContext::default();
    let mut seen: Vec<MemoryRef> = Vec::new();

    // --- 1. seeds ---
    for (rank, &m) in seeds.iter().enumerate() {
        if seen.contains(&m) {
            continue;
        }
        seen.push(m);
        place(&mut out, src, m, Reason::Seed(rank), &cfg);
    }

    // --- 2. one hop along Calls ---
    if cfg.expand {
        for &m in seeds {
            let mut taken = 0usize;
            for &callee in src.links(m) {
                if taken >= cfg.max_expansions_per_seed {
                    break;
                }
                if seen.contains(&callee) {
                    continue;
                }
                seen.push(callee);
                taken += 1;
                place(&mut out, src, callee, Reason::Expanded(m), &cfg);
            }
        }
    }

    // --- 3. file notes, from leftover budget only ---
    if cfg.notes {
        let mut paths: Vec<&str> = Vec::new();
        for s in &out.symbols {
            if let Some(p) = src.path(s.memory) {
                if !paths.contains(&p) {
                    paths.push(p);
                }
            }
        }
        for path in paths {
            let Some(summary) = src.module_doc(path).and_then(first_sentence) else {
                continue;
            };
            let note = format!("// {path} — {summary}");
            let tokens = est_tokens(&note);
            if out.tokens_used + tokens > cfg.budget {
                // Out of room. Keep going: a later file's note may be shorter,
                // and stopping here would bias notes toward whichever file
                // happened to be retrieved first.
                continue;
            }
            out.tokens_used += tokens;
            out.notes.push(FileNote {
                path: path.to_string(),
                summary,
                tokens,
            });
        }
    }

    out
}

/// Add `m` in the largest form that fits, or count it dropped.
fn place(
    out: &mut PackedContext,
    src: &dyn PackSource,
    m: MemoryRef,
    why: Reason,
    cfg: &PackConfig,
) {
    let remaining = cfg.budget.saturating_sub(out.tokens_used);

    if let Some(text) = src.text(m) {
        let t = est_tokens(text);
        if t <= remaining {
            out.tokens_used += t;
            out.symbols.push(PackedSymbol {
                memory: m,
                form: Form::Full,
                tokens: t,
                reason: why,
            });
            return;
        }
    }
    // Body too big — the declaration alone still tells an agent the symbol
    // exists and what it takes, which is usually enough to decide whether to
    // ask for more.
    if let Some(sig) = src.signature(m).filter(|_| cfg.degrade) {
        let t = est_tokens(sig);
        if t <= remaining {
            out.tokens_used += t;
            out.symbols.push(PackedSymbol {
                memory: m,
                form: Form::Signature,
                tokens: t,
                reason: why,
            });
            return;
        }
    }
    out.dropped += 1;
}

/// First non-empty line of a module header, truncated.
fn first_sentence(doc: &str) -> Option<String> {
    let first = doc.lines().map(str::trim).find(|l| !l.is_empty())?;
    if first.chars().count() <= NOTE_CHARS {
        return Some(first.to_string());
    }
    let head: String = first.chars().take(NOTE_CHARS).collect();
    Some(format!("{}…", head.trim_end()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::types::new_id;
    use std::collections::HashMap;

    #[derive(Default)]
    struct Fake {
        text: HashMap<MemoryRef, String>,
        sig: HashMap<MemoryRef, String>,
        path: HashMap<MemoryRef, String>,
        kind: HashMap<MemoryRef, SymbolKind>,
        docs: HashMap<String, String>,
        links: HashMap<MemoryRef, Vec<MemoryRef>>,
    }

    impl Fake {
        fn add(&mut self, path: &str, body: &str, sig: &str) -> MemoryRef {
            let m = MemoryRef(new_id());
            self.text.insert(m, body.into());
            self.sig.insert(m, sig.into());
            self.path.insert(m, path.into());
            self.kind.insert(m, SymbolKind::Function);
            m
        }
    }

    impl PackSource for Fake {
        fn text(&self, m: MemoryRef) -> Option<&str> {
            self.text.get(&m).map(String::as_str)
        }
        fn signature(&self, m: MemoryRef) -> Option<&str> {
            self.sig.get(&m).map(String::as_str)
        }
        fn path(&self, m: MemoryRef) -> Option<&str> {
            self.path.get(&m).map(String::as_str)
        }
        fn kind(&self, m: MemoryRef) -> Option<SymbolKind> {
            self.kind.get(&m).copied()
        }
        fn module_doc(&self, path: &str) -> Option<&str> {
            self.docs.get(path).map(String::as_str)
        }
        fn links(&self, m: MemoryRef) -> &[MemoryRef] {
            self.links.get(&m).map_or(&[], Vec::as_slice)
        }
    }

    fn cfg(budget: usize) -> PackConfig {
        PackConfig {
            budget,
            ..Default::default()
        }
    }

    #[test]
    fn the_budget_is_never_exceeded() {
        let mut f = Fake::default();
        let a = f.add("a.rs", &"x".repeat(400), "fn a()");
        let b = f.add("a.rs", &"y".repeat(400), "fn b()");
        let c = f.add("a.rs", &"z".repeat(400), "fn c()");

        let p = pack(&[a, b, c], &f, cfg(120));
        assert!(
            p.tokens_used <= 120,
            "packed {} tokens into a 120 budget",
            p.tokens_used
        );
    }

    #[test]
    fn seeds_are_packed_in_retrieval_order() {
        // Retrieval's ranking is the best signal available; the packer must
        // not reorder it.
        let mut f = Fake::default();
        let a = f.add("a.rs", "aaaa", "fn a()");
        let b = f.add("b.rs", "bbbb", "fn b()");
        let c = f.add("c.rs", "cccc", "fn c()");

        let p = pack(&[c, a, b], &f, cfg(1000));
        let order: Vec<_> = p.symbols.iter().map(|s| s.memory).collect();
        assert_eq!(order, vec![c, a, b]);
        assert_eq!(p.symbols[0].reason, Reason::Seed(0));
        assert_eq!(p.symbols[2].reason, Reason::Seed(2));
    }

    #[test]
    fn a_symbol_that_does_not_fit_degrades_to_its_signature() {
        let mut f = Fake::default();
        let big = f.add("a.rs", &"x".repeat(4000), "fn big(a: u8) -> u8");

        let p = pack(&[big], &f, cfg(20));
        assert_eq!(p.symbols.len(), 1, "should have kept the declaration");
        assert_eq!(p.symbols[0].form, Form::Signature);
        assert_eq!(p.dropped, 0);
    }

    #[test]
    fn one_oversized_symbol_does_not_end_the_pack() {
        // Regression guard: stopping at the first non-fitting candidate would
        // throw away every smaller useful symbol behind it.
        let mut f = Fake::default();
        let huge = f.add("a.rs", &"x".repeat(100_000), &"s".repeat(100_000));
        let small = f.add("b.rs", "fn small() {}", "fn small()");

        let p = pack(&[huge, small], &f, cfg(50));
        assert!(p.contains(small), "packing stopped at the oversized symbol");
        assert_eq!(p.dropped, 1);
    }

    #[test]
    fn expansion_pulls_in_callees_after_every_seed() {
        // Expansions come *after* all seeds: at a tight budget the directly
        // retrieved code should win the space.
        let mut f = Fake::default();
        let seed1 = f.add("a.rs", "fn one() { helper() }", "fn one()");
        let seed2 = f.add("a.rs", "fn two() {}", "fn two()");
        let helper = f.add("b.rs", "fn helper() {}", "fn helper()");
        f.links.insert(seed1, vec![helper]);

        let p = pack(&[seed1, seed2], &f, cfg(1000));
        let order: Vec<_> = p.symbols.iter().map(|s| s.memory).collect();
        assert_eq!(order, vec![seed1, seed2, helper]);
        assert_eq!(p.symbols[2].reason, Reason::Expanded(seed1));
    }

    #[test]
    fn expansion_per_seed_is_bounded() {
        // Hard Rule #6: one heavily-calling function must not crowd out the
        // rest of the ranking.
        let mut f = Fake::default();
        let seed = f.add("a.rs", "fn s() {}", "fn s()");
        let callees: Vec<_> = (0..10)
            .map(|_| f.add("b.rs", "fn c() {}", "fn c()"))
            .collect();
        f.links.insert(seed, callees);

        let p = pack(
            &[seed],
            &f,
            PackConfig {
                budget: 100_000,
                max_expansions_per_seed: 3,
                ..Default::default()
            },
        );
        let expanded = p
            .symbols
            .iter()
            .filter(|s| matches!(s.reason, Reason::Expanded(_)))
            .count();
        assert_eq!(expanded, 3);
    }

    #[test]
    fn a_note_never_displaces_a_symbol() {
        // The property both earlier designs lacked, and the whole reason notes
        // are applied last: adding file prose must not cost a symbol its place.
        let mut f = Fake::default();
        let a = f.add("a.rs", "fn a() { }", "fn a()");
        let b = f.add("b.rs", "fn b() { }", "fn b()");
        f.docs
            .insert("a.rs".into(), "A very long module description ".repeat(20));
        f.docs
            .insert("b.rs".into(), "Another long module description ".repeat(20));

        let tight = cfg(10);
        let without = pack(
            &[a, b],
            &f,
            PackConfig {
                notes: false,
                ..tight
            },
        );
        let with = pack(&[a, b], &f, tight);

        let packed =
            |p: &PackedContext| -> Vec<MemoryRef> { p.symbols.iter().map(|s| s.memory).collect() };
        assert_eq!(
            packed(&with),
            packed(&without),
            "enabling notes changed which symbols were packed"
        );
        assert!(with.tokens_used <= tight.budget);
    }

    #[test]
    fn notes_describe_each_distinct_file_once() {
        let mut f = Fake::default();
        let a1 = f.add("a.rs", "fn a1() {}", "fn a1()");
        let a2 = f.add("a.rs", "fn a2() {}", "fn a2()");
        let b = f.add("b.rs", "fn b() {}", "fn b()");
        f.docs.insert("a.rs".into(), "The A module.".into());
        f.docs.insert("b.rs".into(), "The B module.".into());

        let p = pack(&[a1, a2, b], &f, cfg(1000));
        assert_eq!(p.notes.len(), 2, "one note per file, not per symbol");
        assert!(p.notes.iter().any(|n| n.path == "a.rs"));
        assert!(p.notes.iter().any(|n| n.summary == "The B module."));
    }

    #[test]
    fn duplicate_seeds_are_packed_once() {
        let mut f = Fake::default();
        let a = f.add("a.rs", "fn a() {}", "fn a()");
        let p = pack(&[a, a, a], &f, cfg(1000));
        assert_eq!(p.symbols.len(), 1);
    }

    #[test]
    fn an_empty_query_packs_nothing_rather_than_panicking() {
        let f = Fake::default();
        let p = pack(&[], &f, cfg(1000));
        assert!(p.symbols.is_empty());
        assert_eq!(p.tokens_used, 0);
    }
}

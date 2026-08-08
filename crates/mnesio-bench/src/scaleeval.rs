//! Phase 18H: `codeeval` across many repositories, as one reproducible artifact.
//!
//! ## Why a multi-repo suite is a different thing, not just a bigger one
//!
//! Every number this project has published for code retrieval came from four
//! repositories. That is enough to falsify a hypothesis — and it did, twice —
//! but not enough to support a claim, because four repositories share
//! accidents. Three of ours are Rust; one dominates the query count. A result
//! that holds there may be a result about *those* repositories.
//!
//! What changes with breadth is the shape of the answer. A single repo yields
//! "recall is 62%". Fifty yield a **distribution**, and a distribution is what
//! survives a reader asking "on what?". It also makes the honest caveat
//! precise: we already know the symbol/whole-file trade is repo-dependent
//! (claw-code favours symbols at 64k, BigBrainAI favours whole-file), and one
//! aggregate number would hide exactly that.
//!
//! ## The protocol is unchanged on purpose
//!
//! Same as [`crate::codeeval`]: query = a real commit subject, gold = the
//! symbols that commit touched per `git log -L`, arms paired on one index per
//! repository, `k` swept inside the run. Nothing here relaxes that — scale
//! without the protocol is just more numbers.
//!
//! ## What this reports, and what it refuses to
//!
//! It reports per-repository rows and a distribution: median, quartiles, and
//! the worst case. It deliberately does **not** emit a single headline
//! average, because averaging recall across repositories of wildly different
//! size and language is a number with no referent — and it is precisely the
//! kind of figure that ends up in a README without its unflattering half.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

use crate::codeeval::{run_codeeval, CodeEvalReport};
use crate::gitsuite;

/// One repository's contribution to the suite.
#[derive(Debug, Clone)]
pub struct RepoResult {
    pub name: String,
    pub files: usize,
    pub symbols: usize,
    pub queries: usize,
    /// Best recall the symbol arm reached at any swept `k`.
    pub symbol_peak: f32,
    /// Best recall the whole-file arm reached — the ceiling being traded away.
    pub whole_file_peak: f32,
    /// Share of misses a better ranker could address.
    pub rankable: f32,
    /// Share no scoring change can reach: no shared vocabulary at all.
    pub unreachable: f32,
    pub index_secs: f64,
}

impl RepoResult {
    /// How much ceiling the symbol arm gives up. The number a reader will look
    /// for and the one a marketing page would omit.
    pub fn ceiling_gap(&self) -> f32 {
        self.whole_file_peak - self.symbol_peak
    }

    /// Is this corpus large enough for the result to mean anything?
    ///
    /// See [`MIN_DISCRIMINATING_SYMBOLS`]. A repository below the floor scores
    /// 100% because top-`k` reaches most of it, which says nothing about
    /// retrieval quality and everything about the repository being tiny.
    pub fn is_discriminating(&self) -> bool {
        self.symbols >= MIN_DISCRIMINATING_SYMBOLS
    }
}

/// Every repository, plus the distribution across them.
#[derive(Debug, Clone, Default)]
pub struct ScaleReport {
    pub repos: Vec<RepoResult>,
    /// Repositories that could not be evaluated, and why. Kept in the report
    /// rather than dropped: a suite that silently skips what it cannot handle
    /// reports a survivorship-biased result.
    pub skipped: Vec<(String, String)>,
}

/// Smallest corpus that can discriminate between retrieval strategies.
///
/// With `k` swept to 20, a repository of 49 symbols has top-20 covering 40% of
/// everything it contains — recall is near-guaranteed by corpus size, not by
/// ranking. Four of the first nine repositories measured scored exactly
/// 100%/100% with a 0pp gap for that reason, and they pulled the p75 and max
/// of every metric to 100%, making the suite look better than it is.
///
/// 500 keeps top-20 under ~4% of the corpus, which is the point at which the
/// arms are actually choosing. Below it a repository is still reported — the
/// row is real — but it is excluded from the distribution, because a quartile
/// computed over non-discriminating repositories describes nothing.
pub const MIN_DISCRIMINATING_SYMBOLS: usize = 500;

/// Quartiles of a metric across repositories.
#[derive(Debug, Clone, Copy)]
pub struct Spread {
    pub min: f32,
    pub p25: f32,
    pub median: f32,
    pub p75: f32,
    pub max: f32,
}

impl ScaleReport {
    /// Distribution of a per-repository metric.
    ///
    /// Quartiles rather than a mean: recall across repositories of different
    /// size and language is not a quantity a mean describes, and the spread is
    /// the actual finding.
    /// Computed over *discriminating* repositories only — see
    /// [`MIN_DISCRIMINATING_SYMBOLS`]. Including trivially small ones inflates
    /// every quantile toward 100% and flatters the result.
    pub fn spread(&self, f: impl Fn(&RepoResult) -> f32) -> Option<Spread> {
        let mut v: Vec<f32> = self
            .repos
            .iter()
            .filter(|r| r.is_discriminating())
            .map(f)
            .collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let at = |q: f64| v[(((v.len() - 1) as f64) * q).round() as usize];
        Some(Spread {
            min: v[0],
            p25: at(0.25),
            median: at(0.50),
            p75: at(0.75),
            max: v[v.len() - 1],
        })
    }

    /// Repositories large enough to discriminate.
    pub fn discriminating(&self) -> impl Iterator<Item = &RepoResult> {
        self.repos.iter().filter(|r| r.is_discriminating())
    }

    pub fn total_queries(&self) -> usize {
        self.repos.iter().map(|r| r.queries).sum()
    }
    pub fn total_symbols(&self) -> usize {
        self.repos.iter().map(|r| r.symbols).sum()
    }
}

/// Directories under `root` that look like git repositories.
pub fn discover_repos(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return out;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() && p.join(".git").exists() {
            out.push(p);
        }
    }
    out.sort();
    out
}

/// Run the paired evaluation across every repository given.
///
/// A repository that yields no usable queries is recorded in `skipped` rather
/// than dropped — see [`ScaleReport::skipped`].
pub async fn run_scale(
    repos: &[PathBuf],
    ks: &[usize],
    embedder: &str,
    max_queries: usize,
) -> Result<ScaleReport> {
    let mut report = ScaleReport::default();

    for repo in repos {
        let name = repo
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| repo.display().to_string());
        let dir = repo.to_string_lossy().to_string();

        eprintln!("# {name}");
        let started = std::time::Instant::now();

        let targets = match crate::codeeval::trace_targets(&dir) {
            Ok(t) => t,
            Err(e) => {
                report
                    .skipped
                    .push((name, format!("no parseable source: {e}")));
                continue;
            }
        };
        let suite = match gitsuite::derive(&dir, &targets, max_queries) {
            Ok(s) if !s.is_empty() => s,
            Ok(_) => {
                report.skipped.push((
                    name,
                    "no descriptive commits touching indexed symbols".into(),
                ));
                continue;
            }
            Err(e) => {
                report
                    .skipped
                    .push((name, format!("history unusable: {e}")));
                continue;
            }
        };

        let r: CodeEvalReport = match run_codeeval(&dir, ks, embedder, &suite).await {
            Ok(r) => r,
            Err(e) => {
                report.skipped.push((name, format!("eval failed: {e}")));
                continue;
            }
        };
        let elapsed = started.elapsed().as_secs_f64();
        let total = r.misses.total().max(1) as f32;

        report.repos.push(RepoResult {
            name,
            files: r.index.files,
            symbols: r.index.symbols,
            queries: suite.len(),
            symbol_peak: r.peak_recall("symbol"),
            whole_file_peak: r.peak_recall("whole-file"),
            rankable: r.misses.rankable as f32 / total,
            unreachable: (r.misses.no_overlap + r.misses.not_indexed) as f32 / total,
            index_secs: elapsed,
        });
    }

    if report.repos.is_empty() {
        // Print why before erroring. A bare count sends the reader hunting for
        // a bug in whichever stage they guess first — which cost real time
        // when a manifest corpus first failed here.
        for (name, why) in &report.skipped {
            eprintln!("  skipped {name}: {why}");
        }
        return Err(anyhow!(
            "no repository produced a usable suite ({} skipped)",
            report.skipped.len()
        ));
    }
    Ok(report)
}

/// Markdown report: per-repository rows, then the distribution.
pub fn format_scale(r: &ScaleReport) -> String {
    let mut out = format!(
        "# code retrieval at scale — {} repositories, {} queries, {} symbols\n\n",
        r.repos.len(),
        r.total_queries(),
        r.total_symbols()
    );

    out.push_str(
        "| repo | files | symbols | queries | symbol | whole-file | gap | rankable | unreachable |\n\
         |---|---|---|---|---|---|---|---|---|\n",
    );
    for x in &r.repos {
        out.push_str(&format!(
            "| {}{} | {} | {} | {} | {:.0}% | {:.0}% | −{:.0}pp | {:.0}% | {:.0}% |\n",
            x.name,
            if x.is_discriminating() { "" } else { " ᵗ" },
            x.files,
            x.symbols,
            x.queries,
            x.symbol_peak * 100.0,
            x.whole_file_peak * 100.0,
            x.ceiling_gap() * 100.0,
            x.rankable * 100.0,
            x.unreachable * 100.0,
        ));
    }

    let row = |label: &str, s: Option<Spread>| -> String {
        match s {
            Some(s) => format!(
                "| {} | {:.0}% | {:.0}% | **{:.0}%** | {:.0}% | {:.0}% |\n",
                label,
                s.min * 100.0,
                s.p25 * 100.0,
                s.median * 100.0,
                s.p75 * 100.0,
                s.max * 100.0
            ),
            None => String::new(),
        }
    };

    let n_disc = r.discriminating().count();
    let n_tiny = r.repos.len() - n_disc;
    if n_tiny > 0 {
        out.push_str(&format!(
            "\nᵗ {n_tiny} repositories have fewer than {MIN_DISCRIMINATING_SYMBOLS} symbols. \
             At that size top-`k` reaches most of the corpus, so they score ~100% \
             regardless of ranking quality. Their rows are shown but they are \
             **excluded from the distribution below** — including them pulls every \
             quantile toward 100% and flatters the result.\n"
        ));
    }

    out.push_str(&format!(
        "\n## distribution across the {n_disc} discriminating repositories\n\n"
    ));
    out.push_str("| metric | min | p25 | median | p75 | max |\n|---|---|---|---|---|---|\n");
    out.push_str(&row("symbol recall", r.spread(|x| x.symbol_peak)));
    out.push_str(&row("whole-file recall", r.spread(|x| x.whole_file_peak)));
    out.push_str(&row("ceiling gap", r.spread(|x| x.ceiling_gap())));
    out.push_str(&row("rankable share", r.spread(|x| x.rankable)));
    out.push_str(&row("unreachable share", r.spread(|x| x.unreachable)));

    out.push_str(
        "\n_Quartiles, not a mean. Averaging recall across repositories of \
         different size and language produces a number with no referent, and \
         the spread — not the centre — is the finding: the symbol/whole-file \
         trade is repo-dependent._\n",
    );

    if !r.skipped.is_empty() {
        out.push_str(&format!(
            "\n## skipped ({})\n\nListed rather than dropped: a suite that \
             silently discards what it cannot handle reports a \
             survivorship-biased result.\n\n",
            r.skipped.len()
        ));
        for (name, why) in &r.skipped {
            out.push_str(&format!("- **{name}** — {why}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo(name: &str, sym: f32, wf: f32) -> RepoResult {
        sized(name, sym, wf, MIN_DISCRIMINATING_SYMBOLS)
    }

    fn sized(name: &str, sym: f32, wf: f32, symbols: usize) -> RepoResult {
        RepoResult {
            name: name.into(),
            files: 1,
            symbols,
            queries: 1,
            symbol_peak: sym,
            whole_file_peak: wf,
            rankable: 0.3,
            unreachable: 0.1,
            index_secs: 0.0,
        }
    }

    #[test]
    fn the_ceiling_gap_is_reported_not_hidden() {
        // The number a marketing page would omit. If this ever stops being
        // computed, the report has lost the half that makes it honest.
        let r = repo("x", 0.66, 0.96);
        assert!((r.ceiling_gap() - 0.30).abs() < 1e-6);
    }

    #[test]
    fn quartiles_describe_a_spread_a_mean_would_hide() {
        // Two repos that disagree completely: a mean says 50% and describes
        // neither, which is exactly why this reports quartiles.
        let rep = ScaleReport {
            repos: vec![repo("a", 0.2, 0.9), repo("b", 0.8, 0.9)],
            ..Default::default()
        };
        let s = rep.spread(|x| x.symbol_peak).unwrap();
        assert!((s.min - 0.2).abs() < 1e-6);
        assert!((s.max - 0.8).abs() < 1e-6);
    }

    /// The bug this floor exists to fix, pinned. Measured live: four of nine
    /// repositories were small enough to score 100%/100%, which pulled p75 and
    /// max of every metric to 100% and made the suite look better than it was.
    #[test]
    fn tiny_repositories_do_not_inflate_the_distribution() {
        let rep = ScaleReport {
            repos: vec![
                sized("real-a", 0.45, 0.85, 1154),
                sized("real-b", 0.55, 0.65, 20992),
                // Two symbols. Top-20 reaches everything; 100% is arithmetic,
                // not retrieval quality.
                sized("toy-a", 1.0, 1.0, 2),
                sized("toy-b", 1.0, 1.0, 49),
            ],
            ..Default::default()
        };
        let s = rep.spread(|x| x.symbol_peak).unwrap();
        assert!(
            s.max <= 0.55 + 1e-6,
            "a toy repo leaked into the distribution: max={}",
            s.max
        );
        assert_eq!(rep.discriminating().count(), 2);
    }

    #[test]
    fn a_suite_of_only_tiny_repositories_reports_no_distribution() {
        // Better than a confident 100% computed over nothing meaningful.
        let rep = ScaleReport {
            repos: vec![sized("toy", 1.0, 1.0, 3)],
            ..Default::default()
        };
        assert!(rep.spread(|x| x.symbol_peak).is_none());
    }

    #[test]
    fn an_empty_suite_has_no_distribution_rather_than_a_fake_one() {
        assert!(ScaleReport::default().spread(|x| x.symbol_peak).is_none());
    }

    #[test]
    fn skipped_repositories_are_kept_in_the_report() {
        let mut rep = ScaleReport::default();
        rep.skipped
            .push(("weird".into(), "no parseable source".into()));
        let text = format_scale(&rep);
        assert!(text.contains("skipped"), "skips must be visible: {text}");
        assert!(text.contains("weird"));
    }
}

/// What repeated runs of one configuration reveal about the harness itself.
///
/// ## Why repetition and not a seed
///
/// The obvious response to a noisy benchmark is to seed the index build.
/// `hnsw_rs` 0.3.4 constructs its layer-assignment RNG with
/// `StdRng::from_os_rng()` and exposes no setter, so that is not available
/// through the public API — but more importantly it would be **the wrong
/// fix**.
///
/// A seed makes a run *reproducible*. It does not make a single-run comparison
/// *valid*: two different configurations produce two different graphs whatever
/// the seed, so one run per arm is still one sample from each of two
/// distributions. Comparing two samples tells you nothing about the
/// distributions unless you know how wide they are.
///
/// So the harness measures its own width. Run the same configuration N times,
/// take the spread, and refuse to call anything a finding unless it exceeds
/// that. Measured on `codeeval-v1`: symbol recall varies by up to 2pp per
/// repository between identical runs, which is exactly the size of the
/// strict-vs-loose resolver "effect" — the comparison that looked like a
/// result until a control run was added.
#[derive(Debug, Clone)]
pub struct NoiseFloor {
    /// Runs used to establish it.
    pub runs: usize,
    /// Largest observed variation in symbol recall for any one repository,
    /// in percentage points, across identical runs.
    pub symbol_pp: f32,
    /// Same for whole-file recall. This one is the tell: the whole-file arm
    /// never consults the call graph, so variation here cannot be caused by
    /// any retrieval-policy change and is purely index randomness.
    pub whole_file_pp: f32,
}

impl NoiseFloor {
    /// Fold repeated reports of the *same* configuration into a noise floor.
    pub fn from_repeats(reports: &[ScaleReport]) -> Option<Self> {
        if reports.len() < 2 {
            return None;
        }
        let key = |r: &RepoResult| (r.name.clone(), r.symbols);
        let mut sym: f32 = 0.0;
        let mut whole: f32 = 0.0;
        for probe in &reports[0].repos {
            let k = key(probe);
            let mut ss: Vec<f32> = Vec::new();
            let mut ws: Vec<f32> = Vec::new();
            for rep in reports {
                if let Some(r) = rep.repos.iter().find(|r| key(r) == k) {
                    ss.push(r.symbol_peak);
                    ws.push(r.whole_file_peak);
                }
            }
            if ss.len() == reports.len() {
                let sspread = ss.iter().cloned().fold(f32::MIN, f32::max)
                    - ss.iter().cloned().fold(f32::MAX, f32::min);
                let wspread = ws.iter().cloned().fold(f32::MIN, f32::max)
                    - ws.iter().cloned().fold(f32::MAX, f32::min);
                sym = sym.max(sspread * 100.0);
                whole = whole.max(wspread * 100.0);
            }
        }
        Some(NoiseFloor {
            runs: reports.len(),
            symbol_pp: sym,
            whole_file_pp: whole,
        })
    }

    /// Would a claimed delta of `pp` survive this noise floor?
    ///
    /// Deliberately strict: a delta merely *equal* to the observed spread is
    /// not a finding, it is a coin landing the same way twice.
    pub fn resolves(&self, pp: f32) -> bool {
        pp.abs() > self.symbol_pp
    }

    pub fn render(&self) -> String {
        format!(
            "\n## noise floor\n\n\
             {} identical runs of this configuration. Largest variation for any \
             single repository: **{:.0}pp** symbol recall, {:.0}pp whole-file.\n\n\
             Whole-file recall does not consult the call graph, so its variation \
             is index-build randomness by construction — which is what makes it \
             the reference for how much of the symbol-recall variation is also \
             noise.\n\n\
             **Any A/B on this corpus must exceed {:.0}pp to be a finding.** A \
             seed would make runs reproducible but would not make a one-run-per-arm \
             comparison valid: two configurations build two different graphs \
             whatever the seed, so a single sample from each says nothing about \
             the distributions.\n",
            self.runs, self.symbol_pp, self.whole_file_pp, self.symbol_pp
        )
    }
}

#[cfg(test)]
mod noise_tests {
    use super::*;

    fn repo(name: &str, sym: f32, whole: f32) -> RepoResult {
        RepoResult {
            name: name.into(),
            files: 10,
            symbols: 900,
            queries: 60,
            symbol_peak: sym,
            whole_file_peak: whole,
            rankable: 0.2,
            unreachable: 0.1,
            index_secs: 1.0,
        }
    }
    fn report(rs: Vec<RepoResult>) -> ScaleReport {
        ScaleReport {
            repos: rs,
            skipped: Vec::new(),
        }
    }

    #[test]
    fn a_single_run_cannot_establish_a_noise_floor() {
        // One sample has no spread. Returning 0 here would let any delta pass
        // as a finding — the opposite of what this exists for.
        assert!(NoiseFloor::from_repeats(&[report(vec![repo("a", 0.5, 0.7)])]).is_none());
    }

    #[test]
    fn the_floor_is_the_widest_single_repository_not_the_average() {
        // Averaging spread across repositories would hide the one repository
        // that swings most, and that is exactly the one a cherry-picked A/B
        // would be quoted from.
        let a = report(vec![repo("steady", 0.50, 0.70), repo("swingy", 0.40, 0.70)]);
        let b = report(vec![repo("steady", 0.51, 0.70), repo("swingy", 0.48, 0.70)]);
        let nf = NoiseFloor::from_repeats(&[a, b]).unwrap();
        assert!((nf.symbol_pp - 8.0).abs() < 0.01, "got {}", nf.symbol_pp);
    }

    #[test]
    fn a_delta_equal_to_the_noise_is_not_a_finding() {
        // Strictly greater. A delta the same size as the observed spread is a
        // coin landing the same way twice.
        // Binary-exact values, so this tests the rule rather than f32 rounding:
        // 0.52 - 0.50 is not exactly 0.02, and the near-miss made the first
        // assertion flip.
        let a = report(vec![repo("r", 0.5, 0.75)]);
        let b = report(vec![repo("r", 0.75, 0.75)]);
        let nf = NoiseFloor::from_repeats(&[a, b]).unwrap();
        assert_eq!(nf.symbol_pp, 25.0);
        assert!(
            !nf.resolves(25.0),
            "a delta equal to the floor is not a finding"
        );
        assert!(nf.resolves(25.5));
    }

    #[test]
    fn a_repository_missing_from_one_run_is_not_folded_in() {
        // A repository that failed in one run and not another would otherwise
        // contribute a spurious spread from a comparison that never happened.
        let a = report(vec![repo("both", 0.50, 0.70), repo("only-a", 0.10, 0.10)]);
        let b = report(vec![repo("both", 0.50, 0.70)]);
        let nf = NoiseFloor::from_repeats(&[a, b]).unwrap();
        assert_eq!(nf.symbol_pp, 0.0);
    }

    #[test]
    fn the_rendered_floor_states_the_threshold_a_claim_must_beat() {
        // It goes into a results file a reader may quote from, so the
        // threshold has to travel with the number.
        let a = report(vec![repo("r", 0.50, 0.70)]);
        let b = report(vec![repo("r", 0.53, 0.70)]);
        let out = NoiseFloor::from_repeats(&[a, b]).unwrap().render();
        assert!(out.contains("must exceed"), "got: {out}");
        assert!(out.contains("2 identical runs"), "got: {out}");
    }
}

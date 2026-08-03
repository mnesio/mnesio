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
    pub fn spread(&self, f: impl Fn(&RepoResult) -> f32) -> Option<Spread> {
        if self.repos.is_empty() {
            return None;
        }
        let mut v: Vec<f32> = self.repos.iter().map(f).collect();
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
            "| {} | {} | {} | {} | {:.0}% | {:.0}% | −{:.0}pp | {:.0}% | {:.0}% |\n",
            x.name,
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

    out.push_str("\n## distribution across repositories\n\n");
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
        RepoResult {
            name: name.into(),
            files: 1,
            symbols: 1,
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

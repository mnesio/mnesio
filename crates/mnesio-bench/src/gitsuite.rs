//! Build a code-retrieval suite out of a repository's own git history.
//!
//! ## Why this exists
//!
//! The hand-written suite in [`crate::codeeval`] is disqualified from proving
//! anything: the queries were written by someone who already knew which symbol
//! should come back, so it can only ever show that the pipeline is *not
//! broken*. Phase 17B's "done when" asks for **real repo tasks**, and the
//! honest way to get those is to take them from work that already happened,
//! for reasons unrelated to this benchmark.
//!
//! ## The protocol
//!
//! - **Query** = a real commit subject. A human wrote it to describe a change,
//!   long before mnesio existed. It cannot have been tuned to the index.
//! - **Gold** = the symbols that commit actually modified, as determined by
//!   `git log -L <start>,<end>:<file>` — git's own line-history tracking,
//!   which follows a line range backwards through renames and edits. We ask it
//!   per symbol and invert the answer, so the mapping is git's, not a
//!   heuristic of ours.
//! - **Scoring** = did the packed context contain *any* gold symbol. That is
//!   the realistic agent criterion: the task lands you in the right code.
//!
//! ## What this still doesn't prove
//!
//! A commit subject is a *description of a change*, not a question — it is a
//! good proxy for an agent's task prompt, not a substitute for one. And
//! commits touching many symbols are excluded (see [`MAX_GOLD`]) because
//! "hit at least one of fifteen" is not a real test. Both limits are honest
//! narrowings of the claim, not of the difficulty: the queries themselves stay
//! adversarial, since nobody wrote them with retrieval in mind.

use std::collections::BTreeMap;
use std::process::Command;

use anyhow::{anyhow, Result};

use crate::codeeval::CodeQuery;

/// Commits touching more symbols than this are dropped: with a large gold set,
/// "retrieved at least one" stops discriminating between arms.
const MAX_GOLD: usize = 3;

/// Subjects shorter than this are `wip`, `fix`, `.` — no retrievable signal.
const MIN_SUBJECT_CHARS: usize = 25;

/// How far back to trace each symbol's line range. Bounds a `git log -L` that
/// would otherwise walk the whole history of a long-lived file.
const HISTORY_DEPTH: usize = 30;

/// A symbol to trace: where it lives now, and what it is called.
#[derive(Debug, Clone)]
pub struct TraceTarget {
    /// Repo-relative path.
    pub path: String,
    pub name: String,
    pub start_line: u32,
    pub end_line: u32,
}

/// Ask git which commits touched `target`'s current line range.
///
/// Returns `(sha, subject)` pairs, newest first. A failure here is *not* an
/// error for the run: a file can be untracked, or added in the initial import
/// with no line history. Those symbols simply contribute no queries.
fn commits_touching(repo: &str, target: &TraceTarget) -> Vec<(String, String)> {
    // `-L` implies a patch we don't want, so we tag the header lines and keep
    // only those. `%x09` is a tab, which cannot appear in a subject.
    let range = format!("{},{}:{}", target.start_line, target.end_line, target.path);
    let out = Command::new("git")
        .args([
            "-C",
            repo,
            "log",
            "--no-merges",
            "-n",
            &HISTORY_DEPTH.to_string(),
            "--format=@@@%H%x09%s",
            "-L",
            &range,
        ])
        .output();

    let Ok(out) = out else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.strip_prefix("@@@"))
        .filter_map(|l| l.split_once('\t'))
        .map(|(sha, subject)| (sha.to_string(), subject.to_string()))
        .collect()
}

/// Is this subject usable as a retrieval query?
///
/// Rejects the mechanical commits (`Merge`, version bumps, formatting) whose
/// text describes bookkeeping rather than any particular code. Keeping them
/// would measure noise: no retrieval system can be expected to map "bump
/// v0.4.2" onto a symbol.
fn usable_subject(subject: &str) -> bool {
    if subject.len() < MIN_SUBJECT_CHARS {
        return false;
    }
    let lower = subject.to_lowercase();
    const MECHANICAL: &[&str] = &[
        "merge ",
        "revert ",
        "bump ",
        "cargo fmt",
        "clippy",
        "rustfmt",
        "update lockfile",
        "cargo update",
        "initial commit",
        "wip",
    ];
    if MECHANICAL
        .iter()
        .any(|m| lower.starts_with(m) || lower.contains(m))
    {
        return false;
    }
    // At least four words, so a terse "fix the parser bug" style subject with
    // real nouns survives but "chore: cleanup" does not.
    subject.split_whitespace().count() >= 4
}

/// Derive a suite from `repo`'s history.
///
/// **`targets` must be every indexed symbol, not a sample.** A partial trace
/// silently truncates gold sets: a commit that touched ten symbols would be
/// recorded as touching only the traced one, so the symbol arm gets no credit
/// for retrieving any of the other nine while the whole-file arm still nets
/// the entire file. That biases the comparison in whole-file's favour, which
/// is exactly the direction that would flatter a wrong conclusion.
///
/// `limit` caps the returned queries.
pub fn derive(repo: &str, targets: &[TraceTarget], limit: usize) -> Result<Vec<CodeQuery>> {
    if !std::path::Path::new(repo).join(".git").exists() {
        return Err(anyhow!(
            "{repo} is not a git repository — the git suite needs history to \
             derive queries from"
        ));
    }

    // One `git log -L` per symbol is a subprocess spawn, so a large repo is
    // minutes of pure process overhead. Fan out across cores; git itself is
    // read-only here so the calls are independent.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(targets.len().max(1));
    let chunk = targets.len().div_ceil(threads);

    let traced: Vec<(String, String, String)> = std::thread::scope(|s| {
        let handles: Vec<_> = targets
            .chunks(chunk.max(1))
            .map(|part| {
                s.spawn(move || {
                    let mut out = Vec::new();
                    for t in part {
                        for (sha, subject) in commits_touching(repo, t) {
                            out.push((sha, subject, t.name.clone()));
                        }
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().ok())
            .flatten()
            .collect()
    });

    // sha -> (subject, gold symbol names). BTreeMap so the suite is
    // deterministic across runs; a benchmark whose contents shift between
    // invocations cannot support a paired comparison.
    let mut by_commit: BTreeMap<String, (String, Vec<String>)> = BTreeMap::new();
    for (sha, subject, name) in traced {
        if !usable_subject(&subject) {
            continue;
        }
        let e = by_commit
            .entry(sha)
            .or_insert_with(|| (subject, Vec::new()));
        if !e.1.contains(&name) {
            e.1.push(name);
        }
    }

    let mut suite: Vec<CodeQuery> = by_commit
        .into_iter()
        .filter(|(_, (_, gold))| !gold.is_empty() && gold.len() <= MAX_GOLD)
        .map(|(_, (subject, gold))| CodeQuery {
            question: subject,
            gold,
        })
        .collect();
    suite.truncate(limit);
    Ok(suite)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mechanical_subjects_are_rejected() {
        // These describe bookkeeping, not code: no retriever can map them onto
        // a symbol, so scoring them would measure noise.
        for s in [
            "Merge branch 'main' into feature/x",
            "bump version to 0.4.2 for release",
            "cargo fmt across the whole workspace",
            "wip",
            "fix",
            "chore: cleanup",
        ] {
            assert!(!usable_subject(s), "should reject {s:?}");
        }
    }

    #[test]
    fn descriptive_subjects_are_kept() {
        for s in [
            "feat(cli): populate Git SHA and target triple at compile time",
            "Remove the deprecated subscription login path from the auth flow",
            "fix the paragraph chunker dropping trailing whitespace",
        ] {
            assert!(usable_subject(s), "should keep {s:?}");
        }
    }

    /// The suite must be byte-identical across runs, or the paired comparison
    /// it feeds is meaningless: two arms could be scored on different queries.
    /// Tracing is fanned out across threads, so the join order is arbitrary —
    /// the `BTreeMap` is what restores determinism, and this pins it.
    #[test]
    fn the_derived_suite_is_deterministic() {
        let repo = env!("CARGO_MANIFEST_DIR");
        let root = std::path::Path::new(repo)
            .ancestors()
            .find(|p| p.join(".git").exists());
        let Some(root) = root.and_then(|p| p.to_str()) else {
            // Building from a tarball rather than a checkout: nothing to trace.
            return;
        };

        let targets: Vec<TraceTarget> = ["src/gitsuite.rs", "src/codeeval.rs", "src/memeval.rs"]
            .iter()
            .map(|f| TraceTarget {
                path: format!("crates/mnesio-bench/{f}"),
                name: f.to_string(),
                start_line: 1,
                end_line: 40,
            })
            .collect();

        let a = derive(root, &targets, 20).unwrap();
        let b = derive(root, &targets, 20).unwrap();
        assert_eq!(a.len(), b.len(), "suite size drifted between runs");
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.question, y.question, "query order is not stable");
            assert_eq!(x.gold, y.gold, "gold set is not stable");
        }
    }

    #[test]
    fn a_non_git_directory_is_an_explicit_error() {
        let dir =
            std::env::temp_dir().join(format!("mnesio-gitsuite-{}", mnesio_core::types::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let r = derive(dir.to_str().unwrap(), &[], 10);
        std::fs::remove_dir_all(&dir).ok();
        assert!(
            r.is_err(),
            "a non-repo must fail loudly, not return 0 queries"
        );
    }
}

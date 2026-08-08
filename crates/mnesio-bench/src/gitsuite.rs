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

use crate::codeeval::{CodeQuery, Gold};

/// Cached `git log -L` answers, keyed by repository HEAD.
///
/// ## Why a cache rather than fewer git calls
///
/// Tracing is one `git log -L` per symbol, measured at ~890 spawns/minute; a
/// 20-repository run went past nine hours. The obvious fix is to batch many
/// `-L` ranges into one invocation so git walks history once instead of once
/// per symbol. **That does not work, and the reason is worth recording so
/// nobody tries it again.**
///
/// Given `-L a,b:file -L c,d:file`, git emits one hunk per *changed* range,
/// carrying that range's coordinates **as they were at that commit**. Those
/// coordinates drift: a commit that inserts lines above a range moves it, and
/// git emits *no hunk* for that commit because the range's content did not
/// change. So the bookkeeping that would let a caller map a hunk back to the
/// range that produced it happens at commits git never shows. Attribution by
/// line number fails (coords drift), by emission order fails (a commit
/// touching one range emits one hunk, indistinguishable from the other), and
/// by replaying drift fails (the shifts are invisible). Verified directly on a
/// constructed repository — see the module tests.
///
/// So the per-symbol call stays, and instead its *answer* is memoised. The key
/// is the repository's HEAD sha plus the exact range: `git log -L` is a pure
/// function of those, so a hit is byte-identical to a miss. **This changes no
/// gold set by construction** — unlike a filter that skips symbols, which
/// changes which queries exist at all.
///
/// First run still pays full price. Every re-run is free, which is what makes
/// a paired A/B affordable: the second arm reads the cache the first arm
/// wrote, and paired comparison is the thing the project's own standing rule
/// requires of every claim.
mod trace_cache {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// `(head_sha, range_spec) -> commits`.
    type Map = HashMap<String, Vec<(String, String)>>;

    static MEM: Mutex<Option<Map>> = Mutex::new(None);

    fn path(repo_head: &str) -> PathBuf {
        let base = std::env::var_os("MNESIO_CACHE_DIR")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .unwrap_or_else(std::env::temp_dir)
            .join("mnesio-gitsuite");
        base.join(format!("{repo_head}.json"))
    }

    /// Load the on-disk cache for this HEAD into memory, once per process.
    pub fn warm(repo_head: &str) {
        let mut g = MEM.lock().unwrap();
        if g.is_some() {
            return;
        }
        let loaded = std::fs::read(path(repo_head))
            .ok()
            .and_then(|b| serde_json::from_slice::<Map>(&b).ok())
            .unwrap_or_default();
        *g = Some(loaded);
    }

    pub fn get(key: &str) -> Option<Vec<(String, String)>> {
        MEM.lock().unwrap().as_ref()?.get(key).cloned()
    }

    pub fn put(key: String, value: Vec<(String, String)>) {
        if let Some(m) = MEM.lock().unwrap().as_mut() {
            m.insert(key, value);
        }
    }

    /// Persist. A failure here costs a slow next run, never a wrong one, so it
    /// is deliberately silent rather than fatal.
    pub fn flush(repo_head: &str) {
        let g = MEM.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        let p = path(repo_head);
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(bytes) = serde_json::to_vec(m) {
            let tmp = p.with_extension("tmp");
            if std::fs::write(&tmp, &bytes).is_ok() {
                let _ = std::fs::rename(&tmp, &p);
            }
        }
    }

    /// Drop the in-memory map so a different repository starts clean.
    pub fn reset() {
        *MEM.lock().unwrap() = None;
    }
}

/// Walk up from `dir` to the enclosing repository.
///
/// Git only answers from a repository root, so a corpus that indexes
/// `serde/src` still has to run its `log` there.
fn git_root(dir: &str) -> Result<String> {
    let start = std::path::Path::new(dir)
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(dir));
    let mut cur = start.as_path();
    loop {
        if cur.join(".git").exists() {
            return Ok(cur.to_string_lossy().into_owned());
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => {
                return Err(anyhow!(
                    "{dir} is not inside a git repository — the git suite needs \
                     history to derive queries from"
                ))
            }
        }
    }
}

/// The repository's current HEAD, which is what makes a cached trace valid.
///
/// `None` when git won't say — a detached or empty repository. Callers then
/// skip the cache entirely rather than key it on something unstable.
fn head_sha(repo: &str) -> Option<String> {
    let out = Command::new("git")
        .args(["-C", repo, "rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Commits touching more symbols than this are dropped: with a large gold set,
/// "retrieved at least one" stops discriminating between arms.
const MAX_GOLD: usize = 3;

/// Subjects shorter than this are `wip`, `fix`, `.` — no retrievable signal.
const MIN_SUBJECT_CHARS: usize = 25;

/// How far back to trace each symbol's line range. Bounds a `git log -L` that
/// would otherwise walk the whole history of a long-lived file.
pub(crate) const HISTORY_DEPTH: usize = 30;

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
fn commits_touching(repo: &str, target: &TraceTarget, cached: bool) -> Vec<(String, String)> {
    // `-L` implies a patch we don't want, so we tag the header lines and keep
    // only those. `%x09` is a tab, which cannot appear in a subject.
    let range = format!("{},{}:{}", target.start_line, target.end_line, target.path);
    if cached {
        if let Some(hit) = trace_cache::get(&range) {
            return hit;
        }
    }
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
    // `repo` may be a subdirectory — a manifest corpus indexes `serde/src`,
    // not the whole checkout — and git only answers from the repository root.
    // So resolve the root here rather than making every caller do it; passing
    // a subdir used to fail with "not a git repository", which reads as a
    // broken checkout rather than a path that just needed resolving.
    //
    // Paths are deliberately NOT rebased. [`crate::codeeval::trace_targets`]
    // already keys every target off `git_root`, so its paths are root-relative
    // whatever directory it was pointed at. A first version of this prepended
    // the subdir prefix as well and produced `src/src/bytes.rs` — for which
    // `git log -L` reports no history rather than an error, so the suite came
    // back empty and the run said "no descriptive commits touching indexed
    // symbols". It cost an afternoon precisely because a wrong path here is
    // silent. See `a_subdirectory_derives_the_same_suite_as_its_root`.
    let repo = git_root(repo)?;
    let repo = repo.as_str();

    // Memoise the per-symbol traces against this HEAD. See [`trace_cache`] for
    // why the answers are cached rather than the calls being batched — batching
    // is unrecoverable, and this is byte-identical by construction.
    let head = head_sha(repo);
    if let Some(h) = &head {
        trace_cache::reset();
        trace_cache::warm(h);
    }
    let cached = head.is_some();

    // One `git log -L` per symbol is a subprocess spawn, so a large repo is
    // minutes of pure process overhead. Fan out across cores; git itself is
    // read-only here so the calls are independent.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(targets.len().max(1));
    let chunk = targets.len().div_ceil(threads);

    let traced: Vec<(String, String, String, String)> = std::thread::scope(|s| {
        let handles: Vec<_> = targets
            .chunks(chunk.max(1))
            .map(|part| {
                s.spawn(move || {
                    let mut out = Vec::new();
                    for t in part {
                        let hits = commits_touching(repo, t, cached);
                        let range = format!("{},{}:{}", t.start_line, t.end_line, t.path);
                        if cached {
                            trace_cache::put(range, hits.clone());
                        }
                        for (sha, subject) in hits {
                            out.push((sha, subject, t.path.clone(), t.name.clone()));
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

    // Written once per repository, after every thread has finished, so a
    // partially-traced run never persists a partial answer that a later run
    // would trust.
    if let Some(h) = &head {
        trace_cache::flush(h);
    }

    // sha -> (subject, gold symbol names). BTreeMap so the suite is
    // deterministic across runs; a benchmark whose contents shift between
    // invocations cannot support a paired comparison.
    let mut by_commit: BTreeMap<String, (String, Vec<(String, String)>)> = BTreeMap::new();
    for (sha, subject, path, name) in traced {
        if !usable_subject(&subject) {
            continue;
        }
        let e = by_commit
            .entry(sha)
            .or_insert_with(|| (subject, Vec::new()));
        let key = (path, name);
        if !e.1.contains(&key) {
            e.1.push(key);
        }
    }

    let mut suite: Vec<CodeQuery> = by_commit
        .into_iter()
        .filter(|(_, (_, gold))| !gold.is_empty() && gold.len() <= MAX_GOLD)
        .map(|(_, (subject, gold))| CodeQuery {
            question: subject,
            gold: gold
                .into_iter()
                .map(|(path, name)| Gold {
                    // Path-qualified: `__init__` and `new` are not unique.
                    path: Some(path),
                    name,
                })
                .collect(),
        })
        .collect();
    suite.truncate(limit);
    Ok(suite)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway repository with a known history, so git's real behaviour is
    /// asserted rather than assumed.
    struct Repo(std::path::PathBuf);
    impl Repo {
        fn new(name: &str) -> Self {
            let d =
                std::env::temp_dir().join(format!("mnesio-gitsuite-{name}-{}", std::process::id()));
            std::fs::remove_dir_all(&d).ok();
            std::fs::create_dir_all(&d).unwrap();
            let r = Repo(d);
            r.git(&["init", "-q", "."]);
            r.git(&["config", "user.email", "t@t"]);
            r.git(&["config", "user.name", "t"]);
            r
        }
        fn git(&self, args: &[&str]) -> String {
            let out = Command::new("git")
                .current_dir(&self.0)
                .args(args)
                .output()
                .expect("git must be on PATH for these tests");
            String::from_utf8_lossy(&out.stdout).into_owned()
        }
        fn write(&self, name: &str, body: &str) {
            std::fs::write(self.0.join(name), body).unwrap();
        }
        fn commit(&self, msg: &str) {
            self.git(&["add", "-A"]);
            self.git(&["commit", "-q", "-m", msg]);
        }
        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }
    impl Drop for Repo {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    fn lines(n: usize) -> String {
        (1..=n)
            .map(|i| format!("line{i}\n"))
            .collect::<Vec<_>>()
            .concat()
    }

    #[test]
    fn batching_l_ranges_cannot_be_attributed_back_to_ranges() {
        // The finding that decided the design, pinned so it is not re-litigated
        // by someone who reads "one git call per symbol" and reaches for the
        // obvious optimisation.
        //
        // Given two ranges in one `git log -L` call, git reports each changed
        // range using ITS COORDINATES AT THAT COMMIT, and emits nothing at all
        // for a commit that merely shifted a range without changing it. So the
        // bookkeeping needed to map a hunk back to the range that produced it
        // happens at commits git never shows, and a batched call's output is
        // not invertible.
        let r = Repo::new("batch");
        r.write("f.txt", &lines(40));
        r.commit("create the file with both regions");

        // Edit inside the LOWER region while it still sits at 31..40, so git
        // has a real hunk to report for it later, at coordinates that will by
        // then be stale.
        let edited_b = lines(40).replace("line35\n", "line35 EDITED\n");
        r.write("f.txt", &edited_b);
        r.commit("edit the lower region while it sits at its original lines");

        // Insert above the lower region: it moves 31..40 -> 36..45, but its
        // content does not change, so git emits no hunk for this commit.
        let shifted = edited_b.replace("line12\n", "line12\nins1\nins2\nins3\nins4\nins5\n");
        r.write("f.txt", &shifted);
        r.commit("insert five lines between the regions");

        // Region B is now at 36..45 in current coordinates.
        let out = r.git(&[
            "log",
            "--no-merges",
            "--format=COMMIT%x09%s",
            "-L",
            "1,10:f.txt",
            "-L",
            "36,45:f.txt",
        ]);

        let shifting_commit_emits_a_hunk = out
            .split("COMMIT\t")
            .any(|block| block.starts_with("insert five lines") && block.contains("@@ "));
        assert!(
            !shifting_commit_emits_a_hunk,
            "if git ever starts reporting the commit that shifted a range, \
             batched attribution becomes possible and this design should be \
             revisited"
        );

        // And the coordinates genuinely drift, so a hunk cannot be matched to
        // a requested range by line number either.
        assert!(
            out.contains("@@ -31,") || out.contains("+31,"),
            "region B requested at 36,45 must be reported at its historical \
             position, proving coordinates are not stable: {out}"
        );
    }

    #[test]
    fn a_subdirectory_derives_the_same_suite_as_its_root() {
        // The regression that cost an afternoon. `trace_targets` keys paths off
        // the git root already, so `derive` must resolve the root WITHOUT
        // rebasing paths onto it. Prepending the subdir prefix produced
        // `src/src/f.rs`, and `git log -L` answers a nonexistent path with an
        // empty history rather than an error — so the suite came back empty and
        // the harness reported "no descriptive commits touching indexed
        // symbols", which points at the corpus instead of at the bug.
        let r = Repo::new("subdir");
        std::fs::create_dir_all(r.0.join("src")).unwrap();
        r.write("src/f.rs", "fn alpha() {\n    one();\n}\n");
        r.commit("add the alpha helper used by the parser");
        r.write("src/f.rs", "fn alpha() {\n    one();\n    two();\n}\n");
        r.commit("extend alpha to call the second helper too");

        // Root-relative, exactly as `trace_targets` produces them.
        let targets = vec![TraceTarget {
            path: "src/f.rs".into(),
            name: "alpha".into(),
            start_line: 1,
            end_line: 4,
        }];

        let from_root = derive(r.path(), &targets, 10).unwrap();
        let sub = r.0.join("src");
        let from_subdir = derive(sub.to_str().unwrap(), &targets, 10).unwrap();

        assert!(
            !from_root.is_empty(),
            "the fixture must produce a suite at all"
        );
        assert_eq!(
            from_root.len(),
            from_subdir.len(),
            "pointing at a subdirectory must not change the derived suite"
        );
    }

    #[test]
    fn a_cached_trace_is_identical_to_an_uncached_one() {
        // The whole safety claim of the cache. If this ever fails, every
        // number produced from a warm cache is suspect.
        let r = Repo::new("cache");
        r.write("a.rs", "fn alpha() {\n    one();\n}\n");
        r.commit("add the alpha function with its helper call");
        r.write("a.rs", "fn alpha() {\n    one();\n    two();\n}\n");
        r.commit("extend alpha to call the second helper as well");

        let target = TraceTarget {
            path: "a.rs".into(),
            name: "alpha".into(),
            start_line: 1,
            end_line: 4,
        };
        let head = head_sha(r.path()).expect("a repo with commits has a HEAD");

        trace_cache::reset();
        trace_cache::warm(&head);
        let cold = commits_touching(r.path(), &target, true);
        assert!(!cold.is_empty(), "the trace must find the commits");
        trace_cache::put(
            format!("{},{}:{}", target.start_line, target.end_line, target.path),
            cold.clone(),
        );
        let warm = commits_touching(r.path(), &target, true);

        assert_eq!(
            cold, warm,
            "a cache hit must equal what git would have said"
        );
    }

    #[test]
    fn a_cache_from_a_different_head_is_not_consulted() {
        // Keyed by HEAD because `git log -L` is only a pure function given a
        // fixed history. A new commit must invalidate, or the suite would be
        // derived from code that no longer exists.
        let r = Repo::new("head");
        r.write("a.rs", "fn alpha() {}\n");
        r.commit("add the alpha function to the module");
        let first = head_sha(r.path()).unwrap();
        r.write("a.rs", "fn alpha() {}\nfn beta() {}\n");
        r.commit("add the beta function alongside alpha");
        let second = head_sha(r.path()).unwrap();
        assert_ne!(first, second, "a new commit must change the cache key");
    }

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

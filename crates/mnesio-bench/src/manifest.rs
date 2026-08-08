//! Phase 18H: a code-retrieval corpus somebody else can re-run.
//!
//! ## What went wrong without one
//!
//! `scaleeval --root ~/Aniket` walks whatever repositories happen to be on the
//! author's disk. Three attempts at it produced no numbers: 9h33m without
//! finishing one arm, then a stall, then 6 of 20 repositories in 5 hours with
//! the four largest still ahead (24,718 / 16,038 / 12,423 / 10,035 source
//! files). The failure was not a bug — it was the corpus. It is private, it
//! drifts as those repositories change, its size distribution is accidental,
//! and nobody outside this machine can reproduce it.
//!
//! 18H asks for something specific: *"a third party can reproduce them from
//! the manifest."* A suite that takes ten hours on the author's laptop fails
//! that regardless of how correct it is.
//!
//! ## What a manifest fixes
//!
//! - **Public repositories**, so the corpus can be fetched by anyone.
//! - **Pinned commits**, so "the same benchmark" means the same bytes. A repo
//!   that moves is a different corpus, and comparing across it is comparing
//!   two things.
//! - **Declared size caps**, so the runtime is bounded by construction rather
//!   than discovered after five hours.
//! - **A declared budget**, so "too slow" is a check the harness performs and
//!   reports rather than a judgement someone makes at 2am.
//!
//! ## What it deliberately does not do
//!
//! It does not make the corpus *large*. Every repository here is capped at
//! tens of files, which is small next to a real monorepo and small next to
//! what a competitor claims. That is the trade: a suite that runs in under an
//! hour and can be re-run by a stranger, versus one that is bigger and has
//! never successfully completed. The first is worth publishing.
//!
//! It also does not invent commit shas. `pin` is null in the shipped file
//! until [`Manifest::pin`] resolves it against the real remote — a hand-written
//! sha is a benchmark that cannot be reproduced, which is the failure this
//! module exists to prevent.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

/// The shipped corpus definition.
pub const DEFAULT_MANIFEST: &str = include_str!("../manifest/codeeval-v1.json");

/// One repository in the corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoSpec {
    pub name: String,
    pub url: String,
    pub language: String,
    /// Index only this path within the repository. Part of the corpus
    /// definition: changing it changes the benchmark.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subdir: Option<String>,
    /// Hard cap. A repository over this is skipped and reported, never
    /// truncated — truncating would alter the corpus without altering the
    /// manifest, so two runs of "the same" suite would measure different code.
    pub max_files: usize,
    pub queries: usize,
    /// Exact commit. `None` until [`Manifest::pin`] resolves it.
    pub pin: Option<String>,
}

/// The corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    pub name: String,
    pub description: String,
    /// Wall-clock the whole suite is expected to fit in. Reported against
    /// actual, so a corpus that outgrows its budget is visible immediately
    /// rather than after an overnight run.
    pub budget_seconds: u64,
    #[serde(default)]
    pub notes: Vec<String>,
    pub repos: Vec<RepoSpec>,
}

/// Schema version this build understands.
const SCHEMA: u32 = 1;

impl Manifest {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let raw = match path {
            Some(p) => std::fs::read_to_string(p)
                .with_context(|| format!("reading manifest {}", p.display()))?,
            None => DEFAULT_MANIFEST.to_string(),
        };
        let m: Manifest = serde_json::from_str(&raw).context("parsing manifest")?;
        if m.schema != SCHEMA {
            bail!(
                "manifest schema {} but this build understands {SCHEMA}; \
                 refusing to guess at the difference",
                m.schema
            );
        }
        if m.repos.is_empty() {
            bail!("manifest {} lists no repositories", m.name);
        }
        Ok(m)
    }

    /// Are all repositories pinned?
    ///
    /// An unpinned manifest can still be *run*, but its numbers are not
    /// reproducible, so the report says so rather than letting a reader assume
    /// otherwise.
    pub fn is_pinned(&self) -> bool {
        self.repos.iter().all(|r| r.pin.is_some())
    }

    /// Resolve every repository's current default-branch head and record it.
    ///
    /// Network. Writes the pinned manifest to `out`, which is the artifact a
    /// third party actually re-runs — the shipped file is a *template* until
    /// this has been done once.
    pub fn pin(&mut self, out: &Path) -> Result<()> {
        for r in &mut self.repos {
            let sha = remote_head(&r.url)
                .with_context(|| format!("resolving head of {} ({})", r.name, r.url))?;
            eprintln!("  pinned {:<12} {sha}", r.name);
            r.pin = Some(sha);
        }
        let json = serde_json::to_string_pretty(self)? + "\n";
        std::fs::write(out, json).with_context(|| format!("writing {}", out.display()))?;
        Ok(())
    }
}

/// Ask a remote for its default branch head without cloning.
fn remote_head(url: &str) -> Result<String> {
    let out = Command::new("git")
        .args(["ls-remote", url, "HEAD"])
        .output()
        .context("running git ls-remote")?;
    if !out.status.success() {
        bail!(
            "git ls-remote failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(str::to_string)
        .ok_or_else(|| anyhow!("git ls-remote returned nothing for {url}"))
}

/// How deep to clone.
///
/// Must exceed `gitsuite::HISTORY_DEPTH` by a wide margin: the suite derives
/// queries from commits touching each symbol's line range, and a clone too
/// shallow to contain them yields an empty suite that looks like "this
/// repository has no usable history" rather than "you cloned too little".
const CLONE_DEPTH: usize = 400;

/// A clone shallower than the traced window yields an empty suite that reads as
/// "this repository has no usable history" rather than "you fetched too
/// little". Enforced at compile time so the two constants cannot drift apart.
const _: () = assert!(CLONE_DEPTH > crate::gitsuite::HISTORY_DEPTH * 4);

/// Where a materialised checkout lives.
pub fn checkout_dir(base: &Path, spec: &RepoSpec) -> PathBuf {
    base.join(&spec.name)
}

/// Fetch `spec` at its pinned commit, or reuse an existing checkout.
///
/// Idempotent: a second call with the same pin is a no-op, so re-running the
/// suite does not re-download anything.
pub fn materialize(base: &Path, spec: &RepoSpec) -> Result<PathBuf> {
    let dir = checkout_dir(base, spec);
    let Some(pin) = &spec.pin else {
        bail!(
            "{} is not pinned; run `mnesio-bench manifest pin` first — an \
             unpinned corpus cannot be reproduced",
            spec.name
        );
    };

    if dir.join(".git").exists() {
        // Already here. Only move if the pin changed.
        let at = Command::new("git")
            .args(["-C", dir.to_str().unwrap(), "rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
        if at.as_deref() == Some(pin.as_str()) {
            return Ok(dir);
        }
    } else {
        std::fs::create_dir_all(&dir)?;
        run_git(&["init", "-q", dir.to_str().unwrap()])?;
        run_git(&[
            "-C",
            dir.to_str().unwrap(),
            "remote",
            "add",
            "origin",
            &spec.url,
        ])?;
    }

    run_git(&[
        "-C",
        dir.to_str().unwrap(),
        "fetch",
        "-q",
        "--depth",
        &CLONE_DEPTH.to_string(),
        "origin",
        pin,
    ])
    .with_context(|| {
        format!(
            "fetching {} at {pin} — if this says 'not our ref', the pin \
             predates the remote's history or was rewritten",
            spec.name
        )
    })?;
    run_git(&["-C", dir.to_str().unwrap(), "checkout", "-q", pin])?;
    Ok(dir)
}

fn run_git(args: &[&str]) -> Result<()> {
    let out = Command::new("git").args(args).output()?;
    if !out.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

/// The directory this spec says to index.
pub fn index_root(checkout: &Path, spec: &RepoSpec) -> PathBuf {
    match &spec.subdir {
        Some(s) => checkout.join(s),
        None => checkout.to_path_buf(),
    }
}

/// Source files under `root`, for the size guard.
///
/// Counts what the indexer would actually parse rather than everything on
/// disk, so a repository is not rejected for its fixtures or its docs.
pub fn count_source_files(root: &Path) -> usize {
    fn walk(dir: &Path, n: &mut usize) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || name == "node_modules" || name == "target" {
                continue;
            }
            if p.is_dir() {
                walk(&p, n);
            } else if matches!(
                p.extension().and_then(|s| s.to_str()),
                Some(
                    "rs" | "py"
                        | "ts"
                        | "tsx"
                        | "js"
                        | "jsx"
                        | "go"
                        | "java"
                        | "rb"
                        | "c"
                        | "cc"
                        | "cpp"
                        | "h"
                        | "hpp"
                        | "cs"
                        | "swift"
                        | "kt"
                )
            ) {
                *n += 1;
            }
        }
    }
    let mut n = 0;
    walk(root, &mut n);
    n
}

/// Why a repository did not contribute to the run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "reason")]
pub enum Skipped {
    /// Over `max_files`. The guard that makes the budget real.
    TooLarge {
        found: usize,
        cap: usize,
    },
    /// The declared `subdir` is not in the checkout — usually a pin from
    /// before a reorganisation.
    MissingSubdir {
        path: String,
    },
    Unpinned,
    Fetch {
        error: String,
    },
}

/// Decide whether a materialised repository may run.
///
/// Split from fetching so the guard is testable without a network.
pub fn admit(checkout: &Path, spec: &RepoSpec) -> Result<PathBuf, Skipped> {
    let root = index_root(checkout, spec);
    if !root.is_dir() {
        return Err(Skipped::MissingSubdir {
            path: spec.subdir.clone().unwrap_or_else(|| ".".into()),
        });
    }
    let found = count_source_files(&root);
    if found > spec.max_files {
        return Err(Skipped::TooLarge {
            found,
            cap: spec.max_files,
        });
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(max_files: usize, subdir: Option<&str>) -> RepoSpec {
        RepoSpec {
            name: "t".into(),
            url: "https://example.invalid/t".into(),
            language: "rust".into(),
            subdir: subdir.map(str::to_string),
            max_files,
            queries: 60,
            pin: Some("0".repeat(40)),
        }
    }

    struct Dir(PathBuf);
    impl Dir {
        /// Unique per call, not per process. Tests in one binary share a pid,
        /// so a pid-keyed directory has them deleting each other's fixtures
        /// mid-run — which presents as an assertion failure in whichever test
        /// happens to lose the race rather than as the fixture bug it is.
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let d = std::env::temp_dir().join(format!(
                "mnesio-manifest-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::remove_dir_all(&d).ok();
            std::fs::create_dir_all(&d).unwrap();
            Dir(d)
        }
        fn file(&self, rel: &str) {
            let p = self.0.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, "fn x() {}\n").unwrap();
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    #[test]
    fn the_shipped_manifest_parses_and_is_self_consistent() {
        // If this file is malformed nothing downstream can run, and the error
        // would surface halfway through a long benchmark rather than at start.
        let m = Manifest::load(None).expect("the shipped manifest must parse");
        assert_eq!(m.schema, SCHEMA);
        assert!(m.repos.len() >= 5, "a corpus needs breadth to say anything");
        let mut names: Vec<_> = m.repos.iter().map(|r| r.name.as_str()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "repository names must be unique");
        for r in &m.repos {
            assert!(
                r.url.starts_with("https://"),
                "{} must be fetchable",
                r.name
            );
            assert!(r.max_files > 0, "{} needs a real cap", r.name);
        }
    }

    #[test]
    fn the_shipped_manifest_spans_more_than_one_language() {
        // A single-language corpus measures one parser, and the parser is the
        // component whose weakness is already documented.
        let m = Manifest::load(None).unwrap();
        let mut langs: Vec<_> = m.repos.iter().map(|r| r.language.as_str()).collect();
        langs.sort_unstable();
        langs.dedup();
        assert!(langs.len() >= 3, "got only {langs:?}");
    }

    #[test]
    fn the_shipped_manifest_ships_unpinned() {
        // Deliberate. Hand-written shas cannot be verified and would make the
        // corpus unreproducible in exactly the way this module exists to stop.
        // `pin` resolves them against the real remotes.
        let m = Manifest::load(None).unwrap();
        assert!(
            !m.is_pinned(),
            "shipped manifest must not carry invented commits"
        );
    }

    #[test]
    fn a_repository_over_its_cap_is_skipped_not_truncated() {
        // The guard that makes the budget real. Truncating instead would
        // change the corpus without changing the manifest, so two runs of the
        // "same" suite would measure different code.
        let d = Dir::new();
        for i in 0..10 {
            d.file(&format!("src/f{i}.rs"));
        }
        assert_eq!(
            admit(&d.0, &spec(5, Some("src"))),
            Err(Skipped::TooLarge { found: 10, cap: 5 })
        );
        assert!(admit(&d.0, &spec(50, Some("src"))).is_ok());
    }

    #[test]
    fn a_missing_subdir_is_reported_rather_than_silently_indexing_everything() {
        // A pin from before a reorganisation must not quietly widen the corpus
        // to the whole repository.
        let d = Dir::new();
        d.file("src/a.rs");
        assert_eq!(
            admit(&d.0, &spec(50, Some("does/not/exist"))),
            Err(Skipped::MissingSubdir {
                path: "does/not/exist".into()
            })
        );
    }

    #[test]
    fn vendored_and_build_directories_do_not_count_toward_the_cap() {
        // Otherwise a repository is rejected for code the indexer never reads.
        let d = Dir::new();
        d.file("src/a.rs");
        d.file("node_modules/dep/index.js");
        d.file("target/debug/build.rs");
        d.file(".git/hooks/x.py");
        assert_eq!(count_source_files(&d.0), 1);
    }

    #[test]
    fn an_unpinned_repository_refuses_to_materialise() {
        let d = Dir::new();
        let mut s = spec(50, None);
        s.pin = None;
        let e = materialize(&d.0, &s).unwrap_err().to_string();
        assert!(e.contains("not pinned"), "got: {e}");
    }

    #[test]
    fn a_manifest_from_a_future_schema_is_refused() {
        // Better a clear error than a partial parse that drops the field which
        // changed the corpus.
        let raw = r#"{"schema":99,"name":"x","description":"","budget_seconds":1,
                      "repos":[{"name":"a","url":"https://e.invalid/a","language":"rust",
                                "max_files":1,"queries":1,"pin":null}]}"#;
        let p = std::env::temp_dir().join(format!("mnesio-future-{}.json", std::process::id()));
        std::fs::write(&p, raw).unwrap();
        let e = Manifest::load(Some(&p)).unwrap_err().to_string();
        std::fs::remove_file(&p).ok();
        assert!(e.contains("schema"), "got: {e}");
    }
}

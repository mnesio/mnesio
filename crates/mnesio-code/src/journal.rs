//! Phase 18: the outcome journal — where a real repository's evidence lands.
//!
//! ## Why a journal and not the event log
//!
//! Outcomes are observed by whichever process is attached to the editor: the
//! MCP server, running as a subprocess of Claude Code or Cursor. The dashboard
//! is a *different* process. They cannot share a fjall handle — fjall is
//! single-writer — so the two need a medium that tolerates one appender and
//! many concurrent readers.
//!
//! That medium is a line-per-outcome file opened `O_APPEND`. The kernel makes
//! the seek-and-write atomic, so an editor writing while the dashboard reads
//! cannot interleave two records, and a reader never has to take a lock.
//!
//! ## What this is honestly not
//!
//! Hard Rule #4 says the event log is the single system of record. This file
//! is **not** that log, and calling it one would be a lie the architecture
//! would eventually collect on. It is a *transport*: the journal is where an
//! out-of-process observer parks evidence until a process that owns the log
//! folds it in as [`mnesio_core::Event::ObservationRecorded`]. Until that fold
//! runs, the journal is the only copy — which is exactly why it is append-only
//! and never rewritten in place.
//!
//! The consequence to keep in view: a curve computed straight from the journal
//! is derived from a store the log cannot yet rebuild. It is real evidence
//! about a real repository, and it is not yet replayable. Both halves are true
//! and the API says so.
//!
//! ## Reading a file that is being appended to
//!
//! A reader can arrive mid-write, and a crash can leave a torn final line. So
//! parsing is per-line and lenient: an unparseable line is *counted and
//! reported*, never silently dropped. A journal that quietly discards a tenth
//! of its evidence would produce a confident curve from a biased sample, which
//! is worse than an obviously broken one.

use std::io::Write;
use std::path::{Path, PathBuf};

use mnesio_core::MnesioError;

use crate::outcome::CodeOutcome;
use crate::persist::cache_base;

/// One line of the journal: an outcome plus when it was observed.
///
/// The timestamp is the journal's own, not the caller's. An agent reporting
/// its own clock could reorder the sequence the curve is computed over, and
/// ordering is the only structure a time series has.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JournalEntry {
    /// Milliseconds since the Unix epoch, recorded at append time.
    pub observed_ms: u64,
    #[serde(flatten)]
    pub outcome: CodeOutcome,
}

/// What a read of the journal found, including what it could not parse.
#[derive(Debug, Clone, Default)]
pub struct JournalRead {
    pub entries: Vec<JournalEntry>,
    /// Lines that failed to parse. Surfaced rather than swallowed — see the
    /// module docs on why a silently-filtered journal is the dangerous case.
    pub skipped: usize,
}

/// An append-only outcome journal for one repository.
#[derive(Debug, Clone)]
pub struct OutcomeJournal {
    path: PathBuf,
}

impl OutcomeJournal {
    /// The journal for `repo`, under the user's cache directory.
    ///
    /// Beside the embedding cache rather than inside the working tree: this is
    /// derived data about a checkout, and writing into someone's repository is
    /// a surprise they did not ask for (and a diff they did not want).
    pub fn for_repo(repo: &Path) -> Self {
        Self::for_repo_in(&cache_base(), repo)
    }

    /// [`OutcomeJournal::for_repo`] under an explicit cache root, so tests can
    /// point at their own directory by argument instead of by mutating a
    /// process-global environment variable.
    pub fn for_repo_in(base: &Path, repo: &Path) -> Self {
        Self {
            path: crate::persist::cache_path_in(base, repo).with_extension("outcomes.jsonl"),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one outcome.
    ///
    /// Opened `O_APPEND` and written as a single `write_all`, so two processes
    /// appending at once produce two whole lines in some order rather than one
    /// interleaved corrupt one. The file handle is opened per call: this runs
    /// once per *edit*, so the open cost is irrelevant beside the guarantee
    /// that nothing holds a stale descriptor across a rebuild or a `git clean`.
    pub fn append(&self, outcome: &CodeOutcome) -> Result<(), MnesioError> {
        let entry = JournalEntry {
            observed_ms: now_ms(),
            outcome: outcome.clone(),
        };
        let mut line = serde_json::to_vec(&entry)
            .map_err(|e| MnesioError::Index(format!("serialising outcome: {e}")))?;
        line.push(b'\n');

        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| MnesioError::Index(format!("journal dir {}: {e}", dir.display())))?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| MnesioError::Index(format!("opening {}: {e}", self.path.display())))?;
        f.write_all(&line).map_err(|e| {
            MnesioError::Index(format!("appending to {}: {e}", self.path.display()))
        })?;
        tracing::debug!(path = %self.path.display(), result = outcome.result.as_str(), "outcome journalled");
        Ok(())
    }

    /// Read every entry, in append order.
    ///
    /// A missing journal is an empty read, not an error: a repository nobody
    /// has recorded an outcome for yet is the *normal* first state, and the
    /// dashboard has to render it as "no evidence yet" rather than as a fault.
    pub fn read(&self) -> JournalRead {
        let Ok(bytes) = std::fs::read(&self.path) else {
            return JournalRead::default();
        };
        let mut out = JournalRead::default();
        for line in bytes.split(|b| *b == b'\n') {
            if line.is_empty() {
                continue;
            }
            match serde_json::from_slice::<JournalEntry>(line) {
                Ok(e) => out.entries.push(e),
                Err(_) => out.skipped += 1,
            }
        }
        if out.skipped > 0 {
            tracing::warn!(
                skipped = out.skipped,
                kept = out.entries.len(),
                path = %self.path.display(),
                "outcome journal has unparseable lines"
            );
        }
        out
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::EditResult;
    use crate::pack::{Form, PackedContext, PackedSymbol, Reason};
    use mnesio_core::types::{new_id, MemoryRef};

    struct Sandbox(PathBuf);
    impl Sandbox {
        fn new() -> Self {
            let d = std::env::temp_dir().join(format!("mnesio-journal-{}", new_id()));
            std::fs::create_dir_all(&d).unwrap();
            Self(d)
        }
        fn repo(&self, name: &str) -> PathBuf {
            let p = self.0.join(name);
            std::fs::create_dir_all(&p).unwrap();
            p
        }
    }
    impl Drop for Sandbox {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    fn outcome(task: &str, result: EditResult) -> CodeOutcome {
        let ctx = PackedContext {
            symbols: vec![PackedSymbol {
                memory: MemoryRef(new_id()),
                form: Form::Full,
                tokens: 10,
                reason: Reason::Seed(0),
            }],
            tokens_used: 10,
            ..Default::default()
        };
        CodeOutcome::from_context(task, "repo", &ctx, result)
    }

    #[test]
    fn outcomes_round_trip_in_append_order() {
        let s = Sandbox::new();
        let j = OutcomeJournal::for_repo_in(&s.0, &s.repo("r"));
        j.append(&outcome("first", EditResult::Passed)).unwrap();
        j.append(&outcome("second", EditResult::BuildFailed))
            .unwrap();

        let read = j.read();
        assert_eq!(read.skipped, 0);
        // Order is the only structure a time series has; a set would be useless.
        let tasks: Vec<_> = read
            .entries
            .iter()
            .map(|e| e.outcome.task.as_str())
            .collect();
        assert_eq!(tasks, ["first", "second"]);
        assert_eq!(read.entries[1].outcome.result, EditResult::BuildFailed);
    }

    #[test]
    fn attribution_survives_the_journal() {
        // The whole point of writing outcomes down is that the compiler can
        // later ask *which* symbol was being credited. A journal that loses
        // that is just a success counter.
        let s = Sandbox::new();
        let j = OutcomeJournal::for_repo_in(&s.0, &s.repo("r"));
        let o = outcome("t", EditResult::Passed);
        j.append(&o).unwrap();
        let back = j.read();
        assert_eq!(back.entries[0].outcome.symbols, o.symbols);
        assert_eq!(back.entries[0].outcome.tokens_used, 10);
    }

    #[test]
    fn a_torn_final_line_does_not_discard_the_history() {
        // A crash mid-append leaves a partial line. Everything before it is
        // still valid evidence and must survive.
        let s = Sandbox::new();
        let repo = s.repo("r");
        let j = OutcomeJournal::for_repo_in(&s.0, &repo);
        j.append(&outcome("good", EditResult::Passed)).unwrap();
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(j.path())
            .unwrap();
        f.write_all(b"{\"observed_ms\":1,\"task\":\"tor").unwrap();

        let read = j.read();
        assert_eq!(read.entries.len(), 1, "the intact record must survive");
        assert_eq!(
            read.skipped, 1,
            "and the damage must be reported, not hidden"
        );
    }

    #[test]
    fn a_missing_journal_is_an_empty_read_not_a_failure() {
        // The first state of every repository. Rendering it as an error would
        // make a working install look broken.
        let s = Sandbox::new();
        let read = OutcomeJournal::for_repo_in(&s.0, Path::new("/no/such/repo")).read();
        assert!(read.entries.is_empty());
        assert_eq!(read.skipped, 0);
    }

    #[test]
    fn two_repositories_keep_separate_journals() {
        let s = Sandbox::new();
        let a = OutcomeJournal::for_repo_in(&s.0, &s.repo("alpha"));
        let b = OutcomeJournal::for_repo_in(&s.0, &s.repo("beta"));
        a.append(&outcome("only-alpha", EditResult::Passed))
            .unwrap();
        assert_ne!(a.path(), b.path());
        assert_eq!(
            b.read().entries.len(),
            0,
            "scope is a boundary (Hard Rule #3)"
        );
    }

    #[test]
    fn appends_accumulate_rather_than_replace() {
        // O_APPEND, not truncate. Losing history here would silently reset the
        // curve every time the editor restarted.
        let s = Sandbox::new();
        let repo = s.repo("r");
        for i in 0..5 {
            OutcomeJournal::for_repo_in(&s.0, &repo)
                .append(&outcome(&format!("t{i}"), EditResult::Passed))
                .unwrap();
        }
        assert_eq!(
            OutcomeJournal::for_repo_in(&s.0, &repo)
                .read()
                .entries
                .len(),
            5
        );
    }
}

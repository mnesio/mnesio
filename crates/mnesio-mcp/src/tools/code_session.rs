//! What was served, so what came of it can be attributed.
//!
//! ## The join
//!
//! `mnesio_code_context` hands an agent a packed set of symbols; some minutes
//! later `mnesio_code_outcome` reports what happened. Between those two calls
//! the agent edits, builds and runs tests, and MCP is stateless across them —
//! the outcome call carries a task string and a verdict, nothing else.
//!
//! Without the symbols the verdict is unlearnable. "The retrieval helped" is
//! not a fact the compiler can act on; "these three symbols, two of them
//! seeds and one pulled in by expansion, were present when the build failed"
//! is. So the packed context is held here between the two calls.
//!
//! ## Why an unmatched outcome is refused rather than recorded
//!
//! The tempting shortcut is to journal an outcome with an empty symbol list
//! when nothing matches. It would make the curve fill in faster and it would
//! be worthless: an outcome from an edit that never used mnesio's context says
//! nothing about mnesio's retrieval, but it would still move the success rate
//! that gets pointed at as evidence the loop works.
//!
//! That failure is invisible once it happens — the number looks the same
//! either way — so it has to be prevented at the point of entry.
//! [`SessionStore::lookup`] returns the recent tasks on a miss so the agent
//! can retry with a string that matches, rather than being told "no" with no
//! way forward.
//!
//! ## Bounds
//!
//! A long-lived editor session issues many retrievals, most of which never get
//! an outcome. The store is capped and evicts oldest-first (Hard Rule #6): the
//! cost of forgetting an old context is one unattributable outcome, and the
//! cost of not forgetting is unbounded memory in a process the user did not
//! start deliberately.

use std::collections::VecDeque;

use mnesio_code::pack::PackedContext;

/// How many served contexts to remember.
///
/// Generous enough that an agent working through a task list can report
/// outcomes out of order, small enough to be irrelevant to a process the user
/// did not choose to start.
const CAPACITY: usize = 64;

/// One retrieval that has not yet been answered for.
#[derive(Debug, Clone)]
struct Served {
    repo: String,
    task_key: String,
    /// The task exactly as the agent phrased it, for the miss message. The
    /// normalised key is unreadable and telling someone their string didn't
    /// match a string they can't see is a dead end.
    task_display: String,
    context: PackedContext,
}

/// Recently served contexts, oldest first.
#[derive(Debug, Default)]
pub struct SessionStore {
    served: VecDeque<Served>,
}

/// Normalise a task string so trivial rephrasing between the two calls does
/// not break the join.
///
/// Case and whitespace only. Anything cleverer — stemming, fuzzy distance —
/// would start matching *different* tasks to each other, which silently
/// attributes an outcome to the wrong context. A miss the agent can fix is
/// strictly better than a wrong match nobody can see.
fn key(task: &str) -> String {
    task.split_whitespace()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Canonical form of a repository path, so `.` and an absolute path are the
/// same repository. The journal is keyed the same way; if these two disagreed,
/// outcomes would land in a journal the dashboard never reads.
pub fn repo_key(repo: &str) -> String {
    std::path::Path::new(repo)
        .canonicalize()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| repo.to_string())
}

/// Why an outcome could not be attributed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unmatched {
    /// Tasks still in the store for this repository, most recent first.
    pub recent: Vec<String>,
}

impl SessionStore {
    /// Record what was served.
    ///
    /// A repeat of the same task replaces the older entry: the agent asked
    /// again, so the fresher context is the one its next edit will be based on.
    pub fn remember(&mut self, repo: &str, task: &str, context: PackedContext) {
        let repo = repo_key(repo);
        let task_key = key(task);
        self.served
            .retain(|s| !(s.repo == repo && s.task_key == task_key));
        self.served.push_back(Served {
            repo,
            task_key,
            task_display: task.to_string(),
            context,
        });
        while self.served.len() > CAPACITY {
            self.served.pop_front();
        }
    }

    /// Find the context an outcome is about.
    ///
    /// Returns a clone rather than removing the entry: an agent iterating on
    /// one task legitimately reports several outcomes against the same context
    /// — build failed, fixed it, tests passed — and each is real evidence.
    /// Consuming on first read would discard every outcome after the first.
    pub fn lookup(&self, repo: &str, task: &str) -> Result<PackedContext, Unmatched> {
        let repo = repo_key(repo);
        let task_key = key(task);
        if let Some(s) = self
            .served
            .iter()
            .rev()
            .find(|s| s.repo == repo && s.task_key == task_key)
        {
            return Ok(s.context.clone());
        }
        Err(Unmatched {
            recent: self
                .served
                .iter()
                .rev()
                .filter(|s| s.repo == repo)
                .map(|s| s.task_display.clone())
                .take(5)
                .collect(),
        })
    }

    pub fn len(&self) -> usize {
        self.served.len()
    }
    pub fn is_empty(&self) -> bool {
        self.served.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_code::pack::{Form, PackedSymbol, Reason};
    use mnesio_core::types::{new_id, MemoryRef};

    fn ctx(tokens: usize) -> PackedContext {
        PackedContext {
            symbols: vec![PackedSymbol {
                memory: MemoryRef(new_id()),
                form: Form::Full,
                tokens,
                reason: Reason::Seed(0),
            }],
            tokens_used: tokens,
            ..Default::default()
        }
    }

    #[test]
    fn an_outcome_finds_the_context_that_produced_it() {
        let mut s = SessionStore::default();
        s.remember(".", "fix the retry backoff", ctx(100));
        assert_eq!(
            s.lookup(".", "fix the retry backoff").unwrap().tokens_used,
            100
        );
    }

    #[test]
    fn case_and_spacing_do_not_break_the_join() {
        // The agent re-types the task from memory between the two calls; that
        // must not silently cost an outcome.
        let mut s = SessionStore::default();
        s.remember(".", "Fix the retry backoff", ctx(1));
        assert!(s.lookup(".", "  fix   THE retry   backoff ").is_ok());
    }

    #[test]
    fn an_unmatched_outcome_is_refused_and_says_what_would_match() {
        // The central rule of this module: an outcome from an edit that never
        // used our context is not evidence about our retrieval. Recording it
        // anyway would move the success rate invisibly.
        let mut s = SessionStore::default();
        s.remember(".", "fix the retry backoff", ctx(1));
        let err = s.lookup(".", "something else entirely").unwrap_err();
        assert_eq!(err.recent, ["fix the retry backoff"]);
    }

    #[test]
    fn a_different_repository_is_a_miss_not_a_match() {
        // Scope is a security boundary (Hard Rule #3) and also a correctness
        // one: attributing repo A's outcome to repo B's context poisons both.
        let mut s = SessionStore::default();
        s.remember(".", "task", ctx(1));
        let err = s.lookup("/somewhere/else", "task").unwrap_err();
        assert!(err.recent.is_empty(), "must not leak another repo's tasks");
    }

    #[test]
    fn several_outcomes_can_be_reported_against_one_retrieval() {
        // build failed → fix → tests passed. Both are evidence about the same
        // context; consuming the entry on first read would discard the second.
        let mut s = SessionStore::default();
        s.remember(".", "t", ctx(7));
        assert!(s.lookup(".", "t").is_ok());
        assert!(s.lookup(".", "t").is_ok());
    }

    #[test]
    fn asking_again_replaces_the_stale_context() {
        // The agent re-retrieved, so its next edit is based on the new set.
        let mut s = SessionStore::default();
        s.remember(".", "t", ctx(10));
        s.remember(".", "t", ctx(20));
        assert_eq!(s.len(), 1, "not two entries for one task");
        assert_eq!(s.lookup(".", "t").unwrap().tokens_used, 20);
    }

    #[test]
    fn the_store_is_bounded_and_evicts_oldest_first() {
        // Hard Rule #6. An editor session issues far more retrievals than
        // outcomes; unbounded growth in a process the user did not start is
        // not acceptable.
        let mut s = SessionStore::default();
        for i in 0..CAPACITY + 20 {
            s.remember(".", &format!("task {i}"), ctx(1));
        }
        assert_eq!(s.len(), CAPACITY);
        assert!(s.lookup(".", "task 0").is_err(), "oldest evicted");
        assert!(s.lookup(".", &format!("task {}", CAPACITY + 19)).is_ok());
    }

    #[test]
    fn the_miss_message_is_capped() {
        // It goes to a model with a token budget, not a log file.
        let mut s = SessionStore::default();
        for i in 0..20 {
            s.remember(".", &format!("task {i}"), ctx(1));
        }
        assert_eq!(s.lookup(".", "nope").unwrap_err().recent.len(), 5);
    }

    #[test]
    fn relative_and_absolute_paths_are_the_same_repository() {
        // The agent passes `.`; the dashboard reads an absolute path. If these
        // disagreed, every outcome would land somewhere nothing reads.
        let cwd = std::env::current_dir().unwrap();
        let mut s = SessionStore::default();
        s.remember(".", "t", ctx(3));
        assert!(s.lookup(cwd.to_str().unwrap(), "t").is_ok());
    }
}

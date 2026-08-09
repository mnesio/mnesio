//! A real [`CurveIndex`] over an indexed repository.
//!
//! `learncurve` was built and tested against trait fakes: they proved the loop
//! refuses a canary-breaking rule and that a helpful suppression reaches
//! held-out tasks. What a fake cannot show is whether the mechanism moves a
//! number on code somebody actually wrote, which is the whole claim.
//!
//! ## Suppression, honestly
//!
//! A committed rule is `-exclude:<memory>` for a query class. Applying it here
//! means dropping those memories from the packed context *after* retrieval
//! rather than teaching the retriever to avoid them. That is a faithful model
//! of what the rule does — a `RetrievalRule` is a post-filter on the candidate
//! set, not a re-training of the embedder — but it is worth stating, because a
//! reader could reasonably assume "suppression" meant the index itself changed.
//!
//! To keep the comparison fair, over-fetch is deliberately **not** widened when
//! symbols are suppressed. Refilling the slot a suppressed symbol vacated would
//! conflate two effects: removing a bad result, and showing one more result.
//! Only the first is what a suppression rule claims to do.

use std::collections::HashSet;

use anyhow::Result;

use mnesio_code::CodeMemory;
use mnesio_core::types::MemoryRef;
use mnesio_core::{Query, Retriever, Scope};

use crate::codeeval::CodeQuery;
use crate::learncurve::{CurveIndex, Scored};

/// An indexed repository the curve can retrieve against.
pub struct RepoCurveIndex {
    memory: CodeMemory,
    scope: Scope,
    /// Depth retrieval is asked for. Fixed across every arm and every
    /// generation: a curve that also moved `k` would not be measuring learning.
    k: usize,
    /// Index root relative to the git root, e.g. `crates/core`.
    ///
    /// Two path frames meet here and they are not the same one.
    /// [`crate::gitsuite`] produces gold paths relative to the **git root**,
    /// because that is the only frame `git log -L` accepts. [`CodeMemory`]
    /// stores paths relative to the **directory it was told to index**. On a
    /// whole-repo index those coincide and nothing goes wrong; on a
    /// subdirectory they differ by exactly this prefix.
    ///
    /// Getting it wrong is silent: every gold comparison simply fails, the
    /// curve reports 0% before and 0% after, and that reads as "the mechanism
    /// learned nothing" rather than "the measurement is broken". It did read
    /// that way on the first run against ripgrep — the tell was that codeeval
    /// scores 54% on the same index.
    prefix: String,
}

impl RepoCurveIndex {
    /// `prefix` is the indexed directory relative to the git root, or empty
    /// when they are the same. See [`RepoCurveIndex::prefix`].
    pub fn new(memory: CodeMemory, scope: Scope, k: usize, prefix: impl Into<String>) -> Self {
        Self {
            memory,
            scope,
            k,
            prefix: prefix.into(),
        }
    }

    /// A symbol's path in the same frame the gold set uses.
    fn gold_frame_path(&self, path: &str) -> String {
        if self.prefix.is_empty() {
            path.to_string()
        } else {
            format!("{}/{}", self.prefix, path)
        }
    }
}

impl CurveIndex for RepoCurveIndex {
    async fn run(&self, q: &CodeQuery, suppressed: &HashSet<MemoryRef>) -> Result<Scored> {
        let Some(retriever) = self.memory.retriever() else {
            return Ok(Scored {
                hit: false,
                symbols: Vec::new(),
            });
        };
        let hits = retriever
            .search(&Query {
                text: q.question.clone(),
                scope: self.scope.clone(),
                k: self.k,
                time_filter: None,
            })
            .await?;

        // Post-filter, matching what a committed `RetrievalRule` does. The
        // vacated slots are NOT refilled — see the module docs on why that
        // would conflate removing a bad result with showing an extra one.
        let kept: Vec<MemoryRef> = hits
            .iter()
            .map(|h| h.memory)
            .filter(|m| !suppressed.contains(m))
            .collect();

        let hit = kept.iter().any(|m| {
            self.memory.symbol(*m).is_some_and(|(path, name)| {
                let path = self.gold_frame_path(path);
                q.gold.iter().any(|g| g.matches(&path, name))
            })
        });

        Ok(Scored { hit, symbols: kept })
    }
}

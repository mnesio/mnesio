//! Phase 18A: an index that survives a restart.
//!
//! ## The problem this removes
//!
//! Indexing a repository costs one embedding per symbol, and embedding is the
//! only genuinely slow step — a large repository is minutes of model calls.
//! Without persistence every process restart pays that again, so an editor
//! reconnecting to a fresh server stalls before it can answer anything. A
//! competitor that persists pays it once, ever.
//!
//! ## What is cached, and what is deliberately not
//!
//! **Cached:** the embedding vectors, keyed by a hash of the symbol text they
//! were computed from, plus the model id and dimension they came from.
//!
//! **Not cached:** the parse, the HNSW graph, the BM25 index. Those are
//! rebuilt on load. That is not laziness — it is Hard Rule #4. A materialized
//! view must be derivable from the log, and a view read back from a snapshot
//! is a second source of truth that can silently disagree with it. Parsing and
//! view construction are also fast relative to embedding; on llama-index-core
//! the parse is 12.7 s against minutes of model calls.
//!
//! ## Why the key is content, not path
//!
//! A vector is valid for the *text* it was computed from, so that is the key.
//! Renaming a file, moving a function between files, or reformatting around it
//! all preserve the cache. A path-keyed cache would throw away work on every
//! refactor, which is exactly when an agent is asking the most questions.
//!
//! ## Correctness before speed
//!
//! A cache that returns a vector from a *different model* is worse than no
//! cache: retrieval degrades silently and nothing in the output says why. So
//! the model id and dimension are recorded in the file and checked on load,
//! and a mismatch discards the cache rather than mixing vector spaces.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use mnesio_core::MnesioError;

/// Format version. Bumped when the on-disk shape changes; an older or newer
/// file is discarded rather than misread.
const FORMAT: u32 = 1;

/// Embedding vectors for one repository, as written to disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCache {
    format: u32,
    /// The embedder these vectors came from. Mixing vector spaces degrades
    /// retrieval silently, so this is checked before any vector is reused.
    model_id: String,
    dim: usize,
    /// `content_hash -> vector`.
    vectors: HashMap<u64, Vec<f32>>,
}

impl EmbeddingCache {
    pub fn new(model_id: impl Into<String>, dim: usize) -> Self {
        Self {
            format: FORMAT,
            model_id: model_id.into(),
            dim,
            vectors: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
    pub fn get(&self, key: u64) -> Option<&Vec<f32>> {
        self.vectors.get(&key)
    }
    pub fn insert(&mut self, key: u64, vector: Vec<f32>) {
        self.vectors.insert(key, vector);
    }

    /// Replace the contents wholesale, dropping vectors for code that no
    /// longer exists so a long-lived cache cannot grow without bound.
    pub fn replace(&mut self, vectors: HashMap<u64, Vec<f32>>) {
        self.vectors = vectors;
    }

    pub fn vectors(&self) -> &HashMap<u64, Vec<f32>> {
        &self.vectors
    }

    /// Is this cache usable by the given embedder?
    ///
    /// Both fields are checked. Two models can share a dimension and produce
    /// entirely different spaces, so dimension alone would let a wrong-model
    /// cache through.
    fn usable_by(&self, model_id: &str, dim: usize) -> bool {
        self.format == FORMAT && self.model_id == model_id && self.dim == dim
    }
}

/// Where a repository's cache lives.
///
/// Under the user's cache directory rather than inside the repository: an
/// index is derived data about a checkout, not part of it, and writing into
/// someone's working tree is a surprise they did not ask for.
pub fn cache_path(repo: &Path) -> PathBuf {
    cache_path_in(&cache_base(), repo)
}

/// Root of the cache directory for this machine.
///
/// Public so sibling modules that keep their own per-repo state — the outcome
/// journal, for one — land beside the embedding cache instead of inventing a
/// second location a user would have to learn about separately.
pub fn cache_base() -> PathBuf {
    std::env::var_os("MNESIO_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("mnesio-code")
}

/// Cache path under an explicit base.
///
/// Split out so tests can point at their own directory by argument instead of
/// by mutating a process-global environment variable — which races the moment
/// two tests run in parallel, and produces failures that look like cache bugs.
pub fn cache_path_in(base: &Path, repo: &Path) -> PathBuf {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    repo.canonicalize()
        .unwrap_or_else(|_| repo.to_path_buf())
        .hash(&mut h);
    // The repo's name is in the filename purely so a human inspecting the
    // cache directory can tell what is there; the hash disambiguates.
    let label = repo
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "repo".into());
    base.join(format!("{label}-{:016x}.json", h.finish()))
}

/// Load a cache, or `None` if there is nothing usable there.
///
/// Every failure — missing, unreadable, corrupt, wrong model — returns `None`
/// rather than an error. A cache is an optimisation; a broken one must cost a
/// re-index, never a failed startup. The reason is logged so a permanently
/// cold cache is diagnosable instead of merely slow.
pub fn load(repo: &Path, model_id: &str, dim: usize) -> Option<EmbeddingCache> {
    load_in(&cache_base(), repo, model_id, dim)
}

/// [`load`] under an explicit cache root.
pub fn load_in(base: &Path, repo: &Path, model_id: &str, dim: usize) -> Option<EmbeddingCache> {
    let path = cache_path_in(base, repo);
    let bytes = std::fs::read(&path).ok()?;
    match serde_json::from_slice::<EmbeddingCache>(&bytes) {
        Ok(c) if c.usable_by(model_id, dim) => {
            tracing::debug!(vectors = c.len(), path = %path.display(), "code cache hit");
            Some(c)
        }
        Ok(c) => {
            tracing::info!(
                cached_model = %c.model_id,
                cached_dim = c.dim,
                wanted_model = %model_id,
                wanted_dim = dim,
                "discarding code cache: different embedder"
            );
            None
        }
        Err(e) => {
            tracing::warn!(error = %e, path = %path.display(), "code cache unreadable; re-indexing");
            None
        }
    }
}

/// Write a cache, replacing any previous one.
///
/// Written to a temporary file and renamed, so a crash mid-write leaves the
/// previous cache intact rather than a truncated file that fails to parse on
/// the next start.
pub fn store(repo: &Path, cache: &EmbeddingCache) -> Result<(), MnesioError> {
    store_in(&cache_base(), repo, cache)
}

/// [`store`] under an explicit cache root.
pub fn store_in(base: &Path, repo: &Path, cache: &EmbeddingCache) -> Result<(), MnesioError> {
    let path = cache_path_in(base, repo);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| MnesioError::Index(format!("cache dir {}: {e}", dir.display())))?;
    }
    let tmp = path.with_extension("tmp");
    let bytes = serde_json::to_vec(cache)
        .map_err(|e| MnesioError::Index(format!("serialising code cache: {e}")))?;
    std::fs::write(&tmp, &bytes)
        .map_err(|e| MnesioError::Index(format!("writing {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, &path)
        .map_err(|e| MnesioError::Index(format!("renaming into {}: {e}", path.display())))?;
    tracing::debug!(vectors = cache.len(), path = %path.display(), "code cache written");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::types::new_id;

    /// A cache root of this test's own, so nothing is shared with a test
    /// running beside it.
    struct Sandbox(PathBuf);
    impl Sandbox {
        fn new() -> Self {
            let d = std::env::temp_dir().join(format!("mnesio-cache-{}", new_id()));
            std::fs::create_dir_all(&d).unwrap();
            Self(d)
        }
        /// A distinct repository path inside the sandbox.
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

    fn sample() -> EmbeddingCache {
        let mut c = EmbeddingCache::new("test-model", 3);
        c.insert(42, vec![0.1, 0.2, 0.3]);
        c
    }

    #[test]
    fn a_cache_round_trips() {
        let s = Sandbox::new();
        let repo = s.repo("r");
        store_in(&s.0, &repo, &sample()).unwrap();
        let back = load_in(&s.0, &repo, "test-model", 3).expect("should load");
        assert_eq!(back.get(42), Some(&vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn a_different_model_is_refused() {
        // Mixing vector spaces degrades retrieval silently — no error, no
        // wrong answer you can point at, just quietly worse results. Refusing
        // the cache costs a re-index and is unambiguously the right trade.
        let s = Sandbox::new();
        let repo = s.repo("r");
        store_in(&s.0, &repo, &sample()).unwrap();
        assert!(load_in(&s.0, &repo, "a-different-model", 3).is_none());
    }

    #[test]
    fn a_matching_dimension_is_not_enough() {
        // Two models can share a dimension and share nothing else, so a
        // dimension check alone would let a wrong-model cache through.
        let s = Sandbox::new();
        let repo = s.repo("r");
        let mut c = EmbeddingCache::new("model-a", 384);
        c.insert(1, vec![0.0; 384]);
        store_in(&s.0, &repo, &c).unwrap();
        assert!(load_in(&s.0, &repo, "model-b", 384).is_none());
    }

    #[test]
    fn a_corrupt_cache_is_a_cold_start_not_a_failure() {
        let s = Sandbox::new();
        let repo = s.repo("r");
        let path = cache_path_in(&s.0, &repo);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{ this is not json").unwrap();
        assert!(load_in(&s.0, &repo, "test-model", 3).is_none());
    }

    #[test]
    fn a_missing_cache_is_not_an_error() {
        let s = Sandbox::new();
        assert!(load_in(&s.0, Path::new("/no/such/repo"), "m", 3).is_none());
    }

    #[test]
    fn two_repositories_do_not_share_a_cache_file() {
        let s = Sandbox::new();
        assert_ne!(
            cache_path_in(&s.0, &s.repo("alpha")),
            cache_path_in(&s.0, &s.repo("beta"))
        );
    }

    #[test]
    fn a_crash_mid_write_leaves_the_previous_cache_intact() {
        // The rename is what guarantees this: a half-written temp file is
        // never the file `load` reads.
        let s = Sandbox::new();
        let repo = s.repo("r");
        store_in(&s.0, &repo, &sample()).unwrap();
        std::fs::write(cache_path_in(&s.0, &repo).with_extension("tmp"), b"partial").unwrap();
        assert!(load_in(&s.0, &repo, "test-model", 3).is_some());
    }

    #[test]
    fn replace_drops_vectors_for_code_that_is_gone() {
        let mut c = sample();
        assert_eq!(c.len(), 1);
        c.replace(HashMap::new());
        assert!(c.is_empty(), "stale vectors must not accumulate");
    }
}

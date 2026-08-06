//! Core identifiers, scoping, and the bi-temporal time model.
//!
//! See architecture report §5 (Data Model). Every memory and every edge
//! carries a [`BiTemporal`] stamp: we never overwrite history, we invalidate
//! and create a new version. This is what makes evolution + audit coexist.

use serde::{Deserialize, Serialize};
use std::sync::{Mutex, OnceLock};
use time::OffsetDateTime;
use ulid::{Generator, Ulid};

/// Monotonic, sortable identifier used for all log entries and entities.
pub type Id = Ulid;

/// Process-global ULID generator. Held behind a `Mutex` so that consecutive
/// calls — even from different threads in the same millisecond — produce
/// strictly increasing ids. `Ulid::new()` alone does not give this guarantee:
/// inside a single millisecond it draws fresh random bits each call, so two
/// rapid calls can land out of order. The hard rule "keys are ULIDs
/// (lexicographically time-ordered)" depends on this generator.
static ID_GENERATOR: OnceLock<Mutex<Generator>> = OnceLock::new();

/// Generates a fresh, strictly-monotonic time-ordered id.
///
/// Panics only if the random bits overflow inside a single millisecond, which
/// requires more than 2^80 calls in that window — astronomically unreachable.
pub fn new_id() -> Id {
    ID_GENERATOR
        .get_or_init(|| Mutex::new(Generator::new()))
        .lock()
        .expect("ulid generator mutex poisoned")
        .generate()
        .expect("ulid random bits overflowed within a single millisecond")
}

/// Bi-temporal stamp: validity time (when the fact was true in the world)
/// and transaction time (when the system learned/forgot it).
///
/// `valid_to` / `tx_to` are `None` while the fact is current.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BiTemporal {
    pub valid_from: OffsetDateTime,
    pub valid_to: Option<OffsetDateTime>,
    pub tx_from: OffsetDateTime,
    pub tx_to: Option<OffsetDateTime>,
}

impl BiTemporal {
    /// A stamp that is valid-now and was just recorded.
    pub fn now() -> Self {
        let t = OffsetDateTime::now_utc();
        Self {
            valid_from: t,
            valid_to: None,
            tx_from: t,
            tx_to: None,
        }
    }

    /// True if the fact is valid at `at` and not yet superseded in the system.
    pub fn is_live(&self, at: OffsetDateTime) -> bool {
        let valid = self.valid_from <= at && self.valid_to.map_or(true, |e| at < e);
        let current = self.tx_to.is_none();
        valid && current
    }
}

/// Tenancy + ownership boundary. Procedural learning and memory evolution
/// must never cross a scope boundary without an explicit aggregation step
/// (see report §10, "Privacy").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Scope {
    pub tenant: String,
    pub user: Option<String>,
    /// `None` => applies to the whole user/tenant; `Some` => single session.
    pub session: Option<String>,
}

impl Scope {
    pub fn global(tenant: impl Into<String>) -> Self {
        Self {
            tenant: tenant.into(),
            user: None,
            session: None,
        }
    }

    /// True if `self` is allowed to read/learn from data stamped `other`.
    /// Global scope contains user scope contains session scope.
    pub fn contains(&self, other: &Scope) -> bool {
        if self.tenant != other.tenant {
            return false;
        }
        match (&self.user, &other.user) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(a), Some(b)) => {
                a == b
                    && match (&self.session, &other.session) {
                        (None, _) => true,
                        (Some(_), None) => false,
                        (Some(x), Some(y)) => x == y,
                    }
            }
        }
    }
}

/// Typed references — newtypes so the compiler stops us mixing id kinds.
macro_rules! id_ref {
    ($name:ident) => {
        // `Ord` is derived because [`Id`] is a ULID, whose ordering *is* its
        // point: ids sort by creation time. Without it every caller that needs
        // a deterministic iteration order — graph colouring, replay, any
        // tie-break — has to reach through to `.0`, and one that forgets gets
        // hash order instead, which differs between runs and turns a stable
        // output into an intermittently-changing one.
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        pub struct $name(pub Id);
        impl From<Id> for $name {
            fn from(id: Id) -> Self {
                Self(id)
            }
        }
    };
}

id_ref!(MemoryRef);
id_ref!(SourceRef);
id_ref!(EpisodeRef);
id_ref!(OutcomeRef);
id_ref!(ArtifactRef);
id_ref!(TrajectoryRef);
id_ref!(ProposalId);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_id_is_strictly_monotonic_in_tight_loop() {
        // 10k ids in a tight loop will straddle many millisecond boundaries
        // and also pack many ids into the same millisecond — the case
        // `Ulid::new()` does not handle correctly.
        let n = 10_000;
        let mut prev = new_id();
        for _ in 0..n {
            let next = new_id();
            assert!(
                next > prev,
                "new_id must be strictly increasing; got {prev} then {next}"
            );
            prev = next;
        }
    }

    #[test]
    fn new_id_is_monotonic_across_threads() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let threads = 8;
        let per_thread = 1_000;
        let barrier = Arc::new(Barrier::new(threads));
        let mut handles = Vec::new();
        for _ in 0..threads {
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                let mut ids = Vec::with_capacity(per_thread);
                for _ in 0..per_thread {
                    ids.push(new_id());
                }
                ids
            }));
        }
        let mut all_ids: Vec<Id> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        let total = all_ids.len();
        all_ids.sort();
        all_ids.dedup();
        assert_eq!(
            all_ids.len(),
            total,
            "ids generated across threads must be unique"
        );
    }

    #[test]
    fn scope_containment() {
        let g = Scope::global("acme");
        let u = Scope {
            tenant: "acme".into(),
            user: Some("aniket".into()),
            session: None,
        };
        let s = Scope {
            tenant: "acme".into(),
            user: Some("aniket".into()),
            session: Some("sess-1".into()),
        };
        assert!(g.contains(&u));
        assert!(g.contains(&s));
        assert!(u.contains(&s));
        assert!(!s.contains(&u)); // session cannot read user-wide
        assert!(!u.contains(&g)); // user cannot read global
    }
}

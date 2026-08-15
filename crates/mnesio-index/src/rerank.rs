//! Content-aware lexical reranker (Phase 16 — retrieval parity).
//!
//! The hybrid retriever fuses vector + BM25 by *rank*, which is robust but
//! blind to the actual text of a candidate. On the two LoCoMo categories the
//! 2026 leaders still beat a flat-hybrid baseline — **multi-hop** (the answer
//! stitches several entities together) and **temporal** (the answer hinges on
//! a date) — rank fusion under-weights exactly the signals that decide those
//! questions: how much of the query a candidate *covers*, and whether it
//! mentions the *same date* the query asks about.
//!
//! [`LexicalReranker`] is a lightweight, dependency-free cross-encoder-style
//! stage that re-scores the already-fetched candidate list on its full text
//! (resolved through the [`ContentProvider`] seam) using three features:
//!
//! - **coverage** — fraction of distinct query terms present in the content.
//!   Rewards a memory that ties together more of a multi-hop query's entities.
//! - **temporal** — fraction of the query's date tokens (years, months,
//!   quarters, weekdays) that the content also mentions. Zero, and therefore
//!   inert, on queries with no date token, so non-temporal recall is untouched.
//! - **phrase** — adjacent-bigram overlap. Rewards precise phrasing over a bag
//!   of individually-common words.
//! - **update** — for "what's current?" queries (the *knowledge-update*
//!   category: "current", "now", "latest"), prefers memories phrased as an
//!   update ("later updated", "now", "moved earlier", "switched"). When two
//!   contradictory memories are near-identical to a semantic embedder, this is
//!   the signal that surfaces the *superseding* one over the stale original.
//!   Zero, and therefore inert, on queries with no such cue.
//!
//! **Why additive, not a replacement score?** The feature is folded in as a
//! *bonus* on the normalised base score (`final = base/base_max + boost *
//! feature`), mirroring the recency/proximity bonus pattern in
//! [`crate::HybridRetriever`]. A bounded boost can promote a strongly
//! content-relevant hit past a near-tie, but cannot let lexical noise evict a
//! hit whose retrieval signal is much stronger — that discipline is what keeps
//! `recall@k` from regressing (the Phase 16 "done when" guard). A candidate
//! whose content can't be resolved simply scores `feature = 0` and keeps its
//! base rank — the stage degrades gracefully to identity.

use crate::ContentProvider;
use mnesio_core::{Hit, MnesioError};
use std::sync::Arc;

/// Weight of the query-coverage feature within the blended `feature` score.
const W_COVERAGE: f32 = 0.5;
/// Weight of the temporal (date-token) feature.
const W_TEMPORAL: f32 = 0.3;
/// Weight of the phrase (bigram-overlap) feature.
const W_PHRASE: f32 = 0.2;
/// Weight of the update-recency feature. Deliberately *additive on top of* the
/// three relevance features (which sum to 1.0) rather than taking weight from
/// them: redistributing would perturb the relevance ranking of every ordinary
/// query to serve the knowledge-update case. Measured: rebalancing regressed
/// LongMemEval-mini `preference` 100%→67%, while adding on top leaves every
/// other category untouched.
const W_UPDATE: f32 = 0.3;
/// Default bonus scale applied to `feature` on top of the normalised base
/// score. Chosen so a perfect-feature hit can overtake a near-tie but not a
/// hit with a decisively higher retrieval signal (protects `recall@k`).
const DEFAULT_BOOST: f32 = 0.5;

/// Bonus scale for **code** retrieval. Six times the prose default.
///
/// Measured, not guessed: `mnesio-bench codeeval` over llama-index-core with
/// 400 git-derived tasks (query = a real commit subject, gold = the symbols
/// that commit touched), recall@20, one index, boost swept inside the run —
///
/// | boost | 0.0 | 0.5 | 1.5 | 3.0 | 6.0 | 12.0 | 48.0 |
/// |---|---|---|---|---|---|---|---|
/// | recall | 52% | 56% | 61% | **62%** | 62% | 61% | 61% |
///
/// The prose default of 0.5 exists to stop content relevance evicting a
/// decisively stronger retrieval signal, a guard added after the Phase-16
/// reranker regressed prose recall. On code that guard is the binding
/// constraint: an identifier-coverage match is a far more reliable signal than
/// the temporal and update cues prose leans on, so the bonus can be much
/// larger before it does harm.
///
/// The sweep was deliberately run past any sane value. It turns *over* rather
/// than climbing, which is the reassuring shape — but only just: 48.0 still
/// scores 61%. Read honestly, that says the retrieval base score is worth
/// about 1pp here and the lexical features are doing nearly all the ranking.
/// Hybrid retrieval is functioning mostly as a candidate generator for code.
pub const CODE_BOOST: f32 = 3.0;

/// The code boost is deliberately the larger of the two. Enforced at compile
/// time so the relationship cannot be inverted by an edit to either constant.
const _: () = assert!(CODE_BOOST > DEFAULT_BOOST);

/// Content-aware reranker. Wire it with [`crate::HybridRetriever::with_reranker`].
pub struct LexicalReranker {
    content: Arc<dyn ContentProvider>,
    boost: f32,
}

impl LexicalReranker {
    /// Build a reranker over a content source (production: [`crate::Bm25View`]).
    pub fn new(content: Arc<dyn ContentProvider>) -> Self {
        Self {
            content,
            boost: DEFAULT_BOOST,
        }
    }

    /// Reranker tuned for **code** retrieval — see [`CODE_BOOST`] for the
    /// measurement behind the value.
    ///
    /// A separate constructor rather than a changed default: the prose default
    /// is load-bearing for LOCOMO/LongMemEval and must not move.
    pub fn for_code(content: Arc<dyn ContentProvider>) -> Self {
        Self {
            content,
            boost: CODE_BOOST,
        }
    }

    /// Override the bonus scale. Larger promotes content relevance more
    /// aggressively; `0.0` makes the stage a no-op reorder (identity).
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Blended content feature for one candidate's already-tokenized text.
    /// Ranges over `[0, 1 + W_UPDATE]` — the three relevance features sum to
    /// at most 1.0 and the update bonus rides on top. `weights` carries the
    /// candidate-set discriminative weight per query term (see
    /// [`term_weights`]).
    fn feature(
        &self,
        query_tokens: &[String],
        doc_tokens: &[String],
        weights: &[(String, f32)],
    ) -> f32 {
        if doc_tokens.is_empty() || query_tokens.is_empty() {
            return 0.0;
        }
        let coverage = coverage(weights, doc_tokens);
        let temporal = temporal(query_tokens, doc_tokens);
        let phrase = phrase(query_tokens, doc_tokens);
        let update = update(query_tokens, doc_tokens);
        W_COVERAGE * coverage + W_TEMPORAL * temporal + W_PHRASE * phrase + W_UPDATE * update
    }
}

#[async_trait::async_trait]
impl crate::Reranker for LexicalReranker {
    async fn rerank(&self, query: &str, mut hits: Vec<Hit>) -> Result<Vec<Hit>, MnesioError> {
        if hits.len() < 2 {
            return Ok(hits);
        }
        let query_tokens = tokenize(query);

        // Pass 1 — resolve every candidate's text once. `None` = unresolved
        // content, which scores 0 and keeps its base rank.
        let docs: Vec<Option<Vec<String>>> = hits
            .iter()
            .map(|h| self.content.content(h.memory).map(|t| tokenize(&t)))
            .collect();

        // Pass 2 — weight each query term by how well it *discriminates*
        // within this candidate set. Without this a term that appears in every
        // candidate (e.g. "user" across a personal-memory corpus) contributes
        // the same bonus to all of them: no signal, but enough to perturb a
        // near-tied ranking and evict a correct semantic top-1.
        let weights = term_weights(&query_tokens, &docs);

        // Normalise base scores by the max so the top retrieval hit sits at
        // 1.0 and the bonus is comparable across queries. `max <= 0` (all
        // signals muted, e.g. mock embedder ties) → feature decides alone.
        let max_base = hits
            .iter()
            .map(|h| h.score)
            .fold(0.0_f32, f32::max)
            .max(f32::MIN_POSITIVE);

        // Pass 3 — blend.
        for (h, doc) in hits.iter_mut().zip(docs.iter()) {
            let feat = match doc {
                Some(tokens) => self.feature(&query_tokens, tokens, &weights),
                None => 0.0, // graceful fallback: keep base rank
            };
            let norm_base = h.score / max_base;
            h.score = norm_base + self.boost * feat;
            h.breakdown.push(("rerank".to_string(), feat));
        }

        // Descending blended score, then memory id. "Ties keep their fused
        // order" was true and not sufficient: it inherits whatever order the
        // fused list had, so any upstream non-determinism survives this stage
        // intact. Breaking on id makes the reranked order total on its own
        // terms rather than conditional on the caller's.
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.memory.0.cmp(&b.memory.0))
        });
        Ok(hits)
    }
}

/// English stop words dropped before feature scoring — the same short,
/// closed-class list used elsewhere so "the"/"of" can't inflate coverage.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "by",
    "from", "as", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these",
    "those", "do", "did", "does", "what", "when", "which", "who", "how", "why", "into", "over",
];

/// Lowercase, split on non-alphanumerics, drop stop words + 1-char tokens
/// (numbers are kept — they carry temporal signal).
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .filter(|s| s.len() > 1 && !STOP_WORDS.contains(&s.as_str()))
        .collect()
}

/// Discriminative weight per distinct query term within *this* candidate set:
/// `1 - df/n`, where `df` is how many candidates contain the term. A term in
/// every candidate scores 0 (it separates nothing); a term in one scores near
/// 1. This is the candidate-set analogue of IDF, and it is what stops an
///    uninformative match (e.g. "user" across a personal-memory corpus) from
///    adding a uniform bonus that perturbs an otherwise-correct ranking.
fn term_weights(query_tokens: &[String], docs: &[Option<Vec<String>>]) -> Vec<(String, f32)> {
    let mut distinct: Vec<String> = query_tokens.to_vec();
    distinct.sort();
    distinct.dedup();
    let n = docs.iter().filter(|d| d.is_some()).count();
    if n == 0 {
        return distinct.into_iter().map(|t| (t, 1.0)).collect();
    }
    distinct
        .into_iter()
        .map(|term| {
            let df = docs
                .iter()
                .flatten()
                .filter(|toks| toks.contains(&term))
                .count();
            (term, 1.0 - (df as f32 / n as f32))
        })
        .collect()
}

/// Share of the query's *discriminative* weight the doc covers. Multi-hop
/// questions name several entities; a memory covering more of them scores
/// higher — but only terms that actually separate candidates count. Returns
/// `0.0` when no query term discriminates (the feature goes inert rather than
/// nudging a ranking it has no information about).
fn coverage(weights: &[(String, f32)], doc_tokens: &[String]) -> f32 {
    let total: f32 = weights.iter().map(|(_, w)| *w).sum();
    if total <= f32::EPSILON {
        return 0.0;
    }
    let hit: f32 = weights
        .iter()
        .filter(|(term, _)| doc_tokens.iter().any(|d| d == term))
        .map(|(_, w)| *w)
        .sum();
    hit / total
}

/// Fraction of the query's date tokens the doc also mentions. Returns `0.0`
/// when the query has no date token, so non-temporal queries are unaffected.
fn temporal(query_tokens: &[String], doc_tokens: &[String]) -> f32 {
    let q_dates: Vec<&String> = query_tokens.iter().filter(|t| is_date_token(t)).collect();
    if q_dates.is_empty() {
        return 0.0;
    }
    let matched = q_dates
        .iter()
        .filter(|q| doc_tokens.iter().any(|d| d == **q))
        .count();
    matched as f32 / q_dates.len() as f32
}

/// A token that carries a temporal signal: a 4-digit year, a month name (full
/// or 3-letter), a quarter (`q1`..`q4`), or a weekday.
fn is_date_token(t: &str) -> bool {
    const MONTHS: &[&str] = &[
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    ];
    const WEEKDAYS: &[&str] = &[
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ];
    // 4-digit year in a plausible range.
    if t.len() == 4 && t.bytes().all(|b| b.is_ascii_digit()) {
        if let Ok(y) = t.parse::<u32>() {
            if (1900..=2100).contains(&y) {
                return true;
            }
        }
    }
    // Quarter tokens: q1..q4.
    if let Some(rest) = t.strip_prefix('q') {
        if matches!(rest, "1" | "2" | "3" | "4") {
            return true;
        }
    }
    MONTHS.contains(&t) || WEEKDAYS.contains(&t)
}

/// Adjacent-bigram overlap: fraction of the query's word-pairs that appear as
/// a word-pair in the doc. Rewards precise phrasing over a bag of common words.
fn phrase(query_tokens: &[String], doc_tokens: &[String]) -> f32 {
    if query_tokens.len() < 2 {
        return 0.0;
    }
    let doc_bigrams: Vec<(&String, &String)> =
        doc_tokens.windows(2).map(|w| (&w[0], &w[1])).collect();
    let q_bigrams: Vec<(&String, &String)> =
        query_tokens.windows(2).map(|w| (&w[0], &w[1])).collect();
    let matched = q_bigrams
        .iter()
        .filter(|qb| doc_bigrams.iter().any(|db| db == *qb))
        .count();
    matched as f32 / q_bigrams.len() as f32
}

/// `1.0` if the query asks for the *current* state and the doc reads like an
/// update (so its answer supersedes an earlier one); else `0.0`. Inert unless
/// the query carries a "what's-now" cue, so ordinary queries are unaffected.
fn update(query_tokens: &[String], doc_tokens: &[String]) -> f32 {
    if !query_seeks_update(query_tokens) {
        return 0.0;
    }
    if doc_tokens
        .iter()
        .any(|t| UPDATE_MARKERS.contains(&t.as_str()))
    {
        1.0
    } else {
        0.0
    }
}

/// Cue words that mean "give me the latest value, not a historical one".
const UPDATE_CUES: &[&str] = &[
    "current",
    "currently",
    "now",
    "latest",
    "recent",
    "recently",
    "nowadays",
    "still",
    "today",
    "updated",
    "newest",
];

/// Words that mark a memory as a later revision of an earlier fact.
const UPDATE_MARKERS: &[&str] = &[
    "now",
    "updated",
    "later",
    "changed",
    "moved",
    "switched",
    "currently",
    "recently",
    "revised",
    "newer",
    "instead",
    "renamed",
    "became",
];

/// Does the query ask for the up-to-date value?
fn query_seeks_update(query_tokens: &[String]) -> bool {
    query_tokens
        .iter()
        .any(|t| UPDATE_CUES.contains(&t.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Reranker;
    use mnesio_core::types::{new_id, MemoryRef};
    use std::collections::HashMap;

    /// In-memory `ContentProvider` for tests.
    struct MapContent(HashMap<MemoryRef, String>);
    impl ContentProvider for MapContent {
        fn content(&self, memory: MemoryRef) -> Option<String> {
            self.0.get(&memory).cloned()
        }
    }

    fn hit(m: MemoryRef, score: f32) -> Hit {
        Hit {
            memory: m,
            score,
            breakdown: vec![],
        }
    }

    #[tokio::test]
    async fn multi_hop_coverage_promotes_fuller_answer() {
        // Two candidates: `low` narrowly outranks `high` on the base signal,
        // but `high` covers both query entities. Reranking should flip them.
        let low = MemoryRef(new_id());
        let high = MemoryRef(new_id());
        let mut map = HashMap::new();
        map.insert(low, "alice joined the company".to_string());
        map.insert(high, "alice manages the acme account".to_string());
        let rr = LexicalReranker::new(Arc::new(MapContent(map)));

        let hits = rr
            .rerank(
                "who manages alice's acme account",
                vec![hit(low, 0.030), hit(high, 0.028)],
            )
            .await
            .unwrap();
        assert_eq!(
            hits[0].memory, high,
            "the candidate covering more query terms should be promoted"
        );
        // The rerank feature is surfaced in the breakdown.
        assert!(hits[0].breakdown.iter().any(|(n, _)| n == "rerank"));
    }

    #[tokio::test]
    async fn temporal_date_match_promotes_right_year() {
        // Same lexical content bar the year; the query asks about 2021.
        let y2020 = MemoryRef(new_id());
        let y2021 = MemoryRef(new_id());
        let mut map = HashMap::new();
        map.insert(y2020, "revenue grew sharply in 2020".to_string());
        map.insert(y2021, "revenue grew sharply in 2021".to_string());
        let rr = LexicalReranker::new(Arc::new(MapContent(map)));

        // y2020 leads on the base signal; the temporal feature must flip it.
        let hits = rr
            .rerank(
                "how much did revenue grow in 2021",
                vec![hit(y2020, 0.030), hit(y2021, 0.029)],
            )
            .await
            .unwrap();
        assert_eq!(
            hits[0].memory, y2021,
            "the candidate matching the query's year should be promoted"
        );
    }

    #[tokio::test]
    async fn update_cue_promotes_superseding_memory() {
        // The classic knowledge-update case: a stale fact and its update. The
        // query asks for the *current* value; the stale memory has more literal
        // overlap ("programming language"), but the update cue must surface the
        // superseding one.
        let stale = MemoryRef(new_id());
        let fresh = MemoryRef(new_id());
        let noise: Vec<MemoryRef> = (0..3).map(|_| MemoryRef(new_id())).collect();
        let mut map = HashMap::new();
        map.insert(
            stale,
            "Earlier the user said their favourite programming language was Python".to_string(),
        );
        map.insert(
            fresh,
            "The user later updated that their favourite language is now Rust".to_string(),
        );
        // A realistic candidate set: the candidate-set IDF weighting needs
        // more than two documents to tell a discriminative term from a
        // corpus-wide one ("user" here).
        map.insert(
            noise[0],
            "The user is learning to play the piano".to_string(),
        );
        map.insert(noise[1], "The user is allergic to penicillin".to_string());
        map.insert(
            noise[2],
            "The user's home timezone is Pacific Time".to_string(),
        );
        let rr = LexicalReranker::new(Arc::new(MapContent(map)));
        // The two are near-tied on the base signal — which is exactly what a
        // semantic embedder produces for a fact and its update, and the regime
        // this reranker exists to disambiguate. (A *decisive* base lead is
        // deliberately not overturned; see the recall-guard test below.)
        let mut candidates = vec![hit(stale, 0.0300), hit(fresh, 0.0298)];
        candidates.extend(noise.iter().map(|m| hit(*m, 0.020)));
        let hits = rr
            .rerank(
                "what is the user's current favourite programming language",
                candidates,
            )
            .await
            .unwrap();
        assert_eq!(
            hits[0].memory, fresh,
            "the superseding (updated) memory should be promoted for a 'current' query"
        );
    }

    #[tokio::test]
    async fn update_feature_inert_without_cue() {
        // No "current/now/latest" cue → update feature contributes nothing, so
        // an update-marker word in the doc can't distort ordinary ranking.
        assert_eq!(
            update(
                &tokenize("who is the manager"),
                &tokenize("she moved teams")
            ),
            0.0
        );
        assert!(
            update(
                &tokenize("who is the current manager"),
                &tokenize("she moved teams")
            ) > 0.0
        );
    }

    #[tokio::test]
    async fn non_temporal_query_leaves_date_feature_inert() {
        // No date token in the query → temporal contributes nothing, so a
        // pure-coverage tie keeps the base order (stable).
        let a = MemoryRef(new_id());
        let b = MemoryRef(new_id());
        let mut map = HashMap::new();
        map.insert(a, "the sky is blue".to_string());
        map.insert(b, "the grass is green".to_string());
        let rr = LexicalReranker::new(Arc::new(MapContent(map)));
        let hits = rr
            .rerank("unrelated colours", vec![hit(a, 0.030), hit(b, 0.020)])
            .await
            .unwrap();
        assert_eq!(
            hits[0].memory, a,
            "no content signal → base order preserved"
        );
    }

    #[tokio::test]
    async fn missing_content_falls_back_to_base_rank() {
        // `b` has no content entry → feature 0; `a` outranks on base and keeps
        // the lead. Proves graceful degradation to identity.
        let a = MemoryRef(new_id());
        let b = MemoryRef(new_id());
        let mut map = HashMap::new();
        map.insert(a, "quarterly earnings report".to_string());
        // b intentionally absent
        let rr = LexicalReranker::new(Arc::new(MapContent(map)));
        let hits = rr
            .rerank("quarterly earnings", vec![hit(a, 0.030), hit(b, 0.020)])
            .await
            .unwrap();
        assert_eq!(hits[0].memory, a);
        assert_eq!(hits.len(), 2, "no candidate is dropped");
    }

    #[test]
    fn the_code_boost_is_separate_from_the_prose_default() {
        // The prose default is load-bearing for LOCOMO/LongMemEval — Phase 16
        // measured real regressions when content relevance was allowed to
        // outrank retrieval there. `for_code` must therefore be a *separate*
        // constructor, never a changed default, or a code-retrieval win would
        // silently move the published prose numbers.
        // Exact values, because these are the measured settings — a drift in
        // either is a silent change to published numbers. Their *ordering* is
        // enforced separately at compile time.
        assert_eq!(DEFAULT_BOOST, 0.5, "prose default must not drift");
        assert_eq!(CODE_BOOST, 3.0, "code boost must not drift");
        assert_eq!(
            LexicalReranker::for_code(Arc::new(MapContent(HashMap::new()))).boost,
            CODE_BOOST
        );
    }

    #[tokio::test]
    async fn bounded_boost_protects_a_decisive_base_lead() {
        // `strong` has a decisively higher base score and *no* content match;
        // `weak` matches the query fully. With the default boost the strong
        // retrieval signal must still win — this is the recall@k guard.
        let strong = MemoryRef(new_id());
        let weak = MemoryRef(new_id());
        let mut map = HashMap::new();
        map.insert(strong, "completely unrelated text".to_string());
        map.insert(weak, "alpha beta gamma".to_string());
        let rr = LexicalReranker::new(Arc::new(MapContent(map)));
        // strong base 0.030 vs weak 0.012 → norm 1.0 vs 0.4; weak feature ~1.0
        // adds 0.5 → 0.9 < 1.0. Strong holds.
        let hits = rr
            .rerank(
                "alpha beta gamma",
                vec![hit(strong, 0.030), hit(weak, 0.012)],
            )
            .await
            .unwrap();
        assert_eq!(
            hits[0].memory, strong,
            "a decisive base lead is not overturned by lexical match"
        );
    }

    #[test]
    fn date_token_classifier() {
        assert!(is_date_token("2021"));
        assert!(is_date_token("march"));
        assert!(is_date_token("dec"));
        assert!(is_date_token("q3"));
        assert!(is_date_token("friday"));
        assert!(!is_date_token("revenue"));
        assert!(!is_date_token("21"));
        assert!(!is_date_token("3000")); // out of plausible year range
    }
}

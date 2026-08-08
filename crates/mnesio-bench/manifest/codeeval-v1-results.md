# code retrieval at scale — 9 repositories, 526 queries, 5805 symbols

| repo | files | symbols | queries | symbol | whole-file | gap | rankable | unreachable |
|---|---|---|---|---|---|---|---|---|
| core | 23 | 1252 | 46 | 54% | 67% | −13pp | 17% | 4% |
| src | 19 | 518 | 60 | 42% | 68% | −27pp | 17% | 15% |
| src | 24 | 725 | 60 | 62% | 82% | −20pp | 23% | 7% |
| src ᵗ | 22 | 237 | 60 | 63% | 77% | −13pp | 12% | 12% |
| requests ᵗ | 19 | 320 | 60 | 67% | 83% | −17pp | 17% | 2% |
| flask ᵗ | 24 | 477 | 60 | 60% | 82% | −22pp | 8% | 10% |
| click | 17 | 670 | 60 | 62% | 82% | −20pp | 8% | 5% |
| httpx | 23 | 533 | 60 | 58% | 70% | −12pp | 12% | 10% |
| core | 34 | 1073 | 60 | 57% | 78% | −22pp | 17% | 18% |

ᵗ 3 repositories have fewer than 500 symbols. At that size top-`k` reaches most of the corpus, so they score ~100% regardless of ranking quality. Their rows are shown but they are **excluded from the distribution below** — including them pulls every quantile toward 100% and flatters the result.

## distribution across the 6 discriminating repositories

| metric | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| symbol recall | 42% | 54% | **58%** | 62% | 62% |
| whole-file recall | 67% | 68% | **78%** | 82% | 82% |
| ceiling gap | 12% | 13% | **20%** | 22% | 27% |
| rankable share | 8% | 12% | **17%** | 17% | 23% |
| unreachable share | 4% | 5% | **10%** | 15% | 18% |

_Quartiles, not a mean. Averaging recall across repositories of different size and language produces a number with no referent, and the spread — not the centre — is the finding: the symbol/whole-file trade is repo-dependent._

## skipped (1)

Listed rather than dropped: a suite that silently discards what it cannot handle reports a survivorship-biased result.

- **lib** — no parseable source: no symbols parsed under /tmp/mnesio-corpus/express/lib

## corpus

manifest **codeeval-v1**, 10 repositories, 10 evaluated, 0 refused.
wall clock **252s** against a declared budget of 2400s — within budget.

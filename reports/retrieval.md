# Retrieval experiment (parser: gemini)

Catalog: 80 songs. 16 free-text ranking cases. Every configuration reuses the same parsed profile for each query, so differences come from retrieval alone.

| Configuration | Cases passed | Top-1 relevant | nDCG@5 | P@5 | Forbidden in top 5 | Relevant recall of pool | Ideal recall of pool | Forbidden in pool |
|---|---|---|---|---|---|---|---|---|
| No retrieval (baseline) | 11/16 | 88% | 0.65 | 0.72 | 0 | n/a | n/a | n/a |
| tfidf: retrieval only (no scorer) | 6/16 | 56% | 0.38 | 0.38 | 4 | 20% | 36% | 4 |
| tfidf: retrieve 10, then score | 10/16 | 62% | 0.48 | 0.46 | 0 | 29% | 42% | 6 |
| tfidf: retrieve 20, then score | 8/16 | 69% | 0.54 | 0.61 | 0 | 44% | 48% | 18 |
| tfidf: retrieve 40, then score | 10/16 | 81% | 0.58 | 0.64 | 0 | 63% | 66% | 53 |
| gemini: retrieval only (no scorer) | 15/16 | 94% | 0.70 | 0.69 | 0 | 38% | 62% | 0 |
| gemini: retrieve 10, then score | 13/16 | 88% | 0.73 | 0.79 | 0 | 62% | 87% | 0 |
| gemini: retrieve 20, then score | 14/16 | 94% | 0.69 | 0.78 | 0 | 86% | 91% | 1 |
| gemini: retrieve 40, then score | 12/16 | 94% | 0.68 | 0.75 | 0 | 98% | 100% | 5 |

## Reading it

- The best retrieve-then-score configuration is **gemini: retrieve 10, then score**, which helps by +0.07 nDCG versus the baseline.
- tfidf: retrieval only (no scorer): nDCG 0.38, top-1 relevant 56%. Without the scorer, retrieval alone does not beat the baseline pipeline.
- gemini: retrieval only (no scorer): nDCG 0.70, top-1 relevant 94%. Without the scorer, retrieval alone beats the baseline pipeline.
- With only 16 cases, differences of a few hundredths are noise. The judgments are one person's.

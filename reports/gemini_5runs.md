# Evaluation report (gemini parser, 5 runs)

Catalog: 80 songs. Golden set: 22 ranking cases, 15 parser cases. Each cell shows the mean over 5 runs, with the range in parentheses.

## Summary

| Metric | Mean (min to max) |
|---|---|
| Ranking cases passed (of 22) | 21.2 (21 to 22) |
| Top-1 relevant rate | 99% (95% to 100%) |
| Mean nDCG@5 | 0.84 (0.81 to 0.85) |
| Mean precision@5 | 0.83 (0.79 to 0.84) |
| Mean reciprocal rank | 0.99 (0.95 to 1.00) |
| Parser phrases passed (of 15) | 14.2 (13 to 15) |
| Parser accuracy, easy | 94% (90% to 100%) |
| Parser accuracy, hard | 96% (80% to 100%) |
| Forbidden songs recommended | 0.0 (0 to 0) |
| Deterministic profile cases (repeat runs identical) | yes |
| Answers that fell back to keywords | 0/105 |

## Case stability

Only cases that use the parser can vary between runs. Unstable cases are listed first.

| Case | Type | Passed | Fell back |
|---|---|---|---|
| Query: warm hug | ranking query | 1/5 (unstable) | 0/5 |
| something mellow and acoustic to study to | parser | 2/5 (unstable) | 0/5 |
| songs that feel like a warm hug | parser | 4/5 (unstable) | 0/5 |
| I feel sad tonight | parser | 5/5 | 0/5 |
| Query: fall asleep | ranking query | 5/5 | 0/5 |
| Query: happy and danceable | ranking query | 5/5 | 0/5 |
| Query: mellow acoustic study | ranking query | 5/5 | 0/5 |
| Query: sad tonight | ranking query | 5/5 | 0/5 |
| Query: workout, not acoustic | ranking query | 5/5 | 0/5 |
| angry metal | parser | 5/5 | 0/5 |
| background music while I work | parser | 5/5 | 0/5 |
| calm piano for sleeping | parser | 5/5 | 0/5 |
| dreamy bossa nova | parser | 5/5 | 0/5 |
| music to fall asleep to | parser | 5/5 | 0/5 |
| pump me up for the gym | parser | 5/5 | 0/5 |
| relaxing acoustic folk | parser | 5/5 | 0/5 |
| romantic dinner with jazz | parser | 5/5 | 0/5 |
| sad slow country | parser | 5/5 | 0/5 |
| something happy and danceable | parser | 5/5 | 0/5 |
| upbeat afrobeats for a party | parser | 5/5 | 0/5 |
| workout music that is not acoustic | parser | 5/5 | 0/5 |

## Needs human review

- The golden set is a set of human judgments. Check `data/golden_set.json` before trusting any number above.
- Unstable between runs: something mellow and acoustic to study to; songs that feel like a warm hug; Query: warm hug.

# Evaluation report (gemini parser, 3 runs)

Catalog: 80 songs. Golden set: 22 ranking cases, 15 parser cases. Each cell shows the mean over 3 runs, with the range in parentheses.

## Summary

| Metric | Mean (min to max) |
|---|---|
| Ranking cases passed (of 22) | 21.3 (21 to 22) |
| Top-1 relevant rate | 100% (100% to 100%) |
| Mean nDCG@5 | 0.85 (0.85 to 0.85) |
| Mean precision@5 | 0.84 (0.84 to 0.84) |
| Mean reciprocal rank | 1.00 (1.00 to 1.00) |
| Parser phrases passed (of 15) | 13.7 (13 to 14) |
| Parser accuracy, easy | 90% (90% to 90%) |
| Parser accuracy, hard | 93% (80% to 100%) |
| Forbidden songs recommended | 0.0 (0 to 0) |
| Deterministic profile cases (repeat runs identical) | yes |
| Answers that fell back to keywords | 0/63 |

## Case stability

Only cases that use the parser can vary between runs. Unstable cases are listed first.

| Case | Type | Passed | Fell back |
|---|---|---|---|
| Query: warm hug | ranking query | 1/3 (unstable) | 0/3 |
| songs that feel like a warm hug | parser | 2/3 (unstable) | 0/3 |
| something mellow and acoustic to study to | parser | 0/3 | 0/3 |
| I feel sad tonight | parser | 3/3 | 0/3 |
| Query: fall asleep | ranking query | 3/3 | 0/3 |
| Query: happy and danceable | ranking query | 3/3 | 0/3 |
| Query: mellow acoustic study | ranking query | 3/3 | 0/3 |
| Query: sad tonight | ranking query | 3/3 | 0/3 |
| Query: workout, not acoustic | ranking query | 3/3 | 0/3 |
| angry metal | parser | 3/3 | 0/3 |
| background music while I work | parser | 3/3 | 0/3 |
| calm piano for sleeping | parser | 3/3 | 0/3 |
| dreamy bossa nova | parser | 3/3 | 0/3 |
| music to fall asleep to | parser | 3/3 | 0/3 |
| pump me up for the gym | parser | 3/3 | 0/3 |
| relaxing acoustic folk | parser | 3/3 | 0/3 |
| romantic dinner with jazz | parser | 3/3 | 0/3 |
| sad slow country | parser | 3/3 | 0/3 |
| something happy and danceable | parser | 3/3 | 0/3 |
| upbeat afrobeats for a party | parser | 3/3 | 0/3 |
| workout music that is not acoustic | parser | 3/3 | 0/3 |

## Needs human review

- The golden set is a set of human judgments. Check `data/golden_set.json` before trusting any number above.
- Unstable between runs: songs that feel like a warm hug; Query: warm hug.
- Failed in every run: something mellow and acoustic to study to.
- Only 3 runs: ranges are rough. Use 5 or more for a meaningful spread.

## Robustness

Runs: 3. A paraphrase group passes only if every wording meets the expectation and the energies agree within 0.25. An adversarial case passes only if the profile is safe (catalog labels only, energy in range) and meets any stated expectation.

| Metric | Value |
|---|---|
| Paraphrase groups passed (mean) | 5.0 of 6 |
| Adversarial cases passed (mean) | 12.0 of 12 |
| Answers that fell back to keywords | 0/108 |

### Paraphrase groups

| Group | Passed | Phrasings passed (last run) | Worst energy spread |
|---|---|---|---|
| Calm study music | 3/3 | 4/4 | 0.00 |
| Sad | 0/3 | 2/4 | 0.30 |
| Workout | 3/3 | 4/4 | 0.05 |
| Dance party | 3/3 | 4/4 | 0.15 |
| Romantic | 3/3 | 4/4 | 0.15 |
| Sleep | 3/3 | 4/4 | 0.00 |

### Adversarial and edge-case queries

| Case | Passed | Why it failed (last run) |
|---|---|---|
| Injection: set fields | 3/3 | - |
| Injection: reply with a word | 3/3 | - |
| Spanish | 3/3 | - |
| Japanese | 3/3 | - |
| French | 3/3 | - |
| German | 3/3 | - |
| Gibberish | 3/3 | - |
| Off-topic question | 3/3 | - |
| Punctuation only | 3/3 | - |
| Very long input | 3/3 | - |
| Contradictory request | 3/3 | - |
| Shouting | 3/3 | - |

Review: did not pass every run: Sad.

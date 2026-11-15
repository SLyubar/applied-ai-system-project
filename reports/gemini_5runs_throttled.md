# Evaluation report (gemini parser, 5 runs)

Catalog: 80 songs. Golden set: 32 ranking cases, 15 parser cases. Each cell shows the mean over 5 runs, with the range in parentheses.

## Summary

| Metric | Mean (min to max) |
|---|---|
| Ranking cases passed (of 32) | 27.6 (26 to 28) |
| Top-1 relevant rate | 90% (87% to 93%) |
| Mean nDCG@5 | 0.75 (0.72 to 0.77) |
| Mean precision@5 | 0.75 (0.72 to 0.79) |
| Mean reciprocal rank | 0.92 (0.89 to 0.95) |
| Parser phrases passed (of 15) | 13.6 (13 to 15) |
| Parser accuracy, easy | 94% (90% to 100%) |
| Parser accuracy, hard | 84% (80% to 100%) |
| Forbidden songs recommended | 0.0 (0 to 0) |
| Deterministic profile cases (repeat runs identical) | yes |
| Answers that fell back to keywords | 0/155 |
| Wall time per run | 121.6s (min 117.0s, max 133.3s) |

## Case stability

Only cases that use the parser can vary between runs. Unstable cases are listed first.

| Case | Type | Passed | Fell back |
|---|---|---|---|
| songs that feel like a warm hug | parser | 1/5 (unstable) | 0/5 |
| something mellow and acoustic to study to | parser | 3/5 (unstable) | 0/5 |
| Query: Sunday morning coffee | ranking query | 4/5 (unstable) | 0/5 |
| Query: rainy afternoon reading | ranking query | 4/5 (unstable) | 0/5 |
| calm piano for sleeping | parser | 4/5 (unstable) | 0/5 |
| Query: dinner party background | ranking query | 0/5 | 0/5 |
| Query: late night city drive | ranking query | 0/5 | 0/5 |
| Query: summer road trip | ranking query | 0/5 | 0/5 |
| Query: warm hug | ranking query | 0/5 | 0/5 |
| I feel sad tonight | parser | 5/5 | 0/5 |
| Query: fall asleep | ranking query | 5/5 | 0/5 |
| Query: furious | ranking query | 5/5 | 0/5 |
| Query: happy and danceable | ranking query | 5/5 | 0/5 |
| Query: mellow acoustic study | ranking query | 5/5 | 0/5 |
| Query: pump up before a game | ranking query | 5/5 | 0/5 |
| Query: sad tonight | ranking query | 5/5 | 0/5 |
| Query: sad winter night | ranking query | 5/5 | 0/5 |
| Query: salsa club | ranking query | 5/5 | 0/5 |
| Query: wedding slow dance | ranking query | 5/5 | 0/5 |
| Query: workout, not acoustic | ranking query | 5/5 | 0/5 |
| angry metal | parser | 5/5 | 0/5 |
| background music while I work | parser | 5/5 | 0/5 |
| dreamy bossa nova | parser | 5/5 | 0/5 |
| music to fall asleep to | parser | 5/5 | 0/5 |
| pump me up for the gym | parser | 5/5 | 0/5 |
| relaxing acoustic folk | parser | 5/5 | 0/5 |
| romantic dinner with jazz | parser | 5/5 | 0/5 |
| sad slow country | parser | 5/5 | 0/5 |
| something happy and danceable | parser | 5/5 | 0/5 |
| upbeat afrobeats for a party | parser | 5/5 | 0/5 |
| workout music that is not acoustic | parser | 5/5 | 0/5 |

## Confidence calibration

Does the confidence score separate good results from bad? Ranking cases with ground truth, grouped by the confidence band the system reported, pooled over 5 runs.

| Confidence | Cases | Passed | Pass rate |
|---|---|---|---|
| high | 104 | 101 | 97% |
| medium | 45 | 27 | 60% |
| low | 1 | 0 | 0% |

Pass rates fall as confidence falls, as they should.
Parser understanding score: mean 0.90 on phrases it got right (68) versus 0.74 on phrases it got wrong (7).

## Needs human review

- The golden set is a set of human judgments. Check `data/golden_set.json` before trusting any number above.
- Unstable between runs: something mellow and acoustic to study to; calm piano for sleeping; songs that feel like a warm hug; Query: rainy afternoon reading; Query: Sunday morning coffee.
- Failed in every run: Query: warm hug; Query: summer road trip; Query: dinner party background; Query: late night city drive.

## Robustness

Runs: 5. A paraphrase group passes only if every wording meets the expectation and the energies agree within 0.25. An adversarial case passes only if the profile is safe (catalog labels only, energy in range) and meets any stated expectation.

| Metric | Value |
|---|---|
| Paraphrase groups passed (mean) | 5.0 of 6 |
| Adversarial cases passed (mean) | 12.0 of 12 |
| Answers that fell back to keywords | 0/180 |

### Paraphrase groups

| Group | Passed | Phrasings passed (last run) | Worst energy spread |
|---|---|---|---|
| Calm study music | 5/5 | 4/4 | 0.00 |
| Sad | 0/5 | 2/4 | 0.30 |
| Workout | 5/5 | 4/4 | 0.05 |
| Dance party | 5/5 | 4/4 | 0.15 |
| Romantic | 5/5 | 4/4 | 0.15 |
| Sleep | 5/5 | 4/4 | 0.05 |

### Adversarial and edge-case queries

| Case | Passed | Why it failed (last run) |
|---|---|---|
| Injection: set fields | 5/5 | - |
| Injection: reply with a word | 5/5 | - |
| Spanish | 5/5 | - |
| Japanese | 5/5 | - |
| French | 5/5 | - |
| German | 5/5 | - |
| Gibberish | 5/5 | - |
| Off-topic question | 5/5 | - |
| Punctuation only | 5/5 | - |
| Very long input | 5/5 | - |
| Contradictory request | 5/5 | - |
| Shouting | 5/5 | - |

Review: did not pass every run: Sad.

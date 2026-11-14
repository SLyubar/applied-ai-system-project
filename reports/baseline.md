# Evaluation report (baseline: keyword parser + deterministic scorer)

Catalog: 80 songs. Golden set: 32 ranking cases, 15 parser cases.

## Summary

| Metric | Value |
|---|---|
| Mean precision@5 | 0.62 |
| Mean nDCG@5 | 0.61 |
| Mean reciprocal rank | 0.77 |
| Top-1 relevant rate | 73% |
| Ranking cases passed | 21/32 |
| Parser accuracy (all) | 11/15 (73%) |
| Parser accuracy (easy / hard) | 100% / 20% |
| Deterministic (repeat runs identical) | yes |
| Forbidden songs recommended (top 5) | 2 |
| Answers that fell back to keywords | 0/31 |
| Wall time per run | 0.0s (min 0.0s, max 0.0s) |

## Ranking cases

| Case | Input | P@5 | nDCG@5 | Result | Top 3 |
|---|---|---|---|---|---|
| Starter lofi | profile | 0.80 | 0.82 | PASS | Library Rain; Midnight Coding; Spacewalk Thoughts |
| Pop workout | profile | 0.80 | 0.91 | PASS | Gym Hero; Glitter Static; Sunrise City |
| Rock intense | profile | 0.80 | 0.77 | PASS | Storm Runner; Cedar Static; Garage Sunday |
| Conflict: high-energy sad classical | profile | 0.80 | 0.87 | PASS | Spring Allegro; Cathedral Suite; Moonlit Sonata Remix |
| Unknown genre and mood | profile | n/a | n/a | PASS | Late Shift Lyrics; Rain on Shibuya; Midnight Han River |
| Out-of-range energy (1.5) | profile | 1.00 | 0.95 | PASS | Sunrise City; Gym Hero; Block Party Gospel |
| No matches, no acoustic pref | profile | n/a | n/a | PASS | Thrift Store Love; Roots in Rain; Corazon Lento |
| Metal fan wants calm | profile | 0.60 | 0.82 | PASS | Deep Field Drift; Spacewalk Thoughts; Library Rain |
| Messy casing and whitespace | profile | 0.80 | 0.82 | PASS | Library Rain; Midnight Coding; Spacewalk Thoughts |
| Acoustic lover wants EDM | profile | 1.00 | 1.00 | PASS | Bassline Cathedral; Sunrise Drop; Noche de Salsa |
| Sad slow country | profile | 0.60 | 0.81 | PASS | Whiskey Letters; Moonlit Sonata Remix; Delta Dust |
| Romantic jazz evening | profile | 1.00 | 1.00 | PASS | Last Call Ballad; Letters to Orchard Lane; Cafe Corcovado |
| Dreamy city pop | profile | 0.80 | 0.84 | PASS | Rainy Tokyo Nights; Midnight Han River; Tokyo Skyline 1985 |
| Angry metal | profile | 1.00 | 1.00 | PASS | Iron Throne; Ashen Crown; Rust and Thunder |
| Happy afrobeats party | profile | 0.40 | 0.57 | PASS | Lagos Sunrise; Sunrise City; Block Party Gospel |
| Hopeful folk | profile | 1.00 | 1.00 | PASS | Porch Light Promise; Cathedral Suite; Letters to Orchard Lane |
| Query: mellow acoustic study | query | 0.80 | 0.83 | PASS | Library Rain; Midnight Coding; Spacewalk Thoughts |
| Query: sad tonight | query | 1.00 | 0.86 | PASS | Delta Dust; Heartbreak Receipts; Whiskey Letters |
| Query: workout, not acoustic | query | 0.60 | 0.57 | PASS | Sunrise Drop; Pulse Garden; Storm Runner |
| Query: fall asleep | query | 0.00 | 0.00 | FAIL | Thrift Store Love; Roots in Rain; Corazon Lento |
| Query: happy and danceable | query | 0.80 | 0.42 | FAIL | Sunday Brunch Swing; Spring Allegro; Sea Salt Skank |
| Query: warm hug | query | 0.00 | 0.00 | FAIL | Thrift Store Love; Roots in Rain; Corazon Lento |
| Query: rainy afternoon reading | query | 0.60 | 0.49 | FAIL | Midnight Coding; Golden Hour Hum; Hammock Dub |
| Query: summer road trip | query | 0.20 | 0.08 | FAIL | Thrift Store Love; Roots in Rain; Corazon Lento |
| Query: wedding slow dance | query | 0.00 | 0.00 | FAIL | Afterhours Static; Sunrise Drop; Pulse Garden |
| Query: dinner party background | query | 0.00 | 0.00 | FAIL | Sunrise Drop; Storm Runner; Calle Fuego |
| Query: late night city drive | query | 0.80 | 0.68 | FAIL | Rain on Shibuya; Smoky Corner; Rainy Motel Blues |
| Query: Sunday morning coffee | query | 0.40 | 0.35 | FAIL | Library Rain; Whiskey Letters; Cathedral Suite |
| Query: pump up before a game | query | 0.00 | 0.00 | FAIL | Thrift Store Love; Roots in Rain; Corazon Lento |
| Query: salsa club | query | 0.20 | 0.09 | FAIL | Afterhours Static; Sunrise Drop; Pulse Garden |
| Query: sad winter night | query | 1.00 | 0.70 | PASS | Delta Dust; Heartbreak Receipts; Whiskey Letters |
| Query: furious | query | 0.80 | 0.93 | PASS | Crown Heavy; Rust and Thunder; Ashen Crown |

### Failed ranking checks

- **Query: fall asleep**: top-1 is relevant, an ideal song is in the top 3
- **Query: happy and danceable**: top-1 is relevant, an ideal song is in the top 3
- **Query: warm hug**: top-1 is relevant, an ideal song is in the top 3
- **Query: rainy afternoon reading**: an ideal song is in the top 3
- **Query: summer road trip**: top-1 is relevant, an ideal song is in the top 3
- **Query: wedding slow dance**: top-1 is relevant, forbidden songs in top k: [14], an ideal song is in the top 3
- **Query: dinner party background**: top-1 is relevant, forbidden songs in top k: [24], an ideal song is in the top 3
- **Query: late night city drive**: an ideal song is in the top 3
- **Query: Sunday morning coffee**: an ideal song is in the top 3
- **Query: pump up before a game**: top-1 is relevant, an ideal song is in the top 3
- **Query: salsa club**: top-1 is relevant, an ideal song is in the top 3

## Confidence calibration

Does the confidence score separate good results from bad? Ranking cases with ground truth, grouped by the confidence band the system reported.

| Confidence | Cases | Passed | Pass rate |
|---|---|---|---|
| high | 22 | 15 | 68% |
| medium | 4 | 4 | 100% |
| low | 4 | 0 | 0% |

Pass rates do not fall cleanly with confidence, so treat the score as a rough guide only.
Parser understanding score: mean 1.00 on phrases it got right (11) versus 0.25 on phrases it got wrong (4).

## Parser cases

| Query | Tier | Result | Wrong fields |
|---|---|---|---|
| something mellow and acoustic to study to | easy | PASS | - |
| I feel sad tonight | easy | PASS | - |
| romantic dinner with jazz | easy | PASS | - |
| upbeat afrobeats for a party | easy | PASS | - |
| dreamy bossa nova | easy | PASS | - |
| angry metal | easy | PASS | - |
| relaxing acoustic folk | easy | PASS | - |
| workout music that is not acoustic | easy | PASS | - |
| sad slow country | easy | PASS | - |
| calm piano for sleeping | easy | PASS | - |
| music to fall asleep to | hard | FAIL | energy_max |
| something happy and danceable | hard | FAIL | energy_min |
| background music while I work | hard | FAIL | mood |
| pump me up for the gym | hard | PASS | - |
| songs that feel like a warm hug | hard | FAIL | mood |

## Needs human review

- The golden set is a set of human judgments. Check `data/golden_set.json` before trusting any number above.
- Cases with no ground truth, checked only for warnings: Unknown genre and mood, No matches, no acoustic pref.
- Decide for each failure whether the system or the golden set is wrong.

## Robustness

Runs: 1. A paraphrase group passes only if every wording meets the expectation and the energies agree within 0.25. An adversarial case passes only if the profile is safe (catalog labels only, energy in range) and meets any stated expectation.

| Metric | Value |
|---|---|
| Paraphrase groups passed (mean) | 0.0 of 6 |
| Adversarial cases passed (mean) | 8.0 of 12 |
| Answers that fell back to keywords | 0/36 |

### Paraphrase groups

| Group | Passed | Phrasings passed (last run) | Worst energy spread |
|---|---|---|---|
| Calm study music | 0/1 | 2/4 | 0.15 |
| Sad | 0/1 | 0/4 | 0.00 |
| Workout | 0/1 | 3/4 | 0.40 |
| Dance party | 0/1 | 2/4 | 0.40 |
| Romantic | 0/1 | 3/4 | 0.00 |
| Sleep | 0/1 | 3/4 | 0.35 |

### Adversarial and edge-case queries

| Case | Passed | Why it failed (last run) |
|---|---|---|
| Injection: set fields | 1/1 | - |
| Injection: reply with a word | 1/1 | - |
| Spanish | 0/1 | likes_acoustic; energy_max; understanding_min |
| Japanese | 0/1 | energy_max; understanding_min |
| French | 0/1 | energy_min; understanding_min |
| German | 0/1 | energy_max; understanding_min |
| Gibberish | 1/1 | - |
| Off-topic question | 1/1 | - |
| Punctuation only | 1/1 | - |
| Very long input | 1/1 | - |
| Contradictory request | 1/1 | - |
| Shouting | 1/1 | - |

Review: did not pass every run: Calm study music; Sad; Workout; Dance party; Romantic; Sleep; Spanish; Japanese; French; German.

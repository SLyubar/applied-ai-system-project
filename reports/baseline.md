# Evaluation report (baseline: keyword parser + deterministic scorer, graded golden set)

Catalog: 80 songs. Golden set: 22 ranking cases, 15 parser cases.

## Summary

| Metric | Value |
|---|---|
| Mean precision@5 | 0.73 |
| Mean nDCG@5 | 0.74 |
| Mean reciprocal rank | 0.88 |
| Top-1 relevant rate | 85% |
| Ranking cases passed | 19/22 |
| Parser accuracy (all) | 11/15 (73%) |
| Parser accuracy (easy / hard) | 100% / 20% |
| Deterministic (repeat runs identical) | yes |
| Forbidden songs recommended (top 5) | 0 |
| Answers that fell back to keywords | 0/21 |

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

### Failed ranking checks

- **Query: fall asleep**: top-1 is relevant, an ideal song is in the top 3
- **Query: happy and danceable**: top-1 is relevant, an ideal song is in the top 3
- **Query: warm hug**: top-1 is relevant, an ideal song is in the top 3

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

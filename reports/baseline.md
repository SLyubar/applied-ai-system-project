# Evaluation report (baseline: keyword parser + deterministic scorer)

Catalog: 80 songs. Golden set: 10 ranking cases, 15 parser cases.

## Summary

| Metric | Value |
|---|---|
| Mean precision@5 | 0.68 |
| Mean reciprocal rank | 1.00 |
| Top-1 relevant rate | 100% |
| Ranking cases passed | 10/10 |
| Parser accuracy (all) | 11/15 (73%) |
| Parser accuracy (easy / hard) | 100% / 20% |
| Deterministic (repeat runs identical) | yes |

## Ranking cases

| Case | P@5 | RR | Result | Top 3 |
|---|---|---|---|---|
| Starter lofi | 0.80 | 1.00 | PASS | Library Rain; Midnight Coding; Spacewalk Thoughts |
| Pop workout | 0.60 | 1.00 | PASS | Gym Hero; Glitter Static; Sunrise City |
| Rock intense | 0.60 | 1.00 | PASS | Storm Runner; Cedar Static; Garage Sunday |
| Conflict: high-energy sad classical | 0.80 | 1.00 | PASS | Spring Allegro; Cathedral Suite; Moonlit Sonata Remix |
| Unknown genre and mood | n/a | n/a | PASS | Late Shift Lyrics; Rain on Shibuya; Midnight Han River |
| Out-of-range energy (1.5) | 0.80 | 1.00 | PASS | Sunrise City; Gym Hero; Block Party Gospel |
| No matches, no acoustic pref | n/a | n/a | PASS | Thrift Store Love; Roots in Rain; Corazon Lento |
| Metal fan wants calm | 0.40 | 1.00 | PASS | Deep Field Drift; Spacewalk Thoughts; Library Rain |
| Messy casing and whitespace | 0.80 | 1.00 | PASS | Library Rain; Midnight Coding; Spacewalk Thoughts |
| Acoustic lover wants EDM | 0.60 | 1.00 | PASS | Bassline Cathedral; Sunrise Drop; Noche de Salsa |

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

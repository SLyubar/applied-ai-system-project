# Weight sensitivity and ablation

Catalog: 80 songs. Scored on the 16 profile cases in the golden set (free-text cases are excluded because they depend on the parser, not the weights).

| Variant | Weights (energy / mood / genre / acoustic) | Cases passed | Top-1 relevant | nDCG@5 | P@5 | Top-1 changed vs baseline | Forbidden songs |
|---|---|---|---|---|---|---|---|
| Baseline | 0.40 / 0.25 / 0.20 / 0.15 | 16/16 | 100% | 0.87 | 0.81 | 0 | 0 |
| Without energy | 0.00 / 0.25 / 0.20 / 0.15 | 16/16 | 100% | 0.81 | 0.74 | 5 | 0 |
| Without mood | 0.40 / 0.00 / 0.20 / 0.15 | 15/16 | 100% | 0.83 | 0.77 | 7 | 0 |
| Without genre | 0.40 / 0.25 / 0.00 / 0.15 | 15/16 | 100% | 0.69 | 0.56 | 2 | 0 |
| Without acoustic | 0.40 / 0.25 / 0.20 / 0.00 | 16/16 | 100% | 0.79 | 0.70 | 3 | 0 |
| Equal weights | 1.00 / 1.00 / 1.00 / 1.00 | 16/16 | 100% | 0.91 | 0.84 | 1 | 0 |
| Energy x0.5 | 0.20 / 0.25 / 0.20 / 0.15 | 16/16 | 100% | 0.84 | 0.77 | 1 | 0 |
| Energy x1.5 | 0.60 / 0.25 / 0.20 / 0.15 | 16/16 | 100% | 0.86 | 0.80 | 0 | 0 |
| Mood x0.5 | 0.40 / 0.12 / 0.20 / 0.15 | 16/16 | 100% | 0.95 | 0.90 | 0 | 0 |
| Mood x1.5 | 0.40 / 0.38 / 0.20 / 0.15 | 16/16 | 100% | 0.79 | 0.67 | 0 | 0 |
| Genre x0.5 | 0.40 / 0.25 / 0.10 / 0.15 | 16/16 | 100% | 0.78 | 0.67 | 0 | 0 |
| Genre x1.5 | 0.40 / 0.25 / 0.30 / 0.15 | 16/16 | 100% | 0.94 | 0.87 | 0 | 0 |
| Acoustic x0.5 | 0.40 / 0.25 / 0.20 / 0.07 | 16/16 | 100% | 0.81 | 0.71 | 2 | 0 |
| Acoustic x1.5 | 0.40 / 0.25 / 0.20 / 0.22 | 16/16 | 100% | 0.89 | 0.84 | 1 | 0 |

## Reading it

- Dropping a feature costs the most for **genre** (nDCG 0.87 to 0.69) and the least for **mood** (to 0.83).
- The weights are sensitive: the +/-50% variants range from nDCG 0.78 to 0.95.
- These cases and their judgments were written by one person, so treat small differences as noise. Only large drops are evidence that a feature matters.

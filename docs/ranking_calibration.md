# Ranking Calibration Notes

## AHP Matrix Rebalancing

### Old matrix (consensus-heavy)
Criteria order: `[norm_rating, fuzzy_quality, fuzzy_consensus, fuzzy_dispute]`

| | norm_rating | fuzzy_quality | fuzzy_consensus | fuzzy_dispute |
|---|---|---|---|---|
| norm_rating | 1 | 1 | 0.5 | 2 |
| fuzzy_quality | 1 | 1 | 0.5 | 2 |
| fuzzy_consensus | 2 | 2 | 1 | 3 |
| fuzzy_dispute | 0.5 | 0.5 | 0.333 | 1 |

**Old weights:** `[0.227, 0.227, 0.423, 0.122]` — fuzzy_consensus dominated at 42%
**Old CR:** 0.0037

### New matrix (star-rating primary)

| | norm_rating | fuzzy_quality | fuzzy_consensus | fuzzy_dispute |
|---|---|---|---|---|
| norm_rating | 1 | 2 | 3 | 4 |
| fuzzy_quality | 0.5 | 1 | 2 | 3 |
| fuzzy_consensus | 0.333 | 0.5 | 1 | 2 |
| fuzzy_dispute | 0.25 | 0.333 | 0.5 | 1 |

**New weights:** `[0.466, 0.277, 0.161, 0.096]`
**New CR:** 0.0113 (< 0.10, consistent)

## Confidence Discount: Bayesian Shrinkage

**Old formula:** `score × (1 - 1/n)` — zeroes out products with n=1 review

**New formula:** `(n × score + k × prior) / (n + k)`, where `k=5, prior=50.0`

| n reviews | old discount | new result (score=80) |
|---|---|---|
| 1 | 0 (zeroed) | ~58 (near prior) |
| 5 | 0.80 × score | ~65 |
| 30 | 0.97 × score | ~78 |

## Calibration Results

| Metric | Before | After |
|---|---|---|
| Pearson corr (avg_star ↔ senticrank_100) | 0.185 | **0.708** |
| Spearman ρ (SenticRank vs baseline rank) | 0.407 | **0.654** |
| n=1 product score | 0.0 (zeroed) | **49–58** (near prior 50) |

## Rationale

The original AHP matrix over-weighted `fuzzy_consensus` (model confidence), which itself
correlates with star_rating. This created an indirect path that paradoxically reduced
the direct star_rating influence while amplifying model artifacts. The new matrix
establishes `norm_rating` as the primary trusted signal, with ML signals providing
calibrated refinement.

# Lessons Learned — SenticRank V2

Four empirical decisions made during development that deviated from initial design.

## 1. Confidence as Precondition for Fake Detection

**Initial design:** Flag all reviews with mismatch ≥ 4 as `very_suspicious`, regardless of model confidence.

**Problem:** A mismatch=4 with confidence=0.30 is likely a model failure, not a fake.
Short or ambiguous texts (e.g. "Норм") give the model no signal, producing low-confidence
polar predictions. Flagging these creates false positives and penalizes legitimate reviews.

**Decision:** Require `confidence > threshold` as a gate before any suspicion label is assigned.
`mismatch=4 + confidence=0.3 → uncertain`, not `very_suspicious`.

**Result:** 95 confirmed fakes (0.57%) instead of 273 over-flagged (1.6%).

---

## 2. Bayesian Shrinkage Instead of 1−1/n Discount

**Initial design:** `final_score = topsis_score × (1 − 1/n_reviews)`

**Problem:** Products with n=1 review receive score=0 regardless of quality.
One highly-rated product with a single 5-star verified review scored below every
product with ≥2 reviews, including products with mediocre ratings.

**Decision:** Replace with Bayesian shrinkage:
`discounted = (n × score + k × prior) / (n + k)`, where `k=5, prior=50`.

**Result:** n=1 products score 49–58 (near neutral prior), not 0.
Cold-start problem handled smoothly without destroying information.

---

## 3. AHP Matrix Rebalance — Star Rating as Primary Signal

**Initial design:** Pairwise matrix gave `fuzzy_consensus` (model confidence) the highest
weight at 42.3%. Rationale was that consensus reflects "certainty of quality signal".

**Problem:** `fuzzy_consensus` is derived from model confidence, which itself correlates
with star_rating. This created an indirect double-path that paradoxically weakened the
direct star signal while amplifying a noisy proxy. Correlation avg_star ↔ senticrank = 0.185.

**Decision:** Rebalance to `norm_rating` as primary criterion.
New weights: [0.466, 0.277, 0.161, 0.096] (star dominates at 47%).

**Result:** Correlation avg_star ↔ senticrank jumped to 0.708.
The system now agrees with human star ratings while still providing differentiation.

---

## 4. Fifth TOPSIS Criterion — Helpful Votes as Tie-Breaker

**Problem discovered late:** 13 products shared rank #1 with identical score 92.857.
All had n=30 reviews (data collection ceiling), all 5⭐, identical feature vectors.
TOPSIS produced a degenerate tie that would be flagged on defense.

**Decision:** Add `fuzzy_engagement = log1p(total_helpful_votes)` as a 5th criterion
with the lowest AHP weight (~0.062). This is an independent crowd-validation signal
orthogonal to star ratings and model predictions.

**Result:** All 13 ties broken. Rank #1 is Apple iPhone 13 with 3,340 helpful votes —
a product the community validated independently of star ratings.
AHP CR remains acceptable at 0.0151 (< 0.10).

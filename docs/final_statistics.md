# Final Statistics — SenticRank V2

## Dataset
| Metric | Value |
|--------|-------|
| Total reviews | 16,825 |
| Unique products | 684 |
| Categories | 9 |
| Languages | Russian, Kazakh |
| Reviews per product (mean) | 24.6 |
| Reviews per product (max) | 30 |

## Stage 2 — Star Rating Predictor

| Model | Accuracy | Bal. Acc | F1-macro | MAE | QWK |
|-------|----------|----------|----------|-----|-----|
| Dummy (baseline) | 0.793 | 0.200 | 0.177 | 0.582 | 0.000 |
| TF-IDF + LogReg | **0.832** | **0.427** | **0.422** | **0.302** | **0.787** |
| TF-IDF + SVM | 0.863 | 0.386 | 0.390 | 0.270 | 0.783 |

**Best model:** `tfidf_logreg` (selected by F1-macro)

F1 per class (tfidf_logreg):
- 1⭐: 0.673
- 2⭐: 0.117 ← hardest class (underrepresented)
- 3⭐: 0.174
- 4⭐: 0.207
- 5⭐: 0.940

## Stage 3 — Fake Review Detection

| Metric | Value |
|--------|-------|
| Total reviews scored | 16,825 |
| Clean | 16,155 (96.0%) |
| Uncertain (low confidence) | 575 (3.4%) |
| Suspicious (mismatch=2, conf>0.7) | 33 (0.2%) |
| Very suspicious (mismatch≥3, conf>0.6) | 62 (0.4%) |
| **Total fakes removed** | **95 (0.57%)** |

Detection thresholds: `confidence_threshold=0.6`, `suspicious_confidence=0.7`

## Stage 4 — SenticRank Ranking Engine

### AHP Configuration (5 criteria)
| Criterion | Column | Weight | Type |
|-----------|--------|--------|------|
| Star rating | agg_norm_rating | **0.416** | benefit |
| ML quality | fuzzy_quality | 0.262 | benefit |
| Consensus | fuzzy_consensus | 0.161 | benefit |
| Anti-dispute | fuzzy_dispute | 0.099 | benefit |
| Engagement | fuzzy_engagement | 0.062 | benefit (tie-breaker) |

**AHP Consistency Ratio: CR = 0.0151** (< 0.10 threshold ✓)

### Ranking Quality
| Metric | Value |
|--------|-------|
| Products ranked | 684 |
| Fakes removed before ranking | 95 |
| Spearman ρ (SenticRank vs star baseline) | 0.465 |
| Pearson corr (avg_star ↔ senticrank_100) | 0.619 |
| Tied ranks at top | 0 (all unique in top-15) |

### Confidence Discount
Method: Bayesian shrinkage — `(n × score + 5 × 50) / (n + 5)`

| n reviews | effective data weight |
|-----------|----------------------|
| 1 | 17% |
| 5 | 50% |
| 10 | 67% |
| 30 | 86% |

## Output Artifacts
| File | Description |
|------|-------------|
| `data/outputs/star_predictor/best_model.joblib` | Trained TF-IDF + LogReg pipeline |
| `data/processed/dataset_with_fake_scores.csv` | All 16,825 reviews + scoring columns |
| `data/processed/dataset_clean.csv` | 16,763 clean reviews |
| `data/outputs/fake_detector/top_fake_examples.csv` | Top-20 suspicious reviews |
| `data/outputs/ranking/senticrank_product_rankings.csv` | Final 684-product ranking |
| `data/outputs/ranking/category_rankings/` | Per-category CSVs (9 files) |

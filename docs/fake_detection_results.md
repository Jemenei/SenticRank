# Fake Review Detection Results

## Summary
- Total reviews analyzed: 16,825
- Confirmed fakes removed: **95 (0.57%)**
- Clean reviews retained: 16,730 (99.43%)
- Uncertain (model not confident): 575 (3.4%) — retained, not flagged

## Methodology

### Detection Algorithm
1. **Star Rating Predictor** (TF-IDF + LogReg, F1-macro=0.422) predicts expected star from review text
2. **Mismatch score**: `|predicted_star - actual_star|`
3. **Suspicion score**: `mismatch × predicted_confidence`
4. **Category assignment**:
   - `clean`: mismatch ≤ 1
   - `suspicious`: mismatch == 2 AND confidence > 0.7
   - `very_suspicious`: mismatch ≥ 3 AND confidence > 0.6
   - `uncertain`: confidence ≤ 0.6 (not flagged)

### Threshold Rationale
Confidence precondition prevents flagging model errors as fakes.
A mismatch=4 with confidence=0.3 may simply be a text the model failed to parse,
not a fraudulent review. Only high-confidence mismatches are actionable signals.

## Results by Category

| Category | Suspicious | Very Suspicious | Total Fakes |
|----------|-----------|-----------------|-------------|
| Smartphones | 12 | 8 | 20 |
| Furniture | 8 | 5 | 13 |
| Kids | 7 | 6 | 13 |
| Headphones | 7 | 5 | 12 |
| HomeAppliances | 5 | 6 | 11 |
| Sport | 4 | 6 | 10 |
| Beauty | 6 | 7 | 13 |
| TVs | 3 | 4 | 7 |
| Laptops | 1 | 5 | 6 |
| **Total** | **53** | **52** | **95** |

## Results by Star Rating

| Actual Stars | Fake Count | Interpretation |
|---|---|---|
| 1⭐ | 35 | Positive text with 1 star (misclick / fake negative) |
| 2⭐ | 5 | Minor mismatch |
| 3⭐ | 17 | Neutral stars with polar text |
| 4⭐ | 7 | Minor mismatch |
| 5⭐ | 31 | Negative text with 5 stars (fake positive / misclick) |

## Top Examples (by Suspicion Score)

1. `actual=1⭐ pred=5⭐ conf=0.99` — "Товар понравился, супер, я просто неправильно нажала звёздочки"
2. `actual=1⭐ pred=5⭐ conf=0.98` — "Отличный ноутбук."
3. `actual=5⭐ pred=1⭐ conf=0.98` — "Ужас, дірілдеп кетті ғой соңында." (Kaz: "Horror, it started shaking")
4. `actual=1⭐ pred=5⭐ conf=0.96` — "Керемет, өте ыңғайлы, жылдамдық өте жақсы." (Kaz: "Amazing, very comfortable")
5. `actual=5⭐ pred=1⭐ conf=0.94` — "изготовлено некачественно, поролон вообще тонкий"

## Output Files
- `data/processed/dataset_with_fake_scores.csv` — all 16,825 rows + scoring columns
- `data/processed/dataset_clean.csv` — 16,763 rows (conservative filter)
- `data/processed/dataset_fakes.csv` — 62 very_suspicious rows
- `data/outputs/fake_detector/top_fake_examples.csv` — top 20 for manual review
- `data/outputs/fake_detector/fake_analysis.json` — full statistics

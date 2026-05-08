# Bilingual Validation — Star Predictor

## Method
Manual qualitative validation: hand-crafted test sentences in Russian and Kazakh
covering positive, negative, and neutral sentiment.

## Test Cases

### Russian
| Text | Expected | Predicted | Confidence | Status |
|------|----------|-----------|------------|--------|
| Отличный товар, всем рекомендую, очень доволен | 5⭐ | 5⭐ | 99.2% | ✓ |
| Полная фигня, не работает, разочарован | 1⭐ | 1⭐ | 87.7% | ✓ |
| Нормально, но есть мелкие недочёты | 3-4⭐ | 4⭐ | 92.1% | ✓ |
| Ужасный товар, обманули, не рекомендую | 1⭐ | 1⭐ | 98.6% | ✓ |

### Kazakh (Cyrillic)
| Text | Translation | Expected | Predicted | Confidence | Status |
|------|-------------|----------|-----------|------------|--------|
| Крем жақсы, ұнады | Cream is good, liked it | 5⭐ | 5⭐ | 52.0% | ✓ |
| Сапасы нашар, жетпей қалды | Quality bad | 1-2⭐ | 1⭐ | 54.1% | ✓ |
| Өте күшті, керемет ұнады | Very strong, awesome | 5⭐ | 5⭐ | 98.0% | ✓ |
| Жаман тауар, ақшамды қайтарыңыз | Bad, return money | 1⭐ | 1⭐ | 72.8% | ✓ |
| Орташа, бірақ жаман емес | Medium, but not bad | 3⭐ | 4⭐ | 66.6% | ~ |

## Summary
- Russian: 4/4 correct, mean confidence 94%
- Kazakh: 4/5 exact match, 5/5 correct sentiment polarity
- Bilingual training data: 2,349 reviews with Kazakh-specific characters (~20% of train)

## Limitations
- Neutral class (3⭐) underrepresented (3.7%), causing 4⭐ lean for neutral texts
- Pure Latin-script Kazakh: only 4 examples in training set
- Pure English: only 5 examples, unreliable predictions expected

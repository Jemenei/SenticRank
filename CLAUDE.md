# SenticRank V2 — Project Context for Claude Code

## Quick reference
- Run tests: `pytest -q`
- Run training: `senticrank train-predictor`
- Run fake detection: `senticrank detect-fakes`
- Run ranking: `senticrank rank`

## Conventions
- Python 3.10+, type hints, Google docstrings
- Logging via `logging` module, NOT print
- Hyperparams in configs/default.yaml ONLY
- Data files NEVER read fully into context (use df.head/sample)

## Architecture
ML Core (star_predictor) → Anomaly Layer (fake_detector) → MCDM Layer (ranking)

## Status
- Phase 1 done: structure + config
- Phase 2 in progress: star predictor
- Phase 3 pending: fake detector
- Phase 4 pending: ranking engine

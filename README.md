# Identify the Author

An authorship-attribution experiment for short literary excerpts. The repository contains a reproducible-looking modeling pipeline built around text preprocessing, TF-IDF and learned features, cross-validation, calibration, and ensemble/submission utilities.

## Status

The implementation and competition data are present, but this repository does not preserve a reliable final leaderboard report or a verified best score. I therefore do not claim a ranking or metric here. Run the pipeline locally and record new results with the exact dependency set, seed, and data snapshot used.

## Problem and metric

Given a short excerpt, predict one of three authors. The competition metric is multiclass log-loss, so probability quality matters more than accuracy alone.

The tracked dataset contains `data/train/train.csv`, `data/test/test.csv`, and `data/sample_submission/sample_submission.csv`. Confirm the competition's data and redistribution terms before using the bundled data outside this repository.

## Validation strategy

The code uses stratified cross-validation with a fixed seed in its evaluation helpers. Keep the test set isolated: use only the training labels for model selection, calibration, feature decisions, and ensemble weights. A Kaggle submission is an external result, not a substitute for a locally reproducible validation report.

## What is implemented

- text preprocessing and feature construction in `src/`
- classical TF-IDF model training with optional richer model paths
- log-loss and accuracy evaluation helpers
- probability calibration and ensemble/submission generation
- diagnostic helpers for learning curves, calibration, and confusion reports

The codebase includes experimental paths for heavier dependencies such as PyTorch, XGBoost, and Transformers. They are not required for the lightweight smoke test below.

## Reproduce the smoke test

```bash
python -m pip install -r requirements.txt
python src/run_pipeline.py --test
```

To run the full pipeline, expect model downloads and materially higher CPU/RAM requirements:

```bash
python src/run_pipeline.py
```

The pipeline writes generated submissions and diagnostics under `outputs/`; those files are intentionally ignored and should not be committed.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/` | competition train, test, and sample-submission files |
| `src/` | loading, preprocessing, features, models, evaluation, ensembling, and submission |
| `diagnostics.py` | learning-curve, calibration, holdout, and confusion-report helpers |
| `main.py` | original end-to-end entry point |
| `requirements.txt` | Python dependencies |

## Limitations and lessons

- No final score is claimed because the preserved repository does not include a verifiable run report.
- The data split, preprocessing choices, calibration method, and ensemble weights can materially change log-loss.
- Optional transformer/GPU paths require separate dependency and hardware validation.
- Do not use test labels, leaderboard feedback, or generated submission artifacts to tune a local validation result.
- Preserve the exact command, commit, data hash, seed, hardware, and dependency lockfile whenever recording a future result.

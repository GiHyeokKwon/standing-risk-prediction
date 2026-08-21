![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green.svg)

# Standing Risk Prediction

**Design of a QR-Based Standing Passenger Risk Notification Service Using AI Models
for Transportation-Vulnerable Individuals**

KT · Korea Road Traffic Authority · Korea (May 2026 ~ Aug 2026)

The AI component of a QR-based standing-risk notification system for
transportation-vulnerable individuals. When a passenger scans a QR code at
a bus stop, the system predicts, based on the boarding/alighting stops,
**whether they will have to stand** and **how long they will stand**,
so it can warn them of standing risk (low/medium/high) before they board.

## Model Overview

Two LightGBM models used sequentially in a two-stage pipeline.

- **Model A (classification)**: predicts whether the passenger will stand (Y/N) at boarding time
- **Model B (regression)**: for passengers Model A predicts as standing, predicts how many seconds until a seat opens up

The ground-truth labels (`is_standing`, `standing_seconds`) were derived from
raw transit card data (TCD) by reconstructing onboard passenger counts and
simulating FIFO seat assignment.

**Input features (8)**: `route_id`, `board_stop_id`, `alight_stop_id`, `weekday`,
`weather`, `hour`, `is_holiday`, `headway_sec`

## Final Model Performance

Based on one year (2024) of Seongbuk-gu data (~330M rows), validated with a
temporal split (trained on Jan–Nov, validated on Dec).

| | Model A (AUC) | Model A (Accuracy) | Model B (MAE) | Model B (RMSE) |
| --- | --- | --- | --- | --- |
| baseline (num_leaves=31) | 0.9141 | 88.12% | 158.3s | 243.1s |
| num_leaves=63 | 0.9167 | 88.24% | 155.3s | 240.8s |
| **num_leaves=127 (final)** | **0.9180** | **88.29%** | **153.5s** | **239.5s** |

The final deployment model was retrained on the full year (Jan–Dec), including
the December data used for validation (`n_estimators=3000`, `num_leaves=127`).

**Feature importance**: the board/alight stop pair was consistently the most
influential variable for both models — interpreted as the route segment being
the strongest direct determinant of congestion.

## Experimentation Process

Started at a smaller scale (Seocho-gu, one month of data) and progressively
scaled up to production scale (Seongbuk-gu, one full year, 300M+ rows).

- **Increasing tree count (300 → 5000)**: confirmed diminishing returns as tree count grew, prompting a shift to other hyperparameters
- **Expanding num_leaves (31 → 63 → 127)**: the largest and most consistent improvement of any parameter tried
- **Switching validation strategy**: found that random splits produced overly optimistic performance, and switched to temporal (past → future) validation
- **Memory optimization for scaling**: for processing 300M+ rows on a local machine (16GB RAM), applied streaming merges, early categorical encoding, and separate train/test storage. Later moved training to a cloud GPU (RTX A5000, 64 cores / 240GB RAM) for full-scale runs

## Backend Integration

Training was done in Python (LightGBM) while the deployment backend runs on
Java (Spring), so models were converted to **PMML** for delivery.

1. Converted LightGBM models (`.txt`) to PMML using [jpmml-lightgbm](https://github.com/jpmml/jpmml-lightgbm)
2. Extracted integer code mappings for categorical features (stop IDs, route IDs, etc.) directly from the model files and delivered them alongside the PMML files
3. Generated **20 golden test samples** (common / rare / boundary cases) to cross-validate that Python's original predictions match the PMML (Java) predictions with the backend team

## Presentation

[View Project Presentation (PDF)](docs/presentation.pdf)

## Structure

```text
.
├── docs/                                   # meeting notes, data spec documents
├── schema/                                 # example schema (not real data)
├── src/
│   ├── merge_data.py                       # merges daily raw files into a single parquet
│   ├── train.py                            # validation training (temporal split, hyperparameter experiments)
│   ├── train_deploy.py                     # final deployment training (full period)
│   ├── category_code.py                    # extracts categorical feature code mappings from trained models
│   └── golden_test_samples.py              # generates input-output sample sets for PMML validation
├── experiments/
│   └── {run_timestamp}/
│       ├── train_log.txt
│       ├── metrics.json
│       ├── model_a.pmml / model_b.pmml
│       ├── category_code_mapping_model_a.json / _model_b.json
│       └── golden_test_samples.json
├── .gitignore
├── LICENSE
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# 1. Merge daily raw data
python src/merge_data.py

# 2. Hyperparameter experiments (temporal split validation)
python src/train.py

# 3. Final deployment training (full period, no holdout)
python src/train_deploy.py

# 4. Extract categorical feature code mappings
python src/category_code.py

# 5. Generate golden test samples for PMML validation
python src/golden_test_samples.py
```

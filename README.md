<p align="center">
  <img src="asset/Logo_02.png" alt="Home Credit Risk Intelligence" width="260" />
</p>

<h1 align="center">Home Credit Default Risk</h1>
<h3 align="center">Open-Finance Credit Risk Prediction — End-to-End MLOps Case Study</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-4.7-00B0F0?style=flat-square" />
  <img src="https://img.shields.io/badge/FastAPI-0.138-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-3.15-0194E2?style=flat-square&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/DVC-3.37-945DD6?style=flat-square&logo=dvc&logoColor=white" />
  <img src="https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Render-2496ED?style=flat-square&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Dashboard-Vercel-000000?style=flat-square&logo=vercel&logoColor=white" />
</p>

---

## Project Definition

This project addresses one of the most critical challenges in consumer lending: **predicting credit default risk for applicants who have little or no formal credit history**. Leveraging the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset — an internationally recognized benchmark in financial machine learning — this case study builds an end-to-end MLOps pipeline, from raw data ingestion through a leakage-aware validation strategy, statistical feature analysis, model training, and a live API with an interactive dashboard, at zero infrastructure cost.

The work covers the full data science lifecycle: rigorous statistical feature analysis across eight relational tables, advanced feature engineering, imbalanced-class treatment, gradient boosting with automated hyperparameter optimization, model explainability, probability calibration, automated tests and CI, and zero-cost cloud deployment. See [Limitations](#limitations) below for what this project does not claim.

---

## Central Objective

> **Determine, with statistical rigor and full interpretability, the probability that a loan applicant will default — enabling financial institutions to extend credit responsibly to underserved populations while minimizing risk exposure.**

Secondary objectives:
- Demonstrate mastery of the complete feature engineering lifecycle across heterogeneous relational tables
- Apply robust statistical validation (Kolmogorov-Smirnov, Chi-Square, Bootstrap CI) to justify feature selection
- Build a reproducible, version-controlled ML pipeline with experiment traceability
- Serve predictions through a production API and visualize model behavior through an interactive dashboard

---

## Dataset

| Table | Rows | Description |
|---|---|---|
| `application_train` | 307,511 | Main application data with TARGET label |
| `application_test` | 48,744 | Test set for final submission |
| `bureau` | 1,716,428 | Credit history from other financial institutions |
| `bureau_balance` | 27,299,925 | Monthly balance snapshots of bureau credits |
| `previous_application` | 1,670,214 | All previous Home Credit loan applications |
| `POS_CASH_balance` | 10,001,358 | Monthly POS and cash loan balance snapshots |
| `credit_card_balance` | 3,840,312 | Monthly credit card balance snapshots |
| `installments_payments` | 13,605,401 | Repayment history for disbursed credits |

**Key characteristic:** Severe class imbalance — only **8.07%** of applicants defaulted (1:11 ratio), handled via `class_weight="balanced"` by default (see [Imbalanced Class Treatment](#3-imbalanced-class-treatment)).

---

## Tech Stack

### Machine Learning & Statistics
| Library | Purpose |
|---|---|
| `scikit-learn` | Preprocessing, calibration, cross-validation |
| `LightGBM` | Primary gradient boosting model |
| `imbalanced-learn` | Optional SMOTE/ADASYN resampling (fold-safe `Pipeline`; off by default — see [Imbalanced Class Treatment](#3-imbalanced-class-treatment)) |
| `Optuna` | Bayesian hyperparameter optimization with MedianPruner |
| `SHAP` | TreeExplainer — global and local model explainability |
| `scipy` | KS test, Chi-Square, statistical analysis |

### MLOps & Experiment Tracking
| Tool | Purpose |
|---|---|
| `MLflow` | Experiment tracking, model registry, metric logging |
| `DVC` | Data versioning, pipeline reproducibility |

### Serving & Deployment
| Tool | Purpose |
|---|---|
| `FastAPI` | REST API with OpenAPI documentation |
| `Docker` | Containerized API deployment |
| `Render` (free tier) | Cloud API hosting — zero cost |
| `Next.js 14` | Interactive TypeScript dashboard |
| `Vercel` | Dashboard deployment — zero cost |

---

## Methodology

### 1. Feature Engineering (200+ Features)
Features were engineered across all eight relational tables, capturing patterns impossible to detect from the main application table alone:

- **Application ratios:** `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`, `EMPLOYED_TO_AGE_RATIO`, `EXT_SOURCE_MEAN/STD/WEIGHTED`
- **Bureau aggregations:** credit count, active/bad debt ratios, overdue days (mean, max), debt-to-credit ratios
- **Payment behavior:** installment DPD/DBD distributions, late payment ratios, payment shortfall patterns
- **Previous application history:** approval/refusal rates, recency of last decision, credit-to-application ratios
- **Credit card behavior:** balance-to-limit ratios, ATM drawing patterns, DPD statistics

### 2. Statistical Validation
All engineered features were validated using formal hypothesis tests before model inclusion, computed **on the training partition only** (see [Validation Strategy](#validation-strategy)):

- **Kolmogorov-Smirnov test** (numeric features): two-sample distribution comparison between DEFAULT=0 and DEFAULT=1 groups
- **Chi-Square test** (categorical features): independence testing between each categorical variable and TARGET
- **Bootstrap confidence intervals:** 1,000 iterations to produce 95% CIs for AUC-ROC, AUC-PR, Precision, Recall, and F1, computed on the held-out test partition

See [Results](#results) for current figures.

### 3. Imbalanced Class Treatment
With only 8.07% positives, class imbalance is handled by **`class_weight="balanced"`** in LightGBM by default — it adds no synthetic data and no extra hyperparameters to tune. SMOTE and ADASYN remain available (`sampling.method` in `params.yaml`) and, when enabled, run inside an `imblearn.pipeline.Pipeline` so resampling is fit only on the training fold of each cross-validation split, never on the held-out fold. See [`reports/sampling_comparison.md`](reports/sampling_comparison.md) for the comparison that motivated the default and the full rationale.

### 4. Hyperparameter Optimization
**Optuna** with `TPESampler` and `MedianPruner` ran 50 trials over:
`n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`

5-fold stratified cross-validation on the **training partition** was used as the objective metric, maximizing AUC-ROC. The final model is then early-stopped (50 rounds) against the **calibration partition** — never the training or test partitions.

### 5. Probability Calibration
After the final fit, **isotonic regression** calibration was applied (`CalibratedClassifierCV` wrapping a `FrozenEstimator`, scikit-learn ≥ 1.6) on the **calibration partition** to ensure that a predicted score of 30% corresponds to an actual 30% default rate — critical for financial risk management applications.

### 6. Threshold Optimization
Rather than defaulting to 0.5, the classification threshold is chosen by grid-searching [0.01, 0.99] in 200 steps on the **calibration partition**. Two selection metrics are supported (`threshold.metric` in `params.yaml`):

- **F1-score (default)** — needs no business assumptions about the relative cost of errors.
- **Cost-based** — minimises `cost_fn * false_negatives + cost_fp * false_positives`. The `cost_fn`/`cost_fp` weights in `params.yaml` (5:1 by default) are an **illustrative placeholder**, not a validated figure for this portfolio, and should be replaced with real unit economics before being used to drive an actual lending decision.

F1 is the default specifically because, absent a validated cost ratio, optimizing for an invented one would create false precision. The cost-based option exists for when real figures are available.

---

## Validation Strategy

Earlier versions of this pipeline reused a single validation split for early
stopping, probability calibration, threshold selection, *and* the final
reported metrics — which overstates performance, since the threshold and
calibration end up tuned against the same data they are then scored on. The
current pipeline instead performs a **three-way, label-stratified split**,
computed once in the features stage before any fitting step:

| Partition | Size | Used for |
|---|---|---|
| **Train** | 70% | Optuna's cross-validated hyperparameter search; the final LightGBM fit |
| **Calibration** | 15% | Early stopping; isotonic calibration; threshold selection |
| **Test** | 15% | Final metrics, bootstrap CI, SHAP importance, fairness and error analysis — untouched until this point |

Splits and the random seed are configured in `params.yaml`
(`data.calibration_size`, `data.test_size`, `data.random_state`); the split
logic itself lives in `src/data/splitting.py`.

Two further leakage sources were removed alongside the split rework:

- **Preprocessing fit scope.** Categorical label encoders and numeric-median
  imputation (`encode_and_impute` in `src/features/engineering.py`) are fit
  on the training partition only, then applied unchanged to calibration,
  test, and the Kaggle submission set. The statistical feature-selection
  report (KS / Chi-square) is likewise computed on the training partition
  only.
- **Resampling inside cross-validation.** When SMOTE/ADASYN is enabled, it
  now runs inside an `imblearn.pipeline.Pipeline`, so it is fit only on the
  training fold of each CV split — never on the fold being scored. The
  default (`class_weight="balanced"`) needs no resampling at all. See
  [`reports/sampling_comparison.md`](reports/sampling_comparison.md).

`tests/test_no_leakage.py` encodes these invariants as automated tests (split
disjointness/stratification, and that a value imputed into the calibration
partition comes from the training partition's median, not its own).

---

## Results

All figures below come from a full pipeline run against the real dataset
(`dvc repro`, 2026-10-04), measured on the **test partition** — 46,127
applications never used for fitting, early stopping, calibration, or
threshold selection.

| Metric | Value | 95% Bootstrap CI | Notes |
|---|---|---|---|
| **AUC-ROC** | 0.7882 | [0.7808, 0.7957] | Test partition |
| **AUC-PR** | 0.2708 | [0.2575, 0.2843] | Imbalanced-aware metric |
| **KS Statistic** | 0.4359 | — | Model discrimination power |
| **F1 Score** | 0.3339 | [0.3208, 0.3461] | At optimal threshold (0.1676) |
| **Precision** | 0.2880 | [0.2751, 0.3003] | At optimal threshold |
| **Recall** | 0.3972 | [0.3812, 0.4121] | At optimal threshold |
| **Optimal Threshold** | 0.1676 | — | Maximizes F1 on the calibration partition |
| **Calibration** | Isotonic | — | Fit on the calibration partition |

Bootstrap CIs use 1,000 resamples of the test partition (percentile method),
computed by `src/utils/stats.py::bootstrap_metrics`.

**Why F1/precision/recall are lower than earlier reported figures, while
AUC-ROC and KS are not:** the pipeline previously selected its decision
threshold on the same split used for calibration and then reported metrics
on that split too — the threshold ends up tuned to that specific sample's
noise, inflating F1 at that operating point. AUC-ROC and the KS statistic
are threshold-independent (they measure ranking quality across *all*
thresholds), so they were less affected by that leakage and, here, came out
slightly *higher* under the corrected methodology (0.7882 vs. a previously
reported 0.781) — the model's ability to rank applicants by risk did not
regress. The optimal threshold also moved substantially (0.35 → 0.1676),
consistent with switching the imbalance strategy from SMOTE to
`class_weight="balanced"` (see [`reports/sampling_comparison.md`](reports/sampling_comparison.md)),
which shifts the model's output probability distribution. In short: this
model discriminates risk at least as well as before; what changed is that
the reported precision/recall/F1 now honestly reflect performance on data
the threshold was never tuned against.

**Top 10 Features by SHAP Importance** (mean |SHAP| on the test partition):
1. `EXT_SOURCE_MEAN` — Mean of the three external credit bureau scores (0.2859)
2. `CODE_GENDER` — Applicant gender (0.1281) — see [Dashboard Pages → Responsible AI](#dashboard-pages) for the corresponding subgroup fairness analysis
3. `POS_CNT_FUTURE_MEAN` — Mean remaining POS/cash installments (0.1194)
4. `GOODS_CREDIT_RATIO` — Goods price to credit amount ratio (0.1013)
5. `EXT_SOURCE_WEIGHTED` — Weighted external credit score blend (0.0968)
6. `EXT_SOURCE_MAX` — Maximum external credit score (0.0899)
7. `NAME_EDUCATION_TYPE` — Applicant education level (0.0875)
8. `EXT_SOURCE_MIN` — Minimum external credit score (0.0871)
9. `CREDIT_TERM` — Credit amount to annuity ratio (0.0766)
10. `BUREAU_AMT_DEBT_MEAN` — Mean outstanding debt across bureau-reported credits (0.0765)

`CODE_GENDER` ranking 2nd by SHAP importance is a direct prompt to check
whether the model treats gender groups equitably. SHAP importance alone is
**not** a fairness audit — see the subgroup fairness metrics surfaced on the
dashboard's Responsible AI page and the [Limitations](#limitations) section
below.

**Drift check (train vs. Kaggle `application_test` holdout):** 12 features
checked, 0 flagged moderate or significant (all PSI < 0.10) — expected,
since both are splits of the same historical competition snapshot rather
than genuinely different time periods (see [Limitations](#limitations)).

**Fairness snapshot (test partition, by `CODE_GENDER`):** disparate impact
ratio **0.53** (below the commonly used 0.8 four-fifths-rule screening
threshold) and an equal-opportunity gap of **13.6 percentage points** TPR
between groups (M: 0.47, F: 0.34) — against base default rates of 10.2% (M)
and 7.0% (F) in this test partition. Part of the gap tracks a real difference
in observed default rates between groups in this data, but a ratio this far
below 0.8 is exactly the kind of signal that should trigger a fair-lending
review before any real lending decision, not just a note in a README — see
[Limitations](#limitations) and the dashboard's Responsible AI page for the
full breakdown (all segments, calibration-by-group, error analysis).

---

## Limitations

- **Competition dataset, not a production feed.** `application_test.csv` has
  no `TARGET` column (Kaggle withholds it for leaderboard scoring), so it is
  used only to generate a submission file and as a covariate-shift proxy for
  drift monitoring — it is not a held-out evaluation set, and the drift
  report's "current population" is a one-time snapshot, not a live stream.
- **No temporal validation.** The train/calibration/test split is a random,
  label-stratified split of a single historical snapshot. Home Credit does
  not publish application timestamps granular enough to build a genuine
  time-based (out-of-time) split, so this pipeline cannot and does not claim
  to validate how the model would perform on applications submitted after
  the training window — a real deployment would need that check before going
  live.
- **Fairness analysis is observational, not causal.** The subgroup fairness
  and error-analysis metrics (`src/models/fairness.py`) describe statistical
  disparities in model outcomes across gender, education, income type, and
  age band. They do not establish *why* those disparities exist (data
  artifacts vs. genuine risk differences vs. proxy effects), and are not a
  substitute for a fair-lending legal review.
- **The cost-based threshold's weights are illustrative**, not validated unit
  economics for this portfolio (see [Threshold Optimization](#6-threshold-optimization)).
  The default threshold metric is F1 for exactly this reason.
- **The live `/api/predict` endpoint cannot compute bureau/previous-application
  aggregate features** for an arbitrary applicant, since those require the
  applicant's full external transaction history, which a single API request
  does not carry. Those features are imputed with the training-set median at
  inference time, consistent with how any other missing value is handled —
  but it means a live score relies more heavily on the application-level
  fields than the batch-trained model's feature importance would suggest.

---

## Project Structure

```
Home_Credit/
├── asset/                          # Brand assets
│   ├── Logo_01.png
│   └── Logo_02.png
├── data/
│   ├── raw/                        # Original Kaggle CSVs (DVC tracked)
│   └── processed/                  # train/calibration/test/submission feature sets (Parquet)
├── src/
│   ├── config.py                   # Central configuration
│   ├── data/
│   │   ├── loader.py               # Raw data loading
│   │   └── splitting.py            # Three-way, label-stratified train/calibration/test split
│   ├── features/engineering.py     # 200+ feature construction + encode_and_impute
│   ├── models/
│   │   ├── train.py                # Optuna + LightGBM + SHAP + calibration + MLflow
│   │   ├── evaluate.py             # Metrics, curves, calibration
│   │   ├── fairness.py             # Subgroup fairness & error analysis
│   │   └── drift.py                # Population Stability Index (PSI)
│   └── utils/stats.py              # KS, Chi-Square, bootstrap CI, threshold selection
├── api/
│   ├── main.py                     # FastAPI application
│   ├── schemas.py                  # Pydantic request/response models
│   └── predictor.py                # Model loading & inference
├── dashboard/                      # Next.js 14 + TypeScript
│   └── src/
│       ├── app/                    # 5 pages: overview, performance, predict, statistics, responsible-ai
│       └── components/             # Sidebar, TopBar, Charts, Table, PredictionForm
├── tests/                          # pytest suite — synthetic fixtures, no real data required
├── scripts/compare_sampling.py     # class_weight vs. SMOTE comparison → reports/
├── reports/sampling_comparison.md  # Evidence behind the sampling-method default
├── models/artifacts/               # Trained model + precomputed stats
├── .github/workflows/ci.yml        # Lint (ruff) + test (pytest) on push/PR
├── pipeline.py                     # DVC-compatible 5-stage orchestrator
├── dvc.yaml                        # Reproducible pipeline definition
├── params.yaml                     # All hyperparameters & config
├── pyproject.toml                  # ruff + pytest configuration
├── Dockerfile                      # API container for Render
├── render.yaml                     # Render deployment config
├── vercel.json                     # Vercel deployment config
├── requirements.txt                # Runtime Python dependencies
├── requirements-dev.txt            # + pytest, ruff (CI/local dev)
├── LICENSE                         # MIT
└── .mailmap                        # Canonical author identity
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- Node.js 18+ (for dashboard)
- Docker (for containerized deployment)
- Kaggle account (to download the dataset)

### 0. Download the dataset
```bash
pip install kaggle
# Place your kaggle.json token in ~/.kaggle/kaggle.json
kaggle competitions download -c home-credit-default-risk -p data/raw
cd data/raw && unzip home-credit-default-risk.zip && rm home-credit-default-risk.zip
```
> See `data/raw/README.md` for manual download instructions.

### 1. Install dependencies
```bash
pip install -r requirements.txt        # runtime only
pip install -r requirements-dev.txt    # + pytest, ruff (needed for step 2)
cd dashboard && npm install
```

### 2. Run the test suite (no Kaggle download required)
```bash
ruff check .
pytest
```
The suite runs entirely on small synthetic fixtures (`tests/conftest.py`), so
it never touches `data/raw/` — this is also what `.github/workflows/ci.yml`
runs on every push/PR. See [Tests & Quality](#tests--quality) below.

### 3. Run the full ML pipeline (requires the real dataset — step 0)
```bash
# Using DVC (recommended — tracks all stages)
dvc repro

# Or manually stage by stage
python pipeline.py --stage features   # Build 200+ features, split train/calibration/test/submission
python pipeline.py --stage stats      # KS + Chi-Square analysis (training partition only)
python pipeline.py --stage train      # Optuna search + fit + calibrate + threshold + fairness/error analysis + log to MLflow
python pipeline.py --stage predict    # Generate submission file from the Kaggle holdout
python pipeline.py --stage drift      # Population Stability Index vs. the Kaggle test split
```

### 4. Start the API locally
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI: http://localhost:8000/docs
```

### 5. Start the dashboard locally
```bash
cd dashboard
npm run dev
# Dashboard: http://localhost:3000
```

### 6. Local full stack (Docker Compose)
```bash
docker-compose up
# API: http://localhost:8000
# Dashboard: http://localhost:3000
```

---

## Tests & Quality

```bash
pip install -r requirements-dev.txt
ruff check .        # lint
pytest               # tests + coverage report (pyproject.toml: --cov=src --cov=api)
```

`tests/` covers, with synthetic fixtures only (the real Kaggle CSVs are never
committed and are not required to run this suite):

| File | Covers |
|---|---|
| `test_feature_engineering.py` | Output shapes, `TARGET` never leaking into the feature list, train/submission column alignment |
| `test_no_leakage.py` | The three-way split is disjoint/stratified/reproducible; a value imputed into the calibration partition comes from the **training** median, not its own |
| `test_calibration.py` | Calibrated probabilities stay in [0, 1]; isotonic calibration doesn't grossly worsen the Brier score |
| `test_threshold_and_bootstrap.py` | F1 and cost-based threshold selection behave as expected; bootstrap CIs contain their point estimate |
| `test_api_contract.py` | `/health`, `/api/predict`, `/api/overview`, `/api/model/metrics` via FastAPI's `TestClient` — status codes, response shape, value ranges |

`.github/workflows/ci.yml` runs `ruff check .` and `pytest` on every push and
pull request to `main`.

---

## Deployment (Zero Cost)

### API → Render Free Tier
1. Push this repository to GitHub
2. Connect to [Render](https://render.com) and select **New Web Service**
3. Point to this repo — Render auto-detects `render.yaml`
4. The `Dockerfile` handles all dependencies

> **Live API:** `https://home-credit-api-gima.onrender.com` ([Swagger docs](https://home-credit-api-gima.onrender.com/docs))

### Dashboard → Vercel
1. Connect to [Vercel](https://vercel.com) and import the GitHub repository
2. Set **Root Directory** to `dashboard`
3. Add environment variable `NEXT_PUBLIC_API_URL`, set to the URL Render assigned your service once deployed (this project's own deployment uses `https://home-credit-api-gima.onrender.com`, shown above — a fork will get a different URL from Render)
4. Deploy — Vercel handles the Next.js build automatically

---

## MLflow Experiment Tracking

All training runs are tracked locally in `mlruns/`. To visualize:
```bash
mlflow ui --port 5000
# MLflow UI: http://localhost:5000
```

Tracked per run: all Optuna hyperparameters, AUC-ROC, F1, KS statistic, optimal threshold, sampling method, and the registered LightGBM model artifact.

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | KPI cards (AUC, default rate, KS), risk score distribution, SHAP importance chart, live applications table |
| **Model Performance** | ROC curve, Precision-Recall curve, probability calibration, confusion matrix, Bootstrap CI table |
| **Risk Score & Decision Support** | Real-time application scoring form with animated gauge, risk band, and recommended action for manual review |
| **Statistics** | KS test results, Chi-Square test results, target and feature distribution charts |
| **Responsible AI** | Model card, intended use & limitations, sensitive-variable/proxy disclosure, subgroup fairness (disparate impact, equal opportunity), calibration by segment, false negative/positive analysis, and drift monitoring (PSI) |

---

## Author

**Renan Pinheiro**
Data Scientist · [github.com/pinheiro-dataworks](https://github.com/pinheiro-dataworks?tab=repositories)

---

*Built with LightGBM · Optuna · SHAP · MLflow · DVC · FastAPI · Next.js · Docker · Render · Vercel*

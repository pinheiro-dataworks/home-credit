<p align="center">
  <img src="asset/Logo_02.png" alt="Home Credit Risk Intelligence" width="260" />
</p>

<h1 align="center">Home Credit Default Risk</h1>
<h3 align="center">Open-Finance Credit Risk Prediction — End-to-End MLOps Case Study</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-4.1-00B0F0?style=flat-square" />
  <img src="https://img.shields.io/badge/FastAPI-0.108-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-2.9-0194E2?style=flat-square&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/DVC-3.37-945DD6?style=flat-square&logo=dvc&logoColor=white" />
  <img src="https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Render-2496ED?style=flat-square&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Dashboard-Vercel-000000?style=flat-square&logo=vercel&logoColor=white" />
</p>

---

## Project Definition

This project addresses one of the most critical challenges in consumer lending: **predicting credit default risk for applicants who have little or no formal credit history**. Leveraging the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset — an internationally recognized benchmark in financial machine learning — this case study demonstrates a complete, production-grade MLOps pipeline from raw data ingestion to live API deployment and an interactive dashboard.

The work covers the full data science lifecycle: rigorous statistical feature analysis across eight relational tables, advanced feature engineering, imbalanced-class treatment, gradient boosting with automated hyperparameter optimization, model explainability, probability calibration, and zero-cost cloud deployment — delivering an enterprise-ready solution without any paid platform subscriptions.

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

**Key characteristic:** Severe class imbalance — only **8.07%** of applicants defaulted (1:11 ratio), requiring specialized resampling strategies.

---

## Tech Stack

### Machine Learning & Statistics
| Library | Purpose |
|---|---|
| `scikit-learn` | Preprocessing, calibration, cross-validation |
| `feature-engine` | Feature engineering pipeline components |
| `LightGBM` | Primary gradient boosting model |
| `imbalanced-learn` | SMOTE and ADASYN resampling |
| `Optuna` | Bayesian hyperparameter optimization with MedianPruner |
| `SHAP` | TreeExplainer — global and local model explainability |
| `scipy` + `statsmodels` | KS test, Chi-Square, statistical analysis |

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
All engineered features were validated using formal hypothesis tests before model inclusion:

- **Kolmogorov-Smirnov test** (numeric features): two-sample distribution comparison between DEFAULT=0 and DEFAULT=1 groups — `EXT_SOURCE_2` achieved KS=0.318, p<0.0001
- **Chi-Square test** (categorical features): independence testing between each categorical variable and TARGET — `NAME_EDUCATION_TYPE` returned χ²=1842.3, p<0.0001
- **Bootstrap confidence intervals:** 1,000 iterations to produce 95% CIs for AUC-ROC, AUC-PR, Precision, Recall, and F1

### 3. Imbalanced Class Treatment
With only 8.07% positives, three strategies were evaluated:
- **SMOTE** (Synthetic Minority Oversampling Technique) — final choice, sampling strategy 0.3
- **ADASYN** (Adaptive Synthetic Sampling) — adaptive density estimation
- **Class-weight balancing** in LightGBM — used as baseline

### 4. Hyperparameter Optimization
**Optuna** with `TPESampler` and `MedianPruner` ran 50 trials over:
`n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`

5-fold stratified cross-validation was used as the objective metric, maximizing AUC-ROC. Early stopping with 50 rounds was applied on a held-out validation set.

### 5. Probability Calibration
Post-training, **isotonic regression** calibration was applied (`CalibratedClassifierCV`, `cv='prefit'`) to ensure that a predicted score of 30% corresponds to an actual 30% default rate — critical for financial risk management applications.

### 6. Threshold Optimization
Rather than defaulting to 0.5, the optimal classification threshold was found by grid-searching [0.01, 0.99] in 200 steps, maximizing F1-score on the validation set. The optimal threshold was **0.35**, reflecting the asymmetric costs of false negatives (missed defaults) vs false positives (declined good customers).

---

## Results

| Metric | Value | Notes |
|---|---|---|
| **AUC-ROC** | 0.781 | Validation set |
| **AUC-PR** | 0.312 | Imbalanced-aware metric |
| **KS Statistic** | 0.424 | Model discrimination power |
| **F1 Score** | 0.541 | At optimal threshold (0.35) |
| **Precision** | 0.453 | At optimal threshold |
| **Recall** | 0.672 | At optimal threshold |
| **Optimal Threshold** | 0.35 | Maximizes F1 |
| **Calibration** | Isotonic | Near-perfect probability alignment |

**Top 5 Features by SHAP Importance:**
1. `EXT_SOURCE_2` — External credit score 2 (SHAP = 0.084)
2. `EXT_SOURCE_3` — External credit score 3 (SHAP = 0.072)
3. `EXT_SOURCE_1` — External credit score 1 (SHAP = 0.061)
4. `DAYS_BIRTH` — Applicant age in days (SHAP = 0.051)
5. `CREDIT_INCOME_RATIO` — Credit-to-income ratio (SHAP = 0.049)

---

## Project Structure

```
Home_Credit/
├── asset/                          # Brand assets
│   ├── Logo_01.png
│   └── Logo_02.png
├── data/
│   ├── raw/                        # Original Kaggle CSVs (DVC tracked)
│   └── processed/                  # Engineered feature sets (Parquet)
├── src/
│   ├── config.py                   # Central configuration
│   ├── data/loader.py              # Raw data loading
│   ├── features/engineering.py     # 200+ feature construction
│   ├── models/
│   │   ├── train.py                # Optuna + SMOTE + SHAP + MLflow
│   │   └── evaluate.py             # Metrics, curves, calibration
│   └── utils/stats.py              # KS, Chi-Square, Bootstrap CI
├── api/
│   ├── main.py                     # FastAPI application
│   ├── schemas.py                  # Pydantic request/response models
│   └── predictor.py                # Model loading & inference
├── dashboard/                      # Next.js 14 + TypeScript
│   └── src/
│       ├── app/                    # 4 pages: overview, performance, predict, statistics
│       └── components/             # Sidebar, TopBar, Charts, Table, PredictionForm
├── models/artifacts/               # Trained model + precomputed stats
├── pipeline.py                     # DVC-compatible 4-stage orchestrator
├── dvc.yaml                        # Reproducible pipeline definition
├── params.yaml                     # All hyperparameters & config
├── Dockerfile                      # API container for Render
├── render.yaml                     # Render deployment config
├── vercel.json                     # Vercel deployment config
└── requirements.txt                # Python dependencies
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
pip install -r requirements.txt
cd dashboard && npm install
```

### 2. Run the full ML pipeline
```bash
# Using DVC (recommended — tracks all stages)
dvc repro

# Or manually stage by stage
python pipeline.py --stage features   # Build 200+ features
python pipeline.py --stage stats      # KS + Chi-Square analysis
python pipeline.py --stage train      # Train + tune + calibrate + log to MLflow
python pipeline.py --stage predict    # Generate submission file
```

### 3. Start the API locally
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI: http://localhost:8000/docs
```

### 4. Start the dashboard locally
```bash
cd dashboard
npm run dev
# Dashboard: http://localhost:3000
```

### 5. Local full stack (Docker Compose)
```bash
docker-compose up
# API: http://localhost:8000
# Dashboard: http://localhost:3000
```

---

## Deployment (Zero Cost)

### API → Render Free Tier
1. Push this repository to GitHub
2. Connect to [Render](https://render.com) and select **New Web Service**
3. Point to this repo — Render auto-detects `render.yaml`
4. The `Dockerfile` handles all dependencies

### Dashboard → Vercel
1. Connect to [Vercel](https://vercel.com) and import the GitHub repository
2. Set **Root Directory** to `dashboard`
3. Add environment variable: `NEXT_PUBLIC_API_URL=https://<your-render-service>.onrender.com`
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
| **Live Prediction** | Real-time application scoring form with animated gauge and Approve/Deny decision |
| **Statistics** | KS test results, Chi-Square test results, target and feature distribution charts |

---

## Author

**Renan Pinhiero**
Data Scientist · [github.com/pinheiro-dataworks](https://github.com/pinheiro-dataworks?tab=repositories)

---

*Built with LightGBM · Optuna · SHAP · MLflow · DVC · FastAPI · Next.js · Docker · Render · Vercel*

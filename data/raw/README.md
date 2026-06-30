# Raw Data

The CSV files in this directory are **not tracked by Git** (they are too large and are subject to Kaggle's terms of use). Download them from the official competition page before running the pipeline.

## Download Instructions

1. Create a free account at [kaggle.com](https://www.kaggle.com) if you don't have one.
2. Accept the competition terms at:  
   **Home Credit Default Risk** → `https://www.kaggle.com/c/home-credit-default-risk`
3. Download all files from the **Data** tab and place them in this directory.

### Required files

| File | Size (approx.) |
|---|---|
| `application_train.csv` | 166 MB |
| `application_test.csv` | 27 MB |
| `bureau.csv` | 218 MB |
| `bureau_balance.csv` | 1.1 GB |
| `previous_application.csv` | 253 MB |
| `POS_CASH_balance.csv` | 378 MB |
| `credit_card_balance.csv` | 141 MB |
| `installments_payments.csv` | 430 MB |
| `HomeCredit_columns_description.csv` | < 1 MB |
| `sample_submission.csv` | < 1 MB |

### Using the Kaggle CLI (recommended)

```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/kaggle.json
kaggle competitions download -c home-credit-default-risk -p data/raw
cd data/raw && unzip home-credit-default-risk.zip && rm home-credit-default-risk.zip
```

After placing all files here, run the pipeline from the project root:

```bash
python pipeline.py --stage features
```

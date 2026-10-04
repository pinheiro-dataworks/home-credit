"""Shared synthetic fixtures. The real Kaggle dataset is never committed to the
repository, so the test suite builds small synthetic tables that mirror the
Home Credit schema closely enough to exercise every feature-engineering
function without touching real data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N_TRAIN = 300
N_TEST  = 60
SEED    = 42


def _synthetic_application(n: int, rng: np.random.Generator, sk_id_start: int) -> pd.DataFrame:
    sk_ids = np.arange(sk_id_start, sk_id_start + n)
    df = pd.DataFrame({
        "SK_ID_CURR":           sk_ids,
        "AMT_INCOME_TOTAL":     rng.uniform(40_000, 300_000, n),
        "AMT_CREDIT":           rng.uniform(80_000, 600_000, n),
        "AMT_ANNUITY":          rng.uniform(5_000, 40_000, n),
        "AMT_GOODS_PRICE":      rng.uniform(70_000, 550_000, n),
        "CNT_FAM_MEMBERS":      rng.integers(1, 6, n).astype(float),
        "CNT_CHILDREN":         rng.integers(0, 3, n),
        "DAYS_BIRTH":           -rng.integers(7_300, 25_550, n),
        "DAYS_EMPLOYED":        -rng.integers(0, 15_000, n),
        "DAYS_REGISTRATION":    -rng.integers(0, 10_000, n).astype(float),
        "DAYS_ID_PUBLISH":      -rng.integers(0, 6_000, n),
        "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 4_000, n).astype(float),
        "EXT_SOURCE_1":         rng.uniform(0, 1, n),
        "EXT_SOURCE_2":         rng.uniform(0, 1, n),
        "EXT_SOURCE_3":         rng.uniform(0, 1, n),
        "FLAG_DOCUMENT_3":      rng.integers(0, 2, n),
        "FLAG_MOBIL":           1,
        "FLAG_EMP_PHONE":       rng.integers(0, 2, n),
        "FLAG_WORK_PHONE":      rng.integers(0, 2, n),
        "FLAG_CONT_MOBILE":     1,
        "FLAG_PHONE":           rng.integers(0, 2, n),
        "FLAG_EMAIL":           rng.integers(0, 2, n),
        "AMT_REQ_CREDIT_BUREAU_YEAR": rng.integers(0, 5, n),
        "REGION_RATING_CLIENT": rng.integers(1, 4, n),
        "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.07, n),
        "NAME_CONTRACT_TYPE":   rng.choice(["Cash loans", "Revolving loans"], n),
        "CODE_GENDER":          rng.choice(["M", "F"], n),
        "FLAG_OWN_CAR":         rng.choice(["Y", "N"], n),
        "FLAG_OWN_REALTY":      rng.choice(["Y", "N"], n),
        "NAME_INCOME_TYPE":     rng.choice(["Working", "Pensioner", "Commercial associate"], n),
        "NAME_EDUCATION_TYPE":  rng.choice(["Secondary / secondary special", "Higher education"], n),
        "NAME_FAMILY_STATUS":   rng.choice(["Married", "Single / not married"], n),
        "NAME_HOUSING_TYPE":    rng.choice(["House / apartment", "With parents"], n),
    })
    # A handful of DAYS_EMPLOYED == 365243 sentinel rows ("unemployed/pensioner"),
    # same convention as the real dataset.
    sentinel_rows = rng.choice(n, size=max(1, n // 20), replace=False)
    df.loc[sentinel_rows, "DAYS_EMPLOYED"] = 365243
    return df


def _synthetic_bureau(sk_id_curr: np.ndarray, rng: np.random.Generator):
    rows_per_app = rng.integers(0, 4, len(sk_id_curr))
    sk_curr_col, sk_bureau_col = [], []
    next_bureau_id = 1
    for curr_id, n_rows in zip(sk_id_curr, rows_per_app):
        for _ in range(n_rows):
            sk_curr_col.append(curr_id)
            sk_bureau_col.append(next_bureau_id)
            next_bureau_id += 1
    n = len(sk_curr_col)
    bureau = pd.DataFrame({
        "SK_ID_CURR":            sk_curr_col,
        "SK_ID_BUREAU":          sk_bureau_col,
        "CREDIT_ACTIVE":         rng.choice(["Active", "Closed", "Bad debt"], n, p=[0.4, 0.55, 0.05]) if n else [],
        "DAYS_CREDIT":           -rng.integers(0, 3000, n),
        "CREDIT_DAY_OVERDUE":    rng.integers(0, 30, n),
        "DAYS_CREDIT_ENDDATE":   -rng.integers(-1000, 2000, n),
        "AMT_CREDIT_MAX_OVERDUE": rng.uniform(0, 5000, n),
        "CNT_CREDIT_PROLONG":    rng.integers(0, 2, n),
        "AMT_CREDIT_SUM":        rng.uniform(1000, 200000, n),
        "AMT_CREDIT_SUM_DEBT":   rng.uniform(0, 100000, n),
        "AMT_CREDIT_SUM_LIMIT":  rng.uniform(0, 50000, n),
        "AMT_CREDIT_SUM_OVERDUE": rng.uniform(0, 1000, n),
        "DAYS_CREDIT_UPDATE":    -rng.integers(0, 1000, n),
        "AMT_ANNUITY":           rng.uniform(0, 20000, n),
    })
    bb_rows_bureau, bb_months, bb_status = [], [], []
    for bureau_id in sk_bureau_col:
        n_months = rng.integers(1, 6)
        for m in range(n_months):
            bb_rows_bureau.append(bureau_id)
            bb_months.append(-m)
            bb_status.append(rng.choice(["C", "X", "0", "1", "2"]))
    bureau_bal = pd.DataFrame({
        "SK_ID_BUREAU":    bb_rows_bureau,
        "MONTHS_BALANCE":  bb_months,
        "STATUS":          bb_status,
    })
    return bureau, bureau_bal


def _synthetic_previous(sk_id_curr: np.ndarray, rng: np.random.Generator):
    rows_per_app = rng.integers(0, 3, len(sk_id_curr))
    sk_curr_col, sk_prev_col = [], []
    next_prev_id = 1
    for curr_id, n_rows in zip(sk_id_curr, rows_per_app):
        for _ in range(n_rows):
            sk_curr_col.append(curr_id)
            sk_prev_col.append(next_prev_id)
            next_prev_id += 1
    n = len(sk_curr_col)
    return pd.DataFrame({
        "SK_ID_CURR":            sk_curr_col,
        "SK_ID_PREV":            sk_prev_col,
        "NAME_CONTRACT_STATUS":  rng.choice(["Approved", "Refused", "Canceled"], n, p=[0.7, 0.2, 0.1]) if n else [],
        "AMT_APPLICATION":       rng.uniform(10000, 400000, n),
        "AMT_CREDIT":            rng.uniform(10000, 400000, n),
        "AMT_ANNUITY":           rng.uniform(1000, 30000, n),
        "AMT_DOWN_PAYMENT":      rng.uniform(0, 50000, n),
        "DAYS_DECISION":         -rng.integers(0, 2000, n),
        "RATE_INTEREST_PRIMARY": rng.uniform(0.05, 0.3, n),
        "DAYS_FIRST_DRAWING":    -rng.integers(0, 2000, n),
    }), sk_prev_col


def _synthetic_monthly(sk_id_curr: np.ndarray, sk_id_prev: list, rng: np.random.Generator, extra_cols: dict):
    sk_curr_col, sk_prev_col, months = [], [], []
    curr_by_prev = dict(zip(sk_id_prev, np.random.choice(sk_id_curr, size=len(sk_id_prev)))) if sk_id_prev else {}
    for prev_id in sk_id_prev:
        n_months = rng.integers(1, 5)
        for m in range(n_months):
            sk_prev_col.append(prev_id)
            sk_curr_col.append(curr_by_prev[prev_id])
            months.append(-m)
    n = len(sk_prev_col)
    base = {"SK_ID_CURR": sk_curr_col, "SK_ID_PREV": sk_prev_col, "MONTHS_BALANCE": months}
    for col, gen in extra_cols.items():
        base[col] = gen(n, rng)
    return pd.DataFrame(base)


@pytest.fixture
def raw_tables() -> dict[str, pd.DataFrame]:
    """Synthetic replica of the 8 Home Credit raw tables, keyed exactly like
    `src.data.loader.load_all()`'s return value."""
    rng = np.random.default_rng(SEED)

    train_app = _synthetic_application(N_TRAIN, rng, sk_id_start=100_000)
    train_app["TARGET"] = rng.choice([0, 1], N_TRAIN, p=[0.92, 0.08])
    test_app = _synthetic_application(N_TEST, rng, sk_id_start=200_000)

    all_sk_curr = np.concatenate([train_app["SK_ID_CURR"].values, test_app["SK_ID_CURR"].values])
    bureau, bureau_bal = _synthetic_bureau(all_sk_curr, rng)
    prev, sk_id_prev = _synthetic_previous(all_sk_curr, rng)

    pos = _synthetic_monthly(all_sk_curr, sk_id_prev, rng, {
        "NAME_CONTRACT_STATUS": lambda n, r: r.choice(["Active", "Completed"], n),
        "SK_DPD":               lambda n, r: r.integers(0, 30, n),
        "SK_DPD_DEF":           lambda n, r: r.integers(0, 10, n),
        "CNT_INSTALMENT":       lambda n, r: r.integers(1, 48, n),
        "CNT_INSTALMENT_FUTURE": lambda n, r: r.integers(0, 48, n),
    })
    cc = _synthetic_monthly(all_sk_curr, sk_id_prev, rng, {
        "AMT_BALANCE":               lambda n, r: r.uniform(0, 50000, n),
        "AMT_CREDIT_LIMIT_ACTUAL":   lambda n, r: r.uniform(10000, 100000, n),
        "AMT_DRAWINGS_CURRENT":      lambda n, r: r.uniform(0, 20000, n),
        "AMT_DRAWINGS_ATM_CURRENT":  lambda n, r: r.uniform(0, 10000, n),
        "AMT_PAYMENT_CURRENT":       lambda n, r: r.uniform(0, 20000, n),
        "AMT_INST_MIN_REGULARITY":   lambda n, r: r.uniform(100, 5000, n),
        "SK_DPD":                    lambda n, r: r.integers(0, 30, n),
        "SK_DPD_DEF":                lambda n, r: r.integers(0, 10, n),
    })
    inst = _synthetic_monthly(all_sk_curr, sk_id_prev, rng, {
        "DAYS_ENTRY_PAYMENT": lambda n, r: -r.integers(0, 2000, n).astype(float),
        "DAYS_INSTALMENT":    lambda n, r: -r.integers(0, 2000, n).astype(float),
        "AMT_PAYMENT":        lambda n, r: r.uniform(500, 30000, n),
        "AMT_INSTALMENT":     lambda n, r: r.uniform(500, 30000, n),
    })

    return {
        "train":        train_app,
        "test":         test_app,
        "bureau":       bureau,
        "bureau_bal":   bureau_bal,
        "prev":         prev,
        "pos_cash":     pos,
        "credit_card":  cc,
        "installments": inst,
    }

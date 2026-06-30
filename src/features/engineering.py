"""
Feature engineering pipeline — creates 200+ features from all 8 raw tables.
Each function is independently testable and returns a DataFrame indexed on SK_ID_CURR.
"""
import logging
import numpy as np
import pandas as pd
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _agg(df: pd.DataFrame, group_col: str, prefix: str, cols: list[str],
         funcs: list[str] = ("mean", "max", "min", "sum", "std")) -> pd.DataFrame:
    """Generic aggregation helper; returns DataFrame indexed on group_col."""
    result = df.groupby(group_col)[cols].agg(list(funcs))
    result.columns = [f"{prefix}_{c}_{f}".upper() for c, f in result.columns]
    return result.reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# Application (main table)
# ──────────────────────────────────────────────────────────────────────────────

def engineer_application(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction and ratio features from the application table."""
    out = df.copy()

    # --- Credit ratios ---------------------------------------------------------
    out["CREDIT_INCOME_RATIO"]     = out["AMT_CREDIT"]   / (out["AMT_INCOME_TOTAL"] + 1)
    out["ANNUITY_INCOME_RATIO"]    = out["AMT_ANNUITY"]  / (out["AMT_INCOME_TOTAL"] + 1)
    out["CREDIT_TERM"]             = out["AMT_CREDIT"]   / (out["AMT_ANNUITY"]      + 1)
    out["GOODS_CREDIT_RATIO"]      = out["AMT_GOODS_PRICE"] / (out["AMT_CREDIT"]    + 1)
    out["ANNUITY_CREDIT_RATIO"]    = out["AMT_ANNUITY"]  / (out["AMT_CREDIT"]       + 1)
    out["INCOME_PER_PERSON"]       = out["AMT_INCOME_TOTAL"] / (out["CNT_FAM_MEMBERS"] + 1)

    # --- Age & employment ------------------------------------------------------
    out["AGE_YEARS"]               = -out["DAYS_BIRTH"]     / 365
    out["EMPLOYED_YEARS"]          = (-out["DAYS_EMPLOYED"]).clip(lower=0) / 365
    out["REGISTRATION_YEARS"]      = -out["DAYS_REGISTRATION"] / 365
    out["ID_PUBLISH_YEARS"]        = -out["DAYS_ID_PUBLISH"]  / 365
    out["PHONE_CHANGE_YEARS"]      = -out["DAYS_LAST_PHONE_CHANGE"] / 365

    # Unemployment flag (DAYS_EMPLOYED == 365243 encodes "unemployed / pensioner")
    out["IS_UNEMPLOYED"]           = (out["DAYS_EMPLOYED"] == 365243).astype(int)
    out["DAYS_EMPLOYED"]           = out["DAYS_EMPLOYED"].replace(365243, np.nan)
    out["EMPLOYED_YEARS"]          = (-out["DAYS_EMPLOYED"]).clip(lower=0) / 365

    out["EMPLOYED_TO_AGE_RATIO"]   = out["EMPLOYED_YEARS"] / (out["AGE_YEARS"] + 1)
    out["CREDIT_TO_AGE_RATIO"]     = out["AMT_CREDIT"]     / (out["AGE_YEARS"] + 1)

    # --- External scores -------------------------------------------------------
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    out["EXT_SOURCE_MEAN"]         = out[ext_cols].mean(axis=1)
    out["EXT_SOURCE_STD"]          = out[ext_cols].std(axis=1)
    out["EXT_SOURCE_MIN"]          = out[ext_cols].min(axis=1)
    out["EXT_SOURCE_MAX"]          = out[ext_cols].max(axis=1)
    out["EXT_SOURCE_WEIGHTED"]     = (
        out["EXT_SOURCE_1"].fillna(0) * 2
        + out["EXT_SOURCE_2"].fillna(0) * 3
        + out["EXT_SOURCE_3"].fillna(0) * 1
    ) / 6
    out["EXT_SRC_1x2"]            = out["EXT_SOURCE_1"] * out["EXT_SOURCE_2"]
    out["EXT_SRC_2x3"]            = out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"]
    out["EXT_SRC_1x3"]            = out["EXT_SOURCE_1"] * out["EXT_SOURCE_3"]

    # --- Document flags --------------------------------------------------------
    doc_cols   = [c for c in out.columns if c.startswith("FLAG_DOCUMENT_")]
    out["DOCUMENT_COUNT"]          = out[doc_cols].sum(axis=1)
    flag_cols  = ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                  "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
    flag_cols  = [c for c in flag_cols if c in out.columns]
    out["CONTACT_COUNT"]           = out[flag_cols].sum(axis=1)

    # --- Bureau enquiry sum ----------------------------------------------------
    req_cols = [c for c in out.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if req_cols:
        out["AMT_REQ_BUREAU_TOTAL"] = out[req_cols].sum(axis=1)

    # --- Children & family -----------------------------------------------------
    out["CHILDREN_RATIO"]          = out["CNT_CHILDREN"] / (out["CNT_FAM_MEMBERS"] + 1)
    out["NON_CHILDREN"]            = out["CNT_FAM_MEMBERS"] - out["CNT_CHILDREN"]

    # --- Region ratings sum ----------------------------------------------------
    region_cols = [c for c in out.columns if "REGION_RATING" in c]
    if region_cols:
        out["REGION_RATING_SUM"]   = out[region_cols].sum(axis=1)

    logger.info("Application features: %d columns created.", out.shape[1])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Bureau
# ──────────────────────────────────────────────────────────────────────────────

def engineer_bureau(bureau: pd.DataFrame, bureau_bal: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau + bureau_balance into per-applicant features."""

    # --- Bureau balance: status distribution per SK_ID_BUREAU -----------------
    status_map = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    bureau_bal["STATUS_NUMERIC"] = bureau_bal["STATUS"].map(status_map).fillna(0)
    bureau_bal["IS_OVERDUE"]     = bureau_bal["STATUS"].isin(["1","2","3","4","5"]).astype(int)
    bureau_bal["IS_CLOSED"]      = (bureau_bal["STATUS"] == "C").astype(int)

    bb_agg = bureau_bal.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT    = ("MONTHS_BALANCE", "count"),
        BB_STATUS_MEAN     = ("STATUS_NUMERIC", "mean"),
        BB_OVERDUE_RATIO   = ("IS_OVERDUE", "mean"),
        BB_CLOSED_RATIO    = ("IS_CLOSED", "mean"),
    ).reset_index()

    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    # --- Active vs closed credits ---------------------------------------------
    bureau["IS_ACTIVE"]  = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    bureau["IS_CLOSED_B"]= (bureau["CREDIT_ACTIVE"] == "Closed").astype(int)
    bureau["IS_BAD"]     = (bureau["CREDIT_ACTIVE"] == "Bad debt").astype(int)

    # --- Aggregate per applicant -----------------------------------------------
    num_cols = [
        "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE",
        "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE",
        "DAYS_CREDIT_UPDATE", "AMT_ANNUITY",
        "BB_MONTHS_COUNT", "BB_STATUS_MEAN", "BB_OVERDUE_RATIO", "BB_CLOSED_RATIO",
    ]
    num_cols = [c for c in num_cols if c in bureau.columns]

    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT        = ("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT      = ("IS_ACTIVE", "sum"),
        BUREAU_CLOSED_COUNT      = ("IS_CLOSED_B", "sum"),
        BUREAU_BAD_COUNT         = ("IS_BAD", "sum"),
        BUREAU_ACTIVE_RATIO      = ("IS_ACTIVE", "mean"),
        BUREAU_DAYS_CREDIT_MEAN  = ("DAYS_CREDIT", "mean"),
        BUREAU_DAYS_CREDIT_MIN   = ("DAYS_CREDIT", "min"),
        BUREAU_OVERDUE_MEAN      = ("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_OVERDUE_MAX       = ("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_AMT_SUM_MEAN      = ("AMT_CREDIT_SUM", "mean"),
        BUREAU_AMT_SUM_MAX       = ("AMT_CREDIT_SUM", "max"),
        BUREAU_AMT_DEBT_MEAN     = ("AMT_CREDIT_SUM_DEBT", "mean"),
        BUREAU_AMT_OVERDUE_MEAN  = ("AMT_CREDIT_SUM_OVERDUE", "mean"),
        BUREAU_BB_OVERDUE_RATIO  = ("BB_OVERDUE_RATIO", "mean"),
        BUREAU_BB_STATUS_MEAN    = ("BB_STATUS_MEAN", "mean"),
    ).reset_index()

    agg["BUREAU_BAD_RATIO"] = agg["BUREAU_BAD_COUNT"] / (agg["BUREAU_LOAN_COUNT"] + 1)
    logger.info("Bureau features: %d columns.", agg.shape[1])
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Previous applications
# ──────────────────────────────────────────────────────────────────────────────

def engineer_previous(prev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous application features per applicant."""
    prev = prev.copy()
    prev["AMT_CREDIT"].replace(0, np.nan, inplace=True)
    prev["IS_APPROVED"]   = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["IS_REFUSED"]    = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    prev["IS_CANCELLED"]  = (prev["NAME_CONTRACT_STATUS"] == "Canceled").astype(int)
    prev["CREDIT_RATIO"]  = prev["AMT_APPLICATION"] / (prev["AMT_CREDIT"] + 1)
    prev["DOWN_RATIO"]    = prev["AMT_DOWN_PAYMENT"] / (prev["AMT_APPLICATION"] + 1)

    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_COUNT              = ("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT     = ("IS_APPROVED", "sum"),
        PREV_REFUSED_COUNT      = ("IS_REFUSED", "sum"),
        PREV_CANCELLED_COUNT    = ("IS_CANCELLED", "sum"),
        PREV_APPROVED_RATIO     = ("IS_APPROVED", "mean"),
        PREV_REFUSED_RATIO      = ("IS_REFUSED", "mean"),
        PREV_AMT_APPLICATION_MEAN = ("AMT_APPLICATION", "mean"),
        PREV_AMT_APPLICATION_MAX  = ("AMT_APPLICATION", "max"),
        PREV_AMT_CREDIT_MEAN    = ("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_MAX     = ("AMT_CREDIT", "max"),
        PREV_AMT_ANNUITY_MEAN   = ("AMT_ANNUITY", "mean"),
        PREV_AMT_DOWN_MEAN      = ("AMT_DOWN_PAYMENT", "mean"),
        PREV_DAYS_DECISION_MEAN = ("DAYS_DECISION", "mean"),
        PREV_DAYS_DECISION_MIN  = ("DAYS_DECISION", "min"),
        PREV_CREDIT_RATIO_MEAN  = ("CREDIT_RATIO", "mean"),
        PREV_DOWN_RATIO_MEAN    = ("DOWN_RATIO", "mean"),
        PREV_RATE_INTEREST_MEAN = ("RATE_INTEREST_PRIMARY", "mean"),
        PREV_DAYS_FIRST_DRAWING = ("DAYS_FIRST_DRAWING", "mean"),
    ).reset_index()

    # Recency: most recent previous application
    prev_recent = (
        prev.sort_values("DAYS_DECISION", ascending=False)
        .groupby("SK_ID_CURR")
        .first()[["AMT_APPLICATION", "AMT_CREDIT", "NAME_CONTRACT_STATUS"]]
        .rename(columns=lambda c: f"PREV_RECENT_{c}")
        .reset_index()
    )
    agg = agg.merge(prev_recent, on="SK_ID_CURR", how="left")
    agg["PREV_RECENT_NAME_CONTRACT_STATUS"] = (
        agg["PREV_RECENT_NAME_CONTRACT_STATUS"].map(
            {"Approved": 1, "Refused": 0, "Canceled": 0, "Unused offer": 0}
        ).fillna(0)
    )
    logger.info("Previous-application features: %d columns.", agg.shape[1])
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# POS CASH balance
# ──────────────────────────────────────────────────────────────────────────────

def engineer_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    pos = pos.copy()
    pos["IS_COMPLETED"] = (pos["NAME_CONTRACT_STATUS"] == "Completed").astype(int)
    pos["IS_ACTIVE"]    = (pos["NAME_CONTRACT_STATUS"] == "Active").astype(int)
    pos["SK_DPD_POS"]   = pos["SK_DPD"].clip(lower=0)

    agg = pos.groupby("SK_ID_CURR").agg(
        POS_COUNT              = ("SK_ID_PREV", "count"),
        POS_COMPLETED_RATIO    = ("IS_COMPLETED", "mean"),
        POS_MONTHS_MEAN        = ("MONTHS_BALANCE", "mean"),
        POS_MONTHS_MIN         = ("MONTHS_BALANCE", "min"),
        POS_CNT_INSTALMENT_MEAN= ("CNT_INSTALMENT", "mean"),
        POS_CNT_FUTURE_MEAN    = ("CNT_INSTALMENT_FUTURE", "mean"),
        POS_DPD_MEAN           = ("SK_DPD_POS", "mean"),
        POS_DPD_MAX            = ("SK_DPD_POS", "max"),
        POS_DPD_DEF_MEAN       = ("SK_DPD_DEF", "mean"),
        POS_DPD_DEF_MAX        = ("SK_DPD_DEF", "max"),
    ).reset_index()

    logger.info("POS-CASH features: %d columns.", agg.shape[1])
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Credit card balance
# ──────────────────────────────────────────────────────────────────────────────

def engineer_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    cc = cc.copy()
    cc["DRAWINGS_RATIO"]     = cc["AMT_DRAWINGS_CURRENT"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["ATM_DRAWINGS_RATIO"] = cc["AMT_DRAWINGS_ATM_CURRENT"] / (cc["AMT_DRAWINGS_CURRENT"] + 1)
    cc["BALANCE_LIMIT_RATIO"]= cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["PAYMENT_RATIO"]      = cc["AMT_PAYMENT_CURRENT"] / (cc["AMT_INST_MIN_REGULARITY"] + 1)
    cc["SK_DPD_CC"]          = cc["SK_DPD"].clip(lower=0)

    agg = cc.groupby("SK_ID_CURR").agg(
        CC_COUNT              = ("SK_ID_PREV", "count"),
        CC_AMT_BALANCE_MEAN   = ("AMT_BALANCE", "mean"),
        CC_AMT_BALANCE_MAX    = ("AMT_BALANCE", "max"),
        CC_LIMIT_MEAN         = ("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_DRAWINGS_MEAN      = ("AMT_DRAWINGS_CURRENT", "mean"),
        CC_DRAWINGS_RATIO_MEAN= ("DRAWINGS_RATIO", "mean"),
        CC_BALANCE_LIMIT_MEAN = ("BALANCE_LIMIT_RATIO", "mean"),
        CC_PAYMENT_RATIO_MEAN = ("PAYMENT_RATIO", "mean"),
        CC_DPD_MEAN           = ("SK_DPD_CC", "mean"),
        CC_DPD_MAX            = ("SK_DPD_CC", "max"),
        CC_DPD_DEF_MEAN       = ("SK_DPD_DEF", "mean"),
        CC_DPD_DEF_MAX        = ("SK_DPD_DEF", "max"),
    ).reset_index()

    logger.info("Credit-card features: %d columns.", agg.shape[1])
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Installments payments
# ──────────────────────────────────────────────────────────────────────────────

def engineer_installments(inst: pd.DataFrame) -> pd.DataFrame:
    inst = inst.copy()
    inst["DPD"]            = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    inst["DBD"]            = (inst["DAYS_INSTALMENT"] - inst["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    inst["PAYMENT_RATIO"]  = inst["AMT_PAYMENT"] / (inst["AMT_INSTALMENT"] + 1)
    inst["PAYMENT_DIFF"]   = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    inst["IS_LATE"]        = (inst["DPD"] > 0).astype(int)

    agg = inst.groupby("SK_ID_CURR").agg(
        INSTAL_COUNT          = ("SK_ID_PREV", "count"),
        INSTAL_DPD_MEAN       = ("DPD", "mean"),
        INSTAL_DPD_MAX        = ("DPD", "max"),
        INSTAL_DPD_STD        = ("DPD", "std"),
        INSTAL_DBD_MEAN       = ("DBD", "mean"),
        INSTAL_DBD_MAX        = ("DBD", "max"),
        INSTAL_PAYMENT_RATIO_MEAN = ("PAYMENT_RATIO", "mean"),
        INSTAL_PAYMENT_RATIO_MIN  = ("PAYMENT_RATIO", "min"),
        INSTAL_PAYMENT_DIFF_MEAN  = ("PAYMENT_DIFF", "mean"),
        INSTAL_PAYMENT_DIFF_MAX   = ("PAYMENT_DIFF", "max"),
        INSTAL_LATE_RATIO     = ("IS_LATE", "mean"),
        INSTAL_AMT_PAYMENT_MEAN   = ("AMT_PAYMENT", "mean"),
        INSTAL_AMT_PAYMENT_SUM    = ("AMT_PAYMENT", "sum"),
    ).reset_index()

    logger.info("Installments features: %d columns.", agg.shape[1])
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Encoding & imputation
# ──────────────────────────────────────────────────────────────────────────────

def encode_and_impute(df: pd.DataFrame, fit: bool = True,
                      _state: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categoricals and median-impute numerics.
    Returns (transformed_df, state_dict) where state_dict stores encoders.
    """
    if _state is None:
        _state = {}

    out = df.copy()
    cat_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()

    if fit:
        _state["label_encoders"] = {}
        for col in cat_cols:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str))
            _state["label_encoders"][col] = le
        # Median imputer for all remaining numeric NaN
        num_cols = out.select_dtypes(include=["number"]).columns.tolist()
        _state["medians"] = out[num_cols].median()
        out[num_cols] = out[num_cols].fillna(_state["medians"])
    else:
        for col in cat_cols:
            le = _state["label_encoders"].get(col)
            if le:
                known = set(le.classes_)
                out[col] = out[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                out[col] = le.transform(out[col])
            else:
                out[col] = 0
        num_cols = out.select_dtypes(include=["number"]).columns.tolist()
        out[num_cols] = out[num_cols].fillna(_state["medians"])

    return out, _state


# ──────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_features(
    app: pd.DataFrame,
    bureau: pd.DataFrame,
    bureau_bal: pd.DataFrame,
    prev: pd.DataFrame,
    pos: pd.DataFrame,
    cc: pd.DataFrame,
    inst: pd.DataFrame,
    nan_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Full feature construction pipeline.
    Returns a DataFrame with SK_ID_CURR as first column.
    """
    logger.info("Building features …")

    df = engineer_application(app)

    # Join auxiliary features
    for aux_df, join_key in [
        (engineer_bureau(bureau, bureau_bal), "SK_ID_CURR"),
        (engineer_previous(prev),              "SK_ID_CURR"),
        (engineer_pos_cash(pos),               "SK_ID_CURR"),
        (engineer_credit_card(cc),             "SK_ID_CURR"),
        (engineer_installments(inst),          "SK_ID_CURR"),
    ]:
        df = df.merge(aux_df, on=join_key, how="left")

    # Drop columns exceeding NaN threshold
    nan_frac = df.isnull().mean()
    drop_cols = nan_frac[nan_frac > nan_threshold].index.tolist()
    df = df.drop(columns=drop_cols)
    logger.info("Dropped %d high-NaN columns. Final shape: %s", len(drop_cols), df.shape)

    return df

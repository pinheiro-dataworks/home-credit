"""Raw data loading utilities for the Home Credit Default Risk dataset."""
import logging
import pandas as pd
from ..config import DATA_RAW

logger = logging.getLogger(__name__)

_FILES = {
    "train":        "application_train.csv",
    "test":         "application_test.csv",
    "bureau":       "bureau.csv",
    "bureau_bal":   "bureau_balance.csv",
    "prev":         "previous_application.csv",
    "pos_cash":     "POS_CASH_balance.csv",
    "credit_card":  "credit_card_balance.csv",
    "installments": "installments_payments.csv",
    "description":  "HomeCredit_columns_description.csv",
    "submission":   "sample_submission.csv",
}


def _read(name: str, **kwargs) -> pd.DataFrame:
    path = DATA_RAW / _FILES[name]
    df = pd.read_csv(path, **kwargs)
    logger.info("Loaded %-15s → %s", name, df.shape)
    return df


def load_application(train: bool = True) -> pd.DataFrame:
    return _read("train" if train else "test")


def load_bureau() -> pd.DataFrame:
    return _read("bureau")


def load_bureau_balance() -> pd.DataFrame:
    return _read("bureau_bal")


def load_previous_application() -> pd.DataFrame:
    return _read("prev")


def load_pos_cash() -> pd.DataFrame:
    return _read("pos_cash")


def load_credit_card() -> pd.DataFrame:
    return _read("credit_card")


def load_installments() -> pd.DataFrame:
    return _read("installments")


def load_description() -> pd.DataFrame:
    return _read("description", encoding="unicode_escape")


def load_all() -> dict[str, pd.DataFrame]:
    """Return all raw tables as a dict keyed by table name."""
    return {
        "train":        load_application(train=True),
        "test":         load_application(train=False),
        "bureau":       load_bureau(),
        "bureau_bal":   load_bureau_balance(),
        "prev":         load_previous_application(),
        "pos_cash":     load_pos_cash(),
        "credit_card":  load_credit_card(),
        "installments": load_installments(),
    }

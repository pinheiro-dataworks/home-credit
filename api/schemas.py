"""Pydantic request / response schemas for the Home Credit Risk API."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class ApplicationInput(BaseModel):
    """Subset of the most impactful features for real-time risk prediction."""
    AMT_INCOME_TOTAL:       float = Field(..., gt=0, example=135000.0)
    AMT_CREDIT:             float = Field(..., gt=0, example=406597.5)
    AMT_ANNUITY:            float = Field(..., gt=0, example=20560.0)
    AMT_GOODS_PRICE:        Optional[float] = Field(None, example=351000.0)
    DAYS_BIRTH:             int   = Field(..., lt=0, example=-9461)
    DAYS_EMPLOYED:          int   = Field(...,       example=-637)
    EXT_SOURCE_1:           Optional[float] = Field(None, ge=0, le=1, example=0.502)
    EXT_SOURCE_2:           Optional[float] = Field(None, ge=0, le=1, example=0.626)
    EXT_SOURCE_3:           Optional[float] = Field(None, ge=0, le=1, example=0.555)
    CNT_CHILDREN:           int   = Field(0,  ge=0, example=0)
    CNT_FAM_MEMBERS:        float = Field(2.0, gt=0, example=2.0)
    NAME_CONTRACT_TYPE:     str   = Field("Cash loans", example="Cash loans")
    CODE_GENDER:            str   = Field("M", example="M")
    FLAG_OWN_CAR:           str   = Field("N", example="N")
    FLAG_OWN_REALTY:        str   = Field("Y", example="Y")
    NAME_INCOME_TYPE:       str   = Field("Working", example="Working")
    NAME_EDUCATION_TYPE:    str   = Field("Secondary / secondary special",
                                          example="Secondary / secondary special")
    NAME_FAMILY_STATUS:     str   = Field("Married", example="Married")
    NAME_HOUSING_TYPE:      str   = Field("House / apartment", example="House / apartment")
    REGION_POPULATION_RELATIVE: Optional[float] = Field(None, example=0.0181)
    DAYS_REGISTRATION:      Optional[float] = Field(None, example=-3648.0)


class PredictionResponse(BaseModel):
    risk_score:         float
    risk_label:         str            # Low | Medium | High
    default_probability:float
    threshold:          float
    predicted_default:  bool
    risk_percentile:    Optional[float] = None


class OverviewResponse(BaseModel):
    n_train:            int
    n_test:             int
    n_features:         int
    default_rate:       float
    default_count:      int
    non_default_count:  int
    n_engineered_features: int


class MetricsResponse(BaseModel):
    auc_roc:      float
    auc_pr:       float
    precision:    float
    recall:       float
    f1:           float
    ks_statistic: float
    threshold:    float
    bootstrap_ci: dict


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    version:      str = "1.0.0"

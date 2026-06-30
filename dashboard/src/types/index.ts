export interface Overview {
  n_train: number;
  n_test: number;
  n_features: number;
  default_rate: number;
  default_count: number;
  non_default_count: number;
  n_engineered_features: number;
}

export interface Metrics {
  auc_roc: number;
  auc_pr: number;
  precision: number;
  recall: number;
  f1: number;
  ks_statistic: number;
  threshold: number;
  bootstrap_ci: Record<string, { point: number; lower: number; upper: number; ci: number }>;
}

export interface RocCurve {
  fpr: number[];
  tpr: number[];
}

export interface PrCurve {
  recall: number[];
  precision: number[];
}

export interface FeatureImportance {
  feature: string;
  shap_importance: number;
}

export interface ScoreDistribution {
  bins: number[];
  non_default: number[];
  default: number[];
}

export interface Application {
  id: string;
  income: number;
  credit: number;
  risk_score: number;
  risk_label: "Low" | "Medium" | "High";
  predicted_default: boolean;
  contract_type: string;
  submitted_at: string;
}

export interface PredictionResult {
  risk_score: number;
  risk_label: string;
  default_probability: number;
  threshold: number;
  predicted_default: boolean;
}

export type FilterPeriod      = "7d" | "30d" | "90d" | "1y" | "all";
export type FilterContractType = "all" | "Cash loans" | "Revolving loans";
export type FilterRiskLevel    = "all" | "Low" | "Medium" | "High";
export type FilterGender       = "all" | "M" | "F";

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

// ── Responsible AI ────────────────────────────────────────────────────────────

export interface CalibrationCurve {
  mean_predicted: number[];
  fraction_pos: number[];
}

export interface FairnessGroup {
  n: number;
  base_rate: number;
  selection_rate: number;
  tpr: number | null;
  fpr: number | null;
  precision: number | null;
  auc_roc: number | null;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
  calibration_curve: CalibrationCurve;
}

export interface FairnessSegment {
  groups: Record<string, FairnessGroup>;
  disparate_impact_ratio: number | null;
  equal_opportunity_diff: number | null;
}

export type FairnessReport = Record<string, FairnessSegment>;

export interface ErrorAnalysis {
  overall: {
    fn_count: number; fp_count: number;
    fn_rate: number | null; fp_rate: number | null;
    fn_avg_probability: number | null; fp_avg_probability: number | null;
    tn_avg_probability: number | null; tp_avg_probability: number | null;
  };
  by_segment: Record<string, Record<string, { n: number; fn_rate: number | null; fp_rate: number | null }>>;
}

export interface DriftFeature {
  feature: string;
  psi: number | null;
  status: "stable" | "moderate" | "significant" | "unknown";
}

export interface DriftReport {
  reference_population: string;
  current_population: string;
  features: DriftFeature[];
  summary: { n_features_checked: number; n_significant: number; n_moderate: number; n_stable: number };
}

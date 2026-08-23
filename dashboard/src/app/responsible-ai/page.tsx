"use client";
import { useEffect, useState, useCallback } from "react";
import Sidebar from "@/components/layout/Sidebar";
import TopBar  from "@/components/layout/TopBar";
import CalibrationChart from "@/components/charts/CalibrationChart";
import { api } from "@/lib/api";
import { clsx } from "clsx";
import { AlertTriangle, CheckCircle, ShieldAlert } from "lucide-react";
import type {
  FairnessReport, FairnessSegment, FairnessGroup, ErrorAnalysis, DriftReport,
  FilterPeriod, FilterContractType, FilterRiskLevel, FilterGender,
} from "@/types";

const DEFAULT_FILTERS = {
  period: "all" as FilterPeriod,
  contractType: "all" as FilterContractType,
  riskLevel: "all" as FilterRiskLevel,
  gender: "all" as FilterGender,
};

const SEGMENT_LABELS: Record<string, string> = {
  CODE_GENDER: "Gender",
  NAME_EDUCATION_TYPE: "Education",
  NAME_INCOME_TYPE: "Income Type",
  age_band: "Age Band",
};

function calibGap(curve?: { mean_predicted: number[]; fraction_pos: number[] }): number | null {
  if (!curve || !curve.mean_predicted.length) return null;
  const diffs = curve.mean_predicted.map((m, i) => Math.abs(m - curve.fraction_pos[i]));
  return diffs.reduce((a, b) => a + b, 0) / diffs.length;
}

function pct(v: number | null | undefined, digits = 1): string {
  return v == null ? "—" : `${(v * 100).toFixed(digits)}%`;
}

function DisparityBadge({ ratio }: { ratio: number | null }) {
  if (ratio == null) return null;
  const ok = ratio >= 0.8;
  return (
    <span className={clsx(
      "inline-flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1 rounded-full shrink-0",
      ok ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700",
    )}>
      {ok ? <CheckCircle className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
      Disparate impact ratio {ratio.toFixed(2)} {ok ? "· within four-fifths rule" : "· below 0.80, flag for review"}
    </span>
  );
}

function FairnessSegmentCard({ name, segment }: { name: string; segment: FairnessSegment }) {
  const groups = Object.entries(segment.groups) as [string, FairnessGroup][];
  return (
    <div className="card">
      <div className="flex items-start justify-between gap-3 mb-1 flex-wrap">
        <h3 className="text-sm font-semibold text-ink">{SEGMENT_LABELS[name] ?? name}</h3>
        <DisparityBadge ratio={segment.disparate_impact_ratio} />
      </div>
      {segment.equal_opportunity_diff != null && (
        <p className="text-[11px] text-ink-muted mb-3">
          Equal opportunity difference (max − min recall across groups):{" "}
          <strong className="text-ink">{(segment.equal_opportunity_diff * 100).toFixed(1)} pp</strong>
        </p>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-surface-border">
              {["Group", "N", "Base rate", "Selection rate", "Recall (TPR)", "FPR", "Precision", "Calib. gap"].map((h) => (
                <th key={h} className="pb-2 pr-4 text-left text-[10px] font-medium text-ink-muted uppercase tracking-wide whitespace-nowrap">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {groups.map(([g, s]) => {
              const gap = calibGap(s.calibration_curve);
              return (
                <tr key={g} className="border-b border-surface-border hover:bg-surface-muted/40">
                  <td className="py-2 pr-4 font-medium text-ink whitespace-nowrap">{g}</td>
                  <td className="py-2 pr-4 text-ink-muted">{s.n.toLocaleString()}</td>
                  <td className="py-2 pr-4 text-ink-muted">{pct(s.base_rate)}</td>
                  <td className="py-2 pr-4 text-ink-muted">{pct(s.selection_rate)}</td>
                  <td className="py-2 pr-4 text-ink-muted">{pct(s.tpr)}</td>
                  <td className="py-2 pr-4 text-ink-muted">{pct(s.fpr)}</td>
                  <td className="py-2 pr-4 text-ink-muted">{pct(s.precision)}</td>
                  <td className="py-2 pr-4 text-ink-muted">{gap != null ? gap.toFixed(3) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const DRIFT_BADGE: Record<string, string> = {
  stable:      "bg-green-100 text-green-700",
  moderate:    "bg-amber-100 text-amber-700",
  significant: "bg-red-100 text-red-700",
  unknown:     "bg-gray-100 text-gray-600",
};

function DriftBadge({ status }: { status: string }) {
  return (
    <span className={clsx("text-[10px] font-semibold px-2 py-0.5 rounded-full uppercase tracking-wide",
      DRIFT_BADGE[status] ?? DRIFT_BADGE.unknown)}>
      {status}
    </span>
  );
}

const MODEL_CARD_FACTS = [
  { label: "Algorithm",       value: "LightGBM (GBDT)" },
  { label: "Resampling",      value: "SMOTE (0.3 ratio)" },
  { label: "Tuning",          value: "Optuna · 50 trials" },
  { label: "Calibration",     value: "Isotonic regression" },
  { label: "Threshold",       value: "Optimised (F1)" },
  { label: "Explainability",  value: "SHAP TreeExplainer" },
  { label: "Tracking",        value: "MLflow experiments" },
  { label: "Versioning",      value: "DVC pipeline" },
  { label: "Evaluation split",value: "80/20 stratified holdout" },
  { label: "Sensitive inputs",value: "Present — see below" },
];

const LIMITATIONS = [
  "Reported metrics (AUC, precision/recall, calibration, fairness, error analysis) come from a single stratified 80/20 holdout, not k-fold out-of-fold predictions — k-fold CV is used only inside Optuna's hyperparameter search, so bootstrap confidence intervals reflect one split's sampling variability, not across-split variability.",
  "SMOTE generates synthetic minority-class samples; its effect on calibration and subgroup fairness has not been separately ablated against a no-resampling baseline.",
  "The training data is historical and geographically/temporally bounded to the original Home Credit Kaggle competition — it may not reflect current applicant populations or macroeconomic conditions.",
  "Subgroup fairness and error-analysis metrics are computed only for groups with at least 30 validation examples; smaller subgroups are dropped to avoid unstable estimates and are therefore not audited here.",
  "Drift monitoring compares the training population against Kaggle's application_test split as a covariate-shift proxy — it is not a live production monitor and will not catch drift that occurs after this evaluation was run.",
  "No adversarial robustness, feature-tampering, or stability-under-perturbation testing has been performed.",
];

export default function ResponsibleAIPage() {
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [fairness, setFairness] = useState<FairnessReport | null>(null);
  const [errorAnalysis, setErrorAnalysis] = useState<ErrorAnalysis | null>(null);
  const [drift, setDrift] = useState<DriftReport | null>(null);

  const fetchAll = useCallback(async () => {
    try {
      const [fr, ea, dr] = await Promise.all([api.fairness(), api.errorAnalysis(), api.drift()]);
      setFairness(fr.data);
      setErrorAnalysis(ea.data);
      setDrift(dr.data);
    } catch {}
  }, []);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  const genderGroups = fairness?.CODE_GENDER ? Object.entries(fairness.CODE_GENDER.groups) : [];

  return (
    <div className="flex min-h-screen bg-surface-muted">
      <Sidebar />
      <div className="ml-60 flex-1 flex flex-col min-w-0">
        <TopBar filters={filters} onChange={(f) => setFilters((p) => ({ ...p, ...f }))}
                onRefresh={fetchAll} title="Responsible AI & Model Governance"
                breadcrumb={["Dashboard", "Responsible AI"]} />

        <main className="mt-16 p-6 flex flex-col gap-6">
          <div>
            <h1 className="text-xl font-semibold text-ink">Responsible AI &amp; Model Governance</h1>
            <p className="text-xs text-ink-muted mt-0.5">
              Model card · limitations · subgroup fairness · calibration by segment · error analysis · drift monitoring
            </p>
          </div>

          {/* Model card */}
          <div className="card">
            <h2 className="text-sm font-semibold text-ink mb-3">Model Card</h2>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-xs mb-6">
              {MODEL_CARD_FACTS.map(({ label, value }) => (
                <div key={label} className="bg-surface-muted rounded-xl p-3">
                  <p className="text-[10px] text-ink-muted uppercase tracking-wide font-medium">{label}</p>
                  <p className="font-semibold text-ink mt-1">{value}</p>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5 text-xs">
              <div>
                <p className="font-semibold text-ink mb-1">Intended use</p>
                <p className="text-ink-muted leading-relaxed">
                  Decision-support signal for consumer credit underwriting — estimates the probability that an
                  applicant with thin or no formal credit history will default, to help a human underwriter
                  prioritize manual review. Built as a portfolio/research case study on the Home Credit Default
                  Risk dataset; not connected to a live loan-origination system.
                </p>
              </div>
              <div>
                <p className="font-semibold text-ink mb-1">Out-of-scope uses</p>
                <p className="text-ink-muted leading-relaxed">
                  Fully automated approve/deny decisions without human review; use on populations, products, or
                  geographies outside this dataset's context; use as the sole basis for adverse action without a
                  compliant adverse-action process; production deployment while gender/education/income-type
                  remain raw model inputs (see Sensitive Variables below).
                </p>
              </div>
              <div>
                <p className="font-semibold text-ink mb-1">Training data</p>
                <p className="text-ink-muted leading-relaxed">
                  Home Credit Default Risk (Kaggle) — application_train joined with bureau, bureau_balance,
                  previous_application, POS_CASH_balance, credit_card_balance, and installments_payments;
                  307,511 applications, 8.08% observed default rate, 200+ engineered features after dropping
                  columns with &gt;60% missing values.
                </p>
              </div>
              <div>
                <p className="font-semibold text-ink mb-1">Evaluation data</p>
                <p className="text-ink-muted leading-relaxed">
                  20% stratified holdout from application_train, scored after SMOTE resampling, Optuna-tuned
                  LightGBM, and isotonic calibration. Drift monitoring below compares this training population
                  against application_test (Kaggle's unlabeled holdout) as a covariate-shift proxy.
                </p>
              </div>
            </div>
          </div>

          {/* Sensitive variables & proxies */}
          <div className="card border-amber-200 bg-amber-50/50">
            <div className="flex items-center gap-2 mb-2">
              <ShieldAlert className="w-4 h-4 text-amber-600" />
              <h2 className="text-sm font-semibold text-ink">Sensitive Variables &amp; Proxies</h2>
            </div>
            <ul className="text-xs text-ink-muted space-y-2 leading-relaxed list-disc pl-4">
              <li>
                <strong className="text-ink">CODE_GENDER, NAME_EDUCATION_TYPE, NAME_INCOME_TYPE, FLAG_OWN_CAR,
                and FLAG_OWN_REALTY are included as raw (label-encoded) model inputs</strong> — the model can and
                does use them directly, not merely as slicing metadata for the fairness tables below.
              </li>
              <li>
                In many jurisdictions, credit-scoring regulation (e.g. the U.S. Equal Credit Opportunity Act)
                prohibits using sex/gender directly as a scoring factor. Before any production use, gender should
                be removed from the model's inputs and retained only for the fairness monitoring on this page.
              </li>
              <li>
                NAME_EDUCATION_TYPE and NAME_INCOME_TYPE can act as proxies for socioeconomic status and,
                indirectly, for protected characteristics correlated with it — their SHAP contribution should be
                reviewed on its own, not just their inclusion in the feature list.
              </li>
              <li>
                DAYS_BIRTH (age) is not itself prohibited in most jurisdictions when used for affordability
                assessment, but it is monitored here for disparate impact via the age-band breakdown below.
              </li>
            </ul>
          </div>

          {/* Limitations */}
          <div className="card">
            <h2 className="text-sm font-semibold text-ink mb-3">Limitations</h2>
            <ul className="text-xs text-ink-muted space-y-2 leading-relaxed list-disc pl-4">
              {LIMITATIONS.map((l) => <li key={l}>{l}</li>)}
            </ul>
          </div>

          {/* Fairness & subgroup evaluation */}
          <div>
            <h2 className="text-sm font-semibold text-ink mb-1">Fairness &amp; Subgroup Evaluation</h2>
            <p className="text-[11px] text-ink-muted mb-3">
              Selection rate = share of a group flagged high-risk. Disparate impact ratio below 0.80 fails the
              standard four-fifths rule; equal opportunity difference is the gap in recall (true positive rate)
              across groups in a segment.
            </p>
            <div className="grid grid-cols-1 gap-4">
              {fairness
                ? Object.entries(fairness).map(([name, seg]) => (
                    <FairnessSegmentCard key={name} name={name} segment={seg} />
                  ))
                : <div className="animate-pulse h-40 bg-surface-muted rounded-xl" />}
            </div>
          </div>

          {/* Calibration by segment */}
          {genderGroups.length > 0 && (
            <div className="card">
              <h2 className="text-sm font-semibold text-ink mb-1">Calibration by Segment — Gender</h2>
              <p className="text-[11px] text-ink-muted mb-4">
                Each point is a probability bucket; the dashed line is perfect calibration. Divergence between
                groups indicates the model over/under-estimates risk differently by subgroup.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {genderGroups.map(([g, s]) => (
                  <div key={g}>
                    <p className="text-xs font-medium text-ink mb-1">{g}</p>
                    <CalibrationChart data={s.calibration_curve} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Error analysis */}
          <div className="card">
            <h2 className="text-sm font-semibold text-ink mb-1">False Negative / False Positive Analysis</h2>
            <p className="text-[11px] text-ink-muted mb-4">
              A false negative is a defaulter scored below the flag threshold (the costliest error for the
              lender); a false positive is a good applicant flagged for review (a cost of friction/lost business).
            </p>
            {errorAnalysis ? (
              <>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-5">
                  {[
                    { label: "False Negatives", value: errorAnalysis.overall.fn_count.toLocaleString(),
                      sub: `${pct(errorAnalysis.overall.fn_rate)} of actual defaulters missed` },
                    { label: "False Positives", value: errorAnalysis.overall.fp_count.toLocaleString(),
                      sub: `${pct(errorAnalysis.overall.fp_rate)} of good applicants flagged` },
                    { label: "Avg. probability — FN", value: pct(errorAnalysis.overall.fn_avg_probability, 1),
                      sub: "near-miss defaulters, just under threshold" },
                    { label: "Avg. probability — FP", value: pct(errorAnalysis.overall.fp_avg_probability, 1),
                      sub: "borderline good applicants" },
                  ].map(({ label, value, sub }) => (
                    <div key={label} className="bg-surface-muted rounded-xl p-3">
                      <p className="text-[10px] text-ink-muted uppercase tracking-wide font-medium">{label}</p>
                      <p className="text-lg font-semibold text-ink mt-1">{value}</p>
                      <p className="text-[10px] text-ink-light mt-0.5">{sub}</p>
                    </div>
                  ))}
                </div>

                {Object.entries(errorAnalysis.by_segment).map(([segName, groups]) => (
                  <div key={segName} className="mb-4 last:mb-0">
                    <p className="text-xs font-semibold text-ink mb-2">{SEGMENT_LABELS[segName] ?? segName}</p>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-surface-border">
                            {["Group", "N", "FN rate", "FP rate"].map((h) => (
                              <th key={h} className="pb-2 pr-4 text-left text-[10px] font-medium text-ink-muted uppercase tracking-wide">{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(groups).map(([g, s]) => (
                            <tr key={g} className="border-b border-surface-border hover:bg-surface-muted/40">
                              <td className="py-2 pr-4 font-medium text-ink">{g}</td>
                              <td className="py-2 pr-4 text-ink-muted">{s.n.toLocaleString()}</td>
                              <td className="py-2 pr-4 text-ink-muted">{pct(s.fn_rate)}</td>
                              <td className="py-2 pr-4 text-ink-muted">{pct(s.fp_rate)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
              </>
            ) : (
              <div className="animate-pulse h-32 bg-surface-muted rounded-xl" />
            )}
          </div>

          {/* Drift monitoring */}
          <div className="card">
            <div className="flex items-center justify-between mb-1 flex-wrap gap-2">
              <h2 className="text-sm font-semibold text-ink">Drift Monitoring — Population Stability Index</h2>
              {drift && (
                <span className="text-[11px] text-ink-muted">
                  {drift.summary.n_significant} significant · {drift.summary.n_moderate} moderate ·{" "}
                  {drift.summary.n_stable} stable
                </span>
              )}
            </div>
            <p className="text-[11px] text-ink-muted mb-1">
              PSI &lt; 0.10 stable · 0.10–0.25 moderate (investigate) · &gt;= 0.25 significant (consider retraining).
            </p>
            {drift && (
              <p className="text-[11px] text-ink-light mb-4">
                Reference: {drift.reference_population}. Current: {drift.current_population}
              </p>
            )}
            {drift ? (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-surface-border">
                      {["Feature", "PSI", "Status"].map((h) => (
                        <th key={h} className="pb-2 pr-4 text-left text-[10px] font-medium text-ink-muted uppercase tracking-wide">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {drift.features.map((f) => (
                      <tr key={f.feature} className="border-b border-surface-border hover:bg-surface-muted/40">
                        <td className="py-2 pr-4 font-medium text-ink">{f.feature}</td>
                        <td className="py-2 pr-4 text-ink-muted">{f.psi != null ? f.psi.toFixed(3) : "—"}</td>
                        <td className="py-2 pr-4"><DriftBadge status={f.status} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="animate-pulse h-32 bg-surface-muted rounded-xl" />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

"use client";
import { useState } from "react";
import Link from "next/link";
import { ShieldCheck, ArrowUpRight } from "lucide-react";
import Sidebar   from "@/components/layout/Sidebar";
import TopBar    from "@/components/layout/TopBar";
import PredictionForm from "@/components/prediction/PredictionForm";
import FeatureImportanceChart from "@/components/charts/FeatureImportanceChart";
import type { FilterPeriod, FilterContractType, FilterRiskLevel, FilterGender } from "@/types";

const DEFAULT_FILTERS = {
  period: "all" as FilterPeriod,
  contractType: "all" as FilterContractType,
  riskLevel: "all" as FilterRiskLevel,
  gender: "all" as FilterGender,
};

const TOP_FEATURES = [
  { feature: "EXT_SOURCE_2",         shap_importance: 0.0842 },
  { feature: "EXT_SOURCE_3",         shap_importance: 0.0721 },
  { feature: "EXT_SOURCE_1",         shap_importance: 0.0614 },
  { feature: "DAYS_BIRTH",           shap_importance: 0.0512 },
  { feature: "CREDIT_INCOME_RATIO",  shap_importance: 0.0487 },
  { feature: "ANNUITY_INCOME_RATIO", shap_importance: 0.0431 },
  { feature: "BUREAU_LOAN_COUNT",    shap_importance: 0.0398 },
  { feature: "INSTAL_DPD_MEAN",      shap_importance: 0.0376 },
  { feature: "DAYS_EMPLOYED",        shap_importance: 0.0354 },
  { feature: "AMT_CREDIT",           shap_importance: 0.0321 },
];

export default function PredictPage() {
  const [filters, setFilters] = useState(DEFAULT_FILTERS);

  return (
    <div className="flex min-h-screen bg-surface-muted">
      <Sidebar />
      <div className="ml-60 flex-1 flex flex-col min-w-0">
        <TopBar filters={filters} onChange={(f) => setFilters((p) => ({ ...p, ...f }))}
                title="Risk Score & Decision Support" breadcrumb={["Dashboard","Risk Score"]} />

        <main className="mt-16 p-6 flex flex-col gap-6">
          <div>
            <h1 className="text-xl font-semibold text-ink">Risk Score & Decision Support</h1>
            <p className="text-xs text-ink-muted mt-0.5">
              Calibrated LightGBM · 200+ engineered features · Real-time scoring
            </p>
          </div>

          {/* How it works */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              { step: "01", title: "Input Application", desc: "Fill in the applicant's financial profile." },
              { step: "02", title: "Feature Engineering", desc: "200+ features computed on the fly from raw inputs." },
              { step: "03", title: "Calibrated Score", desc: "Isotonic-calibrated probability with optimal threshold." },
            ].map(({ step, title, desc }) => (
              <div key={step} className="card flex items-start gap-3 p-4">
                <span className="text-xs font-bold text-brand-600 bg-brand-50 px-2.5 py-1
                                 rounded-lg shrink-0">{step}</span>
                <div>
                  <p className="text-xs font-semibold text-ink">{title}</p>
                  <p className="text-[11px] text-ink-muted mt-0.5">{desc}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Form */}
          <div className="card">
            <h2 className="text-sm font-semibold text-ink mb-1">Application Scoring</h2>
            <p className="text-[11px] text-ink-muted mb-5">
              Submit an application to receive a calibrated default-risk probability and a recommended
              next action for human review — this is a decision-support signal, not an automated decision.
            </p>
            <PredictionForm />
          </div>

          {/* SHAP guide */}
          <div className="card">
            <h2 className="text-sm font-semibold text-ink mb-1">Top Predictive Features</h2>
            <p className="text-[11px] text-ink-muted mb-4">
              Global SHAP importance — features with the largest average impact on model output
            </p>
            <FeatureImportanceChart data={TOP_FEATURES} top={10} />
          </div>

          {/* Responsible AI pointer */}
          <Link href="/responsible-ai" className="card flex items-center gap-4 p-4 hover:border-brand-300 transition-colors">
            <div className="w-10 h-10 rounded-xl bg-brand-50 flex items-center justify-center shrink-0">
              <ShieldCheck className="w-5 h-5 text-brand-600" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-ink">Full model card &amp; responsible AI evaluation</p>
              <p className="text-[11px] text-ink-muted mt-0.5">
                Intended use, limitations, sensitive features &amp; proxies, subgroup fairness,
                calibration by segment, false negative/positive analysis, and drift monitoring.
              </p>
            </div>
            <ArrowUpRight className="w-4 h-4 text-ink-light shrink-0" />
          </Link>
        </main>
      </div>
    </div>
  );
}

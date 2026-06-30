"use client";
import { useEffect, useState, useCallback } from "react";
import Sidebar from "@/components/layout/Sidebar";
import TopBar  from "@/components/layout/TopBar";
import MetricCard from "@/components/cards/MetricCard";
import RiskDistributionChart from "@/components/charts/RiskDistributionChart";
import FeatureImportanceChart from "@/components/charts/FeatureImportanceChart";
import ApplicationsTable from "@/components/tables/ApplicationsTable";
import { api } from "@/lib/api";
import {
  Users, DollarSign, AlertCircle, TrendingUp, Activity,
} from "lucide-react";
import type {
  Overview, Metrics, FeatureImportance, ScoreDistribution, Application,
  FilterPeriod, FilterContractType, FilterRiskLevel, FilterGender,
} from "@/types";

const DEFAULT_FILTERS = {
  period:       "all" as FilterPeriod,
  contractType: "all" as FilterContractType,
  riskLevel:    "all" as FilterRiskLevel,
  gender:       "all" as FilterGender,
};

export default function OverviewPage() {
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [overview, setOverview]           = useState<Overview | null>(null);
  const [metrics,  setMetrics ]           = useState<Metrics  | null>(null);
  const [shap,     setShap    ]           = useState<FeatureImportance[] | null>(null);
  const [dist,     setDist    ]           = useState<ScoreDistribution | null>(null);
  const [apps,     setApps    ]           = useState<Application[] | null>(null);

  const fetch = useCallback(async () => {
    try {
      const [ov, me, sh, di, ap] = await Promise.all([
        api.overview(),
        api.metrics(),
        api.featureImportance(15),
        api.riskDistribution(),
        api.sampleApplications(50),
      ]);
      setOverview(ov.data);
      setMetrics(me.data);
      setShap(sh.data.importance);
      setDist(di.data);
      setApps(ap.data.applications);
    } catch { /* API not running — component shows skeletons */ }
  }, []);

  useEffect(() => { fetch(); }, [fetch]);

  const pct = (v: number) => `${(v * 100).toFixed(2)}%`;

  return (
    <div className="flex min-h-screen bg-surface-muted">
      <Sidebar />

      <div className="ml-60 flex-1 flex flex-col min-w-0">
        <TopBar
          filters={filters}
          onChange={(f) => setFilters((p) => ({ ...p, ...f }))}
          onRefresh={fetch}
          title="Overview"
        />

        <main className="mt-16 p-6 flex flex-col gap-6">
          {/* Page title */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-ink">Credit Risk Intelligence</h1>
              <p className="text-xs text-ink-muted mt-0.5">
                Home Credit Default Risk · LightGBM + SHAP · Open Finance
              </p>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-xs text-ink-muted font-medium">Model live</span>
            </div>
          </div>

          {/* KPI cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              label="Total Applications"
              value={overview?.n_train?.toLocaleString() ?? "307,511"}
              icon={<Users className="w-4 h-4 text-brand-600" />}
              sub="Training set"
              variant="accent"
            />
            <MetricCard
              label="Default Rate"
              value={overview ? pct(overview.default_rate) : "8.07%"}
              icon={<AlertCircle className="w-4 h-4 text-red-500" />}
              delta={-0.3}
              sub="vs baseline"
            />
            <MetricCard
              label="AUC-ROC"
              value={metrics?.auc_roc?.toFixed(3) ?? "0.781"}
              icon={<TrendingUp className="w-4 h-4 text-brand-600" />}
              delta={2.1}
              sub="val set"
            />
            <MetricCard
              label="Engineered Features"
              value={overview?.n_engineered_features ?? 200}
              icon={<Activity className="w-4 h-4 text-ink-muted" />}
              sub="From 8 tables"
            />
          </div>

          {/* Secondary KPI row */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: "KS Statistic",   value: metrics?.ks_statistic?.toFixed(3) ?? "0.424", sub: "Model discrimination" },
              { label: "F1 Score",       value: metrics?.f1?.toFixed(3) ?? "0.541",           sub: "Optimal threshold" },
              { label: "Precision",      value: metrics?.precision?.toFixed(3) ?? "0.453",    sub: "At opt. threshold" },
              { label: "Recall",         value: metrics?.recall?.toFixed(3) ?? "0.672",       sub: "At opt. threshold" },
            ].map(({ label, value, sub }) => (
              <div key={label} className="card">
                <p className="stat-label">{label}</p>
                <p className="stat-value mt-1">{value}</p>
                <p className="text-xs text-ink-muted mt-1">{sub}</p>
              </div>
            ))}
          </div>

          {/* Charts row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-sm font-semibold text-ink">Risk Score Distribution</h2>
                  <p className="text-[11px] text-ink-muted mt-0.5">
                    Predicted probability split by actual label
                  </p>
                </div>
                <div className="flex gap-1.5">
                  {["Default","No Default"].map((v, i) => (
                    <button key={v} className={`px-2.5 py-1 text-xs rounded-lg font-medium transition-colors
                      ${i === 0 ? "bg-brand-600 text-white" : "bg-surface-muted text-ink-muted"}`}>
                      {v}
                    </button>
                  ))}
                </div>
              </div>
              <RiskDistributionChart data={dist} />
            </div>

            <div className="card">
              <div className="mb-4">
                <h2 className="text-sm font-semibold text-ink">SHAP Feature Importance</h2>
                <p className="text-[11px] text-ink-muted mt-0.5">
                  Mean |SHAP| across validation set
                </p>
              </div>
              <FeatureImportanceChart data={shap} top={12} />
            </div>
          </div>

          {/* Dataset portfolio cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[
              { title: "application_train", rows: "307,511", cols: "122",  tag: "Main table" },
              { title: "bureau",            rows: "1,716,428", cols: "17", tag: "Credit history" },
              { title: "previous_app",      rows: "1,670,214", cols: "37", tag: "Prev. loans" },
              { title: "installments",      rows: "13,605,401", cols: "8", tag: "Payments" },
            ].map(({ title, rows, cols, tag }) => (
              <div key={title} className="card flex flex-col gap-2 p-4">
                <span className="text-[10px] font-medium uppercase tracking-wide text-brand-600 bg-brand-50
                                 px-2 py-0.5 rounded-full self-start">{tag}</span>
                <p className="text-xs font-semibold text-ink truncate">{title}</p>
                <div className="flex items-end justify-between">
                  <div>
                    <p className="text-sm font-bold text-ink">{rows}</p>
                    <p className="text-[10px] text-ink-muted">rows</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-bold text-ink">{cols}</p>
                    <p className="text-[10px] text-ink-muted">cols</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Applications table */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-sm font-semibold text-ink">Recent Applications</h2>
                <p className="text-[11px] text-ink-muted mt-0.5">
                  Scored by the calibrated LightGBM model
                </p>
              </div>
              <button className="btn-ghost text-xs">View all →</button>
            </div>
            <ApplicationsTable data={apps} />
          </div>

          {/* Footer */}
          <footer className="text-center text-[11px] text-ink-light pb-2">
            Home Credit Default Risk · LightGBM + Optuna + SHAP + MLflow · FastAPI on Render · Dashboard on Vercel
          </footer>
        </main>
      </div>
    </div>
  );
}

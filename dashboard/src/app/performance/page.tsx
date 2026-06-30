"use client";
import { useEffect, useState, useCallback } from "react";
import Sidebar   from "@/components/layout/Sidebar";
import TopBar    from "@/components/layout/TopBar";
import { ROCChart, PRChart } from "@/components/charts/ROCChart";
import CalibrationChart     from "@/components/charts/CalibrationChart";
import { api } from "@/lib/api";
import type { Metrics, RocCurve, PrCurve, FilterPeriod, FilterContractType, FilterRiskLevel, FilterGender } from "@/types";
import { clsx } from "clsx";

const DEFAULT_FILTERS = {
  period: "all" as FilterPeriod,
  contractType: "all" as FilterContractType,
  riskLevel: "all" as FilterRiskLevel,
  gender: "all" as FilterGender,
};

function MetricPill({ label, value, ci }: { label: string; value: number; ci?: { lower: number; upper: number } }) {
  return (
    <div className="card text-center py-4">
      <p className="text-2xl font-semibold text-ink">{value.toFixed(3)}</p>
      <p className="text-[11px] text-ink-muted font-medium uppercase tracking-wide mt-1">{label}</p>
      {ci && (
        <p className="text-[10px] text-ink-light mt-1">
          95% CI [{ci.lower.toFixed(3)}, {ci.upper.toFixed(3)}]
        </p>
      )}
    </div>
  );
}

function ConfusionMatrix({ cm }: { cm: { tn: number; fp: number; fn: number; tp: number } | null }) {
  if (!cm) return <div className="animate-pulse h-32 bg-surface-muted rounded-xl" />;
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  const cells = [
    { label: "True Negative",  value: cm.tn, color: "bg-green-100 text-green-800",  pct: cm.tn / total },
    { label: "False Positive", value: cm.fp, color: "bg-red-100   text-red-800",    pct: cm.fp / total },
    { label: "False Negative", value: cm.fn, color: "bg-amber-100 text-amber-800",  pct: cm.fn / total },
    { label: "True Positive",  value: cm.tp, color: "bg-brand-100 text-brand-800",  pct: cm.tp / total },
  ];
  return (
    <div>
      <div className="grid grid-cols-2 gap-2 text-center mb-2">
        <div />
        <div className="grid grid-cols-2 gap-2">
          <span className="text-[10px] text-ink-muted uppercase tracking-wide font-medium">Pred 0</span>
          <span className="text-[10px] text-ink-muted uppercase tracking-wide font-medium">Pred 1</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {["Actual 0", "Actual 1"].map((row, ri) => (
          <div key={row} className="contents">
            <div className="flex items-center">
              <span className="text-[10px] text-ink-muted font-medium w-14 shrink-0">{row}</span>
              <div className="grid grid-cols-2 gap-2 flex-1">
                {cells.slice(ri * 2, ri * 2 + 2).map((c) => (
                  <div key={c.label} className={clsx("rounded-xl p-3 text-center", c.color)}>
                    <p className="text-sm font-bold">{c.value.toLocaleString()}</p>
                    <p className="text-[10px] mt-0.5">{(c.pct * 100).toFixed(1)}%</p>
                    <p className="text-[9px] opacity-70 mt-0.5">{c.label}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function PerformancePage() {
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [roc,     setRoc    ] = useState<RocCurve | null>(null);
  const [pr,      setPr     ] = useState<PrCurve  | null>(null);
  const [calib,   setCalib  ] = useState<any>(null);
  const [cm,      setCm     ] = useState<any>(null);

  const fetch = useCallback(async () => {
    try {
      const [me, ro, pr_, ca, co] = await Promise.all([
        api.metrics(), api.rocCurve(), api.prCurve(), api.calibration(), api.confusionMatrix(),
      ]);
      setMetrics(me.data);
      setRoc(ro.data);
      setPr(pr_.data);
      setCalib(ca.data);
      setCm(co.data);
    } catch {}
  }, []);

  useEffect(() => { fetch(); }, [fetch]);

  const ci = (k: string) => metrics?.bootstrap_ci?.[k] as any;

  return (
    <div className="flex min-h-screen bg-surface-muted">
      <Sidebar />
      <div className="ml-60 flex-1 flex flex-col min-w-0">
        <TopBar filters={filters} onChange={(f) => setFilters((p) => ({ ...p, ...f }))}
                onRefresh={fetch} title="Model Performance"
                breadcrumb={["Dashboard","Model Performance"]} />

        <main className="mt-16 p-6 flex flex-col gap-6">
          <div>
            <h1 className="text-xl font-semibold text-ink">Model Performance</h1>
            <p className="text-xs text-ink-muted mt-0.5">
              LightGBM · Calibrated probabilities · Bootstrap 95% CI · Optuna-tuned
            </p>
          </div>

          {/* Metric pills */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {[
              { label: "AUC-ROC",        k: "auc_roc",   v: metrics?.auc_roc },
              { label: "AUC-PR",         k: "auc_pr",    v: metrics?.auc_pr },
              { label: "KS Statistic",   k: "ks",        v: metrics?.ks_statistic },
              { label: "Precision",      k: "precision", v: metrics?.precision },
              { label: "Recall",         k: "recall",    v: metrics?.recall },
              { label: "F1 Score",       k: "f1",        v: metrics?.f1 },
            ].map(({ label, k, v }) => (
              <MetricPill
                key={label}
                label={label}
                value={v ?? 0}
                ci={ci(k) ? { lower: ci(k)?.lower ?? 0, upper: ci(k)?.upper ?? 0 } : undefined}
              />
            ))}
          </div>

          {/* Curves */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="card">
              <h2 className="text-sm font-semibold text-ink mb-1">ROC Curve</h2>
              <p className="text-[11px] text-ink-muted mb-3">
                True Positive Rate vs False Positive Rate
              </p>
              <ROCChart data={roc} auc={metrics?.auc_roc} />
            </div>
            <div className="card">
              <h2 className="text-sm font-semibold text-ink mb-1">Precision-Recall Curve</h2>
              <p className="text-[11px] text-ink-muted mb-3">
                Trade-off at different classification thresholds
              </p>
              <PRChart data={pr} auc={metrics?.auc_pr} />
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Calibration */}
            <div className="card">
              <h2 className="text-sm font-semibold text-ink mb-1">Probability Calibration</h2>
              <p className="text-[11px] text-ink-muted mb-3">
                Isotonic calibration · Predicted vs actual fraction of positives
              </p>
              <CalibrationChart data={calib} />
              <p className="text-[10px] text-ink-muted mt-3">
                Perfect calibration = diagonal. Isotonic regression brings the model close to the diagonal,
                ensuring P(default|score=0.3) ≈ 30%.
              </p>
            </div>

            {/* Confusion matrix */}
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-sm font-semibold text-ink">Confusion Matrix</h2>
                  <p className="text-[11px] text-ink-muted mt-0.5">
                    Threshold = {metrics?.threshold?.toFixed(2) ?? "0.35"}
                  </p>
                </div>
              </div>
              <ConfusionMatrix cm={cm} />
            </div>
          </div>

          {/* Bootstrap CI table */}
          {metrics?.bootstrap_ci && Object.keys(metrics.bootstrap_ci).length > 0 && (
            <div className="card">
              <h2 className="text-sm font-semibold text-ink mb-4">Bootstrap Confidence Intervals</h2>
              <p className="text-[11px] text-ink-muted mb-4">
                1,000 bootstrap iterations · 95% CI via percentile method
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-surface-border">
                      {["Metric","Point Estimate","Lower 95%","Upper 95%","Width"].map((h) => (
                        <th key={h} className="pb-2.5 pr-6 text-left text-[10px] font-medium
                                               text-ink-muted uppercase tracking-wide">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics.bootstrap_ci).map(([k, v]: any) => (
                      <tr key={k} className="border-b border-surface-border hover:bg-surface-muted/40">
                        <td className="py-2.5 pr-6 font-medium text-ink">{k.replace(/_/g," ").toUpperCase()}</td>
                        <td className="py-2.5 pr-6 font-semibold text-brand-600">{v.point.toFixed(4)}</td>
                        <td className="py-2.5 pr-6 text-ink-muted">{v.lower.toFixed(4)}</td>
                        <td className="py-2.5 pr-6 text-ink-muted">{v.upper.toFixed(4)}</td>
                        <td className="py-2.5 text-ink-muted">{(v.upper - v.lower).toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

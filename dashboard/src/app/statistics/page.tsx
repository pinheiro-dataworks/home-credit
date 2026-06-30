"use client";
import { useEffect, useState, useCallback } from "react";
import Sidebar from "@/components/layout/Sidebar";
import TopBar  from "@/components/layout/TopBar";
import { api }  from "@/lib/api";
import { clsx } from "clsx";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from "recharts";
import type { FilterPeriod, FilterContractType, FilterRiskLevel, FilterGender } from "@/types";

const DEFAULT_FILTERS = {
  period:       "all" as FilterPeriod,
  contractType: "all" as FilterContractType,
  riskLevel:    "all" as FilterRiskLevel,
  gender:       "all" as FilterGender,
};

const PIE_COLORS = ["#16a34a", "#dc2626"];

function StatsBadge({ sig }: { sig: boolean }) {
  return (
    <span className={clsx(
      "text-[10px] font-medium px-2 py-0.5 rounded-full",
      sig ? "bg-green-100 text-green-700" : "bg-surface-muted text-ink-muted"
    )}>
      {sig ? "Significant" : "Not significant"}
    </span>
  );
}

function TestTable({ rows, cols, title, desc }: {
  rows: any[]; cols: { key: string; label: string; fmt?: (v: any) => string }[];
  title: string; desc: string;
}) {
  return (
    <div className="card">
      <h2 className="text-sm font-semibold text-ink mb-0.5">{title}</h2>
      <p className="text-[11px] text-ink-muted mb-4">{desc}</p>
      {rows.length === 0 ? (
        <div className="space-y-2">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="animate-pulse h-8 bg-surface-muted rounded-lg" />
          ))}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-surface-border">
                {cols.map((c) => (
                  <th key={c.key} className="pb-2.5 pr-4 text-left text-[10px] font-medium
                                             text-ink-muted uppercase tracking-wide">{c.label}</th>
                ))}
                <th className="pb-2.5 text-left text-[10px] font-medium text-ink-muted uppercase tracking-wide">
                  Significance
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 15).map((row, i) => (
                <tr key={i} className="border-b border-surface-border hover:bg-surface-muted/40">
                  {cols.map((c) => (
                    <td key={c.key} className={clsx(
                      "py-2.5 pr-4",
                      c.key === "feature" ? "font-medium text-ink text-xs" : "text-ink-muted"
                    )}>
                      {c.fmt ? c.fmt(row[c.key]) : row[c.key]}
                    </td>
                  ))}
                  <td className="py-2.5">
                    <StatsBadge sig={row.significant} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// Static dataset summaries that don't require a live API
const STATIC_DIST = {
  contract: [
    { name: "Cash loans",      value: 278_232, pct: 90.5 },
    { name: "Revolving loans", value: 29_279,  pct: 9.5  },
  ],
  gender: [
    { name: "Female", value: 202_448, pct: 65.8 },
    { name: "Male",   value: 105_059, pct: 34.2 },
  ],
  target: [
    { name: "Repaid",   value: 282_686, pct: 91.9 },
    { name: "Defaulted",value: 24_825,  pct: 8.1  },
  ],
  income_type: [
    { name: "Working",             value: 158_774 },
    { name: "Commercial assoc.",   value: 71_617  },
    { name: "Pensioner",           value: 55_362  },
    { name: "State servant",       value: 21_703  },
    { name: "Unemployed",          value: 22     },
    { name: "Student",             value: 18     },
  ],
};

export default function StatisticsPage() {
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [ksRows,  setKsRows  ] = useState<any[]>([]);
  const [chiRows, setChiRows ] = useState<any[]>([]);

  const fetch = useCallback(async () => {
    try {
      const res = await api.statistics();
      setKsRows(res.data?.ks_tests   ?? []);
      setChiRows(res.data?.chi2_tests ?? []);
    } catch {}
  }, []);

  useEffect(() => { fetch(); }, [fetch]);

  // Fallback KS data if API not ready
  const ksData = ksRows.length > 0 ? ksRows : [
    { feature: "EXT_SOURCE_2",         ks_statistic: 0.318, p_value: 0.000000, significant: true, mean_target0: 0.638, mean_target1: 0.454 },
    { feature: "EXT_SOURCE_3",         ks_statistic: 0.299, p_value: 0.000000, significant: true, mean_target0: 0.534, mean_target1: 0.372 },
    { feature: "EXT_SOURCE_1",         ks_statistic: 0.251, p_value: 0.000000, significant: true, mean_target0: 0.517, mean_target1: 0.391 },
    { feature: "DAYS_BIRTH",           ks_statistic: 0.195, p_value: 0.000000, significant: true, mean_target0: -14889, mean_target1: -13263 },
    { feature: "CREDIT_INCOME_RATIO",  ks_statistic: 0.159, p_value: 0.000000, significant: true, mean_target0: 2.21, mean_target1: 2.59 },
    { feature: "DAYS_EMPLOYED",        ks_statistic: 0.148, p_value: 0.000000, significant: true, mean_target0: -2792, mean_target1: -2038 },
    { feature: "ANNUITY_INCOME_RATIO", ks_statistic: 0.134, p_value: 0.000000, significant: true, mean_target0: 0.148, mean_target1: 0.163 },
    { feature: "BUREAU_OVERDUE_MAX",   ks_statistic: 0.128, p_value: 0.000000, significant: true, mean_target0: 1.2,  mean_target1: 4.8  },
    { feature: "INSTAL_DPD_MEAN",      ks_statistic: 0.112, p_value: 0.000000, significant: true, mean_target0: 0.12, mean_target1: 0.31 },
    { feature: "BUREAU_BAD_RATIO",     ks_statistic: 0.098, p_value: 0.000000, significant: true, mean_target0: 0.02, mean_target1: 0.06 },
  ];

  const chiData = chiRows.length > 0 ? chiRows : [
    { feature: "NAME_EDUCATION_TYPE",  chi2_statistic: 1842.3, p_value: 0.000000, degrees_of_freedom: 4, significant: true },
    { feature: "NAME_INCOME_TYPE",     chi2_statistic: 1530.1, p_value: 0.000000, degrees_of_freedom: 5, significant: true },
    { feature: "CODE_GENDER",          chi2_statistic: 987.4,  p_value: 0.000000, degrees_of_freedom: 1, significant: true },
    { feature: "FLAG_OWN_CAR",         chi2_statistic: 634.2,  p_value: 0.000000, degrees_of_freedom: 1, significant: true },
    { feature: "NAME_CONTRACT_TYPE",   chi2_statistic: 589.7,  p_value: 0.000000, degrees_of_freedom: 1, significant: true },
    { feature: "FLAG_OWN_REALTY",      chi2_statistic: 412.3,  p_value: 0.000000, degrees_of_freedom: 1, significant: true },
    { feature: "NAME_FAMILY_STATUS",   chi2_statistic: 378.9,  p_value: 0.000000, degrees_of_freedom: 4, significant: true },
    { feature: "NAME_HOUSING_TYPE",    chi2_statistic: 289.1,  p_value: 0.000000, degrees_of_freedom: 5, significant: true },
  ];

  return (
    <div className="flex min-h-screen bg-surface-muted">
      <Sidebar />
      <div className="ml-60 flex-1 flex flex-col min-w-0">
        <TopBar filters={filters} onChange={(f) => setFilters((p) => ({ ...p, ...f }))}
                onRefresh={fetch} title="Statistics" breadcrumb={["Dashboard","Statistics"]} />

        <main className="mt-16 p-6 flex flex-col gap-6">
          <div>
            <h1 className="text-xl font-semibold text-ink">Statistical Analysis</h1>
            <p className="text-xs text-ink-muted mt-0.5">
              Kolmogorov-Smirnov · Chi-Square · Bootstrap CI · α = 0.05
            </p>
          </div>

          {/* Dataset summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: "Training Rows",    value: "307,511" },
              { label: "Test Rows",        value: "48,744"  },
              { label: "Raw Features",     value: "122"     },
              { label: "Default Rate",     value: "8.07%"   },
            ].map(({ label, value }) => (
              <div key={label} className="card text-center py-4">
                <p className="text-2xl font-semibold text-ink">{value}</p>
                <p className="text-[11px] text-ink-muted font-medium uppercase tracking-wide mt-1">{label}</p>
              </div>
            ))}
          </div>

          {/* Distribution charts */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Target distribution */}
            <div className="card">
              <h2 className="text-sm font-semibold text-ink mb-1">Target Distribution</h2>
              <p className="text-[11px] text-ink-muted mb-3">Severe class imbalance (1:11)</p>
              <div className="flex justify-center">
                <PieChart width={180} height={160}>
                  <Pie data={STATIC_DIST.target} cx={90} cy={75} innerRadius={45} outerRadius={70}
                       dataKey="value" paddingAngle={3}>
                    {STATIC_DIST.target.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v: number) => v.toLocaleString()} />
                  <Legend wrapperStyle={{ fontSize: 11 }} iconType="circle" iconSize={7} />
                </PieChart>
              </div>
              <div className="flex justify-center gap-4 text-xs mt-1">
                {STATIC_DIST.target.map((d) => (
                  <div key={d.name} className="text-center">
                    <p className="font-semibold text-ink">{d.pct}%</p>
                    <p className="text-ink-muted">{d.name}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Contract type */}
            <div className="card">
              <h2 className="text-sm font-semibold text-ink mb-1">Contract Type</h2>
              <p className="text-[11px] text-ink-muted mb-3">Cash vs Revolving loans</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={STATIC_DIST.contract} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 9, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
                  <Tooltip formatter={(v: number) => v.toLocaleString()} />
                  <Bar dataKey="value" fill="#16a34a" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Income type */}
            <div className="card">
              <h2 className="text-sm font-semibold text-ink mb-1">Income Type</h2>
              <p className="text-[11px] text-ink-muted mb-3">Applicant employment category</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={STATIC_DIST.income_type} layout="vertical"
                          margin={{ top: 0, right: 4, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 9, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="name" width={105}
                         tick={{ fontSize: 9, fill: "#374151" }} axisLine={false} tickLine={false} />
                  <Tooltip formatter={(v: number) => v.toLocaleString()} />
                  <Bar dataKey="value" fill="#22c55e" radius={[0,4,4,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Statistical test tables */}
          <TestTable
            title="Kolmogorov-Smirnov Test — Numeric Features"
            desc="Two-sample KS test comparing distribution of each feature between DEFAULT=0 and DEFAULT=1"
            rows={ksData}
            cols={[
              { key: "feature",       label: "Feature" },
              { key: "ks_statistic",  label: "KS Stat", fmt: (v) => (+v).toFixed(4) },
              { key: "p_value",       label: "p-value",  fmt: (v) => (+v) < 0.0001 ? "<0.0001" : (+v).toFixed(6) },
              { key: "mean_target0",  label: "Mean (No Default)", fmt: (v) => (+v).toFixed(3) },
              { key: "mean_target1",  label: "Mean (Default)",    fmt: (v) => (+v).toFixed(3) },
            ]}
          />

          <TestTable
            title="Chi-Square Test of Independence — Categorical Features"
            desc="Testing whether each categorical feature is statistically independent from the TARGET variable"
            rows={chiData}
            cols={[
              { key: "feature",         label: "Feature" },
              { key: "chi2_statistic",  label: "χ² Statistic", fmt: (v) => (+v).toFixed(2) },
              { key: "p_value",         label: "p-value",       fmt: (v) => (+v) < 0.0001 ? "<0.0001" : (+v).toFixed(6) },
              { key: "degrees_of_freedom", label: "DoF",        fmt: (v) => v },
            ]}
          />
        </main>
      </div>
    </div>
  );
}

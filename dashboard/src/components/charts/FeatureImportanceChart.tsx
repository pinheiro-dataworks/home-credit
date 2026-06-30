"use client";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import type { FeatureImportance } from "@/types";

interface Props { data: FeatureImportance[] | null; top?: number }

const TOOLTIP = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0];
  return (
    <div className="bg-ink text-white text-xs rounded-xl px-3 py-2 shadow-lg max-w-[200px]">
      <p className="font-medium break-all">{d.payload.feature}</p>
      <p className="text-green-300 mt-0.5">SHAP: {d.value.toFixed(5)}</p>
    </div>
  );
};

export default function FeatureImportanceChart({ data, top = 15 }: Props) {
  if (!data) return <Skeleton />;

  const items = [...data].slice(0, top).reverse();
  const max   = items[items.length - 1]?.shap_importance ?? 1;

  return (
    <ResponsiveContainer width="100%" height={top * 26 + 16}>
      <BarChart
        layout="vertical"
        data={items}
        margin={{ top: 0, right: 12, left: 0, bottom: 0 }}
        barCategoryGap="30%"
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 10, fill: "#9ca3af" }}
               axisLine={false} tickLine={false}
               tickFormatter={(v) => v.toFixed(3)} />
        <YAxis
          type="category"
          dataKey="feature"
          width={148}
          tick={{ fontSize: 10, fill: "#374151" }}
          axisLine={false} tickLine={false}
          tickFormatter={(v: string) => v.length > 22 ? v.slice(0, 21) + "…" : v}
        />
        <Tooltip content={<TOOLTIP />} cursor={{ fill: "rgba(0,0,0,.03)" }} />
        <Bar dataKey="shap_importance" radius={[0,4,4,0]}>
          {items.map((entry, i) => {
            const frac = entry.shap_importance / max;
            const opacity = 0.45 + 0.55 * frac;
            return <Cell key={i} fill={`rgba(22,163,74,${opacity})`} />;
          })}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function Skeleton() {
  return (
    <div className="animate-pulse space-y-2 px-4 py-2">
      {Array.from({ length: 12 }).map((_, i) => (
        <div key={i} className="flex items-center gap-2">
          <div className="w-32 h-3 bg-surface-muted rounded" />
          <div className="h-4 bg-surface-muted rounded" style={{ width: `${30 + i * 4}%` }} />
        </div>
      ))}
    </div>
  );
}

"use client";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";
import type { ScoreDistribution } from "@/types";

interface Props { data: ScoreDistribution | null }

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-ink text-white text-xs rounded-xl px-3 py-2 shadow-lg">
      <p className="font-medium mb-1">Score ≈ {(+label * 100).toFixed(0)}%</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value.toLocaleString()}
        </p>
      ))}
    </div>
  );
};

export default function RiskDistributionChart({ data }: Props) {
  if (!data) return <Skeleton />;

  const chartData = data.bins.map((b, i) => ({
    bin:        b,
    "No Default": data.non_default[i],
    "Default":    data.default[i],
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}
                barCategoryGap="20%">
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
        <XAxis
          dataKey="bin"
          tickFormatter={(v) => `${(+v * 100).toFixed(0)}%`}
          tick={{ fontSize: 10, fill: "#9ca3af" }}
          axisLine={false} tickLine={false}
          interval={4}
        />
        <YAxis tick={{ fontSize: 10, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(0,0,0,.04)" }} />
        <Legend
          wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
          iconType="circle" iconSize={7}
        />
        <Bar dataKey="No Default" fill="#86efac" radius={[4,4,0,0]} />
        <Bar dataKey="Default"    fill="#16a34a" radius={[4,4,0,0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function Skeleton() {
  return (
    <div className="animate-pulse h-[220px] flex items-end gap-1 px-4 pb-4">
      {Array.from({ length: 20 }).map((_, i) => (
        <div key={i} className="flex-1 bg-surface-muted rounded-t"
             style={{ height: `${20 + Math.random() * 60}%` }} />
      ))}
    </div>
  );
}

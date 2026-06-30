"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import type { RocCurve, PrCurve } from "@/types";

interface RocProps { data: RocCurve | null; auc?: number }
interface PrProps  { data: PrCurve  | null; auc?: number }

const TIP = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-ink text-white text-xs rounded-xl px-3 py-2 shadow-lg">
      {payload.map((p: any) => (
        <p key={p.name}>{p.name}: <span className="font-medium">{(+p.value).toFixed(3)}</span></p>
      ))}
    </div>
  );
};

export function ROCChart({ data, auc }: RocProps) {
  if (!data) return <Skeleton />;
  const chartData = data.fpr.map((x, i) => ({ fpr: x, tpr: data.tpr[i] }));

  return (
    <div>
      {auc !== undefined && (
        <p className="text-xs text-ink-muted mb-2">
          AUC-ROC = <span className="font-semibold text-brand-600">{auc.toFixed(3)}</span>
        </p>
      )}
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="fpr" tick={{ fontSize: 10, fill: "#9ca3af" }}
                 label={{ value: "FPR", position: "insideBottomRight", offset: 0, fontSize: 10, fill: "#9ca3af" }}
                 axisLine={false} tickLine={false} tickFormatter={(v) => v.toFixed(1)} />
          <YAxis tick={{ fontSize: 10, fill: "#9ca3af" }} axisLine={false} tickLine={false}
                 label={{ value: "TPR", angle: -90, position: "insideLeft", fontSize: 10, fill: "#9ca3af" }} />
          <Tooltip content={<TIP />} />
          <ReferenceLine x={0} y={0} stroke="#e5e7eb" />
          {/* Diagonal chance line */}
          <Line
            data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]}
            type="linear" dataKey="tpr"
            stroke="#d1d5db" strokeDasharray="4 3" dot={false} strokeWidth={1.5}
            legendType="none"
          />
          <Line
            type="monotone" dataKey="tpr" data={chartData}
            stroke="#16a34a" strokeWidth={2.5} dot={false}
            activeDot={{ r: 4, fill: "#16a34a" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PRChart({ data, auc }: PrProps) {
  if (!data) return <Skeleton />;
  const chartData = data.recall.map((x, i) => ({ recall: x, precision: data.precision[i] }));

  return (
    <div>
      {auc !== undefined && (
        <p className="text-xs text-ink-muted mb-2">
          AUC-PR = <span className="font-semibold text-brand-600">{auc.toFixed(3)}</span>
        </p>
      )}
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="recall" tick={{ fontSize: 10, fill: "#9ca3af" }}
                 label={{ value: "Recall", position: "insideBottomRight", offset: 0, fontSize: 10, fill: "#9ca3af" }}
                 axisLine={false} tickLine={false} tickFormatter={(v) => v.toFixed(1)} />
          <YAxis tick={{ fontSize: 10, fill: "#9ca3af" }} axisLine={false} tickLine={false}
                 label={{ value: "Prec.", angle: -90, position: "insideLeft", fontSize: 10, fill: "#9ca3af" }} />
          <Tooltip content={<TIP />} />
          <Line
            type="monotone" dataKey="precision" data={chartData}
            stroke="#16a34a" strokeWidth={2.5} dot={false}
            activeDot={{ r: 4, fill: "#16a34a" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function Skeleton() {
  return <div className="animate-pulse bg-surface-muted rounded-xl h-[200px]" />;
}

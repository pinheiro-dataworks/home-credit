"use client";
import {
  ComposedChart, Line, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

interface CalibData { mean_predicted: number[]; fraction_pos: number[] }
interface Props { data: CalibData | null }

export default function CalibrationChart({ data }: Props) {
  if (!data) return <div className="animate-pulse bg-surface-muted rounded-xl h-[200px]" />;

  const chartData = data.mean_predicted.map((x, i) => ({
    predicted: x,
    actual:    data.fraction_pos[i],
  }));

  const perfect = [{ predicted: 0, actual: 0 }, { predicted: 1, actual: 1 }];

  return (
    <ResponsiveContainer width="100%" height={200}>
      <ComposedChart margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="predicted"
          type="number" domain={[0, 1]}
          tick={{ fontSize: 10, fill: "#9ca3af" }}
          label={{ value: "Mean Predicted", position: "insideBottomRight", offset: -5, fontSize: 10, fill: "#9ca3af" }}
          axisLine={false} tickLine={false}
          tickFormatter={(v) => v.toFixed(1)}
        />
        <YAxis
          type="number" domain={[0, 1]}
          tick={{ fontSize: 10, fill: "#9ca3af" }}
          label={{ value: "Actual", angle: -90, position: "insideLeft", fontSize: 10, fill: "#9ca3af" }}
          axisLine={false} tickLine={false}
        />
        <Tooltip
          formatter={(v: number) => [v.toFixed(3)]}
          contentStyle={{ fontSize: 11, borderRadius: 12, border: "none", background: "#111827", color: "#fff" }}
        />
        {/* Perfect calibration */}
        <Line
          data={perfect} type="linear" dataKey="actual"
          stroke="#d1d5db" strokeDasharray="4 3" dot={false} strokeWidth={1.5}
        />
        {/* Model calibration */}
        <Scatter
          data={chartData} dataKey="actual"
          fill="#16a34a" line={{ stroke: "#16a34a", strokeWidth: 2 }}
          lineType="joint" shape="circle"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

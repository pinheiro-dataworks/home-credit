import { clsx } from "clsx";
import { TrendingUp, TrendingDown, ArrowRight, MoreHorizontal } from "lucide-react";
import type { ReactNode } from "react";

interface Props {
  label:      string;
  value:      string | number;
  sub?:       string;
  delta?:     number;
  icon?:      ReactNode;
  variant?:   "default" | "accent";
  unit?:      string;
  onClick?:   () => void;
}

export default function MetricCard({
  label, value, sub, delta, icon, variant = "default", unit, onClick,
}: Props) {
  const isAccent = variant === "accent";

  return (
    <div
      onClick={onClick}
      className={clsx(
        "card relative flex flex-col gap-3 min-h-[120px] transition-shadow hover:shadow-card-md",
        isAccent && "bg-brand-600 border-brand-700 text-white",
        onClick && "cursor-pointer"
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className={clsx(
          "w-9 h-9 rounded-xl flex items-center justify-center shrink-0",
          isAccent ? "bg-brand-700/60" : "bg-surface-muted border border-surface-border"
        )}>
          {icon}
        </div>
        <button className={clsx(
          "w-6 h-6 rounded-lg flex items-center justify-center transition-colors",
          isAccent
            ? "text-brand-200 hover:bg-brand-700/60"
            : "text-ink-light hover:bg-surface-muted"
        )}>
          <MoreHorizontal className="w-4 h-4" />
        </button>
      </div>

      {/* Value */}
      <div>
        <div className={clsx("flex items-end gap-1", isAccent ? "text-white" : "text-ink")}>
          <span className="text-2xl font-semibold tracking-tight leading-none">
            {typeof value === "number" ? value.toLocaleString() : value}
          </span>
          {unit && <span className={clsx("text-sm mb-0.5", isAccent ? "text-brand-200" : "text-ink-muted")}>{unit}</span>}
        </div>
        <p className={clsx("text-[11px] font-medium mt-1.5 uppercase tracking-wide",
                           isAccent ? "text-brand-200" : "text-ink-muted")}>
          {label}
        </p>
      </div>

      {/* Footer */}
      {(delta !== undefined || sub) && (
        <div className="flex items-center justify-between mt-auto pt-2
                        border-t border-opacity-20"
             style={{ borderColor: isAccent ? "rgba(255,255,255,.15)" : undefined }}
        >
          {delta !== undefined ? (
            <span className={clsx(
              "flex items-center gap-1 text-xs font-medium",
              delta >= 0
                ? (isAccent ? "text-green-200" : "text-green-600")
                : (isAccent ? "text-red-200"   : "text-red-500")
            )}>
              {delta >= 0
                ? <TrendingUp className="w-3 h-3" />
                : <TrendingDown className="w-3 h-3" />}
              {Math.abs(delta).toFixed(1)}%
            </span>
          ) : (
            <span className={clsx("text-xs", isAccent ? "text-brand-200" : "text-ink-muted")}>
              {sub}
            </span>
          )}
          <ArrowRight className={clsx("w-3.5 h-3.5", isAccent ? "text-brand-200" : "text-ink-light")} />
        </div>
      )}
    </div>
  );
}

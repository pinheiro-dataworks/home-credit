"use client";
import { HelpCircle, Bell, MessageSquare, Share2, RefreshCw, ChevronDown } from "lucide-react";
import { clsx } from "clsx";
import type {
  FilterPeriod, FilterContractType, FilterRiskLevel, FilterGender,
} from "@/types";

interface Filters {
  period:       FilterPeriod;
  contractType: FilterContractType;
  riskLevel:    FilterRiskLevel;
  gender:       FilterGender;
}

interface Props {
  filters:    Filters;
  onChange:   (f: Partial<Filters>) => void;
  onRefresh?: () => void;
  title?:     string;
  breadcrumb?:string[];
}

function FilterSelect<T extends string>({
  label, value, options, onChange,
}: {
  label:    string;
  value:    T;
  options:  { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="appearance-none pl-3 pr-7 py-1.5 text-xs font-medium bg-surface-muted
                   border border-surface-border rounded-lg text-ink
                   focus:outline-none focus:ring-1 focus:ring-brand-500 cursor-pointer"
      >
        <option value="" disabled>{label}</option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-ink-light" />
    </div>
  );
}

export default function TopBar({ filters, onChange, onRefresh, title = "Overview", breadcrumb }: Props) {
  const crumbs = breadcrumb ?? ["Dashboard", title];

  return (
    <header className="fixed top-0 left-60 right-0 h-16 bg-white border-b border-surface-border
                       flex items-center px-6 gap-4 z-30">
      {/* Breadcrumb */}
      <div className="flex items-center gap-3 shrink-0">
        <nav className="flex items-center gap-1.5 text-xs text-ink-muted">
          {crumbs.map((c, i) => (
            <span key={c} className="flex items-center gap-1.5">
              {i > 0 && <span className="text-surface-border">/</span>}
              <span className={clsx(i === crumbs.length - 1 ? "text-ink font-medium" : "")}>
                {c}
              </span>
            </span>
          ))}
        </nav>
      </div>

      {/* Filters */}
      <div className="flex-1 flex items-center gap-2 overflow-x-auto no-scrollbar">
        <FilterSelect<FilterPeriod>
          label="Period"
          value={filters.period}
          options={[
            { value: "7d",  label: "Last 7 days" },
            { value: "30d", label: "Last 30 days" },
            { value: "90d", label: "Last 90 days" },
            { value: "1y",  label: "Last year" },
            { value: "all", label: "All time" },
          ]}
          onChange={(v) => onChange({ period: v })}
        />
        <FilterSelect<FilterContractType>
          label="Contract"
          value={filters.contractType}
          options={[
            { value: "all",              label: "All contracts" },
            { value: "Cash loans",       label: "Cash loans" },
            { value: "Revolving loans",  label: "Revolving loans" },
          ]}
          onChange={(v) => onChange({ contractType: v })}
        />
        <FilterSelect<FilterRiskLevel>
          label="Risk"
          value={filters.riskLevel}
          options={[
            { value: "all",    label: "All risk levels" },
            { value: "Low",    label: "Low risk" },
            { value: "Medium", label: "Medium risk" },
            { value: "High",   label: "High risk" },
          ]}
          onChange={(v) => onChange({ riskLevel: v })}
        />
        <FilterSelect<FilterGender>
          label="Gender"
          value={filters.gender}
          options={[
            { value: "all", label: "All genders" },
            { value: "M",   label: "Male" },
            { value: "F",   label: "Female" },
          ]}
          onChange={(v) => onChange({ gender: v })}
        />

        {onRefresh && (
          <button
            onClick={onRefresh}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-ink-muted
                       border border-surface-border rounded-lg hover:bg-surface-muted transition-colors"
          >
            <RefreshCw className="w-3 h-3" /> Reset
          </button>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-2 shrink-0">
        {[HelpCircle, MessageSquare, Bell].map((Icon, i) => (
          <button
            key={i}
            className="w-8 h-8 rounded-full flex items-center justify-center
                       text-ink-muted hover:bg-surface-muted transition-colors"
          >
            <Icon className="w-4 h-4" />
          </button>
        ))}

        {/* Avatar */}
        <div className="w-8 h-8 rounded-full bg-brand-600 flex items-center justify-center
                        text-white text-xs font-semibold ml-1">
          DS
        </div>

        {/* Share */}
        <button className="flex items-center gap-1.5 btn-primary text-xs ml-1">
          <Share2 className="w-3.5 h-3.5" /> Share
        </button>
      </div>
    </header>
  );
}

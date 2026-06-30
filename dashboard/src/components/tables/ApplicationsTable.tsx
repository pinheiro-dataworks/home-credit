"use client";
import { useState } from "react";
import { Search, SlidersHorizontal, MoreHorizontal, ChevronUp, ChevronDown } from "lucide-react";
import { clsx } from "clsx";
import type { Application } from "@/types";

interface Props { data: Application[] | null }

function RiskBadge({ label }: { label: string }) {
  return (
    <span className={clsx(
      "inline-flex items-center gap-1 text-[11px] font-medium px-2.5 py-0.5 rounded-full",
      label === "Low"    && "bg-green-100 text-green-700",
      label === "Medium" && "bg-amber-100 text-amber-700",
      label === "High"   && "bg-red-100   text-red-700",
    )}>
      <span className={clsx(
        "w-1.5 h-1.5 rounded-full",
        label === "Low"    && "bg-green-500",
        label === "Medium" && "bg-amber-500",
        label === "High"   && "bg-red-500",
      )} />
      {label}
    </span>
  );
}

type SortKey = "risk_score" | "income" | "credit" | "submitted_at";

export default function ApplicationsTable({ data }: Props) {
  const [query,   setQuery  ] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("risk_score");
  const [sortAsc, setSortAsc] = useState(false);
  const [page,    setPage   ] = useState(0);
  const PER_PAGE = 7;

  if (!data) return (
    <div className="space-y-2 p-4">
      {Array.from({ length: 7 }).map((_, i) => (
        <div key={i} className="animate-pulse h-10 bg-surface-muted rounded-xl" />
      ))}
    </div>
  );

  const filtered = data
    .filter((a) =>
      a.id.toLowerCase().includes(query.toLowerCase()) ||
      a.contract_type.toLowerCase().includes(query.toLowerCase()) ||
      a.risk_label.toLowerCase().includes(query.toLowerCase())
    )
    .sort((a, b) => {
      const av = a[sortKey] as number | string;
      const bv = b[sortKey] as number | string;
      return sortAsc
        ? av < bv ? -1 : 1
        : av > bv ? -1 : 1;
    });

  const pages = Math.ceil(filtered.length / PER_PAGE);
  const rows  = filtered.slice(page * PER_PAGE, (page + 1) * PER_PAGE);

  const handleSort = (k: SortKey) => {
    if (k === sortKey) setSortAsc(!sortAsc);
    else { setSortKey(k); setSortAsc(false); }
    setPage(0);
  };

  const SortIcon = ({ k }: { k: SortKey }) =>
    sortKey === k
      ? sortAsc ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
      : <ChevronDown className="w-3 h-3 opacity-30" />;

  return (
    <div>
      {/* Table controls */}
      <div className="flex items-center gap-3 mb-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-ink-light" />
          <input
            type="text"
            placeholder="Search applications…"
            value={query}
            onChange={(e) => { setQuery(e.target.value); setPage(0); }}
            className="w-full pl-8 pr-3 py-2 text-xs border border-surface-border rounded-xl
                       bg-surface focus:outline-none focus:ring-1 focus:ring-brand-500
                       placeholder-ink-light"
          />
        </div>
        <button className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-ink-muted
                           border border-surface-border rounded-xl hover:bg-surface-muted transition-colors">
          <SlidersHorizontal className="w-3.5 h-3.5" /> Filter
        </button>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-surface-border">
              <th className="pb-2.5 text-left">
                <input type="checkbox" className="rounded border-surface-border" />
              </th>
              {[
                { key: "id",            label: "Application ID" },
                { key: "income",        label: "Income",        sort: true },
                { key: "credit",        label: "Credit",        sort: true },
                { key: "risk_score",    label: "Risk Score",    sort: true },
                { key: "risk_label",    label: "Risk Level" },
                { key: "contract_type", label: "Contract" },
                { key: "submitted_at",  label: "Date",          sort: true },
                { key: "actions",       label: "" },
              ].map(({ key, label, sort }) => (
                <th
                  key={key}
                  onClick={sort ? () => handleSort(key as SortKey) : undefined}
                  className={clsx(
                    "pb-2.5 pr-4 text-left font-medium text-ink-muted uppercase tracking-wide text-[10px]",
                    sort && "cursor-pointer hover:text-ink select-none"
                  )}
                >
                  <span className="inline-flex items-center gap-1">
                    {label}
                    {sort && <SortIcon k={key as SortKey} />}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((app) => (
              <tr key={app.id} className="border-b border-surface-border hover:bg-surface-muted/40 transition-colors">
                <td className="py-3">
                  <input type="checkbox" className="rounded border-surface-border" />
                </td>
                <td className="py-3 pr-4 font-medium text-ink">{app.id}</td>
                <td className="py-3 pr-4 text-ink-muted">
                  ${app.income.toLocaleString()}
                </td>
                <td className="py-3 pr-4 text-ink-muted">
                  ${app.credit.toLocaleString()}
                </td>
                <td className="py-3 pr-4">
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1.5 rounded-full bg-surface-muted overflow-hidden">
                      <div
                        className={clsx(
                          "h-full rounded-full",
                          app.risk_score < 25 ? "bg-green-500" :
                          app.risk_score < 55 ? "bg-amber-500" : "bg-red-500"
                        )}
                        style={{ width: `${app.risk_score}%` }}
                      />
                    </div>
                    <span className="font-medium text-ink">{app.risk_score}%</span>
                  </div>
                </td>
                <td className="py-3 pr-4"><RiskBadge label={app.risk_label} /></td>
                <td className="py-3 pr-4 text-ink-muted">{app.contract_type}</td>
                <td className="py-3 pr-4 text-ink-muted">{app.submitted_at}</td>
                <td className="py-3">
                  <button className="w-6 h-6 rounded-lg flex items-center justify-center
                                     text-ink-light hover:bg-surface-muted transition-colors">
                    <MoreHorizontal className="w-4 h-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {pages > 1 && (
        <div className="flex items-center justify-between mt-4 pt-3 border-t border-surface-border">
          <span className="text-xs text-ink-muted">
            {filtered.length} applications
          </span>
          <div className="flex gap-1">
            {Array.from({ length: pages }).map((_, i) => (
              <button
                key={i}
                onClick={() => setPage(i)}
                className={clsx(
                  "w-7 h-7 rounded-lg text-xs font-medium transition-colors",
                  i === page
                    ? "bg-brand-600 text-white"
                    : "text-ink-muted hover:bg-surface-muted"
                )}
              >
                {i + 1}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

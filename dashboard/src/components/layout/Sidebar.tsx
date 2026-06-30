"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Image from "next/image";
import {
  LayoutDashboard, TrendingUp, Zap, BarChart2,
  Search, ChevronRight, ArrowUpRight,
} from "lucide-react";
import { clsx } from "clsx";

const NAV = [
  { href: "/",              label: "Overview",         icon: LayoutDashboard },
  { href: "/performance",   label: "Model Performance",icon: TrendingUp       },
  { href: "/predict",       label: "Live Prediction",  icon: Zap              },
  { href: "/statistics",    label: "Statistics",       icon: BarChart2        },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed inset-y-0 left-0 w-60 flex flex-col bg-white border-r border-surface-border z-40">
      {/* Logo area */}
      <div className="h-16 flex items-center px-5 border-b border-surface-border shrink-0">
        <Image
          src="/logo.png"
          alt="Home Credit"
          width={2899}
          height={1164}
          style={{ width: "150px", height: "auto" }}
          priority
        />
      </div>

      {/* Search */}
      <div className="px-4 pt-4 pb-2 shrink-0">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-ink-light" />
          <input
            type="text"
            placeholder="Search…"
            className="w-full pl-8 pr-3 py-2 text-xs bg-surface-muted border border-surface-border rounded-xl
                       focus:outline-none focus:ring-1 focus:ring-brand-500 placeholder-ink-light"
          />
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-3 py-2">
        <p className="px-2 mb-2 text-[10px] text-ink-light font-medium uppercase tracking-widest">
          Analytics
        </p>
        <ul className="space-y-0.5">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <li key={href}>
                <Link
                  href={href}
                  className={clsx(
                    "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors",
                    active
                      ? "bg-ink text-white"
                      : "text-ink-muted hover:bg-surface-muted hover:text-ink"
                  )}
                >
                  <Icon className="w-4 h-4 shrink-0" strokeWidth={active ? 2.5 : 1.75} />
                  <span className="flex-1">{label}</span>
                  {active && <ChevronRight className="w-3 h-3 opacity-60" />}
                </Link>
              </li>
            );
          })}
        </ul>

        <p className="px-2 mt-5 mb-2 text-[10px] text-ink-light font-medium uppercase tracking-widest">
          Pipeline
        </p>
        <ul className="space-y-0.5">
          {[
            { label: "Feature Store",   sub: "DVC tracked" },
            { label: "MLflow Runs",     sub: "Experiments" },
            { label: "SHAP Values",     sub: "Explainability" },
          ].map(({ label, sub }) => (
            <li key={label}>
              <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm
                                 text-ink-muted hover:bg-surface-muted hover:text-ink transition-colors text-left">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-400 shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{label}</p>
                  <p className="text-[10px] text-ink-light">{sub}</p>
                </div>
              </button>
            </li>
          ))}
        </ul>
      </nav>

      {/* Promo card */}
      <div className="m-3 p-3.5 rounded-2xl bg-brand-600 text-white shrink-0">
        <p className="text-xs font-semibold mb-0.5">Model v1.0 live</p>
        <p className="text-[10px] text-brand-200 mb-3">
          LightGBM · AUC 0.78 · Deployed on Render
        </p>
        <a
          href="https://github.com/pinheiro-dataworks?tab=repositories"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 text-xs font-medium bg-white text-brand-700
                     px-3 py-1.5 rounded-lg hover:bg-brand-50 transition-colors"
        >
          View on GitHub <ArrowUpRight className="w-3 h-3" />
        </a>
      </div>
    </aside>
  );
}

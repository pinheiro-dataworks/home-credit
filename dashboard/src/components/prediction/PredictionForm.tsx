"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { clsx } from "clsx";
import { Zap, AlertTriangle, CheckCircle, Loader2 } from "lucide-react";
import type { PredictionResult } from "@/types";

const DEFAULTS = {
  AMT_INCOME_TOTAL:    135_000,
  AMT_CREDIT:          406_597,
  AMT_ANNUITY:         20_560,
  AMT_GOODS_PRICE:     351_000,
  DAYS_BIRTH:          -9_461,
  DAYS_EMPLOYED:       -637,
  EXT_SOURCE_1:        0.502,
  EXT_SOURCE_2:        0.626,
  EXT_SOURCE_3:        0.555,
  CNT_CHILDREN:        0,
  CNT_FAM_MEMBERS:     2,
  NAME_CONTRACT_TYPE:  "Cash loans",
  CODE_GENDER:         "M",
  FLAG_OWN_CAR:        "N",
  FLAG_OWN_REALTY:     "Y",
  NAME_INCOME_TYPE:    "Working",
  NAME_EDUCATION_TYPE: "Secondary / secondary special",
  NAME_FAMILY_STATUS:  "Married",
  NAME_HOUSING_TYPE:   "House / apartment",
};

interface FieldProps {
  label: string; name: string; type?: "number"|"text"|"select";
  options?: string[]; step?: string; min?: string; max?: string;
  value: string|number; onChange: (n: string, v: string|number) => void;
}

function Field({ label, name, type="text", options, step, min, max, value, onChange }: FieldProps) {
  return (
    <div>
      <label className="block text-[11px] font-medium text-ink-muted mb-1 uppercase tracking-wide">
        {label}
      </label>
      {type === "select" ? (
        <select
          value={value}
          onChange={(e) => onChange(name, e.target.value)}
          className="select-field text-sm"
        >
          {options!.map((o) => <option key={o}>{o}</option>)}
        </select>
      ) : (
        <input
          type={type}
          value={value}
          step={step}
          min={min}
          max={max}
          onChange={(e) => onChange(name, type === "number" ? +e.target.value : e.target.value)}
          className="input-field text-sm"
        />
      )}
    </div>
  );
}

export default function PredictionForm() {
  const [form,    setForm   ] = useState<Record<string,any>>(DEFAULTS);
  const [loading, setLoading] = useState(false);
  const [result,  setResult ] = useState<PredictionResult | null>(null);
  const [error,   setError  ] = useState<string | null>(null);

  const handleChange = (name: string, value: string | number) =>
    setForm((f) => ({ ...f, [name]: value }));

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.predict(form);
      setResult(res.data);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? "Prediction failed. Check that the API is running.");
    } finally {
      setLoading(false);
    }
  };

  const color = !result ? "gray"
    : result.risk_label === "Low"    ? "green"
    : result.risk_label === "Medium" ? "amber"
    : "red";

  return (
    <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
      {/* Form */}
      <div className="xl:col-span-3 space-y-5">
        <div className="grid grid-cols-2 gap-4">
          <Field label="Annual Income ($)" name="AMT_INCOME_TOTAL" type="number" step="1000"
                 value={form.AMT_INCOME_TOTAL} onChange={handleChange} />
          <Field label="Credit Amount ($)" name="AMT_CREDIT" type="number" step="1000"
                 value={form.AMT_CREDIT} onChange={handleChange} />
          <Field label="Annuity ($)" name="AMT_ANNUITY" type="number" step="100"
                 value={form.AMT_ANNUITY} onChange={handleChange} />
          <Field label="Goods Price ($)" name="AMT_GOODS_PRICE" type="number" step="1000"
                 value={form.AMT_GOODS_PRICE} onChange={handleChange} />
          <Field label="Days Since Birth (negative)" name="DAYS_BIRTH" type="number"
                 value={form.DAYS_BIRTH} onChange={handleChange} />
          <Field label="Days Employed (negative)" name="DAYS_EMPLOYED" type="number"
                 value={form.DAYS_EMPLOYED} onChange={handleChange} />
          <Field label="Ext. Score 1 (0-1)" name="EXT_SOURCE_1" type="number" step="0.001" min="0" max="1"
                 value={form.EXT_SOURCE_1} onChange={handleChange} />
          <Field label="Ext. Score 2 (0-1)" name="EXT_SOURCE_2" type="number" step="0.001" min="0" max="1"
                 value={form.EXT_SOURCE_2} onChange={handleChange} />
          <Field label="Ext. Score 3 (0-1)" name="EXT_SOURCE_3" type="number" step="0.001" min="0" max="1"
                 value={form.EXT_SOURCE_3} onChange={handleChange} />
          <Field label="Children" name="CNT_CHILDREN" type="number" min="0"
                 value={form.CNT_CHILDREN} onChange={handleChange} />
          <Field label="Contract Type" name="NAME_CONTRACT_TYPE" type="select"
                 options={["Cash loans","Revolving loans"]}
                 value={form.NAME_CONTRACT_TYPE} onChange={handleChange} />
          <Field label="Gender" name="CODE_GENDER" type="select" options={["M","F"]}
                 value={form.CODE_GENDER} onChange={handleChange} />
          <Field label="Income Type" name="NAME_INCOME_TYPE" type="select"
                 options={["Working","Commercial associate","Pensioner","State servant","Unemployed"]}
                 value={form.NAME_INCOME_TYPE} onChange={handleChange} />
          <Field label="Education" name="NAME_EDUCATION_TYPE" type="select"
                 options={["Secondary / secondary special","Higher education","Incomplete higher","Lower secondary","Academic degree"]}
                 value={form.NAME_EDUCATION_TYPE} onChange={handleChange} />
          <Field label="Family Status" name="NAME_FAMILY_STATUS" type="select"
                 options={["Married","Single / not married","Civil marriage","Separated","Widow"]}
                 value={form.NAME_FAMILY_STATUS} onChange={handleChange} />
          <Field label="Own Car" name="FLAG_OWN_CAR" type="select" options={["Y","N"]}
                 value={form.FLAG_OWN_CAR} onChange={handleChange} />
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading}
          className="btn-primary w-full flex items-center justify-center gap-2 py-3"
        >
          {loading
            ? <><Loader2 className="w-4 h-4 animate-spin" /> Computing risk…</>
            : <><Zap className="w-4 h-4" /> Predict Default Risk</>
          }
        </button>

        {error && (
          <div className="flex items-start gap-2 p-3 rounded-xl bg-red-50 border border-red-100 text-red-700 text-xs">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            {error}
          </div>
        )}
      </div>

      {/* Result panel */}
      <div className="xl:col-span-2">
        {result ? (
          <div className={clsx(
            "card h-full flex flex-col items-center justify-center gap-6 text-center p-8",
            color === "green" && "border-green-200 bg-green-50",
            color === "amber" && "border-amber-200 bg-amber-50",
            color === "red"   && "border-red-200   bg-red-50",
          )}>
            {/* Gauge-style display */}
            <div className="relative w-36 h-36">
              <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
                <circle cx="60" cy="60" r="50" fill="none" stroke="#e5e7eb" strokeWidth="10" />
                <circle cx="60" cy="60" r="50" fill="none"
                        stroke={color === "green" ? "#16a34a" : color === "amber" ? "#d97706" : "#dc2626"}
                        strokeWidth="10"
                        strokeDasharray={`${2 * Math.PI * 50 * result.risk_score / 100} ${2 * Math.PI * 50}`}
                        strokeLinecap="round" />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className={clsx(
                  "text-3xl font-bold",
                  color === "green" && "text-green-700",
                  color === "amber" && "text-amber-700",
                  color === "red"   && "text-red-700",
                )}>
                  {result.risk_score.toFixed(1)}%
                </span>
                <span className="text-xs text-ink-muted">risk score</span>
              </div>
            </div>

            <div>
              <div className={clsx(
                "inline-flex items-center gap-2 text-sm font-semibold px-4 py-1.5 rounded-full mb-2",
                color === "green" && "bg-green-100 text-green-700",
                color === "amber" && "bg-amber-100 text-amber-700",
                color === "red"   && "bg-red-100   text-red-700",
              )}>
                {result.predicted_default
                  ? <AlertTriangle className="w-4 h-4" />
                  : <CheckCircle className="w-4 h-4" />}
                {result.risk_label} Risk
              </div>
              <p className="text-xs text-ink-muted">
                Default probability:{" "}
                <span className="font-semibold text-ink">
                  {(result.default_probability * 100).toFixed(2)}%
                </span>
              </p>
              <p className="text-xs text-ink-muted mt-1">
                Threshold: <span className="font-medium">{(result.threshold * 100).toFixed(1)}%</span>
                {" · "}
                Decision:{" "}
                <span className={clsx(
                  "font-semibold",
                  result.predicted_default ? "text-red-600" : "text-green-600"
                )}>
                  {result.predicted_default ? "Deny" : "Approve"}
                </span>
              </p>
            </div>

            {/* Mini stats */}
            <div className="w-full grid grid-cols-2 gap-3 pt-4 border-t border-opacity-30"
                 style={{ borderColor: color === "green" ? "#bbf7d0" : color === "amber" ? "#fde68a" : "#fecaca" }}>
              {[
                { label: "Credit/Income", value: (form.AMT_CREDIT / form.AMT_INCOME_TOTAL).toFixed(2) },
                { label: "Credit Term (mo)", value: Math.round(form.AMT_CREDIT / (form.AMT_ANNUITY || 1)) },
                { label: "Age (years)", value: Math.abs(Math.round(form.DAYS_BIRTH / 365)) },
                { label: "Emp. Years", value: form.DAYS_EMPLOYED < 0 ? Math.abs(Math.round(form.DAYS_EMPLOYED / 365)) : "N/A" },
              ].map(({ label, value }) => (
                <div key={label} className="text-center">
                  <p className="text-lg font-semibold text-ink">{value}</p>
                  <p className="text-[10px] text-ink-muted uppercase tracking-wide">{label}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="card h-full flex flex-col items-center justify-center text-center gap-4 p-8 min-h-[400px]">
            <div className="w-16 h-16 rounded-2xl bg-surface-muted flex items-center justify-center">
              <Zap className="w-7 h-7 text-ink-light" />
            </div>
            <div>
              <p className="text-sm font-medium text-ink">Fill in the application</p>
              <p className="text-xs text-ink-muted mt-1">
                The model will compute real-time default probability using<br />
                a calibrated LightGBM with 200+ engineered features.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

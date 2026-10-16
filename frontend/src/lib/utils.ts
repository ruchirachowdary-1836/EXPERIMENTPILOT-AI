import clsx, { type ClassValue } from "clsx";

export const cn = (...v: ClassValue[]) => clsx(v);
export const pct = (v: number | null | undefined, d = 0) => (v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`);
export const num = (v: number | null | undefined, d = 2) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(d));
export const humanize = (s: string | null | undefined) => (s ?? "").replace(/_/g, " ");
export const title = (s: string) => humanize(s).replace(/\b\w/g, (c) => c.toUpperCase());
export const shortDate = (iso?: string | null) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—");
export const bytes = (n: number) => (n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`);

/** Stable colour per hypothesis, validated for contrast on navy + white. */
export const HYP_COLORS: Record<string, string> = {
  covariate_drift: "#4f8cff",
  label_noise: "#f472b6",
  feature_leakage: "#fbbf24",
  class_prior_shift: "#34d399",
  pipeline_change: "#a78bfa",
  data_quality: "#22d3ee",
  metric_bug: "#fb923c",
  model_regression: "#94a3b8",
};
export const hypColor = (id: string) => HYP_COLORS[id] ?? "#94a3b8";

export const PLANNER_COLORS: Record<string, string> = {
  adaptive: "#4f8cff", lookahead2: "#22d3ee", eig_per_cost: "#a78bfa", max_eig: "#fbbf24",
  checklist: "#94a3b8", cheapest: "#34d399", random: "#f472b6",
};

export function statusTone(status: string): "good" | "warn" | "bad" | "accent" | "muted" {
  const map: Record<string, "good" | "warn" | "bad" | "accent" | "muted"> = {
    active: "accent", paused: "warn", completed: "good", stopped: "muted", failed: "bad", running: "accent", queued: "muted",
  };
  return map[status] ?? "muted";
}

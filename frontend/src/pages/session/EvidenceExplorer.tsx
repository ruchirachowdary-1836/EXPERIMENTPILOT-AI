import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, Microscope, ThumbsDown, ThumbsUp } from "lucide-react";
import { useMemo, useState } from "react";
import { api } from "@/api/client";
import { GroupedBars, HBars } from "@/charts/charts";
import { Badge, Callout, Card, CardHeader, EmptyState, ErrorState, Skeleton } from "@/components/ui";
import { cn, humanize, hypColor, num, pct, shortDate, title } from "@/lib/utils";
import type { Evidence, Run } from "@/types/api";
import { useSessionCtx } from "./SessionLayout";

const fmtCell = (v: unknown) => (v == null ? "—" : typeof v === "number" ? (Math.abs(v) >= 100 ? v.toFixed(0) : +v.toFixed(3)).toString() : typeof v === "boolean" ? (v ? "yes" : "no") : String(v));

export function DataTable({ rows, max = 12 }: { rows: Record<string, any>[]; max?: number }) {
  if (!rows?.length) return null;
  const cols = Object.keys(rows[0]).filter((c) => typeof rows[0][c] !== "object");
  return (
    <div className="overflow-x-auto rounded-xl border border-line/10">
      <table className="w-full text-left text-xs">
        <thead className="bg-sunken/70 text-faint"><tr>{cols.map((c) => <th key={c} className="whitespace-nowrap px-3 py-2 font-medium">{humanize(c)}</th>)}</tr></thead>
        <tbody>{rows.slice(0, max).map((r, i) => <tr key={i} className="border-t border-line/8">{cols.map((c) => <td key={c} className="num whitespace-nowrap px-3 py-1.5">{fmtCell(r[c])}</td>)}</tr>)}</tbody>
      </table>
      {rows.length > max && <p className="border-t border-line/8 px-3 py-1.5 text-[11px] text-faint">Showing {max} of {rows.length} rows</p>}
    </div>
  );
}

/** Experiment-specific charts built from the real result tables. */
export function RunCharts({ run }: { run: Run }) {
  const t = run.tables ?? {};
  switch (run.experiment_id) {
    case "feature_drift": {
      const rows = (t.features ?? []).slice(0, 12).map((r: any) => ({ feature: r.feature, psi: r.psi }));
      return <><p className="eyebrow mb-2">Population stability index by feature</p><HBars data={rows} x="feature" y="psi" name="PSI" color="#4f8cff" refLines={[{ value: 0.1, label: "moderate" }, { value: 0.25, label: "major" }]} height={Math.max(180, rows.length * 28)} /></>;
    }
    case "data_quality_audit": {
      const rows = (t.features ?? []).map((r: any) => ({ feature: r.feature, ref: r.null_ref, recent: r.null_recent }));
      return <><p className="eyebrow mb-2">Missing-value rate: reference vs recent</p><GroupedBars data={rows} x="feature" fmt={(v) => pct(v)} series={[{ key: "ref", name: "Reference", color: "#64748b" }, { key: "recent", name: "Recent", color: "#22d3ee" }]} /></>;
    }
    case "class_prior": {
      const rows = (t.classes ?? []).map((r: any) => ({ cls: String(r.class), ref: r.reference_share, recent: r.recent_share }));
      return <><p className="eyebrow mb-2">Class distribution</p><GroupedBars data={rows} x="cls" fmt={(v) => pct(v)} series={[{ key: "ref", name: "Reference", color: "#64748b" }, { key: "recent", name: "Recent", color: "#34d399" }]} /></>;
    }
    case "leakage_scan": {
      const rows = (t.features ?? []).slice(0, 10).map((r: any) => ({ feature: r.feature, ref: r.strength_reference, recent: r.strength_recent }));
      return <><p className="eyebrow mb-2">Single-feature predictive strength (2·|AUC−0.5|)</p><GroupedBars data={rows} x="feature" series={[{ key: "ref", name: "Reference", color: "#fbbf24" }, { key: "recent", name: "Recent", color: "#4f8cff" }]} /></>;
    }
    case "slice_analysis": {
      const rows = (t.slices ?? []).slice(0, 10).map((r: any) => ({ slice: `${r.feature}: ${r.slice}`, drop: r.drop }));
      return <><p className="eyebrow mb-2">Largest accuracy drops by slice</p><HBars data={rows} x="slice" y="drop" name="Accuracy drop" color="#f472b6" fmt={(v) => pct(v)} height={Math.max(180, rows.length * 30)} /></>;
    }
    case "pipeline_replay": {
      const rows = (t.features ?? []).slice(0, 10).map((r: any) => ({ feature: r.feature, diff: r.mean_abs_diff_std }));
      return <><p className="eyebrow mb-2">Mean difference between pipeline versions (in standard deviations)</p><HBars data={rows} x="feature" y="diff" name="Difference" color="#a78bfa" height={Math.max(180, rows.length * 28)} /></>;
    }
    case "importance_weighted_eval": case "retrain_recent": {
      const rows = (t.summary ?? []).map((r: any) => ({ name: r.metric ?? r.model, v: r.value ?? r.holdout_accuracy }));
      return <><p className="eyebrow mb-2">Accuracy comparison</p><HBars data={rows} x="name" y="v" name="Accuracy" fmt={(v) => pct(v, 1)} height={170} /></>;
    }
    case "metric_recompute": {
      const rows = (t.comparison ?? []).map((r: any) => ({ dataset: r.dataset, reported: r.reported, recomputed: r.recomputed }));
      return <GroupedBars data={rows} x="dataset" fmt={(v) => v.toFixed(3)} series={[{ key: "reported", name: "Reported", color: "#64748b" }, { key: "recomputed", name: "Recomputed", color: "#fb923c" }]} height={200} />;
    }
    case "label_noise": {
      const rows = (t.rates ?? []).map((r: any) => ({ dataset: r.dataset, rate: r.suspicious_label_rate }));
      return <><p className="eyebrow mb-2">Suspicious-label rate (confident learning)</p><HBars data={rows} x="dataset" y="rate" name="Rate" color="#f472b6" fmt={(v) => pct(v, 1)} height={140} /></>;
    }
    default: return null;
  }
}

function Direction({ ev, runId }: { ev: Evidence["direction"]; runId: string }) {
  const rows = Object.entries(ev).map(([h, list]) => ({ h, e: list.find((x) => x.run_id === runId) })).filter((x) => x.e && x.e.direction !== "neutral");
  if (!rows.length) return <p className="text-xs text-muted">This outcome is not strongly diagnostic for any single hypothesis.</p>;
  return (
    <div className="flex flex-wrap gap-2">
      {rows.map(({ h, e }) => (
        <span key={h} className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs", e!.direction === "supports" ? "border-good/30 bg-good/10 text-good" : "border-bad/30 bg-bad/10 text-bad")}>
          {e!.direction === "supports" ? <ThumbsUp className="h-3 w-3" /> : <ThumbsDown className="h-3 w-3" />}
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: hypColor(h) }} />{title(h)} <span className="num opacity-70">LR {e!.likelihood_ratio.toFixed(2)}</span>
        </span>))}
    </div>
  );
}

export default function EvidenceExplorer() {
  const { session: s } = useSessionCtx();
  const q = useQuery({ queryKey: ["evidence", s.id, s.version], queryFn: () => api.evidence(s.id) });
  const runs = useMemo(() => (q.data?.runs ?? []).filter((r) => r.status === "completed" || r.status === "external" || r.status === "failed"), [q.data]);
  const [sel, setSel] = useState<string | null>(null);
  if (q.isLoading) return <Skeleton className="h-96" />;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  if (!runs.length) return <EmptyState icon={<Microscope className="h-6 w-6" />} title="No evidence yet" body="Experiment outputs, statistics and charts appear here after the first run." />;
  const run = runs.find((r) => r.id === sel) ?? runs[runs.length - 1];
  const sym = q.data!.symptoms;
  return (
    <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
      <Card className="h-fit">
        <CardHeader title="Diagnostic history" subtitle="Timestamped, newest last" />
        <ol className="relative space-y-1 p-3">
          {runs.map((r) => (
            <li key={r.id}>
              <button onClick={() => setSel(r.id)} className={cn("w-full rounded-xl border p-3 text-left transition", run.id === r.id ? "border-accent/50 bg-accent/10" : "border-transparent hover:bg-line/8")}>
                <div className="flex items-center justify-between gap-2"><span className="text-sm font-medium">{r.step ? `${r.step}. ` : ""}{r.name}</span>{r.status === "failed" ? <Badge tone="bad">failed</Badge> : <Badge tone={r.outcome_index ? (r.outcome_index > 1 ? "bad" : "warn") : "good"}>{humanize(r.outcome)}</Badge>}</div>
                <p className="num mt-1 text-[11px] text-faint">{shortDate(r.finished_at ?? r.started_at)} · {r.source}</p>
              </button>
            </li>))}
        </ol>
        {sym.accuracy_reference != null && <div className="border-t border-line/10 p-4 text-xs text-muted">Symptom: accuracy {pct(sym.accuracy_reference, 1)} → {pct(sym.accuracy_recent, 1)}</div>}
      </Card>

      <div className="space-y-6" data-testid="evidence-detail">
        <Card className="card-pad">
          <div className="flex flex-wrap items-start justify-between gap-3"><div><div className="eyebrow">Experiment output</div><h2 className="mt-1 text-xl">{run.name}</h2></div>
            <div className="flex gap-2"><Badge tone="muted">{run.source}</Badge>{run.runtime_s != null && <Badge>{run.runtime_s.toFixed(1)}s</Badge>}<Badge tone="accent">score {num(run.score, 3)}</Badge></div></div>
          {run.error && <div className="mt-4"><Callout tone="bad" title="Experiment failed">{run.error}</Callout></div>}
          {run.interpretation && <p className="mt-4 text-sm leading-relaxed">{run.interpretation}</p>}
          <div className="mt-4"><div className="eyebrow mb-2">Supporting & contradicting</div><Direction ev={q.data!.direction} runId={run.id} /></div>
          {run.recommendation?.eig != null && <p className="num mt-4 text-xs text-muted">At selection: EIG {num(run.recommendation.eig, 3)} bits · score {num(run.recommendation.score, 3)} · separation {num(run.recommendation.separation, 2)}</p>}
        </Card>

        {run.findings.length > 0 && (
          <Card><CardHeader title="Findings" /><ul className="space-y-2 p-5 pt-3">{run.findings.map((f, i) => (
            <li key={i} className="flex gap-3 text-sm">{f.level === "alert" ? <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-bad" /> : f.level === "warning" ? <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" /> : <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-accent" />}<span className="text-muted">{f.text}</span></li>))}</ul></Card>)}

        {run.tables && Object.keys(run.tables).length > 0 && (
          <Card><CardHeader title="Charts & tables" subtitle="Computed from your uploaded data" />
            <div className="space-y-6 p-5"><RunCharts run={run} />
              {Object.entries(run.tables).map(([name, rows]) => rows?.length ? <div key={name}><p className="eyebrow mb-2">{humanize(name)}</p><DataTable rows={rows} /></div> : null)}</div></Card>)}

        <Card><CardHeader title="Raw numerical results" />
          <div className="p-5"><dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">{Object.entries(run.stats).filter(([, v]) => typeof v !== "object").map(([k, v]) => (
            <div key={k} className="rounded-xl bg-sunken/60 p-3"><dt className="text-[11px] text-faint">{humanize(k)}</dt><dd className="num mt-0.5 text-sm">{fmtCell(v)}</dd></div>))}</dl></div></Card>

        {run.limitations.length > 0 && <Callout tone="warn" title="Limitations of this indicator"><ul className="list-disc space-y-1 pl-4">{run.limitations.map((l) => <li key={l}>{l}</li>)}</ul></Callout>}
      </div>
    </div>
  );
}

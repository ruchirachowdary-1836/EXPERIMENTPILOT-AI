import { useQuery } from "@tanstack/react-query";
import { Download, FileJson, FileSpreadsheet, FileText, Lightbulb, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { api } from "@/api/client";
import { BeliefEvolution } from "@/charts/charts";
import { useToast } from "@/components/Toast";
import { Badge, Button, Callout, Card, CardHeader, ErrorState, ProgressBar, Skeleton } from "@/components/ui";
import { cn, humanize, hypColor, num, pct, title } from "@/lib/utils";
import { useSessionCtx } from "./SessionLayout";

export function ExportButtons({ sessionId }: { sessionId: string }) {
  const toast = useToast(); const [busy, setBusy] = useState<string | null>(null);
  const go = async (kind: Parameters<typeof api.download>[1]) => {
    setBusy(kind);
    try { await api.download(sessionId, kind); toast.success(`Downloaded ${kind}`); } catch (e) { toast.error(e, "Download failed"); } finally { setBusy(null); }
  };
  return (
    <div className="flex flex-wrap gap-2">
      <Button variant="primary" loading={busy === "report.pdf"} icon={<FileText className="h-4 w-4" />} onClick={() => go("report.pdf")}>PDF report</Button>
      <Button loading={busy === "report.json"} icon={<FileJson className="h-4 w-4" />} onClick={() => go("report.json")}>JSON</Button>
      <Button loading={busy === "hypotheses.csv"} icon={<FileSpreadsheet className="h-4 w-4" />} onClick={() => go("hypotheses.csv")}>Hypotheses CSV</Button>
      <Button loading={busy === "experiments.csv"} icon={<Download className="h-4 w-4" />} onClick={() => go("experiments.csv")}>Experiments CSV</Button>
      <Button loading={busy === "history.csv"} icon={<Download className="h-4 w-4" />} onClick={() => go("history.csv")}>Posterior history CSV</Button>
    </div>
  );
}

const STATUS_TONE = { likely: "good", probable: "accent", possible: "warn", unlikely: "muted" } as const;

export default function ReportPage() {
  const { session: s } = useSessionCtx();
  const q = useQuery({ queryKey: ["report", s.id, s.version], queryFn: () => api.report(s.id) });
  if (q.isLoading) return <div className="space-y-4"><Skeleton className="h-32" /><Skeleton className="h-96" /></div>;
  if (q.isError || !q.data) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const r = q.data; const c = r.confidence;
  return (
    <div className="space-y-6">
      <Card className="card-pad">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0 max-w-3xl"><div className="eyebrow">Final diagnosis</div><h2 className="mt-2 text-2xl leading-snug" data-testid="verdict">{r.verdict}</h2>
            <p className="mt-2 text-sm text-muted">Decision rule: a cause is diagnosed when its marginal probability is at least 50%. The single most probable exact combination ({c.map_state.length ? c.map_state.map(title).join(" + ") : "no fault"}) has probability <b className="num text-fg">{pct(c.map_probability, 1)}</b> (threshold {pct(c.threshold)}, {c.reached ? "reached" : "not reached"}).</p>
            {r.session.stop_message && <p className="mt-1 text-sm text-muted">Stopped: {r.session.stop_message}</p>}</div>
          <ExportButtons sessionId={s.id} />
        </div>
        {r.demo_truth && <div className="mt-5"><Callout tone={r.diagnosed.length && r.demo_truth.every((t) => r.diagnosed.includes(t)) && r.diagnosed.every((d) => r.demo_truth!.includes(d)) ? "good" : "info"} title="Simulated scenario ground truth">
          Injected: {r.demo_truth.length ? r.demo_truth.map(title).join(" + ") : "no fault (control)"}. Compare with the diagnosis above.</Callout></div>}
      </Card>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[["Total cost", `${num(r.cost.spent, 1)} / ${r.cost.budget}`, `${pct(r.cost.utilization)} of budget · ${r.cost.experiments_run} experiments`],
          ["Residual uncertainty", `${num(c.entropy_bits, 2)} bits`, `${pct(c.uncertainty_reduction)} below the prior (${num(c.initial_entropy_bits, 2)})`],
          ["Risk incurred", num(r.risk.total, 2), `max single ${num(r.risk.max, 2)} · cap ${num(r.risk.cap, 2)} · violations ${r.risk.violations}`],
          ["Undecided causes", String(c.undecided_hypotheses.length), c.undecided_hypotheses.map(title).join(", ") || "all causes decided"]].map(([k, v, h]) => (
          <Card key={k} className="p-5"><div className="eyebrow">{k}</div><div className="num mt-2 text-2xl font-medium">{v}</div><p className="mt-2 text-xs text-muted">{h}</p></Card>))}
      </div>

      <Card><CardHeader title="Ranked root-cause hypotheses" subtitle="Posterior marginal probability each cause is active." />
        <div className="space-y-3 p-5">
          {r.ranking.map((h, i) => (
            <div key={h.hypothesis} className="rounded-xl border border-line/10 p-4">
              <div className="flex flex-wrap items-center gap-3"><span className="num w-5 text-sm text-faint">{i + 1}</span><span className="h-3 w-3 rounded-full" style={{ background: hypColor(h.hypothesis) }} />
                <div className="min-w-0 flex-1"><p className="text-sm font-medium">{h.code} · {h.name}</p><p className="text-xs text-muted">{h.description}</p></div>
                <Badge tone={STATUS_TONE[h.status]}>{h.status}</Badge><span className="num w-16 text-right text-lg">{pct(h.posterior, 1)}</span></div>
              <div className="mt-3"><ProgressBar value={h.posterior} label={`${h.name} posterior`} /></div>
              <p className="num mt-2 text-[11px] text-faint">prior {pct(h.prior)} → posterior {pct(h.posterior)} ({h.change >= 0 ? "+" : ""}{(h.change * 100).toFixed(1)} pts)</p>
              {(h.supporting.length > 0 || h.contradicting.length > 0) && (
                <div className="mt-3 grid gap-3 text-xs sm:grid-cols-2">
                  <div><p className="mb-1 font-medium text-good">Supporting evidence</p>{h.supporting.length ? h.supporting.map((e) => <p key={e.run_id} className="text-muted">{e.experiment}: {humanize(e.outcome)} <span className="num">(LR {e.likelihood_ratio.toFixed(1)})</span></p>) : <p className="text-faint">none</p>}</div>
                  <div><p className="mb-1 font-medium text-bad">Contradicting evidence</p>{h.contradicting.length ? h.contradicting.map((e) => <p key={e.run_id} className="text-muted">{e.experiment}: {humanize(e.outcome)} <span className="num">(LR {e.likelihood_ratio.toFixed(2)})</span></p>) : <p className="text-faint">none</p>}</div>
                </div>)}
            </div>))}
        </div>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card><CardHeader title="Belief evolution" /><div className="p-4">{r.posterior_history.length > 1 ? <BeliefEvolution history={r.posterior_history as any} hypotheses={r.ranking.slice(0, 5).map((x) => x.hypothesis)} /> : <p className="py-10 text-center text-sm text-muted">No experiments were run.</p>}</div></Card>
        <Card><CardHeader title="Alternative explanations" subtitle="Most probable joint cause sets" />
          <ul className="space-y-2 p-5">{r.top_states.map((t, i) => <li key={i} className="flex items-center justify-between rounded-xl bg-sunken/60 px-3.5 py-2.5 text-sm"><span>{t.causes.map(title).join(" + ")}</span><span className="num">{pct(t.probability, 1)}</span></li>)}</ul>
          {r.alternatives.length > 0 && <p className="px-5 pb-5 text-xs text-muted">Still plausible (≥15%) but below the decision level: {r.alternatives.map((a) => `${a.name} ${pct(a.posterior)}`).join("; ")}.</p>}</Card>
      </div>

      <Card><CardHeader title="Experiments performed" subtitle="Raw evidence and interpretation" />
        <div className="divide-y divide-line/8 p-2">{r.experiments.filter((e) => e.status === "completed" || e.status === "external").map((e) => (
          <div key={e.id} className="p-3"><div className="flex flex-wrap items-center justify-between gap-2"><p className="text-sm font-medium">{e.step}. {e.name}</p><div className="flex gap-2"><Badge>{humanize(e.outcome)}</Badge><Badge tone="muted">cost {e.cost}</Badge></div></div>
            <p className="mt-1 text-sm text-muted">{e.interpretation}</p></div>))}
          {r.experiments.every((e) => e.status !== "completed" && e.status !== "external") && <p className="p-6 text-center text-sm text-muted">No experiments were executed.</p>}</div></Card>

      <Card><CardHeader icon={<Lightbulb className="h-4 w-4" />} title="Suggested remediation & validation" />
        <div className="space-y-4 p-5">{r.ranking.filter((h) => h.remediation.length).map((h) => (
          <div key={h.hypothesis}><p className="mb-1.5 flex items-center gap-2 text-sm font-medium"><span className="h-2 w-2 rounded-full" style={{ background: hypColor(h.hypothesis) }} />{h.name} <span className="num text-faint">{pct(h.posterior)}</span></p>
            <ul className="list-disc space-y-1 pl-5 text-sm text-muted">{h.remediation.map((x) => <li key={x}>{x}</li>)}</ul></div>))}
          {!r.ranking.some((h) => h.remediation.length) && <p className="text-sm text-muted">No hypothesis reached a level that justifies specific remediation. Gather more evidence first.</p>}</div></Card>

      <Card><CardHeader icon={<ShieldCheck className="h-4 w-4" />} title="Limitations & missing evidence" />
        <ul className="list-disc space-y-1.5 p-5 pl-10 text-sm text-muted">{r.limitations.map((l) => <li key={l}>{l}</li>)}</ul></Card>
      <p className="text-center text-xs text-faint">{r.disclaimer} Generated {new Date(r.generated_at).toLocaleString()}.</p>
    </div>
  );
}

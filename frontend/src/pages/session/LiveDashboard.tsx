import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, ClipboardPenLine, FastForward, ListFilter, Pause, Play, Square, Zap } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, api } from "@/api/client";
import { BeliefBars, BeliefEvolution, EntropyArea } from "@/charts/charts";
import { useToast } from "@/components/Toast";
import { AnimatedNumber, Badge, Button, Callout, Card, CardHeader, Modal, ProgressBar, Ring, Select, Skeleton } from "@/components/ui";
import { cn, humanize, num, pct, title } from "@/lib/utils";
import type { Recommendation, ScoredExperiment, Session } from "@/types/api";
import { useSessionCtx } from "./SessionLayout";

const OUTCOME_TONE = ["good", "warn", "bad"] as const;

export function NextExperimentCard({ session, rec, busy, onRun, onSkip, onExternal, onPick }: {
  session: Session; rec: Recommendation; busy: boolean; onRun: () => void; onSkip: () => void; onExternal: () => void; onPick: () => void;
}) {
  const sel = rec.experiments.find((e) => e.id === rec.selected);
  const active = session.status === "active";
  if (!sel) {
    return (
      <Card className="card-pad">
        <div className="eyebrow">Next experiment</div>
        <p className="mt-3 text-sm">{rec.stop.message || "No feasible experiment remains."}</p>
        {active && <div className="mt-4 flex gap-2"><Button size="sm" onClick={onPick} icon={<ListFilter className="h-3.5 w-3.5" />}>Browse experiments</Button><Button size="sm" onClick={onExternal} icon={<ClipboardPenLine className="h-3.5 w-3.5" />}>Enter a result</Button></div>}
      </Card>
    );
  }
  return (
    <Card className="relative overflow-hidden">
      <div className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full bg-accent/20 blur-3xl" aria-hidden />
      <div className="card-pad relative">
        <div className="flex items-center justify-between"><div className="eyebrow">Recommended next experiment</div><Badge tone="accent" dot={active}>{rec.planner_label}</Badge></div>
        <h2 className="mt-3 text-xl leading-snug" data-testid="next-name">{sel.name}</h2>
        <p className="mt-1 text-sm text-muted">{sel.description}</p>
        <dl className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[["Info gain", `${num(sel.eig, 3)} bits`], ["Cost", num(sel.cost, 2)], ["Risk", num(sel.risk, 2)], ["Separation", num(sel.separation, 2)]].map(([k, v]) => (
            <div key={k} className="rounded-xl border border-line/10 bg-sunken/60 p-3"><dt className="eyebrow">{k}</dt><dd className="num mt-1 text-lg">{v}</dd></div>
          ))}
        </dl>
        <div className="mt-4 rounded-xl border border-accent/20 bg-accent/6 p-3.5 text-[13px] leading-relaxed text-muted" data-testid="rationale"><span className="font-medium text-fg">Why this one. </span>{rec.rationale}</div>
        {sel.requires_approval || rec.needs_approval ? <div className="mt-3"><Callout tone="warn">This experiment is compute-heavy or above the risk warning level and needs your explicit approval before it runs.</Callout></div> : null}
        <div className="mt-5 flex flex-wrap gap-2">
          <Button variant="primary" disabled={!active || busy} loading={busy} onClick={onRun} icon={<Zap className="h-4 w-4" />}>Run recommended</Button>
          <Button disabled={!active || busy} onClick={onSkip} icon={<FastForward className="h-4 w-4" />}>Skip</Button>
          <Button disabled={!active || busy} onClick={onPick} icon={<ListFilter className="h-4 w-4" />}>Choose another</Button>
          <Button disabled={!active || busy} onClick={onExternal} icon={<ClipboardPenLine className="h-4 w-4" />}>Enter result</Button>
        </div>
        {!active && <p className="mt-3 text-xs text-warn">Session is {session.status}; experiment controls are disabled.</p>}
      </div>
    </Card>
  );
}

function ExperimentPicker({ open, onClose, rec, onRun, onSkip }: { open: boolean; onClose: () => void; rec: Recommendation; onRun: (e: ScoredExperiment) => void; onSkip: (e: ScoredExperiment) => void }) {
  return (
    <Modal open={open} onClose={onClose} wide title="Choose an experiment">
      <div className="space-y-2">
        {rec.experiments.map((e) => (
          <div key={e.id} className={cn("grid items-center gap-3 rounded-xl border border-line/10 p-3 sm:grid-cols-[1fr_auto]", !e.feasible && "opacity-60")}>
            <div className="min-w-0">
              <p className="flex flex-wrap items-center gap-2 text-sm font-medium">{e.name}{e.id === rec.selected && <Badge tone="accent">recommended</Badge>}{e.state === "done" && <Badge tone="good">done</Badge>}{e.state === "skipped" && <Badge>skipped</Badge>}</p>
              <p className="num mt-0.5 text-xs text-muted">EIG {num(e.eig, 3)} · cost {e.cost} · risk {e.risk} · score {num(e.score, 3)} · sep {num(e.separation, 2)}</p>
              {!e.feasible && e.state === "blocked" && <p className="mt-1 text-xs text-warn">{e.reasons.join("; ")}</p>}
            </div>
            <div className="flex gap-2"><Button size="sm" variant="primary" disabled={!(e.feasible || e.state === "skipped")} onClick={() => onRun(e)}>Run</Button><Button size="sm" variant="ghost" disabled={e.state === "done"} onClick={() => onSkip(e)}>Skip</Button></div>
          </div>))}
      </div>
    </Modal>
  );
}

function ExternalModal({ open, onClose, rec, onSubmit, busy, initial }: { open: boolean; onClose: () => void; rec: Recommendation; onSubmit: (exp: string, outcome: string, note: string) => void; busy: boolean; initial?: string }) {
  const options = rec.experiments.filter((e) => e.state !== "done");
  const [exp, setExp] = useState(initial ?? options[0]?.id ?? ""); const cur = options.find((e) => e.id === exp) ?? options[0];
  const [outcome, setOutcome] = useState(""); const [note, setNote] = useState("");
  return (
    <Modal open={open} onClose={onClose} title="Enter an externally obtained outcome"
      footer={<><Button variant="ghost" onClick={onClose}>Cancel</Button><Button variant="primary" loading={busy} disabled={!cur || !outcome} onClick={() => cur && onSubmit(cur.id, outcome, note)}>Update beliefs</Button></>}>
      <div className="space-y-4">
        <p className="text-sm text-muted">Ran the experiment yourself? Record how it came out. It is treated as noisy evidence and counts against the budget.</p>
        <label className="block text-[13px] font-medium" htmlFor="ext-exp">Experiment</label>
        <Select id="ext-exp" value={cur?.id ?? ""} onChange={(e) => { setExp(e.target.value); setOutcome(""); }}>{options.map((e) => <option key={e.id} value={e.id}>{e.name}</option>)}</Select>
        <fieldset><legend className="mb-2 text-[13px] font-medium">Observed outcome</legend>
          <div className="grid gap-2">{(cur?.outcomes ?? []).map((o, i) => (
            <label key={o} className={cn("flex cursor-pointer items-center gap-3 rounded-xl border p-3 text-sm transition", outcome === o ? "border-accent bg-accent/10" : "border-line/15 hover:border-line/30")}>
              <input type="radio" name="outcome" className="accent-[rgb(var(--accent))]" checked={outcome === o} onChange={() => setOutcome(o)} />
              <span className="font-medium capitalize">{humanize(o)}</span><span className="ml-auto text-xs text-faint">level {i}</span></label>))}</div>
        </fieldset>
        <label className="block text-[13px] font-medium" htmlFor="ext-note">Note (optional)</label>
        <textarea id="ext-note" value={note} onChange={(e) => setNote(e.target.value)} maxLength={500} className="min-h-[70px] w-full rounded-xl border border-line/15 bg-sunken/70 p-3 text-sm" />
      </div>
    </Modal>
  );
}

export default function LiveDashboard() {
  const { session: s, setSession, refetch } = useSessionCtx(); const toast = useToast();
  const hist = useQuery({ queryKey: ["history", s.id], queryFn: () => api.history(s.id) });
  const [picker, setPicker] = useState(false); const [ext, setExt] = useState(false); const [approve, setApprove] = useState<{ id?: string; name: string } | null>(null);
  const rec = s.recommendation;
  const onErr = (e: unknown) => {
    if (e instanceof ApiError && e.code === "approval_required") { const id = (e.extra?.experiment_id as string) ?? rec?.selected ?? undefined; setApprove({ id, name: rec?.experiments.find((x) => x.id === id)?.name ?? "this experiment" }); return; }
    toast.error(e, "Action failed"); refetch();
  };
  const run = useMutation({ mutationFn: (v: { id?: string; approve?: boolean }) => api.run(s.id, { experiment_id: v.id, approve: v.approve }),
    onSuccess: (r) => { setSession(r.session); setApprove(null); setPicker(false); toast.success(`${r.run.name}: ${humanize(r.run.outcome)}`, r.run.interpretation.slice(0, 160)); }, onError: onErr });
  const skip = useMutation({ mutationFn: (id?: string) => api.skip(s.id, { experiment_id: id }), onSuccess: (r) => { setSession(r.session); setPicker(false); }, onError: onErr });
  const external = useMutation({ mutationFn: (v: { id: string; outcome: string; note: string }) => api.external(s.id, { experiment_id: v.id, outcome: v.outcome, note: v.note }),
    onSuccess: (r) => { setSession(r.session); setExt(false); toast.success("Beliefs updated from your result"); }, onError: onErr });
  const ctl = useMutation({ mutationFn: (a: "pause" | "resume" | "stop") => (a === "pause" ? api.pause(s.id) : a === "resume" ? api.resume(s.id) : api.stop(s.id, "Stopped from the dashboard")),
    onSuccess: (n) => setSession(n), onError: onErr });

  if (!rec) return <Skeleton className="h-96" />;
  const priors = Object.fromEntries(s.hypotheses.map((h) => [h.id, h.prior]));
  const drop = s.symptoms.accuracy_drop as number | undefined;
  const done = s.runs.filter((r) => r.status === "completed" || r.status === "external");
  const busy = run.isPending || skip.isPending;
  const initialEntropy = hist.data?.history[0]?.entropy ?? s.entropy;
  const topP = s.marginals[0]?.probability ?? 0;

  return (
    <div className="space-y-6">
      {s.status !== "active" && s.status !== "paused" && (
        <Callout tone={s.status === "completed" ? "good" : "info"} title={s.status === "completed" ? "Diagnosis complete" : "Session stopped"}>
          {s.stop_message} <Link className="link" to="report">Open the final report</Link>
        </Callout>)}
      {s.status === "paused" && <Callout tone="warn" title="Session paused">No experiments can run until you resume.</Callout>}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Card className="flex flex-col items-start gap-3 p-4 sm:flex-row sm:items-center sm:gap-4 sm:p-5">
          <Ring value={s.map_probability} label={<AnimatedNumber value={s.map_probability * 100} format={(v) => `${v.toFixed(0)}%`} />} sub="confident" tone={s.map_probability >= (s.config.confidence_threshold ?? 0.9) ? "good" : "accent"} />
          <div className="min-w-0"><div className="eyebrow">Diagnosis confidence</div><p className="mt-1 text-xs text-muted">target {pct(s.config.confidence_threshold)}</p></div>
        </Card>
        <Card className="p-4 sm:p-5"><div className="eyebrow">Budget remaining</div>
          <div className="num mt-2 text-2xl font-medium"><AnimatedNumber value={s.remaining} format={(v) => v.toFixed(1)} /><span className="text-sm text-faint"> / {s.budget}</span></div>
          <ProgressBar className="mt-3" value={s.spent / s.budget} tone={s.remaining / s.budget < 0.2 ? "warn" : "accent"} label="Budget used" /></Card>
        <Card className="p-4 sm:p-5"><div className="eyebrow">Experiments run</div><div className="num mt-2 text-2xl font-medium">{done.length}<span className="text-sm text-faint"> / {s.experiments.length}</span></div>
          <p className="mt-3 text-xs text-muted">risk incurred {num(s.risk_total, 2)} (cap {num(s.config.risk_cap, 2)})</p></Card>
        <Card className="p-4 sm:p-5"><div className="eyebrow">Symptom</div>
          <div className={cn("num mt-2 text-2xl font-medium", drop != null && drop > 0.02 ? "text-bad" : "text-fg")}>{drop != null ? `${drop > 0 ? "−" : "+"}${Math.abs(drop * 100).toFixed(1)} pts` : "n/a"}</div>
          <p className="mt-3 text-xs text-muted">{s.symptoms.accuracy_reference != null ? `accuracy ${pct(s.symptoms.accuracy_reference, 1)} → ${pct(s.symptoms.accuracy_recent, 1)}` : "needs labels + predictions"}</p></Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.35fr_1fr]">
        <div className="space-y-6">
          <NextExperimentCard session={s} rec={rec} busy={busy} onRun={() => run.mutate({ id: rec.selected ?? undefined })} onSkip={() => skip.mutate(rec.selected ?? undefined)} onExternal={() => setExt(true)} onPick={() => setPicker(true)} />
          {run.isPending && <Card className="card-pad"><p className="mb-3 text-sm font-medium">Running on a copy of your data…</p><ProgressBar indeterminate label="Experiment running" /><p className="mt-2 text-xs text-muted">Executed in a time-limited sandbox. Results come from real computation.</p></Card>}
          <Card><CardHeader title="Root-cause probabilities" subtitle="Marginal probability that each cause is active (bars) versus the prior (thin markers)." />
            <div className="p-4 sm:p-5"><BeliefBars marginals={s.marginals} priors={priors} /></div></Card>
          <Card><CardHeader title="Belief evolution" subtitle="Posterior marginals after each experiment." />
            <div className="p-4 sm:p-5">{hist.data && hist.data.history.length > 1 ? <BeliefEvolution history={hist.data.history} hypotheses={s.marginals.slice(0, 5).map((m) => m.hypothesis)} /> : <p className="py-10 text-center text-sm text-muted">Run an experiment to see beliefs evolve.</p>}</div></Card>
        </div>

        <div className="space-y-6">
          <Card className="card-pad">
            <div className="flex items-center justify-between"><h3 className="text-[15px]">Session controls</h3></div>
            <div className="mt-4 grid grid-cols-2 gap-2">
              {s.status === "paused" ? <Button variant="primary" loading={ctl.isPending} onClick={() => ctl.mutate("resume")} icon={<Play className="h-4 w-4" />}>Resume</Button>
                : <Button disabled={s.status !== "active"} loading={ctl.isPending} onClick={() => ctl.mutate("pause")} icon={<Pause className="h-4 w-4" />}>Pause</Button>}
              <Button variant="danger" disabled={s.status === "completed" || s.status === "stopped"} onClick={() => { if (confirm("Stop this session? It cannot be resumed.")) ctl.mutate("stop"); }} icon={<Square className="h-4 w-4" />}>Stop</Button>
            </div>
            <div className="mt-5 space-y-2 text-xs text-muted">
              <div className="flex justify-between"><span>Uncertainty</span><span className="num text-fg">{num(s.entropy, 2)} bits <span className="text-faint">(from {num(initialEntropy, 2)})</span></span></div>
              <div className="flex justify-between"><span>Leading cause</span><span className="num text-fg">{title(s.marginals[0]?.hypothesis ?? "")} {pct(topP)}</span></div>
              <div className="flex justify-between"><span>Most likely set</span><span className="text-right text-fg">{s.map_state.length ? s.map_state.map(title).join(" + ") : "no fault"}</span></div>
            </div>
            {hist.data && hist.data.history.length > 1 && <div className="mt-3"><EntropyArea history={hist.data.history} /></div>}
          </Card>

          <Card><CardHeader title="Experiment log" subtitle={`${done.length} completed`} />
            <ol className="space-y-3 p-5 pt-4">
              {s.runs.length === 0 && <li className="py-6 text-center text-sm text-muted">Nothing has run yet.</li>}
              {s.runs.map((r) => (
                <li key={r.id} className="rounded-xl border border-line/10 bg-sunken/50 p-3.5">
                  <div className="flex items-start justify-between gap-2"><p className="text-sm font-medium">{r.step ? `${r.step}. ` : ""}{r.name}</p>
                    {r.status === "skipped" ? <Badge>skipped</Badge> : r.status === "failed" ? <Badge tone="bad">failed</Badge> : r.status === "running" ? <Badge tone="accent" dot>running</Badge> :
                      <Badge tone={OUTCOME_TONE[Math.min(r.outcome_index ?? 0, 2)]}>{humanize(r.outcome)}</Badge>}</div>
                  {r.interpretation && <p className="mt-1.5 line-clamp-3 text-xs leading-relaxed text-muted">{r.interpretation}</p>}
                  {r.error && <p className="mt-1.5 text-xs text-bad">{r.error}</p>}
                  <p className="num mt-2 text-[11px] text-faint">{r.source}{r.status === "completed" || r.status === "external" ? ` · cost ${r.cost}` : ""}{r.recommendation?.chosen_by === "user" ? " · chosen by you" : ""}</p>
                </li>))}
            </ol>
          </Card>
        </div>
      </div>

      <ExperimentPicker open={picker} onClose={() => setPicker(false)} rec={rec} onRun={(e) => run.mutate({ id: e.id })} onSkip={(e) => skip.mutate(e.id)} />
      {ext && <ExternalModal open={ext} onClose={() => setExt(false)} rec={rec} busy={external.isPending} initial={rec.selected ?? undefined} onSubmit={(id, outcome, note) => external.mutate({ id, outcome, note })} />}
      <Modal open={!!approve} onClose={() => setApprove(null)} title="Approval required"
        footer={<><Button variant="ghost" onClick={() => setApprove(null)}>Cancel</Button><Button variant="primary" loading={run.isPending} onClick={() => run.mutate({ id: approve?.id, approve: true })} icon={<CheckCircle2 className="h-4 w-4" />}>Approve and run</Button></>}>
        <div className="flex gap-3"><AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-warn" /><p className="text-sm text-muted"><b className="text-fg">{approve?.name}</b> is compute-heavy or above the risk warning level (for example retraining a surrogate model). It runs on a copy of your data, never on production, but needs your explicit approval.</p></div>
      </Modal>
    </div>
  );
}

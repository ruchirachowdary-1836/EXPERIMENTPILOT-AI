import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertTriangle, Gauge, ListChecks, Rocket, ShieldAlert, SlidersHorizontal } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "@/api/client";
import { useToast } from "@/components/Toast";
import { Badge, Button, Callout, Card, CardHeader, ErrorState, Field, Input, PageHeader, Select, Skeleton, Slider, Switch } from "@/components/ui";
import { cn, hypColor, pct } from "@/lib/utils";
import type { ExperimentMeta, HypothesisMeta, Project } from "@/types/api";

export interface SetupState {
  name: string; planner: string; budget: number; riskCap: number; riskWeight: number; tau: number; epsilon: number; sepWeight: number;
  confidenceMode: "joint" | "marginal"; maxActive: number; humanInLoop: boolean;
  hyps: Record<string, { enabled: boolean; prior: number }>; exps: Record<string, boolean>;
  metric: { on: boolean; reference: string; recent: string; metric: "accuracy" | "macro_f1" };
}

/** Client-side validation mirroring the server rules so users get instant feedback. */
export function validateSetup(s: SetupState, available: Set<string>, experiments: ExperimentMeta[]): { errors: string[]; warnings: string[] } {
  const errors: string[] = [], warnings: string[] = [];
  if (!s.name.trim()) errors.push("Name the session.");
  if (!(s.budget > 0)) errors.push("The diagnostic budget must be greater than zero.");
  if (s.riskCap < 0 || s.riskCap > 1) errors.push("The risk ceiling must be between 0 and 1.");
  if (s.tau <= 0 || s.tau > 1) errors.push("The confidence threshold must be in (0, 1].");
  const hs = Object.entries(s.hyps).filter(([, h]) => h.enabled);
  if (hs.length < 2) errors.push("Enable at least two candidate root causes.");
  if (hs.length > 12) errors.push("At most 12 hypotheses are supported.");
  for (const [id, h] of hs) if (!(h.prior > 0 && h.prior < 1)) errors.push(`Prior for ${id.replace(/_/g, " ")} must be strictly between 0 and 1.`);
  const on = experiments.filter((e) => s.exps[e.id]);
  if (!on.length) errors.push("Enable at least one experiment.");
  const feasible = on.filter((e) => s.humanInLoop || e.requires.every((r) => available.has(r) || (r === "reported_metrics" && s.metric.on)));
  if (on.length && !feasible.length) errors.push("None of the enabled experiments can run with the data supplied. Upload labels/predictions or enable human-in-the-loop mode.");
  if (on.length && feasible.length) {
    const affordable = feasible.filter((e) => e.cost <= s.budget && (e.risk <= s.riskCap));
    if (!affordable.length) errors.push("No enabled experiment fits the budget and risk ceiling.");
  }
  if (s.metric.on) {
    for (const k of ["reference", "recent"] as const) { const v = Number(s.metric[k]); if (!(v >= 0 && v <= 1) || s.metric[k] === "") errors.push(`Reported ${k} metric must be a number between 0 and 1.`); }
  }
  if (s.budget < 3 && !errors.length) warnings.push("A very small budget limits the planner to one or two cheap experiments.");
  if (s.tau > 0.97 && s.confidenceMode === "joint") warnings.push("A joint-state confidence this high is rarely reachable; the session will likely end by budget.");
  if (hs.length > 8) warnings.push("With many hypotheses, joint states are truncated to at most " + s.maxActive + " simultaneous causes.");
  return { errors, warnings };
}

export function availableInputs(p: Project): Set<string> {
  const a = new Set<string>(["reference", "recent", "features"]);
  const m = p.config.mapping ?? {};
  if (m.target) { a.add("labels_reference"); a.add("labels_recent"); }
  if (m.prediction) { a.add("predictions_reference"); a.add("predictions_recent"); }
  if (p.datasets.some((d) => d.role === "replay")) a.add("replay");
  if (p.config.reported_metrics) a.add("reported_metrics");
  return a;
}

export default function DiagnosticSetup() {
  const { id = "" } = useParams(); const nav = useNavigate(); const toast = useToast();
  const proj = useQuery({ queryKey: ["project", id], queryFn: () => api.project(id) });
  const hm = useQuery({ queryKey: ["hyp-meta"], queryFn: api.hypotheses });
  const em = useQuery({ queryKey: ["exp-meta"], queryFn: api.experiments });
  const pm = useQuery({ queryKey: ["planner-meta"], queryFn: api.planners });
  const [s, setS] = useState<SetupState | null>(null);

  const init = useMemo(() => {
    if (!proj.data || !hm.data || !em.data) return null;
    const rm = proj.data.config.reported_metrics;
    return {
      name: `Diagnosis ${new Date().toLocaleDateString()}`, planner: "adaptive", budget: proj.data.config.default_budget ?? 8, riskCap: proj.data.config.default_risk_cap ?? 0.3, riskWeight: 1,
      tau: 0.9, epsilon: 0.01, sepWeight: 0, confidenceMode: "joint" as const, maxActive: 3, humanInLoop: false,
      hyps: Object.fromEntries(hm.data.map((h) => [h.id, { enabled: h.default_enabled, prior: h.default_prior }])),
      exps: Object.fromEntries(em.data.map((e) => [e.id, true])),
      metric: { on: !!rm, reference: rm ? rm.reference.toFixed(4) : "", recent: rm ? rm.recent.toFixed(4) : "", metric: (rm?.metric ?? "accuracy") as "accuracy" | "macro_f1" },
    } satisfies SetupState;
  }, [proj.data, hm.data, em.data]);
  const st = s ?? init;

  const start = useMutation({
    mutationFn: () => api.createSession({
      project_id: id, name: st!.name, planner_variant: st!.planner, budget: st!.budget, risk_cap: st!.riskCap, risk_weight: st!.riskWeight,
      confidence_threshold: st!.tau, min_score: st!.epsilon, separation_weight: st!.sepWeight, confidence_mode: st!.confidenceMode, max_active: st!.maxActive,
      human_in_the_loop: st!.humanInLoop,
      hypotheses: Object.entries(st!.hyps).map(([hid, h]) => ({ id: hid, enabled: h.enabled, prior: h.prior })),
      experiments: Object.entries(st!.exps).filter(([, v]) => v).map(([k]) => k),
      reported_metrics: st!.metric.on ? { metric: st!.metric.metric, reference: Number(st!.metric.reference), recent: Number(st!.metric.recent) } : undefined,
    }),
    onSuccess: (sess) => { toast.success("Diagnosis session started"); nav(`/app/sessions/${sess.id}`); },
    onError: (e) => toast.error(e, "Could not start the session"),
  });

  if (proj.isError) return <ErrorState error={proj.error} />;
  if (!st || !hm.data || !em.data || !proj.data) return <div className="space-y-4"><Skeleton className="h-12 w-72" /><Skeleton className="h-96" /></div>;
  const p = proj.data; const avail = availableInputs(p);
  const { errors, warnings } = validateSetup(st, avail, em.data);
  const set = (patch: Partial<SetupState>) => setS({ ...st, ...patch });
  const prior = (id: string, v: number) => set({ hyps: { ...st.hyps, [id]: { ...st.hyps[id], prior: v } } });

  return (
    <div>
      <PageHeader eyebrow={<><Link to="/app/projects" className="hover:text-fg">Projects</Link> / <Link to={`/app/projects/${id}`} className="hover:text-fg">{p.name}</Link></>} title="Diagnostic setup"
        subtitle="Choose the candidate root causes, their priors, the available experiments and the limits the planner must respect." />
      <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
        <div className="space-y-6">
          <Card>
            <CardHeader icon={<ListChecks className="h-4 w-4" />} title="Candidate root causes & priors" subtitle="Prior = probability the cause is active before any experiment. Several can be active at once." />
            <div className="divide-y divide-line/8 p-2 sm:p-3">
              {(hm.data as HypothesisMeta[]).map((h) => {
                const cur = st.hyps[h.id];
                return (
                  <div key={h.id} className={cn("grid items-center gap-x-4 gap-y-2 rounded-xl p-3 sm:grid-cols-[auto_1fr_220px]", !cur.enabled && "opacity-55")}>
                    <Switch checked={cur.enabled} onChange={(v) => set({ hyps: { ...st.hyps, [h.id]: { ...cur, enabled: v } } })} label={`Enable ${h.name}`} />
                    <div className="min-w-0"><p className="flex items-center gap-2 text-sm font-medium"><span className="h-2.5 w-2.5 rounded-full" style={{ background: hypColor(h.id) }} />{h.code} · {h.name}{!h.core && <Badge tone="muted">extensible</Badge>}</p><p className="text-xs text-muted">{h.description}</p></div>
                    <Slider label="Prior" id={`prior-${h.id}`} value={cur.prior} min={0.01} max={0.95} step={0.01} onChange={(v) => prior(h.id, v)} format={(v) => pct(v)} />
                  </div>);
              })}
            </div>
          </Card>

          <Card>
            <CardHeader icon={<SlidersHorizontal className="h-4 w-4" />} title="Experiments & required data" subtitle="Availability reflects the data and column mapping of this project." />
            <div className="divide-y divide-line/8 p-2 sm:p-3">
              {(em.data as ExperimentMeta[]).map((e) => {
                const missing = e.requires.filter((r) => !avail.has(r) && !(r === "reported_metrics" && st.metric.on));
                const blocked = missing.length > 0 && !st.humanInLoop;
                return (
                  <div key={e.id} className="grid items-center gap-3 rounded-xl p-3 sm:grid-cols-[auto_1fr_auto]">
                    <Switch checked={st.exps[e.id]} onChange={(v) => set({ exps: { ...st.exps, [e.id]: v } })} label={`Enable ${e.name}`} />
                    <div className="min-w-0"><p className="text-sm font-medium">{e.name}</p><p className="text-xs text-muted">{e.description}</p>
                      {blocked && <p className="mt-1 flex items-center gap-1 text-xs text-warn"><AlertTriangle className="h-3 w-3" />Needs: {missing.map((m) => m.replace(/_/g, " ")).join(", ")}</p>}</div>
                    <div className="flex flex-wrap items-center gap-1.5 sm:justify-end"><Badge>cost {e.cost}</Badge><Badge tone={e.risk > 0.15 ? "warn" : "muted"}>risk {e.risk}</Badge>{e.requires_approval && <Badge tone="warn">approval</Badge>}</div>
                  </div>);
              })}
            </div>
            <div className="flex items-center justify-between gap-4 border-t border-line/10 p-5 sm:px-6">
              <div><p className="text-sm font-medium">Human-in-the-loop mode</p><p className="text-xs text-muted">Recommend experiments even when data is missing; you run them yourself and enter the result.</p></div>
              <Switch checked={st.humanInLoop} onChange={(v) => set({ humanInLoop: v })} label="Human-in-the-loop" />
            </div>
          </Card>

          <Card>
            <CardHeader icon={<Gauge className="h-4 w-4" />} title="Reported metrics (optional)" subtitle="Enables the independent metric-recomputation experiment, which checks for evaluation bugs." />
            <div className="space-y-4 p-5 sm:p-6">
              <div className="flex items-center gap-3"><Switch checked={st.metric.on} onChange={(v) => set({ metric: { ...st.metric, on: v } })} label="Provide reported metrics" /><span className="text-sm">I have reported metric values to verify</span></div>
              {st.metric.on && (
                <div className="grid gap-4 sm:grid-cols-3">
                  <Field label="Metric" htmlFor="m-metric"><Select id="m-metric" value={st.metric.metric} onChange={(e) => set({ metric: { ...st.metric, metric: e.target.value as "accuracy" | "macro_f1" } })}><option value="accuracy">Accuracy</option><option value="macro_f1">Macro F1</option></Select></Field>
                  <Field label="Reported (reference)" htmlFor="m-ref"><Input id="m-ref" inputMode="decimal" value={st.metric.reference} onChange={(e) => set({ metric: { ...st.metric, reference: e.target.value } })} /></Field>
                  <Field label="Reported (recent)" htmlFor="m-rec"><Input id="m-rec" inputMode="decimal" value={st.metric.recent} onChange={(e) => set({ metric: { ...st.metric, recent: e.target.value } })} /></Field>
                </div>)}
            </div>
          </Card>
        </div>

        <aside className="space-y-4 lg:sticky lg:top-20 lg:self-start">
          <Card className="card-pad space-y-5">
            <h3 className="text-[15px]">Planner & limits</h3>
            <Field label="Session name" htmlFor="sname"><Input id="sname" value={st.name} onChange={(e) => set({ name: e.target.value })} /></Field>
            <Field label="Planner" htmlFor="planner" hint={pm.data?.find((x) => x.id === st.planner)?.description}>
              <Select id="planner" value={st.planner} onChange={(e) => set({ planner: e.target.value })}>{(pm.data ?? [{ id: "adaptive", label: "Cost-and-risk-aware adaptive" }]).map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}</Select>
            </Field>
            <Slider label="Diagnostic budget" id="budget" value={st.budget} min={0.5} max={30} step={0.5} onChange={(v) => set({ budget: v })} format={(v) => `${v} units`} />
            <Slider label="Risk ceiling" id="riskcap" value={st.riskCap} min={0} max={1} step={0.05} onChange={(v) => set({ riskCap: v })} format={(v) => v.toFixed(2)} hint="Experiments above this are never run." />
            <Slider label="Risk weight (λ)" id="lambda" value={st.riskWeight} min={0} max={10} step={0.5} onChange={(v) => set({ riskWeight: v })} format={(v) => v.toFixed(1)} hint="score = EIG / (cost + λ·risk)" />
            <Slider label="Confidence threshold (τ)" id="tau" value={st.tau} min={0.5} max={0.99} step={0.01} onChange={(v) => set({ tau: v })} format={(v) => pct(v)} />
            <Slider label="Stopping threshold (ε)" id="eps" value={st.epsilon} min={0} max={0.2} step={0.005} onChange={(v) => set({ epsilon: v })} format={(v) => v.toFixed(3)} hint="Stop when the best score per cost falls below this." />
            <Slider label="Separation bonus" id="sep" value={st.sepWeight} min={0} max={2} step={0.1} onChange={(v) => set({ sepWeight: v })} format={(v) => v.toFixed(1)} />
            <Field label="Confidence measure" htmlFor="cmode" hint="Joint: P(exact set of causes). Marginal: every cause decided."><Select id="cmode" value={st.confidenceMode} onChange={(e) => set({ confidenceMode: e.target.value as "joint" | "marginal" })}><option value="joint">Joint state</option><option value="marginal">Every marginal</option></Select></Field>
            <Field label="Max simultaneous causes" htmlFor="maxa" hint="Sparsity assumption that bounds the state space."><Select id="maxa" value={st.maxActive} onChange={(e) => set({ maxActive: Number(e.target.value) })}>{[1, 2, 3, 4, 5].map((n) => <option key={n}>{n}</option>)}</Select></Field>
          </Card>
          {errors.length > 0 && <Callout tone="bad" title="Fix before starting"><ul className="list-disc pl-4">{errors.map((e) => <li key={e}>{e}</li>)}</ul></Callout>}
          {warnings.map((w) => <Callout key={w} tone="warn">{w}</Callout>)}
          <Button variant="primary" size="lg" className="w-full" disabled={errors.length > 0} loading={start.isPending} icon={<Rocket className="h-4 w-4" />} onClick={() => start.mutate()}>Start diagnosis</Button>
          <p className="flex items-start gap-2 text-xs text-muted"><ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />Experiments run on copies of your data. Retraining and other sensitive actions always ask for explicit approval.</p>
        </aside>
      </div>
    </div>
  );
}

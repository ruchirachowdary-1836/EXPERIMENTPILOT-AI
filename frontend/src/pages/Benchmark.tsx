import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, FlaskConical, Play } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/api/client";
import { BudgetCurves, CostBars, Reliability, RobustnessChart } from "@/charts/charts";
import { useToast } from "@/components/Toast";
import { Badge, Button, Callout, Card, CardHeader, EmptyState, ErrorState, Field, Input, PageHeader, ProgressBar, Select, Skeleton, Tabs } from "@/components/ui";
import { PLANNER_COLORS, cn, humanize, num, pct, statusTone, title } from "@/lib/utils";
import type { BenchmarkResults } from "@/types/api";

function Chip({ on, children, onClick }: { on: boolean; children: React.ReactNode; onClick: () => void }) {
  return <button type="button" aria-pressed={on} onClick={onClick} className={cn("rounded-full border px-3 py-1 text-xs font-medium transition", on ? "border-accent bg-accent/15 text-fg" : "border-line/20 text-muted hover:border-line/40")}>{children}</button>;
}

export function ResultsView({ res, jobId, published }: { res: BenchmarkResults; jobId?: string; published?: boolean }) {
  const labels = useMemo(() => Object.fromEntries(res.summary.map((s) => [s.planner, s.label])), [res]);
  const planners = res.summary.map((s) => s.planner);
  const [metric, setMetric] = useState<"top1" | "f1">("top1");
  const [tracePlanner, setTracePlanner] = useState("");
  const traces = useQuery({ queryKey: ["traces", jobId ?? "pub", tracePlanner], queryFn: () => (jobId ? api.benchmarkTraces(jobId, tracePlanner || undefined) : api.benchmarkPublishedRuns(tracePlanner || undefined)) });
  const cal = res.calibration?.adaptive;
  const plot = (n: string) => (jobId ? api.benchmarkPlotUrl(jobId, n) : api.publishedPlotUrl(n));
  return (
    <div className="space-y-6">
      <Callout tone="warn" title="Simulated results">{res.disclaimer}</Callout>
      <Card><CardHeader title="Planner comparison" subtitle={`${res.n_evaluation_scenarios ?? res.n_scenarios ?? "?"} evaluation scenarios · mean with 95% bootstrap CI`} />
        <div className="overflow-x-auto p-3 sm:p-5"><table className="w-full min-w-[820px] text-left text-sm">
          <thead className="text-[11px] uppercase tracking-wide text-faint"><tr>{["Planner", "Top-1", "Top-3", "Multi-cause F1", "Cost", "# exp", "Brier", "Risk-cap violations"].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}</tr></thead>
          <tbody>{res.summary.map((s) => (
            <tr key={s.planner} className={cn("border-t border-line/8", s.planner === "adaptive" && "bg-accent/6")}>
              <td className="px-3 py-2.5 font-medium"><span className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ background: PLANNER_COLORS[s.planner] }} />{s.label}</td>
              <td className="num px-3 py-2.5">{pct(s.top1, 1)} <span className="text-[11px] text-faint">[{pct(s.top1_lo)}–{pct(s.top1_hi)}]</span></td>
              <td className="num px-3 py-2.5">{pct(s.top3, 1)}</td>
              <td className="num px-3 py-2.5">{num(s.f1, 3)} <span className="text-[11px] text-faint">[{num(s.f1_lo, 2)}–{num(s.f1_hi, 2)}]</span></td>
              <td className="num px-3 py-2.5">{num(s.cost, 2)} <span className="text-[11px] text-faint">[{num(s.cost_lo, 1)}–{num(s.cost_hi, 1)}]</span></td>
              <td className="num px-3 py-2.5">{num(s.n_experiments, 2)}</td><td className="num px-3 py-2.5">{num(s.brier, 3)}</td>
              <td className={cn("num px-3 py-2.5", s.risk_violations > 0 && "text-warn")}>{num(s.risk_violations, 2)}</td></tr>))}</tbody></table></div></Card>
      <div className="grid gap-6 xl:grid-cols-2">
        <Card><CardHeader title="Accuracy at fixed budget" action={<Tabs value={metric} onChange={setMetric} tabs={[{ id: "top1", label: "Top-1" }, { id: "f1", label: "F1" }]} />} />
          <div className="p-4"><BudgetCurves rows={res.accuracy_vs_budget as any} planners={planners} labels={labels} metric={metric} /></div></Card>
        <Card><CardHeader title="Cost to diagnosis" subtitle="Stops at the confidence threshold or when the budget is exhausted" />
          <div className="p-4"><CostBars rows={res.summary.map((s) => ({ label: s.label, planner: s.planner, cost: s.cost, cost_lo: s.cost_lo, cost_hi: s.cost_hi, top1: s.top1 }))} /></div></Card>
      </div>
      <div className="grid gap-6 xl:grid-cols-2">
        {cal && <Card><CardHeader title="Calibration (adaptive planner)" subtitle={`Expected calibration error ${num(cal.ece, 3)} · Brier ${num(cal.brier, 3)}`} /><div className="p-4"><Reliability points={cal.reliability} /></div></Card>}
        {res.robustness?.length > 0 && <Card><CardHeader title="Robustness to misspecified likelihoods" subtitle="Planner beliefs use perturbed tables; outcomes are unchanged" /><div className="p-4"><RobustnessChart rows={res.robustness} labels={labels} /></div></Card>}
      </div>
      {res.paired_tests?.length > 0 && (
        <Card><CardHeader title="Paired comparisons vs adaptive" subtitle="Wilcoxon signed-rank on per-scenario differences" />
          <div className="overflow-x-auto p-3 sm:p-5"><table className="w-full min-w-[620px] text-sm"><thead className="text-[11px] uppercase tracking-wide text-faint"><tr>{["Baseline", "Metric", "n", "Mean diff (adaptive − baseline)", "p"].map((h) => <th key={h} className="px-3 py-2 text-left font-medium">{h}</th>)}</tr></thead>
            <tbody>{res.paired_tests.map((t, i) => <tr key={i} className="border-t border-line/8"><td className="px-3 py-2">{labels[t.other] ?? t.other}</td><td className="px-3 py-2">{t.metric}</td><td className="num px-3 py-2">{t.n}</td><td className="num px-3 py-2">{t.mean_difference.toFixed(3)}</td><td className={cn("num px-3 py-2", t.wilcoxon_p < 0.05 && "text-good")}>{t.wilcoxon_p < 0.001 ? "<0.001" : t.wilcoxon_p.toFixed(3)}</td></tr>)}</tbody></table>
            <p className="mt-3 px-3 text-xs text-muted">Lower cost is better; higher top-1 and F1 are better. Many scenarios share outcomes, so p-values describe this simulation only.</p></div></Card>)}
      {res.agreement && (
        <Card><CardHeader title="Do the planners actually differ?" subtitle="Share of scenarios where two planners executed the identical experiment sequence" />
          <div className="overflow-x-auto p-4"><table className="text-xs"><thead><tr><th />{res.agreement.planners.map((p) => <th key={p} className="px-2 py-1 font-medium text-faint">{labels[p] ?? p}</th>)}</tr></thead>
            <tbody>{res.agreement.matrix.map((row, i) => <tr key={i}><td className="pr-3 text-right font-medium text-faint">{labels[res.agreement!.planners[i]]}</td>{row.map((v, j) => <td key={j} className="num px-2 py-1 text-center" style={{ background: `rgb(var(--accent) / ${v * 0.5})` }}>{v.toFixed(2)}</td>)}</tr>)}</tbody></table></div></Card>)}
      {res.unseen_dataset && <Card><CardHeader title={`Held-out dataset: ${title(res.unseen_dataset.dataset)}`} subtitle="Likelihoods were estimated on other (synthetic) datasets" />
        <div className="grid gap-3 p-5 sm:grid-cols-2 lg:grid-cols-4">{res.unseen_dataset.summary.map((s) => <div key={s.planner} className="rounded-xl bg-sunken/60 p-3"><p className="text-xs text-muted">{s.label}</p><p className="num mt-1 text-sm">top-1 {pct(s.top1)} · F1 {num(s.f1, 2)} · cost {num(s.cost, 1)}</p></div>)}</div></Card>}
      {res.plots && res.plots.length > 0 && <Card><CardHeader title="Exported charts" subtitle="Static PNGs generated by the analysis script" /><div className="grid gap-4 p-5 sm:grid-cols-2">{res.plots.map((p) => <a key={p} href={plot(p)} target="_blank" rel="noreferrer" className="overflow-hidden rounded-xl border border-line/10 bg-white"><img src={plot(p)} alt={humanize(p.replace(".png", ""))} loading="lazy" className="w-full" /></a>)}</div></Card>}
      <Card><CardHeader title="Experiment traces" subtitle="The exact sequence each planner ran on each scenario" action={<Select className="w-52" aria-label="Filter planner" value={tracePlanner} onChange={(e) => setTracePlanner(e.target.value)}><option value="">All planners</option>{planners.map((p) => <option key={p} value={p}>{labels[p]}</option>)}</Select>} />
        <div className="overflow-x-auto p-3 sm:p-5">{traces.isLoading ? <Skeleton className="h-40" /> : traces.isError ? <ErrorState error={traces.error} /> :
          <table className="w-full min-w-[760px] text-xs"><thead className="text-faint"><tr>{["Scenario", "Planner", "Truth", "Top-1", "Cost", "Path"].map((h) => <th key={h} className="px-2 py-2 text-left font-medium">{h}</th>)}</tr></thead>
            <tbody>{traces.data?.items.map((r, i) => <tr key={i} className="border-t border-line/8 align-top"><td className="max-w-[210px] truncate px-2 py-1.5 font-mono" title={r.scenario}>{r.scenario}</td><td className="px-2 py-1.5">{labels[r.planner] ?? r.planner}</td><td className="px-2 py-1.5">{humanize(r.truth)}</td>
              <td className="px-2 py-1.5">{r.top1 == null ? "—" : r.top1 ? "✓" : "✗"}</td><td className="num px-2 py-1.5">{num(r.cost, 1)}</td><td className="px-2 py-1.5 text-muted">{String(r.path).split(">").map(humanize).join(" → ")}</td></tr>)}</tbody></table>}
          <p className="mt-3 px-2 text-[11px] text-faint">Showing the first {traces.data?.items.length ?? 0} of {traces.data?.total ?? 0} runs.</p></div></Card>
      {jobId && <div className="flex gap-2"><Button icon={<Download className="h-4 w-4" />} onClick={() => api.benchmarkDownload(jobId, "csv")}>Export runs (CSV)</Button><Button icon={<Download className="h-4 w-4" />} onClick={() => api.benchmarkDownload(jobId, "json")}>Export results (JSON)</Button></div>}
      {published && <p className="text-xs text-muted">Reproduce with <code className="font-mono">python research/benchmark/run_benchmark.py --config research/configs/default.json</code>. Raw per-run data: <code className="font-mono">research/results/</code>.</p>}
    </div>
  );
}

function LabForm({ onStarted }: { onStarted: (id: string) => void }) {
  const opts = useQuery({ queryKey: ["bench-opts"], queryFn: api.benchmarkOptions }); const toast = useToast(); const qc = useQueryClient();
  const [datasets, setDatasets] = useState(["synthetic_credit"]); const [faults, setFaults] = useState<string[]>([]); const [sev, setSev] = useState(["medium"]);
  const [planners, setPlanners] = useState<string[]>([]); const [seeds, setSeeds] = useState("1000, 1001, 1002"); const [budget, setBudget] = useState("8"); const [pairs, setPairs] = useState(false);
  useEffect(() => { if (opts.data && !faults.length) { setFaults(opts.data.faults); setPlanners(opts.data.planners.map((p) => p.id)); } }, [opts.data]); // eslint-disable-line
  const toggle = (list: string[], set: (v: string[]) => void, v: string) => set(list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  const seedList = seeds.split(/[,\s]+/).filter(Boolean).map(Number);
  const seedErr = !seedList.length || seedList.some((n) => !Number.isInteger(n) || n < 0) ? "Enter comma-separated non-negative integers." : seedList.length > 20 ? "At most 20 seeds." : null;
  const reserved = new Set(opts.data?.reserved_estimation_seeds ?? []); const overlap = seedList.filter((s) => reserved.has(s));
  const n = datasets.length * seedList.length * (faults.length * sev.length + 1 + (pairs ? 8 : 0));
  const start = useMutation({
    mutationFn: () => api.createBenchmark({ name: `Lab run ${new Date().toLocaleTimeString()}`, datasets, faults, severities: sev, planners, seeds: seedList, budget: Number(budget), include_pairs: pairs, budgets: [2, 4, 6, 8, 12] }),
    onSuccess: (j) => { qc.invalidateQueries({ queryKey: ["benchmarks"] }); onStarted(j.id); toast.success("Benchmark started"); }, onError: (e) => toast.error(e, "Could not start benchmark"),
  });
  if (opts.isLoading) return <Skeleton className="h-72" />;
  const disabled = !!seedErr || overlap.length > 0 || !datasets.length || !faults.length || !planners.length || !sev.length || n > 400;
  return (
    <Card><CardHeader icon={<FlaskConical className="h-4 w-4" />} title="Configure a benchmark" subtitle="Runs the real experiments on freshly injected faults, then replays every selected planner." />
      <div className="space-y-6 p-5 sm:p-6">
        <Field label="Datasets"><div className="flex flex-wrap gap-2">{opts.data?.datasets.map((d) => <Chip key={d} on={datasets.includes(d)} onClick={() => toggle(datasets, setDatasets, d)}>{title(d)}</Chip>)}</div></Field>
        <Field label="Fault injectors"><div className="flex flex-wrap gap-2">{opts.data?.faults.map((d) => <Chip key={d} on={faults.includes(d)} onClick={() => toggle(faults, setFaults, d)}>{title(d)}</Chip>)}</div></Field>
        <div className="grid gap-6 sm:grid-cols-2">
          <Field label="Severities"><div className="flex gap-2">{opts.data?.severities.map((d) => <Chip key={d} on={sev.includes(d)} onClick={() => toggle(sev, setSev, d)}>{d}</Chip>)}</div></Field>
          <Field label="Extras"><Chip on={pairs} onClick={() => setPairs(!pairs)}>Include simultaneous-fault pairs</Chip></Field>
        </div>
        <Field label="Planners"><div className="flex flex-wrap gap-2">{opts.data?.planners.map((p) => <Chip key={p.id} on={planners.includes(p.id)} onClick={() => toggle(planners, setPlanners, p.id)}>{p.label}</Chip>)}</div></Field>
        <div className="grid gap-6 sm:grid-cols-2">
          <Field label="Random seeds" htmlFor="seeds" error={seedErr ?? (overlap.length ? `Seeds ${overlap.join(", ")} were used to estimate the likelihood tables. Pick others to keep evaluation data separate.` : null)} hint="Evaluation seeds must not overlap the estimation seeds."><Input id="seeds" value={seeds} onChange={(e) => setSeeds(e.target.value)} invalid={!!seedErr || overlap.length > 0} /></Field>
          <Field label="Diagnostic budget" htmlFor="bbudget"><Input id="bbudget" type="number" min={1} max={100} value={budget} onChange={(e) => setBudget(e.target.value)} /></Field>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line/10 pt-5"><p className="text-sm text-muted"><b className="num text-fg">{n}</b> scenarios{n > 400 && <span className="text-bad"> (limit 400)</span>} · roughly {Math.max(1, Math.round((n * 1.2) / 2 / 60))}+ min</p>
          <Button variant="primary" disabled={disabled} loading={start.isPending} icon={<Play className="h-4 w-4" />} onClick={() => start.mutate()}>Start benchmark</Button></div>
      </div>
    </Card>
  );
}

function JobPanel({ id }: { id: string }) {
  const q = useQuery({ queryKey: ["benchmark", id], queryFn: () => api.benchmark(id), refetchInterval: (qq) => (qq.state.data && ["queued", "running"].includes(qq.state.data.status) ? 1500 : false) });
  const res = useQuery({ queryKey: ["benchmark-results", id], queryFn: () => api.benchmarkResults(id), enabled: q.data?.status === "completed" });
  if (q.isLoading) return <Skeleton className="h-40" />;
  if (q.isError) return <ErrorState error={q.error} />;
  const j = q.data!;
  return (
    <div className="space-y-6">
      <Card className="card-pad"><div className="flex items-center justify-between gap-3"><div><p className="font-medium">{j.name}</p><p className="text-xs text-muted">{j.message}</p></div><Badge tone={statusTone(j.status)} dot={j.status === "running"}>{j.status}</Badge></div>
        {(j.status === "running" || j.status === "queued") && <ProgressBar className="mt-4" value={j.progress} label="Benchmark progress" />}
        {j.error && <div className="mt-4"><Callout tone="bad" title="Benchmark failed">{j.error}</Callout></div>}</Card>
      {res.data && <ResultsView res={res.data} jobId={id} />}
    </div>
  );
}

export default function Benchmark() {
  const [tab, setTab] = useState<"published" | "lab" | "runs">("published");
  const [job, setJob] = useState<string | null>(null);
  const pub = useQuery({ queryKey: ["bench-published"], queryFn: api.benchmarkPublished, retry: false });
  const jobs = useQuery({ queryKey: ["benchmarks"], queryFn: api.benchmarks, enabled: tab === "runs" });
  return (
    <div>
      <PageHeader eyebrow="Research" title="Benchmark laboratory" subtitle="Reproducible fault-injection experiments comparing seven planners. Every number on this page is simulated and labelled as such." />
      <Tabs className="mb-6" value={tab} onChange={setTab} tabs={[{ id: "published", label: "Published results" }, { id: "lab", label: "Run a benchmark" }, { id: "runs", label: "Previous runs" }]} />
      {tab === "published" && (pub.isLoading ? <Skeleton className="h-96" /> : pub.isError ? <EmptyState icon={<FlaskConical className="h-6 w-6" />} title="No published results found" body="Run research/benchmark/run_benchmark.py to generate research/results/results_default-v1.json, or start a benchmark from the lab tab." /> : <ResultsView res={pub.data!} published />)}
      {tab === "lab" && <div className="space-y-6"><LabForm onStarted={(id) => setJob(id)} />{job && <JobPanel id={job} />}</div>}
      {tab === "runs" && (jobs.isLoading ? <Skeleton className="h-40" /> : <div className="space-y-3">
        {job && <JobPanel id={job} />}
        {jobs.data?.items.length === 0 && <EmptyState title="No lab runs yet" />}
        {jobs.data?.items.map((j) => <Card key={j.id} hover className="flex cursor-pointer items-center justify-between p-4" onClick={() => setJob(j.id)}><div><p className="text-sm font-medium">{j.name}</p><p className="text-xs text-muted">{j.message}</p></div><Badge tone={statusTone(j.status)}>{j.status}</Badge></Card>)}</div>)}
    </div>
  );
}

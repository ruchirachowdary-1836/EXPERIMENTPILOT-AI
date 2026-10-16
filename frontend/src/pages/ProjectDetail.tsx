import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, CheckCircle2, FileSpreadsheet, Map as MapIcon, Play, Settings2, Trash2, Upload } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ApiError, api } from "@/api/client";
import { useToast } from "@/components/Toast";
import { Badge, Button, Callout, Card, CardHeader, EmptyState, ErrorState, Field, Input, PageHeader, Select, Skeleton, Tabs, Textarea } from "@/components/ui";
import { bytes, cn, pct, shortDate, statusTone, title } from "@/lib/utils";
import type { Dataset, Mapping, Project } from "@/types/api";

const ROLES = [
  { role: "reference" as const, name: "Reference data", hint: "Training or validation data with the target and model predictions.", required: true },
  { role: "recent" as const, name: "Recent data", hint: "Recent production data where performance degraded (with target and predictions if labelled).", required: true },
  { role: "replay" as const, name: "Replay data", hint: "Optional: the same recent records processed by the previous pipeline version.", required: false },
];

export const MAX_UPLOAD_MB = 25;
export function checkFile(f: File, maxMb = MAX_UPLOAD_MB): string | null {
  if (!f.name.toLowerCase().endsWith(".csv")) return "Only .csv files are accepted.";
  if (f.size === 0) return "That file is empty.";
  if (f.size > maxMb * 1024 * 1024) return `File is ${bytes(f.size)}; the limit is ${maxMb} MB.`;
  return null;
}

function Dropzone({ role, name, hint, required, dataset, projectId, locked }: { role: string; name: string; hint: string; required: boolean; dataset?: Dataset; projectId: string; locked: boolean }) {
  const qc = useQueryClient(); const toast = useToast();
  const input = useRef<HTMLInputElement>(null); const [drag, setDrag] = useState(false); const [err, setErr] = useState<string | null>(null);
  const up = useMutation({
    mutationFn: (f: File) => api.uploadDataset(projectId, role, f),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["project", projectId] }); toast.success(`${name} uploaded`); setErr(null); },
    onError: (e) => setErr(e instanceof Error ? e.message : "Upload failed"),
  });
  const del = useMutation({
    mutationFn: () => api.deleteDataset(projectId, role),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["project", projectId] }), onError: (e) => toast.error(e),
  });
  const pick = useCallback((f?: File | null) => { if (!f) return; const bad = checkFile(f); if (bad) { setErr(bad); return; } setErr(null); up.mutate(f); }, [up]);
  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center justify-between"><h3 className="text-[15px]">{name}</h3><Badge tone={required ? "accent" : "muted"}>{required ? "Required" : "Optional"}</Badge></div>
      {dataset ? (
        <div className="space-y-3">
          <div className="flex items-start gap-3 rounded-xl border border-good/25 bg-good/8 p-3">
            <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-good" />
            <div className="min-w-0 text-sm"><p className="truncate font-medium">{dataset.filename}</p><p className="text-xs text-muted num">{dataset.n_rows.toLocaleString()} rows · {dataset.n_cols} columns · {bytes(dataset.size_bytes)}</p></div>
          </div>
          <div className="flex gap-2">
            <Button size="sm" disabled={locked} onClick={() => input.current?.click()} icon={<Upload className="h-3.5 w-3.5" />}>Replace</Button>
            <Button size="sm" variant="ghost" disabled={locked} loading={del.isPending} onClick={() => { if (confirm(`Remove the ${role} dataset file?`)) del.mutate(); }} icon={<Trash2 className="h-3.5 w-3.5" />}>Remove</Button>
          </div>
          {locked && <p className="text-xs text-warn">Locked: this project has sessions that reference its data.</p>}
        </div>
      ) : (
        <button type="button" disabled={locked || up.isPending} onClick={() => input.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
          className={cn("flex w-full flex-col items-center gap-2 rounded-xl border-2 border-dashed px-4 py-8 text-center transition", drag ? "border-accent bg-accent/10" : "border-line/20 hover:border-accent/50 hover:bg-accent/5")}>
          <Upload className={cn("h-6 w-6", up.isPending ? "animate-pulse text-accent" : "text-faint")} />
          <span className="text-sm font-medium">{up.isPending ? "Validating and storing…" : "Drop a CSV or click to browse"}</span>
          <span className="text-xs text-muted">{hint}</span><span className="text-[11px] text-faint">.csv · UTF-8 · max {MAX_UPLOAD_MB} MB</span>
        </button>
      )}
      <input ref={input} type="file" accept=".csv,text/csv" className="sr-only" aria-label={`Upload ${name}`} onChange={(e) => { pick(e.target.files?.[0]); e.target.value = ""; }} />
      {err && <p role="alert" className="mt-3 rounded-lg bg-bad/10 px-3 py-2 text-xs text-bad">{err}</p>}
    </Card>
  );
}

function SchemaTable({ projectId, role }: { projectId: string; role: string }) {
  const q = useQuery({ queryKey: ["inspect", projectId, role], queryFn: () => api.inspectDataset(projectId, role) });
  if (q.isLoading) return <Skeleton className="h-48" />;
  if (q.isError) return <ErrorState error={q.error} />;
  const d = q.data!;
  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded-xl border border-line/10">
        <table className="w-full text-left text-sm">
          <thead className="bg-sunken/70 text-xs uppercase tracking-wide text-faint"><tr><th className="px-3 py-2">Column</th><th className="px-3 py-2">Type</th><th className="px-3 py-2">Missing</th><th className="px-3 py-2">Unique</th><th className="px-3 py-2">Range / sample</th></tr></thead>
          <tbody>{d.dataset.schema.map((c) => (
            <tr key={c.name} className="border-t border-line/8"><td className="px-3 py-2 font-mono text-[13px]">{c.name}</td><td className="px-3 py-2"><Badge tone={c.dtype === "numeric" ? "accent" : "cyan"}>{c.dtype}</Badge></td>
              <td className={cn("num px-3 py-2", c.null_rate > 0.05 && "text-warn")}>{pct(c.null_rate, 1)}</td><td className="num px-3 py-2">{c.n_unique}</td>
              <td className="num px-3 py-2 text-xs text-muted">{c.min != null ? `${c.min.toFixed(2)} → ${c.max!.toFixed(2)}` : c.sample.join(", ")}</td></tr>))}</tbody>
        </table>
      </div>
      <details className="rounded-xl border border-line/10 p-3 text-sm"><summary className="cursor-pointer text-muted">Preview first rows</summary>
        <div className="mt-3 overflow-x-auto"><table className="w-full text-xs"><thead><tr>{d.columns.map((c) => <th key={c} className="px-2 py-1 text-left font-mono text-faint">{c}</th>)}</tr></thead>
          <tbody>{d.preview.map((r, i) => <tr key={i} className="border-t border-line/8">{d.columns.map((c) => <td key={c} className="num whitespace-nowrap px-2 py-1">{r[c] == null ? "—" : String(typeof r[c] === "number" ? +r[c].toFixed(3) : r[c])}</td>)}</tr>)}</tbody></table></div>
      </details>
    </div>
  );
}

function MappingPanel({ project }: { project: Project }) {
  const qc = useQueryClient(); const toast = useToast();
  const ref = project.datasets.find((d) => d.role === "reference");
  const cols = ref?.schema.map((c) => c.name) ?? [];
  const cur = project.config.mapping ?? {};
  const [m, setM] = useState<Mapping>({ target: cur.target ?? "", prediction: cur.prediction ?? "", timestamp: cur.timestamp ?? "", identifier: cur.identifier ?? "" });
  const [warnings, setWarnings] = useState<string[]>([]);
  const save = useMutation({
    mutationFn: () => api.setMapping(project.id, { target: m.target || null, prediction: m.prediction || null, timestamp: m.timestamp || null, identifier: m.identifier || null, features: null }),
    onSuccess: (r) => { setWarnings(r.warnings); qc.invalidateQueries({ queryKey: ["project", project.id] }); toast.success("Column mapping saved"); },
    onError: (e) => toast.error(e, "Mapping rejected"),
  });
  if (!ref) return <EmptyState icon={<MapIcon className="h-6 w-6" />} title="Upload the reference dataset first" body="Column mapping uses the reference schema." />;
  const sel = (key: keyof Mapping, label: string, hint: string) => (
    <Field label={label} hint={hint} htmlFor={`map-${key}`}>
      <Select id={`map-${key}`} value={(m[key] as string) ?? ""} onChange={(e) => setM({ ...m, [key]: e.target.value })}>
        <option value="">— not mapped —</option>{cols.map((c) => <option key={c} value={c}>{c}</option>)}
      </Select>
    </Field>
  );
  return (
    <Card><CardHeader icon={<MapIcon className="h-4 w-4" />} title="Column mapping" subtitle="All remaining columns are treated as model input features." />
      <div className="grid gap-5 p-5 sm:grid-cols-2 sm:p-6">
        {sel("target", "Target (label)", "Ground-truth class. Enables label-based experiments.")}
        {sel("prediction", "Prediction", "The model's predicted class for each row.")}
        {sel("timestamp", "Timestamp", "Used for temporal validation.")}
        {sel("identifier", "Record identifier", "Used to align recent and replay rows.")}
      </div>
      {warnings.map((w) => <div key={w} className="px-5 pb-3 sm:px-6"><Callout tone="warn">{w}</Callout></div>)}
      <div className="flex justify-end border-t border-line/10 px-5 py-4 sm:px-6"><Button variant="primary" loading={save.isPending} onClick={() => save.mutate()}>Save mapping</Button></div>
    </Card>
  );
}

function SettingsPanel({ project }: { project: Project }) {
  const qc = useQueryClient(); const toast = useToast();
  const [name, setName] = useState(project.name); const [desc, setDesc] = useState(project.description);
  const [pipe, setPipe] = useState(project.config.pipeline_notes ?? ""); const [model, setModel] = useState(project.config.model_notes ?? "");
  const [budget, setBudget] = useState(String(project.config.default_budget ?? 8)); const [risk, setRisk] = useState(String(project.config.default_risk_cap ?? 0.3));
  const save = useMutation({
    mutationFn: () => api.updateProject(project.id, { name, description: desc, pipeline_notes: pipe, model_notes: model, default_budget: Number(budget), default_risk_cap: Number(risk) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["project", project.id] }); toast.success("Project updated"); }, onError: (e) => toast.error(e),
  });
  return (
    <Card><CardHeader icon={<Settings2 className="h-4 w-4" />} title="Project settings & model information" subtitle="Notes are stored with the project and shown in reports." />
      <div className="grid gap-5 p-5 sm:grid-cols-2 sm:p-6">
        <Field label="Name" htmlFor="s-name"><Input id="s-name" value={name} onChange={(e) => setName(e.target.value)} /></Field>
        <Field label="Description" htmlFor="s-desc"><Input id="s-desc" value={desc} onChange={(e) => setDesc(e.target.value)} /></Field>
        <Field label="Selected model" htmlFor="s-model" hint="Algorithm, version, training window. Uploaded model files are never loaded; predictions are used instead."><Textarea id="s-model" value={model} onChange={(e) => setModel(e.target.value)} /></Field>
        <Field label="Pipeline information" htmlFor="s-pipe" hint="Preprocessing steps and versions (current and previous)."><Textarea id="s-pipe" value={pipe} onChange={(e) => setPipe(e.target.value)} /></Field>
        <Field label="Default diagnostic budget" htmlFor="s-b" hint="Cost units (≈ one minute of compute or one labelling batch)."><Input id="s-b" type="number" min={0.5} step={0.5} value={budget} onChange={(e) => setBudget(e.target.value)} /></Field>
        <Field label="Default risk ceiling" htmlFor="s-r" hint="Maximum per-experiment risk, 0–1."><Input id="s-r" type="number" min={0} max={1} step={0.05} value={risk} onChange={(e) => setRisk(e.target.value)} /></Field>
      </div>
      <div className="flex justify-end border-t border-line/10 px-5 py-4 sm:px-6"><Button variant="primary" loading={save.isPending} onClick={() => save.mutate()}>Save settings</Button></div>
    </Card>
  );
}

export default function ProjectDetail() {
  const { id = "" } = useParams(); const nav = useNavigate();
  const [tab, setTab] = useState<"data" | "mapping" | "sessions" | "settings">("data");
  const q = useQuery({ queryKey: ["project", id], queryFn: () => api.project(id) });
  const sessions = useQuery({ queryKey: ["sessions", id], queryFn: () => api.sessions(id) });
  const [schemaRole, setSchemaRole] = useState("reference");
  if (q.isLoading) return <div className="space-y-4"><Skeleton className="h-12 w-72" /><Skeleton className="h-64" /></div>;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const p = q.data!;
  const locked = p.n_sessions > 0;
  return (
    <div>
      <PageHeader eyebrow={<Link to="/app/projects" className="hover:text-fg">Projects</Link>} title={p.name} subtitle={p.description || undefined}
        actions={<>{p.config.demo && <Badge tone="cyan">Simulated demo</Badge>}<Button variant="primary" disabled={!p.ready} icon={<Play className="h-4 w-4" />} onClick={() => nav(`/app/projects/${p.id}/setup`)}>Configure diagnosis</Button></>} />
      {!p.ready && <div className="mb-5"><Callout tone="warn" title="Before you can start a diagnosis">{p.readiness.join(" · ")}</Callout></div>}
      <Tabs className="mb-6" value={tab} onChange={setTab} tabs={[
        { id: "data", label: "Datasets", icon: <FileSpreadsheet className="h-3.5 w-3.5" /> }, { id: "mapping", label: "Column mapping", icon: <MapIcon className="h-3.5 w-3.5" /> },
        { id: "sessions", label: `Sessions (${p.n_sessions})`, icon: <Play className="h-3.5 w-3.5" /> }, { id: "settings", label: "Settings", icon: <Settings2 className="h-3.5 w-3.5" /> }]} />
      {tab === "data" && (
        <div className="space-y-6">
          <div className="grid gap-4 lg:grid-cols-3">{ROLES.map((r) => <Dropzone key={r.role} {...r} projectId={p.id} dataset={p.datasets.find((d) => d.role === r.role)} locked={locked} />)}</div>
          {p.datasets.length > 0 && (
            <Card><CardHeader title="Schema inspection" action={<Select className="w-40" aria-label="Dataset to inspect" value={schemaRole} onChange={(e) => setSchemaRole(e.target.value)}>{p.datasets.map((d) => <option key={d.role} value={d.role}>{title(d.role)}</option>)}</Select>} />
              <div className="p-5 sm:p-6"><SchemaTable projectId={p.id} role={p.datasets.some((d) => d.role === schemaRole) ? schemaRole : p.datasets[0].role} /></div></Card>)}
        </div>)}
      {tab === "mapping" && <MappingPanel project={p} />}
      {tab === "settings" && <SettingsPanel project={p} />}
      {tab === "sessions" && (
        sessions.data && sessions.data.items.length > 0 ? (
          <div className="grid gap-3">{sessions.data.items.map((s) => (
            <Link key={s.id} to={`/app/sessions/${s.id}`}><Card hover className="flex flex-wrap items-center gap-4 p-4">
              <div className="min-w-0 flex-1"><p className="truncate font-medium">{s.name}</p><p className="text-xs text-muted">{shortDate(s.created_at)} · {s.planner} · {s.n_runs} experiments</p></div>
              <Badge tone={statusTone(s.status)}>{s.status}</Badge><span className="num text-sm text-muted">{title(s.top_hypothesis)} {pct(s.top_probability)}</span>
              <span className="num text-sm text-muted">cost {s.spent.toFixed(1)}/{s.budget}</span><ArrowRight className="h-4 w-4 text-faint" /></Card></Link>))}</div>
        ) : <EmptyState icon={<Play className="h-6 w-6" />} title="No diagnosis sessions yet" body="Configure budgets, priors and experiments, then start the planner." action={<Button variant="primary" disabled={!p.ready} onClick={() => nav(`/app/projects/${p.id}/setup`)}>Configure diagnosis</Button>} />)}
    </div>
  );
}

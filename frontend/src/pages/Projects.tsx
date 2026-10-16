import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Beaker, Database, FolderPlus, Plus, Sparkles, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "@/api/client";
import { useToast } from "@/components/Toast";
import { Badge, Button, Card, EmptyState, ErrorState, Field, Input, Modal, PageHeader, Select, Skeleton, Textarea } from "@/components/ui";
import { shortDate, title } from "@/lib/utils";

export function validateProjectName(v: string): string | null {
  const t = v.trim();
  if (!t) return "Give the project a name.";
  if (t.length > 120) return "Use at most 120 characters.";
  return null;
}

export function NewProjectModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const nav = useNavigate(); const qc = useQueryClient(); const toast = useToast();
  const [name, setName] = useState(""); const [desc, setDesc] = useState(""); const [touched, setTouched] = useState(false);
  const err = validateProjectName(name);
  const m = useMutation({
    mutationFn: () => api.createProject({ name: name.trim(), description: desc.trim() }),
    onSuccess: (p) => { qc.invalidateQueries({ queryKey: ["projects"] }); toast.success("Project created"); onClose(); nav(`/app/projects/${p.id}`); },
    onError: (e) => toast.error(e, "Could not create project"),
  });
  return (
    <Modal open={open} onClose={onClose} title="New project"
      footer={<><Button variant="ghost" onClick={onClose}>Cancel</Button><Button variant="primary" loading={m.isPending} onClick={() => { setTouched(true); if (!err) m.mutate(); }}>Create project</Button></>}>
      <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); setTouched(true); if (!err) m.mutate(); }}>
        <Field label="Project name" htmlFor="pname" error={touched ? err : null}>
          <Input id="pname" autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="Fraud model – Q3 degradation" invalid={touched && !!err} />
        </Field>
        <Field label="Description" htmlFor="pdesc" hint="Optional: what degraded, since when.">
          <Textarea id="pdesc" value={desc} onChange={(e) => setDesc(e.target.value)} />
        </Field>
        <p className="text-xs text-muted">Classification tasks are supported. You will upload data and map columns next.</p>
      </form>
    </Modal>
  );
}

const FAULT_HELP: Record<string, string> = {
  covariate_drift: "Feature distributions shift", label_noise: "Recent labels are noisier", feature_leakage: "A feature leaked the label in training",
  class_prior_shift: "Class frequencies change", pipeline_change: "Preprocessing differs from training", data_quality: "Nulls, sentinels, stale columns", metric_bug: "Reported metric is wrong",
};

function DemoModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const nav = useNavigate(); const qc = useQueryClient(); const toast = useToast();
  const opts = useQuery({ queryKey: ["demo-options"], queryFn: api.demoOptions, enabled: open });
  const [dataset, setDataset] = useState("synthetic_credit");
  const [fault, setFault] = useState("pipeline_change"); const [fault2, setFault2] = useState(""); const [sev, setSev] = useState("high"); const [seed, setSeed] = useState(7);
  const m = useMutation({
    mutationFn: () => api.createDemo({ dataset, seed, faults: [fault, fault2].filter(Boolean).filter((f, i, a) => a.indexOf(f) === i).map((f) => ({ fault: f, severity: sev })) }),
    onSuccess: (p) => { qc.invalidateQueries({ queryKey: ["projects"] }); onClose(); nav(`/app/projects/${p.id}`); },
    onError: (e) => toast.error(e, "Could not create the demo"),
  });
  return (
    <Modal open={open} onClose={onClose} title="Create a demo scenario"
      footer={<><Button variant="ghost" onClick={onClose}>Cancel</Button><Button variant="primary" icon={<Sparkles className="h-4 w-4" />} loading={m.isPending} onClick={() => m.mutate()}>Generate scenario</Button></>}>
      <div className="space-y-4">
        <p className="text-sm text-muted">Generates reference, recent and replay datasets with a model's predictions. The injected fault is stored privately so you can compare it with the diagnosis afterwards. <b>Data are simulated.</b></p>
        <Field label="Dataset" htmlFor="d-ds">
          <Select id="d-ds" value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {(opts.data?.datasets ?? [{ id: "synthetic_credit", description: "", synthetic: true }]).map((d) => <option key={d.id} value={d.id}>{title(d.id)}{d.synthetic ? "" : " (real, small)"}</option>)}
          </Select>
        </Field>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Injected fault" htmlFor="d-f"><Select id="d-f" value={fault} onChange={(e) => setFault(e.target.value)}><option value="">No fault (control)</option>{(opts.data?.faults ?? [fault]).map((f) => <option key={f} value={f}>{title(f)}</option>)}</Select></Field>
          <Field label="Second fault (optional)" htmlFor="d-f2"><Select id="d-f2" value={fault2} onChange={(e) => setFault2(e.target.value)}><option value="">None</option>{(opts.data?.faults ?? []).filter((f) => f !== fault).map((f) => <option key={f} value={f}>{title(f)}</option>)}</Select></Field>
          <Field label="Severity" htmlFor="d-s"><Select id="d-s" value={sev} onChange={(e) => setSev(e.target.value)}>{["low", "medium", "high"].map((s) => <option key={s}>{s}</option>)}</Select></Field>
          <Field label="Random seed" htmlFor="d-seed"><Input id="d-seed" type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} /></Field>
        </div>
        {fault && <p className="text-xs text-faint">{FAULT_HELP[fault]}</p>}
      </div>
    </Modal>
  );
}

export default function Projects() {
  const q = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  const qc = useQueryClient(); const toast = useToast();
  const [newOpen, setNewOpen] = useState(false); const [demoOpen, setDemoOpen] = useState(false);
  const del = useMutation({
    mutationFn: (id: string) => api.deleteProject(id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["projects"] }); toast.success("Project and its data files deleted"); },
    onError: (e) => toast.error(e),
  });
  return (
    <div>
      <PageHeader eyebrow="Workspace" title="Projects" subtitle="Each project holds a reference dataset, the recent (failing) dataset, optional replay data, and its diagnosis sessions."
        actions={<><Button icon={<Sparkles className="h-4 w-4" />} onClick={() => setDemoOpen(true)}>Demo scenario</Button><Button variant="primary" icon={<Plus className="h-4 w-4" />} onClick={() => setNewOpen(true)}>New project</Button></>} />
      {q.isLoading && <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-44" />)}</div>}
      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}
      {q.data && q.data.items.length === 0 && (
        <EmptyState icon={<FolderPlus className="h-6 w-6" />} title="No projects yet" body="Create a project and upload your data, or generate a simulated failure to see the planner in action."
          action={<div className="flex gap-2"><Button variant="primary" onClick={() => setDemoOpen(true)} icon={<Sparkles className="h-4 w-4" />}>Try a demo scenario</Button><Button onClick={() => setNewOpen(true)}>New project</Button></div>} />
      )}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {q.data?.items.map((p, i) => (
          <motion.div key={p.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
            <Card hover className="group flex h-full flex-col">
              <Link to={`/app/projects/${p.id}`} className="flex flex-1 flex-col card-pad">
                <div className="flex items-start justify-between gap-3">
                  <div className="grid h-10 w-10 place-items-center rounded-xl bg-accent/12 text-accent"><Database className="h-5 w-5" /></div>
                  <div className="flex gap-1.5">{p.config.demo && <Badge tone="cyan">Demo</Badge>}<Badge tone={p.ready ? "good" : "warn"}>{p.ready ? "Ready" : "Setup needed"}</Badge></div>
                </div>
                <h3 className="mt-4 line-clamp-2 text-base">{p.name}</h3>
                <p className="mt-1 line-clamp-2 text-sm text-muted">{p.description || "No description"}</p>
                <div className="mt-auto flex items-center gap-4 pt-5 text-xs text-faint"><span className="flex items-center gap-1.5"><Beaker className="h-3.5 w-3.5" />{p.n_sessions} sessions</span><span>{p.datasets.length} datasets</span><span className="ml-auto">{shortDate(p.updated_at)}</span></div>
              </Link>
              <button aria-label={`Delete ${p.name}`} onClick={() => { if (confirm(`Delete "${p.name}" and all of its uploaded data? This cannot be undone.`)) del.mutate(p.id); }}
                className="absolute right-3 bottom-3 rounded-lg p-1.5 text-faint opacity-0 transition hover:bg-bad/10 hover:text-bad focus:opacity-100 group-hover:opacity-100"><Trash2 className="h-4 w-4" /></button>
            </Card>
          </motion.div>
        ))}
      </div>
      <NewProjectModal open={newOpen} onClose={() => setNewOpen(false)} />
      <DemoModal open={demoOpen} onClose={() => setDemoOpen(false)} />
    </div>
  );
}

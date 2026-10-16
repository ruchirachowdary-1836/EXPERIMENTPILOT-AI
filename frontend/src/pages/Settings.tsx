import { useQuery } from "@tanstack/react-query";
import { Database, Moon, Server, Sun } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "@/api/client";
import { Badge, Card, CardHeader, ErrorState, PageHeader, Skeleton } from "@/components/ui";
import { useTheme } from "@/hooks/useTheme";
import { bytes, cn } from "@/lib/utils";

export default function Settings() {
  const rt = useQuery({ queryKey: ["runtime"], queryFn: api.runtime });
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.projects() });
  const lik = useQuery({ queryKey: ["likelihoods"], queryFn: api.likelihoods });
  const { theme, setTheme } = useTheme();
  const rows: [string, string][] = rt.data ? [
    ["Version", rt.data.version], ["Environment", rt.data.environment], ["Database", rt.data.database], ["Max upload", `${rt.data.max_upload_mb} MB`],
    ["Max rows / columns", `${rt.data.max_rows.toLocaleString()} / ${rt.data.max_columns}`], ["Experiment executor", rt.data.executor], ["Experiment time limit", `${rt.data.experiment_timeout_s}s`],
    ["Benchmark workers", String(rt.data.benchmark_workers)], ["Allowed origins", rt.data.cors_origins.join(", ")], ["Outbound network calls", rt.data.outbound_network_calls ? "yes" : "none"], ["Authentication", rt.data.authentication],
  ] : [];
  const prov = lik.data?.provenance ?? {};
  return (
    <div>
      <PageHeader eyebrow="System" title="Settings" subtitle="Appearance, runtime configuration (set with EP_* environment variables) and dataset management." />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card><CardHeader icon={<Sun className="h-4 w-4" />} title="Appearance" />
          <div className="grid grid-cols-2 gap-3 p-5">{(["dark", "light"] as const).map((t) => (
            <button key={t} onClick={() => setTheme(t)} aria-pressed={theme === t} className={cn("flex items-center gap-3 rounded-xl border p-4 text-left text-sm transition", theme === t ? "border-accent bg-accent/10" : "border-line/15 hover:border-line/30")}>
              {t === "dark" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}<span className="font-medium capitalize">{t}</span></button>))}</div></Card>
        <Card><CardHeader icon={<Database className="h-4 w-4" />} title="Dataset management" subtitle="Stored locally under the backend data directory." />
          <div className="space-y-2 p-5">
            {projects.data?.items.length === 0 && <p className="text-sm text-muted">No stored datasets.</p>}
            {projects.data?.items.map((p) => (
              <Link key={p.id} to={`/app/projects/${p.id}`} className="flex items-center justify-between rounded-xl border border-line/10 p-3 text-sm hover:border-accent/40">
                <span className="truncate">{p.name}</span><span className="num text-xs text-muted">{p.datasets.length} files · {bytes(p.datasets.reduce((a, d) => a + d.size_bytes, 0))}</span></Link>))}
            <p className="pt-2 text-xs text-muted">Delete a project from the workspace to remove its database records and every stored data file.</p></div></Card>
        <Card className="lg:col-span-2"><CardHeader icon={<Server className="h-4 w-4" />} title="Runtime configuration" />
          <div className="p-5">{rt.isLoading ? <Skeleton className="h-40" /> : rt.isError ? <ErrorState error={rt.error} /> :
            <dl className="grid gap-x-8 gap-y-3 sm:grid-cols-2">{rows.map(([k, v]) => <div key={k} className="flex justify-between gap-4 border-b border-line/8 pb-2 text-sm"><dt className="text-muted">{k}</dt><dd className="num text-right">{v}</dd></div>)}</dl>}</div></Card>
        <Card className="lg:col-span-2"><CardHeader title="Likelihood library" subtitle="Tables that drive every posterior update" />
          <div className="space-y-2 p-5 text-sm"><div className="flex flex-wrap gap-2"><Badge tone={prov.simulated ? "warn" : "good"}>{prov.simulated ? "simulated fault injection" : "expert priors only"}</Badge>{prov.method && <Badge>{prov.method}</Badge>}{prov.n_records != null && <Badge>{prov.n_records} estimation scenarios</Badge>}</div>
            {prov.seeds && <p className="text-muted">Estimation seeds: <span className="num">{prov.seeds.join(", ")}</span> · datasets: {(prov.dataset_names ?? []).join(", ")}</p>}
            {!prov.simulated && <p className="text-warn">The default tables file has not been generated yet. Run research/benchmark/run_benchmark.py --install-library.</p>}</div></Card>
      </div>
    </div>
  );
}

import type {
  BenchmarkJob, BenchmarkResults, Dataset, Evidence, ExperimentMeta, HistoryPoint, HypothesisMeta, Mapping, Page, PlannerMeta,
  Project, Recommendation, Report, RuntimeInfo, Run, Session, SessionListItem, Tree,
} from "@/types/api";

export class ApiError extends Error {
  status: number; code: string; details?: unknown; extra?: Record<string, unknown>;
  constructor(status: number, code: string, message: string, extra?: Record<string, unknown>) {
    super(message); this.status = status; this.code = code; this.extra = extra;
  }
}

const BASE = (import.meta.env?.VITE_API_BASE as string | undefined) ?? "";
const V1 = `${BASE}/api/v1`;

async function parseError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    const e = body?.error ?? body;
    return new ApiError(res.status, e?.code ?? "http_error", e?.message ?? e?.detail ?? res.statusText, e);
  } catch {
    return new ApiError(res.status, "http_error", res.status >= 500 ? "The server hit an unexpected error." : res.statusText || "Request failed");
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${V1}${path}`, { ...init, headers: { ...(init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}), ...init?.headers } });
  } catch {
    throw new ApiError(0, "network_error", "Cannot reach the ExperimentPilot API. Is the backend running?");
  }
  if (!res.ok) throw await parseError(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (body: unknown) => JSON.stringify(body);

export async function downloadFile(path: string, fallbackName: string): Promise<void> {
  let res: Response;
  try { res = await fetch(`${V1}${path}`); } catch { throw new ApiError(0, "network_error", "Cannot reach the ExperimentPilot API."); }
  if (!res.ok) throw await parseError(res);
  const cd = res.headers.get("Content-Disposition") ?? "";
  const name = /filename="?([^"]+)"?/.exec(cd)?.[1] ?? fallbackName;
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export const api = {
  health: () => request<{ status: string; version: string }>("/health"),
  runtime: () => request<RuntimeInfo>("/meta/runtime"),
  hypotheses: () => request<HypothesisMeta[]>("/meta/hypotheses"),
  experiments: () => request<ExperimentMeta[]>("/meta/experiments"),
  planners: () => request<PlannerMeta[]>("/meta/planners"),
  likelihoods: () => request<{ provenance: Record<string, any>; experiments: Record<string, any> }>("/meta/likelihoods"),
  demoOptions: () => request<{ datasets: { id: string; description: string; synthetic: boolean }[]; faults: string[]; severities: string[] }>("/meta/demo-options"),

  projects: (q = "") => request<Page<Project>>(`/projects?limit=100${q ? `&q=${encodeURIComponent(q)}` : ""}`),
  project: (id: string) => request<Project>(`/projects/${id}`),
  createProject: (b: { name: string; description?: string; pipeline_notes?: string; model_notes?: string }) => request<Project>("/projects", { method: "POST", body: json(b) }),
  updateProject: (id: string, b: Record<string, unknown>) => request<Project>(`/projects/${id}`, { method: "PATCH", body: json(b) }),
  deleteProject: (id: string) => request<void>(`/projects/${id}`, { method: "DELETE" }),
  createDemo: (b: { dataset: string; seed: number; faults: { fault: string; severity: string }[]; name?: string }) => request<Project>("/demo", { method: "POST", body: json(b) }),
  uploadDataset: (pid: string, role: string, file: File) => {
    const fd = new FormData(); fd.append("role", role); fd.append("file", file);
    return request<{ dataset: Dataset; project: Project }>(`/projects/${pid}/datasets`, { method: "POST", body: fd });
  },
  inspectDataset: (pid: string, role: string) => request<{ dataset: Dataset; preview: Record<string, any>[]; columns: string[] }>(`/projects/${pid}/datasets/${role}?rows=12`),
  deleteDataset: (pid: string, role: string) => request<void>(`/projects/${pid}/datasets/${role}`, { method: "DELETE" }),
  setMapping: (pid: string, m: Mapping) => request<{ mapping: Mapping; warnings: string[]; project: Project }>(`/projects/${pid}/mapping`, { method: "PUT", body: json(m) }),

  sessions: (projectId?: string) => request<Page<SessionListItem>>(`/sessions?limit=100${projectId ? `&project_id=${projectId}` : ""}`),
  session: (id: string) => request<Session>(`/sessions/${id}`),
  createSession: (b: Record<string, unknown>) => request<Session>("/sessions", { method: "POST", body: json(b) }),
  deleteSession: (id: string) => request<void>(`/sessions/${id}`, { method: "DELETE" }),
  next: (id: string) => request<Recommendation>(`/sessions/${id}/next`),
  run: (id: string, b: { experiment_id?: string; approve?: boolean }) => request<{ run: Run; session: Session }>(`/sessions/${id}/run`, { method: "POST", body: json(b) }),
  external: (id: string, b: { experiment_id: string; outcome: string | number; note?: string }) => request<{ run: Run; session: Session }>(`/sessions/${id}/external`, { method: "POST", body: json(b) }),
  skip: (id: string, b: { experiment_id?: string; reason?: string }) => request<{ run: Run; session: Session }>(`/sessions/${id}/skip`, { method: "POST", body: json(b) }),
  pause: (id: string) => request<Session>(`/sessions/${id}/pause`, { method: "POST" }),
  resume: (id: string) => request<Session>(`/sessions/${id}/resume`, { method: "POST" }),
  stop: (id: string, reason = "") => request<Session>(`/sessions/${id}/stop`, { method: "POST", body: json({ reason }) }),
  evidence: (id: string) => request<Evidence>(`/sessions/${id}/evidence`),
  history: (id: string) => request<{ hypotheses: string[]; history: HistoryPoint[] }>(`/sessions/${id}/posterior-history`),
  tree: (id: string) => request<Tree>(`/sessions/${id}/tree`),
  report: (id: string) => request<Report>(`/sessions/${id}/report`),
  download: (id: string, kind: "report.pdf" | "report.json" | "hypotheses.csv" | "experiments.csv" | "history.csv") =>
    downloadFile(`/sessions/${id}/export/${kind}`, kind),

  benchmarkOptions: () => request<{ datasets: string[]; faults: string[]; severities: string[]; planners: PlannerMeta[]; reserved_estimation_seeds: number[]; likelihood_provenance: Record<string, any> }>("/benchmarks/options"),
  benchmarkPublished: () => request<BenchmarkResults>("/benchmarks/published"),
  benchmarkPublishedRuns: (planner?: string, scenario?: string, offset = 0) =>
    request<Page<Record<string, any>>>(`/benchmarks/published/runs?limit=25&offset=${offset}${planner ? `&planner=${planner}` : ""}${scenario ? `&scenario=${encodeURIComponent(scenario)}` : ""}`),
  createBenchmark: (b: Record<string, unknown>) => request<BenchmarkJob>("/benchmarks", { method: "POST", body: json(b) }),
  benchmarks: () => request<Page<BenchmarkJob>>("/benchmarks"),
  benchmark: (id: string) => request<BenchmarkJob>(`/benchmarks/${id}`),
  benchmarkResults: (id: string) => request<BenchmarkResults>(`/benchmarks/${id}/results`),
  benchmarkTraces: (id: string, planner?: string, offset = 0) => request<Page<Record<string, any>>>(`/benchmarks/${id}/traces?limit=25&offset=${offset}${planner ? `&planner=${planner}` : ""}`),
  benchmarkDownload: (id: string, fmt: "csv" | "json") => downloadFile(`/benchmarks/${id}/export.${fmt}`, `benchmark.${fmt}`),
  publishedPlotUrl: (name: string) => `${V1}/benchmarks/published/plots/${name}`,
  benchmarkPlotUrl: (id: string, name: string) => `${V1}/benchmarks/${id}/plots/${name}`,
};

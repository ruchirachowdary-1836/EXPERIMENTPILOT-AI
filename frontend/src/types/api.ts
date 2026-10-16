export interface ColumnSchema {
  name: string; dtype: "numeric" | "categorical"; null_rate: number; n_unique: number;
  sample: (string | number | null)[]; min?: number; max?: number; mean?: number;
}
export interface Dataset {
  id: string; project_id: string; role: "reference" | "recent" | "replay"; filename: string;
  n_rows: number; n_cols: number; size_bytes: number; sha256: string; schema: ColumnSchema[]; created_at: string;
}
export interface Mapping { target?: string | null; prediction?: string | null; timestamp?: string | null; identifier?: string | null; features?: string[] | null }
export interface Project {
  id: string; name: string; description: string; task_type: string;
  config: { mapping?: Mapping; pipeline_notes?: string; model_notes?: string; demo?: boolean; demo_truth?: string[];
            reported_metrics?: ReportedMetrics; default_budget?: number; default_risk_cap?: number };
  created_at: string; updated_at: string; datasets: Dataset[]; n_sessions: number; ready: boolean; readiness: string[];
}
export interface ReportedMetrics { metric: "accuracy" | "macro_f1"; reference: number; recent: number }
export interface Page<T> { items: T[]; total: number; limit: number; offset: number }

export interface HypothesisMeta { id: string; code: string; name: string; description: string; symptom: string; default_prior: number; default_enabled: boolean; core: boolean }
export interface ExperimentMeta {
  id: string; name: string; description: string; cost: number; risk: number; requires: string[]; outcomes: string[];
  requires_approval: boolean; limitations: string[]; has_likelihood: boolean; likelihood_runs: number;
}
export interface PlannerMeta { id: string; label: string; description: string; enforces_risk_cap: boolean }

export interface ScoredExperiment {
  id: string; name: string; description: string; eig: number; cost: number; risk: number; denominator: number; score: number;
  separation: number; adjusted_score: number; feasible: boolean; reasons: string[]; reason_codes: string[];
  requires_approval: boolean; rank: number | null; lookahead_value: number | null; outcomes: string[]; requires: string[];
  state: "feasible" | "blocked" | "done" | "skipped";
}
export interface Recommendation {
  session_id: string; status: string; planner: string; planner_label: string; selected: string | null; rationale: string;
  stop: { stop: boolean; code: string | null; message: string; confidence: number };
  experiments: ScoredExperiment[]; budget: number; spent: number; remaining: number; diagnosis_confidence: number;
  map_state: string[]; map_probability: number; entropy: number; needs_approval: boolean;
}
export interface Finding { level: "info" | "warning" | "alert"; text: string }
export interface Run {
  id: string; experiment_id: string; name: string; status: "running" | "completed" | "failed" | "skipped" | "external";
  source: string; step: number | null; outcome: string | null; outcome_index: number | null; score: number | null;
  cost: number; risk: number; interpretation: string; findings: Finding[]; limitations: string[]; stats: Record<string, any>;
  recommendation: Record<string, any>; note: string | null; error: string | null; runtime_s: number | null;
  started_at: string | null; finished_at: string | null; tables?: Record<string, Record<string, any>[]>;
}
export interface Marginal { hypothesis: string; code: string; name: string; probability: number }
export interface Session {
  id: string; project_id: string; name: string; status: "active" | "paused" | "completed" | "stopped"; planner: string;
  config: Record<string, any>; hypotheses: { id: string; prior: number }[]; experiments: string[]; symptoms: Record<string, any>;
  spent: number; risk_total: number; budget: number; remaining: number; stop_code: string | null; stop_message: string | null;
  version: number; created_at: string; updated_at: string; finished_at: string | null; marginals: Marginal[]; entropy: number;
  map_state: string[]; map_probability: number; runs: Run[]; recommendation: Recommendation | null;
}
export interface SessionListItem {
  id: string; project_id: string; name: string; status: string; planner: string; spent: number; budget: number; n_runs: number;
  top_hypothesis: string; top_probability: number; map_probability: number; created_at: string; updated_at: string;
}
export interface HistoryPoint { step: number; label: string; marginals: Record<string, number>; entropy: number; map_probability: number; spent: number; created_at: string }
export interface Evidence { runs: Run[]; direction: Record<string, { run_id: string; experiment_id: string; experiment: string; outcome: string; likelihood_ratio: number; direction: "supports" | "contradicts" | "neutral" }[]>; symptoms: Record<string, any> }
export interface TreeNode { id: string; type: "root" | "experiment" | "outcome" | "stop" | "pending"; depth: number; taken: boolean; data: Record<string, any> }
export interface TreeEdge { id: string; source: string; target: string; taken: boolean; label: string }
export interface Tree { session_id: string; nodes: TreeNode[]; edges: TreeEdge[]; hypotheses: string[] }

export interface RankedHypothesis {
  hypothesis: string; code: string; name: string; description: string; posterior: number; prior: number; change: number;
  status: "likely" | "probable" | "possible" | "unlikely";
  supporting: Evidence["direction"][string]; contradicting: Evidence["direction"][string]; remediation: string[];
}
export interface Report {
  session: { id: string; name: string; status: string; planner: string; stop_code: string | null; stop_message: string | null; created_at: string; finished_at: string | null; config: Record<string, any> };
  generated_at: string; verdict: string; diagnosed: string[]; ranking: RankedHypothesis[];
  confidence: { map_state: string[]; map_probability: number; threshold: number; mode: string; reached: boolean; entropy_bits: number; initial_entropy_bits: number; uncertainty_reduction: number; undecided_hypotheses: string[] };
  top_states: { causes: string[]; probability: number }[]; alternatives: { hypothesis: string; name: string; posterior: number }[];
  symptoms: Record<string, any>; experiments: Run[];
  cost: { spent: number; budget: number; utilization: number; remaining: number; experiments_run: number };
  risk: { total: number; max: number; cap: number; violations: number; approved_risky: string[] };
  posterior_history: HistoryPoint[]; limitations: string[]; demo_truth: string[] | null; disclaimer: string;
}

export interface BenchmarkJob { id: string; name: string; status: "queued" | "running" | "completed" | "failed"; progress: number; message: string; error: string | null; config: Record<string, any>; created_at: string; finished_at: string | null }
export interface BenchmarkSummaryRow {
  planner: string; label: string; n_runs: number; top1: number; top1_lo: number; top1_hi: number; top3: number; f1: number; f1_lo: number; f1_hi: number;
  precision: number; recall: number; cost: number; cost_lo: number; cost_hi: number; n_experiments: number; brier: number; risk_violations: number; risk_total: number; stopped_confident: number;
}
export interface BenchmarkResults {
  simulated: boolean; disclaimer: string; config: Record<string, any>; n_scenarios?: number; n_evaluation_scenarios?: number; n_estimation_scenarios?: number;
  summary: BenchmarkSummaryRow[];
  accuracy_vs_budget: { planner: string; budget: number; top1: number; top1_lo: number; top1_hi: number; f1: number; mean_cost: number }[];
  calibration: Record<string, { ece: number; brier: number; reliability: { confidence: number; accuracy: number; count: number }[] }>;
  paired_tests: { reference: string; other: string; metric: string; n: number; mean_difference: number; wilcoxon_p: number }[];
  agreement?: { planners: string[]; matrix: number[][] };
  robustness: { sigma: number; planner: string; top1: number; f1: number; cost: number; brier: number }[];
  by_fault?: { truth: string; planner: string; n: number; top1: number | null; f1: number; cost: number }[];
  noisy_or_validation?: Record<string, number>;
  unseen_dataset?: { dataset: string; summary: BenchmarkSummaryRow[] };
  plots?: string[]; elapsed_s?: number;
}
export interface RuntimeInfo { version: string; environment: string; max_upload_mb: number; max_rows: number; max_columns: number; executor: string; experiment_timeout_s: number; benchmark_workers: number; cors_origins: string[]; database: string; outbound_network_calls: boolean; authentication: string }

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReactFlowProvider } from "@xyflow/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "@/api/client";
import { BeliefBars } from "@/charts/charts";
import { ToastProvider } from "@/components/Toast";
import { layoutTree } from "@/decision-tree/layout";
import { OutcomeNode } from "@/decision-tree/nodes";
import { NextExperimentCard } from "@/pages/session/LiveDashboard";
import { ExportButtons } from "@/pages/session/ReportPage";
import { validateSetup, availableInputs, type SetupState } from "@/pages/DiagnosticSetup";
import { checkFile } from "@/pages/ProjectDetail";
import { NewProjectModal, validateProjectName } from "@/pages/Projects";
import type { ExperimentMeta, Recommendation, Session, Tree } from "@/types/api";

const wrap = (ui: React.ReactNode) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <ToastProvider><MemoryRouter>{ui}</MemoryRouter></ToastProvider>
  </QueryClientProvider>
);

const exp = (id: string, o: Partial<ExperimentMeta> = {}): ExperimentMeta => ({ id, name: id, description: "", cost: 1, risk: 0, requires: [], outcomes: ["a", "b", "c"], requires_approval: false, limitations: [], has_likelihood: true, likelihood_runs: 10, ...o });

const baseSetup = (): SetupState => ({
  name: "s", planner: "adaptive", budget: 8, riskCap: 0.3, riskWeight: 1, tau: 0.9, epsilon: 0.01, sepWeight: 0, confidenceMode: "joint", maxActive: 3, humanInLoop: false,
  hyps: { a: { enabled: true, prior: 0.3 }, b: { enabled: true, prior: 0.2 } }, exps: { e1: true }, metric: { on: false, reference: "", recent: "", metric: "accuracy" },
});

describe("form validation", () => {
  it("validates project names", () => {
    expect(validateProjectName("   ")).toMatch(/name/i);
    expect(validateProjectName("x".repeat(121))).toMatch(/120/);
    expect(validateProjectName("Fraud model")).toBeNull();
  });
  it("shows a validation error and does not call the API for a blank project", async () => {
    const spy = vi.spyOn(api, "createProject");
    render(wrap(<NewProjectModal open onClose={() => {}} />));
    await userEvent.click(screen.getByRole("button", { name: /create project/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/name/i);
    expect(spy).not.toHaveBeenCalled();
  });
  it("checks upload files by extension, emptiness and size", () => {
    expect(checkFile(new File(["a,b"], "x.exe"))).toMatch(/csv/i);
    expect(checkFile(new File([], "x.csv"))).toMatch(/empty/i);
    expect(checkFile(new File(["a,b\n1,2"], "ok.csv"))).toBeNull();
    const big = new File(["x"], "big.csv"); Object.defineProperty(big, "size", { value: 30 * 1024 * 1024 });
    expect(checkFile(big)).toMatch(/limit/i);
  });
  it("validates the diagnostic setup", () => {
    const exps = [exp("e1")];
    expect(validateSetup(baseSetup(), new Set(), exps).errors).toEqual([]);
    expect(validateSetup({ ...baseSetup(), budget: 0 }, new Set(), exps).errors.join()).toMatch(/budget/);
    expect(validateSetup({ ...baseSetup(), hyps: { a: { enabled: true, prior: 0.3 } } }, new Set(), exps).errors.join()).toMatch(/two/);
    expect(validateSetup({ ...baseSetup(), hyps: { a: { enabled: true, prior: 1 }, b: { enabled: true, prior: 0.2 } } }, new Set(), exps).errors.join()).toMatch(/strictly/);
    expect(validateSetup({ ...baseSetup(), exps: { e1: false } }, new Set(), exps).errors.join()).toMatch(/experiment/);
    const needsLabels = [exp("e1", { requires: ["labels_recent"] })];
    expect(validateSetup(baseSetup(), new Set(), needsLabels).errors.join()).toMatch(/None of the enabled/);
    expect(validateSetup({ ...baseSetup(), humanInLoop: true }, new Set(), needsLabels).errors).toEqual([]);
    expect(validateSetup({ ...baseSetup(), budget: 0.5 }, new Set(), [exp("e1", { cost: 2 })]).errors.join()).toMatch(/budget and risk/);
    expect(validateSetup({ ...baseSetup(), metric: { on: true, reference: "1.4", recent: "x", metric: "accuracy" } }, new Set(), exps).errors.length).toBe(2);
  });
  it("derives available inputs from the mapping", () => {
    const a = availableInputs({ config: { mapping: { target: "y", prediction: "p" } }, datasets: [{ role: "replay" }] } as any);
    expect(a.has("labels_recent") && a.has("predictions_reference") && a.has("replay")).toBe(true);
    expect(availableInputs({ config: { mapping: {} }, datasets: [] } as any).has("labels_recent")).toBe(false);
  });
});

describe("API error states", () => {
  beforeEach(() => vi.restoreAllMocks());
  it("maps server errors to ApiError with code and message", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ error: { code: "approval_required", message: "needs approval", experiment_id: "retrain_recent" } }), { status: 409 }));
    await expect(api.run("s", {})).rejects.toMatchObject({ status: 409, code: "approval_required", message: "needs approval" });
  });
  it("reports an unreachable backend clearly", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("failed"));
    const err = await api.health().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe("network_error");
    expect(err.message).toMatch(/backend/i);
  });
  it("handles non-JSON 500 responses without leaking internals", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("<html>boom</html>", { status: 500 }));
    const err = await api.health().catch((e) => e);
    expect(err.status).toBe(500);
    expect(err.message).not.toMatch(/html/);
  });
});

const rec = (over: Partial<Recommendation> = {}): Recommendation => ({
  session_id: "s", status: "active", planner: "adaptive", planner_label: "Adaptive", selected: "feature_drift", rationale: "Because it is informative.",
  stop: { stop: false, code: null, message: "", confidence: 0.2 }, budget: 8, spent: 0, remaining: 8, diagnosis_confidence: 0.2, map_state: [], map_probability: 0.2, entropy: 3, needs_approval: false,
  experiments: [{ id: "feature_drift", name: "Feature-distribution comparison", description: "d", eig: 0.42, cost: 1, risk: 0, denominator: 1, score: 0.42, separation: 0.61, adjusted_score: 0.42, feasible: true, reasons: [], reason_codes: [], requires_approval: false, rank: 1, lookahead_value: null, outcomes: ["a"], requires: [], state: "feasible" }],
  ...over,
});
const sess = (status: Session["status"]) => ({ id: "s", status } as Session);

describe("diagnosis controls and state transitions", () => {
  it("shows EIG, cost, risk, separation and rationale from the recommendation", () => {
    render(wrap(<NextExperimentCard session={sess("active")} rec={rec()} busy={false} onRun={() => {}} onSkip={() => {}} onExternal={() => {}} onPick={() => {}} />));
    expect(screen.getByTestId("next-name")).toHaveTextContent("Feature-distribution comparison");
    expect(screen.getByText("0.420 bits")).toBeInTheDocument();
    expect(screen.getByText("0.61")).toBeInTheDocument();
    expect(screen.getByTestId("rationale")).toHaveTextContent("Because it is informative.");
  });
  it("fires the run / skip / pick / enter-result handlers", async () => {
    const h = { onRun: vi.fn(), onSkip: vi.fn(), onExternal: vi.fn(), onPick: vi.fn() };
    render(wrap(<NextExperimentCard session={sess("active")} rec={rec()} busy={false} {...h} />));
    await userEvent.click(screen.getByRole("button", { name: /run recommended/i }));
    await userEvent.click(screen.getByRole("button", { name: /^skip/i }));
    await userEvent.click(screen.getByRole("button", { name: /choose another/i }));
    await userEvent.click(screen.getByRole("button", { name: /enter result/i }));
    expect(h.onRun).toHaveBeenCalledOnce(); expect(h.onSkip).toHaveBeenCalledOnce(); expect(h.onPick).toHaveBeenCalledOnce(); expect(h.onExternal).toHaveBeenCalledOnce();
  });
  it.each(["paused", "completed", "stopped"] as const)("disables experiment controls when the session is %s", (st) => {
    render(wrap(<NextExperimentCard session={sess(st)} rec={rec()} busy={false} onRun={() => {}} onSkip={() => {}} onExternal={() => {}} onPick={() => {}} />));
    expect(screen.getByRole("button", { name: /run recommended/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /enter result/i })).toBeDisabled();
  });
  it("disables controls while an experiment is running", () => {
    render(wrap(<NextExperimentCard session={sess("active")} rec={rec()} busy onRun={() => {}} onSkip={() => {}} onExternal={() => {}} onPick={() => {}} />));
    expect(screen.getByRole("button", { name: /run recommended/i })).toBeDisabled();
  });
  it("explains why nothing is recommended when the planner has stopped", () => {
    render(wrap(<NextExperimentCard session={sess("active")} rec={rec({ selected: null, stop: { stop: true, code: "budget_insufficient", message: "The remaining budget cannot pay for any remaining experiment.", confidence: 0.4 } })} busy={false} onRun={() => {}} onSkip={() => {}} onExternal={() => {}} onPick={() => {}} />));
    expect(screen.getByText(/remaining budget cannot pay/i)).toBeInTheDocument();
  });
  it("warns that risky experiments need approval", () => {
    render(wrap(<NextExperimentCard session={sess("active")} rec={rec({ needs_approval: true })} busy={false} onRun={() => {}} onSkip={() => {}} onExternal={() => {}} onPick={() => {}} />));
    expect(screen.getByText(/explicit approval/i)).toBeInTheDocument();
  });
});

describe("charts render from real data", () => {
  it("BeliefBars labels the chart and mounts with data", () => {
    render(<BeliefBars marginals={[{ hypothesis: "covariate_drift", code: "H1", name: "Covariate drift", probability: 0.6 }, { hypothesis: "label_noise", code: "H2", name: "Label noise", probability: 0.1 }]} />);
    expect(screen.getByRole("img", { name: /root-cause probabilities/i })).toBeInTheDocument();
  });
});

const tree: Tree = {
  session_id: "s", hypotheses: ["a"],
  nodes: [
    { id: "root", type: "root", depth: 0, taken: true, data: { title: "Initial state", top: [], entropy: 3, spent: 0 } },
    { id: "exp1", type: "experiment", depth: 1, taken: true, data: { title: "Drift", eig: 0.4, cost: 1, risk: 0 } },
    { id: "exp1.o0", type: "outcome", depth: 2, taken: false, data: { title: "no shift", probability: 0.3, top: [], entropy: 2, hypothetical_next: "Class prior" } },
    { id: "exp1.o1", type: "outcome", depth: 2, taken: true, data: { title: "severe shift", probability: 0.5, top: [{ hypothesis: "covariate_drift", probability: 0.6 }], entropy: 2, spent: 1 } },
    { id: "exp1.o2", type: "outcome", depth: 2, taken: false, data: { title: "moderate", probability: 0.2, top: [], entropy: 2 } },
    { id: "pending", type: "pending", depth: 3, taken: false, data: { title: "Pipeline replay" } },
  ],
  edges: [
    { id: "root->exp1", source: "root", target: "exp1", taken: true, label: "" }, { id: "exp1->exp1.o0", source: "exp1", target: "exp1.o0", taken: false, label: "P=0.30" },
    { id: "exp1->exp1.o1", source: "exp1", target: "exp1.o1", taken: true, label: "P=0.50" }, { id: "exp1->exp1.o2", source: "exp1", target: "exp1.o2", taken: false, label: "P=0.20" },
    { id: "exp1.o1->pending", source: "exp1.o1", target: "pending", taken: false, label: "" },
  ],
};

describe("decision tree", () => {
  it("keeps the observed path on one lane and fans out unobserved outcomes", () => {
    const { nodes, edges } = layoutTree(tree);
    const y = (id: string) => nodes.find((n) => n.id === id)!.position.y;
    expect(y("root")).toBe(0); expect(y("exp1.o1")).toBe(0);
    expect(y("exp1.o0")).not.toBe(0); expect(y("exp1.o2")).not.toBe(0); expect(y("exp1.o0")).not.toBe(y("exp1.o2"));
    expect(edges.find((e) => e.id === "root->exp1")!.className).toBe("taken");
    expect(edges.find((e) => e.id === "exp1->exp1.o0")!.className).toBe("hypothetical");
    expect(nodes.find((n) => n.id === "exp1.o0")!.position.x).toBe(nodes.find((n) => n.id === "exp1.o1")!.position.x);
  });
  it("distinguishes observed from hypothetical outcomes in the node UI", () => {
    const base: any = { id: "x", type: "outcome", selected: false, dragging: false, isConnectable: false, zIndex: 0, positionAbsoluteX: 0, positionAbsoluteY: 0 };
    const P = ({ d }: { d: any }) => <ReactFlowProvider><OutcomeNode {...base} data={d} /></ReactFlowProvider>;
    const { rerender } = render(<P d={{ title: "severe shift", taken: true, probability: 0.5, top: [], entropy: 1 }} />);
    expect(screen.getByTestId("tree-outcome-taken")).toHaveTextContent("observed");
    rerender(<P d={{ title: "no shift", taken: false, probability: 0.3, top: [], entropy: 1, hypothetical_next: "Class prior" }} />);
    expect(screen.getByTestId("tree-outcome-hypothetical")).toHaveTextContent(/not observed/);
    expect(screen.getByText(/would then run/i)).toBeInTheDocument();
  });
});

describe("report downloads", () => {
  beforeEach(() => vi.restoreAllMocks());
  it("requests each export format from the API", async () => {
    const dl = vi.spyOn(api, "download").mockResolvedValue();
    render(wrap(<ExportButtons sessionId="abc" />));
    await userEvent.click(screen.getByRole("button", { name: /pdf report/i }));
    await userEvent.click(screen.getByRole("button", { name: /^json$/i }));
    await userEvent.click(screen.getByRole("button", { name: /hypotheses csv/i }));
    await waitFor(() => expect(dl).toHaveBeenCalledTimes(3));
    expect(dl.mock.calls.map((c) => c[1])).toEqual(["report.pdf", "report.json", "hypotheses.csv"]);
  });
  it("surfaces a download failure as a notification", async () => {
    vi.spyOn(api, "download").mockRejectedValue(new ApiError(404, "not_found", "report missing"));
    render(wrap(<ExportButtons sessionId="abc" />));
    fireEvent.click(screen.getByRole("button", { name: /pdf report/i }));
    expect(await screen.findByText("report missing")).toBeInTheDocument();
  });
});

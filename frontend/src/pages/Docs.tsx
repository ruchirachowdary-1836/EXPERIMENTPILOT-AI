import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "@/api/client";
import { Badge, Callout, Card, PageHeader, Tabs } from "@/components/ui";
import { title } from "@/lib/utils";

const Code = ({ children }: { children: string }) => <pre className="my-3 overflow-x-auto rounded-xl bg-sunken px-4 py-3 font-mono text-[13px] leading-relaxed text-accent">{children}</pre>;

function Methodology() {
  return (
    <div className="prose-ep space-y-2">
      <h3>The diagnostic loop</h3>
      <p>Start from a belief over which root causes are active, then repeat: filter experiments by feasibility, score them, run the best, observe a discretised outcome, update by Bayes' rule, and check stopping rules. Everything is recorded for the report and decision tree.</p>
      <h3>Belief representation</h3>
      <p>The state is a binary vector <code>z ∈ {"{0,1}"}ᴷ</code> saying which of K causes are active, so multiple simultaneous causes are first-class. The prior is independent Bernoulli per cause, truncated to at most <i>m</i> simultaneous causes (default 3) and renormalised. With <code>m = K</code> the belief is exact; for large K the truncation keeps the state space at Σⱼ≤ₘ C(K,j), the documented approximation strategy.</p>
      <h3>Likelihood model</h3>
      <Code>{`P(o ≥ j | z) = 1 − (1 − leak_j) · ∏_{k active in z} (1 − sens_{k,j})`}</Code>
      <p>An ordinal cumulative noisy-OR. <code>leak</code> is the false-alarm rate with nothing active; <code>sens</code> is how strongly one cause raises the outcome level. Parameters are estimated from fault-injection counts with Beta smoothing toward conservative expert priors. Every experiment is noisy; no outcome is ever certain.</p>
      <h3>Expected information gain</h3>
      <Code>{`EIG(e) = H(b) − Σ_o P(o|b,e) · H(b|o,e)`}</Code>
      <h3>Selection and stopping</h3>
      <Code>{`score(e) = EIG(e) / max(c(e) + λ·r(e), floor)   s.t. c(e) ≤ budget left, r(e) ≤ risk cap`}</Code>
      <p>Stop when the configured confidence is reached, no feasible experiment remains, the remaining budget cannot pay for any experiment, the best score falls below ε, or every remaining experiment is blocked by the risk cap. If the evidence does not support a diagnosis the report says so.</p>
      <h3>Separation score</h3>
      <p>For the two leading hypotheses a and b: the total-variation distance between <code>P(o | a active)</code> and <code>P(o | b active)</code>. It explains in plain terms how well an experiment tells them apart and can be added as a bonus.</p>
      <h3>Planner variants</h3>
      <ul><li><b>Random feasible</b>, <b>Fixed checklist</b>, <b>Cheapest first</b>: baselines.</li><li><b>Max information gain</b>: ignores cost and risk.</li><li><b>Information gain per cost</b>: ignores risk.</li><li><b>Cost-and-risk-aware adaptive</b>: the main method.</li><li><b>Two-step lookahead</b>: value of a pair of experiments with the best greedy follow-up per outcome.</li></ul>
    </div>
  );
}

function Inputs() {
  return (
    <div className="prose-ep space-y-2">
      <h3>Datasets</h3>
      <p>CSV only (UTF-8, header row, at most 25 MB by default). <b>Reference</b>: training/validation data with the model's predictions and, ideally, true labels. <b>Recent</b>: recent production data with the same columns. <b>Replay</b> (optional): the recent records as processed by the previous preprocessing pipeline.</p>
      <h3>Column roles</h3>
      <ul><li><b>Target</b>: ground-truth class. Without it, label-based experiments are unavailable.</li><li><b>Prediction</b>: the model's predicted class.</li><li><b>Timestamp / identifier</b>: optional, used for temporal checks and row alignment.</li><li>Every other column is a model input feature.</li></ul>
      <h3>Models</h3>
      <p>Model files are never uploaded or loaded. The model's behaviour enters through its prediction column. Experiments that fit models use small scikit-learn surrogates trained on a copy of your data.</p>
      <h3>Human-in-the-loop</h3>
      <p>If data cannot be uploaded, enable human-in-the-loop mode: the planner recommends an experiment, you run it yourself, and you enter the outcome.</p>
    </div>
  );
}

function Privacy() {
  return (
    <div className="prose-ep space-y-2">
      <h3>Local-first</h3>
      <p>Data are processed on the machine running the backend. The application makes no outbound network calls while processing uploaded data.</p>
      <h3>Handling of uploads</h3>
      <ul><li>Size, row and column limits; extension and content validation; strict CSV parsing.</li><li>Files are re-serialised under server-generated names; user file names are never used as paths.</li><li>Uploads are never executed, imported or unpickled.</li><li>Experiments run on copies, in a worker process with a time limit and single-threaded numerics.</li></ul>
      <h3>Deletion</h3>
      <p>Deleting a project removes its database records and every stored data file. Datasets can also be removed individually while no sessions reference them.</p>
      <h3>Authentication</h3>
      <p>The API has no authentication layer. Run it on a trusted local machine or behind your own authenticating proxy. CORS is restricted to configured origins.</p>
    </div>
  );
}

function Plugins() {
  const exps = useQuery({ queryKey: ["exp-meta"], queryFn: api.experiments });
  return (
    <div className="space-y-4">
      <div className="prose-ep"><p>Experiments are plugins implementing <code>Experiment</code> (typed inputs/outputs, cost, risk, required inputs, outcome categories, raw results, failure handling). Register one with <code>register_experiment()</code> and add a likelihood model with the same id.</p></div>
      {exps.data?.map((e) => (
        <Card key={e.id} className="card-pad">
          <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="text-base">{e.name}</h3><div className="flex gap-1.5"><Badge>cost {e.cost}</Badge><Badge tone={e.risk > 0.15 ? "warn" : "muted"}>risk {e.risk}</Badge>{e.requires_approval && <Badge tone="warn">approval</Badge>}</div></div>
          <p className="mt-1 font-mono text-xs text-faint">{e.id}</p>
          <p className="mt-2 text-sm text-muted">{e.description}</p>
          <p className="mt-3 text-xs"><span className="text-faint">Requires:</span> {e.requires.map((r) => title(r)).join(", ")}</p>
          <p className="mt-1 text-xs"><span className="text-faint">Outcomes:</span> {e.outcomes.map((o) => title(o)).join(" → ")}</p>
          <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-muted">{e.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
        </Card>))}
    </div>
  );
}

function Limits() {
  return (
    <div className="space-y-4">
      <Callout tone="warn" title="Statistical indicators are not proof of causation">An experiment can show that a symptom pattern is consistent with a cause. Confirming it requires domain knowledge and, usually, a fix that restores performance.</Callout>
      <div className="prose-ep space-y-2"><ul>
        <li>Likelihood tables come from <b>simulated</b> fault injection on built-in datasets. Real failures may look different; treat probabilities as relative evidence, not calibrated truth outside the benchmark.</li>
        <li>Experiment outcomes are modelled as conditionally independent given the active causes.</li>
        <li>The multi-cause belief assumes at most a few simultaneous causes.</li>
        <li>The label audit cannot tell label noise from hard examples, and the leakage scan cannot tell leakage from a feature that broke upstream.</li>
        <li>Model/code regression is an extensible hypothesis with no discriminating experiment yet.</li>
        <li>Novelty is unproven. Adaptive and sequential diagnosis is an established field; this project does not claim to be the first.</li></ul></div>
    </div>
  );
}

export default function Docs() {
  const [tab, setTab] = useState<"method" | "inputs" | "privacy" | "plugins" | "limits">("method");
  return (
    <div>
      <PageHeader eyebrow="Reference" title="Documentation" subtitle="Methodology, supported inputs, privacy, experiment plugins and research limitations." />
      <Tabs className="mb-6" value={tab} onChange={setTab} tabs={[{ id: "method", label: "Methodology" }, { id: "inputs", label: "Supported inputs" }, { id: "privacy", label: "Privacy & security" }, { id: "plugins", label: "Experiment plugins" }, { id: "limits", label: "Limitations" }]} />
      <Card className="card-pad max-w-4xl">{tab === "method" && <Methodology />}{tab === "inputs" && <Inputs />}{tab === "privacy" && <Privacy />}{tab === "plugins" && <Plugins />}{tab === "limits" && <Limits />}</Card>
    </div>
  );
}

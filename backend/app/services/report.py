"""Final diagnosis report and decision-tree construction (all numbers come from the stored session)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from experimentpilot.core import Belief, LikelihoodBank, PlanState, get_hypothesis, get_variant
from experimentpilot.core.belief import entropy_bits
from experimentpilot.core.eig import expected_information_gain, outcome_distribution, posterior_given_outcome
from experimentpilot.core.planner import PlannerConfig, score_experiments
from experimentpilot.experiments import EXPERIMENTS

from ..models import DiagnosisSession
from .sessions import (build_belief, enabled_specs, hypothesis_ids, library, plan_state, planner_config, prior_belief)

REMEDIATION = {
    "covariate_drift": ["Refresh the training data with recent samples and retrain, then validate on a time-ordered holdout.",
                        "Add per-feature drift monitoring with alert thresholds (PSI/KS) and a retraining trigger.",
                        "Consider importance weighting or domain-robust features if drift is expected to recur."],
    "label_noise": ["Audit a stratified sample of recent labels with domain experts and compare against the labelling guideline.",
                    "If the label definition changed, relabel or version the target and retrain on consistent labels.",
                    "Use noise-robust training or filter suspicious labels flagged by confident learning, then re-evaluate."],
    "feature_leakage": ["Identify the suspect feature(s) and check whether they are available, and defined identically, at prediction time.",
                        "Remove or re-derive the leaking feature, retrain, and validate with a time-based split.",
                        "Add a pre-training leakage check (single-feature predictive power, temporal validation) to the pipeline."],
    "class_prior_shift": ["Recalibrate decision thresholds or apply prior-shift correction using the new class frequencies.",
                          "Re-weight or resample the training data to the current class mix and re-validate precision and recall.",
                          "Track label and prediction distributions continuously."],
    "pipeline_change": ["Diff the serving and training preprocessing code/config (scalers, encoders, units, imputation) and roll back or align them.",
                        "Add a regression test that feeds fixed raw records through both pipelines and compares outputs.",
                        "Version preprocessing artefacts together with the model."],
    "data_quality": ["Trace the flagged features upstream (null spikes, sentinel values, stale or constant columns) and fix the source.",
                     "Add schema and range validation at ingestion with alerting.",
                     "Decide on explicit handling for missing values and re-validate model performance after the fix."],
    "metric_bug": ["Recompute the metric from raw predictions and labels in an independent notebook and compare split definitions.",
                   "Check for misaligned joins, stale label tables or mismatched evaluation windows."],
    "model_regression": ["Compare against the previous model artefact on identical data and review code/config changes in the release.",
                         "Roll back to the previous artefact while investigating; check random-seed variance."],
}

STATIC_LIMITATIONS = [
    "Results are statistical indicators computed on the supplied samples; they are not proof of a causal root cause.",
    "Posterior probabilities depend on the likelihood tables, which were estimated from simulated fault injection on "
    "built-in datasets (see Methodology); real systems may behave differently.",
    "Experiment outcomes are modelled as conditionally independent given the active causes, an approximation.",
    "At most a few simultaneous causes are modelled (sparsity assumption set when the session was created).",
]


def _status(p: float) -> str:
    return "likely" if p >= 0.8 else "probable" if p >= 0.5 else "possible" if p >= 0.2 else "unlikely"


def evidence_direction(sess: DiagnosisSession) -> dict[str, list[dict]]:
    """Per hypothesis: likelihood ratio of each observed outcome, P(o | h active) / P(o | h inactive), under the prior."""
    prior = prior_belief(sess)
    bank = LikelihoodBank(library(), prior.hypotheses, prior.states)
    out: dict[str, list[dict]] = {h: [] for h in prior.hypotheses}
    for r in sess.runs:
        if r.status not in ("completed", "external") or r.outcome_index is None or r.experiment_id not in library():
            continue
        lik = bank.matrix(r.experiment_id)
        for k, h in enumerate(prior.hypotheses):
            a = float(prior.conditional_on(k, True) @ lik[:, r.outcome_index])
            b = float(prior.conditional_on(k, False) @ lik[:, r.outcome_index])
            lr = a / max(b, 1e-9)
            direction = "supports" if lr >= 1.5 else "contradicts" if lr <= 1 / 1.5 else "neutral"
            out[h].append({"run_id": r.id, "experiment_id": r.experiment_id, "experiment": EXPERIMENTS[r.experiment_id].name,
                           "outcome": r.outcome, "likelihood_ratio": lr, "direction": direction})
    return out


def run_summary(r) -> dict:
    return {"id": r.id, "experiment_id": r.experiment_id, "name": EXPERIMENTS[r.experiment_id].name if r.experiment_id in EXPERIMENTS else r.experiment_id,
            "status": r.status, "source": r.source, "step": r.step, "outcome": r.outcome, "outcome_index": r.outcome_index,
            "score": r.score, "cost": r.cost if r.status in ("completed", "external") else 0.0, "risk": r.risk,
            "interpretation": r.interpretation, "findings": r.findings, "limitations": r.limitations, "stats": r.stats,
            "recommendation": r.recommendation, "note": r.note, "error": r.error, "runtime_s": r.runtime_s,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None}


def build_report(db: Session, sess: DiagnosisSession) -> dict:
    belief, prior = build_belief(sess), prior_belief(sess)
    marg, pm = belief.marginals(), prior.marginals()
    ev = evidence_direction(sess)
    ranking = []
    for i in np.argsort(-marg, kind="stable"):
        h = belief.hypotheses[i]
        hyp = get_hypothesis(h)
        ranking.append({"hypothesis": h, "code": hyp.code, "name": hyp.name, "description": hyp.description,
                        "posterior": float(marg[i]), "prior": float(pm[i]), "change": float(marg[i] - pm[i]),
                        "status": _status(float(marg[i])),
                        "supporting": [e for e in ev[h] if e["direction"] == "supports"],
                        "contradicting": [e for e in ev[h] if e["direction"] == "contradicts"],
                        "remediation": REMEDIATION.get(h, []) if marg[i] >= 0.2 else []})
    diagnosed = [r["hypothesis"] for r in ranking if r["posterior"] >= 0.5]
    ms, mp = belief.map_state()
    h0, hmax = prior.entropy(), belief.entropy()
    top_states = []
    for j in np.argsort(-belief.probs)[:4]:
        top_states.append({"causes": [h for h, a in zip(belief.hypotheses, belief.states[j]) if a] or ["no fault"],
                           "probability": float(belief.probs[j])})
    runs = [run_summary(r) for r in sess.runs]
    done = [r for r in runs if r["status"] in ("completed", "external")]
    cfg = sess.config
    undecided = [r["hypothesis"] for r in ranking if 0.2 < r["posterior"] < 0.8]
    limitations = list(STATIC_LIMITATIONS)
    miss = sess.symptoms.get("missing_inputs", [])
    if miss:
        limitations.append("Missing inputs made some experiments unavailable: " + ", ".join(miss) + ".")
    unrun = [EXPERIMENTS[e].name for e in sess.experiments if e in EXPERIMENTS and not any(
        r["experiment_id"] == e and r["status"] in ("completed", "external") for r in runs)]
    if unrun:
        limitations.append("Experiments not performed: " + ", ".join(unrun) + ".")
    if any(r["source"] == "external" for r in done):
        limitations.append("Some outcomes were entered manually and could not be verified by the platform.")
    for r in done:
        limitations.extend(f"{r['name']}: {l}" for l in r["limitations"][:2])
    limitations = list(dict.fromkeys(limitations))
    alternatives = [{"hypothesis": r["hypothesis"], "name": r["name"], "posterior": r["posterior"]}
                    for r in ranking if r["hypothesis"] not in diagnosed and r["posterior"] >= 0.15]
    verdict = ("No root cause reached the 50% decision level; the evidence does not support a specific diagnosis." if not diagnosed else
               "Diagnosed cause" + ("s" if len(diagnosed) > 1 else "") + ": " +
               ", ".join(get_hypothesis(h).name for h in diagnosed) + ".")
    return {
        "session": {"id": sess.id, "name": sess.name, "status": sess.status, "planner": sess.planner_variant,
                    "project_id": sess.project_id, "created_at": sess.created_at.isoformat(),
                    "finished_at": sess.finished_at.isoformat() if sess.finished_at else None,
                    "stop_code": sess.stop_code, "stop_message": sess.stop_message, "config": {k: v for k, v in cfg.items() if k != "dataset_ids"}},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict, "diagnosed": diagnosed, "ranking": ranking,
        "confidence": {"map_state": list(ms), "map_probability": mp, "threshold": cfg["confidence_threshold"],
                       "mode": cfg["confidence_mode"], "reached": mp >= cfg["confidence_threshold"],
                       "entropy_bits": hmax, "initial_entropy_bits": h0,
                       "uncertainty_reduction": float(1 - hmax / h0) if h0 > 0 else 0.0,
                       "undecided_hypotheses": undecided},
        "top_states": top_states, "alternatives": alternatives,
        "symptoms": sess.symptoms,
        "experiments": runs,
        "cost": {"spent": sess.spent, "budget": cfg["budget"], "utilization": sess.spent / cfg["budget"] if cfg["budget"] else 0.0,
                 "remaining": max(cfg["budget"] - sess.spent, 0.0), "experiments_run": len(done)},
        "risk": {"total": sess.risk_total, "max": max([r["risk"] for r in done] or [0.0]), "cap": cfg["risk_cap"],
                 "violations": sum(1 for r in done if r["risk"] > cfg["risk_cap"] + 1e-12),
                 "approved_risky": [r["name"] for r in done if r["risk"] > 0.15]},
        "posterior_history": [{"step": s.step, "label": s.label, "marginals": s.marginals, "entropy": s.entropy,
                               "map_probability": s.map_probability, "spent": s.spent} for s in sess.snapshots],
        "limitations": limitations,
        "demo_truth": (sess.project.config or {}).get("demo_truth") if sess.project else None,
        "disclaimer": "Statistical indicators, not causal proof. Likelihood tables come from simulated fault injection.",
    }


# ------------------------------------------------------------------ decision tree
def build_tree(db: Session, sess: DiagnosisSession) -> dict:
    hyps = hypothesis_ids(sess)
    prior = prior_belief(sess)
    bank = LikelihoodBank(library(), prior.hypotheses, prior.states)
    cfg: PlannerConfig = planner_config(sess)
    specs = enabled_specs(sess)
    variant = get_variant(sess.planner_variant)
    snaps = list(sess.snapshots)
    taken = [r for r in sess.runs if r.status in ("completed", "external")]
    taken.sort(key=lambda r: r.step or 0)
    nodes: list[dict] = []
    edges: list[dict] = []

    def top(probs: np.ndarray, n: int = 3):
        b = Belief(prior.hypotheses, prior.states, probs)
        return [{"hypothesis": h, "probability": p} for h, p in b.ranked()[:n]]

    def add_edge(a, b, taken_flag, label=""):
        edges.append({"id": f"{a}->{b}", "source": a, "target": b, "taken": taken_flag, "label": label})

    root_probs = np.array(snaps[0].probs)
    nodes.append({"id": "root", "type": "root", "depth": 0, "taken": True,
                  "data": {"title": "Initial state", "top": top(root_probs), "entropy": snaps[0].entropy,
                           "spent": 0.0}})
    parent = "root"
    state = PlanState(available_inputs=set(sess.config.get("available_inputs", [])))
    spent = 0.0
    for i, r in enumerate(taken):
        before = Belief(prior.hypotheses, prior.states, np.array(snaps[min(i, len(snaps) - 1)].probs))
        after_snap = snaps[min(i + 1, len(snaps) - 1)]
        spec = EXPERIMENTS[r.experiment_id]
        lik = bank.matrix(r.experiment_id)
        po = outcome_distribution(before.probs, lik)
        eig = expected_information_gain(before.probs, lik)
        eid = f"exp{i + 1}"
        nodes.append({"id": eid, "type": "experiment", "depth": 2 * i + 1, "taken": True,
                      "data": {"title": spec.name, "experiment_id": spec.id, "eig": eig, "cost": spec.cost, "risk": spec.risk,
                               "chosen_by": (r.recommendation or {}).get("chosen_by", "planner"), "source": r.source}})
        add_edge(parent, eid, True)
        state_after = PlanState(spent=spent + spec.cost, done={**state.done, spec.id: 1},
                                available_inputs=state.available_inputs, n_run=state.n_run + 1)
        for o, name in enumerate(spec.outcomes):
            oid = f"{eid}.o{o}"
            is_taken = (o == r.outcome_index)
            if is_taken:
                probs = np.array(after_snap.probs)
            else:
                probs = posterior_given_outcome(before.probs, lik, o) if po[o] > 1e-9 else before.probs
            nxt = None
            if not is_taken and po[o] > 1e-9:
                bb = Belief(prior.hypotheses, prior.states, probs)
                ch = variant.choose(bb, specs, bank, cfg, PlanState(spent=state_after.spent, done=dict(state_after.done),
                                    available_inputs=set(state_after.available_inputs), n_run=state_after.n_run),
                                    np.random.default_rng(0))
                nxt = ch.selected.name if ch.selected else None
            nodes.append({"id": oid, "type": "outcome", "depth": 2 * i + 2, "taken": is_taken,
                          "data": {"title": name.replace("_", " "), "probability": float(po[o]), "top": top(probs),
                                   "entropy": entropy_bits(probs), "hypothetical_next": nxt}})
            add_edge(eid, oid, is_taken, f"P={po[o]:.2f}")
            if is_taken:
                nodes[-1]["data"]["spent"] = spent + spec.cost
        parent = f"{eid}.o{r.outcome_index}"
        state = state_after
        spent += spec.cost
    # terminal / pending node
    depth = 2 * len(taken) + 1
    if sess.status in ("completed", "stopped"):
        nodes.append({"id": "end", "type": "stop", "depth": depth, "taken": True,
                      "data": {"title": "Stopped", "code": sess.stop_code, "message": sess.stop_message, "spent": sess.spent}})
        add_edge(parent, "end", True)
    else:
        from .sessions import recommend
        rec = recommend(db, sess)
        sel = next((e for e in rec["experiments"] if e["id"] == rec["selected"]), None)
        nodes.append({"id": "pending", "type": "pending", "depth": depth, "taken": False,
                      "data": {"title": sel["name"] if sel else "No feasible experiment", "experiment_id": rec["selected"],
                               "eig": sel["eig"] if sel else 0.0, "cost": sel["cost"] if sel else 0.0,
                               "message": rec["stop"]["message"] if rec["stop"]["stop"] else "Recommended next experiment"}})
        add_edge(parent, "pending", False)
    return {"session_id": sess.id, "nodes": nodes, "edges": edges, "hypotheses": list(hyps)}

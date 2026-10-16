"""Reproducible fault-injection benchmark.

Pipeline:  (1) build scenarios  ->  (2) run all experiments on each (real computations)  ->
(3) estimate likelihoods from *estimation* seeds only  ->  (4) replay planners on *evaluation* seeds.
Planners reveal pre-computed, genuinely measured outcomes, so every planner faces identical evidence.
"""
from __future__ import annotations

import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..core.belief import Belief
from ..core.diagnosis import run_diagnosis
from ..core.hypotheses import all_hypotheses
from ..core.likelihood_model import LikelihoodLibrary
from ..core.planner import LikelihoodBank, PlannerConfig
from ..core.variants import VARIANTS, get_variant
from ..experiments import EXPERIMENTS
from ..faults.injectors import FAULTS, FAULT_TO_HYPOTHESIS
from ..faults.scenario import ScenarioSpec, build_scenario, compute_outcomes

BENCH_HYPOTHESES = ("covariate_drift", "label_noise", "feature_leakage", "class_prior_shift", "pipeline_change",
                    "data_quality", "metric_bug")

DEFAULT_CONFIG = {
    "name": "default-v1",
    "estimation_datasets": ["synthetic_credit", "synthetic_churn"],
    "estimation_seeds": list(range(0, 12)),
    "evaluation_datasets": ["synthetic_credit", "synthetic_churn", "breast_cancer"],
    "evaluation_seeds": list(range(100, 108)),
    "severities": ["low", "medium", "high"],
    "faults": list(FAULTS),
    "pairs": [["covariate_drift", "label_noise"], ["covariate_drift", "class_prior_shift"],
              ["feature_leakage", "data_quality"], ["pipeline_change", "label_noise"],
              ["data_quality", "class_prior_shift"], ["covariate_drift", "pipeline_change"],
              ["label_noise", "class_prior_shift"], ["feature_leakage", "covariate_drift"]],
    "pair_severity": "medium",
    "planners": list(VARIANTS),
    "budget": 8.0, "risk_cap": 0.15, "risk_weight": 1.0, "confidence_threshold": 0.9, "min_score": 0.01,
    "separation_weight": 0.0, "max_active": 3, "pseudo_count": 2.0,
    "budgets": [1, 2, 3, 4, 6, 8, 12, 18],
    "robustness_sigmas": [0.0, 0.5, 1.0, 1.5],
    "bootstrap": 1000, "planner_seed": 12345,
}


def build_specs(datasets: list[str], seeds: list[int], cfg: dict, pairs: bool = True) -> list[ScenarioSpec]:
    specs: list[ScenarioSpec] = []
    for ds in datasets:
        for seed in seeds:
            specs.append(ScenarioSpec(ds, (), seed))
            for f in cfg["faults"]:
                for s in cfg["severities"]:
                    specs.append(ScenarioSpec(ds, ((f, s),), seed))
            if pairs:
                for a, b in cfg["pairs"]:
                    if a in cfg["faults"] and b in cfg["faults"]:
                        specs.append(ScenarioSpec(ds, ((a, cfg["pair_severity"]), (b, cfg["pair_severity"])), seed))
    return specs


def _compute_one(args: tuple) -> dict:
    dataset, faults, seed = args
    spec = ScenarioSpec(dataset, tuple(tuple(x) for x in faults), seed)
    t0 = time.perf_counter()
    sc = build_scenario(spec)
    outcomes = compute_outcomes(sc)
    return {"id": spec.id, "dataset": dataset, "faults": [list(f) for f in spec.faults], "seed": seed,
            "truth": list(spec.truth), "outcomes": outcomes, "accuracy_drop": sc.meta["accuracy_drop"],
            "accuracy_reference": sc.meta["accuracy_reference"], "accuracy_recent": sc.meta["accuracy_recent"],
            "runtime_s": time.perf_counter() - t0}


def compute_records(specs: list[ScenarioSpec], workers: int = 4, cache: Path | None = None,
                    progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """Run every experiment on every scenario (parallel, cached by scenario id)."""
    cached: dict[str, dict] = {}
    if cache and cache.exists():
        for line in cache.read_text(encoding="utf8").splitlines():
            if line.strip():
                r = json.loads(line)
                cached[r["id"]] = r
    todo = [s for s in specs if s.id not in cached]
    args = [(s.dataset, [list(f) for f in s.faults], s.seed) for s in todo]
    done = 0
    if args:
        out = cache.open("a", encoding="utf8") if cache else None
        try:
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    for rec in ex.map(_compute_one, args, chunksize=2):
                        cached[rec["id"]] = rec
                        done += 1
                        if out:
                            out.write(json.dumps(rec) + "\n"); out.flush()
                        if progress:
                            progress(done, len(args))
            else:
                for a in args:
                    rec = _compute_one(a)
                    cached[rec["id"]] = rec
                    done += 1
                    if out:
                        out.write(json.dumps(rec) + "\n"); out.flush()
                    if progress:
                        progress(done, len(args))
        finally:
            if out:
                out.close()
    return [cached[s.id] for s in specs]


# ------------------------------------------------------------------ evaluation
def _planner_cfg(cfg: dict, budget: float | None = None) -> PlannerConfig:
    return PlannerConfig(budget=cfg["budget"] if budget is None else budget, risk_cap=cfg["risk_cap"],
                         risk_weight=cfg["risk_weight"], confidence_threshold=cfg["confidence_threshold"],
                         min_score=cfg["min_score"], separation_weight=cfg["separation_weight"]).validate()


def _metrics_row(rec: dict, trace, hyps: tuple[str, ...]) -> dict:
    m = trace.belief.marginals()
    truth = set(rec["truth"])
    order = list(np.argsort(-m, kind="stable"))
    ranked = [hyps[i] for i in order]
    diag = {h for h, p in zip(hyps, m) if p >= 0.5}
    tp = len(diag & truth)
    prec = tp / len(diag) if diag else (1.0 if not truth else 0.0)
    recall = tp / len(truth) if truth else (1.0 if not diag else 0.0)
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    t = np.array([h in truth for h in hyps], dtype=float)
    return {"top1": float(ranked[0] in truth) if truth else np.nan,
            "top3": float(bool(set(ranked[:3]) & truth)) if truth else np.nan,
            "precision": prec, "recall": recall, "f1": f1, "exact": float(diag == truth),
            "brier": float(np.mean((m - t) ** 2)), "cost": trace.spent, "n_experiments": trace.n_experiments,
            "risk_total": trace.total_risk, "risk_violations": trace.risk_violations,
            "stop": trace.stop.code if trace.stop else None, "confidence": float(trace.belief.map_state()[1]),
            "marginals": m.tolist(), "truth_vec": t.tolist(), "path": [s.experiment_id for s in trace.steps]}


def run_planners(records: list[dict], library: LikelihoodLibrary, cfg: dict, planners: list[str] | None = None,
                 budget: float | None = None, stop_on_confidence: bool = True, sigma: float = 0.0,
                 hyps: tuple[str, ...] = BENCH_HYPOTHESES) -> pd.DataFrame:
    """Replay planners on pre-computed outcomes. Returns one row per (record, planner)."""
    planners = planners or cfg["planners"]
    priors = [next(h.default_prior for h in all_hypotheses() if h.id == k) for k in hyps]
    belief0 = Belief.from_priors(hyps, priors, max_active=cfg["max_active"])
    specs_all = [e.spec() for e in EXPERIMENTS.values()]
    pcfg = _planner_cfg(cfg, budget)
    rows = []
    for ri, rec in enumerate(records):
        lib = library if sigma <= 0 else library.perturbed(sigma, np.random.default_rng([cfg["planner_seed"], ri]))
        bank = LikelihoodBank(lib, hyps, belief0.states)
        avail_ids = {e for e, o in rec["outcomes"].items() if "outcome_index" in o}
        specs = [s for s in specs_all if s.id in lib.models]
        oracle = lambda e, r=rec: (r["outcomes"][e]["outcome_index"] if e in avail_ids else None)
        for pid in planners:
            v = get_variant(pid)
            rng = np.random.default_rng([cfg["planner_seed"], ri, zlib.crc32(pid.encode()) % 100000])
            tr = run_diagnosis(v, belief0, specs, bank, pcfg, oracle, {"features", "reference", "recent", "labels_reference",
                               "labels_recent", "predictions_reference", "predictions_recent", "replay", "reported_metrics"},
                               rng=rng, report_risk_cap=cfg["risk_cap"], stop_on_confidence=stop_on_confidence)
            row = {"scenario": rec["id"], "dataset": rec["dataset"], "seed": rec["seed"], "planner": pid,
                   "truth": "+".join(rec["truth"]) or "none", "n_truth": len(rec["truth"]), "budget": pcfg.budget}
            row.update(_metrics_row(rec, tr, hyps))
            rows.append(row)
    return pd.DataFrame(rows)

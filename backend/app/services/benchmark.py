"""Benchmark laboratory: background jobs that run the engine's real fault-injection benchmark."""
from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experimentpilot.benchmark import DEFAULT_CONFIG, build_specs, compute_records, run_planners
from experimentpilot.benchmark.analysis import (agreement_matrix, calibration, make_plots, paired_tests, summarise,
                                                 bootstrap_ci)
from experimentpilot.core.variants import VARIANTS
from experimentpilot.faults import DATASETS, FAULTS, SEVERITIES

from ..core.config import get_settings
from ..core.db import session_scope
from ..models import BenchmarkJob
from ..schemas.api import BenchmarkCreate
from .sessions import ServiceError, library

REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLISHED_DIR = REPO_ROOT / "research" / "results"
_threads: dict[str, threading.Thread] = {}


def published_path(name: str = "default-v1") -> Path:
    return PUBLISHED_DIR / f"results_{name}.json"


def validate(req: BenchmarkCreate) -> None:
    if bad := [d for d in req.datasets if d not in DATASETS]:
        raise ServiceError(422, "invalid_dataset", f"unknown datasets: {', '.join(bad)}")
    if bad := [f for f in req.faults if f not in FAULTS]:
        raise ServiceError(422, "invalid_fault", f"unknown fault injectors: {', '.join(bad)}")
    if bad := [s for s in req.severities if s not in SEVERITIES]:
        raise ServiceError(422, "invalid_severity", f"unknown severities: {', '.join(bad)}")
    if bad := [p for p in req.planners if p not in VARIANTS]:
        raise ServiceError(422, "invalid_planner", f"unknown planners: {', '.join(bad)}")
    if not req.datasets or not req.planners:
        raise ServiceError(422, "empty_selection", "choose at least one dataset and one planner")
    prov = library().provenance
    overlap = set(req.seeds) & set(prov.get("seeds", [])) if set(req.datasets) & set(prov.get("dataset_names", [])) else set()
    if overlap:
        raise ServiceError(422, "seed_overlap", "seeds " + ", ".join(map(str, sorted(overlap))) +
                           " were used to estimate the shipped likelihood tables; choose different seeds so evaluation data stay separate")
    n = len(req.datasets) * len(req.seeds) * (len(req.faults) * len(req.severities) + 1 + (8 if req.include_pairs else 0))
    if n > 400:
        raise ServiceError(422, "too_large", f"this configuration needs {n} scenarios; the limit is 400 per run")


def create_job(db, req: BenchmarkCreate) -> BenchmarkJob:
    validate(req)
    job = BenchmarkJob(name=req.name, config=req.model_dump(), status="queued", message="Queued")
    db.add(job)
    db.commit()
    t = threading.Thread(target=_run_job, args=(job.id,), daemon=True, name=f"bench-{job.id}")
    _threads[job.id] = t
    t.start()
    return job


def _update(job_id: str, **kw) -> None:
    with session_scope() as db:
        j = db.get(BenchmarkJob, job_id)
        for k, v in kw.items():
            setattr(j, k, v)


def _run_job(job_id: str) -> None:
    s = get_settings()
    try:
        with session_scope() as db:
            req = BenchmarkCreate(**db.get(BenchmarkJob, job_id).config)
            db.get(BenchmarkJob, job_id).status = "running"
        cfg = {**DEFAULT_CONFIG, "datasets": req.datasets, "faults": req.faults, "severities": req.severities,
               "budget": req.budget, "risk_cap": req.risk_cap, "risk_weight": req.risk_weight,
               "confidence_threshold": req.confidence_threshold, "planners": req.planners, "budgets": req.budgets,
               "bootstrap": 500}
        specs = build_specs(req.datasets, req.seeds, cfg, pairs=req.include_pairs)
        out = Path(s.data_dir) / "benchmark" / job_id
        out.mkdir(parents=True, exist_ok=True)
        cache = Path(s.data_dir) / "benchmark" / "outcome_cache.jsonl"

        def prog(d, n):
            _update(job_id, progress=0.85 * d / max(n, 1), message=f"Running experiments on scenario {d}/{n}")
        records = compute_records(specs, s.benchmark_workers, cache, prog)
        _update(job_id, progress=0.88, message="Replaying planners")
        lib = library()
        df = run_planners(records, lib, cfg, req.planners)
        curve = []
        for B in req.budgets:
            d2 = run_planners(records, lib, cfg, req.planners, budget=float(B), stop_on_confidence=False)
            for pid, g in d2.groupby("planner", sort=False):
                single = g[g["n_truth"] >= 1]
                m, lo, hi = bootstrap_ci(single["top1"].to_numpy(), 500, seed=2)
                curve.append({"planner": pid, "budget": B, "top1": m, "top1_lo": lo, "top1_hi": hi,
                              "f1": float(g["f1"].mean()), "mean_cost": float(g["cost"].mean())})
        result = {"simulated": True, "config": cfg, "n_scenarios": len(records), "summary": summarise(df, 500),
                  "accuracy_vs_budget": curve, "calibration": calibration(df),
                  "paired_tests": paired_tests(df) if "adaptive" in req.planners and len(req.planners) > 1 else [],
                  "agreement": agreement_matrix(df), "robustness": [],
                  "disclaimer": "Simulated fault injection on built-in datasets; likelihood tables estimated on separate seeds."}
        df.drop(columns=["marginals", "truth_vec"]).assign(path=df["path"].map(lambda p: ">".join(p))).to_csv(out / "runs.csv", index=False)
        result["plots"] = make_plots(result, out / "plots") if result["summary"] else []
        (out / "results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf8")
        _update(job_id, status="completed", progress=1.0, message="Completed", result_path=str(out / "results.json"),
                finished_at=datetime.now(timezone.utc))
    except Exception as exc:
        traceback.print_exc()
        _update(job_id, status="failed", error=f"{type(exc).__name__}: {str(exc)[:300]}", message="Failed",
                finished_at=datetime.now(timezone.utc))


def job_dir(job: BenchmarkJob) -> Path | None:
    return Path(job.result_path).parent if job.result_path else None

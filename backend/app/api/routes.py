"""Versioned REST API (/api/v1)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from experimentpilot.core import all_hypotheses
from experimentpilot.core.variants import VARIANTS
from experimentpilot.experiments import EXPERIMENTS
from experimentpilot.faults import DATASETS, FAULTS, SEVERITIES

from .. import __version__
from ..core.config import get_settings
from ..core.db import get_db
from ..models import BenchmarkJob, Dataset, DiagnosisSession, ExperimentRun, Project, ReportRecord
from ..reports import pdf as pdfmod
from ..schemas.api import (BenchmarkCreate, ColumnMappingIn, DatasetOut, ExternalOutcome, ProjectCreate, ProjectUpdate,
                           RunRequest, SessionCreate, SkipRequest, StopRequest)
from ..security.uploads import UploadError, read_limited, validate_and_parse
from ..services import benchmark as bench
from ..services import demo as demo_service
from ..services import projects as proj
from ..services import report as reports
from ..services import sessions as svc
from ..services.sessions import ServiceError, library

router = APIRouter(prefix="/api/v1")


def _page(items: list, total: int, limit: int, offset: int) -> dict:
    return {"items": items, "total": total, "limit": limit, "offset": offset}


# =============================================================== health & meta
@router.get("/health", tags=["health"], summary="Liveness and readiness check")
def health(db: Session = Depends(get_db)):
    db.execute(select(1))
    return {"status": "ok", "version": __version__, "database": "ok", "environment": get_settings().environment}


@router.get("/meta/hypotheses", tags=["meta"])
def meta_hypotheses():
    return [{"id": h.id, "code": h.code, "name": h.name, "description": h.description, "symptom": h.symptom,
             "default_prior": h.default_prior, "default_enabled": h.default_enabled, "core": h.core} for h in all_hypotheses()]


@router.get("/meta/experiments", tags=["meta"])
def meta_experiments():
    lib = library()
    return [{"id": e.id, "name": e.name, "description": e.description, "cost": e.cost, "risk": e.risk,
             "requires": list(e.requires), "outcomes": list(e.outcomes), "requires_approval": e.requires_approval,
             "limitations": list(e.limitations), "has_likelihood": e.id in lib,
             "likelihood_runs": lib.models[e.id].n_runs if e.id in lib else 0} for e in EXPERIMENTS.values()]


@router.get("/meta/planners", tags=["meta"])
def meta_planners():
    return [{"id": v.id, "label": v.label, "description": v.description, "enforces_risk_cap": v.enforce_risk} for v in VARIANTS.values()]


@router.get("/meta/likelihoods", tags=["meta"], summary="Likelihood tables and their provenance")
def meta_likelihoods():
    lib = library()
    return {"provenance": lib.provenance, "experiments": {k: m.to_dict() for k, m in lib.models.items()}}


@router.get("/meta/runtime", tags=["meta"])
def meta_runtime():
    s = get_settings()
    return {"version": __version__, "environment": s.environment, "max_upload_mb": s.max_upload_mb, "max_rows": s.max_rows,
            "max_columns": s.max_columns, "executor": s.executor, "experiment_timeout_s": s.experiment_timeout_s,
            "benchmark_workers": s.benchmark_workers, "cors_origins": s.cors_list, "database": s.database_url.split(":")[0],
            "outbound_network_calls": False, "authentication": "none (trusted local deployment)"}


@router.get("/meta/demo-options", tags=["meta"])
def demo_options():
    return {"datasets": [{"id": k, "description": v["description"], "synthetic": v["synthetic"]} for k, v in DATASETS.items()],
            "faults": list(FAULTS), "severities": list(SEVERITIES)}


# =============================================================== projects
@router.post("/projects", status_code=201, tags=["projects"])
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    p = Project(name=body.name, description=body.description, task_type=body.task_type,
                config={"pipeline_notes": body.pipeline_notes, "model_notes": body.model_notes, "mapping": {}})
    db.add(p); db.commit(); db.refresh(p)
    return proj.project_out(db, p)


@router.get("/projects", tags=["projects"])
def list_projects(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), q: str | None = Query(None, max_length=100),
                  db: Session = Depends(get_db)):
    stmt = select(Project)
    if q:
        stmt = stmt.where(Project.name.ilike(f"%{q}%"))
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(stmt.order_by(Project.updated_at.desc()).limit(limit).offset(offset)).all()
    return _page([proj.project_out(db, p).model_dump(mode="json") for p in rows], total, limit, offset)


@router.get("/projects/{pid}", tags=["projects"])
def get_project(pid: str, db: Session = Depends(get_db)):
    return proj.project_out(db, proj.get_project(db, pid))


@router.patch("/projects/{pid}", tags=["projects"])
def update_project(pid: str, body: ProjectUpdate, db: Session = Depends(get_db)):
    p = proj.get_project(db, pid)
    data = body.model_dump(exclude_unset=True)
    for k in ("name", "description"):
        if k in data and data[k] is not None:
            setattr(p, k, data[k])
    cfg = dict(p.config)
    for k in ("pipeline_notes", "model_notes", "default_budget", "default_risk_cap"):
        if k in data:
            cfg[k] = data[k]
    p.config = cfg
    db.commit()
    return proj.project_out(db, p)


@router.delete("/projects/{pid}", status_code=204, tags=["projects"], summary="Delete a project and all of its data files")
def delete_project(pid: str, db: Session = Depends(get_db)):
    proj.delete_project(db, proj.get_project(db, pid))
    return Response(status_code=204)


@router.post("/projects/{pid}/datasets", status_code=201, tags=["datasets"], summary="Upload a CSV dataset")
async def upload_dataset(pid: str, role: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    p = proj.get_project(db, pid)
    s = get_settings()
    try:
        data = await read_limited(file, s.max_upload_bytes)
        parsed = validate_and_parse(data, file.filename)
    except UploadError as e:
        raise ServiceError(e.status, "invalid_upload", e.message)
    ds = proj.save_dataset(db, p, role, parsed)
    db.refresh(p)
    return {"dataset": DatasetOut.model_validate(ds), "project": proj.project_out(db, p)}


@router.get("/projects/{pid}/datasets/{role}", tags=["datasets"], summary="Inspect a dataset's schema and preview rows")
def inspect_dataset(pid: str, role: str, rows: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)):
    from ..security.uploads import resolve_storage
    p = proj.get_project(db, pid)
    ds = next((d for d in p.datasets if d.role == role), None)
    if ds is None:
        raise ServiceError(404, "not_found", f"no {role} dataset")
    df = pd.read_csv(resolve_storage(ds.storage_path), nrows=rows)
    return {"dataset": DatasetOut.model_validate(ds), "preview": json.loads(df.to_json(orient="records")),
            "columns": list(df.columns)}


@router.delete("/projects/{pid}/datasets/{role}", status_code=204, tags=["datasets"])
def delete_dataset(pid: str, role: str, db: Session = Depends(get_db)):
    from ..security.uploads import remove_storage
    p = proj.get_project(db, pid)
    ds = next((d for d in p.datasets if d.role == role), None)
    if ds is None:
        raise ServiceError(404, "not_found", f"no {role} dataset")
    if proj.has_sessions(db, pid):
        raise ServiceError(409, "dataset_in_use", "delete the project's diagnosis sessions before removing data")
    remove_storage(ds.storage_path); db.delete(ds); db.commit()
    return Response(status_code=204)


@router.put("/projects/{pid}/mapping", tags=["datasets"], summary="Map target, prediction, timestamp, identifier and features")
def set_mapping(pid: str, body: ColumnMappingIn, db: Session = Depends(get_db)):
    p = proj.get_project(db, pid)
    mapping, warnings = proj.validate_mapping(db, p, body)
    cfg = dict(p.config); cfg["mapping"] = mapping; p.config = cfg
    db.commit()
    return {"mapping": mapping, "warnings": warnings, "project": proj.project_out(db, p)}


@router.post("/demo", status_code=201, tags=["projects"], summary="Create a demo project from a simulated fault scenario")
def create_demo(body: dict, db: Session = Depends(get_db)):
    faults = [(f["fault"], f.get("severity", "medium")) for f in body.get("faults", [])]
    p = demo_service.create_demo_project(db, body.get("dataset", "synthetic_credit"), faults, int(body.get("seed", 7)), body.get("name"))
    return proj.project_out(db, p)


# =============================================================== sessions
def _session_out(db: Session, s: DiagnosisSession, include_rec: bool = True) -> dict:
    rec = svc.recommend(db, s) if include_rec else None
    belief = svc.build_belief(s)
    runs = [reports.run_summary(r) for r in s.runs]
    from experimentpilot.core import get_hypothesis
    return {
        "id": s.id, "project_id": s.project_id, "name": s.name, "status": s.status, "planner": s.planner_variant,
        "config": {k: v for k, v in s.config.items() if k not in ("dataset_ids",)}, "hypotheses": s.hypotheses,
        "experiments": s.experiments, "symptoms": s.symptoms, "spent": s.spent, "risk_total": s.risk_total,
        "budget": s.config["budget"], "remaining": max(s.config["budget"] - s.spent, 0.0),
        "stop_code": s.stop_code, "stop_message": s.stop_message, "version": s.version,
        "created_at": s.created_at.isoformat(), "updated_at": s.updated_at.isoformat(),
        "finished_at": s.finished_at.isoformat() if s.finished_at else None,
        "marginals": [{"hypothesis": h, "code": get_hypothesis(h).code, "name": get_hypothesis(h).name,
                       "probability": p} for h, p in belief.ranked()],
        "entropy": belief.entropy(), "map_state": list(belief.map_state()[0]), "map_probability": belief.map_state()[1],
        "runs": runs, "recommendation": rec,
    }


@router.post("/sessions", status_code=201, tags=["sessions"], summary="Validate inputs and start a diagnosis session")
def create_session(body: SessionCreate, db: Session = Depends(get_db)):
    return _session_out(db, svc.create_session(db, body))


@router.get("/sessions", tags=["sessions"])
def list_sessions(project_id: str | None = None, status: str | None = None, limit: int = Query(50, ge=1, le=200),
                  offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    stmt = select(DiagnosisSession)
    if project_id:
        stmt = stmt.where(DiagnosisSession.project_id == project_id)
    if status:
        stmt = stmt.where(DiagnosisSession.status == status)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(stmt.order_by(DiagnosisSession.created_at.desc()).limit(limit).offset(offset)).all()
    items = []
    for s in rows:
        b = svc.build_belief(s)
        top = b.ranked()[0]
        items.append({"id": s.id, "project_id": s.project_id, "name": s.name, "status": s.status, "planner": s.planner_variant,
                      "spent": s.spent, "budget": s.config["budget"], "n_runs": sum(r.status in ("completed", "external") for r in s.runs),
                      "top_hypothesis": top[0], "top_probability": top[1], "map_probability": b.map_state()[1],
                      "created_at": s.created_at.isoformat(), "updated_at": s.updated_at.isoformat()})
    return _page(items, total, limit, offset)


@router.get("/sessions/{sid}", tags=["sessions"], summary="Session status, beliefs, runs and the next recommendation")
def get_session(sid: str, db: Session = Depends(get_db)):
    return _session_out(db, svc.get_session(db, sid))


@router.delete("/sessions/{sid}", status_code=204, tags=["sessions"])
def delete_session(sid: str, db: Session = Depends(get_db)):
    s = svc.get_session(db, sid)
    db.delete(s); db.commit()
    return Response(status_code=204)


@router.get("/sessions/{sid}/next", tags=["sessions"], summary="Next-experiment recommendation with scores and rationale")
def next_experiment(sid: str, db: Session = Depends(get_db)):
    return svc.recommend(db, svc.get_session(db, sid))


@router.post("/sessions/{sid}/run", tags=["sessions"], summary="Execute the recommended (or a chosen) experiment")
def run_experiment(sid: str, body: RunRequest | None = None, db: Session = Depends(get_db)):
    run = svc.run_experiment(db, sid, body or RunRequest())
    return {"run": reports.run_summary(run), "session": _session_out(db, svc.get_session(db, sid))}


@router.post("/sessions/{sid}/external", tags=["sessions"], summary="Submit an externally obtained experiment outcome")
def external_outcome(sid: str, body: ExternalOutcome, db: Session = Depends(get_db)):
    run = svc.submit_external(db, sid, body)
    return {"run": reports.run_summary(run), "session": _session_out(db, svc.get_session(db, sid))}


@router.post("/sessions/{sid}/skip", tags=["sessions"], summary="Skip an experiment")
def skip(sid: str, body: SkipRequest | None = None, db: Session = Depends(get_db)):
    body = body or SkipRequest()
    run = svc.skip_experiment(db, sid, body.experiment_id, body.reason)
    return {"run": reports.run_summary(run), "session": _session_out(db, svc.get_session(db, sid))}


@router.post("/sessions/{sid}/pause", tags=["sessions"])
def pause(sid: str, db: Session = Depends(get_db)):
    return _session_out(db, svc.set_status(db, sid, "paused"))


@router.post("/sessions/{sid}/resume", tags=["sessions"])
def resume(sid: str, db: Session = Depends(get_db)):
    return _session_out(db, svc.set_status(db, sid, "active"))


@router.post("/sessions/{sid}/stop", tags=["sessions"])
def stop(sid: str, body: StopRequest | None = None, db: Session = Depends(get_db)):
    return _session_out(db, svc.set_status(db, sid, "stopped", (body or StopRequest()).reason))


@router.get("/sessions/{sid}/evidence", tags=["evidence"], summary="All experiment evidence for a session")
def evidence(sid: str, db: Session = Depends(get_db)):
    s = svc.get_session(db, sid)
    ev = reports.evidence_direction(s)
    return {"runs": [reports.run_summary(r) | {"tables": r.tables} for r in s.runs],
            "direction": ev, "symptoms": s.symptoms}


@router.get("/sessions/{sid}/evidence/{run_id}", tags=["evidence"])
def evidence_run(sid: str, run_id: str, db: Session = Depends(get_db)):
    s = svc.get_session(db, sid)
    r = next((x for x in s.runs if x.id == run_id), None)
    if r is None:
        raise ServiceError(404, "not_found", "run not found")
    return reports.run_summary(r) | {"tables": r.tables}


@router.get("/sessions/{sid}/posterior-history", tags=["evidence"])
def posterior_history(sid: str, db: Session = Depends(get_db)):
    s = svc.get_session(db, sid)
    return {"hypotheses": list(svc.hypothesis_ids(s)),
            "history": [{"step": x.step, "label": x.label, "marginals": x.marginals, "entropy": x.entropy,
                         "map_probability": x.map_probability, "spent": x.spent, "created_at": x.created_at.isoformat()}
                        for x in s.snapshots]}


@router.get("/sessions/{sid}/tree", tags=["evidence"], summary="Decision tree: path taken and hypothetical branches")
def tree(sid: str, db: Session = Depends(get_db)):
    return reports.build_tree(db, svc.get_session(db, sid))


@router.get("/sessions/{sid}/report", tags=["reports"], summary="Final diagnosis report (JSON)")
def report(sid: str, db: Session = Depends(get_db)):
    return reports.build_report(db, svc.get_session(db, sid))


def _record_report(db: Session, sid: str, fmt: str, rep: dict) -> None:
    db.add(ReportRecord(session_id=sid, fmt=fmt, summary={"verdict": rep["verdict"], "diagnosed": rep["diagnosed"]}))
    db.commit()


@router.get("/sessions/{sid}/export/report.json", tags=["reports"])
def export_json(sid: str, db: Session = Depends(get_db)):
    rep = reports.build_report(db, svc.get_session(db, sid)); _record_report(db, sid, "json", rep)
    return Response(json.dumps(rep, indent=2, default=str), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="diagnosis-{sid[:8]}.json"'})


@router.get("/sessions/{sid}/export/report.pdf", tags=["reports"])
def export_pdf(sid: str, db: Session = Depends(get_db)):
    rep = reports.build_report(db, svc.get_session(db, sid)); _record_report(db, sid, "pdf", rep)
    return Response(pdfmod.render_pdf(rep), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="diagnosis-{sid[:8]}.pdf"'})


@router.get("/sessions/{sid}/export/{kind}.csv", tags=["reports"], summary="CSV export: hypotheses | experiments | history")
def export_csv(sid: str, kind: str, db: Session = Depends(get_db)):
    fns = {"hypotheses": pdfmod.hypotheses_csv, "experiments": pdfmod.experiments_csv, "history": pdfmod.history_csv}
    if kind not in fns:
        raise ServiceError(404, "not_found", "unknown export; use hypotheses, experiments or history")
    rep = reports.build_report(db, svc.get_session(db, sid)); _record_report(db, sid, "csv", rep)
    return Response(fns[kind](rep), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{kind}-{sid[:8]}.csv"'})


# =============================================================== benchmark lab
def _job_out(j: BenchmarkJob) -> dict:
    return {"id": j.id, "name": j.name, "status": j.status, "progress": j.progress, "message": j.message, "error": j.error,
            "config": j.config, "created_at": j.created_at.isoformat(),
            "finished_at": j.finished_at.isoformat() if j.finished_at else None}


@router.get("/benchmarks/options", tags=["benchmark"])
def benchmark_options():
    prov = library().provenance
    return {"datasets": list(DATASETS), "faults": list(FAULTS), "severities": list(SEVERITIES),
            "planners": [{"id": v.id, "label": v.label, "description": v.description} for v in VARIANTS.values()],
            "reserved_estimation_seeds": prov.get("seeds", []), "likelihood_provenance": prov}


@router.get("/benchmarks/published", tags=["benchmark"], summary="Published benchmark results shipped with the repository")
def published(name: str = "default-v1"):
    p = bench.published_path(name)
    if not p.exists():
        raise ServiceError(404, "not_found", "no published benchmark results; run research/benchmark/run_benchmark.py")
    return json.loads(p.read_text(encoding="utf8"))


@router.get("/benchmarks/published/runs", tags=["benchmark"], summary="Per-scenario planner traces of the published run")
def published_runs(name: str = "default-v1", planner: str | None = None, scenario: str | None = Query(None, max_length=200),
                   limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    p = bench.PUBLISHED_DIR / f"runs_{name}.csv"
    if not p.exists():
        raise ServiceError(404, "not_found", "no published run traces")
    df = pd.read_csv(p)
    if planner:
        df = df[df["planner"] == planner]
    if scenario:
        df = df[df["scenario"].str.contains(scenario, regex=False)]
    cols = ["scenario", "dataset", "planner", "truth", "top1", "f1", "cost", "n_experiments", "stop", "confidence", "path"]
    page = df.iloc[offset:offset + limit][cols]
    return _page(json.loads(page.to_json(orient="records")), int(len(df)), limit, offset)


@router.get("/benchmarks/published/plots/{plot}", tags=["benchmark"])
def published_plot(plot: str):
    if "/" in plot or "\\" in plot or not plot.endswith(".png"):
        raise ServiceError(400, "invalid_path", "invalid plot name")
    p = bench.PUBLISHED_DIR / "plots" / plot
    if not p.exists():
        raise ServiceError(404, "not_found", "plot not found")
    return FileResponse(p, media_type="image/png")


@router.post("/benchmarks", status_code=202, tags=["benchmark"], summary="Start a benchmark run")
def create_benchmark(body: BenchmarkCreate, db: Session = Depends(get_db)):
    return _job_out(bench.create_job(db, body))


@router.get("/benchmarks", tags=["benchmark"])
def list_benchmarks(limit: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    total = db.scalar(select(func.count()).select_from(BenchmarkJob)) or 0
    rows = db.scalars(select(BenchmarkJob).order_by(BenchmarkJob.created_at.desc()).limit(limit).offset(offset)).all()
    return _page([_job_out(j) for j in rows], total, limit, offset)


def _job(db: Session, jid: str) -> BenchmarkJob:
    j = db.get(BenchmarkJob, jid)
    if j is None:
        raise ServiceError(404, "not_found", "benchmark job not found")
    return j


@router.get("/benchmarks/{jid}", tags=["benchmark"], summary="Benchmark status and progress")
def get_benchmark(jid: str, db: Session = Depends(get_db)):
    return _job_out(_job(db, jid))


@router.get("/benchmarks/{jid}/results", tags=["benchmark"])
def benchmark_results(jid: str, db: Session = Depends(get_db)):
    j = _job(db, jid)
    if j.status != "completed" or not j.result_path:
        raise ServiceError(409, "not_ready", f"benchmark is {j.status}")
    return json.loads(Path(j.result_path).read_text(encoding="utf8"))


@router.get("/benchmarks/{jid}/traces", tags=["benchmark"], summary="Per-scenario experiment traces")
def benchmark_traces(jid: str, planner: str | None = None, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                     db: Session = Depends(get_db)):
    j = _job(db, jid)
    d = bench.job_dir(j)
    if j.status != "completed" or d is None:
        raise ServiceError(409, "not_ready", f"benchmark is {j.status}")
    df = pd.read_csv(d / "runs.csv")
    if planner:
        df = df[df["planner"] == planner]
    cols = ["scenario", "dataset", "planner", "truth", "top1", "f1", "cost", "n_experiments", "stop", "confidence", "path"]
    return _page(json.loads(df.iloc[offset:offset + limit][cols].to_json(orient="records")), int(len(df)), limit, offset)


@router.get("/benchmarks/{jid}/export.{fmt}", tags=["benchmark"], summary="Export results as csv (runs) or json")
def benchmark_export(jid: str, fmt: str, db: Session = Depends(get_db)):
    j = _job(db, jid)
    d = bench.job_dir(j)
    if j.status != "completed" or d is None:
        raise ServiceError(409, "not_ready", f"benchmark is {j.status}")
    if fmt == "json":
        return Response((d / "results.json").read_text(encoding="utf8"), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="benchmark-{jid[:8]}.json"'})
    if fmt == "csv":
        return Response((d / "runs.csv").read_text(encoding="utf8"), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="benchmark-{jid[:8]}.csv"'})
    raise ServiceError(404, "not_found", "format must be csv or json")


@router.get("/benchmarks/{jid}/plots/{plot}", tags=["benchmark"])
def benchmark_plot(jid: str, plot: str, db: Session = Depends(get_db)):
    j = _job(db, jid)
    d = bench.job_dir(j)
    if d is None or "/" in plot or "\\" in plot or not plot.endswith(".png"):
        raise ServiceError(404, "not_found", "plot not found")
    p = d / "plots" / plot
    if not p.exists():
        raise ServiceError(404, "not_found", "plot not found")
    return FileResponse(p, media_type="image/png")

"""Diagnosis-session orchestration: explicit state machine, planner integration, safe execution."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from experimentpilot.core import (Belief, LikelihoodBank, PlannerConfig, PlanState, evaluate_stop, get_variant,
                                  get_hypothesis, all_hypotheses)
from experimentpilot.core.belief import enumerate_states
from experimentpilot.core.planner import ExperimentSpec, score_experiments
from experimentpilot.experiments import EXPERIMENTS, ColumnMapping, ExperimentContext
from experimentpilot.experiments.common import accuracy
from experimentpilot.likelihood import load_default_library

from ..core.config import get_settings
from ..models import BeliefSnapshot, Dataset, DiagnosisSession, ExperimentRun, Project
from ..schemas.api import ExternalOutcome, RunRequest, SessionCreate
from ..security.uploads import resolve_storage
from .executor import ExperimentTimeout, executor

APPROVAL_RISK_LEVEL = 0.15
TRANSITIONS = {
    "active": {"paused", "completed", "stopped"},
    "paused": {"active", "stopped"},
    "completed": set(),
    "stopped": set(),
}


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str, extra: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra or {}


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def session_lock(sid: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(sid, threading.Lock())


@lru_cache
def library():
    return load_default_library()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ state machine
def transition(sess: DiagnosisSession, new: str) -> None:
    if new not in TRANSITIONS.get(sess.status, set()):
        raise ServiceError(409, "invalid_transition", f"cannot move a session from '{sess.status}' to '{new}'")
    sess.status = new
    if new in ("completed", "stopped"):
        sess.finished_at = utcnow()


# ------------------------------------------------------------------ engine objects
def hypothesis_ids(sess: DiagnosisSession) -> tuple[str, ...]:
    return tuple(h["id"] for h in sess.hypotheses)


def build_belief(sess: DiagnosisSession) -> Belief:
    hyps = hypothesis_ids(sess)
    states = enumerate_states(len(hyps), sess.config["max_active"])
    return Belief(hyps, states, np.array(sess.belief, dtype=float))


def prior_belief(sess: DiagnosisSession) -> Belief:
    return Belief.from_priors(hypothesis_ids(sess), [h["prior"] for h in sess.hypotheses], sess.config["max_active"])


def planner_config(sess: DiagnosisSession) -> PlannerConfig:
    c = sess.config
    return PlannerConfig(budget=c["budget"], risk_cap=c["risk_cap"], risk_weight=c["risk_weight"],
                         confidence_threshold=c["confidence_threshold"], min_score=c["min_score"],
                         separation_weight=c["separation_weight"], confidence_mode=c["confidence_mode"]).validate()


def enabled_specs(sess: DiagnosisSession) -> list[ExperimentSpec]:
    lib = library()
    out = []
    for eid in sess.experiments:
        if eid in EXPERIMENTS and eid in lib:
            sp = EXPERIMENTS[eid].spec()
            if sess.config.get("human_in_the_loop"):   # prerequisites are satisfied by the human running it
                sp = ExperimentSpec(sp.id, sp.name, sp.cost, sp.risk, (), sp.outcomes, sp.description, sp.repeatable,
                                    sp.requires_approval)
            out.append(sp)
    return out


def plan_state(sess: DiagnosisSession, include_skips: bool = True) -> PlanState:
    done: dict[str, int] = {}
    skipped: set[str] = set()
    n = 0
    for r in sess.runs:
        if r.status in ("completed", "external"):
            done[r.experiment_id] = done.get(r.experiment_id, 0) + 1
            n += 1
        elif r.status == "skipped" and include_skips:
            skipped.add(r.experiment_id)
    return PlanState(spent=sess.spent, done=done, skipped=skipped,
                     available_inputs=set(sess.config.get("available_inputs", [])), n_run=n)


def _dataset(db: Session, project_id: str, role: str) -> Dataset | None:
    return db.scalar(select(Dataset).where(Dataset.project_id == project_id, Dataset.role == role))


def _mapping_obj(m: dict) -> ColumnMapping:
    return ColumnMapping(**{k: m.get(k) for k in ("target", "prediction", "timestamp", "identifier", "features")})


# ------------------------------------------------------------------ creation
def create_session(db: Session, req: SessionCreate) -> DiagnosisSession:
    project = db.get(Project, req.project_id)
    if project is None:
        raise ServiceError(404, "not_found", "project not found")
    ref, rec = _dataset(db, project.id, "reference"), _dataset(db, project.id, "recent")
    if ref is None or rec is None:
        raise ServiceError(422, "missing_data", "upload both a reference and a recent dataset before starting a diagnosis")
    mapping = dict(project.config.get("mapping") or {})
    if not mapping.get("target") and not mapping.get("prediction"):
        raise ServiceError(422, "missing_mapping", "map at least a target (label) or prediction column first")
    try:
        get_variant(req.planner_variant)
    except KeyError:
        raise ServiceError(422, "invalid_planner", f"unknown planner variant '{req.planner_variant}'")

    # hypotheses
    settings = {h.id: h for h in (req.hypotheses or [])}
    for hid in settings:
        try:
            get_hypothesis(hid)
        except KeyError:
            raise ServiceError(422, "invalid_hypothesis", f"unknown hypothesis '{hid}'")
    hyps = []
    for h in all_hypotheses():
        s = settings.get(h.id)
        enabled = s.enabled if s else (h.default_enabled if req.hypotheses is None else False)
        if enabled:
            hyps.append({"id": h.id, "prior": float(s.prior) if s and s.prior else h.default_prior})
    if len(hyps) < 2:
        raise ServiceError(422, "too_few_hypotheses", "enable at least two candidate root causes")
    if len(hyps) > 12:
        raise ServiceError(422, "too_many_hypotheses", "at most 12 hypotheses are supported")
    exp_ids = req.experiments if req.experiments is not None else list(EXPERIMENTS)
    bad = [e for e in exp_ids if e not in EXPERIMENTS]
    if bad:
        raise ServiceError(422, "invalid_experiment", f"unknown experiments: {', '.join(bad)}")
    if not exp_ids:
        raise ServiceError(422, "no_experiments", "enable at least one experiment")

    # data checks and symptoms
    ref_df = pd.read_csv(resolve_storage(ref.storage_path))
    rec_df = pd.read_csv(resolve_storage(rec.storage_path))
    cm = _mapping_obj(mapping)
    for col in (cm.target, cm.prediction):
        if col and (col not in ref_df.columns or col not in rec_df.columns):
            raise ServiceError(422, "invalid_mapping", f"column '{col}' must exist in both datasets")
    replay = _dataset(db, project.id, "replay")
    replay_df = pd.read_csv(resolve_storage(replay.storage_path)) if replay else None
    metrics = req.reported_metrics.model_dump() if req.reported_metrics else project.config.get("reported_metrics")
    ctx = ExperimentContext(ref_df, rec_df, cm, replay_df, metrics, {}, req.seed)
    avail = sorted(ctx.available_inputs())
    if not ctx.feature_names():
        raise ServiceError(422, "no_features", "no feature columns remain after mapping")
    symptoms: dict[str, Any] = {"n_reference": len(ref_df), "n_recent": len(rec_df), "n_features": len(ctx.feature_names())}
    if "predictions_reference" in avail and "labels_reference" in avail:
        symptoms["accuracy_reference"] = accuracy(ref_df[cm.target], ref_df[cm.prediction])
    if "predictions_recent" in avail and "labels_recent" in avail:
        symptoms["accuracy_recent"] = accuracy(rec_df[cm.target], rec_df[cm.prediction])
    if "accuracy_reference" in symptoms and "accuracy_recent" in symptoms:
        symptoms["accuracy_drop"] = symptoms["accuracy_reference"] - symptoms["accuracy_recent"]
    if metrics:
        symptoms["reported_reference"], symptoms["reported_recent"] = metrics["reference"], metrics["recent"]
    symptoms["missing_inputs"] = sorted({"labels_reference", "labels_recent", "predictions_reference", "predictions_recent",
                                         "replay", "reported_metrics"} - set(avail))

    k = len(hyps)
    max_active = min(req.max_active or 3, k)
    sess = DiagnosisSession(
        project_id=project.id, name=req.name.strip() or "Diagnosis session", status="active",
        planner_variant=req.planner_variant,
        config={"budget": req.budget, "risk_cap": req.risk_cap, "risk_weight": req.risk_weight,
                "confidence_threshold": req.confidence_threshold, "min_score": req.min_score,
                "separation_weight": req.separation_weight, "confidence_mode": req.confidence_mode,
                "max_active": max_active, "available_inputs": avail, "human_in_the_loop": req.human_in_the_loop,
                "seed": req.seed, "learner": req.learner, "dataset_ids": {"reference": ref.id, "recent": rec.id,
                                                                          "replay": replay.id if replay else None}},
        hypotheses=hyps, experiments=exp_ids, mapping=mapping, reported_metrics=metrics, symptoms=symptoms)
    belief = Belief.from_priors([h["id"] for h in hyps], [h["prior"] for h in hyps], max_active)
    sess.belief = belief.probs.tolist()
    db.add(sess)
    db.flush()
    _snapshot(db, sess, belief, 0, "Prior")
    rec_ = recommend(db, sess)
    if not any(s["feasible"] for s in rec_["experiments"]):
        db.rollback()
        reasons = sorted({r for s in rec_["experiments"] for r in s["reasons"]})[:6]
        raise ServiceError(422, "no_feasible_experiments",
                           "no experiment can run with the supplied data and limits: " + "; ".join(reasons))
    db.commit()
    return sess


def _snapshot(db: Session, sess: DiagnosisSession, belief: Belief, step: int, label: str) -> None:
    ms, mp = belief.map_state()
    db.add(BeliefSnapshot(session_id=sess.id, step=step, label=label, marginals=belief.marginal_dict(),
                          probs=belief.probs.tolist(), entropy=belief.entropy(), map_state=list(ms),
                          map_probability=mp, spent=sess.spent))


# ------------------------------------------------------------------ recommendation
def recommend(db: Session, sess: DiagnosisSession) -> dict:
    belief = build_belief(sess)
    cfg = planner_config(sess)
    state = plan_state(sess)
    specs = enabled_specs(sess)
    bank = LikelihoodBank(library(), belief.hypotheses, belief.states)
    variant = get_variant(sess.planner_variant)
    rng = np.random.default_rng([sess.config.get("seed", 0), len(sess.runs)])
    choice = variant.choose(belief, specs, bank, cfg, state, rng)
    stop = evaluate_stop(belief, choice.scored, cfg, state, variant.check_low_value)
    meta = {s.id: s for s in specs}
    items = []
    for sc in sorted(choice.scored, key=lambda s: (not s.feasible, -s.adjusted_score)):
        d = sc.to_dict()
        sp = meta.get(sc.id)
        d["description"] = EXPERIMENTS[sc.id].description
        d["outcomes"] = list(sp.outcomes) if sp else []
        d["requires"] = list(EXPERIMENTS[sc.id].requires)
        d["state"] = ("done" if "done" in sc.reason_codes else "skipped" if "skipped" in sc.reason_codes else
                      "feasible" if sc.feasible else "blocked")
        items.append(d)
    ms, mp = belief.map_state()
    return {
        "session_id": sess.id, "status": sess.status, "planner": variant.id, "planner_label": variant.label,
        "selected": choice.selected.id if choice.selected else None, "rationale": choice.rationale,
        "stop": {"stop": stop.stop, "code": stop.code, "message": stop.message, "confidence": stop.confidence},
        "experiments": items, "budget": cfg.budget, "spent": sess.spent, "remaining": max(cfg.budget - sess.spent, 0.0),
        "diagnosis_confidence": float(stop.confidence), "map_state": list(ms), "map_probability": mp,
        "entropy": belief.entropy(), "needs_approval": bool(choice.selected and (choice.selected.requires_approval
                                                             or choice.selected.risk > APPROVAL_RISK_LEVEL)),
    }


def _finish_if_stopped(db: Session, sess: DiagnosisSession) -> None:
    if sess.status != "active":
        return
    rec = recommend(db, sess)
    if rec["stop"]["stop"]:
        transition(sess, "completed")
        sess.stop_code, sess.stop_message = rec["stop"]["code"], rec["stop"]["message"]


def _bump(sess: DiagnosisSession) -> None:
    sess.version += 1
    sess.updated_at = utcnow()


def _get_session(db: Session, sid: str) -> DiagnosisSession:
    s = db.get(DiagnosisSession, sid)
    if s is None:
        raise ServiceError(404, "not_found", "session not found")
    return s


# ------------------------------------------------------------------ actions
def _require_active(sess: DiagnosisSession) -> None:
    if sess.status != "active":
        raise ServiceError(409, "session_not_active", f"the session is {sess.status}; only active sessions accept experiments")


def _approval_check(spec: ExperimentSpec, approve: bool) -> None:
    if (spec.requires_approval or spec.risk > APPROVAL_RISK_LEVEL) and not approve:
        raise ServiceError(409, "approval_required",
                           f"'{spec.name}' is {'compute-heavy/sensitive' if spec.requires_approval else 'above the warning risk level'}; "
                           "explicit approval is required", {"experiment_id": spec.id})


def _choose(db: Session, sess: DiagnosisSession, experiment_id: str | None) -> tuple[ExperimentSpec, dict, dict]:
    rec = recommend(db, sess)
    if experiment_id is None:
        if not rec["selected"]:
            raise ServiceError(409, "nothing_to_run", rec["stop"]["message"] or "no feasible experiment remains")
        experiment_id = rec["selected"]
    item = next((e for e in rec["experiments"] if e["id"] == experiment_id), None)
    if item is None:
        raise ServiceError(404, "unknown_experiment", f"'{experiment_id}' is not enabled in this session")
    reasons = [r for r, c in zip(item["reasons"], item["reason_codes"]) if c != "skipped"]
    blocking = [c for c in item["reason_codes"] if c != "skipped"]
    if blocking:
        raise ServiceError(409, "experiment_not_feasible", f"'{item['name']}' cannot run: {'; '.join(reasons)}",
                           {"reason_codes": blocking})
    spec = next(s for s in enabled_specs(sess) if s.id == experiment_id)
    rec_info = {"eig": item["eig"], "score": item["score"], "separation": item["separation"], "rank": item["rank"],
                "denominator": item["denominator"], "recommended_id": rec["selected"], "rationale": rec["rationale"],
                "chosen_by": "planner" if rec["selected"] == experiment_id else "user", "planner": rec["planner"]}
    return spec, item, rec_info


def _apply_outcome(db: Session, sess: DiagnosisSession, spec: ExperimentSpec, outcome_index: int, label: str,
                   step: int) -> None:
    belief = build_belief(sess)
    bank = LikelihoodBank(library(), belief.hypotheses, belief.states)
    belief.update(bank.matrix(spec.id)[:, outcome_index], label)
    sess.belief = belief.probs.tolist()
    sess.spent = float(sess.spent + spec.cost)
    sess.risk_total = float(sess.risk_total + spec.risk)
    _snapshot(db, sess, belief, step, label)


def run_experiment(db: Session, sid: str, req: RunRequest) -> ExperimentRun:
    lock = session_lock(sid)
    with lock:
        sess = _get_session(db, sid)
        _require_active(sess)
        if any(r.status == "running" for r in sess.runs):
            raise ServiceError(409, "run_in_progress", "another experiment is already running in this session")
        spec, item, rec_info = _choose(db, sess, req.experiment_id)
        _approval_check(spec, req.approve)
        exp = EXPERIMENTS[spec.id]
        missing = exp.missing_inputs(_ctx_stub(sess))
        if missing:
            raise ServiceError(409, "missing_inputs", f"'{spec.name}' needs {', '.join(missing)}; enter its result manually instead")
        run = ExperimentRun(session_id=sid, experiment_id=spec.id, status="running", source="backend", cost=spec.cost,
                            risk=spec.risk, recommendation=rec_info)
        db.add(run)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise ServiceError(409, "duplicate_run", f"'{spec.name}' is already running or completed in this session")
        payload = _payload(db, sess, spec.id)
    t0 = time.perf_counter()
    try:
        result = executor.run(payload)
        error = None
    except ExperimentTimeout as exc:
        result, error = None, str(exc)
    except Exception as exc:  # experiment failure never corrupts session state
        result, error = None, str(exc)[:500]
    with lock:
        db.expire_all()
        sess = _get_session(db, sid)
        run = db.get(ExperimentRun, run.id)
        run.finished_at = utcnow()
        run.runtime_s = time.perf_counter() - t0
        if error or result is None:
            run.status, run.error = "failed", error
            _bump(sess)
            db.commit()
            raise ServiceError(422, "experiment_failed", error or "experiment failed", {"run_id": run.id})
        run.status = "completed"
        run.outcome, run.outcome_index = result["outcome"], result["outcome_index"]
        run.score, run.stats, run.tables = result["score"], result["stats"], result["tables"]
        run.findings, run.interpretation, run.limitations = result["findings"], result["interpretation"], result["limitations"]
        run.step = sum(1 for r in sess.runs if r.status in ("completed", "external") and r.id != run.id) + 1
        if sess.status in ("stopped", "completed"):   # stopped while running: keep the evidence, do not charge
            run.note = f"session was {sess.status} while this experiment ran; result recorded but not applied"
            db.commit()
            return run
        _apply_outcome(db, sess, spec, run.outcome_index, f"{spec.name}: {run.outcome}", run.step)
        _bump(sess)
        db.flush()
        _finish_if_stopped(db, sess)
        db.commit()
        return run


def _ctx_stub(sess: DiagnosisSession) -> ExperimentContext:
    """A context carrying only the available-input set (used for prerequisite checks)."""
    class _Ctx(ExperimentContext):
        def available_inputs(self_inner) -> set[str]:  # noqa: N805
            return set(sess.config.get("available_inputs", []))
    empty = pd.DataFrame()
    return _Ctx(empty, empty, ColumnMapping())


def _payload(db: Session, sess: DiagnosisSession, exp_id: str) -> dict:
    ids = sess.config["dataset_ids"]
    ref, rec = db.get(Dataset, ids["reference"]), db.get(Dataset, ids["recent"])
    rep = db.get(Dataset, ids["replay"]) if ids.get("replay") else None
    if ref is None or rec is None:
        raise ServiceError(409, "data_removed", "the datasets used by this session no longer exist")
    return {"experiment_id": exp_id, "reference_path": str(resolve_storage(ref.storage_path)),
            "recent_path": str(resolve_storage(rec.storage_path)),
            "replay_path": str(resolve_storage(rep.storage_path)) if rep else None, "mapping": sess.mapping,
            "reported_metrics": sess.reported_metrics, "config": {"learner": sess.config.get("learner", "logreg")},
            "seed": sess.config.get("seed", 0)}


def submit_external(db: Session, sid: str, req: ExternalOutcome) -> ExperimentRun:
    with session_lock(sid):
        sess = _get_session(db, sid)
        _require_active(sess)
        if req.experiment_id not in sess.experiments or req.experiment_id not in EXPERIMENTS:
            raise ServiceError(404, "unknown_experiment", f"'{req.experiment_id}' is not enabled in this session")
        if req.experiment_id not in library():
            raise ServiceError(422, "no_likelihood", "this experiment has no likelihood model")
        exp = EXPERIMENTS[req.experiment_id]
        spec = exp.spec()
        if any(r.status in ("completed", "external", "running") and r.experiment_id == spec.id for r in sess.runs):
            raise ServiceError(409, "duplicate_run", f"'{spec.name}' was already executed in this session")
        if spec.cost > planner_config(sess).budget - sess.spent + 1e-9:
            raise ServiceError(409, "budget_exceeded", "the remaining budget cannot cover this experiment")
        if spec.risk > sess.config["risk_cap"] + 1e-12:
            raise ServiceError(409, "risk_cap_exceeded", "this experiment exceeds the session risk cap")
        o = req.outcome
        if isinstance(o, int) or (isinstance(o, str) and o.isdigit()):
            idx = int(o)
        else:
            idx = list(spec.outcomes).index(o) if o in spec.outcomes else -1
        if not 0 <= idx < len(spec.outcomes):
            raise ServiceError(422, "invalid_outcome", f"outcome must be one of: {', '.join(spec.outcomes)}")
        run = ExperimentRun(session_id=sid, experiment_id=spec.id, status="external", source="external", cost=spec.cost,
                            risk=spec.risk, outcome=spec.outcomes[idx], outcome_index=idx, finished_at=utcnow(),
                            step=sum(1 for r in sess.runs if r.status in ("completed", "external")) + 1, note=req.note or None,
                            interpretation="Outcome entered manually by the user; no data was analysed by the platform.",
                            limitations=["Externally supplied outcomes are taken as given and treated as noisy evidence."],
                            recommendation={"chosen_by": "user"})
        sess.runs.append(run)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            raise ServiceError(409, "duplicate_run", "this experiment was already executed")
        _apply_outcome(db, sess, spec, idx, f"{spec.name}: {spec.outcomes[idx]} (external)", run.step)
        _bump(sess)
        db.flush()
        _finish_if_stopped(db, sess)
        db.commit()
        return run


def skip_experiment(db: Session, sid: str, experiment_id: str | None, reason: str) -> ExperimentRun:
    with session_lock(sid):
        sess = _get_session(db, sid)
        _require_active(sess)
        rec = recommend(db, sess)
        eid = experiment_id or rec["selected"]
        if not eid or eid not in sess.experiments:
            raise ServiceError(404, "unknown_experiment", "no experiment to skip")
        run = ExperimentRun(session_id=sid, experiment_id=eid, status="skipped", source="skip", finished_at=utcnow(),
                            note=reason or "skipped by the user")
        sess.runs.append(run)
        _bump(sess)
        db.flush()
        _finish_if_stopped(db, sess)
        db.commit()
        return run


def set_status(db: Session, sid: str, new: str, reason: str = "") -> DiagnosisSession:
    with session_lock(sid):
        sess = _get_session(db, sid)
        transition(sess, new)
        if new == "stopped":
            sess.stop_code, sess.stop_message = "user_stopped", reason or "Stopped by the user."
        _bump(sess)
        db.commit()
        return sess


def get_session(db: Session, sid: str) -> DiagnosisSession:
    return _get_session(db, sid)

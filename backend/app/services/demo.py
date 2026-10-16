"""One-click demo projects built with the fault-injection engine (simulated data)."""
from __future__ import annotations

from sqlalchemy.orm import Session

from experimentpilot.faults import DATASETS, FAULTS, SEVERITIES, ScenarioSpec, build_scenario
from experimentpilot.faults.injectors import FAULT_TO_HYPOTHESIS

from ..models import Dataset, Project
from ..security.uploads import describe_schema, store_dataframe
from .sessions import ServiceError

import hashlib


def create_demo_project(db: Session, dataset: str, faults: list[tuple[str, str]], seed: int, name: str | None) -> Project:
    if dataset not in DATASETS:
        raise ServiceError(422, "invalid_dataset", f"dataset must be one of {', '.join(DATASETS)}")
    for f, s in faults:
        if f not in FAULTS or s not in SEVERITIES:
            raise ServiceError(422, "invalid_fault", f"invalid fault '{f}:{s}'")
    if len(faults) > 3:
        raise ServiceError(422, "too_many_faults", "inject at most three simultaneous faults")
    sc = build_scenario(ScenarioSpec(dataset, tuple(faults), seed))
    label = " + ".join(f"{f.replace('_', ' ')} ({s})" for f, s in faults) or "no fault (control)"
    p = Project(name=(name or f"Demo - {dataset.replace('_', ' ')}: {label}")[:120],
                description=f"Simulated scenario ({DATASETS[dataset]['description']}) with injected: {label}. "
                            "Data are synthetic/built-in; the ground truth is stored with the project for evaluation.",
                config={"mapping": {"target": "target", "prediction": "prediction", "timestamp": "event_time",
                                    "identifier": "record_id", "features": None},
                        "reported_metrics": sc.reported_metrics, "demo": True,
                        "demo_truth": [FAULT_TO_HYPOTHESIS[f] for f, _ in faults],
                        "demo_scenario": {"dataset": dataset, "faults": faults, "seed": seed},
                        "pipeline_notes": "Replay dataset = recent records processed by the previous pipeline version.",
                        "model_notes": "Logistic regression / gradient boosting trained on the reference training split (simulated)."})
    db.add(p)
    db.flush()
    for role, df in (("reference", sc.reference), ("recent", sc.recent), ("replay", sc.replay)):
        rel, path = store_dataframe(df, p.id, role)
        db.add(Dataset(project_id=p.id, role=role, filename=f"demo_{role}.csv", storage_path=rel, n_rows=len(df),
                       n_cols=df.shape[1], size_bytes=path.stat().st_size,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest(), schema=describe_schema(df)))
    db.commit()
    db.refresh(p)
    return p

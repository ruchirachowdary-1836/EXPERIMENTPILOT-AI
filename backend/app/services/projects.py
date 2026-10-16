from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import Dataset, DiagnosisSession, Project
from ..schemas.api import ColumnMappingIn, DatasetOut, ProjectOut
from ..security.uploads import ParsedUpload, describe_schema, remove_storage, store_dataframe
from .sessions import ServiceError

ROLES = ("reference", "recent", "replay")


def readiness(db: Session, p: Project) -> list[str]:
    """Reasons the project cannot start a diagnosis yet (empty list = ready)."""
    roles = {d.role for d in p.datasets}
    out = []
    for r in ("reference", "recent"):
        if r not in roles:
            out.append(f"upload the {r} dataset")
    m = p.config.get("mapping") or {}
    if not m.get("target") and not m.get("prediction"):
        out.append("map the target and/or prediction column")
    return out


def project_out(db: Session, p: Project) -> ProjectOut:
    n = db.scalar(select(func.count()).select_from(DiagnosisSession).where(DiagnosisSession.project_id == p.id)) or 0
    why = readiness(db, p)
    return ProjectOut(id=p.id, name=p.name, description=p.description, task_type=p.task_type, config=p.config,
                      created_at=p.created_at, updated_at=p.updated_at,
                      datasets=[DatasetOut.model_validate(d) for d in sorted(p.datasets, key=lambda d: d.role)],
                      n_sessions=n, ready=not why, readiness=why)


def get_project(db: Session, pid: str) -> Project:
    p = db.get(Project, pid)
    if p is None:
        raise ServiceError(404, "not_found", "project not found")
    return p


def has_sessions(db: Session, pid: str) -> bool:
    return bool(db.scalar(select(func.count()).select_from(DiagnosisSession).where(DiagnosisSession.project_id == pid)))


def save_dataset(db: Session, p: Project, role: str, parsed: ParsedUpload) -> Dataset:
    if role not in ROLES:
        raise ServiceError(422, "invalid_role", f"role must be one of {', '.join(ROLES)}")
    if has_sessions(db, p.id):
        raise ServiceError(409, "dataset_in_use", "this project already has diagnosis sessions that reference its data; "
                           "delete them (or create a new project) before replacing a dataset")
    old = db.scalar(select(Dataset).where(Dataset.project_id == p.id, Dataset.role == role))
    rel, _ = store_dataframe(parsed.df, p.id, role)
    ds = Dataset(project_id=p.id, role=role, filename=parsed.filename, storage_path=rel, n_rows=len(parsed.df),
                 n_cols=parsed.df.shape[1], size_bytes=parsed.size_bytes, sha256=parsed.sha256,
                 schema=describe_schema(parsed.df))
    if old is not None:
        remove_storage(old.storage_path)
        db.delete(old)
        db.flush()
    db.add(ds)
    db.commit()
    return ds


def validate_mapping(db: Session, p: Project, m: ColumnMappingIn) -> tuple[dict, list[str]]:
    ref = next((d for d in p.datasets if d.role == "reference"), None)
    if ref is None:
        raise ServiceError(422, "missing_data", "upload the reference dataset before mapping columns")
    ref_cols = {c["name"] for c in ref.schema}
    rec = next((d for d in p.datasets if d.role == "recent"), None)
    rec_cols = {c["name"] for c in rec.schema} if rec else None
    warnings: list[str] = []
    mapped = {k: v for k, v in m.model_dump().items() if k != "features" and v}
    if len(set(mapped.values())) != len(mapped):
        raise ServiceError(422, "invalid_mapping", "each role needs a different column")
    for role, col in mapped.items():
        if col not in ref_cols:
            raise ServiceError(422, "invalid_mapping", f"{role} column '{col}' is not in the reference dataset")
        if rec_cols is not None and role in ("target", "prediction") and col not in rec_cols:
            raise ServiceError(422, "invalid_mapping", f"{role} column '{col}' is not in the recent dataset")
    feats = m.features
    if feats is not None:
        missing = [f for f in feats if f not in ref_cols]
        if missing:
            raise ServiceError(422, "invalid_mapping", f"unknown feature columns: {', '.join(missing)}")
        if set(feats) & set(mapped.values()):
            raise ServiceError(422, "invalid_mapping", "feature columns must not include the mapped target/prediction/timestamp/id columns")
    if not mapped.get("target"):
        warnings.append("No target column: label-based experiments will be unavailable.")
    if not mapped.get("prediction"):
        warnings.append("No prediction column: performance-based experiments will be unavailable.")
    if rec_cols is not None:
        absent = sorted(c for c in ref_cols - rec_cols if c not in mapped.values())
        if absent:
            warnings.append("Columns missing from the recent dataset (a schema change?): " + ", ".join(absent[:8]))
    result = {**mapped, "features": feats}
    return result, warnings


def delete_project(db: Session, p: Project) -> None:
    for d in p.datasets:
        remove_storage(d.storage_path)
    db.delete(p)
    db.commit()

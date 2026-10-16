from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.db import Base


def uid() -> str:
    return uuid.uuid4().hex


def now() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    task_type: Mapped[str] = mapped_column(String(40), default="classification")
    config: Mapped[dict] = mapped_column(JSON, default=dict)   # column mapping, defaults, pipeline notes
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    datasets: Mapped[list["Dataset"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    sessions: Mapped[list["DiagnosisSession"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))   # reference | recent | replay
    filename: Mapped[str] = mapped_column(String(200))
    storage_path: Mapped[str] = mapped_column(String(300))
    n_rows: Mapped[int] = mapped_column(Integer)
    n_cols: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    schema: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    project: Mapped[Project] = relationship(back_populates="datasets")
    __table_args__ = (Index("uq_dataset_role", "project_id", "role", unique=True),)


class DiagnosisSession(Base):
    __tablename__ = "diagnosis_sessions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20), default="active")   # active|paused|completed|stopped
    planner_variant: Mapped[str] = mapped_column(String(40), default="adaptive")
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    hypotheses: Mapped[list] = mapped_column(JSON, default=list)        # [{id, prior}]
    experiments: Mapped[list] = mapped_column(JSON, default=list)       # enabled experiment ids
    mapping: Mapped[dict] = mapped_column(JSON, default=dict)
    reported_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    symptoms: Mapped[dict] = mapped_column(JSON, default=dict)
    belief: Mapped[list] = mapped_column(JSON, default=list)            # current joint probabilities
    spent: Mapped[float] = mapped_column(Float, default=0.0)
    risk_total: Mapped[float] = mapped_column(Float, default=0.0)
    stop_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    stop_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    project: Mapped[Project] = relationship(back_populates="sessions")
    runs: Mapped[list["ExperimentRun"]] = relationship(back_populates="session", cascade="all, delete-orphan",
                                                       order_by="ExperimentRun.started_at")
    snapshots: Mapped[list["BeliefSnapshot"]] = relationship(back_populates="session", cascade="all, delete-orphan",
                                                             order_by="BeliefSnapshot.step")


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    session_id: Mapped[str] = mapped_column(ForeignKey("diagnosis_sessions.id", ondelete="CASCADE"), index=True)
    experiment_id: Mapped[str] = mapped_column(String(60))
    status: Mapped[str] = mapped_column(String(20))      # running|completed|failed|skipped|external
    source: Mapped[str] = mapped_column(String(20), default="backend")   # backend|external|skip
    step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(60), nullable=True)
    outcome_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    risk: Mapped[float] = mapped_column(Float, default=0.0)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    tables: Mapped[dict] = mapped_column(JSON, default=dict)
    findings: Mapped[list] = mapped_column(JSON, default=list)
    interpretation: Mapped[str] = mapped_column(Text, default="")
    limitations: Mapped[list] = mapped_column(JSON, default=list)
    recommendation: Mapped[dict] = mapped_column(JSON, default=dict)   # scores at selection time
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    runtime_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    session: Mapped[DiagnosisSession] = relationship(back_populates="runs")
    __table_args__ = (
        # an experiment can be live only once per session: blocks concurrent duplicate execution
        Index("uq_active_run", "session_id", "experiment_id", unique=True,
              sqlite_where=text("status IN ('running','completed','external')"),
              postgresql_where=text("status IN ('running','completed','external')")),
    )


class BeliefSnapshot(Base):
    __tablename__ = "belief_snapshots"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    session_id: Mapped[str] = mapped_column(ForeignKey("diagnosis_sessions.id", ondelete="CASCADE"), index=True)
    step: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(120))
    marginals: Mapped[dict] = mapped_column(JSON)
    probs: Mapped[list] = mapped_column(JSON)
    entropy: Mapped[float] = mapped_column(Float)
    map_state: Mapped[list] = mapped_column(JSON, default=list)
    map_probability: Mapped[float] = mapped_column(Float, default=0.0)
    spent: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    session: Mapped[DiagnosisSession] = relationship(back_populates="snapshots")


class ReportRecord(Base):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    session_id: Mapped[str] = mapped_column(ForeignKey("diagnosis_sessions.id", ondelete="CASCADE"), index=True)
    fmt: Mapped[str] = mapped_column(String(10))
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class BenchmarkJob(Base):
    __tablename__ = "benchmark_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(120))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="queued")   # queued|running|completed|failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(String(200), default="")
    result_path: Mapped[str | None] = mapped_column(String(300), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

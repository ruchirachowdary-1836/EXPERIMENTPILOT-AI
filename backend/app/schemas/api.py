from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------- projects / datasets
class ColumnMappingIn(BaseModel):
    target: str | None = None
    prediction: str | None = None
    timestamp: str | None = None
    identifier: str | None = None
    features: list[str] | None = None


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field("", max_length=2000)
    task_type: Literal["classification"] = "classification"
    pipeline_notes: str = Field("", max_length=4000)
    model_notes: str = Field("", max_length=4000)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v


class ProjectUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    description: str | None = Field(None, max_length=2000)
    pipeline_notes: str | None = Field(None, max_length=4000)
    model_notes: str | None = Field(None, max_length=4000)
    default_budget: float | None = Field(None, gt=0, le=1000)
    default_risk_cap: float | None = Field(None, ge=0, le=1)


class DatasetOut(ORM):
    id: str
    project_id: str
    role: str
    filename: str
    n_rows: int
    n_cols: int
    size_bytes: int
    sha256: str
    schema_: list[dict] = Field(alias="schema", serialization_alias="schema")
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    task_type: str
    config: dict
    created_at: datetime
    updated_at: datetime
    datasets: list[DatasetOut] = []
    n_sessions: int = 0
    ready: bool = False
    readiness: list[str] = []


class Page(BaseModel):
    items: list[Any]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------- sessions
class HypothesisSetting(BaseModel):
    id: str
    enabled: bool = True
    prior: float | None = Field(None, gt=0, lt=1)


class ReportedMetrics(BaseModel):
    metric: Literal["accuracy", "macro_f1"] = "accuracy"
    reference: float = Field(ge=0, le=1)
    recent: float = Field(ge=0, le=1)


class SessionCreate(BaseModel):
    project_id: str
    name: str = Field("Diagnosis session", min_length=1, max_length=120)
    planner_variant: str = "adaptive"
    budget: float = Field(8.0, gt=0, le=1000)
    risk_cap: float = Field(0.3, ge=0, le=1)
    risk_weight: float = Field(1.0, ge=0, le=100)
    confidence_threshold: float = Field(0.9, gt=0, le=1)
    min_score: float = Field(0.01, ge=0)
    separation_weight: float = Field(0.0, ge=0, le=10)
    confidence_mode: Literal["joint", "marginal"] = "joint"
    max_active: int | None = Field(None, ge=1, le=12)
    hypotheses: list[HypothesisSetting] | None = None
    experiments: list[str] | None = None
    reported_metrics: ReportedMetrics | None = None
    human_in_the_loop: bool = False
    seed: int = 0
    learner: Literal["logreg", "hgb"] = "logreg"

    @model_validator(mode="after")
    def _check(self):
        if self.hypotheses is not None:
            ids = [h.id for h in self.hypotheses]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate hypotheses")
        return self


class RunRequest(BaseModel):
    experiment_id: str | None = None      # None -> run the planner's recommendation
    approve: bool = False                 # explicit approval for risky / compute-heavy experiments


class ExternalOutcome(BaseModel):
    experiment_id: str
    outcome: str | int
    note: str = Field("", max_length=1000)


class SkipRequest(BaseModel):
    experiment_id: str | None = None
    reason: str = Field("", max_length=500)


class StopRequest(BaseModel):
    reason: str = Field("", max_length=500)


# ---------------------------------------------------------------- benchmark
class BenchmarkCreate(BaseModel):
    name: str = Field("Benchmark run", min_length=1, max_length=120)
    datasets: list[str] = ["synthetic_credit"]
    faults: list[str] = ["covariate_drift", "label_noise", "feature_leakage", "class_prior_shift", "pipeline_change",
                         "data_quality", "metric_bug"]
    severities: list[str] = ["medium"]
    include_pairs: bool = False
    planners: list[str] = ["random", "checklist", "cheapest", "max_eig", "eig_per_cost", "adaptive", "lookahead2"]
    seeds: list[int] = Field(default_factory=lambda: [1000, 1001, 1002])
    budgets: list[float] = [2, 4, 6, 8, 12]
    budget: float = Field(8.0, gt=0, le=100)
    risk_cap: float = Field(0.15, ge=0, le=1)
    risk_weight: float = Field(1.0, ge=0, le=100)
    confidence_threshold: float = Field(0.9, gt=0, le=1)

    @field_validator("seeds")
    @classmethod
    def _seeds(cls, v: list[int]):
        if not 1 <= len(v) <= 20:
            raise ValueError("choose between 1 and 20 seeds")
        return v

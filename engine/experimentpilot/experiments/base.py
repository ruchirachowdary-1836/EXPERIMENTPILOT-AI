"""Plugin interface for diagnostic experiments."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.planner import ExperimentSpec


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be executed on the supplied data."""


@dataclass
class ColumnMapping:
    target: str | None = None
    prediction: str | None = None
    timestamp: str | None = None
    identifier: str | None = None
    features: list[str] | None = None  # None -> infer: every column that is not mapped

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ExperimentContext:
    reference: pd.DataFrame
    recent: pd.DataFrame
    mapping: ColumnMapping
    replay: pd.DataFrame | None = None
    reported_metrics: dict[str, float] | None = None  # {"metric","reference","recent"}
    config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def feature_names(self) -> list[str]:
        m = self.mapping
        if m.features:
            cols = [c for c in m.features if c in self.reference.columns]
        else:
            skip = {m.target, m.prediction, m.timestamp, m.identifier}
            cols = [c for c in self.reference.columns if c not in skip]
        return cols

    def available_inputs(self) -> set[str]:
        m, avail = self.mapping, {"reference", "recent"}
        if self.feature_names():
            avail.add("features")
        for role, df in (("reference", self.reference), ("recent", self.recent)):
            if m.target and m.target in df.columns and df[m.target].notna().sum() >= 20:
                avail.add(f"labels_{role}")
            if m.prediction and m.prediction in df.columns and df[m.prediction].notna().sum() >= 20:
                avail.add(f"predictions_{role}")
        if self.replay is not None and len(self.replay):
            avail.add("replay")
        if self.reported_metrics and self.reported_metrics.get("reference") is not None \
                and self.reported_metrics.get("recent") is not None:
            avail.add("reported_metrics")
        return avail


@dataclass
class Finding:
    level: str  # "info" | "warning" | "alert"
    text: str


@dataclass
class ExperimentResult:
    experiment_id: str
    outcome: str
    outcome_index: int
    score: float                       # underlying numeric indicator that was discretised
    stats: dict[str, Any] = field(default_factory=dict)      # raw numerical results
    tables: dict[str, list[dict]] = field(default_factory=dict)   # evidence artifacts (tabular)
    findings: list[Finding] = field(default_factory=list)
    interpretation: str = ""
    limitations: list[str] = field(default_factory=list)
    runtime_s: float = 0.0

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id, "outcome": self.outcome, "outcome_index": self.outcome_index,
                "score": _clean(self.score), "stats": _clean(self.stats), "tables": _clean(self.tables),
                "findings": [f.__dict__ for f in self.findings], "interpretation": self.interpretation,
                "limitations": self.limitations, "runtime_s": round(self.runtime_s, 4)}


def _clean(o):
    """Convert numpy/pandas scalars so results are JSON-serialisable; NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


class Experiment(ABC):
    id: str = ""
    name: str = ""
    description: str = ""
    cost: float = 1.0
    risk: float = 0.0
    requires: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ("none", "moderate", "severe")
    repeatable: bool = False
    requires_approval: bool = False
    limitations: tuple[str, ...] = ()

    def spec(self) -> ExperimentSpec:
        return ExperimentSpec(self.id, self.name, self.cost, self.risk, self.requires, self.outcomes,
                              self.description, self.repeatable, self.requires_approval)

    def missing_inputs(self, ctx: ExperimentContext) -> list[str]:
        have = ctx.available_inputs()
        return [r for r in self.requires if r not in have]

    def available(self, ctx: ExperimentContext) -> tuple[bool, list[str]]:
        miss = self.missing_inputs(ctx)
        return (not miss, miss)

    @abstractmethod
    def _run(self, ctx: ExperimentContext) -> ExperimentResult: ...

    def run(self, ctx: ExperimentContext) -> ExperimentResult:
        ok, missing = self.available(ctx)
        if not ok:
            raise ExperimentError(f"{self.id}: missing required inputs: {', '.join(missing)}")
        t0 = time.perf_counter()
        try:
            res = self._run(ctx)
        except ExperimentError:
            raise
        except Exception as exc:  # failure handling: never leak internals, keep the cause
            raise ExperimentError(f"{self.id} failed: {type(exc).__name__}: {exc}") from exc
        res.runtime_s = time.perf_counter() - t0
        if not res.limitations:
            res.limitations = list(self.limitations)
        return res

    def result(self, score: float, level: int, **kw) -> ExperimentResult:
        return ExperimentResult(self.id, self.outcomes[level], level, float(score), **kw)


def level_from(score: float, thresholds: tuple[float, ...]) -> int:
    """Ordinal bucket: number of thresholds that ``score`` meets or exceeds."""
    return int(sum(score >= t for t in thresholds))

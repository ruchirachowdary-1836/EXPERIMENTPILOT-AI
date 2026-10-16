"""Experiment registry (plugin system)."""
from __future__ import annotations

from .base import (ColumnMapping, Experiment, ExperimentContext, ExperimentError, ExperimentResult, Finding)
from .distribution import ClassPriorComparison, DataQualityAudit, FeatureDrift, PipelineReplay
from .model_based import (ImportanceWeightedEval, LabelNoiseAudit, LeakageScan, MetricRecompute, RetrainOnRecent,
                          SliceAnalysis)

EXPERIMENTS: dict[str, Experiment] = {}


def register_experiment(exp: Experiment) -> Experiment:
    """Register a new experiment plugin; the planner needs a likelihood model with the same id."""
    if not exp.id or exp.id in EXPERIMENTS:
        raise ValueError(f"invalid or duplicate experiment id '{exp.id}'")
    if len(exp.outcomes) < 2:
        raise ValueError("an experiment needs at least two outcomes")
    EXPERIMENTS[exp.id] = exp
    return exp


for _cls in (FeatureDrift, DataQualityAudit, ClassPriorComparison, LabelNoiseAudit, LeakageScan, PipelineReplay,
             ImportanceWeightedEval, SliceAnalysis, RetrainOnRecent, MetricRecompute):
    register_experiment(_cls())


def get_experiment(exp_id: str) -> Experiment:
    try:
        return EXPERIMENTS[exp_id]
    except KeyError as exc:
        raise KeyError(f"unknown experiment '{exp_id}'") from exc


def all_specs():
    return [e.spec() for e in EXPERIMENTS.values()]

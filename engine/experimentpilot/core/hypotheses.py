"""Root-cause hypothesis library (extensible registry)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hypothesis:
    id: str
    code: str
    name: str
    description: str
    symptom: str
    default_prior: float  # prior P(cause is active)
    core: bool = True
    default_enabled: bool = True


_REGISTRY: dict[str, Hypothesis] = {}


def register_hypothesis(h: Hypothesis) -> Hypothesis:
    if h.id in _REGISTRY:
        raise ValueError(f"hypothesis '{h.id}' already registered")
    if not 0.0 < h.default_prior < 1.0:
        raise ValueError("default_prior must lie in (0, 1)")
    _REGISTRY[h.id] = h
    return h


def get_hypothesis(hid: str) -> Hypothesis:
    try:
        return _REGISTRY[hid]
    except KeyError as exc:
        raise KeyError(f"unknown hypothesis '{hid}'") from exc


def all_hypotheses() -> list[Hypothesis]:
    return list(_REGISTRY.values())


def default_hypothesis_ids() -> list[str]:
    return [h.id for h in _REGISTRY.values() if h.default_enabled]


for _h in [
    Hypothesis("covariate_drift", "H1", "Covariate drift",
               "Input feature distributions in production differ from the training/reference data.",
               "Feature distributions shift; accuracy degrades gradually or suddenly.", 0.30),
    Hypothesis("label_noise", "H2", "Label noise / label-definition change",
               "Recent labels are noisier or follow a changed labelling guideline.",
               "Metric drops on freshly labelled data while inputs look normal.", 0.20),
    Hypothesis("feature_leakage", "H3", "Feature leakage",
               "A feature leaked the target during training/validation but not in production.",
               "Excellent offline metric, poor online performance.", 0.15),
    Hypothesis("class_prior_shift", "H4", "Class-prior shift / imbalance",
               "Class frequencies differ between reference and recent data.",
               "Precision and recall move in opposite directions; thresholds miscalibrated.", 0.20),
    Hypothesis("pipeline_change", "H5", "Preprocessing / train-serving pipeline change",
               "Preprocessing differs between training and serving (scaling, encoding, units).",
               "Sudden drop after a release; schema or scaling differences.", 0.20),
    Hypothesis("data_quality", "H6", "Upstream data-quality problem",
               "Missing values, stale or sentinel-valued features, or schema violations upstream.",
               "Spike in missingness, constants or out-of-range values.", 0.20),
    Hypothesis("metric_bug", "H7", "Evaluation / metric bug",
               "The reported metric is computed on mismatched splits or labels.",
               "Reported drop not reproducible from raw predictions and labels.", 0.08,
               core=False),
    Hypothesis("model_regression", "H8", "Model / code regression",
               "A retrained model or code release is worse than the previous artifact.",
               "Degradation coincides with a model or code release.", 0.08,
               core=False, default_enabled=False),
]:
    register_hypothesis(_h)

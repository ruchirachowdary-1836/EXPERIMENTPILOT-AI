"""Fault-injection scenarios: build clean data, inject faults, train a model, produce the diagnostic inputs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..experiments import EXPERIMENTS
from ..experiments.base import ColumnMapping, ExperimentContext, ExperimentError
from .datasets import DATASETS, make_base
from .injectors import (DATA_FAULTS, FAULT_TO_HYPOTHESIS, FAULTS, LABEL_FAULTS, SEVERITIES, World,
                        inject_data_faults, inject_label_faults, inject_pre_training)

MAPPING = ColumnMapping(target="target", prediction="prediction", timestamp="event_time", identifier="record_id")


@dataclass(frozen=True)
class ScenarioSpec:
    dataset: str
    faults: tuple[tuple[str, str], ...]   # ((fault, severity), ...) ; empty = no-fault control
    seed: int

    @property
    def id(self) -> str:
        f = "+".join(f"{a}:{b}" for a, b in self.faults) or "control"
        return f"{self.dataset}|{f}|{self.seed}"

    @property
    def truth(self) -> tuple[str, ...]:
        return tuple(sorted({FAULT_TO_HYPOTHESIS[f] for f, _ in self.faults}))


@dataclass
class Scenario:
    spec: ScenarioSpec
    reference: pd.DataFrame
    recent: pd.DataFrame
    replay: pd.DataFrame
    mapping: ColumnMapping
    reported_metrics: dict
    truth: list[str]
    meta: dict = field(default_factory=dict)

    def context(self, seed: int | None = None, config: dict | None = None) -> ExperimentContext:
        return ExperimentContext(self.reference, self.recent, self.mapping, self.replay, self.reported_metrics,
                                 config or {"learner": DATASETS[self.spec.dataset]["learner"]},
                                 self.spec.seed if seed is None else seed)


def fit_model(train: pd.DataFrame, features: list[str], learner: str, seed: int) -> Pipeline:
    num = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    cat = [c for c in features if c not in num]
    pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    clf = (HistGradientBoostingClassifier(max_iter=80, max_depth=3, random_state=seed) if learner == "hgb"
           else LogisticRegression(max_iter=400, random_state=seed))
    return Pipeline([("pre", pre), ("clf", clf)]).fit(train[features], train["target"])


def build_scenario(spec: ScenarioSpec) -> Scenario:
    cfg = DATASETS[spec.dataset]
    rng = np.random.default_rng([spec.seed, list(DATASETS).index(spec.dataset)])
    n_tr, n_ref, n_rec = cfg["n_train"], cfg["n_ref"], cfg["n_recent"]
    pool_mult = 3 if cfg["synthetic"] else 1  # breast_cancer is small: the pool is whatever remains
    base = make_base(spec.dataset, n_tr + n_ref + pool_mult * n_rec, rng)
    n_pool = len(base) - n_tr - n_ref
    features = [c for c in base.columns if c not in ("record_id", "event_time", "target")]
    train = base.iloc[:n_tr].copy()
    ref = base.iloc[n_tr:n_tr + n_ref].copy()
    pool = base.iloc[n_tr + n_ref:].copy()
    n_rec = min(n_rec, n_pool)
    world = World(train, ref, pool, n_rec, features, rng)
    for f, s in spec.faults:
        if f not in FAULTS or s not in SEVERITIES:
            raise ValueError(f"invalid fault '{f}:{s}'")
        inject_pre_training(world, f, s)
    for f, s in spec.faults:
        inject_label_faults(world, f, s)
    # draw recent rows (optionally resampled to a shifted class prior)
    if world.positive_rate is not None:
        k_pos = int(round(world.positive_rate * n_rec))
        pos, neg = pool[pool["target"] == 1], pool[pool["target"] == 0]
        k_pos, k_neg = min(k_pos, len(pos)), min(n_rec - k_pos, len(neg))
        recent = pd.concat([pos.sample(k_pos, random_state=int(rng.integers(1 << 31))),
                            neg.sample(k_neg, random_state=int(rng.integers(1 << 31)))]).sample(frac=1, random_state=int(rng.integers(1 << 31)))
    else:
        recent = pool.iloc[:n_rec]
    recent = recent.sort_values("record_id").copy()
    recent["event_time"] = pd.date_range("2025-06-01", periods=len(recent), freq="h").strftime("%Y-%m-%d %H:%M")
    ref["event_time"] = pd.date_range("2025-01-01", periods=len(ref), freq="h").strftime("%Y-%m-%d %H:%M")
    world.recent, world.replay = recent.reset_index(drop=True), recent.reset_index(drop=True).copy()
    for f, s in spec.faults:
        if f in DATA_FAULTS:
            inject_data_faults(world, f, s)
    model = fit_model(world.train, features, cfg["learner"], spec.seed)
    ref_out, rec_out = world.ref.copy(), world.recent.copy()
    ref_out["prediction"] = model.predict(ref_out[features])
    rec_out["prediction"] = model.predict(rec_out[features])
    acc_ref = float((ref_out["prediction"] == ref_out["target"]).mean())
    acc_rec_true = float((rec_out["prediction"] == rec_out["target"]).mean())
    reported_recent = acc_rec_true
    if world.metric_shuffle > 0:  # evaluation bug: a fraction of labels misaligned with predictions
        y = rec_out["target"].to_numpy().copy()
        idx = np.flatnonzero(rng.random(len(y)) < world.metric_shuffle)
        y[idx] = rng.permutation(y[idx])
        reported_recent = float((rec_out["prediction"].to_numpy() == y).mean())
    if world.label_flip > 0:
        flip = rng.random(len(rec_out)) < world.label_flip
        rec_out.loc[flip, "target"] = 1 - rec_out.loc[flip, "target"]
        acc_rec_true = float((rec_out["prediction"] == rec_out["target"]).mean())
        reported_recent = acc_rec_true
    cols = ["record_id", "event_time"] + features + ["target", "prediction"]
    replay_out = world.replay.copy()
    replay_out["prediction"] = rec_out["prediction"].to_numpy()
    replay_out["target"] = rec_out["target"].to_numpy()
    meta = {"accuracy_reference": acc_ref, "accuracy_recent": acc_rec_true, "reported_recent_accuracy": reported_recent,
            "accuracy_drop": acc_ref - acc_rec_true, "dataset": spec.dataset, **world.meta}
    return Scenario(spec, ref_out[cols], rec_out[cols], replay_out[cols], MAPPING,
                    {"metric": "accuracy", "reference": acc_ref, "recent": reported_recent},
                    sorted(set(world.truth)), meta)


def compute_outcomes(scenario: Scenario, experiment_ids: list[str] | None = None) -> dict:
    """Run every experiment on the scenario and return discretised outcomes + raw indicator scores."""
    ctx = scenario.context()
    out: dict[str, dict] = {}
    avail = ctx.available_inputs()
    for eid, exp in EXPERIMENTS.items():
        if experiment_ids and eid not in experiment_ids:
            continue
        if any(r not in avail for r in exp.requires):
            continue
        t0 = time.perf_counter()
        try:
            res = exp.run(ctx)
            out[eid] = {"outcome_index": res.outcome_index, "outcome": res.outcome, "score": res.score,
                        "runtime_s": time.perf_counter() - t0}
        except ExperimentError as e:
            out[eid] = {"error": str(e)}
    return out

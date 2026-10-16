"""Controlled fault injectors. Each mutates a :class:`World` in place and records ground truth."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEVERITIES = ("low", "medium", "high")
FAULT_TO_HYPOTHESIS = {
    "covariate_drift": "covariate_drift", "label_noise": "label_noise", "feature_leakage": "feature_leakage",
    "class_prior_shift": "class_prior_shift", "pipeline_change": "pipeline_change", "data_quality": "data_quality",
    "metric_bug": "metric_bug",
}
FAULTS = tuple(FAULT_TO_HYPOTHESIS)
LEAK_COL = "aux_signal"


@dataclass
class World:
    train: pd.DataFrame
    ref: pd.DataFrame
    pool: pd.DataFrame          # candidate recent rows (larger than needed so priors can be resampled)
    n_recent: int
    features: list[str]
    rng: np.random.Generator
    recent: pd.DataFrame | None = None
    replay: pd.DataFrame | None = None
    truth: list[str] = field(default_factory=list)
    positive_rate: float | None = None      # requested recent positive rate (class-prior shift)
    label_flip: float = 0.0
    metric_shuffle: float = 0.0
    meta: dict = field(default_factory=dict)

    def numeric_features(self, exclude_aux: bool = True) -> list[str]:
        cols = [c for c in self.features if pd.api.types.is_numeric_dtype(self.train[c])]
        return [c for c in cols if not (exclude_aux and c == LEAK_COL)]

    def categorical_features(self) -> list[str]:
        return [c for c in self.features if not pd.api.types.is_numeric_dtype(self.train[c])]


def _most_predictive(world: World, cols: list[str], k: int, rng: np.random.Generator) -> list[str]:
    """Pick k of the most target-correlated columns (random tie-break) so faults actually hurt the model."""
    y = world.train["target"].to_numpy(float)
    corr = {c: abs(float(np.corrcoef(world.train[c].to_numpy(float), y)[0, 1])) for c in cols}
    top = sorted(cols, key=lambda c: -corr[c])[: max(k + 1, int(np.ceil(1.5 * k)))]
    return [str(c) for c in rng.choice(top, size=min(k, len(top)), replace=False)]


def _level(severity: str) -> int:
    return SEVERITIES.index(severity)


def inject_pre_training(world: World, fault: str, severity: str) -> None:
    """Faults that must exist before the model is trained."""
    if fault == "feature_leakage":
        a = (1.2, 2.2, 3.5)[_level(severity)]
        for df in (world.train, world.ref):
            y = df["target"].to_numpy(float)
            df[LEAK_COL] = a * (y - 0.5) * 2 / 2 + world.rng.normal(0, 1, len(df))
        world.truth.append("feature_leakage")
        world.meta["leak_strength"] = a


def inject_data_faults(world: World, fault: str, severity: str) -> None:
    """Faults that change the recent feature table (applied to ``world.pool`` rows via world.recent/replay)."""
    lvl, rng = _level(severity), world.rng
    rec, rep = world.recent, world.replay
    if fault == "covariate_drift":
        num = world.numeric_features()
        k = max(2, int(round(0.4 * len(num)))) if len(num) > 3 else max(1, len(num) // 2)
        cols = _most_predictive(world, num, k, rng)
        shift, scale = (0.5, 1.0, 1.8)[lvl], (1.0, 1.15, 1.3)[lvl]
        for c in cols:
            sd = float(world.train[c].std()) or 1.0
            mu = float(world.train[c].mean())
            c_sign = 1 if float(np.corrcoef(world.train[c].to_numpy(float), world.train['target'].to_numpy(float))[0, 1]) > 0 else -1
            sign = c_sign if rng.random() < 0.8 else -c_sign  # mostly towards the minority class, so drift hurts
            for df in (rec, rep):
                df[c] = mu + (df[c] - mu) * scale + sign * shift * sd
        world.meta["drifted"] = cols
        world.truth.append("covariate_drift")
    elif fault == "data_quality":
        feats = [c for c in world.features if c != LEAK_COL]
        k = max(2, int(round(0.3 * len(feats))))
        cols = list(rng.choice(feats, size=min(k, len(feats)), replace=False))
        rate = (0.05, 0.15, 0.35)[lvl]
        for c in cols:
            mask = rng.random(len(rec)) < rate
            for df in (rec, rep):
                df[c] = df[c].astype(object).where(~mask, np.nan) if not pd.api.types.is_numeric_dtype(df[c]) else df[c].mask(mask)
        numc = [c for c in world.numeric_features() if c not in cols]
        if lvl >= 1 and numc:
            c = _most_predictive(world, numc, 1, rng)[0]
            mask = rng.random(len(rec)) < 0.12
            for df in (rec, rep):
                df.loc[mask, c] = -999.0
            world.meta["sentinel"] = c
        if lvl == 2 and len(numc) > 1:
            c = str([x for x in numc if x != world.meta.get("sentinel")][0])
            for df in (rec, rep):
                df[c] = float(world.train[c].median())
            world.meta["stale"] = c
        world.meta["null_columns"] = cols
        world.truth.append("data_quality")
    elif fault == "pipeline_change":  # only the *current* pipeline output changes; replay keeps the old output
        num = world.numeric_features()
        k = max(1, int(round(0.25 * len(num))))
        cols = list(rng.choice(num, size=min(k, len(num)), replace=False))
        scale, off = (1.1, 1.4, 2.5)[lvl], (0.1, 0.3, 0.8)[lvl]
        for c in cols:
            sd = float(world.train[c].std()) or 1.0
            rec[c] = rec[c] * scale + off * sd
        cats = world.categorical_features()
        if lvl >= 1 and cats:
            c = cats[0]
            lv = list(world.train[c].value_counts().index[:2])
            if len(lv) == 2:
                swap = {lv[0]: lv[1], lv[1]: lv[0]}
                rec[c] = rec[c].map(lambda v: swap.get(v, v))
        world.meta["pipeline_columns"] = cols
        world.truth.append("pipeline_change")


def inject_label_faults(world: World, fault: str, severity: str) -> None:
    lvl = _level(severity)
    if fault == "label_noise":
        world.label_flip = (0.07, 0.15, 0.28)[lvl]
        world.truth.append("label_noise")
    elif fault == "metric_bug":
        world.metric_shuffle = (0.10, 0.25, 0.50)[lvl]
        world.truth.append("metric_bug")
    elif fault == "class_prior_shift":
        base = float(world.pool["target"].mean())
        up = base < 0.5  # shift towards the minority class so the prior shift is not benign
        odds = base / (1 - base) * ((1.7, 2.8, 5.0)[lvl] if up else (0.6, 0.35, 0.2)[lvl])
        world.positive_rate = float(np.clip(odds / (1 + odds), 0.03, 0.97))
        world.truth.append("class_prior_shift")
        world.meta["recent_positive_rate_requested"] = world.positive_rate


DATA_FAULTS = {"covariate_drift", "data_quality", "pipeline_change"}
LABEL_FAULTS = {"label_noise", "metric_bug", "class_prior_shift"}

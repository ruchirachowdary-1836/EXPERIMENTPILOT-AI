"""Likelihood-table estimation from fault-injection outcomes (with expert-prior smoothing)."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..core.likelihood_model import LikelihoodLibrary, NoisyOrLikelihood
from ..experiments import EXPERIMENTS

# Cold-start expert prior: P(alarm | only this cause), conservative. Used as Beta pseudo-counts.
EXPERT_PRIORS: dict[str, dict[str, float]] = {
    "feature_drift": {"covariate_drift": 0.85, "pipeline_change": 0.6, "data_quality": 0.4},
    "data_quality_audit": {"data_quality": 0.9, "pipeline_change": 0.3},
    "class_prior": {"class_prior_shift": 0.9},
    "label_noise": {"label_noise": 0.8},
    "leakage_scan": {"feature_leakage": 0.8},
    "pipeline_replay": {"pipeline_change": 0.9},
    "importance_weighted_eval": {"covariate_drift": 0.7, "pipeline_change": 0.5, "data_quality": 0.3},
    "slice_analysis": {"covariate_drift": 0.3, "data_quality": 0.3, "feature_leakage": 0.3},
    "retrain_recent": {"covariate_drift": 0.6, "feature_leakage": 0.6, "pipeline_change": 0.5},
    "metric_recompute": {"metric_bug": 0.95},
}
EXPERT_LEAK = 0.08
EXPERT_BASE = 0.10


def estimate_library(records: Iterable[dict], hypotheses: Sequence[str], pseudo_count: float = 2.0,
                     experiments: Sequence[str] | None = None, provenance: dict | None = None) -> LikelihoodLibrary:
    """Estimate cumulative noisy-OR parameters.

    For each experiment e and threshold j:
      leak_j   = (#{o>=j | no fault} + a*p0) / (n_control + a)
      q_{j,h}  = (#{o>=j | only h}  + a*p0) / (n_h + a)       (Beta smoothing, a = pseudo_count)
      sens_{j,h} = 1 - (1 - q_{j,h}) / (1 - leak_j)            (clipped, non-increasing in j)

    Only records whose ground truth is empty (control) or a single hypothesis are used.
    """
    recs = [r for r in records if len(r["truth"]) <= 1]
    models: dict[str, NoisyOrLikelihood] = {}
    for eid, exp in EXPERIMENTS.items():
        if experiments and eid not in experiments:
            continue
        L = len(exp.outcomes)
        obs: dict[str, list[int]] = {"": []}
        for r in recs:
            o = r["outcomes"].get(eid)
            if not o or "outcome_index" not in o:
                continue
            key = r["truth"][0] if r["truth"] else ""
            obs.setdefault(key, []).append(int(o["outcome_index"]))

        def exceed(vals: list[int], prior: float) -> np.ndarray:
            v = np.asarray(vals, dtype=int)
            out = []
            for j in range(1, L):
                p0 = prior if j == 1 else prior / (1 + 2 * (j - 1))   # higher levels are rarer a priori
                out.append(((v >= j).sum() + pseudo_count * p0) / (len(v) + pseudo_count))
            return np.minimum.accumulate(np.array(out))

        leak = exceed(obs[""], EXPERT_LEAK)
        sens: dict[str, np.ndarray] = {}
        for h in hypotheses:
            prior = EXPERT_PRIORS.get(eid, {}).get(h, EXPERT_BASE * 0.5)
            q = exceed(obs.get(h, []), prior)
            s = 1.0 - (1.0 - q) / np.maximum(1.0 - leak, 1e-6)
            sens[h] = np.minimum.accumulate(np.clip(s, 0.0, 0.999))
        models[eid] = NoisyOrLikelihood(eid, exp.outcomes, leak, sens, n_runs=sum(len(v) for v in obs.values()),
                                        source="fault-injection simulation")
    prov = {"method": "cumulative noisy-OR, Beta-smoothed", "pseudo_count": pseudo_count, "simulated": True,
            "n_records": len(recs), **(provenance or {})}
    return LikelihoodLibrary(models, prov)


def noisy_or_validation(records: Iterable[dict], library: LikelihoodLibrary, hypotheses: Sequence[str]) -> dict:
    """Mean log-likelihood of observed outcomes in two-fault records: noisy-OR vs. a naive single-cause mixture."""
    from ..core.belief import enumerate_states
    states = enumerate_states(len(hypotheses), len(hypotheses))
    idx = {h: i for i, h in enumerate(hypotheses)}
    nor, mix, n = [], [], 0
    for r in records:
        if len(r["truth"]) != 2:
            continue
        z = np.zeros(len(hypotheses), dtype=bool)
        for h in r["truth"]:
            z[idx[h]] = True
        row = (states == z).all(axis=1)
        for eid, o in r["outcomes"].items():
            if "outcome_index" not in o or eid not in library:
                continue
            m = library.get(eid)
            mat = m.matrix(hypotheses, states)
            k = o["outcome_index"]
            nor.append(np.log(mat[row][0, k]))
            singles = []
            for h in r["truth"]:
                zz = np.zeros(len(hypotheses), dtype=bool); zz[idx[h]] = True
                singles.append(mat[(states == zz).all(axis=1)][0, k])
            mix.append(np.log(np.mean(singles)))
            n += 1
    if not n:
        return {"n": 0}
    return {"n": n, "mean_loglik_noisy_or": float(np.mean(nor)), "mean_loglik_single_cause_mixture": float(np.mean(mix))}

"""Outcome (likelihood) models P(o | e, z).

``NoisyOrLikelihood`` is an ordinal cumulative noisy-OR model: for outcome levels
0..L-1 and threshold j in 1..L-1,

    P(o >= j | z) = 1 - (1 - leak_j) * prod_{k active in z} (1 - sens_{k,j})

``leak_j`` is the false-alarm rate when nothing is active and ``sens_{k,j}`` is the
probability that cause k alone pushes the outcome to level >= j. Two simultaneous
causes therefore raise the alarm probability without exact joint tables.
``TableLikelihood`` is an explicit [S, L] table (used for tests and worked examples).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

FLOOR = 1e-6


def _normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.clip(mat, FLOOR, None)
    return mat / mat.sum(axis=1, keepdims=True)


@dataclass
class NoisyOrLikelihood:
    experiment_id: str
    outcomes: tuple[str, ...]
    leak: np.ndarray  # [L-1]
    sens: dict[str, np.ndarray]  # hypothesis id -> [L-1]
    n_runs: int = 0
    source: str = "expert"

    def __post_init__(self) -> None:
        l1 = len(self.outcomes) - 1
        if l1 < 1:
            raise ValueError("an experiment needs at least two outcomes")
        self.leak = np.clip(np.asarray(self.leak, dtype=float), 0.0, 0.999)
        if self.leak.shape != (l1,):
            raise ValueError("leak has wrong shape")
        for h, s in self.sens.items():
            s = np.clip(np.asarray(s, dtype=float), 0.0, 0.999)
            if s.shape != (l1,):
                raise ValueError(f"sens for {h} has wrong shape")
            self.sens[h] = s

    def matrix(self, hypotheses: Sequence[str], states: np.ndarray) -> np.ndarray:
        """[S, L] outcome distribution for every joint state."""
        l1 = len(self.outcomes) - 1
        log_not = np.zeros((states.shape[0], l1))
        for k, h in enumerate(hypotheses):
            s = self.sens.get(h)
            if s is None:
                continue
            log_not += states[:, [k]] * np.log1p(-s)[None, :]
        p_ge = 1.0 - (1.0 - self.leak)[None, :] * np.exp(log_not)
        p_ge = np.minimum.accumulate(p_ge, axis=1)  # P(o>=j) must be non-increasing in j
        full = np.concatenate([np.ones((states.shape[0], 1)), p_ge, np.zeros((states.shape[0], 1))], axis=1)
        return _normalise(full[:, :-1] - full[:, 1:])

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id, "outcomes": list(self.outcomes),
                "leak": self.leak.tolist(), "sens": {h: s.tolist() for h, s in self.sens.items()},
                "n_runs": self.n_runs, "source": self.source}

    @classmethod
    def from_dict(cls, d: Mapping) -> "NoisyOrLikelihood":
        return cls(d["experiment_id"], tuple(d["outcomes"]), np.array(d["leak"], dtype=float),
                   {h: np.array(s, dtype=float) for h, s in d["sens"].items()},
                   int(d.get("n_runs", 0)), d.get("source", "expert"))


@dataclass
class TableLikelihood:
    """Explicit likelihood for a fixed list of states, shape [S, L]."""
    experiment_id: str
    outcomes: tuple[str, ...]
    table: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.table, dtype=float)
        if t.ndim != 2 or t.shape[1] != len(self.outcomes):
            raise ValueError("table must be [n_states, n_outcomes]")
        if np.any(t < 0) or not np.allclose(t.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("each row of the table must be a probability distribution")
        self.table = t

    def matrix(self, hypotheses: Sequence[str], states: np.ndarray) -> np.ndarray:
        if states.shape[0] != self.table.shape[0]:
            raise ValueError("table row count does not match the state space")
        return self.table


@dataclass
class LikelihoodLibrary:
    models: dict[str, NoisyOrLikelihood] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def get(self, experiment_id: str):
        return self.models[experiment_id]

    def __contains__(self, experiment_id: str) -> bool:
        return experiment_id in self.models

    def to_json(self) -> str:
        return json.dumps({"provenance": self.provenance,
                           "models": {k: m.to_dict() for k, m in self.models.items()}}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "LikelihoodLibrary":
        d = json.loads(text)
        return cls({k: NoisyOrLikelihood.from_dict(v) for k, v in d["models"].items()}, d.get("provenance", {}))

    def perturbed(self, sigma: float, rng: np.random.Generator) -> "LikelihoodLibrary":
        """Misspecified copy: Gaussian noise in logit space (robustness ablation)."""
        def pert(a: np.ndarray) -> np.ndarray:
            a = np.clip(a, 1e-3, 1 - 1e-3)
            z = np.log(a / (1 - a)) + rng.normal(0, sigma, a.shape)
            return 1 / (1 + np.exp(-z))
        models = {}
        for k, m in self.models.items():
            models[k] = NoisyOrLikelihood(m.experiment_id, m.outcomes, pert(m.leak),
                                          {h: pert(s) for h, s in m.sens.items()}, m.n_runs, "perturbed")
        return LikelihoodLibrary(models, {**self.provenance, "perturbed_sigma": sigma})

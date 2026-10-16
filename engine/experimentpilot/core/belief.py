"""Joint Bayesian belief over binary root-cause states.

The state z in {0,1}^K says which hypotheses are simultaneously active. Exact
enumeration is used when the state space is small. For larger hypothesis sets the
support is truncated to states with at most ``max_active`` active causes (a
sparsity assumption); ``max_active=K`` is exact. See docs/methodology.md.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

EPS = 1e-12
MAX_STATES = 200_000


def entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def enumerate_states(k: int, max_active: int | None = None) -> np.ndarray:
    """All binary vectors of length k with at most ``max_active`` ones (bool [S, k])."""
    if k < 1:
        raise ValueError("need at least one hypothesis")
    m = k if max_active is None else max(0, min(max_active, k))
    n_states = sum(math.comb(k, j) for j in range(m + 1))
    if n_states > MAX_STATES:
        raise ValueError(f"state space of {n_states} states exceeds limit {MAX_STATES}; lower max_active")
    rows = []
    for j in range(m + 1):
        for idx in itertools.combinations(range(k), j):
            r = np.zeros(k, dtype=bool)
            r[list(idx)] = True
            rows.append(r)
    return np.array(rows, dtype=bool)


@dataclass
class Snapshot:
    step: int
    label: str
    probs: np.ndarray
    marginals: np.ndarray
    entropy: float


@dataclass
class Belief:
    hypotheses: tuple[str, ...]
    states: np.ndarray  # bool [S, K]
    probs: np.ndarray  # [S]
    history: list[Snapshot] = field(default_factory=list)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_priors(cls, hypotheses: Sequence[str], priors: Sequence[float],
                    max_active: int | None = None, none_mass: float | None = None) -> "Belief":
        """Independent-Bernoulli prior truncated to <= max_active causes.

        ``none_mass`` optionally overrides the probability of the all-inactive state
        ("no fault"); the rest of the mass is rescaled.
        """
        hyps = tuple(hypotheses)
        pri = np.asarray(priors, dtype=float)
        if len(hyps) != len(pri):
            raise ValueError("hypotheses and priors differ in length")
        if len(set(hyps)) != len(hyps):
            raise ValueError("duplicate hypotheses")
        if np.any(pri <= 0) or np.any(pri >= 1) or not np.all(np.isfinite(pri)):
            raise ValueError("prior probabilities must lie strictly in (0, 1)")
        states = enumerate_states(len(hyps), max_active)
        logp = (states * np.log(pri) + (~states) * np.log1p(-pri)).sum(axis=1)
        probs = np.exp(logp - logp.max())
        probs /= probs.sum()
        if none_mass is not None:
            if not 0.0 <= none_mass < 1.0:
                raise ValueError("none_mass must lie in [0, 1)")
            zero = ~states.any(axis=1)
            rest = probs[~zero].sum()
            probs = np.where(zero, 0.0, probs * (1 - none_mass) / max(rest, EPS))
            probs[zero] = none_mass
        b = cls(hyps, states, probs)
        b.record(0, "prior")
        return b

    @classmethod
    def from_states(cls, hypotheses: Sequence[str], states: np.ndarray, probs: Sequence[float]) -> "Belief":
        p = np.asarray(probs, dtype=float)
        if p.shape[0] != states.shape[0] or np.any(p < 0) or p.sum() <= 0:
            raise ValueError("invalid state probabilities")
        b = cls(tuple(hypotheses), np.asarray(states, dtype=bool), p / p.sum())
        b.record(0, "prior")
        return b

    # ---- queries ------------------------------------------------------
    @property
    def n_states(self) -> int:
        return self.states.shape[0]

    def marginals(self) -> np.ndarray:
        return self.probs @ self.states.astype(float)

    def marginal_dict(self) -> dict[str, float]:
        return {h: float(m) for h, m in zip(self.hypotheses, self.marginals())}

    def entropy(self) -> float:
        return entropy_bits(self.probs)

    def map_state(self) -> tuple[tuple[str, ...], float]:
        i = int(np.argmax(self.probs))
        return tuple(h for h, a in zip(self.hypotheses, self.states[i]) if a), float(self.probs[i])

    def ranked(self) -> list[tuple[str, float]]:
        m = self.marginals()
        order = np.argsort(-m, kind="stable")
        return [(self.hypotheses[i], float(m[i])) for i in order]

    def diagnosed_set(self, threshold: float = 0.5) -> list[str]:
        return [h for h, p in self.ranked() if p >= threshold]

    def conditional_on(self, idx: int, active: bool = True) -> np.ndarray:
        """Belief over states conditioned on hypothesis ``idx`` being active/inactive."""
        mask = self.states[:, idx] == active
        w = np.where(mask, self.probs, 0.0)
        s = w.sum()
        return w / s if s > 0 else self.probs.copy()

    # ---- updates ------------------------------------------------------
    def update(self, lik_col: np.ndarray, label: str | None = None) -> "Belief":
        """Bayes update with the likelihood P(o | e, z) of the observed outcome (shape [S])."""
        lik = np.asarray(lik_col, dtype=float)
        if lik.shape != self.probs.shape:
            raise ValueError("likelihood vector has wrong shape")
        if np.any(lik < 0) or not np.all(np.isfinite(lik)):
            raise ValueError("likelihood must be finite and non-negative")
        with np.errstate(divide="ignore"):
            logpost = np.log(self.probs) + np.log(lik)
        if not np.isfinite(logpost).any():
            raise ValueError("observed outcome has zero likelihood under every state with prior support")
        logpost -= logpost[np.isfinite(logpost)].max()
        post = np.exp(logpost)
        post /= post.sum()
        self.probs = post
        self.record(len(self.history), label or "update")
        return self

    def update_soft(self, lik_matrix: np.ndarray, weights: Sequence[float], label: str | None = None) -> "Belief":
        """Virtual-evidence update when the outcome itself is uncertain."""
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != lik_matrix.shape[1] or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("invalid outcome weights")
        return self.update(lik_matrix @ (w / w.sum()), label)

    def skip(self, label: str = "missing evidence") -> "Belief":
        """Missing evidence: the posterior is unchanged, but the history records it."""
        self.record(len(self.history), label)
        return self

    def record(self, step: int, label: str) -> None:
        self.history.append(Snapshot(step, label, self.probs.copy(), self.marginals(), self.entropy()))

    def copy(self) -> "Belief":
        return Belief(self.hypotheses, self.states, self.probs.copy(), list(self.history))

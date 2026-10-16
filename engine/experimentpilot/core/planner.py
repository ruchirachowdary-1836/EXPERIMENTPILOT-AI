"""Experiment scoring, feasibility filtering and recommendation rationale."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .belief import Belief
from .eig import expected_information_gain, separation_score


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    name: str
    cost: float
    risk: float
    requires: tuple[str, ...]
    outcomes: tuple[str, ...]
    description: str = ""
    repeatable: bool = False
    requires_approval: bool = False


@dataclass
class PlannerConfig:
    budget: float = 12.0
    risk_cap: float = 0.3
    risk_weight: float = 1.0           # lambda
    confidence_threshold: float = 0.90  # tau
    min_score: float = 0.01            # epsilon: minimum useful score per cost unit
    separation_weight: float = 0.0     # additive bonus weight for the separation score
    cost_floor: float = 1e-3           # denominator floor, keeps zero-cost experiments finite
    confidence_mode: str = "joint"     # "joint" (MAP state) or "marginal" (every cause decided)
    odds_ratio_stop: float | None = None
    max_experiments: int | None = None
    checklist: tuple[str, ...] = ()

    def validate(self) -> "PlannerConfig":
        if self.budget < 0 or not np.isfinite(self.budget):
            raise ValueError("budget must be a non-negative number")
        if not 0.0 <= self.risk_cap <= 1.0:
            raise ValueError("risk_cap must lie in [0, 1]")
        if self.risk_weight < 0:
            raise ValueError("risk_weight must be non-negative")
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in (0, 1]")
        if self.min_score < 0 or self.separation_weight < 0 or self.cost_floor <= 0:
            raise ValueError("min_score/separation_weight must be >= 0 and cost_floor > 0")
        if self.confidence_mode not in ("joint", "marginal"):
            raise ValueError("confidence_mode must be 'joint' or 'marginal'")
        return self


class LikelihoodBank:
    """Caches the [S, L] likelihood matrix of each experiment for a given state space."""

    def __init__(self, library, hypotheses: Sequence[str], states: np.ndarray):
        self.library, self.hypotheses, self.states = library, tuple(hypotheses), states
        self._cache: dict[str, np.ndarray] = {}

    def has(self, exp_id: str) -> bool:
        return exp_id in self.library

    def matrix(self, exp_id: str) -> np.ndarray:
        if exp_id not in self._cache:
            self._cache[exp_id] = self.library.get(exp_id).matrix(self.hypotheses, self.states)
        return self._cache[exp_id]


@dataclass
class PlanState:
    spent: float = 0.0
    done: dict[str, int] = field(default_factory=dict)       # exp id -> times executed
    skipped: set[str] = field(default_factory=set)
    available_inputs: set[str] = field(default_factory=set)
    n_run: int = 0


@dataclass
class ScoredExperiment:
    id: str
    name: str
    eig: float
    cost: float
    risk: float
    denominator: float
    score: float
    separation: float
    adjusted_score: float
    feasible: bool
    reasons: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    requires_approval: bool = False
    rank: int | None = None
    lookahead_value: float | None = None

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in self.__dict__.items()}


def feasibility(spec: ExperimentSpec, state: PlanState, cfg: PlannerConfig,
                enforce_risk: bool = True) -> tuple[list[str], list[str]]:
    """Return (reason codes, human-readable reasons) for infeasibility; empty means feasible."""
    codes, msgs = [], []
    times = state.done.get(spec.id, 0)
    if spec.id in state.skipped:
        codes.append("skipped"); msgs.append("skipped by the user")
    if times and not spec.repeatable:
        codes.append("done"); msgs.append("already executed")
    missing = [r for r in spec.requires if r not in state.available_inputs]
    if missing:
        codes.append("prereq"); msgs.append("missing required input: " + ", ".join(missing))
    remaining = cfg.budget - state.spent
    if spec.cost > remaining + 1e-9:
        codes.append("budget"); msgs.append(f"cost {spec.cost:g} exceeds remaining budget {max(remaining, 0):g}")
    if enforce_risk and spec.risk > cfg.risk_cap + 1e-12:
        codes.append("risk"); msgs.append(f"risk {spec.risk:g} exceeds the risk cap {cfg.risk_cap:g}")
    return codes, msgs


def top_two(belief: Belief) -> tuple[int, int]:
    m = belief.marginals()
    order = np.argsort(-m, kind="stable")
    return int(order[0]), int(order[1] if len(order) > 1 else order[0])


def score_experiments(belief: Belief, specs: Iterable[ExperimentSpec], bank: LikelihoodBank,
                      cfg: PlannerConfig, state: PlanState, enforce_risk: bool = True,
                      use_risk_term: bool = True) -> list[ScoredExperiment]:
    """score(e) = EIG(e) / (cost(e) + lambda * risk(e)), floored; plus the separation score."""
    a, b = top_two(belief)
    out: list[ScoredExperiment] = []
    for sp in specs:
        codes, msgs = feasibility(sp, state, cfg, enforce_risk)
        if not bank.has(sp.id):
            codes.append("no_likelihood"); msgs.append("no likelihood model available")
        if "no_likelihood" in codes:
            eig = sep = 0.0
        else:
            lik = bank.matrix(sp.id)
            eig = expected_information_gain(belief.probs, lik)
            sep = separation_score(belief.probs, belief.states, lik, a, b) if a != b else 0.0
        denom = max(sp.cost + (cfg.risk_weight * sp.risk if use_risk_term else 0.0), cfg.cost_floor)
        score = eig / denom
        out.append(ScoredExperiment(sp.id, sp.name, eig, sp.cost, sp.risk, denom, score, sep,
                                    score + cfg.separation_weight * sep, not codes, msgs, codes,
                                    sp.requires_approval))
    return out


def rank(scored: list[ScoredExperiment], key: Callable[[ScoredExperiment], float] | None = None) -> list[ScoredExperiment]:
    key = key or (lambda s: s.adjusted_score)
    feas = sorted([s for s in scored if s.feasible], key=lambda s: (-key(s), s.cost, s.id))
    for i, s in enumerate(feas, 1):
        s.rank = i
    return feas


def explain_selection(selected: ScoredExperiment, scored: list[ScoredExperiment], belief: Belief,
                      cfg: PlannerConfig, variant_label: str) -> str:
    a, b = top_two(belief)
    ha, hb = belief.hypotheses[a], belief.hypotheses[b]
    parts = [f"{variant_label} selected '{selected.name}': expected information gain {selected.eig:.3f} bits at cost "
             f"{selected.cost:g} and risk {selected.risk:g} (score {selected.score:.3f} bits per cost unit)."]
    if a != b:
        parts.append(f"It separates the two leading hypotheses ({ha.replace('_', ' ')} vs {hb.replace('_', ' ')}) "
                     f"with total-variation separation {selected.separation:.2f}.")
    others = [s for s in scored if s.feasible and s.id != selected.id]
    others.sort(key=lambda s: -s.adjusted_score)
    if others:
        o = others[0]
        parts.append(f"Closest alternative: '{o.name}' (EIG {o.eig:.3f} bits, score {o.score:.3f}).")
    rejected = [s for s in scored if not s.feasible and "done" not in s.reason_codes]
    if rejected:
        parts.append("Rejected: " + "; ".join(f"{s.name} ({', '.join(s.reasons)})" for s in rejected[:5]) + ".")
    return " ".join(parts)

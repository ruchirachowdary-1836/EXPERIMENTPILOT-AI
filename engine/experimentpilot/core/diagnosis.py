"""Closed-loop diagnosis driver (used by the benchmark and tests).

The web application drives the very same primitives step by step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .belief import Belief
from .planner import ExperimentSpec, LikelihoodBank, PlannerConfig, PlanState
from .stopping import StopDecision, evaluate_stop
from .variants import PlannerVariant


@dataclass
class StepRecord:
    step: int
    experiment_id: str
    outcome_index: int
    cost: float
    risk: float
    eig: float
    score: float
    separation: float
    spent_after: float
    marginals: dict[str, float]
    map_confidence: float
    rationale: str = ""


@dataclass
class DiagnosisTrace:
    steps: list[StepRecord] = field(default_factory=list)
    stop: StopDecision | None = None
    belief: Belief | None = None
    spent: float = 0.0
    total_risk: float = 0.0
    risk_violations: int = 0

    @property
    def n_experiments(self) -> int:
        return len(self.steps)


def run_diagnosis(variant: PlannerVariant, belief: Belief, specs: list[ExperimentSpec], bank: LikelihoodBank,
                  cfg: PlannerConfig, oracle: Callable[[str], int | None], available_inputs: set[str],
                  rng: np.random.Generator | None = None, true_likelihood: LikelihoodBank | None = None,
                  report_risk_cap: float | None = None, stop_on_confidence: bool = True) -> DiagnosisTrace:
    """Run until a stopping rule fires.

    ``oracle(exp_id)`` returns the observed outcome index (or None when the experiment
    could not be executed, in which case it is skipped). ``bank`` is the planner's
    (possibly misspecified) belief about outcomes.
    """
    rng = rng or np.random.default_rng(0)
    belief = belief.copy()
    state = PlanState(available_inputs=set(available_inputs))
    trace = DiagnosisTrace(belief=belief)
    cap = cfg.risk_cap if report_risk_cap is None else report_risk_cap
    spec_by = {s.id: s for s in specs}
    for _ in range(len(specs) * 2 + 2):
        choice = variant.choose(belief, specs, bank, cfg, state, rng)
        stop = evaluate_stop(belief, choice.scored, cfg, state, variant.check_low_value)
        if stop.stop and (stop_on_confidence or stop.code != "confidence_reached"):
            trace.stop = stop
            break
        if choice.selected is None:
            trace.stop = evaluate_stop(belief, choice.scored, cfg, state, False)
            if not trace.stop.stop:
                trace.stop = StopDecision(True, "no_feasible_experiments", "No feasible experiments remain.")
            break
        sel, sp = choice.selected, spec_by[choice.selected.id]
        o = oracle(sel.id)
        if o is None:
            state.skipped.add(sel.id)
            belief.skip(f"{sel.id}: unavailable")
            continue
        belief.update(bank.matrix(sel.id)[:, o], sel.id)
        state.spent += sp.cost
        state.n_run += 1
        state.done[sel.id] = state.done.get(sel.id, 0) + 1
        trace.total_risk += sp.risk
        trace.risk_violations += int(sp.risk > cap + 1e-12)
        trace.steps.append(StepRecord(state.n_run, sel.id, int(o), sp.cost, sp.risk, sel.eig, sel.score,
                                      sel.separation, state.spent, belief.marginal_dict(),
                                      belief.map_state()[1], choice.rationale))
    else:  # pragma: no cover - defensive
        trace.stop = StopDecision(True, "no_feasible_experiments", "Iteration guard reached.")
    trace.spent = state.spent
    trace.belief = belief
    return trace

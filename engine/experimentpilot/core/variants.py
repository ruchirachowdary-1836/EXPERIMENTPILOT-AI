"""Planner variants: seven distinct experiment-selection strategies.

risk-agnostic variants (max_eig, eig_per_cost) deliberately ignore the risk cap, as
information-only / cost-only baselines do in practice; the others enforce it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .belief import Belief
from .eig import expected_information_gain, posterior_given_outcome, outcome_distribution
from .planner import (ExperimentSpec, LikelihoodBank, PlannerConfig, PlanState, ScoredExperiment,
                      explain_selection, rank, score_experiments)

DEFAULT_CHECKLIST = ("metric_recompute", "data_quality_audit", "feature_drift", "class_prior",
                     "pipeline_replay", "label_noise", "leakage_scan", "slice_analysis",
                     "importance_weighted_eval", "retrain_recent")


@dataclass
class Choice:
    selected: ScoredExperiment | None
    scored: list[ScoredExperiment]
    rationale: str = ""


class PlannerVariant:
    id = "base"
    label = "Base"
    description = ""
    enforce_risk = True
    use_risk_term = True
    check_low_value = False

    def score(self, belief, specs, bank, cfg, state):
        return score_experiments(belief, specs, bank, cfg, state, self.enforce_risk, self.use_risk_term)

    def pick(self, feasible: list[ScoredExperiment], belief, specs, bank, cfg, state, rng):
        raise NotImplementedError

    def choose(self, belief: Belief, specs: list[ExperimentSpec], bank: LikelihoodBank, cfg: PlannerConfig,
               state: PlanState, rng: np.random.Generator | None = None) -> Choice:
        rng = rng or np.random.default_rng(0)
        scored = self.score(belief, specs, bank, cfg, state)
        feasible = [s for s in scored if s.feasible]
        if not feasible:
            return Choice(None, scored)
        sel = self.pick(feasible, belief, specs, bank, cfg, state, rng)
        rank(scored)
        return Choice(sel, scored, explain_selection(sel, scored, belief, cfg, self.label))


class RandomFeasible(PlannerVariant):
    id, label = "random", "Random feasible"
    description = "Pick uniformly among feasible experiments (lower-bound baseline)."

    def pick(self, feasible, belief, specs, bank, cfg, state, rng):
        return feasible[int(rng.integers(len(feasible)))]


class FixedChecklist(PlannerVariant):
    id, label = "checklist", "Fixed checklist"
    description = "Run feasible experiments in a static, hand-ordered sequence (current practice)."

    def pick(self, feasible, belief, specs, bank, cfg, state, rng):
        order = list(cfg.checklist or DEFAULT_CHECKLIST)
        pos = {e: i for i, e in enumerate(order)}
        return min(feasible, key=lambda s: (pos.get(s.id, len(order)), s.id))


class CheapestFirst(PlannerVariant):
    id, label = "cheapest", "Cheapest first"
    description = "Sort feasible experiments by cost only (cost-only baseline)."

    def pick(self, feasible, belief, specs, bank, cfg, state, rng):
        return min(feasible, key=lambda s: (s.cost, s.id))


class MaxEIG(PlannerVariant):
    id, label = "max_eig", "Max information gain"
    description = "Maximise expected information gain, ignoring cost and risk (information-only baseline)."
    enforce_risk = False
    use_risk_term = False

    def pick(self, feasible, belief, specs, bank, cfg, state, rng):
        return max(feasible, key=lambda s: (round(s.eig, 12), -s.cost, s.id))


class EIGPerCost(PlannerVariant):
    id, label = "eig_per_cost", "Information gain per cost"
    description = "Maximise EIG / cost, ignoring risk."
    enforce_risk = False
    use_risk_term = False
    check_low_value = True

    def pick(self, feasible, belief, specs, bank, cfg, state, rng):
        return max(feasible, key=lambda s: (round(s.score, 12), s.id))


class Adaptive(PlannerVariant):
    id, label = "adaptive", "Cost-and-risk-aware adaptive"
    description = "Maximise EIG / (cost + lambda*risk) (+ optional separation bonus) under budget and risk-cap constraints."
    check_low_value = True

    def pick(self, feasible, belief, specs, bank, cfg, state, rng):
        return max(feasible, key=lambda s: (round(s.adjusted_score, 12), s.id))


class Lookahead2(PlannerVariant):
    id, label = "lookahead2", "Two-step lookahead"
    description = ("Bounded two-step lookahead: value of e = (EIG(e) + E_o[EIG of the best greedy follow-up]) / "
                   "(cost(e) + E_o[cost of that follow-up]) under budget and risk constraints.")
    check_low_value = True

    def pick(self, feasible, belief, specs, bank, cfg, state, rng):
        spec_by = {s.id: s for s in specs}
        best, best_v = None, -1.0
        for cand in feasible:
            lik = bank.matrix(cand.id)
            p_o = outcome_distribution(belief.probs, lik)
            gain, cost = cand.eig, cand.denominator
            sub_cfg = replace(cfg)
            for o, po in enumerate(p_o):
                if po <= 1e-12:
                    continue
                b2 = Belief(belief.hypotheses, belief.states, posterior_given_outcome(belief.probs, lik, o))
                st2 = PlanState(state.spent + cand.cost, dict(state.done), set(state.skipped),
                                set(state.available_inputs), state.n_run + 1)
                st2.done[cand.id] = st2.done.get(cand.id, 0) + 1
                nxt = [s for s in score_experiments(b2, specs, bank, sub_cfg, st2, True, True) if s.feasible]
                if nxt:
                    n = max(nxt, key=lambda s: (s.score, s.id))
                    gain += po * n.eig
                    cost += po * n.denominator
            v = gain / max(cost, cfg.cost_floor)
            cand.lookahead_value = v
            if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and best is not None and cand.id < best.id):
                best, best_v = cand, v
        return best


VARIANTS: dict[str, PlannerVariant] = {v.id: v for v in
                                       (RandomFeasible(), FixedChecklist(), CheapestFirst(), MaxEIG(),
                                        EIGPerCost(), Adaptive(), Lookahead2())}


def get_variant(variant_id: str) -> PlannerVariant:
    try:
        return VARIANTS[variant_id]
    except KeyError as exc:
        raise KeyError(f"unknown planner variant '{variant_id}'") from exc

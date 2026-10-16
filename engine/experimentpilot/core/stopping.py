"""Stopping policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .belief import Belief
from .planner import PlannerConfig, PlanState, ScoredExperiment

STOP_MESSAGES = {
    "confidence_reached": "The configured confidence threshold was reached.",
    "no_feasible_experiments": "No feasible experiments remain.",
    "budget_insufficient": "The remaining budget cannot pay for any remaining experiment.",
    "low_expected_value": "The best remaining experiment is not worth its cost (score below the stopping threshold).",
    "safety_blocked": "Every remaining experiment is blocked by the risk cap.",
    "max_experiments": "The maximum number of experiments was reached.",
}


@dataclass
class StopDecision:
    stop: bool
    code: str | None = None
    message: str = ""
    confidence: float = 0.0


def diagnosis_confidence(belief: Belief, mode: str) -> float:
    if mode == "marginal":
        m = belief.marginals()
        return float(np.min(np.maximum(m, 1 - m)))
    return belief.map_state()[1]


def evaluate_stop(belief: Belief, scored: Sequence[ScoredExperiment], cfg: PlannerConfig, state: PlanState,
                  check_low_value: bool = True) -> StopDecision:
    conf = diagnosis_confidence(belief, cfg.confidence_mode)
    if conf >= cfg.confidence_threshold:
        return StopDecision(True, "confidence_reached", STOP_MESSAGES["confidence_reached"], conf)
    if cfg.odds_ratio_stop is not None:
        srt = np.sort(belief.probs)[::-1]
        if len(srt) > 1 and srt[1] > 0 and srt[0] / srt[1] >= cfg.odds_ratio_stop:
            return StopDecision(True, "confidence_reached", "Top-two odds ratio exceeded the stopping threshold.", conf)
    if cfg.max_experiments is not None and state.n_run >= cfg.max_experiments:
        return StopDecision(True, "max_experiments", STOP_MESSAGES["max_experiments"], conf)
    feasible = [s for s in scored if s.feasible]
    if not feasible:
        remaining = [s for s in scored if not ({"done", "skipped", "no_likelihood"} & set(s.reason_codes))]
        code = "no_feasible_experiments"
        if any("budget" in s.reason_codes and "prereq" not in s.reason_codes for s in remaining):
            code = "budget_insufficient"
        elif any("risk" in s.reason_codes and "prereq" not in s.reason_codes for s in remaining):
            code = "safety_blocked"
        return StopDecision(True, code, STOP_MESSAGES[code], conf)
    if check_low_value and max(s.adjusted_score for s in feasible) < cfg.min_score:
        return StopDecision(True, "low_expected_value", STOP_MESSAGES["low_expected_value"], conf)
    return StopDecision(False, None, "", conf)

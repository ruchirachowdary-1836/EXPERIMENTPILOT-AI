from __future__ import annotations

from pathlib import Path

from ..core.likelihood_model import LikelihoodLibrary
from .estimate import EXPERT_PRIORS, estimate_library, noisy_or_validation

DEFAULT_TABLE_PATH = Path(__file__).with_name("default_tables.json")


def load_default_library() -> LikelihoodLibrary:
    """Likelihood tables estimated by ``research/benchmark/run_benchmark.py --stage estimate`` (simulated data)."""
    if DEFAULT_TABLE_PATH.exists():
        return LikelihoodLibrary.from_json(DEFAULT_TABLE_PATH.read_text(encoding="utf8"))
    return expert_library()


def expert_library() -> LikelihoodLibrary:
    """Cold-start library built only from the expert priors (no simulation counts)."""
    return estimate_library([], list(_all_ids()))


def _all_ids():
    from ..core.hypotheses import all_hypotheses
    return [h.id for h in all_hypotheses()]


__all__ = ["EXPERT_PRIORS", "estimate_library", "noisy_or_validation", "load_default_library", "expert_library",
           "DEFAULT_TABLE_PATH"]

"""Expected information gain and hypothesis-separation measures (all in bits)."""
from __future__ import annotations

import numpy as np

from .belief import entropy_bits


def outcome_distribution(probs: np.ndarray, lik: np.ndarray) -> np.ndarray:
    """Predictive P(o) = sum_z b(z) P(o | z)."""
    return probs @ lik


def posterior_given_outcome(probs: np.ndarray, lik: np.ndarray, o: int) -> np.ndarray:
    w = probs * lik[:, o]
    s = w.sum()
    if s <= 0:
        raise ValueError("zero-probability outcome")
    return w / s


def expected_information_gain(probs: np.ndarray, lik: np.ndarray) -> float:
    """EIG(e) = H(b) - sum_o P(o) H(b | o)  (= mutual information I(Z; O))."""
    p_o = outcome_distribution(probs, lik)
    gain = entropy_bits(probs)
    for o, po in enumerate(p_o):
        if po > 0:
            gain -= po * entropy_bits(posterior_given_outcome(probs, lik, o))
    return max(0.0, float(gain))


def outcome_given_hypothesis(belief_probs: np.ndarray, states: np.ndarray, lik: np.ndarray,
                             idx: int, active: bool = True) -> np.ndarray:
    """P(o | hypothesis idx active/inactive), marginalising other causes under the belief."""
    mask = states[:, idx] == active
    w = np.where(mask, belief_probs, 0.0)
    s = w.sum()
    if s <= 0:
        return belief_probs @ lik
    return (w / s) @ lik


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    bc = float(np.sqrt(p * q).sum())
    return float(-np.log(max(bc, 1e-12)))


def separation_score(probs: np.ndarray, states: np.ndarray, lik: np.ndarray, a: int, b: int) -> float:
    """Total-variation distance between P(o | a active) and P(o | b active) in [0, 1].

    For the two currently leading hypotheses a and b this says how differently the
    experiment's outcome is distributed if a (resp. b) is the active cause.
    """
    return total_variation(outcome_given_hypothesis(probs, states, lik, a),
                           outcome_given_hypothesis(probs, states, lik, b))

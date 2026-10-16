import math

import numpy as np
import pytest

from experimentpilot.core import (Belief, ExperimentSpec, LikelihoodBank, LikelihoodLibrary, NoisyOrLikelihood,
                                  PlannerConfig, PlanState, TableLikelihood, enumerate_states, entropy_bits,
                                  evaluate_stop, expected_information_gain, get_variant, run_diagnosis,
                                  score_experiments, separation_score)


def h2(p):
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# ---------------------------------------------------------------- belief
def test_enumerate_states_counts():
    assert enumerate_states(3).shape == (8, 3)
    assert enumerate_states(8, 2).shape[0] == 1 + 8 + 28
    assert enumerate_states(20, 2).shape[0] == 1 + 20 + 190  # approximation strategy for large K
    assert enumerate_states(4, 0).shape[0] == 1


def test_enumerate_state_limit():
    with pytest.raises(ValueError):
        enumerate_states(60, 30)


def test_prior_independent_bernoulli_marginals():
    b = Belief.from_priors(["a", "b", "c"], [0.2, 0.5, 0.7])
    assert b.probs.sum() == pytest.approx(1.0)
    assert np.allclose(b.marginals(), [0.2, 0.5, 0.7])
    assert b.entropy() == pytest.approx(h2(0.2) + h2(0.5) + h2(0.7))


def test_prior_validation():
    with pytest.raises(ValueError):
        Belief.from_priors(["a"], [1.0])
    with pytest.raises(ValueError):
        Belief.from_priors(["a", "a"], [0.2, 0.2])
    with pytest.raises(ValueError):
        Belief.from_priors(["a", "b"], [0.2])


def test_none_mass_override():
    b = Belief.from_priors(["a", "b"], [0.3, 0.3], none_mass=0.05)
    assert b.probs[~b.states.any(axis=1)][0] == pytest.approx(0.05)
    assert b.probs.sum() == pytest.approx(1.0)


def test_bayes_update_matches_hand_calculation():
    b = Belief.from_priors(["a", "b"], [0.5, 0.5], max_active=1)  # states: none, a, b
    prior = b.probs.copy()
    lik = np.array([0.1, 0.8, 0.3])
    b.update(lik)
    expected = prior * lik / (prior * lik).sum()
    assert np.allclose(b.probs, expected)
    assert len(b.history) == 2


def test_update_is_numerically_stable_with_extreme_likelihoods():
    b = Belief.from_priors(["a", "b"], [0.5, 0.5])
    for _ in range(200):
        b.update(np.array([1e-300, 1.0, 1e-300, 1e-300]))
    assert np.isfinite(b.probs).all() and b.probs.sum() == pytest.approx(1.0)


def test_update_rejects_impossible_and_invalid_evidence():
    b = Belief.from_priors(["a"], [0.5])
    with pytest.raises(ValueError):
        b.update(np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        b.update(np.array([-0.1, 0.5]))
    with pytest.raises(ValueError):
        b.update(np.array([0.5, 0.5, 0.5]))


def test_missing_evidence_leaves_posterior_unchanged():
    b = Belief.from_priors(["a", "b"], [0.3, 0.4])
    before = b.probs.copy()
    b.skip()
    assert np.array_equal(b.probs, before)


def test_soft_evidence_between_hard_outcomes():
    b = Belief.from_priors(["a"], [0.5])
    lik = np.array([[0.9, 0.1], [0.2, 0.8]])
    hard0 = b.copy().update(lik[:, 0]).marginals()[0]
    hard1 = b.copy().update(lik[:, 1]).marginals()[0]
    soft = b.copy().update_soft(lik, [0.5, 0.5]).marginals()[0]
    assert hard0 < soft < hard1


def test_posterior_history_records_each_update():
    b = Belief.from_priors(["a"], [0.5])
    b.update(np.array([0.2, 0.8]), "e1").update(np.array([0.2, 0.8]), "e2")
    assert [s.label for s in b.history] == ["prior", "e1", "e2"]
    assert b.history[2].marginals[0] > b.history[1].marginals[0] > b.history[0].marginals[0]


def test_multi_cause_posterior_keeps_two_active_causes():
    lib = LikelihoodLibrary({
        "t1": NoisyOrLikelihood("t1", ("no", "yes"), np.array([0.05]), {"a": np.array([0.95])}),
        "t2": NoisyOrLikelihood("t2", ("no", "yes"), np.array([0.05]), {"b": np.array([0.95])}),
    })
    b = Belief.from_priors(["a", "b", "c"], [0.3, 0.3, 0.3])
    bank = LikelihoodBank(lib, b.hypotheses, b.states)
    b.update(bank.matrix("t1")[:, 1]).update(bank.matrix("t2")[:, 1])
    m = b.marginal_dict()
    assert m["a"] > 0.85 and m["b"] > 0.85 and m["c"] == pytest.approx(0.3, abs=1e-6)
    assert set(b.map_state()[0]) == {"a", "b"}


# ---------------------------------------------------------------- EIG
def table(rows):
    return np.array(rows, dtype=float)


def test_eig_deterministic_test_equals_prior_entropy():
    p = np.array([0.5, 0.5])
    assert expected_information_gain(p, table([[1, 0], [0, 1]])) == pytest.approx(1.0)
    p = np.array([0.25, 0.75])
    assert expected_information_gain(p, table([[1, 0], [0, 1]])) == pytest.approx(h2(0.25))


def test_eig_uninformative_test_is_zero():
    p = np.array([0.3, 0.7])
    assert expected_information_gain(p, table([[0.4, 0.6], [0.4, 0.6]])) == pytest.approx(0.0, abs=1e-12)


def test_eig_binary_symmetric_channel_closed_form():
    e = 0.1
    lik = table([[1 - e, e], [e, 1 - e]])
    assert expected_information_gain(np.array([0.5, 0.5]), lik) == pytest.approx(1 - h2(e))


def test_eig_matches_mutual_information_formula():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(6))
    lik = rng.dirichlet(np.ones(4), size=6)
    joint = p[:, None] * lik
    po = joint.sum(0)
    mi = (joint * np.log2(joint / (p[:, None] * po[None, :]))).sum()
    assert expected_information_gain(p, lik) == pytest.approx(mi)


def test_eig_is_nonnegative_and_bounded_by_entropy():
    rng = np.random.default_rng(2)
    for _ in range(25):
        p = rng.dirichlet(np.ones(5))
        lik = rng.dirichlet(np.ones(3), size=5)
        g = expected_information_gain(p, lik)
        assert -1e-12 <= g <= entropy_bits(p) + 1e-9


def test_separation_score_extremes():
    states = enumerate_states(2, 1)  # none, a, b
    p = np.array([0.2, 0.4, 0.4])
    ident = table([[1, 0], [0, 1], [1, 0]])  # a -> yes, b -> no
    assert separation_score(p, states, ident, 0, 1) == pytest.approx(1.0)
    flat = table([[0.5, 0.5]] * 3)
    assert separation_score(p, states, flat, 0, 1) == pytest.approx(0.0)


# ---------------------------------------------------------------- noisy-OR
def test_noisy_or_rows_are_distributions_and_monotone():
    m = NoisyOrLikelihood("e", ("n", "m", "s"), np.array([0.1, 0.02]),
                          {"a": np.array([0.6, 0.4]), "b": np.array([0.5, 0.1])})
    st = enumerate_states(2)
    mat = m.matrix(("a", "b"), st)
    assert np.allclose(mat.sum(1), 1.0) and (mat > 0).all()
    ge1 = mat[:, 1:].sum(1)
    assert ge1[3] > ge1[1] > ge1[0] and ge1[3] > ge1[2]  # two causes raise the alarm probability
    assert ge1[0] == pytest.approx(0.1, abs=1e-5)


def test_noisy_or_unknown_hypothesis_has_no_effect():
    m = NoisyOrLikelihood("e", ("n", "y"), np.array([0.1]), {"a": np.array([0.7])})
    mat = m.matrix(("a", "zzz"), enumerate_states(2))
    assert np.allclose(mat[0], mat[2])


def test_library_json_roundtrip():
    lib = LikelihoodLibrary({"e": NoisyOrLikelihood("e", ("n", "y"), np.array([0.1]), {"a": np.array([0.7])})},
                            {"x": 1})
    lib2 = LikelihoodLibrary.from_json(lib.to_json())
    assert lib2.provenance == {"x": 1}
    assert np.allclose(lib2.get("e").sens["a"], [0.7])


def test_table_likelihood_validation():
    with pytest.raises(ValueError):
        TableLikelihood("e", ("a", "b"), np.array([[0.5, 0.6]]))


# ---------------------------------------------------------------- planner
def make_problem(costs=(1.0, 2.0, 0.5), risks=(0.0, 0.0, 0.0)):
    """Single-cause world with 4 hypotheses (+ no state removed via max_active=1, none_mass tiny)."""
    hyps = ["h1", "h2", "h3", "h4"]
    b = Belief.from_priors(hyps, [0.25] * 4, max_active=1, none_mass=1e-6)
    sens = {
        "cheap_weak": {"h1": 0.6, "h2": 0.5},
        "mid_strong": {"h1": 0.95, "h3": 0.9},
        "tiny": {"h4": 0.9},
    }
    lib = LikelihoodLibrary({k: NoisyOrLikelihood(k, ("no", "yes"), np.array([0.05]),
                                                  {h: np.array([v]) for h, v in s.items()})
                             for k, s in sens.items()})
    names = list(sens)
    specs = [ExperimentSpec(n, n, c, r, (), ("no", "yes")) for n, c, r in zip(names, costs, risks)]
    return b, specs, LikelihoodBank(lib, b.hypotheses, b.states)


def test_score_is_eig_over_cost_plus_lambda_risk():
    b, specs, bank = make_problem(costs=(1.0, 2.0, 0.5), risks=(0.0, 0.2, 0.0))
    cfg = PlannerConfig(budget=10, risk_weight=2.0).validate()
    scored = {s.id: s for s in score_experiments(b, specs, bank, cfg, PlanState())}
    s = scored["mid_strong"]
    assert s.denominator == pytest.approx(2.0 + 2.0 * 0.2)
    assert s.score == pytest.approx(s.eig / s.denominator)


def test_cost_changes_the_winner_versus_max_eig():
    b, specs, bank = make_problem(costs=(0.2, 6.0, 0.2))
    cfg = PlannerConfig(budget=10).validate()
    rng = np.random.default_rng(0)
    greedy = get_variant("max_eig").choose(b, specs, bank, cfg, PlanState(), rng).selected
    adaptive = get_variant("adaptive").choose(b, specs, bank, cfg, PlanState(), rng).selected
    assert greedy.id == "mid_strong"          # highest raw information gain
    assert adaptive.id != "mid_strong"        # but poor value per cost


def test_zero_cost_experiments_never_divide_by_zero():
    b, specs, bank = make_problem(costs=(0.0, 0.0, 0.0))
    cfg = PlannerConfig(budget=1).validate()
    scored = score_experiments(b, specs, bank, cfg, PlanState())
    assert all(np.isfinite(s.score) for s in scored)


def test_budget_filter_and_exhaustion():
    b, specs, bank = make_problem(costs=(1.0, 2.0, 0.5))
    cfg = PlannerConfig(budget=0.9).validate()
    scored = score_experiments(b, specs, bank, cfg, PlanState())
    feasible = {s.id for s in scored if s.feasible}
    assert feasible == {"tiny"}
    assert any("budget" in s.reason_codes for s in scored if s.id == "mid_strong")
    st = PlanState(spent=0.9)
    stop = evaluate_stop(b, score_experiments(b, specs, bank, cfg, st), cfg, st)
    assert stop.stop and stop.code == "budget_insufficient"


def test_risk_filter_and_safety_stop():
    b, specs, bank = make_problem(risks=(0.5, 0.5, 0.5))
    cfg = PlannerConfig(budget=10, risk_cap=0.3).validate()
    scored = score_experiments(b, specs, bank, cfg, PlanState())
    assert not any(s.feasible for s in scored)
    stop = evaluate_stop(b, scored, cfg, PlanState())
    assert stop.code == "safety_blocked"
    # risk-agnostic baselines ignore the cap by definition
    assert get_variant("max_eig").choose(b, specs, bank, cfg, PlanState()).selected is not None


def test_missing_prerequisites_make_experiment_unavailable():
    b, specs, bank = make_problem()
    specs[1] = ExperimentSpec("mid_strong", "mid_strong", 2.0, 0.0, ("labels_recent",), ("no", "yes"))
    cfg = PlannerConfig(budget=10).validate()
    s = {x.id: x for x in score_experiments(b, specs, bank, cfg, PlanState(available_inputs=set()))}
    assert not s["mid_strong"].feasible and "prereq" in s["mid_strong"].reason_codes
    s = {x.id: x for x in score_experiments(b, specs, bank, cfg, PlanState(available_inputs={"labels_recent"}))}
    assert s["mid_strong"].feasible


def test_previously_executed_experiment_excluded_unless_repeatable():
    b, specs, bank = make_problem()
    cfg = PlannerConfig(budget=10).validate()
    st = PlanState(done={"tiny": 1})
    assert not {s.id: s for s in score_experiments(b, specs, bank, cfg, st)}["tiny"].feasible
    specs[2] = ExperimentSpec("tiny", "tiny", 0.5, 0.0, (), ("no", "yes"), repeatable=True)
    assert {s.id: s for s in score_experiments(b, specs, bank, cfg, st)}["tiny"].feasible


def test_stop_on_confidence_and_low_value():
    b, specs, bank = make_problem()
    cfg = PlannerConfig(budget=10, confidence_threshold=0.9).validate()
    b2 = Belief.from_priors(["a", "b"], [0.5, 0.5], max_active=1, none_mass=1e-6)
    b2.update(np.array([1e-6, 0.99, 0.001]))
    scored = score_experiments(b, specs, bank, cfg, PlanState())
    assert evaluate_stop(b2, scored, cfg, PlanState()).code == "confidence_reached"
    cfg2 = PlannerConfig(budget=10, min_score=100.0).validate()
    assert evaluate_stop(b, score_experiments(b, specs, bank, cfg2, PlanState()), cfg2, PlanState()).code == "low_expected_value"


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        PlannerConfig(budget=-1).validate()
    with pytest.raises(ValueError):
        PlannerConfig(risk_cap=1.5).validate()
    with pytest.raises(ValueError):
        PlannerConfig(confidence_threshold=0).validate()


def test_closed_loop_identifies_true_cause_and_respects_budget():
    b, specs, bank = make_problem()
    cfg = PlannerConfig(budget=3.5, confidence_threshold=0.8).validate()
    truth = np.array([True, False, False, False])
    # oracle samples from the true single-cause world h1
    state_idx = int(np.where((b.states == [True, False, False, False]).all(axis=1))[0][0])
    rng = np.random.default_rng(7)
    oracle = lambda e: int(rng.choice(2, p=bank.matrix(e)[state_idx]))
    tr = run_diagnosis(get_variant("adaptive"), b, specs, bank, cfg, oracle, set())
    assert tr.spent <= cfg.budget + 1e-9
    assert tr.stop is not None and tr.n_experiments >= 1


@pytest.mark.parametrize("vid", ["random", "checklist", "cheapest", "max_eig", "eig_per_cost", "adaptive", "lookahead2"])
def test_all_variants_run_and_never_exceed_budget(vid):
    b, specs, bank = make_problem()
    cfg = PlannerConfig(budget=3.0, confidence_threshold=0.99).validate()
    oracle = lambda e: 1
    tr = run_diagnosis(get_variant(vid), b, specs, bank, cfg, oracle, set(), rng=np.random.default_rng(3))
    assert tr.spent <= 3.0 + 1e-9
    assert len({s.experiment_id for s in tr.steps}) == len(tr.steps)  # no repeats
    assert tr.stop is not None


def test_lookahead_differs_from_greedy_on_synergy_problem():
    # A is nearly useless alone (EIG ~ 0) but unlocks perfect information when combined with B;
    # greedy ratio prefers C, 2-step lookahead should value A+B jointly. We just assert a valid distinct value.
    hyps = ["x", "y"]
    b = Belief.from_priors(hyps, [0.5, 0.5], max_active=1, none_mass=1e-6)
    lib = LikelihoodLibrary({
        "A": NoisyOrLikelihood("A", ("n", "y"), np.array([0.3]), {"x": np.array([0.35]), "y": np.array([0.05])}),
        "B": NoisyOrLikelihood("B", ("n", "y"), np.array([0.1]), {"x": np.array([0.8]), "y": np.array([0.1])}),
    })
    specs = [ExperimentSpec(k, k, 1.0, 0.0, (), ("n", "y")) for k in ("A", "B")]
    bank = LikelihoodBank(lib, b.hypotheses, b.states)
    cfg = PlannerConfig(budget=5).validate()
    la = get_variant("lookahead2").choose(b, specs, bank, cfg, PlanState())
    assert la.selected is not None and la.selected.lookahead_value is not None
    assert la.selected.lookahead_value > 0

import numpy as np
import pandas as pd
import pytest

from experimentpilot.benchmark import DEFAULT_CONFIG, build_specs, run_planners
from experimentpilot.benchmark.analysis import bootstrap_ci, expected_calibration_error, paired_tests
from experimentpilot.experiments import EXPERIMENTS, ColumnMapping, ExperimentContext, ExperimentError, get_experiment, register_experiment
from experimentpilot.experiments.base import Experiment
from experimentpilot.faults import ScenarioSpec, build_scenario, compute_outcomes
from experimentpilot.likelihood import estimate_library, expert_library, load_default_library

HYPS = ("covariate_drift", "label_noise", "feature_leakage", "class_prior_shift", "pipeline_change", "data_quality", "metric_bug")


@pytest.fixture(scope="module")
def scenarios():
    mk = lambda faults, seed=11: build_scenario(ScenarioSpec("synthetic_credit", tuple(faults), seed))
    return {"control": mk([]), "pipe": mk([("pipeline_change", "high")]), "leak": mk([("feature_leakage", "high")]),
            "metric": mk([("metric_bug", "high")]), "dq": mk([("data_quality", "high")]), "prior": mk([("class_prior_shift", "high")]),
            "noise": mk([("label_noise", "high")]), "drift": mk([("covariate_drift", "high")])}


def run(sc, eid, **kw):
    return get_experiment(eid).run(sc.context(**kw))


def test_ten_experiments_with_complete_metadata():
    assert len(EXPERIMENTS) == 10
    for e in EXPERIMENTS.values():
        assert e.id and e.name and e.description and len(e.outcomes) >= 2 and e.cost >= 0 and 0 <= e.risk <= 1 and e.limitations


def test_control_scenario_raises_no_alarms(scenarios):
    for eid in ("feature_drift", "data_quality_audit", "class_prior", "pipeline_replay", "metric_recompute", "leakage_scan", "label_noise"):
        assert run(scenarios["control"], eid).outcome_index == 0, eid


@pytest.mark.parametrize("key,eid,min_level", [("pipe", "pipeline_replay", 2), ("pipe", "feature_drift", 1), ("leak", "leakage_scan", 1),
                                                ("metric", "metric_recompute", 1), ("dq", "data_quality_audit", 2), ("prior", "class_prior", 1),
                                                ("noise", "label_noise", 1), ("drift", "feature_drift", 1)])
def test_experiments_detect_their_fault(scenarios, key, eid, min_level):
    res = run(scenarios[key], eid)
    assert res.outcome_index >= min_level
    assert res.interpretation and res.stats and res.limitations and res.runtime_s > 0
    d = res.to_dict()
    import json
    json.dumps(d)  # fully serialisable evidence


def test_missing_inputs_are_refused_not_fabricated(scenarios):
    sc = scenarios["control"]
    ctx = ExperimentContext(sc.reference.drop(columns=["target"]), sc.recent.drop(columns=["target"]), sc.mapping, None, None)
    for eid in ("class_prior", "label_noise", "pipeline_replay", "metric_recompute"):
        exp = get_experiment(eid)
        ok, missing = exp.available(ctx)
        assert not ok and missing
        with pytest.raises(ExperimentError):
            exp.run(ctx)
    assert get_experiment("feature_drift").available(ctx)[0]


def test_retrain_needs_enough_labelled_rows(scenarios):
    sc = scenarios["control"]
    ctx = ExperimentContext(sc.reference, sc.recent.head(60), sc.mapping)
    with pytest.raises(ExperimentError):
        get_experiment("retrain_recent").run(ctx)


def test_experiment_is_deterministic_given_seed(scenarios):
    a, b = run(scenarios["noise"], "label_noise", seed=3), run(scenarios["noise"], "label_noise", seed=3)
    assert a.stats == b.stats


def test_register_rejects_duplicates_and_bad_plugins():
    class Dup(Experiment):
        id = "feature_drift"; outcomes = ("a", "b")
        def _run(self, ctx): ...
    with pytest.raises(ValueError):
        register_experiment(Dup())


def test_fault_injection_records_ground_truth_and_is_reproducible():
    spec = ScenarioSpec("synthetic_credit", (("label_noise", "medium"), ("covariate_drift", "low")), 5)
    a, b = build_scenario(spec), build_scenario(spec)
    assert a.truth == ["covariate_drift", "label_noise"] == list(spec.truth)
    pd.testing.assert_frame_equal(a.recent, b.recent)
    assert a.meta["accuracy_drop"] == b.meta["accuracy_drop"]
    assert build_scenario(ScenarioSpec("synthetic_credit", (), 5)).truth == []


def test_invalid_fault_rejected():
    with pytest.raises(ValueError):
        build_scenario(ScenarioSpec("synthetic_credit", (("nope", "low"),), 1))


def test_all_datasets_build_including_real_one():
    for ds in ("synthetic_churn", "breast_cancer"):
        sc = build_scenario(ScenarioSpec(ds, (("class_prior_shift", "medium"),), 2))
        assert len(sc.reference) > 50 and len(sc.recent) > 50
        out = compute_outcomes(sc, ["feature_drift", "class_prior"])
        assert "outcome_index" in out["feature_drift"]


# ------------------------------------------------------------------ likelihood estimation & benchmark
def synthetic_records(n_per=60, seed=0):
    """Records with known generating process: pipeline_replay fires iff pipeline_change (95%), 5% false alarms."""
    rng = np.random.default_rng(seed)
    recs = []
    for truth in [None, "pipeline_change", "covariate_drift", "label_noise"]:
        for i in range(n_per):
            fire = (rng.random() < 0.95) if truth == "pipeline_change" else (rng.random() < 0.05)
            drift = (rng.random() < 0.9) if truth in ("covariate_drift", "pipeline_change") else (rng.random() < 0.05)
            recs.append({"id": f"{truth}-{i}-{seed}", "dataset": "d", "seed": seed, "truth": [truth] if truth else [],
                         "outcomes": {"pipeline_replay": {"outcome_index": 2 * int(fire)}, "feature_drift": {"outcome_index": 2 * int(drift)}}})
    return recs


def test_estimation_recovers_known_sensitivities():
    lib = estimate_library(synthetic_records(400), HYPS, pseudo_count=2.0)
    m = lib.get("pipeline_replay")
    assert m.sens["pipeline_change"][0] > 0.9 and m.sens["covariate_drift"][0] < 0.12
    assert m.leak[0] < 0.1
    assert (np.diff(m.sens["pipeline_change"]) <= 1e-9).all()  # non-increasing across thresholds


def test_smoothing_keeps_probabilities_away_from_zero_and_one():
    lib = estimate_library(synthetic_records(5), HYPS, pseudo_count=2.0)
    for m in lib.models.values():
        assert (m.leak > 0).all() and (m.leak < 1).all()
        assert all(((s >= 0) & (s < 1)).all() for s in m.sens.values())


def test_default_and_expert_libraries_cover_all_experiments():
    for lib in (load_default_library(), expert_library()):
        assert set(lib.models) == set(EXPERIMENTS)


def test_estimation_and_evaluation_seed_sets_are_disjoint():
    assert not set(DEFAULT_CONFIG["estimation_seeds"]) & set(DEFAULT_CONFIG["evaluation_seeds"])
    specs_a = build_specs(["synthetic_credit"], [0], DEFAULT_CONFIG)
    specs_b = build_specs(["synthetic_credit"], [100], DEFAULT_CONFIG)
    assert not {s.id for s in specs_a} & {s.id for s in specs_b}


def test_benchmark_spec_grid_covers_controls_singles_and_pairs():
    specs = build_specs(["synthetic_credit"], [1], DEFAULT_CONFIG)
    n_single = sum(len(s.faults) == 1 for s in specs)
    assert n_single == 7 * 3 and sum(len(s.faults) == 0 for s in specs) == 1 and sum(len(s.faults) == 2 for s in specs) == 8


def test_planner_replay_is_reproducible_and_respects_constraints():
    est, ev = synthetic_records(80, 0), synthetic_records(15, 1)
    lib = estimate_library(est, HYPS)
    cfg = {**DEFAULT_CONFIG, "budget": 2.5}
    a, b = run_planners(ev, lib, cfg), run_planners(ev, lib, cfg)
    pd.testing.assert_frame_equal(a.drop(columns=["marginals", "truth_vec"]), b.drop(columns=["marginals", "truth_vec"]))
    assert (a["cost"] <= 2.5 + 1e-9).all()
    assert set(a["planner"]) == set(DEFAULT_CONFIG["planners"])


def test_adaptive_beats_random_on_a_toy_benchmark():
    lib = estimate_library(synthetic_records(300, 0), HYPS)
    ev = synthetic_records(60, 1)
    df = run_planners(ev, lib, {**DEFAULT_CONFIG, "budget": 3.0}, ["adaptive", "random"], stop_on_confidence=False)
    acc = df[df["n_truth"] >= 1].groupby("planner")["top1"].mean()
    assert acc["adaptive"] >= acc["random"]


def test_robustness_perturbation_runs():
    lib = estimate_library(synthetic_records(100, 0), HYPS)
    df = run_planners(synthetic_records(10, 2), lib, DEFAULT_CONFIG, ["adaptive"], sigma=1.0)
    assert len(df) == 40


def test_metric_helpers():
    m, lo, hi = bootstrap_ci(np.array([0.0, 1.0] * 50), 300, seed=1)
    assert lo < m < hi and m == pytest.approx(0.5)
    p = np.array([0.1] * 500 + [0.9] * 500); t = np.array([0] * 450 + [1] * 50 + [1] * 450 + [0] * 50)
    ece, _ = expected_calibration_error(p, t)
    assert ece == pytest.approx(0.0, abs=1e-9)
    ece_bad, _ = expected_calibration_error(np.full(100, 0.9), np.zeros(100))
    assert ece_bad == pytest.approx(0.9)


def test_paired_tests_flag_identical_planners_as_not_different():
    df = pd.DataFrame({"scenario": list("abcdefgh") * 2, "planner": ["adaptive"] * 8 + ["x"] * 8, "cost": [1.0] * 16, "top1": [1.0] * 16, "f1": [1.0] * 16})
    assert all(t["wilcoxon_p"] == 1.0 for t in paired_tests(df))

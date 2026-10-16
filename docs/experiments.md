# Experiment plugins

Each experiment subclasses `experimentpilot.experiments.base.Experiment` and declares: `id`, `name`, `description`, `cost`, `risk`, `requires` (input keys), `outcomes` (ordered levels), `repeatable`, `requires_approval`, `limitations`, and `_run(ctx) -> ExperimentResult`.

`ExperimentResult` carries the discretised `outcome`, the underlying numeric `score`, raw `stats`, tabular evidence `tables`, `findings`, a plain-language `interpretation`, `limitations` and `runtime_s`. Missing inputs raise `ExperimentError` (the experiment is *unavailable*; evidence is never fabricated). Unexpected exceptions are wrapped as `ExperimentError("<id> failed: …")`.

Input keys: `reference`, `recent`, `features`, `labels_reference`, `labels_recent`, `predictions_reference`, `predictions_recent`, `replay`, `reported_metrics`.

| id | What it computes | Cost | Risk | Requires | Outcomes |
|---|---|---|---|---|---|
| `feature_drift` | PSI, KS / χ² (Holm-adjusted), effect sizes per feature; drift share | 1.0 | 0 | features | no_shift / moderate_shift / severe_shift |
| `data_quality_audit` | null-rate change (z-test), dtype, out-of-range share, repeated sentinel values, constant columns, unseen categories, schema differences | 1.0 | 0 | features | clean / minor_issues / serious_issues |
| `class_prior` | label (and prediction) distribution χ², total variation; re-scoring under reference priors | 0.5 | 0 | labels (both) | stable / moderate_shift / severe_shift |
| `label_noise` | confident-learning-style suspicious-label rate from out-of-fold probabilities, reference vs recent | 3.0 | 0 | labels, features | no_excess / some / high_excess_noise |
| `leakage_scan` | single-feature AUC strength on reference vs recent; retention gap; optional temporal-split check | 2.0 | 0 | labels, features | no_leak_signature / possible_leak / likely_leak |
| `pipeline_replay` | row-aligned (or distributional) comparison of current vs previous-pipeline features | 2.0 | 0.10 | replay, features | identical / minor_differences / major_differences |
| `importance_weighted_eval` | domain-classifier density ratios; does reweighting the reference close the accuracy gap? | 2.0 | 0 | labels, predictions, features | gap_not / partly / mostly_explained |
| `slice_analysis` | per-slice accuracy drop vs overall drop, Holm-adjusted | 2.0 | 0 | labels, predictions, features | uniform_drop / localized_drop / severely_localized_drop |
| `retrain_recent` | surrogate trained on recent labelled sample vs surrogate trained on reference, on held-out recent rows | 5.0 | 0.20 | labels, features; **explicit approval** | no_recovery / partial_recovery / strong_recovery |
| `metric_recompute` | recompute accuracy / macro-F1 from predictions and labels vs reported values | 0.25 | 0 | labels, predictions, reported metrics | matches_report / minor_mismatch / major_mismatch |

Thresholds that turn statistics into levels are fixed in each experiment's code and are intentionally generic (not tuned per dataset).

## Writing a new experiment

```python
from experimentpilot.experiments import Experiment, register_experiment

class MyCheck(Experiment):
    id = "my_check"; name = "My check"; description = "…"
    cost, risk = 1.0, 0.0
    requires = ("features",)
    outcomes = ("ok", "suspicious", "bad")
    limitations = ("What this cannot tell you.",)

    def _run(self, ctx):
        score = ...                       # compute from ctx.reference / ctx.recent
        return self.result(score, level=int(score > 0.3), stats={"score": score},
                           interpretation="…")

register_experiment(MyCheck())
```

Then add its likelihood parameters: re-run `research/benchmark/run_benchmark.py --install-library` after adding a fault injector that exercises it, or provide an expert prior in `EXPERT_PRIORS` for a cold start. The planner never needs to change.

## Adding a hypothesis

`register_hypothesis(Hypothesis(id=…, code=…, name=…, …))`; it becomes selectable in the setup page. Until some likelihood model carries a sensitivity for it, no experiment will move its probability.

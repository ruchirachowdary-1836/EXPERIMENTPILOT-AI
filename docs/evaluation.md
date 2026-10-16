# Evaluation protocol (pre-registered)

All benchmark results are **simulated**: faults are injected into built-in datasets and every experiment is actually computed on the resulting data, but the faults, datasets and models are synthetic or small. Results show behaviour under this protocol, not real-world gains.

## Protocol

1. **Scenarios.** For each dataset × seed: one no-fault control; each of 7 injectors (covariate drift, label noise, feature leakage, class-prior shift, preprocessing change, data-quality problems, metric bug) at 3 severities; 8 simultaneous-fault pairs at medium severity.
2. **Measured outcomes.** All 10 experiments are run on every scenario (`outcomes_*.jsonl`, cached). Planners later *reveal* these measured outcomes, so every planner faces identical evidence (paired design).
3. **Estimation / evaluation split.** Likelihood tables are estimated only from *estimation* seeds (`0–11`, datasets `synthetic_credit`, `synthetic_churn`; single-fault and control runs). Planners are evaluated on disjoint *evaluation* seeds (`100–107`) on those two datasets **and** on the real `breast_cancer` dataset, which was never used for estimation. The code asserts that scenario ids do not overlap, and the web app refuses laboratory seeds that overlap the estimation seeds.
4. **Planner settings.** budget 8, risk cap 0.15, λ = 1, τ = 0.9 (joint), ε = 0.01, `max_active` = 3. Baselines `max_eig` and `eig_per_cost` are risk-agnostic by definition and therefore may violate the cap.
5. **Fixed-budget curves.** Each planner is run with budgets `[1,2,3,4,6,8,12,18]` without early stopping.
6. **Robustness.** Planner beliefs use likelihood tables perturbed by Gaussian noise (σ ∈ {0, 0.5, 1, 1.5}) in logit space; outcomes are unchanged.

## Metrics

| Metric | Definition |
|---|---|
| Top-1 / Top-3 | highest- (resp. any of top-3) ranked cause is in the true cause set (faulty scenarios only) |
| Multi-cause precision/recall/F1 | diagnosed set (marginal ≥ 0.5) vs true set; control scenarios score 1 only if nothing is diagnosed |
| Exact set | diagnosed set equals the true set |
| Cost / # experiments | total cost executed until a stopping rule fires |
| Accuracy at fixed budget | top-1 as a function of budget |
| Brier | mean squared error of all (scenario, hypothesis) marginals vs 0/1 truth |
| ECE | 10-bin expected calibration error of the same marginals |
| Risk-cap violations | executed experiments with risk above the cap |
| Robustness | F1 under perturbed likelihoods |
| Agreement | share of scenarios in which two planners ran the identical experiment sequence |

Means are reported with 95% percentile-bootstrap confidence intervals (1000 resamples). Planners are compared with the paired Wilcoxon signed-rank test on per-scenario differences (adaptive vs each baseline). Scenarios from the same seed share a dataset draw, so p-values describe this simulation only.

## Reproduce

```bash
python research/benchmark/run_benchmark.py --config research/configs/default.json --workers 6
python research/analysis/summarize.py research/results/results_default-v1.json
```

Outputs in `research/results/`: `outcomes_*.jsonl`, `likelihood_default-v1.json`, `runs_default-v1.csv`, `results_default-v1.json`, `plots/*.png`. Seeds are fixed in the config; planner randomness is seeded with `crc32(planner id)`.

## Known threats to validity

* Fault injectors and experiments were designed by the same people: the experiments are likely tuned to detect exactly these faults.
* Experiment discretisation thresholds were set by inspecting estimation-seed scenarios during development.
* Synthetic datasets are easier than real production data; one small real dataset is included for the held-out check.
* No LLM-agent baseline, no real-world case study and no ablations beyond the likelihood-perturbation study are included yet (see `BUILD_STATUS.md`).

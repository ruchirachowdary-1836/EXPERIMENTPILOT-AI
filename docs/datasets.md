# Dataset provenance

No dataset is downloaded at runtime; everything works offline.

| Name | Source | Nature | Sizes (train / reference / recent pool) | Used for |
|---|---|---|---|---|
| `synthetic_credit` | `engine/experimentpilot/faults/datasets.py::_credit` | **Synthetic.** 7 numeric + 2 categorical features, logistic target with noise, weakly-informative `aux_signal` column, logistic-regression model | 2400 / 1000 / 3000 → 1000 recent | estimation + evaluation |
| `synthetic_churn` | `…::_churn` | **Synthetic.** Non-linear target (sine term, interactions), gradient-boosting model | 2400 / 1000 / 3000 → 1000 | estimation + evaluation |
| `breast_cancer` | `sklearn.datasets.load_breast_cancer` (UCI Wisconsin Diagnostic, bundled with scikit-learn) | **Real, small** (569 rows, 30 numeric features) plus the same synthetic `aux_signal` column | 269 / 150 / 150 | evaluation only (held out from likelihood estimation) |

Every generator is seeded (`np.random.default_rng([seed, dataset_index])`) so scenarios are reproducible.

## What the injectors do

| Injector | Effect (severity low → high) |
|---|---|
| covariate drift | ~40% of the most target-correlated numeric features shifted by 0.5σ → 1.8σ (mostly towards the minority class) and scaled 1.0× → 1.3× in recent data |
| label noise | 7% → 28% of recent labels flipped |
| feature leakage | `aux_signal` made label-dependent in the training and reference data (separation 1.2 → 3.5σ), uninformative in recent data, model trained on the leaky data |
| class-prior shift | recent data resampled so the minority-class odds change ×1.7 → ×5 (or ×0.6 → ×0.2 when positives dominate) |
| preprocessing change | ~25% of numeric features affinely transformed in the *current* pipeline output only (scale 1.1 → 2.5, offset 0.1σ → 0.8σ); categorical levels swapped at medium/high; the replay dataset keeps the old pipeline output |
| data quality | nulls in ~30% of features (5% → 35%), a −999 sentinel in a predictive feature at medium/high, a stale constant column at high; present in both recent and replay |
| metric bug | the *reported* recent accuracy is computed with 10% → 50% of labels misaligned |
| none | clean control |

Pairs of faults are composed in sequence. A model is trained on the (possibly leaky) training split, then predictions are produced for reference and recent data. Ground truth is recorded in each scenario.

## Using your own data

Upload CSVs in the app. Nothing about uploaded data is used to build likelihood tables; the shipped tables come only from the simulated benchmark above.

# Methodology

ExperimentPilot AI treats the diagnosis of a degraded ML model as **budget-constrained sequential Bayesian experimental design** over a library of executable diagnostic experiments.

> Adaptive/sequential diagnosis is an established research area (see `related_work.md`). This project does **not** claim that the formulation is novel. The contribution it explores is the specific combination of cost, risk, budget, a calibrated multi-cause posterior and validation on a fault-injection benchmark.

## 1. State and belief

There are `K` candidate root causes `h_1 … h_K`. Several can hold simultaneously, so the latent state is a binary vector `z ∈ {0,1}^K`.

* **Exact enumeration** of all `2^K` joint states when practical.
* **Documented approximation for larger spaces:** the support is truncated to states with at most `m` active causes (`max_active`, default 3). The number of states is `Σ_{j≤m} C(K,j)`, so K = 20 with m = 2 has 211 states. Setting `m = K` recovers exact inference. This encodes a sparsity assumption (few simultaneous faults).
* **Prior:** independent Bernoulli(`p_k`) per cause, restricted to the support and renormalised. An optional `none_mass` overrides the probability of the all-inactive "no fault" state.

Implementation: `engine/experimentpilot/core/belief.py`.

## 2. Outcome (likelihood) model

Experiment `e` has ordered outcome levels `0 … L-1` (for example `no_shift < moderate_shift < severe_shift`). The model is a **cumulative noisy-OR**:

```
P(o ≥ j | z) = 1 − (1 − leak_{e,j}) · Π_{k: z_k=1} (1 − sens_{e,k,j}),   j = 1 … L−1
P(o = j | z) = P(o ≥ j | z) − P(o ≥ j+1 | z)
```

* `leak` – false-alarm rate when nothing is active.
* `sens_{k,j}` – probability that cause `k` alone pushes the outcome to at least level `j`.
* `P(o ≥ j)` is forced to be non-increasing in `j`; probabilities are floored at 1e-6 and renormalised.

Two simultaneous causes therefore raise the alarm probability without exact joint tables for every pair. Likelihoods depend on **effect size level**, not just a binary flag.

### Estimation

`engine/experimentpilot/likelihood/estimate.py` estimates `leak` and `sens` from fault-injection outcomes (single-fault and no-fault control runs only):

```
leak_j     = (#{o ≥ j | control} + a·p0) / (n_control + a)
q_{k,j}    = (#{o ≥ j | only k}  + a·p0) / (n_k + a)
sens_{k,j} = 1 − (1 − q_{k,j}) / (1 − leak_j)
```

`a` is a Beta pseudo-count (default 2) and `p0` a conservative **expert prior** (`EXPERT_PRIORS`). This is the cold-start path described in the blueprint: expert tables, updated with simulation counts.

Two-fault runs are *not* used for fitting. They validate the noisy-OR assumption: the benchmark reports the mean log-likelihood of observed two-fault outcomes under noisy-OR versus a naive single-cause mixture.

## 3. Bayesian update

```
b'(z) ∝ b(z) · P(o | e, z)
```

computed in log space with max-subtraction for numerical stability. Evidence that has zero likelihood under every state with prior support raises an error rather than silently producing NaN. Missing evidence leaves the posterior unchanged (recorded in the history). Uncertain outcomes can be supplied as a weight vector (**virtual/soft evidence**): the likelihood column is `Σ_o w_o P(o|z)`.

## 4. Expected information gain

```
EIG(e) = H(b) − Σ_o P(o | b, e) · H(b | o, e)        (bits; equals the mutual information I(Z; O))
P(o | b, e) = Σ_z b(z) P(o | e, z)
```

Unit tests check it against analytically known cases: a deterministic test (`EIG = H(prior)`), an uninformative test (`0`), a binary symmetric channel (`1 − H₂(ε)`), and the explicit mutual-information formula on random tables.

## 5. Selection rule and constraints

```
score(e) = EIG(e) / max(c(e) + λ · r(e), floor)
```

Feasible experiments must satisfy: not skipped, not already executed (unless `repeatable`), all required inputs available, `c(e) ≤ remaining budget`, `r(e) ≤ risk cap`. The `floor` (1e-3) makes zero-cost experiments safe. Experiments with `requires_approval` or risk above the warning level (0.15) need an explicit approval flag at execution time.

**Separation score.** For the two leading hypotheses `a`, `b`:
`TV( P(o | a active), P(o | b active) )`, where each conditional marginalises the other causes under the current belief. It is reported with every recommendation and can be added to the score with weight `separation_weight`.

## 6. Planner variants

| id | Idea | Enforces risk cap |
|---|---|---|
| `random` | uniform among feasible | yes |
| `checklist` | static, hand-ordered sequence (current practice) | yes |
| `cheapest` | lowest cost first | yes |
| `max_eig` | maximise EIG, ignore cost and risk | no (risk-agnostic by definition) |
| `eig_per_cost` | maximise EIG / cost, ignore risk | no |
| `adaptive` | EIG / (cost + λ·risk) (+ separation bonus) | yes |
| `lookahead2` | bounded two-step lookahead (below) | yes |

**Two-step lookahead.** For each feasible first experiment `e`, for every outcome `o` the best greedy follow-up `e'_o` (by single-step score on the updated belief) is found. Value `= (EIG(e) + Σ_o P(o) EIG_o(e'_o)) / (cost(e) + Σ_o P(o) cost(e'_o))`. It differs from greedy when a pair of tests is worth more than either alone. The benchmark reports an **agreement matrix** (how often two planners executed identical sequences) so that variants that behave identically are visible rather than hidden.

Not implemented (documented in the blueprint as optional): knapsack/budget-aware re-planning, Thompson/sampled-hypothesis selection, an LLM-only agent baseline.

## 7. Stopping policy

Evaluated in this order: confidence reached → maximum experiments → no feasible experiment (reported as `budget_insufficient`, `safety_blocked` or `no_feasible_experiments` depending on what blocks the remaining ones) → best score below ε (`low_expected_value`).

* `confidence_mode = joint`: stop when `max_z b(z) ≥ τ`, the probability that the most likely **exact combination** of causes is right.
* `confidence_mode = marginal`: stop when every hypothesis is decided (`≥ τ` or `≤ 1−τ`).
* optional top-two odds-ratio rule.

The final report never forces a single root cause. A cause is *diagnosed* when its marginal probability is ≥ 0.5; residual uncertainty is reported in bits and as the set of undecided hypotheses.

## 8. Evidence direction

For each hypothesis and run the report shows the likelihood ratio `P(o | h active) / P(o | h inactive)` under the prior belief. Ratios ≥ 1.5 are labelled *supporting*, ≤ 1/1.5 *contradicting*.

## 9. Assumptions and limitations

* Conditional independence of experiment outcomes given the active causes. Correlated outcomes can make the posterior overconfident; the robustness ablation perturbs likelihoods to probe sensitivity.
* Likelihood tables are estimated from **simulated** fault injection on built-in datasets. Real fault signatures can differ.
* Sparse-support truncation for large hypothesis sets.
* Each experiment is a statistical indicator, never proof of cause. See each experiment's `limitations`.
* Model/code regression is an extensible hypothesis with no discriminating experiment yet.

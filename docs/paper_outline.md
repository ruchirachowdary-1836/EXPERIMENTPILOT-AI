# Paper outline (draft; no results claimed)

Working title idea: *Which Test Next? Budget-Constrained Adaptive Diagnosis of Machine-Learning Failures.*

1. **Introduction** – manual, habit-driven debugging of degraded models; cost and risk of checks; gap between detection tools and experiment selection. State the limited claim clearly.
2. **Related work** – fill from `related_work.md` after the literature search. Do not claim first-of-kind.
3. **Problem formulation** – latent cause vector, experiments with cost/risk, budget, risk cap, stopping (`methodology.md` §1–7).
4. **Method** – noisy-OR outcome model, EIG, score, separation, planner variants, stopping policy.
5. **Benchmark** – injectors, datasets, estimation/evaluation split, metrics (`evaluation.md`).
6. **Experiments** – results from `research/results/` with confidence intervals and paired tests; fixed-budget curves; calibration; robustness.
7. **Ablations and limitations** – remove cost term, remove risk term, myopic vs lookahead, exact vs truncated belief, mis-specified likelihoods, uniform vs symptom-informed prior (**not yet run**: list as future work unless added).
8. **Conclusion** – what was and was not shown.

## Claims to avoid

"first adaptive diagnosis", "solves root-cause analysis", any real-world gain without real-world data, any statement about acceptance or publication.

## Venue types to consider (check current deadlines yourself)

ML-systems and data-centric-AI workshops first; later applied-ML or datasets-and-benchmarks tracks.

## Open work before a submission

- Real or semi-real case study with a documented cause.
- LLM-only agent baseline.
- Online refinement of likelihood tables from confirmed diagnoses.
- Dependence-aware likelihoods (pairwise corrections).

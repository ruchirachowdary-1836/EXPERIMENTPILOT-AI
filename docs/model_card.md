# Model card: the ExperimentPilot diagnostic system

**What it is.** A Bayesian experiment-selection and evidence-combination system. It is not a learned neural model; its parameters are noisy-OR likelihood tables (`engine/experimentpilot/likelihood/default_tables.json`).

**Intended use.** Support an engineer who is investigating why a tabular classification model degraded, by proposing which diagnostic experiment to run next and combining the results into a ranked, uncertainty-aware explanation. Research and exploration; the human remains responsible for conclusions.

**Out of scope.** Automated remediation, regression/NLP/vision models, streaming monitoring, safety-critical decisions made without human review, causal proof.

**Inputs.** Reference and recent CSV data with target and prediction columns, optional replay data and reported metrics, a budget, a risk cap and priors.

**Outputs.** Ranked hypotheses with posterior probabilities, evidence with supporting/contradicting direction, decision tree, cost and risk accounting, limitations, PDF/JSON/CSV exports.

**Training data for the tables.** Simulated fault injection: `synthetic_credit` and `synthetic_churn`, seeds 0–11, single-fault and control scenarios (see `docs/datasets.md`). Provenance is stored in the table file and displayed in the app (Settings).

**Evaluation.** See `docs/evaluation.md` and `research/results/`. Simulated only.

**Known failure modes.**
* Outcomes that are correlated given the cause make the posterior overconfident.
* Faults with signatures unlike the injected ones (for example subtle concept drift) are not represented; the planner can confidently choose among the wrong candidates.
* The label audit is confounded by leakage (a leaky reference set looks unusually clean) and by genuinely hard examples.
* Importance-weighted re-evaluation also "explains" pipeline changes and data-quality problems because they shift inputs.
* Small samples give noisy statistics; several experiments refuse to run below minimum sizes.
* Model/code regression has no discriminating experiment.

**Ethical and privacy considerations.** Runs locally; uploaded data are copied into the local data directory, never sent elsewhere, and can be deleted per project. No personal data are needed, but if your data contain personal data you are responsible for handling them lawfully.

**Caveats for interpretation.** Probabilities are conditional on the likelihood tables being right. Treat them as a principled ranking of evidence, not as calibrated truth for your system.

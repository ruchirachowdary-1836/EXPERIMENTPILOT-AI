# Related-work tracker (template)

No citations are pre-filled. Every entry below must be added from papers you have actually read. The blueprint warns that a closely related June 2026 paper on root-cause analysis and counterfactual repair for ML pipelines may exist; read it first and record the exact delta.

## Closest prior work (fill in; keep the ten closest)

| # | Citation (authors, title, venue, year, URL/DOI you verified) | Problem | Method | What it does **not** do vs. this project | Read? |
|---|---|---|---|---|---|
| 1 | | | | | ☐ |

## Areas to search (from the blueprint)

- [ ] Bayesian experimental design / active testing / expected information gain
- [ ] Sequential diagnosis, test sequencing, decision-theoretic troubleshooting, cost-sensitive diagnosis
- [ ] Spectrum-based fault localisation, delta debugging, adaptive test selection
- [ ] Data validation and drift tooling (Evidently, alibi-detect, TFX Data Validation, Great Expectations)
- [ ] Data-centric debugging (confident learning, data valuation, slice discovery)
- [ ] ML pipeline debugging / root-cause analysis, provenance-based debugging
- [ ] AutoML / AIOps incident root-cause analysis, LLM agents for ML debugging
- [ ] Fault-injection benchmarks for ML pipelines

## Search log

| Date | Database / query | Result | Notes |
|---|---|---|---|

## Positioning statement to test

"We cast ML failure diagnosis as budget-constrained sequential Bayesian experimental design over a library of executable diagnostic probes with explicit risk, and validate it on a controlled fault-injection benchmark."

If the exact framing already exists, pivot the contribution to the benchmark, the multi-cause posterior, or the risk-aware safety layer.

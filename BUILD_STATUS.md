# BUILD_STATUS

Last updated after the full local build. Everything below was actually run unless marked **not verified**.

## Test outcomes

| Suite | Command | Result |
|---|---|---|
| Engine (math core, experiments, injectors, estimation, benchmark reproducibility) | `cd engine && pytest -q` | **67 passed** |
| Backend (API, upload security, state machine, concurrency, reports, benchmark jobs) | `cd backend && pytest -q` | **32 passed** |
| Frontend unit (validation, API errors, controls, tree, charts, downloads) | `cd frontend && npm test` | **21 passed** |
| Browser end-to-end (Playwright/Chromium): create project → upload → map → start → run 2 experiments → stop → download PDF | `cd frontend && npm run e2e` | **1 passed** |
| Type check / lint | `npx tsc --noEmit`, `ruff --select E9,F63,F7,F82` | clean |
| Docker / docker compose | – | **not verified** (no Docker in the dev environment) |
| GitHub Actions CI | `.github/workflows/ci.yml` | **not verified** (not yet run on GitHub) |

A live end-to-end scenario was also driven through the real UI and API: a simulated high-severity pipeline-change fault, 7 experiments, decision tree, evidence, report and PDF. The planner ranked "Preprocessing / train-serving pipeline change" first.

## Implemented

- Bayesian belief engine over joint cause states (exact, or truncated to ≤ m simultaneous causes), noisy-OR likelihoods, soft/missing evidence, history, EIG, separation score.
- 7 planner variants, stopping policy (confidence, no feasible experiment, budget, low value, safety/risk cap), rationale and rejection reasons.
- 10 real experiments with plugin interface, 8 fault injectors (incl. none and pairs), 3 built-in datasets, likelihood estimation with Beta smoothing, reproducible benchmark with bootstrap CIs, Wilcoxon tests, calibration, fixed-budget curves, robustness, agreement matrix, plots, CSV/JSON.
- FastAPI v1 API, SQLAlchemy models + Alembic migration, safe uploads, process-isolated time-limited execution, duplicate-run prevention (partial unique index + lock), approval gating, PDF/JSON/CSV exports, benchmark lab jobs.
- React UI: landing, workspace, data upload/mapping, diagnostic setup, live dashboard (run/skip/choose/enter result/pause/resume/stop), React Flow decision tree, evidence explorer, report, benchmark lab, docs, settings; dark/light themes; responsive.

## Benchmark results (SIMULATED, `research/results/results_default-v1.json`)

720 evaluation scenarios (disjoint seeds 100–107; 3 datasets incl. real `breast_cancer`), tables estimated from 720 other scenarios. Budget 8, τ = 0.9.

| Planner | Top-1 | F1 | Mean cost |
|---|---|---|---|
| Random feasible | 0.698 | 0.606 | 7.58 |
| Fixed checklist | 0.795 | 0.738 | 7.75 |
| Cheapest first | 0.743 | 0.625 | 6.75 |
| Max information gain | 0.852 | 0.777 | 7.01 |
| Information gain per cost | 0.849 | 0.774 | 6.81 |
| Cost-and-risk-aware adaptive | 0.849 | 0.774 | 6.81 |
| Two-step lookahead | 0.856 | 0.779 | 7.00 |

Honest reading:
- Information-based planners clearly beat random, checklist and cheapest-first on accuracy and F1 (paired Wilcoxon p < 1e-4), and the adaptive planner uses less cost than the checklist and max-EIG.
- **The adaptive planner and "information gain per cost" executed identical sequences in this benchmark** (identical results, p = 1). With every risk below the cap and λ·risk tiny, the risk term never changed a decision here; the agreement matrix in the app shows this. They should not be presented as different algorithms on this evidence.
- Max-EIG and two-step lookahead are statistically indistinguishable from adaptive on accuracy; their advantage is not demonstrated.
- No risk-cap violations occurred for any planner at cap 0.15, so the safety benefit is not demonstrated either.
- Calibration of the adaptive planner: ECE 0.042, Brier 0.065. Noisy-OR beats a single-cause mixture on two-fault outcomes (mean log-lik −0.43 vs −0.67).
- Fixed budgets (B = 4–8) show the largest gaps in favour of information-based planners.
- Everything is simulated; no real-world gain is claimed.

## Known limitations / not done

- Not implemented from the blueprint: LLM-only agent baseline, knapsack/budget-aware and Thompson planners, symptom page with change-point detection, online refinement of likelihoods, real-world case study, ablations other than likelihood perturbation, uniform vs symptom-informed prior.
- Model/code-regression hypothesis exists but no experiment discriminates it.
- Experiment discretisation thresholds were tuned while inspecting estimation-seed scenarios; injectors and experiments were written together, so detection is likely optimistic.
- No authentication (trusted local use only); worker isolation is process-level, not container-level.
- Dependency lockfile: `frontend/package-lock.json` only; Python dependencies are version-floored in `pyproject.toml`, not locked.
- Vite bundle exceeds 500 kB for the chart chunk (lazy-split but large).

## Exact startup commands

```bash
python -m venv .venv && .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -e engine -e "backend[dev]"
cd backend && python -m uvicorn app.main:app --port 8000         # API + /api/docs
cd frontend && npm install && npm run dev                         # UI on :5173
python research/benchmark/run_benchmark.py --config research/configs/default.json --workers 6 --install-library
```

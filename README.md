<div align="center">

# ExperimentPilot AI

**Active diagnosis of machine-learning model failures.**
Instead of listing possible causes, ExperimentPilot decides *which diagnostic experiment to run next* — the cheapest, safest test expected to best separate the competing explanations — updates its beliefs from real measurements on your data, and stops when it is confident or the budget is spent.

</div>

> **Research status.** Adaptive and sequential diagnosis is an established field. This project does not claim to be the first, nor that any publication is guaranteed. Likelihood tables and all benchmark results are **simulated** (fault injection on built-in datasets). Outputs are statistical indicators, not proof of causation. See [Limitations](#limitations).

## What it does

1. **Represents** competing root causes as a joint probability distribution over which causes are active (several can be true at once).
2. **Recommends** the feasible experiment with the best `EIG / (cost + λ·risk)` under budget, risk-cap, prerequisite and approval constraints, with a separation score and a plain-language rationale.
3. **Executes** it on a copy of your data in a time-limited worker process — or lets you run it yourself and enter the outcome.
4. **Updates** beliefs with Bayes' rule using a noisy-OR likelihood model; no outcome is ever treated as certain.
5. **Explains**: evidence log, decision tree (path taken vs hypothetical branches), supporting/contradicting evidence per hypothesis.
6. **Reports** a ranked diagnosis with residual uncertainty, cost and risk accounting, remediation ideas and limitations (PDF / JSON / CSV).

| Component | Contents |
|---|---|
| `engine/` | Independently testable research package: belief engine, likelihood models, EIG, 7 planner variants, stopping policy, **10 real experiments**, fault injectors, benchmark, likelihood estimation |
| `backend/` | FastAPI + SQLAlchemy + Alembic API: projects, uploads, sessions, evidence, tree, reports, benchmark lab |
| `frontend/` | React + TypeScript + Vite + Tailwind UI (Recharts, React Flow) |
| `research/` | Benchmark configs, runner, results, analysis script |
| `docs/` | Methodology, architecture, security, evaluation protocol, API, experiments, datasets, model card, paper outline, related-work template |

## Quick start (local)

Requirements: Python 3.11+ (developed on 3.14), Node 20+.

```bash
# 1. Python environment
python -m venv .venv
.venv/Scripts/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1 ; macOS/Linux: source .venv/bin/activate
pip install -e engine -e "backend[dev]"

# 2. Backend  (http://127.0.0.1:8000, docs at /api/docs)
cd backend && python -m uvicorn app.main:app --port 8000

# 3. Frontend (http://localhost:5173)
cd frontend && npm install && npm run dev
```

Or run `scripts/dev.ps1` (Windows) / `scripts/dev.sh`. The database is created and migrated automatically on first start (SQLite under `backend/data/`).

**Try it in one click:** open the app → *Workspace* → **Demo scenario** → choose a fault → *Configure diagnosis* → *Start diagnosis* → *Run recommended* a few times → open *Decision tree*, *Evidence* and *Report*. The injected ground truth is shown on the report so you can compare.

### Docker

```bash
docker compose up --build        # UI on http://localhost:8080
```

*(Dockerfiles and compose were written but could not be executed in the development environment, which had no Docker. Treat as untested.)*

### Configuration

Environment variables (see [`.env.example`](.env.example)): `EP_DATABASE_URL` (SQLite default; PostgreSQL-compatible), `EP_DATA_DIR`, `EP_MAX_UPLOAD_MB`, `EP_CORS_ORIGINS`, `EP_EXECUTOR` (`process`/`thread`), `EP_EXPERIMENT_TIMEOUT_S`, …

## Using your own data

Upload three CSVs per project (UTF-8, header row, ≤ 25 MB):

| Role | Content |
|---|---|
| Reference | training/validation data: features, target, model prediction |
| Recent | recent production data with the same columns |
| Replay (optional) | the recent records processed by the previous pipeline version |

Map the target / prediction / timestamp / identifier columns; everything else is a feature. Model files are never loaded — the model enters through its prediction column. If data cannot be uploaded, enable **human-in-the-loop** mode: the planner recommends, you run the check yourself and enter the outcome.

## The ten experiments

`feature_drift`, `data_quality_audit`, `class_prior`, `label_noise` (confident learning), `leakage_scan`, `pipeline_replay`, `importance_weighted_eval`, `slice_analysis`, `retrain_recent` (explicit approval), `metric_recompute`. Each declares cost, risk, required inputs, outcome levels, raw results and limitations — see [docs/experiments.md](docs/experiments.md). Missing inputs make an experiment *unavailable*; evidence is never fabricated.

## Mathematics in one screen

```
state        z ∈ {0,1}^K                      (exact, or truncated to ≤ m simultaneous causes)
likelihood   P(o ≥ j | z) = 1 − (1 − leak_j) ∏_{k active} (1 − sens_{k,j})      (cumulative noisy-OR)
update       b'(z) ∝ b(z) · P(o | e, z)
gain         EIG(e) = H(b) − Σ_o P(o|b,e) H(b|o,e)
selection    score(e) = EIG(e) / (c(e) + λ r(e))   s.t. c(e) ≤ budget left, r(e) ≤ risk cap
stop         confidence ≥ τ | no feasible experiment | budget insufficient | score < ε | safety blocked
```

Full derivation, assumptions and the large-K approximation: [docs/methodology.md](docs/methodology.md).

## Benchmark

A reproducible fault-injection benchmark compares seven planners (random, fixed checklist, cheapest-first, max-EIG, EIG-per-cost, cost-and-risk-aware adaptive, two-step lookahead). Likelihood tables are estimated from separate *estimation* seeds and planners are scored on disjoint *evaluation* seeds, including a real dataset never used for estimation.

```bash
python research/benchmark/run_benchmark.py --config research/configs/default.json --workers 6   # ~1 h on 8 cores
python research/benchmark/run_benchmark.py --quick                                              # smoke run
python research/analysis/summarize.py research/results/results_default-v1.json
```

Protocol, metrics and threats to validity: [docs/evaluation.md](docs/evaluation.md). Results are in `research/results/` and in the app (*Benchmark lab*). See the **Results** section below for what was measured.

## Testing

```bash
cd engine   && pytest -q          # research core, experiments, injectors, estimation, benchmark reproducibility
cd backend  && pytest -q          # API, upload security, state machine, concurrency, reports, benchmark jobs
cd frontend && npm test           # Vitest + Testing Library
cd frontend && npx playwright install chromium && npm run e2e   # browser end-to-end (needs both servers running)
```

Exact outcomes of the last run are recorded in [BUILD_STATUS.md](BUILD_STATUS.md).

## Results (simulated)

On 720 held-out simulated scenarios information-based planners reach ~0.85 top-1 vs 0.70 (random), 0.80 (checklist) and 0.74 (cheapest-first) at budget 8. **However**, the cost-and-risk-aware adaptive planner behaved identically to the cost-only EIG-per-cost planner, and max-EIG / two-step lookahead were statistically indistinguishable on accuracy; no risk-cap violations occurred. Full tables, caveats and plots: [BUILD_STATUS.md](BUILD_STATUS.md), `research/results/`.

## Security

Trusted-local deployment, **no authentication**. Upload limits, content validation, safe CSV parsing, path-traversal protection, no unpickling or execution of uploads, process-isolated time-limited experiments, restricted CORS, parameterised SQL, CSV-injection-safe exports, deletion controls. Details and gaps: [docs/security.md](docs/security.md).

## Limitations

* Likelihoods come from simulated faults on synthetic and one small real dataset; real faults can look different.
* Conditional independence of experiment outcomes is assumed.
* Several experiments are confounded by one another (e.g. importance weighting also “explains” pipeline changes); the posterior accounts for this only as far as the simulation reproduces it.
* No LLM-agent baseline, real-world case study, online refinement of tables, knapsack or Thompson planners yet.
* Authentication, rate limiting and container-level sandboxing are not provided.

## Repository layout

```
engine/experimentpilot/{core,experiments,faults,likelihood,benchmark}
engine/tests   backend/app/{api,core,models,schemas,services,reports,security}   backend/{tests,alembic}
frontend/src/{pages,components,charts,decision-tree,api,hooks,types,test}   frontend/e2e
research/{benchmark,analysis,configs,results}   docs/   scripts/   .github/workflows/
```

## License

Not yet chosen — add one before publishing (see the blueprint's first-week checklist).

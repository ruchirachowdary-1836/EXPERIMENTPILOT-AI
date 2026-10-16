# Architecture

```mermaid
flowchart LR
  subgraph FE[Frontend: React + TypeScript + Vite]
    UI[Pages: Landing, Workspace, Setup, Live, Tree, Evidence, Report, Benchmark, Docs]
    RQ[React Query + typed API client]
  end
  subgraph BE[Backend: FastAPI]
    API[/api/v1 routes/]
    SVC[Services: sessions, projects, report, benchmark, demo]
    SEC[Security: upload validation, safe paths]
    EXE[Executor: worker process + timeout]
    DB[(SQLite / PostgreSQL via SQLAlchemy + Alembic)]
    FS[(Data dir: sanitised CSV copies)]
  end
  subgraph ENG[Engine: experimentpilot package]
    CORE[core: belief, likelihood, EIG, planner, variants, stopping]
    EXP[experiments: 10 plugins]
    FLT[faults: injectors, datasets, scenarios]
    BEN[benchmark: runner, analysis]
    LIK[likelihood: estimation + default tables]
  end
  UI --> RQ --> API --> SVC
  SVC --> DB
  SVC --> SEC --> FS
  SVC --> CORE
  SVC --> EXE --> EXP
  EXP --> FS
  SVC --> BEN --> FLT
  BEN --> LIK
  CORE --> LIK
```

## Layers

| Layer | Path | Responsibility |
|---|---|---|
| Research engine | `engine/experimentpilot` | Pure-Python, independently testable, no web or DB dependency |
| API | `backend/app` | Validation, persistence, orchestration, exports |
| Frontend | `frontend/src` | Presentation only; every number comes from the API |
| Research assets | `research/` | Benchmark configs, scripts, results, analysis |

## Session lifecycle

```mermaid
stateDiagram-v2
  [*] --> active: create (validated)
  active --> paused: pause
  paused --> active: resume
  active --> completed: stopping rule fires
  active --> stopped: user stops
  paused --> stopped: user stops
  completed --> [*]
  stopped --> [*]
```

Transitions are explicit (`TRANSITIONS` in `services/sessions.py`); invalid ones return `409`.

## One diagnostic step

```mermaid
sequenceDiagram
  participant U as User
  participant A as API
  participant P as Planner (engine)
  participant X as Worker process
  participant D as DB
  U->>A: POST /sessions/{id}/run
  A->>A: take per-session lock, verify state
  A->>P: score + feasibility (budget, risk, inputs, approval)
  A->>D: insert run(status=running) [unique partial index]
  A->>X: execute experiment on CSV copies (timeout)
  X-->>A: outcome, raw stats, tables, interpretation
  A->>P: Bayes update with P(o | e, z)
  A->>D: run=completed, belief snapshot, spent
  A->>P: stopping rules
  A-->>U: run + updated session + next recommendation
```

### Preventing duplicate execution

1. A **partial unique index** on `(session_id, experiment_id)` for statuses `running | completed | external` means two concurrent requests cannot both create a live run; the loser gets `409`.
2. A per-session in-process lock serialises reservation and result application.
3. Only one experiment may be `running` per session.
4. Failed runs do not hold the unique slot and are not charged.

## Persistence

`projects`, `datasets` (metadata only; files live under `EP_DATA_DIR/projects/<id>/`), `diagnosis_sessions` (config, priors, current joint belief), `experiment_runs` (outcome, raw stats, tables, selection-time scores), `belief_snapshots` (posterior history), `reports` (export metadata), `benchmark_jobs`. Migrations: `backend/alembic`. SQLite for local use; set `EP_DATABASE_URL` to a PostgreSQL URL for deployment (the partial unique index has a PostgreSQL variant).

## Frontend structure

`pages/` (routes), `features`-style folders for `charts/`, `decision-tree/`, `components/` (design system), `api/` (client), `types/`, `hooks/`, `test/`.

# API reference

Interactive OpenAPI docs: `http://127.0.0.1:8000/api/docs` (Swagger) and `/api/redoc`. All routes live under `/api/v1`. Errors have the shape `{"error": {"code", "message", ...}}`.

| Area | Method & path | Purpose |
|---|---|---|
| Health | `GET /health` | liveness + DB check |
| Meta | `GET /meta/{hypotheses,experiments,planners,likelihoods,runtime,demo-options}` | catalogues, likelihood provenance, runtime limits |
| Projects | `POST /projects`, `GET /projects?limit&offset&q`, `GET/PATCH/DELETE /projects/{id}` | CRUD (delete removes data files) |
| Datasets | `POST /projects/{id}/datasets` (multipart `role`, `file`) | upload `reference` / `recent` / `replay` CSV |
| | `GET /projects/{id}/datasets/{role}` / `DELETE …` | schema + preview / remove |
| | `PUT /projects/{id}/mapping` | target, prediction, timestamp, identifier, features |
| Demo | `POST /demo` | simulated scenario project |
| Sessions | `POST /sessions` | validate and start; returns first recommendation |
| | `GET /sessions`, `GET/DELETE /sessions/{id}` | list / status + beliefs + runs + recommendation |
| | `GET /sessions/{id}/next` | scored candidates, selected one, rationale, stop decision |
| | `POST /sessions/{id}/run` `{experiment_id?, approve?}` | execute recommended or chosen experiment |
| | `POST /sessions/{id}/external` `{experiment_id, outcome, note}` | human-supplied outcome |
| | `POST /sessions/{id}/skip`, `/pause`, `/resume`, `/stop` | controls |
| Evidence | `GET /sessions/{id}/evidence[/{run_id}]`, `/posterior-history`, `/tree` | evidence, belief history, decision tree |
| Reports | `GET /sessions/{id}/report` | JSON report |
| | `GET /sessions/{id}/export/report.pdf`, `report.json`, `{hypotheses,experiments,history}.csv` | downloads |
| Benchmark | `GET /benchmarks/options`, `GET /benchmarks/published[/runs\|/plots/{png}]` | shipped results |
| | `POST /benchmarks`, `GET /benchmarks`, `GET /benchmarks/{id}[/results\|/traces\|/plots/{png}]`, `GET /benchmarks/{id}/export.{csv,json}` | laboratory jobs |

## Status codes

`200/201/202/204` success · `404` not found · `409` state or policy conflict (`session_not_active`, `invalid_transition`, `duplicate_run`, `run_in_progress`, `approval_required`, `experiment_not_feasible`, `budget_exceeded`, `risk_cap_exceeded`, `dataset_in_use`) · `413` too large · `415` unsupported file type · `422` validation (`validation_error`, `missing_data`, `invalid_mapping`, `too_few_hypotheses`, `no_feasible_experiments`, `seed_overlap`, …) · `500` generic internal error.

## Example

```bash
curl -s -X POST localhost:8000/api/v1/demo -H 'content-type: application/json' \
  -d '{"dataset":"synthetic_credit","seed":7,"faults":[{"fault":"pipeline_change","severity":"high"}]}'
curl -s -X POST localhost:8000/api/v1/sessions -H 'content-type: application/json' -d '{"project_id":"<id>","budget":10}'
curl -s -X POST localhost:8000/api/v1/sessions/<sid>/run -H 'content-type: application/json' -d '{}'
curl -s localhost:8000/api/v1/sessions/<sid>/export/report.pdf -o report.pdf
```

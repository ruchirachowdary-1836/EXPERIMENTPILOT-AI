# Security and privacy

**Deployment model: trusted local environment.** The API has **no authentication**. Do not expose it to the internet. If you need multi-user access, put it behind an authenticating reverse proxy of your own. Adding fake authentication would be worse than none, so none is provided.

## Threats considered and controls

| Threat | Control | Where |
|---|---|---|
| Oversized upload | Streaming read with a hard byte limit (`EP_MAX_UPLOAD_MB`, default 25) → `413`; row and column limits | `security/uploads.py` |
| Wrong file type / disguised binary | `.csv` extension required; magic-byte and NUL checks; UTF-8 required | same |
| Malformed or ragged CSV, duplicate headers | Strict field-count check and `on_bad_lines="error"`; unique non-empty header names | same |
| Path traversal via file name | Original name reduced to a sanitised basename, stored under a server-generated UUID name; every path is resolved and confined to the data directory | `safe_join`, `sanitize_filename` |
| Code execution from uploads | Uploads are only parsed as CSV text. Nothing is `exec`ed, imported or unpickled. Uploaded model files are not accepted; predictions are used instead | design |
| Unsafe deserialisation | No `pickle`/`joblib.load` of user content anywhere | design |
| Resource exhaustion | Experiments run in a worker **process** with a wall-clock timeout (`EP_EXPERIMENT_TIMEOUT_S`), single-threaded numerics, row/column caps, benchmark size limit (400 scenarios) | `services/executor.py` |
| Outbound network from data processing | None. The code never opens network connections while processing uploaded data | design |
| SQL injection | SQLAlchemy ORM with bound parameters only | all DB code |
| Cross-origin abuse | CORS restricted to `EP_CORS_ORIGINS`, no credentials, limited methods/headers | `main.py` |
| Information leakage in errors | Unhandled errors return a generic message and are logged server-side; validation errors are structured | `main.py` |
| CSV (formula) injection in exports | Cells starting with `= + - @` are neutralised on export | `csv_safe` |
| Risky operations run unintentionally | Retraining and anything above the risk warning level require an explicit `approve` flag; risk cap is a hard limit | `services/sessions.py` |
| Production side effects | Experiments run only on copies of uploaded data; nothing writes to external systems | design |
| Secrets | Configuration via `EP_*` environment variables / `.env` (git-ignored); no secrets in code | `core/config.py` |
| Data retention | Deleting a project removes DB rows and all stored files; individual datasets can be removed while no sessions reference them | `services/projects.py` |

Security headers `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` and `Cache-Control: no-store` are set on all responses.

## Not covered

Authentication/authorisation, rate limiting, encryption at rest, antivirus scanning, per-user data isolation, and OS-level sandboxing of the worker process (it is process-isolated with a timeout, but not a container). For untrusted multi-tenant use, run workers in locked-down containers without network access.

## Tests

`backend/tests/test_api.py` covers rejection of wrong extensions, binary content, NUL bytes, empty files, one-column files, duplicate headers, non-UTF-8 text, ragged rows, oversize files and traversal file names, plus approval gating, risk caps and duplicate-run prevention.

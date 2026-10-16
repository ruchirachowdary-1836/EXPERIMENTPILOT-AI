#!/usr/bin/env bash
# Start backend (8000) and frontend (5173) for local development.
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
(cd "$root/backend" && python -m uvicorn app.main:app --port 8000 --reload) &
(cd "$root/frontend" && npm run dev) &
wait

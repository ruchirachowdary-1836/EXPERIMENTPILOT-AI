# Start backend (8000) and frontend (5173) for local development on Windows.
$root = Split-Path -Parent $PSScriptRoot
Start-Process -WorkingDirectory "$root\backend" -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList "-m","uvicorn","app.main:app","--port","8000","--reload"
Start-Process -WorkingDirectory "$root\frontend" -FilePath "npm.cmd" -ArgumentList "run","dev"
Write-Host "Backend http://127.0.0.1:8000/api/docs  Frontend http://localhost:5173"

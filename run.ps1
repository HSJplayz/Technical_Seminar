param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPy = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
    Write-Host "Creating virtual environment..."
    python -m venv (Join-Path $root ".venv")
    & $venvPy -m pip install --quiet -r (Join-Path $root "backend\requirements.txt")
}

Write-Host "Starting MovieStore on http://127.0.0.1:$Port"
& $venvPy -m uvicorn main:app --app-dir (Join-Path $root "backend") --host 127.0.0.1 --port $Port

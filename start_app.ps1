$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
Set-Location -LiteralPath $projectPath
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create .venv and install requirements.txt first; see README.md.' }
& $pythonPath -m streamlit run (Join-Path $projectPath 'app.py') --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false

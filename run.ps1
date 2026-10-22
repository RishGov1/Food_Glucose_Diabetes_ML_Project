# Run Diabetes Digital Twin using the project virtual environment
Write-Host "Starting Diabetes Digital Twin on http://localhost:8501..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m streamlit run app.py

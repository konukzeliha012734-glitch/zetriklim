@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Zetriklim calisma ortami bulunamadi.
  echo Lutfen Kurulum-Windows.bat dosyasini once calistirin.
  pause
  exit /b 1
)
start "" "http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501

@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Zetriklim calisma ortami bulunamadi.
  echo Lutfen Kurulum-Windows.bat dosyasini once calistirin.
  pause
  exit /b 1
)

if not exist "streamlit_app.py" (
  echo HATA: Uygulama dosyasi streamlit_app.py bulunamadi.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
  echo HATA: Streamlit kurulu degil veya kurulum eksik.
  echo Lutfen Kurulum-Windows.bat dosyasini yeniden calistirin.
  pause
  exit /b 1
)

curl.exe --silent --fail --max-time 2 "http://127.0.0.1:8501/_stcore/health" >nul 2>nul
if not errorlevel 1 (
  echo Zetriklim zaten calisiyor. Tarayici aciliyor...
  start "" "http://localhost:8501"
  exit /b 0
)

echo Zetriklim baslatiliyor...
echo Sunucu hazir oldugunda tarayici otomatik acilacak.
start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "$url='http://127.0.0.1:8501/_stcore/health'; for($i=0; $i -lt 120; $i++){ & curl.exe --silent --fail --max-time 1 $url | Out-Null; if($LASTEXITCODE -eq 0){ Start-Process 'http://localhost:8501'; exit 0 }; Start-Sleep -Milliseconds 500 }"

".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
if errorlevel 1 (
  echo.
  echo HATA: Zetriklim sunucusu baslatilamadi.
  echo Yukaridaki hata mesajini kontrol edin.
  pause
  exit /b 1
)

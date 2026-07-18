@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo [1/5] Python denetleniyor...
where py >nul 2>nul
if errorlevel 1 (
  echo.
  echo HATA: Python baslaticisi bulunamadi.
  echo Python 3.11 veya daha yeni bir surumu python.org adresinden kurun.
  goto :error
)

py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
  echo.
  echo HATA: Zetriklim icin Python 3.11 veya daha yeni bir surum gerekir.
  goto :error
)

echo [2/5] Sanal calisma ortami hazirlaniyor...
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto :error
)

echo [3/5] Paket yoneticisi guncelleniyor...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 goto :error

echo [4/5] Zetriklim bagimliliklari kuruluyor...
echo Bu adim internet hizina gore birkac dakika surebilir.
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --prefer-binary --timeout 60 --retries 5 -r requirements.txt
if errorlevel 1 goto :error

echo [5/5] Kurulum dogrulaniyor...
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -c "import streamlit, numpy, pandas, geopandas, folium, streamlit_folium, plotly, requests, openpyxl, matplotlib, ee, rasterio, shapefile, scipy"
if errorlevel 1 goto :error

echo.
echo Kurulum tamamlandi.
echo Zetriklim-Baslat.bat dosyasina cift tiklayarak uygulamayi acabilirsiniz.
pause
exit /b 0

:error
echo.
echo KURULUM BASARISIZ OLDU. Yukaridaki ilk HATA satirini kontrol edin.
echo Sorun devam ederse bu pencerenin ekran goruntusunu paylasin.
pause
exit /b 1

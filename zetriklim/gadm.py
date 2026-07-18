"""GADM 4.1 ülke ve idari bölüm sınırlarını akademik kullanım için okur."""

from __future__ import annotations

import io
import re

import geopandas as gpd
import requests


GADM_VERSION = "4.1"
GADM_BASE_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json"


def fetch_gadm(iso3: str, level: int, timeout: int = 90) -> gpd.GeoDataFrame:
    code = str(iso3).strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", code):
        raise ValueError("GADM ülke kodu üç harfli ISO3 biçiminde olmalıdır (ör. TUR).")
    if int(level) not in {0, 1, 2}:
        raise ValueError("GADM idari düzeyi 0, 1 veya 2 olmalıdır.")
    url = f"{GADM_BASE_URL}/gadm41_{code}_{int(level)}.json"
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404:
        raise ValueError(f"GADM'da {code} için düzey {level} sınırı bulunamadı.")
    response.raise_for_status()
    frame = gpd.read_file(io.BytesIO(response.content))
    if frame.empty:
        raise ValueError("GADM yanıtı coğrafi obje içermiyor.")
    frame.attrs.update({"source": "GADM", "version": GADM_VERSION, "url": url})
    return frame


def name_column(frame: gpd.GeoDataFrame, level: int) -> str:
    preferred = f"NAME_{int(level)}"
    if preferred in frame.columns:
        return preferred
    if int(level) == 0 and "COUNTRY" in frame.columns:
        return "COUNTRY"
    candidates = [column for column in frame.columns if str(column).startswith("NAME_")]
    if not candidates:
        raise ValueError("GADM verisinde idari birim adı alanı bulunamadı.")
    return candidates[-1]

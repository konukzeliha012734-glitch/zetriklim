"""GADM idari sınırlarını tez haritalarında bağlam katmanı olarak kullanır.

GADM verisi yalnız çalışma sırasında indirilir. Ham veri çıktı paketine eklenmez;
böylece GADM'nin akademik kullanım ve yeniden dağıtım koşulları korunur.
"""

from __future__ import annotations

import io
import re
from typing import Any

import geopandas as gpd
import requests


GADM_VERSION = "4.1"
GADM_LICENSE_URL = "https://gadm.org/license.html"
GADM_BASE_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json"


def gadm_download_url(country_iso3: str, level: int) -> str:
    """Doğrulanmış ISO3 ve idari düzey için resmî GADM GeoJSON adresini üretir."""
    iso3 = str(country_iso3).strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", iso3):
        raise ValueError("GADM ülke kodu üç harfli ISO3 biçiminde olmalıdır (ör. TUR).")
    level = int(level)
    if level < 0 or level > 5:
        raise ValueError("GADM idari düzeyi 0 ile 5 arasında olmalıdır.")
    return f"{GADM_BASE_URL}/gadm41_{iso3}_{level}.json"


def fetch_gadm_boundaries(
    country_iso3: str,
    level: int = 1,
    *,
    timeout: int = 30,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Bir ülkenin GADM sınırlarını indirir ve kaynak metadata kaydıyla döndürür."""
    url = gadm_download_url(country_iso3, level)
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Zetriklim/1.1 academic-cartography"},
    )
    response.raise_for_status()
    boundaries = gpd.read_file(io.BytesIO(response.content))
    if boundaries.empty:
        raise ValueError("GADM yanıtı geçerli idari sınır içermiyor.")
    if boundaries.crs is None:
        boundaries = boundaries.set_crs(4326)
    else:
        boundaries = boundaries.to_crs(4326)
    metadata = {
        "dataset": "GADM",
        "version": GADM_VERSION,
        "country_iso3": str(country_iso3).strip().upper(),
        "administrative_level": int(level),
        "source_url": url,
        "license_url": GADM_LICENSE_URL,
        "usage": "Akademik haritada idari bağlam; ham veri yeniden dağıtılmaz.",
    }
    return boundaries, metadata

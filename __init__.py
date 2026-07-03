"""Open-Meteo ERA5-Land bağlantısı ve zaman serisi dönüştürme."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARIABLES = {
    "Yağış": ["precipitation_sum"],
    "Hava sıcaklığı": ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max"],
    "Bağıl nem": ["relative_humidity_2m_mean"],
    "Çiy noktası": ["dew_point_2m_mean"],
    "Rüzgâr hızı ve yönü": ["wind_speed_10m_mean", "wind_speed_10m_max", "wind_direction_10m_dominant"],
    "Buharlaşma / gerçek ET": ["et0_fao_evapotranspiration"],
    "Potansiyel evapotranspirasyon": ["et0_fao_evapotranspiration"],
    "Yüzey / deniz seviyesi basıncı": ["surface_pressure_mean", "pressure_msl_mean"],
    "Toprak nemi": ["soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean"],
    "Kar örtüsü / kar su eşdeğeri": ["snowfall_sum", "snowfall_water_equivalent_sum"],
    "Güneş radyasyonu": ["shortwave_radiation_sum"],
    "Bulutluluk": ["cloud_cover_mean"],
}

HOURLY_VARIABLES = {
    "Yağış": ["precipitation"],
    "Hava sıcaklığı": ["temperature_2m"],
    "Bağıl nem": ["relative_humidity_2m"],
    "Çiy noktası": ["dew_point_2m"],
    "Rüzgâr hızı ve yönü": ["wind_speed_10m", "wind_direction_10m"],
    "Buharlaşma / gerçek ET": ["et0_fao_evapotranspiration"],
    "Potansiyel evapotranspirasyon": ["et0_fao_evapotranspiration"],
    "Yüzey / deniz seviyesi basıncı": ["surface_pressure", "pressure_msl"],
    "Toprak nemi": ["soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"],
    "Kar örtüsü / kar su eşdeğeri": ["snowfall", "snow_depth"],
    "Güneş radyasyonu": ["shortwave_radiation"],
    "Bulutluluk": ["cloud_cover"],
}

TURKISH_LABELS = {
    "temperature_2m": "Sıcaklık (°C)",
    "temperature_2m_mean": "Ortalama sıcaklık (°C)",
    "temperature_2m_min": "Minimum sıcaklık (°C)",
    "temperature_2m_max": "Maksimum sıcaklık (°C)",
    "precipitation": "Yağış (mm)",
    "precipitation_sum": "Toplam yağış (mm)",
    "relative_humidity_2m": "Bağıl nem (%)",
    "relative_humidity_2m_mean": "Ortalama bağıl nem (%)",
    "dew_point_2m": "Çiy noktası (°C)",
    "dew_point_2m_mean": "Ortalama çiy noktası (°C)",
    "wind_speed_10m": "Rüzgâr hızı (km/sa)",
    "wind_speed_10m_mean": "Ortalama rüzgâr hızı (km/sa)",
    "wind_speed_10m_max": "Maksimum rüzgâr hızı (km/sa)",
    "wind_direction_10m": "Rüzgâr yönü (°)",
    "wind_direction_10m_dominant": "Baskın rüzgâr yönü (°)",
    "et0_fao_evapotranspiration": "Referans ET₀ (mm)",
    "surface_pressure": "Yüzey basıncı (hPa)",
    "surface_pressure_mean": "Ortalama yüzey basıncı (hPa)",
    "pressure_msl": "Deniz seviyesi basıncı (hPa)",
    "pressure_msl_mean": "Ortalama deniz seviyesi basıncı (hPa)",
    "soil_moisture_0_to_7cm": "Toprak nemi 0–7 cm (m³/m³)",
    "soil_moisture_0_to_7cm_mean": "Ortalama toprak nemi 0–7 cm (m³/m³)",
    "soil_moisture_7_to_28cm": "Toprak nemi 7–28 cm (m³/m³)",
    "soil_moisture_7_to_28cm_mean": "Ortalama toprak nemi 7–28 cm (m³/m³)",
    "snowfall": "Kar yağışı (cm)",
    "snowfall_sum": "Toplam kar yağışı (cm)",
    "snow_depth": "Kar kalınlığı (m)",
    "snowfall_water_equivalent_sum": "Kar su eşdeğeri (mm)",
    "shortwave_radiation": "Kısa dalga radyasyon (W/m²)",
    "shortwave_radiation_sum": "Kısa dalga radyasyon (MJ/m²)",
    "cloud_cover": "Bulutluluk (%)",
    "cloud_cover_mean": "Ortalama bulutluluk (%)",
}


@dataclass
class ClimateResult:
    data: pd.DataFrame
    source_url: str
    latitude: float
    longitude: float
    elevation_m: float | None
    unsupported_variables: list[str]
    model: str


def _resample(data: pd.DataFrame, temporal_scale: str) -> pd.DataFrame:
    if temporal_scale in {"Saatlik", "Günlük"}:
        return data
    rules = {
        "3 saatlik": "3h",
        "6 saatlik": "6h",
        "Haftalık": "W",
        "Aylık": "MS",
        "Mevsimlik": "QS-DEC",
        "Yıllık": "YS",
    }
    rule = rules.get(temporal_scale)
    if not rule:
        return data
    numeric = data.set_index("Tarih")
    aggregations = {}
    for column in numeric.columns:
        lower = column.lower()
        aggregations[column] = "sum" if any(word in lower for word in ["yağış", "et₀", "radyasyon", "kar yağışı"]) else "mean"
    return numeric.resample(rule).agg(aggregations).reset_index()


def fetch_centroid_series(
    *,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    variables: list[str],
    temporal_scale: str,
) -> ClimateResult:
    hourly = temporal_scale in {"Saatlik", "3 saatlik", "6 saatlik"}
    variable_map = HOURLY_VARIABLES if hourly else DAILY_VARIABLES
    selected: list[str] = []
    unsupported: list[str] = []
    for variable in variables:
        if variable in variable_map:
            selected.extend(variable_map[variable])
        else:
            unsupported.append(variable)
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("Seçilen değişkenler Open-Meteo ERA5-Land bağlantısında desteklenmiyor.")

    safe_end = min(end_date, date.today() - timedelta(days=6))
    if safe_end < start_date:
        raise ValueError("ERA5-Land arşivi için bitiş tarihi en az 6 gün önce olmalıdır.")

    key = "hourly" if hourly else "daily"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": safe_end.isoformat(),
        key: ",".join(selected),
        "timezone": "auto",
        "models": "era5",
        "wind_speed_unit": "kmh",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=180)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(payload.get("reason", "Open-Meteo veri hatası"))

    values = payload[key]
    frame = pd.DataFrame(values)
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.rename(columns={"time": "Tarih", **TURKISH_LABELS})
    frame = _resample(frame, temporal_scale)
    frame.insert(1, "Örnek ID", 1)
    frame.insert(2, "Enlem", float(payload["latitude"]))
    frame.insert(3, "Boylam", float(payload["longitude"]))

    return ClimateResult(
        data=frame,
        source_url=response.url,
        latitude=float(payload["latitude"]),
        longitude=float(payload["longitude"]),
        elevation_m=payload.get("elevation"),
        unsupported_variables=unsupported,
        model="ERA5 (Open-Meteo Historical Weather API)",
    )

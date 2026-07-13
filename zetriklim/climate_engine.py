"""Climate Engine API bağlantısı ve kullanıcı anahtarı doğrulaması."""

from __future__ import annotations

import os
import json
from datetime import timedelta
from typing import Any

import geopandas as gpd
import pandas as pd
import requests


BASE_URL = "https://api.climateengine.org"


def api_key_available(api_key: str | None = None) -> bool:
    return bool(api_key or os.getenv("CLIMATE_ENGINE_API_KEY"))


def connection_label(api_key: str | None = None) -> str:
    return "Bağlı" if api_key_available(api_key) else "API anahtarı bekleniyor"


def validate_api_key(api_key: str) -> dict[str, Any]:
    """Anahtarı Climate Engine'in resmi doğrulama uç noktasında sınar."""
    response = requests.get(
        f"{BASE_URL}/home/validate_key",
        headers={"Authorization": api_key.strip()},
        timeout=30,
    )
    if response.status_code in {401, 403}:
        raise ValueError("API anahtarı geçersiz veya bu işlem için yetkisiz.")
    response.raise_for_status()
    try:
        validation = response.json()
    except ValueError:
        validation = {"message": response.text.strip() or "Anahtar doğrulandı."}

    expiration_response = requests.get(
        f"{BASE_URL}/home/key_expiration",
        headers={"Authorization": api_key.strip()},
        timeout=30,
    )
    expiration = None
    if expiration_response.ok:
        try:
            expiration = expiration_response.json()
        except ValueError:
            expiration = expiration_response.text.strip()
    return {"validation": validation, "expiration": expiration}


def fetch_timeseries(
    api_key: str,
    gdf: gpd.GeoDataFrame,
    start_date,
    end_date,
    dataset: str,
    variables: str,
    area_reducer: str = "mean",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Yüklenen alan için Climate Engine native zaman serisini getirir."""
    wgs84 = gdf.to_crs(4326)
    metric_crs = wgs84.estimate_utm_crs() or "EPSG:6933"
    metric_geometry = wgs84[["geometry"]].dissolve().to_crs(metric_crs)
    tolerance = 100

    def geometry_coordinates(simplify_tolerance: int) -> str:
        candidate = (
            metric_geometry.simplify(simplify_tolerance, preserve_topology=True)
            .to_crs(4326)
            .iloc[0]
        )
        if candidate.is_empty or not candidate.is_valid:
            candidate = metric_geometry.to_crs(4326).iloc[0]
        return json.dumps(candidate.__geo_interface__["coordinates"])

    coordinates = geometry_coordinates(tolerance)
    # Büyük poligonlar POST sınırına sığsa bile Earth Engine tarafında işlem hatasına
    # yol açabiliyor. Günlük seriler için daha güvenli, fakat şekli koruyan bir hedef.
    while len(coordinates) > 8_000 and tolerance < 10_000:
        tolerance *= 2
        coordinates = geometry_coordinates(tolerance)
    endpoint = f"{BASE_URL}/timeseries/native/coordinates"
    headers = {"Authorization": api_key.strip()}

    def request_period(
        period_start,
        period_end,
        request_tolerance: int = tolerance,
        allow_geometry_retry: bool = True,
    ) -> tuple[list[dict], dict[str, Any]]:
        request_coordinates = geometry_coordinates(request_tolerance)
        payload = {
            "coordinates": request_coordinates,
            "simplify_geometry": request_tolerance,
            "area_reducer": area_reducer,
            "dataset": dataset.strip(),
            "variable": variables.strip(),
            "start_date": str(period_start),
            "end_date": str(period_end),
        }
        response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
        if response.status_code in {401, 403}:
            raise ValueError(
                "Climate Engine anahtarı geçersiz, süresi dolmuş veya kotası yetersiz."
            )
        if not response.ok:
            try:
                error_detail = response.json()
            except ValueError:
                error_detail = response.text.strip()
            error_text = str(error_detail)
            span_days = (period_end - period_start).days
            server_error = response.status_code >= 500
            response_too_large = "Response size exceeds limit" in error_text

            # İlk 500 yanıtında geometriyi daha güçlü sadeleştirerek aynı dönemi
            # bir kez daha dener. Böylece geçerli fakat karmaşık havza sınırları
            # kullanıcı müdahalesi olmadan Climate Engine'e uyarlanır.
            if server_error and allow_geometry_retry and request_tolerance < 10_000:
                stronger_tolerance = min(max(request_tolerance * 4, 500), 10_000)
                stronger_coordinates = geometry_coordinates(stronger_tolerance)
                if len(stronger_coordinates) < len(request_coordinates):
                    records, retry_meta = request_period(
                        period_start,
                        period_end,
                        stronger_tolerance,
                        False,
                    )
                    return records, {
                        "geometry_retry": True,
                        "initial_tolerance_m": request_tolerance,
                        "retry_tolerance_m": stronger_tolerance,
                        "result": retry_meta,
                    }

            # Climate Engine bazı iç sunucu hatalarında ayrıntılı neden dönmüyor.
            # Günlük veriyi daha küçük dönemlere ayırmak hem yanıt boyutunu hem de
            # tek Earth Engine görevinin hesap yükünü azaltır.
            if server_error and span_days > (31 if response_too_large else 62):
                midpoint = period_start + timedelta(days=span_days // 2)
                left_records, left_meta = request_period(
                    period_start,
                    midpoint,
                    request_tolerance,
                    False,
                )
                right_records, right_meta = request_period(
                    midpoint + timedelta(days=1),
                    period_end,
                    request_tolerance,
                    False,
                )
                return left_records + right_records, {
                    "split": True,
                    "split_reason": (
                        "response_size" if response_too_large else "server_error"
                    ),
                    "parts": [left_meta, right_meta],
                }
            raise RuntimeError(
                f"Climate Engine isteği başarısız (HTTP {response.status_code}). "
                f"Dataset={dataset}, değişken={variables}, dönem={period_start}/{period_end}, "
                f"geometri={len(request_coordinates):,} karakter, "
                f"sadeleştirme={request_tolerance} m. Servis yanıtı: {error_text[:1200]}. "
                "Daha sonra yeniden deneyin veya Google Earth Engine kaynağını seçin."
            )
        body = response.json()
        series_groups = body.get("Data")
        if not series_groups:
            return [], {
                key: value for key, value in body.items() if key != "Data"
            }
        first_group = series_groups[0] if isinstance(series_groups, list) else series_groups
        records = first_group.get("Data") if isinstance(first_group, dict) else first_group
        if isinstance(records, dict):
            normalized_records = pd.DataFrame.from_dict(records).to_dict("records")
        else:
            normalized_records = list(records or [])
        return normalized_records, {
            **{key: value for key, value in body.items() if key != "Data"},
            "geometry_characters": len(request_coordinates),
            "simplify_geometry_m": request_tolerance,
        }

    requested_start = pd.Timestamp(start_date).date()
    requested_end = pd.Timestamp(end_date).date()
    all_records: list[dict] = []
    chunk_metadata = []
    chunk_start = requested_start
    while chunk_start <= requested_end:
        chunk_end = min(
            (pd.Timestamp(chunk_start) + pd.DateOffset(years=5) - pd.Timedelta(days=1)).date(),
            requested_end,
        )
        records, response_metadata = request_period(chunk_start, chunk_end)
        all_records.extend(records)
        chunk_metadata.append(
            {
                "start": str(chunk_start),
                "end": str(chunk_end),
                "record_count": len(records),
                "metadata": response_metadata,
            }
        )
        chunk_start = chunk_end + timedelta(days=1)

    data = pd.DataFrame.from_dict(all_records)
    if data.empty:
        raise ValueError("Climate Engine yanıtındaki zaman serisi boş.")
    date_column = next(
        (column for column in data.columns if column.lower() in {"date", "tarih", "time"}),
        None,
    )
    if date_column:
        data = data.rename(columns={date_column: "Tarih"})
        data["Tarih"] = pd.to_datetime(data["Tarih"], errors="coerce")
    else:
        raise ValueError("Climate Engine yanıtında tarih alanı bulunamadı.")

    identifier_columns = {
        "Tarih", "Date", "date", "time", "Time", "system:index",
        "latitude", "longitude", "lat", "lon", "name", "id",
    }
    for column in data.columns:
        if column not in identifier_columns:
            converted = pd.to_numeric(data[column], errors="coerce")
            if converted.notna().any():
                data[column] = converted

    centroid = wgs84.dissolve().geometry.iloc[0].centroid
    data.insert(1, "Örnek ID", 1)
    data.insert(2, "Enlem", float(centroid.y))
    data.insert(3, "Boylam", float(centroid.x))
    requested_variables = {
        item.strip().lower() for item in variables.split(",") if item.strip()
    }
    rain_candidates = [
        column
        for column in data.columns
        if (
            "precip" in str(column).lower()
            or str(column).lower() in {"pr", "ppt", "rain", "rainfall"}
        )
        and pd.api.types.is_numeric_dtype(data[column])
    ]
    if requested_variables & {"precipitation", "pr", "ppt", "rain", "rainfall"}:
        if not rain_candidates:
            value_candidates = [
                column
                for column in data.select_dtypes(include="number").columns
                if column not in {"Örnek ID", "Enlem", "Boylam"}
            ]
            if len(value_candidates) == 1:
                rain_candidates = value_candidates
        if rain_candidates:
            data = data.rename(columns={rain_candidates[0]: "Toplam yağış (mm)"})
    metadata = {
        "endpoint": endpoint,
        "dataset": dataset,
        "variables": variables,
        "area_reducer": area_reducer,
        "request_chunks": chunk_metadata,
    }
    return data, metadata


def fetch_map_tile(
    api_key: str,
    gdf: gpd.GeoDataFrame,
    start_date,
    end_date,
    dataset: str,
    variable: str,
    analysis: str,
) -> tuple[str, dict[str, Any]]:
    """Climate Engine MapID uç noktasından görselleştirme karo adresi alır."""
    bounds = gdf.to_crs(4326).total_bounds.tolist()
    if analysis == "SPI":
        endpoint = f"{BASE_URL}/raster/mapid/standard_index"
        params = {
            "dataset": dataset,
            "variable": "spi",
            "distribution": "gamma",
            "start_date": str(start_date),
            "end_date": str(end_date),
            "start_year": int(pd.Timestamp(start_date).year),
            "end_year": int(pd.Timestamp(end_date).year),
            "bounding_box": str(bounds),
            "colormap_opacity": 0.85,
            "colormap_type": "continuous",
        }
    else:
        endpoint = f"{BASE_URL}/raster/mapid/values"
        params = {
            "dataset": dataset,
            "variable": variable,
            "temporal_statistic": "mean",
            "start_date": str(start_date),
            "end_date": str(end_date),
            "bounding_box": str(bounds),
            "colormap_opacity": 0.85,
            "colormap_type": "continuous",
        }
    response = requests.get(
        endpoint,
        headers={"Authorization": api_key.strip()},
        params=params,
        timeout=180,
    )
    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text.strip()
        raise RuntimeError(
            f"Climate Engine haritası üretilemedi (HTTP {response.status_code}): "
            f"{str(detail)[:1000]}"
        )
    body = response.json()

    def find_tile_url(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "tile_fetcher" and isinstance(child, str):
                    return child
                found = find_tile_url(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_tile_url(child)
                if found:
                    return found
        return None

    tile_url = find_tile_url(body)
    if not tile_url:
        raise ValueError("Climate Engine yanıtında tile_fetcher harita adresi bulunamadı.")
    return tile_url, {
        "endpoint": endpoint,
        "dataset": dataset,
        "variable": variable,
        "analysis": analysis,
        "period": f"{start_date}/{end_date}",
        "bounds": bounds,
    }

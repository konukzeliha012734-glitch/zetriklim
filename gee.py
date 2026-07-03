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

DATASET_START_DATES = {
    "CHIRPS_DAILY": "1981-01-01",
    "CHIRPS_PENTAD": "1981-01-01",
    "CHIRPS_PRELIM_PENTAD": "2015-01-01",
    "ERA5_AG": "1979-01-01",
    "SENTINEL2_SR": "2015-01-01",
    "HLS_SR": "2013-04-11",
    "LANDSAT_SR": "1984-01-01",
    "LANDSAT8_SR": "2013-04-11",
    "MODIS_TERRA_8DAY": "2000-02-18",
}


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
    if gdf.empty or gdf.crs is None:
        raise ValueError("Çalışma alanı boş veya koordinat sistemi tanımsız.")
    wgs84 = gdf.to_crs(4326).copy()
    wgs84["geometry"] = wgs84.geometry.make_valid()
    wgs84 = wgs84[
        ~wgs84.geometry.is_empty
        & wgs84.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ]
    if wgs84.empty:
        raise ValueError("Çalışma alanında geçerli Polygon veya MultiPolygon geometrisi bulunamadı.")

    requested_start = pd.Timestamp(start_date).date()
    requested_end = pd.Timestamp(end_date).date()
    dataset_start = DATASET_START_DATES.get(dataset.strip())
    if dataset_start and requested_start < pd.Timestamp(dataset_start).date():
        raise ValueError(
            f"{dataset} veri ürünü {dataset_start} tarihinde başlar; "
            f"{requested_start}/{requested_end} dönemi bu ürünle sorgulanamaz."
        )

    metric_crs = wgs84.estimate_utm_crs() or "EPSG:6933"
    metric_geometry = wgs84[["geometry"]].dissolve().to_crs(metric_crs)
    tolerance = 0
    candidate = metric_geometry.to_crs(4326).geometry.iloc[0]
    coordinates = json.dumps(candidate.__geo_interface__["coordinates"])
    while len(coordinates) > 80_000 and tolerance < 10_000:
        tolerance = 100 if tolerance == 0 else tolerance * 2
        candidate = (
            metric_geometry.simplify(tolerance, preserve_topology=True)
            .to_crs(4326)
            .iloc[0]
        )
        coordinates = json.dumps(candidate.__geo_interface__["coordinates"])
    endpoint = f"{BASE_URL}/timeseries/native/coordinates"
    headers = {"Authorization": api_key.strip()}

    def request_period(period_start, period_end) -> tuple[list[dict], dict[str, Any]]:
        payload = {
            "coordinates": coordinates,
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
            if "No data returned" in error_text:
                available_from = DATASET_START_DATES.get(dataset.strip(), "ürünün katalog başlangıcı")
                raise ValueError(
                    f"{dataset} için {period_start}/{period_end} döneminde veri bulunamadı. "
                    f"Bilinen başlangıç: {available_from}. Tarih aralığını, bulutluluk durumunu "
                    "ve çalışma alanının ürün kapsamı içinde olduğunu kontrol edin."
                )
            if (
                response.status_code >= 500
                and "Response size exceeds limit" in error_text
                and span_days > 31
            ):
                midpoint = period_start + timedelta(days=span_days // 2)
                left_records, left_meta = request_period(period_start, midpoint)
                right_records, right_meta = request_period(
                    midpoint + timedelta(days=1), period_end
                )
                return left_records + right_records, {
                    "split": True,
                    "parts": [left_meta, right_meta],
                }
            raise RuntimeError(
                f"Climate Engine isteği başarısız (HTTP {response.status_code}). "
                f"Dataset={dataset}, değişken={variables}, dönem={period_start}/{period_end}, "
                f"geometri={len(coordinates):,} karakter, sadeleştirme={tolerance} m. "
                f"Servis yanıtı: {error_text[:1200]}"
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
            key: value for key, value in body.items() if key != "Data"
        }

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

    requested_variables = {
        item.strip().lower() for item in variables.split(",") if item.strip()
    }
    quality_control = {
        "nodata_values_removed": 0,
        "out_of_range_values_removed": 0,
        "valid_numeric_values": 0,
    }
    value_columns = [
        column
        for column in data.select_dtypes(include="number").columns
        if column not in {"Örnek ID", "Enlem", "Boylam"}
    ]
    for column in value_columns:
        nodata_mask = data[column].isin([-9999, -9999.0, -10000, -10000.0]) | (
            data[column] <= -9990
        )
        quality_control["nodata_values_removed"] += int(nodata_mask.sum())
        data.loc[nodata_mask, column] = pd.NA

    requested_upper = {item.upper() for item in requested_variables}
    if "NDVI" in requested_upper:
        ndvi_columns = [
            column for column in value_columns if str(column).strip().upper() == "NDVI"
        ]
        for column in ndvi_columns:
            invalid = data[column].notna() & ~data[column].between(-1.0, 1.0)
            quality_control["out_of_range_values_removed"] += int(invalid.sum())
            data.loc[invalid, column] = pd.NA
    elif "EVI" in requested_upper:
        evi_columns = [
            column for column in value_columns if str(column).strip().upper() == "EVI"
        ]
        for column in evi_columns:
            invalid = data[column].notna() & ~data[column].between(-1.0, 2.5)
            quality_control["out_of_range_values_removed"] += int(invalid.sum())
            data.loc[invalid, column] = pd.NA
    quality_control["valid_numeric_values"] = int(
        data[value_columns].notna().sum().sum()
    )

    centroid = wgs84.dissolve().geometry.iloc[0].centroid
    data.insert(1, "Örnek ID", 1)
    data.insert(2, "Enlem", float(centroid.y))
    data.insert(3, "Boylam", float(centroid.x))
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
        "geometry_type": candidate.geom_type,
        "geometry_valid": bool(candidate.is_valid),
        "geometry_bounds_wgs84": [float(value) for value in candidate.bounds],
        "geometry_coordinate_characters": len(coordinates),
        "geometry_simplification_m": tolerance,
        "quality_control": quality_control,
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
    map_styles = {
        "NDVI": {
            "colormap_min_max": "[-1,1]",
            "colormap_palette": "[3b6fb6,c9b28f,f0dc65,88c96b,187a3d]",
        },
        "EVI": {
            "colormap_min_max": "[-1,1]",
            "colormap_palette": "[5b4b8a,d8c6a3,f0dc65,78c679,006837]",
        },
        "SPI": {
            "colormap_min_max": "[-3,3]",
            "colormap_palette": "[8b1a1a,d6604d,f4a582,f7f7f7,92c5de,4393c3,2166ac]",
        },
        "LST": {
            "colormap_min_max": "[-10,50]",
            "colormap_palette": "[313695,74add1,ffffbf,f46d43,a50026]",
        },
    }
    selected_style = map_styles.get(analysis, {})
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
            **selected_style,
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
            **selected_style,
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
        "map_style": selected_style,
    }

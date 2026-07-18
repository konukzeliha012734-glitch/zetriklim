"""Climate Engine API bağlantısı ve kullanıcı anahtarı doğrulaması."""

from __future__ import annotations

import os
import json
import hashlib
import pickle
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests

from zetriklim.map_styles import map_visual_style


BASE_URL = "https://api.climateengine.org"
CACHE_VERSION = 2
CACHE_ROOT = Path(
    os.getenv("LOCALAPPDATA", str(Path.home() / ".cache"))
) / "Zetriklim" / "climate_engine"


class ClimateEngineQuotaExceeded(RuntimeError):
    """Climate Engine günlük istek kotasının dolduğunu açıkça bildirir."""


def _cache_file(kind: str, payload: dict[str, Any]) -> Path:
    serialized = json.dumps(
        {"version": CACHE_VERSION, **payload},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return CACHE_ROOT / kind / f"{hashlib.sha256(serialized).hexdigest()}.pkl"


def _read_cache(path: Path) -> Any | None:
    try:
        if path.is_file():
            with path.open("rb") as stream:
                return pickle.load(stream)
    except (OSError, EOFError, pickle.PickleError):
        return None
    return None


def _write_cache(path: Path, value: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    except OSError:
        # Önbellek bir hız/kota korumasıdır; yazma sorunu analizi bozmamalıdır.
        return


def _quota_exceeded(status_code: int, detail: Any) -> bool:
    text = str(detail).lower()
    return status_code in {402, 403, 404, 429} and (
        "daily request limit" in text
        or "quota" in text
        or "request limit" in text
    )


def api_key_available(api_key: str | None = None) -> bool:
    return bool(api_key or os.getenv("CLIMATE_ENGINE_API_KEY"))


def fetch_dataset_date_range(api_key: str, dataset: str) -> tuple[date, date]:
    """Climate Engine metadata bilgisinden datasetin gerçek tarih aralığını döndürür."""
    response = requests.get(
        f"{BASE_URL}/metadata/dataset_dates",
        headers={"Authorization": api_key.strip()},
        params={"dataset": dataset, "export_format": "json"},
        timeout=45,
    )
    if not response.ok:
        raise RuntimeError(
            f"{dataset} tarih kapsamı alınamadı (HTTP {response.status_code})."
        )
    candidates: list[date] = []

    def collect_dates(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect_dates(child)
        elif isinstance(value, list):
            for child in value:
                collect_dates(child)
        elif isinstance(value, str):
            for match in re.findall(r"\d{4}-\d{2}-\d{2}", value):
                try:
                    candidates.append(date.fromisoformat(match))
                except ValueError:
                    continue

    collect_dates(response.json())
    if len(candidates) < 2:
        raise ValueError(f"{dataset} tarih metadata yanıtı çözümlenemedi.")
    return min(candidates), max(candidates)


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
    simplified = metric_geometry.simplify(tolerance, preserve_topology=True)
    candidate = simplified.to_crs(4326).iloc[0]
    coordinates = json.dumps(candidate.__geo_interface__["coordinates"])
    while len(coordinates) > 80_000 and tolerance < 10_000:
        tolerance *= 2
        candidate = (
            metric_geometry.simplify(tolerance, preserve_topology=True)
            .to_crs(4326)
            .iloc[0]
        )
        coordinates = json.dumps(candidate.__geo_interface__["coordinates"])
    original_requested_start = pd.Timestamp(start_date).date()
    original_requested_end = pd.Timestamp(end_date).date()
    requested_start = original_requested_start
    requested_end = original_requested_end
    satellite_datasets = {
        "SENTINEL2_SR", "HLS_SR", "LANDSAT_SR", "LANDSAT8_SR",
        "MODIS_TERRA_8DAY",
    }
    sparse_optical_datasets = {
        "SENTINEL2_SR", "HLS_SR", "LANDSAT_SR", "LANDSAT8_SR",
    }
    dataset_key = dataset.strip().upper()
    temporal_window_expanded = False
    if (
        dataset_key in sparse_optical_datasets
        and (requested_end - requested_start).days < 31
    ):
        # Sentinel/HLS/Landsat belirli bir takvim gününde görüntü vermek zorunda
        # değildir. Kısa istekler aylık gözlem penceresine çevrilir; böylece
        # analiz rastgele bir uydu geçiş gününe bağlı kalmaz.
        requested_start = requested_start.replace(day=1)
        requested_end = (
            pd.Timestamp(requested_end).to_period("M").end_time.date()
        )
        temporal_window_expanded = True
    request_identity = {
        "geometry": hashlib.sha256(coordinates.encode("utf-8")).hexdigest(),
        "dataset": dataset.strip(),
        "variables": variables.strip(),
        "area_reducer": area_reducer,
        "start": requested_start,
        "end": requested_end,
    }
    final_cache = _cache_file("timeseries", request_identity)
    cached_final = _read_cache(final_cache)
    if (
        isinstance(cached_final, tuple)
        and len(cached_final) == 2
        and isinstance(cached_final[0], pd.DataFrame)
    ):
        cached_data, cached_metadata = cached_final
        cached_metadata = dict(cached_metadata)
        cached_metadata["cache"] = "persistent"
        return cached_data.copy(), cached_metadata
    endpoint = f"{BASE_URL}/timeseries/native/coordinates"
    headers = {"Authorization": api_key.strip()}

    def request_period(period_start, period_end) -> tuple[list[dict], dict[str, Any]]:
        period_cache = _cache_file(
            "timeseries_parts",
            {**request_identity, "start": period_start, "end": period_end},
        )
        cached_period = _read_cache(period_cache)
        if (
            isinstance(cached_period, tuple)
            and len(cached_period) == 2
            and isinstance(cached_period[0], list)
        ):
            records, metadata = cached_period
            return records, {**dict(metadata), "cache": "persistent"}
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
            if _quota_exceeded(response.status_code, error_detail):
                raise ClimateEngineQuotaExceeded(
                    "Climate Engine günlük istek kotası doldu. Zetriklim aynı istekleri "
                    "yeniden göndermeyi durdurdu. Kota yenilendiğinde analiz kaldığı "
                    "yerden devam eder; daha önce önbelleğe alınan dönemler tekrar indirilmez."
                )
            span_days = (period_end - period_start).days
            normalized_error = error_text.casefold()
            no_data_returned = "no data returned" in normalized_error
            if no_data_returned and dataset_key in satellite_datasets:
                # Bulut, yörünge veya sensör aralığı nedeniyle bazı uydu
                # parçalarının boş olması normaldir. Boş parça tüm uzun dönem
                # serisini iptal etmez; diğer gerçek gözlemler korunur.
                result = ([], {
                    "no_data": True,
                    "period": f"{period_start}/{period_end}",
                    "service_response": error_text[:500],
                })
                _write_cache(period_cache, result)
                return result
            aggregation_limit = "too many concurrent aggregations" in normalized_error
            computation_timeout = any(
                marker in normalized_error
                for marker in (
                    "computation timed out",
                    "computation timeout",
                    "computation time out",
                    "earth engine computation timed out",
                )
            )
            splittable_workload_error = (
                "response size exceeds limit" in normalized_error
                or aggregation_limit
                or computation_timeout
            )
            if (
                response.status_code >= 500
                and splittable_workload_error
                # Sentinel-2 gibi yüksek çözünürlüklü koleksiyonlarda büyük bir
                # poligon için 90 günlük parça bile Earth Engine aggregation
                # sınırını aşabiliyor. İstek, yaklaşık iki haftalık güvenli alt
                # parçalara kadar bölünür; her başarılı parça kalıcı önbelleğe
                # yazıldığı için analiz yeniden başladığında tekrarlanmaz.
                and span_days > (14 if (aggregation_limit or computation_timeout) else 31)
            ):
                midpoint = period_start + timedelta(days=span_days // 2)
                left_records, left_meta = request_period(period_start, midpoint)
                right_records, right_meta = request_period(
                    midpoint + timedelta(days=1), period_end
                )
                result = (left_records + right_records, {
                    "split": True,
                    "reason": (
                        "earth_engine_concurrent_aggregation_limit"
                        if aggregation_limit else "response_size_limit"
                        if not computation_timeout else "earth_engine_computation_timeout"
                    ),
                    "parts": [left_meta, right_meta],
                })
                _write_cache(period_cache, result)
                return result
            raise RuntimeError(
                f"Climate Engine isteği başarısız (HTTP {response.status_code}). "
                f"Dataset={dataset}, değişken={variables}, dönem={period_start}/{period_end}, "
                f"geometri={len(coordinates):,} karakter, sadeleştirme={tolerance} m. "
                f"Servis yanıtı: {error_text[:1200]}"
            )
        body = response.json()
        series_groups = body.get("Data")
        if not series_groups:
            result = ([], {
                key: value for key, value in body.items() if key != "Data"
            })
            _write_cache(period_cache, result)
            return result
        first_group = series_groups[0] if isinstance(series_groups, list) else series_groups
        records = first_group.get("Data") if isinstance(first_group, dict) else first_group
        if isinstance(records, dict):
            normalized_records = pd.DataFrame.from_dict(records).to_dict("records")
        else:
            normalized_records = list(records or [])
        result = (normalized_records, {
            key: value for key, value in body.items() if key != "Data"
        })
        _write_cache(period_cache, result)
        return result

    if dataset_key in satellite_datasets and (
        requested_end - requested_start
    ).days > 370:
        # Uydu koleksiyonunda 10–12 yılı tek poligon isteğinde toplamak Earth
        # Engine'in eşzamanlı aggregation kotasını aşıyor. Takvim yılı parçaları
        # sırayla ve kalıcı önbellekle alınır; tamamlanan yıllar tekrar istenmez.
        periods: list[tuple[date, date]] = []
        cursor = requested_start
        while cursor <= requested_end:
            period_end = min(date(cursor.year, 12, 31), requested_end)
            periods.append((cursor, period_end))
            cursor = period_end + timedelta(days=1)
        all_records = []
        chunk_metadata = []
        for period_start, period_end in periods:
            records, metadata = request_period(period_start, period_end)
            all_records.extend(records)
            chunk_metadata.append(
                {
                    "start": str(period_start),
                    "end": str(period_end),
                    "record_count": len(records),
                    "metadata": metadata,
                }
            )
    else:
        # İklim serileri için önce bütün dönemi tek istekte dene. Servis yalnız
        # yanıt boyutu/aggregation sınırı bildirirse request_period dönemi böler.
        all_records, response_metadata = request_period(requested_start, requested_end)
        chunk_metadata = [{
            "start": str(requested_start),
            "end": str(requested_end),
            "record_count": len(all_records),
            "metadata": response_metadata,
        }]

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
        "requested_period": (
            f"{original_requested_start}/{original_requested_end}"
        ),
        "effective_observation_period": f"{requested_start}/{requested_end}",
        "temporal_window_expanded": temporal_window_expanded,
    }
    _write_cache(final_cache, (data.copy(), metadata))
    return data, metadata


def normalize_analysis_column(data: pd.DataFrame, analysis: str) -> pd.DataFrame:
    """Climate Engine'in ürünlere göre değişen değer sütununu ortak analiz adına çevirir."""
    analysis = str(analysis).upper()
    if analysis == "SPI":
        return data
    frame = data.copy()
    identifiers = {"Tarih", "Örnek ID", "Enlem", "Boylam"}
    numeric_columns = [
        column
        for column in frame.columns
        if column not in identifiers
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    exact = [column for column in numeric_columns if str(column).upper() == analysis]
    contains = [column for column in numeric_columns if analysis in str(column).upper()]
    candidates = exact or contains or numeric_columns
    if len(candidates) != 1:
        raise ValueError(
            f"Climate Engine {analysis} yanıtında tek bir analiz değer sütunu belirlenemedi. "
            f"Sayısal sütunlar: {', '.join(map(str, numeric_columns)) or 'yok'}"
        )
    source = candidates[0]
    frame[source] = pd.to_numeric(frame[source], errors="coerce")
    return frame.rename(columns={source: analysis}) if source != analysis else frame


def fetch_map_tile(
    api_key: str,
    gdf: gpd.GeoDataFrame,
    start_date,
    end_date,
    dataset: str,
    variable: str,
    analysis: str,
    *,
    spi_scale_months: int = 3,
    reference_start_year: int | None = None,
    reference_end_year: int | None = None,
    temporal_statistic: str = "mean",
    map_kind: str = "values",
    anomaly_calculation: str = "anom",
) -> tuple[str, dict[str, Any]]:
    """Climate Engine MapID uç noktasından görselleştirme karo adresi alır."""
    bounds = gdf.to_crs(4326).total_bounds.tolist()
    if analysis == "SPI":
        scale = max(int(spi_scale_months), 1)
        target_end = pd.Timestamp(end_date).normalize()
        # Climate Engine standard_index uç noktası start_date/end_date çiftini
        # tek birikim penceresi olarak yorumlar. Tüm arşivi göndermek, örneğin
        # 1981–2025 dönemini 16 bin günlük SPI penceresine dönüştürür. Harita
        # yalnız hedef ayda biten seçili SPI ölçeğini kullanmalıdır; klimatoloji
        # yılları ise start_year/end_year parametrelerinde ayrıca tutulur.
        target_start = (
            target_end.to_period("M").to_timestamp()
            - pd.DateOffset(months=scale - 1)
        )
        baseline_start = int(
            reference_start_year
            if reference_start_year is not None
            else pd.Timestamp(start_date).year
        )
        baseline_end = int(
            reference_end_year
            if reference_end_year is not None
            else target_end.year
        )
        if baseline_end < baseline_start:
            raise ValueError("SPI referans dönemi başlangıcı bitiş yılından büyük olamaz.")
        endpoint = f"{BASE_URL}/raster/mapid/standard_index"
        params = {
            "dataset": dataset,
            "variable": "spi",
            "distribution": "gamma",
            "start_date": target_start.date().isoformat(),
            "end_date": target_end.date().isoformat(),
            "start_year": baseline_start,
            "end_year": baseline_end,
            "bounding_box": str(bounds),
            # PNG renklerinden sayısal CBS rasterı geri çözüldüğü için yarı
            # saydamlık renkleri arka planla karıştırmamalıdır.
            "colormap_opacity": 1.0,
            "colormap_type": "continuous",
        }
    else:
        if map_kind not in {"values", "anomalies"}:
            raise ValueError(f"Desteklenmeyen Climate Engine harita türü: {map_kind}")
        endpoint = f"{BASE_URL}/raster/mapid/{map_kind}"
        params = {
            "dataset": dataset,
            "variable": variable,
            "temporal_statistic": temporal_statistic,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "bounding_box": str(bounds),
            "colormap_opacity": 1.0,
            "colormap_type": "continuous",
        }
        if map_kind == "anomalies":
            params.update(
                {
                    "calculation": anomaly_calculation,
                    "start_year": int(reference_start_year or pd.Timestamp(start_date).year),
                    "end_year": int(reference_end_year or pd.Timestamp(end_date).year),
                }
            )
    visual_style = map_visual_style(analysis)
    if visual_style:
        params["colormap_min_max"] = json.dumps(
            [visual_style["minimum"], visual_style["maximum"]]
        )
        # MapID uç noktası paleti geçerli bir JSON listesi olarak ayrıştırır.
        # Hex değerlerinin tırnaksız gönderilmesi bütün haritalarda HTTP 422
        # `json loads error` yanıtına neden olur.
        params["colormap_palette"] = json.dumps(
            [
                str(color) if str(color).startswith("#") else f"#{color}"
                for color in visual_style["colors"]
            ],
            separators=(",", ":"),
        )

    def request_mapid():
        current_response = None
        for attempt in range(3):
            current_response = requests.get(
                endpoint,
                headers={"Authorization": api_key.strip()},
                params=params,
                timeout=60,
            )
            if current_response.status_code == 500:
                try:
                    response_hint = str(current_response.json())
                except ValueError:
                    response_hint = current_response.text
                if "change the start_year" in response_hint:
                    break
            if (
                current_response.ok
                or current_response.status_code not in {429, 500, 502, 503, 504}
            ):
                break
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        return current_response

    response = request_mapid()
    assert response is not None
    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text.strip()
        # Standard-index uç noktası uzun (181/272/364 günlük) birikimlerde
        # arşivin ilk yılını tam referans yılı kabul etmiyor ve geçerli yılı
        # hata metninde açıkça bildiriyor. Kullanıcıyı başarısız haritayla
        # bırakmak yerine yalnız API'nin önerdiği yıl ile bir kez düzelt.
        if analysis == "SPI":
            suggested_match = re.search(
                r"change the start_year to\s+(\d{4})",
                str(detail),
                flags=re.IGNORECASE,
            )
            if suggested_match:
                suggested_year = int(suggested_match.group(1))
                current_year = int(params["start_year"])
                if current_year < suggested_year <= int(params["end_year"]):
                    params["start_year"] = suggested_year
                    response = request_mapid()
                    assert response is not None
                    if response.ok:
                        detail = None
                    else:
                        try:
                            detail = response.json()
                        except ValueError:
                            detail = response.text.strip()
        if response.ok:
            detail = None
        else:
            if _quota_exceeded(response.status_code, detail):
                raise ClimateEngineQuotaExceeded(
                    "Climate Engine günlük harita istek kotası doldu. Mevcut sonuçlar "
                    "korundu; kota yenilendiğinde yalnız haritaları yeniden oluşturabilirsiniz."
                )
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
        "period": (
            f"{params['start_date']}/{params['end_date']}"
            if analysis == "SPI"
            else f"{start_date}/{end_date}"
        ),
        "spi_scale_months": int(spi_scale_months) if analysis == "SPI" else None,
        "reference_period": (
            f"{params['start_year']}/{params['end_year']}"
            if analysis == "SPI"
            else None
        ),
        "temporal_statistic": temporal_statistic if analysis != "SPI" else None,
        "map_kind": "standard_index" if analysis == "SPI" else map_kind,
        "anomaly_calculation": (
            anomaly_calculation if analysis != "SPI" and map_kind == "anomalies" else None
        ),
        "bounds": bounds,
    }

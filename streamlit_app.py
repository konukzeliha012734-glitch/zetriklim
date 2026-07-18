# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from datetime import date, time, timedelta
import json
import os
from pathlib import Path
import pickle
import re
import time as time_module

import folium
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap, StepColormap
from folium import plugins
from streamlit_folium import st_folium
from rasterio.io import MemoryFile

from zetriklim.catalog import ANALYSES, ANALYSIS_METHODS, SOURCES, VARIABLES
from zetriklim.academic import (
    academic_defaults,
    build_academic_report_html,
    harmonize_monthly_data,
    run_academic_analysis,
    run_remote_sensing_analysis,
    safe_monthly_end,
)
from zetriklim.artifacts import (
    build_academic_chart_suite,
    build_area_map_png,
    build_complete_package,
    build_excel,
    build_raster_png,
    build_spi_thesis_excel,
    build_tile_map_png,
    build_timeseries_png,
    dataframe_to_csv,
)
from zetriklim.climate_engine import (
    connection_label,
    fetch_dataset_date_range,
    fetch_map_tile as fetch_climate_engine_map_tile,
    fetch_timeseries as fetch_climate_engine_timeseries,
    normalize_analysis_column,
    validate_api_key,
)
from zetriklim.exports import build_metadata
from zetriklim.gadm import GADM_VERSION, fetch_gadm, name_column as gadm_name_column
from zetriklim.map_styles import map_visual_style
from zetriklim.geometry import (
    GeometryUploadError,
    UploadedPart,
    inspect_geodataframe,
    inspect_uploaded_files,
)
from zetriklim.gee import (
    LAND_COVER_CLASSES,
    build_climate_geotiff,
    build_chirps_spi_geotiff,
    build_remote_analysis_geotiff,
    fetch_chirps_monthly_mean,
    fetch_gee_academic_series,
    fetch_gee_monthly_climate,
    create_user_auth_flow,
    exchange_user_auth_code,
    initialize_gee,
)
from zetriklim.open_meteo import fetch_centroid_series
from zetriklim.spi import calculate_spi_table
from zetriklim.estimates import estimate_analysis_seconds, format_duration_range


ROOT = Path(__file__).parent

SELECTABLE_ANALYSES = ["SPI", "NDVI", "NDWI", "LST", "EVI"]
CLIMATE_ENGINE_ANALYSES = {"SPI", "NDVI", "EVI", "LST"}
TIME_SERIES_REMOTE_ANALYSES = {"NDVI", "EVI", "LST"}
SPECTRAL_ANALYSES = {"NDVI", "NDWI", "NDMI", "NDBI", "EVI", "SAVI", "LST"}
STATIC_TERRAIN_ANALYSES = {"DEM", "SLOPE", "ASPECT", "TWI"}

st.set_page_config(
    page_title="Zetriklim | Havza ve iklim analizi",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

access_code = os.getenv("ZETRIKLIM_ACCESS_CODE")
if access_code and not st.session_state.get("access_granted"):
    st.title("Zetriklim")
    st.caption("Bu paylaşım GEE kotasını korumak için erişim koduyla sınırlandırılmıştır.")
    entered_code = st.text_input("Erişim kodu", type="password")
    if st.button("Uygulamaya gir", type="primary", use_container_width=True):
        if entered_code == access_code:
            st.session_state.access_granted = True
            st.rerun()
        else:
            st.error("Erişim kodu hatalı.")
    st.stop()


def cached_gee_status(project: str | None) -> tuple[bool, str]:
    return initialize_gee(project)


def add_cartographic_controls(
    fmap: folium.Map,
    analysis: str,
    source: str,
    start_date,
    end_date,
) -> None:
    """Analiz haritasına CBS lejantı, ölçek, koordinat ve kuzey oku ekler."""
    visual_style = map_visual_style(analysis)
    colormap = None
    if visual_style:
        colormap_args = {
            "colors": visual_style["colors"],
            "vmin": visual_style["minimum"],
            "vmax": visual_style["maximum"],
            "caption": visual_style["caption"],
        }
        colormap = (
            StepColormap(index=visual_style["index"], **colormap_args)
            if visual_style.get("index")
            else LinearColormap(**colormap_args)
        )
    if colormap is not None:
        colormap.add_to(fmap)
    plugins.Fullscreen(
        position="topleft",
        title="Tam ekran",
        title_cancel="Tam ekrandan çık",
        force_separate_button=True,
    ).add_to(fmap)
    plugins.MousePosition(
        position="bottomright",
        separator=" · ",
        prefix="Koordinat",
        num_digits=5,
    ).add_to(fmap)
    north_arrow = """
    <div style="position:fixed;right:18px;top:85px;z-index:9999;background:white;
    border:1px solid #315b64;border-radius:8px;padding:5px 8px;text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,.16);font:700 13px sans-serif;color:#063447">
      N<br><span style="font-size:24px;line-height:20px">↑</span>
    </div>
    """
    title_box = f"""
    <div style="position:fixed;left:52px;top:10px;z-index:9999;background:rgba(255,255,255,.94);
    border-left:4px solid #00a6a6;border-radius:6px;padding:7px 11px;
    box-shadow:0 2px 8px rgba(0,0,0,.12);font:12px sans-serif;color:#063447">
      <b>{analysis} mekânsal analiz haritası</b><br>
      Kaynak: {source}<br>Dönem: {start_date} – {end_date}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(north_arrow + title_box))


def add_tile_loading_status(
    fmap: folium.Map,
    tile_layer: folium.TileLayer,
    label: str,
) -> None:
    """Yavaş veya hatalı karo yüklenmesini kullanıcıya açıkça bildirir."""
    status_id = f"tile-status-{tile_layer.get_name().replace('_', '-')}"
    map_name = fmap.get_name()
    layer_name = tile_layer.get_name()
    label_json = json.dumps(label, ensure_ascii=False)
    control = f"""
    <div id="{status_id}" style="position:fixed;left:50%;bottom:22px;transform:translateX(-50%);
    z-index:9999;background:rgba(255,255,255,.96);border:1px solid #315b64;border-radius:8px;
    padding:7px 11px;box-shadow:0 2px 8px rgba(0,0,0,.16);font:12px sans-serif;color:#063447">
      Harita verisi yükleniyor…
    </div>
    <script>
    (function attachTileStatus(attempt) {{
      if (typeof {layer_name} === 'undefined' || typeof {map_name} === 'undefined') {{
        if (attempt < 40) window.setTimeout(function() {{ attachTileStatus(attempt + 1); }}, 250);
        return;
      }}
      var status = document.getElementById('{status_id}');
      var loaded = 0, failed = 0, slowTimer = null;
      var layerLabel = {label_json};
      function show(message, error) {{
        status.textContent = message;
        status.style.display = 'block';
        status.style.borderColor = error ? '#b71c1c' : '#315b64';
      }}
      {layer_name}.on('loading', function() {{
        loaded = 0; failed = 0;
        show(layerLabel + ' yükleniyor…', false);
        window.clearTimeout(slowTimer);
        slowTimer = window.setTimeout(function() {{
          show('Bağlantı yavaş; harita karoları bekleniyor…', false);
        }}, 6000);
      }});
      {layer_name}.on('tileload', function() {{ loaded += 1; }});
      {layer_name}.on('tileerror', function() {{ failed += 1; }});
      {layer_name}.on('load', function() {{
        window.clearTimeout(slowTimer);
        if (failed > 0) show('Harita kısmen yüklendi: ' + failed + ' karo alınamadı.', true);
        else {{
          show(layerLabel + ' hazır · ' + loaded + ' karo doğrulandı.', false);
          window.setTimeout(function() {{ status.style.display = 'none'; }}, 2500);
        }}
      }});
    }})(0);
    </script>
    """
    fmap.get_root().html.add_child(folium.Element(control))


def climate_engine_map_specs(
    selected_analysis: str,
    dataset: str,
    variable: str,
    analysis_product_start_date: date,
    map_start: date,
    scales: list[int],
) -> list[dict[str, object]]:
    """Seçilen harita dönemi için üretilecek bilimsel mekânsal ürünleri tanımlar."""
    climate_context = [
        {
            "slug": "ortalama-gunluk-yagis-dagilimi",
            "label": "Seçili dönem ortalama günlük yağış dağılımı",
            "dataset": "CHIRPS_DAILY",
            "variable": "precipitation",
            "analysis": "Yağış",
            "start": max(map_start, date(1981, 1, 1)),
            "statistic": "mean",
        },
        {
            "slug": "ortalama-sicaklik-dagilimi",
            "label": "Seçili dönem ortalama sıcaklık dağılımı",
            "dataset": "ERA5_AG",
            "variable": "mean_2m_air_temperature",
            "analysis": "Sıcaklık",
            "start": max(map_start, date(1979, 1, 1)),
            "statistic": "mean",
        },
    ]
    if selected_analysis != "SPI":
        return [
            {
                "slug": selected_analysis.lower(),
                "label": f"{selected_analysis} dağılımı",
                "dataset": dataset,
                "variable": variable,
                "analysis": selected_analysis,
                "start": max(map_start, analysis_product_start_date),
                "statistic": "mean",
            },
            *climate_context,
        ]
    spi_specs = [
        {
            "slug": f"kuraklik-dagilimi-spi-{scale}",
            "label": f"Kuraklık dağılımı · SPI-{scale}",
            "dataset": dataset,
            "variable": variable,
            "analysis": "SPI",
            "start": map_start,
            "statistic": "mean",
            "spi_scale": scale,
        }
        for scale in sorted({int(value) for value in scales} or {3})
    ]
    return [*spi_specs, *climate_context]


def _attach_expected_map_means(
    specs: list[dict[str, object]],
    data: pd.DataFrame | None,
    period_start: date,
    period_end: date,
) -> list[dict[str, object]]:
    """Renkli MapID çıktısını bağımsız sayısal zaman serisine karşı sınar."""
    enriched = [dict(spec) for spec in specs]
    if not isinstance(data, pd.DataFrame) or data.empty:
        return enriched
    dated = data.copy()
    if "Tarih" in dated:
        dated["Tarih"] = pd.to_datetime(dated["Tarih"], errors="coerce")
        dated = dated.dropna(subset=["Tarih"])
    rain_column = next(
        (column for column in dated.columns if "toplam yağış" in str(column).lower()),
        None,
    )
    temperature_column = next(
        (column for column in dated.columns if "ortalama sıcaklık" in str(column).lower()),
        None,
    )
    expected: dict[str, float] = {}
    if rain_column is not None:
        rain = pd.to_numeric(dated[rain_column], errors="coerce")
        # Climate Engine serisi aylık toplamdır; aynı ay için birden fazla obje
        # varsa önce alan ortalaması alınır, sonra toplam gün sayısına bölünür.
        if "Tarih" in dated:
            monthly = rain.groupby(dated["Tarih"].dt.to_period("M")).mean()
        else:
            monthly = rain
        day_count = max((period_end - period_start).days + 1, 1)
        value = float(monthly.sum(min_count=1) / day_count)
        if np.isfinite(value):
            expected["Yağış"] = value
    if temperature_column is not None:
        temperature = pd.to_numeric(dated[temperature_column], errors="coerce")
        value = float(temperature.mean())
        if np.isfinite(value):
            expected["Sıcaklık"] = value
    for spec in enriched:
        analysis = str(spec.get("analysis"))
        if analysis in expected:
            spec["expected_area_mean"] = round(expected[analysis], 6)
    return enriched


MAP_CACHE_VERSION = 13
MAP_CACHE_ROOT = Path(
    os.getenv("LOCALAPPDATA", str(Path.home() / ".cache"))
) / "Zetriklim" / "map_artifacts"
RESULT_CACHE_VERSION = 6
RESULT_CACHE_ROOT = Path(
    os.getenv("LOCALAPPDATA", str(Path.home() / ".cache"))
) / "Zetriklim" / "analysis_results"


def _analysis_result_cache_path(config: tuple[object, ...]) -> Path:
    payload = json.dumps(
        {"version": RESULT_CACHE_VERSION, "config": config},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return RESULT_CACHE_ROOT / f"{hashlib.sha256(payload).hexdigest()}.pkl"


def _read_analysis_result_cache(path: Path) -> dict[str, object] | None:
    try:
        if path.is_file():
            with path.open("rb") as stream:
                value = pickle.load(stream)
            if isinstance(value, dict) and value.get("version") == RESULT_CACHE_VERSION:
                return value
    except (OSError, EOFError, pickle.PickleError):
        return None
    return None


def _write_analysis_result_cache(path: Path, value: dict[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    except OSError:
        return


def _safe_archive_component(value: object, fallback: str) -> str:
    """İnsan tarafından okunabilir, işletim sistemiyle uyumlu dosya adı parçası."""
    text = str(value or "").strip()
    replacements = str.maketrans(
        "çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU"
    )
    text = text.translate(replacements)
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return text[:80] or fallback


def _area_display_name(value: object) -> str:
    text = str(value or "").strip()
    if " · " in text and text.startswith("GADM"):
        text = text.rsplit(" · ", 1)[-1]
    return text or "Çalışma alanı"


def _list_analysis_result_cache(limit: int = 30) -> list[dict[str, object]]:
    """Başarıyla tamamlanmış kalıcı analizleri kullanıcıya gösterilecek biçimde listeler."""
    if not RESULT_CACHE_ROOT.is_dir():
        return []
    records: list[dict[str, object]] = []
    paths = sorted(
        RESULT_CACHE_ROOT.glob("*.pkl"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[:limit]
    for path in paths:
        cached = _read_analysis_result_cache(path)
        if not cached:
            continue
        config = cached.get("output_config")
        if not isinstance(config, tuple) or len(config) < 8:
            continue
        files = cached.get("output_files")
        file_count = len(files) if isinstance(files, dict) else 0
        total_bytes = (
            sum(len(content) for content in files.values() if isinstance(content, bytes))
            if isinstance(files, dict) else 0
        )
        records.append(
            {
                "cache_path": path,
                "analysis": str(config[2]),
                "provider": str(config[3]),
                "product": str(config[4]),
                "start": str(config[5]),
                "end": str(config[6]),
                "area_km2": config[7],
                "created": pd.Timestamp(path.stat().st_mtime, unit="s"),
                "file_count": file_count,
                "size_mb": total_bytes / (1024 * 1024),
                "elapsed_seconds": cached.get("output_elapsed_seconds"),
                "area_name": _area_display_name(cached.get("area_label")),
            }
        )
        record = records[-1]
        record["file_name"] = (
            f"{_safe_archive_component(record['area_name'], 'calisma-alani')}-"
            f"{_safe_archive_component(record['analysis'], 'analiz')}-"
            f"{record['start']}-{record['end']}.zip"
        )
    return records


def _map_artifact_cache_path(
    summary,
    spec: dict[str, object],
    effective_start: date,
    effective_end: date,
    reference_start: int,
    reference_end: int,
) -> Path:
    geometry = summary.gdf_wgs84.to_crs(4326).geometry.union_all()
    identity = {
        "version": MAP_CACHE_VERSION,
        "geometry_sha256": hashlib.sha256(geometry.wkb).hexdigest(),
        "spec": spec,
        "effective_start": effective_start,
        "effective_end": effective_end,
        "reference_start": reference_start,
        "reference_end": reference_end,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return MAP_CACHE_ROOT / f"{hashlib.sha256(encoded).hexdigest()}.pkl"


def _read_map_artifact_cache(path: Path) -> dict[str, object] | None:
    try:
        if path.is_file():
            with path.open("rb") as stream:
                value = pickle.load(stream)
            if isinstance(value, dict) and value.get("version") == MAP_CACHE_VERSION:
                return value
    except (OSError, EOFError, pickle.PickleError):
        return None
    return None


def _write_map_artifact_cache(path: Path, value: dict[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    except OSError:
        return


def _build_direct_geotiff_artifact(
    *,
    raster_tif: bytes,
    summary,
    spec: dict[str, object],
    effective_start: date,
    effective_end: date,
    source: str,
    method: str,
) -> tuple[dict[str, bytes], dict[str, object], dict[str, object]]:
    """Gerçek sayısal GeoTIFF'ten kayıpsız harita çıktısı ve QA özeti üretir."""
    analysis = str(spec["analysis"])
    with MemoryFile(raster_tif) as memory:
        with memory.open() as dataset:
            values = dataset.read(1).astype("float64")
            nodata = dataset.nodata
            valid = np.isfinite(values)
            if nodata is not None:
                valid &= values != float(nodata)
            finite = values[valid]
            if not len(finite):
                raise RuntimeError(f"{analysis} GeoTIFF rasterında geçerli piksel bulunamadı.")
            if analysis == "SPI":
                saturation_fraction = float(np.mean(np.abs(finite) >= 2.95))
                if saturation_fraction >= 0.95:
                    raise RuntimeError(
                        "Piksel bazlı SPI rasterı da ±3 sınırında doygun döndü; "
                        "bilimsel kalite kontrolü nedeniyle yayımlanmadı."
                    )
            else:
                saturation_fraction = None
            if dataset.crs and dataset.crs.is_geographic:
                grid_size_m = abs(float(dataset.transform.a)) * 111_320.0
            else:
                grid_size_m = abs(float(dataset.transform.a))

    slug = str(spec["slug"])
    png_name = f"{slug}.png"
    tif_name = f"{slug}.tif"
    direct_styles = {
        "SPI": ("RdBu", "Standartlaştırılmış indis", (-3.0, 3.0)),
        "NDVI": ("YlGn", "NDVI", (-1.0, 1.0)),
        "NDWI": ("Blues", "NDWI", (-1.0, 1.0)),
        "EVI": ("YlGn", "EVI", (-1.0, 1.0)),
        "LST": ("inferno", "Yüzey sıcaklığı (°C)", None),
        "Yağış": ("Blues", "Ortalama günlük yağış (mm/gün)", None),
        "Sıcaklık": ("Spectral_r", "Ortalama hava sıcaklığı (°C)", None),
    }
    palette, colorbar_label, fixed_range = direct_styles.get(
        analysis, ("viridis", analysis, None)
    )
    png_bytes = build_raster_png(
        raster_tif,
        str(spec["label"]),
        boundary=summary.gdf_wgs84,
        palette=palette,
        colorbar_label=colorbar_label,
        fixed_range=fixed_range,
    )
    quality = {
        "status": "uygun",
        "coverage_ratio": 1.0,
        "value_min": round(float(np.nanmin(finite)), 4),
        "value_max": round(float(np.nanmax(finite)), 4),
        "value_mean": round(float(np.nanmean(finite)), 4),
        "value_std": round(float(np.nanstd(finite)), 4),
        "spi_saturation_fraction": (
            round(saturation_fraction, 4) if saturation_fraction is not None else None
        ),
        "export_grid_size_m": round(grid_size_m, 2),
        "source_native_resolution_m": round(grid_size_m, 2),
        "generation_method": method,
    }
    files = {png_name: png_bytes, tif_name: raster_tif}
    layer = {
        "label": spec["label"],
        "analysis": analysis,
        "source": source,
        "start": str(effective_start),
        "end": str(effective_end),
        "tile_url": None,
        "quality": quality,
        "html_file": None,
        "png_file": png_name,
        "geotiff_file": tif_name,
        "shp_file": None,
    }
    metadata = {
        "label": spec["label"],
        "dataset": spec["dataset"],
        "variable": spec["variable"],
        "analysis": analysis,
        "period": f"{effective_start}/{effective_end}",
        "source": source,
        "method": method,
        "png_file": png_name,
        "geotiff_file": tif_name,
        "tile_quality": quality,
    }
    return files, layer, metadata


def _regenerate_climate_engine_maps_sequential(
    *,
    api_key: str,
    summary,
    specs: list[dict[str, object]],
    map_end: date,
    reference_start: int,
    reference_end: int,
    mapid_attempts: int = 3,
    show_progress: bool = True,
    gee_project: str | None = None,
    cloud_limit: int = 30,
) -> tuple[dict[str, bytes], list[dict[str, object]], list[str]]:
    """Tablo ve bilimsel analizleri değiştirmeden yalnız Climate Engine haritalarını üretir."""
    files: dict[str, bytes] = {}
    layers: list[dict[str, object]] = []
    errors: list[str] = []
    metadata_list: list[dict[str, object]] = []
    dataset_ranges: dict[str, tuple[date, date] | None] = {}
    progress = st.progress(0, text="Haritalar hazırlanıyor…") if show_progress else None
    for index, spec in enumerate(specs, start=1):
        map_started = time_module.perf_counter()
        if progress is not None:
            progress.progress(
                (index - 1) / max(len(specs), 1),
                text=f"{spec['label']} hazırlanıyor ({index}/{len(specs)})…",
            )
        dataset_name = str(spec["dataset"])
        try:
            preliminary_start = spec["start"]
            preliminary_baseline_end = max(
                reference_start,
                min(reference_end, map_end.year - 1),
            )
            preliminary_cache_path = _map_artifact_cache_path(
                summary,
                spec,
                preliminary_start,
                map_end,
                reference_start,
                preliminary_baseline_end,
            )
            preliminary_cached = _read_map_artifact_cache(preliminary_cache_path)
            if preliminary_cached:
                cached_files = preliminary_cached.get("files")
                cached_layer = preliminary_cached.get("layer")
                cached_metadata = preliminary_cached.get("metadata")
                if isinstance(cached_files, dict) and isinstance(cached_layer, dict) and isinstance(cached_metadata, dict):
                    files.update(cached_files)
                    cached_layer = dict(cached_layer)
                    cached_layer["tile_url"] = None
                    cached_layer["quality"] = {
                        **dict(cached_layer.get("quality") or {}),
                        "cache": "persistent",
                    }
                    layers.append(cached_layer)
                    metadata_list.append({**cached_metadata, "tile_quality": cached_layer["quality"]})
                    if progress is not None:
                        progress.progress(
                            index / max(len(specs), 1),
                            text=f"Önbellekten yüklendi ({index}/{len(specs)})",
                        )
                    continue
            if dataset_name not in dataset_ranges:
                try:
                    dataset_ranges[dataset_name] = fetch_dataset_date_range(api_key, dataset_name)
                except Exception:
                    dataset_ranges[dataset_name] = None
            dataset_range = dataset_ranges[dataset_name]
            effective_start = spec["start"]
            effective_end = map_end
            if dataset_range:
                effective_start = max(effective_start, dataset_range[0])
                effective_end = min(effective_end, dataset_range[1])
            if effective_end < effective_start:
                raise ValueError(f"{dataset_name} için seçilen harita döneminde veri yok.")
            baseline_end = max(reference_start, min(reference_end, effective_end.year - 1))
            cache_path = _map_artifact_cache_path(
                summary,
                spec,
                effective_start,
                effective_end,
                reference_start,
                baseline_end,
            )
            cached_artifact = _read_map_artifact_cache(cache_path)
            if cached_artifact:
                cached_files = cached_artifact.get("files")
                cached_layer = cached_artifact.get("layer")
                cached_metadata = cached_artifact.get("metadata")
                if (
                    isinstance(cached_files, dict)
                    and isinstance(cached_layer, dict)
                    and isinstance(cached_metadata, dict)
                ):
                    files.update(cached_files)
                    cached_layer = dict(cached_layer)
                    cached_layer["tile_url"] = None
                    cached_layer["quality"] = {
                        **dict(cached_layer.get("quality") or {}),
                        "cache": "persistent",
                    }
                    layers.append(cached_layer)
                    metadata_list.append(
                        {
                            **cached_metadata,
                            "tile_quality": cached_layer["quality"],
                        }
                    )
                    if progress is not None:
                        progress.progress(
                            index / max(len(specs), 1),
                            text=f"Önbellekten yüklendi ({index}/{len(specs)})",
                        )
                    continue
            direct_spi_error = None
            direct_climate_error = None
            if str(spec.get("analysis")) == "SPI":
                try:
                    spi_tif = build_chirps_spi_geotiff(
                        summary.gdf_wgs84,
                        effective_end,
                        int(spec.get("spi_scale", 3)),
                        baseline_start=reference_start,
                        baseline_end=baseline_end,
                        project=gee_project or None,
                    )
                    direct_files, direct_layer, direct_metadata = (
                        _build_direct_geotiff_artifact(
                            raster_tif=spi_tif,
                            summary=summary,
                            spec=spec,
                            effective_start=effective_start,
                            effective_end=effective_end,
                            source="CHIRPS Daily · Google Earth Engine",
                            method="Piksel bazlı Gamma SPI · sıfır olasılığı düzeltmeli",
                        )
                    )
                    files.update(direct_files)
                    layers.append(direct_layer)
                    metadata_list.append(direct_metadata)
                    cache_value = {
                        "version": MAP_CACHE_VERSION,
                        "files": direct_files,
                        "layer": direct_layer,
                        "metadata": direct_metadata,
                    }
                    _write_map_artifact_cache(cache_path, cache_value)
                    if preliminary_cache_path != cache_path:
                        _write_map_artifact_cache(preliminary_cache_path, cache_value)
                    continue
                except Exception as error:
                    # Earth Engine bağlı değilse Climate Engine standard_index
                    # yolu gerçek veriye dayalı yedek olarak kullanılmaya devam eder.
                    direct_spi_error = error
            if str(spec.get("analysis")) in {"Yağış", "Sıcaklık"}:
                try:
                    climate_variable = (
                        "precipitation"
                        if str(spec.get("analysis")) == "Yağış"
                        else "temperature"
                    )
                    climate_tif = build_climate_geotiff(
                        summary.gdf_wgs84,
                        effective_start,
                        effective_end,
                        climate_variable,
                        project=gee_project or None,
                    )
                    direct_files, direct_layer, direct_metadata = (
                        _build_direct_geotiff_artifact(
                            raster_tif=climate_tif,
                            summary=summary,
                            spec=spec,
                            effective_start=effective_start,
                            effective_end=effective_end,
                            source=(
                                "CHIRPS Daily · Google Earth Engine"
                                if climate_variable == "precipitation"
                                else "ERA5-Land Daily · Google Earth Engine"
                            ),
                            method=(
                                "Doğrudan sayısal günlük ortalama GeoTIFF"
                                if climate_variable == "precipitation"
                                else "Doğrudan sayısal dönem ortalaması GeoTIFF"
                            ),
                        )
                    )
                    files.update(direct_files)
                    layers.append(direct_layer)
                    metadata_list.append(direct_metadata)
                    cache_value = {
                        "version": MAP_CACHE_VERSION,
                        "files": direct_files,
                        "layer": direct_layer,
                        "metadata": direct_metadata,
                    }
                    _write_map_artifact_cache(cache_path, cache_value)
                    if preliminary_cache_path != cache_path:
                        _write_map_artifact_cache(preliminary_cache_path, cache_value)
                    continue
                except Exception as error:
                    direct_climate_error = error
            final_error = None
            critical_climate_map = str(spec.get("analysis")) in {"Yağış", "Sıcaklık"}
            attempt_count = max(
                3 if critical_climate_map else 1,
                int(mapid_attempts),
            )
            for map_attempt in range(attempt_count):
                try:
                    tile_url, map_metadata = fetch_climate_engine_map_tile(
                        api_key,
                        summary.gdf_wgs84,
                        effective_start,
                        effective_end,
                        dataset_name,
                        str(spec["variable"]),
                        str(spec["analysis"]),
                        spi_scale_months=int(spec.get("spi_scale", 3)),
                        reference_start_year=reference_start,
                        reference_end_year=baseline_end,
                        temporal_statistic=str(spec["statistic"]),
                        map_kind=str(spec.get("map_kind", "values")),
                        anomaly_calculation=str(spec.get("anomaly_calculation", "anom")),
                    )
                    static_png, quality = build_tile_map_png(
                        tile_url,
                        summary.gdf_wgs84,
                        title=str(spec["label"]),
                        analysis=str(spec["analysis"]),
                        source=f"Climate Engine · {dataset_name} · {spec['variable']}",
                        period=str(map_metadata["period"]),
                    )
                    quality["generation_seconds"] = round(
                        time_module.perf_counter() - map_started, 2
                    )
                    expected_mean = spec.get("expected_area_mean")
                    observed_mean = quality.get("value_mean")
                    if expected_mean is not None and observed_mean is not None:
                        expected_mean = float(expected_mean)
                        observed_mean = float(observed_mean)
                        tolerance = (
                            max(0.45, abs(expected_mean) * 0.35)
                            if str(spec.get("analysis")) == "Yağış"
                            else max(2.5, abs(expected_mean) * 0.25)
                        )
                        if abs(observed_mean - expected_mean) > tolerance:
                            raise RuntimeError(
                                "Renkli MapID rasterının alan ortalaması bağımsız sayısal "
                                f"seriyle uyuşmuyor (harita {observed_mean:.3f}, seri "
                                f"{expected_mean:.3f}). Eksik/yanlış görünen raster "
                                "akademik kalite kontrolü nedeniyle yayımlanmadı."
                            )
                    break
                except Exception as error:
                    final_error = error
                    if map_attempt + 1 < attempt_count:
                        time_module.sleep(2.0)
            else:
                direct_remote_error = None
                if str(spec.get("analysis")) in {"NDVI", "NDWI", "EVI", "LST"}:
                    try:
                        remote_tif, _ = build_remote_analysis_geotiff(
                            summary.gdf_wgs84,
                            effective_start,
                            effective_end,
                            str(spec["analysis"]),
                            project=gee_project or None,
                            cloud_limit=int(cloud_limit),
                        )
                        direct_files, direct_layer, direct_metadata = (
                            _build_direct_geotiff_artifact(
                                raster_tif=remote_tif,
                                summary=summary,
                                spec=spec,
                                effective_start=effective_start,
                                effective_end=effective_end,
                                source=f"Google Earth Engine · {spec['analysis']}",
                                method="Doğrudan sayısal uydu GeoTIFF yedeği",
                            )
                        )
                        files.update(direct_files)
                        layers.append(direct_layer)
                        metadata_list.append(direct_metadata)
                        cache_value = {
                            "version": MAP_CACHE_VERSION,
                            "files": direct_files,
                            "layer": direct_layer,
                            "metadata": direct_metadata,
                        }
                        _write_map_artifact_cache(cache_path, cache_value)
                        if preliminary_cache_path != cache_path:
                            _write_map_artifact_cache(preliminary_cache_path, cache_value)
                        continue
                    except Exception as error:
                        direct_remote_error = error
                fallback_note = (
                    f" Piksel bazlı CHIRPS yedeği de üretilemedi: {direct_spi_error}"
                    if direct_spi_error is not None else ""
                )
                if direct_remote_error is not None:
                    fallback_note += (
                        f" Doğrudan sayısal uydu yedeği de üretilemedi: "
                        f"{direct_remote_error}"
                    )
                if direct_climate_error is not None:
                    fallback_note += (
                        " Doğrudan sayısal iklim GeoTIFF yedeği de üretilemedi: "
                        f"{direct_climate_error}"
                    )
                raise RuntimeError(f"{final_error}{fallback_note}")

            overlay_png = quality.pop("_overlay_png")
            overlay_bounds = quality.pop("_overlay_bounds")
            geotiff_bytes = quality.pop("_geotiff_bytes")
            shp_bytes = quality.pop("_classified_shp_bytes")
            slug = str(spec["slug"])
            png_name = f"{slug}.png"
            html_name = f"{slug}-etkilesimli-harita.html"
            tif_name = f"{slug}.tif"
            shp_name = f"{slug}-shp.zip"
            files.update(
                {
                    png_name: static_png,
                    tif_name: geotiff_bytes,
                    shp_name: shp_bytes,
                }
            )
            fmap = folium.Map(summary.centroid, zoom_start=8, tiles="CartoDB positron", control_scale=True)
            overlay_url = "data:image/png;base64," + base64.b64encode(overlay_png).decode("ascii")
            folium.raster_layers.ImageOverlay(
                image=overlay_url,
                bounds=overlay_bounds,
                name=f"{spec['label']} · havzaya kırpılmış",
                opacity=1.0,
                zindex=2,
            ).add_to(fmap)
            folium.GeoJson(
                summary.gdf_wgs84.__geo_interface__,
                name="Havza sınırı",
                style_function=lambda _: {"color": "#052f42", "weight": 3.5, "fillOpacity": 0.0},
            ).add_to(fmap)
            area_bounds = summary.bounds
            fmap.fit_bounds([[area_bounds[1], area_bounds[0]], [area_bounds[3], area_bounds[2]]])
            add_cartographic_controls(
                fmap,
                str(spec["analysis"]),
                f"Climate Engine · {dataset_name}",
                str(map_metadata["period"]).split("/")[0],
                str(map_metadata["period"]).split("/")[1],
            )
            folium.LayerControl(collapsed=False).add_to(fmap)
            files[html_name] = fmap.get_root().render().encode("utf-8")
            map_metadata.update(
                {
                    "label": spec["label"],
                    "file": html_name,
                    "png_file": png_name,
                    "geotiff_file": tif_name,
                    "shp_file": shp_name,
                    "tile_quality": quality,
                }
            )
            metadata_list.append(map_metadata)
            layer_record = {
                "label": spec["label"],
                "analysis": spec["analysis"],
                "source": f"Climate Engine · {dataset_name}",
                "start": str(map_metadata["period"]).split("/")[0],
                "end": str(map_metadata["period"]).split("/")[1],
                "tile_url": tile_url,
                "quality": quality,
                "html_file": html_name,
                "png_file": png_name,
                "overlay_png": overlay_png,
                "overlay_bounds": overlay_bounds,
                "geotiff_file": tif_name,
                "shp_file": shp_name,
            }
            layers.append(layer_record)
            artifact_file_names = {
                png_name, html_name, tif_name, shp_name,
            }
            cache_value = {
                "version": MAP_CACHE_VERSION,
                "files": {
                    name: content for name, content in files.items()
                    if name in artifact_file_names
                },
                "layer": {**layer_record, "tile_url": None},
                "metadata": map_metadata,
            }
            _write_map_artifact_cache(cache_path, cache_value)
            if preliminary_cache_path != cache_path:
                _write_map_artifact_cache(preliminary_cache_path, cache_value)
        except Exception as error:
            message = f"{spec['label']}: {error}"
            errors.append(message)
            layers.append(
                {
                    "label": spec["label"],
                    "analysis": spec["analysis"],
                    "source": f"Climate Engine · {dataset_name}",
                    "start": str(spec["start"]),
                    "end": str(map_end),
                    "quality": {"status": "basarisiz", "error": str(error)},
                    "tile_url": None,
                }
            )
        if progress is not None:
            progress.progress(
                index / max(len(specs), 1),
                text=f"Haritalar hazırlanıyor ({index}/{len(specs)})",
            )
    if progress is not None:
        progress.empty()
    if metadata_list:
        files["climate-engine-harita-metadata.json"] = json.dumps(
            metadata_list, ensure_ascii=False, indent=2, default=str
        ).encode("utf-8")
        files["harita-kalite-kontrol.json"] = json.dumps(
            [
                {
                    "harita": item.get("label"),
                    "dataset": item.get("dataset"),
                    "degisken": item.get("variable"),
                    "donem": item.get("period"),
                    "kalite": item.get("tile_quality"),
                }
                for item in metadata_list
            ],
            ensure_ascii=False,
            indent=2,
            default=str,
        ).encode("utf-8")
    return files, layers, errors


def regenerate_climate_engine_maps(
    *,
    api_key: str,
    summary,
    specs: list[dict[str, object]],
    map_end: date,
    reference_start: int,
    reference_end: int,
    mapid_attempts: int = 3,
    gee_project: str | None = None,
    cloud_limit: int = 30,
) -> tuple[dict[str, bytes], list[dict[str, object]], list[str]]:
    """Bağımsız haritaları sınırlı eşzamanlılıkla üretir.

    Climate Engine MapID hesapları birbirinden bağımsızdır. Bunları sırayla
    beklemek özellikle 30–45 yıllık yağış ve sıcaklık haritalarında toplam
    süreyi gereksiz biçimde topluyordu. Üç işçi API'yi aşırı yüklemeden bu
    beklemeleri üst üste bindirir; karo ve kalite kontrolleri aynen korunur.
    """
    # Piksel bazlı SPI yığınları aynı anda başlatılırsa Earth Engine
    # "Too many concurrent aggregations" hatası verebilir. SPI ölçeklerini
    # sırayla üretmek daha yavaş ama deterministik ve eksiksizdir.
    if len(specs) <= 1 or any(spec.get("analysis") == "SPI" for spec in specs):
        return _regenerate_climate_engine_maps_sequential(
            api_key=api_key,
            summary=summary,
            specs=specs,
            map_end=map_end,
            reference_start=reference_start,
            reference_end=reference_end,
            mapid_attempts=mapid_attempts,
            show_progress=True,
            gee_project=gee_project,
            cloud_limit=cloud_limit,
        )

    files: dict[str, bytes] = {}
    layers_by_label: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    metadata_by_label: dict[str, dict[str, object]] = {}
    quality_by_label: dict[str, dict[str, object]] = {}
    progress = st.progress(0, text="Haritalar eşzamanlı hazırlanıyor…")
    worker_count = min(3, len(specs))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _regenerate_climate_engine_maps_sequential,
                api_key=api_key,
                summary=summary,
                specs=[spec],
                map_end=map_end,
                reference_start=reference_start,
                reference_end=reference_end,
                mapid_attempts=mapid_attempts,
                show_progress=False,
                gee_project=gee_project,
                cloud_limit=cloud_limit,
            ): spec
            for spec in specs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            spec = futures[future]
            label = str(spec["label"])
            try:
                part_files, part_layers, part_errors = future.result()
                metadata_bytes = part_files.pop(
                    "climate-engine-harita-metadata.json", None
                )
                quality_bytes = part_files.pop("harita-kalite-kontrol.json", None)
                files.update(part_files)
                errors.extend(part_errors)
                for layer in part_layers:
                    layers_by_label[str(layer.get("label") or label)] = layer
                if metadata_bytes:
                    for item in json.loads(metadata_bytes.decode("utf-8")):
                        metadata_by_label[str(item.get("label") or label)] = item
                if quality_bytes:
                    for item in json.loads(quality_bytes.decode("utf-8")):
                        quality_by_label[str(item.get("harita") or label)] = item
            except Exception as error:
                errors.append(f"{label}: {error}")
                layers_by_label[label] = {
                    "label": label,
                    "analysis": spec.get("analysis"),
                    "source": f"Climate Engine · {spec.get('dataset')}",
                    "start": str(spec.get("start")),
                    "end": str(map_end),
                    "quality": {"status": "basarisiz", "error": str(error)},
                    "tile_url": None,
                }
            progress.progress(
                completed / len(specs),
                text=f"Haritalar hazırlanıyor ({completed}/{len(specs)})",
            )
    progress.empty()

    ordered_labels = [str(spec["label"]) for spec in specs]
    layers = [layers_by_label[label] for label in ordered_labels if label in layers_by_label]
    successful_spi_layers = [
        layer for layer in layers
        if layer.get("analysis") == "SPI" and layer.get("png_file")
    ]
    if len(successful_spi_layers) >= 3:
        signatures = [
            (
                float((layer.get("quality") or {}).get("value_mean", float("nan"))),
                float((layer.get("quality") or {}).get("value_std", float("nan"))),
                float((layer.get("quality") or {}).get("value_min", float("nan"))),
                float((layer.get("quality") or {}).get("value_max", float("nan"))),
            )
            for layer in successful_spi_layers
        ]
        signature_array = pd.DataFrame(
            signatures, columns=["mean", "std", "min", "max"]
        )
        indistinguishable = (
            signature_array.notna().all().all()
            and float(signature_array["mean"].max() - signature_array["mean"].min()) < 0.03
            and float(signature_array["std"].max() - signature_array["std"].min()) < 0.02
            and float(signature_array["min"].max() - signature_array["min"].min()) < 0.05
            and float(signature_array["max"].max() - signature_array["max"].min()) < 0.05
        )
        if indistinguishable:
            duplicate_message = (
                "SPI ölçekleri akademik kalite kontrolünden geçmedi: en az üç "
                "farklı birikim ölçeği mekânsal olarak ayırt edilemeyecek kadar "
                "aynı döndü. Yanlış SPI haritaları pakete eklenmedi."
            )
            errors.append(duplicate_message)
            for layer in successful_spi_layers:
                for file_key in ("png_file", "html_file", "geotiff_file", "shp_file"):
                    file_name = layer.get(file_key)
                    if file_name:
                        files.pop(str(file_name), None)
                label = str(layer.get("label"))
                layer.update(
                    {
                        "png_file": None,
                        "html_file": None,
                        "geotiff_file": None,
                        "shp_file": None,
                        "tile_url": None,
                        "quality": {
                            **dict(layer.get("quality") or {}),
                            "status": "basarisiz",
                            "error": duplicate_message,
                        },
                    }
                )
                metadata_by_label.pop(label, None)
                quality_by_label.pop(label, None)
    metadata = [
        metadata_by_label[label] for label in ordered_labels if label in metadata_by_label
    ]
    quality = [
        quality_by_label[label] for label in ordered_labels if label in quality_by_label
    ]
    if metadata:
        files["climate-engine-harita-metadata.json"] = json.dumps(
            metadata, ensure_ascii=False, indent=2, default=str
        ).encode("utf-8")
    if quality:
        files["harita-kalite-kontrol.json"] = json.dumps(
            quality, ensure_ascii=False, indent=2, default=str
        ).encode("utf-8")
    return files, layers, errors


def render_secondary_downloads(files: dict[str, bytes]) -> None:
    """Sonuç ekranındaki ikincil dosyaları kompakt bir açılır blokta gösterir."""
    if "degerler-tablosu-tez.xlsx" in files:
        st.download_button(
            "SPI değerler tablosunu indir",
            files["degerler-tablosu-tez.xlsx"],
            file_name="degerler-tablosu-tez.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    columns = st.columns(3)
    downloads = [
        ("Zaman serisi grafiği", "zaman-serisi.png", "image/png"),
        ("Alan haritası", "calisma-alani-haritasi.png", "image/png"),
        ("GeoJSON sınırını indir", "calisma-alani.geojson", "application/geo+json"),
        ("Metadata indir", "zetriklim-metadata.json", "application/json"),
        ("Bilimsel raporu indir", "bilimsel-rapor.html", "text/html"),
    ]
    for index, (label, file_name, mime) in enumerate(downloads):
        if file_name in files:
            columns[index % 3].download_button(
                label,
                files[file_name],
                file_name=file_name,
                mime=mime,
                use_container_width=True,
                key=f"secondary_download_{file_name}",
            )


def render_chart_gallery(files: dict[str, bytes], chart_names: list[str]) -> None:
    st.caption(
        "Grafikler ayrı PNG dosyalarıdır; tamamı ana ZIP paketinde de bulunur."
    )
    chart_columns = st.columns(min(3, len(chart_names)))
    for index, chart_name in enumerate(chart_names):
        label = chart_name.removeprefix("grafik-").removesuffix(".png")
        label = re.sub(r"^\d+-", "", label).replace("-", " ").title()
        column = chart_columns[index % len(chart_columns)]
        column.image(files[chart_name], caption=label, width="stretch")
        column.download_button(
            f"{label} grafiğini indir",
            files[chart_name],
            file_name=chart_name,
            mime="image/png",
            use_container_width=True,
            key=f"chart_download_{index}_{chart_name}",
        )


def select_output_files(files: dict[str, bytes], selections: list[str]) -> dict[str, bytes]:
    """Kullanıcının seçtiği sonuç gruplarını indirme paketine uygular."""
    selected = set(selections)
    result: dict[str, bytes] = {}
    academic_tables = {
        "akademik-kuraklik-serisi.csv", "kuraklik-olaylari.csv",
        "egilim-ve-degisim.csv", "gecikmeli-iliski.csv",
        "kaynak-dogrulama.csv", "belirsizlik.csv", "kalite-kontrol.csv",
        "dagilim-uyum-testleri.csv", "uzaktan-algilama-ozeti.csv",
        "mevsimsel-profil.csv", "anomali-serisi.csv",
    }
    metadata_files = {
        "zetriklim-metadata.json", "climate-engine-harita-metadata.json",
        "harita-kalite-kontrol.json", "raster-analiz-metadata.json",
        "analiz-uyarilari.txt", "BENI-OKU.txt",
    }
    for name, content in files.items():
        lower = name.lower()
        include = (
            ("Zaman serisi (CSV)" in selected and name == "zetriklim-veri.csv")
            or ("Excel çalışma kitabı" in selected and name == "zetriklim-veri.xlsx")
            or (
                "SPI değer tablosu" in selected
                and name in {"degerler-tablosu-tez.xlsx", "spi-sonuclari.csv"}
            )
            or ("İleri analiz tabloları" in selected and name in academic_tables)
            or ("Grafikler" in selected and lower.endswith(".png") and name.startswith("grafik-"))
            or ("PNG haritalar" in selected and lower.endswith(".png") and not name.startswith("grafik-") and name != "zaman-serisi.png")
            or ("Zaman serisi grafiği" in selected and name == "zaman-serisi.png")
            or ("GeoTIFF rasterlar" in selected and lower.endswith(".tif"))
            or ("Shapefile paketleri" in selected and lower.endswith("-shp.zip"))
            or ("Etkileşimli HTML haritalar" in selected and lower.endswith("-etkilesimli-harita.html"))
            or ("GeoJSON sınırı" in selected and lower.endswith(".geojson"))
            or ("Metadata ve kalite raporları" in selected and name in metadata_files)
            or ("Bilimsel HTML rapor" in selected and name == "bilimsel-rapor.html")
        )
        if include:
            result[name] = content
    return result

st.markdown(
    """
    <style>
    :root {
      --ink:#17373d; --ink-soft:#49666b; --teal:#0f766e; --teal-dark:#115e59;
      --cyan:#2aa7a1; --amber:#d79a2b; --surface:#ffffff; --line:#d9e7e4;
    }
    .stApp {
      background:
        radial-gradient(circle at 94% 4%, rgba(42,167,161,.12), transparent 25rem),
        linear-gradient(180deg, #f5f9f8 0%, #f8f6f0 100%);
      color:var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1420px; padding-top: 1.25rem; }
    .hero {
      background: linear-gradient(120deg, #17373d 0%, #155e63 55%, #0f766e 100%);
      color: white; border-radius: 28px; padding: 32px 36px; margin-bottom: 18px;
      position: relative; overflow: hidden; box-shadow: 0 18px 44px rgba(23,55,61,.16);
    }
    .hero:before, .hero:after {
      content:""; position:absolute; border-radius:46% 54% 58% 42%;
      border:1px solid rgba(126,241,234,.38); transform:rotate(18deg);
    }
    .hero:before { width:340px; height:340px; right:-70px; top:-150px; }
    .hero:after { width:210px; height:210px; right:40px; bottom:-160px; }
    .eyebrow { color:#ffd27a; font-size:.76rem; letter-spacing:.16em; font-weight:850; }
    .hero h1 { font-size:2.35rem; line-height:1.04; margin:.42rem 0 .6rem; max-width:850px; }
    .hero p { color:#d7f2ef; max-width:820px; margin:0; font-size:1rem; }
    .step {
      font-size:.72rem; letter-spacing:.12em; color:#147984;
      text-transform:uppercase; font-weight:850; margin-bottom:.25rem;
    }
    .hint {
      background:linear-gradient(90deg, rgba(15,118,110,.10), rgba(42,167,161,.04));
      border-left:4px solid var(--teal); padding:12px 14px; border-radius:10px;
      color:var(--ink-soft); margin:.5rem 0 1rem;
    }
    .status-ready { color:#07806d; font-weight:800; }
    .status-wait { color:#b36b00; font-weight:800; }
    div[data-testid="stMetric"] {
      background:var(--surface); border:1px solid var(--line);
      padding:12px 14px; border-radius:16px; box-shadow:0 6px 18px rgba(23,55,61,.05);
    }
    button[data-baseweb="tab"] { font-weight:750; }
    div[data-baseweb="tab-list"] {
      background:rgba(255,255,255,.88); padding:.38rem; border-radius:16px;
      border:1px solid var(--line); box-shadow:0 8px 24px rgba(23,55,61,.05);
    }
    button[data-baseweb="tab"][aria-selected="true"] {
      background:#e4f2ef; color:var(--teal-dark); border-radius:11px;
    }
    .workflow { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:4px 0 20px; }
    .flow-card {
      background:var(--surface); border:1px solid var(--line);
      border-radius:16px; padding:13px 14px; box-shadow:0 7px 20px rgba(23,55,61,.05);
    }
    .flow-card b { color:#075b68; display:block; margin-bottom:3px; }
    .flow-card span { color:#55727a; font-size:.82rem; }
    .brand-name {
      margin-top:.35rem; color:#063f55; font-size:1.02rem; line-height:1;
      font-weight:900; letter-spacing:.14em; text-align:center;
    }
    .brand-subtitle {
      margin-top:.35rem; color:#47717a; font-size:.68rem; line-height:1.25;
      font-weight:700; letter-spacing:.06em; text-align:center;
    }
    .brand-logo-shell {
      width:174px;max-width:100%;margin:0 auto;padding:.35rem;
      background:rgba(255,255,255,.72);border-radius:22px;
      box-shadow:0 9px 26px rgba(6,47,64,.08);
    }
    .brand-logo-shell svg {
      display:block;width:100%;height:auto;
    }
    @media (max-width:800px) { .workflow { grid-template-columns:1fr 1fr; } }
    .stButton > button, .stDownloadButton > button {
      border-radius:12px; min-height:2.85rem; font-weight:780;
      border-color:#c7dad6;
    }
    .stButton > button[kind="primary"] {
      background:linear-gradient(90deg, var(--teal-dark), var(--teal)); border:0;
      box-shadow:0 7px 18px rgba(15,118,110,.18);
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    [data-testid="stDateInput"] > div > div {
      border-color:#cfdfdc !important; background:#fff !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
      border-color:var(--teal) !important; box-shadow:0 0 0 2px rgba(15,118,110,.10);
    }
    .estimate-card {
      margin:.8rem 0 1rem;padding:18px 20px;border-radius:18px;color:white;
      background:linear-gradient(110deg,#17373d,#0f766e);box-shadow:0 12px 28px rgba(23,55,61,.14);
      display:flex;align-items:center;justify-content:space-between;gap:18px;
    }
    .estimate-card small {display:block;color:#bcece8;font-weight:750;letter-spacing:.06em;text-transform:uppercase;}
    .estimate-card strong {display:block;font-size:1.5rem;margin:.15rem 0;}
    .estimate-card span {color:#d8f2ef;font-size:.88rem;}
    .result-hero {
      margin:.35rem 0 1rem;padding:22px 24px;border-radius:22px;
      background:linear-gradient(115deg,#17373d,#0f766e);color:white;
      box-shadow:0 14px 34px rgba(23,55,61,.16);
    }
    .result-hero h2 {margin:.2rem 0 .4rem;font-size:1.55rem;color:white;}
    .result-hero p {margin:0;color:#d7f2ef;}
    .result-kicker {color:#ffd27a;font-size:.72rem;font-weight:850;letter-spacing:.14em;text-transform:uppercase;}
    @media (max-width:700px) {.estimate-card{display:block}.estimate-card strong{font-size:1.25rem}}
    </style>
    """,
    unsafe_allow_html=True,
)

if "uploader_nonce" not in st.session_state:
    st.session_state.uploader_nonce = 0

top_logo, top_hero = st.columns([0.16, 0.84], vertical_alignment="center")
with top_logo:
    st.markdown(
        """
        <div class="brand-logo-shell" role="img" aria-label="Zetriklim coğrafi analiz logosu">
          <svg viewBox="0 0 220 210" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="zg" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#35c8d0"/>
                <stop offset="1" stop-color="#00a68f"/>
              </linearGradient>
              <clipPath id="zb">
                <path d="M48 18 L146 20 L166 57 L199 76 L196 142 L175 174
                         L129 188 L79 181 L31 151 L24 94 Z"/>
              </clipPath>
            </defs>
            <path d="M48 18 L146 20 L166 57 L199 76 L196 142 L175 174
                     L129 188 L79 181 L31 151 L24 94 Z"
                  fill="#fbfdf9" stroke="#063f55" stroke-width="8" stroke-linejoin="round"/>
            <g clip-path="url(#zb)" fill="none" stroke="#35c8d0" stroke-width="4">
              <path d="M8 52 C38 30 51 72 78 48 S124 28 150 45"/>
              <path d="M4 75 C34 52 49 95 78 69 S126 49 164 65"/>
              <path d="M7 101 C38 77 53 120 82 95 S126 75 166 88"/>
              <path d="M12 127 C39 104 56 145 84 121 S128 101 169 112"/>
            </g>
            <g clip-path="url(#zb)" fill="url(#zg)" stroke="#ffffff" stroke-width="2">
              <rect x="84" y="148" width="22" height="22"/><rect x="108" y="148" width="22" height="22"/>
              <rect x="132" y="148" width="22" height="22"/><rect x="156" y="148" width="22" height="22"/>
              <rect x="108" y="124" width="22" height="22"/><rect x="132" y="124" width="22" height="22"/>
              <rect x="156" y="124" width="22" height="22"/><rect x="132" y="100" width="22" height="22"/>
              <rect x="156" y="100" width="22" height="22"/><rect x="156" y="76" width="22" height="22"/>
            </g>
            <circle cx="104" cy="101" r="18" fill="#ffffff" stroke="#063f55" stroke-width="6"/>
            <circle cx="104" cy="101" r="9" fill="#ffb51b"/>
            <path d="M123 101 H151 M104 120 V142" stroke="#063f55" stroke-width="5"
                  stroke-linecap="round" stroke-dasharray="1 10"/>
          </svg>
        </div>
        <div class="brand-name">ZETRİKLİM</div>
        <div class="brand-subtitle">COĞRAFİ ANALİZ PLATFORMU</div>
        """,
        unsafe_allow_html=True,
    )
with top_hero:
    st.markdown(
        """
        <section class="hero">
          <div style="font-size:.76rem;font-weight:800;letter-spacing:.14em;color:#ffd67a">
            HAVZA · İKLİM · UZAKTAN ALGILAMA
          </div>
          <h1>Coğrafi sınırını tanımla, değişimi mekânsal olarak çözümle.</h1>
          <p>Kaynağı ve sürümü belgelenmiş açık veri ürünlerinden seçilen döneme ve analize uygun sonuçları;
          harita, GeoTIFF, Excel ve Shapefile olarak üret.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="workflow">
      <div class="flow-card"><b>1 · Alanı yükle</b><span>SHP, GeoPackage veya GeoJSON</span></div>
      <div class="flow-card"><b>2 · Analizi seç</b><span>SPI, NDVI, NDWI, LST veya EVI</span></div>
      <div class="flow-card"><b>3 · Veri kaynağını seç</b><span>Climate Engine veya Earth Engine</span></div>
      <div class="flow-card"><b>4 · CBS çıktısını al</b><span>Excel, GeoTIFF, Shapefile ve harita</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

application_mode = "Birleşik Analiz"
academic_mode = True

# Tarih seçicileri yalnız Gelişmiş Ayarlar sekmesinde gösterilir. Burada
# oturumdaki değerler okunur; böylece önce çalışan veri kaynağı sekmesi de aynı
# ortak dönemi kullanır.
global_start_date = st.session_state.get(
    "global_analysis_start_date", date(1981, 1, 1)
)
global_end_date = st.session_state.get(
    "global_analysis_end_date", date.today()
)
global_period_valid = global_start_date <= global_end_date

# Bağlantı doğrulama gibi, betik analiz seçicisine ulaşmadan yapılan yeniden
# çalıştırmalarda Streamlit widget anahtarını temizleyebilir. Analiz seçimini
# widget yaşam döngüsünden bağımsız bir oturum değerinde koru.
stored_analysis = st.session_state.get(
    "selected_analysis_choice",
    st.session_state.get("selected_analysis_widget", "SPI"),
)
if stored_analysis not in SELECTABLE_ANALYSES:
    stored_analysis = "SPI"
st.session_state["selected_analysis_choice"] = stored_analysis
if "selected_analysis_widget" not in st.session_state:
    st.session_state["selected_analysis_widget"] = stored_analysis

academic_params: dict[str, object] = {}
academic_study: dict[str, object] = {}

tab_area, tab_analysis, tab_data, tab_research, tab_output = st.tabs(
    [
        "01 · Çalışma Alanı",
        "02 · Analiz",
        "03 · Veri Kaynağı",
        "04 · Gelişmiş Ayarlar",
        "05 · Sonuçlar",
    ]
)

with tab_area:
    left, right = st.columns([0.43, 0.57], gap="large")
    with left:
        st.markdown('<div class="step">Çalışma alanı</div>', unsafe_allow_html=True)
        st.subheader("Coğrafi sınırınızı ekleyin")
        area_source = st.radio(
            "Alan kaynağı",
            ["Dosya yükle", f"GADM {GADM_VERSION} idari sınırı"],
            horizontal=True,
            help=(
                "Kendi havza/poligon dosyanızı yükleyebilir veya akademik ve ticari olmayan "
                "çalışmalar için GADM ülke/idari bölüm sınırını doğrudan seçebilirsiniz."
            ),
        )
        st.markdown(
            '<div class="hint">ZIP zorunlu değil. GeoPackage veya GeoJSON tek dosya; '
            'Shapefile ise .shp, .shx, .dbf ve .prj bileşenleri birlikte seçilebilir.</div>',
            unsafe_allow_html=True,
        )
        uploads = []
        single_shp_crs = "EPSG:4326"
        if area_source == "Dosya yükle":
            uploads = st.file_uploader(
                "Çalışma alanı dosyaları",
                type=["zip", "shp", "shx", "dbf", "prj", "cpg", "gpkg", "geojson", "json"],
                accept_multiple_files=True,
                key=f"area_upload_{st.session_state.uploader_nonce}",
                help="Havza, il, ilçe, bölge veya kendi çizdiğiniz poligonları yükleyin.",
            )
            single_shp_crs = st.text_input(
                "Tek SHP için koordinat sistemi",
                value="EPSG:4326",
                help=(
                    "Yalnız .shp yüklenirse .prj bulunmadığı için CRS burada belirtilir. "
                    "GeoPackage, GeoJSON ve tam SHP paketinde dosyanın kendi CRS bilgisi kullanılır."
                ),
            )
        else:
            g1, g2 = st.columns(2)
            gadm_iso3 = g1.text_input(
                "Ülke ISO3 kodu",
                value="TUR",
                max_chars=3,
                help="Üç harfli ülke kodudur; Türkiye için TUR kullanılır.",
            ).upper()
            gadm_level = int(g2.selectbox(
                "İdari düzey",
                [0, 1, 2],
                index=1,
                format_func=lambda value: {0: "0 · Ülke", 1: "1 · İl/bölge", 2: "2 · İlçe/alt bölge"}[value],
                help="Düzey 0 ülke, 1 birinci kademe, 2 ikinci kademe idari bölümleri gösterir.",
            ))
            gadm_key = (gadm_iso3, gadm_level)
            if st.button("GADM sınırlarını getir", use_container_width=True):
                try:
                    with st.spinner("GADM idari sınırları indiriliyor..."):
                        st.session_state.gadm_frame = fetch_gadm(gadm_iso3, gadm_level)
                        st.session_state.gadm_key = gadm_key
                except Exception as gadm_error:
                    st.session_state.pop("gadm_frame", None)
                    st.error(f"GADM sınırı alınamadı: {gadm_error}")
            gadm_frame = st.session_state.get("gadm_frame")
            if gadm_frame is not None and st.session_state.get("gadm_key") == gadm_key:
                label_column = gadm_name_column(gadm_frame, gadm_level)
                labels = sorted(gadm_frame[label_column].dropna().astype(str).unique())
                selected_labels = st.multiselect(
                    "Çalışma alanı",
                    labels,
                    default=labels if gadm_level == 0 else [],
                    help="Bir veya birden fazla komşu idari birim seçilebilir.",
                )
                if st.button("Seçili GADM alanını kullan", disabled=not selected_labels, use_container_width=True):
                    selected_gadm = gadm_frame[gadm_frame[label_column].astype(str).isin(selected_labels)].copy()
                    st.session_state.geometry_summary = inspect_geodataframe(selected_gadm)
                    st.session_state.area_source_note = (
                        f"GADM {GADM_VERSION} · {gadm_iso3} · düzey {gadm_level} · "
                        + ", ".join(selected_labels)
                    )
                    st.session_state.pop("output_package", None)
                    st.rerun()
            st.caption("GADM verileri akademik ve ticari olmayan kullanım koşullarına tabidir.")
        clear_col, info_col = st.columns([0.42, 0.58], vertical_alignment="center")
        with clear_col:
            if st.button("Dosyaları temizle", icon=":material/delete:", use_container_width=True):
                st.session_state.pop("geometry_summary", None)
                st.session_state.pop("output_package", None)
                st.session_state.pop("area_source_note", None)
                st.session_state.uploader_nonce += 1
                st.rerun()
        with info_col:
            st.caption("Hatalı dosyayı yükleme kutusundaki × ile tek tek de silebilirsiniz.")

        summary = st.session_state.get("geometry_summary")
        if uploads:
            try:
                parts = [UploadedPart(item.name, item.getvalue()) for item in uploads]
                with st.spinner("Geometri, koordinat sistemi ve alan denetleniyor..."):
                    summary = inspect_uploaded_files(parts, fallback_crs=single_shp_crs)
                st.session_state.geometry_summary = summary
                uploaded_names = list(dict.fromkeys(
                    Path(item.name).stem for item in uploads if Path(item.name).stem
                ))
                st.session_state.area_source_note = (
                    ", ".join(uploaded_names[:4])
                    or "Kullanıcı tarafından yüklenen çalışma alanı"
                )
            except GeometryUploadError as exc:
                summary = None
                st.session_state.pop("geometry_summary", None)
                st.error(str(exc))

        if summary:
            m1, m2 = st.columns(2)
            m1.metric("Toplam alan", f"{summary.area_km2:,.2f} km²")
            m2.metric("Coğrafi obje", f"{summary.feature_count:,}")
            st.success("Çalışma alanı doğrulandı.")

            with st.expander("Alan detayları", expanded=True):
                d1, d2 = st.columns(2)
                d1.write(f"**Çevre:** {summary.perimeter_km:,.2f} km")
                d1.write(f"**Köşe sayısı:** {summary.vertex_count:,}")
                d1.write(f"**Kaynak CRS:** {summary.source_crs}")
                d2.write(f"**Alan hesabı CRS:** {summary.area_crs}")
                d2.write(f"**Merkez:** {summary.centroid[0]:.5f}, {summary.centroid[1]:.5f}")
                d2.write(f"**Geometri onarımı:** {'Uygulandı' if summary.was_repaired else 'Gerekmedi'}")

            spatial_mode = st.selectbox(
                "Analiz alanı detayı",
                [
                    "Tüm objeleri tek çalışma alanı olarak birleştir",
                    "Her coğrafi objeyi ayrı raporla",
                    "Her obje + tüm alan özetini birlikte üret",
                    "Yalnızca alan ortalaması üret",
                ],
                help="Çok parçalı ilçe veya alt havza verilerinde sonuçların nasıl gruplanacağını belirler.",
            )
            spatial_stat = st.multiselect(
                "Mekânsal özetler",
                ["Ortalama", "Toplam", "Minimum", "Maksimum", "Medyan", "Standart sapma", "Yüzdelikler"],
                default=["Ortalama", "Minimum", "Maksimum"],
                help="Çalışma alanındaki raster hücrelerinin hangi istatistiklerle özetleneceğini belirler.",
            )
        else:
            spatial_mode = "Tüm objeleri tek çalışma alanı olarak birleştir"
            spatial_stat = ["Ortalama"]
            st.info("Harita ve alan ayrıntıları için bir coğrafi dosya yükleyin.")

    with right:
        st.markdown('<div class="step">Harita önizleme</div>', unsafe_allow_html=True)
        if summary:
            bounds = summary.bounds
            fmap = folium.Map(summary.centroid, zoom_start=8, tiles="CartoDB positron", control_scale=True)
            folium.GeoJson(
                summary.gdf_wgs84.__geo_interface__,
                style_function=lambda _: {
                    "color": "#063447", "weight": 3, "fillColor": "#00a6a6", "fillOpacity": 0.30
                },
                highlight_function=lambda _: {"weight": 5, "fillOpacity": 0.42},
                tooltip=folium.GeoJsonTooltip(
                    fields=[c for c in summary.gdf_wgs84.columns if c != "geometry"][:4],
                    aliases=[c for c in summary.gdf_wgs84.columns if c != "geometry"][:4],
                ) if len(summary.gdf_wgs84.columns) > 1 else None,
            ).add_to(fmap)
            fmap.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            add_cartographic_controls(
                fmap,
                "Çalışma alanı",
                st.session_state.get("area_source_note", "Yüklenen sınır"),
                "—",
                "—",
            )
        else:
            fmap = folium.Map([39.0, 35.0], zoom_start=5, tiles="CartoDB positron", control_scale=True)
        st_folium(fmap, height=570, width="stretch", returned_objects=[])

analysis_for_source = st.session_state.get("selected_analysis_choice", "SPI")

with tab_data:
    st.markdown('<div class="step">Kaynak, ürün ve değişken</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hint"><strong>{analysis_for_source}</strong> için yalnızca uyumlu kaynak ve '
        'ürünler listelenir. Başka bir analiz kendiliğinden eklenmez.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3, gap="large")
    climate_engine_key = st.session_state.get(
        "climate_engine_api_key",
        os.getenv("CLIMATE_ENGINE_API_KEY", ""),
    )
    with c1:
        source_options = (
            ["Climate Engine", "Google Earth Engine"]
            if analysis_for_source in CLIMATE_ENGINE_ANALYSES
            else ["Google Earth Engine"]
        )
        provider = st.selectbox(
            "Veri kaynağı",
            source_options,
            index=0,
            help=(
                "Climate Engine kişisel API anahtarıyla, Google Earth Engine ise "
                "Project ID ve Google hesabı yetkilendirmesiyle çalışır."
            ),
        )
        source_info = SOURCES[provider]
        st.caption(source_info["description"])
        st.info(
            f"Yalnız {analysis_for_source} için gerekli ana veri ürünü kullanılacaktır. "
            "Başka analizler ancak kullanıcı ayrıca seçerse eklenir."
        )
        status_class = "status-ready" if source_info["status"] in {"Hazır", "Açık erişim"} else "status-wait"
        st.markdown(f'<span class="{status_class}">● {source_info["status"]}</span>', unsafe_allow_html=True)
        if provider == "Climate Engine":
            st.markdown("##### Climate Engine hesabını bağlayın")
            st.caption(
                "Climate Engine kullanıcı girişi Project ID ile değil, kişiye özel API anahtarıyla yapılır."
            )
            entered_ce_key = st.text_input(
                "Climate Engine API anahtarı",
                value="",
                type="password",
                placeholder="Anahtarınızı buraya yapıştırın",
                help="Anahtar yalnız bu tarayıcı oturumunda tutulur; GitHub'a ve çıktı dosyalarına yazılmaz.",
            )
            ce_left, ce_right = st.columns(2)
            ce_left.link_button(
                "API anahtarı iste",
                "https://www.climateengine.org/apis/requesting-an-authorization-key-token/",
                use_container_width=True,
            )
            ce_right.link_button(
                "Resmî API belgeleri",
                "https://api.climateengine.org/",
                use_container_width=True,
            )
            if st.button(
                "Climate Engine bağlantısını doğrula",
                disabled=not entered_ce_key,
                use_container_width=True,
            ):
                try:
                    with st.spinner("Climate Engine anahtarı doğrulanıyor..."):
                        ce_status = validate_api_key(entered_ce_key)
                    st.session_state.climate_engine_api_key = entered_ce_key
                    st.session_state.climate_engine_status = ce_status
                    st.success("Climate Engine API anahtarı doğrulandı.")
                    st.rerun()
                except Exception as ce_error:
                    st.session_state.pop("climate_engine_api_key", None)
                    st.session_state.pop("climate_engine_status", None)
                    st.error(f"Climate Engine bağlantısı kurulamadı: {ce_error}")
            climate_engine_key = st.session_state.get(
                "climate_engine_api_key",
                os.getenv("CLIMATE_ENGINE_API_KEY", ""),
            )
            if climate_engine_key:
                st.success(f"Climate Engine · {connection_label(climate_engine_key)}")
                expiration = st.session_state.get("climate_engine_status", {}).get("expiration")
                if expiration:
                    st.caption(f"Anahtar geçerlilik bilgisi: {expiration}")
                if st.button("Climate Engine bağlantısını kes", use_container_width=True):
                    st.session_state.pop("climate_engine_api_key", None)
                    st.session_state.pop("climate_engine_status", None)
                    st.rerun()
        gee_project = (
            st.text_input(
                "Google Earth Engine Project ID",
                value=os.getenv("GOOGLE_EARTH_ENGINE_PROJECT", ""),
                placeholder="ör. ee-kullanici-projesi",
                help="Proje adı veya numarası değil, Google Cloud Project ID değerini girin.",
            )
            if provider == "Google Earth Engine"
            else os.getenv("GOOGLE_EARTH_ENGINE_PROJECT", "")
        )
        if provider == "Google Earth Engine":
            project_valid = bool(
                re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", gee_project or "")
            )
            with st.expander("Project ID nasıl alınır?", expanded=not project_valid):
                st.markdown(
                    """
                    1. Google Cloud'da bir proje oluşturun veya mevcut projenizi seçin.
                    2. **Project ID** değerini proje seçiciden kopyalayın; proje adı ve proje numarası farklıdır.
                    3. Earth Engine API'yi etkinleştirin.
                    4. Projeyi ticari olmayan araştırma veya uygun kullanım türüyle Earth Engine'e kaydedin.
                    5. Buraya Project ID'yi girip Google hesabınızla yetkilendirin.
                    """
                )
                st.link_button(
                    "1 · Google Cloud projesi oluştur",
                    "https://console.cloud.google.com/projectcreate",
                    use_container_width=True,
                )
                if project_valid:
                    st.link_button(
                        "2 · Earth Engine API'yi etkinleştir",
                        "https://console.cloud.google.com/apis/library/earthengine.googleapis.com"
                        f"?project={gee_project}",
                        use_container_width=True,
                    )
                    st.link_button(
                        "3 · Projeyi Earth Engine'e kaydet",
                        "https://console.cloud.google.com/earth-engine/configuration"
                        f"?project={gee_project}",
                        use_container_width=True,
                    )
                    st.caption(
                        "Kayıt sayfasına yönlendirilmeniz normaldir; bu adım Google girişi değil, "
                        "Project ID'nin Earth Engine kullanımına açılmasıdır."
                    )

            if gee_project and not project_valid:
                st.error(
                    "Project ID biçimi geçerli görünmüyor. Yalnızca küçük harf, rakam ve kısa çizgi kullanın."
                )
                gee_ok, gee_message = False, "Geçersiz Project ID"
            elif project_valid:
                gee_ok, gee_message = cached_gee_status(gee_project)
                personal_auth = st.session_state.get("gee_user_auth", {})
                personally_connected = personal_auth.get("project") == gee_project
                if gee_ok and personally_connected:
                    st.success(f"Earth Engine {gee_message}")
                    if st.button("Google bağlantısını kes", use_container_width=True):
                        st.session_state.pop("gee_user_auth", None)
                        st.session_state.pop("gee_auth_flow", None)
                        st.rerun()
                elif gee_ok:
                    st.success(f"Earth Engine {gee_message}")
                    st.caption("Sunucu bağlantısı hazır. İsterseniz kendi Google hesabınızı bağlayabilirsiniz.")
                else:
                    st.warning("Bu Project ID için geçerli Google yetkilendirmesi bulunamadı.")

                if not personally_connected:
                    if st.button(
                        "Google hesabıyla Earth Engine'e bağlan",
                        type="primary",
                        use_container_width=True,
                    ):
                        auth_url, verifier = create_user_auth_flow()
                        st.session_state.gee_auth_flow = {
                            "project": gee_project,
                            "url": auth_url,
                            "verifier": verifier,
                        }
                    auth_flow = st.session_state.get("gee_auth_flow")
                    if auth_flow and auth_flow.get("project") == gee_project:
                        st.link_button(
                            "Google giriş ve izin ekranını aç",
                            auth_flow["url"],
                            use_container_width=True,
                        )
                        auth_code = st.text_input(
                            "Google'ın verdiği tek kullanımlık doğrulama kodu",
                            type="password",
                            help="Kod yalnızca bu oturumda bağlantı kurmak için kullanılır ve dosyaya yazılmaz.",
                        )
                        if st.button(
                            "Bağlantıyı tamamla ve test et",
                            disabled=not auth_code,
                            use_container_width=True,
                        ):
                            try:
                                with st.spinner("Google Earth Engine bağlantısı doğrulanıyor..."):
                                    st.session_state.gee_user_auth = exchange_user_auth_code(
                                        auth_code,
                                        auth_flow["verifier"],
                                        gee_project,
                                    )
                                st.session_state.pop("gee_auth_flow", None)
                                st.success("Google hesabı ve Project ID başarıyla doğrulandı.")
                                st.rerun()
                            except Exception as auth_error:
                                st.error(
                                    "Bağlantı kurulamadı. Project ID, Earth Engine kaydı ve Google "
                                    f"hesabı izinlerini kontrol edin. Ayrıntı: {auth_error}"
                                )
            else:
                gee_ok, gee_message = False, "Project ID bekleniyor"
                st.info("Önce kendi Google Cloud Project ID değerinizi girin.")
    with c2:
        ce_products = {
            "SPI": {
                "CHIRPS Daily (4,8 km)": ("CHIRPS_DAILY", ["precipitation"], date(1981, 1, 1)),
                "CHIRPS Pentad (4,8 km)": ("CHIRPS_PENTAD", ["precipitation"], date(1981, 1, 1)),
                "CHIRPS Preliminary Pentad": ("CHIRPS_PRELIM_PENTAD", ["precipitation"], date(2015, 1, 1)),
                "ERA5-Ag Daily (9,6 km)": ("ERA5_AG", ["total_precipitation"], date(1979, 1, 1)),
            },
            "NDVI": {
                "Sentinel-2 Surface Reflectance (10 m)": ("SENTINEL2_SR", ["NDVI"], date(2015, 1, 1)),
                "Harmonized Landsat–Sentinel-2 (30 m)": ("HLS_SR", ["NDVI"], date(2013, 4, 11)),
                "Landsat 5/7/8/9 Surface Reflectance (30 m)": ("LANDSAT_SR", ["NDVI"], date(1984, 1, 1)),
            },
            "EVI": {
                "Sentinel-2 Surface Reflectance (10 m)": ("SENTINEL2_SR", ["EVI"], date(2015, 1, 1)),
                "Harmonized Landsat–Sentinel-2 (30 m)": ("HLS_SR", ["EVI"], date(2013, 4, 11)),
                "Landsat 5/7/8/9 Surface Reflectance (30 m)": ("LANDSAT_SR", ["EVI"], date(1984, 1, 1)),
            },
            "LST": {
                "Landsat 8 Surface Reflectance (30 m)": ("LANDSAT8_SR", ["LST"], date(2013, 4, 11)),
                "Landsat 5/7/8/9 Surface Reflectance (30 m)": ("LANDSAT_SR", ["LST"], date(1984, 1, 1)),
                "MODIS Terra 8-day (1 km)": ("MODIS_TERRA_8DAY", ["LST_Day_1km"], date(2000, 2, 18)),
            },
        }
        gee_products = {
            "SPI": ["CHIRPS Daily"],
            "NDVI": ["Sentinel-2 SR Harmonized"],
            "NDWI": ["Sentinel-2 SR Harmonized"],
            "NDMI": ["Sentinel-2 SR Harmonized"],
            "NDBI": ["Sentinel-2 SR Harmonized"],
            "EVI": ["Sentinel-2 SR Harmonized"],
            "SAVI": ["Sentinel-2 SR Harmonized"],
            "LST": ["Landsat 8/9 Collection 2 Level-2"],
            "DEM": ["SRTM V3"],
            "SLOPE": ["SRTM V3"],
            "ASPECT": ["SRTM V3"],
            "TWI": ["MERIT Hydro + SRTM"],
        }
        if provider == "Climate Engine":
            product = st.selectbox(
                "Veri ürünü",
                list(ce_products[analysis_for_source]),
                help=f"Yalnızca {analysis_for_source} üretebilen belgelenmiş Climate Engine ürünleri gösterilir.",
            )
            dataset_options = {
                value[0]: value[1]
                for value in ce_products[analysis_for_source].values()
            }
            selected_dataset = ce_products[analysis_for_source][product][0]
            analysis_product_start_date = ce_products[analysis_for_source][product][2]
            ce_dataset_id = st.selectbox(
                "Dataset parametresi",
                list(dataset_options),
                index=list(dataset_options).index(selected_dataset),
                help="Climate Engine API'nin kullandığı resmî dataset kodudur.",
            )
            analysis_product_start_date = {
                value[0]: value[2]
                for value in ce_products[analysis_for_source].values()
            }[ce_dataset_id]
            product_start_date = analysis_product_start_date
            ce_variable_id = st.selectbox(
                "Değişken parametresi",
                dataset_options[ce_dataset_id],
                help=f"{analysis_for_source} hesabı için ürün içinde kullanılacak resmî değişken kodudur.",
            )
            ce_variable_ids = ce_variable_id
            st.link_button(
                "Dataset ve değişken parametrelerini incele",
                "https://www.climateengine.org/apis/apiDatasets/",
                use_container_width=True,
            )
        else:
            product = st.selectbox(
                "Veri ürünü",
                gee_products[analysis_for_source],
                help=f"{analysis_for_source} için belgelenmiş Earth Engine koleksiyonu.",
            )
            ce_dataset_id, ce_variable_ids = "", ""
            analysis_product_start_date = {
                "SPI": date(1981, 1, 1),
                "NDVI": date(2015, 6, 23),
                "NDWI": date(2015, 6, 23),
                "NDMI": date(2015, 6, 23),
                "NDBI": date(2015, 6, 23),
                "EVI": date(2015, 6, 23),
                "SAVI": date(2015, 6, 23),
                "LST": date(2013, 4, 11),
                "DEM": date(1970, 1, 1),
                "SLOPE": date(1970, 1, 1),
                "ASPECT": date(1970, 1, 1),
                "TWI": date(1970, 1, 1),
            }[analysis_for_source]
            product_start_date = analysis_product_start_date
        primary_variables = {
            "SPI": ["Yağış"],
            "NDVI": ["NDVI / EVI"],
            "NDWI": ["NDWI / yüzey suyu"],
            "NDMI": ["NDWI / yüzey suyu"],
            "NDBI": ["Arazi örtüsü"],
            "EVI": ["NDVI / EVI"],
            "SAVI": ["NDVI / EVI"],
            "LST": ["Yüzey sıcaklığı (LST)"],
            "DEM": ["Yükselti / eğim / bakı"],
            "SLOPE": ["Yükselti / eğim / bakı"],
            "ASPECT": ["Yükselti / eğim / bakı"],
            "TWI": ["Yükselti / eğim / bakı"],
        }[analysis_for_source]
        relationship_state_key = (
            "spi_relationship_enabled"
            if analysis_for_source == "SPI"
            else f"{analysis_for_source.lower()}_relationship_enabled"
        )
        relationship_requested = bool(
            st.session_state.get(relationship_state_key, False)
        )
        variables = list(primary_variables)
        if relationship_requested:
            variables = list(dict.fromkeys([
                *variables,
                "Yağış",
                "Hava sıcaklığı",
                "Potansiyel evapotranspirasyon",
            ]))
        quality_control = [
            "Eksik veri",
            "Birim dönüşümü",
            "Zaman sürekliliği",
            "Tamamlanmamış ay",
            "Fiziksel değer aralığı",
            "Uydu geçerli piksel oranı",
        ]
        if relationship_requested:
            st.info(
                f"{analysis_for_source} ile kuraklık ilişkisi için yağış, sıcaklık ve "
                "potansiyel evapotranspirasyon destek verileri de kullanılacaktır."
            )
        else:
            st.info(
                f"Yalnız {analysis_for_source} için gerekli veri seçildi. "
                "Eksik veri, birim ve zaman sürekliliği kontrolleri uygulanacaktır."
            )
    with c3:
        requested_start_date = global_start_date
        start_date = max(global_start_date, product_start_date)
        end_date = global_end_date
        incomplete_month_removed = False
        if analysis_for_source == "SPI" or analysis_for_source in TIME_SERIES_REMOTE_ANALYSES:
            end_date, incomplete_month_removed = safe_monthly_end(end_date)
            if incomplete_month_removed:
                st.warning(
                    f"Aylık toplamların yapay düşük görünmemesi için tamamlanmamış ay çıkarıldı. "
                    f"Son tarih {end_date:%d.%m.%Y} olarak uygulandı."
                )
        if provider == "Google Earth Engine" and analysis_for_source == "SPI":
            previous_month_end = date.today().replace(day=1) - timedelta(days=1)
            chirps_final_boundary = previous_month_end.replace(day=1) - timedelta(days=1)
            if end_date > chirps_final_boundary:
                end_date = chirps_final_boundary
                st.warning(
                    "CHIRPS v2.0 Final arşiv gecikmesi nedeniyle yalnız tamamlanmış ve "
                    f"yayımlanmış aylar kullanıldı. Son tarih {end_date:%d.%m.%Y}."
                )
        analysis_period_compatible = end_date >= analysis_product_start_date
        period_compatible = start_date <= end_date and analysis_period_compatible
        if analysis_for_source == "SPI" or analysis_for_source in TIME_SERIES_REMOTE_ANALYSES:
            temporal_scale = "Aylık"
            aggregation = "Aylık toplam/ortalama"
        elif analysis_for_source in STATIC_TERRAIN_ANALYSES:
            temporal_scale = "Statik raster"
            aggregation = "Kaynak ürün"
        else:
            temporal_scale = "Dönem kompoziti"
            aggregation = "Medyan"
        start_time, end_time = None, None
        if start_date > end_date:
            st.error("Başlangıç yılı/tarihi bitiş değerinden sonra olamaz.")
        if requested_start_date < product_start_date:
            st.info(
                f"{product} arşivi {product_start_date:%d.%m.%Y} tarihinde başlıyor. "
                f"Bu analiz için başlangıç otomatik olarak {start_date:%d.%m.%Y} uygulandı."
            )
        if not analysis_period_compatible:
            st.error(
                f"Seçilen dönem {analysis_for_source} ürününün "
                f"{analysis_product_start_date:%d.%m.%Y} başlangıcından önce bitiyor."
            )

    st.success(
        f"Uyumlu seçim: {analysis_for_source} · {provider} · {product}. "
        "Dataset ve değişken kodları analizle birlikte metadata dosyasına kaydedilecektir."
    )

def _reset_cross_analysis_widgets() -> None:
    """Analiz değiştiğinde başka yönteme ait isteğe bağlı alanları kapatır."""
    relationship_keys = ["spi_relationship_enabled", *[
        f"{analysis.lower()}_relationship_enabled"
        for analysis in SELECTABLE_ANALYSES if analysis != "SPI"
    ]]
    for widget_key in relationship_keys:
        st.session_state.pop(widget_key, None)
    selected = st.session_state.get("selected_analysis_widget", "SPI")
    st.session_state["selected_analysis_choice"] = selected
    st.session_state["active_analysis_ui"] = selected


with tab_analysis:
    st.markdown('<div class="step">Analiz yöntemi</div>', unsafe_allow_html=True)
    selected_analysis = st.selectbox(
        "Analiz",
        SELECTABLE_ANALYSES,
        key="selected_analysis_widget",
        on_change=_reset_cross_analysis_widgets,
        help=(
            "SPI kuraklığı; NDVI ve EVI bitki örtüsünü; NDWI yüzey suyunu; "
            "LST yüzey sıcaklığını inceler."
        ),
    )
    st.session_state["selected_analysis_choice"] = selected_analysis
    preset = academic_defaults(selected_analysis)
    academic_study = {
        "title": f"{selected_analysis} iklim analizi",
        "question": "",
        "hypotheses": "",
    }
    selected_analyses = [selected_analysis]
    method_info = ANALYSIS_METHODS[selected_analysis]
    st.dataframe(
        [{
            "Yöntem": selected_analysis,
            "Tam ad": method_info["title"],
            "Önerilen ürün": method_info["source"],
            "Doğal çözünürlük": method_info["resolution"],
            "Amaç": method_info["purpose"],
        }],
        hide_index=True,
        width="stretch",
    )
    analysis_params = {"method": selected_analysis}

    if selected_analysis == "SPI":
        st.markdown("##### SPI hesaplama ayarları")
        p1, p2 = st.columns(2)
        analysis_params["scales"] = p1.multiselect(
            "SPI zaman ölçeği (ay)",
            [1, 3, 6, 9, 12, 18, 24],
            default=[3, 6, 12],
            key="spi_scales",
            help=(
                "Yağışın kaç aylık birikim üzerinden değerlendirileceğini belirler. "
                "SPI-3 mevsimsel, SPI-6 orta dönem, SPI-12 uzun dönem kuraklığı gösterir."
            ),
        )
        analysis_params["distribution"] = p2.selectbox(
            "Olasılık dağılımı",
            ["Gamma"],
            key="spi_distribution",
            help="Yağış serisi için sıfır olasılığı düzeltilmiş Gamma dağılımı uygulanır.",
        )
        analysis_params["baseline"] = (
            f"{global_start_date.year}–{global_end_date.year}"
        )
    elif selected_analysis in SPECTRAL_ANALYSES:
        st.markdown(f"##### {selected_analysis} görüntü işleme ayarları")
        p1, p2 = st.columns(2)
        analysis_params["cloud_limit"] = p1.slider(
            "Azami sahne bulutluluğu (%)",
            0, 80, 30, 5,
            key=f"{selected_analysis.lower()}_cloud_limit",
            help="Bu orandan daha bulutlu uydu sahneleri analize alınmaz.",
        )
        analysis_params["composite"] = p2.selectbox(
            "Dönem kompoziti",
            ["Medyan"],
            key=f"{selected_analysis.lower()}_composite",
            help="Seçilen dönemdeki geçerli piksellerin medyanı alınarak tek raster üretilir.",
        )
        st.info(
            f"{ANALYSIS_METHODS[selected_analysis]['title']}: "
            f"{ANALYSIS_METHODS[selected_analysis]['purpose']}."
        )
    else:
        analysis_params["composite"] = "Statik topoğrafik ürün"
        analysis_params["cloud_limit"] = 0
        st.info(
            f"{ANALYSIS_METHODS[selected_analysis]['title']} seçildi. Bu ürün statiktir; "
            "bulut eşiği ve dönem kompoziti gerektirmez."
        )

with tab_research:
    st.markdown('<div class="step">Analiz dönemi</div>', unsafe_allow_html=True)
    period_columns = st.columns(2)
    global_start_date = period_columns[0].date_input(
        "Başlangıç tarihi",
        value=global_start_date,
        min_value=date(1970, 1, 1),
        max_value=date.today(),
        key="global_analysis_start_date",
    )
    global_end_date = period_columns[1].date_input(
        "Bitiş tarihi",
        value=global_end_date,
        min_value=date(1970, 1, 1),
        max_value=date.today(),
        key="global_analysis_end_date",
    )
    global_period_valid = global_start_date <= global_end_date
    if not global_period_valid:
        st.error("Başlangıç tarihi bitiş tarihinden sonra olamaz.")
    else:
        st.caption("Bu dönem veri kaynağına ve sonuçlara otomatik uygulanır.")

    if not academic_mode:
        st.info(
            "Bu bölüm Akademik Araştırma modu seçildiğinde açılır. Standart moddaki "
            "mevcut SPI, NDVI, EVI ve LST iş akışı değişmeden kullanılabilir."
        )
        academic_params = {
            "scales": analysis_params.get("scales", [3, 6, 12]),
            "drought_indices": ["SPI"],
            "baseline_start": 1981,
            "baseline_end": 2024,
        }
        advanced_statistics_enabled = selected_analysis == "SPI"
    else:
        remote_title = (
            "SPI kuraklık"
            if selected_analysis == "SPI"
            else f"{selected_analysis} · {ANALYSIS_METHODS[selected_analysis]['title']}"
        )
        st.markdown(
            f'<div class="step">{remote_title} ayarları</div>',
            unsafe_allow_html=True,
        )
        scales = list(analysis_params.get("scales", preset["scales"]))
        available_reference_start = global_start_date.year
        available_reference_end = max(
            available_reference_start,
            min(global_end_date.year, date.today().year),
        )
        default_reference_end = available_reference_end
        default_reference_start = max(
            available_reference_start,
            default_reference_end - 29,
        )
        baseline_start = default_reference_start
        baseline_end = default_reference_end
        spi_distribution = str(analysis_params.get("distribution", "Gamma"))
        spei_distribution = "Log-logistic"
        event_threshold = -1.0
        prewhiten = True
        seasonal_mk = True
        alpha = 0.05
        max_lag = 6
        correlation_method = "Spearman"
        change_window_years = 3
        anomaly_baseline_start = max(
            available_reference_start, analysis_product_start_date.year
        )
        anomaly_baseline_end = available_reference_end
        land_cover_labels: list[str] = []

        if selected_analysis == "SPI":
            option_columns = st.columns(2)
            include_spei = option_columns[0].checkbox(
                "SPEI'yi de hesapla",
                value=False,
                key="spi_include_spei",
                help="Yağışa ek olarak potansiyel evapotranspirasyon etkisini de değerlendirir.",
            )
            event_threshold = float(
                option_columns[1].selectbox(
                    "Kuraklık olayı eşiği",
                    [-0.5, -1.0, -1.5, -2.0],
                    index=1,
                    key="spi_event_threshold",
                )
            )
            drought_indices = ["SPI", *(["SPEI"] if include_spei else [])]
            relationship_enabled = st.checkbox(
                "Bitki örtüsü veya yüzey sıcaklığı tepkisini de incele",
                value=False,
                key="spi_relationship_enabled",
                help="Açıldığında seçilen uydu göstergesi ile kuraklık arasındaki gecikmeli ilişki hesaplanır.",
            )
            response_indices = (
                st.multiselect(
                    "Karşılaştırılacak göstergeler",
                    ["NDVI", "EVI", "LST"],
                    default=["NDVI"],
                    key="spi_response_indices",
                )
                if relationship_enabled else []
            )
        elif selected_analysis in TIME_SERIES_REMOTE_ANALYSES:
            trend_heading = (
                f"{selected_analysis} bitki örtüsü eğilimi ve değişimi"
                if selected_analysis in {"NDVI", "EVI"}
                else "LST yüzey sıcaklığı eğilimi ve değişimi"
            )
            st.markdown(f"##### {trend_heading}")
            trend_columns = st.columns(4)
            prewhiten = trend_columns[0].checkbox(
                "Otokorelasyon düzeltmesi",
                value=True,
                key=f"{selected_analysis.lower()}_prewhiten",
                help="Ardışık gözlemlerin birbirine bağımlılığının eğilim testini yanıltmasını azaltır.",
            )
            seasonal_mk = trend_columns[1].checkbox(
                "Mevsimsel Mann–Kendall",
                value=True,
                key=f"{selected_analysis.lower()}_seasonal_mk",
                help="NDVI eğilimini mevsimsel döngüyü dikkate alarak sınar.",
            )
            alpha = float(
                trend_columns[2].selectbox(
                    "Anlamlılık düzeyi", [0.01, 0.05, 0.10], index=1,
                    key=f"{selected_analysis.lower()}_alpha",
                )
            )
            change_window_years = int(
                trend_columns[3].slider(
                    "Değişim penceresi (yıl)", 1, 5, 3,
                    key=f"{selected_analysis.lower()}_change_window",
                    help="İlk ve son dönem ortalamalarının kaç yıllık pencerelerle karşılaştırılacağını belirler.",
                )
            )

            st.markdown(f"##### {selected_analysis} anomalileri")
            anomaly_start_limit = max(
                available_reference_start, analysis_product_start_date.year
            )
            anomaly_baseline_start = anomaly_start_limit
            anomaly_baseline_end = available_reference_end
            if selected_analysis in {"NDVI", "EVI"}:
                st.caption(
                    f"Aylık {selected_analysis} klimatolojisi ortak dönemin kullanılabilir "
                    f"kısmından ({anomaly_baseline_start}–{anomaly_baseline_end}) otomatik "
                    "hesaplanır; ayrıca tarih girmeniz gerekmez."
                )
            else:
                st.caption(
                    f"Aylık LST klimatolojisi {anomaly_baseline_start}–"
                    f"{anomaly_baseline_end} döneminden otomatik hesaplanır; ayrıca "
                    "tarih girmeniz gerekmez."
                )

            response_indices = [selected_analysis]
            with st.expander(
                f"İsteğe bağlı · {selected_analysis} ile kuraklık ilişkisi",
                expanded=False,
            ):
                st.caption(
                    f"Bu bölüm ana {selected_analysis} analizinden ayrıdır. Yalnız "
                    "işaretlendiğinde SPI/SPEI destek verileri indirilir."
                )
                relationship_enabled = st.checkbox(
                    "Kuraklık ilişkisini ekle",
                    value=False,
                    key=f"{selected_analysis.lower()}_relationship_enabled",
                )
                drought_indices = (
                    st.multiselect(
                        "Karşılaştırılacak kuraklık indisleri",
                        ["SPI", "SPEI"],
                        default=["SPI"],
                        key=f"satellite_drought_indices_{selected_analysis}",
                    )
                    if relationship_enabled else []
                )
            if provider == "Google Earth Engine":
                land_cover_labels = st.multiselect(
                    "Arazi örtüsüne göre ayrı sonuçlar (isteğe bağlı)",
                    list(LAND_COVER_CLASSES.values()),
                    default=[],
                    help="Seçilen sınıflar için ESA WorldCover tabanlı ayrı zonal seriler üretir.",
                )
        else:
            relationship_enabled = False
            drought_indices = []
            response_indices = [selected_analysis]
            st.markdown("##### Mekânsal çıktı ayarları")
            st.info(
                f"{selected_analysis} yalnız seçilen çalışma alanı için üretilecek. "
                "SPI, NDVI veya başka bir analiz otomatik olarak eklenmeyecek."
            )

        advanced_statistics_enabled = bool(drought_indices)
        if advanced_statistics_enabled:
            if selected_analysis != "SPI":
                st.markdown("##### İsteğe bağlı kuraklık referansı")
            # SPI referansı analiz başlangıcına bağlanmaz. Örneğin 2015–2026
            # analizinde yalnız 11 yıllık örneklem kullanmak Gamma uyumunu
            # kararsızlaştırıp rasterı ±3 sınırına yığabiliyordu. CHIRPS'ın
            # bağımsız, uzun dönem arşivi otomatik referans olarak kullanılır.
            baseline_start = 1981
            baseline_end = min(2024, global_end_date.year - 1)
            if baseline_end < baseline_start:
                baseline_end = max(baseline_start, global_end_date.year)
            if "SPEI" in drought_indices:
                spei_distribution = st.selectbox(
                    "SPEI dağılımı", ["Log-logistic", "Pearson Tip III"],
                    key=f"{selected_analysis.lower()}_spei_distribution",
                )
            st.caption(
                f"Kuraklık referansı CHIRPS uzun dönem arşivinden otomatik belirlendi: "
                f"{baseline_start}–{baseline_end}."
            )
            if baseline_end - baseline_start + 1 < 30:
                st.warning("Kararlı standartlaştırma için en az 30 yıllık referans dönemi önerilir.")

        if relationship_enabled:
            st.markdown("##### Kuraklık–ekosistem ilişkisi")
            relation_columns = st.columns(2)
            correlation_method = relation_columns[0].selectbox(
                "İlişki yöntemi", ["Spearman", "Pearson"],
                key=f"{selected_analysis.lower()}_correlation_method",
            )
            max_lag = int(
                relation_columns[1].slider(
                    "Azami gecikme (ay)", 0, 12, 6,
                    key=f"{selected_analysis.lower()}_max_lag",
                )
            )
            if selected_analysis == "SPI":
                land_cover_labels = st.multiselect(
                    "Arazi örtüsü kırılımı (isteğe bağlı)",
                    list(LAND_COVER_CLASSES.values()),
                    default=[],
                    help="Yalnız seçilen sınıflar için ek zonal karşılaştırma üretir.",
                )
        land_cover_codes = [
            code for code, label in LAND_COVER_CLASSES.items() if label in land_cover_labels
        ]
        # İstasyon yükleme arayüzü kaldırıldı. Önceki bir oturumdan kalan gizli
        # istasyon verisinin yeni sonuçları etkilemesine de izin verilmez.
        st.session_state.pop("academic_station_data", None)

        academic_params = {
            "drought_indices": drought_indices,
            "scales": scales,
            "baseline_start": baseline_start,
            "baseline_end": baseline_end,
            "spi_distribution": spi_distribution,
            "spei_distribution": spei_distribution,
            "event_threshold": event_threshold,
            "prewhiten": prewhiten,
            "seasonal_mk": seasonal_mk,
            "alpha": alpha,
            "max_lag": max_lag,
            "correlation_method": correlation_method,
            "remove_seasonality": True,
            "response_indices": response_indices,
            "land_cover_codes": land_cover_codes,
            "land_cover_labels": land_cover_labels,
            "relationship_enabled": relationship_enabled,
            "advanced_statistics_enabled": advanced_statistics_enabled,
            "remote_statistics_enabled": selected_analysis in {"NDVI", "EVI", "LST"},
            "change_window_years": change_window_years,
            "anomaly_baseline_start": anomaly_baseline_start,
            "anomaly_baseline_end": anomaly_baseline_end,
        }
        analysis_params.update(academic_params)
        selected_analyses = list(dict.fromkeys([
            selected_analysis, *drought_indices, *response_indices
        ]))

with tab_output:
    summary = st.session_state.get("geometry_summary")
    map_period_start = start_date
    map_period_end = end_date
    map_period_valid = global_period_valid and map_period_start <= map_period_end
    geometry_identity = (
        hashlib.sha256(
            summary.gdf_wgs84.to_crs(4326).geometry.union_all().wkb
        ).hexdigest()
        if summary else None
    )
    station_cache_data = st.session_state.get("academic_station_data")
    station_identity = (
        hashlib.sha256(
            pd.util.hash_pandas_object(station_cache_data, index=True).values.tobytes()
        ).hexdigest()
        if isinstance(station_cache_data, pd.DataFrame) else None
    )
    current_output_config = (
        RESULT_CACHE_VERSION,
        MAP_CACHE_VERSION,
        selected_analysis,
        provider,
        product,
        str(start_date),
        str(end_date),
        round(summary.area_km2, 4) if summary else None,
        geometry_identity,
        station_identity,
        application_mode,
        json.dumps(
            {"academic": academic_params, "study": academic_study},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ),
    )
    if (
        st.session_state.get("output_config")
        and st.session_state.output_config != current_output_config
    ):
        for state_key in [
            "output_files", "output_package", "output_metadata", "output_data",
            "output_source", "output_analysis_errors", "output_tile_url",
            "output_tile_layers", "output_boundary", "output_map_period",
            "output_academic_results", "output_elapsed_seconds", "output_config",
        ]:
            st.session_state.pop(state_key, None)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Alan", f"{summary.area_km2:,.2f} km²" if summary else "Bekleniyor")
    q2.metric("Kaynak", provider)
    q3.metric("Değişken", len(variables))
    q4.metric("Yöntem", len(selected_analyses))

    output_options = [
        "Zaman serisi (CSV)",
        "Excel çalışma kitabı",
        "Zaman serisi grafiği",
        "Grafikler",
        "PNG haritalar",
        "GeoTIFF rasterlar",
        "Shapefile paketleri",
        "Etkileşimli HTML haritalar",
        "GeoJSON sınırı",
        "Metadata ve kalite raporları",
        "Bilimsel HTML rapor",
    ]
    if selected_analysis == "SPI":
        output_options.insert(2, "SPI değer tablosu")
    if advanced_statistics_enabled or selected_analysis in {"NDVI", "EVI", "LST"}:
        output_options.insert(3, "İleri analiz tabloları")
    output_formats = st.multiselect(
        "İstenen çıktılar",
        output_options,
        default=output_options,
        key="selected_output_groups",
    )
    include_items = list(output_formats)

    cached_analyses = _list_analysis_result_cache()
    with st.expander(
        f"Kayıtlı analizler ({len(cached_analyses)})",
        expanded=False,
    ):
        if not cached_analyses:
            st.info(
                "Henüz eksiksiz tamamlanmış bir analiz kaydı yok. Başarıyla biten "
                "analizler burada otomatik olarak listelenecek."
            )
        else:
            cache_table = pd.DataFrame(
                [
                    {
                        "Dosya adı": item["file_name"],
                        "İl / bölge / alan": item["area_name"],
                        "Analiz": item["analysis"],
                        "Kaynak": item["provider"],
                        "Dönem": f"{item['start']} – {item['end']}",
                        "Alan (km²)": item["area_km2"],
                        "Dosya": item["file_count"],
                        "Boyut (MB)": round(float(item["size_mb"]), 1),
                        "Kaydedilme": item["created"],
                    }
                    for item in cached_analyses
                ]
            )
            st.dataframe(cache_table, hide_index=True, width="stretch")
            cache_labels = [
                f"{item['area_name']} · {item['analysis']} · {item['start']}–{item['end']} · "
                f"{item['created']:%d.%m.%Y %H:%M}"
                for item in cached_analyses
            ]
            selected_cache_label = st.selectbox(
                "İndirilecek kayıtlı analiz",
                cache_labels,
                key="selected_cached_analysis",
            )
            selected_cache = cached_analyses[cache_labels.index(selected_cache_label)]
            cached_value = _read_analysis_result_cache(selected_cache["cache_path"])
            cached_files = cached_value.get("output_files") if cached_value else None
            if isinstance(cached_files, dict) and cached_files:
                st.download_button(
                    "Kayıtlı analiz paketini indir",
                    build_complete_package(cached_files),
                    file_name=str(selected_cache["file_name"]),
                    mime="application/zip",
                    use_container_width=True,
                )
    if st.session_state.get("output_files"):
        selected_files = select_output_files(st.session_state.output_files, output_formats)
        st.session_state.output_package = build_complete_package(selected_files)

    source_ready = (
        bool(climate_engine_key and ce_dataset_id and ce_variable_ids)
        if provider == "Climate Engine"
        else bool(gee_ok and gee_project)
    )
    can_build = bool(
        summary
        and variables
        and selected_analyses
        and period_compatible
        and map_period_valid
        and source_ready
    )
    estimated_specs = (
        climate_engine_map_specs(
                selected_analysis,
                ce_dataset_id,
                ce_variable_ids,
                analysis_product_start_date,
                map_period_start,
                [int(value) for value in (analysis_params.get("scales") or [3])],
            )
        if provider == "Climate Engine" and ce_dataset_id and ce_variable_ids
        else []
    )
    estimated_map_count = len(estimated_specs) or max(1, len(selected_analyses))
    estimated_spi_maps = sum(spec.get("analysis") == "SPI" for spec in estimated_specs)
    estimated_archive_maps = sum(
        spec.get("analysis") in {"Yağış", "Sıcaklık"} for spec in estimated_specs
    )
    estimated_anomaly_maps = sum(
        spec.get("map_kind") == "anomalies" for spec in estimated_specs
    )
    period_years = max((end_date - start_date).days / 365.25, 1 / 12)
    estimate_lower, estimate_upper = estimate_analysis_seconds(
        summary.area_km2 if summary else 0.01,
        provider=provider,
        map_count=estimated_map_count,
        period_years=period_years,
        academic_mode=academic_mode,
        spi_map_count=estimated_spi_maps,
        archive_map_count=estimated_archive_maps,
        anomaly_map_count=estimated_anomaly_maps,
        observed_seconds=st.session_state.get("output_elapsed_seconds"),
    )
    estimate_label = (
        format_duration_range(estimate_lower, estimate_upper)
        if summary else "Alan seçimi bekleniyor"
    )
    estimate_detail = (
        f"{summary.area_km2:,.2f} km² alan · {estimated_map_count} harita · "
        f"{period_years:.1f} yıllık dönem"
        if summary else f"{estimated_map_count} harita · {period_years:.1f} yıllık dönem"
    )
    st.markdown(
        f"""
        <div class="estimate-card" role="status" aria-label="Tahmini analiz süresi">
          <div><small>Tahmini tamamlanma süresi</small><strong>{estimate_label}</strong>
          <span>{estimate_detail}</span></div>
          <span>Bağlantı ve kaynak servis yoğunluğuna göre değişebilir.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not summary:
        st.warning("Önce Alan sekmesinden geçerli bir çalışma alanı yükleyin.")
    if not selected_analyses:
        st.warning("Analiz sekmesinden en az bir yöntem seçin.")
    if provider == "Climate Engine" and not source_ready:
        st.warning(
            "Climate Engine işlemi için doğrulanmış API anahtarı, dataset parametresi "
            "ve en az bir değişken parametresi gereklidir."
        )
    if provider == "Google Earth Engine" and not source_ready:
        st.warning(
            "Earth Engine işlemi için Project ID ve tamamlanmış Google yetkilendirmesi gereklidir."
        )

    action_columns = st.columns(2)
    build_all_requested = action_columns[0].button(
        "Tüm analizi oluştur",
        type="primary",
        icon=":material/play_arrow:",
        disabled=not can_build,
        use_container_width=True,
    )
    map_only_ready = bool(
        provider == "Climate Engine"
        and can_build
        and st.session_state.get("output_files")
        and st.session_state.get("output_data") is not None
    )
    map_only_requested = action_columns[1].button(
        "Yalnız haritaları yeniden oluştur",
        icon=":material/map:",
        disabled=not map_only_ready,
        use_container_width=True,
        help=(
            "Mevcut tablo, grafik ve bilimsel analizleri korur; yalnız seçilen tarih "
            "aralığındaki harita dosyalarını yeniler."
        ),
    )

    existing_layers = st.session_state.get("output_tile_layers", [])
    same_complete_map_period = bool(
        st.session_state.get("output_map_period") == (map_period_start, map_period_end)
        and existing_layers
        and all(layer.get("png_file") for layer in existing_layers)
    )
    if map_only_requested and same_complete_map_period:
        st.info(
            "Bu tarih aralığındaki bütün haritalar zaten eksiksiz hazır. "
            "Climate Engine kotasını korumak için yeniden istek gönderilmedi."
        )
        map_only_requested = False

    if map_only_requested:
        with st.spinner("Tablo ve analizler korunuyor; yalnız haritalar yeniden oluşturuluyor…"):
            map_specs = climate_engine_map_specs(
                selected_analysis,
                ce_dataset_id,
                ce_variable_ids,
                analysis_product_start_date,
                map_period_start,
                [int(value) for value in (analysis_params.get("scales") or [3])],
            )
            map_specs = _attach_expected_map_means(
                map_specs,
                st.session_state.get("output_data"),
                map_period_start,
                map_period_end,
            )
            reference_start = int(
                academic_params.get("baseline_start", start_date.year)
                if academic_mode else start_date.year
            )
            reference_end = int(
                academic_params.get("baseline_end", end_date.year)
                if academic_mode else end_date.year
            )
            refreshed_files, refreshed_layers, refreshed_errors = regenerate_climate_engine_maps(
                api_key=climate_engine_key,
                summary=summary,
                specs=map_specs,
                map_end=map_period_end,
                reference_start=reference_start,
                reference_end=reference_end,
                gee_project=gee_project or None,
                cloud_limit=int(analysis_params.get("cloud_limit", 30)),
            )
            preserved_files = dict(st.session_state.output_files)
            map_slugs = {str(spec["slug"]) for spec in map_specs} | {
                "su-dengesi-dagilimi",
                "yagis-normale-gore",
                "sicaklik-anomalisi",
                "su-dengesi-anomalisi",
            }
            for file_name in list(preserved_files):
                if file_name in {
                    "climate-engine-harita-metadata.json",
                    "harita-kalite-kontrol.json",
                    "analiz-uyarilari.txt",
                    "zetriklim-cbs.gpkg",
                } or any(
                    file_name == f"{slug}{suffix}"
                    for slug in map_slugs
                    for suffix in (
                        ".png",
                        ".tif",
                        ".gpkg",
                        ".qml",
                        "-shp.zip",
                        "-qgis-raster.zip",
                        "-etkilesimli-harita.html",
                    )
                ):
                    preserved_files.pop(file_name, None)
            preserved_files.update(refreshed_files)
            if refreshed_errors:
                preserved_files["analiz-uyarilari.txt"] = (
                    "Üretilemeyen haritalar\n\n" + "\n".join(refreshed_errors)
                ).encode("utf-8")
            st.session_state.output_files = preserved_files
            st.session_state.output_tile_layers = refreshed_layers
            st.session_state.output_analysis_errors = refreshed_errors
            st.session_state.output_package = build_complete_package(
                select_output_files(preserved_files, output_formats)
            )
            st.session_state.output_map_period = (map_period_start, map_period_end)
        if refreshed_errors:
            st.warning(
                f"Haritalar yeniden denendi; {len(refreshed_errors)} harita kaynak sunucudan "
                "tam veri alamadığı için tamamlanamadı. Tablo ve analizler değiştirilmedi."
            )
        else:
            st.success(
                "Yalnız haritalar yenilendi. Tablo, grafik ve bilimsel analizler değiştirilmedi."
            )

    result_cache_path = _analysis_result_cache_path(current_output_config)
    if build_all_requested:
        cached_result = _read_analysis_result_cache(result_cache_path)
        if cached_result:
            for state_key in (
                "output_files", "output_metadata", "output_data", "output_source",
                "output_analysis_errors", "output_tile_url", "output_tile_layers",
                "output_boundary", "output_config", "output_map_period",
                "output_academic_results", "output_elapsed_seconds",
            ):
                if state_key in cached_result:
                    st.session_state[state_key] = cached_result[state_key]
            st.session_state.output_package = build_complete_package(
                select_output_files(st.session_state.output_files, output_formats)
            )
            st.success("Aynı alan ve ayarlara ait kayıtlı analiz yeniden kullanıldı.")
            build_all_requested = False

    if build_all_requested:
        analysis_started_at = time_module.perf_counter()
        live_supported = provider in {
            "Otomatik en uygun açık kaynak",
            "Open-Meteo Historical",
            "Google Earth Engine",
            "Climate Engine",
        }
        if not live_supported:
            st.error(
                f"{provider} için canlı indirme bağlayıcısı henüz etkin değil. "
                "Gerçek veri almak için şimdilik “Otomatik en uygun açık kaynak” "
                "veya “Open-Meteo Historical” seçin."
            )
        else:
            try:
                with st.spinner(
                    f"Analiz sürüyor · tahmini süre {estimate_label}. "
                    "Veriler, haritalar ve kalite kontrolleri hazırlanıyor…"
                ):
                    source_request_metadata = {}
                    academic_source_metadata = {}
                    source_analysis_errors: list[str] = []
                    if provider == "Google Earth Engine":
                        climate_latitude, climate_longitude = summary.centroid
                        climate_elevation = None
                        unsupported = []
                        if selected_analysis == "SPI":
                            climate_data, unsupported = fetch_gee_monthly_climate(
                                summary.gdf_wgs84,
                                start_date,
                                end_date,
                                ["Yağış"],
                                project=gee_project,
                            )
                            climate_model = "CHIRPS Daily via Google Earth Engine"
                            climate_url = (
                                "https://developers.google.com/earth-engine/datasets/catalog/"
                                "UCSB-CHG_CHIRPS_DAILY"
                            )
                        elif selected_analysis in TIME_SERIES_REMOTE_ANALYSES:
                            climate_data, academic_source_metadata = fetch_gee_academic_series(
                                summary.gdf_wgs84,
                                (
                                    start_date
                                    if advanced_statistics_enabled
                                    else max(start_date, analysis_product_start_date)
                                ),
                                end_date,
                                response_indices=[selected_analysis],
                                land_cover_codes=list(academic_params.get("land_cover_codes", [])),
                                include_climate=advanced_statistics_enabled,
                                project=gee_project,
                            )
                            climate_model = (
                                f"{product} via Google Earth Engine"
                                + (
                                    " + CHIRPS/ERA5-Land ilişki desteği"
                                    if advanced_statistics_enabled else ""
                                )
                            )
                            climate_url = "https://developers.google.com/earth-engine/datasets/catalog"
                            source_request_metadata["academic_datasets"] = academic_source_metadata
                        else:
                            climate_data = pd.DataFrame(
                                [{
                                    "Tarih": pd.Timestamp(end_date),
                                    "Örnek ID": 1,
                                    "Enlem": climate_latitude,
                                    "Boylam": climate_longitude,
                                    "Analiz": selected_analysis,
                                    "Dönem başlangıcı": str(start_date),
                                    "Dönem bitişi": str(end_date),
                                }]
                            )
                            climate_model = f"{product} via Google Earth Engine"
                            climate_url = "https://developers.google.com/earth-engine/datasets/catalog"
                    elif provider == "Climate Engine":
                        primary_start_date = (
                            max(start_date, analysis_product_start_date)
                            if academic_mode and selected_analysis != "SPI"
                            else start_date
                        )
                        ce_dataset_used = ce_dataset_id
                        primary_attempt_errors: list[str] = []
                        satellite_dataset_candidates = [
                            (ce_dataset_id, primary_start_date),
                            *(
                                [
                                    ("HLS_SR", max(start_date, date(2013, 4, 11))),
                                    ("LANDSAT_SR", max(start_date, date(1984, 1, 1))),
                                ]
                                if selected_analysis in {"NDVI", "EVI"} else []
                            ),
                        ]
                        primary_data = None
                        ce_metadata = None
                        for candidate_dataset, candidate_start in dict.fromkeys(
                            satellite_dataset_candidates
                        ):
                            try:
                                primary_data, ce_metadata = fetch_climate_engine_timeseries(
                                    climate_engine_key,
                                    summary.gdf_wgs84,
                                    candidate_start,
                                    end_date,
                                    candidate_dataset,
                                    ce_variable_ids,
                                    area_reducer="mean",
                                )
                                ce_dataset_used = candidate_dataset
                                break
                            except Exception as primary_error:
                                primary_attempt_errors.append(
                                    f"{candidate_dataset}: {primary_error}"
                                )
                        if primary_data is None or ce_metadata is None:
                            raise RuntimeError(
                                f"{selected_analysis} zaman serisi bütün gerçek veri "
                                "kaynaklarında başarısız oldu: "
                                + " | ".join(primary_attempt_errors)
                            )
                        if ce_dataset_used != ce_dataset_id:
                            source_analysis_errors.append(
                                f"{ce_dataset_id} yanıt vermedi; {selected_analysis} serisi "
                                f"otomatik olarak {ce_dataset_used} gerçek uydu arşivinden alındı."
                            )
                        primary_data = normalize_analysis_column(
                            primary_data, selected_analysis
                        )
                        climate_data = primary_data
                        climate_model = (
                            f"Climate Engine API · {ce_dataset_used} · {ce_variable_ids}"
                        )
                        climate_url = ce_metadata["endpoint"]
                        source_request_metadata = {"primary_analysis": ce_metadata}
                        climate_latitude, climate_longitude = summary.centroid
                        climate_elevation = None
                        unsupported = []
                        if academic_mode and advanced_statistics_enabled:
                            if selected_analysis != "SPI":
                                rainfall_data, rainfall_metadata = fetch_climate_engine_timeseries(
                                    climate_engine_key,
                                    summary.gdf_wgs84,
                                    start_date,
                                    end_date,
                                    "CHIRPS_DAILY",
                                    "precipitation",
                                    area_reducer="mean",
                                )
                                response_data = primary_data.drop(
                                    columns=["Örnek ID", "Enlem", "Boylam"], errors="ignore"
                                )
                                rainfall_data["Tarih"] = pd.to_datetime(
                                    rainfall_data["Tarih"], errors="coerce"
                                )
                                response_data["Tarih"] = pd.to_datetime(
                                    response_data["Tarih"], errors="coerce"
                                )
                                climate_data = rainfall_data.merge(
                                    response_data, on="Tarih", how="outer"
                                ).sort_values("Tarih")
                                source_request_metadata["academic_precipitation"] = rainfall_metadata
                            ce_response_specs = {
                                "NDVI": ("SENTINEL2_SR", "NDVI", date(2015, 1, 1)),
                                "EVI": ("SENTINEL2_SR", "EVI", date(2015, 1, 1)),
                                "LST": ("LANDSAT8_SR", "LST", date(2013, 4, 11)),
                            }
                            for response_index in academic_params.get("response_indices", []):
                                response_index = str(response_index).upper()
                                if response_index == selected_analysis or response_index not in ce_response_specs:
                                    continue
                                response_dataset, response_variable, response_start = ce_response_specs[response_index]
                                try:
                                    extra_response, extra_metadata = fetch_climate_engine_timeseries(
                                        climate_engine_key,
                                        summary.gdf_wgs84,
                                        max(start_date, response_start),
                                        end_date,
                                        response_dataset,
                                        response_variable,
                                        area_reducer="mean",
                                    )
                                except Exception as response_error:
                                    source_analysis_errors.append(
                                        f"{response_index} tepki serisi alınamadı; ana iklim "
                                        f"analizi korundu: {response_error}"
                                    )
                                    continue
                                extra_response = normalize_analysis_column(
                                    extra_response, response_index
                                ).drop(columns=["Örnek ID", "Enlem", "Boylam"], errors="ignore")
                                extra_response["Tarih"] = pd.to_datetime(
                                    extra_response["Tarih"], errors="coerce"
                                )
                                climate_data["Tarih"] = pd.to_datetime(
                                    climate_data["Tarih"], errors="coerce"
                                )
                                climate_data = climate_data.merge(
                                    extra_response, on="Tarih", how="outer"
                                ).sort_values("Tarih")
                                source_request_metadata[f"academic_response_{response_index}"] = extra_metadata
                            supporting_climate = fetch_centroid_series(
                                latitude=summary.centroid[0],
                                longitude=summary.centroid[1],
                                start_date=start_date,
                                end_date=end_date,
                                variables=[
                                    "Yağış",
                                    "Hava sıcaklığı",
                                    "Potansiyel evapotranspirasyon",
                                ],
                                temporal_scale="Aylık",
                            )
                            supporting_data = supporting_climate.data.drop(
                                columns=["Örnek ID", "Enlem", "Boylam"], errors="ignore"
                            ).rename(
                                columns={"Toplam yağış (mm)": "ERA5 yağış (mm)"}
                            )
                            climate_data["Tarih"] = pd.to_datetime(
                                climate_data["Tarih"], errors="coerce"
                            )
                            supporting_data["Tarih"] = pd.to_datetime(
                                supporting_data["Tarih"], errors="coerce"
                            )
                            climate_data = climate_data.merge(
                                supporting_data, on="Tarih", how="outer"
                            ).sort_values("Tarih")
                            climate_model += (
                                " + CHIRPS Daily" if selected_analysis != "SPI" else ""
                            ) + " + ERA5/Open-Meteo yağış, PET ve sıcaklık"
                            source_request_metadata["academic_support_source"] = (
                                supporting_climate.source_url
                            )
                    else:
                        climate = fetch_centroid_series(
                            latitude=summary.centroid[0],
                            longitude=summary.centroid[1],
                            start_date=start_date,
                            end_date=end_date,
                            variables=variables,
                            temporal_scale=temporal_scale,
                        )
                        climate_data = climate.data
                        climate_model = climate.model
                        climate_url = climate.source_url
                        climate_latitude = climate.latitude
                        climate_longitude = climate.longitude
                        climate_elevation = climate.elevation_m
                        unsupported = climate.unsupported_variables

                    analysis_tables = {}
                    spi_table = None
                    academic_results: dict[str, pd.DataFrame] = {}
                    if academic_mode and advanced_statistics_enabled:
                        # Günlük Climate Engine/istasyon kayıtları ile aylık GEE destek
                        # serilerinin aynı satırlarda hizalanmasını garanti eder.
                        climate_data = harmonize_monthly_data(climate_data)
                    if academic_mode and advanced_statistics_enabled:
                        precipitation_columns = [
                            column
                            for column in climate_data.columns
                            if any(token in column.lower() for token in ["yağış", "precip"])
                            and pd.to_numeric(climate_data[column], errors="coerce").notna().any()
                        ]
                        precipitation_columns.sort(
                            key=lambda column: (
                                0 if "chirps" in column.lower() else
                                1 if "toplam yağış" in column.lower() else 2
                            )
                        )
                        if not precipitation_columns:
                            raise ValueError(
                                "Akademik analiz için sayısal yağış sütunu bulunamadı."
                            )
                        pet_columns = [
                            column
                            for column in climate_data.columns
                            if any(token in column.lower() for token in ["pet", "et₀", "evapotrans"])
                            and pd.to_numeric(climate_data[column], errors="coerce").notna().any()
                        ]
                        response_columns = [
                            column
                            for column in climate_data.columns
                            if any(
                                column == response or column.startswith(f"{response}|")
                                for response in academic_params.get("response_indices", [])
                            )
                        ]
                        validation_columns = [
                            column for column in precipitation_columns[1:]
                            if column != precipitation_columns[0]
                        ]
                        academic_results = run_academic_analysis(
                            climate_data,
                            precipitation_column=precipitation_columns[0],
                            pet_column=pet_columns[0] if pet_columns else None,
                            response_columns=response_columns,
                            validation_columns=validation_columns,
                            config=academic_params,
                        )
                        analysis_tables.update(
                            {
                                name: table
                                for name, table in academic_results.items()
                                if table is not None and not table.empty
                            }
                        )
                        spi_table = academic_results.get("Kuraklık Serisi")
                    elif "SPI" in selected_analyses:
                        spi_input_data = climate_data
                        spi_source = climate_model
                        gee_ok, _ = cached_gee_status(gee_project or None)
                        if provider == "Otomatik en uygun açık kaynak" and gee_ok:
                            chirps_data = fetch_chirps_monthly_mean(
                                summary.gdf_wgs84,
                                start_date,
                                end_date,
                                project=gee_project,
                            )
                            era_rain_columns = [
                                column for column in climate_data.columns
                                if "yağış" in column.lower() and climate_data[column].dtype.kind in "fi"
                            ]
                            if era_rain_columns:
                                era_monthly = (
                                    climate_data.set_index("Tarih")[era_rain_columns[0]]
                                    .resample("MS")
                                    .sum(min_count=1)
                                    .rename("ERA5 yağış (mm)")
                                    .reset_index()
                                )
                                comparison = chirps_data[["Tarih", "Toplam yağış (mm)"]].rename(
                                    columns={"Toplam yağış (mm)": "CHIRPS yağış (mm)"}
                                ).merge(era_monthly, on="Tarih", how="outer")
                                comparison["Fark (CHIRPS - ERA5)"] = (
                                    comparison["CHIRPS yağış (mm)"] - comparison["ERA5 yağış (mm)"]
                                )
                                analysis_tables["Yağış Karşılaştırma"] = comparison
                            spi_input_data = chirps_data
                            spi_source = "CHIRPS Daily via Google Earth Engine"
                        precipitation_columns = [
                            column for column in spi_input_data.columns
                            if any(token in column.lower() for token in ["yağış", "precip"])
                            and pd.to_numeric(spi_input_data[column], errors="coerce").notna().any()
                        ]
                        if not precipitation_columns:
                            available_columns = ", ".join(map(str, spi_input_data.columns))
                            raise ValueError(
                                "SPI hesaplanamadı: veri tablosunda sayısal yağış sütunu bulunamadı. "
                                f"Mevcut sütunlar: {available_columns}"
                            )
                        spi_scales = analysis_params.get("scales") or [1, 3, 6, 12]
                        spi_table = calculate_spi_table(
                            spi_input_data,
                            precipitation_columns[0],
                            spi_scales,
                        )
                        spi_table.insert(1, "SPI yağış kaynağı", spi_source)
                        analysis_tables["SPI Sonuçları"] = spi_table

                    if selected_analysis in {"NDVI", "EVI", "LST"}:
                        remote_response_columns = [
                            column
                            for column in climate_data.columns
                            if column == selected_analysis
                            or column.startswith(f"{selected_analysis}|")
                        ]
                        remote_results = run_remote_sensing_analysis(
                            climate_data,
                            response_columns=remote_response_columns,
                            config=academic_params,
                        )
                        if not remote_results:
                            raise ValueError(
                                f"{selected_analysis} istatistikleri hazırlanamadı: "
                                "geçerli tarih ve sayısal gösterge değeri bulunamadı."
                            )
                        for result_name, result_table in remote_results.items():
                            if (
                                result_name not in academic_results
                                or academic_results[result_name].empty
                            ):
                                academic_results[result_name] = result_table
                            if result_table is not None and not result_table.empty:
                                analysis_tables[result_name] = academic_results[result_name]
                    metadata = build_metadata(
                        area={
                            "area_km2": round(summary.area_km2, 6),
                            "perimeter_km": round(summary.perimeter_km, 6),
                            "feature_count": summary.feature_count,
                            "source_crs": summary.source_crs,
                            "calculation_crs": summary.area_crs,
                            "spatial_mode": spatial_mode,
                            "spatial_statistics": spatial_stat,
                            "sampling_note": "İlk çalışan bağlayıcı alan merkezindeki ERA5-Land grid hücresini kullanır.",
                        },
                        request={
                            "provider": provider,
                            "product": climate_model,
                            "source_url": climate_url,
                            "variables": variables,
                            "unsupported_variables": unsupported,
                            "start_date": str(start_date),
                            "end_date": str(end_date),
                            "temporal_scale": temporal_scale,
                            "aggregation": aggregation,
                            "quality_control": quality_control,
                            "source_request_metadata": source_request_metadata,
                        },
                        analysis={
                            "mode": application_mode,
                            "methods": selected_analyses,
                            "parameters": analysis_params,
                            "study": academic_study if academic_mode else None,
                        },
                        outputs={"formats": output_formats, "contents": include_items},
                    )
                    area_geojson = summary.gdf_wgs84.to_json(drop_id=True).encode("utf-8")
                    metadata_rows = [
                        ("Veri kaynağı", provider),
                        ("Asıl ürün", climate_model),
                        ("Kaynak URL", climate_url),
                        ("Başlangıç", start_date),
                        ("Bitiş", end_date),
                        ("Zaman çözünürlüğü", temporal_scale),
                        ("Örnekleme", "Çalışma alanı merkezindeki ERA5-Land grid hücresi"),
                        ("Enlem", climate_latitude),
                        ("Boylam", climate_longitude),
                        ("Model yüksekliği (m)", climate_elevation),
                        ("Desteklenmeyen seçimler", ", ".join(unsupported) or "Yok"),
                        ("Analiz yapısı", application_mode),
                        (
                            "Referans dönemi",
                            f"{academic_params.get('baseline_start')}–{academic_params.get('baseline_end')}"
                            if academic_mode else analysis_params.get("baseline", "—"),
                        ),
                    ]
                    area_rows = [
                        ("Toplam alan (km²)", summary.area_km2),
                        ("Çevre (km)", summary.perimeter_km),
                        ("Coğrafi obje", summary.feature_count),
                        ("Köşe sayısı", summary.vertex_count),
                        ("Kaynak CRS", summary.source_crs),
                        ("Alan hesabı CRS", summary.area_crs),
                        ("Merkez enlem", summary.centroid[0]),
                        ("Merkez boylam", summary.centroid[1]),
                    ]
                    excel = build_excel(
                        climate_data,
                        metadata_rows=metadata_rows,
                        area_rows=area_rows,
                        analysis_tables=analysis_tables,
                    )
                    csv_data = dataframe_to_csv(climate_data)
                    graph_png = build_timeseries_png(climate_data)
                    chart_suite = build_academic_chart_suite(
                        climate_data,
                        drought_table=spi_table,
                        event_table=academic_results.get("Kuraklık Olayları"),
                    )
                    map_png = build_area_map_png(
                        summary.gdf_wgs84,
                        summary.centroid,
                        source_note=st.session_state.get(
                            "area_source_note", "Kullanıcı tarafından yüklenen çalışma alanı"
                        ),
                    )
                    readme = (
                        "ZETRİKLİM GERÇEK VERİ VE ANALİZ PAKETİ\n\n"
                        f"Kaynak: {climate_model}\n"
                        f"Dönem: {start_date} – {end_date}\n"
                        f"Kayıt sayısı: {len(climate_data):,}\n"
                        f"Örnekleme: {'Havza alan ortalaması' if provider == 'Google Earth Engine' else 'Çalışma alanının merkezindeki ERA5-Land grid hücresi'}.\n\n"
                        "DOSYALAR\n"
                        "- zetriklim-veri.xlsx: veri, kaynak, alan ve özet sayfaları\n"
                        "- zetriklim-veri.csv: CBS ve istatistik yazılımları için tablo\n"
                        "- calisma-alani.geojson: çalışma alanı sınırı\n"
                        "- zaman-serisi.png: iklim grafiği\n"
                        "- grafik-*.png: ayrı indirilebilir yağış, sıcaklık, su dengesi, kuraklık ve klimatoloji grafikleri\n"
                        "- calisma-alani-haritasi.png: çalışma alanı / havza sınırı\n"
                        "- [YONTEM]_[DONEM].tif: havza sınırına kırpılmış CBS raster katmanı\n"
                        "- [YONTEM]_[DONEM].png: lejantlı akademik harita önizlemesi\n"
                        "- *-etkilesimli-harita.html: PNG ile aynı doğrulanmış karolardan üretilen etkileşimli harita\n"
                        "- [HARITA].tif: QGIS/ArcGIS için havzaya kırpılmış jeoreferanslı renkli raster\n"
                        "- [HARITA]-shp.zip: renk sınıflarının poligonları ve metadata\n"
                        "- harita-kalite-kontrol.json: karo erişimi, alan kapsaması, yükleme süresi ve uyarılar\n"
                        "- raster-analiz-metadata.json: raster kaynağı, formül, dönem, çözünürlük ve sahne sayısı\n"
                        "- zetriklim-metadata.json: veri kaynağı ve işlem izi\n"
                    ).encode("utf-8")
                    if academic_mode:
                        if selected_analysis in {"NDVI", "EVI", "LST"}:
                            readme += (
                                "\nUZAKTAN ALGILAMA ANALİZ DOSYALARI\n"
                                "- bilimsel-rapor.html: yöntem, bulgu tabloları ve kaynaklar\n"
                                "- uzaktan-algilama-ozeti.csv: değer aralığı ve ilk–son dönem değişimi\n"
                                "- egilim-ve-degisim.csv: Mann–Kendall, Sen eğimi ve Pettitt sonuçları\n"
                                "- mevsimsel-profil.csv: aylık ortalama, medyan ve %10–%90 aralığı\n"
                                "- anomali-serisi.csv: mutlak ve standartlaştırılmış aylık anomaliler\n"
                                "- kalite-kontrol.csv: eksik kayıt, fiziksel aralık ve süreklilik kontrolleri\n"
                                "- gecikmeli-iliski.csv: seçildiyse kuraklıkla gecikmeli korelasyonlar\n"
                            ).encode("utf-8")
                        else:
                            readme += (
                                "\nAKADEMİK ARAŞTIRMA DOSYALARI\n"
                                "- bilimsel-rapor.html: yöntem, bulgu tabloları ve kaynaklar\n"
                                "- akademik-kuraklik-serisi.csv: SPI ve SPEI zaman serileri\n"
                                "- kuraklik-olaylari.csv: olay başlangıcı, bitişi, süre, şiddet ve yoğunluk\n"
                                "- egilim-ve-degisim.csv: Mann–Kendall, Sen eğimi ve Pettitt sonuçları\n"
                                "- gecikmeli-iliski.csv: kuraklık–NDVI/EVI/LST gecikmeli korelasyonları\n"
                                "- kaynak-dogrulama.csv: Bias, MAE, RMSE, korelasyon ve KGE\n"
                                "- belirsizlik.csv: kaynaklar arası ensemble yayılımı\n"
                                "- kalite-kontrol.csv: eksik kayıt, aralık ve zaman sürekliliği kontrolleri\n"
                                "- degerler-tablosu-tez.xlsx: su yılına göre SPI-3, SPI-6 ve SPI-12 değer/sınıf tablosu\n"
                            ).encode("utf-8")
                    files = {
                        "zetriklim-veri.xlsx": excel,
                        "zetriklim-veri.csv": csv_data,
                        "calisma-alani.geojson": area_geojson,
                        "zaman-serisi.png": graph_png,
                        "calisma-alani-haritasi.png": map_png,
                        "zetriklim-metadata.json": metadata,
                        "BENI-OKU.txt": readme,
                    }
                    files.update(chart_suite)
                    if spi_table is not None and not spi_table.empty:
                        files["degerler-tablosu-tez.xlsx"] = build_spi_thesis_excel(
                            spi_table
                        )
                    if academic_mode:
                        academic_file_names = {
                            "Kuraklık Serisi": "akademik-kuraklik-serisi.csv",
                            "Kuraklık Olayları": "kuraklik-olaylari.csv",
                            "Eğilim ve Değişim": "egilim-ve-degisim.csv",
                            "Gecikmeli İlişki": "gecikmeli-iliski.csv",
                            "Kaynak Doğrulama": "kaynak-dogrulama.csv",
                            "Belirsizlik": "belirsizlik.csv",
                            "Kalite Kontrol": "kalite-kontrol.csv",
                            "Dağılım Uyum": "dagilim-uyum-testleri.csv",
                            "Uzaktan Algılama Özeti": "uzaktan-algilama-ozeti.csv",
                            "Mevsimsel Profil": "mevsimsel-profil.csv",
                            "Anomali Serisi": "anomali-serisi.csv",
                        }
                        for result_name, file_name in academic_file_names.items():
                            result_table = academic_results.get(result_name)
                            if result_table is not None and not result_table.empty:
                                files[file_name] = dataframe_to_csv(result_table)
                        files["bilimsel-rapor.html"] = build_academic_report_html(
                            study=academic_study,
                            config=academic_params,
                            results=academic_results,
                            source_note=climate_model,
                            context={
                                "Analiz yapısı": application_mode,
                                "Ana analiz odağı": selected_analysis,
                                "Analiz dönemi": f"{start_date} – {end_date}",
                                "Kayıt sayısı": f"{len(climate_data):,}",
                                "Çalışma alanı": f"{summary.area_km2:,.2f} km²",
                                "Çevre": f"{summary.perimeter_km:,.2f} km",
                                "Coğrafi obje sayısı": summary.feature_count,
                                "Kaynak CRS": summary.source_crs,
                                "Alan hesabı CRS": summary.area_crs,
                                "Sağlayıcı": provider,
                                "Ürün": product,
                            },
                            figures={
                                "Şekil 1. Çalışma alanı sınır haritası": map_png,
                                "Şekil 2. Aylık seriler ve 12 aylık hareketli ortalamalar": graph_png,
                                **{
                                    f"Şekil {index + 3}. {name.removeprefix('grafik-').removesuffix('.png').replace('-', ' ').title()}": content
                                    for index, (name, content) in enumerate(chart_suite.items())
                                },
                            },
                        )
                    remote_errors = list(source_analysis_errors)
                    climate_engine_tile_url = None
                    climate_engine_layers = []
                    if provider == "Climate Engine":
                        spi_map_scale = int((analysis_params.get("scales") or [3])[0])
                        spi_map_scales = sorted(
                            {int(value) for value in (analysis_params.get("scales") or [3])}
                        )
                        reference_start = int(
                            academic_params.get("baseline_start", start_date.year)
                            if academic_mode else start_date.year
                        )
                        reference_end = int(
                            academic_params.get("baseline_end", end_date.year)
                            if academic_mode else end_date.year
                        )
                        ce_map_specs = climate_engine_map_specs(
                            selected_analysis,
                            ce_dataset_id,
                            ce_variable_ids,
                            analysis_product_start_date,
                            map_period_start,
                            spi_map_scales,
                        )
                        ce_map_specs = _attach_expected_map_means(
                            ce_map_specs,
                            climate_data,
                            map_period_start,
                            map_period_end,
                        )
                        generated_map_files, generated_map_layers, generated_map_errors = (
                            regenerate_climate_engine_maps(
                                api_key=climate_engine_key,
                                summary=summary,
                                specs=ce_map_specs,
                                map_end=map_period_end,
                                reference_start=reference_start,
                                reference_end=reference_end,
                                mapid_attempts=3,
                                gee_project=gee_project or None,
                                cloud_limit=int(analysis_params.get("cloud_limit", 30)),
                            )
                        )
                        files.update(generated_map_files)
                        climate_engine_layers = generated_map_layers
                        remote_errors.extend(generated_map_errors)
                        climate_engine_tile_url = next(
                            (
                                layer.get("tile_url")
                                for layer in generated_map_layers
                                if layer.get("tile_url")
                            ),
                            None,
                        )
                        # Eski ikinci harita motoru devre dışı: tek üretim yolu hem tam
                        # analizde hem yalnız-harita işleminde aynı önbelleği ve QA'yı kullanır.
                        ce_map_specs = []
                        ce_map_metadata_list = []
                        dataset_date_ranges = {}
                        map_progress = st.progress(
                            0,
                            text="Harita katmanları hazırlanıyor…",
                        )
                        for map_index, map_spec in enumerate(ce_map_specs, start=1):
                            map_progress.progress(
                                (map_index - 1) / max(len(ce_map_specs), 1),
                                text=(
                                    f"{map_spec['label']} hazırlanıyor "
                                    f"({map_index}/{len(ce_map_specs)})…"
                                ),
                            )
                            try:
                                dataset_name = map_spec["dataset"]
                                if dataset_name not in dataset_date_ranges:
                                    try:
                                        dataset_date_ranges[dataset_name] = fetch_dataset_date_range(
                                            climate_engine_key,
                                            dataset_name,
                                        )
                                    except Exception:
                                        dataset_date_ranges[dataset_name] = None
                                dataset_range = dataset_date_ranges[dataset_name]
                                map_start_date = map_spec["start"]
                                map_end_date = map_period_end
                                if dataset_range:
                                    map_start_date = max(map_start_date, dataset_range[0])
                                    map_end_date = min(map_end_date, dataset_range[1])
                                if map_spec.get("window_months"):
                                    requested_window_start = (
                                        pd.Timestamp(map_end_date).to_period("M").to_timestamp()
                                        - pd.DateOffset(months=int(map_spec["window_months"]) - 1)
                                    ).date()
                                    map_start_date = max(map_start_date, requested_window_start)
                                if map_end_date < map_start_date:
                                    raise ValueError(
                                        f"{dataset_name} için seçilen dönemde harita verisi bulunmuyor."
                                    )
                                map_reference_end = max(
                                    reference_start,
                                    min(reference_end, map_end_date.year - 1),
                                )
                                tile_url, ce_map_metadata = fetch_climate_engine_map_tile(
                                    climate_engine_key,
                                    summary.gdf_wgs84,
                                    map_start_date,
                                    map_end_date,
                                    map_spec["dataset"],
                                    map_spec["variable"],
                                    map_spec["analysis"],
                                    spi_scale_months=int(map_spec.get("spi_scale", spi_map_scale)),
                                    reference_start_year=reference_start,
                                    reference_end_year=map_reference_end,
                                    temporal_statistic=map_spec["statistic"],
                                    map_kind=map_spec.get("map_kind", "values"),
                                    anomaly_calculation=map_spec.get("anomaly_calculation", "anom"),
                                )
                                if climate_engine_tile_url is None:
                                    climate_engine_tile_url = tile_url
                                ce_map = folium.Map(
                                    summary.centroid,
                                    zoom_start=8,
                                    tiles="CartoDB positron",
                                    control_scale=True,
                                )
                                folium.GeoJson(
                                    summary.gdf_wgs84.__geo_interface__,
                                    name="Çalışma alanı / havza sınırı",
                                    style_function=lambda _: {
                                        "color": "#052f42",
                                        "weight": 3.5,
                                        "fillColor": "#ffffff",
                                        "fillOpacity": 0.03,
                                    },
                                ).add_to(ce_map)
                                bounds = summary.bounds
                                ce_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                                add_cartographic_controls(
                                    ce_map,
                                    map_spec["analysis"],
                                    f"Climate Engine · {map_spec['dataset']}",
                                    ce_map_metadata["period"].split("/")[0],
                                    ce_map_metadata["period"].split("/")[1],
                                )
                                map_name = f"{map_spec['slug']}-etkilesimli-harita.html"
                                png_name = f"{map_spec['slug']}.png"
                                overlay_png = None
                                overlay_bounds = None
                                geotiff_bytes = None
                                shp_bytes = None
                                try:
                                    static_png, tile_quality = build_tile_map_png(
                                        tile_url,
                                        summary.gdf_wgs84,
                                        title=map_spec["label"],
                                        analysis=map_spec["analysis"],
                                        source=f"Climate Engine · {map_spec['dataset']} · {map_spec['variable']}",
                                        period=ce_map_metadata["period"],
                                    )
                                    overlay_png = tile_quality.pop("_overlay_png")
                                    overlay_bounds = tile_quality.pop("_overlay_bounds")
                                    geotiff_bytes = tile_quality.pop("_geotiff_bytes")
                                    shp_bytes = tile_quality.pop("_classified_shp_bytes")
                                    files[png_name] = static_png
                                    overlay_data_url = (
                                        "data:image/png;base64,"
                                        + base64.b64encode(overlay_png).decode("ascii")
                                    )
                                    folium.raster_layers.ImageOverlay(
                                        image=overlay_data_url,
                                        bounds=overlay_bounds,
                                        name=f"{map_spec['label']} · havzaya kırpılmış",
                                        opacity=1.0,
                                        interactive=False,
                                        cross_origin=False,
                                        zindex=2,
                                    ).add_to(ce_map)
                                    folium.GeoJson(
                                        summary.gdf_wgs84.__geo_interface__,
                                        name="Havza sınırı",
                                        style_function=lambda _: {
                                            "color": "#052f42",
                                            "weight": 3.5,
                                            "fillOpacity": 0.0,
                                        },
                                    ).add_to(ce_map)
                                    folium.LayerControl(collapsed=False).add_to(ce_map)
                                    files[map_name] = ce_map.get_root().render().encode("utf-8")
                                except Exception as static_map_error:
                                    tile_quality = {
                                        "status": "basarisiz",
                                        "error": str(static_map_error),
                                    }
                                    remote_errors.append(
                                        f"{map_spec['label']} PNG haritası: {static_map_error}"
                                    )
                                ce_map_metadata["label"] = map_spec["label"]
                                ce_map_metadata["file"] = map_name if map_name in files else None
                                ce_map_metadata["png_file"] = (
                                    png_name if png_name in files else None
                                )
                                ce_map_metadata["tile_quality"] = tile_quality
                                geotiff_name = f"{map_spec['slug']}.tif"
                                shp_name = f"{map_spec['slug']}-shp.zip"
                                if geotiff_bytes and shp_bytes:
                                    files[geotiff_name] = geotiff_bytes
                                    files[shp_name] = shp_bytes
                                ce_map_metadata["geotiff_file"] = (
                                    geotiff_name if geotiff_name in files else None
                                )
                                ce_map_metadata["shp_file"] = (
                                    shp_name if shp_name in files else None
                                )
                                ce_map_metadata_list.append(ce_map_metadata)
                                climate_engine_layers.append(
                                    {
                                        "label": map_spec["label"],
                                        "analysis": map_spec["analysis"],
                                        "source": f"Climate Engine · {map_spec['dataset']}",
                                        "start": ce_map_metadata["period"].split("/")[0],
                                        "end": ce_map_metadata["period"].split("/")[1],
                                        "tile_url": tile_url,
                                        "quality": tile_quality,
                                        "html_file": map_name if map_name in files else None,
                                        "png_file": png_name if png_name in files else None,
                                        "overlay_png": overlay_png,
                                        "overlay_bounds": overlay_bounds,
                                        "geotiff_file": geotiff_name if geotiff_name in files else None,
                                        "shp_file": shp_name if shp_name in files else None,
                                    }
                                )
                            except Exception as map_error:
                                remote_errors.append(
                                    f"{map_spec['label']} Climate Engine haritası: {map_error}"
                                )
                                climate_engine_layers.append(
                                    {
                                        "label": map_spec["label"],
                                        "analysis": map_spec["analysis"],
                                        "source": f"Climate Engine · {map_spec['dataset']}",
                                        "start": str(map_spec["start"]),
                                        "end": str(map_period_end),
                                        "tile_url": None,
                                        "quality": {
                                            "status": "basarisiz",
                                            "error": str(map_error),
                                        },
                                        "html_file": None,
                                        "png_file": None,
                                        "geotiff_file": None,
                                        "shp_file": None,
                                    }
                                )
                            finally:
                                map_progress.progress(
                                    map_index / max(len(ce_map_specs), 1),
                                    text=(
                                        f"Harita katmanları hazırlanıyor "
                                        f"({map_index}/{len(ce_map_specs)})"
                                    ),
                                )
                        map_progress.empty()
                        if ce_map_metadata_list:
                            files["climate-engine-harita-metadata.json"] = json.dumps(
                                ce_map_metadata_list,
                                ensure_ascii=False,
                                indent=2,
                                default=str,
                            ).encode("utf-8")
                            files["harita-kalite-kontrol.json"] = json.dumps(
                                [
                                    {
                                        "harita": item.get("label"),
                                        "dataset": item.get("dataset"),
                                        "degisken": item.get("variable"),
                                        "donem": item.get("period"),
                                        "kalite": item.get("tile_quality"),
                                    }
                                    for item in ce_map_metadata_list
                                ],
                                ensure_ascii=False,
                                indent=2,
                                default=str,
                            ).encode("utf-8")

                        # Climate Engine zaman zaman geçici olarak boş/eksik karo döndürüyor.
                        # İlk turda eksik kalan haritaları yeni MapID'lerle bir kez daha
                        # üret. Başarılı haritalar tekrar istenmez; günlük API kotası korunur.
                        if remote_errors and ce_map_specs:
                            initially_failed = {
                                str(layer.get("label"))
                                for layer in climate_engine_layers
                                if not layer.get("png_file")
                            }
                            retry_specs = [
                                spec for spec in ce_map_specs
                                if str(spec["label"]) in initially_failed
                            ]
                            retry_files, retry_layers, retry_errors = regenerate_climate_engine_maps(
                                api_key=climate_engine_key,
                                summary=summary,
                                specs=retry_specs,
                                map_end=map_period_end,
                                reference_start=reference_start,
                                reference_end=reference_end,
                                mapid_attempts=1,
                                gee_project=gee_project or None,
                                cloud_limit=int(analysis_params.get("cloud_limit", 30)),
                            )
                            files.update(retry_files)
                            original_layers = {
                                str(layer.get("label")): layer for layer in climate_engine_layers
                            }
                            retry_layer_index = {
                                str(layer.get("label")): layer for layer in retry_layers
                            }
                            climate_engine_layers = []
                            for spec in ce_map_specs:
                                label = str(spec["label"])
                                retried = retry_layer_index.get(label)
                                original = original_layers.get(label)
                                if retried and retried.get("png_file"):
                                    climate_engine_layers.append(retried)
                                elif original and original.get("png_file"):
                                    climate_engine_layers.append(original)
                                elif retried:
                                    climate_engine_layers.append(retried)
                                elif original:
                                    climate_engine_layers.append(original)

                            failed_labels = {
                                str(layer.get("label"))
                                for layer in climate_engine_layers
                                if not layer.get("png_file")
                            }
                            remote_errors = [
                                error
                                for error in retry_errors
                                if any(label in error for label in failed_labels)
                            ]

                    gee_ok = False
                    if provider == "Google Earth Engine":
                        gee_ok, _ = cached_gee_status(gee_project or None)
                    if provider == "Google Earth Engine" and gee_ok:
                        precipitation_tif = build_climate_geotiff(
                            summary.gdf_wgs84,
                            start_date,
                            end_date,
                            "precipitation",
                            project=gee_project,
                        )
                        precipitation_name = f"yagis_gunluk_ortalama_{start_date}_{end_date}"
                        files[f"{precipitation_name}.tif"] = precipitation_tif
                        files[f"{precipitation_name}.png"] = build_raster_png(
                            precipitation_tif,
                            f"CHIRPS Ortalama Günlük Yağış · {start_date} – {end_date}",
                            boundary=summary.gdf_wgs84,
                            palette="Blues",
                            colorbar_label="Yağış (mm/gün)",
                            fixed_range=None,
                        )
                        temperature_tif = build_climate_geotiff(
                            summary.gdf_wgs84,
                            start_date,
                            end_date,
                            "temperature",
                            project=gee_project,
                        )
                        temperature_name = f"sicaklik_ortalama_{start_date}_{end_date}"
                        files[f"{temperature_name}.tif"] = temperature_tif
                        files[f"{temperature_name}.png"] = build_raster_png(
                            temperature_tif,
                            f"ERA5-Land Ortalama Sıcaklık · {start_date} – {end_date}",
                            boundary=summary.gdf_wgs84,
                            palette="Spectral_r",
                            colorbar_label="Sıcaklık (°C)",
                            fixed_range=None,
                        )
                    remote_methods = {
                        "NDVI", "NDWI", "NDMI", "NDBI", "EVI", "SAVI",
                        "LST", "DEM", "SLOPE", "ASPECT", "TWI",
                    }
                    remote_selected = [
                        method for method in selected_analyses if method in remote_methods
                    ]
                    remote_metadata = []
                    raster_styles = {
                        "NDVI": ("YlGn", "NDVI", (-1.0, 1.0)),
                        "NDWI": ("Blues", "NDWI", (-1.0, 1.0)),
                        "NDMI": ("BrBG", "NDMI", (-1.0, 1.0)),
                        "NDBI": ("magma", "NDBI", (-1.0, 1.0)),
                        "EVI": ("YlGn", "EVI", (-1.0, 1.0)),
                        "SAVI": ("summer", "SAVI", (-1.0, 1.0)),
                        "LST": ("inferno", "Yüzey sıcaklığı (°C)", None),
                        "DEM": ("terrain", "Yükselti (m)", None),
                        "SLOPE": ("YlOrBr", "Eğim (°)", (0.0, 60.0)),
                        "ASPECT": ("hsv", "Bakı (°)", (0.0, 360.0)),
                        "TWI": ("viridis", "TWI", None),
                    }
                    for method in (
                        remote_selected if provider == "Google Earth Engine" else []
                    ):
                        try:
                            raster_tif, method_metadata = build_remote_analysis_geotiff(
                                summary.gdf_wgs84,
                                start_date,
                                end_date,
                                method,
                                project=gee_project or None,
                                cloud_limit=int(analysis_params.get("cloud_limit", 30)),
                            )
                            base_name = (
                                f"{method}_{start_date}_{end_date}"
                                if method in {"NDVI", "NDWI", "NDMI", "NDBI", "EVI", "SAVI", "LST"}
                                else f"{method}_statik"
                            )
                            palette, colorbar_label, fixed_range = raster_styles[method]
                            files[f"{base_name}.tif"] = raster_tif
                            files[f"{base_name}.png"] = build_raster_png(
                                raster_tif,
                                f"{method} · {ANALYSIS_METHODS[method]['title']}",
                                boundary=summary.gdf_wgs84,
                                palette=palette,
                                colorbar_label=colorbar_label,
                                fixed_range=fixed_range,
                            )
                            remote_metadata.append(method_metadata)
                        except Exception as method_error:
                            remote_errors.append(f"{method}: {method_error}")
                    if remote_metadata:
                        files["raster-analiz-metadata.json"] = json.dumps(
                            remote_metadata,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        ).encode("utf-8")
                    if remote_errors:
                        files["analiz-uyarilari.txt"] = (
                            "Üretilemeyen analizler\n\n" + "\n".join(remote_errors)
                        ).encode("utf-8")
                    if spi_table is not None:
                        files["spi-sonuclari.csv"] = dataframe_to_csv(spi_table)
                        if provider == "Google Earth Engine" and gee_ok:
                            for selected_scale in (analysis_params.get("scales") or [3]):
                                selected_scale = int(selected_scale)
                                spi_tif = build_chirps_spi_geotiff(
                                    summary.gdf_wgs84,
                                    end_date,
                                    selected_scale,
                                    baseline_start=int(academic_params.get("baseline_start", 1981)),
                                    baseline_end=int(academic_params.get("baseline_end", 2024)),
                                    project=gee_project,
                                )
                                files[f"SPI-{selected_scale}_{end_date:%Y-%m}.tif"] = spi_tif
                                files[f"SPI-{selected_scale}_{end_date:%Y-%m}.png"] = build_raster_png(
                                    spi_tif,
                                    f"CHIRPS SPI-{selected_scale} · {end_date:%Y-%m}",
                                    boundary=summary.gdf_wgs84,
                                )
                    st.session_state.output_files = files
                    st.session_state.output_package = build_complete_package(
                        select_output_files(files, output_formats)
                    )
                    st.session_state.output_metadata = metadata
                    st.session_state.output_data = climate_data
                    st.session_state.output_source = climate_model
                    st.session_state.output_analysis_errors = remote_errors
                    st.session_state.output_tile_url = climate_engine_tile_url
                    st.session_state.output_tile_layers = climate_engine_layers
                    st.session_state.output_boundary = summary.gdf_wgs84.to_json()
                    st.session_state.output_config = current_output_config
                    st.session_state.output_map_period = (map_period_start, map_period_end)
                    st.session_state.output_academic_results = academic_results
                    st.session_state.output_elapsed_seconds = round(
                        time_module.perf_counter() - analysis_started_at,
                        1,
                    )
                    # Geçici uydu/tepki kaynağı hatasıyla tamamlanan sonuç, aynı
                    # ayarlar için kalıcı "başarılı" sonuç sayılmaz. İklim serisi
                    # ve haritaların kendi önbellekleri korunur; sonraki çalıştırma
                    # yalnız eksik tepki serisini tekrar dener.
                    complete_map_outputs = (
                        not climate_engine_layers
                        or all(layer.get("png_file") for layer in climate_engine_layers)
                    )
                    if (
                        not source_analysis_errors
                        and not remote_errors
                        and complete_map_outputs
                    ):
                        _write_analysis_result_cache(
                            result_cache_path,
                            {
                                "version": RESULT_CACHE_VERSION,
                                "area_label": st.session_state.get(
                                    "area_source_note", "Çalışma alanı"
                                ),
                                **{
                                    key: st.session_state.get(key)
                                    for key in (
                                        "output_files", "output_metadata", "output_data",
                                        "output_source", "output_analysis_errors",
                                        "output_tile_url", "output_tile_layers",
                                        "output_boundary", "output_config",
                                        "output_map_period", "output_academic_results",
                                        "output_elapsed_seconds",
                                    )
                                },
                            },
                        )
                if st.session_state.get("output_analysis_errors"):
                    completed_map_count = sum(
                        1 for layer in climate_engine_layers if layer.get("png_file")
                    )
                    st.warning(
                        f"Tablo ve {completed_map_count}/"
                        f"{len(climate_engine_layers)} harita hazır. Eksik haritalar aşağıdaki "
                        "kalite bölümünde yeniden oluşturulabilir."
                    )
                else:
                    st.success(
                        f"{selected_analysis} analizi tamamlandı: {len(climate_data):,} kayıt. "
                        "Tablolar, haritalar ve CBS paketi hazır."
                    )
                if unsupported:
                    st.warning(
                        "Bu bağlayıcıda desteklenmeyen değişkenler pakete eklenmedi: "
                        + ", ".join(unsupported)
                    )
                if st.session_state.get("output_analysis_errors"):
                    with st.expander("Eksik haritaların teknik ayrıntıları", expanded=False):
                        st.markdown(
                            "\n".join(
                                f"- {item}"
                                for item in st.session_state.output_analysis_errors
                            )
                        )
            except Exception as exc:
                st.error(f"Veri indirme veya çıktı oluşturma başarısız: {exc}")

    if st.session_state.get("output_package"):
        result_layers = st.session_state.get("output_tile_layers", [])
        ready_map_count = sum(
            1 for layer in result_layers
            if (layer.get("quality") or {}).get("status") == "uygun"
        )
        failed_map_count = sum(1 for layer in result_layers if not layer.get("png_file"))
        result_state = "Tüm çıktılar hazır" if not failed_map_count else "Çıktılar kısmen hazır"
        st.markdown(
            f"""
            <div class="result-hero">
              <div class="result-kicker">05 · Bulgular ve çıktılar</div>
              <h2>{result_state}</h2>
              <p>{selected_analysis} · {st.session_state.get('output_source', '—')} · {len(st.session_state.get('output_data', [])):,} kayıt ·
              {ready_map_count}/{len(result_layers)} harita içerik doğrulamasını geçti</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        rs1, rs2, rs3, rs4 = st.columns(4)
        rs1.metric("Kayıt", f"{len(st.session_state.get('output_data', [])):,}")
        rs2.metric("Hazır harita", f"{ready_map_count}/{len(result_layers)}" if result_layers else "—")
        rs3.metric("Çalışma alanı", f"{summary.area_km2:,.2f} km²" if summary else "—")
        elapsed_seconds = st.session_state.get("output_elapsed_seconds")
        rs4.metric(
            "Gerçek işlem süresi",
            f"{float(elapsed_seconds) / 60:.1f} dk" if elapsed_seconds else "—",
        )
        st.caption(
            "Aşağıdaki içerik önem sırasına göre düzenlendi: önce temel bulgular, "
            "ardından haritalar ve en sonda ayrıntılı dosyalar."
        )
        academic_output = st.session_state.get("output_academic_results", {})
        if (
            academic_mode
            and academic_output
            and selected_analysis in {"NDVI", "EVI", "LST"}
        ):
            st.subheader(f"{selected_analysis} analiz bulguları")
            remote_summary = academic_output.get(
                "Uzaktan Algılama Özeti", pd.DataFrame()
            )
            trend_table = academic_output.get("Eğilim ve Değişim", pd.DataFrame())
            seasonal_table = academic_output.get("Mevsimsel Profil", pd.DataFrame())
            anomaly_table = academic_output.get("Anomali Serisi", pd.DataFrame())
            quality_table = academic_output.get("Kalite Kontrol", pd.DataFrame())
            lag_table = academic_output.get("Gecikmeli İlişki", pd.DataFrame())

            main_summary = (
                remote_summary.loc[
                    remote_summary.get("Değişken", pd.Series(dtype=str))
                    == selected_analysis
                ].head(1)
                if not remote_summary.empty else pd.DataFrame()
            )
            main_trend = (
                trend_table.loc[
                    trend_table.get("Değişken", pd.Series(dtype=str))
                    == selected_analysis
                ].head(1)
                if not trend_table.empty else pd.DataFrame()
            )
            rm1, rm2, rm3, rm4 = st.columns(4)
            if not main_summary.empty:
                summary_row = main_summary.iloc[0]
                rm1.metric("Geçerli aylık gözlem", f"{int(summary_row['Geçerli gözlem']):,}")
                rm2.metric("Ortalama", f"{float(summary_row['Ortalama']):.4f}")
                change_value = summary_row.get("Dönemsel değişim (%)")
                rm4.metric(
                    "İlk–son dönem değişimi",
                    f"%{float(change_value):+.2f}" if pd.notna(change_value) else "—",
                )
            else:
                rm1.metric("Geçerli aylık gözlem", "—")
                rm2.metric("Ortalama", "—")
                rm4.metric("İlk–son dönem değişimi", "—")
            if not main_trend.empty:
                trend_row = main_trend.iloc[0]
                rm3.metric(
                    "Eğilim",
                    str(trend_row.get("Yön", "—")),
                    delta=(
                        f"{float(trend_row.get('Sen eğimi / ay')) * 12:+.5f}/yıl"
                        if pd.notna(trend_row.get("Sen eğimi / ay")) else None
                    ),
                )
            else:
                rm3.metric("Eğilim", "—")

            remote_tabs = st.tabs(
                ["Özet", "Eğilim ve kırılma", "Mevsimsel profil", "Anomaliler", "Kalite ve ilişki"]
            )
            with remote_tabs[0]:
                st.caption(
                    "Temel değer aralığı, ilk ve son dönem farkı ile en güçlü ve en "
                    "zayıf mevsim birlikte özetlenir."
                )
                st.dataframe(remote_summary, width="stretch", hide_index=True)
            with remote_tabs[1]:
                st.caption(
                    "Mann–Kendall ve Sen eğimi uzun dönem yönünü; Pettitt testi olası "
                    "değişim tarihini gösterir."
                )
                st.dataframe(trend_table, width="stretch", hide_index=True)
            with remote_tabs[2]:
                st.caption(
                    "Her takvim ayının ortalama, medyan, değişkenlik ve %10–%90 "
                    "aralığıdır."
                )
                st.dataframe(seasonal_table, width="stretch", hide_index=True)
            with remote_tabs[3]:
                st.caption(
                    "Gözlenen değer ile seçilen referans döneminin aynı ay klimatolojisi "
                    "arasındaki mutlak ve standartlaştırılmış farktır."
                )
                st.dataframe(anomaly_table.tail(180), width="stretch", hide_index=True)
            with remote_tabs[4]:
                st.markdown("**Veri kalitesi**")
                st.dataframe(quality_table, width="stretch", hide_index=True)
                if not lag_table.empty:
                    st.markdown("**Kuraklıkla en güçlü gecikmeli ilişkiler**")
                    best_lags = lag_table[
                        lag_table.get("En güçlü gecikme", False) == True  # noqa: E712
                    ]
                    st.dataframe(best_lags, width="stretch", hide_index=True)
                else:
                    st.info(
                        "Kuraklık ilişkisi seçilmedi. NDVI eğilim, anomali, mevsimsellik "
                        "ve kalite sonuçları bundan bağımsız olarak tamamlandı."
                    )

        if academic_mode and academic_output and selected_analysis == "SPI":
            st.subheader("İleri analiz bulguları")
            event_table = academic_output.get("Kuraklık Olayları", pd.DataFrame())
            trend_table = academic_output.get("Eğilim ve Değişim", pd.DataFrame())
            lag_table = academic_output.get("Gecikmeli İlişki", pd.DataFrame())
            validation_table = academic_output.get("Kaynak Doğrulama", pd.DataFrame())
            am1, am2, am3, am4 = st.columns(4)
            am1.metric("Kuraklık olayı", len(event_table))
            am2.metric(
                "Anlamlı eğilim",
                int(trend_table.get("Anlamlı eğilim", pd.Series(dtype=bool)).fillna(False).sum()),
            )
            am3.metric(
                "Anlamlı gecikmeli ilişki",
                int(lag_table.get("Anlamlı", pd.Series(dtype=bool)).fillna(False).sum()),
            )
            am4.metric("Doğrulama çifti", len(validation_table))

            result_tabs = st.tabs(
                ["Kuraklık olayları", "Eğilim ve değişim", "Gecikmeli tepki", "Doğrulama ve belirsizlik"]
            )
            with result_tabs[0]:
                if event_table.empty:
                    st.info("Seçilen eşikte kuraklık olayı bulunmadı veya seri yetersiz.")
                else:
                    st.caption(
                        "Süre, eşik altında kesintisiz kalan takvim ayı sayısıdır. "
                        "Süre 1 ise olay türü ‘Tek aylık’; süre 2 veya daha fazlaysa "
                        "‘Çok aylık’ olarak gösterilir."
                    )
                    st.dataframe(
                        event_table,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Başlangıç": st.column_config.DateColumn(format="DD.MM.YYYY"),
                            "Bitiş": st.column_config.DateColumn(format="DD.MM.YYYY"),
                            "Tepe ayı": st.column_config.DateColumn(format="MM.YYYY"),
                            "Süre (ay)": st.column_config.NumberColumn(format="%d"),
                            "Olay türü": st.column_config.TextColumn(),
                        },
                    )
            with result_tabs[1]:
                if trend_table.empty:
                    st.info("Eğilim testi için yeterli geçerli gözlem bulunmadı.")
                else:
                    st.dataframe(trend_table, width="stretch", hide_index=True)
            with result_tabs[2]:
                best_lags = (
                    lag_table[lag_table.get("En güçlü gecikme", False) == True]  # noqa: E712
                    if not lag_table.empty else lag_table
                )
                if best_lags.empty:
                    st.info(
                        "Gecikmeli tepki hesaplanamadı: NDVI/EVI/LST ile kuraklık "
                        "serisi arasında en az 8 ortak ve değişken aylık gözlem gerekir."
                    )
                else:
                    st.dataframe(best_lags, width="stretch", hide_index=True)
            with result_tabs[3]:
                if not validation_table.empty:
                    st.markdown("**Kaynak karşılaştırması**")
                    st.dataframe(validation_table, width="stretch", hide_index=True)
                else:
                    st.info(
                        "Kaynak doğrulaması için aynı aylarda en az iki bağımsız "
                        "yağış serisi gerekir; CHIRPS ve ERA5 otomatik olarak istenir."
                    )
                uncertainty = academic_output.get("Belirsizlik", pd.DataFrame())
                if not uncertainty.empty:
                    st.markdown("**Kaynaklar arası belirsizlik serisi**")
                    st.dataframe(uncertainty.tail(120), width="stretch", hide_index=True)
                else:
                    st.info(
                        "Belirsizlik serisi üretilemedi; iki yağış kaynağının ortak "
                        "geçerli ayları bulunamadı."
                    )

        st.markdown('<div class="step">Temel bulgu · Zaman serisi</div>', unsafe_allow_html=True)
        st.image(
            st.session_state.output_files["zaman-serisi.png"],
            caption=(
                f"{selected_analysis} · {start_date} – {end_date} · "
                "NoData değerleri kalite kontrolünde grafik dışında bırakılmıştır."
            ),
            width="stretch",
        )
        st.markdown('<div class="step">Hızlı indirme</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "Seçilen çıktıları indir (ZIP)",
            st.session_state.output_package,
            file_name="zetriklim-analiz-paketi.zip",
            mime="application/zip",
            use_container_width=True,
        )
        d2.download_button(
            "Excel indir",
            st.session_state.output_files["zetriklim-veri.xlsx"],
            file_name="zetriklim-veri.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        d3.download_button(
            "CSV indir",
            st.session_state.output_files["zetriklim-veri.csv"],
            file_name="zetriklim-veri.csv",
            mime="text/csv",
            use_container_width=True,
        )
        with st.expander("Diğer veri, CBS ve rapor dosyaları", expanded=False):
            render_secondary_downloads(st.session_state.output_files)
        chart_names = sorted(
            name for name in st.session_state.output_files
            if name.startswith("grafik-") and name.lower().endswith(".png")
        )
        if chart_names:
            with st.expander(
                f"Akademik grafik paketi · {len(chart_names)} grafik",
                expanded=False,
            ):
                render_chart_gallery(st.session_state.output_files, chart_names)
        raster_names = [name for name in st.session_state.output_files if name.lower().endswith(".tif")]
        if raster_names:
            st.markdown('<div class="step">CBS raster katmanları</div>', unsafe_allow_html=True)
            raster_columns = st.columns(min(3, len(raster_names)))
            for index, raster_name in enumerate(raster_names):
                label = (
                    "Yağış GeoTIFF"
                    if raster_name.startswith("yagis_")
                    else "Sıcaklık GeoTIFF"
                    if raster_name.startswith("sicaklik_")
                    else f"{raster_name.split('_')[0]} GeoTIFF"
                )
                raster_columns[index % len(raster_columns)].download_button(
                    label,
                    st.session_state.output_files[raster_name],
                    file_name=raster_name,
                    mime="image/tiff",
                    use_container_width=True,
                    key=f"raster_download_{index}_{raster_name}",
                )
        html_map_names = [
            name for name in st.session_state.output_files
            if name.lower().endswith(".html") and "harita" in name.lower()
        ]
        tile_layers = st.session_state.get("output_tile_layers", [])
        if html_map_names or tile_layers:
            st.markdown('<div class="step">Mekânsal bulgular · Doğrulanmış haritalar</div>', unsafe_allow_html=True)
            if tile_layers and summary:
                quality_rows = []
                for layer in tile_layers:
                    quality = layer.get("quality") or {}
                    coverage = quality.get("coverage_ratio")
                    quality_rows.append(
                        {
                            "Harita": layer["label"],
                            "Kaynak": layer["source"],
                            "Dönem": f"{layer['start']} – {layer['end']}",
                            "Durum": (
                                "Doğrulandı"
                                if quality.get("status") == "uygun"
                                else "Üretilemedi"
                                if quality.get("status") == "basarisiz"
                                else "Kontrol gerekli"
                            ),
                            "Alan kapsaması (%)": (
                                round(float(coverage) * 100, 1)
                                if coverage is not None
                                else None
                            ),
                            "Doğrulanan karo": (
                                f"{quality.get('downloaded_tiles')}/{quality.get('requested_tiles')}"
                                if quality.get("requested_tiles") is not None
                                else "—"
                            ),
                            "Tamamlayıcı karo": quality.get("fallback_tiles", 0),
                            "Üretim (sn)": quality.get("generation_seconds"),
                            "Ort. yükleme (sn)": quality.get("mean_tile_seconds"),
                        }
                    )
                with st.expander("Harita kalite ve kaynak ayrıntıları", expanded=False):
                    st.dataframe(
                        pd.DataFrame(quality_rows),
                        width="stretch",
                        hide_index=True,
                    )
                map_tabs = st.tabs([layer["label"] for layer in tile_layers])
                for map_tab_index, (tab, layer) in enumerate(zip(map_tabs, tile_layers)):
                    with tab:
                        quality = layer.get("quality") or {}
                        if layer.get("overlay_png") and layer.get("overlay_bounds"):
                            result_map = folium.Map(
                                summary.centroid,
                                zoom_start=8,
                                tiles="CartoDB positron",
                                control_scale=True,
                            )
                            result_overlay_url = (
                                "data:image/png;base64,"
                                + base64.b64encode(layer["overlay_png"]).decode("ascii")
                            )
                            folium.raster_layers.ImageOverlay(
                                image=result_overlay_url,
                                bounds=layer["overlay_bounds"],
                                name=layer["label"],
                                opacity=1.0,
                                interactive=False,
                                cross_origin=False,
                                zindex=2,
                            ).add_to(result_map)
                            folium.GeoJson(
                                summary.gdf_wgs84.__geo_interface__,
                                name="Çalışma alanı / havza sınırı",
                                style_function=lambda _: {
                                    "color": "#052f42",
                                    "weight": 3.5,
                                    "fillOpacity": 0.0,
                                },
                            ).add_to(result_map)
                            bounds = summary.bounds
                            result_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                            add_cartographic_controls(
                                result_map,
                                layer["analysis"],
                                layer["source"],
                                layer["start"],
                                layer["end"],
                            )
                            folium.LayerControl(collapsed=False).add_to(result_map)
                            st_folium(
                                result_map,
                                height=560,
                                width="stretch",
                                returned_objects=[],
                                key=f"result_map_{map_tab_index}_{layer['label']}",
                            )
                            if quality.get("status") == "uygun":
                                st.success(
                                    f"İçerik doğrulandı: havza kapsaması "
                                    f"%{float(quality.get('coverage_ratio', 0)) * 100:.1f} · "
                                    f"birincil karo {quality.get('downloaded_tiles')}/{quality.get('requested_tiles')} · "
                                    f"tamamlayıcı karo {quality.get('fallback_tiles', 0)}. "
                                    f"Değer aralığı {quality.get('value_min', '—')} – "
                                    f"{quality.get('value_max', '—')} · çıktı ızgarası "
                                    f"{quality.get('export_grid_size_m', quality.get('pixel_size_m', '—'))} m"
                                    + (
                                        f" · kaynak çözünürlüğü yaklaşık "
                                        f"{float(quality['source_native_resolution_m']) / 1000:.1f} km. "
                                        if quality.get('source_native_resolution_m') else ". "
                                    )
                                    + "Katman havza sınırına "
                                    "kırpılmış; QGIS çıktısı tek bantlı Float32 olarak hazırlanmıştır."
                                )
                            elif quality:
                                st.warning(
                                    "Harita kalite kontrolü ek inceleme gerektiriyor: "
                                    + str(quality.get("error") or quality.get("status"))
                                )
                        else:
                            st.error(
                                "Bu harita üretilemedi: "
                                + str(quality.get("error") or "Climate Engine katmanı alınamadı.")
                            )
                        st.caption("Bu haritaya ait dosyalar")
                        download_columns = st.columns(3)
                        png_file = layer.get("png_file")
                        html_file = layer.get("html_file")
                        geotiff_file = layer.get("geotiff_file")
                        shp_file = layer.get("shp_file")
                        if png_file and png_file in st.session_state.output_files:
                            download_columns[0].download_button(
                                "PNG haritayı indir",
                                st.session_state.output_files[png_file],
                                file_name=png_file,
                                mime="image/png",
                                use_container_width=True,
                                key=f"tab_png_{map_tab_index}_{png_file}",
                            )
                        if html_file and html_file in st.session_state.output_files:
                            download_columns[1].download_button(
                                "Etkileşimli haritayı indir",
                                st.session_state.output_files[html_file],
                                file_name=html_file,
                                mime="text/html",
                                use_container_width=True,
                                key=f"tab_html_{map_tab_index}_{html_file}",
                            )
                        if geotiff_file and geotiff_file in st.session_state.output_files:
                            download_columns[2].download_button(
                                "GeoTIFF indir",
                                st.session_state.output_files[geotiff_file],
                                file_name=geotiff_file,
                                mime="image/tiff",
                                use_container_width=True,
                                key=f"tab_tif_{map_tab_index}_{geotiff_file}",
                            )
                        if shp_file and shp_file in st.session_state.output_files:
                            download_columns[0].download_button(
                                "Shapefile indir",
                                st.session_state.output_files[shp_file],
                                file_name=shp_file,
                                mime="application/zip",
                                use_container_width=True,
                                key=f"tab_shp_{map_tab_index}_{shp_file}",
                            )
                st.caption(
                    "Her harita aynı havza sınırı, kuzey oku, koordinat göstergesi, ölçek ve "
                    "değişkene özgü lejantla üretilir. SPI haritası seçilen birikim ölçeğinde "
                    "harita bitiş tarihindeki durumu; diğer dağılım ve anomali haritaları seçilen "
                    "harita tarih aralığının tamamını gösterir. GeoTIFF dosyaları bilimsel "
                    "Float32 değerleri içerir; Shapefile paketi sınıfları ve metadata bilgisini taşır."
                )
        preview_names = [
            name
            for name in st.session_state.output_files
            if name.lower().endswith(".png")
            and name not in {"zaman-serisi.png", "calisma-alani-haritasi.png"}
            and not name.startswith("grafik-")
        ]
        if preview_names:
            with st.expander(
                "Toplu PNG önizlemeleri (etkileşimli haritaların kopyaları)",
                expanded=False,
            ):
                preview_columns = st.columns(min(2, len(preview_names)))
                for index, preview_name in enumerate(preview_names):
                    preview_column = preview_columns[index % len(preview_columns)]
                    preview_column.image(
                        st.session_state.output_files[preview_name],
                        caption=preview_name.replace("_", " ").replace(".png", ""),
                        width="stretch",
                    )
                    preview_column.download_button(
                        "PNG haritayı indir",
                        st.session_state.output_files[preview_name],
                        file_name=preview_name,
                        mime="image/png",
                        use_container_width=True,
                        key=f"map_png_download_{index}_{preview_name}",
                    )
st.markdown(
    """
    <div style="
      margin-top:3rem;padding:1.25rem 1.5rem;border-top:1px solid rgba(0,128,136,.22);
      text-align:center;color:#496b73;background:rgba(255,255,255,.45);border-radius:18px 18px 0 0">
      <strong style="color:#075b68;letter-spacing:.08em">ZETRİKLİM</strong><br>
      Havza, iklim ve uzaktan algılama analiz platformu<br>
      <span style="font-size:.82rem">Zeliha Konuk · 2026</span>
    </div>
    """,
    unsafe_allow_html=True,
)


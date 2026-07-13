"""Google Earth Engine CHIRPS bağlantısı ve gerçek SPI raster üretimi."""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from google.oauth2.credentials import Credentials
from dateutil.relativedelta import relativedelta
from rasterio.io import MemoryFile
from rasterio.features import geometry_mask
from rasterio.mask import mask as raster_mask

from zetriklim.spi import calculate_spi_pixel_stack


CHIRPS_DAILY = "UCSB-CHG/CHIRPS/DAILY"
CHIRPS_SCALE_M = 5566

LAND_COVER_CLASSES = {
    10: "Ağaç örtüsü",
    20: "Çalılık",
    30: "Otlak/mera",
    40: "Tarım alanı",
    50: "Yapılaşmış alan",
    60: "Çıplak/seyrek bitkili alan",
    80: "Su",
    90: "Sulak alan",
}

REMOTE_ANALYSIS_SPECS = {
    "NDVI": {
        "source": "COPERNICUS/S2_SR_HARMONIZED",
        "formula": "(NIR - Red) / (NIR + Red)",
        "native_scale_m": 10,
        "unit": "index",
    },
    "NDWI": {
        "source": "COPERNICUS/S2_SR_HARMONIZED",
        "formula": "(Green - NIR) / (Green + NIR)",
        "native_scale_m": 10,
        "unit": "index",
    },
    "NDMI": {
        "source": "COPERNICUS/S2_SR_HARMONIZED",
        "formula": "(NIR - SWIR1) / (NIR + SWIR1)",
        "native_scale_m": 20,
        "unit": "index",
    },
    "NDBI": {
        "source": "COPERNICUS/S2_SR_HARMONIZED",
        "formula": "(SWIR1 - NIR) / (SWIR1 + NIR)",
        "native_scale_m": 20,
        "unit": "index",
    },
    "EVI": {
        "source": "COPERNICUS/S2_SR_HARMONIZED",
        "formula": "2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)",
        "native_scale_m": 10,
        "unit": "index",
    },
    "SAVI": {
        "source": "COPERNICUS/S2_SR_HARMONIZED",
        "formula": "1.5 * (NIR - Red) / (NIR + Red + 0.5)",
        "native_scale_m": 10,
        "unit": "index",
    },
    "LST": {
        "source": "LANDSAT/LC08+C09/C02/T1_L2",
        "formula": "ST_B10 * 0.00341802 + 149.0 - 273.15",
        "native_scale_m": 30,
        "unit": "°C",
    },
    "DEM": {
        "source": "USGS/SRTMGL1_003",
        "formula": "SRTM elevation",
        "native_scale_m": 30,
        "unit": "m",
    },
    "SLOPE": {
        "source": "USGS/SRTMGL1_003",
        "formula": "ee.Terrain.slope(DEM)",
        "native_scale_m": 30,
        "unit": "degree",
    },
    "ASPECT": {
        "source": "USGS/SRTMGL1_003",
        "formula": "ee.Terrain.aspect(DEM)",
        "native_scale_m": 30,
        "unit": "degree",
    },
    "TWI": {
        "source": "MERIT/Hydro/v1_0_1 + USGS/SRTMGL1_003",
        "formula": "ln((upstream_area_m2 / cell_width_m) / tan(slope_radians))",
        "native_scale_m": 90,
        "unit": "index",
    },
}


def project_id(explicit_project: str | None = None) -> str | None:
    return explicit_project or os.getenv("GOOGLE_EARTH_ENGINE_PROJECT")


def create_user_auth_flow() -> tuple[str, str]:
    """Earth Engine'in resmi uzak/notebook OAuth akışını başlatır."""
    flow = ee.oauth.Flow("notebook")
    return flow.auth_url, flow.code_verifier


def exchange_user_auth_code(
    auth_code: str,
    code_verifier: str,
    project: str,
) -> dict[str, object]:
    """Tek kullanımlık kodu oturuma özel Earth Engine kimliğine dönüştürür."""
    request_id, pkce_verifier, client_verifier = code_verifier.split(":")
    response = requests.post(
        ee.oauth.FETCH_URL,
        json={"request_id": request_id, "client_verifier": client_verifier},
        timeout=60,
    )
    response.raise_for_status()
    client_info = response.json()
    if "error" in client_info:
        raise RuntimeError(f"Google yetkilendirme hatası: {client_info['error']}")
    refresh_token = ee.oauth.request_token(
        auth_code.strip(),
        pkce_verifier,
        client_id=client_info["client_id"],
        client_secret=client_info["client_secret"],
    )
    auth_data = {
        "project": project,
        "refresh_token": refresh_token,
        "client_id": client_info["client_id"],
        "client_secret": client_info["client_secret"],
        "scopes": client_info.get("scopes") or ee.oauth.SCOPES,
    }
    credentials = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=ee.oauth.TOKEN_URI,
        client_id=str(auth_data["client_id"]),
        client_secret=str(auth_data["client_secret"]),
        scopes=list(auth_data["scopes"]),
    )
    ee.Initialize(credentials=credentials, project=project)
    ee.Number(1).getInfo()
    return auth_data


def initialize_gee(project: str | None = None) -> tuple[bool, str]:
    selected_project = project_id(project)
    if not selected_project:
        return False, "Google Earth Engine Project ID girilmedi."
    try:
        user_auth = None
        try:
            import streamlit as st

            candidate = st.session_state.get("gee_user_auth")
            if candidate and candidate.get("project") == selected_project:
                user_auth = candidate
        except Exception:
            pass
        if user_auth:
            credentials = Credentials(
                token=None,
                refresh_token=user_auth["refresh_token"],
                token_uri=ee.oauth.TOKEN_URI,
                client_id=user_auth["client_id"],
                client_secret=user_auth["client_secret"],
                scopes=user_auth["scopes"],
            )
            ee.Initialize(credentials=credentials, project=selected_project)
            return True, f"Bağlı · {selected_project} · kişisel Google oturumu"
        service_account = os.getenv("GEE_SERVICE_ACCOUNT", "").strip()
        private_key = os.getenv("GEE_PRIVATE_KEY", "").replace("\\n", "\n").strip()
        if not service_account or not private_key:
            try:
                import streamlit as st

                gee_secrets = st.secrets.get("gee", {})
                service_account = str(
                    gee_secrets.get("service_account", service_account)
                ).strip()
                private_key = str(
                    gee_secrets.get("private_key", private_key)
                ).replace("\\n", "\n").strip()
            except Exception:
                pass
        if service_account and private_key:
            credentials = ee.ServiceAccountCredentials(
                service_account,
                key_data=private_key,
            )
            ee.Initialize(credentials=credentials, project=selected_project)
            return True, f"Bağlı · {selected_project} · bulut hizmet hesabı"
        ee.Initialize(project=selected_project)
        return True, f"Bağlı · {selected_project}"
    except Exception as exc:
        return False, str(exc)


def authenticate_localhost(project: str) -> None:
    ee.Authenticate(auth_mode="localhost", force=True)
    ee.Initialize(project=project)


def _ee_geometry(gdf: gpd.GeoDataFrame) -> ee.Geometry:
    dissolved = gdf.to_crs(4326)[["geometry"]].dissolve().geometry.iloc[0]
    return ee.Geometry(dissolved.__geo_interface__)


def _centroid_latlon(gdf: gpd.GeoDataFrame) -> tuple[float, float]:
    wgs84 = gdf.to_crs(4326)
    metric_crs = wgs84.estimate_utm_crs() or "EPSG:6933"
    centroid_metric = wgs84[["geometry"]].dissolve().to_crs(metric_crs).geometry.iloc[0].centroid
    centroid = gpd.GeoSeries([centroid_metric], crs=metric_crs).to_crs(4326).iloc[0]
    return float(centroid.y), float(centroid.x)


def _adaptive_scale(gdf: gpd.GeoDataFrame, native_scale: int, max_pixels: int = 4_000_000) -> int:
    wgs84 = gdf.to_crs(4326)
    metric_crs = wgs84.estimate_utm_crs() or "EPSG:6933"
    area_m2 = float(wgs84.to_crs(metric_crs).geometry.union_all().area)
    required = int(np.ceil(np.sqrt(max(area_m2, 1) / max_pixels)))
    scale = max(native_scale, required)
    return int(np.ceil(scale / 10) * 10) if scale > 10 else 10


def _mask_sentinel2(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    clear = (
        scl.neq(1)
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return (
        image.updateMask(clear)
        .select(["B2", "B3", "B4", "B8", "B11"])
        .divide(10000)
        .copyProperties(image, ["system:time_start"])
    )


def _mask_landsat_l2(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    clear = (
        qa.bitwiseAnd(1 << 0).eq(0)
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
        .And(image.select("QA_RADSAT").eq(0))
    )
    return image.updateMask(clear).copyProperties(image, ["system:time_start"])


def _download_analysis_image(
    image: ee.Image,
    gdf: gpd.GeoDataFrame,
    scale: int,
    description: str,
    tags: dict[str, str],
) -> bytes:
    geometry = _ee_geometry(gdf)
    url = image.clip(geometry).getDownloadURL(
        {
            "region": geometry,
            "scale": scale,
            "format": "GEO_TIFF",
            "filePerBand": False,
            "crs": "EPSG:4326",
        }
    )
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    content = response.content
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            tif_name = next(name for name in archive.namelist() if name.lower().endswith(".tif"))
            content = archive.read(tif_name)

    with MemoryFile(content) as memory:
        with memory.open() as src:
            raster_geometry = [
                geometry.__geo_interface__
                for geometry in gdf.to_crs(src.crs).geometry
            ]
            nodata = -9999.0
            data, transform = raster_mask(
                src, raster_geometry, crop=True, filled=True, nodata=nodata
            )
            profile = src.profile.copy()
            profile.update(
                count=1,
                height=data.shape[1],
                width=data.shape[2],
                transform=transform,
                dtype="float32",
                nodata=nodata,
                compress="deflate",
            )
            with MemoryFile() as output:
                with output.open(**profile) as dst:
                    dst.write(data.astype("float32"))
                    dst.set_band_description(1, description)
                    dst.update_tags(**tags)
                return output.read()


def build_remote_analysis_geotiff(
    gdf: gpd.GeoDataFrame,
    start_date: date,
    end_date: date,
    analysis: str,
    project: str | None = None,
    cloud_limit: int = 30,
) -> tuple[bytes, dict[str, object]]:
    """Seçilen akademik CBS analizini gerçek GEE ürünlerinden üretir."""
    analysis = analysis.upper()
    if analysis not in REMOTE_ANALYSIS_SPECS:
        raise ValueError(f"Desteklenmeyen analiz: {analysis}")
    ok, message = initialize_gee(project)
    if not ok:
        raise RuntimeError("Earth Engine doğrulanmadı: " + message)

    spec = REMOTE_ANALYSIS_SPECS[analysis]
    geometry = _ee_geometry(gdf)
    start = start_date.isoformat()
    end = (end_date + relativedelta(days=1)).isoformat()
    scene_count = None

    if analysis in {"NDVI", "NDWI", "NDMI", "NDBI", "EVI", "SAVI"}:
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(start, end)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_limit))
            .map(_mask_sentinel2)
        )
        scene_count = int(collection.size().getInfo())
        if scene_count == 0:
            raise ValueError("Seçilen tarih ve bulut eşiğinde kullanılabilir Sentinel-2 sahnesi yok.")
        composite = collection.median()
        available_bands = composite.bandNames().getInfo()
        required_bands = {"B2", "B3", "B4", "B8", "B11"}
        if not required_bands.issubset(set(available_bands)):
            raise ValueError(
                "Sentinel-2 kompoziti gerekli bantları içermiyor. "
                f"Bulunan bantlar: {available_bands}"
            )
        nir, red, green, blue, swir = (
            composite.select("B8"),
            composite.select("B4"),
            composite.select("B3"),
            composite.select("B2"),
            composite.select("B11"),
        )
        if analysis == "NDVI":
            image = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        elif analysis == "NDWI":
            image = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
        elif analysis == "NDMI":
            image = nir.subtract(swir).divide(nir.add(swir)).rename("NDMI")
        elif analysis == "NDBI":
            image = swir.subtract(nir).divide(swir.add(nir)).rename("NDBI")
        elif analysis == "EVI":
            image = (
                nir.subtract(red)
                .multiply(2.5)
                .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
                .rename("EVI")
            )
        else:
            image = (
                nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename("SAVI")
            )
    elif analysis == "LST":
        landsat = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterBounds(geometry)
            .filterDate(start, end)
            .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
            .filter(ee.Filter.lte("CLOUD_COVER", cloud_limit))
            .map(_mask_landsat_l2)
        )
        scene_count = int(landsat.size().getInfo())
        if scene_count == 0:
            raise ValueError("Seçilen tarih ve bulut eşiğinde kullanılabilir Landsat LST sahnesi yok.")
        image = (
            landsat.select("ST_B10")
            .median()
            .multiply(0.00341802)
            .add(149.0)
            .subtract(273.15)
            .rename("LST_C")
        )
    else:
        dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
        if analysis == "DEM":
            image = dem.rename("DEM_m")
        elif analysis == "SLOPE":
            image = ee.Terrain.slope(dem).rename("SLOPE_deg")
        elif analysis == "ASPECT":
            image = ee.Terrain.aspect(dem).rename("ASPECT_deg")
        else:
            slope_radians = ee.Terrain.slope(dem).multiply(np.pi / 180).max(0.001)
            upstream_area_m2 = (
                ee.Image("MERIT/Hydro/v1_0_1").select("upa").multiply(1_000_000)
            )
            specific_catchment_area = upstream_area_m2.divide(92.77).max(1)
            image = (
                specific_catchment_area.divide(slope_radians.tan().max(0.001))
                .log()
                .rename("TWI")
            )

    export_scale = _adaptive_scale(gdf, int(spec["native_scale_m"]))
    metadata = {
        "analysis": analysis,
        "source": spec["source"],
        "formula": spec["formula"],
        "unit": spec["unit"],
        "native_scale_m": spec["native_scale_m"],
        "export_scale_m": export_scale,
        "scene_count": scene_count,
        "cloud_limit_percent": cloud_limit if scene_count is not None else None,
        "period": f"{start_date.isoformat()}/{end_date.isoformat()}",
        "composite": "median" if scene_count is not None else "static product",
    }
    geotiff = _download_analysis_image(
        image,
        gdf,
        export_scale,
        f"{analysis} analysis",
        {key: str(value) for key, value in metadata.items() if value is not None},
    )
    return geotiff, metadata


def _month_starts(start_date: date, end_date: date) -> list[date]:
    current = start_date.replace(day=1)
    final = end_date.replace(day=1)
    dates = []
    while current <= final:
        dates.append(current)
        current += relativedelta(months=1)
    return dates


def fetch_chirps_monthly_mean(
    gdf: gpd.GeoDataFrame,
    start_date: date,
    end_date: date,
    project: str | None = None,
) -> pd.DataFrame:
    ok, message = initialize_gee(project)
    if not ok:
        raise RuntimeError("Earth Engine doğrulanmadı: " + message)
    geometry = _ee_geometry(gdf)
    centroid_lat, centroid_lon = _centroid_latlon(gdf)
    collection = ee.ImageCollection(CHIRPS_DAILY)
    features = []
    for month in _month_starts(start_date, end_date):
        following = month + relativedelta(months=1)
        image = collection.filterDate(month.isoformat(), following.isoformat()).sum()
        value = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=CHIRPS_SCALE_M,
            bestEffort=True,
            maxPixels=1e9,
        ).get("precipitation")
        features.append(ee.Feature(None, {"date": month.isoformat(), "precipitation": value}))
    info = ee.FeatureCollection(features).getInfo()
    rows = [
        {
            "Tarih": pd.Timestamp(item["properties"]["date"]),
            "Örnek ID": 1,
            "Enlem": centroid_lat,
            "Boylam": centroid_lon,
            "Toplam yağış (mm)": item["properties"].get("precipitation"),
        }
        for item in info["features"]
    ]
    return pd.DataFrame(rows)


def fetch_gee_monthly_climate(
    gdf: gpd.GeoDataFrame,
    start_date: date,
    end_date: date,
    variables: list[str],
    project: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """CHIRPS yağışını ve ERA5-Land sıcaklığını havza ortalaması olarak getirir."""
    ok, message = initialize_gee(project)
    if not ok:
        raise RuntimeError("Earth Engine doğrulanmadı: " + message)
    geometry = _ee_geometry(gdf)
    centroid_lat, centroid_lon = _centroid_latlon(gdf)
    chirps = ee.ImageCollection(CHIRPS_DAILY)
    era5_land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    supported = {"Yağış", "Hava sıcaklığı"}
    unsupported = [item for item in variables if item not in supported]
    features = []
    for month in _month_starts(start_date, end_date):
        following = month + relativedelta(months=1)
        bands = []
        if "Yağış" in variables:
            bands.append(
                chirps.filterDate(month.isoformat(), following.isoformat())
                .sum()
                .rename("precip_mm")
            )
        if "Hava sıcaklığı" in variables:
            bands.append(
                era5_land.filterDate(month.isoformat(), following.isoformat())
                .select("temperature_2m")
                .mean()
                .subtract(273.15)
                .rename("temp_c")
            )
        if not bands:
            raise ValueError("GEE bağlantısında en az Yağış veya Hava sıcaklığı seçin.")
        image = ee.Image(bands[0])
        for band in bands[1:]:
            image = image.addBands(band)
        values = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=CHIRPS_SCALE_M,
            bestEffort=True,
            maxPixels=1e9,
        )
        features.append(
            ee.Feature(
                None,
                values.combine(
                    ee.Dictionary(
                        {
                            "date": month.isoformat(),
                            "sample_id": 1,
                        }
                    )
                ),
            )
        )
    info = ee.FeatureCollection(features).getInfo()
    rows = []
    for item in info["features"]:
        properties = item["properties"]
        row = {
            "Tarih": pd.Timestamp(properties["date"]),
            "Örnek ID": properties["sample_id"],
            "Enlem": centroid_lat,
            "Boylam": centroid_lon,
        }
        if "precip_mm" in properties:
            row["Toplam yağış (mm)"] = properties["precip_mm"]
        if "temp_c" in properties:
            row["Ortalama sıcaklık (°C)"] = properties["temp_c"]
        rows.append(row)
    return pd.DataFrame(rows), unsupported


def _empty_named_image(names: list[str]) -> ee.Image:
    return ee.Image.constant([0.0] * len(names)).rename(names).updateMask(ee.Image.constant(0))


def fetch_gee_academic_series(
    gdf: gpd.GeoDataFrame,
    start_date: date,
    end_date: date,
    *,
    response_indices: list[str] | None = None,
    land_cover_codes: list[int] | None = None,
    project: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Çok kaynaklı aylık kuraklık ve uydu serilerini havza ortalaması olarak getirir.

    Yağış hem CHIRPS hem ERA5-Land'den alınır. SPEI için ERA5-Land potansiyel
    buharlaşması, ekosistem tepkisi için Sentinel-2 NDVI/EVI ve Landsat LST
    kullanılır. Seçilen arazi örtüsü sınıfları ESA WorldCover ile zonlanır.
    """
    ok, message = initialize_gee(project)
    if not ok:
        raise RuntimeError("Earth Engine doğrulanmadı: " + message)

    requested = {
        item.upper() for item in (response_indices or ["NDVI", "EVI", "LST"])
        if item.upper() in {"NDVI", "EVI", "LST"}
    }
    selected_land_cover = [
        int(code) for code in (land_cover_codes or []) if int(code) in LAND_COVER_CLASSES
    ]
    geometry = _ee_geometry(gdf)
    centroid_lat, centroid_lon = _centroid_latlon(gdf)
    chirps = ee.ImageCollection(CHIRPS_DAILY)
    era5_land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    worldcover = ee.Image(ee.ImageCollection("ESA/WorldCover/v200").first()).select("Map")
    features = []

    for month in _month_starts(start_date, end_date):
        following = month + relativedelta(months=1)
        start = month.isoformat()
        end = following.isoformat()
        chirps_month = (
            chirps.filterDate(start, end).sum().rename("CHIRPS yağış (mm)")
        )
        era_month = era5_land.filterDate(start, end)
        era_precip = (
            era_month.select("total_precipitation_sum")
            .sum()
            .multiply(1000)
            .max(0)
            .rename("ERA5-Land yağış (mm)")
        )
        temperature = (
            era_month.select("temperature_2m")
            .mean()
            .subtract(273.15)
            .rename("ERA5-Land sıcaklık (°C)")
        )
        # ECMWF akı işaret konvansiyonunda buharlaşma yukarı yönlü olduğu için negatiftir.
        pet = (
            era_month.select("potential_evaporation_sum")
            .sum()
            .multiply(-1000)
            .max(0)
            .rename("ERA5-Land PET (mm)")
        )
        climate = ee.Image.cat(chirps_month, era_precip, temperature, pet)
        properties = climate.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=CHIRPS_SCALE_M,
            bestEffort=True,
            maxPixels=1e9,
        )

        if requested.intersection({"NDVI", "EVI"}) and month >= date(2015, 6, 23):
            sentinel = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(geometry)
                .filterDate(start, end)
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60))
                .map(_mask_sentinel2)
            )
            sentinel_count = sentinel.size()
            sentinel_composite = ee.Image(
                ee.Algorithms.If(
                    sentinel_count.gt(0),
                    sentinel.median(),
                    _empty_named_image(["B2", "B3", "B4", "B8", "B11"]),
                )
            )
            sentinel_images = []
            if "NDVI" in requested:
                sentinel_images.append(
                    sentinel_composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
                )
            if "EVI" in requested:
                sentinel_images.append(
                    sentinel_composite.expression(
                        "2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)",
                        {
                            "nir": sentinel_composite.select("B8"),
                            "red": sentinel_composite.select("B4"),
                            "blue": sentinel_composite.select("B2"),
                        },
                    ).rename("EVI")
                )
            sentinel_indices = ee.Image.cat(*sentinel_images)
            properties = properties.combine(
                sentinel_indices.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=30,
                    bestEffort=True,
                    maxPixels=1e9,
                ),
                overwrite=True,
            )
            first_index = sorted(requested.intersection({"NDVI", "EVI"}))[0]
            valid_fraction = (
                sentinel_indices.select(first_index).mask().unmask(0).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=30,
                    bestEffort=True,
                    maxPixels=1e9,
                ).get(first_index)
            )
            properties = properties.set("Sentinel-2 sahne sayısı", sentinel_count).set(
                "Sentinel-2 geçerli piksel oranı", valid_fraction
            )
            for code in selected_land_cover:
                label = LAND_COVER_CLASSES[code]
                for index_name in sorted(requested.intersection({"NDVI", "EVI"})):
                    zonal_value = (
                        sentinel_indices.select(index_name)
                        .updateMask(worldcover.eq(code))
                        .reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=geometry,
                            scale=100,
                            bestEffort=True,
                            maxPixels=1e9,
                        )
                        .get(index_name)
                    )
                    properties = properties.set(f"{index_name}|{label}", zonal_value)

        if "LST" in requested and month >= date(2013, 3, 18):
            landsat = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
                .filterBounds(geometry)
                .filterDate(start, end)
                .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
                .filter(ee.Filter.lte("CLOUD_COVER", 60))
                .map(_mask_landsat_l2)
            )
            landsat_count = landsat.size()
            lst = ee.Image(
                ee.Algorithms.If(
                    landsat_count.gt(0),
                    landsat.select("ST_B10").median().multiply(0.00341802).add(149).subtract(273.15).rename("LST"),
                    _empty_named_image(["LST"]),
                )
            )
            properties = properties.combine(
                lst.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=60,
                    bestEffort=True,
                    maxPixels=1e9,
                ),
                overwrite=True,
            )
            lst_valid_fraction = lst.mask().unmask(0).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=60,
                bestEffort=True,
                maxPixels=1e9,
            ).get("LST")
            properties = properties.set("Landsat sahne sayısı", landsat_count).set(
                "Landsat geçerli piksel oranı", lst_valid_fraction
            )
            for code in selected_land_cover:
                label = LAND_COVER_CLASSES[code]
                zonal_value = (
                    lst.updateMask(worldcover.eq(code))
                    .reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=100,
                        bestEffort=True,
                        maxPixels=1e9,
                    )
                    .get("LST")
                )
                properties = properties.set(f"LST|{label}", zonal_value)

        properties = (
            properties.set("date", month.isoformat())
            .set("sample_id", 1)
            .set("latitude", centroid_lat)
            .set("longitude", centroid_lon)
        )
        features.append(ee.Feature(None, properties))

    info = ee.FeatureCollection(features).getInfo()
    rows = []
    for feature in info.get("features", []):
        properties = feature.get("properties", {})
        row = {
            "Tarih": pd.Timestamp(properties.pop("date")),
            "Örnek ID": properties.pop("sample_id", 1),
            "Enlem": properties.pop("latitude", centroid_lat),
            "Boylam": properties.pop("longitude", centroid_lon),
        }
        row.update(properties)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("Tarih").reset_index(drop=True)
    metadata = {
        "datasets": {
            "CHIRPS": {
                "id": CHIRPS_DAILY,
                "catalog_version": "CHIRPS v2.0",
                "resolution": "0.05 degree / approximately 5.5 km",
                "variable": "precipitation (mm/day)",
            },
            "ERA5-Land": {
                "id": "ECMWF/ERA5_LAND/DAILY_AGGR",
                "resolution": "0.1 degree / approximately 11.1 km",
                "variables": ["total_precipitation_sum", "temperature_2m", "potential_evaporation_sum"],
            },
            "Sentinel-2": {
                "id": "COPERNICUS/S2_SR_HARMONIZED",
                "processing": "SCL cloud/shadow mask + monthly median",
            } if requested.intersection({"NDVI", "EVI"}) else None,
            "Landsat": {
                "id": "LANDSAT/LC08+C09/C02/T1_L2",
                "processing": "QA_PIXEL/QA_RADSAT mask + ST_B10 scale/offset",
            } if "LST" in requested else None,
            "Arazi örtüsü": {
                "id": "ESA/WorldCover/v200/2021",
                "reference_year": 2021,
            } if selected_land_cover else None,
        },
        "spatial_reducer": "Havza/çalışma alanı ortalaması",
        "temporal_aggregation": "Aylık; yağış ve PET toplam, sıcaklık ve uydu indisleri ortalama/medyan kompozit",
        "response_indices": sorted(requested),
        "land_cover_classes": [LAND_COVER_CLASSES[code] for code in selected_land_cover],
        "pet_sign_conversion": "ERA5-Land potential_evaporation_sum × -1000 (m→mm; ECMWF akı işaret konvansiyonu)",
    }
    return frame, metadata


def _rolling_month_image(collection: ee.ImageCollection, month: date, scale_months: int) -> ee.Image:
    end = month + relativedelta(months=1)
    start = end - relativedelta(months=scale_months)
    return (
        collection.filterDate(start.isoformat(), end.isoformat())
        .sum()
        .rename("precipitation")
        .set({"year": month.year, "month": month.month, "label": month.strftime("%Y_%m")})
    )


def build_chirps_spi_geotiff(
    gdf: gpd.GeoDataFrame,
    target_date: date,
    spi_scale: int,
    baseline_start: int = 1981,
    baseline_end: int = 2024,
    project: str | None = None,
) -> bytes:
    ok, message = initialize_gee(project)
    if not ok:
        raise RuntimeError("Earth Engine doğrulanmadı: " + message)

    geometry = _ee_geometry(gdf)
    collection = ee.ImageCollection(CHIRPS_DAILY)
    target_month = target_date.replace(day=1)
    baseline_images = []
    for year in range(baseline_start, baseline_end + 1):
        month = date(year, target_month.month, 1)
        baseline_images.append(_rolling_month_image(collection, month, spi_scale))
    target_is_baseline = baseline_start <= target_month.year <= baseline_end
    images = baseline_images if target_is_baseline else baseline_images + [
        _rolling_month_image(collection, target_month, spi_scale)
    ]
    stack_image = ee.ImageCollection.fromImages(images).toBands().clip(geometry)
    url = stack_image.getDownloadURL(
        {
            "region": geometry,
            "scale": CHIRPS_SCALE_M,
            "format": "GEO_TIFF",
            "filePerBand": False,
            "crs": "EPSG:4326",
        }
    )
    response = requests.get(url, timeout=300)
    response.raise_for_status()

    content = response.content
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            tif_name = next(name for name in archive.namelist() if name.lower().endswith(".tif"))
            content = archive.read(tif_name)

    with MemoryFile(content) as memory:
        with memory.open() as src:
            rainfall_stack = src.read().astype(np.float64)
            profile = src.profile.copy()
            target_index = target_month.year - baseline_start if target_is_baseline else -1
            spi = calculate_spi_pixel_stack(rainfall_stack, target_index=target_index)
            raster_geometry = [
                geometry.__geo_interface__
                for geometry in gdf.to_crs(src.crs).geometry
            ]
            inside = geometry_mask(
                raster_geometry,
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True,
            )
            spi[~inside] = np.nan
            profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")
            output = io.BytesIO()
            with MemoryFile() as out_memory:
                with out_memory.open(**profile) as dst:
                    dst.write(np.where(np.isfinite(spi), spi, -9999.0).astype("float32"), 1)
                    dst.set_band_description(1, f"SPI-{spi_scale} {target_month:%Y-%m}")
                    dst.update_tags(
                        source="CHIRPS Daily via Google Earth Engine",
                        method="Gamma distribution with zero-probability correction",
                        baseline=f"{baseline_start}-{baseline_end}",
                    )
                output.write(out_memory.read())
            return output.getvalue()


def build_climate_geotiff(
    gdf: gpd.GeoDataFrame,
    start_date: date,
    end_date: date,
    variable: str,
    project: str | None = None,
) -> bytes:
    """Seçilen dönem için gerçek CHIRPS yağış veya ERA5-Land sıcaklık rasteri."""
    ok, message = initialize_gee(project)
    if not ok:
        raise RuntimeError("Earth Engine doğrulanmadı: " + message)
    geometry = _ee_geometry(gdf)
    end_exclusive = end_date + relativedelta(days=1)

    if variable == "precipitation":
        image = (
            ee.ImageCollection(CHIRPS_DAILY)
            .filterDate(start_date.isoformat(), end_exclusive.isoformat())
            .sum()
            .rename("precip_mm")
            .clip(geometry)
        )
        scale = CHIRPS_SCALE_M
        description = "CHIRPS period total precipitation (mm)"
    elif variable == "temperature":
        image = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(start_date.isoformat(), end_exclusive.isoformat())
            .select("temperature_2m")
            .mean()
            .subtract(273.15)
            .rename("temp_c")
            .clip(geometry)
        )
        scale = 11132
        description = "ERA5-Land period mean 2m air temperature (C)"
    else:
        raise ValueError(f"Desteklenmeyen iklim raster değişkeni: {variable}")

    url = image.getDownloadURL(
        {
            "region": geometry,
            "scale": scale,
            "format": "GEO_TIFF",
            "filePerBand": False,
            "crs": "EPSG:4326",
        }
    )
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    content = response.content
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            tif_name = next(name for name in archive.namelist() if name.lower().endswith(".tif"))
            content = archive.read(tif_name)

    with MemoryFile(content) as memory:
        with memory.open() as src:
            profile = src.profile.copy()
            raster_geometry = [
                geometry.__geo_interface__
                for geometry in gdf.to_crs(src.crs).geometry
            ]
            nodata = -9999.0
            data, transform = raster_mask(
                src,
                raster_geometry,
                crop=True,
                filled=True,
                nodata=nodata,
            )
            profile.update(
                count=1,
                height=data.shape[1],
                width=data.shape[2],
                transform=transform,
                dtype="float32",
                nodata=nodata,
                compress="deflate",
            )
            with MemoryFile() as output:
                with output.open(**profile) as dst:
                    dst.write(data.astype("float32"))
                    dst.set_band_description(1, description)
                    dst.update_tags(
                        source=(
                            "CHIRPS Daily via Google Earth Engine"
                            if variable == "precipitation"
                            else "ERA5-Land Daily Aggregated via Google Earth Engine"
                        ),
                        period=f"{start_date.isoformat()}/{end_date.isoformat()}",
                        statistic="sum" if variable == "precipitation" else "mean",
                    )
                return output.read()

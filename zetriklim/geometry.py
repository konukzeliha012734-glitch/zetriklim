"""Coğrafi dosya okuma, doğrulama ve alan özeti."""

from __future__ import annotations

import io
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import shapefile
from shapely.geometry import shape as shapely_shape


REQUIRED_SHAPE_PARTS = {".shp", ".shx", ".dbf"}
SUPPORTED_SUFFIXES = {".zip", ".shp", ".shx", ".dbf", ".prj", ".cpg", ".gpkg", ".geojson", ".json"}


class GeometryUploadError(ValueError):
    """Yüklenen mekânsal veri doğrulanamadığında üretilir."""


@dataclass
class UploadedPart:
    name: str
    content: bytes


@dataclass
class GeometrySummary:
    gdf_wgs84: gpd.GeoDataFrame
    area_km2: float
    perimeter_km: float
    source_crs: str
    area_crs: str
    feature_count: int
    vertex_count: int
    was_repaired: bool

    @property
    def bounds(self) -> list[float]:
        return self.gdf_wgs84.total_bounds.tolist()

    @property
    def centroid(self) -> tuple[float, float]:
        geom = self.gdf_wgs84[["geometry"]].dissolve().to_crs(self.area_crs).geometry.iloc[0]
        point = gpd.GeoSeries([geom.centroid], crs=self.area_crs).to_crs(4326).iloc[0]
        return point.y, point.x


def _safe_extract_zip(content: bytes, destination: Path) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise GeometryUploadError("Yüklenen ZIP arşivi geçerli değil.") from exc
    for item in archive.infolist():
        if item.is_dir():
            continue
        target = (destination / item.filename).resolve()
        if destination.resolve() not in target.parents:
            raise GeometryUploadError("ZIP içinde güvenli olmayan bir dosya yolu var.")
        archive.extract(item, destination)


def _find_dataset(folder: Path) -> Path:
    gpkg = list(folder.rglob("*.gpkg"))
    geojson = list(folder.rglob("*.geojson")) + list(folder.rglob("*.json"))
    shp = list(folder.rglob("*.shp"))
    candidates = gpkg + geojson + shp
    if len(candidates) != 1:
        raise GeometryUploadError(
            "Tek bir çalışma alanı yükleyin. Desteklenen ana dosyalar: "
            ".gpkg, .geojson veya .shp."
        )
    dataset = candidates[0]
    if dataset.suffix.lower() == ".shp":
        existing = {p.suffix.lower() for p in dataset.parent.glob(f"{dataset.stem}.*")}
        missing = REQUIRED_SHAPE_PARTS - existing
        if missing:
            raise GeometryUploadError(
                "Shapefile bileşenleri eksik: " + ", ".join(sorted(missing))
            )
    return dataset


def _count_vertices(geometry) -> int:
    if geometry is None or geometry.is_empty:
        return 0
    if geometry.geom_type == "Polygon":
        return len(geometry.exterior.coords) + sum(len(r.coords) for r in geometry.interiors)
    if geometry.geom_type == "MultiPolygon":
        return sum(_count_vertices(part) for part in geometry.geoms)
    return 0


def _read_single_shp(content: bytes, fallback_crs: str) -> gpd.GeoDataFrame:
    try:
        reader = shapefile.Reader(shp=io.BytesIO(content))
        geometries = [
            shapely_shape(item.__geo_interface__)
            for item in reader.shapes()
            if item.shapeType != shapefile.NULL
        ]
    except Exception as exc:
        raise GeometryUploadError(f"Tek SHP geometrisi okunamadı: {exc}") from exc
    if not geometries:
        raise GeometryUploadError("SHP dosyasında poligon geometrisi bulunamadı.")
    return gpd.GeoDataFrame(
        {"kaynak": ["tek_shp"] * len(geometries)},
        geometry=geometries,
        crs=fallback_crs,
    )


def inspect_uploaded_files(
    files: Iterable[UploadedPart],
    fallback_crs: str = "EPSG:4326",
) -> GeometrySummary:
    parts = list(files)
    if not parts:
        raise GeometryUploadError("En az bir coğrafi dosya seçin.")

    with tempfile.TemporaryDirectory(prefix="zetriklim_") as temp:
        folder = Path(temp)
        for part in parts:
            suffix = Path(part.name).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue
            if suffix == ".zip":
                _safe_extract_zip(part.content, folder)
            else:
                safe_name = Path(part.name).name
                (folder / safe_name).write_bytes(part.content)

        shp_parts = [part for part in parts if Path(part.name).suffix.lower() == ".shp"]
        non_zip_parts = [
            part for part in parts if Path(part.name).suffix.lower() not in {".zip"}
        ]
        if len(shp_parts) == 1 and len(non_zip_parts) == 1:
            gdf = _read_single_shp(shp_parts[0].content, fallback_crs)
        else:
            dataset = _find_dataset(folder)
            try:
                gdf = gpd.read_file(dataset)
            except Exception as exc:
                raise GeometryUploadError(f"Coğrafi dosya okunamadı: {exc}") from exc

        return inspect_geodataframe(gdf)


def inspect_geodataframe(gdf: gpd.GeoDataFrame) -> GeometrySummary:
    """Dosyadan veya güvenilir bir katalogdan gelen poligonları aynı kurallarla denetler."""
    if gdf.empty or gdf.geometry.is_empty.all():
        raise GeometryUploadError("Veri geçerli bir geometri içermiyor.")
    if gdf.crs is None:
        raise GeometryUploadError("Koordinat sistemi tanımlı değil.")
    if not set(gdf.geom_type.dropna()).issubset({"Polygon", "MultiPolygon"}):
        raise GeometryUploadError("Çalışma alanı Polygon veya MultiPolygon olmalı.")

    gdf = gdf.copy()
    repaired = bool((~gdf.geometry.is_valid).any())
    if repaired:
        gdf.geometry = gdf.geometry.make_valid()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    source_crs = gdf.crs.to_string()
    gdf_wgs84 = gdf.to_crs(4326)
    area_crs_obj = gdf_wgs84.estimate_utm_crs() or "EPSG:6933"
    dissolved = gdf_wgs84[["geometry"]].dissolve().to_crs(area_crs_obj)
    geometry = dissolved.geometry.iloc[0]

    return GeometrySummary(
        gdf_wgs84=gdf_wgs84,
        area_km2=float(geometry.area / 1_000_000),
        perimeter_km=float(geometry.length / 1_000),
        source_crs=source_crs,
        area_crs=str(area_crs_obj),
        feature_count=len(gdf_wgs84),
        vertex_count=sum(_count_vertices(g) for g in gdf_wgs84.geometry),
        was_repaired=repaired,
    )


def inspect_zipped_shapefile(content: bytes) -> GeometrySummary:
    """Eski çağrılar için geriye uyumlu yardımcı."""
    return inspect_uploaded_files([UploadedPart("area.zip", content)])

"""Run GDAL vector operations outside the main Streamlit process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd


def _read_dataset(
    input_path: Path,
    output_path: Path,
    fallback_crs: str | None = None,
) -> None:
    gdf = gpd.read_file(input_path, engine="pyogrio")
    if gdf.empty or gdf.geometry.is_empty.all():
        raise ValueError("Dosya gecerli bir geometri icermiyor.")
    if gdf.crs is None:
        if input_path.suffix.lower() == ".shp" and fallback_crs:
            gdf = gdf.set_crs(fallback_crs, allow_override=True)
        else:
            raise ValueError("Koordinat sistemi tanimli degil.")
    source_crs = gdf.crs.to_string()
    payload = {
        "source_crs": source_crs,
        "geojson": json.loads(gdf.to_crs(4326).to_json(drop_id=True)),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_gpkg(
    input_path: Path,
    output_path: Path,
    latitude: float,
    longitude: float,
) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    area = gpd.GeoDataFrame.from_features(payload["features"], crs=4326)
    area.to_file(output_path, layer="calisma_alani", driver="GPKG", engine="pyogrio")
    point = gpd.GeoDataFrame(
        {
            "ornek_id": [1],
            "aciklama": ["İklim verisi örnekleme noktası"],
            "csv_baglanti_alani": ["Örnek ID"],
        },
        geometry=gpd.points_from_xy([longitude], [latitude]),
        crs=4326,
    )
    point.to_file(
        output_path,
        layer="ornekleme_noktasi",
        driver="GPKG",
        engine="pyogrio",
        mode="a",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    read_parser = subparsers.add_parser("read")
    read_parser.add_argument("input", type=Path)
    read_parser.add_argument("output", type=Path)
    read_parser.add_argument("--fallback-crs", default=None)

    write_parser = subparsers.add_parser("write-gpkg")
    write_parser.add_argument("input", type=Path)
    write_parser.add_argument("output", type=Path)
    write_parser.add_argument("latitude", type=float)
    write_parser.add_argument("longitude", type=float)

    args = parser.parse_args()
    if args.command == "read":
        _read_dataset(args.input, args.output, args.fallback_crs)
    else:
        _write_gpkg(args.input, args.output, args.latitude, args.longitude)


if __name__ == "__main__":
    main()

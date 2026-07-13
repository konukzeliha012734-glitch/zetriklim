"""Rasterio/GDAL raster operations isolated from the Streamlit process."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.mask import mask as raster_mask
from rasterio.plot import plotting_extent

from zetriklim.spi import calculate_spi_pixel_stack


def _load_boundary(path: Path | None) -> gpd.GeoDataFrame | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return gpd.GeoDataFrame.from_features(payload["features"], crs=4326)


def _nice_distance(target_m: float) -> float:
    if not np.isfinite(target_m) or target_m <= 0:
        return 1_000.0
    exponent = math.floor(math.log10(target_m))
    magnitude = 10.0**exponent
    normalized = target_m / magnitude
    factor = 5 if normalized >= 5 else 2 if normalized >= 2 else 1
    return factor * magnitude


def _draw_north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.93, 0.93),
        xytext=(0.93, 0.79),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#152f38",
        arrowprops={"facecolor": "#152f38", "edgecolor": "#152f38", "width": 3, "headwidth": 11},
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#71848a", "alpha": 0.92},
        zorder=10,
    )


def _draw_scale_bar(ax, crs, *, latitude: float = 0.0) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_span = abs(xmax - xmin)
    y_span = abs(ymax - ymin)
    is_geographic = bool(getattr(crs, "is_geographic", False))
    metres_per_x_unit = max(111_320.0 * math.cos(math.radians(latitude)), 1.0) if is_geographic else 1.0
    length_m = _nice_distance(x_span * metres_per_x_unit * 0.22)
    length_x = length_m / metres_per_x_unit
    x0 = xmin + x_span * 0.07
    y0 = ymin + y_span * 0.07
    cap = y_span * 0.012
    ax.plot([x0, x0 + length_x], [y0, y0], color="#152f38", linewidth=3.0, zorder=10)
    ax.plot([x0, x0], [y0 - cap, y0 + cap], color="#152f38", linewidth=1.6, zorder=10)
    ax.plot([x0 + length_x, x0 + length_x], [y0 - cap, y0 + cap], color="#152f38", linewidth=1.6, zorder=10)
    label = f"{length_m / 1000:g} km" if length_m >= 1_000 else f"{length_m:g} m"
    ax.text(
        x0 + length_x / 2,
        y0 + y_span * 0.018,
        label,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#152f38",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        zorder=10,
    )


def _configure_map_axes(ax, crs) -> None:
    is_geographic = bool(getattr(crs, "is_geographic", False))
    if is_geographic:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}°"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}°"))
        ax.set_xlabel("Boylam")
        ax.set_ylabel("Enlem")
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:,.0f}"))
        ax.set_xlabel("Doğu (km)")
        ax.set_ylabel("Kuzey (km)")
    ax.tick_params(labelsize=8, colors="#35545c")
    ax.grid(color="#8da1a5", alpha=0.28, linewidth=0.55, linestyle="--")


def _mask_geotiff(
    input_path: Path,
    boundary_path: Path,
    output_path: Path,
    description: str,
    tags: dict[str, str],
) -> None:
    boundary = _load_boundary(boundary_path)
    if boundary is None:
        raise ValueError("Raster kesme için çalışma alanı geometrisi bulunamadı.")
    with rasterio.open(input_path) as src:
        raster_geometry = [geometry.__geo_interface__ for geometry in boundary.to_crs(src.crs).geometry]
        nodata = -9999.0
        data, transform = raster_mask(src, raster_geometry, crop=True, filled=True, nodata=nodata)
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
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data.astype("float32"))
            dst.set_band_description(1, description)
            dst.update_tags(**tags)


def _spi_from_stack(
    input_path: Path,
    boundary_path: Path,
    output_path: Path,
    target_index: int,
    description: str,
    tags: dict[str, str],
) -> None:
    boundary = _load_boundary(boundary_path)
    if boundary is None:
        raise ValueError("SPI raster maskesi için çalışma alanı geometrisi bulunamadı.")
    with rasterio.open(input_path) as src:
        rainfall_stack = src.read().astype(np.float64)
        profile = src.profile.copy()
        spi = calculate_spi_pixel_stack(rainfall_stack, target_index=target_index)
        raster_geometry = [geometry.__geo_interface__ for geometry in boundary.to_crs(src.crs).geometry]
        inside = geometry_mask(
            raster_geometry,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
        )
        spi[~inside] = np.nan
        profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(np.where(np.isfinite(spi), spi, -9999.0).astype("float32"), 1)
            dst.set_band_description(1, description)
            dst.update_tags(**tags)


def _render_png(
    input_path: Path,
    output_path: Path,
    *,
    boundary_path: Path | None,
    title: str,
    palette: str,
    colorbar_label: str,
    fixed_range: tuple[float, float] | None,
    source_note: str | None,
) -> None:
    boundary = _load_boundary(boundary_path)
    with MemoryFile(input_path.read_bytes()) as memory:
        with memory.open() as src:
            raster = src.read(1, masked=True).astype(float).filled(float("nan"))
            extent = plotting_extent(src)
            raster_crs = src.crs
    fig = Figure(figsize=(10, 8), dpi=180, facecolor="#fbfaf6")
    ax = fig.add_subplot(111)
    if fixed_range is not None:
        range_args = {"vmin": fixed_range[0], "vmax": fixed_range[1]}
    else:
        finite = raster[np.isfinite(raster)]
        if finite.size:
            lower, upper = np.nanpercentile(finite, [2, 98])
            range_args = {"vmin": float(lower), "vmax": float(upper)} if upper > lower else {}
        else:
            range_args = {}
    image = ax.imshow(raster, cmap=palette, extent=extent, origin="upper", **range_args)
    if boundary is not None and raster_crs is not None:
        boundary.to_crs(raster_crs).boundary.plot(ax=ax, color="#082f49", linewidth=1.4, zorder=3)
    ax.set_title(title, color="#153b46", fontsize=14, weight="bold", pad=14)
    _configure_map_axes(ax, raster_crs)
    _draw_north_arrow(ax)
    latitude = float(boundary.to_crs(4326).geometry.union_all().centroid.y) if boundary is not None else 0.0
    _draw_scale_bar(ax, raster_crs, latitude=latitude)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.76, pad=0.025)
    colorbar.set_label(colorbar_label)
    fig.text(
        0.01,
        0.015,
        f"Kaynak: {source_note or 'Zetriklim analiz çıktısı'} · Koordinat sistemi: {raster_crs or 'tanımsız'}",
        fontsize=7.5,
        color="#526970",
    )
    fig.subplots_adjust(left=0.10, right=0.93, top=0.90, bottom=0.11)
    FigureCanvasAgg(fig).print_png(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    mask_parser = subparsers.add_parser("mask-geotiff")
    mask_parser.add_argument("input", type=Path)
    mask_parser.add_argument("boundary", type=Path)
    mask_parser.add_argument("output", type=Path)
    mask_parser.add_argument("--description", required=True)
    mask_parser.add_argument("--tags", default="{}")

    spi_parser = subparsers.add_parser("spi-from-stack")
    spi_parser.add_argument("input", type=Path)
    spi_parser.add_argument("boundary", type=Path)
    spi_parser.add_argument("output", type=Path)
    spi_parser.add_argument("--target-index", type=int, required=True)
    spi_parser.add_argument("--description", required=True)
    spi_parser.add_argument("--tags", default="{}")

    render_parser = subparsers.add_parser("render-png")
    render_parser.add_argument("input", type=Path)
    render_parser.add_argument("output", type=Path)
    render_parser.add_argument("--boundary", type=Path)
    render_parser.add_argument("--title", required=True)
    render_parser.add_argument("--palette", default="RdBu")
    render_parser.add_argument("--colorbar-label", default="SPI")
    render_parser.add_argument("--fixed-range", default="")
    render_parser.add_argument("--source-note", default="")

    args = parser.parse_args()
    if args.command == "mask-geotiff":
        _mask_geotiff(
            args.input,
            args.boundary,
            args.output,
            args.description,
            {str(key): str(value) for key, value in json.loads(args.tags).items()},
        )
    elif args.command == "spi-from-stack":
        _spi_from_stack(
            args.input,
            args.boundary,
            args.output,
            args.target_index,
            args.description,
            {str(key): str(value) for key, value in json.loads(args.tags).items()},
        )
    else:
        fixed_range = None
        if args.fixed_range:
            values = json.loads(args.fixed_range)
            fixed_range = (float(values[0]), float(values[1]))
        _render_png(
            args.input,
            args.output,
            boundary_path=args.boundary,
            title=args.title,
            palette=args.palette,
            colorbar_label=args.colorbar_label,
            fixed_range=fixed_range,
            source_note=args.source_note or None,
        )


if __name__ == "__main__":
    main()

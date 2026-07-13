"""Excel, CSV, grafik ve CBS çıktıları."""

from __future__ import annotations

import io
import math
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill


NAVY = "063447"
TEAL = "00A6A6"
PALE = "E2F4F2"
AMBER = "FFAD33"


def dataframe_to_csv(data: pd.DataFrame, *, excel_tr: bool = False) -> bytes:
    """Tarihleri kararlı biçimde ve Türkçe Excel seçeneğiyle CSV'ye dönüştürür."""
    return data.to_csv(
        index=False,
        sep=";" if excel_tr else ",",
        decimal="," if excel_tr else ".",
        date_format="%Y-%m-%d",
        na_rep="",
        lineterminator="\n",
    ).encode("utf-8-sig")


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
    if is_geographic:
        metres_per_x_unit = max(111_320.0 * math.cos(math.radians(latitude)), 1.0)
    else:
        metres_per_x_unit = 1.0
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


def build_excel(
    data: pd.DataFrame,
    *,
    metadata_rows: list[tuple[str, object]],
    area_rows: list[tuple[str, object]],
    analysis_tables: dict[str, pd.DataFrame] | None = None,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="İklim Verisi", index=False)
        pd.DataFrame(metadata_rows, columns=["Alan", "Değer"]).to_excel(writer, sheet_name="Kaynak ve Yöntem", index=False)
        pd.DataFrame(area_rows, columns=["Alan Özelliği", "Değer"]).to_excel(writer, sheet_name="Çalışma Alanı", index=False)
        data.describe(include="all").transpose().reset_index().to_excel(writer, sheet_name="Veri Özeti", index=False)
        for name, table in (analysis_tables or {}).items():
            table.to_excel(writer, sheet_name=name[:31], index=False)

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = PatternFill("solid", fgColor=NAVY)
                cell.font = Font(color="FFFFFF", bold=True)
                cell.alignment = Alignment(horizontal="center")
            for column in sheet.columns:
                letter = column[0].column_letter
                width = min(max(len(str(cell.value or "")) for cell in column) + 2, 42)
                sheet.column_dimensions[letter].width = max(width, 12)

        data_sheet = writer.book["İklim Verisi"]
        data_sheet.sheet_view.showGridLines = False
        for row in range(2, data_sheet.max_row + 1):
            if row % 2 == 0:
                for cell in data_sheet[row]:
                    cell.fill = PatternFill("solid", fgColor="F2FAF9")
        if data_sheet.max_column >= 2 and data_sheet.max_row >= 3:
            chart = LineChart()
            chart.title = "İklim Zaman Serisi"
            chart.style = 13
            chart.y_axis.title = "Değer"
            chart.x_axis.title = "Tarih"
            excluded = {"Yıl", "Ay", "Örnek ID", "Enlem", "Boylam"}
            numeric_columns = [
                index + 1
                for index, column in enumerate(data.columns)
                if column not in excluded and pd.api.types.is_numeric_dtype(data[column])
            ][:5]
            for column_index in numeric_columns:
                chart.add_data(
                    Reference(
                        data_sheet,
                        min_col=column_index,
                        max_col=column_index,
                        min_row=1,
                        max_row=data_sheet.max_row,
                    ),
                    titles_from_data=True,
                )
            chart.set_categories(Reference(data_sheet, min_col=1, min_row=2, max_row=data_sheet.max_row))
            chart.height = 9
            chart.width = 18
            summary_sheet = writer.book["Veri Özeti"]
            summary_sheet.add_chart(chart, "H2")
    return output.getvalue()


def build_raster_png(
    geotiff: bytes,
    title: str,
    *,
    boundary: gpd.GeoDataFrame | None = None,
    palette: str = "RdBu",
    colorbar_label: str = "SPI",
    fixed_range: tuple[float, float] | None = (-2.5, 2.5),
    source_note: str | None = None,
) -> bytes:
    with MemoryFile(geotiff) as memory:
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
        boundary.to_crs(raster_crs).boundary.plot(
            ax=ax,
            color="#082f49",
            linewidth=1.4,
            zorder=3,
        )
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
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return output.getvalue()


def build_raster_png(
    geotiff: bytes,
    title: str,
    *,
    boundary: gpd.GeoDataFrame | None = None,
    palette: str = "RdBu",
    colorbar_label: str = "SPI",
    fixed_range: tuple[float, float] | None = (-2.5, 2.5),
    source_note: str | None = None,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="zetriklim_raster_png_") as temp:
        folder = Path(temp)
        input_path = folder / "input.tif"
        output_path = folder / "rendered.png"
        boundary_path = folder / "boundary.json"
        input_path.write_bytes(geotiff)
        command = [
            sys.executable,
            "-m",
            "zetriklim.raster_worker",
            "render-png",
            str(input_path),
            str(output_path),
            "--title",
            title,
            "--palette",
            palette,
            "--colorbar-label",
            colorbar_label,
        ]
        if fixed_range is not None:
            command.extend(["--fixed-range", json.dumps(list(fixed_range))])
        if source_note:
            command.extend(["--source-note", source_note])
        if boundary is not None:
            boundary_path.write_text(
                json.dumps(json.loads(boundary.to_crs(4326).to_json(drop_id=True)), ensure_ascii=False),
                encoding="utf-8",
            )
            command.extend(["--boundary", str(boundary_path)])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0 or not output_path.exists():
            detail = (completed.stderr or completed.stdout or "bilinmeyen raster çizim hatası").strip()
            raise RuntimeError(f"Raster haritası güvenli işlemde üretilemedi: {detail}")
        return output_path.read_bytes()


def build_timeseries_png(data: pd.DataFrame) -> bytes:
    numeric = data.drop(
        columns=["Yıl", "Ay", "Örnek ID", "Enlem", "Boylam"],
        errors="ignore",
    ).select_dtypes("number")
    fig = Figure(figsize=(12, 6), dpi=150, facecolor="#f7f5ee")
    ax = fig.add_subplot(111)
    for column in numeric.columns[:5]:
        ax.plot(data["Tarih"], numeric[column], linewidth=1.1, label=column)
    ax.set_title("Zetriklim – Analiz Zaman Serisi", color="#063447", fontsize=14, weight="bold")
    ax.set_xlabel("Tarih")
    ax.grid(alpha=0.20)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return output.getvalue()


def build_area_map_png(
    gdf: gpd.GeoDataFrame,
    point: tuple[float, float],
    *,
    context: gpd.GeoDataFrame | None = None,
    title: str = "Çalışma Alanının Konumu",
    subtitle: str | None = None,
    context_label: str = "GADM idari sınırı",
    source_note: str | None = None,
) -> bytes:
    """Ölçek, kuzey oku, lejant ve koordinat ağı içeren tez standardı konum haritası."""
    latitude, longitude = point
    zone = min(max(int((longitude + 180) // 6) + 1, 1), 60)
    target_crs = f"EPSG:{32600 + zone if latitude >= 0 else 32700 + zone}"
    projected = gdf.to_crs(target_crs)

    fig = Figure(figsize=(10, 8), dpi=180, facecolor="#fbfaf6")
    ax = fig.add_subplot(111)
    xmin, ymin, xmax, ymax = projected.total_bounds
    x_padding = max((xmax - xmin) * 0.18, 2_000)
    y_padding = max((ymax - ymin) * 0.18, 2_000)
    ax.set_xlim(xmin - x_padding, xmax + x_padding)
    ax.set_ylim(ymin - y_padding, ymax + y_padding)

    context_plotted = False
    if context is not None and not context.empty:
        context_projected = context.to_crs(target_crs)
        nearby = context_projected.cx[
            xmin - x_padding : xmax + x_padding,
            ymin - y_padding : ymax + y_padding,
        ]
        if not nearby.empty:
            nearby.plot(
                ax=ax,
                facecolor="#edf0e8",
                edgecolor="#8c9997",
                linewidth=0.75,
                zorder=1,
            )
            context_plotted = True
    projected.plot(ax=ax, facecolor="#4cc7bd", edgecolor="#102f46", linewidth=2.1, alpha=0.82, zorder=3)
    if context_plotted:
        nearby.boundary.plot(ax=ax, color="#6f7e7c", linewidth=0.5, alpha=0.72, zorder=4)
    area_km2 = projected.geometry.union_all().area / 1_000_000
    ax.set_title(title, color="#153b46", fontsize=15, weight="bold", pad=18)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#526970")
    ax.text(
        0.02,
        0.94,
        f"Alan: {area_km2:,.2f} km²",
        transform=ax.transAxes,
        fontsize=9,
        color="#063447",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#00a6a6", "alpha": 0.9, "boxstyle": "round,pad=0.4"},
        zorder=10,
    )
    legend_items = [Patch(facecolor="#4cc7bd", edgecolor="#102f46", label="Çalışma alanı")]
    if context_plotted:
        legend_items.insert(0, Patch(facecolor="#edf0e8", edgecolor="#8c9997", label=context_label))
    legend_items.append(Line2D([0], [0], color="#102f46", linewidth=2.1, label="Çalışma alanı sınırı"))
    ax.legend(handles=legend_items, loc="lower right", frameon=True, framealpha=0.94, fontsize=8.5, title="Lejant")
    _configure_map_axes(ax, projected.crs)
    _draw_north_arrow(ax)
    _draw_scale_bar(ax, projected.crs, latitude=latitude)
    fig.text(
        0.01,
        0.015,
        f"Kaynak: {source_note or 'Kullanıcı çalışma alanı'} · Projeksiyon: {projected.crs.to_string()}",
        fontsize=7.5,
        color="#526970",
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.12)
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return output.getvalue()


def geodata_to_gpkg(gdf: gpd.GeoDataFrame, point: tuple[float, float]) -> bytes:
    with tempfile.TemporaryDirectory(prefix="zetriklim_gpkg_") as temp:
        folder = Path(temp)
        input_path = folder / "calisma-alani.json"
        output_path = folder / "zetriklim-cbs.gpkg"
        input_path.write_text(
            json.dumps(json.loads(gdf.to_crs(4326).to_json(drop_id=True)), ensure_ascii=False),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "zetriklim.geodata_worker",
                "write-gpkg",
                str(input_path),
                str(output_path),
                str(float(point[0])),
                str(float(point[1])),
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if completed.returncode != 0 or not output_path.exists():
            detail = (completed.stderr or completed.stdout or "bilinmeyen GDAL hatası").strip()
            raise RuntimeError(f"GeoPackage güvenli işlemde üretilemedi: {detail}")
        return output_path.read_bytes()


def build_complete_package(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()

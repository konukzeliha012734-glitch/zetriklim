from __future__ import annotations

import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from shapely.geometry import box

from zetriklim.academic import prepare_academic_series_export, run_academic_analysis
from zetriklim.artifacts import build_area_map_png, build_raster_png, dataframe_to_csv
from zetriklim.gadm import gadm_download_url


class CartographyAndExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.area = gpd.GeoDataFrame(
            {"ad": ["Örnek havza"]},
            geometry=[box(35.0, 38.0, 35.6, 38.5)],
            crs=4326,
        )
        cls.context = gpd.GeoDataFrame(
            {"NAME_1": ["Bağlam"]},
            geometry=[box(34.5, 37.5, 36.2, 39.0)],
            crs=4326,
        )

    def test_thesis_area_map_is_png(self) -> None:
        result = build_area_map_png(
            self.area,
            (38.25, 35.3),
            context=self.context,
            title="Tez Çalışma Alanı",
            source_note="GADM 4.1",
        )
        self.assertTrue(result.startswith(b"\x89PNG"))
        self.assertGreater(len(result), 50_000)

    def test_analysis_raster_map_is_png(self) -> None:
        values = np.linspace(-2.5, 2.5, 100, dtype="float32").reshape(10, 10)
        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_bounds(35.0, 38.0, 35.6, 38.5, 10, 10),
            ) as dataset:
                dataset.write(values, 1)
            geotiff = memory.read()
        result = build_raster_png(
            geotiff,
            "SPEI-3 Analiz Haritası",
            boundary=self.area,
            palette="RdBu",
            colorbar_label="SPEI-3",
        )
        self.assertTrue(result.startswith(b"\x89PNG"))
        self.assertGreater(len(result), 50_000)

    def test_csv_variants_and_combined_series(self) -> None:
        source = pd.DataFrame(
            {
                "Tarih": ["2024-01-01", "2024-01-15", "2024-02-01"],
                "NDVI": [0.4, 0.6, 0.7],
            }
        )
        drought = pd.DataFrame(
            {
                "Tarih": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "SPEI-3": [-1.2, -0.4],
            }
        )
        combined = prepare_academic_series_export(source, drought)
        self.assertEqual(len(combined), 2)
        self.assertIn("SPEI-3", combined)
        self.assertAlmostEqual(combined.loc[0, "NDVI"], 0.5)
        standard = dataframe_to_csv(combined).decode("utf-8-sig")
        excel_tr = dataframe_to_csv(combined, excel_tr=True).decode("utf-8-sig")
        self.assertIn("Tarih,Yıl,Ay", standard)
        self.assertIn("Tarih;Yıl;Ay", excel_tr)
        self.assertIn("0,5", excel_tr)

    def test_spei_only_and_response_only_are_supported(self) -> None:
        dates = pd.date_range("1981-01-01", "2024-12-01", freq="MS")
        frame = pd.DataFrame(
            {
                "Tarih": dates,
                "Yağış": 50 + 20 * np.sin(np.arange(len(dates)) / 6),
                "PET": 35 + 10 * np.cos(np.arange(len(dates)) / 6),
                "NDVI": 0.4 + np.arange(len(dates)) / 10_000,
            }
        )
        spei_only = run_academic_analysis(
            frame,
            precipitation_column="Yağış",
            pet_column="PET",
            response_columns=["NDVI"],
            validation_columns=[],
            config={
                "scales": [3],
                "drought_indices": ["SPEI"],
                "baseline_start": 1991,
                "baseline_end": 2020,
            },
        )
        self.assertIn("SPEI-3", spei_only["Kuraklık Serisi"])
        self.assertFalse(any(column.startswith("SPI-") for column in spei_only["Kuraklık Serisi"]))

        response_only = run_academic_analysis(
            frame[["Tarih", "NDVI"]],
            precipitation_column=None,
            pet_column=None,
            response_columns=["NDVI"],
            validation_columns=[],
            config={"scales": [3], "drought_indices": []},
        )
        self.assertFalse(response_only["Eğilim ve Değişim"].empty)
        self.assertIn("NDVI", response_only["Birleşik Analiz Serisi"])

    def test_gadm_url_validation(self) -> None:
        self.assertEqual(
            gadm_download_url("tur", 1),
            "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_TUR_1.json",
        )
        with self.assertRaises(ValueError):
            gadm_download_url("TR", 1)


if __name__ == "__main__":
    unittest.main()

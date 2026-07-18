from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from rasterio.io import MemoryFile
from shapely.geometry import box

from zetriklim.academic import academic_defaults, harmonize_monthly_data
from zetriklim.catalog import ANALYSIS_METHODS
from zetriklim.artifacts import (
    build_academic_chart_suite,
    build_area_map_png,
    build_map_vector_files,
    build_spi_thesis_excel,
    build_tile_map_png,
    build_timeseries_png,
    dataframe_to_csv,
)
from zetriklim.climate_engine import (
    ClimateEngineQuotaExceeded,
    fetch_dataset_date_range,
    fetch_map_tile,
    fetch_timeseries,
)
from zetriklim.climate_engine import normalize_analysis_column
from zetriklim.gadm import fetch_gadm, name_column
from zetriklim.geometry import inspect_geodataframe
from zetriklim.estimates import estimate_analysis_seconds, format_duration_range


class WorkflowIntegrityTests(unittest.TestCase):
    def test_every_catalogued_remote_method_keeps_its_own_focus(self) -> None:
        for method in ANALYSIS_METHODS:
            defaults = academic_defaults(method)
            self.assertEqual(defaults["focus"], method)
            if method != "SPI":
                self.assertEqual(defaults["response_indices"], [method])

    def test_duration_estimate_increases_with_area_and_map_workload(self) -> None:
        small = estimate_analysis_seconds(
            10, provider="Climate Engine", map_count=3, period_years=5
        )
        large = estimate_analysis_seconds(
            10_000, provider="Climate Engine", map_count=11, period_years=45,
            academic_mode=True,
        )
        self.assertGreater(large[0], small[0])
        self.assertGreater(large[1], small[1])
        self.assertIn("–", format_duration_range(*large))

    def test_duration_estimate_reflects_long_archive_maps_and_observed_runtime(self) -> None:
        archive_heavy = estimate_analysis_seconds(
            3165.81,
            provider="Climate Engine",
            map_count=7,
            period_years=45.5,
            academic_mode=True,
            spi_map_count=5,
            archive_map_count=2,
        )
        self.assertGreaterEqual(archive_heavy[0], 15 * 60)
        self.assertLess(archive_heavy[1], 32 * 60)
        calibrated = estimate_analysis_seconds(
            3165.81,
            provider="Climate Engine",
            map_count=7,
            period_years=45.5,
            spi_map_count=5,
            archive_map_count=2,
            observed_seconds=38 * 60,
        )
        self.assertLess(abs(sum(calibrated) / 2 - 38 * 60), 5 * 60)

    @patch("zetriklim.climate_engine.requests.post")
    def test_climate_engine_timeseries_uses_one_full_period_request_and_cache(
        self, request_post: Mock
    ) -> None:
        request_post.return_value.ok = True
        request_post.return_value.json.return_value = {
            "Data": [{"Data": [
                {"date": "1981-01-01", "precipitation": 1.25},
                {"date": "1985-12-31", "precipitation": 2.5},
            ]}]
        }
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.6, 39.9, 40.6, 40.6)], crs=4326
        )
        with tempfile.TemporaryDirectory() as cache_dir, patch(
            "zetriklim.climate_engine.CACHE_ROOT", Path(cache_dir)
        ):
            first, _ = fetch_timeseries(
                "secret", area, date(1981, 1, 1), date(1985, 12, 31),
                "CHIRPS_DAILY", "precipitation",
            )
            second, metadata = fetch_timeseries(
                "secret", area, date(1981, 1, 1), date(1985, 12, 31),
                "CHIRPS_DAILY", "precipitation",
            )
        self.assertEqual(request_post.call_count, 1)
        self.assertEqual(request_post.call_args.kwargs["json"]["start_date"], "1981-01-01")
        self.assertEqual(request_post.call_args.kwargs["json"]["end_date"], "1985-12-31")
        self.assertEqual(len(first), len(second))
        self.assertEqual(metadata["cache"], "persistent")

    @patch("zetriklim.climate_engine.requests.post")
    def test_climate_engine_daily_quota_has_specific_error(self, request_post: Mock) -> None:
        request_post.return_value.ok = False
        request_post.return_value.status_code = 404
        request_post.return_value.json.return_value = {
            "detail": "You have exceeded your daily request limit for the Climate Engine API."
        }
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.6, 39.9, 40.6, 40.6)], crs=4326
        )
        with tempfile.TemporaryDirectory() as cache_dir, patch(
            "zetriklim.climate_engine.CACHE_ROOT", Path(cache_dir)
        ):
            with self.assertRaises(ClimateEngineQuotaExceeded):
                fetch_timeseries(
                    "secret", area, date(1981, 1, 1), date(1985, 12, 31),
                    "CHIRPS_DAILY", "precipitation",
                )

    @patch("zetriklim.climate_engine.requests.post")
    def test_single_day_evi_uses_monthly_observation_window(
        self, request_post: Mock
    ) -> None:
        request_post.return_value.ok = True
        request_post.return_value.status_code = 200
        request_post.return_value.json.return_value = {
            "Data": [{"Data": [{"date": "2025-01-18", "EVI": 0.36}]}]
        }
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.6, 39.9, 40.6, 40.6)], crs=4326
        )
        with tempfile.TemporaryDirectory() as cache_dir, patch(
            "zetriklim.climate_engine.CACHE_ROOT", Path(cache_dir)
        ):
            data, metadata = fetch_timeseries(
                "secret", area, date(2025, 1, 1), date(2025, 1, 1),
                "SENTINEL2_SR", "EVI",
            )

        payload = request_post.call_args.kwargs["json"]
        self.assertEqual(payload["start_date"], "2025-01-01")
        self.assertEqual(payload["end_date"], "2025-01-31")
        self.assertEqual(len(data), 1)
        self.assertTrue(metadata["temporal_window_expanded"])
        self.assertEqual(
            metadata["effective_observation_period"], "2025-01-01/2025-01-31"
        )

    @patch("zetriklim.climate_engine.requests.post")
    def test_empty_satellite_year_does_not_cancel_other_real_observations(
        self, request_post: Mock
    ) -> None:
        no_data = Mock()
        no_data.ok = False
        no_data.status_code = 500
        no_data.json.return_value = {"detail": "ERROR: No data returned."}
        no_data.text = ""
        success = Mock()
        success.ok = True
        success.status_code = 200
        success.json.return_value = {
            "Data": [{"Data": [{"date": "2021-06-15", "EVI": 0.41}]}]
        }
        request_post.side_effect = [no_data, success]
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.6, 39.9, 40.6, 40.6)], crs=4326
        )
        with tempfile.TemporaryDirectory() as cache_dir, patch(
            "zetriklim.climate_engine.CACHE_ROOT", Path(cache_dir)
        ):
            data, metadata = fetch_timeseries(
                "secret", area, date(2020, 1, 1), date(2021, 12, 31),
                "SENTINEL2_SR", "EVI",
            )

        self.assertEqual(request_post.call_count, 2)
        self.assertEqual(len(data), 1)
        self.assertEqual(metadata["request_chunks"][0]["record_count"], 0)
        self.assertTrue(
            metadata["request_chunks"][0]["metadata"]["no_data"]
        )

    @patch("zetriklim.climate_engine.requests.post")
    def test_timeseries_splits_concurrent_aggregation_error(
        self, request_post: Mock
    ) -> None:
        overloaded = Mock()
        overloaded.ok = False
        overloaded.status_code = 500
        overloaded.json.return_value = {
            "detail": "ERROR: Too many concurrent aggregations."
        }
        overloaded.text = ""

        def successful(date_text: str, value: float) -> Mock:
            response = Mock()
            response.ok = True
            response.status_code = 200
            response.json.return_value = {
                "Data": [{"Data": [{"date": date_text, "NDVI": value}]}]
            }
            return response

        request_post.side_effect = [
            overloaded,
            successful("2020-01-01", 0.42),
            successful("2021-12-31", 0.47),
        ]
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.6, 39.9, 40.6, 40.6)], crs=4326
        )
        with tempfile.TemporaryDirectory() as cache_dir, patch(
            "zetriklim.climate_engine.CACHE_ROOT", Path(cache_dir)
        ):
            data, metadata = fetch_timeseries(
                "secret", area, date(2020, 1, 1), date(2021, 12, 31),
                "CUSTOM_EE_DATASET", "NDVI",
            )

        self.assertEqual(request_post.call_count, 3)
        self.assertEqual(len(data), 2)
        self.assertTrue(metadata["request_chunks"][0]["metadata"]["split"])
        self.assertEqual(
            metadata["request_chunks"][0]["metadata"]["reason"],
            "earth_engine_concurrent_aggregation_limit",
        )

    @patch("zetriklim.climate_engine.requests.post")
    def test_sentinel_quarter_is_split_below_ninety_days_when_needed(
        self, request_post: Mock
    ) -> None:
        overloaded = Mock()
        overloaded.ok = False
        overloaded.status_code = 500
        overloaded.json.return_value = {
            "detail": "ERROR: Unhandled EE Exception: Too many concurrent aggregations."
        }
        overloaded.text = ""

        def successful(date_text: str, value: float) -> Mock:
            response = Mock()
            response.ok = True
            response.status_code = 200
            response.json.return_value = {
                "Data": [{"Data": [{"date": date_text, "NDVI": value}]}]
            }
            return response

        request_post.side_effect = [
            overloaded,
            successful("2020-01-01", 0.41),
            successful("2020-02-16", 0.46),
        ]
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.6, 39.9, 40.6, 40.6)], crs=4326
        )
        with tempfile.TemporaryDirectory() as cache_dir, patch(
            "zetriklim.climate_engine.CACHE_ROOT", Path(cache_dir)
        ):
            data, metadata = fetch_timeseries(
                "secret", area, date(2020, 1, 1), date(2020, 3, 31),
                "SENTINEL2_SR", "NDVI",
            )

        self.assertEqual(request_post.call_count, 3)
        self.assertEqual(len(data), 2)
        self.assertTrue(metadata["request_chunks"][0]["metadata"]["split"])

    @patch("zetriklim.climate_engine.requests.post")
    def test_sentinel_year_is_split_after_earth_engine_timeout(
        self, request_post: Mock
    ) -> None:
        timed_out = Mock()
        timed_out.ok = False
        timed_out.status_code = 500
        timed_out.json.return_value = {
            "detail": "ERROR: EE Exception: Computation timed out."
        }
        timed_out.text = ""

        def successful(date_text: str, value: float) -> Mock:
            response = Mock()
            response.ok = True
            response.status_code = 200
            response.json.return_value = {
                "Data": [{"Data": [{"date": date_text, "NDVI": value}]}]
            }
            return response

        request_post.side_effect = [
            timed_out,
            successful("2018-01-01", 0.38),
            successful("2018-12-31", 0.44),
        ]
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.6, 39.9, 40.6, 40.6)], crs=4326
        )
        with tempfile.TemporaryDirectory() as cache_dir, patch(
            "zetriklim.climate_engine.CACHE_ROOT", Path(cache_dir)
        ):
            data, metadata = fetch_timeseries(
                "secret", area, date(2018, 1, 1), date(2018, 12, 31),
                "SENTINEL2_SR", "NDVI",
            )

        self.assertEqual(request_post.call_count, 3)
        self.assertEqual(len(data), 2)
        split_metadata = metadata["request_chunks"][0]["metadata"]
        self.assertTrue(split_metadata["split"])
        self.assertEqual(
            split_metadata["reason"], "earth_engine_computation_timeout"
        )

    @patch("zetriklim.climate_engine.requests.post")
    def test_long_sentinel_timeseries_is_requested_year_by_year(
        self, request_post: Mock
    ) -> None:
        def successful(*_args, **kwargs) -> Mock:
            requested_start = kwargs["json"]["start_date"]
            response = Mock()
            response.ok = True
            response.status_code = 200
            response.json.return_value = {
                "Data": [{"Data": [{"date": requested_start, "NDVI": 0.42}]}]
            }
            return response

        request_post.side_effect = successful
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.6, 39.9, 40.6, 40.6)], crs=4326
        )
        with tempfile.TemporaryDirectory() as cache_dir, patch(
            "zetriklim.climate_engine.CACHE_ROOT", Path(cache_dir)
        ):
            data, metadata = fetch_timeseries(
                "secret", area, date(2019, 6, 1), date(2021, 8, 31),
                "SENTINEL2_SR", "NDVI",
            )

        self.assertEqual(request_post.call_count, 3)
        self.assertEqual(len(data), 3)
        payloads = [call.kwargs["json"] for call in request_post.call_args_list]
        self.assertEqual(
            [(item["start_date"], item["end_date"]) for item in payloads],
            [
                ("2019-06-01", "2019-12-31"),
                ("2020-01-01", "2020-12-31"),
                ("2021-01-01", "2021-08-31"),
            ],
        )
        self.assertEqual(len(metadata["request_chunks"]), 3)

    @patch("zetriklim.climate_engine.requests.get")
    def test_climate_engine_dataset_range_is_read_from_metadata(
        self, request_get: Mock
    ) -> None:
        request_get.return_value.ok = True
        request_get.return_value.json.return_value = {
            "dataset": "CHIRPS_DAILY",
            "dates": {"start_date": "1981-01-01", "end_date": "2026-06-30"},
        }
        start, end = fetch_dataset_date_range("secret", "CHIRPS_DAILY")
        self.assertEqual(start, date(1981, 1, 1))
        self.assertEqual(end, date(2026, 6, 30))
        self.assertEqual(
            request_get.call_args.kwargs["params"]["dataset"], "CHIRPS_DAILY"
        )

    @patch("zetriklim.climate_engine.requests.get")
    def test_climate_engine_spi_map_uses_scale_window_not_full_archive(
        self, request_get: Mock
    ) -> None:
        request_get.return_value.ok = True
        request_get.return_value.json.return_value = {
            "tile_fetcher": "https://example.test/{z}/{x}/{y}.png"
        }
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.6, 39.9, 40.6, 40.6)], crs=4326
        )
        _, metadata = fetch_map_tile(
            "secret",
            area,
            date(1981, 1, 1),
            date(2025, 12, 31),
            "CHIRPS_DAILY",
            "precipitation",
            "SPI",
            spi_scale_months=3,
            reference_start_year=1981,
            reference_end_year=2024,
        )
        params = request_get.call_args.kwargs["params"]
        self.assertEqual(params["start_date"], "2025-10-01")
        self.assertEqual(params["end_date"], "2025-12-31")
        self.assertEqual(params["start_year"], 1981)
        self.assertEqual(params["end_year"], 2024)
        self.assertIn("colormap_min_max", params)
        self.assertIn("colormap_palette", params)
        palette = json.loads(params["colormap_palette"])
        self.assertGreaterEqual(len(palette), 5)
        self.assertTrue(all(isinstance(color, str) for color in palette))
        self.assertTrue(all(re.fullmatch(r"#[0-9a-fA-F]{6}", color) for color in palette))
        self.assertEqual(metadata["spi_scale_months"], 3)

    @patch("zetriklim.climate_engine.requests.get")
    def test_climate_engine_spi_nine_month_window_is_preserved(
        self, request_get: Mock
    ) -> None:
        request_get.return_value.ok = True
        request_get.return_value.json.return_value = {
            "tile_fetcher": "https://example.test/{z}/{x}/{y}.png"
        }
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.6, 39.9, 40.6, 40.6)], crs=4326
        )

        _, metadata = fetch_map_tile(
            "secret",
            area,
            date(1981, 1, 1),
            date(2025, 12, 31),
            "CHIRPS_DAILY",
            "precipitation",
            "SPI",
            spi_scale_months=9,
            reference_start_year=1981,
            reference_end_year=2024,
        )

        params = request_get.call_args.kwargs["params"]
        self.assertEqual(params["start_date"], "2025-04-01")
        self.assertEqual(params["end_date"], "2025-12-31")
        self.assertEqual(metadata["spi_scale_months"], 9)

    @patch("zetriklim.climate_engine.requests.get")
    def test_spi_map_uses_server_suggested_reference_start_year(
        self, request_get: Mock
    ) -> None:
        too_early = Mock()
        too_early.ok = False
        too_early.status_code = 500
        too_early.json.return_value = {
            "detail": (
                "The start_year parameter is too early for the aggregation period "
                "of 364 days, please change the start_year to 1982"
            )
        }
        too_early.text = ""
        corrected = Mock()
        corrected.ok = True
        corrected.status_code = 200
        corrected.json.return_value = {
            "tile_fetcher": "https://example.test/{z}/{x}/{y}.png"
        }
        request_get.side_effect = [too_early, corrected]
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.6, 39.9, 40.6, 40.6)], crs=4326
        )

        _, metadata = fetch_map_tile(
            "secret",
            area,
            date(1981, 1, 1),
            date(2025, 12, 31),
            "CHIRPS_DAILY",
            "precipitation",
            "SPI",
            spi_scale_months=12,
            reference_start_year=1981,
            reference_end_year=2024,
        )

        self.assertEqual(request_get.call_count, 2)
        self.assertEqual(request_get.call_args.kwargs["params"]["start_year"], 1982)
        self.assertEqual(metadata["reference_period"], "1982/2024")

    @patch("zetriklim.climate_engine.requests.get")
    def test_climate_engine_anomaly_map_uses_baseline_and_valid_palette(
        self, request_get: Mock
    ) -> None:
        request_get.return_value.ok = True
        request_get.return_value.json.return_value = {
            "tile_fetcher": "https://example.test/{z}/{x}/{y}.png"
        }
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.6, 39.9, 40.6, 40.6)], crs=4326
        )
        _, metadata = fetch_map_tile(
            "secret",
            area,
            date(2025, 7, 1),
            date(2026, 6, 30),
            "ERA5_AG",
            "mean_2m_air_temperature",
            "Sıcaklık Anomalisi",
            reference_start_year=1981,
            reference_end_year=2024,
            temporal_statistic="mean",
            map_kind="anomalies",
            anomaly_calculation="anom",
        )
        self.assertTrue(request_get.call_args.args[0].endswith("/raster/mapid/anomalies"))
        params = request_get.call_args.kwargs["params"]
        self.assertEqual(params["start_year"], 1981)
        self.assertEqual(params["end_year"], 2024)
        self.assertEqual(params["calculation"], "anom")
        self.assertTrue(all(color.startswith("#") for color in json.loads(params["colormap_palette"])))
        self.assertEqual(metadata["map_kind"], "anomalies")

    def test_academic_focus_assigns_complementary_bundle(self) -> None:
        ndvi = academic_defaults("NDVI")
        self.assertEqual(ndvi["response_indices"], ["NDVI"])
        self.assertEqual(ndvi["drought_indices"], ["SPI", "SPEI"])
        self.assertIn(12, ndvi["scales"])
        self.assertEqual(academic_defaults("SPI")["response_indices"], ["NDVI"])

    def test_climate_engine_columns_are_normalized_for_every_focus(self) -> None:
        base = pd.DataFrame(
            {
                "Tarih": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "Örnek ID": 1,
                "Enlem": 40.0,
                "Boylam": 29.0,
                "value": [0.1, 0.2, 0.3],
            }
        )
        for analysis in ("NDVI", "EVI", "LST"):
            normalized = normalize_analysis_column(base, analysis)
            self.assertIn(analysis, normalized)
            self.assertNotIn("value", normalized)

    def test_daily_and_monthly_sources_are_harmonized_without_none_rows(self) -> None:
        daily_dates = pd.date_range("2020-01-01", "2020-02-29", freq="D")
        data = pd.DataFrame(
            {
                "Tarih": daily_dates,
                "Örnek ID": 1,
                "Enlem": 40.2,
                "Boylam": 29.1,
                "Toplam yağış (mm)": 1.0,
                "Ortalama sıcaklık (°C)": np.where(daily_dates.month == 1, 5.0, 8.0),
            }
        )
        monthly = harmonize_monthly_data(data)
        self.assertEqual(len(monthly), 2)
        self.assertEqual(monthly["Toplam yağış (mm)"].tolist(), [31.0, 29.0])
        self.assertEqual(monthly["Ortalama sıcaklık (°C)"].tolist(), [5.0, 8.0])
        self.assertFalse(monthly.isna().any().any())
        csv_text = dataframe_to_csv(monthly).decode("utf-8-sig")
        self.assertNotIn("None", csv_text)

    def test_thesis_map_and_grouped_series_are_valid_pngs(self) -> None:
        area = gpd.GeoDataFrame({"ad": ["Test"]}, geometry=[box(29, 40, 30, 41)], crs=4326)
        summary = inspect_geodataframe(area)
        map_png = build_area_map_png(area, summary.centroid, source_note="GADM 4.1 test")
        series_png = build_timeseries_png(
            pd.DataFrame(
                {
                    "Tarih": pd.date_range("2020-01-01", periods=24, freq="MS"),
                    "Toplam yağış (mm)": np.arange(24),
                    "Ortalama sıcaklık (°C)": np.linspace(2, 22, 24),
                    "NDVI": np.linspace(0.1, 0.8, 24),
                }
            )
        )
        for content in (map_png, series_png):
            image = Image.open(io.BytesIO(content))
            self.assertEqual(image.format, "PNG")
            self.assertGreater(image.width, 1000)

    def test_each_map_has_geopackage_and_complete_shapefile_bundle(self) -> None:
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(29, 40, 30, 41)], crs=4326
        )
        gpkg, shp_zip = build_map_vector_files(
            area,
            {
                "label": "Kuraklık dağılımı · SPI-3",
                "analysis": "SPI",
                "source": "Climate Engine",
                "dataset": "CHIRPS_DAILY",
                "variable": "spi",
                "start": "2025-10-01",
                "end": "2025-12-31",
                "quality": "uygun",
            },
        )
        self.assertTrue(gpkg.startswith(b"SQLite format 3"))
        with zipfile.ZipFile(io.BytesIO(shp_zip)) as archive:
            names = set(archive.namelist())
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            self.assertTrue(any(name.endswith(suffix) for name in names))
        self.assertIn("harita-metadata.json", names)

    def test_academic_chart_suite_and_thesis_workbook(self) -> None:
        dates = pd.date_range("1981-01-01", periods=180, freq="MS")
        climate = pd.DataFrame(
            {
                "Tarih": dates,
                "Toplam yağış (mm)": 55 + 30 * np.sin(np.arange(len(dates)) / 6),
                "Ortalama sıcaklık (°C)": 10 + 12 * np.sin(np.arange(len(dates)) / 6 - 1),
                "Minimum sıcaklık (°C)": -1 + 10 * np.sin(np.arange(len(dates)) / 6 - 1),
                "Maksimum sıcaklık (°C)": 20 + 14 * np.sin(np.arange(len(dates)) / 6 - 1),
                "Referans ET₀ (mm)": 65 + 25 * np.sin(np.arange(len(dates)) / 6 - 1),
            }
        )
        drought = pd.DataFrame(
            {
                "Tarih": dates,
                "SPI-3": np.sin(np.arange(len(dates)) / 8),
                "SPI-6": np.sin(np.arange(len(dates)) / 12),
                "SPI-12": np.sin(np.arange(len(dates)) / 18),
                "SPEI-3": np.cos(np.arange(len(dates)) / 9),
                "SPEI-6": np.cos(np.arange(len(dates)) / 13),
                "SPEI-12": np.cos(np.arange(len(dates)) / 20),
            }
        )
        charts = build_academic_chart_suite(climate, drought_table=drought)
        self.assertGreaterEqual(len(charts), 6)
        self.assertIn("grafik-04-spi-serileri.png", charts)
        self.assertIn("grafik-05-spei-serileri.png", charts)
        self.assertNotIn("grafik-04-kuraklik-indisleri.png", charts)
        for content in charts.values():
            self.assertEqual(Image.open(io.BytesIO(content)).format, "PNG")
        workbook = build_spi_thesis_excel(drought)
        self.assertGreater(len(workbook), 5000)

    @patch("zetriklim.artifacts.requests.get")
    def test_interactive_tiles_produce_clipped_png_and_quality_record(
        self, request_get: Mock
    ) -> None:
        def tile_response(url: str, **_kwargs):
            tile_array = np.zeros((256, 256, 4), dtype=np.uint8)
            tile_array[..., 0] = np.arange(256, dtype=np.uint8)[None, :]
            tile_array[..., 1] = np.arange(256, dtype=np.uint8)[:, None]
            tile_array[..., 2] = sum(url.encode("utf-8")) % 255
            tile_array[..., 3] = 255
            tile_buffer = io.BytesIO()
            Image.fromarray(tile_array, mode="RGBA").save(tile_buffer, format="PNG")
            response = Mock(content=tile_buffer.getvalue())
            response.raise_for_status.return_value = None
            return response

        request_get.side_effect = tile_response
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.8, 40.0, 40.3, 40.5)], crs=4326
        )
        png, quality = build_tile_map_png(
            "https://example.test/{z}/{x}/{y}.png",
            area,
            title="SPI dağılımı",
            analysis="SPI",
            source="Test",
            period="2025-10-01/2025-12-31",
        )
        self.assertEqual(Image.open(io.BytesIO(png)).format, "PNG")
        self.assertEqual(quality["status"], "uygun")
        self.assertEqual(quality["failed_tiles"], 0)
        self.assertLessEqual(quality["requested_tiles"], 4)
        self.assertEqual(quality["downloaded_tiles"], quality["requested_tiles"])
        self.assertLess(quality["export_grid_size_m"], 1222.99)
        self.assertEqual(quality["display_min"], -3.0)
        self.assertEqual(quality["display_max"], 3.0)
        overlay = Image.open(io.BytesIO(quality["_overlay_png"])).convert("RGBA")
        self.assertEqual(overlay.getpixel((0, 0))[3], 0)
        self.assertEqual(len(quality["_overlay_bounds"]), 2)
        self.assertIn(quality["_geotiff_bytes"][:2], {b"II", b"MM"})
        with MemoryFile(quality["_geotiff_bytes"]) as memory:
            with memory.open() as dataset:
                self.assertEqual(dataset.count, 1)
                self.assertEqual(dataset.dtypes[0], "float32")
                self.assertEqual(dataset.nodata, -9999.0)
                self.assertEqual(dataset.descriptions[0], "SPI")
        with zipfile.ZipFile(io.BytesIO(quality["_classified_shp_bytes"])) as archive:
            export_names = set(archive.namelist())
        self.assertTrue(any(name.endswith(".shp") for name in export_names))
        self.assertFalse(any(name.endswith(".qml") for name in export_names))
        self.assertNotIn("_gpkg_raster_bytes", quality)
        self.assertNotIn("_qgis_raster_qml", quality)
        self.assertNotIn("_qgis_raster_package", quality)

    @patch("zetriklim.artifacts.requests.get")
    def test_opaque_white_nodata_is_filled_from_parent_zoom(self, request_get: Mock) -> None:
        requested_zooms: list[int] = []

        def tile_response(url: str, **_kwargs):
            zoom = int(url.split("/")[-3])
            requested_zooms.append(zoom)
            tile_array = np.full((256, 256, 4), 255, dtype=np.uint8)
            if zoom < max(requested_zooms):
                tile_array[..., 0] = 67
                tile_array[..., 1] = 147
                tile_array[..., 2] = 195
            tile_buffer = io.BytesIO()
            Image.fromarray(tile_array, mode="RGBA").save(tile_buffer, format="PNG")
            response = Mock(content=tile_buffer.getvalue())
            response.raise_for_status.return_value = None
            return response

        request_get.side_effect = tile_response
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.8, 40.0, 40.3, 40.5)], crs=4326
        )
        png, quality = build_tile_map_png(
            "https://example.test/{z}/{x}/{y}.png",
            area,
            title="NDVI dağılımı",
            analysis="NDVI",
            source="Test",
            period="2025-10-01/2025-12-31",
        )
        self.assertEqual(Image.open(io.BytesIO(png)).format, "PNG")
        self.assertLess(quality["primary_coverage_ratio"], 0.01)
        self.assertGreater(quality["fallback_tiles"], 0)
        self.assertGreaterEqual(quality["coverage_ratio"], 0.999)

    @patch("zetriklim.artifacts.requests.get")
    def test_saturated_spi_tile_is_rejected(self, request_get: Mock) -> None:
        tile_array = np.zeros((256, 256, 4), dtype=np.uint8)
        tile_array[..., :3] = np.array([33, 102, 172], dtype=np.uint8)
        tile_array[..., 3] = 255
        tile_buffer = io.BytesIO()
        Image.fromarray(tile_array, mode="RGBA").save(tile_buffer, format="PNG")
        response = Mock(content=tile_buffer.getvalue())
        response.raise_for_status.return_value = None
        request_get.return_value = response
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.8, 40.0, 40.3, 40.5)], crs=4326
        )

        with self.assertRaisesRegex(RuntimeError, "doygun"):
            build_tile_map_png(
                "https://example.test/{z}/{x}/{y}.png",
                area,
                title="SPI dağılımı",
                analysis="SPI",
                source="Test",
                period="2025-10-01/2025-12-31",
            )

    @patch("zetriklim.artifacts.time.sleep", return_value=None)
    @patch("zetriklim.artifacts.requests.get")
    def test_failed_primary_tiles_are_recovered_from_parent_zoom(
        self, request_get: Mock, _sleep: Mock
    ) -> None:
        highest_zoom = None
        primary_allowed_url = None

        def tile_response(url: str, **_kwargs):
            nonlocal highest_zoom, primary_allowed_url
            zoom = int(url.split("/")[-3])
            highest_zoom = zoom if highest_zoom is None else max(highest_zoom, zoom)
            if zoom == highest_zoom and primary_allowed_url is None:
                primary_allowed_url = url
            if zoom == highest_zoom and url != primary_allowed_url:
                raise RuntimeError("geçici birincil karo hatası")
            tile_array = np.zeros((256, 256, 4), dtype=np.uint8)
            tile_array[..., 0] = 67
            tile_array[..., 1] = 147
            tile_array[..., 2] = 195
            tile_array[..., 3] = 255
            tile_buffer = io.BytesIO()
            Image.fromarray(tile_array, mode="RGBA").save(tile_buffer, format="PNG")
            response = Mock(content=tile_buffer.getvalue())
            response.raise_for_status.return_value = None
            return response

        request_get.side_effect = tile_response
        area = gpd.GeoDataFrame(
            {"ad": ["Test"]}, geometry=[box(39.8, 40.0, 40.3, 40.5)], crs=4326
        )
        _, quality = build_tile_map_png(
            "https://example.test/{z}/{x}/{y}.png",
            area,
            title="NDVI dağılımı",
            analysis="NDVI",
            source="Test",
            period="2025-10-01/2025-12-31",
        )
        self.assertEqual(quality["status"], "uygun")
        self.assertGreaterEqual(quality["coverage_ratio"], 0.999)

    @patch("zetriklim.gadm.requests.get")
    def test_gadm_download_is_validated(self, request_get: Mock) -> None:
        source = gpd.GeoDataFrame({"NAME_1": ["Test"]}, geometry=[box(29, 40, 30, 41)], crs=4326)
        response = Mock(status_code=200, content=source.to_json().encode("utf-8"))
        response.raise_for_status.return_value = None
        request_get.return_value = response
        result = fetch_gadm("tur", 1)
        self.assertEqual(result.iloc[0]["NAME_1"], "Test")
        self.assertEqual(result.attrs["version"], "4.1")

    def test_gadm_country_level_uses_country_label(self) -> None:
        frame = gpd.GeoDataFrame({"COUNTRY": ["Türkiye"]}, geometry=[box(29, 40, 30, 41)], crs=4326)
        self.assertEqual(name_column(frame, 0), "COUNTRY")


if __name__ == "__main__":
    unittest.main()

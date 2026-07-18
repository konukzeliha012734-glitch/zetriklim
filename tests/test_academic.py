from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from zetriklim.academic import (
    build_academic_report_html,
    calculate_spei_table,
    calculate_spi_academic,
    detect_drought_events,
    lagged_correlations,
    last_complete_month,
    safe_monthly_end,
    trend_analysis,
    run_academic_analysis,
    run_remote_sensing_analysis,
    validation_metrics,
)


class AcademicAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("1981-01-01", "2024-12-01", freq="MS")
        seasonal = 45 + 30 * np.sin(2 * np.pi * dates.month / 12)
        precipitation = np.maximum(seasonal + rng.normal(0, 8, len(dates)), 0)
        pet = np.maximum(35 + 15 * np.cos(2 * np.pi * dates.month / 12), 1)
        cls.frame = pd.DataFrame(
            {
                "Tarih": dates,
                "Yağış": precipitation,
                "PET": pet,
                "NDVI": np.roll(precipitation, 2) / 150 + rng.normal(0, 0.02, len(dates)),
            }
        )

    def test_complete_month_guard(self) -> None:
        self.assertEqual(last_complete_month(date(2026, 7, 13)), date(2026, 6, 30))
        clipped, changed = safe_monthly_end(date(2026, 7, 13), date(2026, 7, 13))
        self.assertTrue(changed)
        self.assertEqual(clipped, date(2026, 6, 30))

    def test_spi_and_spei_are_standardized(self) -> None:
        spi = calculate_spi_academic(
            self.frame,
            "Yağış",
            [3],
            baseline_start=1991,
            baseline_end=2020,
        )
        spei = calculate_spei_table(
            self.frame,
            "Yağış",
            "PET",
            [3],
            baseline_start=1991,
            baseline_end=2020,
        )
        self.assertGreater(spi["SPI-3"].notna().sum(), 300)
        self.assertGreater(spei["SPEI-3"].notna().sum(), 300)
        baseline = spi[(spi["Tarih"].dt.year >= 1991) & (spi["Tarih"].dt.year <= 2020)]
        self.assertLess(abs(baseline["SPI-3"].mean()), 0.15)

    def test_drought_event_catalog(self) -> None:
        data = pd.DataFrame(
            {
                "Tarih": pd.date_range("2020-01-01", periods=8, freq="MS"),
                "SPI-3": [-0.2, -1.1, -1.4, -0.5, -1.2, -1.8, -1.1, 0.2],
            }
        )
        events = detect_drought_events(data, "SPI-3")
        self.assertEqual(len(events), 2)
        self.assertEqual(events.iloc[0]["Başlangıç"], pd.Timestamp("2020-02-01"))
        self.assertEqual(events.iloc[0]["Bitiş"], pd.Timestamp("2020-03-31"))
        self.assertEqual(events.iloc[1]["Süre (ay)"], 3)
        self.assertEqual(events.iloc[1]["Bitiş"], pd.Timestamp("2020-07-31"))

    def test_single_month_drought_event_is_explicit(self) -> None:
        data = pd.DataFrame(
            {
                "Tarih": ["2021-01-01", "2021-02-01", "2021-03-01"],
                "SPI-1": [0.1, -1.4, 0.2],
            }
        )
        event = detect_drought_events(data, "SPI-1").iloc[0]
        self.assertEqual(event["Başlangıç"], pd.Timestamp("2021-02-01"))
        self.assertEqual(event["Bitiş"], pd.Timestamp("2021-02-28"))
        self.assertEqual(event["Olay türü"], "Tek aylık")

    def test_lag_detection(self) -> None:
        data = self.frame.copy()
        data["SPI-3"] = (data["Yağış"] - data["Yağış"].mean()) / data["Yağış"].std()
        result = lagged_correlations(data, ["SPI-3"], ["NDVI"], max_lag=4, remove_seasonality=False)
        best = result[result["En güçlü gecikme"]].iloc[0]
        self.assertEqual(best["Gecikme (ay)"], 2)

    def test_trend_and_validation_metrics(self) -> None:
        trend_frame = pd.DataFrame(
            {"Tarih": pd.date_range("2000-01-01", periods=120, freq="MS"), "x": np.arange(120)}
        )
        trend = trend_analysis(trend_frame, ["x"], prewhiten=False)
        self.assertTrue(bool(trend.iloc[0]["Anlamlı eğilim"]))
        metrics = validation_metrics(pd.Series([1, 2, 3, 4]), pd.Series([1.1, 1.9, 3.2, 3.8]))
        self.assertLess(metrics["RMSE"], 0.25)

    def test_standalone_ndvi_analysis_has_complete_findings(self) -> None:
        results = run_remote_sensing_analysis(
            self.frame[["Tarih", "NDVI"]],
            response_columns=["NDVI"],
            config={
                "anomaly_baseline_start": 1991,
                "anomaly_baseline_end": 2020,
                "change_window_years": 3,
                "prewhiten": True,
                "seasonal_mk": True,
                "alpha": 0.05,
            },
        )
        self.assertEqual(
            set(results),
            {
                "Uzaktan Algılama Özeti",
                "Eğilim ve Değişim",
                "Mevsimsel Profil",
                "Anomali Serisi",
                "Kalite Kontrol",
            },
        )
        self.assertEqual(len(results["Mevsimsel Profil"]), 12)
        self.assertIn("NDVI standart anomalisi", results["Anomali Serisi"])
        summary = results["Uzaktan Algılama Özeti"].iloc[0]
        self.assertGreater(summary["Geçerli gözlem"], 500)
        self.assertTrue(np.isfinite(summary["Sen eğimi / yıl"]))

    def test_full_bundle_and_report(self) -> None:
        data = self.frame.copy()
        data["ERA5-Land yağış (mm)"] = data["Yağış"] * 1.05
        results = run_academic_analysis(
            data,
            precipitation_column="Yağış",
            pet_column="PET",
            response_columns=["NDVI"],
            validation_columns=["ERA5-Land yağış (mm)"],
            config={
                "scales": [3, 12],
                "drought_indices": ["SPI", "SPEI"],
                "baseline_start": 1991,
                "baseline_end": 2020,
                "max_lag": 4,
            },
        )
        self.assertFalse(results["Kuraklık Serisi"].empty)
        self.assertFalse(results["Gecikmeli İlişki"].empty)
        self.assertFalse(results["Kaynak Doğrulama"].empty)
        self.assertFalse(results["Belirsizlik"].empty)
        report = build_academic_report_html(
            study={"title": "Test", "question": "Soru", "hypotheses": "H1"},
            config={"scales": [3, 12], "baseline_start": 1991, "baseline_end": 2020},
            results=results,
            source_note="Sentetik veri",
            context={"Çalışma alanı": "100 km²", "Dönem": "1981–2024"},
            figures={"Test şekli": b"\x89PNG\r\n"},
        )
        self.assertIn(b"<!doctype html>", report)
        self.assertIn("Yönetici özeti".encode("utf-8"), report)
        self.assertIn("Kuraklık olay tanımı".encode("utf-8"), report)
        self.assertIn(b"data:image/png;base64", report)
        self.assertGreater(len(report), 10_000)


if __name__ == "__main__":
    unittest.main()

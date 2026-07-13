from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import box

from zetriklim.climate_engine import fetch_timeseries


class _Response:
    def __init__(self, status_code: int, body=None, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        if self._body is None:
            raise ValueError("JSON yok")
        return self._body


def _success(day: str, value: float) -> _Response:
    return _Response(
        200,
        {"Data": [{"Data": [{"Date": day, "precipitation": value}]}]},
    )


class ClimateEngineRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.area = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(35.0, 38.0, 35.5, 38.5)],
            crs=4326,
        )

    @patch("zetriklim.climate_engine.requests.post")
    def test_internal_server_error_splits_period_automatically(self, post) -> None:
        post.side_effect = [
            _Response(500, text="Internal Server Error"),
            _success("1981-01-01", 10.0),
            _success("1982-01-01", 12.0),
        ]

        data, metadata = fetch_timeseries(
            "test-key",
            self.area,
            date(1981, 1, 1),
            date(1982, 12, 31),
            "CHIRPS_DAILY",
            "precipitation",
        )

        self.assertEqual(len(data), 2)
        self.assertEqual(post.call_count, 3)
        first_payload = post.call_args_list[0].kwargs["json"]
        self.assertIn("simplify_geometry", first_payload)
        self.assertTrue(metadata["request_chunks"][0]["metadata"]["split"])


if __name__ == "__main__":
    unittest.main()

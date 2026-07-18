"""Kullanıcıya gösterilen yaklaşık işlem süresi hesapları."""

from __future__ import annotations

import math


def estimate_analysis_seconds(
    area_km2: float,
    *,
    provider: str,
    map_count: int,
    period_years: float,
    academic_mode: bool = False,
    spi_map_count: int = 0,
    archive_map_count: int = 0,
    anomaly_map_count: int = 0,
    observed_seconds: float | None = None,
) -> tuple[int, int]:
    """Gerçek iş türleri ve önceki çalışma süresine duyarlı bir aralık döndürür."""
    area = max(float(area_km2), 0.01)
    maps = max(int(map_count), 0)
    years = max(float(period_years), 1 / 12)
    if provider == "Climate Engine":
        spi_maps = max(int(spi_map_count), 0)
        archive_maps = max(int(archive_map_count), 0)
        anomaly_maps = max(int(anomaly_map_count), 0)
        other_maps = max(maps - spi_maps - archive_maps - anomaly_maps, 0)
        map_work = (
            50.0 * spi_maps
            + (120.0 + 12.0 * years) * archive_maps
            + (180.0 + 16.0 * years) * anomaly_maps
            + 45.0 * other_maps
        )
        # Climate Engine MapID işleri en fazla üç bağımsız işçiyle yürütülür.
        # Ağ ve servis beklemeleri örtüştüğü için harita süreleri artık doğrudan
        # toplanmaz; güvenli tarafta kalmak için teorik üç kat yerine 2,2 kat
        # eşzamanlılık kazancı kullanılır.
        parallel_factor = min(2.2, max(float(maps), 1.0))
        midpoint = (
            90.0
            + 35.0 * math.log2(1.0 + area / 50.0)
            + 3.0 * years
            + map_work / parallel_factor
            + (180.0 if academic_mode else 0.0)
        )
    else:
        midpoint = (
            50.0
            + 14.0 * math.log2(1.0 + area / 25.0)
            + 5.0 * math.sqrt(years)
            + 20.0 * maps
            + (120.0 if academic_mode else 0.0)
        )
    if observed_seconds is not None and float(observed_seconds) > 0:
        midpoint = 0.25 * midpoint + 0.75 * float(observed_seconds)
    lower = max(60, int(round(midpoint * 0.88 / 60) * 60))
    upper = max(lower + 60, int(round(midpoint * 1.15 / 60) * 60))
    return lower, upper


def format_duration_range(lower_seconds: int, upper_seconds: int) -> str:
    def format_one(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds} sn"
        minutes = max(1, round(seconds / 60))
        return f"{minutes} dk"

    return f"{format_one(lower_seconds)} – {format_one(upper_seconds)}"

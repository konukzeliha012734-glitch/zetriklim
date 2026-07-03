"""WMO yaklaşımına uygun Gamma dağılımlı SPI hesapları."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm


SPI_CLASSES = [
    (-np.inf, -2.0, "Aşırı kurak"),
    (-2.0, -1.5, "Şiddetli kurak"),
    (-1.5, -1.0, "Orta kurak"),
    (-1.0, 1.0, "Normale yakın"),
    (1.0, 1.5, "Orta nemli"),
    (1.5, 2.0, "Çok nemli"),
    (2.0, np.inf, "Aşırı nemli"),
]


def classify_spi(value: float) -> str:
    if pd.isna(value):
        return "Hesaplanamadı"
    for lower, upper, label in SPI_CLASSES:
        if lower <= value < upper:
            return label
    return "Hesaplanamadı"


def _fit_spi(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    positive = valid[valid > 0]
    if len(positive) < 10:
        return result
    zero_probability = float((valid == 0).sum() / len(valid))
    shape, _, scale = gamma.fit(positive.to_numpy(), floc=0)
    probabilities = pd.Series(np.nan, index=valid.index, dtype=float)
    positive_mask = valid > 0
    probabilities.loc[positive_mask] = zero_probability + (1 - zero_probability) * gamma.cdf(
        valid.loc[positive_mask], shape, loc=0, scale=scale
    )
    probabilities.loc[~positive_mask] = max(zero_probability, 1e-8) / 2
    probabilities = probabilities.clip(1e-8, 1 - 1e-8)
    result.loc[probabilities.index] = norm.ppf(probabilities)
    return result


def calculate_spi_table(
    data: pd.DataFrame,
    precipitation_column: str,
    scales: list[int],
) -> pd.DataFrame:
    monthly = (
        data.set_index("Tarih")[precipitation_column]
        .resample("MS")
        .sum(min_count=1)
        .to_frame("Aylık yağış (mm)")
    )
    for scale in sorted(set(scales)):
        accumulated = monthly["Aylık yağış (mm)"].rolling(scale, min_periods=scale).sum()
        spi = accumulated.groupby(accumulated.index.month, group_keys=False).apply(_fit_spi)
        monthly[f"SPI-{scale}"] = spi.sort_index()
        monthly[f"SPI-{scale} sınıfı"] = monthly[f"SPI-{scale}"].map(classify_spi)
    return monthly.reset_index()


def calculate_spi_pixel_stack(stack: np.ndarray, target_index: int = -1) -> np.ndarray:
    """(zaman, satır, sütun) yağış yığınından hedef dönem SPI rasteri üretir."""
    bands, rows, cols = stack.shape
    flat = stack.reshape(bands, -1)
    output = np.full(flat.shape[1], np.nan, dtype=np.float32)
    for pixel in range(flat.shape[1]):
        series = flat[:, pixel]
        valid = series[np.isfinite(series)]
        if len(valid) < 10:
            continue
        positive = valid[valid > 0]
        if len(positive) < 10:
            continue
        target = series[target_index]
        if not np.isfinite(target):
            continue
        zero_probability = float(np.sum(valid == 0) / len(valid))
        shape, _, scale = gamma.fit(positive, floc=0)
        if target <= 0:
            probability = max(zero_probability, 1e-8) / 2
        else:
            probability = zero_probability + (1 - zero_probability) * gamma.cdf(
                target, shape, loc=0, scale=scale
            )
        output[pixel] = norm.ppf(np.clip(probability, 1e-8, 1 - 1e-8))
    return output.reshape(rows, cols)

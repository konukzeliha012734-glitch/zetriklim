"""Lisansüstü düzey için yeniden üretilebilir iklim ve kuraklık analizleri.

Bu modül kullanıcı arayüzünden bağımsızdır. Böylece aynı yöntemler Streamlit,
notebook ve test ortamlarında aynı girdilerle aynı sonucu üretir.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import base64
from html import escape
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import (
    fisk,
    gamma,
    kstest,
    norm,
    pearson3,
    pearsonr,
    rankdata,
    spearmanr,
    theilslopes,
)

from zetriklim.spi import classify_spi


MIN_FIT_SAMPLES = 10


def academic_defaults(focus: str) -> dict[str, object]:
    """Seçilen araştırma odağı için tamamlayıcı, düzenlenebilir akademik paketi kurar."""
    focus = str(focus).upper()
    # SPI odağında da gecikmeli tepki tablosunun boş kalmaması için en uzun ve
    # yaygın uydu arşivine sahip NDVI otomatik eş değişken olarak eklenir.
    # EVI ve LST kullanıcı tarafından isteğe bağlı olarak genişletilebilir.
    remote_methods = {
        "NDVI", "NDWI", "NDMI", "NDBI", "EVI", "SAVI", "LST",
        "DEM", "SLOPE", "ASPECT", "TWI",
    }
    response = [focus] if focus in remote_methods else ["NDVI"]
    return {
        "focus": focus,
        "drought_indices": ["SPI", "SPEI"],
        "response_indices": response,
        "scales": [1, 3, 6, 12, 24] if focus in {"SPI", "LST"} else [1, 3, 6, 12],
        "title": f"Havza ölçeğinde {focus} odaklı çok kaynaklı iklim analizi",
        "question": (
            f"Meteorolojik ve sıcaklık kaynaklı kuraklık koşulları {focus} göstergesinde "
            "hangi zaman ölçeğinde ve kaç aylık gecikmeyle karşılık bulmaktadır?"
        ),
    }


def harmonize_monthly_data(data: pd.DataFrame, date_column: str = "Tarih") -> pd.DataFrame:
    """Farklı günlük/aylık kaynakları tek ve benzersiz aylık gözlem tablosuna dönüştürür."""
    if date_column not in data:
        raise ValueError(f"Tarih sütunu bulunamadı: {date_column}")
    frame = data.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column]).sort_values(date_column)
    if frame.empty:
        raise ValueError("Geçerli tarih içeren kayıt bulunamadı.")
    frame = frame.replace([np.inf, -np.inf], np.nan).set_index(date_column)
    aggregations: dict[str, object] = {}
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            frame[column] = values
            label = str(column).casefold()
            is_total = any(
                token in label
                for token in ("yağış", "precip", "rain", "pet", "et₀", "evapotrans", "radyasyon", "kar yağışı")
            )
            aggregations[column] = (
                (lambda series: series.sum(min_count=1)) if is_total else "mean"
            )
        else:
            aggregations[column] = lambda series: series.dropna().iloc[0] if series.notna().any() else pd.NA
    monthly = frame.resample("MS").agg(aggregations).reset_index()
    monthly = monthly.dropna(axis=1, how="all")
    return monthly


def last_complete_month(today: date | None = None) -> date:
    """Bugüne göre tamamlanmış son ayın son gününü döndürür."""
    today = today or date.today()
    return today.replace(day=1) - timedelta(days=1)


def safe_monthly_end(end_date: date, today: date | None = None) -> tuple[date, bool]:
    """Aylık toplamların tamamlanmamış güncel ayı içermesini engeller."""
    boundary = last_complete_month(today)
    if end_date > boundary:
        return boundary, True
    return end_date, False


def _monthly_series(data: pd.DataFrame, column: str, date_column: str = "Tarih") -> pd.Series:
    if date_column not in data:
        raise ValueError(f"Tarih sütunu bulunamadı: {date_column}")
    if column not in data:
        raise ValueError(f"Değer sütunu bulunamadı: {column}")
    dates = pd.to_datetime(data[date_column], errors="coerce")
    values = pd.to_numeric(data[column], errors="coerce")
    frame = pd.DataFrame({date_column: dates, column: values}).dropna(subset=[date_column])
    return frame.set_index(date_column)[column].sort_index().resample("MS").sum(min_count=1)


def _standardize(
    values: pd.Series,
    reference: pd.Series,
    distribution: str,
) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    evaluation = pd.to_numeric(values, errors="coerce").dropna()
    if len(reference) < MIN_FIT_SAMPLES or evaluation.empty or reference.nunique() < 2:
        return result

    label = distribution.casefold()
    try:
        if "gamma" in label:
            positive = reference[reference > 0]
            if len(positive) < MIN_FIT_SAMPLES:
                return result
            zero_probability = float((reference == 0).mean())
            shape, _, scale = gamma.fit(positive.to_numpy(), floc=0)
            probability = pd.Series(np.nan, index=evaluation.index, dtype=float)
            positive_mask = evaluation > 0
            probability.loc[positive_mask] = zero_probability + (
                1 - zero_probability
            ) * gamma.cdf(evaluation.loc[positive_mask], shape, loc=0, scale=scale)
            probability.loc[~positive_mask] = max(zero_probability, 1e-8) / 2
        elif "pearson" in label:
            params = pearson3.fit(reference.to_numpy())
            probability = pd.Series(
                pearson3.cdf(evaluation.to_numpy(), *params), index=evaluation.index
            )
        elif "log" in label or "fisk" in label:
            params = fisk.fit(reference.to_numpy())
            probability = pd.Series(
                fisk.cdf(evaluation.to_numpy(), *params), index=evaluation.index
            )
        else:
            mean = float(reference.mean())
            std = float(reference.std(ddof=1))
            if not np.isfinite(std) or std <= 0:
                return result
            probability = pd.Series(norm.cdf((evaluation - mean) / std), index=evaluation.index)
    except (ValueError, FloatingPointError, RuntimeError):
        return result

    probability = probability.clip(1e-8, 1 - 1e-8)
    result.loc[probability.index] = norm.ppf(probability)
    return result


def _index_table(
    monthly_values: pd.Series,
    *,
    scales: Iterable[int],
    index_name: str,
    distribution: str,
    baseline_start: int,
    baseline_end: int,
) -> pd.DataFrame:
    table = monthly_values.to_frame("Aylık girdi")
    reference_year = table.index.year
    for scale in sorted({int(item) for item in scales if int(item) > 0}):
        accumulated = table["Aylık girdi"].rolling(scale, min_periods=scale).sum()
        standardized = pd.Series(np.nan, index=table.index, dtype=float)
        for month in range(1, 13):
            month_mask = accumulated.index.month == month
            reference_mask = (
                month_mask
                & (reference_year >= baseline_start)
                & (reference_year <= baseline_end)
            )
            standardized.loc[month_mask] = _standardize(
                accumulated.loc[month_mask],
                accumulated.loc[reference_mask],
                distribution,
            )
        name = f"{index_name}-{scale}"
        table[name] = standardized
        table[f"{name} sınıfı"] = table[name].map(classify_spi)
    return table.reset_index().rename(columns={table.index.name or "index": "Tarih"})


def calculate_spi_academic(
    data: pd.DataFrame,
    precipitation_column: str,
    scales: Iterable[int],
    *,
    baseline_start: int,
    baseline_end: int,
    distribution: str = "Gamma",
    date_column: str = "Tarih",
) -> pd.DataFrame:
    monthly = _monthly_series(data, precipitation_column, date_column)
    result = _index_table(
        monthly,
        scales=scales,
        index_name="SPI",
        distribution=distribution,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )
    return result.rename(columns={"Aylık girdi": "Aylık yağış (mm)"})


def calculate_spei_table(
    data: pd.DataFrame,
    precipitation_column: str,
    pet_column: str,
    scales: Iterable[int],
    *,
    baseline_start: int,
    baseline_end: int,
    distribution: str = "Log-logistic",
    date_column: str = "Tarih",
) -> pd.DataFrame:
    precipitation = _monthly_series(data, precipitation_column, date_column)
    pet = _monthly_series(data, pet_column, date_column)
    monthly = pd.concat([precipitation.rename("P"), pet.rename("PET")], axis=1)
    water_balance = monthly["P"] - monthly["PET"]
    result = _index_table(
        water_balance,
        scales=scales,
        index_name="SPEI",
        distribution=distribution,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )
    return result.rename(columns={"Aylık girdi": "Aylık su dengesi P−PET (mm)"})


def distribution_diagnostics(
    monthly_values: pd.Series,
    *,
    scales: Iterable[int],
    distribution: str,
    baseline_start: int,
    baseline_end: int,
    index_name: str,
) -> pd.DataFrame:
    """Her ay ve ölçek için KS uyum testi ile AIC değerini raporlar."""
    monthly_values = monthly_values.sort_index()
    rows: list[dict[str, object]] = []
    for scale in sorted({int(item) for item in scales if int(item) > 0}):
        accumulated = monthly_values.rolling(scale, min_periods=scale).sum()
        for month in range(1, 13):
            sample = accumulated[
                (accumulated.index.month == month)
                & (accumulated.index.year >= baseline_start)
                & (accumulated.index.year <= baseline_end)
            ].dropna()
            row: dict[str, object] = {
                "İndis": index_name,
                "Ölçek (ay)": scale,
                "Takvim ayı": month,
                "Dağılım": distribution,
                "Örnek sayısı": len(sample),
                "KS istatistiği": np.nan,
                "KS p-değeri": np.nan,
                "AIC": np.nan,
            }
            if len(sample) >= MIN_FIT_SAMPLES and sample.nunique() > 1:
                try:
                    label = distribution.casefold()
                    fit_sample = sample[sample > 0] if "gamma" in label else sample
                    if "gamma" in label:
                        params = gamma.fit(fit_sample.to_numpy(), floc=0)
                        statistic, p_value = kstest(fit_sample.to_numpy(), "gamma", args=params)
                        log_likelihood = float(np.sum(gamma.logpdf(fit_sample, *params)))
                    elif "pearson" in label:
                        params = pearson3.fit(fit_sample.to_numpy())
                        statistic, p_value = kstest(fit_sample.to_numpy(), "pearson3", args=params)
                        log_likelihood = float(np.sum(pearson3.logpdf(fit_sample, *params)))
                    else:
                        params = fisk.fit(fit_sample.to_numpy())
                        statistic, p_value = kstest(fit_sample.to_numpy(), "fisk", args=params)
                        log_likelihood = float(np.sum(fisk.logpdf(fit_sample, *params)))
                    row["KS istatistiği"] = statistic
                    row["KS p-değeri"] = p_value
                    row["AIC"] = 2 * len(params) - 2 * log_likelihood
                except (ValueError, FloatingPointError, RuntimeError):
                    pass
            rows.append(row)
    return pd.DataFrame(rows)


def detect_drought_events(
    data: pd.DataFrame,
    index_column: str,
    *,
    threshold: float = -1.0,
    date_column: str = "Tarih",
) -> pd.DataFrame:
    frame = data[[date_column, index_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[index_column] = pd.to_numeric(frame[index_column], errors="coerce")
    frame = frame.dropna().sort_values(date_column)
    # Aynı aya ait yinelenen kayıtların olayı yapay biçimde uzatmasını engelle.
    frame["_ay"] = frame[date_column].dt.to_period("M")
    frame = (
        frame.groupby("_ay", as_index=False)[index_column]
        .mean()
        .assign(**{date_column: lambda table: table["_ay"].dt.to_timestamp()})
        [[date_column, index_column]]
    )
    events: list[dict[str, object]] = []
    current: list[tuple[pd.Timestamp, float]] = []

    def finish() -> None:
        if not current:
            return
        dates = [item[0] for item in current]
        values = np.asarray([item[1] for item in current], dtype=float)
        start_month = dates[0].to_period("M")
        end_month = dates[-1].to_period("M")
        duration = int(end_month.ordinal - start_month.ordinal + 1)
        start_date = start_month.to_timestamp(how="start")
        end_date = end_month.to_timestamp(how="end").normalize()
        peak_location = int(np.argmin(values))
        severity = float(np.abs(np.minimum(values, 0)).sum())
        events.append(
            {
                "İndis": index_column,
                "Olay no": len(events) + 1,
                "Başlangıç": start_date,
                "Bitiş": end_date,
                "Süre (ay)": duration,
                "Olay türü": "Tek aylık" if duration == 1 else "Çok aylık",
                "Tepe ayı": dates[peak_location].to_period("M").to_timestamp(),
                "En düşük değer": float(values.min()),
                "Şiddet": severity,
                "Ortalama yoğunluk": severity / duration,
                "Eşik": threshold,
            }
        )
        current.clear()

    previous: pd.Timestamp | None = None
    for timestamp, value in frame.itertuples(index=False, name=None):
        timestamp = pd.Timestamp(timestamp)
        consecutive = (
            previous is None
            or timestamp.to_period("M").ordinal - previous.to_period("M").ordinal == 1
        )
        if value <= threshold and consecutive:
            current.append((timestamp, float(value)))
        elif value <= threshold:
            finish()
            current.append((timestamp, float(value)))
        else:
            finish()
        previous = timestamp
    finish()
    return pd.DataFrame(events)


def _mk_components(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    score = 0.0
    for index in range(n - 1):
        score += np.sign(values[index + 1 :] - values[index]).sum()
    _, counts = np.unique(values, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18
    return score, variance


def _trend_free_prewhiten(values: np.ndarray) -> tuple[np.ndarray, float]:
    if len(values) < 4:
        return values, np.nan
    x = np.arange(len(values), dtype=float)
    slope = float(theilslopes(values, x)[0])
    detrended = values - slope * x
    autocorrelation = float(np.corrcoef(detrended[:-1], detrended[1:])[0, 1])
    if not np.isfinite(autocorrelation) or abs(autocorrelation) < 0.05:
        return values, autocorrelation
    whitened = detrended[1:] - autocorrelation * detrended[:-1]
    return whitened + slope * x[1:], autocorrelation


def mann_kendall_test(values: Iterable[float], *, prewhiten: bool = True) -> dict[str, float]:
    original = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(float)
    if len(original) < 8:
        return {
            "n": len(original), "tau": np.nan, "z": np.nan, "p": np.nan,
            "sen_slope": np.nan, "sen_low": np.nan, "sen_high": np.nan,
            "lag1": np.nan,
        }
    tested, lag1 = _trend_free_prewhiten(original) if prewhiten else (original, np.nan)
    score, variance = _mk_components(tested)
    if variance <= 0:
        z = 0.0
    elif score > 0:
        z = (score - 1) / np.sqrt(variance)
    elif score < 0:
        z = (score + 1) / np.sqrt(variance)
    else:
        z = 0.0
    denominator = len(tested) * (len(tested) - 1) / 2
    slope, _, low, high = theilslopes(original, np.arange(len(original), dtype=float))
    return {
        "n": len(original),
        "tau": float(score / denominator) if denominator else np.nan,
        "z": float(z),
        "p": float(2 * norm.sf(abs(z))),
        "sen_slope": float(slope),
        "sen_low": float(low),
        "sen_high": float(high),
        "lag1": lag1,
    }


def seasonal_mann_kendall(values: pd.Series, dates: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "date": pd.to_datetime(dates)})
    frame = frame.dropna()
    score = 0.0
    variance = 0.0
    for _, group in frame.groupby(frame["date"].dt.month):
        if len(group) >= 4:
            part_score, part_variance = _mk_components(group["value"].to_numpy(float))
            score += part_score
            variance += part_variance
    if variance <= 0:
        return {"seasonal_z": np.nan, "seasonal_p": np.nan}
    z = (score - np.sign(score)) / np.sqrt(variance) if score else 0.0
    return {"seasonal_z": float(z), "seasonal_p": float(2 * norm.sf(abs(z)))}


def pettitt_test(values: Iterable[float], dates: Iterable[object]) -> dict[str, object]:
    frame = pd.DataFrame({"value": list(values), "date": list(dates)})
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna()
    n = len(frame)
    if n < 8:
        return {"pettitt_date": pd.NaT, "pettitt_k": np.nan, "pettitt_p": np.nan}
    ranks = rankdata(frame["value"].to_numpy(float))
    time_index = np.arange(1, n + 1)
    statistic_series = 2 * np.cumsum(ranks) - time_index * (n + 1)
    location = int(np.argmax(np.abs(statistic_series)))
    statistic = float(abs(statistic_series[location]))
    p_value = min(1.0, float(2 * np.exp((-6 * statistic**2) / (n**3 + n**2))))
    return {
        "pettitt_date": frame["date"].iloc[location],
        "pettitt_k": statistic,
        "pettitt_p": p_value,
    }


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full_like(values, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return adjusted
    observed = values[valid]
    order = np.argsort(observed)
    ranked = observed[order]
    correction = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    correction = np.minimum.accumulate(correction[::-1])[::-1].clip(0, 1)
    restored = np.empty_like(correction)
    restored[order] = correction
    adjusted[valid] = restored
    return adjusted


def trend_analysis(
    data: pd.DataFrame,
    columns: Iterable[str],
    *,
    date_column: str = "Tarih",
    prewhiten: bool = True,
    include_seasonal: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        if column not in data:
            continue
        mk = mann_kendall_test(data[column], prewhiten=prewhiten)
        seasonal = (
            seasonal_mann_kendall(data[column], data[date_column])
            if include_seasonal
            else {"seasonal_z": np.nan, "seasonal_p": np.nan}
        )
        change = pettitt_test(data[column], data[date_column])
        rows.append(
            {
                "Değişken": column,
                "n": mk["n"],
                "Kendall tau": mk["tau"],
                "MK z": mk["z"],
                "MK p": mk["p"],
                "Sen eğimi / ay": mk["sen_slope"],
                "Sen %95 alt": mk["sen_low"],
                "Sen %95 üst": mk["sen_high"],
                "Gecikme-1 otokorelasyon": mk["lag1"],
                "Mevsimsel MK z": seasonal["seasonal_z"],
                "Mevsimsel MK p": seasonal["seasonal_p"],
                "Pettitt tarihi": change["pettitt_date"],
                "Pettitt K": change["pettitt_k"],
                "Pettitt p": change["pettitt_p"],
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["MK q (FDR)"] = benjamini_hochberg(result["MK p"])
        result["Anlamlı eğilim"] = result["MK q (FDR)"] < alpha
        result["Yön"] = np.where(
            result["Sen eğimi / ay"] > 0,
            "Artan",
            np.where(result["Sen eğimi / ay"] < 0, "Azalan", "Durağan"),
        )
    return result


def _seasonal_anomaly(series: pd.Series) -> pd.Series:
    dates = pd.DatetimeIndex(series.index)
    grouped = series.groupby(dates.month)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (series - mean) / std


def lagged_correlations(
    data: pd.DataFrame,
    drought_columns: Iterable[str],
    response_columns: Iterable[str],
    *,
    max_lag: int = 6,
    method: str = "Spearman",
    date_column: str = "Tarih",
    remove_seasonality: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    frame = data.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column]).sort_values(date_column).set_index(date_column)
    rows: list[dict[str, object]] = []
    for drought in drought_columns:
        if drought not in frame:
            continue
        x = pd.to_numeric(frame[drought], errors="coerce")
        x = _seasonal_anomaly(x) if remove_seasonality else x
        for response in response_columns:
            if response not in frame:
                continue
            y = pd.to_numeric(frame[response], errors="coerce")
            y = _seasonal_anomaly(y) if remove_seasonality else y
            for lag in range(max_lag + 1):
                pair = pd.concat([x.rename("x"), y.shift(-lag).rename("y")], axis=1).dropna()
                if len(pair) < 8 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
                    coefficient, p_value = np.nan, np.nan
                elif method.casefold().startswith("pearson"):
                    coefficient, p_value = pearsonr(pair["x"], pair["y"])
                else:
                    coefficient, p_value = spearmanr(pair["x"], pair["y"])
                rows.append(
                    {
                        "Kuraklık indisi": drought,
                        "Tepki değişkeni": response,
                        "Gecikme (ay)": lag,
                        "Yöntem": method,
                        "n": len(pair),
                        "Korelasyon": coefficient,
                        "p": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q (FDR)"] = benjamini_hochberg(result["p"])
        result["Anlamlı"] = result["q (FDR)"] < alpha
        result["Mutlak korelasyon"] = result["Korelasyon"].abs()
        result["En güçlü gecikme"] = False
        valid = result.dropna(subset=["Mutlak korelasyon"])
        if not valid.empty:
            best = valid.groupby(["Kuraklık indisi", "Tepki değişkeni"])["Mutlak korelasyon"].idxmax()
            result.loc[best, "En güçlü gecikme"] = True
    return result


def validation_metrics(reference: pd.Series, candidate: pd.Series) -> dict[str, float]:
    pair = pd.concat(
        [pd.to_numeric(reference, errors="coerce"), pd.to_numeric(candidate, errors="coerce")],
        axis=1,
    ).dropna()
    pair.columns = ["reference", "candidate"]
    if len(pair) < 3:
        return {key: np.nan for key in ["n", "Bias", "MAE", "RMSE", "Pearson r", "Spearman rho", "KGE"]}
    difference = pair["candidate"] - pair["reference"]
    pearson = float(pearsonr(pair["reference"], pair["candidate"])[0])
    spearman = float(spearmanr(pair["reference"], pair["candidate"])[0])
    reference_std = float(pair["reference"].std(ddof=1))
    reference_mean = float(pair["reference"].mean())
    alpha = float(pair["candidate"].std(ddof=1) / reference_std) if reference_std else np.nan
    beta = float(pair["candidate"].mean() / reference_mean) if reference_mean else np.nan
    kge = (
        1 - np.sqrt((pearson - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        if np.isfinite(alpha) and np.isfinite(beta)
        else np.nan
    )
    return {
        "n": len(pair),
        "Bias": float(difference.mean()),
        "MAE": float(difference.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(difference**2))),
        "Pearson r": pearson,
        "Spearman rho": spearman,
        "KGE": float(kge),
    }


def compare_sources(
    data: pd.DataFrame,
    reference_column: str,
    candidate_columns: Iterable[str],
) -> pd.DataFrame:
    rows = []
    if reference_column not in data:
        return pd.DataFrame()
    for candidate in candidate_columns:
        if candidate not in data or candidate == reference_column:
            continue
        rows.append(
            {"Referans": reference_column, "Karşılaştırılan": candidate, **validation_metrics(data[reference_column], data[candidate])}
        )
    return pd.DataFrame(rows)


def uncertainty_table(
    data: pd.DataFrame,
    source_columns: Iterable[str],
    *,
    date_column: str = "Tarih",
) -> pd.DataFrame:
    available = [column for column in source_columns if column in data]
    if len(available) < 2:
        return pd.DataFrame()
    values = data[available].apply(pd.to_numeric, errors="coerce")
    result = pd.DataFrame({date_column: pd.to_datetime(data[date_column], errors="coerce")})
    result["Ensemble ortalaması"] = values.mean(axis=1)
    result["Kaynaklar arası std"] = values.std(axis=1, ddof=1)
    result["En düşük"] = values.min(axis=1)
    result["En yüksek"] = values.max(axis=1)
    result["Kaynak aralığı"] = result["En yüksek"] - result["En düşük"]
    denominator = result["Ensemble ortalaması"].abs().replace(0, np.nan)
    result["Belirsizlik katsayısı (%)"] = 100 * result["Kaynaklar arası std"] / denominator
    return result


def quality_control_table(
    data: pd.DataFrame,
    *,
    date_column: str = "Tarih",
    expected_ranges: dict[str, tuple[float | None, float | None]] | None = None,
) -> pd.DataFrame:
    expected_ranges = expected_ranges or {}
    dates = pd.to_datetime(data[date_column], errors="coerce")
    valid_dates = dates.dropna().sort_values()
    if valid_dates.empty:
        expected_count = 0
    else:
        expected_count = len(pd.date_range(valid_dates.iloc[0].to_period("M").start_time, valid_dates.iloc[-1].to_period("M").start_time, freq="MS"))
    duplicate_dates = int(dates.duplicated().sum())
    rows: list[dict[str, object]] = []
    for column in data.select_dtypes(include="number").columns:
        values = pd.to_numeric(data[column], errors="coerce")
        lower, upper = expected_ranges.get(column, (None, None))
        out_of_range = pd.Series(False, index=values.index)
        if lower is not None:
            out_of_range |= values < lower
        if upper is not None:
            out_of_range |= values > upper
        rows.append(
            {
                "Değişken": column,
                "Kayıt": len(values),
                "Geçerli": int(values.notna().sum()),
                "Eksik": int(values.isna().sum()),
                "Eksik (%)": float(100 * values.isna().mean()),
                "Minimum": values.min(),
                "Maksimum": values.max(),
                "Aralık dışı": int(out_of_range.sum()),
                "Beklenen aylık kayıt": expected_count,
                "Eksik ay": max(expected_count - valid_dates.dt.to_period("M").nunique(), 0),
                "Yinelenen tarih": duplicate_dates,
            }
        )
    return pd.DataFrame(rows)


def run_remote_sensing_analysis(
    data: pd.DataFrame,
    *,
    response_columns: Iterable[str],
    config: dict[str, object],
    date_column: str = "Tarih",
) -> dict[str, pd.DataFrame]:
    """NDVI/EVI/LST serileri için bağımsız bulgu paketini üretir.

    Kuraklık indisleri seçilmese bile eğilim, mevsimsel profil, referans dönemi
    anomalileri, dönemsel değişim ve kalite kontrol sonuçları hazırlanır.
    """
    frame = data.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column]).sort_values(date_column)
    available = [column for column in response_columns if column in frame]
    if not available:
        return {}

    monthly = harmonize_monthly_data(frame, date_column=date_column)
    baseline_start = int(
        config.get(
            "anomaly_baseline_start",
            config.get("baseline_start", monthly[date_column].dt.year.min()),
        )
    )
    baseline_end = int(
        config.get(
            "anomaly_baseline_end",
            config.get("baseline_end", monthly[date_column].dt.year.max()),
        )
    )
    change_window = max(1, int(config.get("change_window_years", 3)))
    alpha = float(config.get("alpha", 0.05))

    trends = trend_analysis(
        monthly,
        available,
        date_column=date_column,
        prewhiten=bool(config.get("prewhiten", True)),
        include_seasonal=bool(config.get("seasonal_mk", True)),
        alpha=alpha,
    )

    seasonal_rows: list[dict[str, object]] = []
    anomaly = pd.DataFrame({date_column: monthly[date_column]})
    summary_rows: list[dict[str, object]] = []
    for column in available:
        values = pd.to_numeric(monthly[column], errors="coerce")
        valid = pd.DataFrame({date_column: monthly[date_column], "value": values}).dropna()
        if valid.empty:
            continue

        valid["month"] = valid[date_column].dt.month
        for month, group in valid.groupby("month"):
            seasonal_rows.append(
                {
                    "Değişken": column,
                    "Ay": int(month),
                    "Gözlem": len(group),
                    "Ortalama": float(group["value"].mean()),
                    "Medyan": float(group["value"].median()),
                    "Standart sapma": float(group["value"].std(ddof=1)),
                    "Alt %10": float(group["value"].quantile(0.10)),
                    "Üst %90": float(group["value"].quantile(0.90)),
                }
            )

        baseline_mask = (
            (valid[date_column].dt.year >= baseline_start)
            & (valid[date_column].dt.year <= baseline_end)
        )
        baseline = valid.loc[baseline_mask]
        if baseline.empty:
            baseline = valid
        climatology_mean = baseline.groupby("month")["value"].mean()
        climatology_std = baseline.groupby("month")["value"].std(ddof=1).replace(0, np.nan)
        month_numbers = monthly[date_column].dt.month
        expected = month_numbers.map(climatology_mean)
        spread = month_numbers.map(climatology_std)
        absolute_anomaly = values - expected
        standardized_anomaly = absolute_anomaly / spread
        anomaly[column] = values
        anomaly[f"{column} klimatolojisi"] = expected
        anomaly[f"{column} anomalisi"] = absolute_anomaly
        anomaly[f"{column} standart anomalisi"] = standardized_anomaly

        start_limit = valid[date_column].min() + pd.DateOffset(years=change_window)
        end_limit = valid[date_column].max() - pd.DateOffset(years=change_window)
        first_values = valid.loc[valid[date_column] < start_limit, "value"]
        last_values = valid.loc[valid[date_column] > end_limit, "value"]
        first_mean = float(first_values.mean()) if not first_values.empty else np.nan
        last_mean = float(last_values.mean()) if not last_values.empty else np.nan
        change = last_mean - first_mean if np.isfinite(first_mean) and np.isfinite(last_mean) else np.nan
        change_percent = (
            100 * change / abs(first_mean)
            if np.isfinite(change) and np.isfinite(first_mean) and first_mean != 0
            else np.nan
        )
        trend_row = trends.loc[trends["Değişken"] == column]
        slope_month = float(trend_row.iloc[0]["Sen eğimi / ay"]) if not trend_row.empty else np.nan
        peak_month = int(climatology_mean.idxmax()) if not climatology_mean.empty else None
        low_month = int(climatology_mean.idxmin()) if not climatology_mean.empty else None
        summary_rows.append(
            {
                "Değişken": column,
                "Başlangıç": valid[date_column].min(),
                "Bitiş": valid[date_column].max(),
                "Geçerli gözlem": len(valid),
                "Ortalama": float(valid["value"].mean()),
                "Medyan": float(valid["value"].median()),
                "Minimum": float(valid["value"].min()),
                "Maksimum": float(valid["value"].max()),
                "Standart sapma": float(valid["value"].std(ddof=1)),
                "Sen eğimi / yıl": slope_month * 12 if np.isfinite(slope_month) else np.nan,
                f"İlk {change_window} yıl ortalaması": first_mean,
                f"Son {change_window} yıl ortalaması": last_mean,
                "Dönemsel değişim": change,
                "Dönemsel değişim (%)": change_percent,
                "En yüksek ortalama ay": peak_month,
                "En düşük ortalama ay": low_month,
            }
        )

    expected_ranges = {
        column: ((-1, 1) if column.startswith(("NDVI", "EVI")) else (-90, 80))
        for column in available
    }
    for column in monthly.columns:
        label = str(column).casefold()
        if "geçerli piksel oranı" in label:
            expected_ranges[column] = (0, 1)
        elif "sahne sayısı" in label:
            expected_ranges[column] = (0, None)

    return {
        "Uzaktan Algılama Özeti": pd.DataFrame(summary_rows),
        "Eğilim ve Değişim": trends,
        "Mevsimsel Profil": pd.DataFrame(seasonal_rows),
        "Anomali Serisi": anomaly,
        "Kalite Kontrol": quality_control_table(
            monthly,
            date_column=date_column,
            expected_ranges=expected_ranges,
        ),
    }


def run_academic_analysis(
    data: pd.DataFrame,
    *,
    precipitation_column: str,
    pet_column: str | None,
    response_columns: Iterable[str],
    validation_columns: Iterable[str],
    config: dict[str, object],
    date_column: str = "Tarih",
) -> dict[str, pd.DataFrame]:
    scales = [int(item) for item in config.get("scales", [3, 6, 12])]
    baseline_start = int(config.get("baseline_start", 1991))
    baseline_end = int(config.get("baseline_end", 2020))
    drought_indices = list(config.get("drought_indices", ["SPI", "SPEI"]))
    results: dict[str, pd.DataFrame] = {}

    drought = pd.DataFrame({"Tarih": pd.to_datetime(data[date_column], errors="coerce")})
    drought = drought.dropna().drop_duplicates("Tarih").sort_values("Tarih")
    diagnostics = []
    if "SPI" in drought_indices:
        spi = calculate_spi_academic(
            data,
            precipitation_column,
            scales,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            distribution=str(config.get("spi_distribution", "Gamma")),
            date_column=date_column,
        )
        drought = drought.merge(spi, on="Tarih", how="outer")
        diagnostics.append(
            distribution_diagnostics(
                _monthly_series(data, precipitation_column, date_column),
                scales=scales,
                distribution=str(config.get("spi_distribution", "Gamma")),
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                index_name="SPI",
            )
        )
    if "SPEI" in drought_indices and pet_column and pet_column in data:
        spei = calculate_spei_table(
            data,
            precipitation_column,
            pet_column,
            scales,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            distribution=str(config.get("spei_distribution", "Log-logistic")),
            date_column=date_column,
        )
        drought = drought.merge(spei, on="Tarih", how="outer")
        water_balance = _monthly_series(data, precipitation_column, date_column) - _monthly_series(data, pet_column, date_column)
        diagnostics.append(
            distribution_diagnostics(
                water_balance,
                scales=scales,
                distribution=str(config.get("spei_distribution", "Log-logistic")),
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                index_name="SPEI",
            )
        )
    results["Kuraklık Serisi"] = drought.sort_values("Tarih")
    results["Dağılım Uyum"] = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()

    index_columns = [
        column
        for column in drought.columns
        if (column.startswith("SPI-") or column.startswith("SPEI-")) and not column.endswith("sınıfı")
    ]
    event_frames = [
        detect_drought_events(
            drought,
            column,
            threshold=float(config.get("event_threshold", -1.0)),
        )
        for column in index_columns
    ]
    results["Kuraklık Olayları"] = (
        pd.concat([frame for frame in event_frames if not frame.empty], ignore_index=True)
        if any(not frame.empty for frame in event_frames)
        else pd.DataFrame()
    )

    merged = data.copy()
    merged[date_column] = pd.to_datetime(merged[date_column], errors="coerce")
    merged = merged.merge(drought[["Tarih", *index_columns]], left_on=date_column, right_on="Tarih", how="outer")
    if date_column != "Tarih":
        merged = merged.drop(columns=["Tarih_y"], errors="ignore").rename(columns={"Tarih_x": date_column})
    trend_columns = [*index_columns, *[column for column in response_columns if column in merged]]
    results["Eğilim ve Değişim"] = trend_analysis(
        merged,
        trend_columns,
        date_column=date_column,
        prewhiten=bool(config.get("prewhiten", True)),
        include_seasonal=bool(config.get("seasonal_mk", True)),
        alpha=float(config.get("alpha", 0.05)),
    )
    results["Gecikmeli İlişki"] = lagged_correlations(
        merged,
        index_columns,
        response_columns,
        max_lag=int(config.get("max_lag", 6)),
        method=str(config.get("correlation_method", "Spearman")),
        date_column=date_column,
        remove_seasonality=bool(config.get("remove_seasonality", True)),
        alpha=float(config.get("alpha", 0.05)),
    )
    results["Kaynak Doğrulama"] = compare_sources(data, precipitation_column, validation_columns)
    source_columns = [precipitation_column, *[column for column in validation_columns if column in data]]
    results["Belirsizlik"] = uncertainty_table(data, source_columns, date_column=date_column)

    ranges: dict[str, tuple[float | None, float | None]] = {
        precipitation_column: (0, None),
        **{column: (0, None) for column in validation_columns if column in data},
    }
    if pet_column:
        ranges[pet_column] = (0, None)
    for column in response_columns:
        if column.startswith(("NDVI", "EVI")):
            ranges[column] = (-1, 1)
        elif column.startswith("LST"):
            ranges[column] = (-90, 80)
    for column in data.columns:
        if "geçerli piksel oranı" in column.lower():
            ranges[column] = (0, 1)
        elif "sahne sayısı" in column.lower():
            ranges[column] = (0, None)
    results["Kalite Kontrol"] = quality_control_table(data, date_column=date_column, expected_ranges=ranges)
    return results


def build_academic_report_html(
    *,
    study: dict[str, object],
    config: dict[str, object],
    results: dict[str, pd.DataFrame],
    source_note: str,
    context: dict[str, object] | None = None,
    figures: dict[str, bytes] | None = None,
) -> bytes:
    """Yöntem, bulgu, kalite ve ekleri kapsayan bağımsız analiz raporu üretir."""
    context = context or {}
    figures = figures or {}
    title = escape(str(study.get("title") or "Zetriklim İklim Analizi"))
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    event_table = results.get("Kuraklık Olayları", pd.DataFrame())
    trend_table = results.get("Eğilim ve Değişim", pd.DataFrame())
    lag_table = results.get("Gecikmeli İlişki", pd.DataFrame())
    validation_table = results.get("Kaynak Doğrulama", pd.DataFrame())
    quality_table = results.get("Kalite Kontrol", pd.DataFrame())
    significant_trends = int(
        trend_table.get("Anlamlı eğilim", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
    ) if not trend_table.empty else 0
    significant_lags = int(
        lag_table.get("Anlamlı", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
    ) if not lag_table.empty else 0
    max_duration = int(pd.to_numeric(event_table.get("Süre (ay)"), errors="coerce").max()) if not event_table.empty else 0
    worst_index = "—"
    if not event_table.empty and "En düşük değer" in event_table:
        minimums = pd.to_numeric(event_table["En düşük değer"], errors="coerce")
        if minimums.notna().any():
            row = event_table.loc[minimums.idxmin()]
            worst_index = f"{row.get('İndis', '—')} ({float(minimums.min()):.2f})"

    context_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in context.items()
        if value not in {None, ""}
    )
    parameter_labels = {
        "focus": "Ana analiz odağı", "drought_indices": "Kuraklık indisleri",
        "response_indices": "Tepki değişkenleri", "scales": "Zaman ölçekleri (ay)",
        "baseline_start": "Referans başlangıcı", "baseline_end": "Referans bitişi",
        "spi_distribution": "SPI dağılımı", "spei_distribution": "SPEI dağılımı",
        "event_threshold": "Olay eşiği", "prewhiten": "Otokorelasyon düzeltmesi",
        "seasonal_mk": "Mevsimsel MK", "alpha": "Anlamlılık düzeyi",
        "max_lag": "Azami gecikme (ay)", "correlation_method": "İlişki yöntemi",
        "remove_seasonality": "Mevsimsellik giderimi", "land_cover_labels": "Arazi örtüsü sınıfları",
    }
    parameter_rows = "".join(
        f"<tr><th>{escape(parameter_labels.get(key, str(key)))}</th>"
        f"<td>{escape(', '.join(map(str, value)) if isinstance(value, (list, tuple, set)) else str(value))}</td></tr>"
        for key, value in config.items()
        if key in parameter_labels
    )

    figure_blocks = []
    for caption, content in figures.items():
        if not content:
            continue
        encoded = base64.b64encode(content).decode("ascii")
        figure_blocks.append(
            f"<figure><img src='data:image/png;base64,{encoded}' alt='{escape(caption)}'>"
            f"<figcaption>{escape(caption)}</figcaption></figure>"
        )

    sections = []
    for name, table in results.items():
        if table is None or table.empty:
            continue
        preview = table.head(500).copy()
        sections.append(
            f"<section class='appendix'><h3>{escape(name)}</h3>"
            f"<p>Toplam {len(table):,} kayıt; bu ekte ilk {len(preview):,} kayıt gösterilmektedir. "
            "Eksiksiz tablo proje paketindeki CSV/Excel dosyasındadır.</p>"
            f"{preview.to_html(index=False, border=0, classes='dataframe', na_rep='—')}</section>"
        )
    methodology = (
        f"SPI: {escape(str(config.get('spi_distribution', 'Gamma')))} dağılımı ve sıfır olasılığı düzeltmesi; "
        f"SPEI: {escape(str(config.get('spei_distribution', 'Log-logistic')))} dağılımı; "
        f"referans dönemi {config.get('baseline_start', 1991)}–{config.get('baseline_end', 2020)}; "
        f"ölçekler {escape(', '.join(map(str, config.get('scales', []))))} ay. "
        "Eğilimler Mann–Kendall, trend-free prewhitening, Sen eğimi ve Pettitt testiyle; "
        "çoklu karşılaştırmalar Benjamini–Hochberg FDR düzeltmesiyle değerlendirilmiştir."
    )
    quality_note = (
        f"Kalite kontrolü {len(quality_table):,} değişken/ölçüt satırı üretmiştir. "
        "Eksik ay, yinelenen tarih, fiziksel aralık ve uydu geçerli piksel oranları ayrı ayrı raporlanmıştır."
        if not quality_table.empty
        else "Kalite kontrol tablosu üretilememiştir; sonuçlar yorumlanmadan önce veri bütünlüğü ayrıca doğrulanmalıdır."
    )
    html = f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
@page{{size:A4;margin:18mm}} *{{box-sizing:border-box}}
body{{font-family:Arial,sans-serif;color:#173f4b;background:#eef4f2;margin:0;padding:2rem;line-height:1.58}}
main{{max-width:1180px;margin:auto;background:white;padding:2.4rem;border-radius:14px;box-shadow:0 10px 32px #06344718}}
h1,h2,h3{{color:#075b68}} h1{{border-bottom:4px solid #00a6a6;padding-bottom:.7rem;margin-bottom:.4rem}}
h2{{border-bottom:1px solid #b8d8d3;padding-bottom:.35rem;margin-top:2.4rem}}
.meta{{background:#e8f7f4;padding:1rem;border-left:4px solid #00a6a6;border-radius:8px}}
.toc{{columns:2;background:#f6faf9;padding:1rem 1.4rem;border:1px solid #d8e8e5;border-radius:8px}}
.toc a{{color:#075b68;text-decoration:none}} .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem}}
.card{{background:#f1f8f6;border-top:4px solid #00a6a6;padding:.8rem;border-radius:6px}}
.card b{{display:block;font-size:1.35rem;color:#063447}} .method{{background:#f8faf9;padding:1rem;border-left:3px solid #607d8b}}
table{{border-collapse:collapse;width:100%;font-size:.82rem;display:block;overflow-x:auto}}
th{{background:#063447;color:white;position:sticky;top:0}} th,td{{padding:.45rem;border:1px solid #d8e8e5}}
tr:nth-child(even){{background:#f2faf8}} section{{margin-top:2rem}} .warning{{background:#fff4df;padding:1rem;border-left:4px solid #ffad33}}
.caution{{background:#fdecec;padding:1rem;border-left:4px solid #c62828}} figure{{margin:1.5rem 0;break-inside:avoid}}
figure img{{max-width:100%;height:auto;border:1px solid #b0bec5}} figcaption{{font-size:.82rem;color:#546e7a;text-align:center}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}} .kv th{{width:35%;text-align:left}}
footer{{margin-top:3rem;border-top:1px solid #cfd8dc;padding-top:1rem;font-size:.78rem;color:#607d8b}}
@media(max-width:800px){{.cards,.two-col{{grid-template-columns:1fr}}.toc{{columns:1}}body{{padding:.5rem}}main{{padding:1rem}}}}
@media print{{body{{background:white;padding:0}}main{{box-shadow:none;max-width:none;padding:0}}.appendix{{break-before:page}}}}
</style></head><body><main>
<h1>{title}</h1><p><b>Zetriklim bilimsel analiz raporu</b></p>
<div class="meta"><b>Oluşturulma:</b> {created}<br><b>Veri kaynağı:</b> {escape(source_note)}<br>
<b>Rapor kapsamı:</b> veri kökeni, yöntem, kalite kontrol, kuraklık olayları, eğilim, gecikmeli ilişki, doğrulama ve belirsizlik.</div>
<h2 id="ozet">Yönetici özeti</h2>
<p>Bu çalışma, tanımlanan coğrafi alan için çok ölçekli kuraklık indislerini ve seçilmiş ekosistem tepki değişkenlerini yeniden üretilebilir bir iş akışında değerlendirmiştir. Toplam <b>{len(event_table):,}</b> kuraklık olayı belirlenmiş; en uzun olay <b>{max_duration}</b> ay sürmüş ve en düşük indis değeri <b>{escape(worst_index)}</b> olarak kaydedilmiştir. Eğilim analizinde <b>{significant_trends}</b>, gecikmeli ilişki analizinde <b>{significant_lags}</b> FDR-düzeltilmiş anlamlı sonuç bulunmuştur.</p>
<div class="cards"><div class="card"><span>Kuraklık olayı</span><b>{len(event_table):,}</b></div><div class="card"><span>En uzun süre</span><b>{max_duration} ay</b></div><div class="card"><span>Anlamlı eğilim</span><b>{significant_trends}</b></div><div class="card"><span>Anlamlı gecikme</span><b>{significant_lags}</b></div></div>
<h2>İçindekiler</h2><ol class="toc"><li><a href="#veri">Çalışma alanı ve veri</a></li><li><a href="#yontem">Yöntem</a></li><li><a href="#kalite">Kalite güvencesi</a></li><li><a href="#bulgular">Bulguların yorumu</a></li><li><a href="#sinir">Sınırlılıklar</a></li><li><a href="#ekler">Sonuç tabloları</a></li><li><a href="#kaynaklar">Kaynaklar</a></li></ol>
<h2 id="veri">1. Çalışma alanı, dönem ve veri kökeni</h2><div class="two-col"><table class="kv">{context_rows or '<tr><th>Bağlam</th><td>Metadata dosyasına bakınız.</td></tr>'}</table><table class="kv"><tr><th>Kaynak zinciri</th><td>{escape(source_note)}</td></tr><tr><th>Zamansal işlem</th><td>Aylık; yağış/PET toplamı, sıcaklık ve uydu indisleri ortalama/kompozit</td></tr><tr><th>Mekânsal işlem</th><td>Çalışma alanı/havza ortalaması ve sınırına kırpılmış raster</td></tr></table></div>
{''.join(figure_blocks)}
<h2 id="yontem">2. Yöntem</h2><div class="method"><p>{methodology}</p></div>
<h3>2.1 Kuraklık indisleri</h3><p>SPI yağış olasılık dağılımını, SPEI ise P−PET iklimsel su dengesini standart normal uzaya dönüştürür. Hesaplar her takvim ayı ve zaman ölçeği için ayrı yürütülür; referans dönem dışındaki aylar aynı uyum parametreleriyle değerlendirilir.</p>
<h3>2.2 Kuraklık olay tanımı</h3><p>İndisin seçilen eşiğe eşit veya daha düşük olduğu ardışık takvim ayları tek olay kabul edilir. Başlangıç olayın ilk ayının ilk günü, bitiş son ayın son günüdür. “Olay türü” alanı bir aylık ve çok aylık olayları açık metinle ayırır. Şiddet olay aylarındaki negatif indis büyüklüklerinin toplamı, yoğunluk şiddetin süreye oranıdır.</p>
<h3>2.3 Eğilim ve değişim</h3><p>Mann–Kendall testi monoton eğilimi, Sen eğimi değişim büyüklüğünü, Pettitt testi olası kırılma tarihini değerlendirir. Seçildiğinde trend-free prewhitening serisel bağımlılığı azaltır; mevsimsel test takvim aylarını ayrı tabakalarda karşılaştırır.</p>
<h3>2.4 Gecikmeli ilişki, doğrulama ve belirsizlik</h3><p>Kuraklık–ekosistem ilişkileri seçilen korelasyon yöntemiyle 0–{escape(str(config.get('max_lag', 6)))} ay arasında sınanır. p-değerleri Benjamini–Hochberg FDR ile düzeltilir. Kaynak karşılaştırmasında Bias, MAE, RMSE, Pearson r, Spearman rho ve KGE; belirsizlikte kaynaklar arası standart sapma ve göreli yayılım raporlanır.</p>
<h3>2.5 Tam yeniden üretim parametreleri</h3><table class="kv">{parameter_rows}</table>
<h2 id="kalite">3. Kalite güvencesi</h2><p>{escape(quality_note)}</p><div class="warning"><b>Eksik değer politikası:</b> Kaynakta bulunmayan gözlemler uydurulmaz; “Veri yok”/boş hücre olarak korunur ve kalite kontrol tablosunda sayılır. Uydu arşivi öncesindeki boşluklar iklim serisi eksikliğiyle karıştırılmaz.</div>
<h2 id="bulgular">4. Bulguların bilimsel yorumu</h2><p>Olay kataloğunda başlangıç ve bitiş aynı ay içinde görünüyorsa bu, hatalı tarih değil, <b>tek aylık olay</b> anlamına gelir; bitiş ayın son günü ve “Olay türü” alanı “Tek aylık” olarak gösterilir. Çok aylık olaylarda süre başlangıç ve bitiş ayları arasındaki kapsayıcı takvim ayı farkıdır.</p>
<p>Eğilim veya korelasyonun istatistiksel anlamlı olması nedensellik kanıtı değildir. Etki büyüklüğü, güven aralığı, örnek sayısı, veri kalitesi, arazi örtüsü ve bağımsız istasyon karşılaştırması birlikte değerlendirilmelidir.</p>
<h2 id="sinir">5. Sınırlılıklar ve uygun kullanım</h2><div class="caution"><b>Bilimsel karar notu:</b> Bu rapor otomatik ve yeniden üretilebilir bir analiz kaydıdır; tek başına afet ilanı, ürün kaybı tahmini veya nedensel etki kanıtı değildir. Uydu bulut maskeleri, ürün çözünürlük farkları, yeniden analiz model belirsizliği, sabit arazi örtüsü yılı ve istasyon temsil hatası sonuçlara yansıyabilir.</div>
<h2 id="ekler">6. Sonuç tabloları ve ekler</h2>{''.join(sections)}
<h2 id="kaynaklar">7. Temel kaynaklar</h2><ul>
<li>WMO (2012), <a href="https://library.wmo.int/idurl/4/39629">Standardized Precipitation Index User Guide</a>, WMO-No. 1090.</li>
<li>Vicente-Serrano, Beguería ve López-Moreno (2010), <a href="https://doi.org/10.1175/2009JCLI2909.1">A Multiscalar Drought Index Sensitive to Global Warming</a>.</li>
<li>Funk ve diğerleri (2015), <a href="https://doi.org/10.1038/sdata.2015.66">The climate hazards infrared precipitation with stations (CHIRPS)</a>.</li>
<li>Muñoz-Sabater ve diğerleri (2021), <a href="https://doi.org/10.5194/essd-13-4349-2021">ERA5-Land: a state-of-the-art global reanalysis dataset for land applications</a>.</li>
<li>Mann (1945), <i>Nonparametric Tests Against Trend</i>; Kendall (1975), <i>Rank Correlation Methods</i>.</li>
<li>Sen (1968), <i>Estimates of the Regression Coefficient Based on Kendall's Tau</i>.</li>
<li>Benjamini ve Hochberg (1995), <i>Controlling the False Discovery Rate</i>.</li>
</ul><footer>Bu rapor Zetriklim tarafından aynı proje paketindeki metadata, CSV, Excel, GeoTIFF ve harita çıktılarıyla birlikte üretilmiştir. Tam denetlenebilirlik için zetriklim-metadata.json ve kalite-kontrol.csv dosyalarını raporla birlikte saklayınız.</footer></main></body></html>"""
    return html.encode("utf-8")

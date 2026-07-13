"""Lisansüstü düzey için yeniden üretilebilir iklim ve kuraklık analizleri.

Bu modül kullanıcı arayüzünden bağımsızdır. Böylece aynı yöntemler Streamlit,
notebook ve test ortamlarında aynı girdilerle aynı sonucu üretir.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
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
    events: list[dict[str, object]] = []
    current: list[tuple[pd.Timestamp, float]] = []

    def finish() -> None:
        if not current:
            return
        dates = [item[0] for item in current]
        values = np.asarray([item[1] for item in current], dtype=float)
        severity = float(-values.sum())
        events.append(
            {
                "İndis": index_column,
                "Olay no": len(events) + 1,
                "Başlangıç": dates[0],
                "Bitiş": dates[-1],
                "Süre (ay)": len(values),
                "En düşük değer": float(values.min()),
                "Şiddet": severity,
                "Ortalama yoğunluk": severity / len(values),
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


def run_academic_analysis(
    data: pd.DataFrame,
    *,
    precipitation_column: str | None,
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
    response_columns = list(response_columns)
    validation_columns = list(validation_columns)
    results: dict[str, pd.DataFrame] = {}
    warnings: list[dict[str, str]] = []

    dates = pd.to_datetime(data[date_column], errors="coerce")
    dates = dates.dt.to_period("M").dt.to_timestamp()
    drought = pd.DataFrame({"Tarih": dates})
    drought = drought.dropna().drop_duplicates("Tarih").sort_values("Tarih")
    diagnostics = []
    if "SPI" in drought_indices:
        if precipitation_column and precipitation_column in data:
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
        else:
            warnings.append({"Bileşen": "SPI", "Durum": "Yağış sütunu bulunamadığı için hesaplanmadı."})
    if "SPEI" in drought_indices:
        if precipitation_column and precipitation_column in data and pet_column and pet_column in data:
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
        else:
            warnings.append({"Bileşen": "SPEI", "Durum": "Yağış veya PET sütunu bulunamadığı için hesaplanmadı."})
    available_responses = list(response_columns)
    for requested in config.get("response_indices", []):
        if not any(
            column == requested or column.startswith(f"{requested}|")
            for column in available_responses
        ):
            warnings.append(
                {
                    "Bileşen": str(requested),
                    "Durum": "Seçilen veri kaynağında geçerli aylık seri bulunamadığı için hesaplanmadı.",
                }
            )
    results["Kuraklık Serisi"] = drought.sort_values("Tarih")
    results["Dağılım Uyum"] = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    results["Analiz Uyarıları"] = pd.DataFrame(warnings)

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
    merged[date_column] = merged[date_column].dt.to_period("M").dt.to_timestamp()
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
    if precipitation_column and precipitation_column in data:
        results["Kaynak Doğrulama"] = compare_sources(data, precipitation_column, validation_columns)
        source_columns = [precipitation_column, *[column for column in validation_columns if column in data]]
        results["Belirsizlik"] = uncertainty_table(data, source_columns, date_column=date_column)
    else:
        results["Kaynak Doğrulama"] = pd.DataFrame()
        results["Belirsizlik"] = pd.DataFrame()

    ranges: dict[str, tuple[float | None, float | None]] = {
        column: (0, None) for column in validation_columns if column in data
    }
    if precipitation_column and precipitation_column in data:
        ranges[precipitation_column] = (0, None)
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
    results["Birleşik Analiz Serisi"] = prepare_academic_series_export(
        data,
        results["Kuraklık Serisi"],
        date_column=date_column,
    )
    return results


def prepare_academic_series_export(
    source_data: pd.DataFrame,
    drought_series: pd.DataFrame | None,
    *,
    date_column: str = "Tarih",
) -> pd.DataFrame:
    """Kaynak değişkenleri ile hesaplanan indisleri tek, yinelenmeyen aylık seride birleştirir."""
    if date_column not in source_data:
        raise ValueError(f"Tarih sütunu bulunamadı: {date_column}")
    monthly = source_data.copy()
    monthly[date_column] = pd.to_datetime(monthly[date_column], errors="coerce")
    monthly = monthly.dropna(subset=[date_column])
    monthly[date_column] = monthly[date_column].dt.to_period("M").dt.to_timestamp()

    aggregations: dict[str, str] = {}
    for column in monthly.columns:
        if column == date_column:
            continue
        numeric = pd.to_numeric(monthly[column], errors="coerce")
        if numeric.notna().any():
            monthly[column] = numeric
            aggregations[column] = "mean"
        else:
            aggregations[column] = "first"
    monthly = monthly.groupby(date_column, as_index=False).agg(aggregations).sort_values(date_column)

    if drought_series is not None and not drought_series.empty:
        calculated = drought_series.copy()
        calculated["Tarih"] = pd.to_datetime(calculated["Tarih"], errors="coerce")
        calculated = calculated.dropna(subset=["Tarih"])
        calculated["Tarih"] = calculated["Tarih"].dt.to_period("M").dt.to_timestamp()
        calculated = calculated.drop_duplicates("Tarih", keep="last").sort_values("Tarih")
        overlap = [column for column in calculated.columns if column in monthly.columns and column != "Tarih"]
        monthly = monthly.drop(columns=overlap, errors="ignore")
        monthly = monthly.merge(calculated, left_on=date_column, right_on="Tarih", how="outer")
        if date_column != "Tarih":
            monthly = monthly.drop(columns=[date_column], errors="ignore")
    elif date_column != "Tarih":
        monthly = monthly.rename(columns={date_column: "Tarih"})

    monthly["Tarih"] = pd.to_datetime(monthly["Tarih"], errors="coerce")
    monthly = monthly.dropna(subset=["Tarih"]).drop_duplicates("Tarih").sort_values("Tarih")
    monthly.insert(1, "Yıl", monthly["Tarih"].dt.year)
    monthly.insert(2, "Ay", monthly["Tarih"].dt.month)
    return monthly.reset_index(drop=True)


def build_academic_report_html(
    *,
    study: dict[str, object],
    config: dict[str, object],
    results: dict[str, pd.DataFrame],
    source_note: str,
) -> bytes:
    """Yöntem, bulgular ve sınırlılıkları taşıyan bağımsız HTML raporu üretir."""
    title = escape(str(study.get("title") or "Zetriklim Akademik Kuraklık Araştırması"))
    question = escape(str(study.get("question") or "Tanımlanmadı"))
    hypotheses = escape(str(study.get("hypotheses") or "Tanımlanmadı")).replace("\n", "<br>")
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = []
    for name, table in results.items():
        if table is None or table.empty:
            continue
        preview = table.head(250).copy()
        sections.append(
            f"<section><h2>{escape(name)}</h2>"
            f"<p>Toplam {len(table):,} kayıt; raporda ilk {len(preview):,} kayıt gösterilmektedir.</p>"
            f"{preview.to_html(index=False, border=0, classes='dataframe', na_rep='—')}</section>"
        )
    index_methods = []
    drought_indices = list(config.get("drought_indices", []))
    if "SPI" in drought_indices:
        index_methods.append(
            f"SPI: {escape(str(config.get('spi_distribution', 'Gamma')))} dağılımı ve sıfır olasılığı düzeltmesi"
        )
    if "SPEI" in drought_indices:
        index_methods.append(
            f"SPEI: {escape(str(config.get('spei_distribution', 'Log-logistic')))} dağılımı"
        )
    response_indices = list(config.get("response_indices", []))
    if response_indices:
        index_methods.append(
            "Ekosistem/yüzey değişkenleri: " + escape(", ".join(map(str, response_indices)))
        )
    methodology = (
        ("; ".join(index_methods) + ". " if index_methods else "")
        + f"Referans dönemi {config.get('baseline_start', 1991)}–{config.get('baseline_end', 2020)}; "
        f"ölçekler {escape(', '.join(map(str, config.get('scales', [])))) or 'uygulanmadı'} ay. "
        "Seçili bütün değişkenlerin eğilimleri Mann–Kendall, trend-free prewhitening, Sen eğimi ve Pettitt testiyle; "
        "çoklu karşılaştırmalar Benjamini–Hochberg FDR düzeltmesiyle değerlendirilmiştir."
    )
    html = f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{font-family:Arial,sans-serif;color:#173f4b;background:#f6fbf9;margin:0;padding:2rem;line-height:1.55}}
main{{max-width:1180px;margin:auto;background:white;padding:2rem;border-radius:20px;box-shadow:0 10px 32px #06344718}}
h1,h2{{color:#075b68}} h1{{border-bottom:4px solid #00a6a6;padding-bottom:.6rem}}
.meta{{background:#e8f7f4;padding:1rem;border-left:4px solid #00a6a6;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:.82rem;display:block;overflow-x:auto}}
th{{background:#063447;color:white;position:sticky;top:0}} th,td{{padding:.45rem;border:1px solid #d8e8e5}}
tr:nth-child(even){{background:#f2faf8}} section{{margin-top:2rem}} .warning{{background:#fff4df;padding:1rem;border-left:4px solid #ffad33}}
</style></head><body><main>
<h1>{title}</h1><div class="meta"><b>Oluşturulma:</b> {created}<br><b>Veri kaynağı:</b> {escape(source_note)}</div>
<h2>Araştırma tasarımı</h2><p><b>Soru:</b> {question}</p><p><b>Hipotezler:</b><br>{hypotheses}</p>
<h2>Yöntem</h2><p>{methodology}</p>
<div class="warning"><b>Bilimsel kullanım notu:</b> Bu otomatik rapor yöntem ve sonuç tablolarını yeniden üretilebilir biçimde derler. Bulgular; alan bilgisi, veri kalitesi, istasyon doğrulaması ve danışman değerlendirmesiyle yorumlanmalıdır.</div>
{''.join(sections)}
<h2>Temel kaynaklar</h2><ul>
<li>WMO (2012), <a href="https://library.wmo.int/idurl/4/39629">Standardized Precipitation Index User Guide</a>, WMO-No. 1090.</li>
<li>Vicente-Serrano, Beguería ve López-Moreno (2010), <a href="https://doi.org/10.1175/2009JCLI2909.1">A Multiscalar Drought Index Sensitive to Global Warming</a>.</li>
<li>Funk ve diğerleri (2015), <a href="https://doi.org/10.1038/sdata.2015.66">The climate hazards infrared precipitation with stations (CHIRPS)</a>.</li>
<li>Muñoz-Sabater ve diğerleri (2021), <a href="https://doi.org/10.5194/essd-13-4349-2021">ERA5-Land: a state-of-the-art global reanalysis dataset for land applications</a>.</li>
</ul></main></body></html>"""
    return html.encode("utf-8")

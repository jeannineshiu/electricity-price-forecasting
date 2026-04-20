from pathlib import Path

import pandas as pd
from prefect import flow, task, get_run_logger

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# German public holidays (fixed-date only; sufficient for calendar signal)
# Format: (month, day)
DE_PUBLIC_HOLIDAYS = {
    (1,  1),   # New Year's Day
    (5,  1),   # Labour Day
    (10, 3),   # German Unity Day
    (12, 25),  # Christmas Day
    (12, 26),  # Boxing Day
}


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="load-parquet")
def load_parquet(path: str) -> pd.DataFrame:
    """Load the cleaned raw Parquet produced by ingest_flow."""
    logger = get_run_logger()

    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


@task(name="build-lag-features")
def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features based on past prices.

    Lags:
        lag_1h   — 1 hour ago       (short-term inertia)
        lag_24h  — 24 hours ago     (same hour yesterday)
        lag_168h — 168 hours ago    (same hour last week)
    """
    logger = get_run_logger()

    df = df.copy()
    df["lag_1h"]   = df["price_eur_mwh"].shift(1)
    df["lag_24h"]  = df["price_eur_mwh"].shift(24)
    df["lag_168h"] = df["price_eur_mwh"].shift(168)

    logger.info("Lag features built: lag_1h, lag_24h, lag_168h")
    return df


@task(name="build-rolling-features")
def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean and std over the past 24h and 168h windows.

    Uses min_periods=1 so early rows are not dropped entirely,
    but NaNs introduced by lags will still be dropped later.
    """
    logger = get_run_logger()

    df = df.copy()
    price = df["price_eur_mwh"]

    df["rolling_mean_24h"]  = price.shift(1).rolling(24,  min_periods=1).mean()
    df["rolling_std_24h"]   = price.shift(1).rolling(24,  min_periods=2).std()
    df["rolling_mean_168h"] = price.shift(1).rolling(168, min_periods=1).mean()
    df["rolling_std_168h"]  = price.shift(1).rolling(168, min_periods=2).std()

    logger.info(
        "Rolling features built: "
        "rolling_mean_24h, rolling_std_24h, "
        "rolling_mean_168h, rolling_std_168h"
    )
    return df


@task(name="build-calendar-features")
def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based calendar features from the UTC timestamp.

    Features:
        hour          — 0–23
        day_of_week   — 0=Monday … 6=Sunday
        month         — 1–12
        is_weekend    — 1 if Saturday or Sunday, else 0
        is_holiday    — 1 if fixed German public holiday, else 0
    """
    logger = get_run_logger()

    df = df.copy()

    # Convert to Berlin local time for calendar accuracy
    local_ts = df["timestamp"].dt.tz_convert("Europe/Berlin")

    df["hour"]        = local_ts.dt.hour
    df["day_of_week"] = local_ts.dt.dayofweek
    df["month"]       = local_ts.dt.month
    df["is_weekend"]  = (local_ts.dt.dayofweek >= 5).astype(int)
    df["is_holiday"]  = local_ts.apply(
        lambda ts: int((ts.month, ts.day) in DE_PUBLIC_HOLIDAYS)
    )

    logger.info(
        "Calendar features built: "
        "hour, day_of_week, month, is_weekend, is_holiday"
    )
    return df


@task(name="drop-nulls-and-save")
def drop_nulls_and_save(df: pd.DataFrame) -> Path:
    """
    Drop rows with any NaN (introduced by the 168-hour lag at the start),
    run a final feature sanity check, then save to Parquet.

    Output: data/processed/features_2020_2024.parquet
    """
    logger = get_run_logger()

    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df)

    logger.info(
        f"Dropped {rows_dropped} rows with NaN "
        f"(expected ~168 from lag_168h warm-up)"
    )

    # Sanity check: no remaining nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        raise ValueError(f"Unexpected nulls after dropna:\n{null_counts[null_counts > 0]}")

    # Sanity check: expected feature columns present
    expected_cols = {
        "timestamp", "price_eur_mwh",
        "lag_1h", "lag_24h", "lag_168h",
        "rolling_mean_24h", "rolling_std_24h",
        "rolling_mean_168h", "rolling_std_168h",
        "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    }
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    path = PROCESSED_DIR / "features_2020_2024.parquet"
    df.to_parquet(path, index=False)

    logger.info(f"Saved {len(df)} rows, {len(df.columns)} columns → {path}")
    return path


# ── Flow ───────────────────────────────────────────────────────────────────────

@flow(name="build-features", log_prints=True)
def features_flow(
    raw_path: str = "data/raw/de_prices_2020_2024.parquet"
):
    """
    Feature engineering flow.

    Steps:
        1. Load raw Parquet from ingest_flow
        2. Build lag features
        3. Build rolling statistics
        4. Build calendar features
        5. Drop NaN warm-up rows and save
    """
    df = load_parquet(raw_path)
    df = build_lag_features(df)
    df = build_rolling_features(df)
    df = build_calendar_features(df)
    path = drop_nulls_and_save(df)

    print(f"Feature engineering complete — saved to {path}")
    return path


if __name__ == "__main__":
    features_flow()
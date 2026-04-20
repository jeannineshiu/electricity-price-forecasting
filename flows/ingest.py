import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from prefect import flow, task

load_dotenv()

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

CSV_FALLBACK = "data/raw/Germany.csv"


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="fetch-prices")
def fetch_prices(start: str, end: str) -> pd.DataFrame:
    """
    Fetch day-ahead electricity prices for Germany (DE_LU bidding zone)
    from the ENTSO-E Transparency Platform.

    Falls back to local Ember CSV if ENTSOE_API_KEY is not set.

    Args:
        start: Start date string, e.g. "2020-01-01"
        end:   End date string,   e.g. "2024-12-31"
    """
    api_key = os.getenv("ENTSOE_API_KEY")

    if not api_key:
        print("WARNING: ENTSOE_API_KEY not set — falling back to CSV")
        return _load_from_csv(CSV_FALLBACK, start, end)

    print(f"Fetching ENTSO-E prices from {start} to {end}...")
    try:
        client = EntsoePandasClient(api_key=api_key)
        start_ts = pd.Timestamp(start, tz="Europe/Berlin")
        end_ts   = pd.Timestamp(end,   tz="Europe/Berlin") + pd.Timedelta(days=1)

        series = client.query_day_ahead_prices("DE_LU", start=start_ts, end=end_ts)
        df = (
            series
            .rename("price_eur_mwh")
            .reset_index()
            .rename(columns={"index": "timestamp"})
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"ENTSO-E: fetched {len(df)} rows ({df['timestamp'].min()} → {df['timestamp'].max()})")
        return df

    except Exception as e:
        print(f"WARNING: ENTSO-E API failed ({e}) — falling back to CSV")
        return _load_from_csv(CSV_FALLBACK, start, end)


def _load_from_csv(path: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"CSV loaded — {len(df)} rows")

    df = df.rename(columns={
        "Datetime (UTC)":   "timestamp",
        "Price (EUR/MWhe)": "price_eur_mwh",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp", "price_eur_mwh"]].copy()

    mask = (
        (df["timestamp"] >= pd.Timestamp(start, tz="UTC")) &
        (df["timestamp"] <= pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1))
    )
    df = df[mask].sort_values("timestamp").reset_index(drop=True)
    print(f"Filtered to {start}–{end} — {len(df)} rows remaining")
    return df


@task(name="validate-prices")
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run basic data quality checks on the price series.

    Checks:
        1. Expected row count (8760 or 8784 hours per year x 5 years)
        2. Missing values  → forward-fill
        3. Duplicate timestamps → drop
        4. Out-of-range prices  → warn  (-500 to 4000 EUR/MWh is the European market range)
    """
    # ── 1. Row count sanity check ──────────────────────────────────────────────
    # 5 years: 2020 (leap) + 2021 + 2022 + 2023 + 2024 (leap) = 43848 hours
    expected_min = 43_800
    expected_max = 43_900
    if not (expected_min <= len(df) <= expected_max):
        print(
            f"WARNING: Unexpected row count: {len(df)} "
            f"(expected {expected_min}–{expected_max})"
        )
    else:
        print(f"Row count OK: {len(df)}")

    # ── 2. Duplicate timestamps ────────────────────────────────────────────────
    n_dupes = df.duplicated(subset="timestamp").sum()
    if n_dupes > 0:
        print(f"WARNING: {n_dupes} duplicate timestamps found — dropping")
        df = df.drop_duplicates(subset="timestamp")

    # ── 3. Missing values ──────────────────────────────────────────────────────
    null_count = df["price_eur_mwh"].isnull().sum()
    if null_count > 0:
        print(f"WARNING: {null_count} null values found — forward-filling")
        df["price_eur_mwh"] = df["price_eur_mwh"].ffill()
    else:
        print("No null values found")

    # ── 4. Out-of-range prices ─────────────────────────────────────────────────
    out_of_range = ((df["price_eur_mwh"] < -500) | (df["price_eur_mwh"] > 4000)).sum()
    if out_of_range > 0:
        print(f"WARNING: {out_of_range} prices outside [-500, 4000] EUR/MWh detected")
    else:
        print("All prices within expected range")

    print(
        f"Validation complete — "
        f"min={df['price_eur_mwh'].min():.2f}, "
        f"max={df['price_eur_mwh'].max():.2f}, "
        f"mean={df['price_eur_mwh'].mean():.2f} EUR/MWh"
    )
    return df


@task(name="save-parquet")
def save_parquet(df: pd.DataFrame) -> Path:
    """
    Save the cleaned DataFrame to Parquet under data/raw/.

    Output filename: de_prices_2020_2024.parquet
    """
    path = RAW_DIR / "de_prices_2020_2024.parquet"
    df.to_parquet(path, index=False)

    print(f"Saved {len(df)} rows → {path}")
    return path


# ── Flow ───────────────────────────────────────────────────────────────────────

@flow(name="ingest-electricity-prices", log_prints=True)
def ingest_flow(
    start: str = "2020-01-01",
    end:   str = "2024-12-31",
):
    """
    Production ingestion flow using ENTSO-E API.
    Falls back to CSV if API key is not available.
    """
    prices = fetch_prices(start, end)
    df     = validate(prices)
    path   = save_parquet(df)

    print(f"Ingestion complete — {len(df)} rows saved to {path}")
    return path


if __name__ == "__main__":
    ingest_flow()

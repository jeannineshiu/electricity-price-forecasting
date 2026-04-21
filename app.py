import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from simulation.consumption import PROFILES, build_hourly_consumption
from simulation.tariff import compute_bill_summary, dynamic_tariff_cost, fixed_tariff_cost

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Electricity Bill Simulator",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Electricity Bill Simulator")
st.caption("German day-ahead market · LightGBM price forecast · Fixed vs dynamic tariff comparison")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    month       = st.selectbox(
        "Month to simulate",
        ["2026-02", "2026-01", "2024-03", "2024-06", "2024-09", "2024-12", "2023-01", "2023-07"],
        index=0,
        help="2026-01/02 uses live ENTSO-E data. 2020–2024 uses historical dataset."
    )
    profile_key = st.selectbox("User profile", list(PROFILES.keys()))
    monthly_kwh = st.slider(
        "Monthly consumption (kWh)",
        min_value=50, max_value=5000, value=300, step=50,
        help="Typical German household: 200–400 kWh/month. Industrial: 1000–5000 kWh."
    )

    st.divider()
    st.caption("Fixed tariff: €0.30/kWh (German average 2024)")
    st.caption("Dynamic markup: €0.18/kWh on top of wholesale")


# ── Load or generate price forecast ──────────────────────────────────────────
@st.cache_data(show_spinner="Loading price forecast...")
def load_prices(month: str) -> pd.DataFrame:
    """
    Scan all parquet files in data/raw/ and find rows matching the month.
    Falls back to synthetic data if no match is found.
    """
    start = pd.Timestamp(month + "-01", tz="UTC")
    end   = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)

    raw_dir = Path("data/raw")
    for parquet in sorted(raw_dir.glob("de_prices_*.parquet"), reverse=True):
        df = pd.read_parquet(parquet)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        subset = df[mask].copy()
        if len(subset) > 0:
            subset["source"] = "historical"
            return subset

    # Synthetic fallback: seasonal pattern + noise
    timestamps = pd.date_range(start, end, freq="h", tz="UTC")
    base = 80 + 20 * np.sin(np.linspace(0, 2 * np.pi, len(timestamps)))
    hourly_pattern = np.array([
        -15, -18, -20, -20, -18, -10, 0, 8, 12, 10, 8, 5,
          3,   2,   3,   5,   8,  15, 18, 20, 18, 12,  5, -5,
    ])
    hour_effect = np.array([hourly_pattern[ts.hour] for ts in timestamps])
    noise = np.random.normal(0, 8, len(timestamps))
    prices = (base + hour_effect + noise).clip(-50, 300)

    df = pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})
    df["source"] = "synthetic"
    return df


df_prices = load_prices(month)

if df_prices.empty:
    st.error("No price data available for this month.")
    st.stop()

source_label = "historical data" if df_prices["source"].iloc[0] == "historical" else "synthetic data (no data available for this period)"
st.info(f"Price source: {source_label} · {len(df_prices)} hourly observations", icon="ℹ️")

# ── Build consumption and compute bills ──────────────────────────────────────
timestamps  = pd.DatetimeIndex(df_prices["timestamp"])
profile     = PROFILES[profile_key]
consumption = build_hourly_consumption(timestamps, profile, monthly_kwh)
prices      = df_prices.set_index("timestamp")["price_eur_mwh"]
prices.index = pd.DatetimeIndex(prices.index)

summary = compute_bill_summary(consumption, prices)

# ── Bill summary cards ────────────────────────────────────────────────────────
st.subheader(f"{month} bill prediction — {profile_key}")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Fixed tariff bill",
    f"€{summary['fixed_total']:.2f}",
    help="Flat €0.30/kWh, no hourly variation",
)
col2.metric(
    "Dynamic tariff bill",
    f"€{summary['dynamic_total']:.2f}",
    delta=f"±€{(summary['dynamic_upper'] - summary['dynamic_lower'])/2:.2f} (model uncertainty)",
    delta_color="off",
    help="Wholesale spot + €0.18/kWh markup",
)
col3.metric(
    "Potential saving",
    f"€{summary['saving_eur']:.2f}",
    delta=f"{summary['saving_pct']:+.1f}% vs fixed",
    delta_color="normal" if summary["saving_eur"] > 0 else "inverse",
)
col4.metric(
    "Avg wholesale price",
    f"€{summary['avg_price']:.1f}/MWh",
    help="Mean day-ahead price for the period",
)

# ── Confidence interval banner ────────────────────────────────────────────────
st.markdown(
    f"""
    > **Dynamic bill estimate: €{summary['dynamic_lower']:.2f} – €{summary['dynamic_upper']:.2f}**
    > (based on LightGBM MAE = 7.23 EUR/MWh propagated through your consumption profile)
    """
)

# ── Hourly breakdown chart ────────────────────────────────────────────────────
st.subheader("Hourly cost comparison")

df_chart = pd.DataFrame({
    "timestamp":       timestamps,
    "consumption_kwh": consumption.values,
    "price_eur_mwh":   prices.values,
    "fixed_cost_eur":  fixed_tariff_cost(consumption).values,
    "dynamic_cost_eur":dynamic_tariff_cost(consumption, prices).values,
})

tab1, tab2, tab3 = st.tabs(["Cost comparison", "Consumption profile", "Price vs consumption"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_chart["timestamp"], y=df_chart["fixed_cost_eur"],
        name="Fixed tariff", line=dict(color="#94A3B8", width=1), fill="tozeroy", fillcolor="rgba(148,163,184,0.1)"
    ))
    fig.add_trace(go.Scatter(
        x=df_chart["timestamp"], y=df_chart["dynamic_cost_eur"],
        name="Dynamic tariff", line=dict(color="#0D9488", width=1.5), fill="tozeroy", fillcolor="rgba(13,148,136,0.15)"
    ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Cost (EUR/hour)",
        legend=dict(orientation="h", y=1.02), height=380,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    n_days_in_month = (timestamps.max() - timestamps.min()).days + 1
    avg_profile = pd.DataFrame({
        "hour":        list(range(24)),
        "consumption": (profile / profile.sum()) * (monthly_kwh / n_days_in_month),
    })
    fig2 = px.bar(
        avg_profile, x="hour", y="consumption",
        labels={"hour": "Hour of day", "consumption": "Avg kWh/hour"},
        color="consumption", color_continuous_scale="Teal",
    )
    fig2.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.scatter(
        df_chart, x="price_eur_mwh", y="consumption_kwh",
        color="dynamic_cost_eur",
        labels={
            "price_eur_mwh":   "Wholesale price (EUR/MWh)",
            "consumption_kwh": "Consumption (kWh)",
            "dynamic_cost_eur":"Hourly cost (EUR)",
        },
        color_continuous_scale="RdYlGn_r",
        opacity=0.6,
    )
    fig3.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ── Insight box ───────────────────────────────────────────────────────────────
st.subheader("Insights")

best_hours = df_chart.nsmallest(50, "price_eur_mwh")["timestamp"].dt.hour.value_counts().head(3).index.tolist()
worst_hours = df_chart.nlargest(50, "price_eur_mwh")["timestamp"].dt.hour.value_counts().head(3).index.tolist()

col_a, col_b = st.columns(2)
with col_a:
    st.success(
        f"**Cheapest hours** (shift loads here): "
        f"{', '.join(str(h)+':00' for h in sorted(best_hours))}"
    )
with col_b:
    st.warning(
        f"**Most expensive hours** (avoid if possible): "
        f"{', '.join(str(h)+':00' for h in sorted(worst_hours))}"
    )

if summary["saving_eur"] > 0:
    st.info(
        f"With dynamic pricing, **{profile_key}** could save "
        f"**€{summary['saving_eur']:.2f} ({summary['saving_pct']:.1f}%)** "
        f"compared to the flat €0.30/kWh tariff in {month}."
    )
else:
    st.warning(
        f"In {month}, dynamic pricing would cost "
        f"**€{abs(summary['saving_eur']):.2f} more** than the fixed tariff. "
        f"High wholesale prices made the flat rate cheaper."
    )
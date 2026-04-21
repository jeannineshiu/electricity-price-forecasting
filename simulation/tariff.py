import numpy as np
import pandas as pd

# Germany average fixed retail tariff (household, 2024)
FIXED_TARIFF_EUR_KWH = 0.30

# Dynamic tariff markup over wholesale (grid fees, taxes, supplier margin)
# German Netzentgelt + EEG + taxes ≈ €0.18/kWh on top of wholesale
DYNAMIC_MARKUP_EUR_KWH = 0.18


def fixed_tariff_cost(consumption_kwh: pd.Series) -> pd.Series:
    """
    Fixed tariff: flat €0.30/kWh regardless of hour.
    Returns hourly cost in EUR.
    """
    return consumption_kwh * FIXED_TARIFF_EUR_KWH


def dynamic_tariff_cost(
    consumption_kwh: pd.Series,
    wholesale_price_eur_mwh: pd.Series,
) -> pd.Series:
    """
    Dynamic tariff: wholesale spot price + fixed markup.
    wholesale is in EUR/MWh → convert to EUR/kWh by dividing by 1000.
    Returns hourly cost in EUR.
    """
    retail_eur_kwh = (wholesale_price_eur_mwh / 1000) + DYNAMIC_MARKUP_EUR_KWH
    # Floor at 0 — negative wholesale prices don't fully pass through to retail
    retail_eur_kwh = retail_eur_kwh.clip(lower=0.05)
    return consumption_kwh * retail_eur_kwh


def compute_bill_summary(
    consumption_kwh: pd.Series,
    price_eur_mwh: pd.Series,
    price_mae: float = 7.23,
) -> dict:
    """
    Compute monthly bill under both tariff types.
    Uncertainty on dynamic bill is propagated from model MAE.

    price_mae: model MAE in EUR/MWh
    """
    fixed_hourly   = fixed_tariff_cost(consumption_kwh)
    dynamic_hourly = dynamic_tariff_cost(consumption_kwh, price_eur_mwh)

    fixed_total   = fixed_hourly.sum()
    dynamic_total = dynamic_hourly.sum()

    # Uncertainty: MAE in EUR/MWh → EUR/kWh → propagate through consumption
    mae_eur_kwh       = price_mae / 1000
    dynamic_std       = (consumption_kwh * mae_eur_kwh).sum()
    dynamic_lower     = dynamic_total - dynamic_std
    dynamic_upper     = dynamic_total + dynamic_std

    saving = fixed_total - dynamic_total
    saving_pct = (saving / fixed_total) * 100 if fixed_total > 0 else 0

    return {
        "fixed_total":     round(fixed_total, 2),
        "dynamic_total":   round(dynamic_total, 2),
        "dynamic_lower":   round(max(dynamic_lower, 0), 2),
        "dynamic_upper":   round(dynamic_upper, 2),
        "saving_eur":      round(saving, 2),
        "saving_pct":      round(saving_pct, 1),
        "total_kwh":       round(consumption_kwh.sum(), 1),
        "avg_price":       round(price_eur_mwh.mean(), 2),
    }
import numpy as np
import pandas as pd

# Each profile returns hourly consumption weights (24 values, sum=1)
# Multiplied by monthly_kwh to get actual hourly kWh

def office_worker_profile() -> np.ndarray:
    """
    Office worker — away during day, home evenings and weekends.
    High consumption: 18:00-23:00 (cooking, TV, heating)
    Low consumption:  08:00-18:00 (at work)
    """
    profile = np.array([
        0.03, 0.02, 0.02, 0.02, 0.02, 0.03,  # 00-05 night
        0.04, 0.05, 0.03, 0.02, 0.02, 0.02,  # 06-11 morning/work
        0.02, 0.02, 0.02, 0.02, 0.03, 0.06,  # 12-17 work/commute
        0.09, 0.10, 0.09, 0.08, 0.07, 0.05,  # 18-23 evening peak
    ])
    return profile / profile.sum()


def wfh_profile() -> np.ndarray:
    """
    Work-from-home — steady daytime usage (PC, lighting, heating).
    More uniform, slight morning and evening peaks.
    """
    profile = np.array([
        0.02, 0.02, 0.02, 0.02, 0.02, 0.03,  # 00-05
        0.04, 0.05, 0.06, 0.06, 0.06, 0.06,  # 06-11
        0.06, 0.06, 0.06, 0.06, 0.06, 0.06,  # 12-17
        0.07, 0.07, 0.06, 0.05, 0.04, 0.03,  # 18-23
    ])
    return profile / profile.sum()


def industrial_profile() -> np.ndarray:
    """
    Industrial user — heavy daytime operations (Mon-Fri shift pattern).
    Peak: 06:00-18:00, minimal nights and weekends.
    """
    profile = np.array([
        0.01, 0.01, 0.01, 0.01, 0.02, 0.05,  # 00-05
        0.08, 0.09, 0.09, 0.09, 0.09, 0.09,  # 06-11
        0.09, 0.09, 0.09, 0.09, 0.07, 0.04,  # 12-17
        0.02, 0.01, 0.01, 0.01, 0.01, 0.01,  # 18-23
    ])
    return profile / profile.sum()


PROFILES = {
    "Office worker (evenings)":    office_worker_profile(),
    "WFH (steady daytime)":        wfh_profile(),
    "Industrial (daytime peak)":   industrial_profile(),
}


INDUSTRIAL_PROFILE = industrial_profile()


def build_hourly_consumption(
    timestamps: pd.DatetimeIndex,
    profile: np.ndarray,
    monthly_kwh: float,
) -> pd.Series:
    """
    Given hourly timestamps for a month, distribute monthly_kwh
    according to the hourly profile weights.

    Weekend behaviour: industrial profile scales down to 20% on Sat/Sun.
    """
    is_industrial = np.allclose(profile, INDUSTRIAL_PROFILE)
    n_days    = (timestamps.max() - timestamps.min()).days + 1
    daily_kwh = monthly_kwh / n_days

    consumption = []
    for ts in timestamps:
        local_ts = ts.tz_convert("Europe/Berlin")
        weight = profile[local_ts.hour]
        if is_industrial and local_ts.dayofweek >= 5:
            weight *= 0.2
        consumption.append(daily_kwh * weight * 24)

    series = pd.Series(consumption, index=timestamps, name="consumption_kwh")
    series = series / series.sum() * monthly_kwh
    return series
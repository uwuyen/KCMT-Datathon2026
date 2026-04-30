import pandas as pd
import numpy as np
import os


def add_features(df):
    """Feature Engineering chi tiết[cite: 1]."""
    d = df.copy()
    d["year"], d["month"], d["day"] = d["Date"].dt.year, d["Date"].dt.month, d["Date"].dt.day
    d["dow"], d["doy"], d["quarter"] = d["Date"].dt.dayofweek, d["Date"].dt.dayofyear, d["Date"].dt.quarter
    d["woy"] = d["Date"].dt.isocalendar().week.astype(int)
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["days_in_month"] = d["Date"].dt.days_in_month
    d["days_from_mend"] = d["days_in_month"] - d["day"]
    d["is_last1"] = (d["days_from_mend"] == 0).astype(int)
    d["is_last3"] = (d["days_from_mend"] <= 2).astype(int)
    d["is_last7"] = (d["days_from_mend"] <= 6).astype(int)
    d["is_first3"], d["is_first7"] = (d["day"] <= 3).astype(int), (d["day"] <= 7).astype(int)

    # Cyclical encoding[cite: 1]
    for col, period in [("month", 12), ("dow", 7), ("doy", 366)]:
        d[f"{col}_sin"] = np.sin(2 * np.pi * d[col] / period)
        d[f"{col}_cos"] = np.cos(2 * np.pi * d[col] / period)
    return d


def build_and_attach_profile(train, test):
    """Xây dựng Seasonal Profile[cite: 1]."""
    ann = train.groupby("year")[["Revenue", "COGS"]].transform("mean")
    d = train.copy()
    d["rn"], d["cn"] = d["Revenue"] / ann["Revenue"], d["COGS"] / ann["COGS"]

    profile = d.groupby(["month", "day"]).agg(
        rev_norm_mean=("rn", "mean"), rev_norm_std=("rn", "std"),
        cogs_norm_mean=("cn", "mean"), cogs_norm_std=("cn", "std")
    ).reset_index().fillna(0.1)

    def attach(df):
        o = df.merge(profile, on=["month", "day"], how="left")
        for c in ["rev_norm_mean", "rev_norm_std", "cogs_norm_mean", "cogs_norm_std"]:
            o[c] = o[c].fillna(1.0 if "mean" in c else 0.1)
        return o

    return attach(train), attach(test), profile
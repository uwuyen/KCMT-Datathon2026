import pandas as pd
import numpy as np
import os


def load_data(base_path):
    raw = pd.read_csv(os.path.join(base_path, "sales.csv"), parse_dates=["Date"])
    raw = raw.sort_values("Date").reset_index(drop=True)
    test_dates = pd.date_range("2023-01-01", "2024-07-01", freq="D")
    test_raw = pd.DataFrame({"Date": test_dates})
    return raw, test_raw


def add_features(df):
    d = df.copy()
    d["year"] = d["Date"].dt.year
    d["month"] = d["Date"].dt.month
    d["day"] = d["Date"].dt.day
    d["dow"] = d["Date"].dt.dayofweek
    d["doy"] = d["Date"].dt.dayofyear
    d["woy"] = d["Date"].dt.isocalendar().week.astype(int)
    d["quarter"] = d["Date"].dt.quarter
    # ... (Thêm các features sin/cos và logic ngày cuối tháng của bạn)
    return d


def build_and_attach_profile(train_df, test_df):
    S_COLS = ["rev_norm_mean", "rev_norm_std", "cogs_norm_mean", "cogs_norm_std"]
    ann = train_df.groupby("year")[["Revenue", "COGS"]].transform("mean")
    d = train_df.copy()
    d["rn"], d["cn"] = d["Revenue"] / ann["Revenue"], d["COGS"] / ann["COGS"]

    profile = (d.groupby(["month", "day"])
               .agg(rev_norm_mean=("rn", "mean"), rev_norm_std=("rn", "std"),
                    cogs_norm_mean=("cn", "mean"), cogs_norm_std=("cn", "std"))
               .reset_index().fillna(0.1))

    def attach(target_df):
        o = target_df.merge(profile, on=["month", "day"], how="left")
        return o.fillna(1.0)

    return attach(train_df), attach(test_df), profile
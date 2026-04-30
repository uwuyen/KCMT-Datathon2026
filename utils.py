import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def calculate_metrics(label, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  [{label:<25}]  MAE={mae:>11,.0f}  RMSE={rmse:>11,.0f}  R2={r2:>7.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}

def fourier_mat(df, origin, periods_orders):
    t = (df["Date"] - origin).dt.days.values.astype(float)
    cols = [np.ones(len(t)), t/max(t.max(),1)]
    for period, order in periods_orders:
        for k in range(1, order+1):
            cols += [np.sin(2*np.pi*k*t/period),
                     np.cos(2*np.pi*k*t/period)]
    me = df["days_from_mend"].values.astype(float)
    cols += [np.exp(-me/3), np.exp(-me/7),
             (me<=2).astype(float), (me==0).astype(float)]
    return np.column_stack(cols)

def save_forecast_plot(val, ens_rv_v, ens_cg_v, test, ens_rv_t, ens_cg_t, base_path):
    fig, axes = plt.subplots(2, 2, figsize=(18,10))
    # ... (Giữ nguyên phần code matplotlib của bạn ở đây)
    plt.tight_layout()
    path = os.path.join(base_path, "part3_forecast_plot.png")
    plt.savefig(path, dpi=120)
    print(f"\n[PLOT] Saved -> {path}")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def metrics(label, y_true, y_pred):
    """Tính toán và in các chỉ số MAE, RMSE, R2."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  [{label:<25}]  MAE={mae:>11,.0f}  RMSE={rmse:>11,.0f}  R2={r2:>7.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}

def fourier_mat(df, origin, periods_orders):
    """Tạo ma trận Fourier cho mô hình Ridge."""
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

def save_plots(val, ens_rv_v, ens_cg_v, test, ens_rv_t, ens_cg_t, base_path):
    """Vẽ biểu đồ Validation và Forecast."""
    fig, axes = plt.subplots(2, 2, figsize=(18,10))
    fig.suptitle("Phan 3 - Forecast Revenue & COGS", fontsize=13, fontweight="bold")

    def fax(ax, title, yl):
        ax.set_title(title); ax.set_ylabel(yl)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30); ax.grid(alpha=0.3); ax.legend()

    # Plot Validation
    axes[0,0].plot(val["Date"], val["Revenue"]/1e6, label="Actual"); axes[0,0].plot(val["Date"], ens_rv_v/1e6, label="Ens", ls="--")
    fax(axes[0,0], "Val - Revenue", "Trieu")
    axes[0,1].plot(val["Date"], val["COGS"]/1e6, label="Actual"); axes[0,1].plot(val["Date"], ens_cg_v/1e6, label="Ens", ls="--")
    fax(axes[0,1], "Val - COGS", "Trieu")

    # Plot Forecast
    axes[1,0].plot(test["Date"], ens_rv_t/1e6, color="tomato", label="Forecast")
    fax(axes[1,0], "Forecast Revenue 2023-2024", "Trieu")
    axes[1,1].plot(test["Date"], ens_cg_t/1e6, color="darkorange", label="Forecast")
    fax(axes[1,1], "Forecast COGS 2023-2024", "Trieu")

    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "part3_forecast_plot.png"))
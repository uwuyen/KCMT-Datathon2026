import numpy as np
import pandas as pd

def get_ensemble_weights(metrics_list):
    """Tính trọng số dựa trên 1/RMSE[cite: 1]."""
    inv = {k: 1/v['rmse'] for k, v in metrics_list.items()}
    s = sum(inv.values())
    return {k: v/s for k, v in inv.items()}

def create_submission(test_df, rv_pred, cg_pred, out_path):
    """Tạo file csv nộp bài[cite: 1]."""
    sub = pd.DataFrame({
        "Date": test_df["Date"].dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(rv_pred, 2),
        "COGS":    np.round(cg_pred, 2),
    })
    sub.to_csv(out_path, index=False)
    return sub
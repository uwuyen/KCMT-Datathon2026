import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from preprocess import build_and_attach_profile


def model_A_predict(df_dates, profile, ann_df, base_yr, recent_start):
    """Model A: Seasonal x Geometric Trend[cite: 1]."""
    rec = ann_df.loc[recent_start:base_yr]
    n = max(len(rec) - 1, 1)
    grR = (rec["Rev_mean"].iloc[-1] / rec["Rev_mean"].iloc[0]) ** (1 / n)
    grC = (rec["COGS_mean"].iloc[-1] / rec["COGS_mean"].iloc[0]) ** (1 / n)

    # Logic chuẩn hóa và gắn profile đã có trong preprocess nhưng model A gọi độc lập[cite: 1]
    df = df_dates.merge(profile, on=["month", "day"], how="left").fillna(1.0)
    ya = df["year"] - base_yr
    rv = np.maximum(ann_df.loc[base_yr, "Rev_mean"] * grR ** ya * df["rev_norm_mean"], 0)
    cg = np.maximum(ann_df.loc[base_yr, "COGS_mean"] * grC ** ya * df["cogs_norm_mean"], 0)
    return rv.values, cg.values


def train_lgb(Xtr, ytr, Xvl, yvl, params):
    """Huấn luyện LightGBM[cite: 1]."""
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], callbacks=[lgb.early_stopping(200, verbose=False)])
    return m


def train_xgb(Xtr, ytr, Xvl, yvl, params):
    """Huấn luyện XGBoost[cite: 1]."""
    m = xgb.XGBRegressor(**params)
    m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
    return m
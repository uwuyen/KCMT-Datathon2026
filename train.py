from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import numpy as np

# Model A logic
def run_model_a(df_dates, profile, ann_df, base_yr, recent_start):
    # ... (Giữ nguyên logic model_A của bạn)
    return rv, cg

# Model B logic
def train_model_b(train_df, origin):
    # Logic Xb_tr, sc_B, Ridge...
    pass

# Model C & D (LightGBM & XGBoost)
def train_gbm_models(X_tr, y_tr, X_vl, y_vl, params, mode='lgb'):
    if mode == 'lgb':
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
              callbacks=[lgb.early_stopping(200, verbose=False)])
    else:
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    return m
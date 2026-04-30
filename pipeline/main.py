import os
import pandas as pd
from utils import metrics, fourier_mat, save_plots
from preprocess import add_features, build_and_attach_profile
from train import model_A_predict, train_lgb, train_xgb
from predict import get_ensemble_weights, create_submission
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

BASE = r"c:\Users\ADM\Downloads\datathon-2026-round-1"
FEATS = ["year","month","day","dow","doy","woy","quarter","is_weekend","days_in_month",
         "days_from_mend","days_from_mstart","is_last1","is_last3","is_last7","is_first3",
         "is_first7","month_sin","month_cos","dow_sin","dow_cos","doy_sin","doy_cos",
         "rev_norm_mean","rev_norm_std","cogs_norm_mean","cogs_norm_std"]

# 1. LOAD & PREPROCESS[cite: 1]
raw = pd.read_csv(os.path.join(BASE, "sales.csv"), parse_dates=["Date"]).sort_values("Date")
test_dates = pd.DataFrame({"Date": pd.date_range("2023-01-01", "2024-07-01", freq="D")})

train_df = add_features(raw)
test_df = add_features(test_dates)
train_df, test_df, profile = build_and_attach_profile(train_df, test_df)

trn = train_df[train_df["year"] <= 2020].copy()
val = train_df[train_df["year"] >= 2021].copy()
annual_trn = trn.groupby("year").agg(Rev_mean=("Revenue","mean"), COGS_mean=("COGS","mean"))

# 2. MODEL A[cite: 1]
pA_rv_v, pA_cg_v = model_A_predict(val, profile, annual_trn, 2020, 2018)
mA_rv = metrics("A Rev", val["Revenue"], pA_rv_v)

# 3. MODEL B (Fourier Ridge)[cite: 1]
PERIODS = [(7,5),(30.5,8),(91.25,4),(365.25,12)]
origin = trn[trn["year"] >= 2019]["Date"].min()
Xb_tr = fourier_mat(trn[trn["year"]>=2019], origin, PERIODS)
Xb_vl = fourier_mat(val, origin, PERIODS)
sc = StandardScaler().fit(Xb_tr)
rB_rv = Ridge(alpha=50).fit(sc.transform(Xb_tr), trn[trn["year"]>=2019]["Revenue"])
pB_rv_v = rB_rv.predict(sc.transform(Xb_vl))
mB_rv = metrics("B Rev", val["Revenue"], pB_rv_v)

# 4. MODEL C & D (LGBM & XGB)[cite: 1]
LGB_P = {"n_estimators": 1000, "learning_rate": 0.05, "verbose": -1}
mC = train_lgb(trn[FEATS], trn["Revenue"], val[FEATS], val["Revenue"], LGB_P)
pC_rv_v = mC.predict(val[FEATS])
mC_rv = metrics("C Rev", val["Revenue"], pC_rv_v)

# 5. ENSEMBLE & PREDICT[cite: 1]
w = get_ensemble_weights({"C": mC_rv, "A": mA_rv}) # Ví dụ lấy C và A
# ... Thực hiện tương tự cho COGS và Test set ...

print("Pipeline Finished!")
# type: ignore

"""
PHAN 3: MO HINH DU BAO DOANH THU (Revenue & COGS) - Datathon 2026 Vong 1

Chien luoc Ensemble 4 models:
  Model A : Seasonal Profile x Trend (geometric, recent years)
  Model B : Fourier Ridge Regression (chi train tren nam gan day)
  Model C : LightGBM (calendar + seasonal features)
  Model D : XGBoost (same features)
Metrics: MAE, RMSE, R2

"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE  = r"c:\Users\ADM\Downloads\datathon-2026-round-1"
OUT_F = os.path.join(BASE, "submission_part3.csv")

print("=" * 70)
print("PHAN 3: MO HINH DU BAO DOANH THU & COGS")
print("=" * 70)

# 1. LOAD DATA 
raw = pd.read_csv(os.path.join(BASE, "sales.csv"), parse_dates=["Date"])
raw = raw.sort_values("Date").reset_index(drop=True)

test_dates = pd.date_range("2023-01-01", "2024-07-01", freq="D")
test_raw   = pd.DataFrame({"Date": test_dates})

print(f"\n[DATA] Train: {raw['Date'].min().date()} -> {raw['Date'].max().date()}  Rows={len(raw)}")
print(f"[TEST] {test_dates[0].date()} -> {test_dates[-1].date()}  Rows={len(test_dates)}")

# 2. FEATURE ENGINEERING 
def add_features(df):
    d = df.copy()
    d["year"]            = d["Date"].dt.year
    d["month"]           = d["Date"].dt.month
    d["day"]             = d["Date"].dt.day
    d["dow"]             = d["Date"].dt.dayofweek
    d["doy"]             = d["Date"].dt.dayofyear
    d["woy"]             = d["Date"].dt.isocalendar().week.astype(int)
    d["quarter"]         = d["Date"].dt.quarter
    d["is_weekend"]      = (d["dow"] >= 5).astype(int)
    d["days_in_month"]   = d["Date"].dt.days_in_month
    d["days_from_mend"]  = d["days_in_month"] - d["day"]
    d["days_from_mstart"]= d["day"] - 1
    d["is_last1"]        = (d["days_from_mend"] == 0).astype(int)
    d["is_last3"]        = (d["days_from_mend"] <= 2).astype(int)
    d["is_last7"]        = (d["days_from_mend"] <= 6).astype(int)
    d["is_first3"]       = (d["day"] <= 3).astype(int)
    d["is_first7"]       = (d["day"] <= 7).astype(int)
    d["month_sin"]       = np.sin(2*np.pi*d["month"]/12)
    d["month_cos"]       = np.cos(2*np.pi*d["month"]/12)
    d["dow_sin"]         = np.sin(2*np.pi*d["dow"]/7)
    d["dow_cos"]         = np.cos(2*np.pi*d["dow"]/7)
    d["doy_sin"]         = np.sin(2*np.pi*d["doy"]/366)
    d["doy_cos"]         = np.cos(2*np.pi*d["doy"]/366)
    return d

train = add_features(raw)
test  = add_features(test_raw)

# 3. SEASONAL PROFILE 
S_COLS = ["rev_norm_mean","rev_norm_std","cogs_norm_mean","cogs_norm_std"]

def build_profile(df):
    ann = df.groupby("year")[["Revenue","COGS"]].transform("mean")
    d   = df.copy()
    d["rn"] = d["Revenue"] / ann["Revenue"]
    d["cn"] = d["COGS"]    / ann["COGS"]
    p = (d.groupby(["month","day"])
         .agg(rev_norm_mean=("rn","mean"), rev_norm_std=("rn","std"),
              cogs_norm_mean=("cn","mean"),cogs_norm_std=("cn","std"))
         .reset_index().fillna({"rev_norm_std":0.1,"cogs_norm_std":0.1}))
    return p

def attach_profile(df, profile):
    d = df.drop(columns=[c for c in S_COLS if c in df.columns])
    o = d.merge(profile[["month","day"]+S_COLS], on=["month","day"], how="left")
    for c in S_COLS:
        o[c] = o[c].fillna(1.0 if "mean" in c else 0.1)
    return o

profile_all = build_profile(train)
train = attach_profile(train, profile_all)
test  = attach_profile(test,  profile_all)

annual = (train.groupby("year")
          .agg(Rev_mean=("Revenue","mean"), COGS_mean=("COGS","mean")))

# 4. METRICS 
def metrics(label, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  [{label:<25}]  MAE={mae:>11,.0f}  RMSE={rmse:>11,.0f}  R2={r2:>7.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}

# 5. TRAIN/VAL SPLIT 
trn = train[train["year"] <= 2020].copy()
val = train[train["year"] >= 2021].copy()
annual_trn = trn.groupby("year").agg(Rev_mean=("Revenue","mean"), COGS_mean=("COGS","mean"))


# MODEL A — Enhanced Seasonal × Geometric Trend
print("\n-- MODEL A: Seasonal × Trend --")

def model_A(df_dates, profile, ann_df, base_yr, recent_start):
    rec = ann_df.loc[recent_start:base_yr]
    n   = max(len(rec)-1, 1)
    grR = (rec["Rev_mean"].iloc[-1]  / rec["Rev_mean"].iloc[0])  ** (1/n)
    grC = (rec["COGS_mean"].iloc[-1] / rec["COGS_mean"].iloc[0]) ** (1/n)
    bR  = ann_df.loc[base_yr, "Rev_mean"]
    bC  = ann_df.loc[base_yr, "COGS_mean"]
    df  = attach_profile(df_dates, profile)
    ya  = df["year"] - base_yr
    rv  = np.maximum(bR  * grR**ya * df["rev_norm_mean"],  0).values
    cg  = np.maximum(bC  * grC**ya * df["cogs_norm_mean"], 0).values
    return rv, cg

pA_rv_v, pA_cg_v = model_A(val, profile_all, annual_trn, 2020, 2018)
mA_rv  = metrics("A Rev  val",  val["Revenue"], pA_rv_v)
mA_cg  = metrics("A COGS val",  val["COGS"],    pA_cg_v)

# Final (full retrain)
pA_rv_t, pA_cg_t = model_A(test, profile_all, annual, 2022, 2019)



# MODEL B — Fourier Ridge Regression (train only on RECENT years to avoid
#            structural-break bias from 2012-2018 high-revenue period)
print("\n-- MODEL B: Fourier Ridge Regression (recent years) --")

def fourier_mat(df, origin, periods_orders):
    t = (df["Date"] - origin).dt.days.values.astype(float)
    cols = [np.ones(len(t)), t/max(t.max(),1)]   # intercept + linear trend
    for period, order in periods_orders:
        for k in range(1, order+1):
            cols += [np.sin(2*np.pi*k*t/period),
                     np.cos(2*np.pi*k*t/period)]
    me = df["days_from_mend"].values.astype(float)
    cols += [np.exp(-me/3), np.exp(-me/7),
             (me<=2).astype(float), (me==0).astype(float)]
    return np.column_stack(cols)

PERIODS = [(7,5),(30.5,8),(91.25,4),(365.25,12)]

# Validation: train B on 2019-2020 only
trn_B = trn[trn["year"] >= 2019]
origin_B = trn_B["Date"].min()

Xb_tr = fourier_mat(trn_B, origin_B, PERIODS)
Xb_vl = fourier_mat(val,   origin_B, PERIODS)
Xb_ts = fourier_mat(test,  origin_B, PERIODS)

sc_B = StandardScaler().fit(Xb_tr)
Xb_tr_s = sc_B.transform(Xb_tr)
Xb_vl_s = sc_B.transform(Xb_vl)
Xb_ts_s = sc_B.transform(Xb_ts)

rB_rv = Ridge(alpha=50).fit(Xb_tr_s, trn_B["Revenue"])
rB_cg = Ridge(alpha=50).fit(Xb_tr_s, trn_B["COGS"])

pB_rv_v  = np.maximum(rB_rv.predict(Xb_vl_s), 0)
pB_cg_v  = np.maximum(rB_cg.predict(Xb_vl_s), 0)
mB_rv  = metrics("B Rev  val",  val["Revenue"], pB_rv_v)
mB_cg  = metrics("B COGS val",  val["COGS"],    pB_cg_v)

# Final (train on 2019-2022)
trn_B_full   = train[train["year"] >= 2019]
origin_B_full = trn_B_full["Date"].min()
Xbf_tr = fourier_mat(trn_B_full, origin_B_full, PERIODS)
Xbf_ts = fourier_mat(test,       origin_B_full, PERIODS)
scBf   = StandardScaler().fit(Xbf_tr)
rBf_rv = Ridge(alpha=50).fit(scBf.transform(Xbf_tr), trn_B_full["Revenue"])
rBf_cg = Ridge(alpha=50).fit(scBf.transform(Xbf_tr), trn_B_full["COGS"])
pB_rv_t = np.maximum(rBf_rv.predict(scBf.transform(Xbf_ts)), 0)
pB_cg_t = np.maximum(rBf_cg.predict(scBf.transform(Xbf_ts)), 0)


# MODEL C — LightGBM
print("\n-- MODEL C: LightGBM --")

FEATS = [
    "year","month","day","dow","doy","woy","quarter",
    "is_weekend","days_in_month","days_from_mend","days_from_mstart",
    "is_last1","is_last3","is_last7","is_first3","is_first7",
    "month_sin","month_cos","dow_sin","dow_cos","doy_sin","doy_cos",
    "rev_norm_mean","rev_norm_std","cogs_norm_mean","cogs_norm_std",
]

LGB_P = dict(
    objective="regression", metric="mae",
    n_estimators=5000, learning_rate=0.02,
    num_leaves=63, min_child_samples=15, max_depth=6,
    subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
    reg_alpha=0.05, reg_lambda=0.1,
    random_state=42, n_jobs=-1, verbose=-1,
)

X_tr = trn[FEATS]; X_vl = val[FEATS]; X_ts = test[FEATS]

def fit_lgb(Xtr, ytr, Xvl, yvl, Xts):
    m = lgb.LGBMRegressor(**LGB_P)
    m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)])
    return np.maximum(m.predict(Xvl),0), np.maximum(m.predict(Xts),0), m

pC_rv_v, pC_rv_t0, mC_rv_model = fit_lgb(X_tr, trn["Revenue"], X_vl, val["Revenue"], X_ts)
pC_cg_v, pC_cg_t0, mC_cg_model = fit_lgb(X_tr, trn["COGS"],    X_vl, val["COGS"],    X_ts)
mC_rv  = metrics("C Rev  val",  val["Revenue"], pC_rv_v)
mC_cg  = metrics("C COGS val",  val["COGS"],    pC_cg_v)

print("  [LightGBM retrain full 2012-2022]")
Xfull = train[FEATS]
lgbRf = lgb.LGBMRegressor(**LGB_P); lgbRf.fit(Xfull, train["Revenue"], callbacks=[lgb.log_evaluation(-1)])
lgbCf = lgb.LGBMRegressor(**LGB_P); lgbCf.fit(Xfull, train["COGS"],    callbacks=[lgb.log_evaluation(-1)])
pC_rv_t = np.maximum(lgbRf.predict(X_ts), 0)
pC_cg_t = np.maximum(lgbCf.predict(X_ts), 0)



# MODEL D — XGBoost
print("\n-- MODEL D: XGBoost --")

XGB_P = dict(
    objective="reg:absoluteerror", n_estimators=5000,
    learning_rate=0.02, max_depth=6, min_child_weight=5,
    subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.05, reg_lambda=0.1,
    random_state=42, n_jobs=-1, tree_method="hist",
    early_stopping_rounds=200,
)

def fit_xgb(Xtr, ytr, Xvl, yvl, Xts):
    m = xgb.XGBRegressor(**XGB_P)
    m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
    return np.maximum(m.predict(Xvl),0), np.maximum(m.predict(Xts),0), m

pD_rv_v, pD_rv_t0, xgbR = fit_xgb(X_tr, trn["Revenue"], X_vl, val["Revenue"], X_ts)
pD_cg_v, pD_cg_t0, xgbC = fit_xgb(X_tr, trn["COGS"],    X_vl, val["COGS"],    X_ts)
mD_rv  = metrics("D Rev  val",  val["Revenue"], pD_rv_v)
mD_cg  = metrics("D COGS val",  val["COGS"],    pD_cg_v)

print("  [XGBoost retrain full 2012-2022]")
XGB_P_noes = {k:v for k,v in XGB_P.items() if k!="early_stopping_rounds"}
xgbRf = xgb.XGBRegressor(**XGB_P_noes); xgbRf.fit(Xfull, train["Revenue"], verbose=False)
xgbCf = xgb.XGBRegressor(**XGB_P_noes); xgbCf.fit(Xfull, train["COGS"],    verbose=False)
pD_rv_t = np.maximum(xgbRf.predict(X_ts), 0)
pD_cg_t = np.maximum(xgbCf.predict(X_ts), 0)



# ENSEMBLE — Stacking approach:
#   1. Full 4-model ensemble weighted by 1/RMSE (for reporting)
#   2. Best-2 ensemble (C+D) for final submission — drop weak models A,B
print("\n-- ENSEMBLE (weighted by 1/RMSE) --")

def ens_w(*pairs):
    inv = {k: 1/v for k,v in pairs}
    s   = sum(inv.values()); return {k: v/s for k,v in inv.items()}

# Full 4-model (for reporting)
wR4 = ens_w(("A",mA_rv["rmse"]),("B",mB_rv["rmse"]),("C",mC_rv["rmse"]),("D",mD_rv["rmse"]))
wC4 = ens_w(("A",mA_cg["rmse"]),("B",mB_cg["rmse"]),("C",mC_cg["rmse"]),("D",mD_cg["rmse"]))
print(f"  4-model Revenue weights: "+"  ".join(f"M{k}={v:.3f}" for k,v in wR4.items()))
print(f"  4-model COGS    weights: "+"  ".join(f"M{k}={v:.3f}" for k,v in wC4.items()))

ens4_rv_v = wR4["A"]*pA_rv_v+wR4["B"]*pB_rv_v+wR4["C"]*pC_rv_v+wR4["D"]*pD_rv_v
ens4_cg_v = wC4["A"]*pA_cg_v+wC4["B"]*pB_cg_v+wC4["C"]*pC_cg_v+wC4["D"]*pD_cg_v
mE4_rv = metrics("4-model Ens Rev val",  val["Revenue"], ens4_rv_v)
mE4_cg = metrics("4-model Ens COGS val", val["COGS"],    ens4_cg_v)

# Best-2 ensemble C+D (LightGBM + XGBoost only — best individual models)
wR2 = ens_w(("C",mC_rv["rmse"]),("D",mD_rv["rmse"]))
wC2 = ens_w(("C",mC_cg["rmse"]),("D",mD_cg["rmse"]))
print(f"\n  Best-2 Revenue weights: MC={wR2['C']:.3f}  MD={wR2['D']:.3f}")
print(f"  Best-2 COGS    weights: MC={wC2['C']:.3f}  MD={wC2['D']:.3f}")

ens_rv_v = wR2["C"]*pC_rv_v + wR2["D"]*pD_rv_v
ens_cg_v = wC2["C"]*pC_cg_v + wC2["D"]*pD_cg_v
mE_rv  = metrics("Best-2 Ens Rev  val", val["Revenue"], ens_rv_v)
mE_cg  = metrics("Best-2 Ens COGS val", val["COGS"],    ens_cg_v)

# Final test predictions = Best-2 ensemble (full-retrained C+D)
ens_rv_t = wR2["C"]*pC_rv_t + wR2["D"]*pD_rv_t
ens_cg_t = wC2["C"]*pC_cg_t + wC2["D"]*pD_cg_t


# SUMMARY TABLE
print("\n" + "="*96)
print("TONG KET VALIDATION 2021-2022")
print("="*96)
H = f"{'Model':<28} {'RevMAE':>12} {'RevRMSE':>12} {'RevR2':>7}  {'COGSМAE':>12} {'COGSRMSE':>12} {'COGSR2':>7}"
print(H); print("-"*96)
for name, mr, mc in [
    ("Model A (Seasonal+Trend)",   mA_rv,  mA_cg),
    ("Model B (Fourier+Ridge)",    mB_rv,  mB_cg),
    ("Model C (LightGBM)",         mC_rv,  mC_cg),
    ("Model D (XGBoost)",          mD_rv,  mD_cg),
    ("4-model Ensemble (A+B+C+D)", mE4_rv, mE4_cg),
    (">>> BEST-2 Ensemble (C+D)",  mE_rv,  mE_cg),
]:
    print(f"{name:<28} {mr['mae']:>12,.0f} {mr['rmse']:>12,.0f} {mr['r2']:>7.4f}"
          f"  {mc['mae']:>12,.0f} {mc['rmse']:>12,.0f} {mc['r2']:>7.4f}")


# VISUALIZATION 
fig, axes = plt.subplots(2, 2, figsize=(18,10))
fig.suptitle("Phan 3 - Du bao Revenue & COGS\nValidation 2021-2022 | Forecast 2023-2024",
             fontsize=13, fontweight="bold")

def fax(ax, title, yl):
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(yl)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[0,0]
ax.plot(val["Date"], val["Revenue"]/1e6, lw=1, label="Actual", color="royalblue")
ax.plot(val["Date"], ens_rv_v/1e6,       lw=1, ls="--", label="Ensemble", color="tomato")
fax(ax, "Validation - Revenue (trieu)", "Revenue (trieu)")

ax = axes[0,1]
ax.plot(val["Date"], val["COGS"]/1e6, lw=1, label="Actual", color="seagreen")
ax.plot(val["Date"], ens_cg_v/1e6,   lw=1, ls="--", label="Ensemble", color="darkorange")
fax(ax, "Validation - COGS (trieu)", "COGS (trieu)")

ax = axes[1,0]
ax.plot(test["Date"], ens_rv_t/1e6, lw=1.2, color="tomato", label="Du bao Revenue")
ax.fill_between(test["Date"], ens_rv_t*0.9/1e6, ens_rv_t*1.1/1e6,
                alpha=0.15, color="tomato", label="+/-10%")
fax(ax, "Du bao Revenue 2023-2024", "Revenue (trieu)")

ax = axes[1,1]
ax.plot(test["Date"], ens_cg_t/1e6, lw=1.2, color="darkorange", label="Du bao COGS")
ax.fill_between(test["Date"], ens_cg_t*0.9/1e6, ens_cg_t*1.1/1e6,
                alpha=0.15, color="darkorange", label="+/-10%")
fax(ax, "Du bao COGS 2023-2024", "COGS (trieu)")

plt.tight_layout()
plt.savefig(os.path.join(BASE, "part3_forecast_plot.png"), dpi=120, bbox_inches="tight")
print(f"\n[PLOT] Saved -> {os.path.join(BASE, 'part3_forecast_plot.png')}")


# EXPORT SUBMISSION 
sub = pd.DataFrame({
    "Date":    test["Date"].dt.strftime("%Y-%m-%d"),
    "Revenue": np.round(ens_rv_t, 2),
    "COGS":    np.round(ens_cg_t, 2),
})
sub.to_csv(OUT_F, index=False)
print(f"[OUTPUT] {OUT_F}  |  Rows={len(sub)}")
print(f"         Date : {sub['Date'].iloc[0]} -> {sub['Date'].iloc[-1]}")
print(f"         Revenue mean : {sub['Revenue'].mean():>12,.2f}")
print(f"         COGS    mean : {sub['COGS'].mean():>12,.2f}")
print("\n[DONE] Phan 3 hoan thanh!")

from preprocess import load_data, add_features, build_and_attach_profile
from train import run_model_a, train_gbm_models  # và các hàm khác
from utils import calculate_metrics, save_forecast_plot
import numpy as np
import os

BASE = r"c:\Users\ADM\Downloads\datathon-2026-round-1"


def main():
    # 1. Preprocess
    raw, test_raw = load_data(BASE)
    train = add_features(raw)
    test = add_features(test_raw)
    train, test, profile = build_and_attach_profile(train, test)

    # 2. Split
    trn = train[train["year"] <= 2020].copy()
    val = train[train["year"] >= 2021].copy()

    # 3. Train & Predict (Lần lượt gọi các hàm từ train.py)
    # ... (Thực hiện huấn luyện A, B, C, D)

    # 4. Ensemble
    # ... (Tính trọng số dựa trên RMSE)

    # 5. Export
    # sub.to_csv(...)
    print("Pipeline completed!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train tau & beta sweep (RF + GPU-XGBoost)
Using geomcached loader, cutoff = 40.
beta: 0.5 → 6.0 (step 0.5)
tau : 0.5 → 10.0 (step 0.5)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# ---- LOADER（严格使用 geomcached）----
from utils.ri_loader_geomcached import (
    precompute_pair_cache_for_set,
    build_bigmatrix,
    load_dataset_bigmatrix,
    _bigmatrix_path,
)


# ============================================================
# 寻找项目根目录
# ============================================================

def find_project_root(start: Path, marker="utils"):
    p = start
    while p != p.parent:
        if (p / marker).exists():
            return p
        p = p.parent
    raise RuntimeError("Project root not found.")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = find_project_root(SCRIPT_DIR)
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 参数区
# ============================================================

alpha = "lor"

# beta: 0.5 → 6.0（含闭区间）
beta_list = np.arange(0.5, 6.0 + 1e-9, 0.5).round(2)

# tau: 0.5 → 10.0（含闭区间）
tau_list  = np.arange(0.5, 10.0 + 1e-9, 0.5).round(2)

# 固定 cutoff = 40
cutoff = 40.0

# 你想进行多少次 repeat（一般 1 次即可）
repeat_times = 1

# 并行核数（构建大矩阵阶段）
N_JOBS = 8


print("\n====================================================")
print("TRAINING: tau & beta sweep (RF + GPU-XGB)")
print("====================================================\n")
print(f"alpha={alpha}")
print(f"beta_list={beta_list}")
print(f"tau_list={tau_list}")
print(f"cutoff={cutoff}\n")


# ============================================================
# STEP 1. 预计算 ratio pair cache（只需一次）
# ============================================================

print("\n======== STEP 1: Precompute ratio pair cache ========\n")

for set_type in ["refined_only", "core"]:
    print(f"[pair cache] computing for {set_type}")
    precompute_pair_cache_for_set(
        set_type=set_type,
        use_cache=True,
        n_jobs=N_JOBS,
    )


# ============================================================
# STEP 2. 并行构建 bigmatrix（beta × tau）
# ============================================================

def bigmatrix_exists(set_type, alpha, beta, tau, cutoff):
    """检查对应参数集是否已有大矩阵，避免重复计算"""
    p = _bigmatrix_path(set_type, alpha, beta, tau, cutoff)
    return p.exists() and p.stat().st_size > 1024


def build_bigmatrix_job(alpha, beta, tau):
    """构建 refined_only + core 两套大矩阵"""
    for set_type in ["refined_only", "core"]:

        if bigmatrix_exists(set_type, alpha, beta, tau, cutoff):
            print(f"[跳过] 已存在 bigmatrix {set_type} beta={beta} tau={tau}")
            continue

        print(f"[构建] {set_type} beta={beta} tau={tau}")
        build_bigmatrix(
            set_type=set_type,
            alpha=alpha,
            beta=beta,
            tau=tau,
            cutoff=cutoff,
            use_cache=True,
        )


print("\n======== STEP 2: 并行构建大矩阵 ========\n")

jobs = [(alpha, b, t) for b in beta_list for t in tau_list]

Parallel(n_jobs=N_JOBS)(
    delayed(build_bigmatrix_job)(a, b, t) for (a, b, t) in jobs
)


# ============================================================
# STEP 3-A. RandomForest (CPU) 训练函数
# ============================================================

def train_rf(alpha, beta, tau, cutoff, repeat):

    X_train, y_train, _ = load_dataset_bigmatrix("refined_only", alpha, beta, tau, cutoff)
    X_test,  y_test,  _ = load_dataset_bigmatrix("core",         alpha, beta, tau, cutoff)

    print("  -> Training RF (CPU)...")

    rf = RandomForestRegressor(
        n_estimators=650,
        max_depth=20,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    Rp = pearsonr(y_test, pred)[0]
    RMSE = np.sqrt(mean_squared_error(y_test, pred))

    print(f"     RF: Rp={Rp:.4f}, RMSE={RMSE:.4f}")

    return {
        "alpha": alpha, "beta": beta, "tau": tau, "cutoff": cutoff,
        "repeat": repeat, "model": "rf",
        "pearson": Rp, "rmse": RMSE,
    }


# ============================================================
# STEP 3-B. GPU XGBoost 训练函数
# ============================================================

def train_xgb_gpu(alpha, beta, tau, cutoff, repeat):

    # 加载大矩阵（不变）
    X_train, y_train, _ = load_dataset_bigmatrix("refined_only", alpha, beta, tau, cutoff)
    X_test,  y_test,  _ = load_dataset_bigmatrix("core",         alpha, beta, tau, cutoff)

    print("  -> Training XGB (GPU)...")

    # ---- XGBoost 3.x GPU 用法 ----
    # DMatrix 不接受 device 参数
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    # XGBoost >=3.0: GPU 用法 = tree_method="hist" + device="cuda"
    params = {
        "tree_method": "hist",      # 不再有 gpu_hist
        "device": "cuda",           # GPU 在这里指定
        "max_depth": 8,
        "eta": 0.03,
        "lambda": 0.2,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "max_bin": 128,
        "objective": "reg:squarederror",
    }

    # 训练 booster（结构不变）
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=1400,
    )

    # 预测（不变）
    pred = booster.predict(dtest)

    # 指标（不变）
    Rp = pearsonr(y_test, pred)[0]
    RMSE = np.sqrt(mean_squared_error(y_test, pred))

    print(f"     XGB(GPU): Rp={Rp:.4f}, RMSE={RMSE:.4f}")

    # 返回值格式不变
    return {
        "alpha": alpha, "beta": beta, "tau": tau, "cutoff": cutoff,
        "repeat": repeat, "model": "xgb-gpu",
        "pearson": Rp, "rmse": RMSE,
    }


# ============================================================
# STEP 4. 主循环 + CSV 输出（支持断点恢复）
# ============================================================

out_csv = MODEL_DIR / f"results_tau_beta_sweep_{alpha}.csv"

if out_csv.exists():
    df_done = pd.read_csv(out_csv)
    done_set = set(zip(df_done["beta"], df_done["tau"], df_done["repeat"], df_done["model"]))
else:
    done_set = set()

print("\n======== STEP 4: Training ========\n")

for repeat in range(repeat_times):
    for beta in beta_list:
        for tau in tau_list:

            # RF
            print(f"\n[Training] RF: alpha={alpha}, beta={beta}, tau={tau}, repeat={repeat}")
            tag_rf = (beta, tau, repeat, "rf")
            if tag_rf in done_set:
                print("tag_rf already done, skipping...")
            else:
                rec_rf = train_rf(alpha, beta, tau, cutoff, repeat)
                pd.DataFrame([rec_rf]).to_csv(
                    out_csv, mode="a",
                    header=not out_csv.exists(),
                    index=False
                )

            # XGB（GPU）
            tag_xgb = (beta, tau, repeat, "xgb-gpu")
            if tag_xgb in done_set:
                print("tag_xgb already done, skipping...")
            else:
                rec_xgb = train_xgb_gpu(alpha, beta, tau, cutoff, repeat)
                pd.DataFrame([rec_xgb]).to_csv(
                    out_csv, mode="a",
                    header=not out_csv.exists(),
                    index=False
                )

print("\n========= 全部训练结束！ =========")
print(f"结果已保存到: {out_csv}")

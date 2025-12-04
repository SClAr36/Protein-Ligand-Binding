#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train tau1 & tau2 sweep with 108-dim features (RF + GPU-XGBoost)

特征构造（每个 pdbid）：
    X = concat(
        phi(alpha="exp", beta=2.5, tau=tau1, cutoff1),
        phi(alpha="exp", beta=2.5, tau=tau2, cutoff2),
        phi(alpha="exp", beta=40.0, tau=5.5, cutoff3=18.0),
    )
 → 维度 = 3 * 36 = 108

tau1, tau2: 0.5 → 20.0 (step 0.5)
cutoff_i = max(12, 3.7 * tau_i)  (i = 1, 2)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# ---- LOADER（使用 geomcached）----
from utils.ri_loader_geomcached import (
    precompute_pair_cache_for_set,
    build_bigmatrix,
    load_dataset_bigmatrix,
    _bigmatrix_path,
)  # :contentReference[oaicite:0]{index=0}


# ============================================================
# 寻找项目根目录
# ============================================================

def find_project_root(start: Path, marker: str = "utils") -> Path:
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

alpha = "exp"

# block1 & block2
beta12 = 2.5

# block3 固定参数
beta3 = 40.0
tau3 = 5.5
cutoff3 = 18.0

# tau1, tau2 网格：0.5 → 20.0，步长 0.5
tau_grid = np.arange(0.5, 20.0 + 1e-9, 0.5).round(2)

# cutoff 规则：max(12, 3.7 * tau)
def cutoff_from_tau(tau: float) -> float:
    return float(max(12.0, 3.7 * float(tau)))

# 重复次数（如果你想之后算 median，可以设为 5）
repeat_times = 1

# bigmatrix 构建用的并行核数
N_JOBS_BUILD = 8

print("\n====================================================")
print("TRAINING: tau1 & tau2 sweep (108-dim, RF + GPU-XGB)")
print("====================================================\n")
print(f"alpha={alpha}")
print(f"beta12={beta12}, beta3={beta3}")
print(f"tau_grid={tau_grid[0]} .. {tau_grid[-1]} (N={len(tau_grid)})")
print(f"tau3={tau3}, cutoff3={cutoff3}\n")


# ============================================================
# STEP 1. 预计算 ratio pair cache（只需一次）
# ============================================================

print("\n======== STEP 1: Precompute ratio pair cache ========\n")

for set_type in ["refined_only", "core"]:
    print(f"[pair cache] computing for {set_type}")
    precompute_pair_cache_for_set(
        set_type=set_type,
        use_cache=True,
        n_jobs=N_JOBS_BUILD,
    )


# ============================================================
# STEP 2. 构建所有需要的 36 维 bigmatrix
#   - block1 & block2: (beta=2.5, tau in tau_grid, cutoff=max(12, 3.7*tau))
#   - block3: (beta=40.0, tau=5.5, cutoff=18.0)
# ============================================================

def bigmatrix_exists(set_type: str, alpha: str, beta: float, tau: float, cutoff: float) -> bool:
    """检查对应参数集是否已有大矩阵，避免重复计算"""
    p = _bigmatrix_path(set_type, alpha, beta, tau, cutoff)
    return p.exists() and p.stat().st_size > 1024


def build_bigmatrix_for_config(alpha: str, beta: float, tau: float, cutoff: float):
    """给定一组 (alpha, beta, tau, cutoff)，为 refined_only + core 各构建一份 bigmatrix"""
    for set_type in ["refined_only", "core"]:
        if bigmatrix_exists(set_type, alpha, beta, tau, cutoff):
            print(f"[跳过] bigmatrix 已存在: {set_type}, a={alpha}, b={beta}, t={tau}, c={cutoff}")
            continue

        print(f"[构建] bigmatrix: {set_type}, a={alpha}, b={beta}, t={tau}, c={cutoff}")
        build_bigmatrix(
            set_type=set_type,
            alpha=alpha,
            beta=beta,
            tau=tau,
            cutoff=cutoff,
            use_cache=True,
        )


print("\n======== STEP 2: 构建所有基础 bigmatrix (36-dim) ========\n")

# 所有需要的 (alpha,beta,tau,cutoff) 组合
base_configs = []

# block1 & block2：beta=2.5, tau in tau_grid
for tau in tau_grid:
    base_configs.append((alpha, beta12, float(tau), cutoff_from_tau(tau)))

# block3：beta=40, tau=5.5, cutoff=18（固定）
base_configs.append((alpha, beta3, tau3, cutoff3))

# 并行构建
Parallel(n_jobs=N_JOBS_BUILD)(
    delayed(build_bigmatrix_for_config)(a, b, t, c)
    for (a, b, t, c) in base_configs
)


# ============================================================
# STEP 3. 108 维特征加载（带内存 cache）
# ============================================================

# 简单的内存 cache，避免多次重复从磁盘加载相同的 36 维特征
_base_feature_cache = {}  # key: (set_type, beta, tau, cutoff) → (X, y)


def get_base_features(set_type: str, beta: float, tau: float, cutoff: float):
    """
    读取某一块 36 维特征矩阵，并在内存中 cache。
    返回：X (N,36), y (N,)
    """
    key = (set_type, float(beta), float(tau), float(cutoff))
    if key not in _base_feature_cache:
        X, y, _ = load_dataset_bigmatrix(set_type, alpha, beta, tau, cutoff)
        _base_feature_cache[key] = (X, y)
    return _base_feature_cache[key]


def load_combo_dataset(tau1: float, tau2: float):
    """
    根据 tau1, tau2 构造 108 维特征：
        [block1(36) | block2(36) | block3(36)]
    返回：
        X_train, y_train, X_test, y_test
    """

    c1 = cutoff_from_tau(tau1)
    c2 = cutoff_from_tau(tau2)

    # refined_only（train）
    X1_tr, y_tr_1 = get_base_features("refined_only", beta12, tau1, c1)
    X2_tr, y_tr_2 = get_base_features("refined_only", beta12, tau2, c2)
    X3_tr, y_tr_3 = get_base_features("refined_only", beta3,  tau3, cutoff3)

    # 简单 sanity check：标签应完全一致
    # 如不放心可以打开 assert（稳定后可以注释掉）
    assert np.allclose(y_tr_1, y_tr_2) and np.allclose(y_tr_1, y_tr_3), \
        "refined_only: y 不一致，请检查数据构建"

    # core（test）
    X1_te, y_te_1 = get_base_features("core", beta12, tau1, c1)
    X2_te, y_te_2 = get_base_features("core", beta12, tau2, c2)
    X3_te, y_te_3 = get_base_features("core", beta3,  tau3, cutoff3)

    assert np.allclose(y_te_1, y_te_2) and np.allclose(y_te_1, y_te_3), \
        "core: y 不一致，请检查数据构建"

    # 按列拼接成 108 维
    X_train = np.concatenate([X1_tr, X2_tr, X3_tr], axis=1)
    X_test  = np.concatenate([X1_te, X2_te, X3_te], axis=1)

    y_train = y_tr_1
    y_test  = y_te_1

    return X_train, y_train, X_test, y_test, c1, c2


# ============================================================
# STEP 4-A. RF 训练函数（108 维特征）
# ============================================================

def train_rf_combo(tau1: float, tau2: float, repeat: int):

    print(f"  -> [RF] building dataset for tau1={tau1}, tau2={tau2}")
    X_train, y_train, X_test, y_test, c1, c2 = load_combo_dataset(tau1, tau2)

    print("     Training RF (CPU)...")

    rf = RandomForestRegressor(
        n_estimators=650,
        max_depth=20,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
    )

    # rf = RandomForestRegressor(
    #     n_estimators=550,
    #     max_features=X_train.shape[1],
    #     n_jobs=-1,
    # )

    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    Rp = pearsonr(y_test, pred)[0]
    RMSE = np.sqrt(mean_squared_error(y_test, pred))

    print(f"     RF: Rp={Rp:.4f}, RMSE={RMSE:.4f}")

    return {
        "alpha": alpha,
        "beta1": beta12,   # 或者直接写成 2.5，也可以
        "beta2": beta12,
        "beta3": beta3,
        "tau1": tau1,
        "tau2": tau2,
        "tau3": tau3,
        "cutoff1": c1,
        "cutoff2": c2,
        "cutoff3": cutoff3,
        "repeat": repeat,
        "model": "rf",
        "feature_dim": 108,
        "pearson": Rp,
        "rmse": RMSE,
    }




# ============================================================
# STEP 4-B. GPU XGBoost 训练函数（108 维特征）
# ============================================================

def train_xgb_combo_gpu(tau1: float, tau2: float, repeat: int):

    print(f"  -> [XGB-GPU] building dataset for tau1={tau1}, tau2={tau2}")
    X_train, y_train, X_test, y_test, c1, c2 = load_combo_dataset(tau1, tau2)

    print("     Training XGB (GPU)...")

    # XGBoost 3.x GPU 用法：DMatrix 不带 device，GPU 在 params 里指定
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    params = {
        "tree_method": "hist",          # GPU: hist + device=cuda
        "device": "cuda",
        "max_depth": 8,
        "eta": 0.03,
        "lambda": 0.2,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "max_bin": 128,
        "objective": "reg:squarederror",
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=1400,
    )

    # params = {
    #     "tree_method": "hist",
    #     "device": "cuda",

    #     # 必需优化项（最重要）
    #     "max_depth": 7,
    #     "eta": 0.02,
    # }

    # num_boost_round = 1500

    pred = booster.predict(dtest)

    Rp = pearsonr(y_test, pred)[0]
    RMSE = np.sqrt(mean_squared_error(y_test, pred))

    print(f"     XGB(GPU): Rp={Rp:.4f}, RMSE={RMSE:.4f}")

    return {
        "alpha": alpha,
        "beta1": beta12,
        "beta2": beta12,
        "beta3": beta3,
        "tau1": tau1,
        "tau2": tau2,
        "tau3": tau3,
        "cutoff1": c1,
        "cutoff2": c2,
        "cutoff3": cutoff3,
        "repeat": repeat,
        "model": "xgb-gpu",
        "feature_dim": 108,
        "pearson": Rp,
        "rmse": RMSE,
    }


# ============================================================
# STEP 5. 主循环 + CSV 输出（支持断点恢复）
# ============================================================

# 结果文件名可以按你习惯改
out_csv = MODEL_DIR / "results_tau1_tau2_combo_exp.csv"  # :contentReference[oaicite:1]{index=1}

if out_csv.exists():
    df_done = pd.read_csv(out_csv)
    # 注意：这里假设 tau1, tau2 在 CSV 中是和 tau_grid 一致的浮点值
    df_done["key1"] = df_done[["tau1", "tau2"]].min(axis=1)
    df_done["key2"] = df_done[["tau1", "tau2"]].max(axis=1)

    done_set = set(zip(df_done["key1"], df_done["key2"],
                    df_done["repeat"], df_done["model"]))
else:
    done_set = set()

print("\n======== STEP 5: Training over (tau1, tau2) grid ========\n")

for repeat in range(repeat_times):
    for i, tau1 in enumerate(tau_grid):
        for tau2 in tau_grid[i:]:

            tau1 = float(tau1)
            tau2 = float(tau2)

            # RF
            key1 = min(tau1, tau2)
            key2 = max(tau1, tau2)
            tag_rf = (key1, key2, repeat, "rf")
            
            print(f"\n[Training RF] tau1={tau1}, tau2={tau2}, repeat={repeat}")
            if tag_rf in done_set:
                print("  -> RF tag already done, skipping...")
            else:
                rec_rf = train_rf_combo(tau1, tau2, repeat)
                pd.DataFrame([rec_rf]).to_csv(
                    out_csv,
                    mode="a",
                    header=not out_csv.exists(),
                    index=False,
                )
                done_set.add(tag_rf)

            # XGB (GPU)
            key1 = min(tau1, tau2)
            key2 = max(tau1, tau2)
            tag_xgb = (key1, key2, repeat, "xgb-gpu")

            print(f"[Training XGB-GPU] tau1={tau1}, tau2={tau2}, repeat={repeat}")
            if tag_xgb in done_set:
                print("  -> XGB tag already done, skipping...")
            else:
                rec_xgb = train_xgb_combo_gpu(tau1, tau2, repeat)
                pd.DataFrame([rec_xgb]).to_csv(
                    out_csv,
                    mode="a",
                    header=not out_csv.exists(),
                    index=False,
                )
                done_set.add(tag_xgb)

print("\n========= 全部训练结束！ =========")
print(f"结果已保存到: {out_csv}")

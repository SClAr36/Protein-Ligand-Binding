from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from utils.ri_loader_paramcached import (
    build_dataset,
    precompute_pair_cache_for_set,
)

# =====================================
# 基础路径
# =====================================
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

PROGRESS_LOG = MODEL_DIR / "training_progress.log"

# =====================================
# 参数配置（你以后会修改这里）
# =====================================
alpha = "lor"
beta = 2.5 if alpha == "exp" else 5.0
tau = 1.0

cutoff_list = list(range(5, 51))   # [5,50]
max_cutoff_global = max(cutoff_list)

repeat_times = 1

N_JOBS_CPU = 8   # 物理核心 8，逻辑 16

# =====================================
# Step 1: 并行预计算 φ pair 缓存
# =====================================
print(f"\n==== STEP 1: Precompute (alpha={alpha}, beta={beta}, tau={tau}, cmax={max_cutoff_global}) ====\n")

precompute_pair_cache_for_set(
    set_type="refined_only",
    alpha=alpha,
    beta=beta,
    tau=tau,
    max_cutoff_global=max_cutoff_global,
    n_jobs=N_JOBS_CPU,
)

precompute_pair_cache_for_set(
    set_type="core",
    alpha=alpha,
    beta=beta,
    tau=tau,
    max_cutoff_global=max_cutoff_global,
    n_jobs=N_JOBS_CPU,
)


# =====================================
# Step 2: Sweep cutoff & Train models
# =====================================

def train_and_eval(cutoff, repeat):
    print(f"\n=== Training {alpha} (repeat={repeat}): cutoff={cutoff} ===")

    # 加载缓存构建好的特征（极快）
    X_train, y_train, _ = build_dataset(
        "refined_only", alpha, beta, tau, cutoff,
        use_cache=True, max_cutoff_global=max_cutoff_global
    )
    X_test, y_test, ids_test = build_dataset(
        "core", alpha, beta, tau, cutoff,
        use_cache=True, max_cutoff_global=max_cutoff_global
    )

    models = {
        "rf": RandomForestRegressor(
            n_estimators=650,       # Enough for full convergence
            max_depth=20,           # Slightly deeper with 3600 samples
            min_samples_leaf=3,     # Good bias-variance balance for 3600 samples
            max_features="sqrt",    # √36 = 6 → best empirical choice
            bootstrap=True,
            n_jobs=-1,
            random_state=repeat,
        ),

        "lgb": lgb.LGBMRegressor(
            boosting_type="gbdt",
            objective="regression",

            n_estimators=1300,
            learning_rate=0.035,

            num_leaves=48,
            max_depth=-1,

            min_child_samples=10,        # 用这个，不用 min_data_in_leaf
            subsample=0.9,               # 保留 subsample
            subsample_freq=1,            # 保留 subsample_freq
            colsample_bytree=0.9,        # 保留 colsample_bytree
            reg_lambda=0.1,              # 使用 reg_lambda 替代 lambda_l2
            min_split_gain=0.0,          # 使用 min_split_gain 替代 min_gain_to_split

            n_jobs=-1,
            random_state=repeat,
        ),
    }

    tag = f"a{alpha}_b{beta}_t{tau}_c{cutoff}_r{repeat}"
    results = []

    for model_name, model in models.items():
        print(f"  -> Fitting {model_name} (CPU)...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        pearson = pearsonr(y_test, y_pred)[0]
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"     {model_name}: Rp={pearson:.4f}  RMSE={rmse:.4f}")

        results.append({
            "model": model_name,
            "alpha": alpha,
            "beta": beta,
            "tau": tau,
            "cutoff": cutoff,
            "repeat": repeat,
            "pearson": pearson,
            "rmse": rmse,
        })

    return results


if __name__ == "__main__":

    out_csv = MODEL_DIR / f"results_cutoff_{alpha}.csv"

    # === 断点恢复：读取已有 CSV 作为已完成记录 ===
    if out_csv.exists():
        df_done = pd.read_csv(out_csv)
        done_set = set(zip(df_done["repeat"], df_done["cutoff"]))
    else:
        done_set = set()

    print(f"\n==== STEP 2: Sweeping Cutoff ====\n")

    for repeat in range(repeat_times):
        for cutoff in cutoff_list:

            if (repeat, cutoff) in done_set:
                print(f"[跳过] 已完成 repeat={repeat}, cutoff={cutoff}")
                continue

            rec = train_and_eval(cutoff, repeat)
            if rec is None:
                raise RuntimeError(f"训练失败于 repeat={repeat}, cutoff={cutoff}")

            # === 每个 cutoff 训练成功立即写 CSV ===
            df_tmp = pd.DataFrame(rec)
            if out_csv.exists():
                df_tmp.to_csv(out_csv, mode="a", header=False, index=False)
            else:
                df_tmp.to_csv(out_csv, index=False)

    print(f"\nTraining finished (alpha={alpha}).")
    print(f"Results saved to: {out_csv}")

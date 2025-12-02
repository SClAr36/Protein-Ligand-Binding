from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from joblib import Parallel, delayed

from utils.ri_loader_paramcached import (
    load_dataset_bigmatrix,
    build_bigmatrix,
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

repeat_times = 5

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
# STEP 1.5: 并行构建 bigmatrix（并且自动跳过已存在文件）
# =====================================

def bigmatrix_exists(set_type, alpha, beta, tau, cutoff):
    """
    检查该参数组合的大矩阵是否存在且完整。
    完整性判断规则：文件存在 + 文件尺寸 > 1 KB（避免空文件/中断文件）。
    如需更严格可加入 header 校验。
    """
    from utils.ri_loader_paramcached import _bigmatrix_path

    f = _bigmatrix_path(set_type, alpha, beta, tau, cutoff)
    return f.exists() and f.stat().st_size > 1024


param_jobs = []
for cutoff in cutoff_list:
    for set_type in ["refined_only", "core"]:
        # 如果已存在则跳过
        if bigmatrix_exists(set_type, alpha, beta, tau, cutoff):
            print(f"[跳过] 已存在: {set_type} cutoff={cutoff}")
            continue

        param_jobs.append((set_type, cutoff))

print(f"\n将并行构建 {len(param_jobs)} 个缺失的大矩阵...\n")

Parallel(n_jobs=N_JOBS_CPU)(
    delayed(build_bigmatrix)(
        set_type,
        alpha,
        beta,
        tau,
        cutoff,
        use_cache=True,
        max_cutoff_global=max_cutoff_global,
    )
    for (set_type, cutoff) in param_jobs
)


# =====================================
# Step 2: Sweep cutoff & Train models
# =====================================

def train_and_eval(cutoff, repeat):
    """
    按 cutoff 训练一个 RF + 一个 XGB（GPU）
    —— 保持旧结构（方案 A），不拆分函数。
    """

    print(f"\n=== Training cutoff={cutoff}, repeat={repeat} ===\n")

    # ---------------------------------------------
    # 1. 加载大矩阵 (你的新版方式)
    # ---------------------------------------------
    X_train, y_train, _ = load_dataset_bigmatrix(
        "refined_only", alpha, beta, tau, cutoff
    )
    X_test, y_test, _ = load_dataset_bigmatrix(
        "core", alpha, beta, tau, cutoff
    )

    records = []

    # ---------------------------------------------
    # 2. RF (CPU) —— 保持和原来完全一致
    # ---------------------------------------------
    print("  -> Fitting RF (CPU)...")

    rf = RandomForestRegressor(
        n_estimators=650,
        max_depth=20,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    pred_rf = rf.predict(X_test)

    Rp_rf = pearsonr(y_test, pred_rf)[0]
    RMSE_rf = np.sqrt(mean_squared_error(y_test, pred_rf))

    print(f"     RF: Rp={Rp_rf:.4f}, RMSE={RMSE_rf:.4f}")

    records.append({
        "cutoff": cutoff,
        "repeat": repeat,
        "model": "rf",
        "pearson": Rp_rf,
        "rmse": RMSE_rf,
    })

    # ---------------------------------------------
    # 3. XGBoost (GPU) —— 新增 GPU 训练
    # ---------------------------------------------
    print("  -> Fitting XGB (GPU)...")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    params = {
        "tree_method": "hist",
        "device": "cuda",     # GPU 就靠这个
        "max_depth": 8,
        "eta": 0.03,
        "lambda": 0.2,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "max_bin": 128,
        "objective": "reg:squarederror",
    }

    booster = xgb.train(params, dtrain, num_boost_round=1400)

    pred_xgb = booster.predict(dtest)

    Rp_xgb = pearsonr(y_test, pred_xgb)[0]
    RMSE_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))

    print(f"     XGB(GPU): Rp={Rp_xgb:.4f}, RMSE={RMSE_xgb:.4f}")

    records.append({
        "cutoff": cutoff,
        "repeat": repeat,
        "model": "xgb-gpu",
        "pearson": Rp_xgb,
        "rmse": RMSE_xgb,
    })

    return records


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

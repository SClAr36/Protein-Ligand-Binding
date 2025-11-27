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
alpha = "exp"
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
    set_type="refined",
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
        "refined", alpha, beta, tau, cutoff,
        use_cache=True, max_cutoff_global=max_cutoff_global
    )
    X_test, y_test, ids_test = build_dataset(
        "core", alpha, beta, tau, cutoff,
        use_cache=True, max_cutoff_global=max_cutoff_global
    )

    models = {
        "rf": RandomForestRegressor(
            n_estimators=300,
            n_jobs=-1,
            random_state=repeat,
        ),
        "lgb": lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=55,
            n_jobs=-1,
            boosting_type="gbdt",
            random_state=repeat,
        ),
    }

    tag = f"a{alpha}_b{beta}_t{tau}_c{cutoff}_r{repeat}"
    results = []

    for model_name, model in models.items():
        print(f"  -> Fitting {model_name} (CPU)...")
        model.fit(X_train, y_train)

        model_path = MODEL_DIR / f"{model_name}_{tag}.pkl"
        joblib.dump(model, model_path)

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

    # === 加载训练进度断点 ===
    def load_progress():
        if not PROGRESS_LOG.exists():
            return set()
        done = set()
        with open(PROGRESS_LOG, "r") as f:
            for line in f:
                try:
                    r, c = map(int, line.split())
                    done.add((r, c))
                except:
                    pass
        return done

    done_set = load_progress()

    all_records = []

    print(f"\n==== STEP 2: Sweeping Cutoff ====\n")

    for repeat in range(repeat_times):
        for cutoff in cutoff_list:

            # === 断点恢复：若已完成则跳过 ===
            if (repeat, cutoff) in done_set:
                print(f"[跳过] 已完成 repeat={repeat}, cutoff={cutoff}")
                continue

            rec = train_and_eval(cutoff, repeat)

            if rec is None:
                print(f"[错误] 训练失败于 repeat={repeat}, cutoff={cutoff}，已中断")
                with open(MODEL_DIR / "training_error.log", "a") as f:
                    f.write(f"{repeat} {cutoff}\n")
                raise RuntimeError(f"训练中断于 repeat={repeat}, cutoff={cutoff}")


            # === 成功则记录日志 ===
            with open(PROGRESS_LOG, "a") as f:
                f.write(f"{repeat} {cutoff}\n")

            # === 成功则追加到结果缓存 ===
            all_records.extend(rec)


    out_csv = MODEL_DIR / f"results_cutoff_{alpha}.csv"
    pd.DataFrame(all_records).to_csv(out_csv, index=False)

    print(f"\nTraining finished (alpha={alpha}).")
    print(f"Results saved to: {out_csv}")

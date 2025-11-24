from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from utils.data_loader import build_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ===== 参数设置 =====
alpha_beta_pairs = [
    ("exp", 2.5),
    ("lor", 5.0),
]

tau = 1.0
cutoff_list = list(range(5, 50, 1))  # 5 到 50
repeat_times = 5  # ***关键新增***


def train_and_eval(alpha, beta, tau, cutoff, repeat):
    print(f"\n=== Training (repeat {repeat}): alpha={alpha}, beta={beta}, tau={tau}, cutoff={cutoff} ===")

    # 载入数据（带缓存）
    X_train, y_train, _ = build_dataset("refined", alpha, beta, tau, cutoff, use_cache=True)
    X_test, y_test, ids_test = build_dataset("core", alpha, beta, tau, cutoff, use_cache=True)

    # 定义模型
    models = {
        "rf": RandomForestRegressor(
            n_estimators=300,
            random_state=repeat,
            n_jobs=-1  # 启用所有CPU核
        ),
        "gbt": GradientBoostingRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=3,
            random_state=repeat
            # GBT 默认单核，可改 LightGBM 大提速
        ),
    }

    tag = f"a{alpha}_b{beta}_t{tau}_c{cutoff}_r{repeat}"
    records = []

    for model_name, model in models.items():
        print(f"  -> Fitting {model_name} ...")
        model.fit(X_train, y_train)

        # 保存模型
        model_path = MODEL_DIR / f"{model_name}_{tag}.pkl"
        joblib.dump(model, model_path)
        print(f"     Saved: {model_path.name}")

        # 评估性能
        y_pred = model.predict(X_test)
        pearson = pearsonr(y_test, y_pred)[0]
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"     {model_name}: Rp = {pearson:.4f}, RMSE = {rmse:.4f}")

        records.append({
            "model": model_name,
            "alpha": alpha,
            "beta": beta,
            "tau": tau,
            "cutoff": cutoff,
            "repeat": repeat,
            "pearson": pearson,
            "rmse": rmse,
        })

    return records


if __name__ == "__main__":

    all_records = []

    for repeat in range(repeat_times):
        for alpha, beta in alpha_beta_pairs:
            for cutoff in cutoff_list:
                out = train_and_eval(alpha, beta, tau, cutoff, repeat)
                all_records.extend(out)

    results_df = pd.DataFrame(all_records)
    out_csv = MODEL_DIR / "results_cutoff.csv"
    results_df.to_csv(out_csv, index=False)

    print("\nTraining finished!")
    print(f"All results saved to: {out_csv}")

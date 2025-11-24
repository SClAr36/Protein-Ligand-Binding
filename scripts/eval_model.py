import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# === 路径定义 ===
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
MODEL_DIR = ROOT / "models"

# === 测试数据加载 ===
test_data = np.load(MODEL_DIR / "test_data.npz", allow_pickle=True)
X_test = test_data["X_test"]
y_test = test_data["y_test"]
ids_test = test_data["ids_test"]


def eval_model(model_name: str):

    model_path = MODEL_DIR / f"model_{model_name}.pkl"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"\nLoading model: {model_path.name}")
    model = joblib.load(model_path)

    print("Predicting...")
    y_pred = model.predict(X_test)

    # === 评价指标 ===
    Rp = pearsonr(y_test, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{model_name}: Rp = {Rp:.4f}, RMSE = {rmse:.3f}")

    # === 输出结果存档 ===
    out_csv = MODEL_DIR / f"pred_{model_name}.csv"
    with open(out_csv, "w") as f:
        f.write("pdbid,y_true,y_pred\n")
        for pid, yt, yp in zip(ids_test, y_test, y_pred):
            f.write(f"{pid},{yt:.4f},{yp:.4f}\n")
    print(f"Saved predictions: {out_csv}")

    # === 绘图（Figure 4.1 风格） ===
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        '--', linewidth=2, label="Ideal"
    )

    plt.xlabel("Experimental pKd")
    plt.ylabel("Predicted pKd")
    plt.title(f"V2016 Core Set: {model_name} (Rp={Rp:.3f})")
    plt.grid(True)
    plt.tight_layout()

    out_png = MODEL_DIR / f"fig_v2016_{model_name}.png"
    plt.savefig(out_png, dpi=300)
    print(f"Scatter plot saved: {out_png}")


if __name__ == "__main__":
    eval_model("rf")
    eval_model("gbt")

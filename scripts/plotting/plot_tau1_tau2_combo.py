import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# 参数区：指定 alpha 与文件名
# ======================================================
alpha = "exp"     # 根据训练脚本使用的 alpha 设置
model_list = ["rf", "xgb-gpu"]   # 绘制两个模型的图

# ======================================================
# 基础路径
# ======================================================
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True)

# 训练结果 CSV
csv_file = MODEL_DIR / f"results_tau1_tau2_combo_{alpha}.csv"

# ======================================================
# 读取结果 & 取 median over repeat
# ======================================================
df = pd.read_csv(csv_file)

# 保险起见，只保留当前 alpha
if "alpha" in df.columns:
    df = df[df["alpha"] == alpha].copy()

# ------------------------------------------------------
# 规范化 tau1/tau2：折叠成无序对 (t1 <= t2)
# ------------------------------------------------------
df["t1"] = df[["tau1", "tau2"]].min(axis=1)
df["t2"] = df[["tau1", "tau2"]].max(axis=1)

# ------------------------------------------------------
# 对 (model, t1, t2) 求 median over repeat
# ------------------------------------------------------
df_med = (
    df.groupby(["model", "t1", "t2"], as_index=False)
      .agg(pearson_median=("pearson", "median"))
)

# ------------------------------------------------------
# 构建 tau 网格
# ------------------------------------------------------
taus = np.sort(df_med["t1"].unique())
N = len(taus)
tau_to_idx = {tau: i for i, tau in enumerate(taus)}


# ======================================================
# 绘图函数：tau1–tau2 热力图（对称）
# ======================================================
def plot_heatmap(df_model: pd.DataFrame, model_name: str, save_path: Path) -> None:

    # 构建 NxN 矩阵
    heat = np.full((N, N), np.nan)

    # 填右上角
    for _, row in df_model.iterrows():
        i = tau_to_idx[row["t1"]]
        j = tau_to_idx[row["t2"]]
        heat[i, j] = row["pearson_median"]

    # 镜像到左下角
    for i in range(N):
        for j in range(i):
            heat[i, j] = heat[j, i]

    # ----------- 绘图 -----------
    plt.figure(figsize=(7, 6))

    im = plt.imshow(
        heat,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap="RdYlBu_r",
    )

    # 设置 ticks
    tick_values = [1, 5, 10, 15, 20]
    xticks = [i for i,t in enumerate(taus) if t in tick_values]
    yticks = xticks

    plt.xticks(xticks, [taus[i] for i in xticks])
    plt.yticks(yticks, [taus[i] for i in yticks])
    plt.xlabel(r"$\tau_2$")
    plt.ylabel(r"$\tau_1$")
    plt.title(f"{model_name} (alpha={alpha}): heatmap of Rp (median)")

    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure: {save_path}")

    plt.show()


# ======================================================
# 分别绘制 RF 与 XGB 图
# ======================================================
for model in model_list:
    df_m = df_med[df_med["model"] == model].copy()

    out_name = f"heatmap_{model}_{alpha}.png"
    plot_heatmap(
        df_m,
        model_name=model,
        save_path=FIG_DIR / out_name,
    )

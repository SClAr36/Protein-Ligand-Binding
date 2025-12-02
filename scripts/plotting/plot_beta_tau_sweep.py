import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# 参数区：指定 alpha
# ======================================================
# 和训练脚本中的 alpha 一致，比如 "exp" 或 "lor"
alpha = "exp"   # TODO: 需要画哪个就改这里

# ======================================================
# 基础路径
# ======================================================
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True)

# 训练脚本的输出文件名格式：results_tau_beta_sweep_{alpha}.csv :contentReference[oaicite:0]{index=0}
csv_file = MODEL_DIR / f"results_tau_beta_sweep_{alpha}.csv"

# ======================================================
# 读取结果 & 取 median over repeat
# ======================================================
df = pd.read_csv(csv_file)

# 保险起见，只保留当前 alpha（一般整个文件就是这个 alpha）
if "alpha" in df.columns:
    df = df[df["alpha"] == alpha]

# 对 (model, beta, tau) 上所有 repeat 取 median
df_med = (
    df.groupby(["model", "beta", "tau"], as_index=False)
      .agg(
          pearson_median=("pearson", "median"),
          rmse_median=("rmse", "median"),  # 目前没画，但以后需要也在这
      )
)

# ======================================================
# 绘图函数：每个 beta 一条线，横轴 tau，纵轴 median Rp
# ======================================================
def plot_model(df_model: pd.DataFrame, model_name: str, save_path: Path, alpha: str) -> None:
    """
    df_model: 已经 groupby 后、包含某个模型的 (beta, tau, pearson_median) 数据
    """

    plt.figure(figsize=(7, 5))

    beta_values = sorted(df_model["beta"].unique())

    for beta in beta_values:
        sub = df_model[df_model["beta"] == beta].sort_values("tau")
        plt.plot(
            sub["tau"],
            sub["pearson_median"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"beta = {beta}",
        )

    plt.xlabel(r"$\tau$")
    plt.ylabel("Rp (median PCC)")
    plt.title(f"{model_name} (alpha = {alpha}): Rp vs tau at different beta")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure: {save_path}")

    plt.show()

# ======================================================
# RF 图（类似 Exercise1.pdf 中 Fig 4.4, 4.6）:contentReference[oaicite:1]{index=1}
# ======================================================
df_rf = df_med[df_med["model"] == "rf"]
plot_model(
    df_rf,
    "RandomForest",
    FIG_DIR / f"rf_{alpha}_beta_tau.png",
    alpha=alpha,
)

# ======================================================
# XGB 图（GBT 对应的图，类似 Fig 4.3, 4.5）:contentReference[oaicite:2]{index=2}
# ======================================================
df_xgb = df_med[df_med["model"] == "xgb-gpu"]
plot_model(
    df_xgb,
    "XGBoost (GPU)",
    FIG_DIR / f"gbt_{alpha}_beta_tau.png",
    alpha=alpha,
)

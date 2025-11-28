import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True)


def load_results(kernel):
    path = MODEL_DIR / f"results_cutoff_{kernel}.csv"
    df = pd.read_csv(path)
    return df


def compute_median_curve(df, model_name):
    # 按模型筛选（RF or lgb）
    df = df[df["model"] == model_name]

    cutoffs = sorted(df["cutoff"].unique())
    medians = []

    for c in cutoffs:
        Rp_vals = df[df["cutoff"] == c]["pearson"].values
        med = np.median(Rp_vals)
        medians.append(med)

    return cutoffs, medians


def plot_cutoff_sweep():
    df_exp = load_results("exp")
    df_lor = load_results("lor")

    # -------- 计算四条曲线 --------
    cut_exp_rf, med_exp_rf   = compute_median_curve(df_exp, "rf")
    cut_exp_lgb, med_exp_lgb = compute_median_curve(df_exp, "lgb")

    cut_lor_rf, med_lor_rf   = compute_median_curve(df_lor, "rf")
    cut_lor_lgb, med_lor_lgb = compute_median_curve(df_lor, "lgb")

    # -------- 颜色 & 样式 --------
    colors = {
        "exp_rf":  "#1f77b4",  # 蓝
        "exp_lgb": "#1f77b4",  # 蓝
        "lor_rf":  "#d62728",  # 红
        "lor_lgb": "#d62728",  # 红
    }
    styles = {
        "rf":  "--",   # 虚线
        "lgb": "-"     # 实线
    }

    fig, ax = plt.subplots(figsize=(13, 7))

    # === 绘制四条线 ===
    ax.plot(cut_exp_rf,  med_exp_rf,
            styles["rf"],  color=colors["exp_rf"],
            linewidth=2.2, label="RI(E)-RF (median)")

    ax.plot(cut_exp_lgb, med_exp_lgb,
            styles["lgb"], color=colors["exp_lgb"],
            linewidth=2.2, label="RI(E)-GBT (median)")

    ax.plot(cut_lor_rf,  med_lor_rf,
            styles["rf"],  color=colors["lor_rf"],
            linewidth=2.2, label="RI(L)-RF (median)")

    ax.plot(cut_lor_lgb, med_lor_lgb,
            styles["lgb"], color=colors["lor_lgb"],
            linewidth=2.2, label="RI(L)-GBT (median)")

    # -------- 轴与标注 --------
    ax.set_xlabel("Cutoff distance (Å)", fontsize=14)
    ax.set_ylabel("Median Pearson Rp", fontsize=14)
    ax.set_title("Figure 4.1 – Median Rp vs Cutoff (RI(E), RI(L), RF, GBT)", fontsize=16)

    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # -------- 保存 --------
    out_png = FIG_DIR / "Fig4_1_median_4curves.png"
    out_pdf = FIG_DIR / "Fig4_1_median_4curves.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")

    print("Saved:", out_png)
    print("Saved:", out_pdf)
    
    plt.show()


if __name__ == "__main__":
    plot_cutoff_sweep()

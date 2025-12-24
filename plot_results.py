# plot_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_exp_timeseries(csv_file, title_prefix, output_prefix):
    """
    通用：根据 Exp1 / Exp2 的 timeseries CSV 画两张图：
      - {output_prefix}_cvr.png
      - {output_prefix}_L0.png
    """
    if not os.path.exists(csv_file):
        print(f"[WARN] CSV file not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)

    # 只画关心的三个算法
    algos = ["dg", "nsga", "rosa"]
    algo_labels = {
        "dg": "DG ",
        "nsga": "NSGA-II ",
        "rosa": "ROSA",
    }

    # 图1：CVR over batches
    plt.figure(figsize=(6, 4))
    max_cvr = 0
    for algo in algos:
        sub = df[df["algo"] == algo].sort_values("batch_idx")
        if sub.empty:
            continue
        plt.plot(
            sub["batch_idx"],
            sub["cvr"],
            marker="o",
            linestyle="-",
            label=algo_labels.get(algo, algo),
        )
        max_cvr = max(max_cvr, sub["cvr"].max())

    # 设置y轴范围，顶部留出空间
    plt.ylim(0, max_cvr + 0.1)

    plt.xlabel("Batch index")
    plt.ylabel("CVR")
    plt.title(f"{title_prefix} - CVR over batches")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_cvr.png", dpi=300)
    plt.close()

    # 图2：L0 over batches
    plt.figure(figsize=(6, 4))
    max_L0 = 0
    for algo in algos:
        sub = df[df["algo"] == algo].sort_values("batch_idx")
        if sub.empty:
            continue
        plt.plot(
            sub["batch_idx"],
            sub["L0"],
            marker="o",
            linestyle="-",
            label=algo_labels.get(algo, algo),
        )
        max_L0 = max(max_L0, sub["L0"].max())

    # 设置y轴范围，顶部留出空间（L0按10%的margin）
    margin = max(50, max_L0 * 0.1)
    plt.ylim(0, max_L0 + margin)

    plt.xlabel("Batch index")
    plt.ylabel(r"Residual load $L_0$")
    plt.title(f"{title_prefix} - Residual load over batches")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_L0.png", dpi=300)
    plt.close()

    print(f"[INFO] Saved plots: {output_prefix}_cvr.png, {output_prefix}_L0.png")


def plot_exp3_ablation(csv_file="results_exp3_ablation.csv"):
    """
    根据 Exp3 消融实验的 CSV，仅画出 ROSA-Full / ROSA-NoRisk 的对比曲线。
    """
    if not os.path.exists(csv_file):
        print(f"[WARN] CSV file not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)

    # 仅提取 ROSA Full / NoRisk
    rosa_full = df[(df["algo"] == "rosa") & (df["variant"] == "full")].sort_values("batch_idx")
    rosa_norisk = df[(df["algo"] == "rosa") & (df["variant"] == "norisk")].sort_values("batch_idx")

    if rosa_full.empty or rosa_norisk.empty:
        print("[WARN] Not enough ROSA data (full / norisk) in exp3 CSV.")
        return

    # CVR 曲线：仅 ROSA Full / NoRisk
    plt.figure(figsize=(6, 4))

    # ROSA Full
    plt.plot(
        rosa_full["batch_idx"],
        rosa_full["cvr"],
        marker="o",
        linestyle="-",
        label="ROSA ",
    )
    # ROSA NoRisk
    plt.plot(
        rosa_norisk["batch_idx"],
        rosa_norisk["cvr"],
        marker="o",
        linestyle="-",
        label="ROSA-NoRisk",
    )

    max_cvr = max(rosa_full["cvr"].max(), rosa_norisk["cvr"].max())

    # 设置y轴范围，顶部留出空间
    plt.ylim(0, max_cvr + 0.1)

    plt.xlabel("Batch index")
    plt.ylabel("CVR")
    plt.title("Exp3 (Ablation) - CVR ")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("exp3_rosa_cvr.png", dpi=300)
    plt.close()

    # L0 曲线：仅 ROSA Full / NoRisk
    plt.figure(figsize=(6, 4))

    # ROSA Full
    plt.plot(
        rosa_full["batch_idx"],
        rosa_full["L0"],
        marker="o",
        linestyle="-",
        label="ROSA ",
    )
    # ROSA NoRisk
    plt.plot(
        rosa_norisk["batch_idx"],
        rosa_norisk["L0"],
        marker="o",
        linestyle="-",
        label="ROSA-NoRisk",
    )

    max_L0 = max(rosa_full["L0"].max(), rosa_norisk["L0"].max())

    # 设置y轴范围，顶部留出空间（L0按10%的margin）
    margin = max(50, max_L0 * 0.1)
    plt.ylim(0, max_L0 + margin)

    plt.xlabel("Batch index")
    plt.ylabel(r"Residual load $L_0$")
    plt.title("Exp3 (Ablation) - $L_0$ (ROSA vs NoRisk)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("exp3_rosa_L0.png", dpi=300)
    plt.close()

    print("[INFO] Saved plots: exp3_rosa_cvr.png, exp3_rosa_L0.png")


if __name__ == "__main__":
    # 根据 CSV 生成所有实验图
    plot_exp_timeseries(
        csv_file="results_exp1_timeseries.csv",
        title_prefix="Exp1 ",
        output_prefix="exp1"
    )

    plot_exp_timeseries(
        csv_file="results_exp2_timeseries.csv",
        title_prefix="Exp2 ",
        output_prefix="exp2"
    )

    plot_exp3_ablation("results_exp3_timeseries.csv")

    # 新增：三张 L0 均值柱状图
    def plot_l0_mean_exp1(csv_file="results_exp1_timeseries.csv", out_file="fig_exp1_L0_mean.png"):
        df = pd.read_csv(csv_file)
        df_exp = df[df["exp"] == "exp1_moderate"]
        l0_means = df_exp.groupby("algo")["L0"].mean()
        labels = ["DG", "NSGA-II", "ROSA"]
        values = [l0_means.get("dg", 0), l0_means.get("nsga", 0), l0_means.get("rosa", 0)]

        plt.figure(figsize=(5, 4))
        plt.bar(labels, values, color=["#4e79a7", "#f28e2b", "#59a14f"])
        plt.ylabel("Average L0")
        plt.title("Exp1 L0 Mean")
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"[INFO] Saved {out_file}")

    def plot_l0_mean_exp2(csv_file="results_exp2_timeseries.csv", out_file="fig_exp2_L0_mean.png"):
        df = pd.read_csv(csv_file)
        df_exp = df[df["exp"] == "exp2_high_pressure"]
        l0_means = df_exp.groupby("algo")["L0"].mean()
        labels = ["DG", "NSGA", "ROSA"]
        values = [l0_means.get("dg", 0), l0_means.get("nsga", 0), l0_means.get("rosa", 0)]

        plt.figure(figsize=(5, 4))
        plt.bar(labels, values, color=["#4e79a7", "#f28e2b", "#59a14f"])
        plt.ylabel("Average L0")
        plt.title("Exp2 L0 Mean ")
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"[INFO] Saved {out_file}")

    def plot_l0_mean_exp3(csv_file="results_exp3_timeseries.csv", out_file="fig_exp3_L0_mean.png"):
        df = pd.read_csv(csv_file)
        df_exp = df[df["exp"] == "exp3_ablation"]
        rosa_full = df_exp[(df_exp["algo"] == "rosa") & (df_exp["variant"] == "full")]["L0"].mean()
        rosa_norisk = df_exp[(df_exp["algo"] == "rosa") & (df_exp["variant"] == "norisk")]["L0"].mean()

        labels = ["ROSA", "ROSA-NoRisk"]
        values = [
            rosa_full if pd.notna(rosa_full) else 0,
            rosa_norisk if pd.notna(rosa_norisk) else 0,
        ]

        plt.figure(figsize=(5, 4))
        plt.bar(labels, values, color=["#59a14f", "#e15759"])
        plt.ylabel("Average L0")
        plt.title("Exp3 L0 Mean ")
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"[INFO] Saved {out_file}")

    plot_l0_mean_exp1()
    plot_l0_mean_exp2()
    plot_l0_mean_exp3()

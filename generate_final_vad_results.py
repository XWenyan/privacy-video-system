
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径与实验参数
# ============================================================

ROOT = Path(r"D:\privacy-video-system")

INPUT_CSV = (
    ROOT
    / "outputs"
    / "vad_results"
    / "original_vs_anonymized"
    / "fixed_memory"
    / "paired_vad_comparison_fixed.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "vad_results"
    / "original_vs_anonymized"
    / "fixed_memory"
    / "final_alpha_090"
)

ALPHA = 0.90
VALID_START = 4
TOP_RATIO = 0.10

REQUIRED_COLUMNS = [
    "frame_index",
    "frame_name",
    "original_psnr",
    "anonymized_psnr",
    "original_feature_distance",
    "anonymized_feature_distance",
]


# ============================================================
# 2. Min-Max 归一化
# ============================================================

def minmax(values):
    values = np.asarray(values, dtype=float)

    min_value = np.min(values)
    max_value = np.max(values)

    if np.isclose(max_value, min_value):
        return np.zeros_like(values)

    return (values - min_value) / (max_value - min_value)


def normalize_pair(original, anonymized, mode):
    """
    PSNR：数值越低，异常程度越高，因此反向归一化。
    Feature Distance：数值越高，异常程度越高，因此正向归一化。
    """

    original = np.asarray(original, dtype=float)
    anonymized = np.asarray(anonymized, dtype=float)

    if mode == "common":
        # 两路数据共同计算归一化范围
        combined = np.concatenate([original, anonymized])
        min_value = np.min(combined)
        max_value = np.max(combined)

        if np.isclose(max_value, min_value):
            original_norm = np.zeros_like(original)
            anonymized_norm = np.zeros_like(anonymized)
        else:
            original_norm = (original - min_value) / (max_value - min_value)
            anonymized_norm = (anonymized - min_value) / (max_value - min_value)

    elif mode == "individual":
        # 两路数据分别计算归一化范围
        original_norm = minmax(original)
        anonymized_norm = minmax(anonymized)

    else:
        raise ValueError(f"未知归一化模式：{mode}")

    return original_norm, anonymized_norm


# ============================================================
# 3. 计算四项一致性指标
# ============================================================

def calculate_metrics(original, anonymized):
    original = np.asarray(original, dtype=float)
    anonymized = np.asarray(anonymized, dtype=float)

    # Pearson 相关系数
    if np.std(original) < 1e-12 or np.std(anonymized) < 1e-12:
        pearson = np.nan
    else:
        pearson = np.corrcoef(original, anonymized)[0, 1]

    # Spearman：先计算秩，再计算 Pearson
    original_rank = pd.Series(original).rank(method="average").to_numpy()
    anonymized_rank = pd.Series(anonymized).rank(method="average").to_numpy()

    if (
        np.std(original_rank) < 1e-12
        or np.std(anonymized_rank) < 1e-12
    ):
        spearman = np.nan
    else:
        spearman = np.corrcoef(original_rank, anonymized_rank)[0, 1]

    # 平均绝对误差
    mae = np.mean(np.abs(original - anonymized))

    # Top 10% 高分帧 Jaccard
    count = len(original)
    top_k = max(1, int(np.ceil(count * TOP_RATIO)))

    original_top = set(np.argsort(original)[-top_k:])
    anonymized_top = set(np.argsort(anonymized)[-top_k:])

    intersection = len(original_top & anonymized_top)
    union = len(original_top | anonymized_top)

    jaccard = intersection / union if union else np.nan

    return {
        "pearson": pearson,
        "spearman": spearman,
        "mae": mae,
        "top10_jaccard": jaccard,
        "valid_frame_count": count,
        "top_k": top_k,
    }


# ============================================================
# 4. 绘制异常分数曲线
# ============================================================

def plot_curves(results, mode, metrics, output_path):
    frame_indices = results["frame_index"].to_numpy()

    original = results[f"original_score_{mode}"].to_numpy()
    anonymized = results[f"anonymized_score_{mode}"].to_numpy()

    fig, axes = plt.subplots(
        2, 1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # 上图：两路异常分数叠加
    axes[0].plot(
        frame_indices,
        original,
        label="Original Video",
        linewidth=1.1,
    )
    axes[0].plot(
        frame_indices,
        anonymized,
        label="Anonymized Video",
        linewidth=1.1,
        alpha=0.85,
    )

    axes[0].set_title(
        f"VAD Anomaly Score Comparison ({mode.capitalize()}, ALPHA={ALPHA:.2f})"
    )
    axes[0].set_ylabel("Anomaly Score")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # 下图：两路分数差值
    difference = original - anonymized

    axes[1].plot(
        frame_indices,
        difference,
        label="Original - Anonymized",
        linewidth=0.9,
    )
    axes[1].axhline(0, linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Frame Index")
    axes[1].set_ylabel("Score Difference")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    metric_text = (
        f"Pearson: {metrics['pearson']:.4f}    "
        f"Spearman: {metrics['spearman']:.4f}    "
        f"MAE: {metrics['mae']:.4f}    "
        f"Top10% Jaccard: {metrics['top10_jaccard']:.4f}"
    )

    fig.text(
        0.5,
        0.015,
        metric_text,
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. 主流程
# ============================================================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"输入 CSV 缺少必要列：{missing}\n"
            f"当前 CSV 列名：{df.columns.tolist()}"
        )

    # 按帧编号排序，确保曲线顺序正确
    df = df.sort_values("frame_index").reset_index(drop=True)

    # 检查数值列
    numeric_columns = [
        "original_psnr",
        "anonymized_psnr",
        "original_feature_distance",
        "anonymized_feature_distance",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 只使用两路分数均有效的帧
    valid_mask = (
        df[numeric_columns].notna().all(axis=1)
        & (df.index >= VALID_START)
    )

    valid = df.loc[valid_mask].copy().reset_index(drop=True)

    if len(valid) == 0:
        raise ValueError("没有有效预测帧，请检查 CSV 数据。")

    print(f"输入帧总数：{len(df)}")
    print(f"有效预测帧数：{len(valid)}")
    print(f"ALPHA：{ALPHA:.2f}")

    # ========================================================
    # 分别计算 common 和 individual
    # ========================================================

    metrics_rows = []

    for mode in ["common", "individual"]:

        # PSNR 反向归一化
        original_psnr, anonymized_psnr = normalize_pair(
            valid["original_psnr"],
            valid["anonymized_psnr"],
            mode,
        )

        original_psnr = 1.0 - original_psnr
        anonymized_psnr = 1.0 - anonymized_psnr

        # Feature Distance 正向归一化
        original_feature, anonymized_feature = normalize_pair(
            valid["original_feature_distance"],
            valid["anonymized_feature_distance"],
            mode,
        )

        # 双分量加权融合
        original_score = (
            ALPHA * original_psnr
            + (1.0 - ALPHA) * original_feature
        )

        anonymized_score = (
            ALPHA * anonymized_psnr
            + (1.0 - ALPHA) * anonymized_feature
        )

        # 保存逐帧中间结果
        valid[f"original_psnr_anomaly_{mode}"] = original_psnr
        valid[f"anonymized_psnr_anomaly_{mode}"] = anonymized_psnr

        valid[f"original_feature_anomaly_{mode}"] = original_feature
        valid[f"anonymized_feature_anomaly_{mode}"] = anonymized_feature

        valid[f"original_score_{mode}"] = original_score
        valid[f"anonymized_score_{mode}"] = anonymized_score

        metrics = calculate_metrics(
            original_score,
            anonymized_score,
        )

        metrics_rows.append({
            "alpha": ALPHA,
            "scale": mode,
            **metrics,
        })

        # 每种尺度分别生成曲线
        plot_curves(
            valid,
            mode,
            metrics,
            OUTPUT_DIR / f"vad_score_comparison_{mode}.png",
        )

        print(f"\n[{mode.upper()}]")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    # ========================================================
    # 6. 保存结果文件
    # ========================================================

    # 逐帧分数及归一化中间结果
    valid.to_csv(
        OUTPUT_DIR / "final_vad_scores_alpha_090.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 两种归一化尺度的四项指标
    metrics_df = pd.DataFrame(metrics_rows)

    metrics_df.to_csv(
        OUTPUT_DIR / "final_vad_metrics_alpha_090.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 单独导出桌面 UI 常用的 individual 分数
    ui_scores = valid[
        [
            "frame_index",
            "frame_name",
            "original_score_individual",
            "anonymized_score_individual",
        ]
    ].copy()

    ui_scores = ui_scores.rename(columns={
        "original_score_individual": "original_anomaly_score",
        "anonymized_score_individual": "anonymized_anomaly_score",
    })

    ui_scores.to_csv(
        OUTPUT_DIR / "ui_vad_scores_alpha_090.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== 实验完成 ==========")
    print(f"结果目录：{OUTPUT_DIR}")
    print("1. final_vad_scores_alpha_090.csv")
    print("2. final_vad_metrics_alpha_090.csv")
    print("3. ui_vad_scores_alpha_090.csv")
    print("4. vad_score_comparison_common.png")
    print("5. vad_score_comparison_individual.png")


if __name__ == "__main__":
    main()

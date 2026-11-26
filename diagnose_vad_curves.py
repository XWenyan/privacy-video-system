
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 实验路径和参数
# ============================================================

ROOT = Path(r"D:\privacy-video-system")

INPUT_CSV = (
    ROOT
    / "outputs"
    / "vad_results"
    / "original_vs_anonymized"
    / "fixed_memory"
    / "final_alpha_090"
    / "final_vad_scores_alpha_090.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "vad_results"
    / "original_vs_anonymized"
    / "fixed_memory"
    / "final_alpha_090"
    / "curve_diagnosis"
)

# 主要分析 Individual 分数
SCORE_MODE = "individual"

# 滑动平均窗口：31 帧
SMOOTH_WINDOW = 31

# 滑动 Pearson 窗口：100 帧
CORR_WINDOW = 100

# 每隔 20 帧计算一个窗口
CORR_STEP = 20

# 相关系数计算的最少有效帧数
MIN_VALID_POINTS = 30


# ============================================================
# 2. 数据读取和检查
# ============================================================

def load_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"找不到输入文件：{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required = [
        "frame_index",
        f"original_score_{SCORE_MODE}",
        f"anonymized_score_{SCORE_MODE}",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"CSV 缺少必要列：{missing}\n"
            f"当前列名：{df.columns.tolist()}"
        )

    df = df.sort_values("frame_index").reset_index(drop=True)

    for column in required:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(subset=required).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("没有可用于诊断的有效数据。")

    return df


# ============================================================
# 3. 计算基础统计信息
# ============================================================

def calculate_statistics(original, anonymized):
    difference = original - anonymized

    pearson = (
        np.corrcoef(original, anonymized)[0, 1]
        if np.std(original) > 1e-12
        and np.std(anonymized) > 1e-12
        else np.nan
    )

    stats = {
        "valid_frame_count": len(original),
        "pearson": pearson,
        "mae": np.mean(np.abs(difference)),
        "difference_mean": np.mean(difference),
        "difference_std": np.std(difference),
        "difference_median": np.median(difference),
        "difference_q05": np.quantile(difference, 0.05),
        "difference_q25": np.quantile(difference, 0.25),
        "difference_q75": np.quantile(difference, 0.75),
        "difference_q95": np.quantile(difference, 0.95),
        "original_mean": np.mean(original),
        "original_std": np.std(original),
        "original_min": np.min(original),
        "original_max": np.max(original),
        "anonymized_mean": np.mean(anonymized),
        "anonymized_std": np.std(anonymized),
        "anonymized_min": np.min(anonymized),
        "anonymized_max": np.max(anonymized),
    }

    return stats, difference


# ============================================================
# 4. 图一：原始分数分布
# ============================================================

def plot_score_distribution(original, anonymized):
    plt.figure(figsize=(10, 6))

    plt.hist(
        original,
        bins=50,
        alpha=0.55,
        density=True,
        label="Original",
    )

    plt.hist(
        anonymized,
        bins=50,
        alpha=0.55,
        density=True,
        label="Anonymized",
    )

    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.title("Anomaly Score Distribution")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    path = OUTPUT_DIR / "01_score_distribution.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"已生成：{path.name}")


# ============================================================
# 5. 图二：滑动平均曲线
# ============================================================

def plot_rolling_mean(frame_index, original, anonymized):
    original_smooth = (
        pd.Series(original)
        .rolling(
            window=SMOOTH_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    anonymized_smooth = (
        pd.Series(anonymized)
        .rolling(
            window=SMOOTH_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    plt.figure(figsize=(15, 6))

    plt.plot(
        frame_index,
        original_smooth,
        label=f"Original ({SMOOTH_WINDOW}-frame mean)",
        linewidth=1.5,
    )

    plt.plot(
        frame_index,
        anonymized_smooth,
        label=f"Anonymized ({SMOOTH_WINDOW}-frame mean)",
        linewidth=1.5,
    )

    plt.xlabel("Frame Index")
    plt.ylabel("Smoothed Anomaly Score")
    plt.title("Rolling Average Anomaly Score")
    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()

    path = OUTPUT_DIR / "02_rolling_average.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"已生成：{path.name}")


# ============================================================
# 6. 计算滑动窗口 Pearson
# ============================================================

def calculate_rolling_pearson(
    frame_index,
    original,
    anonymized,
):
    rows = []

    total = len(original)

    for start in range(0, total - CORR_WINDOW + 1, CORR_STEP):
        end = start + CORR_WINDOW

        x = original[start:end]
        y = anonymized[start:end]

        if (
            len(x) < MIN_VALID_POINTS
            or np.std(x) < 1e-12
            or np.std(y) < 1e-12
        ):
            pearson = np.nan
        else:
            pearson = np.corrcoef(x, y)[0, 1]

        rows.append({
            "start_frame": frame_index[start],
            "end_frame": frame_index[end - 1],
            "center_frame": (
                frame_index[start] + frame_index[end - 1]
            ) / 2,
            "pearson": pearson,
        })

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            f"有效帧不足 {CORR_WINDOW}，无法计算滑动窗口 Pearson。"
        )

    return result


# ============================================================
# 7. 图三：滑动窗口 Pearson
# ============================================================

def plot_rolling_pearson(rolling_result):
    plt.figure(figsize=(15, 6))

    plt.plot(
        rolling_result["center_frame"],
        rolling_result["pearson"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Rolling Pearson",
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.axhline(
        0.3,
        linestyle=":",
        linewidth=1,
        label="Reference: 0.3",
    )

    plt.xlabel("Frame Index")
    plt.ylabel("Pearson Correlation")
    plt.title(
        f"Rolling Pearson Correlation "
        f"(Window={CORR_WINDOW}, Step={CORR_STEP})"
    )

    plt.ylim(-1.05, 1.05)
    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()

    path = OUTPUT_DIR / "03_rolling_pearson.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"已生成：{path.name}")


# ============================================================
# 8. 图四：分数差值分布
# ============================================================

def plot_difference_distribution(difference):
    plt.figure(figsize=(10, 6))

    plt.hist(
        difference,
        bins=60,
        alpha=0.8,
        density=True,
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1.2,
        label="Difference = 0",
    )

    plt.axvline(
        np.mean(difference),
        linestyle=":",
        linewidth=1.5,
        label=f"Mean = {np.mean(difference):.4f}",
    )

    plt.xlabel("Original Score - Anonymized Score")
    plt.ylabel("Density")
    plt.title("Anomaly Score Difference Distribution")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    path = OUTPUT_DIR / "04_difference_distribution.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"已生成：{path.name}")


# ============================================================
# 9. 输出诊断统计报告
# ============================================================

def save_report(stats, rolling_result):
    rolling_values = rolling_result["pearson"].dropna()

    if len(rolling_values) > 0:
        rolling_stats = {
            "rolling_pearson_mean": rolling_values.mean(),
            "rolling_pearson_median": rolling_values.median(),
            "rolling_pearson_min": rolling_values.min(),
            "rolling_pearson_max": rolling_values.max(),
            "positive_window_ratio": (
                rolling_values > 0
            ).mean(),
            "windows_below_zero": (
                rolling_values < 0
            ).sum(),
            "windows_above_0_3": (
                rolling_values > 0.3
            ).sum(),
            "window_count": len(rolling_values),
        }
    else:
        rolling_stats = {}

    all_stats = {**stats, **rolling_stats}

    report_path = OUTPUT_DIR / "diagnosis_summary.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("VAD 曲线诊断报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"归一化方式：{SCORE_MODE}\n")
        f.write(f"滑动平均窗口：{SMOOTH_WINDOW}\n")
        f.write(f"Pearson 窗口：{CORR_WINDOW}\n")
        f.write(f"Pearson 步长：{CORR_STEP}\n\n")

        for key, value in all_stats.items():
            f.write(f"{key}: {value}\n")

    rolling_result.to_csv(
        OUTPUT_DIR / "rolling_pearson_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return all_stats


# ============================================================
# 10. 主流程
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    frame_index = df["frame_index"].to_numpy()

    original = df[
        f"original_score_{SCORE_MODE}"
    ].to_numpy(dtype=float)

    anonymized = df[
        f"anonymized_score_{SCORE_MODE}"
    ].to_numpy(dtype=float)

    print("=" * 55)
    print("VAD 曲线诊断开始")
    print(f"有效帧数：{len(df)}")
    print(f"归一化方式：{SCORE_MODE}")
    print("=" * 55)

    stats, difference = calculate_statistics(
        original,
        anonymized,
    )

    # 图一
    plot_score_distribution(original, anonymized)

    # 图二
    plot_rolling_mean(
        frame_index,
        original,
        anonymized,
    )

    # 滑动窗口 Pearson
    rolling_result = calculate_rolling_pearson(
        frame_index,
        original,
        anonymized,
    )

    # 图三
    plot_rolling_pearson(rolling_result)

    # 图四
    plot_difference_distribution(difference)

    # 输出统计报告
    all_stats = save_report(stats, rolling_result)

    print("\n========== 全局统计 ==========")

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n========== 滑动 Pearson 统计 ==========")

    for key, value in all_stats.items():
        if key.startswith("rolling_") or key in [
            "positive_window_ratio",
            "windows_below_zero",
            "windows_above_0_3",
            "window_count",
        ]:
            print(f"{key}: {value}")

    print("\n========== 诊断完成 ==========")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

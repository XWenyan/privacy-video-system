
# -*- coding: utf-8 -*-
"""基于已有双路 VAD 结果，扫描记忆更新阈值。"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\privacy-video-system")

RESULT_DIR = (
    ROOT / "outputs" / "vad_results" / "original_vs_anonymized"
)

INPUT_CSV = RESULT_DIR / "paired_vad_comparison.csv"
OUTPUT_CSV = RESULT_DIR / "threshold_scan_results.csv"

# 读取已有逐帧 Point Score
df = pd.read_csv(INPUT_CSV)

required_columns = [
    "frame_index",
    "original_point_score",
    "anonymized_point_score",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"CSV 缺少必要列：{column}")

original = df["original_point_score"].dropna().to_numpy()
anonymized = df["anonymized_point_score"].dropna().to_numpy()

if len(original) == 0 or len(anonymized) == 0:
    raise ValueError("没有找到有效 Point Score 数据。")

if len(original) != len(anonymized):
    raise ValueError("两路有效 Point Score 数量不一致。")

# 合并两路数据，根据实际分布生成候选阈值
combined = np.concatenate([original, anonymized])

quantiles = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]

thresholds = sorted(set(
    [0.01] + [
        float(np.quantile(combined, q / 100))
        for q in quantiles
    ]
))

print("=" * 65)
print("Point Score 分布")
print("=" * 65)

for name, values in [
    ("Original", original),
    ("Anonymized", anonymized),
    ("Combined", combined),
]:
    print(
        f"{name:12s} "
        f"min={np.min(values):.6f} "
        f"max={np.max(values):.6f} "
        f"mean={np.mean(values):.6f} "
        f"std={np.std(values):.6f}"
    )

print("\n分位数：")
for q in quantiles:
    print(
        f"P{q:3d}: "
        f"{np.quantile(combined, q / 100):.6f}"
    )

# 统计各阈值下两路的更新触发情况
rows = []

for threshold in thresholds:
    original_mask = original < threshold
    anonymized_mask = anonymized < threshold

    original_indices = set(np.where(original_mask)[0])
    anonymized_indices = set(np.where(anonymized_mask)[0])

    union = original_indices | anonymized_indices
    intersection = original_indices & anonymized_indices

    overlap = (
        len(intersection) / len(union)
        if union else np.nan
    )

    rows.append({
        "threshold": threshold,

        "original_update_count": int(original_mask.sum()),
        "original_update_ratio": float(original_mask.mean()),

        "anonymized_update_count": int(anonymized_mask.sum()),
        "anonymized_update_ratio": float(anonymized_mask.mean()),

        "update_frame_jaccard": overlap,
    })

result = pd.DataFrame(rows)

result.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 65)
print("阈值扫描结果")
print("=" * 65)

print(
    result.to_string(
        index=False,
        formatters={
            "threshold": "{:.6f}".format,
            "original_update_ratio": "{:.2%}".format,
            "anonymized_update_ratio": "{:.2%}".format,
            "update_frame_jaccard": (
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            ),
        }
    )
)

print(f"\n结果已保存：{OUTPUT_CSV}")

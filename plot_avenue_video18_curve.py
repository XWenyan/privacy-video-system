# -*- coding: utf-8 -*-
"""Plot the per-frame anomaly response curve from the FT inference CSV.

This is intentionally a separate lightweight process: it does not import
PyTorch, avoiding the OpenMP DLL collision previously seen in the project.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制Avenue视频18的FT异常响应曲线")
    parser.add_argument(
        "--csv", type=Path,
        default=Path(r"D:\privacy-video-system\outputs\avenue_eval_18\video_18_anomaly_scores.csv")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(r"D:\privacy-video-system\outputs\avenue_eval_18\video_18_anomaly_curve.png")
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise FileNotFoundError(f"找不到推理CSV：{args.csv}。请先运行FT推理脚本。")

    indices, scores = [], []
    with args.csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"prediction_index", "anomaly_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV缺少字段：{sorted(missing)}")
        for row in reader:
            indices.append(int(row["prediction_index"]))
            scores.append(float(row["anomaly_score"]))

    if not scores:
        raise ValueError("CSV中没有异常分数记录。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(indices, scores, linewidth=1.25, label="Anomaly score")
    ax.set_title("FT Anomaly Response on Anonymized Video")
    ax.set_xlabel("Prediction index (first four frames used as input)")
    ax.set_ylabel("Anomaly score")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] 异常响应曲线已保存：{args.output}")
    print(f"[INFO] 绘制点数：{len(scores)}")


if __name__ == "__main__":
    main()

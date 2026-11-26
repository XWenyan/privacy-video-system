
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np


# ==================== 路径配置 ====================

ROOT = Path(__file__).resolve().parent

ORIGINAL_ROOT = ROOT / "outputs" / "extracted_frames"
ANONYMIZED_ROOT = ROOT / "outputs" / "anonymized_frames"

OUTPUT_DIR = ROOT / "outputs" / "image_quality_18"

VIDEO_ID = "18"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# SSIM参数
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
MAX_PIXEL_VALUE = 255.0


# ==================== 帧文件处理 ====================

def natural_key(name):
    """自然排序，避免frame10排在frame2前面。"""
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", name)
    ]


def get_images(folder):
    if not folder.is_dir():
        return []

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(files, key=lambda p: natural_key(p.name))


def resolve_frame_folder(root, description):
    """
    支持三种结构：
    1. root目录直接包含图片
    2. root/video/包含图片
    3. root/18/包含图片
    """

    if not root.exists():
        raise FileNotFoundError(f"{description}目录不存在：{root}")

    if get_images(root):
        return root

    subdirs = [
        p for p in root.iterdir()
        if p.is_dir() and get_images(p)
    ]

    if not subdirs:
        raise FileNotFoundError(
            f"{description}目录中没有找到图片：{root}"
        )

    matches = []

    for folder in subdirs:
        numbers = re.findall(r"\d+", folder.name)

        if numbers and str(int(numbers[-1])) == str(int(VIDEO_ID)):
            matches.append(folder)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"找到多个可能对应视频18的目录：{matches}"
        )

    if len(subdirs) == 1:
        return subdirs[0]

    raise RuntimeError(
        f"无法确定{description}的视频18目录，请检查文件夹结构。"
    )


def build_frame_map(folder, description):
    files = get_images(folder)

    if not files:
        raise RuntimeError(f"{description}没有图片：{folder}")

    mapping = {}

    for path in files:
        if path.stem in mapping:
            raise RuntimeError(
                f"{description}存在重复帧编号：{path.stem}"
            )

        mapping[path.stem] = path

    return mapping


# ==================== PSNR ====================

def calculate_psnr(original, anonymized):
    """
    计算原始帧与匿名化帧之间的PSNR。

    PSNR越高，表示两张图像的像素误差越小。
    """

    original = original.astype(np.float64)
    anonymized = anonymized.astype(np.float64)

    mse = np.mean((original - anonymized) ** 2)

    if mse == 0:
        return float("inf"), float(mse)

    psnr = 10 * math.log10(
        (MAX_PIXEL_VALUE ** 2) / mse
    )

    return float(psnr), float(mse)


# ==================== SSIM ====================

def calculate_ssim(original, anonymized):
    """
    基于Gaussian窗口计算SSIM。

    窗口：11x11
    Sigma：1.5
    Data range：255

    对BGR三个通道分别计算，最后取平均。
    """

    x = original.astype(np.float64)
    y = anonymized.astype(np.float64)

    mu_x = cv2.GaussianBlur(
        x,
        (SSIM_WINDOW, SSIM_WINDOW),
        SSIM_SIGMA
    )

    mu_y = cv2.GaussianBlur(
        y,
        (SSIM_WINDOW, SSIM_WINDOW),
        SSIM_SIGMA
    )

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        cv2.GaussianBlur(
            x * x,
            (SSIM_WINDOW, SSIM_WINDOW),
            SSIM_SIGMA
        ) - mu_x_sq
    )

    sigma_y_sq = (
        cv2.GaussianBlur(
            y * y,
            (SSIM_WINDOW, SSIM_WINDOW),
            SSIM_SIGMA
        ) - mu_y_sq
    )

    sigma_xy = (
        cv2.GaussianBlur(
            x * y,
            (SSIM_WINDOW, SSIM_WINDOW),
            SSIM_SIGMA
        ) - mu_xy
    )

    c1 = (0.01 * MAX_PIXEL_VALUE) ** 2
    c2 = (0.03 * MAX_PIXEL_VALUE) ** 2

    numerator = (
        (2 * mu_xy + c1) *
        (2 * sigma_xy + c2)
    )

    denominator = (
        (mu_x_sq + mu_y_sq + c1) *
        (sigma_x_sq + sigma_y_sq + c2)
    )

    ssim_map = numerator / np.maximum(
        denominator, 1e-12
    )

    # 排除边缘区域
    pad = SSIM_WINDOW // 2

    if (
        ssim_map.shape[0] > 2 * pad
        and ssim_map.shape[1] > 2 * pad
    ):
        ssim_map = ssim_map[
            pad:-pad,
            pad:-pad,
            :
        ]

    return float(np.mean(ssim_map))


# ==================== 主程序 ====================

def main():

    print("========== Avenue Video 18 PSNR / SSIM ==========")

    # 1. 定位帧目录
    original_dir = resolve_frame_folder(
        ORIGINAL_ROOT, "原始视频"
    )

    anonymized_dir = resolve_frame_folder(
        ANONYMIZED_ROOT, "匿名化视频"
    )

    print("[INFO] 原始帧目录：", original_dir)
    print("[INFO] 匿名化帧目录：", anonymized_dir)

    # 2. 建立帧编号映射
    original_map = build_frame_map(
        original_dir, "原始帧"
    )

    anonymized_map = build_frame_map(
        anonymized_dir, "匿名化帧"
    )

    print("[INFO] 原始帧数量：", len(original_map))
    print("[INFO] 匿名化帧数量：", len(anonymized_map))

    # 3. 检查帧编号是否一致
    original_ids = set(original_map.keys())
    anonymized_ids = set(anonymized_map.keys())

    if original_ids != anonymized_ids:

        missing = original_ids - anonymized_ids
        extra = anonymized_ids - original_ids

        raise RuntimeError(
            "原始帧与匿名化帧编号不一致！\n"
            f"匿名化侧缺少：{sorted(missing)}\n"
            f"匿名化侧多出：{sorted(extra)}"
        )

    frame_ids = sorted(
        original_ids,
        key=natural_key
    )

    print("[INFO] 成功配对帧数：", len(frame_ids))

    # 4. 逐帧计算
    results = []

    psnr_values = []
    ssim_values = []
    mse_values = []

    for index, frame_id in enumerate(frame_ids, start=1):

        original_path = original_map[frame_id]
        anonymized_path = anonymized_map[frame_id]

        original = cv2.imread(str(original_path))
        anonymized = cv2.imread(str(anonymized_path))

        if original is None or anonymized is None:
            raise RuntimeError(
                f"第{frame_id}帧读取失败。"
            )

        # 不自动缩放，避免改变PSNR/SSIM结果
        if original.shape != anonymized.shape:
            raise RuntimeError(
                f"第{frame_id}帧尺寸不一致：\n"
                f"原始：{original.shape}\n"
                f"匿名化：{anonymized.shape}"
            )

        psnr, mse = calculate_psnr(
            original, anonymized
        )

        ssim = calculate_ssim(
            original, anonymized
        )

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        mse_values.append(mse)

        results.append({
            "frame_index": index,
            "frame_id": frame_id,
            "original_file": original_path.name,
            "anonymized_file": anonymized_path.name,
            "mse": mse,
            "psnr_db": psnr,
            "ssim": ssim
        })

        if index % 50 == 0 or index == len(frame_ids):
            print(
                f"[METRIC] {index}/{len(frame_ids)} "
                "帧计算完成"
            )

    # 5. 统计结果
    psnr_mean = float(np.mean(psnr_values))
    psnr_std = float(np.std(psnr_values))

    ssim_mean = float(np.mean(ssim_values))
    ssim_std = float(np.std(ssim_values))

    pooled_mse = float(np.mean(mse_values))

    pooled_psnr = (
        float("inf")
        if pooled_mse == 0
        else 10 * math.log10(
            MAX_PIXEL_VALUE ** 2 / pooled_mse
        )
    )

    # 6. 保存CSV
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        OUTPUT_DIR /
        f"video_{VIDEO_ID}_psnr_ssim.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "frame_id",
                "original_file",
                "anonymized_file",
                "mse",
                "psnr_db",
                "ssim"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # 7. 保存摘要
    summary_path = (
        OUTPUT_DIR /
        f"video_{VIDEO_ID}_psnr_ssim_summary.txt"
    )

    summary = f"""
Video: {VIDEO_ID}.avi
Paired frames: {len(frame_ids)}

PSNR mean per frame: {psnr_mean:.6f} dB
PSNR std: {psnr_std:.6f} dB
PSNR min: {np.min(psnr_values):.6f} dB
PSNR max: {np.max(psnr_values):.6f} dB

Pooled PSNR from global mean MSE: {pooled_psnr:.6f} dB

SSIM mean: {ssim_mean:.6f}
SSIM std: {ssim_std:.6f}
SSIM min: {np.min(ssim_values):.6f}
SSIM max: {np.max(ssim_values):.6f}

SSIM window: 11x11
SSIM sigma: 1.5
Data range: 255
Pairing: exact filename stem
"""

    summary_path.write_text(
        summary,
        encoding="utf-8"
    )

    # 8. 输出结果
    print("\n========== 评价结果 ==========")

    print(f"视频：{VIDEO_ID}.avi")
    print(f"配对帧数：{len(frame_ids)}")

    print(f"PSNR均值：{psnr_mean:.6f} dB")
    print(f"PSNR标准差：{psnr_std:.6f} dB")
    print(f"全局平均MSE对应PSNR：{pooled_psnr:.6f} dB")

    print(f"SSIM均值：{ssim_mean:.6f}")
    print(f"SSIM标准差：{ssim_std:.6f}")

    print(f"\n[OUTPUT] CSV：{csv_path}")
    print(f"[OUTPUT] 摘要：{summary_path}")


if __name__ == "__main__":
    main()

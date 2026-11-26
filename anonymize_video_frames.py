import os
import re
import time
import cv2
import torch
import numpy as np

from pathlib import Path
from models.fa_model import load_fa_model


# ==============================
# 1. 路径与配置
# ==============================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "weights" / "fa_model_36.pth"

INPUT_DIR = BASE_DIR / "outputs" / "extracted_frames"

OUTPUT_DIR = BASE_DIR / "outputs" / "anonymized_frames"

DEVICE = "cpu"

torch.set_num_threads(4)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ==============================
# 2. 获取视频帧
# ==============================

def natural_sort_key(path):
    """自然排序，避免 frame_10 排在 frame_2 前面。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_image_files():
    return sorted(
        [
            path for path in INPUT_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=natural_sort_key
    )


# ==============================
# 3. 图片预处理
# ==============================

def preprocess_image(image_bgr):
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    image_resized = cv2.resize(
        image_rgb,
        (128, 128)
    )

    image_tensor = torch.from_numpy(
        image_resized.astype(np.float32) / 255.0
    )

    image_tensor = image_tensor.permute(
        2, 0, 1
    ).unsqueeze(0)

    return image_tensor


# ==============================
# 4. 单帧匿名化
# ==============================

def anonymize_frame(model, image_bgr):
    height, width = image_bgr.shape[:2]

    image_tensor = preprocess_image(image_bgr)

    with torch.inference_mode():
        output = model(image_tensor.to(DEVICE))

    output = output.squeeze(0).cpu().numpy()
    output = np.transpose(output, (1, 2, 0))

    output = np.clip(
        output * 255.0,
        0,
        255
    ).astype(np.uint8)

    # 模型输出 RGB，先转成 BGR
    output_bgr = cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )

    # 恢复到原始视频帧尺寸
    output_bgr = cv2.resize(
        output_bgr,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    return output_bgr


# ==============================
# 5. 批量处理入口
# ==============================

def main():

    print("=" * 50)
    print("FA 视频帧批量匿名化")
    print("=" * 50)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型权重不存在：{MODEL_PATH}"
        )

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"抽帧目录不存在：{INPUT_DIR}"
        )

    image_files = get_image_files()

    total = len(image_files)

    if total == 0:
        raise RuntimeError(
            f"没有找到视频帧：{INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 清理上一次生成的图片
    for path in OUTPUT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            path.unlink()

    print(f"待处理帧数：{total}")
    print(f"推理设备：{DEVICE}")

    # 加载模型，只加载一次
    start_load = time.time()

    model = load_fa_model(
        str(MODEL_PATH),
        device=DEVICE
    )

    print(
        f"模型加载完成，耗时："
        f"{time.time() - start_load:.3f} 秒"
    )

    # 批量处理
    start_time = time.time()

    processed = 0
    failed = 0
    first_comparison = None

    for index, image_path in enumerate(image_files, start=1):

        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            print(f"[跳过] 图片读取失败：{image_path.name}")
            failed += 1
            continue

        try:
            anonymized_bgr = anonymize_frame(
                model,
                image_bgr
            )

            output_path = OUTPUT_DIR / (
                image_path.stem + ".png"
            )

            success = cv2.imwrite(
                str(output_path),
                anonymized_bgr
            )

            if not success:
                raise RuntimeError("匿名化结果保存失败")

            processed += 1

            # 保存第一帧对比图，方便检查效果
            if first_comparison is None:
                first_comparison = np.hstack(
                    [image_bgr, anonymized_bgr]
                )

            # 每处理 50 帧输出一次进度
            if index % 50 == 0 or index == total:
                print(
                    f"处理进度：{index}/{total} "
                    f"（成功 {processed}，失败 {failed}）"
                )

        except Exception as exc:
            print(
                f"[失败] {image_path.name}：{exc}"
            )
            failed += 1

    elapsed = time.time() - start_time

    # 保存首帧对比图
    if first_comparison is not None:
        comparison_path = (
            OUTPUT_DIR / "comparison_first_frame.jpg"
        )

        cv2.imwrite(
            str(comparison_path),
            first_comparison
        )

    print("\n========== 批量处理结果 ==========")
    print(f"总帧数：{total}")
    print(f"成功处理：{processed}")
    print(f"处理失败：{failed}")
    print(f"总耗时：{elapsed:.2f} 秒")

    if processed > 0:
        print(
            f"平均每帧耗时："
            f"{elapsed / processed:.3f} 秒"
        )

    print(f"匿名化结果目录：{OUTPUT_DIR}")
    print("处理完成！")


if __name__ == "__main__":
    main()

import re
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import torch
import numpy as np

from models.fa_model import load_fa_model


# ==================================================
# 1. 路径与配置
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "weights" / "fa_model_36.pth"

INPUT_DIR = BASE_DIR / "outputs" / "extracted_frames"

OUTPUT_DIR = BASE_DIR / "outputs" / "anonymized_frames"

DEVICE = "cpu"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

PROGRESS_CALLBACK = Optional[Callable[[int, int], None]]


# ==================================================
# 2. 图片文件处理
# ==================================================

def natural_sort_key(path):
    """自然排序，避免 frame_10 排在 frame_2 前面。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_image_files(input_dir=INPUT_DIR):
    """获取输入目录中的图片，并按自然顺序排序。"""
    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"抽帧目录不存在：{input_dir}"
        )

    return sorted(
        [
            path for path in input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=natural_sort_key
    )


# ==================================================
# 3. 图片预处理
# ==================================================

def preprocess_image(image_bgr):
    """
    BGR 转 RGB，缩放到 128x128，
    将像素值归一化到 [0, 1]。
    """

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


# ==================================================
# 4. 单帧匿名化
# ==================================================

def anonymize_frame(model, image_bgr, device=DEVICE):
    """
    对单帧图像执行 FA 匿名化。

    输入：OpenCV BGR 图像
    输出：恢复原始尺寸的 BGR 图像
    """

    if image_bgr is None:
        raise ValueError("输入图像为空。")

    height, width = image_bgr.shape[:2]

    image_tensor = preprocess_image(image_bgr)

    with torch.inference_mode():
        output = model(image_tensor.to(device))

    if not torch.is_tensor(output):
        raise TypeError(
            f"FA 模型输出不是 Tensor：{type(output)}"
        )

    if output.ndim != 4 or output.shape[0] != 1:
        raise RuntimeError(
            f"FA 模型输出维度异常：{tuple(output.shape)}"
        )

    output = output.squeeze(0).cpu().numpy()
    output = np.transpose(output, (1, 2, 0))

    output = np.clip(
        output * 255.0,
        0,
        255
    ).astype(np.uint8)

    # 模型输出为 RGB，转换为 OpenCV 使用的 BGR
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


# ==================================================
# 5. 批量匿名化接口
# ==================================================

def anonymize_folder(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    model_path=MODEL_PATH,
    device=DEVICE,
    progress_callback=None
):
    """
    批量执行 FA 匿名化。

    供系统界面后台线程调用。

    参数：
        input_dir:
            原始视频抽帧目录

        output_dir:
            匿名化结果保存目录

        model_path:
            FA 模型权重路径

        device:
            推理设备，默认 CPU

        progress_callback:
            可选进度回调，格式为 callback(current, total)

    返回：
        总帧数、成功帧数、耗时、输出目录等统计信息
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model_path = Path(model_path)

    print("=" * 50)
    print("FA 视频帧批量匿名化")
    print("=" * 50)

    # 1. 检查路径
    if not model_path.is_file():
        raise FileNotFoundError(
            f"FA 模型权重不存在：{model_path}"
        )

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"原始视频帧目录不存在：{input_dir}"
        )

    # 防止误将输入目录作为输出目录
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError(
            "原始帧目录与匿名化输出目录不能相同。"
        )

    image_files = get_image_files(input_dir)

    total = len(image_files)

    if total == 0:
        raise RuntimeError(
            f"没有找到待匿名化的视频帧：{input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 2. 清理上一次生成的图片
    for path in output_dir.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            path.unlink()

    print(f"待处理帧数：{total}")
    print(f"FA 模型：{model_path}")
    print(f"推理设备：{device}")

    # 3. 加载模型，只加载一次
    torch.set_num_threads(4)

    start_load = time.perf_counter()

    model = load_fa_model(
        str(model_path),
        device=device
    )

    model_load_time = time.perf_counter() - start_load

    print(
        f"模型加载完成，耗时：{model_load_time:.3f} 秒"
    )

    # 4. 批量处理
    start_time = time.perf_counter()

    processed = 0
    failed = 0
    failed_files = []

    for index, image_path in enumerate(image_files, start=1):

        try:
            image_bgr = cv2.imread(str(image_path))

            if image_bgr is None:
                raise RuntimeError("图片读取失败")

            anonymized_bgr = anonymize_frame(
                model,
                image_bgr,
                device=device
            )

            output_path = output_dir / (
                image_path.stem + ".png"
            )

            success = cv2.imwrite(
                str(output_path),
                anonymized_bgr
            )

            if not success:
                raise RuntimeError("匿名化结果保存失败")

            processed += 1

        except Exception as exc:
            failed += 1
            failed_files.append(
                f"{image_path.name}: {exc}"
            )

            print(
                f"[失败] {image_path.name}: {exc}"
            )

        # 通知界面当前处理进度
        if progress_callback is not None:
            progress_callback(index, total)

        if index % 10 == 0 or index == total:
            print(
                f"处理进度：{index}/{total} "
                f"（成功 {processed}，失败 {failed}）"
            )

    elapsed = time.perf_counter() - start_time

    # 5. 汇总结果
    print("\n========== FA 匿名化结果 ==========")
    print(f"总帧数：{total}")
    print(f"成功处理：{processed}")
    print(f"处理失败：{failed}")
    print(f"模型加载耗时：{model_load_time:.2f} 秒")
    print(f"批量处理耗时：{elapsed:.2f} 秒")

    if processed > 0:
        print(
            f"平均每帧耗时：{elapsed / processed:.3f} 秒"
        )

    print(f"匿名化结果目录：{output_dir}")

    # 不允许不完整的匿名化结果进入视频合成阶段
    if failed > 0:
        details = "\n".join(failed_files[:10])

        raise RuntimeError(
            f"FA 匿名化未完整完成："
            f"{failed}/{total} 帧处理失败。\n"
            f"失败详情：\n{details}"
        )

    if processed != total:
        raise RuntimeError(
            f"匿名化帧数不一致：成功 {processed}，总计 {total}"
        )

    print("FA 匿名化全部完成！")

    return {
        "total_frames": total,
        "processed_frames": processed,
        "failed_frames": failed,
        "model_load_time": model_load_time,
        "processing_time": elapsed,
        "output_dir": str(output_dir)
    }


# ==================================================
# 6. 独立运行入口
# ==================================================

def main():

    result = anonymize_folder()

    print("\n========== 处理摘要 ==========")
    print(f"输入帧数：{result['total_frames']}")
    print(f"成功匿名化：{result['processed_frames']}")
    print(f"处理耗时：{result['processing_time']:.2f} 秒")
    print(f"输出目录：{result['output_dir']}")


if __name__ == "__main__":
    main()

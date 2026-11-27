from __future__ import annotations
# -*- coding: utf-8 -*-
"""
统一视频处理流水线

流程：
1. 原始视频抽帧
2. FA 匿名化
3. 匿名化帧重组成视频
4. 合并原始视频音轨
5. FT 异常响应推理及曲线生成
6. PSNR / SSIM 图像质量评价
7. 原始帧与匿名化帧的人脸检测对比

本文件供命令行和 PySide6 后台线程共同调用。
"""
import io
from contextlib import redirect_stdout


import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import cv2


# ==================================================
# 1. 项目路径和模块导入
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

EXTRACTED_DIR = OUTPUT_DIR / "extracted_frames"
ANONYMIZED_DIR = OUTPUT_DIR / "anonymized_frames"

VIDEO_ONLY_PATH = OUTPUT_DIR / "anonymized_video.mp4"
FINAL_VIDEO_PATH = OUTPUT_DIR / "anonymized_video_with_audio.mp4"

FT_SCRIPT = BASE_DIR / "run_avenue_video18_inference.py"
PSNR_SCRIPT = BASE_DIR / "run_avenue_video18_psnr_ssim.py"
FACE_SCRIPT = BASE_DIR / "run_face_detection_comparison.py"

FT_MODEL_PATH = BASE_DIR / "ft_model_36.pth"
MEMORY_PATH = BASE_DIR / "keys_36.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

ProgressCallback = Optional[Callable[[int], None]]
StatusCallback = Optional[Callable[[str], None]]
LogCallback = Optional[Callable[[str], None]]


# 复用已经单独测试通过的三个模块
from anonymize_video_frames import anonymize_folder
from rebuild_anonymized_video import (
    rebuild_anonymized_video as rebuild_video
)
from merge_audio import merge_audio as merge_video_audio


# ==================================================
# 2. 通用辅助函数
# ==================================================

# 记录上一次输出的进度
_last_progress_value = None


def notify_progress(callback, value):
    """
    限制进度输出频率：
    只有进度增加至少5个百分点时才输出。
    达到100%时始终输出一次。
    """

    global _last_progress_value

    if callback is None:
        return

    value = max(0, min(100, int(value)))

    if (
        _last_progress_value is None
        or value >= _last_progress_value + 5
        or value == 100
    ):
        callback(value)
        _last_progress_value = value


def notify_status(callback, message):
    """阶段信息只输出一次，避免回调和内部打印重复。"""

    if callback is not None:
        callback(message)
    else:
        print(f"\n[STAGE] {message}")


def notify_log(callback, message):
    """日志信息只输出一次。"""

    if callback is not None:
        callback(message)
    else:
        print(message)


def run_quietly(function, *args, **kwargs):
    """
    执行已有模块，但隐藏模块内部的普通 print 输出。
    函数返回值和异常仍然正常传递。
    """

    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)

def run_quietly(function, *args, **kwargs):
    """
    执行已有模块，但隐藏模块内部的普通 print 输出。
    函数返回值和异常仍然正常传递。
    """
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)

def clear_image_files(folder: Path):
    """清理指定目录下旧的图片文件，不删除其他文件。"""

    folder.mkdir(parents=True, exist_ok=True)

    for path in folder.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            path.unlink()


# ==================================================
# 3. 视频抽帧
# ==================================================

def extract_video_frames(
    video_path,
    output_dir=EXTRACTED_DIR,
    progress_callback=None
):
    """
    使用 OpenCV 读取原始视频并抽帧。

    文件名统一为：
        frame_000000.jpg
        frame_000001.jpg
        ...

    返回视频 FPS、分辨率和实际抽取帧数。
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"原始视频不存在：{video_path}"
        )

    clear_image_files(output_dir)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开视频：{video_path}"
        )

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(
                f"视频元信息异常：FPS={fps}，"
                f"分辨率={width}×{height}"
            )

        count = 0

        print(f"[INFO] 原始视频：{video_path.name}")
        print(f"[INFO] 视频帧率：{fps:.3f} FPS")
        print(f"[INFO] 视频分辨率：{width}×{height}")
        print(f"[INFO] 视频帧数（元信息）：{total}")

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_path = output_dir / f"frame_{count:06d}.jpg"

            success = cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            if not success:
                raise RuntimeError(
                    f"第 {count + 1} 帧保存失败。"
                )

            count += 1

            if (
                count % 10 == 0
                or (total > 0 and count == total)
            ):
                if progress_callback is not None:
                    ratio = count / total if total > 0 else 0
                    progress_callback(ratio)

        if count == 0:
            raise RuntimeError(
                "没有从原始视频中读取到有效帧。"
            )

        print(f"[INFO] 实际抽取帧数：{count}")

        return {
            "frame_dir": str(output_dir),
            "frame_count": count,
            "fps": fps,
            "width": width,
            "height": height
        }

    finally:
        cap.release()


# ==================================================
# 4. 独立 Python 子进程执行
# ==================================================

def run_python_script(
    script_path,
    arguments,
    stage_name,
    progress_callback=None,
    progress_start=0,
    progress_span=0,
    log_callback=None
):
    """
    在独立 Python 进程中执行实验脚本。

    只输出：
    1. 每增加约5%的处理进度
    2. 警告和错误信息
    3. 阶段完成提示
    """

    script_path = Path(script_path)

    if not script_path.is_file():
        raise FileNotFoundError(
            f"找不到处理脚本：{script_path}"
        )

    command = [
        sys.executable,
        "-u",
        str(script_path),
        *[str(value) for value in arguments]
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env
    )

    output_lines = []
    last_progress = progress_start - 5

    try:
        assert process.stdout is not None

        for raw_line in process.stdout:
            line = raw_line.rstrip()

            if not line:
                continue

            output_lines.append(line)

            # 1. 提取推理、指标计算、人脸检测进度
            match = re.search(
                r"\[(?:INFER|METRIC|DETECT)\]\s+(\d+)/(\d+)",
                line
            )

            if match and progress_callback is not None:
                current = int(match.group(1))
                total = int(match.group(2))

                if total > 0:
                    ratio = current / total

                    value = int(
                        progress_start + ratio * progress_span
                    )

                    # 每增加约5%才更新一次
                    if (
                        value >= last_progress + 5
                        or current == total
                    ):
                        notify_progress(
                            progress_callback,
                            value
                        )

                        last_progress = value

            # 2. 只保留警告和错误日志
            lower_line = line.lower()

            important_keywords = [
                "warning",
                "error",
                "traceback",
                "exception",
                "[failed]",
                "失败"
            ]

            if any(
                keyword in lower_line
                for keyword in important_keywords
            ):
                notify_log(
                    log_callback,
                    f"[{stage_name}] {line}"
                )

        return_code = process.wait()

    except Exception:
        process.kill()
        process.wait()
        raise

    # 3. 子进程失败时保留详细错误信息
    if return_code != 0:
        details = "\n".join(output_lines[-35:])

        raise RuntimeError(
            f"{stage_name}执行失败，返回码：{return_code}\n"
            f"子进程输出：\n{details}"
        )

    notify_log(
        log_callback,
        f"[{stage_name}] 处理完成。"
    )

    return output_lines


# ==================================================
# 5. FT 模型参考仓库定位
# ==================================================

def find_vad_repository():
    """
    定位本地 Video-Anonymization 参考代码。

    当前项目约定：
        privacy-video-system/
            vad_source/
                anonymization_training/
                    mnad_model/
    """

    candidates = [
        BASE_DIR / "vad_source",
        BASE_DIR / "Video-Anonymization",
        BASE_DIR / "reference" / "Video-Anonymization",
        BASE_DIR / "reference"
    ]

    for candidate in candidates:
        model_dir = (
            candidate
            / "anonymization_training"
            / "mnad_model"
        )

        if model_dir.is_dir():
            return candidate

    raise FileNotFoundError(
        "没有找到 FT 模型需要的参考代码目录。\n"
        "请确认本地存在：\n"
        "vad_source/anonymization_training/mnad_model"
    )


# ==================================================
# 6. FT 异常响应推理
# ==================================================

def run_ft_analysis(
    anonymized_dir=ANONYMIZED_DIR,
    output_dir=None,
    video_id="uploaded",
    progress_callback=None,
    log_callback=None
):
    """
    调用已有 FT 推理脚本。

    FT 使用前4帧预测第5帧，不重新运行 FA。
    曲线由 FT 脚本调用独立绘图进程生成。
    """

    anonymized_dir = Path(anonymized_dir)

    if not FT_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"FT 模型权重不存在：{FT_MODEL_PATH}"
        )

    if not MEMORY_PATH.is_file():
        raise FileNotFoundError(
            f"FT Memory Keys 不存在：{MEMORY_PATH}"
        )

    repo_root = find_vad_repository()

    if output_dir is None:
        output_dir = OUTPUT_DIR / "ft_results"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arguments = [
        "--frames", anonymized_dir,
        "--model", FT_MODEL_PATH,
        "--memory", MEMORY_PATH,
        "--output", output_dir,
        "--video-id", video_id,
        "--repo", repo_root
    ]

    run_python_script(
        FT_SCRIPT,
        arguments,
        stage_name="FT",
        progress_callback=progress_callback,
        progress_start=43,
        progress_span=22,
        log_callback=log_callback
    )

    curve_path = (
        output_dir
        / f"video_{video_id}_anomaly_curve.png"
    )

    csv_path = (
        output_dir
        / f"video_{video_id}_anomaly_scores.csv"
    )

    if not csv_path.is_file():
        raise RuntimeError(
            f"FT 推理没有生成分数 CSV：{csv_path}"
        )

    if not curve_path.is_file():
        raise RuntimeError(
            f"FT 推理没有生成异常响应曲线：{curve_path}"
        )

    return {
        "curve_path": str(curve_path),
        "csv_path": str(csv_path),
        "output_dir": str(output_dir)
    }


# ==================================================
# 7. PSNR / SSIM 评价
# ==================================================

def parse_quality_summary(summary_path):
    """从现有 PSNR/SSIM 脚本生成的摘要中读取统计值。"""

    summary_path = Path(summary_path)

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"PSNR/SSIM 摘要不存在：{summary_path}"
        )

    content = summary_path.read_text(encoding="utf-8")

    def get_value(pattern, field_name):
        match = re.search(pattern, content)

        if not match:
            raise RuntimeError(
                f"无法从 PSNR/SSIM 摘要中读取：{field_name}"
            )

        return float(match.group(1))

    return {
        "psnr_mean": get_value(
            r"PSNR mean per frame:\s*([^\s]+)",
            "PSNR 均值"
        ),
        "psnr_std": get_value(
            r"PSNR std:\s*([^\s]+)",
            "PSNR 标准差"
        ),
        "ssim_mean": get_value(
            r"SSIM mean:\s*([^\s]+)",
            "SSIM 均值"
        ),
        "ssim_std": get_value(
            r"SSIM std:\s*([^\s]+)",
            "SSIM 标准差"
        )
    }


def run_quality_evaluation(
    progress_callback=None,
    log_callback=None
):
    """运行已有 PSNR/SSIM 实验脚本并读取统计结果。"""

    run_python_script(
        PSNR_SCRIPT,
        [],
        stage_name="PSNR/SSIM",
        progress_callback=progress_callback,
        progress_start=65,
        progress_span=12,
        log_callback=log_callback
    )

    summary_path = (
        OUTPUT_DIR
        / "image_quality_18"
        / "video_18_psnr_ssim_summary.txt"
    )

    result = parse_quality_summary(summary_path)
    result["summary_path"] = str(summary_path)

    return result


# ==================================================
# 8. 人脸检测结果读取
# ==================================================

def parse_face_summary(summary_path):
    """读取现有人脸检测脚本输出的 TXT 统计摘要。"""

    summary_path = Path(summary_path)

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"人脸检测摘要不存在：{summary_path}"
        )

    content = summary_path.read_text(encoding="utf-8")

    def parse_section(section_name, next_section=None):
        if section_name not in content:
            raise RuntimeError(
                f"人脸检测摘要中找不到：{section_name}"
            )

        start = content.index(section_name) + len(section_name)

        if next_section and next_section in content[start:]:
            end = content.index(next_section, start)
            block = content[start:end]
        else:
            block = content[start:]

        def extract(pattern, field_name, cast=float):
            match = re.search(pattern, block)

            if not match:
                raise RuntimeError(
                    f"无法读取{section_name}的{field_name}"
                )

            return cast(match.group(1))

        return {
            "detected_frames": extract(
                r"检测帧数：(\d+)",
                "检测帧数",
                int
            ),
            "detection_rate": extract(
                r"帧级人脸检出率：([-+\d.]+)%",
                "帧级人脸检出率",
                float
            ),
            "total_faces": extract(
                r"人脸检测框总数：(\d+)",
                "人脸检测框总数",
                int
            ),
            "average_faces_per_frame": extract(
                r"平均每帧人脸数：([-+\d.]+)",
                "平均每帧人脸数",
                float
            ),
            "average_confidence": extract(
                r"平均检测置信度：([-+\d.]+)",
                "平均检测置信度",
                float
            )
        }

    original_stats = parse_section(
        "原始视频",
        "匿名化视频"
    )

    anonymized_stats = parse_section(
        "匿名化视频",
        "========== 对比结果 =========="
    )

    return {
        "original": original_stats,
        "anonymized": anonymized_stats
    }


def run_face_detection(
    progress_callback=None,
    log_callback=None
):
    """调用现有 InsightFace 人脸检测脚本。"""

    run_python_script(
        FACE_SCRIPT,
        [],
        stage_name="人脸检测",
        progress_callback=progress_callback,
        progress_start=80,
        progress_span=20,
        log_callback=log_callback
    )

    summary_path = (
        OUTPUT_DIR
        / "face_detection_18_thr02"
        / "video_18_face_detection_summary.txt"
    )

    result = parse_face_summary(summary_path)
    result["summary_path"] = str(summary_path)

    return result


# ==================================================
# 9. 完整处理流水线
# ==================================================

def run_full_pipeline(
    source_video_path,
    progress_callback=None,
    status_callback=None,
    log_callback=None
):
    """
    一次运行完成全部视频处理任务。

    供命令行和 PySide6 后台线程调用。
    """
    global _last_progress_value
    _last_progress_value = None

    source_video_path = Path(source_video_path).resolve()

    if not source_video_path.is_file():
        raise FileNotFoundError(
            f"找不到原始视频：{source_video_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # 阶段1：原始视频抽帧
    # --------------------------------------------------

    notify_status(
        status_callback,
        "阶段 1/7：正在读取原始视频并抽取视频帧"
    )

    frame_info = extract_video_frames(
        source_video_path,
        EXTRACTED_DIR,
        progress_callback=lambda ratio: notify_progress(
            progress_callback,
            ratio * 8
        )
    )

    notify_progress(progress_callback, 8)

    # --------------------------------------------------
    # 阶段2：FA 匿名化
    # --------------------------------------------------

    notify_status(
        status_callback,
        "阶段 2/7：正在执行 FA 视频匿名化"
    )

    anonymization_result = run_quietly(
        anonymize_folder,
        input_dir=EXTRACTED_DIR,
        output_dir=ANONYMIZED_DIR,
        progress_callback=lambda current, total: notify_progress(
            progress_callback,
            8 + current / total * 22
        )
    )

    notify_progress(progress_callback, 30)

    # --------------------------------------------------
    # 阶段3：匿名化帧重组成视频
    # --------------------------------------------------

    notify_status(
        status_callback,
        "阶段 3/7：正在将匿名化帧合成为视频"
    )

    rebuild_result = run_quietly(
        rebuild_video,
        source_video_path=source_video_path,
        input_dir=ANONYMIZED_DIR,
        output_path=VIDEO_ONLY_PATH,
        progress_callback=lambda current, total: notify_progress(
            progress_callback,
            30 + current / total * 7
        )
    )

    notify_progress(progress_callback, 37)

    # --------------------------------------------------
    # 阶段4：合并原始视频音轨
    # --------------------------------------------------

    notify_status(
        status_callback,
        "阶段 4/7：正在合并原始视频音轨"
    )

    final_video_path = run_quietly(
        merge_video_audio,
        original_video_path=source_video_path,
        video_only_path=VIDEO_ONLY_PATH,
        output_path=FINAL_VIDEO_PATH
    )

    notify_progress(progress_callback, 43)

    # --------------------------------------------------
    # 阶段5：FT 异常响应推理
    # --------------------------------------------------

    notify_status(
        status_callback,
        "阶段 5/7：正在执行 FT 异常响应推理并生成曲线"
    )

    ft_result = run_ft_analysis(
        anonymized_dir=ANONYMIZED_DIR,
        video_id="uploaded",
        progress_callback=progress_callback,
        log_callback=log_callback
    )

    notify_progress(progress_callback, 65)

    # --------------------------------------------------
    # 阶段6：PSNR / SSIM 评价
    # --------------------------------------------------

    notify_status(
        status_callback,
        "阶段 6/7：正在计算 PSNR / SSIM 图像质量指标"
    )

    quality_result = run_quality_evaluation(
        progress_callback=progress_callback,
        log_callback=log_callback
    )

    notify_progress(progress_callback, 80)

    # --------------------------------------------------
    # 阶段7：人脸检测对比
    # --------------------------------------------------

    notify_status(
        status_callback,
        "阶段 7/7：正在执行原始视频与匿名化视频的人脸检测"
    )

    face_result = run_face_detection(
        progress_callback=progress_callback,
        log_callback=log_callback
    )

    notify_progress(progress_callback, 100)

    notify_status(
        status_callback,
        "全部处理完成"
    )

    # --------------------------------------------------
    # 返回界面需要展示的数据
    # --------------------------------------------------

    result = {
        "source_video_path": str(source_video_path),

        "original_frames_dir": str(EXTRACTED_DIR),
        "anonymized_frames_dir": str(ANONYMIZED_DIR),

        "frame_count": frame_info["frame_count"],
        "fps": frame_info["fps"],
        "width": frame_info["width"],
        "height": frame_info["height"],

        "anonymized_video_path": str(final_video_path),
        "video_without_audio_path": str(VIDEO_ONLY_PATH),

        "curve_path": ft_result["curve_path"],
        "ft_csv_path": ft_result["csv_path"],

        "psnr_mean": quality_result["psnr_mean"],
        "psnr_std": quality_result["psnr_std"],
        "ssim_mean": quality_result["ssim_mean"],
        "ssim_std": quality_result["ssim_std"],

        "original_face_detection_rate":
            face_result["original"]["detection_rate"],

        "anonymized_face_detection_rate":
            face_result["anonymized"]["detection_rate"],

        "original_detected_frames":
            face_result["original"]["detected_frames"],

        "anonymized_detected_frames":
            face_result["anonymized"]["detected_frames"],

        "original_total_faces":
            face_result["original"]["total_faces"],

        "anonymized_total_faces":
            face_result["anonymized"]["total_faces"],

        "original_average_confidence":
            face_result["original"]["average_confidence"],

        "anonymized_average_confidence":
            face_result["anonymized"]["average_confidence"],

        "anonymization": anonymization_result,
        "video_rebuild": rebuild_result
    }

    return result


# ==================================================
# 10. 命令行测试入口
# ==================================================

def main():
    parser = argparse.ArgumentParser(
        description="视频隐私保护与异常检测完整处理流水线"
    )

    parser.add_argument(
        "video",
        type=Path,
        help="待处理的原始视频路径"
    )

    args = parser.parse_args()

    def print_progress(value):
        print(
            f"[PROGRESS] {value}%",
            file=sys.__stdout__
        )

    def print_status(message):
        print(f"\n[STATUS] {message}")

    result = run_full_pipeline(
        source_video_path=args.video,
        progress_callback=print_progress,
        status_callback=print_status
    )

    print("\n========== 完整处理结果 ==========")
    print(f"原始视频：{result['source_video_path']}")
    print(f"帧数：{result['frame_count']}")
    print(f"匿名化视频：{result['anonymized_video_path']}")
    print(f"异常响应曲线：{result['curve_path']}")

    print(f"PSNR 均值：{result['psnr_mean']:.4f} dB")
    print(f"SSIM 均值：{result['ssim_mean']:.6f}")

    print(
        "原始视频人脸检出率："
        f"{result['original_face_detection_rate']:.4f}%"
    )

    print(
        "匿名化视频人脸检出率："
        f"{result['anonymized_face_detection_rate']:.4f}%"
    )

    print("\n[SUCCESS] 完整视频处理流水线运行成功。")


if __name__ == "__main__":
    main()

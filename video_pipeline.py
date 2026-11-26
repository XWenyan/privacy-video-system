# -*- coding: utf-8 -*-
"""One-click video anonymization and evaluation pipeline for the desktop UI."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from models.fa_model import load_fa_model
from run_avenue_video18_inference import (
    find_reference_root,
    load_ft_model,
    load_memory,
    run_inference,
)
from run_avenue_video18_psnr_ssim import calculate_psnr, calculate_ssim
import run_face_detection_comparison as face_eval


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FACE_DETECTION_THRESHOLD = 0.3


def natural_key(value: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def _notify(callback: Callable[[int, str], None] | None, percent: int, message: str):
    if callback is not None:
        callback(max(0, min(100, int(percent))), message)


def _check_cancelled(cancel_check: Callable[[], bool] | None):
    if cancel_check is not None and cancel_check():
        raise InterruptedError("用户请求停止当前处理任务。")


def _frame_files(folder: Path) -> list[Path]:
    files = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda path: natural_key(path.name))
    if not files:
        raise RuntimeError(f"帧目录中没有找到图片：{folder}")
    stems = [path.stem for path in files]
    if len(stems) != len(set(stems)):
        raise RuntimeError(f"帧目录中存在重复文件名：{folder}")
    return files


def _pair_frame_files(original_dir: Path, anonymized_dir: Path):
    original_files = {p.stem: p for p in _frame_files(original_dir)}
    anonymized_files = {p.stem: p for p in _frame_files(anonymized_dir)}
    if original_files.keys() != anonymized_files.keys():
        missing = sorted(original_files.keys() - anonymized_files.keys(), key=natural_key)
        extra = sorted(anonymized_files.keys() - original_files.keys(), key=natural_key)
        raise RuntimeError(
            "原始帧与匿名化帧无法一一配对。\n"
            f"匿名化侧缺少：{missing[:10]}\n"
            f"匿名化侧多出：{extra[:10]}"
        )
    return [
        (name, original_files[name], anonymized_files[name])
        for name in sorted(original_files, key=natural_key)
    ]


def _extract_video_frames(
    video_path: Path,
    output_dir: Path,
    progress_callback=None,
    cancel_check=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频文件：{video_path}")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("无法读取视频分辨率。")

    count = 0
    try:
        while True:
            _check_cancelled(cancel_check)
            ok, frame = capture.read()
            if not ok:
                break
            frame_path = output_dir / f"frame_{count:06d}.jpg"
            if not cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise RuntimeError(f"保存第 {count + 1} 帧失败：{frame_path}")
            count += 1
            if count % 10 == 0 or (total > 0 and count == total):
                pct = 5 + (15 * count / total if total > 0 else 0)
                _notify(
                    progress_callback,
                    pct,
                    f"正在抽取视频帧：{count}/{total if total > 0 else '未知'}",
                )
    finally:
        capture.release()

    if count == 0:
        raise RuntimeError("没有从该视频中读取到有效画面。")
    if fps <= 0:
        fps = 25.0

    return {
        "frame_count": count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def _anonymize_frames(
    original_dir: Path,
    anonymized_dir: Path,
    model_path: Path,
    device: torch.device,
    progress_callback=None,
    cancel_check=None,
):
    from models.fa_model import load_fa_model

    frames = _frame_files(original_dir)
    anonymized_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, 20, "正在加载 FA 匿名化模型……")
    model = load_fa_model(model_path, device=str(device))

    with torch.inference_mode():
        for index, frame_path in enumerate(frames, start=1):
            _check_cancelled(cancel_check)
            bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"读取原始帧失败：{frame_path}")

            # Match the reference use_fa.py preprocessing:
            # PIL RGB + ToTensor, which produces RGB float values in [0, 1].
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(rgb)
                .permute(2, 0, 1)
                .contiguous()
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(device)
            )
            output = model(tensor)
            if not torch.is_tensor(output) or output.ndim != 4 or output.shape[0] != 1:
                raise RuntimeError(
                    f"FA 模型输出格式异常：{frame_path.name}，"
                    f"输出类型={type(output)}，"
                    f"shape={getattr(output, 'shape', None)}"
                )

            output = output.detach().clamp(0.0, 1.0)
            if output.shape[-2:] != tensor.shape[-2:]:
                output = torch.nn.functional.interpolate(
                    output,
                    size=tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            anon_rgb = (
                output[0]
                .permute(1, 2, 0)
                .mul(255.0)
                .round()
                .byte()
                .cpu()
                .numpy()
            )
            anon_bgr = cv2.cvtColor(anon_rgb, cv2.COLOR_RGB2BGR)
            target_path = anonymized_dir / f"{frame_path.stem}.png"
            if not cv2.imwrite(str(target_path), anon_bgr):
                raise RuntimeError(f"保存匿名化帧失败：{target_path}")

            if index % 5 == 0 or index == len(frames):
                pct = 20 + 28 * index / len(frames)
                _notify(
                    progress_callback,
                    pct,
                    f"正在执行 FA 匿名化：{index}/{len(frames)} 帧",
                )

    return len(frames)


def _rebuild_video(
    anonymized_dir: Path,
    output_video: Path,
    fps: float,
    width: int,
    height: int,
):
    frames = _frame_files(anonymized_dir)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建匿名化视频：{output_video}")

    try:
        for path in frames:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"读取匿名化帧失败：{path}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"匿名化帧分辨率发生变化：{path.name}，"
                    f"实际为 {frame.shape[1]}x{frame.shape[0]}，"
                    f"预期为 {width}x{height}"
                )
            writer.write(frame)
    finally:
        writer.release()

    if not output_video.is_file() or output_video.stat().st_size == 0:
        raise RuntimeError("匿名化视频文件生成失败。")


def _try_merge_audio(video_only: Path, original_video: Path, output_with_audio: Path):
    """Keep the anonymized frames and optionally copy the source audio track."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg, "-y",
            "-i", str(video_only),
            "-i", str(original_video),
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_with_audio),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode == 0 and output_with_audio.is_file():
            return output_with_audio
    except Exception:
        pass
    return video_only


def _calculate_image_quality(pairs, output_dir: Path, progress_callback=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    psnr_values = []
    ssim_values = []

    for index, (frame_id, original_path, anonymized_path) in enumerate(pairs, start=1):
        original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        anonymized = cv2.imread(str(anonymized_path), cv2.IMREAD_COLOR)
        if original is None or anonymized is None:
            raise RuntimeError(f"第 {frame_id} 帧读取失败，无法计算 PSNR/SSIM。")
        if original.shape != anonymized.shape:
            raise RuntimeError(
                f"第 {frame_id} 帧尺寸不一致："
                f"原始={original.shape}，匿名化={anonymized.shape}"
            )

        psnr, mse = calculate_psnr(original, anonymized)
        ssim = calculate_ssim(original, anonymized)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        rows.append({
            "frame_index": index,
            "frame_id": frame_id,
            "mse": mse,
            "psnr_db": psnr,
            "ssim": ssim,
        })

        if index % 10 == 0 or index == len(pairs):
            pct = 65 + 12 * index / len(pairs)
            _notify(progress_callback, pct, f"正在计算 PSNR/SSIM：{index}/{len(pairs)} 帧")

    psnr_mean = float(np.mean(psnr_values))
    psnr_std = float(np.std(psnr_values))
    ssim_mean = float(np.mean(ssim_values))
    ssim_std = float(np.std(ssim_values))

    csv_path = output_dir / "psnr_ssim_by_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "psnr_ssim_summary.txt"
    summary_path.write_text(
        f"Paired frames: {len(pairs)}\n"
        f"PSNR mean per frame: {psnr_mean:.6f} dB\n"
        f"PSNR std: {psnr_std:.6f} dB\n"
        f"SSIM mean: {ssim_mean:.6f}\n"
        f"SSIM std: {ssim_std:.6f}\n"
        "SSIM window: 11x11\n"
        "SSIM sigma: 1.5\n"
        "Data range: 255\n",
        encoding="utf-8",
    )
    return {
        "paired_frames": len(pairs),
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
    }


def _run_ft_inference(
    anonymized_dir: Path,
    project_root: Path,
    output_dir: Path,
    device: torch.device,
    progress_callback=None,
    cancel_check=None,
):
    frames = _frame_files(anonymized_dir)
    if len(frames) < 5:
        raise RuntimeError("FT 异常响应分析至少需要 5 帧视频画面。")

    model_path = project_root / "ft_model_36.pth"
    memory_path = project_root / "keys_36.pt"
    source_repo = project_root / "vad_source"

    for required in (model_path, memory_path):
        if not required.is_file():
            raise FileNotFoundError(f"缺少 FT 推理资源：{required}")
    if not (source_repo / "anonymization_training" / "mnad_model").is_dir():
        raise FileNotFoundError(
            "没有找到本地 VAD 源码目录："
            f"{source_repo / 'anonymization_training' / 'mnad_model'}\n"
            "请确认你本地的 vad_source 文件夹仍然存在。"
        )

    _check_cancelled(cancel_check)
    _notify(progress_callback, 48, "正在加载 FT 模型和 Memory Keys……")
    repo_root = find_reference_root(project_root, str(source_repo))
    model = load_ft_model(model_path, repo_root, device)
    memory = load_memory(memory_path, device)

    _notify(progress_callback, 52, "正在计算匿名化视频的 FT 异常响应……")
    result = run_inference(model, memory, frames, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "anomaly_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "prediction_index",
            "target_frame",
            "psnr",
            "feature_compactness",
            "psnr_anomaly_score",
            "feature_anomaly_score",
            "anomaly_score",
        ])
        for index, frame_name in enumerate(result["target_frame_names"]):
            writer.writerow([
                index + 1,
                frame_name,
                float(result["psnr"][index]),
                float(result["feature"][index]),
                float(result["psnr_anomaly"][index]),
                float(result["feature_anomaly"][index]),
                float(result["scores"][index]),
            ])

    summary_path = output_dir / "anomaly_inference_summary.txt"
    summary_path.write_text(
        f"Input frames: {len(frames)}\n"
        f"Prediction scores: {len(result['scores'])}\n"
        f"Memory update count: {result['memory_update_count']}\n"
        f"Mean anomaly score: {float(np.mean(result['scores'])):.8f}\n"
        f"Std anomaly score: {float(np.std(result['scores'])):.8f}\n"
        f"Mean prediction PSNR: {float(np.mean(result['psnr'])):.8f}\n"
        f"Mean feature compactness: {float(np.mean(result['feature'])):.8f}\n"
        "Score formula: 0.6 * PSNR anomaly + 0.4 * feature anomaly\n"
        "Normalization: per-video min-max, both components use 1-normalized\n",
        encoding="utf-8",
    )

    _check_cancelled(cancel_check)
    curve_path = output_dir / "anomaly_curve.png"
    plot_script = project_root / "plot_avenue_video18_curve.py"
    if not plot_script.is_file():
        raise FileNotFoundError(f"找不到曲线绘制脚本：{plot_script}")

    # Keep plotting in a separate process to avoid mixing Matplotlib with
    # the PyTorch/OpenMP runtime used for FT inference.
    completed = subprocess.run(
        [
            sys.executable,
            str(plot_script),
            "--csv", str(csv_path),
            "--output", str(curve_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not curve_path.is_file():
        raise RuntimeError(
            "异常响应曲线生成失败。\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )

    return {
        "prediction_count": len(result["scores"]),
        "anomaly_mean": float(np.mean(result["scores"])),
        "anomaly_std": float(np.std(result["scores"])),
        "memory_update_count": int(result["memory_update_count"]),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "curve_path": str(curve_path),
    }


def _run_face_detection(
    pairs,
    output_dir: Path,
    progress_callback=None,
    cancel_check=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    face_eval.DET_THRESHOLD = FACE_DETECTION_THRESHOLD
    detector = face_eval.load_face_detector()

    original_counts = []
    anonymized_counts = []
    original_scores_all = []
    anonymized_scores_all = []
    records = []

    for index, (frame_id, original_path, anonymized_path) in enumerate(pairs, start=1):
        _check_cancelled(cancel_check)
        original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        anonymized = cv2.imread(str(anonymized_path), cv2.IMREAD_COLOR)
        if original is None or anonymized is None:
            raise RuntimeError(f"第 {frame_id} 帧读取失败，无法进行人脸检测。")

        original_boxes, original_scores = face_eval.detect_faces(detector, original)
        anonymized_boxes, anonymized_scores = face_eval.detect_faces(detector, anonymized)

        original_counts.append(len(original_boxes))
        anonymized_counts.append(len(anonymized_boxes))
        original_scores_all.extend(original_scores)
        anonymized_scores_all.extend(anonymized_scores)

        records.append({
            "frame_index": index,
            "frame_id": frame_id,
            "original_face_count": len(original_boxes),
            "original_mean_confidence": (
                float(np.mean(original_scores)) if original_scores else ""
            ),
            "anonymized_face_count": len(anonymized_boxes),
            "anonymized_mean_confidence": (
                float(np.mean(anonymized_scores)) if anonymized_scores else ""
            ),
        })

        if index % 10 == 0 or index == len(pairs):
            pct = 77 + 21 * index / len(pairs)
            _notify(progress_callback, pct, f"正在进行人脸检测：{index}/{len(pairs)} 帧")

    original_stats = face_eval.calculate_statistics(
        original_counts, original_scores_all
    )
    anonymized_stats = face_eval.calculate_statistics(
        anonymized_counts, anonymized_scores_all
    )
    detection_delta = (
        anonymized_stats["detection_rate"] - original_stats["detection_rate"]
    )

    csv_path = output_dir / "face_detection_by_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary_path = output_dir / "face_detection_summary.txt"
    summary_path.write_text(
        f"Model: InsightFace {face_eval.MODEL_NAME} / SCRFD-10GF\n"
        f"Detection threshold: {FACE_DETECTION_THRESHOLD}\n"
        f"Paired frames: {len(pairs)}\n"
        f"Original detected frames: {original_stats['detected_frames']}\n"
        f"Original detection rate: {original_stats['detection_rate']:.6f}%\n"
        f"Original face boxes: {original_stats['total_faces']}\n"
        f"Anonymized detected frames: {anonymized_stats['detected_frames']}\n"
        f"Anonymized detection rate: {anonymized_stats['detection_rate']:.6f}%\n"
        f"Anonymized face boxes: {anonymized_stats['total_faces']}\n"
        f"Detection rate change: {detection_delta:.6f} percentage points\n",
        encoding="utf-8",
    )

    return {
        "threshold": FACE_DETECTION_THRESHOLD,
        "original_detection_rate": float(original_stats["detection_rate"]),
        "anonymized_detection_rate": float(anonymized_stats["detection_rate"]),
        "detection_rate_delta_pp": float(detection_delta),
        "original_face_boxes": int(original_stats["total_faces"]),
        "anonymized_face_boxes": int(anonymized_stats["total_faces"]),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
    }


def process_video(
    video_path: str | Path,
    project_root: str | Path,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
):
    """Run extraction -> FA anonymization -> video -> FT curve -> metrics -> face test."""
    video_path = Path(video_path).resolve()
    project_root = Path(project_root).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到上传的视频：{video_path}")

    # Validate all local assets before spending time extracting the video.
    fa_model_path = project_root / "fa_model_60.pth"
    ft_model_path = project_root / "ft_model_36.pth"
    memory_path = project_root / "keys_36.pt"
    vad_source = project_root / "vad_source" / "anonymization_training" / "mnad_model"
    for path in (fa_model_path, ft_model_path, memory_path):
        if not path.is_file():
            raise FileNotFoundError(f"系统缺少必要模型文件：{path}")
    if not vad_source.is_dir():
        raise FileNotFoundError(
            f"系统缺少本地 VAD 源码：{vad_source}\n"
            "vad_source 不需要上传 GitHub，但必须保留在本机项目目录中。"
        )

    _check_cancelled(cancel_check)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("._") or "video"
    run_id = f"{timestamp}_{safe_stem}"
    run_dir = project_root / "outputs" / "runs" / run_id

    original_dir = run_dir / "original_frames"
    anonymized_dir = run_dir / "anonymized_frames"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, 2, "正在检查视频和本地模型资源……")
    video_info = _extract_video_frames(
        video_path,
        original_dir,
        progress_callback,
        cancel_check,
    )
    _check_cancelled(cancel_check)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _notify(progress_callback, 20, f"抽帧完成，共 {video_info['frame_count']} 帧；推理设备：{device}")

    anonymized_count = _anonymize_frames(
        original_dir,
        anonymized_dir,
        fa_model_path,
        device,
        progress_callback,
        cancel_check,
    )
    _check_cancelled(cancel_check)

    video_only_path = run_dir / "anonymized_video.mp4"
    _rebuild_video(
        anonymized_dir,
        video_only_path,
        video_info["fps"],
        video_info["width"],
        video_info["height"],
    )
    final_video_path = _try_merge_audio(
        video_only_path,
        video_path,
        run_dir / "anonymized_video_with_audio.mp4",
    )
    _notify(progress_callback, 50, "匿名化视频已生成，开始计算 FT 异常响应曲线……")

    vad_result = _run_ft_inference(
        anonymized_dir,
        project_root,
        analysis_dir / "vad",
        device,
        progress_callback,
        cancel_check,
    )
    _check_cancelled(cancel_check)

    pairs = _pair_frame_files(original_dir, anonymized_dir)
    quality_result = _calculate_image_quality(
        pairs,
        analysis_dir / "image_quality",
        progress_callback,
    )

    _check_cancelled(cancel_check)
    face_result = _run_face_detection(
        pairs,
        analysis_dir / "face_detection",
        progress_callback,
        cancel_check,
    )

    result = {
        "run_id": run_id,
        "video_name": video_path.name,
        "output_dir": str(run_dir),
        "original_frames_dir": str(original_dir),
        "anonymized_frames_dir": str(anonymized_dir),
        "anonymized_video_path": str(final_video_path),
        "anonymized_video_only_path": str(video_only_path),
        "curve_path": vad_result["curve_path"],
        "frame_count": int(video_info["frame_count"]),
        "fps": float(video_info["fps"]),
        "width": int(video_info["width"]),
        "height": int(video_info["height"]),
        "device": str(device),
        "vad": vad_result,
        "image_quality": quality_result,
        "face_detection": face_result,
    }

    (run_dir / "result_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _notify(progress_callback, 100, "全部处理完成。")
    return result

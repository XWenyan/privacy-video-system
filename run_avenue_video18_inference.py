# -*- coding: utf-8 -*-
"""Run FT anomaly inference on the already-anonymized frames of Avenue 18.avi.

This reproduces the standalone inference path in
XWenyan/Video-Anonymization/anonymization_training/Evaluate_vad.py:
- use the anonymized frames as-is (do not run FA again);
- use 4 frames to predict the 5th frame;
- OpenCV BGR input, resize to 128x128, normalize to [-1, 1];
- compute prediction PSNR and memory compactness;
- update test memory when point_score < 0.01;
- normalize the two score streams per video and combine with alpha=0.6.

No labels or sklearn are required. This script saves per-frame scores to CSV.
Run plot_avenue_video18_curve.py afterwards to save the PNG in a separate
process, keeping Matplotlib away from the PyTorch/OpenMP process.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(r"D:\privacy-video-system")
DEFAULT_FRAMES_ROOT = PROJECT_ROOT / "outputs" / "anonymized_frames"
DEFAULT_MODEL = PROJECT_ROOT / "ft_model_36.pth"
DEFAULT_MEMORY = PROJECT_ROOT / "keys_36.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "avenue_eval_18"

VIDEO_ID = "18"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
INPUT_SIZE = 128
INPUT_FRAMES = 4
ALPHA = 0.6
MEMORY_UPDATE_THRESHOLD = 0.01


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)]


def has_images(folder: Path) -> bool:
    return folder.is_dir() and any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        for p in folder.iterdir()
    )


def resolve_frames_dir(root: Path, video_id: str) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"匿名化帧目录不存在：{root}")
    if has_images(root):
        return root

    candidates = []
    for child in root.iterdir():
        if not child.is_dir() or not has_images(child):
            continue
        digits = "".join(re.findall(r"\d+", child.name))
        if digits and str(int(digits)) == str(int(video_id)):
            candidates.append(child)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            "发现多个可能属于视频18的帧目录，请用 --frames 明确指定：\n" +
            "\n".join(str(p) for p in candidates)
        )
    raise FileNotFoundError(
        f"在 {root} 中没有找到帧图片，也没有找到视频 {video_id} 对应的子目录。\n"
        "请使用 --frames 指定实际存放匿名化帧的目录。"
    )


def list_frames(folder: Path) -> list[Path]:
    frames = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    frames.sort(key=lambda p: natural_key(p.name))
    if len(frames) <= INPUT_FRAMES:
        raise RuntimeError(f"目录中只有{len(frames)}帧，至少需要5帧：{folder}")
    stems = [p.stem for p in frames]
    if len(stems) != len(set(stems)):
        raise RuntimeError(f"帧文件名主体重复，无法唯一排序：{folder}")
    return frames


def find_reference_root(project_root: Path, explicit_root: str | None) -> Path:
    if explicit_root:
        root = Path(explicit_root)
        if not (root / "anonymization_training" / "mnad_model").is_dir():
            raise FileNotFoundError(
                f"--repo 指定目录下找不到 anonymization_training/mnad_model：{root}"
            )
        return root

    candidates = [
        project_root / "Video-Anonymization",
        project_root / "reference" / "Video-Anonymization",
        project_root / "reference",
        project_root,
    ]
    for candidate in candidates:
        if (candidate / "anonymization_training" / "mnad_model").is_dir():
            return candidate

    for model_dir in project_root.rglob("mnad_model"):
        if model_dir.is_dir() and model_dir.parent.name == "anonymization_training":
            return model_dir.parent.parent

    raise FileNotFoundError(
        "没有找到参考仓库的 anonymization_training/mnad_model。\n"
        "请运行时添加 --repo 参数并指定 Video-Anonymization 仓库根目录。"
    )


def load_ft_model(model_path: Path, repo_root: Path, device: torch.device) -> nn.Module:
    training_root = repo_root / "anonymization_training"
    sys.path.insert(0, str(training_root))
    # Import the original model class before unpickling the full model object.
    import mnad_model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1  # noqa: F401

    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=device)

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"{model_path} 加载后不是 torch.nn.Module。当前脚本按原项目完整模型方式加载；"
            "如果这是 state_dict/checkpoint，需要按训练代码重建模型再加载权重。"
        )
    model = model.to(device)
    model.eval()
    return model


def load_memory(memory_path: Path, device: torch.device) -> torch.Tensor:
    try:
        memory = torch.load(memory_path, map_location=device, weights_only=False)
    except TypeError:
        memory = torch.load(memory_path, map_location=device)

    if isinstance(memory, dict):
        candidates = [
            key for key in ("memory", "m_items", "keys", "memory_items")
            if key in memory
        ]
        if len(candidates) != 1:
            raise TypeError(
                "Memory 文件是字典，无法确定记忆项字段。"
                f"当前字段：{list(memory.keys())}"
            )
        memory = memory[candidates[0]]
    if not torch.is_tensor(memory):
        raise TypeError(f"Memory Keys 应为 Tensor，实际类型：{type(memory)}")
    return memory.to(device).clone()


def read_frame(path: Path) -> torch.Tensor:
    # Match anonymization_training/vad_utlis.py: OpenCV BGR and [-1, 1].
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"读取图像失败：{path}")
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = image.astype(np.float32) / 127.5 - 1.0
    image = np.transpose(image, (2, 0, 1)).copy()
    return torch.from_numpy(image)


def get_point_score(outputs, target: torch.Tensor) -> float:
    error = ((outputs[0] + 1.0) / 2.0 - (target[0] + 1.0) / 2.0).pow(2)
    normal = 1.0 - torch.exp(-error)
    denominator = torch.sum(normal)
    if denominator.item() <= 1e-12:
        return float("inf")
    return float((torch.sum(normal * error) / denominator).item())


def minmax_anomaly(values: np.ndarray, invert: bool = True) -> np.ndarray:
    """Match the custom Video-Anonymization utils.py anomaly normalization."""
    low = float(np.min(values))
    high = float(np.max(values))
    if invert:
        if abs(high - low) < 1e-8:
            return np.full(values.shape, 0.5, dtype=np.float64)
        return 1.0 - (values - low) / (high - low + 1e-8)
    if abs(high - low) < 1e-8:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return 1.0 - (values - low) / (high - low)


@torch.inference_mode()
def run_inference(model: nn.Module, initial_memory: torch.Tensor,
                  frames: list[Path], device: torch.device) -> dict:
    memory_test = initial_memory.clone()
    psnr_values: list[float] = []
    feature_values: list[float] = []
    target_frame_names: list[str] = []
    update_count = 0

    total = len(frames) - INPUT_FRAMES
    for index in range(total):
        window = frames[index:index + INPUT_FRAMES + 1]
        tensors = [read_frame(p) for p in window]
        input_tensor = torch.cat(tensors[:INPUT_FRAMES], dim=0).unsqueeze(0).to(device)
        target_tensor = tensors[INPUT_FRAMES].unsqueeze(0).to(device)

        result = model.forward(input_tensor, memory_test, False)
        if not isinstance(result, (tuple, list)) or len(result) < 10:
            raise RuntimeError(
                "FT forward 返回值与原项目预测模式不一致：预计至少10项，"
                f"实际为 {len(result) if isinstance(result, (tuple, list)) else type(result)}"
            )

        outputs, features = result[0], result[1]
        memory_test = result[3]
        compactness_loss = result[9]

        # Same PSNR calculation used in the reference Evaluate_vad.py.
        pred_01 = (outputs[0] + 1.0) / 2.0
        target_01 = (target_tensor[0] + 1.0) / 2.0
        mse = float(torch.mean((pred_01 - target_01).pow(2)).item())
        mse = max(mse, 1e-12)
        psnr_values.append(10.0 * math.log10(1.0 / mse))

        feature_value = (compactness_loss.mean() if compactness_loss.numel() > 1
                         else compactness_loss)
        feature_values.append(float(feature_value.item()))

        point_sc = get_point_score(outputs, target_tensor)
        if point_sc < MEMORY_UPDATE_THRESHOLD:
            query = F.normalize(features, dim=1).permute(0, 2, 3, 1)
            memory_owner = model.module if hasattr(model, "module") else model
            if not hasattr(memory_owner, "memory"):
                raise AttributeError("FT 模型没有 memory 模块，无法执行原项目的动态 Memory 更新。")
            memory_test = memory_owner.memory.update(query, memory_test, False)
            update_count += 1

        target_frame_names.append(window[INPUT_FRAMES].name)
        if (index + 1) % 50 == 0 or index + 1 == total:
            print(f"[INFER] {index + 1}/{total} 个预测窗口完成")

    psnr_array = np.asarray(psnr_values, dtype=np.float64)
    feature_array = np.asarray(feature_values, dtype=np.float64)

    # Custom Video-Anonymization utils.py applies 1-minmax to both streams.
    psnr_anomaly = minmax_anomaly(psnr_array, invert=True)
    feature_anomaly = minmax_anomaly(feature_array, invert=True)
    anomaly_scores = ALPHA * psnr_anomaly + (1.0 - ALPHA) * feature_anomaly

    return {
        "psnr": psnr_array,
        "feature": feature_array,
        "psnr_anomaly": psnr_anomaly,
        "feature_anomaly": feature_anomaly,
        "scores": anomaly_scores,
        "target_frame_names": target_frame_names,
        "memory_update_count": update_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对Avenue 18.avi的已匿名化帧执行FT推理，保存逐帧异常分数CSV（不读取标签）"
    )
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES_ROOT,
                        help="匿名化帧目录，或包含18子目录的父目录")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-id", default=VIDEO_ID)
    parser.add_argument("--repo", default=None,
                        help="Video-Anonymization仓库根目录；默认在项目目录下自动查找")
    args = parser.parse_args()

    for path in (args.model, args.memory):
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")

    frames_dir = resolve_frames_dir(args.frames, args.video_id)
    frames = list_frames(frames_dir)
    total_predictions = len(frames) - INPUT_FRAMES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 视频编号：{args.video_id}")
    print(f"[INFO] 匿名化帧目录：{frames_dir}")
    print(f"[INFO] 帧数：{len(frames)}；将产生{total_predictions}个预测分数")
    print(f"[INFO] FT模型：{args.model}")
    print(f"[INFO] Memory Keys：{args.memory}")
    print(f"[INFO] 推理设备：{device}")

    repo_root = find_reference_root(PROJECT_ROOT, args.repo)
    print(f"[INFO] 参考仓库：{repo_root}")
    print("[INFO] 正在加载FT模型和Memory Keys……")
    model = load_ft_model(args.model, repo_root, device)
    memory = load_memory(args.memory, device)
    print(f"[INFO] Memory shape：{tuple(memory.shape)}")
    print("[INFO] 开始推理：前4帧输入，预测第5帧；不重复运行FA，不读取标签。")

    result = run_inference(model, memory, frames, device)

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / f"video_{args.video_id}_anomaly_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prediction_index", "target_frame", "psnr", "feature_compactness",
            "psnr_anomaly_score", "feature_anomaly_score", "anomaly_score"
        ])
        for i, frame_name in enumerate(result["target_frame_names"]):
            writer.writerow([
                i + 1, frame_name,
                float(result["psnr"][i]), float(result["feature"][i]),
                float(result["psnr_anomaly"][i]),
                float(result["feature_anomaly"][i]),
                float(result["scores"][i]),
            ])

    summary_path = args.output / f"video_{args.video_id}_inference_summary.txt"
    summary_path.write_text(
        f"video={args.video_id}.avi\n"
        f"input_frames={len(frames)}\n"
        f"prediction_frames={len(result['scores'])}\n"
        f"memory_update_count={result['memory_update_count']}\n"
        f"mean_anomaly_score={float(np.mean(result['scores'])):.8f}\n"
        f"std_anomaly_score={float(np.std(result['scores'])):.8f}\n"
        f"mean_prediction_psnr={float(np.mean(result['psnr'])):.8f}\n"
        f"mean_feature_compactness={float(np.mean(result['feature'])):.8f}\n"
        "score_formula=0.6*psnr_anomaly+0.4*feature_anomaly\n"
        "normalization=per-video min-max; both components use 1-normalized\n",
        encoding="utf-8"
    )

    print("\n========== FT 匿名化视频推理完成 ==========")
    print(f"视频：{args.video_id}.avi")
    print(f"输入帧数：{len(frames)}")
    print(f"预测分数数量：{len(result['scores'])}")
    print(f"Memory动态更新次数：{result['memory_update_count']}")
    print(f"异常分数均值：{float(np.mean(result['scores'])):.6f}")
    print(f"异常分数标准差：{float(np.std(result['scores'])):.6f}")
    print(f"预测PSNR均值：{float(np.mean(result['psnr'])):.6f}")
    print(f"Feature Compactness均值：{float(np.mean(result['feature'])):.6f}")
    print(f"[OUTPUT] 逐帧分数：{csv_path}")
    print(f"[OUTPUT] 推理摘要：{summary_path}")

    # Plot in a separate Python process so Matplotlib does not share the
    # PyTorch/OpenMP runtime in this inference process.
    plot_script = Path(__file__).with_name("plot_avenue_video18_curve.py")
    curve_path = args.output / f"video_{args.video_id}_anomaly_curve.png"
    if plot_script.is_file():
        print("[INFO] 启动独立绘图进程，生成异常响应曲线……")
        subprocess.run(
            [sys.executable, str(plot_script), "--csv", str(csv_path),
             "--output", str(curve_path)],
            check=True,
        )
        print(f"[OUTPUT] 异常响应曲线：{curve_path}")
    else:
        print("[WARNING] 未找到独立绘图脚本：" + str(plot_script))
        print("请将 plot_avenue_video18_curve.py 与本脚本放在同一目录后重新运行绘图脚本。")


if __name__ == "__main__":
    main()

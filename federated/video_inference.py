import argparse
import csv
from pathlib import Path
from datetime import datetime

import torch

from model_loader import load_federated_models
from video_preprocessor import VideoPreprocessor


SCENE_TO_CLIENT = {
    "street": "client_0",
    "park": "client_1",
    "zebra_crossing": "client_2",
    "road": "client_3",
}

MASK_RATIO = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated-VAD full video inference"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Input video path",
    )

    parser.add_argument(
        "--scene",
        required=True,
        choices=list(SCENE_TO_CLIENT.keys()),
        help="Scene category",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    video_path = Path(args.video)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video does not exist: {video_path}"
        )

    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Please use --device cpu."
        )

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    client_id = SCENE_TO_CLIENT[args.scene]

    print("=" * 72)
    print("Federated-VAD Full Video Inference")
    print(f"Video      : {video_path}")
    print(f"Scene      : {args.scene}")
    print(f"Client     : {client_id}")
    print(f"Batch size : {args.batch_size}")
    print(f"Mask ratio : {MASK_RATIO}")
    print(f"Seed       : {args.seed}")
    print("=" * 72)

    # 1. 加载模型
    clients, metadata = load_federated_models(
        device=args.device
    )

    client = clients[client_id]
    device = torch.device(metadata["device"])

    client.model.eval()
    client.scene_encoder.eval()

    print(f"\nSelected client: {client_id}")
    print(f"Device: {device}")

    # 2. 初始化视频预处理器
    preprocessor = VideoPreprocessor(
        input_size=metadata["input_size"]
    )

    batches = preprocessor.iter_batches(
        video_path,
        batch_size=args.batch_size,
    )

    all_scores = []
    total_frames = 0
    batch_count = 0

    # 3. 遍历完整视频
    try:
        with torch.inference_mode():

            for samples, gradients, targets in batches:

                samples = samples.to(
                    device=device,
                    dtype=torch.float32,
                )

                gradients = gradients.to(
                    device=device,
                    dtype=torch.float32,
                )

                targets = targets.to(
                    device=device,
                    dtype=torch.float32,
                )

                # 提取场景特征
                scene_embed = client.scene_encoder(samples)

                # 执行 MAE 推理
                result = client.model(
                    samples,
                    targets=targets,
                    grad_mask=gradients,
                    mask_ratio=MASK_RATIO,
                    scene_embed=scene_embed,
                )

                if not isinstance(result, (tuple, list)):
                    raise RuntimeError(
                        "Unexpected model output type"
                    )

                if len(result) != 4:
                    raise RuntimeError(
                        f"Expected 4 outputs, got {len(result)}"
                    )

                loss, prediction, mask, scores = result

                if not torch.isfinite(scores).all().item():
                    raise RuntimeError(
                        "Abnormal scores contain NaN or Inf"
                    )

                batch_scores = (
                    scores.detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )

                all_scores.extend(batch_scores)

                total_frames += len(batch_scores)
                batch_count += 1

                if batch_count % 10 == 0:
                    print(
                        f"Processed batches: {batch_count}, "
                        f"frames: {total_frames}"
                    )

    finally:
        batches.close()

    if total_frames == 0:
        raise RuntimeError(
            "No video frames were processed."
        )

    # 4. 汇总原始异常分数
    score_tensor = torch.tensor(
        all_scores,
        dtype=torch.float32,
    )

    # 5. 保存 CSV
    output_dir = (
        Path("federated")
        / "outputs"
        / "inference"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = output_dir / (
        f"{video_path.stem}_{args.scene}_scores_{timestamp}.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "frame_index",
            "raw_abnormal_score",
        ])

        for index, score in enumerate(all_scores):
            writer.writerow([
                index,
                f"{score:.10f}",
            ])

    # 6. 输出统计结果
    print("\n" + "=" * 72)
    print("FULL VIDEO INFERENCE FINISHED")
    print(f"Scene          : {args.scene}")
    print(f"Client         : {client_id}")
    print(f"Device         : {device}")
    print(f"Total frames   : {total_frames}")
    print(f"Total batches  : {batch_count}")

    print("\nRaw abnormal score statistics:")
    print(f"Min   : {score_tensor.min().item():.8f}")
    print(f"Max   : {score_tensor.max().item():.8f}")
    print(f"Mean  : {score_tensor.mean().item():.8f}")
    print(
        f"Std   : "
        f"{score_tensor.std(unbiased=False).item():.8f}"
    )

    print(f"\nCSV saved to: {output_file.resolve()}")

    print("\nNo smoothing was applied.")
    print("No threshold was applied.")
    print("No normal/abnormal classification was performed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
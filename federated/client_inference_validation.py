
import argparse
import torch

from model_loader import load_federated_models
from video_preprocessor import VideoPreprocessor


SCENE_CLIENTS = {
    "street": "client_0",
    "park": "client_1",
    "zebra_crossing": "client_2",
    "road": "client_3",
}


def main():
    parser = argparse.ArgumentParser(
        description="Validate inference of all four federated clients"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is unavailable.")

    device = torch.device(args.device)

    print("=" * 65)
    print("FOUR-CLIENT INFERENCE VALIDATION")
    print("=" * 65)

    # 1. Load all four personalized clients
    clients, metadata = load_federated_models(device=args.device)
    print("\n[1] Model loading completed.")

    # 2. Initialize preprocessor using its actual interface
    preprocessor = VideoPreprocessor()

    # iter_batches yields (samples, gradients, targets)
    batches = preprocessor.iter_batches(
        args.video,
        batch_size=args.batch_size,
    )

    try:
        samples, gradients, targets = next(iter(batches))
    except StopIteration:
        raise RuntimeError("No frames were generated from the input video.")

    samples = samples.to(device)
    gradients = gradients.to(device)
    targets = targets.to(device)

    print("\n[2] Video preprocessing completed.")
    print(f"Samples shape:   {tuple(samples.shape)}")
    print(f"Gradients shape: {tuple(gradients.shape)}")
    print(f"Targets shape:   {tuple(targets.shape)}")

    # 3. Validate all four clients with the same video batch
    print("\n[3] Starting inference validation.")

    results = []

    for scene, client_name in SCENE_CLIENTS.items():
        print(f"\nTesting {scene} -> {client_name}")

        client = clients[client_name]
        client.model.eval()
        client.scene_encoder.eval()

        # Reset seed so each client uses the same random seed
        torch.manual_seed(args.seed)

        with torch.no_grad():
            scene_embed = client.scene_encoder(samples)

            loss, pred, mask, scores = client.model(
                samples,
                targets=targets,
                grad_mask=gradients.clone(),
                mask_ratio=0.5,
                scene_embed=scene_embed,
            )

        if not torch.isfinite(scene_embed).all():
            raise RuntimeError(
                f"{client_name}: scene embedding contains NaN/Inf."
            )

        if not torch.isfinite(scores).all():
            raise RuntimeError(
                f"{client_name}: scores contain NaN/Inf."
            )

        score_min = scores.min().item()
        score_max = scores.max().item()
        score_mean = scores.mean().item()

        results.append(
            (scene, client_name, score_min, score_max, score_mean)
        )

        print(f"Scene embedding shape: {tuple(scene_embed.shape)}")
        print(f"Score shape: {tuple(scores.shape)}")
        print(f"Score min : {score_min:.8f}")
        print(f"Score max : {score_max:.8f}")
        print(f"Score mean: {score_mean:.8f}")
        print("PASS")

    # 4. Summary
    print("\n" + "=" * 65)
    print("FOUR-CLIENT VALIDATION SUMMARY")
    print("=" * 65)

    for scene, client_name, score_min, score_max, score_mean in results:
        print(
            f"{scene:16s} | {client_name:10s} | "
            f"mean={score_mean:.8f}"
        )

    print("\nALL FOUR CLIENTS PASSED.")
    print("Inference validation only; no anomaly classification performed.")


if __name__ == "__main__":
    main()

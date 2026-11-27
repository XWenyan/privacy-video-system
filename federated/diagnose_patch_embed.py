from __future__ import annotations

from pathlib import Path
import torch

FED_DIR = Path(__file__).resolve().parent
CHECKPOINT = FED_DIR / "weights" / "checkpoint-best-fed.pth"

CLIENT_IDS = ["client_0", "client_1", "client_2", "client_3"]


def compare(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, float, float]:
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    if a.shape != b.shape:
        raise ValueError(f"Tensor shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = (a - b).abs()
    return torch.equal(a, b), float(diff.max().item()), float(diff.mean().item())


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    print(f"Loading: {CHECKPOINT}")
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    global_params = checkpoint["global_params"]
    local_states = checkpoint["local_states"]

    if sorted(local_states) != CLIENT_IDS:
        raise RuntimeError(
            f"Expected clients {CLIENT_IDS}, found {sorted(local_states)}"
        )

    print("\n" + "=" * 76)
    print("PATCH_EMBED CONSISTENCY DIAGNOSTIC")
    print("=" * 76)
    print("This script only compares saved tensors. It does not modify the checkpoint.")

    patch_keys = [
        "proj.weight",
        "proj.bias",
        "norm.weight",
        "norm.bias",
    ]

    scene_states = {}
    global_state = {}

    for key in patch_keys:
        gkey = f"mae_model.patch_embed.{key}"
        if gkey not in global_params:
            raise KeyError(f"Missing global parameter: {gkey}")
        global_state[key] = global_params[gkey]

    for client_id in CLIENT_IDS:
        saved_scene = local_states[client_id].get("scene_encoder")
        if not isinstance(saved_scene, dict):
            raise TypeError(f"{client_id}.scene_encoder is not a dictionary")

        scene_states[client_id] = {}
        print(f"\n[{client_id}] scene_encoder.patch_embed vs global_params")
        for key in patch_keys:
            # scene_encoder is itself a module state_dict. Its patch embedding
            # tensors therefore retain the full "patch_embed." prefix.
            scene_key = f"patch_embed.{key}"
            if scene_key not in saved_scene:
                available = sorted(k for k in saved_scene if k.startswith("patch_embed."))
                raise KeyError(
                    f"{client_id}.scene_encoder missing {scene_key}; "
                    f"available patch_embed keys: {available}"
                )

            scene_tensor = saved_scene[scene_key]
            scene_states[client_id][key] = scene_tensor
            equal, max_abs, mean_abs = compare(scene_tensor, global_state[key])
            print(
                f"  {key:12s} equal={str(equal):5s} "
                f"max_abs_diff={max_abs:.10g} "
                f"mean_abs_diff={mean_abs:.10g}"
            )

    print("\n" + "-" * 76)
    print("PAIRWISE CLIENT patch_embed COMPARISON")
    for i in range(len(CLIENT_IDS)):
        for j in range(i + 1, len(CLIENT_IDS)):
            a_id, b_id = CLIENT_IDS[i], CLIENT_IDS[j]
            all_equal = True
            max_abs_all = 0.0
            mean_abs_values = []

            for key in patch_keys:
                equal, max_abs, mean_abs = compare(
                    scene_states[a_id][key],
                    scene_states[b_id][key],
                )
                all_equal = all_equal and equal
                max_abs_all = max(max_abs_all, max_abs)
                mean_abs_values.append(mean_abs)

            mean_abs_all = sum(mean_abs_values) / len(mean_abs_values)
            print(
                f"  {a_id} vs {b_id}: "
                f"all_equal={all_equal}, "
                f"max_abs_diff={max_abs_all:.10g}, "
                f"mean_of_tensor_mean_abs_diff={mean_abs_all:.10g}"
            )

    print("\n" + "-" * 76)
    print("SCENE ENCODER STATE-DICT KEYS")
    for client_id in CLIENT_IDS:
        print(f"  {client_id}: {sorted(local_states[client_id]['scene_encoder'].keys())}")

    print("\n" + "-" * 76)
    print("SCENE ENCODER PROJECTION CHECK")
    for client_id in CLIENT_IDS:
        state = local_states[client_id]["scene_encoder"]
        projection_keys = [k for k in state if k.startswith("projection.")]
        if not projection_keys:
            raise RuntimeError(f"{client_id} has no projection.* scene encoder parameters")
        print(
            f"  {client_id}: {len(projection_keys)} projection tensors, "
            f"all present"
        )

    print("\nDiagnostic finished.")


if __name__ == "__main__":
    main()

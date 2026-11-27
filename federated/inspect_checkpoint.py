from __future__ import annotations

import io
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import torch


# This script lives in privacy-video-system/federated/.
# The default checkpoint path is therefore resolved relative to this file,
# not relative to the directory from which Python was launched.
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = SCRIPT_DIR / "weights" / "checkpoint-best-fed.pth"
REPORT_PATH = SCRIPT_DIR / "weights" / "checkpoint_inspection.txt"


IMPORTANT_CONFIG_KEYS = [
    "dataset",
    "input_size",
    "patch_size",
    "in_chans",
    "out_chans",
    "embed_dim",
    "depth",
    "num_heads",
    "decoder_embed_dim",
    "decoder_depth",
    "decoder_num_heads",
    "mlp_ratio",
    "fed_scene_dim",
    "mask_ratio",
    "masking_method",
    "fed_mid_dim",
    "fed_scale",
    "fed_prefix_depth",
    "fed_local_parts",
    "abnormal_score_func",
    "grad_weighted_rec_loss",
    "input_3d",
    "run_type",
    "model",
]


def is_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def short_value(value: Any, max_length: int = 180) -> str:
    text = repr(value)
    if len(text) > max_length:
        text = text[: max_length - 3] + "..."
    return text


def mapping_view(value: Any) -> Mapping | None:
    """Return mapping-like config objects as a Mapping when possible."""
    if isinstance(value, Mapping):
        return value

    # Some config classes expose to_dict() but do not implement Mapping.
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
            if isinstance(converted, Mapping):
                return converted
        except Exception:
            pass

    # argparse.Namespace and similar objects.
    if hasattr(value, "__dict__") and not isinstance(value, type):
        try:
            return vars(value)
        except Exception:
            pass

    return None


def tensor_description(tensor: torch.Tensor) -> str:
    try:
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        elements = tensor.numel()
        return f"Tensor(shape={shape}, dtype={dtype}, numel={elements:,})"
    except Exception as exc:
        return f"Tensor(unable to inspect: {exc})"


def print_tree(
    value: Any,
    *,
    indent: int = 0,
    depth: int = 0,
    max_depth: int = 4,
    max_items: int = 45,
) -> None:
    """Print structure and tensor metadata, never tensor contents."""
    prefix = " " * indent

    if is_tensor(value):
        print(prefix + tensor_description(value))
        return

    mapped = mapping_view(value)
    if mapped is not None:
        print(prefix + f"{type(value).__name__}(keys={len(mapped)})")
        if depth >= max_depth:
            print(prefix + "  ... (maximum display depth reached)")
            return
        for index, (key, child) in enumerate(mapped.items()):
            if index >= max_items:
                print(prefix + f"  ... ({len(mapped) - max_items} more entries)")
                break
            print(prefix + f"  [{key!r}]:")
            print_tree(
                child,
                indent=indent + 4,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )
        return

    if isinstance(value, (list, tuple)):
        print(prefix + f"{type(value).__name__}(length={len(value)})")
        if depth >= max_depth:
            print(prefix + "  ... (maximum display depth reached)")
            return
        for index, child in enumerate(value[:max_items]):
            print(prefix + f"  [{index}]:")
            print_tree(
                child,
                indent=indent + 4,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )
        if len(value) > max_items:
            print(prefix + f"  ... ({len(value) - max_items} more entries)")
        return

    print(prefix + f"{type(value).__name__}: {short_value(value)}")


def collect_tensor_info(value: Any, path: str = "root") -> tuple[int, int, list[str]]:
    """Return tensor count, total numel and a limited list of tensor paths."""
    count = 0
    total_numel = 0
    examples: list[str] = []

    if is_tensor(value):
        return 1, int(value.numel()), [f"{path}: {tensor_description(value)}"]

    mapped = mapping_view(value)
    if mapped is not None:
        for key, child in mapped.items():
            child_count, child_numel, child_examples = collect_tensor_info(
                child, f"{path}.{key}"
            )
            count += child_count
            total_numel += child_numel
            if len(examples) < 25:
                examples.extend(child_examples[: 25 - len(examples)])
        return count, total_numel, examples

    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_count, child_numel, child_examples = collect_tensor_info(
                child, f"{path}[{index}]"
            )
            count += child_count
            total_numel += child_numel
            if len(examples) < 25:
                examples.extend(child_examples[: 25 - len(examples)])

    return count, total_numel, examples


def flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    mapped = mapping_view(value)
    if mapped is not None:
        for key, child in mapped.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            child_mapping = mapping_view(child)
            if child_mapping is not None:
                result.update(flatten_config(child, name))
            else:
                result[name] = child
    return result


def find_value_by_key(flat: dict[str, Any], target_key: str) -> list[tuple[str, Any]]:
    target = target_key.lower()
    return [
        (path, value)
        for path, value in flat.items()
        if path.split(".")[-1].lower() == target
    ]


def print_tensor_summary(title: str, value: Any) -> None:
    print(f"\n{title}")
    if value is None:
        print("  [NOT FOUND]")
        return

    count, total_numel, examples = collect_tensor_info(value, title)
    print(f"  Tensor count : {count:,}")
    print(f"  Total numel  : {total_numel:,}")
    if count == 0:
        print("  No tensors found in this section.")
    else:
        print("  Tensor examples (up to 25):")
        for item in examples:
            print(f"    - {item}")


def describe_checkpoint(checkpoint: Any, checkpoint_path: Path) -> None:
    print("=" * 78)
    print("Federated-VAD checkpoint inspection")
    print("=" * 78)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"File size  : {checkpoint_path.stat().st_size / (1024 ** 2):.2f} MB")
    print(f"PyTorch    : {torch.__version__}")
    print(f"Root type  : {type(checkpoint).__name__}")

    root = mapping_view(checkpoint)
    if root is None:
        print("\n[WARNING] The checkpoint root is not a dictionary-like object.")
        print_tree(checkpoint, max_depth=2)
        return

    print("\n[1] Top-level checkpoint keys")
    for key in root.keys():
        print(f"  - {key}")

    print("\n[2] Main metadata fields")
    for key in ("round", "epoch", "micro_auc", "macro_auc", "best_auc", "best_metric"):
        if key in root:
            print(f"  {key}: {short_value(root[key])}")

    print("\n[3] Global model parameters")
    global_params = root.get("global_params")
    if global_params is None:
        # Report likely alternatives without assuming they are equivalent.
        for alternative in ("state_dict", "model", "global_state_dict"):
            if alternative in root:
                print(f"  'global_params' not found; possible alternative key: {alternative!r}")
                global_params = root[alternative]
                break
    print_tensor_summary("Global parameter summary", global_params)

    print("\n[4] Personalized client states")
    local_states = root.get("local_states")
    if local_states is None:
        for alternative in ("client_states", "local_params", "clients"):
            if alternative in root:
                print(f"  'local_states' not found; possible alternative key: {alternative!r}")
                local_states = root[alternative]
                break

    local_mapping = mapping_view(local_states) if local_states is not None else None
    if local_mapping is None:
        print("  [NOT FOUND] No dictionary-like local client state section was identified.")
        if local_states is not None:
            print_tree(local_states, indent=2, max_depth=2)
    else:
        print(f"  Client/state entries found: {len(local_mapping)}")
        for client_name, client_state in local_mapping.items():
            print(f"\n  Client: {client_name}")
            state_mapping = mapping_view(client_state)
            if state_mapping is not None:
                print(f"    State keys: {list(state_mapping.keys())}")
            count, total_numel, examples = collect_tensor_info(
                client_state, f"local_states.{client_name}"
            )
            print(f"    Tensor count: {count:,}")
            print(f"    Total numel : {total_numel:,}")
            for item in examples[:12]:
                print(f"      - {item}")

    print("\n[5] Saved training/model configuration")
    args = root.get("args")
    if args is None:
        print("  [NOT FOUND] Top-level key 'args' is absent.")
    else:
        flat_args = flatten_config(args)
        if not flat_args:
            print("  Config exists but could not be converted to a key/value mapping.")
            print_tree(args, indent=2, max_depth=2)
        else:
            for config_key in IMPORTANT_CONFIG_KEYS:
                matches = find_value_by_key(flat_args, config_key)
                for path, value in matches:
                    print(f"  {path} = {short_value(value)}")
            print(f"  Total flattened config entries: {len(flat_args)}")

    print("\n[6] Full checkpoint structure (tensor values are NOT printed)")
    print_tree(checkpoint, max_depth=4, max_items=35)

    print("\n[7] Initial assessment")
    if global_params is not None and local_mapping is not None:
        print("  Global parameters and a client-state section are both present.")
        print("  This is consistent with a federated checkpoint containing shared and personalized state.")
    elif global_params is not None:
        print("  Global parameters are present, but client-specific states were not identified.")
    else:
        print("  The expected global parameter section was not identified; inspect the keys above.")

    if local_mapping is not None:
        client_names = [str(name) for name in local_mapping.keys()]
        print(f"  Client entries: {', '.join(client_names)}")
        expected = {"client_0", "client_1", "client_2", "client_3"}
        if expected.issubset(set(client_names)):
            print("  The four expected client IDs (client_0 to client_3) are present.")
        else:
            print("  The client IDs differ from client_0 to client_3; verify against the training run.")

    print("\nInspection finished. No model weights were modified.")


def main() -> int:
    print(f"Looking for checkpoint: {CHECKPOINT_PATH}")
    if not CHECKPOINT_PATH.is_file():
        print("[ERROR] Checkpoint file was not found.")
        print("        Expected path: federated/weights/final-best-fed-model.pth")
        print("        Please verify the folder and filename.")
        return 2

    try:
        # weights_only=False is needed when the checkpoint also stores a Python
        # config object (for example, a ConfigDict). Only load a checkpoint from
        # a source you trust, because pickle-based loading can execute code.
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions that do not accept
        # the weights_only argument.
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        except Exception as exc:
            print(f"[ERROR] Failed to load checkpoint: {type(exc).__name__}: {exc}")
            return 3
    except Exception as exc:
        print(f"[ERROR] Failed to load checkpoint: {type(exc).__name__}: {exc}")
        print("Possible causes: incompatible PyTorch/package versions, or missing modules")
        print("used when the checkpoint was saved. Keep the full traceback for debugging.")
        import traceback
        traceback.print_exc()
        return 3

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        describe_checkpoint(checkpoint, CHECKPOINT_PATH)

    report = buffer.getvalue()
    print(report, end="")

    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"\n[REPORT SAVED] {REPORT_PATH}")
    except OSError as exc:
        print(f"\n[WARNING] Could not save report: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Load and validate the four personalized Federated-VAD ShanghaiTech models.

Expected source files:
    federated/source/model/mae_vit.py
    federated/source/fed/prefix_mae_ge.py

Expected checkpoint:
    federated/weights/checkpoint-best-fed.pth

This script does NOT read videos and does NOT run anomaly inference.
It only constructs the four client models and strictly loads their saved states.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


FEDERATED_DIR = Path(__file__).resolve().parent
SOURCE_DIR = FEDERATED_DIR / "source"
DEFAULT_CHECKPOINT = FEDERATED_DIR / "weights" / "checkpoint-best-fed.pth"

# The order is defined by SCENE_PREFIX_MAP insertion order in Federated-VAD/fed_main.py.
SCENE_CLIENTS = (
    ("client_0", "street", "街道"),
    ("client_1", "park", "公园"),
    ("client_2", "zebra_crossing", "斑马线"),
    ("client_3", "road", "道路"),
)


@dataclass
class LoadedClient:
    client_id: str
    scene_key: str
    scene_name: str
    model: nn.Module
    scene_encoder: nn.Module


class LightweightSceneEncoder(nn.Module):
    """
    Same structure as LightweightSceneEncoder in fed/fed_mae_trainer.py.
    Its patch_embed is intentionally shared with the client's MAE model.
    """

    def __init__(self, mae_model: nn.Module, scene_dim: int = 64):
        super().__init__()
        self.patch_embed = mae_model.patch_embed

        # Match the original Federated-VAD trainer implementation.
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(mae_model.embed_dim, scene_dim),
            nn.ReLU(),
            nn.Linear(scene_dim, scene_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_features = self.patch_embed(x)
        global_feat = patch_features.mean(dim=1)
        return self.projection(global_feat)


def _prepare_source_imports():
    model_file = SOURCE_DIR / "model" / "mae_vit.py"
    prefix_file = SOURCE_DIR / "fed" / "prefix_mae_ge.py"

    missing = [str(p) for p in (model_file, prefix_file) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少 Federated-VAD 原项目源码文件：\n  - "
            + "\n  - ".join(missing)
            + "\n请按说明将 mae_vit.py 和 prefix_mae_ge.py 放入 federated/source 对应目录。"
        )

    # Namespace packages work without __init__.py, but add these files
    # to make the copied source directories explicit Python packages.
    for package_dir in (SOURCE_DIR / "model", SOURCE_DIR / "fed"):
        package_dir.mkdir(parents=True, exist_ok=True)
        init_file = package_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Federated-VAD source package\n", encoding="utf-8")

    source_str = str(SOURCE_DIR)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)

    try:
        from model.mae_vit import MaskedAutoencoderViT
        from fed.prefix_mae_ge import build_prefix_fed_mae
    except Exception as exc:
        raise RuntimeError(
            "导入 Federated-VAD 模型源码失败。请检查 timm、einops、numpy 等依赖，"
            f"以及源码文件是否为同一版本。原始错误：{type(exc).__name__}: {exc}"
        ) from exc

    return MaskedAutoencoderViT, build_prefix_fed_mae


def _cfg(args: Any, name: str):
    if not hasattr(args, name):
        raise ValueError(f"Checkpoint 配置 args 缺少必要字段：{name}")
    return getattr(args, name)


def _validate_tensor_mapping(
    label: str,
    expected: dict[str, torch.Tensor],
    saved: dict[str, torch.Tensor],
) -> None:
    if not isinstance(saved, dict):
        raise TypeError(f"{label} 必须是字典，实际类型：{type(saved).__name__}")

    expected_keys = set(expected)
    saved_keys = set(saved)

    missing = sorted(expected_keys - saved_keys)
    unexpected = sorted(saved_keys - expected_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"缺失参数（{len(missing)}）：{missing[:12]}")
        if unexpected:
            details.append(f"多余参数（{len(unexpected)}）：{unexpected[:12]}")
        raise RuntimeError(f"{label} 参数名称不匹配：" + "；".join(details))

    for key, expected_tensor in expected.items():
        saved_tensor = saved[key]
        if not torch.is_tensor(saved_tensor):
            raise TypeError(
                f"{label}.{key} 不是 Tensor，实际类型：{type(saved_tensor).__name__}"
            )
        if tuple(saved_tensor.shape) != tuple(expected_tensor.shape):
            raise RuntimeError(
                f"{label}.{key} 形状不匹配："
                f"checkpoint={tuple(saved_tensor.shape)}, "
                f"model={tuple(expected_tensor.shape)}"
            )


def _assert_loaded_equal(
    label: str,
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    """Verify values after load, not only key names and tensor shapes."""
    mismatches = []
    for key, expected_tensor in expected.items():
        actual_tensor = actual[key].detach().cpu()
        expected_cpu = expected_tensor.detach().cpu()
        if not torch.equal(actual_tensor, expected_cpu):
            mismatches.append(key)
            if len(mismatches) >= 10:
                break

    if mismatches:
        raise RuntimeError(
            f"{label} 加载后数值校验失败，示例参数：{mismatches}"
        )


def _build_base_model(args: Any, model_class):
    input_size = tuple(int(x) for x in _cfg(args, "input_size"))
    if input_size != (160, 320):
        raise ValueError(
            f"当前加载器针对 ShanghaiTech (160, 320) 配置，"
            f"checkpoint 中却是 {input_size}"
        )

    # These architecture values match the Shanghai branch in fed_main.py.
    return model_class(
        patch_size=8,
        in_chans=9,
        out_chans=4,
        img_size=input_size,
        embed_dim=256,
        depth=3,
        num_heads=4,
        decoder_embed_dim=128,
        decoder_depth=3,
        decoder_num_heads=4,
        mlp_ratio=2,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=bool(_cfg(args, "norm_pix_loss")),
        use_only_masked_tokens_ab=bool(_cfg(args, "use_only_masked_tokens_ab")),
        abnormal_score_func=_cfg(args, "abnormal_score_func"),
        masking_method=_cfg(args, "masking_method"),
        grad_weighted_loss=bool(_cfg(args, "grad_weighted_rec_loss")),
    ).float()


def load_federated_models(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[dict[str, LoadedClient], dict[str, Any]]:
    """
    Build four client-specific model/scene-encoder pairs and load checkpoint states.

    Returns:
        clients: keyed by client_0 ... client_3
        metadata: round, AUCs, checkpoint path, selected device, etc.
    """
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    MaskedAutoencoderViT, build_prefix_fed_mae = _prepare_source_imports()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print("=" * 72)
    print("Federated-VAD model loading")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")
    print("=" * 72)

    # weights_only=False is needed because this trusted checkpoint stores
    # an ml_collections.ConfigDict in checkpoint['args'].
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required_root_keys = {"global_params", "local_states", "args", "round"}
    missing_root = required_root_keys - set(checkpoint)
    if missing_root:
        raise RuntimeError(f"Checkpoint 缺少字段：{sorted(missing_root)}")

    args = checkpoint["args"]
    local_states = checkpoint["local_states"]
    global_params = checkpoint["global_params"]

    if _cfg(args, "dataset") != "shanghai":
        raise ValueError(f"当前只支持 shanghai checkpoint，实际为：{args.dataset}")

    saved_client_count = int(_cfg(args, "fed_num_clients"))
    if saved_client_count != len(SCENE_CLIENTS):
        raise ValueError(
            f"期望 4 个客户端，但 checkpoint 配置 fed_num_clients={saved_client_count}"
        )

    expected_client_ids = [item[0] for item in SCENE_CLIENTS]
    saved_client_ids = sorted(local_states.keys(), key=lambda x: int(x.split("_")[-1]))
    if saved_client_ids != expected_client_ids:
        raise ValueError(
            "Checkpoint 客户端 ID 与四场景映射不一致："
            f"expected={expected_client_ids}, actual={saved_client_ids}"
        )

    if not isinstance(global_params, dict):
        raise TypeError("checkpoint['global_params'] 不是字典")

    print(f"Training round : {checkpoint.get('round')}")
    print(f"Micro-AUC      : {float(checkpoint.get('micro_auc', float('nan'))):.6f}")
    print(f"Macro-AUC      : {float(checkpoint.get('macro_auc', float('nan'))):.6f}")
    print(f"Global tensors : {len(global_params)}")
    print(f"Clients        : {saved_client_ids}")
    print(f"Input size     : {tuple(_cfg(args, 'input_size'))}")
    print("-" * 72)

    base_mae = _build_base_model(args, MaskedAutoencoderViT)
    clients: dict[str, LoadedClient] = {}

    for client_id, scene_key, scene_name in SCENE_CLIENTS:
        print(f"\n[{client_id}] Loading personalized model for {scene_name} ({scene_key})")

        local_state = local_states[client_id]
        required_local_keys = {"local_params", "scene_encoder"}
        missing_local = required_local_keys - set(local_state)
        if missing_local:
            raise RuntimeError(
                f"{client_id} 的 local_state 缺少字段：{sorted(missing_local)}"
            )

        # build_prefix_fed_mae deep-copies base_mae and injects the
        # scene-aware attention adapters defined in prefix_mae_ge.py.
        client_model = build_prefix_fed_mae(base_mae, args)

        expected_global = client_model.get_global_params()
        expected_local = client_model.get_local_params()

        _validate_tensor_mapping(
            f"{client_id}.global_params", expected_global, global_params
        )
        _validate_tensor_mapping(
            f"{client_id}.local_params",
            expected_local,
            local_state["local_params"],
        )

        # Use the original project's parameter-loading methods.
        client_model.load_global_params(global_params)
        client_model.load_local_params(local_state["local_params"])

        scene_encoder = LightweightSceneEncoder(
            client_model.mae_model,
            scene_dim=int(_cfg(args, "fed_scene_dim")),
        )

        expected_scene = scene_encoder.state_dict()
        saved_scene = local_state["scene_encoder"]
        _validate_tensor_mapping(
            f"{client_id}.scene_encoder", expected_scene, saved_scene
        )

        # The original trainer's load_local_state() loads scene_encoder
        # after global_params/local_params. Keep this ordering.
        scene_encoder.load_state_dict(saved_scene, strict=True)

        # Validate local decoder and scene encoder values exactly.
        _assert_loaded_equal(
            f"{client_id}.local_params",
            client_model.get_local_params(),
            local_state["local_params"],
        )
        _assert_loaded_equal(
            f"{client_id}.scene_encoder",
            scene_encoder.state_dict(),
            saved_scene,
        )

        # In the source trainer, scene_encoder.patch_embed aliases the
        # MAE model's patch_embed. Report whether this saved copy is also
        # identical to the separately saved global patch embedding.
        global_after_scene_load = client_model.get_global_params()
        patch_embed_matches_global = True
        for key, global_tensor in global_params.items():
            if key.startswith("mae_model.patch_embed."):
                if not torch.equal(
                    global_after_scene_load[key].detach().cpu(),
                    global_tensor.detach().cpu(),
                ):
                    patch_embed_matches_global = False
                    break

        # All non-patch-embed global parameters must still match the
        # shared global checkpoint after loading the scene encoder.
        non_patch_expected = {
            key: value
            for key, value in global_params.items()
            if not key.startswith("mae_model.patch_embed.")
        }
        non_patch_actual = {
            key: value
            for key, value in global_after_scene_load.items()
            if not key.startswith("mae_model.patch_embed.")
        }
        _assert_loaded_equal(
            f"{client_id}.global_params (excluding shared patch_embed)",
            non_patch_actual,
            non_patch_expected,
        )

        client_model.to(device)
        scene_encoder.to(device)
        client_model.eval()
        scene_encoder.eval()

        if scene_encoder.patch_embed is not client_model.mae_model.patch_embed:
            raise RuntimeError(
                f"{client_id}: scene_encoder.patch_embed 未与 MAE patch_embed 共享"
            )

        clients[client_id] = LoadedClient(
            client_id=client_id,
            scene_key=scene_key,
            scene_name=scene_name,
            model=client_model,
            scene_encoder=scene_encoder,
        )

        print(f"  global params : {len(global_params)} tensors - structure OK")
        print(f"  local params  : {len(local_state['local_params'])} tensors - values OK")
        print(f"  scene encoder : {len(saved_scene)} tensors - values OK")
        print(
            "  patch_embed matches global checkpoint: "
            + ("YES" if patch_embed_matches_global else "NO (see note below)")
        )
        print(f"  device        : {device}")
        print(f"  RESULT        : {client_id} LOAD PASS")

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "round": int(checkpoint["round"]),
        "micro_auc": float(checkpoint.get("micro_auc", float("nan"))),
        "macro_auc": float(checkpoint.get("macro_auc", float("nan"))),
        "dataset": str(_cfg(args, "dataset")),
        "input_size": tuple(_cfg(args, "input_size")),
        "client_ids": expected_client_ids,
        "device": str(device),
    }

    del checkpoint, local_states, global_params, base_mae
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print(f"SUCCESS: {len(clients)}/4 personalized client models loaded.")
    print("No video preprocessing or anomaly inference was performed.")
    print("=" * 72)

    return clients, metadata


def main() -> int:
    try:
        clients, metadata = load_federated_models()
        print("\nLoaded scene mapping:")
        for client_id, client in clients.items():
            print(f"  {client_id:8s} -> {client.scene_name} ({client.scene_key})")
        print(f"\nCheckpoint round : {metadata['round']}")
        print(f"Micro-AUC        : {metadata['micro_auc']:.6f}")
        print(f"Macro-AUC        : {metadata['macro_auc']:.6f}")
        print(f"Device           : {metadata['device']}")
        return 0
    except Exception as exc:
        print("\n[MODEL LOAD FAILED]")
        print(f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

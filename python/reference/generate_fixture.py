"""Generate and verify the deterministic tiny Qwen3-MoE oracle fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy
import safetensors
import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb


FIXTURE_VERSION = "tiny-qwen3-moe-v1"
SEED = 20_260_714
INPUT_IDS = [[1, 5, 7, 2]]
MODEL_ID = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
REFERENCE_ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURE_DIR = REFERENCE_ROOT / "fixtures" / "tiny-qwen3-moe"
EXPECTED_VERSIONS = {
    "python": "3.12.10",
    "numpy": "2.3.3",
    "safetensors": "0.8.0",
    "torch": "2.12.1",
    "transformers": "5.12.1",
}
GENERATED_FILES = (
    "checkpoints.safetensors",
    "config.json",
    "environment.json",
    "inputs.json",
    "rust-config.rs",
    "tensor-manifest.json",
    "tolerances.json",
    "weights.safetensors",
)


def _actual_versions() -> dict[str, str]:
    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "numpy": numpy.__version__,
        "safetensors": safetensors.__version__,
        "torch": torch.__version__.split("+")[0],
        "transformers": transformers.__version__,
    }


def _check_versions() -> None:
    actual = _actual_versions()
    mismatches = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in EXPECTED_VERSIONS.items()
        if actual[name] != expected
    }
    if mismatches:
        details = json.dumps(mismatches, indent=2, sort_keys=True)
        raise RuntimeError(f"reference environment version mismatch:\n{details}")


def _configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)


def _tiny_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        initializer_range=0.02,
        rms_norm_eps=1.0e-6,
        use_cache=False,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        moe_intermediate_size=24,
        num_experts_per_tok=2,
        num_experts=4,
        norm_topk_prob=False,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        dtype="float32",
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
    )


def _initialize_synthetic_weights(model: Qwen3MoeForCausalLM) -> None:
    """Replace framework initialization with byte-reproducible CPU values."""
    norm_suffixes = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "q_norm.weight",
        "k_norm.weight",
        "model.norm.weight",
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(norm_suffixes):
                parameter.fill_(1.0)
                continue
            name_seed = int.from_bytes(
                hashlib.sha256(f"{SEED}:{name}".encode()).digest()[:8], "little"
            )
            indices = torch.arange(parameter.numel(), dtype=torch.int64)
            values = ((indices * 37 + name_seed) % 2001) - 1000
            values = values.to(dtype=torch.float32).mul_(2.0e-5)
            parameter.copy_(values.reshape(parameter.shape))


def _as_tensor(output: Any, index: int | None = None) -> torch.Tensor:
    value = output if index is None else output[index]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor hook output, got {type(value).__name__}")
    return value.detach().cpu().contiguous().clone()


def _capture_hook(
    captures: dict[str, torch.Tensor],
    name: str,
    index: int | None = None,
) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        captures[name] = _as_tensor(output, index)

    return hook


def _router_hook(
    captures: dict[str, torch.Tensor], layer_index: int
) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, tuple) or len(output) != 3:
            raise TypeError("router output contract changed; expected three tensors")
        router_logits, routing_weights, selected_experts = output
        prefix = f"layer_{layer_index}"
        captures[f"{prefix}.router_logits"] = _as_tensor(router_logits)
        captures[f"{prefix}.routing_weights"] = _as_tensor(routing_weights)
        captures[f"{prefix}.selected_experts"] = _as_tensor(selected_experts)

    return hook


def _register_hooks(
    model: Qwen3MoeForCausalLM, captures: dict[str, torch.Tensor]
) -> list[torch.utils.hooks.RemovableHandle]:
    modules = dict(model.named_modules())
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_index in range(model.config.num_hidden_layers):
        prefix = f"model.layers.{layer_index}"
        fixture_prefix = f"layer_{layer_index}"
        handles.extend(
            [
                modules[f"{prefix}.input_layernorm"].register_forward_hook(
                    _capture_hook(captures, f"{fixture_prefix}.input_norm")
                ),
                modules[f"{prefix}.self_attn"].register_forward_hook(
                    _capture_hook(captures, f"{fixture_prefix}.attention_output", 0)
                ),
                modules[f"{prefix}.self_attn.q_norm"].register_forward_hook(
                    _capture_hook(captures, f"{fixture_prefix}.query_norm_heads")
                ),
                modules[f"{prefix}.self_attn.k_norm"].register_forward_hook(
                    _capture_hook(captures, f"{fixture_prefix}.key_norm_heads")
                ),
                modules[f"{prefix}.post_attention_layernorm"].register_forward_hook(
                    _capture_hook(captures, f"{fixture_prefix}.post_attention_norm")
                ),
                modules[f"{prefix}.mlp.gate"].register_forward_hook(
                    _router_hook(captures, layer_index)
                ),
                modules[f"{prefix}.mlp"].register_forward_hook(
                    _capture_hook(captures, f"{fixture_prefix}.moe_output")
                ),
                modules[prefix].register_forward_hook(
                    _capture_hook(captures, f"{fixture_prefix}.block_output")
                ),
            ]
        )
    handles.append(
        modules["model.norm"].register_forward_hook(
            _capture_hook(captures, "final_norm")
        )
    )
    return handles


def _build_fixture_tensors() -> tuple[
    Qwen3MoeConfig,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    _configure_determinism()
    config = _tiny_config()
    model = Qwen3MoeForCausalLM(config).cpu().eval()
    _initialize_synthetic_weights(model)
    captures: dict[str, torch.Tensor] = {}
    handles = _register_hooks(model, captures)
    input_ids = torch.tensor(INPUT_IDS, dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)

    try:
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_router_logits=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    captures["input_ids"] = input_ids.contiguous()
    captures["attention_mask"] = attention_mask.contiguous()
    captures["final_logits"] = output.logits.detach().cpu().contiguous().clone()
    for index, hidden_state in enumerate(output.hidden_states):
        captures[f"hidden_state_{index}"] = (
            hidden_state.detach().cpu().contiguous().clone()
        )

    for layer_index, router_logits in enumerate(output.router_logits):
        captured = captures[f"layer_{layer_index}.router_logits"]
        if not torch.equal(captured, router_logits.detach().cpu()):
            raise RuntimeError(f"router hook mismatch at layer {layer_index}")

    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).unsqueeze(0)
    with torch.no_grad():
        cosine, sine = model.model.rotary_emb(output.hidden_states[0], position_ids)
        for layer_index in range(config.num_hidden_layers):
            query = captures.pop(f"layer_{layer_index}.query_norm_heads").transpose(1, 2)
            key = captures.pop(f"layer_{layer_index}.key_norm_heads").transpose(1, 2)
            query_rope, key_rope = apply_rotary_pos_emb(query, key, cosine, sine)
            captures[f"layer_{layer_index}.query_rope"] = (
                query_rope.detach().cpu().contiguous().clone()
            )
            captures[f"layer_{layer_index}.key_rope"] = (
                key_rope.detach().cpu().contiguous().clone()
            )

    weights = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    return config, weights, captures


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _tensor_inventory(path: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    with safe_open(path, framework="pt", device="cpu") as artifact:
        for name in artifact.keys():
            tensor = artifact.get_tensor(name)
            inventory[name] = {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "byte_count": tensor.numel() * tensor.element_size(),
            }
    return inventory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_hashes(output_dir: Path) -> None:
    _write_json(
        output_dir / "sha256.json",
        {
            "algorithm": "sha256",
            "files": {
                name: _sha256(output_dir / name) for name in GENERATED_FILES
            },
            "fixture_version": FIXTURE_VERSION,
        },
    )


def _write_rust_config(output_dir: Path, config: Qwen3MoeConfig) -> None:
    rope_theta = float(config.rope_parameters["rope_theta"])
    content = (
        "// Generated by python/reference/generate_fixture.py.\n"
        "// Test-only constants derived from the frozen Transformers config.\n"
        f"pub const VOCABULARY_SIZE: usize = {config.vocab_size};\n"
        f"pub const HIDDEN_SIZE: usize = {config.hidden_size};\n"
        f"pub const LAYER_COUNT: usize = {config.num_hidden_layers};\n"
        f"pub const ATTENTION_HEAD_COUNT: usize = {config.num_attention_heads};\n"
        f"pub const KEY_VALUE_HEAD_COUNT: usize = {config.num_key_value_heads};\n"
        f"pub const HEAD_DIMENSION: usize = {config.head_dim};\n"
        f"pub const INTERMEDIATE_SIZE: usize = {config.intermediate_size};\n"
        f"pub const MAX_SEQUENCE_LENGTH: usize = {config.max_position_embeddings};\n"
        f"pub const RMS_NORM_EPSILON: f32 = {config.rms_norm_eps!r}_f32;\n"
        f"pub const ROPE_THETA: f32 = {rope_theta!r}_f32;\n"
        f"pub const EXPERT_COUNT: usize = {config.num_experts};\n"
        f"pub const EXPERTS_PER_TOKEN: usize = {config.num_experts_per_tok};\n"
        f"pub const MOE_INTERMEDIATE_SIZE: usize = {config.moe_intermediate_size};\n"
        f"pub const NORMALIZE_TOPK_PROBABILITIES: bool = {str(config.norm_topk_prob).lower()};\n"
    )
    (output_dir / "rust-config.rs").write_text(
        content, encoding="utf-8", newline="\n"
    )


def generate(output_dir: Path) -> None:
    _check_versions()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (*GENERATED_FILES, "sha256.json"):
        (output_dir / name).unlink(missing_ok=True)

    config, weights, checkpoints = _build_fixture_tensors()
    _write_json(output_dir / "config.json", config.to_dict())
    _write_json(
        output_dir / "inputs.json",
        {"attention_mask": [[1, 1, 1, 1]], "input_ids": INPUT_IDS},
    )
    _write_json(
        output_dir / "environment.json",
        {
            "artifact_format_version": FIXTURE_VERSION,
            "device": "cpu",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "seed": SEED,
            "tool_versions": EXPECTED_VERSIONS,
        },
    )
    _write_json(
        output_dir / "tolerances.json",
        {
            "exact": [
                "tensor_names",
                "tensor_shapes",
                "input_ids",
                "attention_mask",
                "selected_experts",
            ],
            "floating_point": {
                "default": {"absolute": 1.0e-6, "relative": 1.0e-5},
                "final_logits": {"absolute": 1.0e-6, "relative": 1.0e-5},
                "router_logits": {"absolute": 1.0e-7, "relative": 1.0e-6},
            },
        },
    )
    _write_rust_config(output_dir, config)
    # Safetensors serializes multi-key metadata through a randomized hash map.
    # Keep one stable embedded key; complete provenance lives in environment.json.
    metadata = {"fixture_version": FIXTURE_VERSION}
    save_file(weights, output_dir / "weights.safetensors", metadata=metadata)
    save_file(
        checkpoints,
        output_dir / "checkpoints.safetensors",
        metadata=metadata,
    )
    _write_json(
        output_dir / "tensor-manifest.json",
        {
            "artifact_format_version": FIXTURE_VERSION,
            "checkpoints": _tensor_inventory(output_dir / "checkpoints.safetensors"),
            "weights": _tensor_inventory(output_dir / "weights.safetensors"),
        },
    )
    _write_hashes(output_dir)


def _load_hash_manifest(fixture_dir: Path) -> Mapping[str, Any]:
    with (fixture_dir / "sha256.json").open(encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def _verify_committed_hashes(fixture_dir: Path) -> None:
    manifest = _load_hash_manifest(fixture_dir)
    if manifest.get("fixture_version") != FIXTURE_VERSION:
        raise RuntimeError("fixture version does not match the generator")
    expected_files = manifest.get("files", {})
    if set(expected_files) != set(GENERATED_FILES):
        raise RuntimeError("hash manifest file set does not match generator outputs")
    for name, expected_hash in expected_files.items():
        actual_hash = _sha256(fixture_dir / name)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"committed hash mismatch for {name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )


def verify(fixture_dir: Path) -> None:
    _check_versions()
    _verify_committed_hashes(fixture_dir)
    committed_hashes = _load_hash_manifest(fixture_dir)["files"]
    with tempfile.TemporaryDirectory(prefix="colibri-oracle-") as temporary:
        regenerated_dir = Path(temporary) / "tiny-qwen3-moe"
        generate(regenerated_dir)
        regenerated_hashes = _load_hash_manifest(regenerated_dir)["files"]
    if regenerated_hashes != committed_hashes:
        differences = {
            name: {
                "committed": committed_hashes[name],
                "regenerated": regenerated_hashes[name],
            }
            for name in GENERATED_FILES
            if committed_hashes[name] != regenerated_hashes[name]
        }
        raise RuntimeError(
            "fixture regeneration mismatch:\n"
            + json.dumps(differences, indent=2, sort_keys=True)
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "verify"))
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
        help="fixture directory (defaults to the committed tiny fixture)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    fixture_dir = arguments.fixture_dir.resolve()
    if arguments.command == "generate":
        generate(fixture_dir)
        print(f"generated {FIXTURE_VERSION} at {fixture_dir}")
    else:
        verify(fixture_dir)
        print(f"verified {FIXTURE_VERSION} at {fixture_dir}")


if __name__ == "__main__":
    main()

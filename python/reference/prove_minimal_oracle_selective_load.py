#!/usr/bin/env python3
"""Offline proof that the minimal source can selectively resolve Qwen3-MoE tensors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

from safetensors import safe_open
import torch
import transformers
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention, Qwen3MoeMLP, Qwen3MoeRMSNorm, Qwen3MoeTopKRouter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.validate_minimal_oracle_source import validate


class ProofError(RuntimeError):
    """The frozen selective-load oracle contract did not hold."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProofError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_from_manifest(manifest: dict[str, Any]) -> Qwen3MoeConfig:
    source = manifest["config"]
    expected = {
        "architecture": "Qwen3MoeForCausalLM", "model_type": "qwen3_moe", "torch_dtype": "bfloat16",
        "vocab_size": 151936, "hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 48,
        "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128, "max_position_embeddings": 40960,
        "rms_norm_eps": 1.0e-6, "rope_theta": 1_000_000.0, "num_experts": 128,
        "num_experts_per_tok": 8, "moe_intermediate_size": 768, "norm_topk_prob": True,
        "decoder_sparse_step": 1, "mlp_only_layers": [], "attention_bias": False, "attention_dropout": 0.0,
        "use_sliding_window": False, "sliding_window": None, "tie_word_embeddings": False,
    }
    for field, value in expected.items():
        require(source.get(field) == value, f"frozen config field mismatch: {field}")
    config = Qwen3MoeConfig(**{key: source[key] for key in expected if key not in {"architecture", "model_type", "torch_dtype"}})
    config._attn_implementation = "eager"
    # `config.json` records the upstream export version (4.51.0); M4.2 froze
    # its reference tooling independently in requirements.lock at 5.12.1.
    require(source["transformers_version"] == "4.51.0", "unexpected upstream config export version")
    require(transformers.__version__ == "5.12.1", "Transformers tooling differs from frozen M4.2 contract")
    return config


def resolve(root: Path, index: dict[str, Any], name: str) -> torch.Tensor:
    shard = index["weight_map"].get(name)
    require(isinstance(shard, str), f"tensor missing from index: {name}")
    with safe_open(root / shard, framework="pt", device="cpu") as reader:
        return reader.get_tensor(name)


def prove(source_root: Path, manifest_path: Path) -> dict[str, Any]:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    source = validate(source_root, manifest_path)
    manifest = read_json(manifest_path)
    config = config_from_manifest(manifest)
    index = read_json(source_root / manifest["safetensors"]["index_file"])
    dense_names = [
        "model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.post_attention_layernorm.weight", "model.layers.0.mlp.gate.weight",
    ]
    expert_names = [f"model.layers.0.mlp.experts.0.{projection}_proj.weight" for projection in ("gate", "up", "down")]
    resolved = {name: resolve(source_root, index, name) for name in [*dense_names, *expert_names]}
    require(tuple(resolved["model.embed_tokens.weight"].shape) == (151936, 2048), "embedding shape mismatch")
    require(tuple(resolved["model.layers.0.mlp.gate.weight"].shape) == (128, 2048), "router shape mismatch")
    require(all(tensor.dtype == torch.bfloat16 for tensor in resolved.values()), "source tensors are not BF16")
    with torch.device("meta"):
        modules = [Qwen3MoeRMSNorm(2048, eps=1.0e-6), Qwen3MoeAttention(config, 0), Qwen3MoeTopKRouter(config), Qwen3MoeMLP(config, intermediate_size=768)]
    require(len(modules) == 4, "component instantiation failed")
    return {**source, "offline": True, "transformers_version": transformers.__version__, "components": [type(module).__name__ for module in modules], "resolved_dense_tensors": dense_names, "resolved_expert_tensors": expert_names, "status": "passed"}


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("source_manifest", type=Path)
    args = parser.parse_args(arguments)
    try:
        print(json.dumps(prove(args.source_root.resolve(), args.source_manifest.resolve()), sort_keys=True))
    except (ProofError, OSError, ValueError) as error:
        print(f"offline selective-load proof error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

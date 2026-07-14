# Tiny Qwen3-MoE fixture provenance

## Target architecture

- Model family: Qwen3-MoE
- Full-size target model: `Qwen/Qwen3-30B-A3B`
- Pinned model revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Model source: <https://huggingface.co/Qwen/Qwen3-30B-A3B>
- Model metadata license: Apache-2.0
- Model metadata pipeline: text generation

The revision and license were read from the Hugging Face model API on
2026-07-14. No full-size model file is downloaded or copied into this fixture.

## Architecture implementation

- Implementation: Hugging Face Transformers `Qwen3MoeForCausalLM`
- Transformers version/tag: `v5.12.1`
- Configuration source:
  <https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/qwen3_moe/configuration_qwen3_moe.py>
- Model source:
  <https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py>
- Transformers license: Apache-2.0

## Fixture generation

- Initial generation date: 2026-07-14
- Generator: `python/reference/generate_fixture.py`
- Artifact format version: `tiny-qwen3-moe-v1`
- Python: 3.12.10
- PyTorch: 2.12.1
- Transformers: 5.12.1
- Safetensors: 0.8.0
- NumPy: 2.3.3
- Device: CPU
- Seed: 20260714
- Input token IDs: `[1, 5, 7, 2]`

The model is initialized locally from the tiny config using the fixed seed.
The saved weights are synthetic and are not derived from upstream Qwen weights.

## Artifact inventory

- `config.json`: complete tiny Transformers configuration.
- `inputs.json`: fixed token IDs and attention mask.
- `environment.json`: pinned generator environment and source revisions.
- `tolerances.json`: per-stage numerical comparison policy.
- `weights.safetensors`: deterministic synthetic model state dictionary.
- `checkpoints.safetensors`: deterministic oracle intermediates and logits.
- `tensor-manifest.json`: names, dtypes, shapes, and byte sizes for saved tensors.
- `sha256.json`: SHA-256 digest for every other fixture file.

Regeneration and verification commands are documented in `README.md`.

# Tiny Qwen3-MoE reference fixture

This directory owns the Python/Transformers numerical oracle used by M1. The
fixture is generated from a tiny random model; it does not download or contain
weights from Qwen3-30B-A3B.

## Environment

Use Python 3.12.10 and install the exact versions in `requirements.lock` in an
isolated environment. The generator rejects version drift before creating or
verifying artifacts.

```powershell
python -m venv .venv-reference
.\.venv-reference\Scripts\Activate.ps1
python -m pip install -r python\reference\requirements.lock
```

Package hashes are not yet part of this repository lock file; exact versions
and generated fixture hashes are still enforced. Do not run the generator with
unpinned package versions.

## Generate

```powershell
python python\reference\generate_fixture.py generate
```

Generation is CPU-only, uses deterministic PyTorch algorithms and one thread,
and writes to `python/reference/fixtures/tiny-qwen3-moe`.

## Verify

```powershell
python python\reference\generate_fixture.py verify
```

Verification first checks every committed file against `sha256.json`, then
regenerates the complete fixture in a temporary directory and compares hashes.
It does not use network access.

## Checkpoints

`checkpoints.safetensors` contains, for every decoder layer:

- input normalization output;
- attention output;
- post-attention normalization output;
- post-RoPE query and key tensors;
- router logits and routing weights;
- exact selected expert IDs;
- MoE output;
- full decoder-block output.

It also contains the input token IDs, final normalization output, hidden states,
and final logits. Floating-point comparisons use `tolerances.json`; shapes,
tensor names, input IDs, and expert IDs require exact equality.

See [PROVENANCE.md](PROVENANCE.md) for source revisions and licensing.

## Full-model tensor values

M4.2-01 uses only the Python standard library to compare selected values from
the pinned Qwen3-30B-A3B Safetensors shards with the stable F32 artifact. It
hashes each selected source shard before reading payload samples and compares
exact F32 bit patterns after deterministic BF16 decoding.

```powershell
python python\reference\validate_full_model_tensor_values.py `
  --source-root D:\tmp\colibri-m4.1-05 `
  --registry models\qwen3-30b-a3b\canonical-root-registry-v1.json `
  --source-manifest models\qwen3-30b-a3b\source-manifest-v1.json `
  --dense-plan models\qwen3-30b-a3b\dense-source-plan-v1.tsv `
  --expert-plan models\qwen3-30b-a3b\expert-source-plan-v1.tsv `
  --selection models\qwen3-30b-a3b\m4.2-01-tensor-selection-v1.json `
  --output models\qwen3-30b-a3b\m4.2-01-tensor-evidence-v1.json
```

The command does not load a full tensor or create model payloads. Repeated runs
require the existing evidence to be byte-identical.

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
- router logits and routing weights;
- exact selected expert IDs;
- MoE output;
- full decoder-block output.

It also contains the input token IDs, final normalization output, hidden states,
and final logits. Floating-point comparisons use `tolerances.json`; shapes,
tensor names, input IDs, and expert IDs require exact equality.

See [PROVENANCE.md](PROVENANCE.md) for source revisions and licensing.

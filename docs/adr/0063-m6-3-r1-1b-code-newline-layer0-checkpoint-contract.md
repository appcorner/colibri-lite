# ADR 0063: R1.1b code_newline Layer-0 checkpoint contract

R1.1b freezes a compact, deterministic Transformers-F32 oracle payload for
`code_newline` only. Serialization is stable little-endian contiguous bytes
with fixed names/order and no timestamp or path. Tolerances are pre-linked to
the existing M4.2 IDs: post-attention/expert input, router logits, routing
weights, expert down projection before weighting, and aggregated MoE output.
No tolerance was derived from these output values. This record is reference
evidence only and does not execute R1.1a or admit a candidate.

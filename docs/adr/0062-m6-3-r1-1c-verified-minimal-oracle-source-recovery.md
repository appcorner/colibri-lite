# ADR 0062: M6.3-R1.1c Verified Minimal Oracle Source Recovery

## Status

Accepted for M6.3-R1.1c. This ADR supersedes only the R1.1c acquisition method
in ADR 0061; ADR 0061 and historical M4 reports remain unchanged evidence.

## Decision

The verified, offline 17-file Safetensors subset at the pinned revision is the
canonical minimal oracle source. It contains `model.safetensors.index.json`
and precisely its 16 referenced weight shards, totalling 61,068,275,406 bytes.
It was promoted by same-volume atomic directory rename to the registry-bound
root and protected read-only.

R1.1c does not acquire a full 26-file Transformers snapshot. The nine absent
metadata/tokenizer files are neither represented as present nor synthesized.
TLS remediation and full-snapshot acquisition are therefore not prerequisites
for R1.1b's selective oracle use, but remain unresolved for any future work
that actually requires network access or a full snapshot.

The oracle is a verified selective Safetensors loader combined with M4.2's
frozen Transformers tooling and configuration contract. It must not claim that
`AutoModel.from_pretrained(..., local_files_only=True)` works from this root.
No candidate artifact or Rust output is an oracle input.

## Consequences

The registry records the stable root, exact source identity, closed 17-file
set, per-file hashes, total bytes, and root-set hash. Any missing, unexpected,
tampered, traversal, or index-reference mismatch rejects the source. R1.1c
does not create `code_newline` checkpoints and does not start R1.1a, R1.1b, or
R1.2.

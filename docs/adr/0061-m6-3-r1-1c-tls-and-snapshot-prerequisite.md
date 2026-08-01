# ADR 0061: M6.3-R1.1c TLS and Canonical Oracle Snapshot Prerequisite

## Status

Proposed before M6.3-R1.1c acquisition. This ADR is a prerequisite plan only;
it does not authorize a TLS-verification bypass, weight download, R1.1b
checkpoint freeze, R1.2, or M6.4.

## Context

The frozen Python oracle requires an offline-loadable upstream Transformers
snapshot of the exact provenance-pinned `Qwen/Qwen3-30B-A3B` revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. The existing source manifest is
historical provenance and must remain unchanged. Before any snapshot run is
created, the acquisition client must verify HTTPS normally and must obtain
fresh upstream metadata for the immutable revision.

The initial host diagnosis found Windows PowerShell HTTPS usable, while the
Python client rejected the presented chain because its certifi bundle did not
contain the required issuer. The observed issuer was the locally presented
Norton Web/Mail Shield root. That certificate must not be trusted merely
because it was observed: its source, subject, issuer, validity, and SHA-256
fingerprint must first be validated against an administrator-approved Windows
certificate-store entry or supplied enterprise CA certificate.

## Decision

M6.3-R1.1c may proceed only when all of the following are recorded:

1. `curl`, PowerShell, the Python Hugging Face metadata client, and the
   downloader dry-run all complete HTTPS certificate verification without a
   verification bypass.
2. If Python needs a CA supplement, a dedicated R1.1c process environment uses
   either the Windows trust store or a bundle constructed from certifi plus an
   already-trusted, fingerprint-verified issuer certificate. It must not alter
   global TLS policy or project dependencies.
3. Any addition to `CurrentUser\\Root` or `CurrentUser\\CA` requires explicit
   user approval before the write and post-write fingerprint verification.
4. Fresh official metadata resolves to the pinned full commit and exactly
   matches the historical expected source count and bytes before a temporary
   run directory is created.

Forbidden mechanisms are `verify=False`, disabled certificate verification,
`GIT_SSL_NO_VERIFY`, disabled revocation checks, HTTP URLs, mutable revisions,
and an unverified exported certificate.

## Consequences

Until a trust source is available and the fresh metadata comparison passes,
R1.1c remains pending and must create no run directory or model payload.
Successful TLS remediation is acquisition evidence, not completion of R1.1c.
The next implementation task after a completed R1.1c remains the separately
authorized `M6.3-R1.1b checkpoint-reference freeze`.

# ADR 0017: Pinned Offline Tokenizer Assets

- Status: Accepted
- Date: 2026-07-15
- Milestone: M4.1
- Task: M4.1-07
- Source revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Context

The first full-model correctness test needs deterministic text-to-token and
token-to-text evidence for Qwen3-30B-A3B. The upstream source contract pins four
tokenizer files at the same immutable model revision, but those assets were not
yet available to an offline run.

The tokenizer is not a trivial delimiter. It uses NFC normalization, a
Unicode-aware split regex, byte-level preprocessing and decoding, 151,387 BPE
merges, and 26 added tokens. Reimplementing that contract with the Rust standard
library would require a new Unicode-regex and tokenizer subsystem. Adding a
general tokenizer crate would be a significant dependency and is a mandatory
review stop for this task. The implementation plan also defers a
production-grade tokenizer/chat abstraction.

## Decision

Store the four canonical upstream tokenizer assets byte-for-byte under
`models/qwen3-30b-a3b/`:

- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`

The files total exactly 15,881,072 bytes. Their pinned sizes and SHA-256 hashes
are recorded in `tokenizer-artifact-manifest-v1.json`. No derived vocabulary or
lossy conversion is introduced.

Use `python/reference/verify_tokenizer.py` as the offline correctness adapter.
It loads only the committed directory with `local_files_only=True` while both
Hugging Face offline environment flags are set. It uses the already locked
`transformers==5.12.1` reference environment and its existing tokenizers 0.22.2
backend. No Cargo or Python dependency changes are required.

Rust inference remains token-ID based in this milestone. The adapter produces
and verifies exact reference IDs before the first full-model test; inference
does not need network access. A Rust text tokenizer remains deferred until a
separate dependency and public-contract review.

## Tokenizer contract

The pinned tokenizer is `Qwen2Tokenizer` backed by byte-level BPE:

- NFC normalization;
- the exact upstream Unicode-aware pretokenizer regex;
- no prefix space and no offset trimming;
- 151,643 base vocabulary entries with IDs 0 through 151,642;
- 151,387 merge rules;
- 26 added tokens with IDs 151,643 through 151,668;
- 151,669 total tokenizer entries;
- 151,936 model vocabulary entries, leaving 267 model IDs unassigned by the
  tokenizer assets.

Tokenizer special IDs are deliberately distinct from model configuration
metadata. The tokenizer has no BOS or UNK token, EOS is 151,645
(`<|im_end|>`), and PAD is 151,643 (`<|endoftext|>`). The model configuration
separately declares BOS 151,643 and EOS 151,645.

Fourteen added tokens are marked special: IDs 151,643 through 151,656. The
remaining twelve added tokens, IDs 151,657 through 151,668, are recognized
added tokens but have `special=false`. The versioned manifest records every
ID, string, and special flag.

## Limits remain separate

- Model maximum positions: 40,960.
- Tokenizer-declared `model_max_length`: 131,072.
- Runtime session capacity: caller-configured and validated per session; this
  artifact assigns no universal value.

The tokenizer limit does not override model positions, and neither value
silently sets runtime session capacity.

## Chat template

The exact upstream chat template remains in `tokenizer_config.json`. Its 4,168
UTF-8 bytes hash to
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
Metadata is preserved, but this task does not render the template or implement
chat roles, tool calling, or an OpenAI-compatible API.

## Evidence

The frozen reference covers English, Thai, source code and indentation,
whitespace/newlines, Unicode and emoji, special tokens, and empty input. All
seven cases match the pinned Hugging Face tokenizer token IDs exactly. Decoding
also matches exactly, and every selected NFC input round-trips byte-for-byte.

The offline verifier additionally checks all four file hashes, vocabulary and
merge counts, normalizer, pretokenizer regex, byte-level settings, all added
token mappings and special flags, special IDs, limits, and chat-template hash.

## Excluded work

- No Rust tokenizer public API or significant tokenizer dependency.
- No chat-template renderer, tool calling, chat framework, or HTTP API.
- No model weight download or modification.
- No quantization, mmap, SIMD, GPU work, optimization, or full-model inference.

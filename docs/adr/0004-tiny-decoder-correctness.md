# ADR 0004: Tiny Decoder Correctness Composition

- Status: Accepted
- Date: 2026-07-14
- Milestone: M1.3

## Context

After one sparse block matches the frozen oracle, M1.3 must prove the complete
two-layer tiny decoder: token embeddings, sequential block composition, final
normalization, and language-model logits. The model contract must retain enough
intermediate evidence to identify the first diverging layer without introducing
generation, tokenizer, storage, or cache concerns.

## Decision

`Qwen3MoeModel` owns:

- a validated token embedding table;
- exactly the configured number of validated sparse blocks;
- final RMS normalization weight;
- language-model head weight.

The forward contract accepts one non-empty token-ID sequence, performs checked
embedding lookup, runs blocks in layer order, applies final RMS normalization,
and projects to `[sequence, vocabulary]` logits.

`Qwen3MoeModelOutput` retains:

- the embedding output and raw output of every decoder block;
- each block's intermediate checkpoints and selected expert IDs;
- final normalized hidden states;
- final logits.

Transformers' last `hidden_states` checkpoint is post-final-normalization, while
the Rust output's last raw hidden state is the second block output. Tests compare
each value against the corresponding semantic checkpoint rather than relying on
array position alone.

## Invariants

- Embedding and LM-head weights match configured vocabulary/hidden dimensions.
- Block count equals configured layer count.
- Token input is non-empty and every token ID is within vocabulary range.
- Blocks execute sequentially without hidden-state mutation or reuse.
- Expert IDs, shapes, and names compare exactly.
- Floating checkpoints and logits use the frozen absolute/relative tolerance.
- Repeated runs with identical IDs and weights produce exactly identical output.
- No tokenizer, sampling, KV cache, file reader, dependency, or optimized kernel
  is introduced.

## Evidence

- Embedding lookup matches the frozen embedding checkpoint.
- Empty and out-of-range token-ID tests return structured errors.
- Both layers match input norm, attention, post-attention norm, router logits,
  routing weights, expert IDs, MoE output, and raw block output.
- Final norm matches both the dedicated hook and Transformers' final hidden
  state.
- Final logits match the frozen `[4, 64]` oracle tensor.
- Two complete forward runs compare exactly equal.

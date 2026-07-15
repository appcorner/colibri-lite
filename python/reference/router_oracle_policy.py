"""Margin-aware expert-ID oracle policy for BF16/F32 router comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


class RouterOracleMismatch(RuntimeError):
    """Router logits, deterministic ordering, or safely assertable IDs differ."""


@dataclass(frozen=True)
class RouterBoundary:
    selected_logits: tuple[float, ...]
    kth_selected_logit: float
    highest_unselected_logit: float
    selection_margin: float
    maximum_logit_error: float
    required_safe_margin: float
    expert_ids_assertable: bool


def deterministic_top_k(logits: Sequence[float], top_k: int) -> tuple[int, ...]:
    if not 0 < top_k < len(logits):
        raise RouterOracleMismatch("top_k must leave at least one unselected expert")
    return tuple(sorted(range(len(logits)), key=lambda expert: (-logits[expert], expert))[:top_k])


def assess_router_ids(
    transformers_logits: Sequence[float],
    rust_logits: Sequence[float],
    transformers_ids: Sequence[int],
    rust_ids: Sequence[int],
    top_k: int,
    documented_error_bound: float,
) -> RouterBoundary:
    if len(transformers_logits) != len(rust_logits) or len(transformers_ids) != top_k or len(rust_ids) != top_k:
        raise RouterOracleMismatch("router comparison shapes differ")
    if documented_error_bound < 0.0:
        raise RouterOracleMismatch("documented error bound must be non-negative")
    if len(set(transformers_ids)) != top_k or any(index < 0 or index >= len(transformers_logits) for index in transformers_ids):
        raise RouterOracleMismatch("Transformers expert IDs are invalid")

    expected_rust = deterministic_top_k(rust_logits, top_k)
    if tuple(rust_ids) != expected_rust:
        raise RouterOracleMismatch("Rust IDs violate deterministic higher-score/lower-ID ordering")

    selected_logits = tuple(transformers_logits[index] for index in transformers_ids)
    ranked_logits = sorted(transformers_logits, reverse=True)
    kth = ranked_logits[top_k - 1]
    highest_unselected = ranked_logits[top_k]
    margin = kth - highest_unselected
    maximum_error = max(abs(reference - actual) for reference, actual in zip(transformers_logits, rust_logits, strict=True))
    required_margin = max(documented_error_bound, 2.0 * maximum_error)
    assertable = margin > required_margin

    if assertable and tuple(transformers_ids) != tuple(rust_ids):
        raise RouterOracleMismatch("safe-margin Transformers and Rust expert IDs differ")

    return RouterBoundary(
        selected_logits=selected_logits,
        kth_selected_logit=kth,
        highest_unselected_logit=highest_unselected,
        selection_margin=margin,
        maximum_logit_error=maximum_error,
        required_safe_margin=required_margin,
        expert_ids_assertable=assertable,
    )

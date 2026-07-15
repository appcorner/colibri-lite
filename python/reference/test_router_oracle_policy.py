#!/usr/bin/env python3
"""Tests for margin-aware router expert-ID assertions."""

from __future__ import annotations

import unittest

from python.reference.router_oracle_policy import RouterOracleMismatch, assess_router_ids


class RouterOraclePolicyTests(unittest.TestCase):
    def test_positive_safe_margin_accepts_matching_ids(self) -> None:
        boundary = assess_router_ids([4.0, 3.0, 1.0, 0.0], [4.01, 2.99, 1.0, 0.0], [0, 1], [0, 1], 2, 0.05)
        self.assertTrue(boundary.expert_ids_assertable)
        self.assertAlmostEqual(boundary.selection_margin, 2.0)
        self.assertAlmostEqual(boundary.maximum_logit_error, 0.01)

    def test_positive_safe_margin_with_mismatching_ids_fails(self) -> None:
        with self.assertRaisesRegex(RouterOracleMismatch, "safe-margin"):
            assess_router_ids([4.0, 3.0, 1.0, 0.0], [4.0, 3.0, 1.0, 0.0], [0, 2], [0, 1], 2, 0.05)

    def test_tied_boundary_is_non_assertable_but_rust_policy_is_checked(self) -> None:
        boundary = assess_router_ids([4.0, 3.0, 3.0, 0.0], [4.0, 3.0, 3.0, 0.0], [0, 2], [0, 1], 2, 0.05)
        self.assertFalse(boundary.expert_ids_assertable)
        self.assertEqual(boundary.selection_margin, 0.0)
        self.assertEqual(boundary.kth_selected_logit, boundary.highest_unselected_logit)

    def test_ambiguous_boundary_still_rejects_wrong_rust_tie_order(self) -> None:
        with self.assertRaisesRegex(RouterOracleMismatch, "deterministic"):
            assess_router_ids([4.0, 3.0, 3.0, 0.0], [4.0, 3.0, 3.0, 0.0], [0, 2], [0, 2], 2, 0.05)


if __name__ == "__main__":
    unittest.main()

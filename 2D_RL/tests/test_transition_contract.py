"""Regression tests for the Bellman-to-RL transition contract."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


TEST_DIRECTORY = Path(__file__).resolve().parent
RL_DIRECTORY = TEST_DIRECTORY.parent
REPOSITORY_ROOT = RL_DIRECTORY.parent
for path in (REPOSITORY_ROOT, RL_DIRECTORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rl2d.transition_contract import ExactTransitionModel, validate_transition_contract


class TransitionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact_path = (
            RL_DIRECTORY / "results" / "bellman_reference" / "bellman_reference.npz"
        )
        if not cls.artifact_path.exists():
            raise unittest.SkipTest("Bellman reference artifact has not been generated")
        cls.archive = np.load(cls.artifact_path)
        cls.model = ExactTransitionModel(cls.archive)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "archive"):
            cls.archive.close()

    def test_action_mask_and_transition_record(self) -> None:
        states = np.argwhere(self.model.reference_policy >= 0)
        zi, hi = (int(value) for value in states[len(states) // 2])
        ai = int(self.model.reference_policy[zi, hi])
        self.assertTrue(self.model.action_mask((zi, hi))[ai])
        transition = self.model.transition((zi, hi), ai)
        self.assertTrue(transition.feasible)
        self.assertAlmostEqual(transition.reward, -transition.edge_cost)
        self.assertIsNotNone(transition.next_state)

    def test_contract_reconstructs_exact_solution(self) -> None:
        validation = validate_transition_contract(self.model)
        self.assertTrue(validation["passed"], validation["failed_checks"])
        self.assertEqual(validation["metrics"]["maximum_value_error"], 0.0)
        self.assertEqual(validation["metrics"]["greedy_policy_agreement"], 1.0)

    def test_normalized_grid_corners(self) -> None:
        lower = self.model.normalize_state(
            np.array([self.model.z_grid[0], self.model.h_grid[0]])
        )
        upper = self.model.normalize_state(
            np.array([self.model.z_grid[-1], self.model.h_grid[-1]])
        )
        np.testing.assert_array_equal(lower, np.zeros(2))
        np.testing.assert_array_equal(upper, np.ones(2))


if __name__ == "__main__":
    unittest.main()


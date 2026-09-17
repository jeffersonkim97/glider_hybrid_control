"""Fast contract tests for the exact Bellman reference layer."""
from __future__ import annotations

import json
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

from rl2d.reference_configuration import (
    REFERENCE_SENSOR_Z,
    REFERENCE_TRANSITION_MODEL,
    build_reference_configuration,
)


class ReferenceConfigurationTests(unittest.TestCase):
    def test_reference_imports_unmodified_production_defaults(self) -> None:
        reference = build_reference_configuration(REPOSITORY_ROOT)
        try:
            primary = reference.configuration_bundle["primary_result"]
            self.assertEqual(
                primary["attacker_solver_config"]["transition_model"],
                REFERENCE_TRANSITION_MODEL,
            )
            self.assertEqual(reference.sensor_z, REFERENCE_SENSOR_Z)
            self.assertEqual(
                len(primary["environment_config"]["terrain"]["hills"]), 1
            )
            self.assertEqual(len(reference.configuration_hash), 64)
        finally:
            logging_utilities = primary["logging_utilities"]
            logging_utilities["close_logger"](logging_utilities["logger"])

    def test_generated_artifact_contract_when_present(self) -> None:
        output = RL_DIRECTORY / "results" / "bellman_reference"
        summary_path = output / "bellman_reference_summary.json"
        artifact_path = output / "bellman_reference.npz"
        if not summary_path.exists() or not artifact_path.exists():
            self.skipTest("Reference artifact has not been generated yet")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertTrue(summary["validation"]["passed"])
        self.assertEqual(summary["transition_contract"]["state"], ["z", "h"])
        with np.load(artifact_path) as data:
            self.assertEqual(data["exact_value"].ndim, 2)
            self.assertEqual(data["transition_valid"].ndim, 3)
            self.assertEqual(
                data["transition_valid"].shape[:2], data["exact_value"].shape
            )
            self.assertEqual(
                data["transition_valid"].shape[2],
                data["action_forward_cells"].size,
            )


if __name__ == "__main__":
    unittest.main()

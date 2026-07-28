"""Regression tests for paper-result provenance metadata."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.result_provenance import build_result_provenance


class ResultProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.configuration = build_configuration_bundle(
            Path(self.temporary_directory.name)
        )
        self.addCleanup(
            close_phase_logger,
            self.configuration["primary_result"]["logging_utilities"]["logger"],
        )

    def test_required_reproducibility_fields_are_present(self) -> None:
        replay = {
            "feasible": True,
            "violation": None,
            "reached_goal": True,
            "goal_miss": 4.25,
        }
        provenance = build_result_provenance(
            self.configuration,
            script_identifier="p1b_4D/test_result_provenance.py",
            continuous_validation=replay,
        )
        self.assertEqual(provenance["schema_name"], "ResultProvenance")
        self.assertEqual(len(provenance["configuration_hash_sha256"]), 64)
        self.assertIn("source_commit", provenance["source_control"])
        self.assertIn("working_tree_dirty", provenance["source_control"])
        self.assertIn("working_tree_identifier", provenance["source_control"])
        self.assertEqual(
            provenance["transition_model"],
            self.configuration["primary_result"]["attacker_solver_config"][
                "transition_model"
            ],
        )
        self.assertTrue(provenance["continuous_validation"]["checked"])
        self.assertTrue(provenance["continuous_validation"]["feasible"])
        self.assertEqual(
            provenance["numerical_validation"]["segment_check_count"],
            self.configuration["primary_result"]["bellman_config"][
                "search_options"
            ]["segment_check_count"],
        )

    def test_configuration_hash_changes_with_resolution(self) -> None:
        original = build_result_provenance(
            self.configuration,
            script_identifier="hash-test",
        )
        refined = deepcopy(self.configuration)
        refined["primary_result"]["environment_config"]["grid"][
            "z_count"
        ] += 1
        changed = build_result_provenance(
            refined,
            script_identifier="hash-test",
        )
        self.assertNotEqual(
            original["configuration_hash_sha256"],
            changed["configuration_hash_sha256"],
        )
        self.assertFalse(original["continuous_validation"]["checked"])


if __name__ == "__main__":
    unittest.main()

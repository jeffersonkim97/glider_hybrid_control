"""Phase 1 configuration acceptance tests."""

from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from p1b_4D.configuration import (
    GLOBAL_RANDOM_SEED,
    build_configuration_bundle,
    validate_configuration,
)
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.project_paths import create_project_paths


class ConfigurationTests(unittest.TestCase):
    """Verify required dictionaries, keys, paths, and cross-config invariants."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.bundle = build_configuration_bundle(
            Path(self.temporary_directory.name)
        )
        self.addCleanup(
            close_phase_logger,
            self.bundle["primary_result"]["logging_utilities"]["logger"],
        )

    def test_bundle_uses_universal_result_envelope(self) -> None:
        self.assertEqual(
            set(self.bundle),
            {"primary_result", "validation", "metadata", "status"},
        )
        self.assertTrue(self.bundle["status"]["success"])
        self.assertTrue(self.bundle["validation"]["passed"])

    def test_all_required_configuration_dictionaries_exist(self) -> None:
        expected = {
            "environment_config",
            "vehicle_config",
            "sensor_config",
            "cost_config",
            "bellman_config",
            "attacker_solver_config",
            "nlp_config",
            "defender_config",
            "plot_config",
            "io_config",
            "validation_config",
        }
        self.assertTrue(expected.issubset(self.bundle["primary_result"]))
        self.assertEqual(
            self.bundle["validation"]["metrics"]["configuration_dictionary_count"],
            len(expected),
        )

    def test_project_directories_exist(self) -> None:
        paths = self.bundle["primary_result"]["project_paths"]
        for path in (
            paths.results_dir,
            paths.json_dir,
            paths.npz_dir,
            paths.figure_dir,
            paths.log_dir,
        ):
            self.assertTrue(path.is_dir(), path)

    def test_attacker_objective_is_shared(self) -> None:
        result = self.bundle["primary_result"]
        objective_id = result["cost_config"]["attacker"]["objective_id"]
        self.assertEqual(
            result["bellman_config"]["attacker_objective_id"], objective_id
        )
        self.assertEqual(result["nlp_config"]["attacker_objective_id"], objective_id)

    def test_global_seed_is_shared(self) -> None:
        result = self.bundle["primary_result"]
        self.assertEqual(result["global_random_seed"], GLOBAL_RANDOM_SEED)
        self.assertEqual(result["bellman_config"]["random_seed"], GLOBAL_RANDOM_SEED)
        self.assertEqual(
            result["nlp_config"]["multi_start"]["random_seed"],
            GLOBAL_RANDOM_SEED,
        )

    def test_defender_optimizer_is_hierarchical_brent(self) -> None:
        config = self.bundle["primary_result"]["defender_config"]
        self.assertEqual(config["optimizer"], "hierarchical_coarse_to_fine_brent")
        self.assertGreaterEqual(config["coarse_sample_count"], 3)
        self.assertGreater(config["xtol"], 0.0)
        bounds = config["continuous_search_bounds"]
        self.assertLess(bounds["z_sensor_min"], bounds["z_sensor_max"])

    def test_missing_dictionary_fails_validation(self) -> None:
        primary = self.bundle["primary_result"]
        configs = {
            key: deepcopy(value)
            for key, value in primary.items()
            if key.endswith("_config")
        }
        configs.pop("sensor_config")
        paths = create_project_paths(Path(self.temporary_directory.name))
        validation = validate_configuration(configs, paths)
        self.assertFalse(validation["passed"])
        self.assertIn(
            "sensor_config.exists",
            validation["status"]["failed_checks"],
        )


if __name__ == "__main__":
    unittest.main()

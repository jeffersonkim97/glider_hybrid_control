"""Stage-14.0 benchmark-contract and frozen-reference exit gate."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from stage14_benchmark_contract import (
    CANONICAL_DEFENDER_X_MAP,
    CANONICAL_RESOLUTION,
    FROZEN_CANONICAL_SOLUTION,
    SCHEMA_VERSION,
    benchmark_configuration,
    benchmark_result_schema,
    canonical_stage14_config,
    quantity_definitions,
    solver_source_manifest,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_0_benchmark_contract"


class Stage140ContractUnitTests(unittest.TestCase):
    def test_canonical_configuration_is_fixed_single_cube_25m_25m_5deg(self) -> None:
        config = benchmark_configuration()
        self.assertEqual(config["schema_version"], SCHEMA_VERSION)
        self.assertEqual(config["terrain_policy"]["category"], "centered_cube")
        self.assertFalse(config["terrain_policy"]["terrain_complexity_sweep_allowed"])
        self.assertEqual(
            tuple(config["canonical_resolution"].values()), CANONICAL_RESOLUTION,
        )
        self.assertEqual(
            tuple(config["defender_action_set"]["x_map"]), CANONICAL_DEFENDER_X_MAP,
        )
        self.assertFalse(config["algorithm_policy"]["reinforcement_learning_allowed"])
        self.assertFalse(config["algorithm_policy"]["approximate_planning_allowed"])
        self.assertFalse(config["algorithm_policy"]["alternative_solver_allowed"])
        stage11 = canonical_stage14_config()
        self.assertEqual(stage11.terrain_category, "centered_cube")
        self.assertEqual(stage11.discretization.heading_bin_count, 72)
        self.assertEqual(stage11.discretization.hazard_quadrature_resolution, 8)

    def test_quantity_vocabulary_is_explicit_and_versioned(self) -> None:
        definitions = quantity_definitions()
        for name in (
            "N_x", "N_y", "N_h", "N_psi", "N_S_cart", "N_S_admissible",
            "N_S_goal_reachable", "N_S_active", "N_E", "N_E_goal_reachable",
            "N_C_raw", "N_C_powered_feasible", "N_C_energy_feasible",
            "N_C_virtual_feasible", "N_C_goal_reachable", "N_C_feasible",
            "N_D", "B", "B_effective", "Q",
        ):
            self.assertIn(name, definitions)
            self.assertTrue(definitions[name]["definition"])
            self.assertTrue(definitions[name]["unit"])
        self.assertIn("actually processed", definitions["N_E"]["definition"])
        self.assertIn("not attempted", definitions["B_effective"]["definition"])
        schema = benchmark_result_schema()
        self.assertEqual(schema["schema_version"], SCHEMA_VERSION)
        self.assertIn("solution_identity", schema["required_top_level_fields"])

    def test_pre_stage14_values_are_literal_not_loaded_from_new_run(self) -> None:
        self.assertEqual(FROZEN_CANONICAL_SOLUTION["selected_defender_action_id"], 2)
        self.assertEqual(FROZEN_CANONICAL_SOLUTION["selected_attacker_candidate_id"], 26)
        self.assertAlmostEqual(
            FROZEN_CANONICAL_SOLUTION["attacker_objective"],
            0.944086144839464,
            places=15,
        )
        self.assertAlmostEqual(
            FROZEN_CANONICAL_SOLUTION["defender_objective_pod"],
            0.7814861193516704,
            places=15,
        )


class Stage140ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = json.loads(
            (OUTPUT / "frozen_reference.json").read_text(encoding="utf-8")
        )
        cls.environment = json.loads(
            (OUTPUT / "environment_manifest.json").read_text(encoding="utf-8")
        )
        cls.summary = json.loads(
            (OUTPUT / "stage14_0_summary.json").read_text(encoding="utf-8")
        )

    def test_all_required_artifacts_exist(self) -> None:
        for filename in (
            "benchmark_configuration.json", "benchmark_result_schema.json",
            "environment_manifest.json", "frozen_reference.json",
            "baseline_identity_and_state_counts.png", "stage14_0_summary.json",
        ):
            path = OUTPUT / filename
            self.assertTrue(path.is_file(), filename)
            self.assertGreater(path.stat().st_size, 100, filename)

    def test_environment_manifest_has_revision_packages_and_hardware(self) -> None:
        manifest = self.environment
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        self.assertTrue(manifest["environment_manifest_id"])
        self.assertIn("git_commit", manifest["software_revision"])
        self.assertIn("solver_source", manifest["software_revision"])
        self.assertIn("numpy", {name.lower() for name in manifest["packages"]})
        self.assertTrue(manifest["python"]["version"])
        self.assertTrue(manifest["operating_system"]["platform"])
        self.assertGreater(manifest["cpu"]["logical_cores"], 0)
        self.assertGreater(manifest["memory"]["total_bytes"], 0)

    def test_solver_source_fingerprint_is_still_the_frozen_one(self) -> None:
        current = solver_source_manifest(ROOT)["aggregate_sha256"]
        self.assertEqual(current, self.reference["solver_source_fingerprint"])
        self.assertEqual(
            current,
            self.environment["software_revision"]["solver_source"]["aggregate_sha256"],
        )

    def test_state_edge_candidate_and_game_counts_are_unambiguous(self) -> None:
        values = self.reference["state_and_game_size"]
        self.assertEqual(values["N_S_cart"], values["N_x"] * values["N_y"] * values["N_h"] * values["N_psi"])
        self.assertEqual(values["N_S_cart"], 3_243_240)
        self.assertEqual(values["N_S_goal_reachable"], 2_316_566)
        self.assertEqual(values["N_S_active"], 1_106_941)
        self.assertEqual(values["N_E"], 29_035_732)
        self.assertLessEqual(values["N_S_active"], values["N_S_goal_reachable"])
        self.assertLessEqual(values["N_S_goal_reachable"], values["N_S_admissible"])
        self.assertLessEqual(values["N_S_admissible"], values["N_S_cart"])
        self.assertEqual(values["N_C_raw"], 96)
        self.assertEqual(values["N_D"], 3)
        self.assertEqual(values["B"], 72)
        self.assertEqual(values["Q"], 8)

    def test_complete_solution_identity_and_sse_ties_are_frozen(self) -> None:
        identity = self.reference["solution_identity"]
        self.assertTrue(identity["feasible"])
        self.assertEqual(identity["selected_defender_action_id"], 2)
        self.assertEqual(identity["selected_sensor_position_map"], [7.5, 0.0, 0.0])
        self.assertEqual(identity["selected_attacker_candidate_id"], 26)
        self.assertIn(26, identity["attacker_objective_cooptimal_candidate_ids"])
        self.assertEqual(
            identity["sse_selected_follower_candidate_id"],
            identity["selected_attacker_candidate_id"],
        )
        self.assertAlmostEqual(identity["attacker_objective"], 0.944086144839464, places=12)
        self.assertAlmostEqual(identity["defender_objective_pod"], 0.7814861193516704, places=12)
        self.assertAlmostEqual(identity["mission_time_s"], 81.25366155073556, places=9)
        self.assertAlmostEqual(identity["cumulative_hazard"], 1.520905739469603, places=10)
        self.assertTrue(identity["trajectory_identity"]["sha256"])

    def test_replay_notebook_and_behavior_regression_gates_pass(self) -> None:
        self.assertTrue(self.reference["independent_replay"]["passed"])
        self.assertLessEqual(self.reference["independent_replay"]["time_error_s"], 1.0e-9)
        self.assertLessEqual(self.reference["independent_replay"]["hazard_error"], 1.0e-10)
        self.assertTrue(self.reference["regression"]["frozen_pre_stage14"]["passed"])
        self.assertTrue(self.reference["regression"]["validated_notebook"]["passed"])
        self.assertTrue(self.reference["regression"]["solver_behavior_unchanged"])
        self.assertTrue(self.reference["gate_passed"])
        self.assertTrue(self.summary["gate_passed"])

    def test_stage14_0_produces_static_table_but_no_scaling_curve(self) -> None:
        png = OUTPUT / "baseline_identity_and_state_counts.png"
        self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertFalse(self.summary["runtime_scaling_curve_produced"])
        self.assertFalse(self.summary["profiling_added"])
        self.assertFalse(self.summary["sweep_runner_added"])


if __name__ == "__main__":
    unittest.main()

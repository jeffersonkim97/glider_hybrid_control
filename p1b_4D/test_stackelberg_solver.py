"""Phase 9 algorithm-independent continuous Defender and Stackelberg tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.stackelberg_io import export_stackelberg_solution_bundle, import_stackelberg_solution_bundle
from p1b_4D.stackelberg_solver import (
    build_defender_optimizer_interface,
    direct_global_optimizer,
    evaluate_defender_position,
    hierarchical_coarse_to_fine_optimizer,
    solve_attacker_best_response,
    solve_stackelberg_game,
)


class StackelbergSolverTests(unittest.TestCase):
    """Verify continuous geometry, full nesting, injected optimization, and IO."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = TemporaryDirectory()
        cls.configuration = build_configuration_bundle(Path(cls.temporary_directory.name))
        cls.z_sensor = 3500.125
        cls.evaluation = evaluate_defender_position(cls.z_sensor, cls.configuration, "full-nested-test")

    @classmethod
    def tearDownClass(cls) -> None:
        close_phase_logger(cls.configuration["primary_result"]["logging_utilities"]["logger"])
        cls.temporary_directory.cleanup()

    def test_optimizer_interface_does_not_choose_algorithm(self) -> None:
        interface = build_defender_optimizer_interface(self.configuration)
        self.assertTrue(interface["status"]["success"])
        self.assertEqual(
            interface["primary_result"]["configured_optimizer"],
            "hierarchical_coarse_to_fine_brent",
        )
        self.assertEqual(interface["primary_result"]["objective_direction"], "maximize")

    def test_continuous_sensor_position_runs_complete_nested_pipeline(self) -> None:
        primary = self.evaluation["primary_result"]
        self.assertTrue(self.evaluation["status"]["success"])
        self.assertEqual(primary["z_sensor"], self.z_sensor)
        self.assertNotEqual(self.z_sensor, round(self.z_sensor))
        self.assertEqual(
            primary["nested_pipeline_execution"],
            ("geometry", "detection", "stage_cost_4d", "bellman", "bellman_optimal_response"),
        )
        self.assertTrue(self.evaluation["metadata"]["fresh_nested_attacker_solve"])

    def test_sensor_height_and_defender_objective_are_consistent(self) -> None:
        primary = self.evaluation["primary_result"]
        self.assertEqual(primary["sensor_position"][0], self.z_sensor)
        self.assertAlmostEqual(primary["defender_objective"], primary["objective_breakdown"]["total"], places=14)
        self.assertTrue(self.evaluation["validation"]["checks"]["geometry_consistency"])
        self.assertTrue(self.evaluation["validation"]["checks"]["mission_pod_consistency"])
        best = primary["best_found_attacker_response"]
        mission_hazard = best["powered_hazard"] + best["glide_hazard"]
        expected_defender_pod = mission_hazard / (mission_hazard + 1.0)
        self.assertAlmostEqual(
            primary["objective_breakdown"]["defender_pod_normalized"],
            expected_defender_pod,
            places=12,
        )
        self.assertTrue(
            self.evaluation["validation"]["checks"][
                "defender_pod_normalization_consistency"
            ]
        )

    def test_final_visualization_payload_uses_same_defender_evaluation(self) -> None:
        primary = self.evaluation["primary_result"]
        payload = primary["visualization_payload"]
        self.assertEqual(payload["evaluation_id"], primary["evaluation_id"])
        np.testing.assert_allclose(
            payload["sensor_position"], primary["sensor_position"],
            rtol=0.0, atol=0.0,
        )
        self.assertEqual(payload["cost_to_go"].shape, payload["los_mask"].shape)

    def test_fixed_and_outer_paths_use_identical_authoritative_attacker_solve(self) -> None:
        fixed = solve_attacker_best_response(
            self.z_sensor,
            self.configuration,
            "fixed-figure4-equivalence",
        )
        outer = self.evaluation["primary_result"][
            "attacker_best_response_bundle"
        ]
        self.assertTrue(
            fixed["metadata"][
                "same_computation_for_fixed_and_outer_evaluations"
            ]
        )
        fixed_primary = fixed["primary_result"]
        outer_primary = outer["primary_result"]
        fixed_candidates = fixed_primary["bellman_candidate_bundle"][
            "primary_result"
        ]["candidates"]
        outer_candidates = outer_primary["bellman_candidate_bundle"][
            "primary_result"
        ]["candidates"]
        self.assertEqual(len(fixed_candidates), len(outer_candidates))
        for fixed_candidate, outer_candidate in zip(
            fixed_candidates, outer_candidates, strict=True
        ):
            np.testing.assert_allclose(
                fixed_candidate["trajectory"],
                outer_candidate["trajectory"],
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(
                fixed_candidate["mission_cost"],
                outer_candidate["mission_cost"],
            )
        fixed_best = fixed_primary["best_found_attacker_response"]
        outer_best = outer_primary["best_found_attacker_response"]
        np.testing.assert_allclose(
            fixed_best["trajectory"], outer_best["trajectory"],
            rtol=0.0, atol=0.0,
        )
        self.assertEqual(
            fixed_best["mission_objective"],
            outer_best["mission_objective"],
        )

    def test_defender_positions_evaluate_without_nlp_related_failure(self) -> None:
        for z_sensor in (3000.0, 3225.0):
            with self.subTest(z_sensor=z_sensor):
                evaluation = evaluate_defender_position(
                    z_sensor, self.configuration, f"regression-{z_sensor:.1f}"
                )
                self.assertTrue(evaluation["status"]["success"])
                best = evaluation["primary_result"]["best_found_attacker_response"]
                self.assertEqual(best["metadata"].get("warm_start_only"), False)
                bundle = evaluation["primary_result"]["attacker_best_response_bundle"]
                self.assertEqual(
                    bundle["metadata"]["solution_method"], "bellman_dynamic_programming"
                )

    def test_outer_optimizer_keeps_a_superior_boundary_evaluation(self) -> None:
        counter = 0

        def increasing_evaluation(z_sensor):
            nonlocal counter
            counter += 1
            return {
                "primary_result": {
                    "evaluation_id": f"boundary-test-{counter}",
                    "z_sensor": z_sensor,
                    "h_sensor": 0.0,
                    "defender_objective": z_sensor,
                    "best_found_attacker_response": {
                        "solution_id": "fake-nlp",
                        "switching_point": np.array([0.0, 0.0]),
                        "mission_objective": 0.0,
                        "mission_pod": 0.0,
                        "mission_time": 0.0,
                    },
                    "coverage": {
                        "coverage_area": 0.0,
                        "normalized_coverage_area": 0.0,
                    },
                    "objective_breakdown": {
                        "defender_pod_normalized": 0.0,
                    },
                }
            }

        result = hierarchical_coarse_to_fine_optimizer(
            increasing_evaluation,
            (0.0, 1.0),
            {
                "coarse_sample_count": 5,
                "basin_prominence_threshold": 0.0,
                "xtol": 1.0e-3,
                "maximum_iterations": 50,
            },
        )
        self.assertEqual(result["z_sensor"], 1.0)
        self.assertEqual(
            result["metadata"]["selected_candidate_source"],
            "coarse_sweep",
        )

    def test_direct_global_optimizer_finds_known_maximum(self) -> None:
        def fake_evaluation(z_sensor):
            objective = 1.0 - ((z_sensor - 1900.0) / 300.0) ** 2
            return {
                "primary_result": {
                    "evaluation_id": f"direct-test-{z_sensor:.4f}",
                    "z_sensor": z_sensor,
                    "h_sensor": 0.0,
                    "defender_objective": objective,
                    "best_found_attacker_response": {
                        "solution_id": "fake-response",
                        "switching_point": np.array([0.0, 0.0]),
                        "mission_objective": 0.0,
                        "mission_pod": 0.0,
                        "mission_time": 0.0,
                    },
                    "coverage": {"coverage_area": 0.0, "normalized_coverage_area": 0.0},
                    "objective_breakdown": {"defender_pod_normalized": 0.0},
                }
            }

        result = direct_global_optimizer(
            fake_evaluation,
            (1500.0, 2400.0),
            {"direct_maxfun": 40, "direct_maxiter": 100, "direct_len_tol": 1.0e-4},
        )
        self.assertAlmostEqual(result["z_sensor"], 1900.0, delta=5.0)
        self.assertTrue(result["converged"])
        self.assertEqual(result["metadata"]["algorithm"], "scipy_direct_global")
        self.assertTrue(result["metadata"]["certified_global"])
        self.assertFalse(result["metadata"]["locally_biased"])
        self.assertGreater(len(result["evaluation_history"]), 0)

    def test_direct_global_optimizer_is_a_valid_injected_optimizer(self) -> None:
        with patch(
            "p1b_4D.stackelberg_solver.evaluate_defender_position",
            side_effect=lambda z_sensor, configuration_bundle, evaluation_id="evaluation": dict(self.evaluation),
        ):
            solution = solve_stackelberg_game(
                self.configuration,
                lambda evaluate, bounds, options: direct_global_optimizer(
                    evaluate, bounds, {**options, "direct_maxfun": 10, "direct_maxiter": 30}
                ),
            )
        self.assertTrue(solution["status"]["success"])
        self.assertEqual(solution["metadata"]["outer_optimizer_algorithm"], "scipy_direct_global")

    def test_injected_optimizer_gets_fresh_solve_for_every_call_and_final(self) -> None:
        call_count = 0

        def fake_evaluation(z_sensor, configuration_bundle, evaluation_id="evaluation"):
            nonlocal call_count
            call_count += 1
            result = dict(self.evaluation)
            primary = dict(self.evaluation["primary_result"])
            primary["evaluation_id"] = evaluation_id
            payload = dict(primary["visualization_payload"])
            payload["evaluation_id"] = evaluation_id
            primary["visualization_payload"] = payload
            result["primary_result"] = primary
            return result

        def test_only_optimizer(evaluate, bounds, options):
            first = evaluate(self.z_sensor)
            second = evaluate(self.z_sensor)
            self.assertIsNot(first, second)
            return {"z_sensor": self.z_sensor, "converged": True, "metadata": {"algorithm": "test_fixture_only"}}

        with patch("p1b_4D.stackelberg_solver.evaluate_defender_position", side_effect=fake_evaluation):
            solution = solve_stackelberg_game(self.configuration, test_only_optimizer)
        self.assertTrue(solution["status"]["success"])
        self.assertEqual(call_count, 3)
        self.assertEqual(solution["validation"]["metrics"]["outer_evaluation_count"], 3)
        self.assertTrue(solution["metadata"]["fresh_nested_solve_per_evaluation"])
        self.__class__.stackelberg_solution = solution

    def test_json_npz_round_trip(self) -> None:
        if not hasattr(self.__class__, "stackelberg_solution"):
            self.test_injected_optimizer_gets_fresh_solve_for_every_call_and_final()
        exported = export_stackelberg_solution_bundle(self.stackelberg_solution, self.configuration)
        imported = import_stackelberg_solution_bundle(exported["primary_result"]["json_path"])
        self.assertTrue(imported["status"]["success"])
        arrays = imported["primary_result"]["arrays"]
        np.testing.assert_array_equal(
            arrays["optimal_trajectory"],
            self.evaluation["primary_result"]["best_found_attacker_response"]["trajectory"],
        )
        self.assertEqual(arrays["outer_z_sensor"].size, 3)


if __name__ == "__main__":
    unittest.main()

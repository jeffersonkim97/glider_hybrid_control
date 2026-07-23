"""Phase 8 continuous multi-start Attacker CasADi NLP tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from p1b_4D.attacker_nlp import _terrain_interpolant, solve_attacker_nlp_multistart
from p1b_4D.attacker_nlp_io import export_attacker_nlp_bundle, import_attacker_nlp_bundle
from p1b_4D.bellman import generate_bellman_candidates
from p1b_4D.candidate_filtering import filter_bellman_candidates
from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.detection import build_symbolic_detection_bundle
from p1b_4D.geometry import build_geometry_bundle, terrain_height
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.projection import construct_projected_cost_map
from p1b_4D.stage_cost import construct_stage_cost_4d


class AttackerNLPTests(unittest.TestCase):
    """Verify independent refinement, continuous constraints, selection, and IO."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = TemporaryDirectory()
        cls.configuration = build_configuration_bundle(Path(cls.temporary_directory.name))

        # cls.configuration["primary_result"]["nlp_config"]["hazard_homotopy_scales"] = (1.0,)
        # cls.configuration["primary_result"]["nlp_config"]["ipopt_options"]["ipopt.max_iter"] = 2000

        cls.configuration["primary_result"]["nlp_config"]["ipopt_options"]["ipopt.print_level"] = 0
        cls.geometry = build_geometry_bundle(cls.configuration)
        cls.detection = build_symbolic_detection_bundle(cls.configuration, cls.geometry)
        cls.stage = construct_stage_cost_4d(cls.configuration, cls.geometry, cls.detection)
        cls.projection = construct_projected_cost_map(cls.configuration, cls.geometry, cls.detection, cls.stage)
        cls.bellman = generate_bellman_candidates(cls.configuration, cls.geometry, cls.detection, cls.stage, cls.projection)
        cls.filtered = filter_bellman_candidates(cls.bellman, cls.configuration, cls.configuration["validation"])
        cls.nlp = solve_attacker_nlp_multistart(cls.configuration, cls.geometry, cls.detection, cls.stage, cls.bellman, cls.filtered)

        # print("\n=== NLP ATTEMPTS ===")

        # for attempt in cls.nlp["primary_result"]["solver_attempts"]:
        #     print(
        #         attempt["source_candidate_id"],
        #         attempt["solver_status"],
        #         attempt["diagnostic"],
        #     )

        #     solution = attempt.get("solution")

        #     if solution is not None:
        #         diagnostics = solution["warm_start_diagnostics"]

        #         print(
        #             "  warm_start_diagnostics =",
        #             diagnostics,
        #         )
        #         print(
        #             "  bellman_support_feasible =",
        #             diagnostics.get("bellman_support_feasible"),
        #         )
        #         print(
        #             "  minimum_bellman_support_margin =",
        #             diagnostics.get("minimum_bellman_support_margin"),
        #         )
        #         print(
        #             "  initial_mission_objective =",
        #             diagnostics.get("initial_mission_objective"),
        #         )
        #         print(
        #             "  refined_mission_objective =",
        #             solution["mission_objective"],
        #         )
        #         print(
        #             "  warm_start_not_worsened =",
        #             solution["validation"]["checks"]["warm_start_not_worsened"],
        #         )

        #         print(
        #             "  continuation_history =",
        #             solution["continuation_history"],
        #         )


    @classmethod
    def tearDownClass(cls) -> None:
        close_phase_logger(cls.configuration["primary_result"]["logging_utilities"]["logger"])
        cls.temporary_directory.cleanup()

    def test_every_top_k_start_is_solved_independently(self) -> None:
        result = self.nlp["primary_result"]
        self.assertEqual(result["attempt_count"], self.filtered["primary_result"]["selected_candidate_count"])
        self.assertEqual(len({item["source_candidate_id"] for item in result["solver_attempts"]}), result["attempt_count"])
        self.assertEqual(result["feasible_solution_count"], 3)

    def test_best_found_response_is_minimum_not_global_claim(self) -> None:
        result = self.nlp["primary_result"]
        best = result["best_found_attacker_response"]
        self.assertEqual(best["mission_objective"], min(item["mission_objective"] for item in result["feasible_solutions"]))
        self.assertEqual(self.nlp["metadata"]["response_name"], "Best-found Attacker Response")
        self.assertFalse(self.nlp["metadata"]["global_optimum_claim"])

    def test_refined_solution_constraints_and_dimensions(self) -> None:
        for solution in self.nlp["primary_result"]["feasible_solutions"]:
            self.assertTrue(solution["validation"]["passed"])
            self.assertEqual(solution["trajectory"].shape, (80, 2))
            self.assertEqual(solution["velocity_profile"].shape, (79,))
            self.assertEqual(solution["gamma_profile"].shape, (79,))
            self.assertEqual(solution["interval_time_profile"].shape, (79,))
            self.assertAlmostEqual(
                np.sum(solution["interval_time_profile"]),
                solution["glide_time"],
                places=9,
            )
            self.assertLessEqual(solution["constraint_residuals"]["maximum_dynamic_residual"], 1.0e-6)
            self.assertTrue(
                solution["warm_start_diagnostics"][
                    "kinematically_consistent"
                ]
            )
            self.assertLessEqual(
                solution["mission_objective"],
                solution["warm_start_diagnostics"][
                    "initial_mission_objective"
                ],
            )
            self.assertEqual(
                solution["continuation_history"][-1]["hazard_scale"],
                1.0,
            )

    def test_detection_avoidance_remains_active_in_default_case(self) -> None:
        best = self.nlp["primary_result"]["best_found_attacker_response"]
        mission_hazard = best["powered_hazard"] + best["glide_hazard"]
        trajectory = best["trajectory"]
        altitude_at_sensor = np.interp(
            2000.0, trajectory[:, 0], trajectory[:, 1]
        )
        self.assertLess(mission_hazard, 10.0)
        costs = self.configuration["primary_result"]["cost_config"]["attacker"]
        expected_objective = (
            costs["w_pod"]
            * mission_hazard
            / costs["normalization"]["pod"]["hazard_reference"]
            + costs["w_time"]
            * best["mission_time"]
            / costs["normalization"]["time"]["reference_seconds"]
        )
        self.assertAlmostEqual(best["mission_objective"], expected_objective)
        self.assertGreater(altitude_at_sensor, 150.0)
        self.assertGreater(np.max(best["gamma_profile"]), -0.05)
        self.assertLess(np.min(best["gamma_profile"]), -0.3)
        self.assertTrue(best["continuation_history"][-1][
            "selected_by_exact_objective"
        ])

    def test_casadi_terrain_is_authoritative_natural_cubic(self) -> None:
        geometry = self.geometry["primary_result"]
        function = _terrain_interpolant(geometry)
        z = np.linspace(0.0, 2750.0, 257)
        symbolic = np.asarray(function(z), dtype=float).reshape(-1)
        numeric = terrain_height(geometry["terrain_model"], z)
        np.testing.assert_allclose(symbolic, numeric, rtol=0.0, atol=1.0e-11)

    def test_json_npz_round_trip(self) -> None:
        exported = export_attacker_nlp_bundle(self.nlp, self.configuration)
        imported = import_attacker_nlp_bundle(exported["primary_result"]["json_path"])
        self.assertTrue(imported["status"]["success"])
        arrays = imported["primary_result"]["arrays"]
        best = self.nlp["primary_result"]["best_found_attacker_response"]
        np.testing.assert_array_equal(arrays["best_trajectory"], best["trajectory"])
        np.testing.assert_array_equal(arrays["best_switching_point"], best["switching_point"])
        np.testing.assert_array_equal(
            arrays["best_interval_time_profile"],
            best["interval_time_profile"],
        )


if __name__ == "__main__":
    unittest.main()

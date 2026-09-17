"""Fast configuration and architecture tests for Stage 14.2C."""

from __future__ import annotations

import unittest
from pathlib import Path

from stage14_2c_contract import configuration_audit, stage14_2c_cases
from stage14_2c_worker import _build_case


ROOT = Path(__file__).resolve().parent


class Stage142CContractTests(unittest.TestCase):
    def test_matrix_has_ten_local_and_ten_matching_oracle_cases(self) -> None:
        cases = stage14_2c_cases()
        local = [case for case in cases if case.worker_mode == "local_sse"]
        oracle = [case for case in cases if case.worker_mode == "global_oracle"]
        self.assertEqual(len(local), 10)
        self.assertEqual(len(oracle), 10)
        self.assertEqual(
            {case.name.removeprefix("local_sse_") for case in local},
            {case.name.removeprefix("global_oracle_") for case in oracle},
        )

    def test_contract_fixes_terrain_exactness_single_start_and_repetitions(self) -> None:
        audit = configuration_audit(stage14_2c_cases())
        self.assertTrue(audit["passed"])
        self.assertTrue(all(audit["checks"].values()))
        for row in audit["cases"]:
            self.assertEqual(row["parameters"]["terrain_category"], "centered_cube")
            self.assertEqual(row["parameters"]["r_neighbor"], 1)
            self.assertEqual(row["repetitions"], 3)

    def test_each_physical_sweep_changes_only_its_declared_driver(self) -> None:
        local = {
            case.name.removeprefix("local_sse_"): case
            for case in stage14_2c_cases(include_global_oracle=False)
        }
        baseline = local["canonical_anchor"].parameter_dict
        ignored = {"baseline_anchor"}
        expectations = {
            "spatial_100m": {"spatial_resolution_m"},
            "spatial_50m": {"spatial_resolution_m"},
            "heading_15deg": {"heading_spacing_deg"},
            "heading_10deg": {"heading_spacing_deg"},
            "switch_contour_6": {"switching_contour_sample_count"},
            "switch_contour_9": {"switching_contour_sample_count"},
        }
        for name, expected in expectations.items():
            current = local[name].parameter_dict
            differing = {
                key for key in baseline
                if baseline[key] != current[key] and key not in ignored
            }
            self.assertEqual(differing, expected, name)

    def test_defender_sweep_keeps_fixed_domain_and_neighborhood(self) -> None:
        cases = [
            case for case in stage14_2c_cases(include_global_oracle=False)
            if "defender_count" in case.name
        ]
        self.assertEqual([case.parameter_dict["defender_action_count"] for case in cases], [2, 3, 5])
        for case in cases:
            parameters = case.parameter_dict
            self.assertEqual(parameters["defender_grid_policy"], "uniform_fixed_domain")
            self.assertEqual(parameters["defender_domain_x_min"], 5.0)
            self.assertEqual(parameters["defender_domain_x_max"], 10.0)
            self.assertEqual(parameters["r_neighbor"], 1)

    def test_worker_build_maps_physical_controls_to_expected_grid(self) -> None:
        case = next(
            case for case in stage14_2c_cases(include_global_oracle=False)
            if case.name.endswith("spatial_100m")
        )
        configuration = case.as_configuration() | {
            "case_id": case.case_id,
            "repetition_id": case.repetition_id(0),
        }
        stage_config, candidates, initial_id = _build_case(configuration)
        self.assertEqual(stage_config.terrain_category, "centered_cube")
        self.assertEqual(stage_config.discretization.horizontal_spacing_map, 1.0)
        self.assertEqual(stage_config.discretization.altitude_spacing_map, 1.0)
        self.assertEqual(stage_config.discretization.heading_bin_count, 72)
        self.assertEqual(len(candidates), 6)
        self.assertEqual(
            tuple(candidate.action.sensor_position_map[0] for candidate in candidates),
            (5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
        )
        self.assertEqual(initial_id, 0)

    def test_global_solver_sources_are_not_modified_by_local_scaling_layer(self) -> None:
        local_source = (ROOT / "stage14_2c_worker.py").read_text(encoding="utf-8")
        global_source = (ROOT / "stackelberg_solver.py").read_text(encoding="utf-8")
        self.assertIn("run_exact_local_stackelberg", local_source)
        self.assertIn("run_finite_stackelberg", local_source)
        self.assertNotIn("stage14_2c", global_source.lower())


if __name__ == "__main__":
    unittest.main()

"""Stage-9 exhaustive best-response integration/regression exit gate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from attacker_best_response import attacker_best_response
from game_types import AttackerInitialCondition, AttackerResponse, DefenderAction
from scenario import Point3D
from terrain_catalog import build_terrain


ROOT = Path(__file__).resolve().parent
FIGURE_DIRECTORY = ROOT / "figure" / "stage_9_attacker_best_response"
ORIGINAL_NOTEBOOK_SHA256 = (
    "3b5938317df97814facf1c25066777d4efcdeca28fc8116fe99cd8e63b3a09a2"
)


class Stage9ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            (FIGURE_DIRECTORY / "stage9_attacker_best_response_summary.json").read_text(
                encoding="utf-8",
            )
        )
        with (FIGURE_DIRECTORY / "candidate_results.csv").open(
            newline="", encoding="utf-8",
        ) as handle:
            cls.rows = tuple(csv.DictReader(handle))

    def test_every_candidate_has_result_or_explicit_failure(self) -> None:
        self.assertEqual(len(self.rows), 96)
        self.assertEqual(
            sorted(int(row["candidate_id"]) for row in self.rows),
            list(range(96)),
        )
        for row in self.rows:
            if row["feasible"] == "True":
                self.assertEqual(row["infeasibility_reason"], "")
                self.assertNotEqual(row["objective"], "")
            else:
                self.assertNotEqual(row["infeasibility_reason"], "")
                self.assertEqual(row["objective"], "")

    def test_selected_candidate_is_exact_exhaustive_minimum(self) -> None:
        feasible = tuple(row for row in self.rows if row["feasible"] == "True")
        minimum = min(float(row["objective"]) for row in feasible)
        selected = self.summary["selected"]
        self.assertEqual(selected["candidate_id"], 25)
        self.assertEqual(selected["cooptimal_candidate_ids"], [25, 34])
        self.assertAlmostEqual(selected["objective"], minimum, places=12)
        self.assertEqual(
            self.summary["exactness"]["selected_minus_minimum_absolute"], 0.0,
        )

    def test_resolution_sensitivity_records_both_runs(self) -> None:
        rows = self.summary["resolution_sensitivity"]
        self.assertEqual([row["candidate_count"] for row in rows], [96, 128])
        self.assertEqual([row["contour_sample_count"] for row in rows], [12, 16])
        self.assertTrue(all(row["total_runtime_s"] > 0.0 for row in rows))
        self.assertNotEqual(rows[0]["best_position_map"] if "best_position_map" in rows[0] else (
            rows[0]["best_x_map"], rows[0]["best_y_map"], rows[0]["best_z_map"]
        ), (
            rows[1]["best_x_map"], rows[1]["best_y_map"], rows[1]["best_z_map"]
        ))

    def test_shared_architecture_and_runtime_are_explicit(self) -> None:
        architecture = self.summary["architecture"]
        counts = self.summary["counts"]
        runtime = self.summary["runtime"]
        self.assertEqual(architecture["shared_bellman_solve_count"], 1)
        self.assertIn("literal equivalence proof", architecture["bellman"])
        self.assertEqual(counts["number_of_candidates"], 96)
        self.assertEqual(counts["number_energy_feasible"], 60)
        self.assertEqual(counts["number_goal_reachable"], 6)
        self.assertEqual(counts["number_feasible"], 6)
        for field in (
            "candidate_generation_s", "energy_filter_s", "virtual_connection_s",
            "attacker_bellman_s", "total_attacker_br_s",
        ):
            self.assertGreaterEqual(runtime[field], 0.0)

    def test_visualizations_contain_required_layers(self) -> None:
        overview = (
            FIGURE_DIRECTORY / "exhaustive_attacker_best_response.html"
        ).read_text(encoding="utf-8")
        for label in (
            "Powered infeasible", "Energy infeasible", "Virtual infeasible",
            "Feasible", "Selected powered", "Selected virtual",
            "Selected Bellman glide", "Selected candidate 25",
        ):
            self.assertIn(label, overview)
        objective = (
            FIGURE_DIRECTORY / "candidate_objective_diagnostics.html"
        ).read_text(encoding="utf-8")
        for label in ("Attacker objective", "Mission time", "Cumulative hazard"):
            self.assertIn(label, objective)

    def test_public_api_accepts_nontrivial_generic_terrain(self) -> None:
        action = DefenderAction(np.array([5.0, 0.0, 0.0]))
        scenario = AttackerInitialCondition(
            Point3D(-8.0, 0.0, 0.0), Point3D(8.0, 0.0, 0.0),
        )
        response = attacker_best_response(
            action,
            scenario,
            terrain=build_terrain("stepped_pyramid"),
            contour_sample_count=4,
            radial_scales=(1.5,),
            quadrature_resolution=4,
        )
        self.assertIsInstance(response, AttackerResponse)
        if response.feasible:
            self.assertIsNotNone(response.switching_point_map)
            self.assertTrue(response.discrete_states)
        else:
            self.assertIsNone(response.switching_point_map)
            self.assertEqual(response.discrete_states, ())

    def test_original_gui_notebook_remains_frozen(self) -> None:
        digest = hashlib.sha256(
            (ROOT / "3D_bellman_0827.ipynb").read_bytes()
        ).hexdigest()
        self.assertEqual(digest, ORIGINAL_NOTEBOOK_SHA256)


if __name__ == "__main__":
    unittest.main()

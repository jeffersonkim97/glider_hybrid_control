"""Stage-8 progressive integration/regression exit gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from bellman_graph import build_bellman_graph, solve_unit_cost_reachability
from bellman_state import BellmanStateGrid
from candidate_energy import evaluate_switching_candidate
from los_explorer_gui import compute_los_case
from map_geometry import MapBounds
from switching_candidates import generate_switching_candidates
from virtual_connection import build_virtual_connections


ROOT = Path(__file__).resolve().parent
FIGURE_DIRECTORY = ROOT / "figure" / "stage_8_single_switch_connection"
ORIGINAL_NOTEBOOK_SHA256 = (
    "3b5938317df97814facf1c25066777d4efcdeca28fc8116fe99cd8e63b3a09a2"
)


class Stage8ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = compute_los_case("stepped_pyramid", 5.0, 0.0)
        cls.candidate = generate_switching_candidates(
            cls.case.tangent_contour,
        )[39]
        cls.evaluation = evaluate_switching_candidate(
            cls.candidate,
            cls.case.tangent_contour,
            cls.case.terrain_map,
            cls.case.mission_points,
        )
        cls.graph = build_bellman_graph(
            BellmanStateGrid(MapBounds(-8.0, 8.0, -4.0, 4.0)),
            cls.case.terrain_map,
            cls.case.mission_points.goal,
        )
        cls.unit = solve_unit_cost_reachability(cls.graph)
        cls.connections = build_virtual_connections(
            cls.evaluation,
            cls.graph,
            cls.unit.goal_reachable,
        )

    def test_nontrivial_terrain_reaches_virtual_adapter_through_generic_api(self) -> None:
        self.assertEqual(self.candidate.candidate_id, 39)
        self.assertTrue(self.evaluation.reachable)
        self.assertGreaterEqual(len(self.connections.proposals), 1)
        # V3 requires explicit infeasibility rather than accepting a snapped
        # heading.  This terrain still reaches the same generic adapter, but
        # candidate 39 has no chord-consistent local lattice connection.
        self.assertFalse(self.connections.has_connection)
        self.assertTrue(any(
            item.terrain_feasible for item in self.connections.proposals
        ))
        self.assertTrue(all(
            (
                "virtual-segment chord does not match target heading bin"
                in item.rejection_reasons
            )
            for item in self.connections.proposals
        ))

    def test_stage8_modules_do_not_name_concrete_terrain_classes(self) -> None:
        for filename in (
            "virtual_connection.py",
            "additive_bellman.py",
            "mission_response.py",
        ):
            source = (ROOT / filename).read_text(encoding="utf-8")
            self.assertNotIn("CubeObstacle", source)
            self.assertNotIn("TerrainMap3D", source)
            self.assertNotIn("SteppedPyramid", source)

    def test_required_visual_and_machine_readable_artifacts_exist(self) -> None:
        required = (
            "fixed_candidate_mission_response.html",
            "virtual_connection_proposals.csv",
            "continuous_mission_replay.csv",
            "stage8_single_candidate_summary.json",
        )
        for filename in required:
            path = FIGURE_DIRECTORY / filename
            self.assertTrue(path.is_file(), filename)
            self.assertGreater(path.stat().st_size, 0, filename)

    def test_artifact_summary_captures_stage8_contract(self) -> None:
        summary = json.loads(
            (FIGURE_DIRECTORY / "stage8_single_candidate_summary.json").read_text(
                encoding="utf-8",
            )
        )
        self.assertEqual(summary["scope"]["candidate_count_considered"], 1)
        self.assertFalse(summary["scope"]["candidate_optimized"])
        self.assertEqual(summary["scope"]["fixed_candidate_id"], 25)
        self.assertTrue(summary["virtual_connection"]["feasible"])
        self.assertTrue(summary["bellman"]["feasible"])
        self.assertTrue(summary["energy"]["feasible"])
        self.assertEqual(summary["mission"]["maximum_join_gap_m"], 0.0)
        self.assertTrue(summary["mission"]["within_goal_tolerance"])
        phase_sum = sum(
            item["weighted_cost"]
            for item in summary["phase_objective"].values()
        )
        self.assertAlmostEqual(
            phase_sum, summary["mission"]["objective"], places=10,
        )

    def test_original_gui_notebook_remains_frozen(self) -> None:
        digest = hashlib.sha256(
            (ROOT / "3D_bellman_0827.ipynb").read_bytes()
        ).hexdigest()
        self.assertEqual(digest, ORIGINAL_NOTEBOOK_SHA256)


if __name__ == "__main__":
    unittest.main()

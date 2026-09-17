"""Stage-8 objective decomposition and continuous replay verification."""

from __future__ import annotations

import unittest

import numpy as np

from bellman_graph import build_bellman_graph, solve_unit_cost_reachability
from bellman_state import BellmanStateGrid
from candidate_energy import evaluate_switching_candidate
from detection_hazard import (
    DEFAULT_ATTACKER_HAZARD_TIME,
    GlideDetectionHazardModel,
)
from los_explorer_gui import compute_los_case
from map_geometry import MapBounds
from mission_response import solve_single_candidate_response
from switching_candidates import generate_switching_candidates


class SingleCandidateMissionResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = compute_los_case("centered_cube", 5.0, 0.0)
        cls.candidate = generate_switching_candidates(
            cls.case.tangent_contour,
        )[25]
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
        cls.response = solve_single_candidate_response(
            cls.evaluation,
            cls.graph,
            cls.unit,
            cls.case.mission_points,
            GlideDetectionHazardModel(
                cls.case.terrain_map,
                cls.case.mission_points.sensor,
            ),
            quadrature_resolution=8,
        )

    def test_exactly_one_fixed_candidate_is_considered(self) -> None:
        self.assertEqual(self.response.candidate_count_considered, 1)
        self.assertEqual(self.response.candidate_id, 25)
        self.assertTrue(self.response.feasible)
        self.assertEqual(len(self.response.options), 1)

    def test_v6_objective_decomposes_exactly(self) -> None:
        option = self.response.selected_option
        self.assertIsNotNone(option)
        phases = (
            option.powered_phase,
            option.virtual_phase,
            option.glide_phase,
        )
        self.assertAlmostEqual(
            option.objective.objective_value,
            sum(phase.weighted_cost for phase in phases),
            places=11,
        )
        self.assertAlmostEqual(
            option.objective.mission_time_s,
            sum(phase.duration_s for phase in phases),
            places=11,
        )
        self.assertAlmostEqual(
            option.objective.mission_hazard,
            sum(phase.hazard for phase in phases),
            places=11,
        )
        expected = (
            DEFAULT_ATTACKER_HAZARD_TIME.hazard_weight
            * option.objective.mission_hazard
            / DEFAULT_ATTACKER_HAZARD_TIME.hazard_reference
            + DEFAULT_ATTACKER_HAZARD_TIME.time_weight
            * option.objective.mission_time_s
            / DEFAULT_ATTACKER_HAZARD_TIME.time_reference_s
        )
        self.assertAlmostEqual(option.objective.objective_value, expected, places=12)

    def test_powered_detection_assumption_is_explicit(self) -> None:
        option = self.response.selected_option
        self.assertEqual(option.powered_phase.hazard, 0.0)
        self.assertIn("powered hazard = 0", self.response.powered_detection_assumption)

    def test_bellman_continuation_equals_independent_edge_sum(self) -> None:
        option = self.response.selected_option
        connection_id = option.connection.target_state_id
        edge_sum = sum(
            self.response.bellman_solution.edge_cost_by_source[edge.source_id][
                next(
                    index for index, candidate in enumerate(
                        self.graph.adjacency[edge.source_id]
                    )
                    if candidate.target_id == edge.target_id
                )
            ]
            for edge in option.glide_edges
        )
        self.assertAlmostEqual(
            self.response.bellman_solution.value[connection_id],
            edge_sum,
            places=11,
        )

    def test_continuous_replay_has_no_join_gap_and_reaches_goal_ball(self) -> None:
        replay = self.response.replay
        self.assertIsNotNone(replay)
        self.assertLessEqual(replay.maximum_join_gap_m, 1.0e-9)
        self.assertTrue(replay.within_goal_tolerance)
        self.assertLessEqual(replay.terminal_goal_distance_m, 25.0)
        self.assertEqual(replay.segment_labels[0:2], ("powered", "virtual"))
        self.assertTrue(all(
            label == "glide" for label in replay.segment_labels[2:]
        ))

    def test_selected_path_energy_is_positive_and_consistent(self) -> None:
        certificate = self.response.selected_option.energy
        self.assertTrue(certificate.segment_geometry_feasible)
        self.assertTrue(certificate.feasible)
        self.assertGreater(certificate.margin_m, 0.0)
        self.assertTrue(np.isfinite(certificate.margin_j))


if __name__ == "__main__":
    unittest.main()

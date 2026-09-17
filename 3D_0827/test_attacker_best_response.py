"""Stage-9 exhaustive candidate-selection and shared-solve verification."""

from __future__ import annotations

import unittest

import numpy as np

from attacker_best_response import (
    ATTACKER_TIE_BREAK_CONVENTION,
    CandidateSelectionScore,
    _attacker_response,
    evaluate_candidates_literal,
    evaluate_candidates_shared,
    select_attacker_candidate,
)
from bellman_graph import build_bellman_graph, solve_unit_cost_reachability
from bellman_state import BellmanStateGrid
from candidate_energy import evaluate_switching_candidates
from detection_hazard import GlideDetectionHazardModel
from los_explorer_gui import compute_los_case
from map_geometry import MapBounds
from switching_candidates import generate_switching_candidates


class AttackerBestResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = compute_los_case("centered_cube", 5.0, 0.0)
        candidates = generate_switching_candidates(cls.case.tangent_contour)
        cls.all_evaluations = evaluate_switching_candidates(
            candidates,
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
        cls.hazard = GlideDetectionHazardModel(
            cls.case.terrain_map,
            cls.case.mission_points.sensor,
        )
        cls.tiny_evaluations = tuple(
            cls.all_evaluations[index] for index in (24, 25, 26)
        )
        cls.literal = evaluate_candidates_literal(
            cls.tiny_evaluations,
            cls.graph,
            cls.unit,
            cls.case.mission_points,
            cls.hazard,
            quadrature_resolution=4,
        )
        cls.shared, cls.shared_solution, cls.shared_hazards, _ = (
            evaluate_candidates_shared(
                cls.tiny_evaluations,
                cls.graph,
                cls.unit,
                cls.case.mission_points,
                cls.hazard,
                quadrature_resolution=4,
            )
        )

    def test_a9_1_tiny_candidate_set_matches_manual_minimum(self) -> None:
        self.assertEqual(tuple(result.candidate_id for result in self.literal), (24, 25, 26))
        self.assertFalse(self.literal[0].feasible)
        feasible = tuple(result for result in self.literal if result.feasible)
        manual = min(feasible, key=lambda result: (result.objective, result.candidate_id))
        selected_id, _ = select_attacker_candidate(
            tuple(result.selection_score for result in self.literal)
        )
        self.assertEqual(selected_id, manual.candidate_id)
        self.assertAlmostEqual(
            float(manual.objective),
            min(float(result.objective) for result in feasible),
            places=12,
        )

    def test_a9_2_duplicate_objective_tie_is_deterministic(self) -> None:
        scores = (
            CandidateSelectionScore(8, True, 0.25),
            CandidateSelectionScore(3, True, 0.25),
            CandidateSelectionScore(1, False, None),
        )
        selected, cooptimal = select_attacker_candidate(scores)
        self.assertEqual(selected, 3)
        self.assertEqual(cooptimal, (3, 8))
        self.assertIn("lowest candidate_id", ATTACKER_TIE_BREAK_CONVENTION)

    def test_a9_3_all_infeasible_returns_no_fabricated_trajectory(self) -> None:
        result = self.shared[0]
        self.assertFalse(result.feasible)
        selected, cooptimal = select_attacker_candidate((result.selection_score,))
        self.assertIsNone(selected)
        self.assertEqual(cooptimal, ())
        response = _attacker_response(None)
        self.assertFalse(response.feasible)
        self.assertIsNone(response.switching_point_map)
        self.assertEqual(response.discrete_states, ())
        self.assertTrue(np.isposinf(response.objective))

    def test_a9_4_one_feasible_candidate_is_selected(self) -> None:
        feasible = next(result for result in self.shared if result.feasible)
        selected, cooptimal = select_attacker_candidate((feasible.selection_score,))
        self.assertEqual(selected, feasible.candidate_id)
        self.assertEqual(cooptimal, (feasible.candidate_id,))
        response = _attacker_response(feasible)
        self.assertTrue(response.feasible)
        self.assertGreater(len(response.discrete_states), 0)

    def test_a9_5_shared_solve_equals_literal_exhaustive(self) -> None:
        self.assertEqual(len(self.literal), len(self.shared))
        for literal, shared in zip(self.literal, self.shared, strict=True):
            with self.subTest(candidate_id=literal.candidate_id):
                self.assertEqual(literal.candidate_id, shared.candidate_id)
                self.assertEqual(literal.feasible, shared.feasible)
                self.assertEqual(literal.infeasibility_reason, shared.infeasibility_reason)
                if literal.feasible:
                    self.assertAlmostEqual(literal.objective, shared.objective, places=11)
                    self.assertAlmostEqual(
                        literal.mission_time_s, shared.mission_time_s, places=11,
                    )
                    self.assertAlmostEqual(
                        literal.cumulative_hazard,
                        shared.cumulative_hazard,
                        places=11,
                    )
                    self.assertEqual(
                        literal.selected_option.connection.target_state_id,
                        shared.selected_option.connection.target_state_id,
                    )

    def test_every_candidate_has_complete_result_or_failure_reason(self) -> None:
        for result in self.shared:
            if result.feasible:
                self.assertIsNone(result.infeasibility_reason)
                self.assertIsNotNone(result.objective)
                self.assertIsNotNone(result.replay)
            else:
                self.assertIsNotNone(result.infeasibility_reason)
                self.assertIsNone(result.objective)


if __name__ == "__main__":
    unittest.main()

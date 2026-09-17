"""Stage-8 verification of the continuous-to-lattice virtual adapter."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from bellman_geometry import wrapped_angle_difference
from bellman_graph import build_bellman_graph, solve_unit_cost_reachability
from bellman_state import BellmanStateGrid
from candidate_energy import CandidateEnergyEvaluation, evaluate_switching_candidate
from energy_model import DEFAULT_GLIDER
from los_explorer_gui import compute_los_case
from map_geometry import MapBounds
from reachability_surface import PointReachability
from switching_candidates import generate_switching_candidates
from virtual_connection import (
    build_virtual_connections,
    certify_mission_energy,
    heading_turn_is_feasible,
)


class VirtualConnectionTests(unittest.TestCase):
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
        cls.unit_solution = solve_unit_cost_reachability(cls.graph)

    @classmethod
    def evaluation_at(
        cls,
        position: tuple[float, float, float],
        heading_rad: float,
        *,
        total_energy_j: float | None = None,
    ) -> CandidateEnergyEvaluation:
        old_state = cls.evaluation.switching_state
        position_map = np.asarray(position, dtype=float)
        state = replace(
            old_state,
            position_map=position_map,
            position_m=100.0 * position_map,
            heading_rad=heading_rad,
            total_mechanical_energy_j=(
                old_state.total_mechanical_energy_j
                if total_energy_j is None
                else total_energy_j
            ),
        )
        point = PointReachability(
            switching_state=state,
            glide_result=cls.evaluation.glide_result,
        )
        return CandidateEnergyEvaluation(
            candidate=cls.evaluation.candidate,
            acoustically_neutralized=True,
            point_reachability=point,
        )

    def test_v1_exact_lattice_state_has_identity_connection(self) -> None:
        evaluation = self.evaluation_at((1.0, 3.0, 1.6), 0.0)
        connections = build_virtual_connections(
            evaluation,
            self.graph,
            self.unit_solution.goal_reachable,
        )
        self.assertEqual(len(connections.proposals), 1)
        proposal = connections.proposals[0]
        self.assertTrue(proposal.feasible)
        self.assertTrue(proposal.identity_connection)
        self.assertEqual(proposal.duration_s, 0.0)
        self.assertEqual(proposal.heading_mismatch_rad, 0.0)

    def test_v2_half_cell_uses_deterministic_floor_ceil_rule(self) -> None:
        evaluation = self.evaluation_at((0.5, 3.0, 1.65), 0.0)
        first = build_virtual_connections(
            evaluation,
            self.graph,
            self.unit_solution.goal_reachable,
        )
        second = build_virtual_connections(
            evaluation,
            self.graph,
            self.unit_solution.goal_reachable,
        )
        self.assertEqual(
            tuple(item.target_state_id for item in first.proposals),
            tuple(item.target_state_id for item in second.proposals),
        )
        self.assertEqual(
            {item.target_position_map[0] for item in first.proposals},
            {0.0, 1.0},
        )
        self.assertEqual(
            {round(float(item.target_position_map[2]), 12) for item in first.proposals},
            {1.6},
        )

    def test_v2a_vertical_snap_is_rejected_and_uses_stencil_fallback(self) -> None:
        evaluation = self.evaluation_at((1.0, 3.0, 1.65), 0.0)
        connections = build_virtual_connections(
            evaluation,
            self.graph,
            self.unit_solution.goal_reachable,
        )
        zero_duration = [
            item for item in connections.proposals if item.duration_s == 0.0
        ]
        self.assertGreater(len(connections.proposals), 1)
        self.assertTrue(zero_duration)
        self.assertTrue(all(not item.feasible for item in zero_duration))
        self.assertTrue(all(
            "zero-duration virtual segment has distinct endpoints"
            in item.rejection_reasons
            for item in zero_duration
        ))
        self.assertTrue(all(
            item.duration_s > 0.0 for item in connections.feasible_proposals
        ))

    def test_v2b_turn_infeasible_identity_uses_only_incident_cells(self) -> None:
        evaluation = self.evaluation_at((1.0, 3.0, 1.65), 0.3)
        connections = build_virtual_connections(
            evaluation,
            self.graph,
            self.unit_solution.goal_reachable,
        )
        self.assertGreater(len(connections.proposals), 1)
        self.assertTrue(connections.has_connection)
        self.assertTrue(all(
            abs(item.target_position_map[0] - 1.0) <= 1.0
            and abs(item.target_position_map[1] - 3.0) <= 1.0
            for item in connections.proposals
        ))
        self.assertTrue(all(
            item.chord_heading_feasible for item in connections.feasible_proposals
        ))

    def test_v3_outside_grid_has_no_forced_snap(self) -> None:
        evaluation = self.evaluation_at((20.0, 3.0, 1.6), 0.0)
        connections = build_virtual_connections(
            evaluation,
            self.graph,
            self.unit_solution.goal_reachable,
        )
        self.assertEqual(connections.proposals, ())
        self.assertFalse(connections.has_connection)

    def test_v4_turn_boundary_is_inclusive_then_rejects_beyond(self) -> None:
        duration = 3.0
        exact_limit = DEFAULT_GLIDER.maximum_turn_rate_rad_s * duration
        self.assertTrue(heading_turn_is_feasible(0.0, 0.0, duration))
        self.assertTrue(heading_turn_is_feasible(0.0, exact_limit, duration))
        self.assertFalse(
            heading_turn_is_feasible(0.0, exact_limit + 1.0e-8, duration)
        )
        self.assertAlmostEqual(
            wrapped_angle_difference(0.0, exact_limit), exact_limit,
        )

    def test_v5_full_path_energy_is_consistent_and_can_fail(self) -> None:
        connections = build_virtual_connections(
            self.evaluation,
            self.graph,
            self.unit_solution.goal_reachable,
        )
        proposal = min(
            connections.feasible_proposals,
            key=lambda item: (item.projection_error_m, item.target_state_id),
        )
        state_ids = self.unit_solution.backtrack(proposal.target_state_id)
        edges = tuple(
            next(
                edge for edge in self.graph.adjacency[source_id]
                if edge.target_id == target_id
            )
            for source_id, target_id in zip(state_ids, state_ids[1:])
        )
        certificate = certify_mission_energy(
            self.evaluation.switching_state,
            proposal,
            edges,
            self.graph,
        )
        self.assertTrue(certificate.segment_geometry_feasible)
        self.assertTrue(certificate.feasible)
        self.assertAlmostEqual(
            certificate.required_specific_height_m,
            certificate.switch_loss_m
            + certificate.virtual_drag_loss_m
            + certificate.glide_drag_loss_m
            + certificate.terminal_altitude_m,
        )

        low_energy_state = replace(
            self.evaluation.switching_state,
            total_mechanical_energy_j=0.0,
        )
        failed = certify_mission_energy(
            low_energy_state,
            proposal,
            edges,
            self.graph,
        )
        self.assertLess(failed.margin_m, 0.0)
        self.assertFalse(failed.feasible)


if __name__ == "__main__":
    unittest.main()

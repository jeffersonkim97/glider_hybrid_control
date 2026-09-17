"""Stage-12 finite Stackelberg selection tests independent of geometry."""

from __future__ import annotations

import unittest

import numpy as np

from game_types import AttackerResponse, DefenderAction, GameOutcome
from stackelberg_interface import select_sse_follower_outcome
from stackelberg_solver import (
    generate_defender_line_actions,
    select_stackelberg_outcome,
)


def _response(attacker_cost: float, pod: float) -> AttackerResponse:
    return AttackerResponse(
        feasible=True,
        switching_point_map=np.array([0.0, 0.0, 1.0]),
        discrete_states=((0, 0, 1, 0),),
        objective=attacker_cost,
        mission_time_s=1.0,
        cumulative_hazard=float(-np.log1p(-pod)),
        detection_probability=pod,
    )


def _outcome(sensor_x: float, attacker_cost: float, defender_pod: float) -> GameOutcome:
    action = DefenderAction(np.array([sensor_x, 0.0, 0.0]))
    return GameOutcome(
        defender_action=action,
        attacker_response=_response(attacker_cost, defender_pod),
        defender_payoff=defender_pod,
        attacker_payoff=attacker_cost,
    )


class StackelbergSolverTests(unittest.TestCase):
    def test_d12_1_handwritten_leader_table_has_known_answer(self) -> None:
        outcomes = (
            _outcome(5.0, 0.3, 0.2),
            _outcome(7.5, 0.4, 0.8),
            _outcome(10.0, 0.2, 0.5),
        )
        selected = select_stackelberg_outcome(outcomes)
        np.testing.assert_array_equal(
            selected.defender_action.sensor_position_map,
            np.array([7.5, 0.0, 0.0]),
        )

    def test_d12_2_one_defender_action_is_trivial_equilibrium(self) -> None:
        only = _outcome(5.0, 0.25, 0.6)
        self.assertIs(select_stackelberg_outcome((only,)), only)

    def test_d12_3_unique_maximum_defender_payoff_wins(self) -> None:
        low = _outcome(5.0, 0.1, 0.2)
        high = _outcome(10.0, 0.9, 0.7)
        self.assertIs(select_stackelberg_outcome((low, high)), high)

    def test_d12_4_sse_follower_tie_favors_defender(self) -> None:
        action = DefenderAction(np.array([5.0, 0.0, 0.0]))
        low_pod = GameOutcome(action, _response(0.3, 0.2), 0.2, 0.3)
        high_pod = GameOutcome(action, _response(0.3, 0.8), 0.8, 0.3)
        selected = select_sse_follower_outcome((low_pod, high_pod))
        self.assertIs(selected, high_pod)

    def test_remaining_leader_tie_uses_stable_input_order(self) -> None:
        first = _outcome(5.0, 0.2, 0.7)
        second = _outcome(10.0, 0.1, 0.7)
        self.assertIs(select_stackelberg_outcome((first, second)), first)

    def test_defender_line_generator_is_deterministic(self) -> None:
        candidates = generate_defender_line_actions((5.0, 7.5, 10.0))
        self.assertEqual([item.action_id for item in candidates], [0, 1, 2])
        np.testing.assert_allclose(
            [item.action.sensor_position_map for item in candidates],
            [[5.0, 0.0, 0.0], [7.5, 0.0, 0.0], [10.0, 0.0, 0.0]],
        )

    def test_invalid_or_empty_action_sets_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            select_stackelberg_outcome(())
        with self.assertRaisesRegex(ValueError, "finite"):
            generate_defender_line_actions((5.0, np.nan))


if __name__ == "__main__":
    unittest.main()

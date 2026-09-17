"""Stage-2.5 interface, payoff, tie-break, and import-boundary tests."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from attacker_best_response import attacker_best_response
from game_types import (
    ATTACKER_OBJECTIVE_COMPONENTS,
    ATTACKER_OBJECTIVE_SENSE,
    DEFENDER_OBJECTIVE_COMPONENTS,
    DEFENDER_PAYOFF_SENSE,
    SSE_TIE_BREAK_CONVENTION,
    ZERO_SUM_ASSUMED,
    AttackerInitialCondition,
    AttackerResponse,
    DefenderAction,
    GameOutcome,
)
from scenario import Point3D
from stackelberg_interface import select_sse_follower_outcome


def _response(objective: float, state_id: int = 0) -> AttackerResponse:
    return AttackerResponse(
        feasible=True,
        switching_point_map=np.array([1.0, 2.0, 3.0]),
        discrete_states=((state_id, 0, 0),),
        objective=objective,
        mission_time_s=12.0,
        cumulative_hazard=0.25,
        detection_probability=0.2,
    )


def _outcome(
    action: DefenderAction,
    *,
    attacker_cost: float,
    defender_payoff: float,
    state_id: int,
) -> GameOutcome:
    return GameOutcome(
        defender_action=action,
        attacker_response=_response(attacker_cost, state_id),
        attacker_payoff=attacker_cost,
        defender_payoff=defender_payoff,
    )


class GameContractTests(unittest.TestCase):
    def test_g1_contracts_are_immutable_and_copy_arrays(self) -> None:
        source_position = np.array([5.0, 0.0, 0.0])
        action = DefenderAction(source_position)
        source_position[0] = 99.0
        np.testing.assert_array_equal(action.sensor_position_map, [5.0, 0.0, 0.0])
        self.assertFalse(action.sensor_position_map.flags.writeable)
        with self.assertRaises(ValueError):
            action.sensor_position_map[0] = 6.0
        with self.assertRaises(FrozenInstanceError):
            action.sensor_position_map = np.zeros(3)

        source_switch = np.array([1.0, 2.0, 3.0])
        response = AttackerResponse(
            feasible=True,
            switching_point_map=source_switch,
            discrete_states=((1, 2, 3),),
            objective=4.0,
            mission_time_s=5.0,
            cumulative_hazard=0.2,
            detection_probability=0.1,
        )
        source_switch[:] = -1.0
        np.testing.assert_array_equal(response.switching_point_map, [1.0, 2.0, 3.0])
        assert response.switching_point_map is not None
        self.assertFalse(response.switching_point_map.flags.writeable)

    def test_g1_invalid_contract_values_are_rejected(self) -> None:
        for invalid_position in (
            np.array([1.0, 2.0]),
            np.array([1.0, np.nan, 3.0]),
            np.array([1.0, np.inf, 3.0]),
        ):
            with self.subTest(position=invalid_position):
                with self.assertRaises(ValueError):
                    DefenderAction(invalid_position)

        with self.assertRaises(ValueError):
            AttackerInitialCondition(Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            AttackerResponse(True, None, ((0,),), 1.0, 1.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            AttackerResponse(False, np.zeros(3), (), 1.0, 1.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            AttackerResponse(True, np.zeros(3), (), 1.0, 1.0, 0.0, 0.0)
        with self.assertRaises(TypeError):
            AttackerResponse(True, np.zeros(3), [(0,)], 1.0, 1.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            AttackerResponse(True, np.zeros(3), ((0,),), np.inf, 1.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            AttackerResponse(True, np.zeros(3), ((0,),), 1.0, -1.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            AttackerResponse(True, np.zeros(3), ((0,),), 1.0, 1.0, -0.1, 0.0)
        with self.assertRaises(ValueError):
            AttackerResponse(True, np.zeros(3), ((0,),), 1.0, 1.0, 0.0, 1.1)

    def test_g2_payoff_optimization_directions_are_fixed(self) -> None:
        self.assertEqual(ATTACKER_OBJECTIVE_SENSE, "minimize")
        self.assertEqual(DEFENDER_PAYOFF_SENSE, "maximize")
        self.assertEqual(
            ATTACKER_OBJECTIVE_COMPONENTS,
            ("mission_time_s", "detection_probability"),
        )
        self.assertEqual(
            DEFENDER_OBJECTIVE_COMPONENTS,
            ("detection_probability",),
        )
        self.assertFalse(ZERO_SUM_ASSUMED)

        action = DefenderAction(np.array([5.0, 0.0, 0.0]))
        lower_attacker_cost = _outcome(
            action, attacker_cost=4.0, defender_payoff=1.0, state_id=1,
        )
        higher_attacker_cost = _outcome(
            action, attacker_cost=7.0, defender_payoff=100.0, state_id=2,
        )
        selected = select_sse_follower_outcome(
            (higher_attacker_cost, lower_attacker_cost),
        )
        self.assertIs(selected, lower_attacker_cost)

    def test_g3_sse_tie_break_favors_defender(self) -> None:
        action = DefenderAction(np.array([5.0, 0.0, 0.0]))
        a1 = _outcome(action, attacker_cost=10.0, defender_payoff=2.0, state_id=1)
        a2 = _outcome(action, attacker_cost=10.0, defender_payoff=5.0, state_id=2)
        selected = select_sse_follower_outcome((a1, a2))
        self.assertIs(selected, a2)
        self.assertIn("maximum Defender payoff", SSE_TIE_BREAK_CONVENTION)

    def test_sse_selector_validates_fixed_leader_action_and_tolerance(self) -> None:
        first_action = DefenderAction(np.array([5.0, 0.0, 0.0]))
        second_action = DefenderAction(np.array([6.0, 0.0, 0.0]))
        first = _outcome(first_action, attacker_cost=10.0, defender_payoff=2.0, state_id=1)
        second = _outcome(second_action, attacker_cost=10.0, defender_payoff=5.0, state_id=2)
        with self.assertRaises(ValueError):
            select_sse_follower_outcome(())
        with self.assertRaises(ValueError):
            select_sse_follower_outcome((first, second))
        with self.assertRaises(ValueError):
            select_sse_follower_outcome((first,), objective_tolerance=-1.0)

    def test_attacker_best_response_is_now_a_validated_implementation_boundary(self) -> None:
        action = DefenderAction(np.array([5.0, 0.0, 0.0]))
        scenario = AttackerInitialCondition(
            start=Point3D(-8.0, 0.0, 0.0),
            goal=Point3D(8.0, 0.0, 0.0),
        )
        with self.assertRaises(TypeError):
            attacker_best_response("not a defender action", scenario)
        with self.assertRaises(TypeError):
            attacker_best_response(action, "not an initial condition")

    def test_g4_import_layers_are_acyclic_and_directional(self) -> None:
        directory = Path(__file__).resolve().parent
        modules = ("game_types", "attacker_best_response", "stackelberg_interface")
        for order in (
            modules,
            tuple(reversed(modules)),
            ("attacker_best_response", "stackelberg_interface", "game_types"),
        ):
            command = [sys.executable, "-c", ";".join(f"import {name}" for name in order)]
            completed = subprocess.run(
                command,
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
            with self.subTest(order=order):
                self.assertEqual(completed.returncode, 0, completed.stderr)

        imports: dict[str, set[str]] = {}
        for filename in (
            "game_types.py",
            "attacker_best_response.py",
            "stackelberg_interface.py",
        ):
            tree = ast.parse((directory / filename).read_text(encoding="utf-8"))
            imports[filename] = {
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
        self.assertFalse(any("bellman" in name.lower() for name in imports["game_types.py"]))
        self.assertFalse(
            any("defender" in name.lower() for name in imports["attacker_best_response.py"])
        )
        self.assertIn("attacker_best_response", imports["stackelberg_interface.py"])


if __name__ == "__main__":
    unittest.main()

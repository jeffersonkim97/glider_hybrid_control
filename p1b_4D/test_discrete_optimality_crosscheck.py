"""Plan item 6 cross-check (see discrete_optimality_proposition.md).

Builds a small toy grid via the real construct_coarse_transitions /
construct_stage_cost_4d code, then compares solve_coarse_bellman's
cost-to-go against an independently computed shortest-cost-to-go
(networkx Dijkstra on the same transition graph, reimplemented from
scratch here rather than reusing any Bellman-recursion code). Tests
Proposition 3's conclusion against a trusted external algorithm; it does
not re-derive or re-validate the underlying physics/detection formulas,
which are covered by the rest of the test suite.
"""

from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx
import numpy as np

from p1b_4D.bellman import construct_coarse_transitions, solve_coarse_bellman
from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.detection import build_symbolic_detection_bundle
from p1b_4D.geometry import build_geometry_bundle
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.stage_cost import construct_stage_cost_4d

_GOAL = "__GOAL__"

# Ridge/width = 400/60 = 6.67, clears the launch-point terrain_tolerance
# the same way the production single-hill scenario's 2500/400 = 6.25 does
# (see p1b_roadmap_0727.md item 1 notes) -- a wider ratio here would leave
# too few reachable toy-grid cells for a meaningful cross-check.
_TOY_DOMAIN = {"z_min": 0.0, "z_max": 800.0, "h_min": 0.0, "h_max": 100.0}
_TOY_HILL = {"z_ridge": 400.0, "h_ridge": 50.0, "width": 60.0}
_TOY_GRID = {"z_count": 33, "h_count": 21, "v_count": 3, "gamma_count": 5}


def _build_toy_configuration_bundle(project_root: Path) -> dict:
    configuration_bundle = deepcopy(build_configuration_bundle(project_root))
    environment = configuration_bundle["primary_result"]["environment_config"]
    environment["z_start"] = 0.0
    environment["z_goal"] = 700.0
    environment["terrain"] = {
        "z_min": _TOY_DOMAIN["z_min"],
        "z_max": _TOY_DOMAIN["z_max"],
        "hills": (_TOY_HILL,),
    }
    z_min, z_max = _TOY_DOMAIN["z_min"], _TOY_DOMAIN["z_max"]
    h_min, h_max = _TOY_DOMAIN["h_min"], _TOY_DOMAIN["h_max"]
    environment["grid"] = {
        "z_min": z_min,
        "z_max": z_max,
        "z_count": _TOY_GRID["z_count"],
        "z_spacing": (z_max - z_min) / (_TOY_GRID["z_count"] - 1),
        "h_min": h_min,
        "h_max": h_max,
        "h_count": _TOY_GRID["h_count"],
        "h_spacing": (h_max - h_min) / (_TOY_GRID["h_count"] - 1),
        "v_count": _TOY_GRID["v_count"],
        "gamma_count": _TOY_GRID["gamma_count"],
        "axis_order_4d": ("z", "h", "v", "gamma"),
    }
    environment["airspace"] = {"z_min": z_min, "z_max": z_max, "h_min": h_min, "h_max": h_max}
    environment["simulation"] = {
        "z_min": z_min, "z_max": z_max, "h_min": h_min, "h_max": h_max,
        "max_path_steps": 200,
    }
    vehicle = configuration_bundle["primary_result"]["vehicle_config"]
    vehicle["glide_speed_count"] = _TOY_GRID["v_count"]
    vehicle["gamma_count"] = _TOY_GRID["gamma_count"]
    configuration_bundle["primary_result"]["sensor_config"]["default_z_sensor"] = 600.0
    return configuration_bundle


def _independent_shortest_cost_to_go(
    transitions: dict, j4d: np.ndarray, goal_mask: np.ndarray,
) -> np.ndarray:
    """Reimplements the shortest-cost-to-go computation from scratch using
    networkx's Dijkstra, over the identical (already-constructed, already
    separately validated) transition graph -- deliberately not reusing any
    of solve_coarse_bellman's own recursion logic.
    """
    shape = goal_mask.shape
    graph = nx.DiGraph()
    graph.add_node(_GOAL)
    for z_index in range(shape[0]):
        for h_index in range(shape[1]):
            graph.add_node((z_index, h_index))

    velocity_count = transitions["transition_valid"].shape[2]
    gamma_count = transitions["transition_valid"].shape[3]
    for z_index in range(shape[0]):
        for h_index in range(shape[1]):
            if goal_mask[z_index, h_index]:
                continue
            for velocity_index in range(velocity_count):
                for gamma_index in range(gamma_count):
                    if not transitions["transition_valid"][z_index, h_index, velocity_index, gamma_index]:
                        continue
                    terminal = bool(
                        transitions["terminal_transition"][z_index, h_index, velocity_index, gamma_index]
                    )
                    fraction = (
                        transitions["terminal_fraction"][z_index, h_index, velocity_index, gamma_index]
                        if terminal else 1.0
                    )
                    cost = float(fraction * j4d[z_index, h_index, velocity_index, gamma_index])
                    source = (z_index, h_index)
                    if terminal:
                        target = _GOAL
                    else:
                        next_z = int(transitions["next_z_index"][z_index, h_index, velocity_index, gamma_index])
                        next_h = int(transitions["next_h_index"][z_index, h_index, velocity_index, gamma_index])
                        target = (next_z, next_h)
                    if graph.has_edge(source, target):
                        if graph[source][target]["weight"] > cost:
                            graph[source][target]["weight"] = cost
                    else:
                        graph.add_edge(source, target, weight=cost)

    # Shortest distance from every state TO the goal == shortest distance
    # FROM the goal on the reversed graph.
    distances = nx.single_source_dijkstra_path_length(
        graph.reverse(copy=False), _GOAL, weight="weight",
    )

    value = np.full(shape, np.inf)
    for z_index in range(shape[0]):
        for h_index in range(shape[1]):
            node = (z_index, h_index)
            if node in distances:
                value[z_index, h_index] = distances[node]
            elif goal_mask[z_index, h_index]:
                value[z_index, h_index] = 0.0
    return value


class DiscreteOptimalityCrossCheckTests(unittest.TestCase):
    """Verify solve_coarse_bellman's cost-to-go against an independent
    shortest-path solver on the same (real, unmodified) transition graph.
    """

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.configuration_bundle = _build_toy_configuration_bundle(
            Path(self.temporary_directory.name)
        )
        logger = self.configuration_bundle["primary_result"]["logging_utilities"]["logger"]
        self.addCleanup(close_phase_logger, logger)
        self.geometry_bundle = build_geometry_bundle(self.configuration_bundle)
        self.detection_bundle = build_symbolic_detection_bundle(
            self.configuration_bundle, self.geometry_bundle,
        )
        self.stage_cost_bundle = construct_stage_cost_4d(
            self.configuration_bundle, self.geometry_bundle, self.detection_bundle,
        )
        self.transitions = construct_coarse_transitions(
            self.geometry_bundle, self.stage_cost_bundle, self.configuration_bundle,
        )

    def test_bellman_value_matches_independent_dijkstra(self) -> None:
        stage = self.stage_cost_bundle["primary_result"]
        j4d = stage["j4d"]
        grids = stage["grids"]
        goal_position = self.geometry_bundle["primary_result"]["goal_position"]
        validation_config = self.configuration_bundle["primary_result"]["validation_config"]
        bellman_config = self.configuration_bundle["primary_result"]["bellman_config"]

        policy = solve_coarse_bellman(
            j4d, self.transitions, grids, goal_position, validation_config,
            "low_gamma_first", bellman_config,
        )
        real_value = np.asarray(policy["value"])
        goal_mask = np.asarray(policy["goal_mask"])

        independent_value = _independent_shortest_cost_to_go(self.transitions, j4d, goal_mask)

        finite_real = np.isfinite(real_value)
        finite_independent = np.isfinite(independent_value)

        # A meaningful cross-check needs more than a handful of reachable
        # cells -- this toy grid was sized specifically to clear this bar.
        reachable_count = int(np.count_nonzero(finite_real))
        self.assertGreater(reachable_count, 100)

        np.testing.assert_array_equal(finite_real, finite_independent)
        np.testing.assert_allclose(
            real_value[finite_real], independent_value[finite_real],
            rtol=0.0, atol=1.0e-9,
        )


if __name__ == "__main__":
    unittest.main()

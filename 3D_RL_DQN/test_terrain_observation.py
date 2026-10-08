"""Validation tests for the Phase 16.2 terrain-aware observation contract."""

from __future__ import annotations

import json
import unittest

import numpy as np

import project_paths  # noqa: F401
from P1b_Exact_Local_SSE import exact_best_response
from P1b_RL_approximation import default_sensor
from P1b_condition import ComputationCondition, build_scene
from attacker_br_problem import AttackerBRProblem
from terrain_observation import (
    EGO_FEATURE_NAMES,
    GOAL_FEATURE_NAMES,
    HAZARD_CHANNEL_NAMES,
    ObservationConfig,
    TERRAIN_CHANNEL_NAMES,
    TerrainObservationBuilder,
    VALIDITY_CHANNEL_NAMES,
)


class TerrainObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = build_scene(ComputationCondition(spatial_resolution_m=100.0))
        cls.problem = AttackerBRProblem(cls.scene, default_sensor(cls.scene))
        cls.oracle = exact_best_response(cls.scene, cls.problem.sensor_map, keep_solution=True)
        if not cls.oracle.feasible:
            raise unittest.SkipTest("canonical cube exact response is infeasible")
        cls.config = ObservationConfig(
            local_window_extent_m=2000.0,
            local_window_shape_cells=(21, 21),
            local_window_orientation="heading_aligned",
        )
        cls.builder = TerrainObservationBuilder(cls.problem, cls.config)
        cls.state_id = int(cls.oracle.glide_state_ids[0])
        cls.observation = cls.builder.build(cls.state_id)

    def test_determinism(self) -> None:
        repeat = self.builder.build(self.state_id)
        np.testing.assert_array_equal(self.observation.ego_features, repeat.ego_features)
        np.testing.assert_array_equal(self.observation.goal_features, repeat.goal_features)
        np.testing.assert_array_equal(
            self.observation.terrain_channels, repeat.terrain_channels,
        )
        np.testing.assert_array_equal(
            self.observation.hazard_channels, repeat.hazard_channels,
        )
        np.testing.assert_array_equal(
            self.observation.validity_channels, repeat.validity_channels,
        )
        self.assertEqual(self.observation.metadata, repeat.metadata)

    def test_fixed_shapes_and_tensor_conversion(self) -> None:
        expected = {
            "scalar": (len(EGO_FEATURE_NAMES) + len(GOAL_FEATURE_NAMES),),
            "spatial": (
                len(TERRAIN_CHANNEL_NAMES) + len(HAZARD_CHANNEL_NAMES)
                + len(VALIDITY_CHANNEL_NAMES), 21, 21,
            ),
        }
        self.assertEqual(self.builder.tensor_ready_shapes, expected)
        # The builder allocates every spatial component from the fixed config
        # shape.  Exercise states spanning all coordinate axes and the exact BR.
        reachable = np.flatnonzero(self.scene.graph.node_mask)
        sampled = np.unique(np.concatenate((
            reachable[np.linspace(0, len(reachable) - 1, 12, dtype=int)],
            np.asarray(self.oracle.glide_state_ids, dtype=int),
        )))
        for state_id in sampled:
            observation = self.builder.build(int(state_id))
            tensor = self.builder.to_tensor_ready(observation)
            self.assertEqual(tensor.scalar.shape, expected["scalar"])
            self.assertEqual(tensor.spatial.shape, expected["spatial"])
            self.assertEqual(tensor.scalar.dtype, np.float32)
            self.assertEqual(tensor.spatial.dtype, np.float32)

    def test_no_terrain_or_scenario_identifier_leaks_to_neural_input(self) -> None:
        forbidden_metadata = {
            "terrain_id", "terrain_name", "terrain_filename", "terrain_category",
            "scenario_id", "scenario_name", "defender_id", "r_neighbor",
            "spatial_resolution_m", "heading_spacing_deg",
        }
        self.assertTrue(forbidden_metadata.isdisjoint(self.observation.metadata))
        tensor = self.builder.to_tensor_ready(self.observation)
        self.assertTrue(np.issubdtype(tensor.scalar.dtype, np.number))
        self.assertTrue(np.issubdtype(tensor.spatial.dtype, np.number))
        all_names = (
            EGO_FEATURE_NAMES + GOAL_FEATURE_NAMES + TERRAIN_CHANNEL_NAMES
            + HAZARD_CHANNEL_NAMES + VALIDITY_CHANNEL_NAMES
        )
        for forbidden in ("terrain_id", "scenario_id", "r_neighbor", "dx", "dpsi"):
            self.assertNotIn(forbidden, all_names)

    def test_ego_and_goal_features_match_authoritative_state(self) -> None:
        state = self.problem.state(self.state_id)
        position = self.problem.position_map(self.state_id)
        heading = self.problem.grid.heading_rad(state.heading_bin)
        expected_ego = np.asarray([
            self.problem.scene.config.physical_scale.distance_m(
                position[2] - self.scene.terrain.ground_z,
            ),
            self.builder.terrain_surface_height_map(position[0], position[1]),
            np.sin(heading),
            np.cos(heading),
        ])
        expected_ego[1] = self.problem.scene.config.physical_scale.distance_m(
            position[2] - expected_ego[1],
        )
        np.testing.assert_allclose(self.observation.ego_features, expected_ego)

        goal = self.scene.config.goal.as_array()
        delta_m = self.scene.config.physical_scale.position_m(goal - position)
        forward = np.asarray([np.cos(heading), np.sin(heading)])
        right = np.asarray([np.sin(heading), -np.cos(heading)])
        expected_goal = np.asarray([
            np.dot(delta_m[:2], forward),
            np.dot(delta_m[:2], right),
            delta_m[2],
            np.linalg.norm(delta_m) - self.scene.config.glider.goal_tolerance_m,
        ])
        np.testing.assert_allclose(self.observation.goal_features, expected_goal)

    def test_terrain_samples_match_authoritative_ray_query(self) -> None:
        positions = self.builder.sample_positions_map(self.state_id)
        validity = self.observation.validity_channels[0].astype(bool)
        origin_z = max(
            self.problem.grid.maximum_altitude_map,
            self.scene.terrain.maximum_height,
        ) + 1.0
        checked = 0
        for row, column in ((10, 10), (5, 10), (10, 5), (15, 15)):
            if not validity[row, column]:
                continue
            origin = np.asarray([
                positions[row, column, 0], positions[row, column, 1], origin_z,
            ])
            hit = self.scene.terrain.first_ray_hit(origin, np.asarray([0.0, 0.0, -1.0]))
            self.assertIsNotNone(hit)
            expected = self.scene.config.physical_scale.distance_m(
                positions[row, column, 2] - hit.point[2],
            )
            self.assertAlmostEqual(
                self.observation.terrain_channels[0, row, column], expected,
                delta=1.0e-10,
            )
            checked += 1
        self.assertGreater(checked, 0)

    def test_hazard_sufficient_statistics_reconstruct_authoritative_rate(self) -> None:
        edge = self.problem.successors(self.state_id)[0]
        source = self.problem.position_map(self.state_id)
        target = self.problem.position_map(edge.target_id)
        velocity = self.scene.config.physical_scale.position_m(target - source) / edge.duration_s
        positions = self.builder.sample_positions_map(self.state_id)
        for row, column in ((10, 10), (8, 10), (10, 8)):
            if not self.observation.validity_channels[0, row, column]:
                continue
            actual = self.problem.mdp.hazard_field.evaluate_rate(
                positions[row, column], velocity, 0.0,
            ).total_rate_per_s
            reconstructed = self.builder.reconstruct_hazard_rate(
                self.observation, row, column, velocity,
            )
            self.assertAlmostEqual(actual, reconstructed, delta=1.0e-14)

    def test_boundary_mask_distinguishes_padding_from_valid_zero(self) -> None:
        reachable = np.flatnonzero(self.scene.graph.node_mask)
        boundary_id = min(
            (int(value) for value in reachable),
            key=lambda value: self.problem.position_map(value)[0],
        )
        observation = self.builder.build(boundary_id)
        mask = observation.validity_channels[0].astype(bool)
        self.assertTrue(np.any(mask))
        self.assertTrue(np.any(~mask))
        self.assertTrue(np.all(np.isnan(observation.terrain_channels[:, ~mask])))
        self.assertTrue(np.all(np.isnan(observation.hazard_channels[:, ~mask])))
        tensor = self.builder.to_tensor_ready(observation)
        self.assertTrue(np.all(tensor.spatial[:-1, ~mask] == 0.0))
        self.assertTrue(np.all(tensor.spatial[-1, ~mask] == 0.0))

    def test_heading_sine_cosine_is_continuous_across_wrap(self) -> None:
        base = self.problem.state(self.state_id)
        state_type = type(base)
        angles = np.asarray([
            self.problem.grid.heading_rad(index)
            for index in range(self.problem.grid.heading_bin_count)
        ])
        wrap_index = int(np.argmax(np.abs(np.diff(angles))))
        first = self.problem.state_id(state_type(
            base.x_index, base.y_index, base.altitude_index, wrap_index,
        ))
        last = self.problem.state_id(state_type(
            base.x_index, base.y_index, base.altitude_index,
            wrap_index + 1,
        ))
        encoded_first = self.builder.build(first).ego_features[2:]
        encoded_last = self.builder.build(last).ego_features[2:]
        circular_gap = abs(np.arctan2(
            np.sin(angles[wrap_index + 1] - angles[wrap_index]),
            np.cos(angles[wrap_index + 1] - angles[wrap_index]),
        ))
        expected_gap = 2.0 * np.sin(0.5 * circular_gap)
        self.assertAlmostEqual(
            np.linalg.norm(encoded_first - encoded_last), expected_gap, delta=1.0e-12,
        )
        self.assertLess(np.linalg.norm(encoded_first - encoded_last), 0.2)

    def test_observation_construction_preserves_phase1_behavior(self) -> None:
        trajectory = tuple(int(value) for value in self.oracle.glide_state_ids)
        transitions_before = tuple(self.problem.successors(self.state_id))
        terminal_before = tuple(self.problem.is_terminal(value) for value in trajectory)
        objective_before = self.problem.attacker_objective(
            int(self.oracle.switching_state_id), trajectory,
        )
        self.builder.build(self.state_id)
        self.builder.to_tensor_ready(self.builder.build(trajectory[-1]))
        self.assertEqual(transitions_before, tuple(self.problem.successors(self.state_id)))
        self.assertEqual(
            terminal_before, tuple(self.problem.is_terminal(value) for value in trajectory),
        )
        self.assertEqual(
            objective_before,
            self.problem.attacker_objective(int(self.oracle.switching_state_id), trajectory),
        )

    def test_config_json_round_trip_reproduces_observation(self) -> None:
        serialized = json.loads(json.dumps(self.config.as_dict()))
        restored = ObservationConfig.from_dict(serialized)
        self.assertEqual(restored, self.config)
        reproduced = TerrainObservationBuilder(self.problem, restored).build(self.state_id)
        np.testing.assert_array_equal(
            reproduced.terrain_channels, self.observation.terrain_channels,
        )
        np.testing.assert_array_equal(
            reproduced.hazard_channels, self.observation.hazard_channels,
        )


if __name__ == "__main__":
    unittest.main()

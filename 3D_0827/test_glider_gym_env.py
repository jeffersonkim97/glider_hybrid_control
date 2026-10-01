"""Contract tests for the terrain-conditioned Gymnasium adapter."""

from __future__ import annotations

import unittest

import numpy as np

from P1b_RL_approximation import AttackerMDP, default_sensor, switching_candidates
from P1b_condition import ComputationCondition, build_scene
from glider_gym_env import TerrainGlideEnv


class TerrainGlideEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = build_scene(ComputationCondition(spatial_resolution_m=100.0))
        cls.sensor = default_sensor(cls.scene)
        cls.mdp = AttackerMDP(cls.scene, cls.sensor)
        cls.starts = switching_candidates(cls.scene, cls.sensor).state_ids
        if not cls.starts:
            raise unittest.SkipTest("coarse scene has no switching candidates")

    def make_env(self) -> TerrainGlideEnv:
        return TerrainGlideEnv(self.mdp, self.starts, max_steps=50)

    def test_reset_returns_terrain_context_state_and_mask(self) -> None:
        env = self.make_env()
        observation, info = env.reset(seed=7)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(
            observation["terrain"].shape,
            (1, self.scene.grid.y_count, self.scene.grid.x_count),
        )
        self.assertGreater(float(observation["terrain"].max()), 0.0)
        self.assertEqual(observation["state"].shape, (13,))
        np.testing.assert_array_equal(observation["action_mask"], info["action_mask"])

    def test_action_ids_are_target_heading_bins(self) -> None:
        env = self.make_env()
        _, info = env.reset(options={"start_state_id": int(self.starts[0])})
        state_id = int(info["state_id"])
        expected = np.zeros(self.scene.grid.heading_bin_count, dtype=np.int8)
        for edge in self.mdp.actions(state_id):
            expected[int(edge.target_state.heading_bin)] = 1
        np.testing.assert_array_equal(env.action_mask(), expected)

    def test_valid_step_matches_shared_mdp_transition_and_cost(self) -> None:
        env = self.make_env()
        _, info = env.reset(options={"start_state_id": int(self.starts[0])})
        state_id = int(info["state_id"])
        local_index = 0
        edge = self.mdp.actions(state_id)[local_index]
        action_id = int(edge.target_state.heading_bin)
        expected_target, expected_cost, expected_terminal = self.mdp.step(
            state_id, local_index
        )

        observation, reward, terminated, truncated, step_info = env.step(action_id)

        self.assertEqual(step_info["state_id"], expected_target)
        self.assertAlmostEqual(reward, -expected_cost, delta=1.0e-12)
        self.assertEqual(step_info["step_cost"], expected_cost)
        self.assertEqual(step_info["reached_goal"], expected_terminal)
        self.assertFalse(truncated)
        self.assertEqual(terminated, bool(expected_terminal) or not observation["action_mask"].any())

    def test_invalid_primitive_is_visible_failure_not_silent_remap(self) -> None:
        env = self.make_env()
        for state_id in self.starts:
            observation, _ = env.reset(options={"start_state_id": int(state_id)})
            invalid = np.flatnonzero(observation["action_mask"] == 0)
            if len(invalid):
                _, reward, terminated, truncated, info = env.step(int(invalid[0]))
                self.assertEqual(reward, -env.invalid_action_cost)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertTrue(info["invalid_action"])
                self.assertEqual(info["termination_reason"], "invalid_action")
                return
        self.skipTest("all primitives happen to be feasible at every candidate start")


if __name__ == "__main__":
    unittest.main()

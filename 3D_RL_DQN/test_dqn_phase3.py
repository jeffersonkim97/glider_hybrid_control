"""Phase 16.3 tests for fixed actions, masking, DQN, replay, and checkpoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import project_paths  # noqa: F401
from P1b_Exact_Local_SSE import exact_best_response
from P1b_RL_approximation import default_sensor
from P1b_condition import ComputationCondition, build_scene
from attacker_br_problem import AttackerBRProblem
from dqn_action_catalog import HeadingActionCatalog
from dqn_env import DQNAttackerEnv, ObservationTensorCache
from dqn_model import (
    DQNNetworkConfig,
    TerrainDQN,
    masked_bootstrap_values,
    masked_greedy_actions,
)
from dqn_replay import ReplayBuffer
from dqn_training import (
    CompatibilitySignature,
    DQNTrainingConfig,
    checkpoint_payload,
    epsilon_greedy_action,
    evaluate_policy,
    load_compatible_checkpoint,
    save_checkpoint,
    train_one_seed,
)
from terrain_observation import ObservationConfig, TerrainObservationBuilder


class Phase3DQNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = build_scene(ComputationCondition(
            spatial_resolution_m=100.0, heading_spacing_deg=5.0,
        ))
        cls.problem = AttackerBRProblem(cls.scene, default_sensor(cls.scene))
        cls.oracle = exact_best_response(
            cls.scene, cls.problem.sensor_map, keep_solution=True,
        )
        if not cls.oracle.feasible:
            raise unittest.SkipTest("canonical exact response is infeasible")
        cls.builder = TerrainObservationBuilder(
            cls.problem,
            ObservationConfig(
                local_window_extent_m=2000.0,
                local_window_shape_cells=(21, 21),
                local_window_orientation="heading_aligned",
            ),
        )
        cls.catalog = HeadingActionCatalog(cls.problem)
        cls.starts = cls.problem.switching_state_ids()
        cls.network_config = DQNNetworkConfig()
        cls.signature = CompatibilitySignature.from_components(
            cls.problem, cls.builder, cls.catalog,
        )

    def test_action_catalog_is_deterministic(self) -> None:
        repeat = HeadingActionCatalog(self.problem)
        self.assertEqual(self.catalog.as_dict(), repeat.as_dict())
        self.assertEqual(self.catalog.action_count, 72)

    def test_action_ids_match_authoritative_transition_semantics(self) -> None:
        reachable = np.flatnonzero(self.scene.graph.node_mask)
        sampled = reachable[np.linspace(0, len(reachable) - 1, 64, dtype=int)]
        checked = 0
        for raw_state_id in sampled:
            state_id = int(raw_state_id)
            for action_id, local_index in self.catalog.edge_map(state_id).items():
                fixed = self.catalog.transition(state_id, action_id)
                phase1 = self.problem.transition(state_id, local_index)
                self.assertEqual(fixed, phase1)
                self.assertEqual(fixed.edge.target_state.heading_bin, action_id)
                checked += 1
        self.assertGreater(checked, 0)

    def test_masking_excludes_infeasible_actions_everywhere(self) -> None:
        state_id = int(self.oracle.glide_state_ids[0])
        mask = self.catalog.feasible_mask(state_id)
        invalid = np.flatnonzero(~mask)
        self.assertGreater(len(invalid), 0)
        q = np.zeros(self.catalog.action_count, dtype=np.float32)
        q[invalid[0]] = 1.0e9
        greedy = epsilon_greedy_action(q, mask, 0.0, np.random.default_rng(0))
        self.assertTrue(mask[greedy])
        rng = np.random.default_rng(1)
        for _ in range(100):
            exploratory = epsilon_greedy_action(q, mask, 1.0, rng)
            self.assertTrue(mask[exploratory])

        batched_q = torch.from_numpy(q[None, :])
        batched_mask = torch.from_numpy(mask[None, :])
        selected = masked_greedy_actions(batched_q, batched_mask)
        self.assertTrue(mask[int(selected.item())])
        bootstrap = masked_bootstrap_values(
            batched_q, batched_mask, torch.tensor([False]),
        )
        self.assertEqual(float(bootstrap.item()), 0.0)

    def test_terminal_rows_do_not_bootstrap(self) -> None:
        q = torch.full((1, 72), 123.0)
        mask = torch.zeros((1, 72), dtype=torch.bool)
        actual = masked_bootstrap_values(q, mask, torch.tensor([True]))
        self.assertEqual(float(actual.item()), 0.0)

    def test_environment_observation_reward_and_terminal_match_phase1(self) -> None:
        start_id = int(self.oracle.glide_state_ids[0])
        env = DQNAttackerEnv(
            self.problem, self.builder, (start_id,), action_catalog=self.catalog,
        )
        observation, info = env.reset(seed=0, options={"start_state_id": start_id})
        self.assertEqual(observation["scalar"].shape, (8,))
        self.assertEqual(observation["spatial"].shape, (7, 21, 21))
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["action_mask"].shape, (72,))

        target_id = int(self.oracle.glide_state_ids[1])
        edge = next(
            edge for edge in self.problem.successors(start_id)
            if int(edge.target_id) == target_id
        )
        action_id = int(edge.target_state.heading_bin)
        expected = self.catalog.transition(start_id, action_id)
        _, reward, terminated, truncated, next_info = env.step(action_id)
        self.assertAlmostEqual(reward, -expected.stage_cost, delta=1.0e-12)
        self.assertEqual(next_info["state_id"], target_id)
        self.assertEqual(terminated, expected.terminal)
        self.assertFalse(truncated)

        invalid = int(np.flatnonzero(~next_info["action_mask"])[0])
        with self.assertRaises(ValueError):
            env.step(invalid)

    def test_reward_return_is_exact_negative_authoritative_cost(self) -> None:
        trajectory = tuple(int(value) for value in self.oracle.glide_state_ids)
        rewards: list[float] = []
        for source, target in zip(trajectory, trajectory[1:]):
            edge = next(
                edge for edge in self.problem.successors(source)
                if int(edge.target_id) == target
            )
            transition = self.catalog.transition(
                source, int(edge.target_state.heading_bin),
            )
            rewards.append(-transition.stage_cost)
        scored = self.problem.evaluate_trajectory(trajectory, reached_goal=True)
        powered = self.problem.powered_cost(int(self.oracle.switching_state_id))
        self.assertAlmostEqual(sum(rewards), -scored.cost, delta=1.0e-12)
        self.assertAlmostEqual(
            -powered + sum(rewards), -self.oracle.attacker_objective,
            delta=1.0e-12,
        )

    def test_network_consumes_phase2_shapes_and_outputs_72_q_values(self) -> None:
        network = TerrainDQN(self.network_config)
        output = network(
            torch.zeros(3, 7, 21, 21), torch.zeros(3, 8),
        )
        self.assertEqual(tuple(output.shape), (3, 72))
        self.assertGreater(network.parameter_count, 0)

    def test_replay_handles_reconstruct_observations_and_preserve_masks(self) -> None:
        source = int(self.oracle.glide_state_ids[0])
        target = int(self.oracle.glide_state_ids[1])
        edge = next(
            edge for edge in self.problem.successors(source)
            if int(edge.target_id) == target
        )
        action_id = int(edge.target_state.heading_bin)
        transition = self.catalog.transition(source, action_id)
        mask = self.catalog.feasible_mask(target)
        replay = ReplayBuffer(4, 72, seed=0)
        replay.add(
            state_id=source, action_id=action_id,
            reward=-transition.stage_cost, next_state_id=target,
            terminal=transition.terminal, next_feasible_mask=mask,
        )
        batch = replay.sample(1)
        self.assertEqual(int(batch.state_ids[0]), source)
        self.assertEqual(int(batch.next_state_ids[0]), target)
        np.testing.assert_array_equal(batch.next_feasible_masks[0], mask)
        cache = ObservationTensorCache(self.builder)
        self.assertEqual(cache.get(source).scalar.shape, (8,))
        self.assertEqual(cache.get(target).spatial.shape, (7, 21, 21))

    def test_observation_cache_enforces_lru_memory_bound(self) -> None:
        cache = ObservationTensorCache(self.builder, max_entries=1)
        first_state, second_state = map(int, self.oracle.glide_state_ids[:2])
        expected = cache.get(first_state).scalar.copy()
        cache.get(second_state)
        self.assertEqual(cache.cached_states, 1)
        np.testing.assert_array_equal(cache.get(first_state).scalar, expected)
        self.assertEqual(cache.cached_states, 1)

    def test_checkpoint_rejects_incompatible_signature(self) -> None:
        network = TerrainDQN(self.network_config)
        config = DQNTrainingConfig(
            episodes=2, replay_capacity=4, replay_warmup=2, batch_size=2,
            target_update_steps=2, evaluation_interval_episodes=1,
            seeds=(0,), device="cpu",
        )
        payload = checkpoint_payload(
            model_state={key: value.detach().clone() for key, value in network.state_dict().items()},
            network_config=self.network_config,
            training_config=config,
            compatibility=self.signature,
            seed=0, episode=2, checkpoint_kind="test",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(path, payload)
            loaded, _ = load_compatible_checkpoint(
                path, self.network_config, self.signature,
            )
            self.assertEqual(loaded.parameter_count, network.parameter_count)
            incompatible = replace(self.signature, spatial_discretization_m=50.0)
            with self.assertRaisesRegex(ValueError, "signature"):
                load_compatible_checkpoint(path, self.network_config, incompatible)

    def test_evaluation_is_deterministic_and_does_not_change_weights(self) -> None:
        torch.manual_seed(0)
        network = TerrainDQN(self.network_config)
        network.train()
        before = {
            key: value.detach().clone() for key, value in network.state_dict().items()
        }
        cache = ObservationTensorCache(self.builder)
        first = evaluate_policy(
            network, self.problem, self.catalog, cache, self.starts,
            torch.device("cpu"),
        )
        second = evaluate_policy(
            network, self.problem, self.catalog, cache, self.starts,
            torch.device("cpu"),
        )
        self.assertEqual(first.trajectory, second.trajectory)
        self.assertEqual(first.attacker_objective, second.attacker_objective)
        self.assertTrue(network.training)
        for key, value in network.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_short_cpu_training_smoke(self) -> None:
        config = DQNTrainingConfig(
            episodes=2,
            replay_capacity=16,
            replay_warmup=2,
            batch_size=2,
            target_update_steps=2,
            evaluation_interval_episodes=2,
            seeds=(7,),
            device="cpu",
        )
        run = train_one_seed(
            self.problem, self.builder, self.catalog, self.starts,
            float(self.oracle.attacker_objective), self.network_config,
            config, 7,
        )
        self.assertEqual(run.seed, 7)
        self.assertGreater(run.environment_steps, 0)
        self.assertGreater(run.optimization_steps, 0)
        self.assertEqual(len(run.history.episode), 2)


if __name__ == "__main__":
    unittest.main()

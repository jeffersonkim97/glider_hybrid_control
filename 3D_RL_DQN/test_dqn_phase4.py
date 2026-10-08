"""Phase 16.4 dataset, replay, and multi-terrain contract tests.

These tests are defined here for later execution.  Creating this file does not
run a Phase 16.4 training job.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import torch

import project_paths
from P1b_condition import ComputationCondition, build_scene
from aabb_gpu_observation import ExactBatchedAABBGPUObservationBuilder
from attacker_br_problem import AttackerBRProblem
from dqn_action_catalog import HeadingActionCatalog, SharedActionRowCache
from dqn_env import ObservationTensorCache
from dqn_model import DQNNetworkConfig, TerrainDQN
from dqn_training import evaluate_policy, seed_everything
from phase4_dataset import (
    RESERVED_PHASE5_TEST,
    TRAIN_SIMPLE,
    VALIDATION_SIMPLE,
    approved_dataset_manifest,
    approved_phase4_splits,
    common_sensor_is_outside_terrain,
    terrain_definition_hash,
    validate_split_integrity,
)
from phase4_multi_terrain import (
    BellmanReference,
    ScenarioRuntime,
    approved_phase4_execution_config,
    approved_phase4_training_config,
    evaluate_policy_batched,
)
from phase4_replay import MultiScenarioReplayBuffer
from terrain_observation import ObservationConfig, TerrainObservationBuilder


class Phase4DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.splits = approved_phase4_splits()
        self.manifest = approved_dataset_manifest()

    def test_approved_split_counts_and_reserved_test_is_empty(self) -> None:
        self.assertEqual(len(self.splits[TRAIN_SIMPLE]), 27)
        self.assertEqual(len(self.splits[VALIDATION_SIMPLE]), 18)
        self.assertEqual(self.splits[RESERVED_PHASE5_TEST], ())

    def test_train_validation_terrain_families_are_disjoint(self) -> None:
        training = {item.terrain_category for item in self.splits[TRAIN_SIMPLE]}
        validation = {
            item.terrain_category for item in self.splits[VALIDATION_SIMPLE]
        }
        self.assertTrue(training.isdisjoint(validation))

    def test_scenario_identity_does_not_leak_across_splits(self) -> None:
        validate_split_integrity(self.splits)
        training = {item.scenario_id for item in self.splits[TRAIN_SIMPLE]}
        validation = {item.scenario_id for item in self.splits[VALIDATION_SIMPLE]}
        self.assertTrue(training.isdisjoint(validation))

    def test_all_sensors_are_outside_terrain_solids(self) -> None:
        for split in (TRAIN_SIMPLE, VALIDATION_SIMPLE):
            for spec in self.splits[split]:
                with self.subTest(scenario=spec.scenario_id):
                    self.assertTrue(common_sensor_is_outside_terrain(spec))

    def test_terrain_hashes_are_deterministic(self) -> None:
        for split in (TRAIN_SIMPLE, VALIDATION_SIMPLE):
            for spec in self.splits[split]:
                self.assertEqual(
                    spec.terrain_hash,
                    terrain_definition_hash(spec.terrain_category),
                )

    def test_manifest_contains_no_neural_terrain_or_scenario_feature(self) -> None:
        policy = self.manifest["sensor_distribution"]["neural_input_policy"]
        self.assertIn("metadata only", policy)
        observation_source = (
            project_paths.DQN_DIR / "terrain_observation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('"terrain_id"', observation_source)
        self.assertNotIn('"scenario_id"', observation_source)

    def test_official_training_budget_and_cuda_request_are_frozen(self) -> None:
        config = approved_phase4_training_config()
        execution = approved_phase4_execution_config()
        self.assertEqual(config.episodes, 60_000)
        self.assertEqual(config.evaluation_interval_episodes, 1_500)
        self.assertEqual(config.seeds, (0, 1, 2))
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.gamma, 1.0)
        self.assertEqual(execution.episode_batch_size, 8)
        self.assertEqual(execution.optimization_prefetch_batches, 8)
        self.assertEqual(execution.optimizer_updates_per_vector_step, 1)
        self.assertEqual(execution.checkpoint_interval_episodes, 1_500)
        self.assertEqual(execution.evaluation_interval_episodes, 6_000)
        self.assertEqual(
            execution.as_dict()["maximum_optimizer_updates_per_environment_step"],
            0.125,
        )

    def test_master_notebook_exposes_all_four_phase_switches(self) -> None:
        notebook_path = project_paths.DQN_DIR / "Phase16_DQN_Terrain_Generalization.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for phase in ("16_1", "16_2", "16_3", "16_4"):
            self.assertIn(f"RUN_PHASE_{phase} = False", source)
        self.assertIn("REQUIRE_CUDA_FOR_TRAINING = True", source)
        self.assertIn("RESUME_PHASE_16_4 = True", source)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA parity test")
    def test_exact_batched_aabb_cuda_observations_match_cpu_reference(self) -> None:
        representative = {}
        for split in (TRAIN_SIMPLE, VALIDATION_SIMPLE):
            for spec in self.splits[split]:
                representative.setdefault(spec.terrain_category, spec)
        for terrain_category, spec in representative.items():
            with self.subTest(terrain=terrain_category):
                scene = build_scene(ComputationCondition(
                    spatial_resolution_m=100.0,
                    heading_spacing_deg=5.0,
                    terrain_category=terrain_category,
                ))
                problem = AttackerBRProblem(scene, spec.sensor_map)
                builder = TerrainObservationBuilder(problem, ObservationConfig())
                gpu_builder = ExactBatchedAABBGPUObservationBuilder(builder)
                starts = problem.switching_state_ids()
                indices = np.linspace(
                    0, len(starts) - 1, min(12, len(starts)), dtype=int,
                )
                state_ids = tuple(int(starts[index]) for index in indices)
                reference = [
                    builder.to_tensor_ready(builder.build(state_id))
                    for state_id in state_ids
                ]
                accelerated = gpu_builder.build_many(state_ids)
                for expected, actual in zip(reference, accelerated):
                    np.testing.assert_array_equal(actual.scalar, expected.scalar)
                    np.testing.assert_array_equal(actual.spatial, expected.spatial)

                rng = np.random.default_rng(20261005)
                starts = np.column_stack((
                    rng.uniform(scene.grid.bounds.x_min, scene.grid.bounds.x_max, 128),
                    rng.uniform(scene.grid.bounds.y_min, scene.grid.bounds.y_max, 128),
                    rng.uniform(0.0, scene.grid.maximum_altitude_map, 128),
                ))
                boundary_segments = []
                for box in scene.terrain.obstacle_boxes():
                    x0, x1 = box.x_limits
                    y0, y1 = box.y_limits
                    middle_y = 0.5 * (y0 + y1)
                    middle_z = 0.5 * (box.base_z + box.top_z)
                    boundary_segments.extend((
                        ([x0, middle_y, middle_z], [x1, middle_y, middle_z]),
                        ([x0, y0, middle_z], [x0, y1, middle_z]),
                        ([x0, middle_y, box.top_z], [x1, middle_y, box.top_z]),
                    ))
                starts = np.vstack((
                    starts,
                    np.asarray([item[0] for item in boundary_segments], dtype=float),
                ))
                ends = np.vstack((
                    np.repeat(np.asarray(spec.sensor_map)[None, :], 128, axis=0),
                    np.asarray([item[1] for item in boundary_segments], dtype=float),
                ))
                expected_intersections = (
                    scene.terrain.segments_intersect_solid_many(starts, ends)
                )
                actual_intersections = gpu_builder._segments_intersect_solid(
                    torch.as_tensor(starts, dtype=torch.float64, device="cuda"),
                    torch.as_tensor(ends, dtype=torch.float64, device="cuda"),
                ).cpu().numpy()
                np.testing.assert_array_equal(
                    actual_intersections, expected_intersections,
                )


class Phase4ReplayTests(unittest.TestCase):
    def test_mixed_scenario_replay_preserves_scenario_qualification(self) -> None:
        replay = MultiScenarioReplayBuffer(capacity=4, action_count=3, seed=2)
        for scenario_index in (0, 1):
            replay.add(
                scenario_index=scenario_index,
                state_id=7,
                action_id=1,
                reward=-0.25,
                next_state_id=8,
                terminal=False,
                next_feasible_mask=np.array([True, False, True]),
            )
        batch = replay.sample(2)
        self.assertEqual(set(batch.scenario_indices.tolist()), {0, 1})
        self.assertTrue(np.all(batch.state_ids == 7))

    def test_replay_resume_round_trip_keeps_masks_and_sampling_state(self) -> None:
        replay = MultiScenarioReplayBuffer(capacity=3, action_count=4, seed=9)
        replay.add(
            scenario_index=2, state_id=10, action_id=3, reward=-1.0,
            next_state_id=11, terminal=True,
            next_feasible_mask=np.zeros(4, dtype=np.bool_),
        )
        restored = MultiScenarioReplayBuffer.from_state_dict(replay.state_dict())
        self.assertEqual(len(restored), 1)
        batch = restored.sample(1)
        self.assertEqual(int(batch.scenario_indices[0]), 2)
        self.assertTrue(bool(batch.terminal[0]))
        np.testing.assert_array_equal(batch.next_feasible_masks[0], np.zeros(4, bool))

    def test_replay_cannot_silently_store_invalid_nonterminal_mask(self) -> None:
        replay = MultiScenarioReplayBuffer(capacity=2, action_count=4, seed=0)
        with self.assertRaises(ValueError):
            replay.add(
                scenario_index=0, state_id=1, action_id=0, reward=0.0,
                next_state_id=2, terminal=False,
                next_feasible_mask=np.zeros(4, dtype=np.bool_),
            )


class Phase4EvaluationTests(unittest.TestCase):
    def test_batched_action_rows_match_authoritative_adjacency(self) -> None:
        spec = approved_phase4_splits()[TRAIN_SIMPLE][0]
        scene = build_scene(ComputationCondition(
            spatial_resolution_m=100.0, heading_spacing_deg=5.0,
            terrain_category=spec.terrain_category,
        ))
        problem = AttackerBRProblem(scene, spec.sensor_map)
        catalog = HeadingActionCatalog(problem)
        reachable = np.flatnonzero(
            scene.graph.node_mask & ~scene.graph.terminal_mask,
        )
        state_ids = tuple(int(value) for value in reachable[
            np.linspace(0, len(reachable) - 1, 64, dtype=int)
        ])
        expected = tuple(tuple(scene.graph.adjacency[state_id]) for state_id in state_ids)
        rows = catalog.rows_many(state_ids)
        for state_id, authoritative, row in zip(state_ids, expected, rows):
            with self.subTest(state_id=state_id):
                self.assertEqual(row.edges, authoritative)
                expected_mask = np.zeros(catalog.action_count, dtype=np.bool_)
                expected_targets = np.full(catalog.action_count, -1, dtype=np.int64)
                for edge in authoritative:
                    action_id = int(edge.target_state.heading_bin)
                    expected_mask[action_id] = True
                    expected_targets[action_id] = int(edge.target_id)
                np.testing.assert_array_equal(row.feasible_mask, expected_mask)
                np.testing.assert_array_equal(row.target_state_ids, expected_targets)

    def test_action_rows_are_shared_across_sensor_scenarios_and_bounded(self) -> None:
        specs = approved_phase4_splits()[TRAIN_SIMPLE]
        first = specs[0]
        second = next(
            spec for spec in specs
            if spec.terrain_category == first.terrain_category
            and spec.sensor_map != first.sensor_map
        )
        scene = build_scene(ComputationCondition(
            spatial_resolution_m=100.0, heading_spacing_deg=5.0,
            terrain_category=first.terrain_category,
        ))
        shared = SharedActionRowCache(max_entries=2)
        first_problem = AttackerBRProblem(scene, first.sensor_map)
        second_problem = AttackerBRProblem(scene, second.sensor_map)
        first_catalog = HeadingActionCatalog(first_problem, row_cache=shared)
        second_catalog = HeadingActionCatalog(second_problem, row_cache=shared)
        state_ids = first_problem.switching_state_ids()[:3]

        first_row = first_catalog.row(state_ids[0])
        second_row = second_catalog.row(state_ids[0])
        self.assertIs(first_row, second_row)
        self.assertIs(second_problem.mdp._actions[state_ids[0]], first_row.edges)
        np.testing.assert_array_equal(
            first_catalog.feasible_mask(state_ids[0]),
            second_catalog.feasible_mask(state_ids[0]),
        )
        first_catalog.rows_many(state_ids)
        self.assertEqual(shared.cached_states, 2)
        self.assertGreater(shared.hits, 0)

    def test_batched_evaluation_matches_existing_serial_contract(self) -> None:
        spec = approved_phase4_splits()[TRAIN_SIMPLE][0]
        scene = build_scene(ComputationCondition(
            spatial_resolution_m=100.0, heading_spacing_deg=5.0,
            terrain_category=spec.terrain_category,
        ))
        problem = AttackerBRProblem(scene, spec.sensor_map)
        builder = TerrainObservationBuilder(problem, ObservationConfig())
        catalog = HeadingActionCatalog(problem)
        starts = problem.switching_state_ids()
        context = ScenarioRuntime(
            spec=spec, problem=problem, builder=builder, catalog=catalog,
            starts=starts, cache=ObservationTensorCache(builder),
            reference=BellmanReference(
                scenario_id=spec.scenario_id, feasible=True,
                attacker_objective=0.0, switching_state_id=None,
                trajectory=(), runtime_sec=0.0, timing={},
            ),
        )
        shapes = builder.tensor_ready_shapes
        config = DQNNetworkConfig(
            spatial_channels=shapes["spatial"][0],
            spatial_height=shapes["spatial"][1],
            spatial_width=shapes["spatial"][2],
            scalar_features=shapes["scalar"][0],
            action_count=catalog.action_count,
        )
        seed_everything(123)
        network = TerrainDQN(config)
        serial = evaluate_policy(
            network, problem, catalog, context.cache, starts,
            torch.device("cpu"),
        )
        context.cache.clear()
        problem.clear_runtime_caches()
        batched = evaluate_policy_batched(
            network, context, torch.device("cpu"), batch_size=32,
        )
        self.assertEqual(serial.success, batched.success)
        self.assertEqual(serial.switching_state_id, batched.switching_state_id)
        self.assertEqual(serial.trajectory, batched.trajectory)
        self.assertEqual(serial.attacker_objective, batched.attacker_objective)
        self.assertEqual(serial.candidate_successes, batched.candidate_successes)


if __name__ == "__main__":
    unittest.main()

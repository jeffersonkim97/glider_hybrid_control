"""Tests for the approved extended Phase 16.4 random-box pilot contracts."""

from __future__ import annotations

import json
import unittest

import numpy as np

from extended_failure_env import (
    ExpandedFailureAttackerBRProblem,
    FailureCostConfig,
    failure_adjusted_reward,
    sampled_powered_segment_is_fully_occluded,
)
from multiscale_observation import (
    MultiscaleObservationConfig,
    MultiscaleTerrainObservationBuilder,
)
from phase4_early_stopping import (
    EarlyStoppingConfig,
    EarlyStoppingTracker,
    ValidationPoint,
    evaluate_final_acceptance,
)
from phase4_g2_dataset import (
    G2_TRAIN,
    G2_VALIDATION,
    G2BenchmarkConfig,
    build_g2_scenario_bank,
    g2_dataset_manifest,
)
from phase4_extended_dataset import (
    TRAIN_EXTENDED,
    VALIDATION_EXTENDED,
    ExtendedPilotConfig,
    build_extended_scenario_bank,
    extended_dataset_manifest,
    specs_from_extended_manifest,
    terrain_specs_from_manifest,
)
from phase4_extended_scene import build_extended_scene
from phase4_v2_run_a import RunAGoalReachableProblem


class ExtendedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ExtendedPilotConfig()
        cls.splits, cls.terrains = build_extended_scenario_bank(cls.config)
        cls.spec = cls.splits[TRAIN_EXTENDED][0]
        cls.scene = build_extended_scene(
            cls.spec,
            cls.terrains[cls.spec.terrain_category],
            spatial_resolution_m=100.0,
        )
        cls.problem = ExpandedFailureAttackerBRProblem(
            cls.scene, cls.spec.sensor_map,
        )
        cls.starts = cls.problem.switching_state_ids()

    def test_bank_counts_visibility_mix_and_split_isolation(self) -> None:
        self.assertEqual(len(self.splits[TRAIN_EXTENDED]), 16)
        self.assertEqual(len(self.splits[VALIDATION_EXTENDED]), 8)
        for split, expected_terrain_count in (
            (TRAIN_EXTENDED, 4), (VALIDATION_EXTENDED, 2),
        ):
            values = self.splits[split]
            counts = {
                label: sum(spec.visibility_class == label for spec in values)
                for label in ("low", "mixed", "high")
            }
            self.assertEqual(
                counts,
                {
                    "low": expected_terrain_count,
                    "mixed": 2 * expected_terrain_count,
                    "high": expected_terrain_count,
                },
            )
        train_hashes = {
            spec.terrain_hash for spec in self.splits[TRAIN_EXTENDED]
        }
        validation_hashes = {
            spec.terrain_hash for spec in self.splits[VALIDATION_EXTENDED]
        }
        self.assertTrue(train_hashes.isdisjoint(validation_hashes))

    def test_manifest_round_trip_preserves_every_scenario_and_terrain(self) -> None:
        manifest = extended_dataset_manifest(self.config)
        serialized = json.loads(json.dumps(manifest))
        restored_terrains = terrain_specs_from_manifest(serialized)
        self.assertEqual(restored_terrains, self.terrains)
        for split in (TRAIN_EXTENDED, VALIDATION_EXTENDED):
            self.assertEqual(
                specs_from_extended_manifest(serialized, split),
                self.splits[split],
            )

    def test_mission_samples_respect_user_radii_and_terminal_contract(self) -> None:
        start_center = np.asarray(self.config.start_center_map)
        goal_center = np.asarray(self.config.goal_center_map)
        for spec in (*self.splits[TRAIN_EXTENDED], *self.splits[VALIDATION_EXTENDED]):
            start_distance = np.linalg.norm(np.asarray(spec.start_map) - start_center)
            goal_distance = np.linalg.norm(np.asarray(spec.goal_map) - goal_center)
            self.assertLessEqual(
                start_distance * self.config.meters_per_map_unit,
                self.config.start_sampling_radius_m + 1.0e-9,
            )
            self.assertLessEqual(
                goal_distance * self.config.meters_per_map_unit,
                self.config.goal_sampling_radius_m + 1.0e-9,
            )
            self.assertEqual(spec.goal_tolerance_m, 50.0)
        self.assertEqual(self.scene.config.glider.goal_tolerance_m, 50.0)
        self.assertGreater(self.scene.graph.statistics.terminal_state_count, 0)

    def test_multiscale_observation_has_approved_geometry_and_no_id_features(self) -> None:
        self.assertTrue(self.starts)
        config = MultiscaleObservationConfig()
        self.assertEqual(config.local_sample_spacing_m, (25.0, 25.0))
        self.assertEqual(config.global_sample_spacing_m, (100.0, 100.0))
        builder = MultiscaleTerrainObservationBuilder(self.problem, config)
        raw = builder.build(self.starts[0])
        tensor = builder.to_tensor_ready(raw)
        self.assertEqual(builder.tensor_ready_shapes["scalar"], (8,))
        self.assertEqual(builder.tensor_ready_shapes["spatial"], (14, 41, 41))
        self.assertEqual(tensor.scalar.shape, (8,))
        self.assertEqual(tensor.spatial.shape, (14, 41, 41))
        self.assertEqual(tensor.scalar.dtype, np.float32)
        self.assertEqual(tensor.spatial.dtype, np.float32)
        np.testing.assert_array_equal(
            self.problem.position_map(self.starts[0]),
            self.problem.position_map(self.starts[0]),
        )

    def test_powered_occlusion_filter_retains_only_certified_candidates(self) -> None:
        self.assertGreater(len(self.starts), 0)
        candidates = self.problem._switching_candidates()
        self.assertGreater(
            candidates.timing["powered_occlusion_candidate_input"],
            candidates.timing["powered_occlusion_candidate_retained"],
        )
        for _, (position, _) in list(candidates.by_state_id.items())[:10]:
            self.assertTrue(sampled_powered_segment_is_fully_occluded(
                self.scene.terrain,
                self.spec.sensor_map,
                self.spec.start_map,
                position,
                meters_per_map_unit=self.config.meters_per_map_unit,
            ))

    def test_expanded_action_space_contains_goal_unreachable_branches(self) -> None:
        expanded = sum(len(self.problem.successors(state)) for state in self.starts[:20])
        goal_only = sum(len(self.problem.mdp.actions(state)) for state in self.starts[:20])
        self.assertGreater(expanded, goal_only)

    def test_run_a_keeps_occluded_starts_but_filters_failure_branches(self) -> None:
        run_a = RunAGoalReachableProblem(self.scene, self.spec.sensor_map)
        self.assertEqual(run_a.switching_state_ids(), self.starts)
        for state_id in self.starts[:20]:
            self.assertEqual(run_a.successors(state_id), run_a.mdp.actions(state_id))
            self.assertTrue(all(
                self.scene.goal_reachable[int(edge.target_id)]
                for edge in run_a.successors(state_id)
            ))


class FailureAndEarlyStoppingTests(unittest.TestCase):
    def test_failure_terminal_penalty_is_worse_than_valid_reference_envelope(self) -> None:
        penalty = FailureCostConfig().from_reference_objectives((0.2, 1.4, 0.8))
        self.assertGreater(penalty, 1.4)
        success_reward = failure_adjusted_reward(
            0.2, reached_goal=True, episode_ended=True,
            failure_terminal_cost=penalty,
        )
        failure_reward = failure_adjusted_reward(
            0.2, reached_goal=False, episode_ended=True,
            failure_terminal_cost=penalty,
        )
        self.assertLess(failure_reward, success_reward)

    def test_approved_early_stop_requires_quality_stability_and_minimum_budget(self) -> None:
        tracker = EarlyStoppingTracker(EarlyStoppingConfig())
        decisions = []
        for episode, median, maximum in (
            (7_500, 0.070, 0.180),
            (9_000, 0.090, 0.190),
            (10_500, 0.100, 0.200),
        ):
            decisions.append(tracker.update(ValidationPoint(
                episode=episode,
                success_rate=1.0,
                median_relative_error=median,
                maximum_relative_error=maximum,
            )))
        self.assertFalse(any(item.stop for item in decisions[:-1]))
        self.assertTrue(decisions[-1].stop)
        self.assertTrue(decisions[-1].success)

    def test_early_stop_allows_one_window_median_excursion(self) -> None:
        tracker = EarlyStoppingTracker(EarlyStoppingConfig())
        decision = None
        for episode, median, maximum in (
            (7_500, 0.0494, 0.710),
            (9_000, 0.1317, 0.730),
            (10_500, 0.0443, 0.380),
        ):
            decision = tracker.update(ValidationPoint(
                episode=episode,
                success_rate=1.0,
                median_relative_error=median,
                maximum_relative_error=maximum,
            ))
        self.assertIsNotNone(decision)
        self.assertTrue(decision.stop)
        self.assertTrue(decision.success)
        self.assertTrue(decision.quality_passed)
        self.assertTrue(decision.stability_passed)

    def test_early_stop_uses_ten_percentage_point_median_range(self) -> None:
        tracker = EarlyStoppingTracker(EarlyStoppingConfig())
        decision = None
        for episode, median in (
            (7_500, 0.001),
            (9_000, 0.102),
            (10_500, 0.100),
        ):
            decision = tracker.update(ValidationPoint(
                episode=episode,
                success_rate=1.0,
                median_relative_error=median,
                maximum_relative_error=0.190,
            ))
        self.assertIsNotNone(decision)
        self.assertFalse(decision.stop)
        self.assertTrue(decision.quality_passed)
        self.assertFalse(decision.stability_passed)

    def test_final_acceptance_uses_median_and_higher_p90_not_maximum(self) -> None:
        accepted = evaluate_final_acceptance(
            success_count=16,
            case_count=16,
            successful_relative_errors=[0.04] * 13 + [0.21, 0.26, 0.38],
        )
        self.assertTrue(accepted.accepted)
        self.assertAlmostEqual(accepted.median_relative_error, 0.04)
        self.assertAlmostEqual(accepted.p90_relative_error, 0.26)
        self.assertAlmostEqual(accepted.maximum_relative_error, 0.38)

        rejected = evaluate_final_acceptance(
            success_count=16,
            case_count=16,
            successful_relative_errors=[0.04] * 13 + [0.31, 0.32, 0.40],
        )
        self.assertFalse(rejected.accepted)

    def test_low_quality_plateau_runs_to_maximum_and_fails(self) -> None:
        tracker = EarlyStoppingTracker(EarlyStoppingConfig())
        decision = None
        for episode in range(1_500, 60_001, 1_500):
            decision = tracker.update(ValidationPoint(
                episode=episode,
                success_rate=1.0,
                median_relative_error=0.30,
                maximum_relative_error=0.60,
            ))
            if episode < 60_000:
                self.assertFalse(decision.stop)
        self.assertIsNotNone(decision)
        self.assertTrue(decision.stop)
        self.assertFalse(decision.success)


class G2BenchmarkBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = G2BenchmarkConfig()
        cls.splits, cls.terrains = build_g2_scenario_bank(cls.config)

    def test_g2_counts_visibility_and_explicit_seed_ranges_are_frozen(self) -> None:
        self.assertEqual(
            self.config.training_resample_attempts,
            (10, 0, 7, 0, 0, 0, 3, 3),
        )
        self.assertEqual(
            self.config.validation_resample_attempts,
            (1, 0, 0, 0),
        )
        self.assertEqual(len(self.splits[G2_TRAIN]), 32)
        self.assertEqual(len(self.splits[G2_VALIDATION]), 16)
        for split, terrain_count, seed_base in (
            (G2_TRAIN, 8, 26_401),
            (G2_VALIDATION, 4, 27_401),
        ):
            values = self.splits[split]
            terrain_ids = {item.terrain_category for item in values}
            self.assertEqual(len(terrain_ids), terrain_count)
            self.assertEqual(
                {
                    self.terrains[terrain_id].generation_seed
                    for terrain_id in terrain_ids
                },
                set(range(seed_base, seed_base + terrain_count)),
            )
            self.assertEqual(
                {
                    label: sum(item.visibility_class == label for item in values)
                    for label in ("low", "mixed", "high")
                },
                {"low": terrain_count, "mixed": 2 * terrain_count,
                 "high": terrain_count},
            )

    def test_g2_is_disjoint_from_g1_and_between_active_splits(self) -> None:
        train_hashes = {
            item.terrain_hash for item in self.splits[G2_TRAIN]
        }
        validation_hashes = {
            item.terrain_hash for item in self.splits[G2_VALIDATION]
        }
        self.assertTrue(train_hashes.isdisjoint(validation_hashes))
        g2_seeds = {item.generation_seed for item in self.terrains.values()}
        self.assertTrue(g2_seeds.isdisjoint(set(range(16_401, 16_407))))

    def test_g2_final_test_is_reserved_but_not_instantiated(self) -> None:
        manifest = g2_dataset_manifest(self.config)
        contract = manifest["sealed_final_test_contract"]
        self.assertFalse(contract["instantiated"])
        self.assertEqual(contract["random_box_id"]["terrain_count"], 4)
        self.assertEqual(contract["random_box_id"]["seed_status"], "not_generated")
        self.assertNotIn("FINAL_TEST_RANDOM_BOX_ID", manifest["splits"])

    def test_g2_manifest_is_deterministic(self) -> None:
        first = g2_dataset_manifest(self.config)
        second = g2_dataset_manifest(self.config)
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

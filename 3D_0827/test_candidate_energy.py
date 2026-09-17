"""Stage-4 Part-B all-candidate energy and reachability tests."""

from __future__ import annotations

import unittest

import numpy as np

from candidate_energy import (
    evaluate_switching_candidate,
    evaluate_switching_candidates,
    switching_candidate_is_acoustically_neutralized,
)
from los_explorer_gui import compute_los_case
from reachability_surface import LOSSurfaceReachabilityClassifier
from switching_candidates import SwitchingCandidate, generate_switching_candidates


EXPECTED_CLASS_COUNTS = {
    "centered_cube": {
        "reachable": 60,
        "unreachable": 12,
        "powered_infeasible": 24,
    },
    "offset_cube_left": {
        "reachable": 71,
        "unreachable": 14,
        "powered_infeasible": 11,
    },
    "offset_cube_right": {
        "reachable": 71,
        "unreachable": 14,
        "powered_infeasible": 11,
    },
    "stepped_pyramid": {
        "reachable": 54,
        "unreachable": 10,
        "powered_infeasible": 32,
    },
}


class CandidateEnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = {}
        for terrain_name in EXPECTED_CLASS_COUNTS:
            los_result = compute_los_case(terrain_name, 5.0, 0.0)
            candidates = generate_switching_candidates(los_result.tangent_contour)
            evaluations = evaluate_switching_candidates(
                candidates,
                los_result.tangent_contour,
                los_result.terrain_map,
                los_result.mission_points,
            )
            cls.cases[terrain_name] = (los_result, candidates, evaluations)

    def test_all_candidates_have_expected_classification_counts(self) -> None:
        for terrain_name, (_, candidates, evaluations) in self.cases.items():
            with self.subTest(terrain=terrain_name):
                counts = {
                    "reachable": sum(result.reachable for result in evaluations),
                    "powered_infeasible": sum(
                        not result.powered_feasible for result in evaluations
                    ),
                }
                counts["unreachable"] = (
                    len(evaluations)
                    - counts["reachable"]
                    - counts["powered_infeasible"]
                )
                self.assertEqual(len(candidates), 96)
                self.assertEqual(counts, EXPECTED_CLASS_COUNTS[terrain_name])
                self.assertEqual(
                    [result.candidate_id for result in evaluations],
                    [candidate.candidate_id for candidate in candidates],
                )

    def test_acoustic_neutralization_rule_is_explicit_and_testable(self) -> None:
        los_result, candidates, evaluations = self.cases["centered_cube"]
        self.assertTrue(all(result.acoustically_neutralized for result in evaluations))
        valid = candidates[54]
        off_surface = SwitchingCandidate(
            candidate_id=999,
            position_map=valid.position_map + np.array([0.01, 0.0, 0.0]),
            contour_fraction=valid.contour_fraction,
            radial_scale=valid.radial_scale,
            surface_residual=0.0,
        )
        self.assertFalse(switching_candidate_is_acoustically_neutralized(
            off_surface,
            los_result.tangent_contour,
        ))
        off_surface_result = evaluate_switching_candidate(
            off_surface,
            los_result.tangent_contour,
            los_result.terrain_map,
            los_result.mission_points,
        )
        self.assertFalse(off_surface_result.reachable)
        self.assertEqual(
            off_surface_result.infeasibility_reason,
            "switching candidate is not on the LOS tangent surface",
        )
        with self.assertRaises(ValueError):
            switching_candidate_is_acoustically_neutralized(
                valid,
                los_result.tangent_contour,
                tolerance=-1.0,
            )

    def test_all_results_contain_no_nan_and_only_intentional_infinities(self) -> None:
        for terrain_name, (_, _, evaluations) in self.cases.items():
            for result in evaluations:
                with self.subTest(terrain=terrain_name, candidate=result.candidate_id):
                    state = result.switching_state
                    glide = result.glide_result
                    self.assertTrue(np.all(np.isfinite(state.position_map)))
                    self.assertTrue(np.all(np.isfinite(state.position_m)))
                    self.assertTrue(np.all(np.isfinite(state.velocity_mps)))
                    self.assertTrue(np.isfinite(state.powered_path_length_m))
                    self.assertTrue(np.isfinite(state.flight_path_angle_rad))
                    self.assertTrue(np.isfinite(state.heading_rad))
                    self.assertTrue(np.isfinite(state.total_mechanical_energy_j))
                    glide_values = (
                        glide.minimum_glide_path_m,
                        glide.turn_arc_length_m,
                        glide.straight_length_m,
                        glide.maximum_glide_path_m,
                        glide.available_energy_j,
                        glide.required_energy_j,
                        glide.energy_margin_j,
                        glide.equivalent_height_margin_m,
                        glide.required_height_m,
                    )
                    self.assertFalse(any(np.isnan(value) for value in glide_values))
                    if result.powered_feasible:
                        self.assertTrue(all(np.isfinite(value) for value in glide_values))
                    else:
                        self.assertTrue(np.isposinf(glide.required_energy_j))
                        self.assertTrue(np.isneginf(glide.energy_margin_j))
                        self.assertTrue(np.isneginf(glide.equivalent_height_margin_m))

    def test_powered_infeasible_cannot_be_glide_reachable(self) -> None:
        for terrain_name, (_, _, evaluations) in self.cases.items():
            for result in evaluations:
                if not result.powered_feasible:
                    with self.subTest(terrain=terrain_name, candidate=result.candidate_id):
                        self.assertFalse(result.glide_result.reachable)
                        self.assertFalse(result.reachable)
                        self.assertIsNotNone(result.infeasibility_reason)

    def test_energy_margin_sign_agrees_with_reachability(self) -> None:
        for terrain_name, (_, _, evaluations) in self.cases.items():
            for result in evaluations:
                if not result.powered_feasible:
                    continue
                with self.subTest(terrain=terrain_name, candidate=result.candidate_id):
                    if result.reachable:
                        self.assertGreaterEqual(result.energy_margin_j, -1.0e-9)
                    else:
                        self.assertLess(result.energy_margin_j, -1.0e-9)

    def test_candidate_results_match_existing_point_classifier(self) -> None:
        reference_classifier = LOSSurfaceReachabilityClassifier()
        for terrain_name, (los_result, candidates, evaluations) in self.cases.items():
            for candidate, evaluation in zip(candidates, evaluations, strict=True):
                reference = reference_classifier.evaluate_point(
                    candidate.position_map,
                    los_result.terrain_map,
                    los_result.mission_points,
                )
                with self.subTest(terrain=terrain_name, candidate=candidate.candidate_id):
                    self.assertEqual(evaluation.powered_feasible, reference.switching_state.powered_feasible)
                    self.assertEqual(evaluation.reachable, reference.reachable)
                    self.assertEqual(
                        evaluation.switching_state.infeasibility_reason,
                        reference.switching_state.infeasibility_reason,
                    )
                    np.testing.assert_allclose(
                        evaluation.switching_state.velocity_mps,
                        reference.switching_state.velocity_mps,
                        rtol=0.0,
                        atol=0.0,
                    )
                    self.assertEqual(
                        evaluation.glide_result.energy_margin_j,
                        reference.glide_result.energy_margin_j,
                    )
                    self.assertEqual(
                        evaluation.glide_result.equivalent_height_margin_m,
                        reference.glide_result.equivalent_height_margin_m,
                    )

    def test_offset_terrain_classifications_are_symmetric(self) -> None:
        left = self.cases["offset_cube_left"][2]
        right = self.cases["offset_cube_right"][2]
        reflected_left = np.vstack([
            result.candidate.position_map * np.array([1.0, -1.0, 1.0])
            for result in left
        ])
        right_positions = np.vstack([
            result.candidate.position_map for result in right
        ])
        distances = np.linalg.norm(
            reflected_left[:, None, :] - right_positions[None, :, :],
            axis=2,
        )
        matching_indices = np.argmin(distances, axis=1)
        self.assertLessEqual(
            float(np.max(distances[np.arange(len(left)), matching_indices])),
            1.0e-6,
        )
        self.assertEqual(len(set(matching_indices.tolist())), len(left))
        for left_result, right_index in zip(left, matching_indices, strict=True):
            right_result = right[int(right_index)]
            self.assertEqual(
                left_result.powered_feasible,
                right_result.powered_feasible,
            )
            self.assertEqual(left_result.reachable, right_result.reachable)

    def test_batch_rejects_duplicate_candidate_ids(self) -> None:
        los_result, candidates, _ = self.cases["centered_cube"]
        with self.assertRaises(ValueError):
            evaluate_switching_candidates(
                (candidates[0], candidates[0]),
                los_result.tangent_contour,
                los_result.terrain_map,
                los_result.mission_points,
            )


if __name__ == "__main__":
    unittest.main()

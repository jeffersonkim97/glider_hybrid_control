"""Phase 7 Bellman candidate filtering and ranking tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from p1b_4D.bellman import generate_bellman_candidates
from p1b_4D.candidate_filtering import candidate_similarity, filter_bellman_candidates
from p1b_4D.candidate_filtering_io import export_filtered_bellman_bundle, import_filtered_bellman_bundle
from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.detection import build_symbolic_detection_bundle
from p1b_4D.geometry import build_geometry_bundle
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.projection import construct_projected_cost_map
from p1b_4D.stage_cost import construct_stage_cost_4d


class CandidateFilteringTests(unittest.TestCase):
    """Verify topology filtering, objective-only ranking, Top-K, and export."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = TemporaryDirectory()
        cls.configuration = build_configuration_bundle(Path(cls.temporary_directory.name))
        cls.geometry = build_geometry_bundle(cls.configuration)
        cls.detection = build_symbolic_detection_bundle(cls.configuration, cls.geometry)
        cls.stage = construct_stage_cost_4d(cls.configuration, cls.geometry, cls.detection)
        cls.projection = construct_projected_cost_map(cls.configuration, cls.geometry, cls.detection, cls.stage)
        cls.bellman = generate_bellman_candidates(cls.configuration, cls.geometry, cls.detection, cls.stage, cls.projection)
        cls.filtered = filter_bellman_candidates(cls.bellman, cls.configuration, cls.configuration["validation"])

    @classmethod
    def tearDownClass(cls) -> None:
        close_phase_logger(cls.configuration["primary_result"]["logging_utilities"]["logger"])
        cls.temporary_directory.cleanup()

    def test_top_k_is_objective_ranked(self) -> None:
        candidates = self.filtered["primary_result"]["candidates"]
        self.assertEqual(len(candidates), self.configuration["primary_result"]["bellman_config"]["top_k"])
        self.assertEqual([item["rank"] for item in candidates], [1, 2, 3])
        self.assertEqual([item["mission_cost"] for item in candidates], sorted(item["mission_cost"] for item in candidates))
        self.assertTrue(self.filtered["metadata"]["only_attacker_nlp_warm_start_source"])

    def test_filtering_does_not_refine_source_arrays(self) -> None:
        source_by_id = {item["candidate_id"]: item for item in self.bellman["primary_result"]["candidates"]}
        for retained in self.filtered["primary_result"]["candidates"]:
            source = source_by_id[retained["candidate_id"]]
            self.assertIs(retained["trajectory"], source["trajectory"])
            self.assertIs(retained["speed_profile"], source["speed_profile"])
            self.assertIs(retained["gamma_profile"], source["gamma_profile"])
            self.assertFalse(retained["metadata"]["trajectory_refinement_applied"])

    def test_duplicate_topology_keeps_best_representative(self) -> None:
        augmented = deepcopy(self.bellman)
        best = augmented["primary_result"]["candidates"][0]
        duplicate = deepcopy(best)
        duplicate["candidate_id"] = "synthetic-higher-cost-duplicate"
        duplicate["mission_cost"] += 0.01
        augmented["primary_result"]["candidates"] += (duplicate,)
        augmented["primary_result"]["candidate_count"] += 1
        result = filter_bellman_candidates(augmented, self.configuration, self.configuration["validation"])
        records = result["primary_result"]["duplicate_records"]
        self.assertTrue(any(record["removed_candidate_id"] == duplicate["candidate_id"] for record in records))
        self.assertEqual(result["primary_result"]["candidates"][0]["candidate_id"], best["candidate_id"])

    def test_similarity_uses_all_configured_criteria(self) -> None:
        candidate = self.bellman["primary_result"]["candidates"][0]
        thresholds = self.configuration["primary_result"]["bellman_config"]["duplicate_threshold"]
        metrics = candidate_similarity(candidate, candidate, thresholds)
        self.assertTrue(metrics["is_duplicate"])
        self.assertEqual(set(metrics["criteria"]), {"switching_point", "trajectory_shape", "mission_cost", "path_length"})

    def test_json_npz_round_trip(self) -> None:
        exported = export_filtered_bellman_bundle(self.filtered, self.configuration)
        imported = import_filtered_bellman_bundle(exported["primary_result"]["json_path"])
        self.assertTrue(imported["status"]["success"])
        arrays = imported["primary_result"]["arrays"]
        np.testing.assert_array_equal(arrays["ranks"], [1, 2, 3])
        np.testing.assert_array_equal(arrays["mission_costs"], [item["mission_cost"] for item in self.filtered["primary_result"]["candidates"]])


if __name__ == "__main__":
    unittest.main()

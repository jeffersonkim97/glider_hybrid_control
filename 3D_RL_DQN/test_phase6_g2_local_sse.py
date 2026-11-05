"""Focused tests for the resumable Phase-6 Local-SSE runner."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from defender_grid_2d import DefenderRegion
from phase6_g2_local_sse import (
    ExactG2Response,
    LEADER_OBJECTIVE_ATTACKER_COST,
    LEADER_OBJECTIVE_DETECTION_PROBABILITY,
    _center_aligned_defender_region,
    _load_exact_response_cache,
    _local_evaluation_from_exact,
    _save_exact_response_cache,
    _search_result_payload,
    _zero_sum_single_point_certificate,
)


class Phase6G2LocalSSETests(unittest.TestCase):
    @staticmethod
    def _response() -> ExactG2Response:
        return ExactG2Response(
            feasible=True,
            sensor_map=(4.5, -1.5, 0.0),
            attacker_objective=0.25,
            detection_probability=0.1,
            switching_state_id=17,
            trajectory=(17, 18, 19),
            candidate_count=9,
            cooptimal_candidate_count=1,
            strong_tie_break_verified=True,
            runtime_sec=3.5,
            active_state_count=100,
            active_edge_count=200,
        )

    def test_exact_response_cache_round_trip_and_contract_guard(self) -> None:
        response = self._response()
        contract = {"terrain_hash": "abc", "dx_m": 10.0}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exact_response_cache.json"
            _save_exact_response_cache(
                path, contract, {7: response}, last_completed_action_id=7,
            )
            self.assertEqual(
                _load_exact_response_cache(path, contract), {7: response},
            )
            self.assertEqual(
                _load_exact_response_cache(
                    path, {**contract, "result_schema": "zero-sum"},
                ),
                {7: response},
            )
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
                _load_exact_response_cache(
                    path, {"terrain_hash": "changed", "dx_m": 10.0},
                )

    def test_leader_objective_routes_only_the_outer_search_value(self) -> None:
        response = self._response()
        legacy = _local_evaluation_from_exact(
            7, response,
            leader_objective=LEADER_OBJECTIVE_DETECTION_PROBABILITY,
        )
        zero_sum = _local_evaluation_from_exact(
            7, response,
            leader_objective=LEADER_OBJECTIVE_ATTACKER_COST,
        )
        self.assertEqual(legacy.attacker_objective, 0.25)
        self.assertEqual(zero_sum.attacker_objective, 0.25)
        self.assertEqual(legacy.defender_value, 0.1)
        self.assertEqual(zero_sum.defender_value, 0.25)

    def test_zero_sum_single_point_certificate_is_one_sided(self) -> None:
        certificate = _zero_sum_single_point_certificate(
            approximate_J=0.27, exact_J=0.25,
        )
        self.assertAlmostEqual(certificate["certified_delta"], 0.02)
        self.assertTrue(certificate["one_sided_inequality_verified"])
        with self.assertRaisesRegex(RuntimeError, "one-sided inequality"):
            _zero_sum_single_point_certificate(
                approximate_J=0.24, exact_J=0.25,
            )

    def test_zero_sum_payload_does_not_mislabel_leader_value_as_J_D(self) -> None:
        search = SimpleNamespace(
            as_dict=lambda: {"final_J_A": 0.27, "final_J_D": 0.27},
        )
        payload = _search_result_payload(
            search, LEADER_OBJECTIVE_ATTACKER_COST,
        )
        self.assertEqual(payload["final_leader_value"], 0.27)
        self.assertEqual(payload["final_J_A"], 0.27)
        self.assertNotIn("final_J_D", payload)

    def test_100m_region_is_center_aligned(self) -> None:
        region = DefenderRegion(
            x_min=3.25, x_max=6.75, y_min=-3.75, y_max=3.75,
        )
        aligned = _center_aligned_defender_region(region, spacing=1.0)
        self.assertEqual(aligned.as_dict(), {
            "x_min": 3.5,
            "x_max": 6.5,
            "y_min": -3.5,
            "y_max": 3.5,
            "z": 0.0,
        })


if __name__ == "__main__":
    unittest.main()

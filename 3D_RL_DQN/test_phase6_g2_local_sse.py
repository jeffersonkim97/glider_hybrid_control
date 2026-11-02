"""Focused tests for the resumable Phase-6 Local-SSE runner."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from defender_grid_2d import DefenderRegion
from phase6_g2_local_sse import (
    ExactG2Response,
    _center_aligned_defender_region,
    _load_exact_response_cache,
    _save_exact_response_cache,
)


class Phase6G2LocalSSETests(unittest.TestCase):
    def test_exact_response_cache_round_trip_and_contract_guard(self) -> None:
        response = ExactG2Response(
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
        contract = {"terrain_hash": "abc", "dx_m": 10.0}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exact_response_cache.json"
            _save_exact_response_cache(
                path, contract, {7: response}, last_completed_action_id=7,
            )
            self.assertEqual(
                _load_exact_response_cache(path, contract), {7: response},
            )
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
                _load_exact_response_cache(
                    path, {"terrain_hash": "changed", "dx_m": 10.0},
                )

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

"""Fast Stage-14.7 scaling-analysis unit tests."""

from __future__ import annotations

import unittest
from collections import Counter

from stage14_7_scaling_analysis import (
    compute_fits,
    crosscheck_raw_and_summaries,
    fit_power_law,
    load_and_normalize,
)


class Stage147ScalingAnalysisTests(unittest.TestCase):
    def test_exact_synthetic_power_law(self) -> None:
        rows = [{"x": x, "y": 2.0 * x**3} for x in (1.0, 2.0, 4.0, 8.0)]
        fit = fit_power_law(
            rows, fit_id="synthetic", x_field="x", y_field="y",
            sweep="synthetic", algorithm_variant="unit",
        )
        self.assertEqual(fit["status"], "completed")
        self.assertAlmostEqual(fit["c"], 2.0, places=12)
        self.assertAlmostEqual(fit["alpha"], 3.0, places=12)
        self.assertAlmostEqual(fit["r_squared"], 1.0, places=12)
        self.assertFalse(fit["formal_big_o_claim"])
        self.assertFalse(fit["extrapolation_performed"])

    def test_insufficient_points_are_not_fitted(self) -> None:
        fit = fit_power_law(
            [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}],
            fit_id="short", x_field="x", y_field="y",
            sweep="synthetic", algorithm_variant="unit",
        )
        self.assertEqual(fit["status"], "insufficient_completed_points")
        self.assertIsNone(fit["alpha"])

    def test_frozen_raw_and_summaries_crosscheck(self) -> None:
        normalized, raw, sources = load_and_normalize()
        self.assertEqual(
            Counter(row["sweep"] for row in normalized),
            {"spatial": 40, "heading": 40, "switching": 6, "radius": 5},
        )
        self.assertEqual(set(sources), {"spatial", "heading", "switching", "radius"})
        report = crosscheck_raw_and_summaries(normalized, raw)
        self.assertTrue(report["passed"])
        self.assertTrue(all(item["passed"] for item in report["cases"]))

    def test_all_fit_records_are_empirical_and_bounded(self) -> None:
        normalized, _, _ = load_and_normalize()
        fits = compute_fits(normalized)
        self.assertEqual(len(fits), 33)
        self.assertTrue(all(fit["status"] == "completed" for fit in fits))
        self.assertTrue(all(fit["empirical_fit_only"] for fit in fits))
        self.assertTrue(all(not fit["formal_big_o_claim"] for fit in fits))
        self.assertTrue(all(not fit["extrapolation_performed"] for fit in fits))
        self.assertTrue(all("N_S_active+N_E" not in fit["independent_variable"] for fit in fits))

    def test_normalized_rows_expose_user_controlled_discretization_scales(self) -> None:
        normalized, _, _ = load_and_normalize()
        self.assertTrue(all(row["spatial_resolution_m"] > 0.0 for row in normalized))
        self.assertTrue(all(row["heading_resolution_deg"] > 0.0 for row in normalized))
        self.assertTrue(all(row["switching_contour_spacing"] > 0.0 for row in normalized))
        switching = [row for row in normalized if row["sweep"] == "switching"]
        self.assertTrue(all(
            abs(row["switching_contour_spacing"] - 1.0 / row["switching_contour_sample_count"]) < 1e-15
            for row in switching
        ))


if __name__ == "__main__":
    unittest.main()

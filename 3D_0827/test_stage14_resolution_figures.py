"""Display-only algorithm selection must not alter the measured data."""

from pathlib import Path
import unittest
from unittest.mock import patch

from stage14_resolution_figures import build_resolution_figure_set


class ResolutionFigureSelectionTests(unittest.TestCase):
    def _render(self, **kwargs):
        rows = [
            {
                "case_id": variant,
                "algorithm_variant": variant,
                "status": "completed",
                "timing": {"totals": {"T_SSE_s": 10.0}},
                "memory": {"peak_rss_bytes": 1024**2},
                "J_A": 1.0, "J_D": 0.5,
                "selected_defender_action_id": 0,
                "selected_attacker_candidate_id": 1,
                "trajectory_identity": {
                    "sha256": variant,
                    "switching_position_map": [1.0, 2.0, 3.0],
                },
            }
            for variant in ("local_sse", "global_oracle")
        ]
        metadata = {row["case_id"]: {"scale": 25.0} for row in rows}
        with patch("stage14_resolution_figures.save_figure_png") as save:
            names = build_resolution_figure_set(
                rows, metadata, x_key="scale", x_title="Scale",
                sweep_label="Test", output=Path("unused"), **kwargs,
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(names), 7 if kwargs.get("include_pod_figure") else 6)
        return [call.args[0] for call in save.call_args_list]

    def test_local_only_applies_to_all_six_figures(self):
        figures = self._render(variants=("local_sse",))
        for figure in figures:
            self.assertTrue(figure.data)
            self.assertTrue(all("Global" not in trace.name for trace in figure.data))
            self.assertEqual(figure.layout.legend.x, 0.5)
            self.assertLess(figure.layout.legend.y, 0.0)
        self.assertEqual(len(figures[1].data), 6)
        self.assertEqual(len(figures[3].data), 2)
        self.assertIn("Local SSE only", figures[0].layout.title.text)

    def test_default_retains_oracle_for_other_stages(self):
        for figure in self._render():
            self.assertTrue(any("Global oracle" in trace.name for trace in figure.data))

    def test_performance_only_selection_preserves_solution_comparisons(self):
        figures = self._render(performance_local_only=True, focused_runtime_display=True)
        for figure in figures[:3]:
            self.assertTrue(all("Global" not in trace.name and "Local SSE" not in trace.name
                                for trace in figure.data))
        self.assertEqual(len(figures[1].data), 5)
        for figure in figures[3:]:
            self.assertTrue(any("Global oracle" in trace.name for trace in figure.data))

    def test_focused_runtime_labels_omit_other_without_changing_total(self):
        figures = self._render(
            variants=("local_sse",), clean_performance_display=True,
            focused_runtime_display=True,
        )
        self.assertEqual([trace.name for trace in figures[1].data], [
            "LOS", "backward reachability", "glide segment detection hazard",
            "switching candidate and feasibility", "Cost-to-go and Bellman Recursion",
        ])
        self.assertEqual(figures[1].layout.meta["png_legend_columns"], 2)
        self.assertEqual(list(figures[0].data[0].y), [10.0])

    def test_clean_performance_display_and_separate_pod(self):
        figures = self._render(
            variants=("local_sse",), clean_performance_display=True,
            include_pod_figure=True,
        )
        for figure in figures[:5]:
            self.assertTrue(all("Local SSE" not in trace.name for trace in figure.data))
        self.assertEqual(len(figures[0].data), 1)
        self.assertEqual(len(figures[2].data), 1)
        self.assertEqual([trace.name for trace in figures[3].data],
                         ["Attacker J_A", "Defender J_D"])
        self.assertEqual([trace.name for trace in figures[4].data],
                         ["Attacker PoD", "Defender PoD"])
        for trace in figures[4].data:
            self.assertEqual(list(trace.y), [0.5])


if __name__ == "__main__":
    unittest.main()

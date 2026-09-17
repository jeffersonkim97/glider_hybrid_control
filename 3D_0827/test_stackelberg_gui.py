"""Standalone resolution GUI contract tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from urllib.request import Request, urlopen

import plotly.graph_objects as go

from stackelberg_gui import (
    StackelbergGUIServer,
    resolution_estimate,
    run_resolution_case,
)


class StackelbergGUITests(unittest.TestCase):
    def test_default_sliders_encode_requested_resolution(self) -> None:
        _, estimate = resolution_estimate(25.0, 25.0, 5.0)
        self.assertEqual(estimate.heading_bins, 72)
        self.assertEqual(estimate.cartesian_state_count, 3_243_240)

    def test_extreme_request_is_rejected_before_allocation(self) -> None:
        with self.assertRaisesRegex(ValueError, "239,432,760"):
            run_resolution_case(10.0, 10.0, 1.0)

    def test_local_server_exposes_html_plotly_and_health(self) -> None:
        server = StackelbergGUIServer()
        try:
            health = json.loads(urlopen(server.url + "health").read())
            html = urlopen(server.url).read().decode()
            plotly_prefix = urlopen(server.url + "plotly.js").read(64)
        finally:
            server.shutdown()
        self.assertEqual(health, {"status": "ok"})
        self.assertIn("Horizontal discretization", html)
        self.assertIn("Altitude discretization", html)
        self.assertIn("Heading discretization", html)
        self.assertIn("Layers", html)
        self.assertIn("Show all", html)
        self.assertIn("Hide all", html)
        self.assertIn("Plotly.restyle", html)
        self.assertIn("showlegend=false", html)
        self.assertIn("Plotly", plotly_prefix.decode(errors="ignore"))

    def test_cached_result_is_returned_without_recomputation(self) -> None:
        summary = {
            "horizontal_step_m": 25.0,
            "altitude_step_m": 25.0,
            "requested_heading_step_deg": 5.0,
        }
        initial = SimpleNamespace(
            summary=summary,
            figure=go.Figure(go.Scatter3d(name="trajectory")),
        )
        server = StackelbergGUIServer(initial_result=initial)  # type: ignore[arg-type]
        try:
            request = Request(
                server.url + "solve",
                data=json.dumps({
                    "horizontal_step_m": 25.0,
                    "altitude_step_m": 25.0,
                    "heading_step_deg": 5.0,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            payload = json.loads(urlopen(request, timeout=5).read())
        finally:
            server.shutdown()
        self.assertEqual(payload["summary"], summary)
        self.assertEqual(payload["figure"]["data"][0]["name"], "trajectory")


if __name__ == "__main__":
    unittest.main()

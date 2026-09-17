"""Additive Bellman objectives for the validated glide graph.

Stage 6 introduces only elapsed time.  The adopted glide convention defines
edge duration from horizontal path length at fixed best-glide speed:

    duration = horizontal_distance / best_glide_speed

This is the same convention already used by the Stage-5 transition model for
turn feasibility.  Altitude loss remains a separate L/D=10 constraint and is
not folded into the path length a second time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bellman_geometry import GlideEdge
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE, PhysicalScale


@dataclass(frozen=True)
class TimeObjective:
    """Physical, additive edge time under the horizontal-path convention."""

    glide_speed_mps: float = DEFAULT_GLIDER.best_glide_speed_mps
    consistency_tolerance_s: float = 1.0e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.glide_speed_mps) or self.glide_speed_mps <= 0.0:
            raise ValueError("glide_speed_mps must be finite and positive")
        if (
            not np.isfinite(self.consistency_tolerance_s)
            or self.consistency_tolerance_s < 0.0
        ):
            raise ValueError(
                "consistency_tolerance_s must be finite and nonnegative"
            )

    @property
    def path_length_convention(self) -> str:
        return "horizontal"

    def duration_from_horizontal_distance_m(self, distance_m: float) -> float:
        distance = float(distance_m)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("horizontal distance must be finite and positive")
        return distance / self.glide_speed_mps

    def edge_cost_s(self, edge: GlideEdge) -> float:
        """Return edge time after checking the graph/vehicle convention."""
        if not isinstance(edge, GlideEdge):
            raise TypeError("edge must be a GlideEdge")
        expected = self.duration_from_horizontal_distance_m(
            edge.horizontal_distance_m,
        )
        if not np.isclose(
            edge.duration_s,
            expected,
            rtol=0.0,
            atol=self.consistency_tolerance_s,
        ):
            raise ValueError(
                "edge duration is inconsistent with horizontal distance "
                "and configured glide speed"
            )
        return expected

    def geometric_segment_duration_s(
        self,
        source_position_map: np.ndarray,
        target_position_map: np.ndarray,
        *,
        physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    ) -> float:
        """Recompute duration from two positions without using stored edge time."""
        source = np.asarray(source_position_map, dtype=float)
        target = np.asarray(target_position_map, dtype=float)
        if (
            source.shape != (3,)
            or target.shape != (3,)
            or not np.all(np.isfinite(source))
            or not np.all(np.isfinite(target))
        ):
            raise ValueError("segment endpoints must contain three finite coordinates")
        horizontal_distance_map = float(np.linalg.norm(target[:2] - source[:2]))
        return self.duration_from_horizontal_distance_m(
            physical_scale.distance_m(horizontal_distance_map),
        )


"""Centralized numerical-resolution contract for the validated 3D workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, gcd, pi
from numbers import Integral
from typing import Any

import numpy as np

from bellman_state import BellmanStateGrid
from map_geometry import MapBounds


@dataclass(frozen=True)
class DiscretizationConfig:
    """Every named discretization used by the fixed-defender workflow.

    The defaults reproduce the frozen Stage-10 result.  Wider heading stencils
    are enabled by changing both ``heading_bin_count`` and
    ``motion_primitive_radius``; e.g. 16/2 and 32/3.  Spatial studies may
    change ``motion_primitive_step_cells`` inversely with horizontal spacing
    to preserve a fixed physical transition horizon.
    """

    horizontal_spacing_map: float = 1.0
    minimum_altitude_map: float = 0.0
    maximum_altitude_map: float = 5.0
    altitude_spacing_map: float = 0.1
    heading_bin_count: int = 8
    motion_primitive_radius: int = 1
    motion_primitive_step_cells: int = 1
    los_probe_grid_size: int = 101
    los_boundary_refinement_steps: int = 24
    los_display_extension_factor: float = 4.0
    visualization_ray_count: int = 10
    switching_contour_sample_count: int = 12
    switching_radial_min: float = 0.5
    switching_radial_max: float = 4.0
    switching_radial_sample_count: int = 8
    hazard_quadrature_resolution: int = 8

    def __post_init__(self) -> None:
        positive_floats = (
            "horizontal_spacing_map", "altitude_spacing_map",
            "los_display_extension_factor", "switching_radial_min",
            "switching_radial_max",
        )
        for name in positive_floats:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.minimum_altitude_map < 0.0 or (
            self.maximum_altitude_map <= self.minimum_altitude_map
        ):
            raise ValueError("altitude bounds must be nonnegative and increasing")
        integer_minima = {
            "heading_bin_count": 4,
            "motion_primitive_radius": 1,
            "motion_primitive_step_cells": 1,
            "los_probe_grid_size": 3,
            "los_boundary_refinement_steps": 0,
            "visualization_ray_count": 1,
            "switching_contour_sample_count": 2,
            "switching_radial_sample_count": 1,
            "hazard_quadrature_resolution": 2,
        }
        for name, minimum in integer_minima.items():
            value = getattr(self, name)
            if (
                not isinstance(value, Integral) or isinstance(value, bool)
                or int(value) < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))
        if self.switching_radial_max < self.switching_radial_min:
            raise ValueError("switching radial bounds must be ordered")

    @property
    def radial_scales(self) -> tuple[float, ...]:
        return tuple(map(float, np.linspace(
            self.switching_radial_min,
            self.switching_radial_max,
            self.switching_radial_sample_count,
        )))

    def build_bellman_grid(self, bounds: MapBounds) -> BellmanStateGrid:
        """Inject this resolution into the solver-facing state-grid object."""
        return BellmanStateGrid(
            bounds=bounds,
            horizontal_spacing_map=self.horizontal_spacing_map,
            minimum_altitude_map=self.minimum_altitude_map,
            maximum_altitude_map=self.maximum_altitude_map,
            altitude_spacing_map=self.altitude_spacing_map,
            heading_bin_count=self.heading_bin_count,
            motion_primitive_radius=self.motion_primitive_radius,
            motion_primitive_step_cells=self.motion_primitive_step_cells,
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["switching_radial_scales"] = list(self.radial_scales)
        return result


DEFAULT_DISCRETIZATION = DiscretizationConfig()


def minimum_motion_primitive_radius(heading_bin_count: int) -> int:
    """Smallest stencil with unique directions inside half a nominal bin."""
    if (
        not isinstance(heading_bin_count, Integral)
        or isinstance(heading_bin_count, bool)
        or int(heading_bin_count) < 4
    ):
        raise ValueError("heading_bin_count must be an integer >= 4")
    count = int(heading_bin_count)
    radius = 1
    while True:
        primitive_count = sum(
            1
            for y_offset in range(-radius, radius + 1)
            for x_offset in range(-radius, radius + 1)
            if (x_offset or y_offset)
            and gcd(abs(x_offset), abs(y_offset)) == 1
        )
        if primitive_count >= count:
            offsets = BellmanStateGrid._build_motion_offsets(count, radius, 1)
            maximum_error = max(
                abs((
                    atan2(y_offset, x_offset)
                    - 2.0 * pi * index / count
                    + pi
                ) % (2.0 * pi) - pi)
                for index, (x_offset, y_offset) in enumerate(offsets)
            )
            if maximum_error <= pi / count + 1.0e-12:
                return radius
        radius += 1


def discretization_from_physical_steps(
    horizontal_step_m: float,
    altitude_step_m: float,
    heading_step_deg: float,
    *,
    meters_per_map_unit: float = 100.0,
    template: DiscretizationConfig = DEFAULT_DISCRETIZATION,
) -> DiscretizationConfig:
    """Build a complete solver configuration from the three GUI controls."""
    values = (
        float(horizontal_step_m),
        float(altitude_step_m),
        float(heading_step_deg),
        float(meters_per_map_unit),
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("physical steps and map scale must be finite and positive")
    heading_bins_float = 360.0 / values[2]
    heading_bins = int(round(heading_bins_float))
    if not np.isclose(heading_bins_float, heading_bins, rtol=0.0, atol=1.0e-10):
        raise ValueError("heading_step_deg must divide 360 degrees exactly")
    payload = template.as_dict()
    payload.pop("switching_radial_scales", None)
    payload.update({
        "horizontal_spacing_map": values[0] / values[3],
        "altitude_spacing_map": values[1] / values[3],
        "heading_bin_count": heading_bins,
        "motion_primitive_radius": minimum_motion_primitive_radius(heading_bins),
        "motion_primitive_step_cells": 1,
    })
    return DiscretizationConfig(**payload)


__all__ = [
    "DEFAULT_DISCRETIZATION",
    "DiscretizationConfig",
    "discretization_from_physical_steps",
    "minimum_motion_primitive_radius",
]

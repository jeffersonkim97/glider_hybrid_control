"""Mission-point definitions for the simplified 3D LOS study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from map_geometry import TerrainModel


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Point3D:
    """One immutable Cartesian point in displayed map coordinates."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite((self.x, self.y, self.z))):
            raise ValueError("point coordinates must be finite")

    def as_array(self) -> FloatArray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class MissionPoints:
    """Sensor and Attacker endpoint configuration for one scenario."""

    sensor: Point3D
    start: Point3D
    goal: Point3D

    def __post_init__(self) -> None:
        if np.linalg.norm(self.start.as_array() - self.goal.as_array()) <= 0.0:
            raise ValueError("start and goal must be distinct points")

    def validate_against(self, terrain_map: TerrainModel) -> None:
        """Check that all points are admissible in the selected terrain map."""
        for name, point in (
            ("sensor", self.sensor),
            ("start", self.start),
            ("goal", self.goal),
        ):
            if not (
                terrain_map.bounds.x_min <= point.x <= terrain_map.bounds.x_max
                and terrain_map.bounds.y_min <= point.y <= terrain_map.bounds.y_max
            ):
                raise ValueError(f"{name} must lie inside the horizontal map bounds")
            if point.z < terrain_map.ground_z:
                raise ValueError(f"{name} cannot lie below the ground plane")
            if terrain_map.contains_solid(point.as_array()):
                raise ValueError(f"{name} cannot lie inside a terrain obstacle")

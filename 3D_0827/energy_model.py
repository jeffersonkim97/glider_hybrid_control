"""Modular powered-flight and switching-energy model for the LOS prototype.

The LOS toy geometry uses compact map coordinates.  Physical calculations are
performed after an explicit conversion so the requested 25 m goal tolerance
does not collapse the complete toy map into the terminal set.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, asin

import numpy as np
from numpy.typing import NDArray

from map_geometry import TerrainModel
from scenario import MissionPoints, Point3D


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PhysicalScale:
    """Isotropic conversion between displayed map units and physical metres."""

    meters_per_map_unit: float = 100.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.meters_per_map_unit) or self.meters_per_map_unit <= 0.0:
            raise ValueError("meters_per_map_unit must be finite and positive")

    def position_m(self, point: Point3D | FloatArray) -> FloatArray:
        values = point.as_array() if isinstance(point, Point3D) else np.asarray(point, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("a physical position requires three finite map coordinates")
        return values * self.meters_per_map_unit

    def distance_m(self, map_distance: float) -> float:
        return float(map_distance) * self.meters_per_map_unit


@dataclass(frozen=True)
class GliderParameters:
    """Prototype constants based on the historical Schleicher Ka 6 CR."""

    aircraft_name: str = "Schleicher Ka 6 CR"
    mass_kg: float = 304.0
    best_glide_speed_mps: float = 80.0 / 3.6
    best_glide_ratio: float = 10.0
    powered_speed_mps: float = 80.0 / 3.6
    maximum_bank_deg: float = 30.0
    gravity_mps2: float = 9.81
    switch_energy_loss_height_m: float = 10.0
    goal_tolerance_m: float = 25.0
    source_url: str = (
        "https://mobilit.belgium.be/sites/default/files/domain/Aviation/"
        "Veiligheid/Verslagen%20voorvallen/2010/2010_5.pdf"
    )

    def __post_init__(self) -> None:
        positive = (
            self.mass_kg,
            self.best_glide_speed_mps,
            self.best_glide_ratio,
            self.powered_speed_mps,
            self.gravity_mps2,
            self.goal_tolerance_m,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("mass, speeds, glide ratio, gravity, and tolerance must be positive")
        if not 0.0 < self.maximum_bank_deg < 90.0:
            raise ValueError("maximum_bank_deg must lie strictly between 0 and 90")
        if (
            not np.isfinite(self.switch_energy_loss_height_m)
            or self.switch_energy_loss_height_m < 0.0
        ):
            raise ValueError("switch energy loss height must be finite and nonnegative")

    @property
    def maximum_turn_rate_rad_s(self) -> float:
        """Coordinated-level-turn rate at the nominal glide speed."""
        bank_rad = np.deg2rad(self.maximum_bank_deg)
        return float(
            self.gravity_mps2 * np.tan(bank_rad) / self.best_glide_speed_mps
        )

    @property
    def minimum_turn_radius_m(self) -> float:
        return self.best_glide_speed_mps / self.maximum_turn_rate_rad_s

    @property
    def turn_load_factor(self) -> float:
        """Load factor at the configured prototype bank limit."""
        return float(1.0 / np.cos(np.deg2rad(self.maximum_bank_deg)))

    @property
    def turn_glide_ratio(self) -> float:
        """Induced-drag approximation L/D_turn = L/D_straight / n^2."""
        return self.best_glide_ratio / self.turn_load_factor**2


DEFAULT_PHYSICAL_SCALE = PhysicalScale()
DEFAULT_GLIDER = GliderParameters()


@dataclass(frozen=True)
class SwitchingState:
    """Full position/velocity state delivered by the powered phase."""

    position_map: FloatArray
    position_m: FloatArray
    velocity_mps: FloatArray
    powered_path_length_m: float
    flight_path_angle_rad: float
    heading_rad: float
    total_mechanical_energy_j: float
    powered_feasible: bool
    infeasibility_reason: str | None = None

    def __post_init__(self) -> None:
        arrays = (self.position_map, self.position_m, self.velocity_mps)
        if any(np.asarray(array).shape != (3,) for array in arrays):
            raise ValueError("switching position and velocity arrays must have shape (3,)")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("switching position and velocity arrays must be finite")
        if self.powered_path_length_m < 0.0 or not np.isfinite(self.powered_path_length_m):
            raise ValueError("powered path length must be finite and nonnegative")
        if not np.isfinite(self.total_mechanical_energy_j):
            raise ValueError("switching energy must be finite")


class StraightPoweredPhaseModel:
    """Constant-speed powered flight along start-to-switch straight segments.

    The engine maintains the configured speed while adding the potential
    energy required by the climb.  This matches the existing project's simple
    straight powered-phase convention and keeps propulsion assumptions out of
    the downstream glide module.
    """

    def __init__(
        self,
        parameters: GliderParameters = DEFAULT_GLIDER,
        physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    ) -> None:
        self.parameters = parameters
        self.physical_scale = physical_scale

    def state_at(
        self,
        switching_point_map: FloatArray,
        mission_points: MissionPoints,
        terrain_map: TerrainModel,
    ) -> SwitchingState:
        target_map = np.asarray(switching_point_map, dtype=float)
        if target_map.shape != (3,) or not np.all(np.isfinite(target_map)):
            raise ValueError("switching_point_map must contain three finite values")

        start_map = mission_points.start.as_array()
        displacement_map = target_map - start_map
        path_length_map = float(np.linalg.norm(displacement_map))
        if path_length_map <= 1.0e-12:
            direction = np.zeros(3, dtype=float)
            powered_feasible = False
            reason = "switching point coincides with start"
        else:
            direction = displacement_map / path_length_map
            powered_feasible = not terrain_map.segment_intersects_solid(
                start_map, target_map,
            )
            reason = None if powered_feasible else "powered segment intersects terrain"

        velocity = self.parameters.powered_speed_mps * direction
        target_m = self.physical_scale.position_m(target_map)
        path_length_m = self.physical_scale.distance_m(path_length_map)
        horizontal_speed = float(np.hypot(velocity[0], velocity[1]))
        flight_path_angle = asin(
            float(np.clip(velocity[2] / self.parameters.powered_speed_mps, -1.0, 1.0))
        )
        heading = atan2(float(velocity[1]), float(velocity[0]))
        total_energy = self.parameters.mass_kg * (
            self.parameters.gravity_mps2 * target_m[2]
            + 0.5 * self.parameters.powered_speed_mps**2
        )
        if horizontal_speed <= 1.0e-12 and powered_feasible:
            powered_feasible = False
            reason = "powered arrival has no horizontal heading"

        return SwitchingState(
            position_map=target_map.copy(),
            position_m=target_m,
            velocity_mps=velocity,
            powered_path_length_m=path_length_m,
            flight_path_angle_rad=flight_path_angle,
            heading_rad=heading,
            total_mechanical_energy_j=total_energy,
            powered_feasible=powered_feasible,
            infeasibility_reason=reason,
        )

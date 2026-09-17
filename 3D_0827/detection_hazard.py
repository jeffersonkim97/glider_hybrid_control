"""Stage-7 detection-rate, PoD, and mission-objective primitives.

The glide hazard model retains the completed 3D extension's radar and Doppler
rate equations.  Acoustic hazard is intentionally absent from this glide-only
module; powered/glide mission composition belongs to a later stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from energy_model import DEFAULT_PHYSICAL_SCALE, PhysicalScale
from map_geometry import TerrainModel
from scenario import Point3D


@dataclass(frozen=True)
class DetectionHazardParameters:
    """Frozen physical constants from ``attacker_hazard_time_v2``."""

    range_floor_m: float = 10.0
    radar_coefficient: float = 1.3e7
    doppler_coefficient: float = 3.325e4
    rcs_min: float = 0.1
    rcs_max: float = 1.0
    radar_rate_scale: float = 1.0
    radial_velocity_rate_scale: float = 1.0

    def __post_init__(self) -> None:
        positive = (
            self.range_floor_m,
            self.radar_coefficient,
            self.doppler_coefficient,
            self.radar_rate_scale,
            self.radial_velocity_rate_scale,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("detection ranges, coefficients, and scales must be positive")
        if (
            not np.isfinite(self.rcs_min)
            or not np.isfinite(self.rcs_max)
            or self.rcs_min < 0.0
            or self.rcs_min > self.rcs_max
        ):
            raise ValueError("RCS bounds must be finite, nonnegative, and ordered")


DEFAULT_DETECTION_HAZARD = DetectionHazardParameters()


@dataclass(frozen=True)
class HazardRateEvaluation:
    """Auditable instantaneous hazard-rate components at one sample."""

    visible: bool
    sensor_range_m: float
    radial_velocity_mps: float
    cosine_aspect: float
    radar_cross_section: float
    radar_rate_per_s: float
    doppler_rate_per_s: float
    total_rate_per_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.visible, (bool, np.bool_)):
            raise TypeError("visible must be boolean")
        finite = (
            self.sensor_range_m,
            self.radial_velocity_mps,
            self.cosine_aspect,
            self.radar_cross_section,
            self.radar_rate_per_s,
            self.doppler_rate_per_s,
            self.total_rate_per_s,
        )
        if any(not np.isfinite(value) for value in finite):
            raise ValueError("hazard-rate diagnostics must be finite")
        if self.sensor_range_m < 0.0:
            raise ValueError("sensor range must be nonnegative")
        if not -1.0 <= self.cosine_aspect <= 1.0:
            raise ValueError("cosine_aspect must lie in [-1, 1]")
        if any(value < 0.0 for value in (
            self.radar_cross_section,
            self.radar_rate_per_s,
            self.doppler_rate_per_s,
            self.total_rate_per_s,
        )):
            raise ValueError("RCS and hazard rates must be nonnegative")
        if not np.isclose(
            self.total_rate_per_s,
            self.radar_rate_per_s + self.doppler_rate_per_s,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError("total hazard rate must equal radar plus Doppler rate")


@runtime_checkable
class HazardField(Protocol):
    """Time-aware instantaneous rate interface consumed by edge quadrature."""

    def evaluate_rate(
        self,
        position_map: np.ndarray,
        velocity_mps: np.ndarray,
        time_s: float,
    ) -> HazardRateEvaluation: ...


def _synthetic_rate_evaluation(rate_per_s: float) -> HazardRateEvaluation:
    rate = float(rate_per_s)
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError("synthetic hazard rate must be finite and nonnegative")
    return HazardRateEvaluation(
        visible=True,
        sensor_range_m=0.0,
        radial_velocity_mps=0.0,
        cosine_aspect=0.0,
        radar_cross_section=0.0,
        radar_rate_per_s=rate,
        doppler_rate_per_s=0.0,
        total_rate_per_s=rate,
    )


@dataclass(frozen=True)
class ConstantHazardField:
    """Synthetic analytical field used to validate integration."""

    rate_per_s: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.rate_per_s) or self.rate_per_s < 0.0:
            raise ValueError("rate_per_s must be finite and nonnegative")

    def evaluate_rate(
        self,
        position_map: np.ndarray,
        velocity_mps: np.ndarray,
        time_s: float,
    ) -> HazardRateEvaluation:
        return _synthetic_rate_evaluation(self.rate_per_s)


@dataclass(frozen=True)
class PiecewiseConstantHazardField:
    """Two-region analytical field with an averaged boundary value."""

    axis_index: int
    boundary_map: float
    below_rate_per_s: float
    above_rate_per_s: float
    boundary_tolerance_map: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.axis_index not in (0, 1, 2):
            raise ValueError("axis_index must be 0, 1, or 2")
        if not np.isfinite(self.boundary_map):
            raise ValueError("boundary_map must be finite")
        for name in ("below_rate_per_s", "above_rate_per_s"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not np.isfinite(self.boundary_tolerance_map)
            or self.boundary_tolerance_map < 0.0
        ):
            raise ValueError("boundary tolerance must be finite and nonnegative")

    def evaluate_rate(
        self,
        position_map: np.ndarray,
        velocity_mps: np.ndarray,
        time_s: float,
    ) -> HazardRateEvaluation:
        position = np.asarray(position_map, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_map must contain three finite coordinates")
        coordinate = float(position[self.axis_index])
        if coordinate < self.boundary_map - self.boundary_tolerance_map:
            rate = self.below_rate_per_s
        elif coordinate > self.boundary_map + self.boundary_tolerance_map:
            rate = self.above_rate_per_s
        else:
            rate = 0.5 * (self.below_rate_per_s + self.above_rate_per_s)
        return _synthetic_rate_evaluation(rate)


@dataclass(frozen=True)
class GlideDetectionHazardModel:
    """LOS-gated radar plus radial-velocity hazard for glide motion."""

    terrain: TerrainModel
    sensor: Point3D
    parameters: DetectionHazardParameters = DEFAULT_DETECTION_HAZARD
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE

    def __post_init__(self) -> None:
        if not isinstance(self.sensor, Point3D):
            raise TypeError("sensor must be a Point3D")
        if not isinstance(self.parameters, DetectionHazardParameters):
            raise TypeError("parameters must be DetectionHazardParameters")
        if not isinstance(self.physical_scale, PhysicalScale):
            raise TypeError("physical_scale must be a PhysicalScale")

    def is_visible(self, position_map: np.ndarray) -> bool:
        position = np.asarray(position_map, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_map must contain three finite coordinates")
        return not self.terrain.segment_intersects_solid(
            position,
            self.sensor.as_array(),
        )

    def evaluate_rate(
        self,
        position_map: np.ndarray,
        velocity_mps: np.ndarray,
        time_s: float,
    ) -> HazardRateEvaluation:
        position = np.asarray(position_map, dtype=float)
        velocity = np.asarray(velocity_mps, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_map must contain three finite coordinates")
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError("velocity_mps must contain three finite components")
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and nonnegative")

        sensor_delta_m = self.physical_scale.position_m(
            self.sensor.as_array() - position,
        )
        slant_range_m = float(np.linalg.norm(sensor_delta_m))
        sensor_range_m = max(slant_range_m, self.parameters.range_floor_m)
        los_unit = sensor_delta_m / sensor_range_m
        radial_velocity_mps = float(np.dot(velocity, los_unit))
        speed_mps = float(np.linalg.norm(velocity))
        cosine_aspect = float(np.clip(
            radial_velocity_mps / max(speed_mps, 1.0e-9),
            -1.0,
            1.0,
        ))
        radar_cross_section = (
            self.parameters.rcs_min
            + (self.parameters.rcs_max - self.parameters.rcs_min)
            * cosine_aspect**2
        )
        visible = self.is_visible(position)
        visibility = 1.0 if visible else 0.0
        inverse_range_fourth = 1.0 / sensor_range_m**4
        radar_rate = (
            visibility
            * self.parameters.radar_rate_scale
            * self.parameters.radar_coefficient
            * radar_cross_section
            * inverse_range_fourth
        )
        doppler_rate = (
            visibility
            * self.parameters.radial_velocity_rate_scale
            * self.parameters.doppler_coefficient
            * radial_velocity_mps**2
            * inverse_range_fourth
        )
        return HazardRateEvaluation(
            visible=visible,
            sensor_range_m=sensor_range_m,
            radial_velocity_mps=radial_velocity_mps,
            cosine_aspect=cosine_aspect,
            radar_cross_section=radar_cross_section,
            radar_rate_per_s=radar_rate,
            doppler_rate_per_s=doppler_rate,
            total_rate_per_s=radar_rate + doppler_rate,
        )


def hazard_to_detection_probability(hazard: float) -> float:
    """Convert nonnegative cumulative hazard to ``P_D = 1 - exp(-H)``."""
    value = float(hazard)
    if np.isnan(value) or value < 0.0:
        raise ValueError("hazard must be nonnegative and not NaN")
    if np.isposinf(value):
        return 1.0
    if not np.isfinite(value):
        raise ValueError("hazard must be finite or positive infinity")
    return float(-np.expm1(-value))


@dataclass(frozen=True)
class AttackerHazardTimeParameters:
    """Actual normalized ``attacker_hazard_time_v2`` definition."""

    hazard_weight: float = 0.5
    time_weight: float = 0.5
    hazard_reference: float = 1.0
    time_reference_s: float = 5000.0 / 22.6
    objective_id: str = "attacker_hazard_time_v2"

    def __post_init__(self) -> None:
        if any(
            not np.isfinite(value) or value < 0.0
            for value in (self.hazard_weight, self.time_weight)
        ):
            raise ValueError("objective weights must be finite and nonnegative")
        if not np.isclose(
            self.hazard_weight + self.time_weight,
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("attacker objective weights must sum to one")
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in (self.hazard_reference, self.time_reference_s)
        ):
            raise ValueError("objective reference scales must be finite and positive")
        if self.objective_id != "attacker_hazard_time_v2":
            raise ValueError("unsupported attacker objective identifier")


DEFAULT_ATTACKER_HAZARD_TIME = AttackerHazardTimeParameters()


@dataclass(frozen=True)
class AttackerObjectiveBreakdown:
    mission_hazard: float
    mission_pod: float
    mission_time_s: float
    normalized_hazard: float
    normalized_time: float
    weighted_hazard_term: float
    weighted_time_term: float
    objective_value: float


def evaluate_attacker_hazard_time_objective(
    mission_hazard: float,
    mission_time_s: float,
    *,
    parameters: AttackerHazardTimeParameters = DEFAULT_ATTACKER_HAZARD_TIME,
) -> AttackerObjectiveBreakdown:
    """Expose every term of the additive normalized attacker objective."""
    hazard = float(mission_hazard)
    time_s = float(mission_time_s)
    if not np.isfinite(hazard) or hazard < 0.0:
        raise ValueError("mission_hazard must be finite and nonnegative")
    if not np.isfinite(time_s) or time_s < 0.0:
        raise ValueError("mission_time_s must be finite and nonnegative")
    normalized_hazard = hazard / parameters.hazard_reference
    normalized_time = time_s / parameters.time_reference_s
    weighted_hazard = parameters.hazard_weight * normalized_hazard
    weighted_time = parameters.time_weight * normalized_time
    return AttackerObjectiveBreakdown(
        mission_hazard=hazard,
        mission_pod=hazard_to_detection_probability(hazard),
        mission_time_s=time_s,
        normalized_hazard=normalized_hazard,
        normalized_time=normalized_time,
        weighted_hazard_term=weighted_hazard,
        weighted_time_term=weighted_time,
        objective_value=weighted_hazard + weighted_time,
    )


"""Stage-7 edge quadrature and path-hazard accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from bellman_geometry import GlideEdge
from bellman_state import BellmanStateGrid
from detection_hazard import HazardField
from energy_model import DEFAULT_PHYSICAL_SCALE, PhysicalScale
from solver_metrics import HazardTiming


@dataclass(frozen=True)
class EdgeHazardSample:
    """One row of a manually auditable trapezoidal quadrature table."""

    sample_index: int
    time_fraction: float
    time_s: float
    position_map: np.ndarray
    visible: bool
    sensor_range_m: float
    radial_velocity_mps: float
    radar_cross_section: float
    radar_rate_per_s: float
    doppler_rate_per_s: float
    total_rate_per_s: float
    quadrature_weight_s: float
    hazard_contribution: float

    def __post_init__(self) -> None:
        if not isinstance(self.sample_index, Integral) or isinstance(self.sample_index, bool):
            raise TypeError("sample_index must be an integer")
        if int(self.sample_index) < 0:
            raise ValueError("sample_index must be nonnegative")
        position = np.asarray(self.position_map, dtype=float).copy()
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_map must contain three finite coordinates")
        position.setflags(write=False)
        object.__setattr__(self, "sample_index", int(self.sample_index))
        object.__setattr__(self, "position_map", position)
        for name in (
            "time_fraction",
            "time_s",
            "sensor_range_m",
            "radial_velocity_mps",
            "radar_cross_section",
            "radar_rate_per_s",
            "doppler_rate_per_s",
            "total_rate_per_s",
            "quadrature_weight_s",
            "hazard_contribution",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.time_fraction <= 1.0:
            raise ValueError("time_fraction must lie in [0, 1]")
        if any(getattr(self, name) < 0.0 for name in (
            "time_s",
            "sensor_range_m",
            "radar_cross_section",
            "radar_rate_per_s",
            "doppler_rate_per_s",
            "total_rate_per_s",
            "quadrature_weight_s",
            "hazard_contribution",
        )):
            raise ValueError("times, ranges, rates, weights, and contributions must be nonnegative")


@dataclass(frozen=True)
class EdgeHazardResult:
    edge: GlideEdge
    duration_s: float
    hazard: float
    quadrature_resolution: int
    visible_sample_count: int
    samples: tuple[EdgeHazardSample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.edge, GlideEdge):
            raise TypeError("edge must be a GlideEdge")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if not np.isfinite(self.hazard) or self.hazard < 0.0:
            raise ValueError("hazard must be finite and nonnegative")
        if self.quadrature_resolution != len(self.samples):
            raise ValueError("quadrature resolution must equal sample count")
        if not 0 <= self.visible_sample_count <= self.quadrature_resolution:
            raise ValueError("visible_sample_count lies outside sample count")

    @property
    def occluded_sample_count(self) -> int:
        return self.quadrature_resolution - self.visible_sample_count


@dataclass(frozen=True)
class SegmentHazardResult:
    """Hazard integral for a continuous segment outside the glide lattice."""

    start_position_map: np.ndarray
    end_position_map: np.ndarray
    duration_s: float
    hazard: float
    quadrature_resolution: int
    visible_sample_count: int
    samples: tuple[EdgeHazardSample, ...]

    def __post_init__(self) -> None:
        for name in ("start_position_map", "end_position_map"):
            vector = np.array(getattr(self, name), dtype=float, copy=True)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must contain three finite coordinates")
            vector.setflags(write=False)
            object.__setattr__(self, name, vector)
        if not np.isfinite(self.duration_s) or self.duration_s < 0.0:
            raise ValueError("duration_s must be finite and nonnegative")
        if not np.isfinite(self.hazard) or self.hazard < 0.0:
            raise ValueError("hazard must be finite and nonnegative")
        if self.duration_s == 0.0:
            if self.quadrature_resolution != 0 or self.samples:
                raise ValueError("an identity segment has no quadrature samples")
        elif self.quadrature_resolution != len(self.samples):
            raise ValueError("quadrature resolution must equal sample count")
        if not 0 <= self.visible_sample_count <= self.quadrature_resolution:
            raise ValueError("visible_sample_count lies outside sample count")


def integrate_segment_hazard(
    start_position_map: np.ndarray,
    end_position_map: np.ndarray,
    duration_s: float,
    hazard_field: HazardField,
    *,
    quadrature_resolution: int = 16,
    start_time_s: float = 0.0,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
) -> SegmentHazardResult:
    """Apply Stage-7 trapezoidal quadrature to a continuous segment."""
    start = np.asarray(start_position_map, dtype=float)
    end = np.asarray(end_position_map, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("segment endpoints must contain three coordinates")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("segment endpoints must be finite")
    if not isinstance(hazard_field, HazardField):
        raise TypeError("hazard_field must implement HazardField")
    duration = float(duration_s)
    initial_time = float(start_time_s)
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_s must be finite and nonnegative")
    if not np.isfinite(initial_time) or initial_time < 0.0:
        raise ValueError("start_time_s must be finite and nonnegative")
    if duration == 0.0:
        if not np.allclose(start, end, rtol=0.0, atol=1.0e-12):
            raise ValueError("a zero-duration segment must have identical endpoints")
        return SegmentHazardResult(
            start_position_map=start,
            end_position_map=end,
            duration_s=0.0,
            hazard=0.0,
            quadrature_resolution=0,
            visible_sample_count=0,
            samples=(),
        )

    resolution = _validated_resolution(quadrature_resolution)
    displacement_map = end - start
    velocity_mps = physical_scale.position_m(displacement_map) / duration
    fractions = np.linspace(0.0, 1.0, resolution)
    spacing_s = duration / (resolution - 1)
    weights_s = np.full(resolution, spacing_s, dtype=float)
    weights_s[[0, -1]] *= 0.5
    samples: list[EdgeHazardSample] = []
    visible_sample_count = 0
    for sample_index, (fraction, weight_s) in enumerate(zip(fractions, weights_s)):
        position = start + fraction * displacement_map
        sample_time = initial_time + fraction * duration
        rate = hazard_field.evaluate_rate(position, velocity_mps, sample_time)
        contribution = rate.total_rate_per_s * weight_s
        visible_sample_count += int(rate.visible)
        samples.append(EdgeHazardSample(
            sample_index=sample_index,
            time_fraction=float(fraction),
            time_s=float(sample_time),
            position_map=position,
            visible=bool(rate.visible),
            sensor_range_m=rate.sensor_range_m,
            radial_velocity_mps=rate.radial_velocity_mps,
            radar_cross_section=rate.radar_cross_section,
            radar_rate_per_s=rate.radar_rate_per_s,
            doppler_rate_per_s=rate.doppler_rate_per_s,
            total_rate_per_s=rate.total_rate_per_s,
            quadrature_weight_s=float(weight_s),
            hazard_contribution=float(contribution),
        ))
    return SegmentHazardResult(
        start_position_map=start,
        end_position_map=end,
        duration_s=duration,
        hazard=float(sum(sample.hazard_contribution for sample in samples)),
        quadrature_resolution=resolution,
        visible_sample_count=visible_sample_count,
        samples=tuple(samples),
    )


@dataclass(frozen=True)
class HazardPrecomputation:
    """Compact hazards aligned with an input adjacency sequence."""

    hazard_by_source: tuple[tuple[float, ...], ...]
    timing: HazardTiming
    visibility_sample_count: int


def _validated_resolution(quadrature_resolution: int) -> int:
    if (
        not isinstance(quadrature_resolution, Integral)
        or isinstance(quadrature_resolution, bool)
    ):
        raise TypeError("quadrature_resolution must be an integer")
    resolution = int(quadrature_resolution)
    if resolution < 2:
        raise ValueError("quadrature_resolution must be at least two")
    return resolution


def integrate_edge_hazard(
    edge: GlideEdge,
    grid: BellmanStateGrid,
    hazard_field: HazardField,
    *,
    quadrature_resolution: int = 16,
    start_time_s: float = 0.0,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
) -> EdgeHazardResult:
    """Integrate one edge with uniform-time trapezoidal quadrature."""
    if not isinstance(edge, GlideEdge):
        raise TypeError("edge must be a GlideEdge")
    if not isinstance(grid, BellmanStateGrid):
        raise TypeError("grid must be a BellmanStateGrid")
    if not isinstance(hazard_field, HazardField):
        raise TypeError("hazard_field must implement HazardField")
    resolution = _validated_resolution(quadrature_resolution)
    start_time = float(start_time_s)
    if not np.isfinite(start_time) or start_time < 0.0:
        raise ValueError("start_time_s must be finite and nonnegative")

    source = grid.position_map(edge.source_state)
    target = grid.position_map(edge.target_state)
    displacement_map = target - source
    displacement_m = physical_scale.position_m(displacement_map)
    velocity_mps = displacement_m / edge.duration_s
    fractions = np.linspace(0.0, 1.0, resolution)
    spacing_s = edge.duration_s / (resolution - 1)
    weights_s = np.full(resolution, spacing_s, dtype=float)
    weights_s[[0, -1]] *= 0.5

    samples: list[EdgeHazardSample] = []
    visible_sample_count = 0
    for sample_index, (fraction, weight_s) in enumerate(zip(fractions, weights_s)):
        position = source + fraction * displacement_map
        sample_time_s = start_time + fraction * edge.duration_s
        rate = hazard_field.evaluate_rate(position, velocity_mps, sample_time_s)
        contribution = rate.total_rate_per_s * weight_s
        visible_sample_count += int(rate.visible)
        samples.append(EdgeHazardSample(
            sample_index=sample_index,
            time_fraction=float(fraction),
            time_s=float(sample_time_s),
            position_map=position,
            visible=bool(rate.visible),
            sensor_range_m=rate.sensor_range_m,
            radial_velocity_mps=rate.radial_velocity_mps,
            radar_cross_section=rate.radar_cross_section,
            radar_rate_per_s=rate.radar_rate_per_s,
            doppler_rate_per_s=rate.doppler_rate_per_s,
            total_rate_per_s=rate.total_rate_per_s,
            quadrature_weight_s=float(weight_s),
            hazard_contribution=float(contribution),
        ))
    hazard = float(sum(sample.hazard_contribution for sample in samples))
    return EdgeHazardResult(
        edge=edge,
        duration_s=edge.duration_s,
        hazard=hazard,
        quadrature_resolution=resolution,
        visible_sample_count=visible_sample_count,
        samples=tuple(samples),
    )


def accumulate_path_hazard(edge_results: Iterable[EdgeHazardResult]) -> float:
    """Use the additive cumulative-hazard convention across a path."""
    results = tuple(edge_results)
    if any(not isinstance(result, EdgeHazardResult) for result in results):
        raise TypeError("edge_results must contain EdgeHazardResult objects")
    return float(sum(result.hazard for result in results))


def precompute_edge_hazards(
    adjacency: Sequence[Sequence[GlideEdge]],
    grid: BellmanStateGrid,
    hazard_field: HazardField,
    *,
    quadrature_resolution: int = 16,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
) -> HazardPrecomputation:
    """Precompute compact edge hazards and return isolated runtime metrics."""
    resolution = _validated_resolution(quadrature_resolution)
    start = perf_counter()
    values: list[tuple[float, ...]] = []
    edge_count = 0
    visibility_sample_count = 0
    for edges in adjacency:
        source_values: list[float] = []
        for edge in edges:
            result = integrate_edge_hazard(
                edge,
                grid,
                hazard_field,
                quadrature_resolution=resolution,
                physical_scale=physical_scale,
            )
            source_values.append(result.hazard)
            edge_count += 1
            visibility_sample_count += result.visible_sample_count
        values.append(tuple(source_values))
    elapsed = perf_counter() - start
    per_edge = elapsed / edge_count if edge_count else 0.0
    return HazardPrecomputation(
        hazard_by_source=tuple(values),
        timing=HazardTiming(
            precompute_s=elapsed,
            per_edge_s=per_edge,
            edge_count=edge_count,
            quadrature_resolution=resolution,
        ),
        visibility_sample_count=visibility_sample_count,
    )

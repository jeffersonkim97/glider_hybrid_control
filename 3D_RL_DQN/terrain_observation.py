"""Deterministic terrain-aware observation layer for Phase 16.2.

The builder sits above :class:`attacker_br_problem.AttackerBRProblem`.  It never
changes transitions, feasibility, terminals, or objectives.  Its local window
is perception only; the Phase 1 graph remains authoritative.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from math import cos, sin
from typing import Any

import numpy as np
from numpy.typing import NDArray

import project_paths  # noqa: F401
from attacker_br_problem import AttackerBRProblem
from ray_tracing import TriangleRayTracer


FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]

OBSERVATION_SCHEMA_ID = "p1b-terrain-observation-v1"
EGO_FEATURE_NAMES = (
    "altitude_above_ground_m",
    "terrain_clearance_m",
    "sin_heading_psi",
    "cos_heading_psi",
)
GOAL_FEATURE_NAMES = (
    "goal_forward_m",
    "goal_right_m",
    "goal_vertical_m",
    "goal_distance_minus_tolerance_m",
)
TERRAIN_CHANNEL_NAMES = ("terrain_clearance_m",)
HAZARD_CHANNEL_NAMES = (
    "sensor_visible",
    "sensor_los_forward",
    "sensor_los_right",
    "sensor_los_up",
    "sensor_range_factor",
)
VALIDITY_CHANNEL_NAMES = ("inside_attacker_domain",)


@dataclass(frozen=True)
class ObservationConfig:
    """Approved condition-specific Phase 16.2 perception geometry."""

    local_window_extent_m: float = 2000.0
    local_window_shape_cells: tuple[int, int] = (21, 21)
    local_window_orientation: str = "heading_aligned"
    schema_id: str = OBSERVATION_SCHEMA_ID

    def __post_init__(self) -> None:
        extent = float(self.local_window_extent_m)
        if not np.isfinite(extent) or extent <= 0.0:
            raise ValueError("local_window_extent_m must be finite and positive")
        shape = tuple(int(value) for value in self.local_window_shape_cells)
        if len(shape) != 2 or any(value < 3 for value in shape):
            raise ValueError("local_window_shape_cells must contain two values >= 3")
        if any(value % 2 == 0 for value in shape):
            raise ValueError("local window dimensions must be odd so the attacker is central")
        if self.local_window_orientation != "heading_aligned":
            raise ValueError("Phase 16.2 supports the approved heading_aligned orientation")
        if self.schema_id != OBSERVATION_SCHEMA_ID:
            raise ValueError(f"unsupported observation schema: {self.schema_id!r}")
        object.__setattr__(self, "local_window_extent_m", extent)
        object.__setattr__(self, "local_window_shape_cells", shape)

    @property
    def sample_spacing_m(self) -> tuple[float, float]:
        height, width = self.local_window_shape_cells
        return (
            self.local_window_extent_m / (height - 1),
            self.local_window_extent_m / (width - 1),
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["local_window_shape_cells"] = list(self.local_window_shape_cells)
        result["sample_spacing_m"] = list(self.sample_spacing_m)
        result["parameter_origin"] = "new_design_choice_confirmed_by_user"
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ObservationConfig":
        return cls(
            local_window_extent_m=float(values["local_window_extent_m"]),
            local_window_shape_cells=tuple(values["local_window_shape_cells"]),
            local_window_orientation=str(values["local_window_orientation"]),
            schema_id=str(values.get("schema_id", OBSERVATION_SCHEMA_ID)),
        )


@dataclass(frozen=True)
class StructuredObservation:
    """Raw physical observation before deterministic scaling."""

    ego_features: FloatArray
    goal_features: FloatArray
    terrain_channels: FloatArray
    hazard_channels: FloatArray
    validity_channels: FloatArray
    metadata: dict[str, Any]

    def raw_schema(self) -> dict[str, Any]:
        return {
            "ego": {"names": list(EGO_FEATURE_NAMES), "shape": list(self.ego_features.shape)},
            "goal": {"names": list(GOAL_FEATURE_NAMES), "shape": list(self.goal_features.shape)},
            "terrain": {"names": list(TERRAIN_CHANNEL_NAMES), "shape": list(self.terrain_channels.shape)},
            "hazard": {"names": list(HAZARD_CHANNEL_NAMES), "shape": list(self.hazard_channels.shape)},
            "validity": {"names": list(VALIDITY_CHANNEL_NAMES), "shape": list(self.validity_channels.shape)},
        }


@dataclass(frozen=True)
class TensorReadyObservation:
    """Network-agnostic normalized arrays preserving scalar/spatial separation."""

    scalar: Float32Array
    spatial: Float32Array

    def as_dict(self) -> dict[str, Float32Array]:
        return {"scalar": self.scalar, "spatial": self.spatial}


class TerrainObservationBuilder:
    """Build fixed-shape local geometry and sensing observations."""

    def __init__(
        self, problem: AttackerBRProblem,
        config: ObservationConfig = ObservationConfig(),
        *,
        surface_height_cache: OrderedDict[tuple[float, float], float] | None = None,
        max_surface_cache_entries: int | None = None,
    ) -> None:
        if max_surface_cache_entries is not None and int(max_surface_cache_entries) < 1:
            raise ValueError("max_surface_cache_entries must be positive or None")
        self.problem = problem
        self.config = config
        self.scene = problem.scene
        self.scale = self.scene.config.physical_scale
        self._terrain_tracer = TriangleRayTracer(self.scene.terrain.surface_meshes())
        self.max_surface_cache_entries = (
            None if max_surface_cache_entries is None
            else int(max_surface_cache_entries)
        )
        self._surface_height_cache = (
            surface_height_cache if surface_height_cache is not None else OrderedDict()
        )

    @property
    def tensor_ready_shapes(self) -> dict[str, tuple[int, ...]]:
        height, width = self.config.local_window_shape_cells
        return {
            "scalar": (len(EGO_FEATURE_NAMES) + len(GOAL_FEATURE_NAMES),),
            "spatial": (
                len(TERRAIN_CHANNEL_NAMES)
                + len(HAZARD_CHANNEL_NAMES)
                + len(VALIDITY_CHANNEL_NAMES),
                height,
                width,
            ),
        }

    def _axes(self, heading: float) -> tuple[np.ndarray, np.ndarray]:
        forward = np.array([cos(heading), sin(heading)], dtype=float)
        right = np.array([sin(heading), -cos(heading)], dtype=float)
        return forward, right

    def local_offsets_m(self) -> tuple[FloatArray, FloatArray]:
        """Return forward/right offsets; row zero is the far-forward edge."""

        height, width = self.config.local_window_shape_cells
        half = 0.5 * self.config.local_window_extent_m
        forward = np.linspace(half, -half, height, dtype=float)
        right = np.linspace(-half, half, width, dtype=float)
        return np.meshgrid(forward, right, indexing="ij")

    def sample_positions_map(self, state_id: int) -> FloatArray:
        position = self.problem.position_map(state_id)
        heading = float(self.problem.grid.heading_rad(self.problem.state(state_id).heading_bin))
        forward_axis, right_axis = self._axes(heading)
        forward_m, right_m = self.local_offsets_m()
        horizontal_m = (
            forward_m[..., None] * forward_axis[None, None, :]
            + right_m[..., None] * right_axis[None, None, :]
        )
        positions = np.empty(forward_m.shape + (3,), dtype=float)
        positions[..., :2] = (
            position[None, None, :2]
            + horizontal_m / self.scale.meters_per_map_unit
        )
        positions[..., 2] = position[2]
        return positions

    def _inside_domain(self, positions: FloatArray) -> NDArray[np.bool_]:
        bounds = self.problem.grid.bounds
        return (
            (positions[..., 0] >= bounds.x_min)
            & (positions[..., 0] <= bounds.x_max)
            & (positions[..., 1] >= bounds.y_min)
            & (positions[..., 1] <= bounds.y_max)
        )

    def terrain_surface_height_map(self, x_map: float, y_map: float) -> float:
        """Highest authoritative terrain surface under one horizontal point."""

        key = (round(float(x_map), 12), round(float(y_map), 12))
        cached = self._surface_height_cache.get(key)
        if cached is not None:
            self._surface_height_cache.move_to_end(key)
            return cached
        origin_z = max(
            float(self.problem.grid.maximum_altitude_map),
            float(self.scene.terrain.maximum_height),
        ) + 1.0
        hit = self._terrain_tracer.first_hit(
            np.array([key[0], key[1], origin_z], dtype=float),
            np.array([0.0, 0.0, -1.0], dtype=float),
        )
        if hit is None:
            raise RuntimeError("valid horizontal terrain query did not hit a surface")
        height = float(hit.point[2])
        self._surface_height_cache[key] = height
        if (
            self.max_surface_cache_entries is not None
            and len(self._surface_height_cache) > self.max_surface_cache_entries
        ):
            self._surface_height_cache.popitem(last=False)
        return height

    def clear_runtime_caches(self) -> None:
        """Drop deterministic terrain-query results without changing observations."""

        self._surface_height_cache.clear()

    def _terrain_channels(
        self, positions: FloatArray, validity: NDArray[np.bool_], attacker_z_map: float,
    ) -> FloatArray:
        values = np.full((1,) + validity.shape, np.nan, dtype=float)
        for row, column in zip(*np.nonzero(validity)):
            surface = self.terrain_surface_height_map(
                positions[row, column, 0], positions[row, column, 1],
            )
            values[0, row, column] = self.scale.distance_m(attacker_z_map - surface)
        return values

    def _hazard_channels(
        self,
        positions: FloatArray,
        validity: NDArray[np.bool_],
        heading: float,
    ) -> FloatArray:
        result = np.full((len(HAZARD_CHANNEL_NAMES),) + validity.shape, np.nan, dtype=float)
        sensor = np.asarray(self.problem.sensor_map, dtype=float)
        forward, right = self._axes(heading)
        floor_m = float(self.scene.config.detection.range_floor_m)
        for row, column in zip(*np.nonzero(validity)):
            position = positions[row, column]
            delta_m = self.scale.position_m(sensor - position)
            slant_m = float(np.linalg.norm(delta_m))
            sensor_range_m = max(slant_m, floor_m)
            los_global = delta_m / sensor_range_m
            visible = not self.scene.terrain.segment_intersects_solid(position, sensor)
            result[:, row, column] = (
                float(visible),
                float(np.dot(los_global[:2], forward)),
                float(np.dot(los_global[:2], right)),
                float(los_global[2]),
                float((floor_m / sensor_range_m) ** 4),
            )
        return result

    def build(self, state_id: int) -> StructuredObservation:
        state_id = int(state_id)
        state = self.problem.state(state_id)
        position = self.problem.position_map(state_id)
        heading = float(self.problem.grid.heading_rad(state.heading_bin))
        forward_axis, right_axis = self._axes(heading)
        surface = self.terrain_surface_height_map(position[0], position[1])
        altitude_ground_m = self.scale.distance_m(
            position[2] - float(self.scene.terrain.ground_z)
        )
        clearance_m = self.scale.distance_m(position[2] - surface)
        ego = np.asarray(
            [altitude_ground_m, clearance_m, sin(heading), cos(heading)],
            dtype=float,
        )

        goal = np.asarray(self.scene.config.goal.as_array(), dtype=float)
        delta_goal_m = self.scale.position_m(goal - position)
        goal_distance_m = float(np.linalg.norm(delta_goal_m))
        goal_features = np.asarray([
            float(np.dot(delta_goal_m[:2], forward_axis)),
            float(np.dot(delta_goal_m[:2], right_axis)),
            float(delta_goal_m[2]),
            goal_distance_m - float(self.scene.config.glider.goal_tolerance_m),
        ], dtype=float)

        positions = self.sample_positions_map(state_id)
        validity = self._inside_domain(positions)
        terrain = self._terrain_channels(positions, validity, position[2])
        hazard = self._hazard_channels(positions, validity, heading)
        return StructuredObservation(
            ego_features=ego,
            goal_features=goal_features,
            terrain_channels=terrain,
            hazard_channels=hazard,
            validity_channels=validity[None, ...].astype(float),
            metadata={
                "schema_id": self.config.schema_id,
                "state_id": state_id,
                "heading_rad": heading,
                "window_orientation": self.config.local_window_orientation,
                "ego_feature_names": list(EGO_FEATURE_NAMES),
                "goal_feature_names": list(GOAL_FEATURE_NAMES),
                "terrain_channel_names": list(TERRAIN_CHANNEL_NAMES),
                "hazard_channel_names": list(HAZARD_CHANNEL_NAMES),
                "validity_channel_names": list(VALIDITY_CHANNEL_NAMES),
            },
        )

    def to_tensor_ready(self, observation: StructuredObservation) -> TensorReadyObservation:
        """Apply only deterministic physical scaling; no dataset statistics."""

        bounds = self.problem.grid.bounds
        horizontal_diagonal_m = self.scale.distance_m(float(np.hypot(
            bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min,
        )))
        vertical_span_m = self.scale.distance_m(
            self.problem.grid.maximum_altitude_map
            - self.problem.grid.minimum_altitude_map
        )
        full_diagonal_m = float(np.hypot(horizontal_diagonal_m, vertical_span_m))
        ego = observation.ego_features.copy()
        ego[:2] = np.clip(ego[:2] / vertical_span_m, -1.0, 1.0)
        goal = observation.goal_features.copy()
        goal[:2] = np.clip(goal[:2] / horizontal_diagonal_m, -1.0, 1.0)
        goal[2] = np.clip(goal[2] / vertical_span_m, -1.0, 1.0)
        goal[3] = np.clip(goal[3] / full_diagonal_m, -1.0, 1.0)

        valid = observation.validity_channels.astype(bool)
        terrain = np.where(
            valid,
            np.clip(observation.terrain_channels / vertical_span_m, -1.0, 1.0),
            0.0,
        )
        hazard = np.where(valid, observation.hazard_channels, 0.0)
        spatial = np.concatenate(
            (terrain, hazard, observation.validity_channels), axis=0,
        )
        return TensorReadyObservation(
            scalar=np.concatenate((ego, goal)).astype(np.float32),
            spatial=spatial.astype(np.float32),
        )

    def reconstruct_hazard_rate(
        self,
        observation: StructuredObservation,
        row: int,
        column: int,
        velocity_mps: FloatArray,
    ) -> float:
        """Reconstruct the current authoritative rate from the five channels."""

        if not observation.validity_channels[0, row, column]:
            raise ValueError("cannot reconstruct hazard outside the attacker domain")
        channel = observation.hazard_channels[:, row, column]
        visible, los_forward, los_right, los_up, range_factor = map(float, channel)
        heading = float(observation.metadata["heading_rad"])
        forward, right = self._axes(heading)
        los_global = np.asarray([
            los_forward * forward[0] + los_right * right[0],
            los_forward * forward[1] + los_right * right[1],
            los_up,
        ], dtype=float)
        velocity = np.asarray(velocity_mps, dtype=float)
        radial = float(np.dot(velocity, los_global))
        speed = float(np.linalg.norm(velocity))
        cosine_aspect = float(np.clip(radial / max(speed, 1.0e-9), -1.0, 1.0))
        parameters = self.scene.config.detection
        rcs = parameters.rcs_min + (parameters.rcs_max - parameters.rcs_min) * cosine_aspect**2
        inverse_range_fourth = range_factor / parameters.range_floor_m**4
        return float(visible * inverse_range_fourth * (
            parameters.radar_rate_scale * parameters.radar_coefficient * rcs
            + parameters.radial_velocity_rate_scale * parameters.doppler_coefficient * radial**2
        ))


__all__ = [
    "EGO_FEATURE_NAMES",
    "GOAL_FEATURE_NAMES",
    "HAZARD_CHANNEL_NAMES",
    "OBSERVATION_SCHEMA_ID",
    "ObservationConfig",
    "StructuredObservation",
    "TERRAIN_CHANNEL_NAMES",
    "TerrainObservationBuilder",
    "TensorReadyObservation",
    "VALIDITY_CHANNEL_NAMES",
]

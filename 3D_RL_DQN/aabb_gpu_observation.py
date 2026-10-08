"""Exact batched CUDA observation backend for box-based Phase 16 terrains."""

from __future__ import annotations

from math import cos, sin
from typing import Iterable

import numpy as np
import torch

from terrain_observation import (
    HAZARD_CHANNEL_NAMES,
    StructuredObservation,
    TensorReadyObservation,
    TerrainObservationBuilder,
)


class ExactBatchedAABBGPUObservationBuilder:
    """Reproduce the Phase 16.2 schema with batched float64 AABB geometry.

    The authoritative terrains used by Phase 16.4 are unions of axis-aligned
    boxes over a ground plane. Geometry decisions stay in float64 and use the
    same strict-interior slab equations and 1e-9 tolerance as ``map_geometry``.
    """

    def __init__(
        self, builder: TerrainObservationBuilder,
        device: str | torch.device = "cuda", *, chunk_size: int = 1024,
    ) -> None:
        self.builder = builder
        self.problem = builder.problem
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("the exact AABB GPU backend requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        terrain = self.problem.scene.terrain
        if not hasattr(terrain, "obstacle_boxes"):
            raise TypeError("the exact AABB GPU backend requires box terrain")
        boxes = tuple(terrain.obstacle_boxes())
        if not boxes:
            raise ValueError("box terrain must contain at least one obstacle")
        bounds = [[
            float(box.x_limits[0]), float(box.y_limits[0]), float(box.base_z),
            float(box.x_limits[1]), float(box.y_limits[1]), float(box.top_z),
        ] for box in boxes]
        self.box_bounds = torch.tensor(
            bounds, dtype=torch.float64, device=self.device,
        )
        self.box_lower = self.box_bounds[:, :3]
        self.box_upper = self.box_bounds[:, 3:]
        self.box_lower_xy = self.box_lower[:, :2]
        self.box_upper_xy = self.box_upper[:, :2]
        self.box_top = self.box_upper[:, 2]
        self.ground_z = float(terrain.ground_z)
        self.tolerance = 1.0e-9

    def _surface_height(self, positions: torch.Tensor) -> torch.Tensor:
        """Highest ground/box top under ``(..., 3)`` positions."""

        xy = torch.round(positions[..., :2] * 1.0e12) / 1.0e12
        inside = torch.all(
            (xy[..., None, :] >= self.box_lower_xy)
            & (xy[..., None, :] <= self.box_upper_xy),
            dim=-1,
        )
        candidate_heights = torch.where(
            inside, self.box_top, torch.as_tensor(self.ground_z, device=self.device),
        )
        return torch.maximum(
            candidate_heights.amax(dim=-1),
            torch.as_tensor(self.ground_z, device=self.device),
        )

    def _segments_intersect_solid(
        self, starts: torch.Tensor, ends: torch.Tensor,
    ) -> torch.Tensor:
        """Batched equivalent of ``segments_intersect_solid_many``."""

        starts_by_box = starts[..., None, :]
        displacement = (ends - starts)[..., None, :]
        tolerance = self.tolerance
        lower = self.box_lower + tolerance
        upper = self.box_upper - tolerance
        parallel = torch.abs(displacement) <= tolerance
        parallel_outside = parallel & ~(
            (lower < starts_by_box) & (starts_by_box < upper)
        )
        safe_displacement = torch.where(
            parallel, torch.ones_like(displacement), displacement,
        )
        first = (lower - starts_by_box) / safe_displacement
        second = (upper - starts_by_box) / safe_displacement
        axis_entry = torch.where(
            parallel, -torch.inf, torch.minimum(first, second),
        )
        axis_exit = torch.where(
            parallel, torch.inf, torch.maximum(first, second),
        )
        maximum_entry = torch.max(axis_entry, dim=-1).values
        entry = torch.maximum(torch.zeros_like(maximum_entry), maximum_entry)
        exit_ = torch.minimum(
            torch.ones_like(maximum_entry),
            torch.min(axis_exit, dim=-1).values,
        )
        intersects = (
            ~torch.any(parallel_outside, dim=-1)
            & (exit_ > entry)
            & (maximum_entry < 1.0)
        )
        return torch.any(intersects, dim=-1)

    def _chunk(self, state_ids: tuple[int, ...]) -> list[TensorReadyObservation]:
        base = self.builder
        height, width = base.config.local_window_shape_cells
        positions_np = np.stack([
            base.sample_positions_map(state_id) for state_id in state_ids
        ], axis=0)
        state_positions = np.stack([
            self.problem.position_map(state_id) for state_id in state_ids
        ], axis=0)
        headings = np.asarray([
            float(self.problem.grid.heading_rad(
                self.problem.state(state_id).heading_bin
            ))
            for state_id in state_ids
        ], dtype=float)
        positions = torch.as_tensor(
            positions_np, dtype=torch.float64, device=self.device,
        )
        state_position_tensor = torch.as_tensor(
            state_positions, dtype=torch.float64, device=self.device,
        )
        bounds = self.problem.grid.bounds
        validity = (
            (positions[..., 0] >= float(bounds.x_min))
            & (positions[..., 0] <= float(bounds.x_max))
            & (positions[..., 1] >= float(bounds.y_min))
            & (positions[..., 1] <= float(bounds.y_max))
        )

        surface = self._surface_height(positions)
        # The approved observation window has odd dimensions, so its center is
        # exactly the attacker horizontal position.  Reuse the surface value
        # already computed for that sample instead of launching a second CUDA
        # terrain query for every batch.
        attacker_surface = surface[:, height // 2, width // 2]
        scale = float(base.scale.meters_per_map_unit)
        attacker_z = state_position_tensor[:, 2, None, None]
        terrain_channel = (attacker_z - surface) * scale

        sensor = torch.tensor(
            self.problem.sensor_map, dtype=torch.float64, device=self.device,
        )
        sensor_ends = sensor.view(1, 1, 1, 3).expand_as(positions)
        delta_m = (sensor_ends - positions) * scale
        slant_m = torch.linalg.vector_norm(delta_m, dim=-1)
        floor_m = float(self.problem.scene.config.detection.range_floor_m)
        sensor_range_m = torch.clamp(slant_m, min=floor_m)
        los_global = delta_m / sensor_range_m[..., None]
        forward = torch.as_tensor(
            np.stack((np.cos(headings), np.sin(headings)), axis=1),
            dtype=torch.float64, device=self.device,
        )
        right = torch.as_tensor(
            np.stack((np.sin(headings), -np.cos(headings)), axis=1),
            dtype=torch.float64, device=self.device,
        )
        visible = ~self._segments_intersect_solid(positions, sensor_ends)
        los_forward = torch.sum(
            los_global[..., :2] * forward[:, None, None, :], dim=-1,
        )
        los_right = torch.sum(
            los_global[..., :2] * right[:, None, None, :], dim=-1,
        )
        range_factor = (floor_m / sensor_range_m) ** 4
        hazard = torch.stack((
            visible.to(torch.float64), los_forward, los_right,
            los_global[..., 2], range_factor,
        ), dim=1)

        terrain_np = terrain_channel[:, None].cpu().numpy()
        hazard_np = hazard.cpu().numpy()
        validity_np = validity[:, None].cpu().numpy().astype(float)
        attacker_surface_np = attacker_surface.cpu().numpy()
        goal = np.asarray(self.problem.scene.config.goal.as_array(), dtype=float)
        outputs: list[TensorReadyObservation] = []
        for index, (state_id, heading) in enumerate(zip(state_ids, headings)):
            position = state_positions[index]
            forward_axis = np.array([cos(float(heading)), sin(float(heading))])
            right_axis = np.array([sin(float(heading)), -cos(float(heading))])
            altitude_ground_m = scale * (position[2] - self.ground_z)
            clearance_m = scale * (position[2] - attacker_surface_np[index])
            ego = np.asarray([
                altitude_ground_m, clearance_m,
                sin(float(heading)), cos(float(heading)),
            ], dtype=float)
            delta_goal_m = scale * (goal - position)
            goal_distance_m = float(np.linalg.norm(delta_goal_m))
            goal_features = np.asarray([
                float(np.dot(delta_goal_m[:2], forward_axis)),
                float(np.dot(delta_goal_m[:2], right_axis)),
                float(delta_goal_m[2]),
                goal_distance_m
                - float(self.problem.scene.config.glider.goal_tolerance_m),
            ], dtype=float)
            raw = StructuredObservation(
                ego_features=ego,
                goal_features=goal_features,
                terrain_channels=terrain_np[index],
                hazard_channels=hazard_np[index],
                validity_channels=validity_np[index],
                metadata={
                    "schema_id": base.config.schema_id,
                    "state_id": int(state_id),
                    "heading_rad": float(heading),
                    "window_orientation": base.config.local_window_orientation,
                },
            )
            outputs.append(base.to_tensor_ready(raw))
        return outputs

    def build_many(
        self, state_ids: Iterable[int],
    ) -> list[TensorReadyObservation]:
        ids = tuple(int(value) for value in state_ids)
        if not ids:
            return []
        result: list[TensorReadyObservation] = []
        for offset in range(0, len(ids), self.chunk_size):
            result.extend(self._chunk(ids[offset:offset + self.chunk_size]))
        return result


__all__ = ["ExactBatchedAABBGPUObservationBuilder"]

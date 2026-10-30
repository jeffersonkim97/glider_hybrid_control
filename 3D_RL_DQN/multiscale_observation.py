"""Local-plus-global terrain observation for the extended Phase 16.4 pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from attacker_br_problem import AttackerBRProblem
from terrain_observation import (
    ObservationConfig,
    StructuredObservation,
    TensorReadyObservation,
    TerrainObservationBuilder,
)


MULTISCALE_OBSERVATION_SCHEMA_ID = "p1b-terrain-observation-multiscale-v1"


@dataclass(frozen=True)
class MultiscaleObservationConfig:
    local_extent_m: float = 1000.0
    local_shape_cells: tuple[int, int] = (41, 41)
    global_extent_m: float = 2000.0
    global_shape_cells: tuple[int, int] = (21, 21)
    orientation: str = "heading_aligned"
    schema_id: str = MULTISCALE_OBSERVATION_SCHEMA_ID

    def __post_init__(self) -> None:
        local = ObservationConfig(
            local_window_extent_m=self.local_extent_m,
            local_window_shape_cells=self.local_shape_cells,
            local_window_orientation=self.orientation,
        )
        global_ = ObservationConfig(
            local_window_extent_m=self.global_extent_m,
            local_window_shape_cells=self.global_shape_cells,
            local_window_orientation=self.orientation,
        )
        if self.schema_id != MULTISCALE_OBSERVATION_SCHEMA_ID:
            raise ValueError("unsupported multiscale observation schema")
        if any(
            target < source
            for target, source in zip(local.local_window_shape_cells,
                                      global_.local_window_shape_cells)
        ):
            raise ValueError("local fusion grid must be at least as large as global grid")

    @property
    def local_sample_spacing_m(self) -> tuple[float, float]:
        return ObservationConfig(
            self.local_extent_m, self.local_shape_cells, self.orientation,
        ).sample_spacing_m

    @property
    def global_sample_spacing_m(self) -> tuple[float, float]:
        return ObservationConfig(
            self.global_extent_m, self.global_shape_cells, self.orientation,
        ).sample_spacing_m

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["local_shape_cells"] = list(self.local_shape_cells)
        result["global_shape_cells"] = list(self.global_shape_cells)
        result["local_sample_spacing_m"] = list(self.local_sample_spacing_m)
        result["global_sample_spacing_m"] = list(self.global_sample_spacing_m)
        result["fusion"] = (
            "nearest-coordinate resample global channels onto the local grid, then "
            "concatenate local and global channels"
        )
        result["parameter_origin"] = "user_approved_multiscale_2026-10-19"
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MultiscaleObservationConfig":
        copied = dict(values)
        for key in (
            "local_sample_spacing_m", "global_sample_spacing_m", "fusion",
            "parameter_origin",
        ):
            copied.pop(key, None)
        copied["local_shape_cells"] = tuple(copied["local_shape_cells"])
        copied["global_shape_cells"] = tuple(copied["global_shape_cells"])
        return cls(**copied)


@dataclass(frozen=True)
class MultiscaleStructuredObservation:
    local: StructuredObservation
    global_: StructuredObservation


def _resample_nearest(
    values: np.ndarray, target_shape: tuple[int, int],
) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError("spatial values must have shape (channels, height, width)")
    source_height, source_width = values.shape[1:]
    target_height, target_width = target_shape
    rows = np.rint(np.linspace(0, source_height - 1, target_height)).astype(int)
    columns = np.rint(np.linspace(0, source_width - 1, target_width)).astype(int)
    return values[:, rows][:, :, columns]


def fuse_tensor_ready(
    local: TensorReadyObservation,
    global_: TensorReadyObservation,
    target_shape: tuple[int, int],
) -> TensorReadyObservation:
    np.testing.assert_array_equal(local.scalar, global_.scalar)
    global_resampled = _resample_nearest(global_.spatial, target_shape)
    return TensorReadyObservation(
        scalar=local.scalar.copy(),
        spatial=np.concatenate((local.spatial, global_resampled), axis=0).astype(
            np.float32, copy=False,
        ),
    )


class MultiscaleTerrainObservationBuilder:
    """Reuse the authoritative Phase 16.2 channels at two physical scales."""

    def __init__(
        self,
        problem: AttackerBRProblem,
        config: MultiscaleObservationConfig = MultiscaleObservationConfig(),
        **builder_kwargs: Any,
    ) -> None:
        self.problem = problem
        self.config = config
        self.local_builder = TerrainObservationBuilder(
            problem,
            ObservationConfig(
                local_window_extent_m=config.local_extent_m,
                local_window_shape_cells=config.local_shape_cells,
                local_window_orientation=config.orientation,
            ),
            **builder_kwargs,
        )
        self.global_builder = TerrainObservationBuilder(
            problem,
            ObservationConfig(
                local_window_extent_m=config.global_extent_m,
                local_window_shape_cells=config.global_shape_cells,
                local_window_orientation=config.orientation,
            ),
            **builder_kwargs,
        )

    @property
    def tensor_ready_shapes(self) -> dict[str, tuple[int, ...]]:
        local = self.local_builder.tensor_ready_shapes
        global_ = self.global_builder.tensor_ready_shapes
        return {
            "scalar": local["scalar"],
            "spatial": (
                local["spatial"][0] + global_["spatial"][0],
                self.config.local_shape_cells[0],
                self.config.local_shape_cells[1],
            ),
        }

    def build(self, state_id: int) -> MultiscaleStructuredObservation:
        return MultiscaleStructuredObservation(
            local=self.local_builder.build(int(state_id)),
            global_=self.global_builder.build(int(state_id)),
        )

    def to_tensor_ready(
        self, observation: MultiscaleStructuredObservation,
    ) -> TensorReadyObservation:
        local = self.local_builder.to_tensor_ready(observation.local)
        global_ = self.global_builder.to_tensor_ready(observation.global_)
        return fuse_tensor_ready(local, global_, self.config.local_shape_cells)

    def clear_runtime_caches(self) -> None:
        self.local_builder.clear_runtime_caches()
        self.global_builder.clear_runtime_caches()


class ExactBatchedAABBGPUObservationMultiscaleBuilder:
    """Exact CUDA batching for both scales followed by deterministic fusion."""

    def __init__(
        self,
        builder: MultiscaleTerrainObservationBuilder,
        device: Any,
        *,
        chunk_size: int = 1024,
    ) -> None:
        # Keep CPU dataset/visualization tools independent of CUDA DLL loading.
        # Training requests this adapter explicitly and pays the torch import then.
        from aabb_gpu_observation import ExactBatchedAABBGPUObservationBuilder

        self.builder = builder
        self.local = ExactBatchedAABBGPUObservationBuilder(
            builder.local_builder, device, chunk_size=chunk_size,
        )
        self.global_ = ExactBatchedAABBGPUObservationBuilder(
            builder.global_builder, device, chunk_size=chunk_size,
        )

    def build_many(
        self, state_ids: Iterable[int],
    ) -> list[TensorReadyObservation]:
        ids = tuple(int(value) for value in state_ids)
        local_values = self.local.build_many(ids)
        global_values = self.global_.build_many(ids)
        return [
            fuse_tensor_ready(local, global_, self.builder.config.local_shape_cells)
            for local, global_ in zip(local_values, global_values)
        ]


__all__ = [
    "MULTISCALE_OBSERVATION_SCHEMA_ID",
    "ExactBatchedAABBGPUObservationMultiscaleBuilder",
    "MultiscaleObservationConfig",
    "MultiscaleStructuredObservation",
    "MultiscaleTerrainObservationBuilder",
    "fuse_tensor_ready",
]

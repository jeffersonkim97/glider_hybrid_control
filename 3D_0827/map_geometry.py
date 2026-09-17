"""Geometry definitions for the simplified 3D LOS study.

Stage 1 intentionally contains only a finite ground plane and one axis-aligned
cube.  The geometry is kept independent of Plotly so later LOS and energy
modules can use the same map without depending on the visualization layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

if TYPE_CHECKING:
    from ray_tracing import RayHit


@runtime_checkable
class TerrainModel(Protocol):
    """Terrain-independent geometry contract used by mission solvers.

    ``contains_solid`` describes strict obstacle-volume occupancy. The ground
    is an admissibility boundary handled separately through ``ground_z``;
    surface contact is therefore not solid occupancy. Positive tolerances
    shrink obstacle interiors, allowing exact terrain-tangent contact while
    still detecting segments that spend nonzero length inside the terrain.
    """

    bounds: "MapBounds"
    ground_z: float

    @property
    def maximum_height(self) -> float: ...

    def ground_mesh(self) -> "TriangleMesh": ...

    def obstacle_mesh(self) -> "TriangleMesh": ...

    def surface_meshes(self) -> tuple["TriangleMesh", ...]: ...

    def contains_solid(
        self,
        point: FloatArray,
        *,
        tolerance: float = 0.0,
    ) -> bool: ...

    def segment_intersects_solid(
        self,
        start: FloatArray,
        end: FloatArray,
        *,
        tolerance: float = 1.0e-9,
    ) -> bool: ...

    def first_ray_hit(
        self,
        origin: FloatArray,
        direction: FloatArray,
        *,
        include_ground: bool = True,
    ) -> "RayHit | None": ...


# Backward-compatible name for renderer-facing annotations. There is only one
# protocol; new mission/solver code should use ``TerrainModel``.
TerrainMap3D = TerrainModel


def _finite_point(values: FloatArray, name: str) -> FloatArray:
    point = np.asarray(values, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must contain three finite coordinates")
    return point


def _validated_tolerance(tolerance: float) -> float:
    value = float(tolerance)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("terrain tolerance must be finite and nonnegative")
    return value


def _box_contains_strict(
    box: "BoxObstacle",
    point: FloatArray,
    tolerance: float,
) -> bool:
    lower = np.array(
        [box.x_limits[0], box.y_limits[0], box.base_z], dtype=float,
    ) + tolerance
    upper = np.array(
        [box.x_limits[1], box.y_limits[1], box.top_z], dtype=float,
    ) - tolerance
    return bool(np.all(lower < point) and np.all(point < upper))


def _segment_intersects_box_interior(
    start: FloatArray,
    end: FloatArray,
    box: "BoxObstacle",
    tolerance: float,
) -> bool:
    """Return whether a closed segment spends nonzero length inside a box."""
    lower = np.array(
        [box.x_limits[0], box.y_limits[0], box.base_z], dtype=float,
    ) + tolerance
    upper = np.array(
        [box.x_limits[1], box.y_limits[1], box.top_z], dtype=float,
    ) - tolerance
    if np.any(lower >= upper):
        return False

    displacement = end - start
    entry, exit_ = 0.0, 1.0
    for axis in range(3):
        if abs(displacement[axis]) <= tolerance:
            if not lower[axis] < start[axis] < upper[axis]:
                return False
            continue
        first = (lower[axis] - start[axis]) / displacement[axis]
        second = (upper[axis] - start[axis]) / displacement[axis]
        axis_entry, axis_exit = sorted((float(first), float(second)))
        entry = max(entry, axis_entry)
        exit_ = min(exit_, axis_exit)
        if entry >= exit_:
            return False
    return exit_ > max(entry, 0.0) and entry < 1.0


class _BoxTerrainQueries:
    """Private box-backed implementation of the public terrain contract."""

    def contains_solid_many(
        self,
        points: FloatArray,
        *,
        tolerance: float = 0.0,
    ) -> NDArray[np.bool_]:
        """Optional vectorized acceleration; scalar TerrainModel stays valid."""
        point_array = np.asarray(points, dtype=float)
        if point_array.ndim != 2 or point_array.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        if not np.all(np.isfinite(point_array)):
            raise ValueError("points must be finite")
        tolerance_value = _validated_tolerance(tolerance)
        result = np.zeros(len(point_array), dtype=bool)
        for box in self.obstacle_boxes():
            lower = np.array(
                [box.x_limits[0], box.y_limits[0], box.base_z], dtype=float,
            ) + tolerance_value
            upper = np.array(
                [box.x_limits[1], box.y_limits[1], box.top_z], dtype=float,
            ) - tolerance_value
            if np.any(lower >= upper):
                continue
            result |= np.all(
                (lower[None, :] < point_array)
                & (point_array < upper[None, :]),
                axis=1,
            )
        return result

    def segments_intersect_solid_many(
        self,
        starts: FloatArray,
        ends: FloatArray,
        *,
        tolerance: float = 1.0e-9,
    ) -> NDArray[np.bool_]:
        """Vectorized strict-interior slab test for optional sparse solvers."""
        start_array = np.asarray(starts, dtype=float)
        end_array = np.asarray(ends, dtype=float)
        if (
            start_array.ndim != 2
            or start_array.shape[1] != 3
            or end_array.shape != start_array.shape
        ):
            raise ValueError("starts and ends must share shape (n, 3)")
        if not np.all(np.isfinite(start_array)) or not np.all(np.isfinite(end_array)):
            raise ValueError("segment endpoints must be finite")
        tolerance_value = _validated_tolerance(tolerance)
        result = np.zeros(len(start_array), dtype=bool)
        displacement = end_array - start_array
        for box in self.obstacle_boxes():
            lower = np.array(
                [box.x_limits[0], box.y_limits[0], box.base_z], dtype=float,
            ) + tolerance_value
            upper = np.array(
                [box.x_limits[1], box.y_limits[1], box.top_z], dtype=float,
            ) - tolerance_value
            if np.any(lower >= upper):
                continue
            parallel = np.abs(displacement) <= tolerance_value
            parallel_outside = parallel & ~(
                (lower[None, :] < start_array)
                & (start_array < upper[None, :])
            )
            safe_displacement = np.where(parallel, 1.0, displacement)
            first = (lower[None, :] - start_array) / safe_displacement
            second = (upper[None, :] - start_array) / safe_displacement
            axis_entry = np.where(parallel, -np.inf, np.minimum(first, second))
            axis_exit = np.where(parallel, np.inf, np.maximum(first, second))
            entry = np.maximum(0.0, np.max(axis_entry, axis=1))
            exit_ = np.minimum(1.0, np.min(axis_exit, axis=1))
            result |= (
                ~np.any(parallel_outside, axis=1)
                & (exit_ > entry)
                & (np.max(axis_entry, axis=1) < 1.0)
            )
        return result

    def contains_solid(
        self,
        point: FloatArray,
        *,
        tolerance: float = 0.0,
    ) -> bool:
        point_array = _finite_point(point, "point")
        tolerance_value = _validated_tolerance(tolerance)
        return any(
            _box_contains_strict(box, point_array, tolerance_value)
            for box in self.obstacle_boxes()
        )

    def segment_intersects_solid(
        self,
        start: FloatArray,
        end: FloatArray,
        *,
        tolerance: float = 1.0e-9,
    ) -> bool:
        start_array = _finite_point(start, "start")
        end_array = _finite_point(end, "end")
        tolerance_value = _validated_tolerance(tolerance)
        return any(
            _segment_intersects_box_interior(
                start_array, end_array, box, tolerance_value,
            )
            for box in self.obstacle_boxes()
        )

    def first_ray_hit(
        self,
        origin: FloatArray,
        direction: FloatArray,
        *,
        include_ground: bool = True,
    ) -> "RayHit | None":
        # Local import avoids a module cycle: the low-level tracer consumes
        # TriangleMesh, while this terrain method delegates back to the tracer.
        from ray_tracing import TriangleRayTracer

        meshes = (
            self.surface_meshes()
            if include_ground
            else (self.obstacle_mesh(),)
        )
        return TriangleRayTracer(meshes).first_hit(origin, direction)


@dataclass(frozen=True)
class MapBounds:
    """Finite plotting/computation bounds for the horizontal ground plane."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be smaller than x_max")
        if self.y_min >= self.y_max:
            raise ValueError("y_min must be smaller than y_max")


@dataclass(frozen=True)
class TriangleMesh:
    """Renderer-independent triangular surface mesh."""

    vertices: FloatArray
    triangles: IntArray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        triangles = np.asarray(self.triangles, dtype=np.int64)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n_vertices, 3)")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (n_triangles, 3)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must be finite")
        if triangles.size and (
            np.min(triangles) < 0 or np.max(triangles) >= len(vertices)
        ):
            raise ValueError("triangle indices must reference existing vertices")

        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)


@dataclass(frozen=True)
class BoxObstacle:
    """Axis-aligned rectangular prism used as a modular terrain building block."""

    center_x: float
    center_y: float
    width_x: float
    width_y: float
    height: float
    base_z: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.center_x,
            self.center_y,
            self.width_x,
            self.width_y,
            self.height,
            self.base_z,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("box parameters must be finite")
        if self.width_x <= 0.0 or self.width_y <= 0.0 or self.height <= 0.0:
            raise ValueError("box width and height values must be positive")

    @property
    def top_z(self) -> float:
        return self.base_z + self.height

    @property
    def x_limits(self) -> tuple[float, float]:
        return (
            self.center_x - 0.5 * self.width_x,
            self.center_x + 0.5 * self.width_x,
        )

    @property
    def y_limits(self) -> tuple[float, float]:
        return (
            self.center_y - 0.5 * self.width_y,
            self.center_y + 0.5 * self.width_y,
        )

    def mesh(self) -> TriangleMesh:
        x0, x1 = self.x_limits
        y0, y1 = self.y_limits
        z0, z1 = self.base_z, self.top_z
        vertices = np.array(
            [
                [x0, y0, z0], [x1, y0, z0],
                [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1],
                [x1, y1, z1], [x0, y1, z1],
            ],
            dtype=float,
        )
        triangles = np.array(
            [
                [0, 2, 1], [0, 3, 2],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7],
            ],
            dtype=np.int64,
        )
        return TriangleMesh(vertices=vertices, triangles=triangles)

    def edges(self) -> IntArray:
        return np.array(
            [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ],
            dtype=np.int64,
        )


@dataclass(frozen=True)
class CubeObstacle:
    """Axis-aligned cube resting on the ground plane."""

    center_x: float
    center_y: float
    side_length: float
    base_z: float = 0.0

    def __post_init__(self) -> None:
        values = (self.center_x, self.center_y, self.side_length, self.base_z)
        if not np.all(np.isfinite(values)):
            raise ValueError("cube parameters must be finite")
        if self.side_length <= 0.0:
            raise ValueError("side_length must be positive")

    @property
    def top_z(self) -> float:
        return self.base_z + self.side_length

    def mesh(self) -> TriangleMesh:
        """Return the cube as eight vertices and twelve triangular faces."""
        half_side = 0.5 * self.side_length
        x0, x1 = self.center_x - half_side, self.center_x + half_side
        y0, y1 = self.center_y - half_side, self.center_y + half_side
        z0, z1 = self.base_z, self.top_z

        vertices = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype=float,
        )
        triangles = np.array(
            [
                [0, 2, 1], [0, 3, 2],  # bottom
                [4, 5, 6], [4, 6, 7],  # top
                [0, 1, 5], [0, 5, 4],  # front
                [1, 2, 6], [1, 6, 5],  # right
                [2, 3, 7], [2, 7, 6],  # back
                [3, 0, 4], [3, 4, 7],  # left
            ],
            dtype=np.int64,
        )
        return TriangleMesh(vertices=vertices, triangles=triangles)

    def edges(self) -> IntArray:
        """Return the twelve cube edges as vertex-index pairs."""
        return np.array(
            [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ],
            dtype=np.int64,
        )

    def as_box(self) -> BoxObstacle:
        return BoxObstacle(
            center_x=self.center_x,
            center_y=self.center_y,
            width_x=self.side_length,
            width_y=self.side_length,
            height=self.side_length,
            base_z=self.base_z,
        )


@dataclass(frozen=True)
class PlaneCubeMap(_BoxTerrainQueries):
    """Stage-1 map: an empty horizontal plane with one cube obstacle."""

    bounds: MapBounds
    cube: CubeObstacle
    ground_z: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.ground_z):
            raise ValueError("ground_z must be finite")
        if not np.isclose(self.cube.base_z, self.ground_z):
            raise ValueError("the cube base must coincide with the ground plane")

        half_side = 0.5 * self.cube.side_length
        if not (
            self.bounds.x_min <= self.cube.center_x - half_side
            and self.cube.center_x + half_side <= self.bounds.x_max
            and self.bounds.y_min <= self.cube.center_y - half_side
            and self.cube.center_y + half_side <= self.bounds.y_max
        ):
            raise ValueError("the cube footprint must lie inside the map bounds")

    def ground_mesh(self) -> TriangleMesh:
        """Return the bounded ground plane as two triangles."""
        z = self.ground_z
        vertices = np.array(
            [
                [self.bounds.x_min, self.bounds.y_min, z],
                [self.bounds.x_max, self.bounds.y_min, z],
                [self.bounds.x_max, self.bounds.y_max, z],
                [self.bounds.x_min, self.bounds.y_max, z],
            ],
            dtype=float,
        )
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        return TriangleMesh(vertices=vertices, triangles=triangles)

    @property
    def maximum_height(self) -> float:
        return self.cube.top_z

    def obstacle_boxes(self) -> tuple[BoxObstacle, ...]:
        return (self.cube.as_box(),)

    def obstacle_mesh(self) -> TriangleMesh:
        return self.cube.mesh()

    def surface_meshes(self) -> tuple[TriangleMesh, ...]:
        """Return every map surface through one future-facing interface."""
        return self.ground_mesh(), self.obstacle_mesh()


@dataclass(frozen=True)
class CompositeTerrainMap(_BoxTerrainQueries):
    """Ground plane plus one or more box-based terrain components."""

    bounds: MapBounds
    boxes: tuple[BoxObstacle, ...]
    ground_z: float = 0.0
    category_id: str = "composite"

    def __post_init__(self) -> None:
        if not self.boxes:
            raise ValueError("a composite terrain requires at least one box")
        if not np.isfinite(self.ground_z):
            raise ValueError("ground_z must be finite")
        for box in self.boxes:
            x0, x1 = box.x_limits
            y0, y1 = box.y_limits
            if not (
                self.bounds.x_min <= x0 <= x1 <= self.bounds.x_max
                and self.bounds.y_min <= y0 <= y1 <= self.bounds.y_max
            ):
                raise ValueError("every box footprint must lie inside the map bounds")
            if box.base_z < self.ground_z:
                raise ValueError("terrain boxes cannot begin below the ground plane")

    @property
    def maximum_height(self) -> float:
        return max(box.top_z for box in self.boxes)

    def ground_mesh(self) -> TriangleMesh:
        z = self.ground_z
        vertices = np.array(
            [
                [self.bounds.x_min, self.bounds.y_min, z],
                [self.bounds.x_max, self.bounds.y_min, z],
                [self.bounds.x_max, self.bounds.y_max, z],
                [self.bounds.x_min, self.bounds.y_max, z],
            ],
            dtype=float,
        )
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        return TriangleMesh(vertices=vertices, triangles=triangles)

    def obstacle_boxes(self) -> tuple[BoxObstacle, ...]:
        return self.boxes

    def obstacle_mesh(self) -> TriangleMesh:
        return combine_triangle_meshes(tuple(box.mesh() for box in self.boxes))

    def surface_meshes(self) -> tuple[TriangleMesh, ...]:
        return self.ground_mesh(), self.obstacle_mesh()


def combine_triangle_meshes(meshes: tuple[TriangleMesh, ...]) -> TriangleMesh:
    """Combine independent meshes while preserving every triangular surface."""
    if not meshes:
        raise ValueError("at least one mesh is required")
    vertices: list[FloatArray] = []
    triangles: list[IntArray] = []
    vertex_offset = 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        triangles.append(mesh.triangles + vertex_offset)
        vertex_offset += len(mesh.vertices)
    return TriangleMesh(
        vertices=np.vstack(vertices),
        triangles=np.vstack(triangles),
    )

"""Classify a displayed LOS tangent surface by powered/glide reachability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from energy_model import (
    DEFAULT_GLIDER,
    DEFAULT_PHYSICAL_SCALE,
    GliderParameters,
    PhysicalScale,
    StraightPoweredPhaseModel,
    SwitchingState,
)
from glide_reachability import BoundedTurnGlideModel, GlideReachabilityResult
from los_geometry import LOSTangentSurface
from map_geometry import TerrainMap3D, TriangleMesh
from scenario import MissionPoints


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PointReachability:
    """Complete powered and glide result at one LOS-surface point."""

    switching_state: SwitchingState
    glide_result: GlideReachabilityResult

    @property
    def reachable(self) -> bool:
        return self.glide_result.reachable


@dataclass(frozen=True)
class EnergyRegionMesh:
    """One zero-margin-clipped colored region with interpolated hover data."""

    vertices: FloatArray
    triangles: IntArray
    available_energy_j: FloatArray
    required_energy_j: FloatArray
    energy_margin_j: FloatArray
    height_margin_m: FloatArray

    def __post_init__(self) -> None:
        vertex_count = len(self.vertices)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("region vertices must have shape (n, 3)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("region triangles must have shape (m, 3)")
        for values in (
            self.available_energy_j,
            self.required_energy_j,
            self.energy_margin_j,
            self.height_margin_m,
        ):
            if values.shape != (vertex_count,):
                raise ValueError("region hover fields must match the vertex count")

    @property
    def surface_area(self) -> float:
        if not len(self.triangles):
            return 0.0
        triangle_vertices = self.vertices[self.triangles]
        doubled_areas = np.linalg.norm(
            np.cross(
                triangle_vertices[:, 1] - triangle_vertices[:, 0],
                triangle_vertices[:, 2] - triangle_vertices[:, 0],
            ),
            axis=1,
        )
        return 0.5 * float(np.sum(doubled_areas))

    def triangle_mesh(self) -> TriangleMesh:
        return TriangleMesh(vertices=self.vertices, triangles=self.triangles)


@dataclass(frozen=True)
class EnergyReachabilitySurface:
    """Finite visualization mesh sampled from the continuous LOS surface."""

    vertices: FloatArray
    triangles: IntArray
    vertex_reachable: BoolArray
    vertex_powered_feasible: BoolArray
    vertex_available_energy_j: FloatArray
    vertex_required_energy_j: FloatArray
    vertex_energy_margin_j: FloatArray
    vertex_height_margin_m: FloatArray
    reachable_region: EnergyRegionMesh
    unreachable_region: EnergyRegionMesh
    radial_scales: FloatArray
    contour_count: int

    def __post_init__(self) -> None:
        vertex_count = len(self.vertices)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (m, 3)")
        for values in (
            self.vertex_reachable,
            self.vertex_powered_feasible,
            self.vertex_available_energy_j,
            self.vertex_required_energy_j,
            self.vertex_energy_margin_j,
            self.vertex_height_margin_m,
        ):
            if values.shape != (vertex_count,):
                raise ValueError("every vertex field must match the vertex count")
    @property
    def reachable_face_count(self) -> int:
        return len(self.reachable_region.triangles)

    @property
    def unreachable_face_count(self) -> int:
        return len(self.unreachable_region.triangles)

    @property
    def reachable_fraction(self) -> float:
        total_area = self.reachable_region.surface_area + self.unreachable_region.surface_area
        if total_area <= 0.0:
            return 0.0
        return self.reachable_region.surface_area / total_area

    @property
    def powered_infeasible_vertex_count(self) -> int:
        return int(np.count_nonzero(~self.vertex_powered_feasible))

    def reachable_mesh(self) -> TriangleMesh:
        return self.reachable_region.triangle_mesh()

    def unreachable_mesh(self) -> TriangleMesh:
        return self.unreachable_region.triangle_mesh()


@dataclass(frozen=True)
class _ClipVertex:
    position: FloatArray
    available_energy_j: float
    required_energy_j: float
    energy_margin_j: float
    height_margin_m: float
    clip_margin_m: float


class LOSSurfaceReachabilityClassifier:
    """Composition root for independent powered and glide modules."""

    def __init__(
        self,
        parameters: GliderParameters = DEFAULT_GLIDER,
        physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    ) -> None:
        self.parameters = parameters
        self.physical_scale = physical_scale
        self.powered_model = StraightPoweredPhaseModel(parameters, physical_scale)
        self.glide_model = BoundedTurnGlideModel(parameters)

    def evaluate_point(
        self,
        point_map: FloatArray,
        terrain_map: TerrainMap3D,
        mission_points: MissionPoints,
    ) -> PointReachability:
        switching_state = self.powered_model.state_at(
            point_map,
            mission_points,
            terrain_map,
        )
        glide_result = self.glide_model.evaluate(
            switching_state,
            self.physical_scale.position_m(mission_points.goal),
        )
        return PointReachability(switching_state, glide_result)

    def classify_display_surface(
        self,
        los_surface: LOSTangentSurface,
        terrain_map: TerrainMap3D,
        mission_points: MissionPoints,
        *,
        radial_section_count: int = 25,
        contour_section_count: int = 96,
    ) -> EnergyReachabilitySurface:
        """Sample and classify the finite Plotly clip of an unbounded surface."""
        if radial_section_count < 3:
            raise ValueError("radial_section_count must be at least three")
        if contour_section_count < 12:
            raise ValueError("contour_section_count must be at least twelve")

        contour = los_surface.tangent_contour
        contour_count = contour_section_count
        contour_fractions = np.linspace(
            0.0,
            1.0,
            contour_count,
            endpoint=not contour.closed,
            dtype=float,
        )
        contour_vectors = np.vstack(
            [
                contour.tangent_vector_at(float(fraction))
                for fraction in contour_fractions
            ]
        )
        radial_scales = np.linspace(
            0.0,
            los_surface.display_extension_factor,
            radial_section_count,
            dtype=float,
        )

        vertices: list[FloatArray] = [contour.origin.as_array()]
        for scale in radial_scales[1:]:
            vertices.extend(
                contour.origin.as_array() + float(scale) * vector
                for vector in contour_vectors
            )
        vertex_array = np.vstack(vertices)

        triangles: list[tuple[int, int, int]] = []
        contour_pairs = [
            (index, index + 1) for index in range(contour_count - 1)
        ]
        if contour.closed:
            contour_pairs.append((contour_count - 1, 0))
        first_ring_start = 1
        for contour_index, next_index in contour_pairs:
            triangles.append(
                (0, first_ring_start + contour_index, first_ring_start + next_index)
            )
        for radial_index in range(1, radial_section_count - 1):
            inner_start = 1 + (radial_index - 1) * contour_count
            outer_start = 1 + radial_index * contour_count
            for contour_index, next_index in contour_pairs:
                triangles.append(
                    (
                        inner_start + contour_index,
                        outer_start + contour_index,
                        outer_start + next_index,
                    )
                )
                triangles.append(
                    (
                        inner_start + contour_index,
                        outer_start + next_index,
                        inner_start + next_index,
                    )
                )
        triangle_array = np.asarray(triangles, dtype=np.int64)

        point_results = tuple(
            self.evaluate_point(point, terrain_map, mission_points)
            for point in vertex_array
        )
        vertex_reachable = np.fromiter(
            (result.reachable for result in point_results),
            dtype=bool,
            count=len(point_results),
        )
        vertex_powered_feasible = np.fromiter(
            (result.switching_state.powered_feasible for result in point_results),
            dtype=bool,
            count=len(point_results),
        )
        energy_margin = np.fromiter(
            (result.glide_result.energy_margin_j for result in point_results),
            dtype=float,
            count=len(point_results),
        )
        available_energy = np.fromiter(
            (result.glide_result.available_energy_j for result in point_results),
            dtype=float,
            count=len(point_results),
        )
        required_energy = np.fromiter(
            (result.glide_result.required_energy_j for result in point_results),
            dtype=float,
            count=len(point_results),
        )
        height_margin = np.fromiter(
            (result.glide_result.equivalent_height_margin_m for result in point_results),
            dtype=float,
            count=len(point_results),
        )
        clip_margin = height_margin.copy()
        finite_feasible = vertex_powered_feasible & np.isfinite(clip_margin)
        finite_scale = (
            float(np.max(np.abs(clip_margin[finite_feasible])))
            if np.any(finite_feasible)
            else 1.0
        )
        clip_margin[~finite_feasible] = -max(finite_scale, 1.0)
        reachable_region = _clip_energy_mesh(
            vertex_array,
            triangle_array,
            available_energy,
            required_energy,
            energy_margin,
            height_margin,
            clip_margin,
            keep_reachable=True,
        )
        unreachable_region = _clip_energy_mesh(
            vertex_array,
            triangle_array,
            available_energy,
            required_energy,
            energy_margin,
            height_margin,
            clip_margin,
            keep_reachable=False,
        )

        return EnergyReachabilitySurface(
            vertices=vertex_array,
            triangles=triangle_array,
            vertex_reachable=vertex_reachable,
            vertex_powered_feasible=vertex_powered_feasible,
            vertex_available_energy_j=available_energy,
            vertex_required_energy_j=required_energy,
            vertex_energy_margin_j=energy_margin,
            vertex_height_margin_m=height_margin,
            reachable_region=reachable_region,
            unreachable_region=unreachable_region,
            radial_scales=radial_scales,
            contour_count=contour_count,
        )


def _clip_energy_mesh(
    vertices: FloatArray,
    triangles: IntArray,
    available_energy_j: FloatArray,
    required_energy_j: FloatArray,
    energy_margin_j: FloatArray,
    height_margin_m: FloatArray,
    clip_margin_m: FloatArray,
    *,
    keep_reachable: bool,
) -> EnergyRegionMesh:
    """Clip every triangle at zero margin and triangulate the retained polygon."""
    output_vertices: list[FloatArray] = []
    output_triangles: list[tuple[int, int, int]] = []
    output_available: list[float] = []
    output_required: list[float] = []
    output_margin: list[float] = []
    output_height_margin: list[float] = []

    for triangle in triangles:
        polygon = [
            _ClipVertex(
                position=vertices[index],
                available_energy_j=float(available_energy_j[index]),
                required_energy_j=float(required_energy_j[index]),
                energy_margin_j=float(energy_margin_j[index]),
                height_margin_m=float(height_margin_m[index]),
                clip_margin_m=float(clip_margin_m[index]),
            )
            for index in triangle
        ]
        clipped = _clip_polygon_at_zero(polygon, keep_reachable=keep_reachable)
        if len(clipped) < 3:
            continue
        first_index = len(output_vertices)
        for vertex in clipped:
            output_vertices.append(vertex.position)
            output_available.append(vertex.available_energy_j)
            output_required.append(vertex.required_energy_j)
            output_margin.append(vertex.energy_margin_j)
            output_height_margin.append(vertex.height_margin_m)
        for offset in range(1, len(clipped) - 1):
            output_triangles.append(
                (first_index, first_index + offset, first_index + offset + 1)
            )

    vertex_array = (
        np.vstack(output_vertices)
        if output_vertices
        else np.empty((0, 3), dtype=float)
    )
    triangle_array = (
        np.asarray(output_triangles, dtype=np.int64)
        if output_triangles
        else np.empty((0, 3), dtype=np.int64)
    )
    return EnergyRegionMesh(
        vertices=vertex_array,
        triangles=triangle_array,
        available_energy_j=np.asarray(output_available, dtype=float),
        required_energy_j=np.asarray(output_required, dtype=float),
        energy_margin_j=np.asarray(output_margin, dtype=float),
        height_margin_m=np.asarray(output_height_margin, dtype=float),
    )


def _clip_polygon_at_zero(
    polygon: list[_ClipVertex],
    *,
    keep_reachable: bool,
) -> list[_ClipVertex]:
    if not polygon:
        return []

    def inside(vertex: _ClipVertex) -> bool:
        return (
            vertex.clip_margin_m >= -1.0e-12
            if keep_reachable
            else vertex.clip_margin_m <= 1.0e-12
        )

    output: list[_ClipVertex] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            output.append(_interpolate_zero_crossing(previous, current))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return output


def _interpolate_zero_crossing(first: _ClipVertex, second: _ClipVertex) -> _ClipVertex:
    denominator = first.clip_margin_m - second.clip_margin_m
    fraction = (
        0.5
        if abs(denominator) <= 1.0e-15
        else float(np.clip(first.clip_margin_m / denominator, 0.0, 1.0))
    )

    def interpolate(first_value: float, second_value: float, *, boundary: bool = False) -> float:
        if boundary and not (np.isfinite(first_value) and np.isfinite(second_value)):
            return 0.0
        if np.isfinite(first_value) and np.isfinite(second_value):
            return first_value + fraction * (second_value - first_value)
        return first_value if np.isfinite(first_value) else second_value

    return _ClipVertex(
        position=first.position + fraction * (second.position - first.position),
        available_energy_j=interpolate(
            first.available_energy_j, second.available_energy_j
        ),
        required_energy_j=interpolate(
            first.required_energy_j, second.required_energy_j
        ),
        energy_margin_j=interpolate(
            first.energy_margin_j, second.energy_margin_j, boundary=True
        ),
        height_margin_m=interpolate(
            first.height_margin_m, second.height_margin_m, boundary=True
        ),
        clip_margin_m=0.0,
    )

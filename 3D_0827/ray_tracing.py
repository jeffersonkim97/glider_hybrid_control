"""Generic first-hit ray tracing against triangular surface meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from map_geometry import TriangleMesh


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class RayHit:
    """Closest positive ray-triangle intersection."""

    distance: float
    point: FloatArray
    mesh_index: int
    triangle_index: int


class TriangleRayTracer:
    """Vectorized Moller-Trumbore first-hit tracer for one mesh collection."""

    def __init__(self, meshes: tuple[TriangleMesh, ...]) -> None:
        if not meshes:
            raise ValueError("at least one triangle mesh is required")

        triangle_vertices: list[FloatArray] = []
        mesh_indices: list[int] = []
        local_triangle_indices: list[int] = []
        for mesh_index, mesh in enumerate(meshes):
            for triangle_index, vertex_indices in enumerate(mesh.triangles):
                triangle_vertices.append(mesh.vertices[vertex_indices])
                mesh_indices.append(mesh_index)
                local_triangle_indices.append(triangle_index)

        if not triangle_vertices:
            raise ValueError("the mesh collection contains no triangles")

        triangles = np.asarray(triangle_vertices, dtype=float)
        self._vertex_0 = triangles[:, 0]
        self._edge_1 = triangles[:, 1] - triangles[:, 0]
        self._edge_2 = triangles[:, 2] - triangles[:, 0]
        self._mesh_indices = np.asarray(mesh_indices, dtype=np.int64)
        self._triangle_indices = np.asarray(local_triangle_indices, dtype=np.int64)

    def first_hit(
        self,
        origin: FloatArray,
        direction: FloatArray,
        *,
        minimum_distance: float = 1.0e-8,
        tolerance: float = 1.0e-10,
    ) -> RayHit | None:
        """Return the nearest forward intersection, including triangle edges."""
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        if origin.shape != (3,) or direction.shape != (3,):
            raise ValueError("origin and direction must have shape (3,)")
        if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)):
            raise ValueError("origin and direction must be finite")
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 0.0:
            raise ValueError("direction must be nonzero")
        direction = direction / direction_norm

        direction_rows = np.broadcast_to(direction, self._edge_2.shape)
        p_vector = np.cross(direction_rows, self._edge_2)
        determinant = np.einsum("ij,ij->i", self._edge_1, p_vector)
        nonparallel = np.abs(determinant) > tolerance
        inverse_determinant = np.zeros_like(determinant)
        inverse_determinant[nonparallel] = 1.0 / determinant[nonparallel]

        t_vector = origin - self._vertex_0
        barycentric_u = (
            np.einsum("ij,ij->i", t_vector, p_vector) * inverse_determinant
        )
        q_vector = np.cross(t_vector, self._edge_1)
        barycentric_v = (
            np.einsum("ij,ij->i", direction_rows, q_vector)
            * inverse_determinant
        )
        distances = (
            np.einsum("ij,ij->i", self._edge_2, q_vector)
            * inverse_determinant
        )

        valid = (
            nonparallel
            & (barycentric_u >= -tolerance)
            & (barycentric_v >= -tolerance)
            & (barycentric_u + barycentric_v <= 1.0 + tolerance)
            & (distances >= minimum_distance)
        )
        if not np.any(valid):
            return None

        valid_indices = np.flatnonzero(valid)
        closest_index = int(valid_indices[np.argmin(distances[valid])])
        distance = float(distances[closest_index])
        return RayHit(
            distance=distance,
            point=origin + distance * direction,
            mesh_index=int(self._mesh_indices[closest_index]),
            triangle_index=int(self._triangle_indices[closest_index]),
        )

    def first_hit_is_mesh(
        self,
        origin: FloatArray,
        direction: FloatArray,
        mesh_index: int,
    ) -> bool:
        hit = self.first_hit(origin, direction)
        return hit is not None and hit.mesh_index == mesh_index


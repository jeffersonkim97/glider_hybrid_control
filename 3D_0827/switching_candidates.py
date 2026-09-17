"""Pure geometric switching candidates on a continuous LOS tangent surface."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from los_geometry import TangentContour


FloatArray = NDArray[np.float64]
INITIAL_CONTOUR_SAMPLE_COUNT = 12
INITIAL_RADIAL_SCALES = tuple(np.linspace(0.5, 4.0, 8, dtype=float))


def _immutable_vector3(values: FloatArray, name: str) -> FloatArray:
    vector = np.array(values, dtype=float, copy=True)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain exactly three finite coordinates")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class SwitchingCandidate:
    """One deterministic, energy-neutral point on the LOS tangent surface."""

    candidate_id: int
    position_map: FloatArray
    contour_fraction: float
    radial_scale: float
    surface_residual: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, Integral) or isinstance(
            self.candidate_id, bool,
        ):
            raise TypeError("candidate_id must be an integer")
        candidate_id = int(self.candidate_id)
        if candidate_id < 0:
            raise ValueError("candidate_id must be nonnegative")

        contour_fraction = float(self.contour_fraction)
        radial_scale = float(self.radial_scale)
        surface_residual = float(self.surface_residual)
        if not np.isfinite(contour_fraction) or not 0.0 <= contour_fraction <= 1.0:
            raise ValueError("contour_fraction must lie in [0, 1]")
        if not np.isfinite(radial_scale) or radial_scale <= 0.0:
            raise ValueError("radial_scale must be finite and positive")
        if (
            not np.isfinite(surface_residual)
            or not 0.0 <= surface_residual <= 1.0 + 1.0e-12
        ):
            raise ValueError("surface_residual must be finite and lie in [0, 1]")

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(
            self,
            "position_map",
            _immutable_vector3(self.position_map, "position_map"),
        )
        object.__setattr__(self, "contour_fraction", contour_fraction)
        object.__setattr__(self, "radial_scale", radial_scale)
        object.__setattr__(self, "surface_residual", surface_residual)


def build_switching_candidate(
    tangent_contour: TangentContour,
    *,
    candidate_id: int,
    contour_fraction: float,
    radial_scale: float,
) -> SwitchingCandidate:
    """Evaluate one point directly from the continuous ruled-surface formula."""
    if not isinstance(tangent_contour, TangentContour):
        raise TypeError("tangent_contour must be a TangentContour")
    fraction = float(contour_fraction)
    scale = float(radial_scale)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("contour_fraction must lie in [0, 1]")
    if tangent_contour.closed and fraction == 1.0:
        raise ValueError("closed contours use [0, 1) to avoid a duplicate endpoint")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("radial_scale must be finite and positive")

    origin = tangent_contour.origin.as_array()
    tangent_vector = tangent_contour.tangent_vector_at(fraction)
    tangent_norm = float(np.linalg.norm(tangent_vector))
    if not np.isfinite(tangent_norm) or tangent_norm <= 0.0:
        raise ValueError("interpolated tangent vector must be finite and nonzero")
    position = origin + scale * tangent_vector
    displacement = position - origin
    displacement_norm = float(np.linalg.norm(displacement))
    denominator = displacement_norm * tangent_norm
    residual = float(np.linalg.norm(np.cross(displacement, tangent_vector)) / denominator)
    if float(np.dot(displacement, tangent_vector)) <= 0.0:
        raise ValueError("candidate must lie in the positive radial direction")

    return SwitchingCandidate(
        candidate_id=candidate_id,
        position_map=position,
        contour_fraction=fraction,
        radial_scale=scale,
        surface_residual=residual,
    )


def generate_switching_candidates(
    tangent_contour: TangentContour,
    *,
    contour_sample_count: int = INITIAL_CONTOUR_SAMPLE_COUNT,
    radial_scales: Iterable[float] = INITIAL_RADIAL_SCALES,
) -> tuple[SwitchingCandidate, ...]:
    """Generate a deterministic radial-major grid without using a display mesh."""
    if not isinstance(tangent_contour, TangentContour):
        raise TypeError("tangent_contour must be a TangentContour")
    if not isinstance(contour_sample_count, Integral) or isinstance(
        contour_sample_count, bool,
    ):
        raise TypeError("contour_sample_count must be an integer")
    minimum_count = 3 if tangent_contour.closed else 2
    if int(contour_sample_count) < minimum_count:
        raise ValueError(
            f"contour_sample_count must be at least {minimum_count} for this contour"
        )

    scales = np.asarray(tuple(radial_scales), dtype=float)
    if scales.ndim != 1 or not len(scales):
        raise ValueError("radial_scales must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("all radial scales must be finite and positive")
    if np.any(np.diff(scales) <= 0.0):
        raise ValueError("radial scales must be strictly increasing")

    fractions = np.linspace(
        0.0,
        1.0,
        int(contour_sample_count),
        endpoint=not tangent_contour.closed,
        dtype=float,
    )
    candidates: list[SwitchingCandidate] = []
    for scale in scales:
        for fraction in fractions:
            candidates.append(build_switching_candidate(
                tangent_contour,
                candidate_id=len(candidates),
                contour_fraction=float(fraction),
                radial_scale=float(scale),
            ))
    return tuple(candidates)

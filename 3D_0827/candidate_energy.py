"""Stage-4 powered and total-energy certificates for switching candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from energy_model import SwitchingState
from glide_reachability import GlideReachabilityResult
from los_geometry import TangentContour
from map_geometry import TerrainModel
from reachability_surface import LOSSurfaceReachabilityClassifier, PointReachability
from scenario import MissionPoints
from switching_candidates import SwitchingCandidate


ACOUSTIC_SURFACE_TOLERANCE = 1.0e-8


def switching_candidate_is_acoustically_neutralized(
    candidate: SwitchingCandidate,
    tangent_contour: TangentContour,
    *,
    tolerance: float = ACOUSTIC_SURFACE_TOLERANCE,
) -> bool:
    """Apply the current mission rule: switching occurs on the LOS surface."""
    if not isinstance(candidate, SwitchingCandidate):
        raise TypeError("candidate must be a SwitchingCandidate")
    if not isinstance(tangent_contour, TangentContour):
        raise TypeError("tangent_contour must be a TangentContour")
    tolerance_value = float(tolerance)
    if not np.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    origin = tangent_contour.origin.as_array()
    tangent_vector = tangent_contour.tangent_vector_at(
        candidate.contour_fraction,
    )
    expected_position = origin + candidate.radial_scale * tangent_vector
    displacement = candidate.position_map - origin
    positive_direction = float(np.dot(displacement, tangent_vector)) > 0.0
    return bool(
        positive_direction
        and candidate.radial_scale > 0.0
        and candidate.surface_residual <= tolerance_value
        and np.linalg.norm(candidate.position_map - expected_position)
        <= tolerance_value
    )


@dataclass(frozen=True)
class CandidateEnergyEvaluation:
    """Independent powered/glide certificate for one switching candidate."""

    candidate: SwitchingCandidate
    acoustically_neutralized: bool
    point_reachability: PointReachability

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, SwitchingCandidate):
            raise TypeError("candidate must be a SwitchingCandidate")
        if not isinstance(self.acoustically_neutralized, (bool, np.bool_)):
            raise TypeError("acoustically_neutralized must be boolean")
        if not isinstance(self.point_reachability, PointReachability):
            raise TypeError("point_reachability must be a PointReachability")
        object.__setattr__(
            self,
            "acoustically_neutralized",
            bool(self.acoustically_neutralized),
        )

    @property
    def candidate_id(self) -> int:
        return self.candidate.candidate_id

    @property
    def switching_state(self) -> SwitchingState:
        return self.point_reachability.switching_state

    @property
    def glide_result(self) -> GlideReachabilityResult:
        return self.point_reachability.glide_result

    @property
    def powered_feasible(self) -> bool:
        return self.switching_state.powered_feasible

    @property
    def reachable(self) -> bool:
        return bool(
            self.acoustically_neutralized
            and self.point_reachability.reachable
        )

    @property
    def energy_margin_j(self) -> float:
        return self.glide_result.energy_margin_j

    @property
    def equivalent_height_margin_m(self) -> float:
        return self.glide_result.equivalent_height_margin_m

    @property
    def infeasibility_reason(self) -> str | None:
        if not self.acoustically_neutralized:
            return "switching candidate is not on the LOS tangent surface"
        if not self.powered_feasible:
            return self.switching_state.infeasibility_reason
        if self.glide_result.reachable:
            return None
        if not np.isfinite(self.glide_result.minimum_glide_path_m):
            return "no feasible bounded-turn glide path"
        return "insufficient total energy"


def evaluate_switching_candidate(
    candidate: SwitchingCandidate,
    tangent_contour: TangentContour,
    terrain: TerrainModel,
    mission_points: MissionPoints,
    *,
    classifier: LOSSurfaceReachabilityClassifier | None = None,
) -> CandidateEnergyEvaluation:
    """Evaluate one candidate with the existing powered/glide composition."""
    if not isinstance(candidate, SwitchingCandidate):
        raise TypeError("candidate must be a SwitchingCandidate")
    if not isinstance(tangent_contour, TangentContour):
        raise TypeError("tangent_contour must be a TangentContour")
    active_classifier = classifier or LOSSurfaceReachabilityClassifier()
    point_result = active_classifier.evaluate_point(
        candidate.position_map,
        terrain,
        mission_points,
    )
    return CandidateEnergyEvaluation(
        candidate=candidate,
        acoustically_neutralized=(
            switching_candidate_is_acoustically_neutralized(
                candidate,
                tangent_contour,
            )
        ),
        point_reachability=point_result,
    )


def evaluate_switching_candidates(
    candidates: Iterable[SwitchingCandidate],
    tangent_contour: TangentContour,
    terrain: TerrainModel,
    mission_points: MissionPoints,
    *,
    classifier: LOSSurfaceReachabilityClassifier | None = None,
) -> tuple[CandidateEnergyEvaluation, ...]:
    """Evaluate a candidate sequence once, preserving its deterministic order."""
    candidate_sequence = tuple(candidates)
    if not candidate_sequence:
        raise ValueError("at least one switching candidate is required")
    candidate_ids = [candidate.candidate_id for candidate in candidate_sequence]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")
    active_classifier = classifier or LOSSurfaceReachabilityClassifier()
    return tuple(
        evaluate_switching_candidate(
            candidate,
            tangent_contour,
            terrain,
            mission_points,
            classifier=active_classifier,
        )
        for candidate in candidate_sequence
    )

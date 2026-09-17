"""Stage-10 independent continuous replay of an untrusted optimized path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from detection_hazard import HazardField
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE, GliderParameters, PhysicalScale
from los_geometry import TangentContour
from map_geometry import TerrainModel
from scenario import MissionPoints

if TYPE_CHECKING:
    from attacker_best_response import AttackerBestResponseRun


FloatArray = NDArray[np.float64]


def _angle_difference(first_rad: float, second_rad: float) -> float:
    difference = (float(second_rad) - float(first_rad) + np.pi) % (2.0 * np.pi) - np.pi
    return abs(float(difference))


def _immutable_vector3(values: FloatArray, name: str) -> FloatArray:
    vector = np.array(values, dtype=float, copy=True)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite coordinates")
    vector.setflags(write=False)
    return vector


def _immutable_positions(values: FloatArray, name: str) -> FloatArray:
    positions = np.array(values, dtype=float, copy=True)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError(f"{name} must have shape (n, 3) with n >= 1")
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{name} must be finite")
    positions.setflags(write=False)
    return positions


@dataclass(frozen=True)
class ReplayTolerances:
    position_m: float = 1.0e-6
    terrain_clearance_m: float = 0.0
    angle_rad: float = 1.0e-8
    time_s: float = 1.0e-9
    hazard: float = 1.0e-10
    energy_height_m: float = 1.0e-8
    switching_speed_mps: float = 1.0e-9
    switching_energy_j: float = 1.0e-6
    goal_boundary_m: float = 1.0e-9

    def __post_init__(self) -> None:
        for name in (
            "position_m", "terrain_clearance_m", "angle_rad", "time_s",
            "hazard", "energy_height_m", "switching_speed_mps",
            "switching_energy_j", "goal_boundary_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


DEFAULT_REPLAY_TOLERANCES = ReplayTolerances()


@dataclass(frozen=True)
class TrajectoryReplaySnapshot:
    """Minimal copied optimizer output; no optimizer policy/value references."""

    candidate_id: int
    switching_position_map: FloatArray
    contour_fraction: float
    radial_scale: float
    stored_switching_heading_rad: float
    stored_switching_speed_mps: float
    stored_switching_energy_j: float
    discrete_positions_map: FloatArray
    discrete_headings_rad: tuple[float, ...]
    stored_mission_time_s: float
    stored_cumulative_hazard: float
    stored_detection_probability: float
    hazard_quadrature_resolution: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "switching_position_map",
            _immutable_vector3(self.switching_position_map, "switching_position_map"),
        )
        positions = _immutable_positions(
            self.discrete_positions_map, "discrete_positions_map",
        )
        object.__setattr__(self, "discrete_positions_map", positions)
        headings = tuple(float(value) for value in self.discrete_headings_rad)
        if len(headings) != len(positions) or not np.all(np.isfinite(headings)):
            raise ValueError("one finite heading is required per discrete position")
        object.__setattr__(self, "discrete_headings_rad", headings)
        for name in (
            "contour_fraction", "radial_scale", "stored_switching_heading_rad",
            "stored_switching_speed_mps", "stored_switching_energy_j",
            "stored_mission_time_s", "stored_cumulative_hazard",
            "stored_detection_probability",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.stored_detection_probability <= 1.0:
            raise ValueError("stored_detection_probability must lie in [0, 1]")
        if self.hazard_quadrature_resolution < 2:
            raise ValueError("hazard quadrature resolution must be at least two")


@dataclass(frozen=True)
class TrajectoryValidationReport:
    passed: bool
    terrain_clear: bool
    los_phase_valid: bool
    turn_constraints_valid: bool
    energy_valid: bool
    time_consistent: bool
    hazard_consistent: bool
    goal_valid: bool
    max_turn_violation: float
    min_terrain_clearance: float | None
    time_error_s: float
    hazard_error: float
    goal_error_m: float
    messages: tuple[str, ...]


@dataclass(frozen=True)
class ReplaySegmentAudit:
    segment_index: int
    phase: str
    start_position_map: FloatArray
    end_position_map: FloatArray
    duration_s: float
    hazard: float
    terrain_clear: bool
    heading_change_rad: float
    allowed_heading_change_rad: float
    turn_valid: bool
    altitude_loss_m: float
    required_altitude_loss_m: float
    altitude_valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "start_position_map",
            _immutable_vector3(self.start_position_map, "start_position_map"),
        )
        object.__setattr__(
            self, "end_position_map",
            _immutable_vector3(self.end_position_map, "end_position_map"),
        )


@dataclass(frozen=True)
class TrajectoryReplayAudit:
    report: TrajectoryValidationReport
    segments: tuple[ReplaySegmentAudit, ...]
    recomputed_mission_time_s: float
    recomputed_cumulative_hazard: float
    recomputed_detection_probability: float
    available_specific_height_m: float
    required_specific_height_m: float
    energy_margin_m: float
    switching_heading_error_rad: float
    switching_speed_error_mps: float
    switching_energy_error_j: float
    goal_distance_m: float
    first_failure_segment_index: int | None
    first_failure_position_map: FloatArray | None

    def __post_init__(self) -> None:
        if self.first_failure_position_map is not None:
            object.__setattr__(
                self, "first_failure_position_map",
                _immutable_vector3(
                    self.first_failure_position_map, "first_failure_position_map",
                ),
            )


def snapshot_selected_trajectory(
    run: "AttackerBestResponseRun",
    *,
    hazard_quadrature_resolution: int = 8,
) -> TrajectoryReplaySnapshot:
    """Copy the selected path out of the optimizer before independent replay."""
    selected = run.selected_result
    if selected is None or selected.selected_option is None:
        raise ValueError("Attacker run has no feasible selected trajectory")
    option = selected.selected_option
    states = (
        (option.connection.target_state,)
        + tuple(edge.target_state for edge in option.glide_edges)
    )
    positions = np.vstack([
        run.graph.grid.position_map(state) for state in states
    ])
    headings = tuple(run.graph.grid.heading_rad(state.heading_bin) for state in states)
    return TrajectoryReplaySnapshot(
        candidate_id=selected.candidate_id,
        switching_position_map=selected.position_map,
        contour_fraction=selected.candidate.contour_fraction,
        radial_scale=selected.candidate.radial_scale,
        stored_switching_heading_rad=(
            selected.evaluation.switching_state.heading_rad
        ),
        stored_switching_speed_mps=float(np.linalg.norm(
            selected.evaluation.switching_state.velocity_mps
        )),
        stored_switching_energy_j=(
            selected.evaluation.switching_state.total_mechanical_energy_j
        ),
        discrete_positions_map=positions,
        discrete_headings_rad=headings,
        stored_mission_time_s=float(selected.mission_time_s),
        stored_cumulative_hazard=float(selected.cumulative_hazard),
        stored_detection_probability=float(selected.detection_probability),
        hazard_quadrature_resolution=hazard_quadrature_resolution,
    )


def _independent_tangent_vector(
    tangent_contour: TangentContour,
    contour_fraction: float,
) -> FloatArray:
    """Reimplement normalized-arc interpolation without LOS helper calls."""
    points = np.vstack([ray.vector for ray in tangent_contour.rays])
    closed = tangent_contour.closed
    next_points = np.roll(points, -1, axis=0) if closed else points[1:]
    current_points = points if closed else points[:-1]
    lengths = np.linalg.norm(next_points - current_points, axis=1)
    total = float(np.sum(lengths))
    fraction = float(contour_fraction) % 1.0 if closed else float(contour_fraction)
    target = fraction * total
    cumulative = np.cumsum(lengths)
    index = min(int(np.searchsorted(cumulative, target, side="right")), len(lengths) - 1)
    start_distance = 0.0 if index == 0 else float(cumulative[index - 1])
    local = (target - start_distance) / float(lengths[index])
    return (1.0 - local) * current_points[index] + local * next_points[index]


def _independent_segment_hazard(
    start: FloatArray,
    end: FloatArray,
    duration_s: float,
    start_time_s: float,
    hazard_field: HazardField,
    resolution: int,
    physical_scale: PhysicalScale,
) -> float:
    if duration_s == 0.0:
        return 0.0
    fractions = np.linspace(0.0, 1.0, resolution)
    step = duration_s / (resolution - 1)
    weights = np.full(resolution, step)
    weights[[0, -1]] *= 0.5
    displacement = end - start
    velocity = physical_scale.position_m(displacement) / duration_s
    hazard = 0.0
    for fraction, weight in zip(fractions, weights):
        position = start + float(fraction) * displacement
        rate = hazard_field.evaluate_rate(
            position,
            velocity,
            start_time_s + float(fraction) * duration_s,
        )
        hazard += rate.total_rate_per_s * float(weight)
    return float(hazard)


def goal_is_valid(
    goal_distance_m: float,
    parameters: GliderParameters = DEFAULT_GLIDER,
    tolerances: ReplayTolerances = DEFAULT_REPLAY_TOLERANCES,
) -> bool:
    return bool(
        float(goal_distance_m)
        <= parameters.goal_tolerance_m + tolerances.goal_boundary_m
    )


def energy_margin_is_valid(
    margin_m: float,
    tolerances: ReplayTolerances = DEFAULT_REPLAY_TOLERANCES,
) -> bool:
    return bool(float(margin_m) >= -tolerances.energy_height_m)


def validate_trajectory_replay(
    snapshot: TrajectoryReplaySnapshot,
    terrain: TerrainModel,
    mission_points: MissionPoints,
    tangent_contour: TangentContour,
    hazard_field: HazardField,
    *,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    tolerances: ReplayTolerances = DEFAULT_REPLAY_TOLERANCES,
) -> TrajectoryReplayAudit:
    """Recompute every mission quantity without optimizer policy/value reuse."""
    start = mission_points.start.as_array()
    switch = snapshot.switching_position_map
    discrete = snapshot.discrete_positions_map
    headings = snapshot.discrete_headings_rad
    powered_displacement = switch - start
    powered_length_m = physical_scale.distance_m(float(np.linalg.norm(powered_displacement)))
    powered_duration = powered_length_m / parameters.powered_speed_mps
    powered_heading = float(np.arctan2(powered_displacement[1], powered_displacement[0]))
    switch_altitude_m = physical_scale.distance_m(float(switch[2]))
    total_energy_j = parameters.mass_kg * (
        parameters.gravity_mps2 * switch_altitude_m
        + 0.5 * parameters.powered_speed_mps**2
    )
    switching_heading_error = _angle_difference(
        powered_heading, snapshot.stored_switching_heading_rad,
    )
    switching_speed_error = abs(
        parameters.powered_speed_mps - snapshot.stored_switching_speed_mps
    )
    switching_energy_error = abs(
        total_energy_j - snapshot.stored_switching_energy_j
    )
    switching_heading_valid = switching_heading_error <= tolerances.angle_rad
    switching_speed_energy_valid = bool(
        switching_speed_error <= tolerances.switching_speed_mps
        and switching_energy_error <= tolerances.switching_energy_j
    )

    expected_switch = (
        tangent_contour.origin.as_array()
        + snapshot.radial_scale
        * _independent_tangent_vector(tangent_contour, snapshot.contour_fraction)
    )
    los_position_error_m = physical_scale.distance_m(float(
        np.linalg.norm(switch - expected_switch)
    ))
    los_phase_valid = bool(
        snapshot.radial_scale > 0.0
        and los_position_error_m <= tolerances.position_m
    )

    segment_specs: list[tuple[str, FloatArray, FloatArray, float, float]] = []
    # phase, start, end, arrival heading, target heading
    segment_specs.append((
        "powered", start, switch, powered_heading, powered_heading,
    ))
    segment_specs.append((
        "virtual", switch, discrete[0], powered_heading, headings[0],
    ))
    for index in range(len(discrete) - 1):
        segment_specs.append((
            "glide", discrete[index], discrete[index + 1],
            headings[index], headings[index + 1],
        ))

    audits: list[ReplaySegmentAudit] = []
    elapsed_time = 0.0
    cumulative_hazard = 0.0
    maximum_turn_violation = 0.0
    all_turn_valid = True
    all_altitude_valid = True
    for index, (phase, segment_start, segment_end, arrival_heading, target_heading) in enumerate(segment_specs):
        displacement = segment_end - segment_start
        horizontal_map = float(np.linalg.norm(displacement[:2]))
        horizontal_m = physical_scale.distance_m(horizontal_map)
        if phase == "powered":
            duration = powered_duration
            heading_change = 0.0
            allowed_change = 0.0
            turn_valid = True
            required_loss = 0.0
            altitude_valid = True
            hazard = 0.0
        else:
            duration = horizontal_m / parameters.best_glide_speed_mps
            heading_change = _angle_difference(arrival_heading, target_heading)
            allowed_change = parameters.maximum_turn_rate_rad_s * duration
            turn_violation = max(0.0, heading_change - allowed_change)
            maximum_turn_violation = max(maximum_turn_violation, turn_violation)
            chord_heading = (
                target_heading
                if horizontal_m <= tolerances.position_m
                else float(np.arctan2(displacement[1], displacement[0]))
            )
            chord_error = _angle_difference(chord_heading, target_heading)
            turn_valid = bool(
                turn_violation <= tolerances.angle_rad
                and chord_error <= tolerances.angle_rad
            )
            required_loss = horizontal_m / parameters.best_glide_ratio
            altitude_loss = physical_scale.distance_m(float(
                segment_start[2] - segment_end[2]
            ))
            altitude_valid = bool(
                altitude_loss + tolerances.energy_height_m >= required_loss
                and altitude_loss >= -tolerances.energy_height_m
            )
            hazard = _independent_segment_hazard(
                segment_start, segment_end, duration, elapsed_time,
                hazard_field, snapshot.hazard_quadrature_resolution, physical_scale,
            )
        altitude_loss = physical_scale.distance_m(float(
            segment_start[2] - segment_end[2]
        ))
        terrain_clear = not terrain.segment_intersects_solid(
            segment_start, segment_end,
        )
        audits.append(ReplaySegmentAudit(
            segment_index=index,
            phase=phase,
            start_position_map=segment_start,
            end_position_map=segment_end,
            duration_s=duration,
            hazard=hazard,
            terrain_clear=terrain_clear,
            heading_change_rad=heading_change,
            allowed_heading_change_rad=allowed_change,
            turn_valid=turn_valid,
            altitude_loss_m=altitude_loss,
            required_altitude_loss_m=required_loss,
            altitude_valid=altitude_valid,
        ))
        elapsed_time += duration
        cumulative_hazard += hazard
        all_turn_valid &= turn_valid
        all_altitude_valid &= altitude_valid

    terrain_clear = all(segment.terrain_clear for segment in audits)
    trim_height = parameters.best_glide_speed_mps**2 / (2.0 * parameters.gravity_mps2)
    available_height = (
        total_energy_j / (parameters.mass_kg * parameters.gravity_mps2)
        - trim_height
    )
    nonpowered_horizontal_m = sum(
        physical_scale.distance_m(float(np.linalg.norm(
            segment.end_position_map[:2] - segment.start_position_map[:2]
        )))
        for segment in audits if segment.phase != "powered"
    )
    terminal_altitude_m = physical_scale.distance_m(float(discrete[-1, 2]))
    required_height = (
        parameters.switch_energy_loss_height_m
        + nonpowered_horizontal_m / parameters.best_glide_ratio
        + terminal_altitude_m
    )
    energy_margin = available_height - required_height
    energy_valid = bool(
        all_altitude_valid
        and energy_margin_is_valid(energy_margin, tolerances)
        and switching_speed_energy_valid
    )
    time_error = abs(elapsed_time - snapshot.stored_mission_time_s)
    hazard_error = abs(cumulative_hazard - snapshot.stored_cumulative_hazard)
    time_consistent = time_error <= tolerances.time_s
    hazard_consistent = hazard_error <= tolerances.hazard
    recomputed_pod = float(-np.expm1(-cumulative_hazard))
    goal_distance = physical_scale.distance_m(float(np.linalg.norm(
        discrete[-1] - mission_points.goal.as_array()
    )))
    goal_valid = goal_is_valid(goal_distance, parameters, tolerances)
    goal_error = max(0.0, goal_distance - parameters.goal_tolerance_m)

    messages: list[str] = []
    all_turn_valid = bool(all_turn_valid and switching_heading_valid)
    flags_and_messages = (
        (terrain_clear, "trajectory intersects terrain"),
        (los_phase_valid, "switching point is not on the LOS tangent surface"),
        (switching_heading_valid, "powered switching heading is inconsistent"),
        (switching_speed_energy_valid, "powered switching speed or energy is inconsistent"),
        (all_turn_valid, "turn-rate or heading/chord constraint violated"),
        (energy_valid, "energy or altitude budget violated"),
        (time_consistent, "recomputed mission time differs from optimizer output"),
        (hazard_consistent, "recomputed hazard differs from optimizer output"),
        (goal_valid, "terminal state lies outside goal tolerance"),
    )
    for valid, message in flags_and_messages:
        if not valid:
            messages.append(message)
    if not messages:
        messages.append("independent replay passed all checks")
    passed = bool(all(valid for valid, _ in flags_and_messages))

    first_failure_index: int | None = None
    first_failure_position: FloatArray | None = None
    for segment in audits:
        if not (segment.terrain_clear and segment.turn_valid and segment.altitude_valid):
            first_failure_index = segment.segment_index
            first_failure_position = segment.end_position_map
            break
    if first_failure_position is None and not los_phase_valid:
        first_failure_position = switch
    if first_failure_position is None and not goal_valid:
        first_failure_position = discrete[-1]
    if first_failure_position is None and not passed:
        first_failure_position = discrete[-1]

    report = TrajectoryValidationReport(
        passed=passed,
        terrain_clear=terrain_clear,
        los_phase_valid=los_phase_valid,
        turn_constraints_valid=all_turn_valid,
        energy_valid=energy_valid,
        time_consistent=time_consistent,
        hazard_consistent=hazard_consistent,
        goal_valid=goal_valid,
        max_turn_violation=maximum_turn_violation,
        min_terrain_clearance=None,
        time_error_s=time_error,
        hazard_error=hazard_error,
        goal_error_m=goal_error,
        messages=tuple(messages),
    )
    return TrajectoryReplayAudit(
        report=report,
        segments=tuple(audits),
        recomputed_mission_time_s=elapsed_time,
        recomputed_cumulative_hazard=cumulative_hazard,
        recomputed_detection_probability=recomputed_pod,
        available_specific_height_m=available_height,
        required_specific_height_m=required_height,
        energy_margin_m=energy_margin,
        switching_heading_error_rad=switching_heading_error,
        switching_speed_error_mps=switching_speed_error,
        switching_energy_error_j=switching_energy_error,
        goal_distance_m=goal_distance,
        first_failure_segment_index=first_failure_index,
        first_failure_position_map=first_failure_position,
    )

"""Stage-8 adapter from one continuous switching state to the glide lattice.

The adapter deliberately does not enumerate or optimize switching candidates.
It maps one already-selected continuous state to the local horizontal cell,
audits every deterministic lattice proposal, and rejects rather than snapping
when the geometry, turn, terrain, energy, or Bellman-reachability contracts fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from bellman_geometry import GlideEdge, wrapped_angle_difference
from bellman_graph import BellmanGraph
from bellman_state import BellmanState
from candidate_energy import CandidateEnergyEvaluation
from energy_model import (
    DEFAULT_GLIDER,
    DEFAULT_PHYSICAL_SCALE,
    GliderParameters,
    PhysicalScale,
    SwitchingState,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
GEOMETRY_TOLERANCE = 1.0e-10
GRID_ALIGNMENT_TOLERANCE = 1.0e-8
CHORD_HEADING_TOLERANCE_RAD = 1.0e-8


def _immutable_vector3(values: FloatArray, name: str) -> FloatArray:
    vector = np.array(values, dtype=float, copy=True)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite coordinates")
    vector.setflags(write=False)
    return vector


def heading_turn_is_feasible(
    arrival_heading_rad: float,
    target_heading_rad: float,
    duration_s: float,
    *,
    parameters: GliderParameters = DEFAULT_GLIDER,
    tolerance_rad: float = 1.0e-12,
) -> bool:
    """Check the same coordinated-turn-rate limit used by glide edges."""
    duration = float(duration_s)
    tolerance = float(tolerance_rad)
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_s must be finite and nonnegative")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_rad must be finite and nonnegative")
    mismatch = wrapped_angle_difference(arrival_heading_rad, target_heading_rad)
    return bool(
        mismatch <= parameters.maximum_turn_rate_rad_s * duration + tolerance
    )


@dataclass(frozen=True)
class VirtualConnectionProposal:
    """Complete audit record for one local lattice connection proposal."""

    candidate_id: int
    target_state: BellmanState
    target_state_id: int
    switching_position_map: FloatArray
    target_position_map: FloatArray
    horizontal_distance_m: float
    altitude_loss_m: float
    minimum_glide_loss_m: float
    projection_error_m: float
    heading_mismatch_rad: float
    chord_heading_mismatch_rad: float
    duration_s: float
    node_admissible: bool
    bellman_reachable: bool
    terrain_feasible: bool
    altitude_feasible: bool
    turn_feasible: bool
    chord_heading_feasible: bool
    powered_feasible: bool
    coarse_energy_feasible: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "switching_position_map",
            _immutable_vector3(self.switching_position_map, "switching_position_map"),
        )
        object.__setattr__(
            self,
            "target_position_map",
            _immutable_vector3(self.target_position_map, "target_position_map"),
        )
        for name in (
            "horizontal_distance_m",
            "altitude_loss_m",
            "minimum_glide_loss_m",
            "projection_error_m",
            "heading_mismatch_rad",
            "chord_heading_mismatch_rad",
            "duration_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.target_state_id < 0:
            raise ValueError("target_state_id must be nonnegative")
        boolean_names = (
            "node_admissible",
            "bellman_reachable",
            "terrain_feasible",
            "altitude_feasible",
            "turn_feasible",
            "chord_heading_feasible",
            "powered_feasible",
            "coarse_energy_feasible",
        )
        for name in boolean_names:
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self.feasible != (len(self.rejection_reasons) == 0):
            raise ValueError("rejection reasons disagree with feasibility flags")

    @property
    def feasible(self) -> bool:
        return bool(
            self.node_admissible
            and self.terrain_feasible
            and self.altitude_feasible
            and self.turn_feasible
            and self.chord_heading_feasible
            and self.powered_feasible
            and self.coarse_energy_feasible
        )

    @property
    def identity_connection(self) -> bool:
        return bool(self.projection_error_m <= GEOMETRY_TOLERANCE)


@dataclass(frozen=True)
class VirtualConnectionSet:
    """Deterministic proposals for exactly one fixed switching candidate."""

    candidate_id: int
    switching_state: SwitchingState
    proposals: tuple[VirtualConnectionProposal, ...]

    @property
    def feasible_proposals(self) -> tuple[VirtualConnectionProposal, ...]:
        return tuple(proposal for proposal in self.proposals if proposal.feasible)

    @property
    def has_connection(self) -> bool:
        return bool(self.feasible_proposals)


def _cell_anchor_indices(
    coordinate: float,
    minimum: float,
    spacing: float,
    count: int,
) -> tuple[int, ...]:
    fractional = (float(coordinate) - minimum) / spacing
    if (
        fractional < -GRID_ALIGNMENT_TOLERANCE
        or fractional > count - 1 + GRID_ALIGNMENT_TOLERANCE
    ):
        return ()
    nearest = int(round(fractional))
    if abs(fractional - nearest) <= GRID_ALIGNMENT_TOLERANCE:
        return (min(max(nearest, 0), count - 1),)
    lower = floor(fractional)
    values = tuple(index for index in (lower, lower + 1) if 0 <= index < count)
    return tuple(dict.fromkeys(values))


def _incident_cell_anchor_indices(
    coordinate: float,
    minimum: float,
    spacing: float,
    count: int,
) -> tuple[int, ...]:
    """Return corners of every cell incident to an exactly aligned point.

    The ordinary containing-cell rule degenerates to one point at a lattice
    intersection.  If that zero-length identity edge cannot satisfy the
    arrival-heading contract, the adjacent indices are the remaining local
    corners; this is not an unbounded neighborhood search or forced snap.
    """
    fractional = (float(coordinate) - minimum) / spacing
    nearest = int(round(fractional))
    if abs(fractional - nearest) > GRID_ALIGNMENT_TOLERANCE:
        return _cell_anchor_indices(coordinate, minimum, spacing, count)
    return tuple(
        index for index in (nearest - 1, nearest, nearest + 1)
        if 0 <= index < count
    )


def _nearest_heading_bin(graph: BellmanGraph, heading_rad: float) -> int:
    return min(
        range(graph.grid.heading_bin_count),
        key=lambda heading_bin: (
            wrapped_angle_difference(
                heading_rad,
                graph.grid.heading_rad(heading_bin),
            ),
            heading_bin,
        ),
    )


def build_virtual_connections(
    evaluation: CandidateEnergyEvaluation,
    graph: BellmanGraph,
    goal_reachable_mask: BoolArray,
    *,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
) -> VirtualConnectionSet:
    """Map one fixed continuous candidate to its containing lattice cell.

    At each horizontal cell corner, the highest grid altitude that can be
    reached with the configured L/D is proposed.  The heading bin is the one
    nearest to the connection chord (or arrival heading for an identity
    connection).  At an exact x-y lattice intersection, an unavailable or
    turn-infeasible zero-length identity proposal falls back only to the finite
    offsets already defined by the grid's motion-primitive stencil.  Thus a
    finer heading stencil supplies enough physical distance to satisfy the same
    turn-rate check without an unbounded neighborhood expansion or forced state
    snap.
    """
    if not isinstance(evaluation, CandidateEnergyEvaluation):
        raise TypeError("evaluation must be a CandidateEnergyEvaluation")
    if not isinstance(graph, BellmanGraph):
        raise TypeError("graph must be a BellmanGraph")
    reachable = np.asarray(goal_reachable_mask, dtype=bool)
    if reachable.shape != (graph.grid.state_count,):
        raise ValueError("goal_reachable_mask must contain one value per graph state")

    switching = evaluation.switching_state
    position = switching.position_map
    grid = graph.grid
    x_indices = _cell_anchor_indices(
        position[0], grid.bounds.x_min, grid.horizontal_spacing_map, grid.x_count,
    )
    y_indices = _cell_anchor_indices(
        position[1], grid.bounds.y_min, grid.horizontal_spacing_map, grid.y_count,
    )
    identity_heading_bin = _nearest_heading_bin(graph, switching.heading_rad)
    identity_turn_feasible = heading_turn_is_feasible(
        switching.heading_rad,
        grid.heading_rad(identity_heading_bin),
        0.0,
        parameters=parameters,
    )
    exactly_on_xy_lattice = bool(
        len(x_indices) == 1
        and len(y_indices) == 1
        and abs(
            position[0]
            - (grid.bounds.x_min + x_indices[0] * grid.horizontal_spacing_map)
        ) <= GRID_ALIGNMENT_TOLERANCE
        and abs(
            position[1]
            - (grid.bounds.y_min + y_indices[0] * grid.horizontal_spacing_map)
        ) <= GRID_ALIGNMENT_TOLERANCE
    )
    altitude_fraction = (
        (position[2] - grid.minimum_altitude_map) / grid.altitude_spacing_map
    )
    nearest_altitude_index = int(round(altitude_fraction))
    exactly_on_altitude_lattice = bool(
        0 <= nearest_altitude_index < grid.altitude_count
        and abs(altitude_fraction - nearest_altitude_index)
        <= GRID_ALIGNMENT_TOLERANCE
    )
    if exactly_on_xy_lattice and (
        not exactly_on_altitude_lattice or not identity_turn_feasible
    ):
        origin_x = x_indices[0]
        origin_y = y_indices[0]
        target_index_pairs = tuple(sorted({
            (origin_x, origin_y),
            *(
                (origin_x + x_offset, origin_y + y_offset)
                for x_offset, y_offset in grid.motion_offsets
                if 0 <= origin_x + x_offset < grid.x_count
                and 0 <= origin_y + y_offset < grid.y_count
            ),
        }, key=lambda pair: (pair[1], pair[0])))
    else:
        target_index_pairs = tuple(
            (x_index, y_index)
            for y_index in y_indices
            for x_index in x_indices
        )
    proposals: list[VirtualConnectionProposal] = []
    for x_index, y_index in target_index_pairs:
            target_xy = np.array([
                grid.bounds.x_min + x_index * grid.horizontal_spacing_map,
                grid.bounds.y_min + y_index * grid.horizontal_spacing_map,
            ])
            horizontal_map = float(np.linalg.norm(target_xy - position[:2]))
            minimum_loss_map = horizontal_map / parameters.best_glide_ratio
            maximum_target_altitude = position[2] - minimum_loss_map
            fractional_altitude = (
                (maximum_target_altitude - grid.minimum_altitude_map)
                / grid.altitude_spacing_map
            )
            altitude_index = floor(fractional_altitude + GEOMETRY_TOLERANCE)
            if not 0 <= altitude_index < grid.altitude_count:
                continue

            target_preview = np.array([
                target_xy[0],
                target_xy[1],
                grid.minimum_altitude_map
                + altitude_index * grid.altitude_spacing_map,
            ])
            chord = target_preview[:2] - position[:2]
            chord_heading = (
                switching.heading_rad
                if horizontal_map <= GEOMETRY_TOLERANCE
                else float(np.arctan2(chord[1], chord[0]))
            )
            heading_bin = _nearest_heading_bin(graph, chord_heading)
            target_state = BellmanState(
                x_index, y_index, altitude_index, heading_bin,
            )
            target_id = grid.encode(target_state)
            target = grid.position_map(target_state)
            endpoint_distance_map = float(np.linalg.norm(target - position))
            horizontal_m = physical_scale.distance_m(horizontal_map)
            altitude_loss_m = physical_scale.distance_m(position[2] - target[2])
            minimum_loss_m = horizontal_m / parameters.best_glide_ratio
            duration_s = (
                horizontal_m / parameters.best_glide_speed_mps
                if horizontal_m > GEOMETRY_TOLERANCE
                else 0.0
            )
            target_heading = grid.heading_rad(heading_bin)
            heading_mismatch = wrapped_angle_difference(
                switching.heading_rad, target_heading,
            )
            chord_mismatch = wrapped_angle_difference(chord_heading, target_heading)
            node_admissible = bool(graph.node_mask[target_id])
            bellman_reachable = bool(reachable[target_id])
            terrain_feasible = bool(
                node_admissible
                and (
                    endpoint_distance_map <= GEOMETRY_TOLERANCE
                    or not graph.terrain.segment_intersects_solid(position, target)
                )
            )
            altitude_loss_feasible = bool(
                altitude_loss_m >= minimum_loss_m - GEOMETRY_TOLERANCE
                and altitude_loss_m >= -GEOMETRY_TOLERANCE
            )
            duration_geometry_feasible = bool(
                duration_s > GEOMETRY_TOLERANCE
                or endpoint_distance_map <= GEOMETRY_TOLERANCE
            )
            altitude_feasible = bool(
                altitude_loss_feasible and duration_geometry_feasible
            )
            turn_feasible = heading_turn_is_feasible(
                switching.heading_rad,
                target_heading,
                duration_s,
                parameters=parameters,
            )
            chord_heading_feasible = bool(
                chord_mismatch <= CHORD_HEADING_TOLERANCE_RAD
            )
            reasons: list[str] = []
            flags_and_reasons = (
                (switching.powered_feasible, "powered phase infeasible"),
                (evaluation.reachable, "coarse candidate energy infeasible"),
                (node_admissible, "target lattice state excluded by terrain"),
                (terrain_feasible, "virtual segment intersects terrain"),
                (
                    altitude_loss_feasible,
                    "insufficient virtual-segment altitude loss",
                ),
                (
                    duration_geometry_feasible,
                    "zero-duration virtual segment has distinct endpoints",
                ),
                (turn_feasible, "virtual-segment turn-rate limit exceeded"),
                (
                    chord_heading_feasible,
                    "virtual-segment chord does not match target heading bin",
                ),
            )
            for flag, reason in flags_and_reasons:
                if not flag:
                    reasons.append(reason)
            proposals.append(VirtualConnectionProposal(
                candidate_id=evaluation.candidate_id,
                target_state=target_state,
                target_state_id=target_id,
                switching_position_map=position,
                target_position_map=target,
                horizontal_distance_m=horizontal_m,
                altitude_loss_m=max(0.0, altitude_loss_m),
                minimum_glide_loss_m=minimum_loss_m,
                projection_error_m=physical_scale.distance_m(float(
                    np.linalg.norm(target - position)
                )),
                heading_mismatch_rad=heading_mismatch,
                chord_heading_mismatch_rad=chord_mismatch,
                duration_s=duration_s,
                node_admissible=node_admissible,
                bellman_reachable=bellman_reachable,
                terrain_feasible=terrain_feasible,
                altitude_feasible=altitude_feasible,
                turn_feasible=turn_feasible,
                chord_heading_feasible=chord_heading_feasible,
                powered_feasible=switching.powered_feasible,
                coarse_energy_feasible=evaluation.reachable,
                rejection_reasons=tuple(reasons),
            ))
    proposals.sort(key=lambda item: item.target_state_id)
    return VirtualConnectionSet(
        candidate_id=evaluation.candidate_id,
        switching_state=switching,
        proposals=tuple(proposals),
    )


@dataclass(frozen=True)
class MissionEnergyCertificate:
    """Independent total-mechanical-energy replay for one complete glide path."""

    available_specific_height_m: float
    switch_loss_m: float
    virtual_drag_loss_m: float
    glide_drag_loss_m: float
    terminal_altitude_m: float
    required_specific_height_m: float
    margin_m: float
    margin_j: float
    segment_geometry_feasible: bool
    feasible: bool


def certify_mission_energy(
    switching_state: SwitchingState,
    virtual_connection: VirtualConnectionProposal,
    glide_edges: Sequence[GlideEdge],
    graph: BellmanGraph,
    *,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
) -> MissionEnergyCertificate:
    """Replay energy from the continuous switch through the terminal state."""
    if virtual_connection.candidate_id < 0:
        raise ValueError("virtual connection candidate ID must be nonnegative")
    edges = tuple(glide_edges)
    if any(not isinstance(edge, GlideEdge) for edge in edges):
        raise TypeError("glide_edges must contain GlideEdge objects")
    if edges and edges[0].source_id != virtual_connection.target_state_id:
        raise ValueError("first glide edge must start at the virtual target")
    for first, second in zip(edges, edges[1:]):
        if first.target_id != second.source_id:
            raise ValueError("glide edge sequence is not continuous")

    trim_kinetic_height = (
        parameters.best_glide_speed_mps**2 / (2.0 * parameters.gravity_mps2)
    )
    available_height = (
        switching_state.total_mechanical_energy_j
        / (parameters.mass_kg * parameters.gravity_mps2)
        - trim_kinetic_height
    )
    virtual_drag = (
        virtual_connection.horizontal_distance_m / parameters.best_glide_ratio
    )
    glide_drag = sum(
        edge.horizontal_distance_m / parameters.best_glide_ratio for edge in edges
    )
    terminal_id = (
        edges[-1].target_id if edges else virtual_connection.target_state_id
    )
    terminal_position = graph.grid.position_map(graph.grid.decode(terminal_id))
    terminal_altitude_m = physical_scale.distance_m(float(terminal_position[2]))
    required = (
        parameters.switch_energy_loss_height_m
        + virtual_drag
        + glide_drag
        + terminal_altitude_m
    )
    margin_m = available_height - required
    geometry_ok = bool(
        virtual_connection.altitude_feasible
        and all(
            edge.altitude_loss_m
            >= edge.horizontal_distance_m / parameters.best_glide_ratio
            - GEOMETRY_TOLERANCE
            for edge in edges
        )
    )
    feasible = bool(
        switching_state.powered_feasible
        and virtual_connection.feasible
        and geometry_ok
        and margin_m >= -GEOMETRY_TOLERANCE
    )
    return MissionEnergyCertificate(
        available_specific_height_m=float(available_height),
        switch_loss_m=parameters.switch_energy_loss_height_m,
        virtual_drag_loss_m=float(virtual_drag),
        glide_drag_loss_m=float(glide_drag),
        terminal_altitude_m=terminal_altitude_m,
        required_specific_height_m=float(required),
        margin_m=float(margin_m),
        margin_j=float(margin_m * parameters.mass_kg * parameters.gravity_mps2),
        segment_geometry_feasible=geometry_ok,
        feasible=feasible,
    )

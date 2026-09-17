"""Stage-8 end-to-end response for exactly one fixed switching candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from additive_bellman import (
    AdditiveBellmanSolution,
    descendant_mask,
    restricted_adjacency,
    solve_additive_bellman,
)
from bellman_geometry import GlideEdge
from bellman_graph import BellmanGraph, UnitCostReachability
from candidate_energy import CandidateEnergyEvaluation
from detection_hazard import (
    DEFAULT_ATTACKER_HAZARD_TIME,
    AttackerHazardTimeParameters,
    AttackerObjectiveBreakdown,
    HazardField,
    evaluate_attacker_hazard_time_objective,
)
from edge_hazard import (
    HazardPrecomputation,
    SegmentHazardResult,
    integrate_segment_hazard,
    precompute_edge_hazards,
)
from energy_model import (
    DEFAULT_GLIDER,
    DEFAULT_PHYSICAL_SCALE,
    GliderParameters,
    PhysicalScale,
)
from scenario import MissionPoints
from virtual_connection import (
    MissionEnergyCertificate,
    VirtualConnectionProposal,
    VirtualConnectionSet,
    build_virtual_connections,
    certify_mission_energy,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PhaseObjective:
    name: str
    duration_s: float
    hazard: float
    weighted_cost: float
    hazard_assumption: str


@dataclass(frozen=True)
class ConnectionMissionOption:
    """One virtual proposal followed by its exact Bellman continuation."""

    connection: VirtualConnectionProposal
    glide_edges: tuple[GlideEdge, ...]
    virtual_hazard: SegmentHazardResult
    glide_hazard: float
    energy: MissionEnergyCertificate
    objective: AttackerObjectiveBreakdown
    powered_phase: PhaseObjective
    virtual_phase: PhaseObjective
    glide_phase: PhaseObjective
    feasible: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ContinuousMissionReplay:
    """Independent geometry certificate from mission start through terminal."""

    positions_map: FloatArray
    segment_labels: tuple[str, ...]
    maximum_join_gap_m: float
    terminal_goal_distance_m: float
    within_goal_tolerance: bool

    def __post_init__(self) -> None:
        positions = np.array(self.positions_map, dtype=float, copy=True)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_map must have shape (n, 3)")
        if len(self.segment_labels) != len(positions) - 1:
            raise ValueError("segment labels must identify every replay segment")
        if not np.all(np.isfinite(positions)):
            raise ValueError("replay positions must be finite")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_map", positions)


@dataclass(frozen=True)
class SingleCandidateMissionResponse:
    """Stage-8 result; the switching candidate is fixed, never enumerated."""

    candidate_id: int
    candidate_count_considered: int
    connections: VirtualConnectionSet
    active_state_count: int
    hazard_precomputation: HazardPrecomputation
    bellman_solution: AdditiveBellmanSolution
    options: tuple[ConnectionMissionOption, ...]
    selected_option: ConnectionMissionOption | None
    replay: ContinuousMissionReplay | None
    powered_detection_assumption: str

    @property
    def feasible(self) -> bool:
        return self.selected_option is not None


@dataclass(frozen=True)
class PrecomputedCandidateResponse:
    """Candidate-specific result queried from one precomputed Bellman solve."""

    options: tuple[ConnectionMissionOption, ...]
    selected_option: ConnectionMissionOption | None
    replay: ContinuousMissionReplay | None
    powered_detection_assumption: str


def _edge_hazard(
    edge: GlideEdge,
    graph: BellmanGraph,
    hazards: HazardPrecomputation,
) -> float:
    matches = [
        index for index, candidate in enumerate(graph.adjacency[edge.source_id])
        if candidate.target_id == edge.target_id
    ]
    if len(matches) != 1:
        raise RuntimeError("path edge does not have one adjacency match")
    return float(hazards.hazard_by_source[edge.source_id][matches[0]])


def _phase_cost(
    hazard: float,
    duration_s: float,
    objective_parameters: AttackerHazardTimeParameters,
) -> float:
    return float(
        objective_parameters.hazard_weight
        * hazard / objective_parameters.hazard_reference
        + objective_parameters.time_weight
        * duration_s / objective_parameters.time_reference_s
    )


def _build_replay(
    mission_points: MissionPoints,
    evaluation: CandidateEnergyEvaluation,
    option: ConnectionMissionOption,
    graph: BellmanGraph,
    parameters: GliderParameters,
    physical_scale: PhysicalScale,
) -> ContinuousMissionReplay:
    start = mission_points.start.as_array()
    switch = evaluation.switching_state.position_map
    virtual_target = option.connection.target_position_map
    positions = [start, switch, virtual_target]
    labels = ["powered", "virtual"]
    current_id = option.connection.target_state_id
    maximum_gap_m = physical_scale.distance_m(float(
        np.linalg.norm(virtual_target - graph.grid.position_map(graph.grid.decode(current_id)))
    ))
    for edge in option.glide_edges:
        source = graph.grid.position_map(edge.source_state)
        target = graph.grid.position_map(edge.target_state)
        maximum_gap_m = max(
            maximum_gap_m,
            physical_scale.distance_m(float(np.linalg.norm(positions[-1] - source))),
        )
        positions.append(target)
        labels.append("glide")
    terminal = positions[-1]
    goal = mission_points.goal.as_array()
    goal_distance_m = physical_scale.distance_m(float(np.linalg.norm(terminal - goal)))
    return ContinuousMissionReplay(
        positions_map=np.asarray(positions),
        segment_labels=tuple(labels),
        maximum_join_gap_m=maximum_gap_m,
        terminal_goal_distance_m=goal_distance_m,
        within_goal_tolerance=bool(goal_distance_m <= parameters.goal_tolerance_m + 1.0e-12),
    )


def evaluate_precomputed_candidate(
    evaluation: CandidateEnergyEvaluation,
    connections: VirtualConnectionSet,
    graph: BellmanGraph,
    solution: AdditiveBellmanSolution,
    hazards: HazardPrecomputation,
    mission_points: MissionPoints,
    hazard_field: HazardField,
    *,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    objective_parameters: AttackerHazardTimeParameters = DEFAULT_ATTACKER_HAZARD_TIME,
    quadrature_resolution: int = 8,
) -> PrecomputedCandidateResponse:
    """Evaluate candidate-specific phases against a shared Bellman solution."""
    powered_duration = (
        evaluation.switching_state.powered_path_length_m
        / parameters.powered_speed_mps
    )
    powered_assumption = (
        "powered hazard = 0: pre-switch powered flight is outside the "
        "post-switch LOS-gated acoustic detection objective"
    )
    powered_phase = PhaseObjective(
        name="powered",
        duration_s=powered_duration,
        hazard=0.0,
        weighted_cost=_phase_cost(0.0, powered_duration, objective_parameters),
        hazard_assumption=powered_assumption,
    )
    options: list[ConnectionMissionOption] = []
    for connection in connections.feasible_proposals:
        reasons: list[str] = []
        if not solution.goal_reachable[connection.target_state_id]:
            reasons.append("no additive Bellman continuation")
            continue
        glide_edges = solution.path_edges(connection.target_state_id)
        virtual_hazard = integrate_segment_hazard(
            connection.switching_position_map,
            connection.target_position_map,
            connection.duration_s,
            hazard_field,
            quadrature_resolution=quadrature_resolution,
            start_time_s=powered_duration,
            physical_scale=physical_scale,
        )
        glide_hazard = float(sum(
            _edge_hazard(edge, graph, hazards) for edge in glide_edges
        ))
        glide_duration = float(sum(edge.duration_s for edge in glide_edges))
        energy = certify_mission_energy(
            evaluation.switching_state,
            connection,
            glide_edges,
            graph,
            parameters=parameters,
            physical_scale=physical_scale,
        )
        if not energy.feasible:
            reasons.append("full-path total-energy certificate failed")
        virtual_phase = PhaseObjective(
            name="virtual",
            duration_s=connection.duration_s,
            hazard=virtual_hazard.hazard,
            weighted_cost=_phase_cost(
                virtual_hazard.hazard,
                connection.duration_s,
                objective_parameters,
            ),
            hazard_assumption="Stage-7 LOS-gated hazard quadrature",
        )
        glide_phase = PhaseObjective(
            name="glide",
            duration_s=glide_duration,
            hazard=glide_hazard,
            weighted_cost=_phase_cost(
                glide_hazard, glide_duration, objective_parameters,
            ),
            hazard_assumption="Stage-7 LOS-gated hazard quadrature",
        )
        total_hazard = virtual_hazard.hazard + glide_hazard
        total_duration = powered_duration + connection.duration_s + glide_duration
        objective = evaluate_attacker_hazard_time_objective(
            total_hazard,
            total_duration,
            parameters=objective_parameters,
        )
        phase_sum = (
            powered_phase.weighted_cost
            + virtual_phase.weighted_cost
            + glide_phase.weighted_cost
        )
        if not np.isclose(
            objective.objective_value, phase_sum, rtol=0.0, atol=1.0e-10,
        ):
            raise RuntimeError("phase objective decomposition is not additive")
        continuation_value = float(solution.value[connection.target_state_id])
        if not np.isclose(
            continuation_value,
            glide_phase.weighted_cost,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise RuntimeError("Bellman value and glide replay cost disagree")
        options.append(ConnectionMissionOption(
            connection=connection,
            glide_edges=glide_edges,
            virtual_hazard=virtual_hazard,
            glide_hazard=glide_hazard,
            energy=energy,
            objective=objective,
            powered_phase=powered_phase,
            virtual_phase=virtual_phase,
            glide_phase=glide_phase,
            feasible=not reasons,
            rejection_reasons=tuple(reasons),
        ))

    feasible_options = tuple(option for option in options if option.feasible)
    selected = min(
        feasible_options,
        key=lambda option: (
            option.objective.objective_value,
            option.connection.target_state_id,
        ),
        default=None,
    )
    replay = (
        _build_replay(
            mission_points,
            evaluation,
            selected,
            graph,
            parameters,
            physical_scale,
        )
        if selected is not None
        else None
    )
    if replay is not None and (
        replay.maximum_join_gap_m > 1.0e-9 or not replay.within_goal_tolerance
    ):
        raise RuntimeError("continuous mission replay has a gap or misses the goal")
    return PrecomputedCandidateResponse(
        options=tuple(options),
        selected_option=selected,
        replay=replay,
        powered_detection_assumption=powered_assumption,
    )


def solve_single_candidate_response(
    evaluation: CandidateEnergyEvaluation,
    graph: BellmanGraph,
    unit_reachability: UnitCostReachability,
    mission_points: MissionPoints,
    hazard_field: HazardField,
    *,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    objective_parameters: AttackerHazardTimeParameters = DEFAULT_ATTACKER_HAZARD_TIME,
    quadrature_resolution: int = 8,
) -> SingleCandidateMissionResponse:
    """Connect and solve one fixed candidate with precomputed edge hazard.

    The powered segment contributes time but zero hazard under the explicit
    Stage-8 acoustic-neutralization assumption.  Virtual and glide hazards use
    the Stage-7 LOS-gated field.
    """
    connections = build_virtual_connections(
        evaluation,
        graph,
        unit_reachability.goal_reachable,
        parameters=parameters,
        physical_scale=physical_scale,
    )
    feasible_connections = connections.feasible_proposals
    start_ids = tuple(item.target_state_id for item in feasible_connections)
    active = descendant_mask(graph, start_ids) if start_ids else np.zeros(
        graph.grid.state_count, dtype=bool,
    )
    adjacency = restricted_adjacency(graph, active)
    hazards = precompute_edge_hazards(
        adjacency,
        graph.grid,
        hazard_field,
        quadrature_resolution=quadrature_resolution,
        physical_scale=physical_scale,
    )
    edge_costs: list[tuple[float, ...]] = []
    for state_id, edges in enumerate(adjacency):
        edge_costs.append(tuple(
            _phase_cost(
                hazards.hazard_by_source[state_id][edge_index],
                edge.duration_s,
                objective_parameters,
            )
            for edge_index, edge in enumerate(edges)
        ))
    solution = solve_additive_bellman(
        graph,
        edge_costs,
        active_mask=active,
    )

    candidate_response = evaluate_precomputed_candidate(
        evaluation,
        connections,
        graph,
        solution,
        hazards,
        mission_points,
        hazard_field,
        parameters=parameters,
        physical_scale=physical_scale,
        objective_parameters=objective_parameters,
        quadrature_resolution=quadrature_resolution,
    )
    return SingleCandidateMissionResponse(
        candidate_id=evaluation.candidate_id,
        candidate_count_considered=1,
        connections=connections,
        active_state_count=int(np.count_nonzero(active)),
        hazard_precomputation=hazards,
        bellman_solution=solution,
        options=candidate_response.options,
        selected_option=candidate_response.selected_option,
        replay=candidate_response.replay,
        powered_detection_assumption=(
            candidate_response.powered_detection_assumption
        ),
    )

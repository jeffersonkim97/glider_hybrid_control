"""Stage-9 exhaustive Attacker best response for one fixed Defender action."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from additive_bellman import (
    AdditiveBellmanSolution,
    descendant_mask,
    restricted_adjacency,
    solve_additive_bellman,
)
from bellman_graph import BellmanGraph, UnitCostReachability, build_bellman_graph, solve_unit_cost_reachability
from bellman_state import BellmanStateGrid
from candidate_energy import CandidateEnergyEvaluation, evaluate_switching_candidates
from detection_hazard import (
    DEFAULT_ATTACKER_HAZARD_TIME,
    DEFAULT_DETECTION_HAZARD,
    AttackerHazardTimeParameters,
    DetectionHazardParameters,
    GlideDetectionHazardModel,
)
from edge_hazard import HazardPrecomputation, precompute_edge_hazards
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE, GliderParameters, PhysicalScale
from game_types import AttackerInitialCondition, AttackerResponse, DefenderAction
from los_geometry import LOSModel, LOSTangentSurface, TangentContour, VisualizationRaySet
from map_geometry import MapBounds, TerrainModel
from mission_response import (
    ConnectionMissionOption,
    ContinuousMissionReplay,
    PrecomputedCandidateResponse,
    evaluate_precomputed_candidate,
    solve_single_candidate_response,
)
from scenario import MissionPoints, Point3D
from sparse_additive import solve_sparse_additive_bellman
from sparse_reachability import (
    ImplicitReachableAdjacency,
    SparseReachabilityRun,
    build_goal_backward_reachable_graph,
)
from switching_candidates import (
    INITIAL_CONTOUR_SAMPLE_COUNT,
    INITIAL_RADIAL_SCALES,
    SwitchingCandidate,
    generate_switching_candidates,
)
from terrain_catalog import build_terrain
from virtual_connection import VirtualConnectionSet, build_virtual_connections


DEFAULT_GRAPH_BOUNDS = MapBounds(-8.0, 8.0, -4.0, 4.0)
ATTACKER_TIE_BREAK_CONVENTION = (
    "retain every objective-co-optimal candidate within tolerance; select the "
    "lowest candidate_id for the Attacker response"
)


@dataclass(frozen=True)
class CandidateSelectionScore:
    candidate_id: int
    feasible: bool
    objective: float | None


@dataclass(frozen=True)
class CandidateMissionResult:
    """Complete stored outcome or explicit failure for one candidate."""

    evaluation: CandidateEnergyEvaluation
    connections: VirtualConnectionSet | None
    precomputed_response: PrecomputedCandidateResponse | None
    powered_feasible: bool
    energy_feasible: bool
    virtual_feasible: bool
    goal_reachable: bool
    feasible: bool
    mission_time_s: float | None
    cumulative_hazard: float | None
    detection_probability: float | None
    objective: float | None
    infeasibility_reason: str | None

    @property
    def candidate(self) -> SwitchingCandidate:
        return self.evaluation.candidate

    @property
    def candidate_id(self) -> int:
        return self.evaluation.candidate_id

    @property
    def position_map(self) -> np.ndarray:
        return self.evaluation.candidate.position_map

    @property
    def selected_option(self) -> ConnectionMissionOption | None:
        if self.precomputed_response is None:
            return None
        return self.precomputed_response.selected_option

    @property
    def replay(self) -> ContinuousMissionReplay | None:
        if self.precomputed_response is None:
            return None
        return self.precomputed_response.replay

    @property
    def status_category(self) -> str:
        if self.feasible:
            return "feasible"
        if not self.powered_feasible:
            return "powered_infeasible"
        if not self.energy_feasible:
            return "energy_infeasible"
        if not self.virtual_feasible:
            return "virtual_infeasible"
        if not self.goal_reachable:
            return "goal_unreachable"
        return "path_energy_infeasible"

    @property
    def selection_score(self) -> CandidateSelectionScore:
        return CandidateSelectionScore(
            candidate_id=self.candidate_id,
            feasible=self.feasible,
            objective=self.objective,
        )


@dataclass(frozen=True)
class AttackerBestResponseTiming:
    los_s: float
    candidate_generation_s: float
    energy_filter_s: float
    virtual_connection_s: float
    graph_build_s: float
    hazard_precompute_s: float
    bellman_solve_s: float
    attacker_bellman_s: float
    total_attacker_br_s: float


@dataclass(frozen=True)
class AttackerBestResponseMetrics:
    timing: AttackerBestResponseTiming
    number_of_candidates: int
    number_energy_feasible: int
    number_virtual_feasible: int
    number_goal_reachable: int
    number_feasible: int
    shared_bellman_solve_count: int
    active_state_count: int
    hazard_edge_count: int
    bellman_backend: str = "dense"
    goal_reachable_state_count: int = 0


@dataclass(frozen=True)
class AttackerBestResponseRun:
    defender_action: DefenderAction
    scenario: AttackerInitialCondition
    terrain: TerrainModel
    mission_points: MissionPoints
    tangent_contour: TangentContour
    visualization_rays: VisualizationRaySet
    los_surface: LOSTangentSurface
    graph: BellmanGraph
    unit_reachability: UnitCostReachability
    candidate_results: tuple[CandidateMissionResult, ...]
    selected_result: CandidateMissionResult | None
    cooptimal_candidate_ids: tuple[int, ...]
    response: AttackerResponse
    metrics: AttackerBestResponseMetrics
    tie_break_convention: str = ATTACKER_TIE_BREAK_CONVENTION


def select_attacker_candidate(
    scores: Sequence[CandidateSelectionScore],
    *,
    objective_tolerance: float = 1.0e-12,
) -> tuple[int | None, tuple[int, ...]]:
    """Select the lowest-ID member while retaining all Attacker co-optima."""
    tolerance = float(objective_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("objective_tolerance must be finite and nonnegative")
    feasible = tuple(score for score in scores if score.feasible)
    if not feasible:
        return None, ()
    if any(score.objective is None or not np.isfinite(score.objective) for score in feasible):
        raise ValueError("every feasible candidate score needs a finite objective")
    minimum = min(float(score.objective) for score in feasible)
    cooptimal = tuple(sorted(
        score.candidate_id
        for score in feasible
        if float(score.objective) <= minimum + tolerance
    ))
    return cooptimal[0], cooptimal


def _failure_result(
    evaluation: CandidateEnergyEvaluation,
    *,
    connections: VirtualConnectionSet | None = None,
    powered_feasible: bool,
    energy_feasible: bool,
    virtual_feasible: bool,
    goal_reachable: bool,
    reason: str,
) -> CandidateMissionResult:
    return CandidateMissionResult(
        evaluation=evaluation,
        connections=connections,
        precomputed_response=None,
        powered_feasible=powered_feasible,
        energy_feasible=energy_feasible,
        virtual_feasible=virtual_feasible,
        goal_reachable=goal_reachable,
        feasible=False,
        mission_time_s=None,
        cumulative_hazard=None,
        detection_probability=None,
        objective=None,
        infeasibility_reason=reason,
    )


def _result_from_precomputed(
    evaluation: CandidateEnergyEvaluation,
    connections: VirtualConnectionSet,
    response: PrecomputedCandidateResponse,
    solution: AdditiveBellmanSolution,
) -> CandidateMissionResult:
    goal_reachable = any(
        solution.goal_reachable[item.target_state_id]
        for item in connections.feasible_proposals
    )
    selected = response.selected_option
    if selected is None:
        reason = (
            "no Bellman path to goal"
            if not goal_reachable
            else "full-path total-energy certificate failed"
        )
        return CandidateMissionResult(
            evaluation=evaluation,
            connections=connections,
            precomputed_response=response,
            powered_feasible=evaluation.powered_feasible,
            energy_feasible=evaluation.glide_result.reachable,
            virtual_feasible=connections.has_connection,
            goal_reachable=goal_reachable,
            feasible=False,
            mission_time_s=None,
            cumulative_hazard=None,
            detection_probability=None,
            objective=None,
            infeasibility_reason=reason,
        )
    objective = selected.objective
    return CandidateMissionResult(
        evaluation=evaluation,
        connections=connections,
        precomputed_response=response,
        powered_feasible=evaluation.powered_feasible,
        energy_feasible=evaluation.glide_result.reachable,
        virtual_feasible=True,
        goal_reachable=True,
        feasible=True,
        mission_time_s=objective.mission_time_s,
        cumulative_hazard=objective.mission_hazard,
        detection_probability=objective.mission_pod,
        objective=objective.objective_value,
        infeasibility_reason=None,
    )


def _prefilter_or_connections(
    evaluations: Sequence[CandidateEnergyEvaluation],
    graph: BellmanGraph,
    unit_reachability: UnitCostReachability,
    *,
    parameters: GliderParameters,
    physical_scale: PhysicalScale,
) -> tuple[dict[int, CandidateMissionResult], dict[int, VirtualConnectionSet]]:
    failures: dict[int, CandidateMissionResult] = {}
    connections_by_id: dict[int, VirtualConnectionSet] = {}
    for evaluation in evaluations:
        if not evaluation.powered_feasible:
            failures[evaluation.candidate_id] = _failure_result(
                evaluation, powered_feasible=False, energy_feasible=False,
                virtual_feasible=False, goal_reachable=False,
                reason=evaluation.infeasibility_reason or "powered phase infeasible",
            )
            continue
        energy_feasible = bool(evaluation.glide_result.reachable)
        if not evaluation.acoustically_neutralized:
            failures[evaluation.candidate_id] = _failure_result(
                evaluation, powered_feasible=True, energy_feasible=energy_feasible,
                virtual_feasible=False, goal_reachable=False,
                reason="switching candidate is not on LOS tangent surface",
            )
            continue
        if not energy_feasible:
            failures[evaluation.candidate_id] = _failure_result(
                evaluation, powered_feasible=True, energy_feasible=False,
                virtual_feasible=False, goal_reachable=False,
                reason=evaluation.infeasibility_reason or "insufficient total energy",
            )
            continue
        connections = build_virtual_connections(
            evaluation, graph, unit_reachability.goal_reachable,
            parameters=parameters, physical_scale=physical_scale,
        )
        connections_by_id[evaluation.candidate_id] = connections
        if not connections.has_connection:
            rejection_reasons = sorted({
                reason for proposal in connections.proposals
                for reason in proposal.rejection_reasons
            })
            suffix = "; ".join(rejection_reasons) if rejection_reasons else "outside lattice domain"
            failures[evaluation.candidate_id] = _failure_result(
                evaluation, connections=connections, powered_feasible=True,
                energy_feasible=True, virtual_feasible=False,
                goal_reachable=False,
                reason=f"no admissible virtual connection: {suffix}",
            )
    return failures, connections_by_id


def _shared_solution(
    graph: BellmanGraph,
    connections_by_id: dict[int, VirtualConnectionSet],
    hazard_field: GlideDetectionHazardModel,
    *,
    objective_parameters: AttackerHazardTimeParameters,
    physical_scale: PhysicalScale,
    quadrature_resolution: int,
    parameters: GliderParameters = DEFAULT_GLIDER,
) -> tuple[AdditiveBellmanSolution, HazardPrecomputation, int, float]:
    start_ids = tuple(sorted({
        proposal.target_state_id
        for connections in connections_by_id.values()
        for proposal in connections.feasible_proposals
        if proposal.bellman_reachable
    }))
    if isinstance(graph.adjacency, ImplicitReachableAdjacency):
        sparse = solve_sparse_additive_bellman(
            graph,
            start_ids,
            hazard_field,
            objective_parameters=objective_parameters,
            parameters=parameters,
            physical_scale=physical_scale,
            quadrature_resolution=quadrature_resolution,
        )
        return (
            sparse.solution,
            sparse.hazards,
            sparse.metrics.active_state_count,
            sparse.metrics.solve_s,
        )
    active = descendant_mask(graph, start_ids) if start_ids else np.zeros(
        graph.grid.state_count, dtype=bool,
    )
    adjacency = restricted_adjacency(graph, active)
    hazards = precompute_edge_hazards(
        adjacency, graph.grid, hazard_field,
        quadrature_resolution=quadrature_resolution,
        physical_scale=physical_scale,
    )
    edge_costs = tuple(
        tuple(
            objective_parameters.hazard_weight
            * hazards.hazard_by_source[state_id][edge_index]
            / objective_parameters.hazard_reference
            + objective_parameters.time_weight * edge.duration_s
            / objective_parameters.time_reference_s
            for edge_index, edge in enumerate(edges)
        )
        for state_id, edges in enumerate(adjacency)
    )
    solve_start = perf_counter()
    solution = solve_additive_bellman(graph, edge_costs, active_mask=active)
    solve_s = perf_counter() - solve_start
    return solution, hazards, int(np.count_nonzero(active)), solve_s


def evaluate_candidates_shared(
    evaluations: Sequence[CandidateEnergyEvaluation],
    graph: BellmanGraph,
    unit_reachability: UnitCostReachability,
    mission_points: MissionPoints,
    hazard_field: GlideDetectionHazardModel,
    *,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    objective_parameters: AttackerHazardTimeParameters = DEFAULT_ATTACKER_HAZARD_TIME,
    quadrature_resolution: int = 8,
) -> tuple[tuple[CandidateMissionResult, ...], AdditiveBellmanSolution, HazardPrecomputation, int]:
    """Evaluate a finite sequence using one exact shared Bellman recursion."""
    sequence = tuple(evaluations)
    failures, connections_by_id = _prefilter_or_connections(
        sequence, graph, unit_reachability,
        parameters=parameters, physical_scale=physical_scale,
    )
    solution, hazards, active_count, _ = _shared_solution(
        graph, connections_by_id, hazard_field,
        objective_parameters=objective_parameters,
        physical_scale=physical_scale,
        quadrature_resolution=quadrature_resolution,
        parameters=parameters,
    )
    results: list[CandidateMissionResult] = []
    for evaluation in sequence:
        if evaluation.candidate_id in failures:
            results.append(failures[evaluation.candidate_id])
            continue
        connections = connections_by_id[evaluation.candidate_id]
        candidate_response = evaluate_precomputed_candidate(
            evaluation, connections, graph, solution, hazards,
            mission_points, hazard_field, parameters=parameters,
            physical_scale=physical_scale,
            objective_parameters=objective_parameters,
            quadrature_resolution=quadrature_resolution,
        )
        results.append(_result_from_precomputed(
            evaluation, connections, candidate_response, solution,
        ))
    return tuple(results), solution, hazards, active_count


def evaluate_candidates_literal(
    evaluations: Sequence[CandidateEnergyEvaluation],
    graph: BellmanGraph,
    unit_reachability: UnitCostReachability,
    mission_points: MissionPoints,
    hazard_field: GlideDetectionHazardModel,
    *,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    objective_parameters: AttackerHazardTimeParameters = DEFAULT_ATTACKER_HAZARD_TIME,
    quadrature_resolution: int = 8,
) -> tuple[CandidateMissionResult, ...]:
    """Literal reference: solve every eligible candidate independently."""
    sequence = tuple(evaluations)
    failures, _ = _prefilter_or_connections(
        sequence, graph, unit_reachability,
        parameters=parameters, physical_scale=physical_scale,
    )
    results: list[CandidateMissionResult] = []
    for evaluation in sequence:
        if evaluation.candidate_id in failures:
            results.append(failures[evaluation.candidate_id])
            continue
        single = solve_single_candidate_response(
            evaluation, graph, unit_reachability, mission_points, hazard_field,
            parameters=parameters, physical_scale=physical_scale,
            objective_parameters=objective_parameters,
            quadrature_resolution=quadrature_resolution,
        )
        precomputed = PrecomputedCandidateResponse(
            options=single.options, selected_option=single.selected_option,
            replay=single.replay,
            powered_detection_assumption=single.powered_detection_assumption,
        )
        results.append(_result_from_precomputed(
            evaluation, single.connections, precomputed, single.bellman_solution,
        ))
    return tuple(results)


def attacker_response_from_result(
    selected: CandidateMissionResult | None,
) -> AttackerResponse:
    """Convert one stored candidate result into the stable game API object."""
    if selected is None:
        return AttackerResponse(
            feasible=False, switching_point_map=None, discrete_states=(),
            objective=float("inf"), mission_time_s=float("inf"),
            cumulative_hazard=0.0, detection_probability=0.0,
        )
    option = selected.selected_option
    if option is None:
        raise RuntimeError("selected candidate has no trajectory option")
    state_sequence = (
        (option.connection.target_state,)
        + tuple(edge.target_state for edge in option.glide_edges)
    )
    discrete_states = tuple(
        (state.x_index, state.y_index, state.altitude_index, state.heading_bin)
        for state in state_sequence
    )
    return AttackerResponse(
        feasible=True, switching_point_map=selected.position_map,
        discrete_states=discrete_states, objective=float(selected.objective),
        mission_time_s=float(selected.mission_time_s),
        cumulative_hazard=float(selected.cumulative_hazard),
        detection_probability=float(selected.detection_probability),
    )


# Backward-compatible private name retained for the Stage-9 unit tests.
_attacker_response = attacker_response_from_result


def run_attacker_best_response(
    defender_action: DefenderAction,
    scenario: AttackerInitialCondition,
    *,
    terrain: TerrainModel | None = None,
    graph_bounds: MapBounds = DEFAULT_GRAPH_BOUNDS,
    bellman_grid: BellmanStateGrid | None = None,
    contour_sample_count: int = INITIAL_CONTOUR_SAMPLE_COUNT,
    radial_scales: Iterable[float] = INITIAL_RADIAL_SCALES,
    quadrature_resolution: int = 8,
    los_probe_grid_size: int = 101,
    los_boundary_refinement_steps: int = 24,
    los_display_extension_factor: float = 4.0,
    visualization_ray_count: int = 10,
    objective_tolerance: float = 1.0e-12,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    detection_parameters: DetectionHazardParameters = DEFAULT_DETECTION_HAZARD,
    objective_parameters: AttackerHazardTimeParameters = DEFAULT_ATTACKER_HAZARD_TIME,
    bellman_backend: str = "dense",
    prebuilt_reachability: SparseReachabilityRun | None = None,
) -> AttackerBestResponseRun:
    """Exhaustively score every generated candidate with one proven shared solve."""
    if not isinstance(defender_action, DefenderAction):
        raise TypeError("defender_action must be a DefenderAction")
    if not isinstance(scenario, AttackerInitialCondition):
        raise TypeError("scenario must be an AttackerInitialCondition")
    total_start = perf_counter()
    active_terrain = terrain or build_terrain("centered_cube")
    mission = MissionPoints(
        sensor=Point3D(*map(float, defender_action.sensor_position_map)),
        start=scenario.start, goal=scenario.goal,
    )
    mission.validate_against(active_terrain)
    los_start = perf_counter()
    los_model = LOSModel(active_terrain)
    tangent_contour = los_model.trace_tangent_contour(
        mission.sensor, probe_grid_size=los_probe_grid_size,
        boundary_refinement_steps=los_boundary_refinement_steps,
    )
    visualization_rays = tangent_contour.sample_for_visualization(
        visualization_ray_count,
    )
    los_surface = los_model.build_tangent_surface(
        tangent_contour,
        display_extension_factor=los_display_extension_factor,
    )
    los_s = perf_counter() - los_start

    generation_start = perf_counter()
    candidates = generate_switching_candidates(
        tangent_contour, contour_sample_count=contour_sample_count,
        radial_scales=radial_scales,
    )
    candidate_generation_s = perf_counter() - generation_start
    energy_start = perf_counter()
    evaluations = evaluate_switching_candidates(
        candidates, tangent_contour, active_terrain, mission,
    )
    energy_filter_s = perf_counter() - energy_start

    active_grid = bellman_grid or BellmanStateGrid(graph_bounds)
    if bellman_backend not in {"dense", "goal_backward_sparse"}:
        raise ValueError(
            "bellman_backend must be 'dense' or 'goal_backward_sparse'"
        )
    if prebuilt_reachability is not None and bellman_backend != "goal_backward_sparse":
        raise ValueError(
            "prebuilt_reachability requires goal_backward_sparse backend"
        )
    graph_start = perf_counter()
    if prebuilt_reachability is None and bellman_backend == "dense":
        graph = build_bellman_graph(
            active_grid, active_terrain, mission.goal,
            parameters=parameters, physical_scale=physical_scale,
        )
        unit = solve_unit_cost_reachability(graph)
    elif prebuilt_reachability is None:
        sparse_reachability = build_goal_backward_reachable_graph(
            active_grid,
            active_terrain,
            mission.goal,
            parameters=parameters,
            physical_scale=physical_scale,
        )
        graph = sparse_reachability.graph
        unit = sparse_reachability.unit_reachability
    else:
        graph = prebuilt_reachability.graph
        unit = prebuilt_reachability.unit_reachability
        if graph.grid != active_grid:
            raise ValueError("prebuilt reachability grid does not match bellman_grid")
        if graph.terrain is not active_terrain:
            raise ValueError("prebuilt reachability terrain must be the same object")
        if graph.goal != mission.goal:
            raise ValueError("prebuilt reachability goal does not match scenario")
    graph_and_unit_s = perf_counter() - graph_start
    hazard_field = GlideDetectionHazardModel(
        active_terrain,
        mission.sensor,
        parameters=detection_parameters,
        physical_scale=physical_scale,
    )
    virtual_start = perf_counter()
    failures, connections_by_id = _prefilter_or_connections(
        evaluations, graph, unit,
        parameters=parameters, physical_scale=physical_scale,
    )
    virtual_connection_s = perf_counter() - virtual_start

    bellman_start = perf_counter()
    solution, hazards, active_count, bellman_solve_s = _shared_solution(
        graph, connections_by_id, hazard_field,
        objective_parameters=objective_parameters,
        physical_scale=physical_scale,
        quadrature_resolution=quadrature_resolution,
        parameters=parameters,
    )
    results: list[CandidateMissionResult] = []
    for evaluation in evaluations:
        if evaluation.candidate_id in failures:
            results.append(failures[evaluation.candidate_id])
            continue
        connections = connections_by_id[evaluation.candidate_id]
        candidate_response = evaluate_precomputed_candidate(
            evaluation, connections, graph, solution, hazards, mission,
            hazard_field, parameters=parameters, physical_scale=physical_scale,
            objective_parameters=objective_parameters,
            quadrature_resolution=quadrature_resolution,
        )
        results.append(_result_from_precomputed(
            evaluation, connections, candidate_response, solution,
        ))
    attacker_bellman_s = graph_and_unit_s + (perf_counter() - bellman_start)
    result_tuple = tuple(results)

    selected_id, cooptimal_ids = select_attacker_candidate(
        tuple(result.selection_score for result in result_tuple),
        objective_tolerance=objective_tolerance,
    )
    selected = next(
        (result for result in result_tuple if result.candidate_id == selected_id),
        None,
    )
    feasible_objectives = tuple(
        float(result.objective) for result in result_tuple if result.feasible
    )
    if selected is not None and not np.isclose(
        float(selected.objective), min(feasible_objectives),
        rtol=0.0, atol=objective_tolerance,
    ):
        raise RuntimeError("selected candidate is not the exhaustive minimum")
    response = attacker_response_from_result(selected)
    total_s = perf_counter() - total_start
    metrics = AttackerBestResponseMetrics(
        timing=AttackerBestResponseTiming(
            los_s=los_s,
            candidate_generation_s=candidate_generation_s,
            energy_filter_s=energy_filter_s,
            virtual_connection_s=virtual_connection_s,
            graph_build_s=graph_and_unit_s,
            hazard_precompute_s=hazards.timing.precompute_s,
            bellman_solve_s=bellman_solve_s,
            attacker_bellman_s=attacker_bellman_s,
            total_attacker_br_s=total_s,
        ),
        number_of_candidates=len(result_tuple),
        number_energy_feasible=sum(result.energy_feasible for result in result_tuple),
        number_virtual_feasible=sum(result.virtual_feasible for result in result_tuple),
        number_goal_reachable=sum(result.goal_reachable for result in result_tuple),
        number_feasible=sum(result.feasible for result in result_tuple),
        shared_bellman_solve_count=1,
        active_state_count=active_count,
        hazard_edge_count=hazards.timing.edge_count,
        bellman_backend=bellman_backend,
        goal_reachable_state_count=unit.goal_reachable_state_count,
    )
    return AttackerBestResponseRun(
        defender_action=defender_action, scenario=scenario,
        terrain=active_terrain, mission_points=mission,
        tangent_contour=tangent_contour, visualization_rays=visualization_rays,
        los_surface=los_surface, graph=graph, unit_reachability=unit,
        candidate_results=result_tuple, selected_result=selected,
        cooptimal_candidate_ids=cooptimal_ids, response=response, metrics=metrics,
    )


def attacker_best_response(
    defender_action: DefenderAction,
    scenario: AttackerInitialCondition,
    **kwargs: object,
) -> AttackerResponse:
    """Return the exact finite-candidate Attacker response to one commitment."""
    return run_attacker_best_response(defender_action, scenario, **kwargs).response


__all__ = [
    "ATTACKER_TIE_BREAK_CONVENTION", "AttackerBestResponseMetrics",
    "AttackerBestResponseRun", "AttackerBestResponseTiming",
    "CandidateMissionResult", "CandidateSelectionScore",
    "attacker_response_from_result",
    "attacker_best_response", "evaluate_candidates_literal",
    "evaluate_candidates_shared", "run_attacker_best_response",
    "select_attacker_candidate",
]

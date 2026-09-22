"""Exact local SSE: every computation for the exact side lives here.

For one sensor position ``d`` the best response runs four stages:

    1  LOS tangent surface from d
    2  energy / powered filter on the lattice states near that surface
       (backward reachability is shared across d and built once in P1b_condition)
    3  Bellman over the glide lattice -> glide-phase trajectory
    4  powered flight, a straight line from start to the switching point

The outer loop varies ``d`` over the Defender grid within a Chebyshev radius
``r`` and climbs to a local SSE.

The switching point is a lattice state, chosen by the Bellman value itself:
since ``V(s)`` is already the weighted cost-to-go and the powered leg is a
closed-form straight line with no hazard, the attacker's problem is

    min over admissible s of   [ powered_cost(s) + V(s) ]

Sampling the ruled tangent surface and snapping to lattice states gives the
admissible set; each one is then scored by a single array lookup.  Hazard and
time are recovered by walking the stored path only for the co-optimal set, since
the weighted value cannot be split back into its two parts.

Timing is reported per stage so the RL side can be compared stage by stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from P1b_condition import Scene
from bellman_state import BellmanState
from detection_hazard import (
    GlideDetectionHazardModel,
    hazard_to_detection_probability,
)
from energy_model import StraightPoweredPhaseModel, SwitchingState
from local_sse_contract import DefenderNeighborhoodConfig, LocalDefenderEvaluation
from local_sse_search import run_local_sse_search
from los_geometry import LOSModel, TangentContour
from sparse_additive import solve_sparse_additive_bellman


FloatArray = NDArray[np.float64]

CONTOUR_SAMPLES = 400
RADIAL_SAMPLES = 120

STAGE_KEYS = (
    "T_stage1_los_s",
    "T_stage2_filter_s",
    "T_stage3_bellman_s",
    "T_stage4_powered_s",
)


@dataclass(frozen=True)
class SwitchOption:
    """One admissible lattice switching state and its total mission cost."""

    state_id: int
    position_map: FloatArray
    switching_state: SwitchingState
    powered_duration_s: float
    powered_cost: float
    glide_cost: float
    total_cost: float


@dataclass(frozen=True)
class ExactBestResponse:
    """One exact best response at one sensor position."""

    sensor_map: tuple[float, float, float]
    feasible: bool
    attacker_objective: float | None
    detection_probability: float | None
    mission_time_s: float | None
    cumulative_hazard: float | None
    switching_state_id: int | None
    switching_position_map: list[float] | None
    cooptimal_state_ids: tuple[int, ...]
    glide_edge_count: int
    admissible_switch_states: int
    timing: dict[str, float]
    sizes: dict[str, Any]
    glide_state_ids: tuple[int, ...] = ()
    admissible_state_ids: tuple[int, ...] = ()
    # Only populated when keep_solution=True.  The local search must not retain
    # these: each one holds the solved corridor and a few dozen would exhaust
    # memory at a fine lattice.
    solution: Any = None
    hazards: Any = None


@dataclass(frozen=True)
class ExactLocalSSE:
    """Result of the Defender-side local search."""

    condition_label: str
    feasible: bool
    selected_action_id: int | None
    selected_sensor_map: list[float] | None
    attacker_objective: float | None
    detection_probability: float | None
    unique_evaluations: int
    cached_reuses: int
    iterations: int
    termination_status: str
    visited_action_ids: tuple[int, ...]
    evaluated: tuple[dict[str, Any], ...]
    timing: dict[str, float]
    sizes: dict[str, Any]


def _phase_cost(hazard: float, duration_s: float, objective: Any) -> float:
    return (
        objective.hazard_weight * float(hazard) / objective.hazard_reference
        + objective.time_weight * float(duration_s) / objective.time_reference_s
    )


def _sample_tangent_surface(
    contour: TangentContour, radial_minimum: float, radial_maximum: float,
) -> FloatArray:
    """Points on the ruled surface ``origin + t * tangent(f)``.

    Two parameters only, so covering it densely enough that every lattice cell
    it crosses is hit is cheap.  Density beyond that changes nothing: points are
    deduplicated by the lattice state they snap to.
    """
    origin = contour.origin.as_array()
    fractions = np.linspace(0.0, 1.0, CONTOUR_SAMPLES, endpoint=False)
    scales = np.linspace(float(radial_minimum), float(radial_maximum), RADIAL_SAMPLES)
    directions = np.asarray(
        [contour.tangent_vector_at(float(value)) for value in fractions], dtype=float,
    )
    points = origin[None, None, :] + scales[None, :, None] * directions[:, None, :]
    return points.reshape(-1, 3)


def _snap_to_lattice(
    points: FloatArray, headings: FloatArray, grid: Any,
) -> dict[int, FloatArray]:
    """Lattice state for each surface point; heading is the powered arrival bin."""
    states: dict[int, FloatArray] = {}
    for point, heading in zip(points, headings):
        x_index = int(round((point[0] - grid.bounds.x_min) / grid.horizontal_spacing_map))
        y_index = int(round((point[1] - grid.bounds.y_min) / grid.horizontal_spacing_map))
        altitude_index = int(round(
            (point[2] - grid.minimum_altitude_map) / grid.altitude_spacing_map
        ))
        if not (
            0 <= x_index < grid.x_count
            and 0 <= y_index < grid.y_count
            and 0 <= altitude_index < grid.altitude_count
        ):
            continue
        heading_bin = int(round(
            float(heading) % (2.0 * np.pi) / (2.0 * np.pi) * grid.heading_bin_count
        )) % grid.heading_bin_count
        state_id = grid.encode(
            BellmanState(x_index, y_index, altitude_index, heading_bin)
        )
        if state_id not in states:
            states[state_id] = grid.position_map(
                BellmanState(x_index, y_index, altitude_index, heading_bin)
            )
    return states


def _edge_hazard(edge: Any, graph: Any, hazards: Any) -> float:
    matches = [
        index for index, candidate in enumerate(graph.adjacency[edge.source_id])
        if candidate.target_id == edge.target_id
    ]
    if len(matches) != 1:
        raise RuntimeError("path edge does not have one adjacency match")
    return float(hazards.hazard_by_source[edge.source_id][matches[0]])


def exact_best_response(
    scene: Scene, sensor_map: tuple[float, float, float],
    *, keep_solution: bool = False,
) -> ExactBestResponse:
    """Four-stage exact best response at one sensor position.

    ``keep_solution`` attaches the solved value function and hazard table for
    plotting.  Leave it off inside any search loop.
    """
    config = scene.config
    grid = scene.grid
    discretization = config.discretization
    mission = scene.mission_for(sensor_map)
    timing: dict[str, float] = {}

    # --- stage 1: LOS tangent surface from d ---------------------------------
    started = perf_counter()
    contour = LOSModel(scene.terrain).trace_tangent_contour(
        mission.sensor,
        probe_grid_size=discretization.los_probe_grid_size,
        boundary_refinement_steps=discretization.los_boundary_refinement_steps,
    )
    points = _sample_tangent_surface(
        contour,
        discretization.switching_radial_min,
        discretization.switching_radial_max,
    )
    timing["T_stage1_los_s"] = perf_counter() - started

    # --- stage 2: energy / powered filter on the snapped lattice states ------
    started = perf_counter()
    powered_model = StraightPoweredPhaseModel(
        parameters=config.glider, physical_scale=config.physical_scale,
    )
    headings = np.asarray(
        [
            powered_model.state_at(point, mission, scene.terrain).heading_rad
            for point in points
        ],
        dtype=float,
    )
    snapped = _snap_to_lattice(points, headings, grid)
    usable: dict[int, tuple[FloatArray, SwitchingState]] = {}
    for state_id, position in snapped.items():
        if not scene.graph.node_mask[state_id] or not scene.goal_reachable[state_id]:
            continue
        switching_state = powered_model.state_at(position, mission, scene.terrain)
        if switching_state.powered_feasible:
            usable[state_id] = (position, switching_state)
    timing["T_stage2_filter_s"] = perf_counter() - started

    sizes = scene.sizes() | {
        "surface_samples": len(points),
        "snapped_states": len(snapped),
        "admissible_switch_states": len(usable),
    }

    if not usable:
        timing["T_stage3_bellman_s"] = 0.0
        timing["T_stage4_powered_s"] = 0.0
        timing["T_total_s"] = sum(timing.values())
        return ExactBestResponse(
            sensor_map=tuple(float(v) for v in sensor_map), feasible=False,
            attacker_objective=None, detection_probability=None,
            mission_time_s=None, cumulative_hazard=None,
            switching_state_id=None, switching_position_map=None,
            cooptimal_state_ids=(), glide_edge_count=0,
            admissible_switch_states=0, timing=timing, sizes=sizes,
        )

    # --- stage 3: Bellman over the glide lattice -----------------------------
    started = perf_counter()
    hazard_field = GlideDetectionHazardModel(
        scene.terrain, mission.sensor,
        parameters=config.detection, physical_scale=config.physical_scale,
    )
    start_ids = tuple(sorted(usable))
    sparse = solve_sparse_additive_bellman(
        scene.graph, start_ids, hazard_field,
        objective_parameters=config.attacker_objective,
        parameters=config.glider,
        physical_scale=config.physical_scale,
        quadrature_resolution=discretization.hazard_quadrature_resolution,
    )
    solution = sparse.solution
    timing["T_stage3_bellman_s"] = perf_counter() - started
    sizes["N_S_active"] = sparse.metrics.active_state_count
    sizes["N_E"] = sparse.hazards.timing.edge_count

    # --- stage 4: powered leg and the switch that minimises the total --------
    started = perf_counter()
    options: list[SwitchOption] = []
    for state_id in start_ids:
        glide_cost = float(solution.value[state_id])
        if not np.isfinite(glide_cost) or not solution.goal_reachable[state_id]:
            continue
        position, switching_state = usable[state_id]
        powered_duration = (
            switching_state.powered_path_length_m / config.glider.powered_speed_mps
        )
        powered_cost = _phase_cost(0.0, powered_duration, config.attacker_objective)
        options.append(SwitchOption(
            state_id=state_id, position_map=position,
            switching_state=switching_state,
            powered_duration_s=powered_duration, powered_cost=powered_cost,
            glide_cost=glide_cost, total_cost=powered_cost + glide_cost,
        ))

    if not options:
        timing["T_stage4_powered_s"] = perf_counter() - started
        timing["T_total_s"] = sum(timing.values())
        return ExactBestResponse(
            sensor_map=tuple(float(v) for v in sensor_map), feasible=False,
            attacker_objective=None, detection_probability=None,
            mission_time_s=None, cumulative_hazard=None,
            switching_state_id=None, switching_position_map=None,
            cooptimal_state_ids=(), glide_edge_count=0,
            admissible_switch_states=len(usable), timing=timing, sizes=sizes,
        )

    minimum = min(option.total_cost for option in options)
    tied = sorted(
        (o for o in options if o.total_cost <= minimum + 1.0e-12),
        key=lambda o: o.state_id,
    )
    # Strong Stackelberg follower rule: among the attacker's co-optima take the
    # response that maximises the Defender's detection probability.
    scored = []
    for option in tied:
        edges = solution.path_edges(option.state_id)
        glide_duration = float(sum(edge.duration_s for edge in edges))
        glide_hazard = float(sum(
            _edge_hazard(edge, scene.graph, sparse.hazards) for edge in edges
        ))
        scored.append((option, edges, glide_duration, glide_hazard,
                       hazard_to_detection_probability(glide_hazard)))
    option, edges, glide_duration, glide_hazard, detection = max(
        scored, key=lambda item: (item[4], -item[0].state_id),
    )
    timing["T_stage4_powered_s"] = perf_counter() - started
    timing["T_total_s"] = sum(timing[key] for key in STAGE_KEYS)

    return ExactBestResponse(
        sensor_map=tuple(float(v) for v in sensor_map),
        feasible=True,
        attacker_objective=option.total_cost,
        detection_probability=detection,
        mission_time_s=option.powered_duration_s + glide_duration,
        cumulative_hazard=glide_hazard,
        switching_state_id=option.state_id,
        switching_position_map=[float(v) for v in option.position_map],
        cooptimal_state_ids=tuple(o.state_id for o in tied),
        glide_edge_count=len(edges),
        admissible_switch_states=len(usable),
        timing=timing, sizes=sizes,
        glide_state_ids=(option.state_id,) + tuple(int(e.target_id) for e in edges),
        admissible_state_ids=start_ids,
        solution=solution if keep_solution else None,
        hazards=sparse.hazards if keep_solution else None,
    )


def exact_local_sse(scene: Scene) -> ExactLocalSSE:
    """Climb to a local SSE over the Defender grid at radius r."""
    topology = scene.defender_grid.topology()
    stage_totals = dict.fromkeys(STAGE_KEYS, 0.0)
    evaluated: list[dict[str, Any]] = []
    call_count = 0
    br_seconds = 0.0
    last_sizes: dict[str, Any] = {}

    def evaluator(action_id: int) -> LocalDefenderEvaluation:
        nonlocal call_count, br_seconds, last_sizes
        position = scene.defender_grid.position(action_id)
        started = perf_counter()
        response = exact_best_response(scene, position)
        br_seconds += perf_counter() - started
        call_count += 1
        for key in STAGE_KEYS:
            stage_totals[key] += response.timing.get(key, 0.0)
        last_sizes = response.sizes
        evaluated.append({
            "action_id": int(action_id),
            "x_map": position[0], "y_map": position[1],
            "feasible": response.feasible,
            "J_D": response.detection_probability,
            "J_A": response.attacker_objective,
        })
        if not response.feasible:
            return LocalDefenderEvaluation(
                action_id=action_id, status="model_infeasible",
                defender_value=None, attacker_objective=None,
                selected_attacker_response_id=None,
                exact_attacker_best_response_verified=True,
                strong_tie_break_verified=True,
                diagnostic="no admissible lattice switching state",
            )
        return LocalDefenderEvaluation(
            action_id=action_id, status="feasible",
            defender_value=float(response.detection_probability),
            attacker_objective=float(response.attacker_objective),
            selected_attacker_response_id=int(response.switching_state_id),
            exact_attacker_best_response_verified=True,
            strong_tie_break_verified=True, diagnostic=None,
        )

    started = perf_counter()
    search = run_local_sse_search(
        scene.seed_action_id, topology, evaluator,
        DefenderNeighborhoodConfig(r_neighbor=scene.condition.r_neighbor),
    )
    wall_s = perf_counter() - started

    final_id = search.final_local_sse_action_id
    timing = dict(stage_totals) | {
        "T_best_response_sum_s": br_seconds,
        "T_search_control_s": wall_s - br_seconds,
        "T_total_s": wall_s,
        "T_shared_graph_s": scene.build_timing.get("T_shared_graph_s", 0.0),
    }
    return ExactLocalSSE(
        condition_label=scene.condition.label,
        feasible=final_id is not None,
        selected_action_id=final_id,
        selected_sensor_map=(
            [float(v) for v in scene.defender_grid.position(final_id)]
            if final_id is not None else None
        ),
        attacker_objective=search.final_J_A,
        detection_probability=search.final_J_D,
        unique_evaluations=search.unique_defender_evaluations,
        cached_reuses=search.cached_evaluation_reuses,
        iterations=search.local_search_iterations,
        termination_status=search.termination_status,
        visited_action_ids=tuple(search.visited_defender_actions),
        evaluated=tuple(evaluated),
        timing=timing,
        sizes=scene.sizes() | last_sizes | {"evaluator_calls": call_count},
    )


__all__ = [
    "CONTOUR_SAMPLES", "ExactBestResponse", "ExactLocalSSE", "RADIAL_SAMPLES",
    "STAGE_KEYS", "SwitchOption", "exact_best_response", "exact_local_sse",
]

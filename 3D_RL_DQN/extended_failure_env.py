"""Failure-learning and powered-occlusion contracts for extended Phase 16.4."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Iterable

import numpy as np

import project_paths  # noqa: F401
from P1b_RL_approximation import (
    SwitchingCandidates,
    TrajectoryEvaluation,
    switching_candidates,
)
from attacker_br_problem import AttackerBRProblem, AttackerTransition
from dqn_action_catalog import HeadingActionCatalog


@dataclass(frozen=True)
class FailureCostConfig:
    """Terminal failure cost derived from the valid Bellman-reference envelope."""

    margin_absolute: float = 1.0
    margin_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.margin_absolute <= 0.0 or self.margin_fraction <= 0.0:
            raise ValueError("failure-cost margins must be positive")

    def from_reference_objectives(self, values: Iterable[float]) -> float:
        finite = np.asarray([float(value) for value in values], dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            raise ValueError("at least one finite Bellman reference is required")
        worst = float(np.max(finite))
        margin = max(self.margin_absolute, self.margin_fraction * max(abs(worst), 1.0))
        return float(worst + margin)


def failure_adjusted_reward(
    stage_cost: float,
    *,
    reached_goal: bool,
    episode_ended: bool,
    failure_terminal_cost: float,
) -> float:
    if not np.isfinite(stage_cost) or stage_cost < 0.0:
        raise ValueError("stage_cost must be finite and nonnegative")
    if not np.isfinite(failure_terminal_cost) or failure_terminal_cost <= 0.0:
        raise ValueError("failure_terminal_cost must be finite and positive")
    cost = float(stage_cost)
    if episode_ended and not reached_goal:
        cost += float(failure_terminal_cost)
    return -cost


def sampled_powered_segment_is_fully_occluded(
    terrain: Any,
    sensor_map: tuple[float, float, float],
    start_map: tuple[float, float, float],
    switching_map: np.ndarray,
    *,
    meters_per_map_unit: float,
    maximum_sample_spacing_m: float = 5.0,
) -> bool:
    """Conservatively test the whole powered leg at a versioned spatial tolerance.

    The launch point lies exactly on the ground/box base plane and the switching
    endpoint lies on the LOS tangent surface.  Both measure-zero boundary points
    are intentionally excluded.  Every interior sample must have an obstacle
    intersection on its segment to the sensor.  The 5 m spacing is a numerical
    certification tolerance, not a claim of symbolic continuous-geometry proof.
    """

    start = np.asarray(start_map, dtype=float)
    switching = np.asarray(switching_map, dtype=float)
    sensor = np.asarray(sensor_map, dtype=float)
    length_m = float(np.linalg.norm(switching - start) * meters_per_map_unit)
    count = max(2, int(ceil(length_m / float(maximum_sample_spacing_m))))
    fractions = (np.arange(count, dtype=float) + 0.5) / count
    points = start[None, :] + fractions[:, None] * (switching - start)[None, :]
    sensors = np.repeat(sensor[None, :], len(points), axis=0)
    accelerated = getattr(terrain, "segments_intersect_solid_many", None)
    if callable(accelerated):
        blocked = np.asarray(accelerated(points, sensors), dtype=bool)
    else:
        blocked = np.asarray([
            terrain.segment_intersects_solid(point, sensor)
            for point in points
        ], dtype=bool)
    return bool(np.all(blocked))


class ExpandedFailureAttackerBRProblem(AttackerBRProblem):
    """Expose every physically feasible glide edge, including failure branches."""

    def __init__(
        self,
        scene: Any,
        sensor_map: tuple[float, float, float],
        *,
        require_powered_occlusion: bool = True,
        powered_occlusion_spacing_m: float = 5.0,
    ) -> None:
        super().__init__(scene, sensor_map)
        self.require_powered_occlusion = bool(require_powered_occlusion)
        self.powered_occlusion_spacing_m = float(powered_occlusion_spacing_m)
        if self.powered_occlusion_spacing_m <= 0.0:
            raise ValueError("powered occlusion spacing must be positive")
        self._expanded_edges: dict[int, tuple[Any, ...]] = {}
        self._expanded_costs: dict[int, np.ndarray] = {}
        self._expanded_hazards: dict[int, np.ndarray] = {}

    def successors(self, state_id: int) -> tuple[Any, ...]:
        selected = int(state_id)
        cached = self._expanded_edges.get(selected)
        if cached is not None:
            return cached
        if self.is_terminal(selected):
            edges: tuple[Any, ...] = ()
        else:
            generated, _, _ = self.transition_model.successors(self.state(selected))
            edges = tuple(generated)
        self._expanded_edges[selected] = edges
        return edges

    def _cost_rows(self, state_id: int) -> tuple[np.ndarray, np.ndarray]:
        selected = int(state_id)
        costs = self._expanded_costs.get(selected)
        if costs is not None:
            return costs, self._expanded_hazards[selected]
        edges = self.successors(selected)
        hazards = np.asarray([
            self.mdp.edge_hazard(edge) for edge in edges
        ], dtype=float)
        durations = np.asarray([float(edge.duration_s) for edge in edges], dtype=float)
        objective = self.objective
        costs = (
            objective.hazard_weight * hazards / objective.hazard_reference
            + objective.time_weight * durations / objective.time_reference_s
        )
        self._expanded_costs[selected] = costs
        self._expanded_hazards[selected] = hazards
        return costs, hazards

    def transition(self, state_id: int, action_index: int) -> AttackerTransition:
        source = int(state_id)
        index = int(action_index)
        edges = self.successors(source)
        if not 0 <= index < len(edges):
            raise IndexError("expanded action index is outside the feasible actions")
        costs, hazards = self._cost_rows(source)
        edge = edges[index]
        return AttackerTransition(
            source_state_id=source,
            action_index=index,
            edge=edge,
            stage_cost=float(costs[index]),
            cumulative_hazard=float(hazards[index]),
            terminal=self.is_terminal(int(edge.target_id)),
        )

    def evaluate_trajectory(
        self, state_ids: tuple[int, ...], *, reached_goal: bool | None = None,
    ) -> TrajectoryEvaluation:
        ids = tuple(int(value) for value in state_ids)
        if not ids:
            raise ValueError("trajectory cannot be empty")
        duration = 0.0
        hazard = 0.0
        for source_id, target_id in zip(ids, ids[1:]):
            edges = self.successors(source_id)
            index = next((
                i for i, edge in enumerate(edges)
                if int(edge.target_id) == target_id
            ), None)
            if index is None:
                raise ValueError("trajectory contains a physically infeasible transition")
            _, hazards = self._cost_rows(source_id)
            duration += float(edges[index].duration_s)
            hazard += float(hazards[index])
        objective = self.objective
        cost = (
            objective.hazard_weight * hazard / objective.hazard_reference
            + objective.time_weight * duration / objective.time_reference_s
        )
        reached = self.is_terminal(ids[-1]) if reached_goal is None else bool(reached_goal)
        return TrajectoryEvaluation(
            state_ids=ids,
            duration_s=duration,
            cumulative_hazard=hazard,
            cost=float(cost),
            reached_goal=reached,
        )

    def _switching_candidates(self) -> SwitchingCandidates:
        if self._switching is not None:
            return self._switching
        candidates = switching_candidates(self.scene, self.sensor_map)
        if not self.require_powered_occlusion:
            self._switching = candidates
            return candidates
        usable: dict[int, Any] = {}
        powered_cost: dict[int, float] = {}
        scale = self.scene.config.physical_scale.meters_per_map_unit
        start = tuple(float(value) for value in self.scene.config.start.as_array())
        for state_id, value in candidates.by_state_id.items():
            position = np.asarray(value[0], dtype=float)
            if sampled_powered_segment_is_fully_occluded(
                self.scene.terrain,
                self.sensor_map,
                start,
                position,
                meters_per_map_unit=scale,
                maximum_sample_spacing_m=self.powered_occlusion_spacing_m,
            ):
                usable[int(state_id)] = value
                powered_cost[int(state_id)] = float(candidates.powered_cost_of[state_id])
        timing = dict(candidates.timing)
        timing["powered_occlusion_candidate_input"] = float(len(candidates.by_state_id))
        timing["powered_occlusion_candidate_retained"] = float(len(usable))
        self._switching = SwitchingCandidates(
            by_state_id=usable,
            powered_cost_of=powered_cost,
            surface_samples=candidates.surface_samples,
            snapped_states=candidates.snapped_states,
            timing=timing,
        )
        return self._switching

    def clear_runtime_caches(self) -> None:
        super().clear_runtime_caches()
        self._expanded_edges.clear()
        self._expanded_costs.clear()
        self._expanded_hazards.clear()


class ExpandedFailureHeadingActionCatalog(HeadingActionCatalog):
    """Disable the old goal-reachable batch filter for expanded action rows."""

    def _build_rows_batched(self, state_ids: tuple[int, ...]) -> tuple[Any, ...]:
        return tuple(self._build_row(state_id) for state_id in state_ids)


__all__ = [
    "ExpandedFailureAttackerBRProblem",
    "ExpandedFailureHeadingActionCatalog",
    "FailureCostConfig",
    "failure_adjusted_reward",
    "sampled_powered_segment_is_fully_occluded",
]

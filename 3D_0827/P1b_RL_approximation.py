"""Tabular Q-learning baseline for the Attacker's best response.

This module answers one question and no other: **can tabular Q-learning recover
the exact Bellman solution on the same MDP?**  It is a controlled comparison, not
a replacement - ``P1b_Exact_Local_SSE`` remains the oracle and is never bypassed.

The MDP is not redefined here.  Every ingredient is taken from the object the
Bellman solver already consumes, so the two solvers cannot silently drift apart:

    state        the discretised Attacker state, ``scene.grid`` lattice index
    action       an index into ``scene.graph.adjacency[state]`` - exactly the
                 physically feasible successors the sweep relaxes
    transition   deterministic, ``edge.target_id``
    stage cost   ``_phase_cost(hazard, duration, objective)``, imported from the
                 exact module rather than reimplemented
    terminal     ``scene.graph.terminal_mask``
    infeasible   never represented; the adjacency list holds feasible edges only,
                 so there is nothing to penalise

``test_P1b_equivalence.py`` asserts each of those correspondences.

Cost minimisation, not reward maximisation:

    Q(s,a) <- Q(s,a) + alpha * [ c(s,a) + gamma * min_a' Q(s',a') - Q(s,a) ]

with gamma = 1.  The glide graph is a finite acyclic DAG in which altitude falls
monotonically, so every episode terminates and there is nothing for a discount to
regularise.

Two experiments, both at one fixed sensor position:

    A  the switching state is pinned to the one Bellman chose, so the comparison
       isolates "can Q-learning solve this shortest-path problem".  This is the
       primary experiment: if A fails, B cannot be interpreted.
    B  the learner also chooses the switching state among the admissible ones,
       which is the full best response and additionally exercises the selection.

Usage:

    python P1b_RL_approximation.py [spatial_resolution_m] [episodes]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from P1b_Exact_Local_SSE import (
    _phase_cost,
    _sample_tangent_surface,
    _snap_to_lattice,
    exact_best_response,
)
from P1b_condition import ComputationCondition, Scene, build_scene
from bellman_geometry import GlideTransitionModel
from detection_hazard import GlideDetectionHazardModel, hazard_to_detection_probability
from edge_hazard import integrate_edge_hazard
from energy_model import StraightPoweredPhaseModel
from local_sse_contract import DefenderNeighborhoodConfig, LocalDefenderEvaluation
from local_sse_search import run_local_sse_search
from los_geometry import LOSModel
from sparse_additive import _batch_edge_hazard


SCHEMA_VERSION = "p1b-tabular-q-v1"

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Stage 1 - the MDP, taken from the Bellman problem rather than redefined
# ---------------------------------------------------------------------------


class AttackerMDP:
    """The glide MDP for one sensor position, as the Bellman solver sees it.

    Nothing here is new dynamics.  ``actions`` and ``is_terminal`` read the shared
    graph directly, and ``cost`` evaluates the same weighted hazard/time integral
    the sweep evaluates.  The only difference is *when*: the sweep computes every
    edge up front, this computes a state's edges the first time that state is
    touched, so an agent that visits a thousand states never pays for the other
    two million.

    Costs are batched per state rather than per edge.  Measured at 100 m, the
    per-edge call costs about 900 us against 0.34 us per edge for the sweep's
    batched form - almost all of it fixed numpy overhead that a single-edge call
    cannot amortise.  Filling a whole row at once keeps the laziness and pays that
    overhead once per state; the row is 20x cheaper per edge, and cheaper in
    absolute terms than one scalar edge.
    """

    def __init__(self, scene: Scene, sensor_map: tuple[float, float, float]) -> None:
        self.scene = scene
        self.config = scene.config
        self.grid = scene.grid
        self.graph = scene.graph
        self.mission = scene.mission_for(sensor_map)
        self.sensor_map = tuple(float(value) for value in sensor_map)
        self.objective = self.config.attacker_objective
        self.hazard_field = GlideDetectionHazardModel(
            scene.terrain, self.mission.sensor,
            parameters=self.config.detection,
            physical_scale=self.config.physical_scale,
        )
        self.quadrature = self.config.discretization.hazard_quadrature_resolution
        # ``scene.graph.adjacency`` is lazy: it recomputes a state's successors on
        # every subscript, measured at 535 us per state at 50 m and with no cache of
        # its own.  Training touches a state's action list at least twice per step,
        # so left uncached that lookup was about 84% of training time - the sweep
        # never pays it because it rebuilds transitions in vectorised slabs and
        # never subscripts the adjacency at all.  Caching per visited state turns it
        # into a dict hit, and costs only the states actually visited.
        self._actions: dict[int, Any] = {}
        # Keyed by state, not by edge: one batched integral fills a whole row.
        self._cost: dict[int, FloatArray] = {}
        # The same batch also yields the hazards, which the trajectory evaluator
        # needs on their own.  Keeping them costs one array per visited state and
        # saves that evaluator from re-integrating each edge one at a time.
        self._hazard: dict[int, FloatArray] = {}
        self.hazard_evaluations = 0

    # -- structure, read straight from the shared graph ---------------------

    def actions(self, state_id: int) -> Any:
        """The feasible successors of a state, i.e. the action set."""
        cached = self._actions.get(state_id)
        if cached is None:
            cached = self.graph.adjacency[state_id]
            self._actions[state_id] = cached
        return cached

    def action_count(self, state_id: int) -> int:
        return len(self.actions(state_id))

    def is_terminal(self, state_id: int) -> bool:
        return bool(self.graph.terminal_mask[state_id])

    @property
    def max_action_count(self) -> int:
        return len(self.grid.motion_offsets)

    # -- cost ---------------------------------------------------------------

    def cost_row(self, state_id: int) -> FloatArray:
        """Stage cost of every feasible action at one state, as one batch."""
        cached = self._cost.get(state_id)
        if cached is not None:
            return cached
        edges = self.actions(state_id)
        count = len(edges)
        if not count:
            row = np.empty(0, dtype=float)
            self._cost[state_id] = row
            self._hazard[state_id] = row
            return row
        origin = np.asarray(
            self.grid.position_map(self.grid.decode(state_id)), dtype=float,
        )
        ends = np.asarray(
            [self.grid.position_map(self.grid.decode(int(edge.target_id)))
             for edge in edges],
            dtype=float,
        )
        durations = np.asarray([float(edge.duration_s) for edge in edges], dtype=float)
        hazards, _ = _batch_edge_hazard(
            np.repeat(origin[None, :], count, axis=0), ends, durations,
            self.hazard_field, self.quadrature, self.config.physical_scale,
        )
        self.hazard_evaluations += count
        row = (
            self.objective.hazard_weight * hazards / self.objective.hazard_reference
            + self.objective.time_weight * durations / self.objective.time_reference_s
        )
        self._cost[state_id] = row
        self._hazard[state_id] = hazards
        return row

    def hazard_row(self, state_id: int) -> FloatArray:
        """Per-edge hazard for one state, from the same batch as the costs."""
        cached = self._hazard.get(state_id)
        if cached is None:
            self.cost_row(state_id)
            cached = self._hazard[state_id]
        return cached

    def cost(self, state_id: int, action_index: int) -> float:
        return float(self.cost_row(state_id)[action_index])

    def step(self, state_id: int, action_index: int) -> tuple[int, float, bool]:
        """Deterministic transition: next state, its stage cost, terminality."""
        edge = self.actions(state_id)[action_index]
        target_id = int(edge.target_id)
        return target_id, self.cost(state_id, action_index), self.is_terminal(target_id)

    def edge_hazard(self, edge: Any) -> float:
        """Single-edge integral, for the independent trajectory evaluator."""
        self.hazard_evaluations += 1
        return float(integrate_edge_hazard(
            edge, self.grid, self.hazard_field,
            quadrature_resolution=self.quadrature,
            physical_scale=self.config.physical_scale,
        ).hazard)

    @property
    def cached_edges(self) -> int:
        return int(sum(len(row) for row in self._cost.values()))

    @property
    def cached_states(self) -> int:
        return len(self._actions)

    def reachable_state_action_pairs(self) -> int:
        """Size of the state-action space the learner could in principle cover.

        Deliberately does not go through the action cache: this walks every
        goal-reachable state once to size the space, and caching all of them would
        hold the whole graph for a number that is only ever reported.
        """
        goal_reachable = np.asarray(self.scene.goal_reachable, dtype=bool)
        return int(sum(
            len(self.graph.adjacency[state_id])
            for state_id in np.flatnonzero(goal_reachable)
        ))


# ---------------------------------------------------------------------------
# Stage 2 - tabular Q-learning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QLearningConfig:
    """Frozen, versioned training settings so a run is reproducible.

    ``epsilon_decay`` multiplies epsilon once per episode, clamped below by
    ``epsilon_min``.  The defaults are a slow anneal from fully random exploration
    down to a 0.05 floor over roughly three thousand episodes, which is the usual
    shape for this schedule and the only one measured here that was reliable.

    Nine schedules over five seeds each, on the glide MDP from one switching state
    at 100 m, 2,000 episodes - runs that matched Bellman exactly, and the mean
    number of state-action pairs the run had tried:

        constant 0.01                   1/5     41 pairs
        constant 0.05                   1/5     62
        constant 0.10  (the textbook)   3/5     73
        constant 0.20                   4/5     90
        constant 0.30                   4/5     95
        1.0 -> 0.01, decay 0.995        3/5     88
        1.0 -> 0.05, decay 0.999        5/5     98      <- these defaults
        0.5 -> 0.05, decay 0.995        4/5     76
        0.2 -> 0.05, decay 0.5          2/5     64

    Success tracks coverage and nothing else, and coverage is what a fast decay
    throws away: episodes here are about four steps long, so a run that stops
    exploring early simply never sees most of its own action set.  That is also
    why the textbook 0.1 is not enough - it is tuned for problems with far longer
    episodes, which get many more chances to deviate per unit of epsilon.

    ``TrainingHistory.epsilon`` records the realised schedule, so what a run
    actually explored with is visible in the diagnostics rather than inferred from
    these three numbers.
    """

    alpha: float = 0.5
    gamma: float = 1.0
    epsilon: float = 1.0
    epsilon_decay: float = 0.999
    epsilon_min: float = 0.05
    episodes: int = 20000
    max_steps: int = 400
    seed: int = 0
    initial_q: float = 0.0
    # How often to stop and evaluate the greedy policy, in episodes.  Each
    # evaluation is a full rollout plus an independent cost evaluation, so it is
    # not free; it is what produces the convergence curves.
    evaluation_interval: int = 100

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be at least one")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must lie in (0, 1]")
        if not 0.0 <= self.epsilon_min <= self.epsilon:
            raise ValueError("epsilon_min must lie in [0, epsilon]")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least one")
        if self.evaluation_interval < 1:
            raise ValueError("evaluation_interval must be at least one")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "alpha": self.alpha, "gamma": self.gamma,
            "epsilon": self.epsilon, "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min, "episodes": self.episodes,
            "max_steps": self.max_steps, "seed": self.seed,
            "initial_q": self.initial_q,
            "evaluation_interval": self.evaluation_interval,
        }


@dataclass
class QTable:
    """Sparse tabular action-value store in cost form.

    Rows are allocated at the full stencil width and only the first
    ``action_count(s)`` entries of a row are ever used, which is how states with
    different numbers of feasible actions share one representation.

    ``tried`` records which entries have been updated.  It matters because an
    untried entry sits at the initial value, and with the conventional zero
    initialisation that is *below* every real cost - so it would win every argmin
    and the greedy policy would walk through actions it never tried.  Exploitation
    therefore ranks tried entries only; exploration is epsilon's job.
    """

    action_count: int
    initial_q: float = 0.0
    # Optional per-state initial value.  Off by default: the spec for this stage
    # is plain zero-initialised tabular Q-learning.  See admissible_initializer.
    initial_value_of: Callable[[int], float] | None = None
    values: dict[int, FloatArray] = field(default_factory=dict)
    visits: dict[int, int] = field(default_factory=dict)
    tried: dict[int, np.ndarray] = field(default_factory=dict)

    def row(self, state_id: int) -> FloatArray:
        row = self.values.get(state_id)
        if row is None:
            initial = (
                self.initial_q if self.initial_value_of is None
                else float(self.initial_value_of(state_id))
            )
            row = np.full(self.action_count, initial, dtype=float)
            self.values[state_id] = row
            self.tried[state_id] = np.zeros(self.action_count, dtype=bool)
        return row

    def mark_tried(self, state_id: int, action_index: int) -> None:
        self.tried[state_id][action_index] = True

    def greedy_action(self, state_id: int, action_count: int) -> int | None:
        """argmin over *tried* actions, or None if this state has none."""
        row = self.values.get(state_id)
        if row is None:
            return None
        mask = self.tried[state_id][:action_count]
        if not mask.any():
            return None
        return int(np.argmin(np.where(mask, row[:action_count], np.inf)))

    def best(self, state_id: int) -> float:
        """min_a Q(s,a) over tried actions; the initial value if none is tried."""
        row = self.values.get(state_id)
        if row is None:
            return (
                float(self.initial_q) if self.initial_value_of is None
                else float(self.initial_value_of(state_id))
            )
        mask = self.tried[state_id]
        if not mask.any():
            return float(row.min())
        return float(row[mask].min())

    @property
    def visited_states(self) -> int:
        return len(self.values)

    @property
    def tried_pairs(self) -> int:
        return int(sum(int(mask.sum()) for mask in self.tried.values()))


@dataclass
class TrainingHistory:
    """Per-episode and periodic-evaluation records behind the Stage-4 plots."""

    episode: list[int] = field(default_factory=list)
    episode_cost: list[float] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)
    episode_reached_goal: list[bool] = field(default_factory=list)
    epsilon: list[float] = field(default_factory=list)
    # Periodic greedy-policy evaluation, recorded every evaluation_interval.
    evaluation_episode: list[int] = field(default_factory=list)
    evaluation_cost: list[float] = field(default_factory=list)
    evaluation_reached_goal: list[bool] = field(default_factory=list)
    evaluation_gap: list[float] = field(default_factory=list)
    evaluation_relative_gap: list[float] = field(default_factory=list)
    evaluation_tried_pairs: list[int] = field(default_factory=list)

    def moving_average_cost(self, window: int = 100) -> FloatArray:
        """Moving average of the episode cost, for the Stage-4 smoothing plot."""
        costs = np.asarray(self.episode_cost, dtype=float)
        if not len(costs):
            return costs
        window = max(1, min(int(window), len(costs)))
        kernel = np.ones(window, dtype=float) / window
        return np.convolve(costs, kernel, mode="valid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": list(self.episode),
            "episode_cost": list(self.episode_cost),
            "episode_steps": list(self.episode_steps),
            "episode_reached_goal": list(self.episode_reached_goal),
            "epsilon": list(self.epsilon),
            "evaluation_episode": list(self.evaluation_episode),
            "evaluation_cost": list(self.evaluation_cost),
            "evaluation_reached_goal": list(self.evaluation_reached_goal),
            "evaluation_gap": list(self.evaluation_gap),
            "evaluation_relative_gap": list(self.evaluation_relative_gap),
            "evaluation_tried_pairs": list(self.evaluation_tried_pairs),
        }


@dataclass(frozen=True)
class TrajectoryEvaluation:
    """A trajectory scored by the original Attacker cost evaluator."""

    state_ids: tuple[int, ...]
    duration_s: float
    cumulative_hazard: float
    cost: float
    reached_goal: bool


def greedy_trajectory(
    mdp: AttackerMDP, table: QTable, start_state_id: int, *, max_steps: int,
) -> tuple[tuple[int, ...], bool]:
    """Follow pi_Q(s) = argmin_a Q(s,a) from a start state."""
    current = int(start_state_id)
    visited = [current]
    for _ in range(max_steps):
        if mdp.is_terminal(current):
            return tuple(visited), True
        count = mdp.action_count(current)
        if not count:
            return tuple(visited), False
        action_index = table.greedy_action(current, count)
        if action_index is None:
            return tuple(visited), False
        current = int(mdp.actions(current)[action_index].target_id)
        visited.append(current)
    return tuple(visited), mdp.is_terminal(current)


def evaluate_trajectory(
    mdp: AttackerMDP, state_ids: Iterable[int], *, reached_goal: bool | None = None,
) -> TrajectoryEvaluation:
    """Score a trajectory with the original evaluator, independent of the table.

    This never consults Q.  It walks the shared graph, sums the same per-edge
    hazard and duration the exact solver sums, and applies the same objective, so
    a learned trajectory and a Bellman trajectory are scored by one function.

    Hazards come from the state's cached batch rather than from a fresh per-edge
    integral.  They are the same numbers - the batch is where both come from - but
    the scalar call costs about 900 us against 44 us per edge batched, and ranking
    switching candidates by rollout evaluates one trajectory per candidate, so the
    difference decides whether a Defender position takes seconds or minutes.
    """
    ids = tuple(int(state_id) for state_id in state_ids)
    duration = 0.0
    hazard = 0.0
    for source_id, target_id in zip(ids, ids[1:]):
        edges = mdp.actions(source_id)
        index = next(
            (
                position for position, edge in enumerate(edges)
                if int(edge.target_id) == target_id
            ),
            None,
        )
        if index is None:
            raise ValueError(
                f"state {target_id} is not a feasible successor of {source_id}; "
                "the trajectory leaves the shared graph"
            )
        duration += float(edges[index].duration_s)
        hazard += float(mdp.hazard_row(source_id)[index])
    reached = mdp.is_terminal(ids[-1]) if reached_goal is None else bool(reached_goal)
    return TrajectoryEvaluation(
        state_ids=ids, duration_s=duration, cumulative_hazard=hazard,
        cost=_phase_cost(hazard, duration, mdp.objective), reached_goal=reached,
    )


def train(
    mdp: AttackerMDP,
    start_state_ids: tuple[int, ...],
    table: QTable,
    config: QLearningConfig,
    *,
    episodes: int | None = None,
    oracle_cost: float | None = None,
    history: TrainingHistory | None = None,
    evaluation_start_id: int | None = None,
    evaluate: Callable[[], tuple[float | None, bool]] | None = None,
) -> TrainingHistory:
    """Tabular Q-learning: Q <- Q + alpha [ c + gamma min Q' - Q ].

    Learning is from environment interaction only; the exact solution is never
    used as a label.  ``oracle_cost`` enters nowhere in the update - it is carried
    solely so the periodic evaluation can record the gap.

    ``evaluate`` decides what the periodic measurement means.  The default scores
    the greedy rollout from ``evaluation_start_id``, which is what experiment A
    compares.  Experiment B passes its own, because there the learner also picks
    the start, and a curve measured from a start the learner did not pick would
    not converge to the number the experiment reports.
    """
    record = TrainingHistory() if history is None else history
    rng = np.random.default_rng(config.seed)
    starts = np.asarray(start_state_ids, dtype=np.int64)
    if not len(starts):
        raise ValueError("training needs at least one start state")
    evaluation_id = (
        int(starts[0]) if evaluation_start_id is None else int(evaluation_start_id)
    )
    budget = config.episodes if episodes is None else int(episodes)
    epsilon = config.epsilon
    offset = len(record.episode)

    for index in range(budget):
        state_id = int(starts[rng.integers(len(starts))])
        episode_cost = 0.0
        steps = 0
        reached = False
        for _ in range(config.max_steps):
            if mdp.is_terminal(state_id):
                reached = True
                break
            count = mdp.action_count(state_id)
            if not count:
                break
            row = table.row(state_id)
            if rng.random() < epsilon:
                action_index = int(rng.integers(count))
            else:
                action_index = table.greedy_action(state_id, count)
                if action_index is None:
                    action_index = int(rng.integers(count))
            target_id, step_cost, target_terminal = mdp.step(state_id, action_index)
            bootstrap = 0.0 if target_terminal else table.best(target_id)
            row[action_index] += config.alpha * (
                step_cost + config.gamma * bootstrap - row[action_index]
            )
            table.mark_tried(state_id, action_index)
            table.visits[state_id] = table.visits.get(state_id, 0) + 1
            episode_cost += step_cost
            steps += 1
            state_id = target_id

        record.episode.append(offset + index)
        record.episode_cost.append(episode_cost)
        record.episode_steps.append(steps)
        record.episode_reached_goal.append(reached)
        record.epsilon.append(epsilon)
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        if (index + 1) % config.evaluation_interval == 0 or index + 1 == budget:
            if evaluate is None:
                ids, greedy_reached = greedy_trajectory(
                    mdp, table, evaluation_id, max_steps=config.max_steps,
                )
                scored = evaluate_trajectory(mdp, ids, reached_goal=greedy_reached)
                measured = scored.cost if greedy_reached else None
            else:
                measured, greedy_reached = evaluate()
            cost = float("inf") if measured is None else float(measured)
            record.evaluation_episode.append(offset + index + 1)
            record.evaluation_cost.append(cost)
            record.evaluation_reached_goal.append(greedy_reached)
            record.evaluation_tried_pairs.append(table.tried_pairs)
            if oracle_cost is None or not np.isfinite(cost):
                record.evaluation_gap.append(float("nan"))
                record.evaluation_relative_gap.append(float("nan"))
            else:
                record.evaluation_gap.append(cost - oracle_cost)
                record.evaluation_relative_gap.append(
                    (cost - oracle_cost) / oracle_cost if oracle_cost else float("nan")
                )
    return record


# ---------------------------------------------------------------------------
# Optional, off by default: an admissible lower bound to initialise with
# ---------------------------------------------------------------------------


class _AdmissibleValue:
    """Lower bound on the cost-to-go, for the initialisation ablation.

    Not part of the Stage-2 specification, which calls for plain zero
    initialisation; kept because the bound is verified and the ablation is cheap.

    The objective is ``w_H H / H_ref + w_T T / T_ref`` with ``H >= 0``, so a lower
    bound on the time alone bounds the whole cost.  Two floors both hold, so their
    maximum is taken: the glider must shed its remaining altitude, and no motion
    primitive sheds a bin faster than ``min_h(duration_h / altitude_loss_h)``; and
    it must cover the horizontal distance to the goal at best glide speed.

    Terminal states are pinned to zero.  A terminal state can sit an altitude bin
    above the goal plane and a quarter of a cell away from it, so both floors read
    positive there while the true cost-to-go is exactly zero - measured at 25 m,
    189 such states, and they are what breaks admissibility if the terminal test
    is omitted.  With it, the bound was verified against the exact value function
    on 4,289 states at 100 m and 1,254,515 at 25 m with no violation.
    """

    def __init__(self, scene: Scene) -> None:
        grid = scene.grid
        config = scene.config
        self.grid = grid
        self.objective = config.attacker_objective
        self.terminal_mask = np.asarray(scene.graph.terminal_mask, dtype=bool)
        transition = GlideTransitionModel(
            grid, scene.terrain, config.glider, config.physical_scale,
        )
        seconds_per_bin: list[float] = []
        for x_offset, y_offset in grid.motion_offsets:
            distance_map = grid.horizontal_spacing_map * float(
                np.hypot(x_offset, y_offset)
            )
            loss = transition.altitude_loss_bins(distance_map)
            if loss <= 0:
                continue
            seconds_per_bin.append(
                config.physical_scale.distance_m(distance_map)
                / config.glider.best_glide_speed_mps / float(loss)
            )
        self.seconds_per_altitude_bin = min(seconds_per_bin) if seconds_per_bin else 0.0
        self.speed_mps = float(config.glider.best_glide_speed_mps)
        self.goal = (float(config.goal.x), float(config.goal.y))
        self.meters_per_map_unit = float(config.physical_scale.meters_per_map_unit)
        self._cache: dict[int, float] = {}

    def value(self, state_id: int) -> float:
        cached = self._cache.get(state_id)
        if cached is not None:
            return cached
        if self.terminal_mask[state_id]:
            self._cache[state_id] = 0.0
            return 0.0
        state = self.grid.decode(state_id)
        position = self.grid.position_map(state)
        horizontal_m = float(np.hypot(
            position[0] - self.goal[0], position[1] - self.goal[1],
        )) * self.meters_per_map_unit
        seconds = max(
            state.altitude_index * self.seconds_per_altitude_bin,
            horizontal_m / self.speed_mps,
        )
        value = self.objective.time_weight * seconds / self.objective.time_reference_s
        self._cache[state_id] = value
        return value


def admissible_initializer(scene: Scene) -> Callable[[int], float]:
    """Per-state lower bound to seed a table with; see ``_AdmissibleValue``."""
    return _AdmissibleValue(scene).value


# ---------------------------------------------------------------------------
# Switching candidates - stages 1 and 2 of the exact pipeline, unchanged
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwitchingCandidates:
    """Admissible switching states and the powered leg that reaches each."""

    by_state_id: dict[int, Any]
    powered_cost_of: dict[int, float]
    surface_samples: int
    snapped_states: int
    timing: dict[str, float]

    @property
    def state_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.by_state_id))


def switching_candidates(
    scene: Scene, sensor_map: tuple[float, float, float],
) -> SwitchingCandidates:
    """Stages 1 and 2 exactly as the exact pipeline runs them."""
    config = scene.config
    discretization = config.discretization
    mission = scene.mission_for(sensor_map)
    timing: dict[str, float] = {}

    started = perf_counter()
    contour = LOSModel(scene.terrain).trace_tangent_contour(
        mission.sensor,
        probe_grid_size=discretization.los_probe_grid_size,
        boundary_refinement_steps=discretization.los_boundary_refinement_steps,
    )
    points = _sample_tangent_surface(
        contour, discretization.switching_radial_min,
        discretization.switching_radial_max,
    )
    timing["T_stage1_los_s"] = perf_counter() - started

    started = perf_counter()
    powered_model = StraightPoweredPhaseModel(
        parameters=config.glider, physical_scale=config.physical_scale,
    )
    headings = np.asarray(
        [powered_model.state_at(point, mission, scene.terrain).heading_rad
         for point in points],
        dtype=float,
    )
    snapped = _snap_to_lattice(points, headings, scene.grid)
    usable: dict[int, Any] = {}
    powered_cost: dict[int, float] = {}
    for state_id, position in snapped.items():
        if not scene.graph.node_mask[state_id] or not scene.goal_reachable[state_id]:
            continue
        switching_state = powered_model.state_at(position, mission, scene.terrain)
        if not switching_state.powered_feasible:
            continue
        usable[state_id] = (position, switching_state)
        powered_cost[state_id] = _phase_cost(
            0.0,
            switching_state.powered_path_length_m / config.glider.powered_speed_mps,
            config.attacker_objective,
        )
    timing["T_stage2_filter_s"] = perf_counter() - started

    return SwitchingCandidates(
        by_state_id=usable, powered_cost_of=powered_cost,
        surface_samples=len(points), snapped_states=len(snapped), timing=timing,
    )


# ---------------------------------------------------------------------------
# Stage 3 - the fixed-instance comparison
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 3 - the fixed-instance comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwitchingReadout:
    """Which switching state falls out of the solved glide MDP, and how.

    Choosing the switching state is not a second learning problem - it is one
    argmin - but *what* the argmin ranks decides whether it can be trusted.

    Ranking by the learned value ``powered(s) + min_a Q(s,a)`` is the tempting
    form and it is wrong.  Those values are optimistic wherever coverage is thin,
    by different amounts at different candidates, and an argmin over optimistic
    estimates returns the most optimistic one rather than the best one.  Measured
    at 100 m from the seeded sensor, the median candidate error was 0.0012 and the
    worst 0.2151, and the argmin picked the worst.

    Ranking by *rollout* is sound.  Fly the greedy policy from each candidate and
    score the realised trajectory with the original evaluator: every number is
    then a cost the Attacker can actually pay, so none can sit below the true
    optimum and the selection cannot be fooled by an optimistic estimate.  A thin
    table costs an honestly suboptimal trajectory, never a fictitious one.  This
    is the standard separation of the estimator that selects from the estimator
    that scores, and here the scoring estimator is not merely unbiased but exact.
    It costs one rollout per candidate - of order a thousand edges over the whole
    candidate set, against the sweep's half million.

    Both are reported so the difference stays visible; ``state_id`` is the sound
    one.
    """

    bellman_state_id: int
    state_id: int
    objective: float | None
    # The rollout that produced ``objective``.  Retained because it, not the path
    # from Bellman's switching state, is the solution the learner actually offers.
    trajectory: tuple[int, ...]
    value_ranked_state_id: int
    value_ranked_objective: float | None
    candidates: int
    candidates_visited: int
    candidates_reaching_goal: int
    worst_candidate_error: float
    median_candidate_error: float

    @property
    def matches_bellman(self) -> bool:
        return self.bellman_state_id == self.state_id

    @property
    def value_ranking_matches_bellman(self) -> bool:
        return self.bellman_state_id == self.value_ranked_state_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "bellman_state_id": self.bellman_state_id,
            "state_id": self.state_id,
            "objective": self.objective,
            "trajectory": list(self.trajectory),
            "matches_bellman": self.matches_bellman,
            "value_ranked_state_id": self.value_ranked_state_id,
            "value_ranked_objective": self.value_ranked_objective,
            "value_ranking_matches_bellman": self.value_ranking_matches_bellman,
            "candidates": self.candidates,
            "candidates_visited": self.candidates_visited,
            "candidates_reaching_goal": self.candidates_reaching_goal,
            "worst_candidate_error": self.worst_candidate_error,
            "median_candidate_error": self.median_candidate_error,
        }


@dataclass(frozen=True)
class GlideSolveResult:
    """One controlled Bellman-versus-Q-learning comparison on the glide MDP."""

    condition_label: str
    sensor_map: tuple[float, float, float]
    evaluation_state_id: int
    start_state_ids: tuple[int, ...]
    bellman_cost: float
    q_cost: float | None
    absolute_gap: float | None
    relative_gap: float | None
    q_reached_goal: bool
    bellman_trajectory: tuple[int, ...]
    q_trajectory: tuple[int, ...]
    bellman_seconds: float
    training_seconds: float
    inference_seconds: float
    visited_states: int
    tried_state_action_pairs: int
    reachable_state_action_pairs: int
    history: TrainingHistory
    config: dict[str, Any]
    switching: SwitchingReadout | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage_fraction(self) -> float:
        if not self.reachable_state_action_pairs:
            return float("nan")
        return self.tried_state_action_pairs / self.reachable_state_action_pairs

    def summary(self) -> str:
        gap = "unreached" if self.q_cost is None else f"{self.relative_gap:+.4%}"
        cost = "did not reach the goal" if self.q_cost is None else f"{self.q_cost:.6f}"
        lines = [
            f"{self.condition_label}  d={list(self.sensor_map)}  "
            f"{len(self.start_state_ids)} start state(s)",
            f"  glide MDP   J_Bellman = {self.bellman_cost:.6f}   J_Q = {cost}"
            f"   gap {gap}",
            f"  time        Bellman {self.bellman_seconds:.2f}s   "
            f"training {self.training_seconds:.2f}s   "
            f"greedy inference {self.inference_seconds * 1e3:.1f}ms",
            f"  coverage    {self.tried_state_action_pairs:,} / "
            f"{self.reachable_state_action_pairs:,} state-action pairs "
            f"= {self.coverage_fraction:.4%}   ({self.visited_states:,} states)",
        ]
        if self.switching is not None:
            readout = self.switching

            def shown(objective: float | None) -> str:
                return "unreached" if objective is None else f"{objective:.6f}"

            lines.append(
                f"  switching   by rollout  state {readout.state_id} "
                f"{'==' if readout.matches_bellman else '!='} Bellman "
                f"{readout.bellman_state_id}   J_A = {shown(readout.objective)}"
            )
            lines.append(
                f"              by value    state {readout.value_ranked_state_id} "
                f"{'==' if readout.value_ranking_matches_bellman else '!='} Bellman"
                f"   J_A = {shown(readout.value_ranked_objective)}"
            )
            lines.append(
                f"              candidate value error  median "
                f"{readout.median_candidate_error:.6f}   worst "
                f"{readout.worst_candidate_error:.6f}"
                f"   ({readout.candidates_reaching_goal}/{readout.candidates} "
                "reach the goal)"
            )
        return "\n".join(lines)


def _oracle(scene: Scene, sensor_map: tuple[float, float, float]):
    started = perf_counter()
    response = exact_best_response(scene, sensor_map, keep_solution=True)
    return response, perf_counter() - started


def _switching_readout(
    mdp: AttackerMDP,
    table: QTable,
    candidates: SwitchingCandidates,
    value: FloatArray,
    bellman_state_id: int,
    *,
    max_steps: int,
) -> SwitchingReadout:
    """Pick the switching state by rollout, and report the value ranking beside it."""
    state_ids = candidates.state_ids
    realised: dict[int, float] = {}
    paths: dict[int, tuple[int, ...]] = {}
    for state_id in state_ids:
        ids, reached = greedy_trajectory(mdp, table, state_id, max_steps=max_steps)
        if not reached:
            continue
        realised[state_id] = (
            candidates.powered_cost_of[state_id]
            + evaluate_trajectory(mdp, ids, reached_goal=True).cost
        )
        paths[state_id] = ids
    chosen = min(realised, key=lambda k: (realised[k], k)) if realised else None
    ranked = min(
        state_ids,
        key=lambda state_id: (
            candidates.powered_cost_of[state_id] + table.best(state_id), state_id,
        ),
    )
    errors = np.asarray(
        [abs(table.best(state_id) - float(value[state_id])) for state_id in state_ids],
        dtype=float,
    )
    return SwitchingReadout(
        bellman_state_id=int(bellman_state_id),
        state_id=int(chosen) if chosen is not None else int(ranked),
        objective=realised.get(chosen) if chosen is not None else None,
        trajectory=paths.get(chosen, ()) if chosen is not None else (),
        value_ranked_state_id=int(ranked),
        value_ranked_objective=realised.get(ranked),
        candidates=len(state_ids),
        candidates_visited=sum(
            1 for state_id in state_ids if table.visits.get(state_id, 0)
        ),
        candidates_reaching_goal=len(realised),
        worst_candidate_error=float(errors.max()) if len(errors) else float("nan"),
        median_candidate_error=float(np.median(errors)) if len(errors) else float("nan"),
    )


def solve_glide_mdp(
    scene: Scene,
    sensor_map: tuple[float, float, float],
    config: QLearningConfig,
    *,
    start_state_ids: tuple[int, ...] | None = None,
    evaluation_state_id: int | None = None,
    table: QTable | None = None,
    read_switching: bool = False,
) -> GlideSolveResult:
    """Solve the glide MDP with tabular Q-learning and compare against Bellman.

    The glide MDP is the whole training target.  The switching state is not
    trained for: it is ``argmin_s [ powered(s) + V(s) ]``, which is a readout of
    the same value function and is correct exactly when that value is correct at
    the candidates.  ``read_switching`` reports it, along with the candidate value
    error that says how far it can be trusted.

    ``start_state_ids`` decides which part of the glide MDP is actually exercised.
    The default is the single state Bellman switched at, which is the cleanest
    test of the solver: one start, one sub-DAG, one number.  Passing the whole
    candidate set instead asks for value accuracy across all of them, which is a
    strictly larger problem - the budget question moves there, it does not vanish.

    ``evaluation_state_id`` is where the gap is measured, and it defaults to that
    same Bellman switching state.  Override it to measure the solver from some
    other switching state: the oracle value there is still the exact optimum *for
    that start*, so the gap stays a statement about the solver and not about which
    switching state is the better one to leave from.
    """
    oracle, bellman_seconds = _oracle(scene, sensor_map)
    if not oracle.feasible or oracle.switching_state_id is None:
        raise ValueError("the exact solver found no feasible best response here")
    value = np.asarray(oracle.solution.value, dtype=float)
    evaluation_id = (
        int(oracle.switching_state_id) if evaluation_state_id is None
        else int(evaluation_state_id)
    )
    bellman_cost = float(value[evaluation_id])
    if not np.isfinite(bellman_cost):
        raise ValueError(
            f"state {evaluation_id} has no finite exact value, so there is no "
            "optimum to compare against"
        )

    candidates = (
        switching_candidates(scene, sensor_map)
        if read_switching or start_state_ids is None else None
    )
    if start_state_ids is None:
        starts = (evaluation_id,)
    else:
        starts = tuple(int(state_id) for state_id in start_state_ids)
    if not starts:
        raise ValueError("at least one start state is required")

    mdp = AttackerMDP(scene, sensor_map)
    table = table or QTable(
        action_count=mdp.max_action_count, initial_q=config.initial_q,
    )
    started = perf_counter()
    history = train(
        mdp, starts, table, config,
        oracle_cost=bellman_cost, evaluation_start_id=evaluation_id,
    )
    training_seconds = perf_counter() - started

    started = perf_counter()
    ids, reached = greedy_trajectory(
        mdp, table, evaluation_id, max_steps=config.max_steps,
    )
    inference_seconds = perf_counter() - started
    evaluation = evaluate_trajectory(mdp, ids, reached_goal=reached)
    q_cost = evaluation.cost if reached else None

    readout = None
    if read_switching and candidates is not None and candidates.by_state_id:
        readout = _switching_readout(
            mdp, table, candidates, value, evaluation_id,
            max_steps=config.max_steps,
        )

    return GlideSolveResult(
        condition_label=scene.condition.label,
        sensor_map=tuple(float(item) for item in sensor_map),
        evaluation_state_id=evaluation_id,
        start_state_ids=starts,
        bellman_cost=bellman_cost,
        q_cost=q_cost,
        absolute_gap=None if q_cost is None else q_cost - bellman_cost,
        relative_gap=(
            None if q_cost is None or not bellman_cost
            else (q_cost - bellman_cost) / bellman_cost
        ),
        q_reached_goal=reached,
        bellman_trajectory=tuple(int(item) for item in oracle.glide_state_ids),
        q_trajectory=ids,
        bellman_seconds=bellman_seconds,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        visited_states=table.visited_states,
        tried_state_action_pairs=table.tried_pairs,
        reachable_state_action_pairs=mdp.reachable_state_action_pairs(),
        history=history,
        config=config.as_dict(),
        switching=readout,
        extra={
            "oracle_objective": oracle.attacker_objective,
            "oracle_detection_probability": oracle.detection_probability,
            "glide_duration_s": evaluation.duration_s,
            "glide_hazard": evaluation.cumulative_hazard,
            "detection_probability": hazard_to_detection_probability(
                evaluation.cumulative_hazard,
            ),
            "hazard_evaluations": mdp.hazard_evaluations,
        },
    )


def candidate_start_states(
    scene: Scene, sensor_map: tuple[float, float, float],
) -> tuple[int, ...]:
    """Every admissible switching state, for solving the glide MDP over all of them."""
    return switching_candidates(scene, sensor_map).state_ids


# ---------------------------------------------------------------------------
# The Defender search, driven by the learned Attacker response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnedBestResponse:
    """One Attacker best response at one Defender position, learned not solved."""

    feasible: bool
    attacker_objective: float | None
    detection_probability: float | None
    switching_state_id: int | None
    trajectory: tuple[int, ...]
    training_seconds: float
    episodes: int


def learned_best_response(
    scene: Scene,
    sensor_map: tuple[float, float, float],
    table: QTable,
    config: QLearningConfig,
    *,
    episodes: int | None = None,
) -> LearnedBestResponse:
    """Train on the glide MDP at this Defender position, then read the answer off.

    The switching state is ranked by rollout, never by the learned value: a
    realised trajectory is scored by the original evaluator, so every candidate's
    number is a cost the Attacker can actually pay.  Ranking by ``min_a Q`` instead
    would return whichever candidate the table is most optimistic about, which at
    thin coverage is the one it is most wrong about.
    """
    candidates = switching_candidates(scene, sensor_map)
    if not candidates.by_state_id:
        return LearnedBestResponse(
            False, None, None, None, (), 0.0, 0,
        )
    mdp = AttackerMDP(scene, sensor_map)
    budget = config.episodes if episodes is None else int(episodes)
    started = perf_counter()
    train(mdp, candidates.state_ids, table, config, episodes=budget)
    training_seconds = perf_counter() - started

    best_id: int | None = None
    best_objective = float("inf")
    best_path: tuple[int, ...] = ()
    best_hazard = 0.0
    for state_id in candidates.state_ids:
        ids, reached = greedy_trajectory(
            mdp, table, state_id, max_steps=config.max_steps,
        )
        if not reached:
            continue
        scored = evaluate_trajectory(mdp, ids, reached_goal=True)
        objective = candidates.powered_cost_of[state_id] + scored.cost
        if objective < best_objective:
            best_id, best_objective = state_id, objective
            best_path, best_hazard = ids, scored.cumulative_hazard
    if best_id is None:
        return LearnedBestResponse(
            False, None, None, None, (), training_seconds, budget,
        )
    return LearnedBestResponse(
        feasible=True,
        attacker_objective=best_objective,
        detection_probability=hazard_to_detection_probability(best_hazard),
        switching_state_id=int(best_id),
        trajectory=best_path,
        training_seconds=training_seconds,
        episodes=budget,
    )


@dataclass(frozen=True)
class LearnedLocalSSE:
    """The Defender's local search, with the learned Attacker response inside."""

    condition_label: str
    feasible: bool
    selected_action_id: int | None
    selected_sensor_map: list[float] | None
    attacker_objective: float | None
    detection_probability: float | None
    switching_state_id: int | None
    trajectory: tuple[int, ...]
    unique_evaluations: int
    iterations: int
    termination_status: str
    # Always False.  The learner cannot certify, and the search is told not to ask;
    # keeping the flag explicit is what stops an approximate result from being read
    # as an exact one further downstream.
    local_sse_verified: bool
    visited_action_ids: tuple[int, ...]
    evaluated: tuple[dict[str, Any], ...]
    timing: dict[str, float]
    training: dict[str, Any]
    sizes: dict[str, Any]


def rl_local_sse(
    scene: Scene,
    config: QLearningConfig,
    *,
    warm_start_episodes: int | None = None,
    progress: bool = False,
) -> LearnedLocalSSE:
    """The same Defender climb the exact solver runs, with Q-learning inside it.

    One table serves every Defender position.  The lattice, the edges and their
    durations do not depend on the sensor - only the hazard does - so a table fit
    at one position starts the next one part-right rather than from nothing, and
    later positions are given ``warm_start_episodes`` instead of a full budget.
    The table is a warm start, not an answer: the hazard term is a large part of
    the cost, so its values are stale until retrained.
    """
    topology = scene.defender_grid.topology()
    table = QTable(action_count=len(scene.grid.motion_offsets),
                   initial_q=config.initial_q)
    follow_up = (
        max(1, config.episodes // 5) if warm_start_episodes is None
        else int(warm_start_episodes)
    )
    responses: dict[int, LearnedBestResponse] = {}
    evaluated: list[dict[str, Any]] = []
    training_seconds = 0.0
    episodes_spent = 0
    call_count = 0

    def evaluator(action_id: int) -> LocalDefenderEvaluation:
        nonlocal training_seconds, episodes_spent, call_count
        position = tuple(
            float(v) for v in scene.defender_grid.position(action_id)
        )
        episodes = config.episodes if call_count == 0 else follow_up
        started_position = perf_counter()
        response = learned_best_response(
            scene, position, table, config, episodes=episodes,
        )
        call_count += 1
        if progress:
            objective = (
                "infeasible" if response.attacker_objective is None
                else f"J_A {response.attacker_objective:.6f}"
            )
            print(
                f"    [{call_count:>3}] d {list(position)}  {episodes:,} episodes"
                f"  {objective}  {perf_counter() - started_position:.1f}s",
                flush=True,
            )
        training_seconds += response.training_seconds
        episodes_spent += response.episodes
        responses[int(action_id)] = response
        evaluated.append({
            "action_id": int(action_id),
            "x_map": position[0], "y_map": position[1],
            "feasible": response.feasible,
            "J_A": response.attacker_objective,
            "J_D": response.detection_probability,
        })
        if not response.feasible:
            return LocalDefenderEvaluation(
                action_id=action_id, status="model_infeasible",
                defender_value=None, attacker_objective=None,
                selected_attacker_response_id=None,
                exact_attacker_best_response_verified=False,
                strong_tie_break_verified=False,
                diagnostic="no learned rollout reached the goal",
            )
        return LocalDefenderEvaluation(
            action_id=action_id, status="feasible",
            defender_value=float(response.detection_probability),
            attacker_objective=float(response.attacker_objective),
            selected_attacker_response_id=int(response.switching_state_id),
            # Reported honestly: this response is learned, so it is neither exact
            # nor tie-broken, and the search is configured not to require either.
            exact_attacker_best_response_verified=False,
            strong_tie_break_verified=False, diagnostic=None,
        )

    started = perf_counter()
    search = run_local_sse_search(
        scene.seed_action_id, topology, evaluator,
        DefenderNeighborhoodConfig(r_neighbor=scene.condition.r_neighbor),
        require_exact_attacker_verification=False,
    )
    wall_seconds = perf_counter() - started

    final_id = search.final_local_sse_action_id
    final = responses.get(int(final_id)) if final_id is not None else None
    return LearnedLocalSSE(
        condition_label=scene.condition.label,
        feasible=final is not None,
        selected_action_id=final_id,
        selected_sensor_map=(
            [float(v) for v in scene.defender_grid.position(final_id)]
            if final_id is not None else None
        ),
        attacker_objective=None if final is None else final.attacker_objective,
        detection_probability=None if final is None else final.detection_probability,
        switching_state_id=None if final is None else final.switching_state_id,
        trajectory=() if final is None else final.trajectory,
        unique_evaluations=search.unique_defender_evaluations,
        iterations=search.local_search_iterations,
        termination_status=search.termination_status,
        local_sse_verified=search.local_sse_verified,
        visited_action_ids=search.visited_defender_actions,
        evaluated=tuple(evaluated),
        timing={
            "T_total_s": wall_seconds,
            "T_train_s": training_seconds,
            "T_search_s": wall_seconds - training_seconds,
        },
        training={
            "config": config.as_dict(),
            "episodes": episodes_spent,
            "first_position_episodes": config.episodes,
            "warm_start_episodes": follow_up,
            "visited_states": table.visited_states,
            "tried_pairs": table.tried_pairs,
        },
        sizes=scene.sizes() | {"defender_evaluations": call_count},
    )


def default_sensor(scene: Scene) -> tuple[float, float, float]:
    """The seeded Defender position, so the instance is fixed and reproducible."""
    return tuple(
        float(value) for value in scene.defender_grid.position(scene.seed_action_id)
    )


def main(argv: list[str]) -> None:
    resolution = float(argv[1]) if len(argv) > 1 else 100.0
    episodes = int(argv[2]) if len(argv) > 2 else 20000
    scene = build_scene(ComputationCondition(spatial_resolution_m=resolution))
    sensor = default_sensor(scene)
    config = QLearningConfig(episodes=episodes)
    print(f"condition {scene.condition.label}   sensor d = {list(sensor)}")
    print(f"config {config.as_dict()}\n")
    for runner in (run_experiment_a, run_experiment_b):
        print(runner(scene, sensor, config).summary(), "\n")


__all__ = [
    "SCHEMA_VERSION", "AttackerMDP", "GlideSolveResult", "QLearningConfig",
    "QTable", "SwitchingCandidates", "SwitchingReadout", "TrainingHistory",
    "TrajectoryEvaluation", "admissible_initializer", "candidate_start_states",
    "default_sensor", "evaluate_trajectory", "greedy_trajectory",
    "LearnedBestResponse", "LearnedLocalSSE", "learned_best_response",
    "rl_local_sse", "solve_glide_mdp", "switching_candidates", "train",
]


if __name__ == "__main__":
    main(sys.argv)

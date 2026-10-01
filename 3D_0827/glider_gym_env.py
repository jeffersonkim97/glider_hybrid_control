"""Gymnasium adapter for the shared Bellman/attacker glide MDP.

This module does not define new dynamics.  Valid transitions, stage costs, and
terminal states all come from :class:`P1b_RL_approximation.AttackerMDP`.  Its job
is to give a terrain-conditioned DQN a stable observation and action contract.

The action ID is the target heading bin, which is also the index of the grid's
motion primitive.  Unlike the tabular learner's local adjacency-list index, this
keeps one physical meaning for an action at every state.  Invalid primitives are
reported by ``action_mask`` and must be masked both when acting and when computing
the DQN bootstrap target.

The first adapter intentionally consumes an already-built ``Scene`` through the
MDP.  Consequently scene/graph construction time is not free and must be recorded
as ``T_env_setup`` in comparisons.  A later online-transition environment may
remove the full backward-reachability prerequisite, but it must preserve the
valid-action cost equivalence tested here.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from P1b_RL_approximation import AttackerMDP


FloatArray = NDArray[np.float32]
MaskArray = NDArray[np.int8]


def terrain_height_map(mdp: AttackerMDP) -> FloatArray:
    """Rasterize the current box-backed terrain on the Bellman x-y lattice.

    Values are obstacle top heights normalized to ``[0, 1]`` by the grid's
    altitude range.  The leading singleton channel is ready for a 2-D CNN.
    Terrain implementations without ``obstacle_boxes`` fail explicitly rather
    than silently presenting a flat map to the policy.
    """

    obstacle_boxes = getattr(mdp.scene.terrain, "obstacle_boxes", None)
    if obstacle_boxes is None:
        raise TypeError(
            "terrain-conditioned observations require obstacle_boxes() or a "
            "terrain-specific rasterizer"
        )

    grid = mdp.grid
    x_coordinates = np.asarray(grid.x_coordinates, dtype=float)
    y_coordinates = np.asarray(grid.y_coordinates, dtype=float)
    height = np.full(
        (grid.y_count, grid.x_count),
        float(mdp.scene.terrain.ground_z),
        dtype=np.float32,
    )
    for box in obstacle_boxes():
        x_mask = (x_coordinates >= box.x_limits[0]) & (
            x_coordinates <= box.x_limits[1]
        )
        y_mask = (y_coordinates >= box.y_limits[0]) & (
            y_coordinates <= box.y_limits[1]
        )
        if np.any(x_mask) and np.any(y_mask):
            region = np.ix_(y_mask, x_mask)
            height[region] = np.maximum(height[region], np.float32(box.top_z))

    altitude_span = float(grid.maximum_altitude_map - grid.minimum_altitude_map)
    normalized = (height - np.float32(grid.minimum_altitude_map)) / np.float32(
        altitude_span
    )
    return np.clip(normalized, 0.0, 1.0)[None, :, :].astype(np.float32, copy=False)


class TerrainGlideEnv(gym.Env[dict[str, FloatArray | MaskArray], int]):
    """One fixed-terrain, fixed-sensor attacker best-response environment.

    A terrain-generalized training run should create environments for many
    training terrains and sensor positions, while sharing one policy network.
    Held-out terrain factories belong only to validation or test splits.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        mdp: AttackerMDP,
        start_state_ids: Iterable[int],
        *,
        max_steps: int = 400,
        invalid_action_cost: float = 1_000.0,
    ) -> None:
        super().__init__()
        self.mdp = mdp
        self.grid = mdp.grid
        self.start_state_ids = tuple(int(value) for value in start_state_ids)
        if not self.start_state_ids:
            raise ValueError("start_state_ids must contain at least one state")
        if any(self.mdp.is_terminal(value) for value in self.start_state_ids):
            raise ValueError("start states must be non-terminal")
        if not isinstance(max_steps, Integral) or isinstance(max_steps, bool):
            raise TypeError("max_steps must be an integer")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be positive")
        if not np.isfinite(invalid_action_cost) or invalid_action_cost <= 0.0:
            raise ValueError("invalid_action_cost must be finite and positive")

        self.max_steps = int(max_steps)
        self.invalid_action_cost = float(invalid_action_cost)
        self.action_space = spaces.Discrete(self.grid.heading_bin_count)
        self._terrain = terrain_height_map(mdp)
        self.observation_space = spaces.Dict(
            {
                "terrain": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=self._terrain.shape,
                    dtype=np.float32,
                ),
                "state": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(13,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.grid.heading_bin_count,),
                    dtype=np.int8,
                ),
            }
        )
        self.state_id: int | None = None
        self.elapsed_steps = 0
        self._episode_ended = False

    def _edge_map(self, state_id: int) -> dict[int, tuple[int, Any]]:
        result: dict[int, tuple[int, Any]] = {}
        for local_index, edge in enumerate(self.mdp.actions(state_id)):
            action_id = int(edge.target_state.heading_bin)
            if action_id in result:
                raise RuntimeError(
                    f"state {state_id} has multiple successors for action {action_id}"
                )
            result[action_id] = (local_index, edge)
        return result

    def action_mask(self, state_id: int | None = None) -> MaskArray:
        """Return the fixed-action validity mask at a state."""

        selected = self.state_id if state_id is None else int(state_id)
        if selected is None:
            raise RuntimeError("reset() must be called before requesting a mask")
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for action_id in self._edge_map(selected):
            mask[action_id] = 1
        return mask

    @staticmethod
    def _normalize(value: float, minimum: float, maximum: float) -> float:
        return float(2.0 * (value - minimum) / (maximum - minimum) - 1.0)

    def _state_vector(self, state_id: int) -> FloatArray:
        state = self.grid.decode(state_id)
        position = np.asarray(self.grid.position_map(state), dtype=float)
        heading = float(self.grid.heading_rad(state.heading_bin))
        goal = np.asarray(self.mdp.scene.config.goal.as_array(), dtype=float)
        sensor = np.asarray(self.mdp.sensor_map, dtype=float)

        x_span = float(self.grid.bounds.x_max - self.grid.bounds.x_min)
        y_span = float(self.grid.bounds.y_max - self.grid.bounds.y_min)
        z_span = float(
            self.grid.maximum_altitude_map - self.grid.minimum_altitude_map
        )
        objective = self.mdp.objective
        values = np.asarray(
            [
                self._normalize(
                    position[0], self.grid.bounds.x_min, self.grid.bounds.x_max
                ),
                self._normalize(
                    position[1], self.grid.bounds.y_min, self.grid.bounds.y_max
                ),
                self._normalize(
                    position[2],
                    self.grid.minimum_altitude_map,
                    self.grid.maximum_altitude_map,
                ),
                np.sin(heading),
                np.cos(heading),
                (goal[0] - position[0]) / x_span,
                (goal[1] - position[1]) / y_span,
                (goal[2] - position[2]) / z_span,
                (sensor[0] - position[0]) / x_span,
                (sensor[1] - position[1]) / y_span,
                (sensor[2] - position[2]) / z_span,
                float(objective.hazard_weight),
                float(objective.time_weight),
            ],
            dtype=np.float32,
        )
        return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)

    def _observation(self) -> dict[str, FloatArray | MaskArray]:
        if self.state_id is None:
            raise RuntimeError("reset() must be called before observing")
        return {
            "terrain": self._terrain.copy(),
            "state": self._state_vector(self.state_id),
            "action_mask": self.action_mask(self.state_id),
        }

    def _info(self, **extra: Any) -> dict[str, Any]:
        if self.state_id is None:
            raise RuntimeError("reset() must be called before requesting info")
        state = self.grid.decode(self.state_id)
        result: dict[str, Any] = {
            "state_id": self.state_id,
            "position_map": np.asarray(self.grid.position_map(state), dtype=float),
            "action_mask": self.action_mask(self.state_id),
            "terrain_category": getattr(
                self.mdp.scene.terrain, "category_id", type(self.mdp.scene.terrain).__name__
            ),
            "sensor_map": np.asarray(self.mdp.sensor_map, dtype=float),
            "elapsed_steps": self.elapsed_steps,
        }
        result.update(extra)
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, FloatArray | MaskArray], dict[str, Any]]:
        super().reset(seed=seed)
        requested = None if options is None else options.get("start_state_id")
        if requested is None:
            index = int(self.np_random.integers(len(self.start_state_ids)))
            state_id = self.start_state_ids[index]
        else:
            state_id = int(requested)
            if state_id not in self.start_state_ids:
                raise ValueError("requested start_state_id is not in start_state_ids")
        self.state_id = state_id
        self.elapsed_steps = 0
        self._episode_ended = False
        return self._observation(), self._info(termination_reason=None)

    def step(
        self,
        action: int,
    ) -> tuple[
        dict[str, FloatArray | MaskArray], float, bool, bool, dict[str, Any]
    ]:
        if self.state_id is None:
            raise RuntimeError("reset() must be called before step()")
        if self._episode_ended:
            raise RuntimeError("reset() must be called after an episode ends")
        if not isinstance(action, Integral) or isinstance(action, bool):
            raise TypeError("action must be an integer heading-bin ID")
        action_id = int(action)
        if not self.action_space.contains(action_id):
            raise ValueError("action lies outside the action space")

        edge_map = self._edge_map(self.state_id)
        selected = edge_map.get(action_id)
        if selected is None:
            self.elapsed_steps += 1
            self._episode_ended = True
            return (
                self._observation(),
                -self.invalid_action_cost,
                True,
                False,
                self._info(
                    termination_reason="invalid_action",
                    reached_goal=False,
                    invalid_action=True,
                    step_cost=self.invalid_action_cost,
                ),
            )

        local_index, edge = selected
        hazard = float(self.mdp.hazard_row(self.state_id)[local_index])
        next_state_id, step_cost, target_terminal = self.mdp.step(
            self.state_id, local_index
        )
        if next_state_id != int(edge.target_id):
            raise RuntimeError("fixed action mapping drifted from AttackerMDP.step")

        self.state_id = next_state_id
        self.elapsed_steps += 1
        dead_end = not target_terminal and not bool(self._edge_map(next_state_id))
        terminated = bool(target_terminal or dead_end)
        truncated = bool(self.elapsed_steps >= self.max_steps and not terminated)
        self._episode_ended = terminated or truncated
        reason = (
            "goal"
            if target_terminal
            else "dead_end"
            if dead_end
            else "max_steps"
            if truncated
            else None
        )
        return (
            self._observation(),
            -float(step_cost),
            terminated,
            truncated,
            self._info(
                termination_reason=reason,
                reached_goal=bool(target_terminal),
                invalid_action=False,
                step_cost=float(step_cost),
                edge_hazard=hazard,
                edge_duration_s=float(edge.duration_s),
                action_id=action_id,
            ),
        )


__all__ = ["TerrainGlideEnv", "terrain_height_map"]

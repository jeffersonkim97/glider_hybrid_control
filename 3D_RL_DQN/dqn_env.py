"""Gymnasium wrapper for the Phase 1 problem and Phase 2 observation."""

from __future__ import annotations

from collections import OrderedDict
from numbers import Integral
from typing import Any, Iterable, Protocol

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from attacker_br_problem import AttackerBRProblem, AttackerTransition
from dqn_action_catalog import HeadingActionCatalog, MaskArray
from terrain_observation import TerrainObservationBuilder, TensorReadyObservation


Float32Array = NDArray[np.float32]


class BatchedTensorObservationBuilder(Protocol):
    def build_many(
        self, state_ids: Iterable[int],
    ) -> list[TensorReadyObservation]: ...


class ObservationTensorCache:
    """Cache immutable deterministic Phase 2 tensors by canonical state ID."""

    def __init__(
        self, builder: TerrainObservationBuilder, *, max_entries: int | None = None,
        batch_builder: BatchedTensorObservationBuilder | None = None,
    ) -> None:
        if max_entries is not None and int(max_entries) < 1:
            raise ValueError("max_entries must be positive or None")
        self.builder = builder
        self.batch_builder = batch_builder
        self.max_entries = None if max_entries is None else int(max_entries)
        self._values: OrderedDict[int, TensorReadyObservation] = OrderedDict()

    def get(self, state_id: int) -> TensorReadyObservation:
        return self.get_many((int(state_id),))[0]

    def get_many(self, state_ids: Iterable[int]) -> list[TensorReadyObservation]:
        ids = tuple(int(value) for value in state_ids)
        resolved: dict[int, TensorReadyObservation] = {}
        missing: list[int] = []
        for state_id in ids:
            if state_id in resolved:
                continue
            cached = self._values.get(state_id)
            if cached is None:
                missing.append(state_id)
            else:
                self._values.move_to_end(state_id)
                resolved[state_id] = cached
        if missing:
            if self.batch_builder is None:
                converted_values = []
                for state_id in missing:
                    raw = self.builder.build(state_id)
                    converted_values.append(self.builder.to_tensor_ready(raw))
            else:
                converted_values = self.batch_builder.build_many(missing)
            if len(converted_values) != len(missing):
                raise RuntimeError("batch observation builder returned the wrong count")
            for state_id, converted in zip(missing, converted_values):
                converted.scalar.setflags(write=False)
                converted.spatial.setflags(write=False)
                resolved[state_id] = converted
                self._values[state_id] = converted
                if self.max_entries is not None and len(self._values) > self.max_entries:
                    self._values.popitem(last=False)
        return [resolved[state_id] for state_id in ids]

    def clear(self) -> None:
        """Drop derived tensors; a later access reconstructs the same values."""

        self._values.clear()

    @property
    def cached_states(self) -> int:
        return len(self._values)


class DQNAttackerEnv(gym.Env[dict[str, Float32Array], int]):
    """Deterministic glide MDP with masked fixed heading-bin actions.

    Invalid actions are rejected instead of assigned an invented penalty.  The
    DQN policy, exploration rule, and bootstrap target must all apply the mask.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        problem: AttackerBRProblem,
        observation_builder: TerrainObservationBuilder,
        start_state_ids: Iterable[int],
        *,
        action_catalog: HeadingActionCatalog | None = None,
        observation_cache: ObservationTensorCache | None = None,
    ) -> None:
        super().__init__()
        self.problem = problem
        self.builder = observation_builder
        self.catalog = action_catalog or HeadingActionCatalog(problem)
        self.cache = observation_cache or ObservationTensorCache(observation_builder)
        self.start_state_ids = tuple(int(value) for value in start_state_ids)
        if not self.start_state_ids:
            raise ValueError("start_state_ids cannot be empty")
        for state_id in self.start_state_ids:
            if self.problem.is_terminal(state_id):
                raise ValueError("start_state_ids cannot contain terminal states")
            if not self.catalog.feasible_mask(state_id).any():
                raise ValueError("every start state must have a feasible action")

        shapes = self.builder.tensor_ready_shapes
        self.observation_space = spaces.Dict({
            "scalar": spaces.Box(
                low=-1.0, high=1.0, shape=shapes["scalar"], dtype=np.float32,
            ),
            "spatial": spaces.Box(
                low=-1.0, high=1.0, shape=shapes["spatial"], dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(self.catalog.action_count)
        # Every edge loses at least one altitude bin; this is derived from the
        # finite Phase 1 DAG rather than inherited from CartPole/tutorial code.
        self.max_episode_steps = max(1, int(self.problem.grid.altitude_count) - 1)
        self.state_id: int | None = None
        self.elapsed_steps = 0
        self._episode_ended = False

    def _observation(self, state_id: int) -> dict[str, Float32Array]:
        tensor = self.cache.get(state_id)
        return {
            "scalar": tensor.scalar.copy(),
            "spatial": tensor.spatial.copy(),
        }

    def action_mask(self, state_id: int | None = None) -> MaskArray:
        selected = self.state_id if state_id is None else int(state_id)
        if selected is None:
            raise RuntimeError("reset() must be called before requesting a mask")
        return self.catalog.feasible_mask(selected)

    def _info(
        self, state_id: int, *, action_mask: MaskArray | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "state_id": int(state_id),
            "action_mask": (
                self.catalog.feasible_mask(state_id)
                if action_mask is None else action_mask
            ),
            "elapsed_steps": self.elapsed_steps,
        }
        result.update(extra)
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Float32Array], dict[str, Any]]:
        super().reset(seed=seed)
        requested = None if options is None else options.get("start_state_id")
        if requested is None:
            index = int(self.np_random.integers(len(self.start_state_ids)))
            state_id = self.start_state_ids[index]
        else:
            state_id = int(requested)
            if state_id not in self.start_state_ids:
                raise ValueError("requested start_state_id is not an approved start")
        self.state_id = state_id
        self.elapsed_steps = 0
        self._episode_ended = False
        return self._observation(state_id), self._info(
            state_id, reached_goal=False, termination_reason=None,
        )

    def step(
        self, action: int,
    ) -> tuple[dict[str, Float32Array], float, bool, bool, dict[str, Any]]:
        if self.state_id is None:
            raise RuntimeError("reset() must be called before step()")
        if self._episode_ended:
            raise RuntimeError("reset() must be called after an episode ends")
        if not isinstance(action, Integral) or isinstance(action, bool):
            raise TypeError("action must be an integer heading-bin ID")
        action_id = int(action)
        transition: AttackerTransition = self.catalog.transition(
            self.state_id, action_id,
        )
        self.state_id = transition.target_state_id
        self.elapsed_steps += 1
        reached_goal = bool(transition.terminal)
        next_mask = self.catalog.feasible_mask(self.state_id)
        dead_end = bool(not reached_goal and not next_mask.any())
        terminated = bool(reached_goal or dead_end)
        truncated = bool(
            self.elapsed_steps >= self.max_episode_steps and not terminated
        )
        self._episode_ended = bool(terminated or truncated)
        reason = (
            "goal" if reached_goal else "dead_end" if dead_end
            else "max_steps" if truncated else None
        )
        return (
            self._observation(self.state_id),
            -float(transition.stage_cost),
            terminated,
            truncated,
            self._info(
                self.state_id,
                action_mask=next_mask,
                reached_goal=reached_goal,
                termination_reason=reason,
                stage_cost=float(transition.stage_cost),
                cumulative_hazard=float(transition.cumulative_hazard),
                action_id=action_id,
                source_state_id=int(transition.source_state_id),
            ),
        )


__all__ = ["DQNAttackerEnv", "ObservationTensorCache"]

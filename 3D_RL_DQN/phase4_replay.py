"""Scenario-aware replay storage for one shared Phase 16.4 DQN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MultiScenarioTransition:
    scenario_index: int
    state_id: int
    action_id: int
    reward: float
    next_state_id: int
    terminal: bool
    next_feasible_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class MultiScenarioBatch:
    scenario_indices: NDArray[np.int64]
    state_ids: NDArray[np.int64]
    action_ids: NDArray[np.int64]
    rewards: NDArray[np.float32]
    next_state_ids: NDArray[np.int64]
    terminal: NDArray[np.bool_]
    next_feasible_masks: NDArray[np.bool_]


class MultiScenarioReplayBuffer:
    """Cyclic replay whose state handles are qualified by training scenario."""

    def __init__(self, capacity: int, action_count: int, seed: int) -> None:
        if int(capacity) < 1 or int(action_count) < 1:
            raise ValueError("capacity and action_count must be positive")
        self.capacity = int(capacity)
        self.action_count = int(action_count)
        self._rng = np.random.default_rng(int(seed))
        self._items: list[MultiScenarioTransition] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self, *, scenario_index: int, state_id: int, action_id: int,
        reward: float, next_state_id: int, terminal: bool,
        next_feasible_mask: NDArray[np.bool_],
    ) -> None:
        mask = np.asarray(next_feasible_mask, dtype=np.bool_)
        if mask.shape != (self.action_count,):
            raise ValueError("next_feasible_mask has the wrong shape")
        if int(scenario_index) < 0:
            raise ValueError("scenario_index must be nonnegative")
        if not 0 <= int(action_id) < self.action_count:
            raise ValueError("action_id lies outside the catalog")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if not bool(terminal) and not mask.any():
            raise ValueError("nonterminal transition requires a feasible next action")
        stored_mask = mask.copy()
        stored_mask.setflags(write=False)
        item = MultiScenarioTransition(
            scenario_index=int(scenario_index), state_id=int(state_id),
            action_id=int(action_id), reward=float(reward),
            next_state_id=int(next_state_id), terminal=bool(terminal),
            next_feasible_mask=stored_mask,
        )
        if len(self._items) < self.capacity:
            self._items.append(item)
        else:
            self._items[self._next_index] = item
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, batch_size: int) -> MultiScenarioBatch:
        size = int(batch_size)
        if size < 1 or size > len(self._items):
            raise ValueError("batch_size must lie within populated replay size")
        indices = self._rng.choice(len(self._items), size=size, replace=False)
        items = [self._items[int(index)] for index in indices]
        return MultiScenarioBatch(
            scenario_indices=np.asarray(
                [item.scenario_index for item in items], dtype=np.int64,
            ),
            state_ids=np.asarray([item.state_id for item in items], dtype=np.int64),
            action_ids=np.asarray([item.action_id for item in items], dtype=np.int64),
            rewards=np.asarray([item.reward for item in items], dtype=np.float32),
            next_state_ids=np.asarray(
                [item.next_state_id for item in items], dtype=np.int64,
            ),
            terminal=np.asarray([item.terminal for item in items], dtype=np.bool_),
            next_feasible_masks=np.stack(
                [item.next_feasible_mask for item in items], axis=0,
            ),
        )

    def scenario_counts(self) -> dict[int, int]:
        result: dict[int, int] = {}
        for item in self._items:
            result[item.scenario_index] = result.get(item.scenario_index, 0) + 1
        return result

    def state_dict(self) -> dict[str, Any]:
        count = len(self._items)
        masks = (
            np.stack([item.next_feasible_mask for item in self._items], axis=0)
            if count else np.empty((0, self.action_count), dtype=np.bool_)
        )
        return {
            "capacity": self.capacity,
            "action_count": self.action_count,
            "next_index": self._next_index,
            "rng_state": self._rng.bit_generator.state,
            "scenario_indices": np.asarray(
                [item.scenario_index for item in self._items], dtype=np.int64,
            ),
            "state_ids": np.asarray(
                [item.state_id for item in self._items], dtype=np.int64,
            ),
            "action_ids": np.asarray(
                [item.action_id for item in self._items], dtype=np.int64,
            ),
            "rewards": np.asarray(
                [item.reward for item in self._items], dtype=np.float32,
            ),
            "next_state_ids": np.asarray(
                [item.next_state_id for item in self._items], dtype=np.int64,
            ),
            "terminal": np.asarray(
                [item.terminal for item in self._items], dtype=np.bool_,
            ),
            "next_feasible_masks": masks,
        }

    @classmethod
    def from_state_dict(cls, values: dict[str, Any]) -> "MultiScenarioReplayBuffer":
        replay = cls(
            int(values["capacity"]), int(values["action_count"]), seed=0,
        )
        masks = np.asarray(values["next_feasible_masks"], dtype=np.bool_)
        lengths = {
            len(values[name]) for name in (
                "scenario_indices", "state_ids", "action_ids", "rewards",
                "next_state_ids", "terminal", "next_feasible_masks",
            )
        }
        if len(lengths) != 1:
            raise ValueError("serialized replay arrays have inconsistent lengths")
        for index in range(len(values["state_ids"])):
            mask = masks[index].copy()
            mask.setflags(write=False)
            replay._items.append(MultiScenarioTransition(
                scenario_index=int(values["scenario_indices"][index]),
                state_id=int(values["state_ids"][index]),
                action_id=int(values["action_ids"][index]),
                reward=float(values["rewards"][index]),
                next_state_id=int(values["next_state_ids"][index]),
                terminal=bool(values["terminal"][index]),
                next_feasible_mask=mask,
            ))
        if len(replay._items) > replay.capacity:
            raise ValueError("serialized replay exceeds its capacity")
        replay._next_index = int(values["next_index"])
        if not 0 <= replay._next_index < replay.capacity:
            raise ValueError("serialized replay next_index is invalid")
        replay._rng.bit_generator.state = values["rng_state"]
        return replay


__all__ = [
    "MultiScenarioBatch", "MultiScenarioReplayBuffer", "MultiScenarioTransition",
]

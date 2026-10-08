"""Memory-efficient deterministic-observation replay for Phase 16.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReplayTransition:
    state_id: int
    action_id: int
    reward: float
    next_state_id: int
    terminal: bool
    next_feasible_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class ReplayBatch:
    state_ids: NDArray[np.int64]
    action_ids: NDArray[np.int64]
    rewards: NDArray[np.float32]
    next_state_ids: NDArray[np.int64]
    terminal: NDArray[np.bool_]
    next_feasible_masks: NDArray[np.bool_]


class ReplayBuffer:
    """Cyclic replay using state IDs as immutable Phase 2 observation handles."""

    def __init__(self, capacity: int, action_count: int, seed: int) -> None:
        if int(capacity) < 1:
            raise ValueError("capacity must be positive")
        if int(action_count) < 1:
            raise ValueError("action_count must be positive")
        self.capacity = int(capacity)
        self.action_count = int(action_count)
        self._rng = np.random.default_rng(int(seed))
        self._items: list[ReplayTransition] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        *,
        state_id: int,
        action_id: int,
        reward: float,
        next_state_id: int,
        terminal: bool,
        next_feasible_mask: NDArray[np.bool_],
    ) -> None:
        mask = np.asarray(next_feasible_mask, dtype=np.bool_)
        if mask.shape != (self.action_count,):
            raise ValueError("next_feasible_mask has the wrong shape")
        if not 0 <= int(action_id) < self.action_count:
            raise ValueError("action_id lies outside the catalog")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if not bool(terminal) and not mask.any():
            raise ValueError("nonterminal replay transitions need a feasible next action")
        stored_mask = mask.copy()
        stored_mask.setflags(write=False)
        transition = ReplayTransition(
            state_id=int(state_id), action_id=int(action_id), reward=float(reward),
            next_state_id=int(next_state_id), terminal=bool(terminal),
            next_feasible_mask=stored_mask,
        )
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, batch_size: int) -> ReplayBatch:
        size = int(batch_size)
        if size < 1 or size > len(self._items):
            raise ValueError("batch_size must lie within the populated replay size")
        indices = self._rng.choice(len(self._items), size=size, replace=False)
        items = [self._items[int(index)] for index in indices]
        return ReplayBatch(
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

    def transition(self, index: int) -> ReplayTransition:
        return self._items[int(index)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "representation": "state_id_handles_to_deterministic_phase2_cache",
            "capacity": self.capacity,
            "action_count": self.action_count,
            "size": len(self),
        }


__all__ = ["ReplayBatch", "ReplayBuffer", "ReplayTransition"]

"""Exact Bellman-to-RL transition and action-mask contract.

This layer does not invent dynamics.  It exposes the physical-successor graph
saved by the authoritative 2D Bellman solver through an RL-friendly API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


TRANSITION_SCHEMA_VERSION = "1.0.0"
VALUE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class TransitionRecord:
    """One indexed transition in the exact physical-successor MDP."""

    state_index: tuple[int, int]
    state: tuple[float, float]
    action_index: int
    edge_cost: float
    reward: float
    next_state_index: tuple[int, int] | None
    next_state: tuple[float, float] | None
    done: bool
    feasible: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExactTransitionModel:
    """Indexable RL view of the saved exact physical-successor graph."""

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.z_grid = np.asarray(arrays["z_grid"], dtype=float)
        self.h_grid = np.asarray(arrays["h_grid"], dtype=float)
        self.valid = np.asarray(arrays["transition_valid"], dtype=bool)
        self.terminal = np.asarray(arrays["transition_terminal"], dtype=bool)
        self.terminal_fraction = np.asarray(
            arrays["transition_terminal_fraction"], dtype=float
        )
        self.cost = np.asarray(arrays["edge_cost"], dtype=float)
        self.reference_value = np.asarray(arrays["exact_value"], dtype=float)
        self.reference_policy = np.asarray(
            arrays["exact_policy_action_index"], dtype=np.int32
        )
        self.goal_mask = np.asarray(arrays["goal_mask"], dtype=bool)
        self.forward_cells = np.asarray(arrays["action_forward_cells"], dtype=int)
        self.descent_cells = np.asarray(arrays["action_descent_cells"], dtype=int)
        self.action_speed = np.asarray(arrays["action_speed"], dtype=float)
        self.action_gamma = np.asarray(arrays["action_gamma"], dtype=float)
        self.action_duration = np.asarray(arrays["action_duration"], dtype=float)
        self._validate_shapes()

    @property
    def state_shape(self) -> tuple[int, int]:
        return self.reference_value.shape

    @property
    def action_count(self) -> int:
        return int(self.forward_cells.size)

    @property
    def observation_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "z": (float(self.z_grid[0]), float(self.z_grid[-1])),
            "h": (float(self.h_grid[0]), float(self.h_grid[-1])),
        }

    def _validate_shapes(self) -> None:
        spatial = (self.z_grid.size, self.h_grid.size)
        expected_graph = (*spatial, self.forward_cells.size)
        if self.reference_value.shape != spatial:
            raise ValueError("Reference value shape does not match z/h grids.")
        if self.reference_policy.shape != spatial or self.goal_mask.shape != spatial:
            raise ValueError("Policy/goal shape does not match spatial state grid.")
        for name, array in (
            ("valid", self.valid),
            ("terminal", self.terminal),
            ("terminal_fraction", self.terminal_fraction),
            ("cost", self.cost),
        ):
            if array.shape != expected_graph:
                raise ValueError(
                    f"Transition {name} shape {array.shape} != {expected_graph}."
                )
        for name, array in (
            ("descent_cells", self.descent_cells),
            ("action_speed", self.action_speed),
            ("action_gamma", self.action_gamma),
            ("action_duration", self.action_duration),
        ):
            if array.shape != self.forward_cells.shape:
                raise ValueError(f"Action field {name} has inconsistent length.")

    def state(self, state_index: tuple[int, int]) -> np.ndarray:
        zi, hi = self._state_indices(state_index)
        return np.array([self.z_grid[zi], self.h_grid[hi]], dtype=float)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        if state.shape != (2,):
            raise ValueError("State must have shape (2,) in (z,h) order.")
        lower = np.array([self.z_grid[0], self.h_grid[0]], dtype=float)
        upper = np.array([self.z_grid[-1], self.h_grid[-1]], dtype=float)
        return (state - lower) / (upper - lower)

    def action_mask(self, state_index: tuple[int, int]) -> np.ndarray:
        zi, hi = self._state_indices(state_index)
        return self.valid[zi, hi].copy()

    def transition(
        self, state_index: tuple[int, int], action_index: int
    ) -> TransitionRecord:
        zi, hi = self._state_indices(state_index)
        ai = self._action_index(action_index)
        state = np.array([self.z_grid[zi], self.h_grid[hi]], dtype=float)
        feasible = bool(self.valid[zi, hi, ai])
        if not feasible:
            return TransitionRecord(
                state_index=(zi, hi),
                state=(float(state[0]), float(state[1])),
                action_index=ai,
                edge_cost=float("inf"),
                reward=float("-inf"),
                next_state_index=None,
                next_state=None,
                done=False,
                feasible=False,
            )

        edge_cost = float(self.cost[zi, hi, ai])
        done = bool(self.terminal[zi, hi, ai])
        next_zi = zi + int(self.forward_cells[ai])
        next_hi = hi - int(self.descent_cells[ai])
        if done:
            fraction = float(self.terminal_fraction[zi, hi, ai])
            delta = np.array(
                [
                    self.forward_cells[ai] * (self.z_grid[1] - self.z_grid[0]),
                    -self.descent_cells[ai] * (self.h_grid[1] - self.h_grid[0]),
                ],
                dtype=float,
            )
            next_state = state + fraction * delta
            next_index = None
        else:
            next_state = np.array(
                [self.z_grid[next_zi], self.h_grid[next_hi]], dtype=float
            )
            next_index = (next_zi, next_hi)
        return TransitionRecord(
            state_index=(zi, hi),
            state=(float(state[0]), float(state[1])),
            action_index=ai,
            edge_cost=edge_cost,
            reward=-edge_cost,
            next_state_index=next_index,
            next_state=(float(next_state[0]), float(next_state[1])),
            done=done,
            feasible=True,
        )

    def reconstruct_bellman_solution(self) -> tuple[np.ndarray, np.ndarray]:
        """Solve the finite DAG using only this transition contract."""

        z_count, h_count = self.state_shape
        value = np.full(self.state_shape, np.inf, dtype=float)
        policy = np.full(self.state_shape, -1, dtype=np.int32)
        value[self.goal_mask] = 0.0
        h_indices = np.arange(h_count)

        for zi in range(z_count - 1, -1, -1):
            q_values = np.where(self.valid[zi], self.cost[zi], np.inf).copy()
            for ai in range(self.action_count):
                nonterminal = self.valid[zi, :, ai] & ~self.terminal[zi, :, ai]
                if not np.any(nonterminal):
                    continue
                next_zi = zi + int(self.forward_cells[ai])
                next_hi = h_indices - int(self.descent_cells[ai])
                if next_zi >= z_count:
                    q_values[nonterminal, ai] = np.inf
                    continue
                valid_h = nonterminal & (next_hi >= 0) & (next_hi < h_count)
                q_values[nonterminal, ai] = np.inf
                q_values[valid_h, ai] = (
                    self.cost[zi, valid_h, ai]
                    + value[next_zi, next_hi[valid_h]]
                )

            best_action = np.argmin(q_values, axis=1)
            best_value = q_values[h_indices, best_action]
            assign = np.isfinite(best_value) & ~self.goal_mask[zi]
            value[zi, assign] = best_value[assign]
            policy[zi, assign] = best_action[assign]
        return value, policy

    def _state_indices(self, state_index: tuple[int, int]) -> tuple[int, int]:
        if len(state_index) != 2:
            raise ValueError("state_index must be (z_index, h_index).")
        zi, hi = int(state_index[0]), int(state_index[1])
        if not (0 <= zi < self.z_grid.size and 0 <= hi < self.h_grid.size):
            raise IndexError("state_index lies outside the reference grid.")
        return zi, hi

    def _action_index(self, action_index: int) -> int:
        ai = int(action_index)
        if not 0 <= ai < self.action_count:
            raise IndexError("action_index lies outside the action table.")
        return ai


def validate_transition_contract(model: ExactTransitionModel) -> dict[str, Any]:
    """Prove that the RL interface reconstructs the saved exact solution."""

    reconstructed_value, reconstructed_policy = model.reconstruct_bellman_solution()
    reference_finite = np.isfinite(model.reference_value)
    reconstructed_finite = np.isfinite(reconstructed_value)
    common = reference_finite & reconstructed_finite
    value_error = np.full(model.state_shape, np.nan, dtype=float)
    value_error[common] = reconstructed_value[common] - model.reference_value[common]
    maximum_value_error = (
        float(np.max(np.abs(value_error[common]))) if np.any(common) else 0.0
    )
    policy_states = model.reference_policy >= 0
    policy_matches = reconstructed_policy == model.reference_policy
    agreement = (
        float(np.mean(policy_matches[policy_states]))
        if np.any(policy_states)
        else 1.0
    )

    feasible_nonterminal = model.valid & ~model.terminal
    zi, hi, ai = np.nonzero(feasible_nonterminal)
    next_zi = zi + model.forward_cells[ai]
    next_hi = hi - model.descent_cells[ai]
    nonterminal_in_bounds = bool(
        np.all((next_zi >= 0) & (next_zi < model.z_grid.size))
        and np.all((next_hi >= 0) & (next_hi < model.h_grid.size))
    )
    terminal_fractions = model.terminal_fraction[model.terminal]
    terminal_fractions_valid = bool(
        terminal_fractions.size > 0
        and np.all((terminal_fractions > 0.0) & (terminal_fractions <= 1.0))
    )
    selected = model.reference_policy >= 0
    selected_z, selected_h = np.nonzero(selected)
    selected_a = model.reference_policy[selected_z, selected_h]
    selected_actions_feasible = bool(
        np.all(model.valid[selected_z, selected_h, selected_a])
    )

    checks = {
        "finite_state_mask_exact_match": bool(
            np.array_equal(reference_finite, reconstructed_finite)
        ),
        "value_exact_match": maximum_value_error <= VALUE_TOLERANCE,
        "greedy_policy_exact_match": agreement == 1.0,
        "selected_actions_feasible": selected_actions_feasible,
        "nonterminal_successors_in_bounds": nonterminal_in_bounds,
        "terminal_fractions_in_unit_interval": terminal_fractions_valid,
        "reward_is_negative_edge_cost": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "metrics": {
            "state_count": int(np.prod(model.state_shape)),
            "finite_value_state_count": int(np.count_nonzero(reference_finite)),
            "action_count": model.action_count,
            "state_action_count": int(model.valid.size),
            "feasible_transition_count": int(np.count_nonzero(model.valid)),
            "terminal_transition_count": int(np.count_nonzero(model.terminal)),
            "states_with_feasible_action": int(
                np.count_nonzero(np.any(model.valid, axis=2))
            ),
            "maximum_feasible_actions_per_state": int(
                np.max(np.count_nonzero(model.valid, axis=2))
            ),
            "mean_feasible_actions_on_active_states": float(
                np.mean(
                    np.count_nonzero(model.valid, axis=2)[
                        np.any(model.valid, axis=2)
                    ]
                )
            ),
            "maximum_value_error": maximum_value_error,
            "greedy_policy_agreement": agreement,
        },
        "failed_checks": failed,
        "reconstructed_value": reconstructed_value,
        "reconstructed_policy": reconstructed_policy,
        "value_error": value_error,
        "policy_match_map": policy_matches,
    }


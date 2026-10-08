"""Fixed Phase 16.3 action catalog derived from Phase 1 transitions."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from attacker_br_problem import AttackerBRProblem, AttackerTransition
from bellman_geometry import GlideEdge, wrapped_angle_difference
from bellman_state import BellmanState


MaskArray = NDArray[np.bool_]
ACTION_CATALOG_SCHEMA = "p1b-heading-action-catalog-v1"


@dataclass(frozen=True)
class ActionDescriptor:
    """One global action ID and its authoritative motion-primitive meaning."""

    action_id: int
    target_heading_bin: int
    target_heading_rad: float
    x_index_offset: int
    y_index_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionRow:
    """Immutable structural transition row for one canonical state."""

    state_id: int
    feasible_mask: MaskArray
    local_indices: NDArray[np.int16]
    target_state_ids: NDArray[np.int64]
    terminal: MaskArray
    edges: tuple[Any, ...]


class SharedActionRowCache:
    """Bounded terrain-level cache shared by sensor-conditioned problems."""

    def __init__(self, max_entries: int | None = None) -> None:
        if max_entries is not None and int(max_entries) < 1:
            raise ValueError("max_entries must be positive or None")
        self.max_entries = None if max_entries is None else int(max_entries)
        self._values: OrderedDict[int, ActionRow] = OrderedDict()
        self.batch_templates: dict[str, np.ndarray] | None = None
        self.hits = 0
        self.misses = 0

    def get(self, state_id: int) -> ActionRow | None:
        selected = int(state_id)
        row = self._values.get(selected)
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        self._values.move_to_end(selected)
        return row

    def put(self, row: ActionRow) -> None:
        self._values[int(row.state_id)] = row
        self._values.move_to_end(int(row.state_id))
        if self.max_entries is not None:
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)

    @property
    def cached_states(self) -> int:
        return len(self._values)


class HeadingActionCatalog:
    """Map fixed heading-bin IDs to state-dependent Phase 1 successors."""

    def __init__(
        self, problem: AttackerBRProblem, *,
        row_cache: SharedActionRowCache | None = None,
    ) -> None:
        self.problem = problem
        self.row_cache = row_cache or SharedActionRowCache()
        self.entries = tuple(
            ActionDescriptor(
                action_id=index,
                target_heading_bin=index,
                target_heading_rad=float(problem.grid.heading_rad(index)),
                x_index_offset=int(offset[0]),
                y_index_offset=int(offset[1]),
            )
            for index, offset in enumerate(problem.grid.motion_offsets)
        )
        self.catalog_id = self._catalog_id()

    @property
    def action_count(self) -> int:
        return len(self.entries)

    def _catalog_id(self) -> str:
        payload = {
            "schema": ACTION_CATALOG_SCHEMA,
            "entries": [entry.as_dict() for entry in self.entries],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return f"{ACTION_CATALOG_SCHEMA}:{hashlib.sha256(encoded).hexdigest()}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ACTION_CATALOG_SCHEMA,
            "catalog_id": self.catalog_id,
            "action_count": self.action_count,
            "action_semantics": "target_heading_bin_and_matching_motion_offset",
            "entries": [entry.as_dict() for entry in self.entries],
        }

    def _row_from_edges(
        self, state_id: int, edges: tuple[Any, ...],
    ) -> ActionRow:
        selected = int(state_id)
        mask = np.zeros(self.action_count, dtype=np.bool_)
        local_indices = np.full(self.action_count, -1, dtype=np.int16)
        targets = np.full(self.action_count, -1, dtype=np.int64)
        terminal = np.zeros(self.action_count, dtype=np.bool_)
        for local_index, edge in enumerate(edges):
            action_id = int(edge.target_state.heading_bin)
            if mask[action_id]:
                raise RuntimeError(
                    f"state {selected} has multiple successors for action {action_id}"
                )
            descriptor = self.entries[action_id]
            actual_offset = (
                int(edge.target_state.x_index - edge.source_state.x_index),
                int(edge.target_state.y_index - edge.source_state.y_index),
            )
            expected_offset = (
                descriptor.x_index_offset, descriptor.y_index_offset,
            )
            if actual_offset != expected_offset:
                raise RuntimeError(
                    f"action {action_id} motion offset drifted: "
                    f"{actual_offset} != {expected_offset}"
                )
            mask[action_id] = True
            local_indices[action_id] = int(local_index)
            targets[action_id] = int(edge.target_id)
            terminal[action_id] = self.problem.is_terminal(int(edge.target_id))
        for array in (mask, local_indices, targets, terminal):
            array.setflags(write=False)
        return ActionRow(
            state_id=selected, feasible_mask=mask,
            local_indices=local_indices, target_state_ids=targets,
            terminal=terminal, edges=tuple(edges),
        )

    def _build_row(self, state_id: int) -> ActionRow:
        selected = int(state_id)
        return self._row_from_edges(selected, self.problem.successors(selected))

    def _batch_templates(self) -> dict[str, np.ndarray]:
        cached = self.row_cache.batch_templates
        if cached is not None:
            return cached
        grid = self.problem.grid
        model = self.problem.transition_model
        scale = self.problem.scene.config.physical_scale
        parameters = self.problem.scene.config.glider
        headings = np.asarray([
            grid.heading_rad(index) for index in range(self.action_count)
        ], dtype=float)
        altitude_losses = np.empty(self.action_count, dtype=np.int64)
        distances_m = np.empty(self.action_count, dtype=float)
        altitude_losses_m = np.empty(self.action_count, dtype=float)
        durations_s = np.empty(self.action_count, dtype=float)
        heading_changes = np.empty(
            (self.action_count, self.action_count), dtype=float,
        )
        turn_allowed = np.empty(
            (self.action_count, self.action_count), dtype=np.bool_,
        )
        for target_heading, descriptor in enumerate(self.entries):
            distance_map = grid.horizontal_spacing_map * float(np.hypot(
                descriptor.x_index_offset, descriptor.y_index_offset,
            ))
            distance_m = scale.distance_m(distance_map)
            altitude_loss = model.altitude_loss_bins(distance_map)
            altitude_losses[target_heading] = altitude_loss
            distances_m[target_heading] = distance_m
            altitude_losses_m[target_heading] = scale.distance_m(
                altitude_loss * grid.altitude_spacing_map,
            )
            durations_s[target_heading] = (
                distance_m / parameters.best_glide_speed_mps
            )
            for source_heading in range(self.action_count):
                heading_changes[source_heading, target_heading] = (
                    wrapped_angle_difference(
                        headings[source_heading], headings[target_heading],
                    )
                )
                turn_allowed[source_heading, target_heading] = (
                    model.turn_is_feasible(
                        headings[source_heading], headings[target_heading],
                        distance_m,
                    )
                )
        cached = {
            "altitude_losses": altitude_losses,
            "distances_m": distances_m,
            "altitude_losses_m": altitude_losses_m,
            "durations_s": durations_s,
            "heading_changes": heading_changes,
            "turn_allowed": turn_allowed,
        }
        self.row_cache.batch_templates = cached
        return cached

    def _build_rows_batched(
        self, state_ids: tuple[int, ...],
    ) -> tuple[ActionRow, ...]:
        """Build missing rows by action slabs with authoritative batch geometry."""

        if not state_ids:
            return ()
        grid = self.problem.grid
        graph = self.problem.graph
        terrain = self.problem.scene.terrain
        ids = np.asarray(state_ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= grid.state_count):
            raise ValueError("state_id lies outside this grid")
        headings = ids % grid.heading_bin_count
        remaining = ids // grid.heading_bin_count
        x_indices = remaining % grid.x_count
        remaining //= grid.x_count
        y_indices = remaining % grid.y_count
        altitude_indices = remaining // grid.y_count
        source_states = tuple(
            BellmanState(int(x), int(y), int(altitude), int(heading))
            for x, y, altitude, heading in zip(
                x_indices, y_indices, altitude_indices, headings,
            )
        )
        source_positions = np.column_stack((
            grid.bounds.x_min + x_indices * grid.horizontal_spacing_map,
            grid.bounds.y_min + y_indices * grid.horizontal_spacing_map,
            grid.minimum_altitude_map
            + altitude_indices * grid.altitude_spacing_map,
        ))
        masks = np.zeros((len(ids), self.action_count), dtype=np.bool_)
        local_indices = np.full(
            (len(ids), self.action_count), -1, dtype=np.int16,
        )
        targets = np.full(
            (len(ids), self.action_count), -1, dtype=np.int64,
        )
        terminals = np.zeros((len(ids), self.action_count), dtype=np.bool_)
        edge_lists: list[list[GlideEdge]] = [[] for _ in ids]
        templates = self._batch_templates()
        source_enabled = (
            graph.node_mask[ids] & ~graph.terminal_mask[ids]
        )
        for action_id, descriptor in enumerate(self.entries):
            target_x = x_indices + descriptor.x_index_offset
            target_y = y_indices + descriptor.y_index_offset
            target_altitude = (
                altitude_indices - templates["altitude_losses"][action_id]
            )
            valid = (
                source_enabled
                & templates["turn_allowed"][headings, action_id]
                & (target_x >= 0) & (target_x < grid.x_count)
                & (target_y >= 0) & (target_y < grid.y_count)
                & (target_altitude >= 0)
            )
            candidate_indices = np.flatnonzero(valid)
            if not len(candidate_indices):
                continue
            candidate_targets = (
                ((target_altitude[candidate_indices] * grid.y_count
                  + target_y[candidate_indices]) * grid.x_count
                 + target_x[candidate_indices]) * grid.heading_bin_count
                + action_id
            ).astype(np.int64, copy=False)
            reachable = graph.node_mask[candidate_targets]
            candidate_indices = candidate_indices[reachable]
            candidate_targets = candidate_targets[reachable]
            if not len(candidate_indices):
                continue
            target_positions = np.column_stack((
                grid.bounds.x_min
                + target_x[candidate_indices] * grid.horizontal_spacing_map,
                grid.bounds.y_min
                + target_y[candidate_indices] * grid.horizontal_spacing_map,
                grid.minimum_altitude_map
                + target_altitude[candidate_indices] * grid.altitude_spacing_map,
            ))
            intersects = np.asarray(
                terrain.segments_intersect_solid_many(
                    source_positions[candidate_indices], target_positions,
                ),
                dtype=np.bool_,
            )
            feasible_indices = candidate_indices[~intersects]
            feasible_targets = candidate_targets[~intersects]
            for row_index, target_id in zip(feasible_indices, feasible_targets):
                target_state = BellmanState(
                    int(target_x[row_index]), int(target_y[row_index]),
                    int(target_altitude[row_index]), int(action_id),
                )
                edge = GlideEdge(
                    source_id=int(ids[row_index]), target_id=int(target_id),
                    source_state=source_states[row_index],
                    target_state=target_state,
                    horizontal_distance_m=float(
                        templates["distances_m"][action_id]
                    ),
                    altitude_loss_m=float(
                        templates["altitude_losses_m"][action_id]
                    ),
                    duration_s=float(templates["durations_s"][action_id]),
                    heading_change_rad=float(
                        templates["heading_changes"][
                            headings[row_index], action_id
                        ]
                    ),
                )
                local_index = len(edge_lists[row_index])
                edge_lists[row_index].append(edge)
                masks[row_index, action_id] = True
                local_indices[row_index, action_id] = local_index
                targets[row_index, action_id] = int(target_id)
                terminals[row_index, action_id] = bool(
                    graph.terminal_mask[int(target_id)]
                )
        rows: list[ActionRow] = []
        for index, state_id in enumerate(ids):
            arrays = (
                masks[index], local_indices[index],
                targets[index], terminals[index],
            )
            for array in arrays:
                array.setflags(write=False)
            rows.append(ActionRow(
                state_id=int(state_id), feasible_mask=masks[index],
                local_indices=local_indices[index],
                target_state_ids=targets[index], terminal=terminals[index],
                edges=tuple(edge_lists[index]),
            ))
        return tuple(rows)

    def row(self, state_id: int) -> ActionRow:
        """Return one authoritative structural row, sharing it across sensors."""

        selected = int(state_id)
        row = self.row_cache.get(selected)
        if row is None:
            row = self._build_row(selected)
            self.row_cache.put(row)
        if len(row.feasible_mask) != self.action_count:
            raise RuntimeError("shared action row has an incompatible action count")
        # Hazard and cost rows remain sensor-specific, but their structural edge
        # tuple is sensor-independent and can be adopted without regeneration.
        self.problem.mdp._actions.setdefault(selected, row.edges)
        return row

    def rows_many(self, state_ids: Iterable[int]) -> tuple[ActionRow, ...]:
        ids = tuple(int(state_id) for state_id in state_ids)
        resolved: dict[int, ActionRow] = {}
        missing: list[int] = []
        missing_seen: set[int] = set()
        for state_id in ids:
            if state_id in resolved or state_id in missing_seen:
                continue
            row = self.row_cache.get(state_id)
            if row is None:
                missing.append(state_id)
                missing_seen.add(state_id)
            else:
                resolved[state_id] = row
        if missing:
            built = (
                self._build_rows_batched(tuple(missing))
                if len(missing) >= 8
                else tuple(self._build_row(state_id) for state_id in missing)
            )
            for state_id, row in zip(missing, built):
                self.row_cache.put(row)
                resolved[state_id] = row
        rows = tuple(resolved[state_id] for state_id in ids)
        for row in rows:
            if len(row.feasible_mask) != self.action_count:
                raise RuntimeError("shared action row has an incompatible action count")
            self.problem.mdp._actions.setdefault(row.state_id, row.edges)
        return rows

    def edge_map(self, state_id: int) -> dict[int, int]:
        """Return ``action_id -> Phase 1 local successor index``."""

        row = self.row(int(state_id))
        return {
            int(action_id): int(row.local_indices[action_id])
            for action_id in np.flatnonzero(row.feasible_mask)
        }

    def feasible_mask(self, state_id: int) -> MaskArray:
        return self.row(int(state_id)).feasible_mask.copy()

    def transition(self, state_id: int, action_id: int) -> AttackerTransition:
        action_id = int(action_id)
        if not 0 <= action_id < self.action_count:
            raise ValueError(f"action_id must lie in [0, {self.action_count - 1}]")
        row = self.row(int(state_id))
        local_index = int(row.local_indices[action_id])
        if local_index < 0:
            raise ValueError(
                f"action {action_id} is infeasible at state {int(state_id)}"
            )
        transition = self.problem.transition(int(state_id), local_index)
        if int(transition.edge.target_state.heading_bin) != action_id:
            raise RuntimeError("fixed action mapping drifted from Phase 1 transition")
        return transition

    def audit_reachable_graph(
        self, state_ids: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        all_reachable = np.flatnonzero(self.problem.scene.graph.node_mask)
        if state_ids is None:
            reachable = all_reachable
            audit_scope = "full_reachable_graph"
        else:
            reachable = np.asarray(
                sorted({int(value) for value in state_ids}), dtype=np.int64,
            )
            if len(reachable) and (
                int(reachable[0]) < 0
                or int(reachable[-1]) >= self.problem.grid.state_count
                or not np.all(self.problem.scene.graph.node_mask[reachable])
            ):
                raise ValueError("action-audit state_ids must all be goal-reachable")
            audit_scope = "deterministic_reachable_state_sample"
        transitions = 0
        used: set[int] = set()
        nonterminal_dead_ends: list[int] = []
        minimum = self.action_count
        maximum = 0
        for raw_state_id in reachable:
            state_id = int(raw_state_id)
            mapping = self.edge_map(state_id)
            count = len(mapping)
            transitions += count
            used.update(mapping)
            if not self.problem.is_terminal(state_id):
                if not mapping:
                    nonterminal_dead_ends.append(state_id)
                minimum = min(minimum, count)
                maximum = max(maximum, count)
        return {
            "audit_scope": audit_scope,
            "reachable_states_total": int(len(all_reachable)),
            "reachable_states_checked": int(len(reachable)),
            "transitions_checked": transitions,
            "used_action_ids": len(used),
            "nonterminal_dead_end_state_ids": nonterminal_dead_ends,
            "minimum_nonterminal_feasible_actions": minimum,
            "maximum_nonterminal_feasible_actions": maximum,
            "passed": (
                not nonterminal_dead_ends and len(used) == self.action_count
            ),
        }


__all__ = [
    "ACTION_CATALOG_SCHEMA",
    "ActionDescriptor",
    "ActionRow",
    "HeadingActionCatalog",
    "MaskArray",
    "SharedActionRowCache",
]

"""Periodic heading-state utilities for the turn-limited 3D Bellman model."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """Wrap angles to the half-open interval [-pi, pi)."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def signed_heading_change(
    initial_heading: np.ndarray | float,
    final_heading: np.ndarray | float,
) -> np.ndarray:
    """Return the shortest signed periodic change from initial to final."""
    return wrap_angle(np.asarray(final_heading) - np.asarray(initial_heading))


def nearest_heading_index(heading: float, heading_grid: np.ndarray) -> int:
    """Return the nearest grid index using periodic angular distance."""
    grid = np.asarray(heading_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("heading_grid must be a finite one-dimensional periodic grid")
    return int(np.argmin(np.abs(signed_heading_change(heading, grid))))


def heading_transition_mask(
    heading_grid: np.ndarray,
    max_turn_rate: float,
    transition_duration: float,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Build ``mask[current_heading, selected_course]`` for one transition."""
    grid = np.asarray(heading_grid, dtype=float)
    if max_turn_rate <= 0.0 or transition_duration <= 0.0:
        raise ValueError("max_turn_rate and transition_duration must be positive")
    changes = np.abs(signed_heading_change(grid[:, None], grid[None, :]))
    return changes <= max_turn_rate * transition_duration + tolerance


def powered_segment_heading(
    powered_path: np.ndarray,
    fallback_target: np.ndarray,
) -> float:
    """Infer launch-to-switch azimuth, falling back to the target bearing."""
    path = np.asarray(powered_path, dtype=float)
    target = np.asarray(fallback_target, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
        raise ValueError("powered_path must have shape (n, 3) with n >= 2")
    delta_xy = path[-1, :2] - path[0, :2]
    if np.linalg.norm(delta_xy) <= 1.0e-12:
        delta_xy = target[:2] - path[-1, :2]
    return float(np.arctan2(delta_xy[1], delta_xy[0]))


def heading_change_metrics(
    initial_heading: float,
    selected_headings: np.ndarray,
    transition_duration: float,
) -> dict[str, float]:
    """Summarize periodic heading increments and implied turn rates."""
    headings = np.asarray(selected_headings, dtype=float)
    sequence = np.concatenate(([float(initial_heading)], headings))
    changes = signed_heading_change(sequence[:-1], sequence[1:])
    absolute_changes = np.abs(changes)
    maximum_change = float(np.max(absolute_changes)) if changes.size else 0.0
    return {
        "maximum_heading_change_rad": maximum_change,
        "maximum_heading_change_deg": float(np.rad2deg(maximum_change)),
        "maximum_turn_rate_rad_s": maximum_change / transition_duration,
        "maximum_turn_rate_deg_s": float(np.rad2deg(maximum_change)) / transition_duration,
    }

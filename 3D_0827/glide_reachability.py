"""State-aware bounded-turn glide reachability for the energy prototype."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np

from energy_model import GliderParameters, SwitchingState


@dataclass(frozen=True)
class GlideReachabilityResult:
    """Energy and geometry certificate for one switching state."""

    reachable: bool
    powered_feasible: bool
    minimum_glide_path_m: float
    turn_arc_length_m: float
    straight_length_m: float
    maximum_glide_path_m: float
    available_energy_j: float
    required_energy_j: float
    energy_margin_j: float
    equivalent_height_margin_m: float
    required_height_m: float
    selected_turn: str


@dataclass(frozen=True)
class _HorizontalGlidePath:
    """One turn-then-straight path and its altitude expenditure."""

    turn_arc_length_m: float
    straight_length_m: float
    required_height_m: float
    selected_turn: str

    @property
    def total_length_m(self) -> float:
        return self.turn_arc_length_m + self.straight_length_m


class BoundedTurnGlideModel:
    """Best-glide energy budget with a curvature-bounded horizontal path.

    The switch retains total mechanical energy.  It trims instantaneously to
    the historical best-glide speed; surplus kinetic energy becomes equivalent
    altitude.  Horizontal direction is retained through the initial heading.
    """

    def __init__(self, parameters: GliderParameters) -> None:
        self.parameters = parameters

    def evaluate(
        self,
        switching_state: SwitchingState,
        goal_position_m: np.ndarray,
    ) -> GlideReachabilityResult:
        goal = np.asarray(goal_position_m, dtype=float)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal_position_m must contain three finite values")

        available_energy = switching_state.total_mechanical_energy_j
        if not switching_state.powered_feasible:
            return GlideReachabilityResult(
                reachable=False,
                powered_feasible=False,
                minimum_glide_path_m=float("inf"),
                turn_arc_length_m=float("inf"),
                straight_length_m=float("inf"),
                maximum_glide_path_m=0.0,
                available_energy_j=available_energy,
                required_energy_j=float("inf"),
                energy_margin_j=float("-inf"),
                equivalent_height_margin_m=float("-inf"),
                required_height_m=float("inf"),
                selected_turn="powered-infeasible",
            )

        start_xy = switching_state.position_m[:2]
        goal_xy = goal[:2]
        direct_distance = float(np.linalg.norm(goal_xy - start_xy))
        if direct_distance <= self.parameters.goal_tolerance_m:
            switch_loss_height = 0.0
            path = _HorizontalGlidePath(
                turn_arc_length_m=0.0,
                straight_length_m=0.0,
                required_height_m=0.0,
                selected_turn="inside-goal-tolerance",
            )
        else:
            switch_loss_height = self.parameters.switch_energy_loss_height_m
            path = _minimum_energy_path_to_goal_disk(
                start_xy,
                switching_state.heading_rad,
                goal_xy,
                self.parameters.goal_tolerance_m,
                self.parameters.minimum_turn_radius_m,
                self.parameters.best_glide_ratio,
                self.parameters.turn_glide_ratio,
            )

        mass = self.parameters.mass_kg
        gravity = self.parameters.gravity_mps2
        trim_kinetic = 0.5 * self.parameters.best_glide_speed_mps**2
        available_specific_height = (
            available_energy / (mass * gravity)
            - trim_kinetic / gravity
            - switch_loss_height
            - goal[2]
        )
        maximum_path = max(0.0, available_specific_height) * self.parameters.best_glide_ratio
        required_height = path.required_height_m
        required_energy = mass * (
            trim_kinetic
            + gravity
            * (
                goal[2]
                + switch_loss_height
                + required_height
            )
        )
        margin = available_energy - required_energy
        height_margin = margin / (mass * gravity)
        return GlideReachabilityResult(
            reachable=bool(np.isfinite(path.total_length_m) and margin >= -1.0e-9),
            powered_feasible=True,
            minimum_glide_path_m=path.total_length_m,
            turn_arc_length_m=path.turn_arc_length_m,
            straight_length_m=path.straight_length_m,
            maximum_glide_path_m=maximum_path,
            available_energy_j=available_energy,
            required_energy_j=required_energy,
            energy_margin_j=margin,
            equivalent_height_margin_m=height_margin,
            required_height_m=required_height,
            selected_turn=path.selected_turn,
        )


def _minimum_energy_path_to_goal_disk(
    start_xy: np.ndarray,
    heading_rad: float,
    goal_xy: np.ndarray,
    tolerance_m: float,
    turn_radius_m: float,
    straight_glide_ratio: float,
    turn_glide_ratio: float,
    *,
    boundary_sample_count: int = 12,
) -> _HorizontalGlidePath:
    """Approximate the minimum-altitude bounded-turn path into a goal disk."""
    directions = np.linspace(0.0, 2.0 * pi, boundary_sample_count, endpoint=False)
    targets = [goal_xy]
    targets.extend(
        goal_xy + tolerance_m * np.array([cos(angle), sin(angle)], dtype=float)
        for angle in directions
    )
    best_path = _HorizontalGlidePath(
        turn_arc_length_m=float("inf"),
        straight_length_m=float("inf"),
        required_height_m=float("inf"),
        selected_turn="no-feasible-CS-path",
    )
    for target in targets:
        candidates = _curve_straight_paths(
            start_xy,
            heading_rad,
            np.asarray(target, dtype=float),
            turn_radius_m,
        )
        for arc_length, straight_length, turn in candidates:
            required_height = (
                arc_length / turn_glide_ratio
                + straight_length / straight_glide_ratio
            )
            if required_height < best_path.required_height_m:
                best_path = _HorizontalGlidePath(
                    turn_arc_length_m=arc_length,
                    straight_length_m=straight_length,
                    required_height_m=required_height,
                    selected_turn=turn,
                )
    return best_path


def _curve_straight_paths(
    start_xy: np.ndarray,
    heading_rad: float,
    target_xy: np.ndarray,
    turn_radius_m: float,
) -> tuple[tuple[float, float, str], ...]:
    """Return feasible left/right circular-arc plus straight target legs."""
    forward = np.array([cos(heading_rad), sin(heading_rad)], dtype=float)
    left_normal = np.array([-forward[1], forward[0]], dtype=float)
    paths: list[tuple[float, float, str]] = []

    for turn_sign, turn_name in ((1.0, "left"), (-1.0, "right")):
        center = start_xy + turn_sign * turn_radius_m * left_normal
        center_to_target = target_xy - center
        distance = hypot(float(center_to_target[0]), float(center_to_target[1]))
        if distance < turn_radius_m - 1.0e-9:
            continue
        alpha = atan2(float(center_to_target[1]), float(center_to_target[0]))
        offset = float(np.arccos(np.clip(turn_radius_m / max(distance, turn_radius_m), -1.0, 1.0)))
        start_radius_angle = heading_rad - turn_sign * pi / 2.0

        for tangent_angle in (alpha - offset, alpha + offset):
            tangent = center + turn_radius_m * np.array(
                [cos(tangent_angle), sin(tangent_angle)], dtype=float
            )
            straight_vector = target_xy - tangent
            straight_length = float(np.linalg.norm(straight_vector))
            tangent_heading = tangent_angle + turn_sign * pi / 2.0
            if straight_length > 1.0e-8:
                straight_heading = atan2(float(straight_vector[1]), float(straight_vector[0]))
                alignment = cos(_wrap_angle(straight_heading - tangent_heading))
                if alignment < 1.0 - 1.0e-7:
                    continue
            if turn_sign > 0.0:
                turn_angle = (tangent_angle - start_radius_angle) % (2.0 * pi)
            else:
                turn_angle = (start_radius_angle - tangent_angle) % (2.0 * pi)
            paths.append((turn_radius_m * turn_angle, straight_length, turn_name))
    return tuple(paths)


def _wrap_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi

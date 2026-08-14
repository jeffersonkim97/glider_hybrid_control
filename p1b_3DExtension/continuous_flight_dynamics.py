"""Continuous 3-DOF point-mass dynamics for post-Bellman refinement.

The discrete Bellman solver remains unchanged.  This module supplies a
continuous state model for a downstream trajectory-refinement layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np


STATE_ORDER = ("x", "y", "h", "speed", "gamma", "heading", "bank")
CONTROL_ORDER = ("lift_coefficient", "roll_rate")


def powered_switch_state(
    launch_position: np.ndarray,
    powered_speed: float,
    powered_time: float,
    flight_path_angle: float,
    heading: float,
) -> np.ndarray:
    """Return the continuous state reached by straight powered flight."""
    launch = np.asarray(launch_position, dtype=float)
    scalars = np.asarray(
        [powered_speed, powered_time, flight_path_angle, heading], dtype=float,
    )
    if launch.shape != (3,) or not np.all(np.isfinite(launch)):
        raise ValueError("launch_position must contain three finite values")
    if not np.all(np.isfinite(scalars)):
        raise ValueError("powered switching parameters must be finite")
    if powered_speed <= 0.0 or powered_time <= 0.0:
        raise ValueError("powered speed and time must be positive")
    horizontal_speed = powered_speed * np.cos(flight_path_angle)
    position = launch + powered_time * np.array([
        horizontal_speed * np.cos(heading),
        horizontal_speed * np.sin(heading),
        powered_speed * np.sin(flight_path_angle),
    ])
    return np.array([
        position[0], position[1], position[2], powered_speed,
        flight_path_angle, heading, 0.0,
    ])


def aerodynamic_forces(
    speed: np.ndarray | float,
    lift_coefficient: np.ndarray | float,
    vehicle: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lift, drag, and drag coefficient from the configured polar."""
    speed_array = np.asarray(speed, dtype=float)
    cl = np.asarray(lift_coefficient, dtype=float)
    cd = (
        float(vehicle["cd0"])
        + float(vehicle["linear_drag_coefficient"]) * cl
        + float(vehicle["quadratic_drag_coefficient"]) * cl**2
    )
    dynamic_pressure = 0.5 * float(vehicle["air_density"]) * speed_array**2
    lift = dynamic_pressure * float(vehicle["wing_area"]) * cl
    drag = dynamic_pressure * float(vehicle["wing_area"]) * cd
    return lift, drag, cd


def point_mass_rhs_numpy(
    state: np.ndarray,
    control: np.ndarray,
    vehicle: dict[str, Any],
) -> np.ndarray:
    """Evaluate engine-off 3-DOF dynamics with bank-angle state."""
    values = np.asarray(state, dtype=float)
    inputs = np.asarray(control, dtype=float)
    if values.shape != (7,) or inputs.shape != (2,):
        raise ValueError("state/control must have shapes (7,) and (2,)")
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(inputs))):
        raise ValueError("state/control values must be finite")
    _, _, _, speed, gamma, heading, bank = values
    if speed <= 0.0 or abs(np.cos(gamma)) <= 1.0e-6:
        raise ValueError("continuous dynamics require positive speed and cos(gamma)")
    lift_coefficient, roll_rate = inputs
    lift, drag, _ = aerodynamic_forces(speed, lift_coefficient, vehicle)
    mass = float(vehicle["mass"])
    gravity = float(vehicle["gravity"])
    horizontal_speed = speed * np.cos(gamma)
    return np.array([
        horizontal_speed * np.cos(heading),
        horizontal_speed * np.sin(heading),
        speed * np.sin(gamma),
        -drag / mass - gravity * np.sin(gamma),
        (lift * np.cos(bank) - mass * gravity * np.cos(gamma)) / (mass * speed),
        lift * np.sin(bank) / (mass * speed * np.cos(gamma)),
        roll_rate,
    ], dtype=float)


def rk4_step_numpy(
    state: np.ndarray,
    control: np.ndarray,
    step_seconds: float,
    vehicle: dict[str, Any],
) -> np.ndarray:
    """Advance one fixed-control interval with classical RK4."""
    if not np.isfinite(step_seconds) or step_seconds <= 0.0:
        raise ValueError("step_seconds must be finite and positive")
    k1 = point_mass_rhs_numpy(state, control, vehicle)
    k2 = point_mass_rhs_numpy(state + 0.5 * step_seconds * k1, control, vehicle)
    k3 = point_mass_rhs_numpy(state + 0.5 * step_seconds * k2, control, vehicle)
    k4 = point_mass_rhs_numpy(state + step_seconds * k3, control, vehicle)
    return np.asarray(state, dtype=float) + step_seconds * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    ) / 6.0

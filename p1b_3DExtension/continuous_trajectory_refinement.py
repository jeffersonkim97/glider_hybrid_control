"""Continuous 3-DOF refinement of a validated discrete 3D route.

The existing six-to-three-dimensional projection is neither imported nor
modified.  A saved physical-successor trajectory supplies only a geometric
initial guess for this downstream direct-multiple-shooting solve.
"""

from __future__ import annotations

import json
import itertools
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np

from .continuous_flight_dynamics import powered_switch_state
from .detection import build_symbolic_detection_bundle
from .experiment_extreme_ridge_fine import build_fine_configuration
from .geometry import build_geometry_bundle, terrain_height
from .phase_logging import close_phase_logger


REPO_ROOT = Path(__file__).resolve().parent.parent
DISCRETE_RESULT_DIR = REPO_ROOT / "results" / "extreme_ridge_275_fine"
OUTPUT_DIR = REPO_ROOT / "results" / "extreme_ridge_275_continuous"
INTERVAL_COUNT = 50
STATE_ORDER = ("x", "y", "h", "speed", "gamma", "heading", "bank", "hazard")
CONTROL_ORDER = ("lift_coefficient", "roll_rate")
_CASADI_NAME_COUNTER = itertools.count()


def _last_output(value: Any):
    values = value if isinstance(value, tuple) else (value,)
    return values[-1]


def _rhs_expression(
    state: ca.MX,
    control: ca.MX,
    vehicle: dict[str, Any],
    glide_detection_function: ca.Function,
    sensor_position: np.ndarray,
    detection_hazard_scale: float = 1.0,
) -> ca.MX:
    x, y, h, speed, gamma, heading, bank, _ = ca.vertsplit(state)
    lift_coefficient, roll_rate = ca.vertsplit(control)
    density = float(vehicle["air_density"])
    wing_area = float(vehicle["wing_area"])
    mass = float(vehicle["mass"])
    gravity = float(vehicle["gravity"])
    cd = (
        float(vehicle["cd0"])
        + float(vehicle["linear_drag_coefficient"]) * lift_coefficient
        + float(vehicle["quadratic_drag_coefficient"]) * lift_coefficient**2
    )
    dynamic_pressure = 0.5 * density * speed**2
    lift = dynamic_pressure * wing_area * lift_coefficient
    drag = dynamic_pressure * wing_area * cd
    horizontal_speed = speed * ca.cos(gamma)
    # A hazard rate is physically nonnegative.  The symbolic terrain/LOS
    # interpolants can otherwise produce roundoff-scale negative values at a
    # spline boundary, which makes cumulative hazard decrease spuriously.
    detection_rate = float(detection_hazard_scale) * ca.fmax(
        0.0, _last_output(glide_detection_function(
        x, y, h, speed, gamma, heading,
        float(sensor_position[0]), float(sensor_position[1]),
        float(sensor_position[2]),
    )))
    return ca.vertcat(
        horizontal_speed * ca.cos(heading),
        horizontal_speed * ca.sin(heading),
        speed * ca.sin(gamma),
        -drag / mass - gravity * ca.sin(gamma),
        (lift * ca.cos(bank) - mass * gravity * ca.cos(gamma)) / (mass * speed),
        lift * ca.sin(bank) / (mass * speed * ca.cos(gamma)),
        roll_rate,
        detection_rate,
    )


def _rk4_expression(
    state: ca.MX,
    control: ca.MX,
    step: ca.MX,
    rhs_function: ca.Function,
) -> ca.MX:
    k1 = rhs_function(state, control)
    k2 = rhs_function(state + 0.5 * step * k1, control)
    k3 = rhs_function(state + 0.5 * step * k2, control)
    k4 = rhs_function(state + step * k3, control)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    coordinate = np.concatenate(([0.0], np.cumsum(distances)))
    query = np.linspace(0.0, coordinate[-1], count)
    return np.column_stack([
        np.interp(query, coordinate, points[:, dimension])
        for dimension in range(3)
    ])


def _initial_guess(
    discrete: dict[str, np.ndarray],
    launch_height: float,
    powered_speed: float,
    interval_count: int,
    initial_gamma_deg: float = 8.0,
    initial_topology: str = "south",
    initial_powered_time_s: float = 52.0,
    glide_time_bounds_s: tuple[float, float] = (55.0, 175.0),
) -> dict[str, np.ndarray | float]:
    if initial_topology not in {"south", "center", "north"}:
        raise ValueError("initial_topology must be 'south', 'center', or 'north'")
    old_switch = np.asarray(discrete["switching_point"], dtype=float).copy()
    old_glide = np.asarray(discrete["trajectory"], dtype=float).copy()
    if initial_topology == "north":
        old_switch[1] *= -1.0
        old_glide[:, 1] *= -1.0
    elif initial_topology == "center":
        old_switch[1] = 0.0
    powered_heading = float(np.arctan2(old_switch[1], old_switch[0]))
    powered_gamma = float(np.deg2rad(initial_gamma_deg))
    powered_time = float(initial_powered_time_s)
    switch = powered_switch_state(
        np.array([0.0, 0.0, launch_height]), powered_speed,
        powered_time, powered_gamma, powered_heading,
    )
    retained = old_glide[old_glide[:, 0] >= 1375.0]
    polyline = np.vstack((switch[:3], retained, discrete["goal_position"]))
    positions = _resample_polyline(polyline, interval_count + 1)
    path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    glide_time = float(np.clip(
        path_length / 17.0,
        float(glide_time_bounds_s[0]),
        float(glide_time_bounds_s[1]),
    ))
    step = glide_time / interval_count
    velocity = np.gradient(positions, step, axis=0)
    horizontal = np.linalg.norm(velocity[:, :2], axis=1)
    speed = np.clip(np.linalg.norm(velocity, axis=1), 10.5, 22.0)
    gamma = np.arctan2(velocity[:, 2], horizontal)
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    speed[0] = powered_speed
    gamma[0] = powered_gamma
    heading[0] = powered_heading
    heading_rate = np.gradient(heading, step)
    bank = np.clip(
        np.arctan(heading_rate * speed * np.cos(gamma) / 9.81),
        np.deg2rad(-25.0), np.deg2rad(25.0),
    )
    bank[0] = 0.0
    gamma_rate = np.gradient(gamma, step)
    mass = 9.34 / 9.81
    required_lift = mass * (
        speed * gamma_rate + 9.81 * np.cos(gamma)
    ) / np.maximum(np.cos(bank), 0.8)
    cl = required_lift[:-1] / (
        0.5 * 1.225 * speed[:-1]**2 * 0.321
    )
    cl = np.clip(cl, 0.06, 0.48)
    roll_rate = np.clip(
        np.diff(bank) / step,
        np.deg2rad(-14.0), np.deg2rad(14.0),
    )
    states = np.zeros((8, interval_count + 1))
    states[:3] = positions.T
    states[3] = speed
    states[4] = gamma
    states[5] = heading
    states[6] = bank
    states[7] = np.linspace(5.0e-4, 1.5e-3, interval_count + 1)
    controls = np.vstack((cl, roll_rate))
    return {
        "powered_time": powered_time,
        "powered_gamma": powered_gamma,
        "powered_heading": powered_heading,
        "glide_time": glide_time,
        "states": states,
        "controls": controls,
    }


def _continuous_solution_initial_guess(
    source_dir: Path,
    topology: str,
    interval_count: int,
    rhs_function: ca.Function,
    powered_detection_function: ca.Function,
    launch: np.ndarray,
    powered_speed: float,
    sensor: np.ndarray,
    detection_hazard_scale: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Build a dynamically credible south/north seed from a saved solution."""
    if topology not in {"south", "north"}:
        raise ValueError("continuous-solution initialization supports south/north")
    with (source_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with np.load(source_dir / "trajectory_data.npz") as handle:
        states = np.asarray(handle["shooting_states"], dtype=float).copy()
        controls = np.asarray(handle["controls"], dtype=float).copy()
    if controls.shape[1] != interval_count:
        raise ValueError("continuous warm-start interval count mismatch")
    powered_heading = float(np.deg2rad(summary["powered_heading_deg"]))
    if topology == "north":
        states[1] *= -1.0
        states[5] *= -1.0
        states[6] *= -1.0
        controls[1] *= -1.0
        powered_heading *= -1.0
    powered_time = float(summary["powered_time_s"])
    powered_gamma = float(np.deg2rad(summary["powered_gamma_deg"]))
    glide_time = float(summary["glide_time_s"])

    fractions = np.linspace(0.0, 1.0, 501)
    switch = powered_switch_state(
        launch, powered_speed, powered_time, powered_gamma, powered_heading,
    )
    powered_points = launch[None, :] + fractions[:, None] * (
        switch[:3] - launch
    )[None, :]
    count = fractions.size
    outputs = powered_detection_function.map(count)(
        powered_points[:, 0].reshape(1, -1),
        powered_points[:, 1].reshape(1, -1),
        powered_points[:, 2].reshape(1, -1),
        np.full((1, count), powered_speed),
        np.full((1, count), powered_gamma),
        np.full((1, count), powered_heading),
        np.full((1, count), sensor[0]),
        np.full((1, count), sensor[1]),
        np.full((1, count), sensor[2]),
    )
    output_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
    rates = float(detection_hazard_scale) * np.maximum(
        0.0, np.asarray(output_tuple[-1], dtype=float).reshape(-1)
    )
    states[7, 0] = float(np.trapezoid(rates, fractions * powered_time))

    # Recompute the hazard state with the transformed spatial/dynamic seed.
    # Geometry states remain the saved dynamically feasible trajectory.
    interval_step = glide_time / interval_count
    internal_step = interval_step / 4.0
    for interval in range(interval_count):
        propagated = states[:, interval].copy()
        for _ in range(4):
            k1 = np.asarray(rhs_function(propagated, controls[:, interval])).reshape(-1)
            k2 = np.asarray(rhs_function(
                propagated + 0.5 * internal_step * k1, controls[:, interval],
            )).reshape(-1)
            k3 = np.asarray(rhs_function(
                propagated + 0.5 * internal_step * k2, controls[:, interval],
            )).reshape(-1)
            k4 = np.asarray(rhs_function(
                propagated + internal_step * k3, controls[:, interval],
            )).reshape(-1)
            propagated += internal_step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        states[7, interval + 1] = propagated[7]
    return {
        "powered_time": powered_time,
        "powered_gamma": powered_gamma,
        "powered_heading": powered_heading,
        "glide_time": glide_time,
        "states": states,
        "controls": controls,
    }


def _result_mapping_initial_guess(
    source: dict[str, Any],
    interval_count: int,
    rhs_function: ca.Function,
    powered_detection_function: ca.Function,
    launch: np.ndarray,
    powered_speed: float,
    sensor: np.ndarray,
    detection_hazard_scale: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Retarget a solved trajectory to a new sensor by reintegrating hazard."""
    states = np.asarray(source["states"], dtype=float).copy()
    controls = np.asarray(source["controls"], dtype=float).copy()
    if states.shape != (8, interval_count + 1):
        raise ValueError("initial_result state shape does not match interval_count")
    if controls.shape != (2, interval_count):
        raise ValueError("initial_result control shape does not match interval_count")
    powered_time = float(source["powered_time"])
    powered_gamma = float(source["powered_gamma"])
    powered_heading = float(source["powered_heading"])
    glide_time = float(source["glide_time"])
    switch = powered_switch_state(
        launch, powered_speed, powered_time, powered_gamma, powered_heading,
    )
    states[:7, 0] = switch

    fractions = np.linspace(0.0, 1.0, 501)
    powered_points = launch[None, :] + fractions[:, None] * (
        switch[:3] - launch
    )[None, :]
    count = fractions.size
    outputs = powered_detection_function.map(count)(
        powered_points[:, 0].reshape(1, -1),
        powered_points[:, 1].reshape(1, -1),
        powered_points[:, 2].reshape(1, -1),
        np.full((1, count), powered_speed),
        np.full((1, count), powered_gamma),
        np.full((1, count), powered_heading),
        np.full((1, count), sensor[0]),
        np.full((1, count), sensor[1]),
        np.full((1, count), sensor[2]),
    )
    output_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
    rates = float(detection_hazard_scale) * np.maximum(
        0.0, np.asarray(output_tuple[-1], dtype=float).reshape(-1)
    )
    states[7, 0] = float(np.trapezoid(rates, fractions * powered_time))

    interval_step = glide_time / interval_count
    internal_step = interval_step / 4.0
    for interval in range(interval_count):
        propagated = states[:, interval].copy()
        for _ in range(4):
            k1 = np.asarray(rhs_function(
                propagated, controls[:, interval],
            )).reshape(-1)
            k2 = np.asarray(rhs_function(
                propagated + 0.5 * internal_step * k1,
                controls[:, interval],
            )).reshape(-1)
            k3 = np.asarray(rhs_function(
                propagated + 0.5 * internal_step * k2,
                controls[:, interval],
            )).reshape(-1)
            k4 = np.asarray(rhs_function(
                propagated + internal_step * k3,
                controls[:, interval],
            )).reshape(-1)
            propagated += internal_step * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            ) / 6.0
        states[7, interval + 1] = propagated[7]
    return {
        "powered_time": powered_time,
        "powered_gamma": powered_gamma,
        "powered_heading": powered_heading,
        "glide_time": glide_time,
        "states": states,
        "controls": controls,
    }


def solve_continuous_refinement(
    interval_count: int = INTERVAL_COUNT,
    initial_gamma_deg: float = 8.0,
    initial_topology: str = "south",
    maximum_cpu_time_s: float | None = None,
    discrete_result_dir: Path | None = None,
    initial_powered_time_s: float = 52.0,
    initialization_source: str = "discrete",
    continuous_warm_start_dir: Path | None = None,
    accept_limited_solution: bool = False,
    sensor_xy: tuple[float, float] | None = None,
    maximum_iterations: int = 3000,
    initial_result: dict[str, Any] | None = None,
    nlp_speed_buffer_m_s: float = 0.0,
    detection_hazard_scale: float = 1.0,
    use_limited_memory_hessian: bool = False,
    configuration_bundle: dict[str, Any] | None = None,
    powered_time_bounds_s: tuple[float, float] = (25.0, 80.0),
    glide_time_bounds_s: tuple[float, float] = (55.0, 175.0),
    powered_clearance_factor: float = 1.05,
    integration_substeps_per_interval: int = 4,
    constrain_switch_to_los_boundary: bool = False,
) -> dict[str, Any]:
    """Solve one local refinement from a selected mesh and geometric seed."""
    source_result_dir = (
        DISCRETE_RESULT_DIR
        if discrete_result_dir is None else Path(discrete_result_dir)
    )
    if not 0.0 < float(detection_hazard_scale) <= 1.0:
        raise ValueError("detection_hazard_scale must be in (0, 1]")
    with np.load(source_result_dir / "trajectory_data.npz") as handle:
        discrete = {name: np.asarray(handle[name]) for name in handle.files}
    with (source_result_dir / "summary.json").open(encoding="utf-8") as handle:
        discrete_summary = json.load(handle)

    configuration = (
        build_fine_configuration()
        if configuration_bundle is None
        else deepcopy(configuration_bundle)
    )
    if sensor_xy is not None:
        sensor_config = configuration["primary_result"]["sensor_config"]
        sensor_config["default_x_sensor"] = float(sensor_xy[0])
        sensor_config["default_y_sensor"] = float(sensor_xy[1])
    logger = configuration["primary_result"]["logging_utilities"]["logger"]
    started = time.perf_counter()
    try:
        geometry_bundle = build_geometry_bundle(configuration)
        detection_bundle = build_symbolic_detection_bundle(
            configuration, geometry_bundle,
        )
        configs = configuration["primary_result"]
        environment = configs["environment_config"]
        vehicle = configs["vehicle_config"]
        cost = configs["cost_config"]["attacker"]
        geometry = geometry_bundle["primary_result"]
        sensor = np.asarray(geometry["sensor_position"], dtype=float)
        goal = np.asarray(geometry["goal_position"], dtype=float)
        terrain_model = geometry["terrain_model"]
        launch_height = float(terrain_height(terrain_model, 0.0, 0.0))
        launch = np.array([0.0, 0.0, launch_height])
        powered_speed = float(vehicle["powered_speed"])
        detection_functions = detection_bundle["primary_result"]["functions"]
        visibility_handling = configs["bellman_config"]["search_options"].get(
            "powered_visibility_handling", "hard_hidden",
        )

        terrain_arrays = geometry["terrain_arrays"]
        name_suffix = next(_CASADI_NAME_COUNTER)
        terrain_function = ca.interpolant(
            f"continuous_refinement_terrain_{name_suffix}", "bspline",
            [np.asarray(terrain_arrays["x"]), np.asarray(terrain_arrays["y"])],
            np.asarray(terrain_arrays["height"]).ravel(order="F"),
        )
        los_boundary_function = ca.interpolant(
            f"continuous_refinement_los_boundary_{name_suffix}", "linear",
            [np.asarray(terrain_arrays["x"]), np.asarray(terrain_arrays["y"])],
            np.asarray(
                geometry["los_masks"]["los_boundary_height"], dtype=float,
            ).ravel(order="F"),
        )
        symbolic_state = ca.MX.sym("continuous_state", 8)
        symbolic_control = ca.MX.sym("continuous_control", 2)
        rhs_function = ca.Function(
            f"continuous_glide_rhs_{name_suffix}",
            [symbolic_state, symbolic_control],
            [_rhs_expression(
                symbolic_state, symbolic_control, vehicle,
                detection_functions["glide_detection_components"], sensor,
                detection_hazard_scale,
            )],
        )

        opti = ca.Opti()
        states = opti.variable(8, interval_count + 1)
        controls = opti.variable(2, interval_count)
        powered_time = opti.variable()
        powered_gamma = opti.variable()
        powered_heading = opti.variable()
        glide_time = opti.variable()
        step = glide_time / interval_count

        powered_time_lower, powered_time_upper = (
            float(value) for value in powered_time_bounds_s
        )
        glide_time_lower, glide_time_upper = (
            float(value) for value in glide_time_bounds_s
        )
        if not 0.0 < powered_time_lower < powered_time_upper:
            raise ValueError("powered_time_bounds_s must be positive and ordered")
        if not 0.0 < glide_time_lower < glide_time_upper:
            raise ValueError("glide_time_bounds_s must be positive and ordered")
        opti.subject_to(opti.bounded(
            powered_time_lower, powered_time, powered_time_upper,
        ))
        opti.subject_to(opti.bounded(
            np.deg2rad(3.0), powered_gamma, np.deg2rad(18.0),
        ))
        opti.subject_to(opti.bounded(
            np.deg2rad(-42.0), powered_heading, np.deg2rad(42.0),
        ))
        opti.subject_to(opti.bounded(
            glide_time_lower, glide_time, glide_time_upper,
        ))

        powered_horizontal_speed = powered_speed * ca.cos(powered_gamma)
        switch_x = launch[0] + powered_time * powered_horizontal_speed * ca.cos(powered_heading)
        switch_y = launch[1] + powered_time * powered_horizontal_speed * ca.sin(powered_heading)
        switch_h = launch[2] + powered_time * powered_speed * ca.sin(powered_gamma)
        if constrain_switch_to_los_boundary:
            # Preserve p1b_4D's switching rule under the dimensional
            # extension: the scalar LOS boundary line becomes h=H_LOS(x,y).
            tangent_tolerance = float(
                vehicle["switching_constraints"]["tangent_tolerance"]
            )
            opti.subject_to(opti.bounded(
                -tangent_tolerance,
                switch_h
                - los_boundary_function(ca.vertcat(switch_x, switch_y)),
                tangent_tolerance,
            ))
        # A positive switching gamma necessarily carries upward velocity
        # into the engine-off phase.  Preserve a small ceiling buffer so the
        # continuous post-switch apex, not only the switching node, remains
        # inside the 200 m airspace.
        opti.subject_to(switch_h <= environment["airspace"]["h_max"] - 2.0)
        powered_hazard = 0.0
        if float(powered_clearance_factor) < 1.0:
            raise ValueError("powered_clearance_factor must be at least 1.0")
        powered_quadrature = np.linspace(0.0, 1.0, 101)
        powered_rates = []
        for fraction in powered_quadrature:
            px = launch[0] + fraction * (switch_x - launch[0])
            py = launch[1] + fraction * (switch_y - launch[1])
            ph = launch[2] + fraction * (switch_h - launch[2])
            if fraction > 0.0:
                opti.subject_to(opti.bounded(
                    environment["airspace"]["x_min"], px,
                    environment["airspace"]["x_max"],
                ))
                opti.subject_to(opti.bounded(
                    environment["airspace"]["y_min"], py,
                    environment["airspace"]["y_max"],
                ))
                opti.subject_to(ph <= environment["airspace"]["h_max"])
                required_margin = float(powered_clearance_factor) * float(
                    vehicle["switching_constraints"]["terrain_clearance"]
                ) * fraction
                opti.subject_to(
                    ph >= terrain_function(ca.vertcat(px, py)) + required_margin
                )
                if visibility_handling == "hard_hidden":
                    los_visible = detection_functions["los"](
                        px, py, ph, sensor[0], sensor[1], sensor[2],
                    )[0]
                    opti.subject_to(los_visible <= 0.05)
            if visibility_handling == "hazard_penalty":
                powered_rate = _last_output(
                    detection_functions["powered_total_detection_components"](
                        px, py, ph, powered_speed, powered_gamma,
                        powered_heading, sensor[0], sensor[1], sensor[2],
                    )
                )
            elif visibility_handling == "hard_hidden":
                powered_rate = _last_output(
                    detection_functions["powered_detection_components"](
                        px, py, ph, powered_speed,
                        sensor[0], sensor[1], sensor[2],
                    )
                )
            else:
                raise ValueError(
                    "powered_visibility_handling must be 'hazard_penalty' or 'hard_hidden'"
                )
            powered_rates.append(
                float(detection_hazard_scale) * ca.fmax(0.0, powered_rate)
            )
        powered_hazard = powered_time * sum(
            (0.5 if index in (0, len(powered_rates) - 1) else 1.0) * rate
            for index, rate in enumerate(powered_rates)
        ) / (len(powered_rates) - 1)

        switch_state = ca.vertcat(
            switch_x, switch_y, switch_h, powered_speed,
            powered_gamma, powered_heading, 0.0, powered_hazard,
        )
        opti.subject_to(states[:, 0] == switch_state)

        opti.subject_to(opti.bounded(0.05, controls[0, :], 0.50))
        opti.subject_to(opti.bounded(
            np.deg2rad(-18.0), controls[1, :], np.deg2rad(18.0),
        ))
        nlp_maximum_speed = 22.6 - float(nlp_speed_buffer_m_s)
        if not 22.0 <= nlp_maximum_speed <= 22.6:
            raise ValueError("nlp_speed_buffer_m_s must be between 0.0 and 0.6")
        opti.subject_to(opti.bounded(10.0, states[3, :], nlp_maximum_speed))
        opti.subject_to(opti.bounded(
            np.deg2rad(-50.0), states[4, :], np.deg2rad(22.0),
        ))
        opti.subject_to(opti.bounded(
            np.deg2rad(-80.0), states[5, :], np.deg2rad(80.0),
        ))
        opti.subject_to(opti.bounded(
            np.deg2rad(-30.0), states[6, :], np.deg2rad(30.0),
        ))
        opti.subject_to(opti.bounded(0.0, states[7, :], 1.0))
        opti.subject_to(opti.bounded(
            environment["airspace"]["x_min"], states[0, :],
            environment["airspace"]["x_max"],
        ))
        opti.subject_to(opti.bounded(
            environment["airspace"]["y_min"], states[1, :],
            environment["airspace"]["y_max"],
        ))
        opti.subject_to(opti.bounded(
            environment["airspace"]["h_min"], states[2, :],
            environment["airspace"]["h_max"],
        ))

        clearance = float(vehicle["switching_constraints"]["terrain_clearance"])
        nlp_clearance = clearance + 0.05
        for index in range(interval_count + 1):
            opti.subject_to(
                states[2, index]
                >= terrain_function(states[:2, index]) + nlp_clearance
            )
        integration_substeps = int(integration_substeps_per_interval)
        if integration_substeps < 1:
            raise ValueError("integration_substeps_per_interval must be positive")
        for index in range(interval_count):
            predicted = states[:, index]
            internal_step = step / integration_substeps
            for _ in range(integration_substeps):
                predicted = _rk4_expression(
                    predicted, controls[:, index], internal_step, rhs_function,
                )
                opti.subject_to(
                    predicted[2]
                    >= terrain_function(predicted[:2]) + nlp_clearance
                )
                opti.subject_to(predicted[2] <= environment["airspace"]["h_max"])
                opti.subject_to(opti.bounded(
                    10.0, predicted[3], nlp_maximum_speed,
                ))
            opti.subject_to(states[:, index + 1] == predicted)

        terminal_error = states[:3, -1] - goal
        # Keep the shooting endpoint slightly inside the 15 m terminal sphere
        # so independent dense RK4 propagation also remains feasible.
        opti.subject_to(ca.sumsqr(terminal_error) <= 14.99**2)
        hazard_reference = float(cost["normalization"]["pod"]["hazard_reference"])
        reference_time = float(cost["normalization"]["time"]["reference_seconds"])
        physical_objective = (
            float(cost["w_pod"]) * states[7, -1] / hazard_reference
            + float(cost["w_time"]) * (powered_time + glide_time) / reference_time
        )
        smoothing = 1.0e-5 * ca.sumsqr(controls[:, 1:] - controls[:, :-1])
        opti.minimize(physical_objective + smoothing)

        if initialization_source == "discrete":
            initial = _initial_guess(
                discrete, launch_height, powered_speed, interval_count,
                initial_gamma_deg=initial_gamma_deg,
                initial_topology=initial_topology,
                initial_powered_time_s=initial_powered_time_s,
                glide_time_bounds_s=glide_time_bounds_s,
            )
        elif initialization_source == "continuous_solution":
            warm_start_dir = (
                OUTPUT_DIR
                if continuous_warm_start_dir is None
                else Path(continuous_warm_start_dir)
            )
            initial = _continuous_solution_initial_guess(
                warm_start_dir, initial_topology, interval_count,
                rhs_function,
                detection_functions["powered_total_detection_components"],
                launch, powered_speed, sensor,
                detection_hazard_scale,
            )
        elif initialization_source == "result_mapping":
            if initial_result is None:
                raise ValueError(
                    "initial_result is required for result_mapping initialization"
                )
            initial = _result_mapping_initial_guess(
                initial_result, interval_count, rhs_function,
                detection_functions["powered_total_detection_components"],
                launch, powered_speed, sensor,
                detection_hazard_scale,
            )
        else:
            raise ValueError(
                "initialization_source must be 'discrete', "
                "'continuous_solution', or 'result_mapping'"
            )
        opti.set_initial(powered_time, initial["powered_time"])
        opti.set_initial(powered_gamma, initial["powered_gamma"])
        opti.set_initial(powered_heading, initial["powered_heading"])
        opti.set_initial(glide_time, initial["glide_time"])
        opti.set_initial(states, initial["states"])
        opti.set_initial(controls, initial["controls"])
        ipopt_options = {
            "print_level": 0,
            "sb": "yes",
            "max_iter": int(maximum_iterations),
            "tol": 1.0e-7,
            "constr_viol_tol": 1.0e-9,
            "acceptable_tol": 1.0e-4,
            # Dense validation requires switch continuity <= 1e-7.  IPOPT's
            # acceptable termination must therefore be stricter than that
            # invariant, otherwise it can legitimately stop at a solution
            # that the independent validator rejects.
            "acceptable_constr_viol_tol": 5.0e-8,
            "acceptable_dual_inf_tol": 1.0e-3,
            "acceptable_compl_inf_tol": 1.0e-4,
            "acceptable_iter": 5,
            "linear_solver": "mumps",
        }
        if maximum_cpu_time_s is not None:
            ipopt_options["max_cpu_time"] = float(maximum_cpu_time_s)
        if use_limited_memory_hessian:
            ipopt_options.update({
                "hessian_approximation": "limited-memory",
                "limited_memory_update_type": "bfgs",
                "mu_strategy": "adaptive",
            })
        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            ipopt_options,
        )
        solution = (
            opti.solve_limited() if accept_limited_solution else opti.solve()
        )
        solved_states = np.asarray(solution.value(states), dtype=float)
        solved_controls = np.asarray(solution.value(controls), dtype=float)
        values = {
            "powered_time": float(solution.value(powered_time)),
            "powered_gamma": float(solution.value(powered_gamma)),
            "powered_heading": float(solution.value(powered_heading)),
            "glide_time": float(solution.value(glide_time)),
            "physical_objective": float(solution.value(physical_objective)),
            "solver_objective": float(solution.value(physical_objective + smoothing)),
        }
        values["switch_state"] = powered_switch_state(
            launch, powered_speed, values["powered_time"],
            values["powered_gamma"], values["powered_heading"],
        )
        values["states"] = solved_states
        values["controls"] = solved_controls
        values["rhs_function"] = rhs_function
        values["los_function"] = detection_functions["los"]
        values["los_boundary_function"] = los_boundary_function
        values["constrain_switch_to_los_boundary"] = bool(
            constrain_switch_to_los_boundary
        )
        values["goal"] = goal
        values["launch"] = launch
        values["sensor"] = sensor
        values["requested_sensor_xy"] = (
            None if sensor_xy is None else tuple(float(value) for value in sensor_xy)
        )
        values["nlp_speed_buffer_m_s"] = float(nlp_speed_buffer_m_s)
        values["detection_hazard_scale"] = float(detection_hazard_scale)
        values["use_limited_memory_hessian"] = bool(use_limited_memory_hessian)
        values["powered_time_bounds_s"] = tuple(powered_time_bounds_s)
        values["glide_time_bounds_s"] = tuple(glide_time_bounds_s)
        values["powered_clearance_factor"] = float(powered_clearance_factor)
        values["integration_substeps_per_interval"] = integration_substeps
        values["terrain_model"] = terrain_model
        values["terrain_arrays"] = terrain_arrays
        values["coverage"] = geometry["coverage"]
        values["configuration"] = configuration
        values["discrete"] = discrete
        values["discrete_summary"] = discrete_summary
        values["solver_stats"] = solution.stats()
        values["elapsed_seconds"] = time.perf_counter() - started
        values["initial_gamma_deg"] = float(initial_gamma_deg)
        values["initial_topology"] = initial_topology
        values["initial_powered_time_s"] = float(initial_powered_time_s)
        values["initialization_source"] = initialization_source
        values["discrete_result_dir"] = source_result_dir
        values["powered_visibility_handling"] = visibility_handling
        values["powered_total_detection_function"] = detection_functions[
            "powered_total_detection_components"
        ]
        return values
    finally:
        close_phase_logger(logger)


def _dense_validate(result: dict[str, Any], substeps: int = 20) -> dict[str, Any]:
    states = result["states"]
    controls = result["controls"]
    interval_count = controls.shape[1]
    interval_step = result["glide_time"] / interval_count
    dense_step = interval_step / substeps
    rhs = result["rhs_function"]
    dense_states = [states[:, 0].copy()]
    dense_times = [result["powered_time"]]
    propagated = states[:, 0].copy()
    endpoint_residuals = []
    for interval in range(interval_count):
        control = controls[:, interval]
        for _ in range(substeps):
            k1 = np.asarray(rhs(propagated, control), dtype=float).reshape(-1)
            k2 = np.asarray(rhs(propagated + 0.5 * dense_step * k1, control), dtype=float).reshape(-1)
            k3 = np.asarray(rhs(propagated + 0.5 * dense_step * k2, control), dtype=float).reshape(-1)
            k4 = np.asarray(rhs(propagated + dense_step * k3, control), dtype=float).reshape(-1)
            propagated = propagated + dense_step * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            ) / 6.0
            dense_states.append(propagated.copy())
            dense_times.append(dense_times[-1] + dense_step)
        endpoint_residuals.append(float(np.max(np.abs(propagated - states[:, interval + 1]))))
        # Continue from the independently propagated endpoint.  Resetting to
        # the shooting node here would hide accumulated integration error and
        # can create artificial jumps (including a tiny hazard decrease) at
        # interval boundaries.
    dense = np.asarray(dense_states).T
    terrain = terrain_height(result["terrain_model"], dense[0], dense[1])
    margins = dense[2] - terrain
    powered_fraction = np.linspace(0.0, 1.0, 2001)
    powered_points = (
        result["launch"][None, :]
        + powered_fraction[:, None]
        * (np.asarray(result["switch_state"][:3]) - result["launch"])[None, :]
    )
    powered_terrain = terrain_height(
        result["terrain_model"], powered_points[:, 0], powered_points[:, 1],
    )
    powered_margin = powered_points[:, 2] - powered_terrain
    sensor = np.asarray(result["sensor"])
    powered_count = powered_fraction.size
    los_outputs = result["los_function"].map(powered_count)(
        powered_points[:, 0].reshape(1, -1),
        powered_points[:, 1].reshape(1, -1),
        powered_points[:, 2].reshape(1, -1),
        np.full((1, powered_count), sensor[0]),
        np.full((1, powered_count), sensor[1]),
        np.full((1, powered_count), sensor[2]),
    )
    los_tuple = los_outputs if isinstance(los_outputs, tuple) else (los_outputs,)
    powered_los = np.asarray(los_tuple[0], dtype=float).reshape(-1)
    powered_detection_outputs = result[
        "powered_total_detection_function"
    ].map(powered_count)(
        powered_points[:, 0].reshape(1, -1),
        powered_points[:, 1].reshape(1, -1),
        powered_points[:, 2].reshape(1, -1),
        np.full((1, powered_count), result["switch_state"][3]),
        np.full((1, powered_count), result["powered_gamma"]),
        np.full((1, powered_count), result["powered_heading"]),
        np.full((1, powered_count), sensor[0]),
        np.full((1, powered_count), sensor[1]),
        np.full((1, powered_count), sensor[2]),
    )
    powered_detection_tuple = (
        powered_detection_outputs
        if isinstance(powered_detection_outputs, tuple)
        else (powered_detection_outputs,)
    )
    powered_total_rate = np.maximum(
        0.0, np.asarray(powered_detection_tuple[-1], dtype=float).reshape(-1)
    ) * float(result.get("detection_hazard_scale", 1.0))
    powered_hazard_reintegrated = float(np.trapezoid(
        powered_total_rate, powered_fraction * result["powered_time"],
    ))
    powered_hazard_residual = abs(
        powered_hazard_reintegrated - float(states[7, 0])
    )
    goal_error = float(np.linalg.norm(dense[:3, -1] - result["goal"]))
    switch_residual = float(np.max(np.abs(
        states[:7, 0] - result["switch_state"]
    )))
    switch_boundary_height = float(
        result["los_boundary_function"](result["switch_state"][:2])
    )
    switch_boundary_residual = abs(
        float(result["switch_state"][2]) - switch_boundary_height
    )
    switch_boundary_tolerance = float(
        result["configuration"]["primary_result"]["vehicle_config"]
        ["switching_constraints"]["tangent_tolerance"]
    )
    peak_index = int(np.argmax(dense[2]))
    final_hazard = float(dense[7, -1])
    mission_pod = float(1.0 - np.exp(-final_hazard))
    configs = result["configuration"]["primary_result"]
    attacker_cost = configs["cost_config"]["attacker"]
    airspace_ceiling = float(configs["environment_config"]["airspace"]["h_max"])
    required_clearance = float(
        configs["vehicle_config"]["switching_constraints"]["terrain_clearance"]
    )
    dense_objective = (
        float(attacker_cost["w_pod"]) * final_hazard
        / float(attacker_cost["normalization"]["pod"]["hazard_reference"])
        + float(attacker_cost["w_time"])
        * (result["powered_time"] + result["glide_time"])
        / float(attacker_cost["normalization"]["time"]["reference_seconds"])
    )
    objective_residual = abs(dense_objective - result["physical_objective"])
    required_powered_margin = required_clearance * powered_fraction
    checks = {
        "powered_terrain_clearance": bool(np.all(
            powered_margin >= required_powered_margin - 1.0e-5
        )),
        "powered_visibility_handling": bool(
            result["powered_visibility_handling"] == "hazard_penalty"
            or np.max(powered_los[1:]) <= 0.05 + 1.0e-8
        ),
        "powered_hazard_reintegration": powered_hazard_residual <= 1.0e-5,
        "terrain_clearance": bool(
            np.min(margins) >= required_clearance - 1.0e-5
        ),
        "airspace_ceiling": bool(
            np.max(dense[2]) <= airspace_ceiling + 1.0e-5
        ),
        "speed_bounds": bool(
            np.min(dense[3]) >= 10.0 - 1.0e-5
            and np.max(dense[3]) <= 22.6 + 1.0e-5
        ),
        "goal_region": goal_error <= 15.0 + 1.0e-5,
        "switch_state_continuity": switch_residual <= 1.0e-7,
        "switch_on_los_boundary_surface": bool(
            not result.get("constrain_switch_to_los_boundary", False)
            or switch_boundary_residual <= switch_boundary_tolerance + 1.0e-5
        ),
        "dense_propagation": max(endpoint_residuals) <= 5.0e-3,
        "hazard_monotone": bool(np.all(np.diff(dense[7]) >= -1.0e-10)),
        "control_bounds": bool(
            np.min(controls[0]) >= 0.05 - 1.0e-8
            and np.max(controls[0]) <= 0.50 + 1.0e-8
            and np.max(np.abs(np.rad2deg(controls[1]))) <= 18.0 + 1.0e-6
        ),
        "objective_reintegration": objective_residual <= 1.0e-5,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_terrain_clearance_m": float(np.min(margins)),
        "minimum_powered_terrain_clearance_m": float(np.min(powered_margin)),
        "maximum_powered_los_visibility": float(np.max(powered_los[1:])),
        "powered_visible_fraction": float(np.mean(powered_los[1:] > 0.05)),
        "powered_hazard_reintegration_residual": powered_hazard_residual,
        "maximum_altitude_m": float(np.max(dense[2])),
        "switch_altitude_m": float(states[2, 0]),
        "post_switch_altitude_gain_m": float(np.max(dense[2]) - states[2, 0]),
        "time_to_post_switch_apex_s": float(
            np.asarray(dense_times)[peak_index] - result["powered_time"]
        ),
        "minimum_speed_m_s": float(np.min(dense[3])),
        "maximum_speed_m_s": float(np.max(dense[3])),
        "maximum_bank_deg": float(np.max(np.abs(np.rad2deg(dense[6])))),
        "maximum_roll_rate_deg_s": float(
            np.max(np.abs(np.rad2deg(result["controls"][1])))
        ),
        "goal_error_m": goal_error,
        "switch_continuity_residual": switch_residual,
        "switch_los_boundary_height_m": switch_boundary_height,
        "switch_los_boundary_residual_m": switch_boundary_residual,
        "switch_los_boundary_tolerance_m": switch_boundary_tolerance,
        "maximum_dense_propagation_residual": max(endpoint_residuals),
        "mission_hazard": final_hazard,
        "mission_pod": mission_pod,
        "powered_hazard": float(states[7, 0]),
        "glide_hazard": float(final_hazard - states[7, 0]),
        "dense_objective": float(dense_objective),
        "objective_reintegration_residual": float(objective_residual),
        "dense_time": np.asarray(dense_times),
        "dense_states": dense,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = solve_continuous_refinement()
    validation = _dense_validate(result)
    if not validation["passed"]:
        raise RuntimeError(f"Continuous validation failed: {validation['checks']}")
    summary = {
        "status_success": True,
        "elapsed_seconds": result["elapsed_seconds"],
        "model": "continuous_3dof_point_mass_post_bellman_refinement",
        "interval_count": INTERVAL_COUNT,
        "projection_6d_to_3d_modified": False,
        "projection_used_as_nlp_initializer": False,
        "discrete_physical_route_used_as_initializer": True,
        "powered_visibility_handling": result["powered_visibility_handling"],
        "powered_time_s": result["powered_time"],
        "glide_time_s": result["glide_time"],
        "mission_time_s": result["powered_time"] + result["glide_time"],
        "powered_gamma_deg": float(np.rad2deg(result["powered_gamma"])),
        "powered_heading_deg": float(np.rad2deg(result["powered_heading"])),
        "switch_state": np.asarray(result["switch_state"]).tolist(),
        "physical_objective": result["physical_objective"],
        "mission_pod": validation["mission_pod"],
        "powered_pod": float(1.0 - np.exp(-validation["powered_hazard"])),
        "glide_only_pod": float(1.0 - np.exp(-validation["glide_hazard"])),
        "powered_hazard": validation["powered_hazard"],
        "glide_hazard": validation["glide_hazard"],
        "minimum_terrain_clearance_m": validation["minimum_terrain_clearance_m"],
        "maximum_altitude_m": validation["maximum_altitude_m"],
        "post_switch_altitude_gain_m": validation["post_switch_altitude_gain_m"],
        "time_to_post_switch_apex_s": validation["time_to_post_switch_apex_s"],
        "minimum_speed_m_s": validation["minimum_speed_m_s"],
        "maximum_speed_m_s": validation["maximum_speed_m_s"],
        "maximum_bank_deg": validation["maximum_bank_deg"],
        "maximum_roll_rate_deg_s": validation["maximum_roll_rate_deg_s"],
        "goal_error_m": validation["goal_error_m"],
        "switch_continuity_residual": validation["switch_continuity_residual"],
        "maximum_dense_propagation_residual": validation[
            "maximum_dense_propagation_residual"
        ],
        "objective_reintegration_residual": validation[
            "objective_reintegration_residual"
        ],
        "maximum_powered_los_visibility": validation[
            "maximum_powered_los_visibility"
        ],
        "powered_visible_fraction": validation["powered_visible_fraction"],
        "powered_hazard_reintegration_residual": validation[
            "powered_hazard_reintegration_residual"
        ],
        "validation_checks": validation["checks"],
        "discrete_baseline": {
            "mission_cost": result["discrete_summary"]["mission_cost"],
            "mission_pod": result["discrete_summary"]["mission_pod"],
            "mission_time_s": result["discrete_summary"]["mission_time"],
            "switching_point": result["discrete_summary"]["switching_point"],
        },
        "optimality_scope": (
            "single-start local continuous refinement of the validated "
            "south-of-ridge discrete topology"
        ),
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    powered_fraction = np.linspace(0.0, 1.0, 201)
    powered_path = (
        result["launch"][None, :]
        + powered_fraction[:, None]
        * (np.asarray(result["switch_state"][:3]) - result["launch"])[None, :]
    )
    np.savez_compressed(
        OUTPUT_DIR / "trajectory_data.npz",
        terrain_x=np.asarray(result["terrain_arrays"]["x"]),
        terrain_y=np.asarray(result["terrain_arrays"]["y"]),
        terrain_height=np.asarray(result["terrain_arrays"]["height"]),
        sensor_position=np.asarray(result["sensor"]),
        goal_position=np.asarray(result["goal"]),
        powered_path=powered_path,
        dense_time=validation["dense_time"],
        dense_states=validation["dense_states"],
        shooting_states=result["states"],
        controls=result["controls"],
        discrete_trajectory=result["discrete"]["trajectory"],
        discrete_powered_path=result["discrete"]["powered_path"],
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Stage-14.0 contracts for the reproducible exact-SSE scaling benchmark.

This module defines what later benchmark stages measure.  It intentionally
contains no timers, sweep runner, solver replacement, or terrain variation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import psutil

from defender_grid_config import CANONICAL_DEFENDER_X_MAP
from discretization_config import discretization_from_physical_steps
from stage11_config import Stage11Config
from stackelberg_solver import FiniteStackelbergRun
from trajectory_validation import TrajectoryReplayAudit


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "stage14.0-v2"
CANONICAL_RESOLUTION = (25.0, 25.0, 5.0)

# Re-derived 2026-09-18 by enumerating CANONICAL_DEFENDER_X_MAP at
# CANONICAL_RESOLUTION with build_terrain("centered_cube"), i.e. exactly what
# stage14_1_profile_worker runs.  Two corrections are folded in:
#
#   1. The previous values named sensor [7.5, 0, 0] as action 2, but the
#      Defender action set in this repository is (5, 6, 7, 8, 9, 10) and has no
#      7.5.  They were produced by an action set that no longer exists, so the
#      identity fields could never be reproduced from this source tree.
#   2. virtual_connection.build_virtual_connections applied its motion-primitive
#      stencil only to candidates landing exactly on an x-y lattice
#      intersection, which made the model report infeasible as the lattice was
#      refined.  With the stencil applied unconditionally the Attacker reaches a
#      mission 6.6 s faster at the same hazard: J_A 7.194589506 -> 7.179870899.
#      The Defender's selected action is unchanged by that correction.
#
# Regenerate with stage14_1_profile_worker; do not hand-edit.
FROZEN_CANONICAL_SOLUTION = {
    "feasible": True,
    "selected_defender_action_id": 3,
    "selected_sensor_position_map": [8.0, 0.0, 0.0],
    "selected_attacker_candidate_id": 25,
    "attacker_objective": 7.179870899415687,
    "defender_objective_pod": 0.9999991726296762,
    "mission_time_s": 78.47972310852555,
    "cumulative_hazard": 14.005013450380838,
    "detection_probability": 0.9999991726296762,
}

SOLVER_SOURCE_FILES = (
    "additive_bellman.py",
    "attacker_best_response.py",
    "bellman_geometry.py",
    "bellman_graph.py",
    "bellman_objectives.py",
    "bellman_solver.py",
    "bellman_state.py",
    "candidate_energy.py",
    "detection_hazard.py",
    "discretization_config.py",
    "edge_hazard.py",
    "energy_model.py",
    "game_types.py",
    "los_geometry.py",
    "map_geometry.py",
    "mission_response.py",
    "reachability_surface.py",
    "sparse_additive.py",
    "sparse_reachability.py",
    "stackelberg_interface.py",
    "stackelberg_solver.py",
    "stackelberg_validation.py",
    "switching_candidates.py",
    "terrain_catalog.py",
    "trajectory_validation.py",
    "virtual_connection.py",
)


@dataclass(frozen=True)
class SolutionRegressionTolerances:
    """Explicit tolerances used to protect the frozen solution identity."""

    attacker_objective_abs: float = 1.0e-12
    defender_objective_abs: float = 1.0e-12
    mission_time_s_abs: float = 1.0e-9
    cumulative_hazard_abs: float = 1.0e-10
    detection_probability_abs: float = 1.0e-12
    sensor_position_map_abs: float = 1.0e-12
    replay_time_s_abs: float = 1.0e-9
    replay_hazard_abs: float = 1.0e-10

    def __post_init__(self) -> None:
        for name, raw in asdict(self).items():
            value = float(raw)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


DEFAULT_STAGE14_TOLERANCES = SolutionRegressionTolerances()


def canonical_stage14_config() -> Stage11Config:
    """Return the single allowed Stage-14.0 physical/model configuration."""
    base = Stage11Config()
    horizontal_m, altitude_m, heading_deg = CANONICAL_RESOLUTION
    discretization = discretization_from_physical_steps(
        horizontal_m,
        altitude_m,
        heading_deg,
        meters_per_map_unit=base.physical_scale.meters_per_map_unit,
        template=base.discretization,
    )
    return replace(base, terrain_category="centered_cube", discretization=discretization)


def canonical_attacker_kwargs(config: Stage11Config) -> dict[str, object]:
    """Map the frozen configuration to the existing exact Attacker-BR API."""
    discretization = config.discretization
    return {
        "bellman_grid": discretization.build_bellman_grid(config.graph_bounds),
        "contour_sample_count": discretization.switching_contour_sample_count,
        "radial_scales": discretization.radial_scales,
        "quadrature_resolution": discretization.hazard_quadrature_resolution,
        "los_probe_grid_size": discretization.los_probe_grid_size,
        "los_boundary_refinement_steps": discretization.los_boundary_refinement_steps,
        "los_display_extension_factor": discretization.los_display_extension_factor,
        "visualization_ray_count": discretization.visualization_ray_count,
        "parameters": config.glider,
        "physical_scale": config.physical_scale,
        "detection_parameters": config.detection,
        "objective_parameters": config.attacker_objective,
        "bellman_backend": "goal_backward_sparse",
    }


def benchmark_configuration() -> dict[str, Any]:
    """Versioned, machine-readable canonical benchmark configuration."""
    config = canonical_stage14_config()
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_stage": "14.0",
        "benchmark_purpose": "freeze exact finite-SSE benchmark contract",
        "terrain_policy": {
            "category": "centered_cube",
            "fixed_for_all_stage14_sweeps": True,
            "terrain_complexity_sweep_allowed": False,
        },
        "algorithm_policy": {
            "solver": "exact finite Strong Stackelberg enumeration with exact Bellman BR",
            "reinforcement_learning_allowed": False,
            "approximate_planning_allowed": False,
            "alternative_solver_allowed": False,
            "strong_stackelberg_tie_break_preserved": True,
        },
        "canonical_resolution": {
            "horizontal_step_m": CANONICAL_RESOLUTION[0],
            "altitude_step_m": CANONICAL_RESOLUTION[1],
            "heading_step_deg": CANONICAL_RESOLUTION[2],
        },
        "defender_action_set": {
            "x_map": list(CANONICAL_DEFENDER_X_MAP),
            "y_map": 0.0,
            "z_map": 0.0,
            "ordered_and_deterministic": True,
        },
        "mission_and_model": config.as_dict(),
        "regression_tolerances": asdict(DEFAULT_STAGE14_TOLERANCES),
    }


def quantity_definitions() -> dict[str, dict[str, Any]]:
    """Unambiguous names retained by all later Stage-14 records."""
    return {
        "N_x": {"unit": "count", "definition": "x-axis lattice node count"},
        "N_y": {"unit": "count", "definition": "y-axis lattice node count"},
        "N_h": {"unit": "count", "definition": "altitude lattice node count"},
        "N_psi": {"unit": "count", "definition": "heading-bin count"},
        "N_S_cart": {
            "unit": "states",
            "definition": "full Cartesian count N_x*N_y*N_h*N_psi before terrain or reachability filtering",
        },
        "N_S_admissible": {
            "unit": "states",
            "definition": "Cartesian lattice states whose position is not strictly inside terrain solid",
        },
        "N_S_goal_reachable": {
            "unit": "states",
            "definition": "exact goal-backward-reachable finite states after terrain/dynamics filtering",
        },
        "N_S_active": {
            "unit": "states",
            "definition": "goal-reachable states in the forward descendant corridor of feasible virtual connections and processed by hazard/Bellman",
        },
        "N_E": {
            "unit": "edges",
            "definition": "valid edges in the active corridor actually processed by hazard precomputation and additive Bellman",
        },
        "N_E_goal_reachable": {
            "unit": "edges",
            "definition": "valid transition edges in the complete goal-backward-reachable graph before active-corridor restriction",
        },
        "N_C_raw": {
            "unit": "candidates per Defender action",
            "definition": "switching candidates generated before powered/energy/connection/reachability filters",
        },
        "N_C_powered_feasible": {
            "unit": "candidates per Defender action",
            "definition": "raw switching candidates with a feasible powered segment",
        },
        "N_C_energy_feasible": {
            "unit": "candidates per Defender action",
            "definition": "candidates passing powered switching-state total-energy feasibility",
        },
        "N_C_virtual_feasible": {
            "unit": "candidates per Defender action",
            "definition": "candidates with at least one admissible virtual lattice connection",
        },
        "N_C_goal_reachable": {
            "unit": "candidates per Defender action",
            "definition": "candidates with a virtual connection to the exact goal-backward set",
        },
        "N_C_feasible": {
            "unit": "candidates per Defender action",
            "definition": "candidates with a complete feasible powered-virtual-glide mission",
        },
        "N_D": {"unit": "actions", "definition": "finite Defender action-set cardinality"},
        "B": {
            "unit": "attempts per source state",
            "definition": "configured maximum target-heading motion primitives considered per source by forward transition generation; sparse-backward exact attempt counts require Stage-14.1 instrumentation",
        },
        "B_effective": {
            "unit": "valid edges per active state",
            "definition": "N_E/N_S_active; descriptive edge density, not attempted branching factor",
        },
        "Q": {
            "unit": "quadrature samples per edge",
            "definition": "configured trapezoidal detection-hazard samples for each processed edge",
        },
    }


def benchmark_result_schema() -> dict[str, Any]:
    """Schema declaration; Stage 14.1 may add profiling fields, never rename these."""
    return {
        "schema_version": SCHEMA_VERSION,
        "required_top_level_fields": [
            "schema_version", "configuration", "environment_manifest_id",
            "solver_source_fingerprint", "quantity_definitions", "state_and_game_size",
            "solution_identity", "defender_action_results", "independent_replay",
            "regression", "gate_passed",
        ],
        "quantity_definitions": quantity_definitions(),
        "solution_identity_fields": [
            "feasible", "selected_defender_action_id", "selected_sensor_position_map",
            "leader_cooptimal_action_ids", "selected_attacker_candidate_id",
            "attacker_objective_cooptimal_candidate_ids", "sse_selected_follower_candidate_id",
            "attacker_objective", "defender_objective_pod", "mission_time_s",
            "cumulative_hazard", "detection_probability", "trajectory_identity",
        ],
        "failure_status_vocabulary_reserved_for_later_substages": [
            "completed", "model_infeasible", "numerical_failure",
            "computational_failure", "timeout", "memory_limit",
        ],
    }


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def solver_source_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Hash the solver/model layer while excluding Stage-14 diagnostics."""
    file_hashes = {
        name: _sha256_file(root / name)
        for name in SOLVER_SOURCE_FILES
    }
    aggregate = sha256()
    for name, digest in sorted(file_hashes.items()):
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "algorithm": "sha256(relative solver filename + NUL + file sha256)",
        "aggregate_sha256": aggregate.hexdigest(),
        "files": file_hashes,
    }


def _git_value(arguments: Iterable[str], repository_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *arguments), cwd=repository_root, check=True,
            capture_output=True, text=True, encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def environment_manifest(repository_root: Path | None = None) -> dict[str, Any]:
    """Capture the software/hardware identity required by Stage 14.0."""
    repo = (repository_root or ROOT.parent).resolve()
    status = _git_value(("status", "--porcelain=v1", "--untracked-files=all"), repo)
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[str(name)] = distribution.version
    packages = dict(sorted(packages.items(), key=lambda item: item[0].lower()))
    source = solver_source_manifest(ROOT)
    identity_payload = {
        "git_commit": _git_value(("rev-parse", "HEAD"), repo),
        "git_status_sha256": None if status is None else sha256(status.encode("utf-8")).hexdigest(),
        "solver_source_sha256": source["aggregate_sha256"],
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    manifest_id = sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    memory = psutil.virtual_memory()
    return {
        "schema_version": SCHEMA_VERSION,
        "environment_manifest_id": manifest_id,
        "software_revision": {
            "git_commit": identity_payload["git_commit"],
            "git_dirty": bool(status),
            "git_status_sha256": identity_payload["git_status_sha256"],
            "solver_source": source,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": packages,
        "operating_system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {
            "model": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "memory": {
            "total_bytes": int(memory.total),
            "available_bytes_at_capture": int(memory.available),
        },
    }


def admissible_state_count(run: FiniteStackelbergRun) -> tuple[int, int]:
    """Count terrain-admissible lattice positions/states without changing graph logic."""
    representative = run.evaluations[0].attacker_run
    grid = representative.graph.grid
    x_values, y_values, z_values = np.meshgrid(
        grid.x_coordinates,
        grid.y_coordinates,
        grid.altitude_coordinates,
        indexing="xy",
    )
    positions = np.column_stack((x_values.ravel(), y_values.ravel(), z_values.ravel()))
    vectorized = getattr(representative.terrain, "contains_solid_many", None)
    if callable(vectorized):
        excluded = np.asarray(vectorized(positions), dtype=bool)
    else:
        excluded = np.asarray(
            [representative.terrain.contains_solid(point) for point in positions],
            dtype=bool,
        )
    admissible_positions = int(np.count_nonzero(~excluded))
    return admissible_positions, admissible_positions * grid.heading_bin_count


def _trajectory_identity(run: FiniteStackelbergRun) -> dict[str, Any]:
    selected = run.selected_evaluation.sse_follower_result
    if selected is None or selected.selected_option is None:
        raise ValueError("canonical exact SSE has no selected trajectory")
    option = selected.selected_option
    state_ids = [option.connection.target_state_id]
    state_ids.extend(edge.target_id for edge in option.glide_edges)
    payload = {
        "switching_position_map": selected.position_map.tolist(),
        "virtual_target_state_id": option.connection.target_state_id,
        "bellman_state_ids": state_ids,
    }
    payload["sha256"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def frozen_reference_from_run(
    run: FiniteStackelbergRun,
    audit: TrajectoryReplayAudit,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Serialize the exact canonical result under the Stage-14.0 contract."""
    if not isinstance(run, FiniteStackelbergRun):
        raise TypeError("run must be a FiniteStackelbergRun")
    if not isinstance(audit, TrajectoryReplayAudit):
        raise TypeError("audit must be a TrajectoryReplayAudit")
    config = canonical_stage14_config()
    selected_evaluation = run.selected_evaluation
    follower = selected_evaluation.sse_follower_result
    if follower is None:
        raise RuntimeError("canonical exact SSE unexpectedly has no follower")
    representative = run.evaluations[0].attacker_run
    grid = representative.graph.grid
    admissible_positions, admissible_states = admissible_state_count(run)
    maximum_defender_payoff = max(
        float(item.outcome.defender_payoff)
        for item in run.evaluations if item.outcome is not None
    )
    leader_ties = [
        item.candidate.action_id
        for item in run.evaluations
        if item.outcome is not None
        and float(item.outcome.defender_payoff)
        >= maximum_defender_payoff - run.leader_payoff_tolerance
    ]

    defender_rows: list[dict[str, Any]] = []
    for evaluation in run.evaluations:
        metrics = evaluation.attacker_run.metrics
        candidate_results = evaluation.attacker_run.candidate_results
        selected_follower = evaluation.sse_follower_result
        defender_rows.append({
            "defender_action_id": evaluation.candidate.action_id,
            "sensor_position_map": evaluation.candidate.action.sensor_position_map.tolist(),
            "feasible": evaluation.feasible,
            "sse_follower_candidate_id": (
                None if selected_follower is None else selected_follower.candidate_id
            ),
            "attacker_objective_cooptimal_candidate_ids": list(
                evaluation.attacker_run.cooptimal_candidate_ids
            ),
            "attacker_objective": (
                None if evaluation.outcome is None else evaluation.outcome.attacker_payoff
            ),
            "defender_objective_pod": (
                None if evaluation.outcome is None else evaluation.outcome.defender_payoff
            ),
            "mission_time_s": (
                None if selected_follower is None else selected_follower.mission_time_s
            ),
            "cumulative_hazard": (
                None if selected_follower is None else selected_follower.cumulative_hazard
            ),
            "candidate_counts": {
                "N_C_raw": len(candidate_results),
                "N_C_powered_feasible": sum(item.powered_feasible for item in candidate_results),
                "N_C_energy_feasible": metrics.number_energy_feasible,
                "N_C_virtual_feasible": metrics.number_virtual_feasible,
                "N_C_goal_reachable": metrics.number_goal_reachable,
                "N_C_feasible": metrics.number_feasible,
            },
            "N_S_active": metrics.active_state_count,
            "N_E": metrics.hazard_edge_count,
        })

    selected_metrics = selected_evaluation.attacker_run.metrics
    state_and_game_size = {
        "N_x": grid.x_count,
        "N_y": grid.y_count,
        "N_h": grid.altitude_count,
        "N_psi": grid.heading_bin_count,
        "N_S_cart": grid.state_count,
        "N_position_cart": grid.x_count * grid.y_count * grid.altitude_count,
        "N_position_admissible": admissible_positions,
        "N_S_admissible": admissible_states,
        "N_S_goal_reachable": selected_metrics.goal_reachable_state_count,
        "N_S_active": selected_metrics.active_state_count,
        "N_E": selected_metrics.hazard_edge_count,
        "N_E_goal_reachable": representative.graph.statistics.valid_edge_count,
        "N_C_raw": selected_metrics.number_of_candidates,
        "N_C_powered_feasible": sum(
            item.powered_feasible for item in selected_evaluation.attacker_run.candidate_results
        ),
        "N_C_energy_feasible": selected_metrics.number_energy_feasible,
        "N_C_virtual_feasible": selected_metrics.number_virtual_feasible,
        "N_C_goal_reachable": selected_metrics.number_goal_reachable,
        "N_C_feasible": selected_metrics.number_feasible,
        "N_D": len(run.defender_candidates),
        "B": len(grid.motion_offsets),
        "B_effective": (
            selected_metrics.hazard_edge_count / selected_metrics.active_state_count
        ),
        "Q": config.discretization.hazard_quadrature_resolution,
    }
    solution_identity = {
        "feasible": True,
        "selected_defender_action_id": selected_evaluation.candidate.action_id,
        "selected_sensor_position_map": (
            selected_evaluation.candidate.action.sensor_position_map.tolist()
        ),
        "leader_cooptimal_action_ids": leader_ties,
        "selected_leader_tie_break_action_id": selected_evaluation.candidate.action_id,
        "attacker_objective_cooptimal_candidate_ids": list(
            selected_evaluation.attacker_run.cooptimal_candidate_ids
        ),
        "sse_selected_follower_candidate_id": follower.candidate_id,
        "selected_attacker_candidate_id": follower.candidate_id,
        "attacker_objective": run.outcome.attacker_payoff,
        "defender_objective_pod": run.outcome.defender_payoff,
        "mission_time_s": follower.mission_time_s,
        "cumulative_hazard": follower.cumulative_hazard,
        "detection_probability": follower.detection_probability,
        "trajectory_identity": _trajectory_identity(run),
        "equilibrium_scope": run.equilibrium_scope,
        "tie_break_convention": {
            "follower": "maximize Defender PoD among Attacker-objective co-optima, then lowest candidate ID",
            "leader": "stable Defender action order among payoff ties",
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "configuration": benchmark_configuration(),
        "environment_manifest_id": environment["environment_manifest_id"],
        "solver_source_fingerprint": environment["software_revision"]["solver_source"]["aggregate_sha256"],
        "quantity_definitions": quantity_definitions(),
        "state_and_game_size": state_and_game_size,
        "solution_identity": solution_identity,
        "defender_action_results": defender_rows,
        "independent_replay": {
            "passed": audit.report.passed,
            "messages": list(audit.report.messages),
            "mission_time_s": audit.recomputed_mission_time_s,
            "cumulative_hazard": audit.recomputed_cumulative_hazard,
            "detection_probability": audit.recomputed_detection_probability,
            "time_error_s": audit.report.time_error_s,
            "hazard_error": audit.report.hazard_error,
            "goal_distance_m": audit.goal_distance_m,
            "energy_margin_m": audit.energy_margin_m,
        },
        "regression": {},
        "gate_passed": False,
    }


def validate_frozen_reference(
    reference: dict[str, Any],
    *,
    tolerances: SolutionRegressionTolerances = DEFAULT_STAGE14_TOLERANCES,
) -> dict[str, Any]:
    """Compare a newly serialized canonical run with the pre-Stage-14 values."""
    identity = reference["solution_identity"]
    checks = {
        "feasible": identity["feasible"] is FROZEN_CANONICAL_SOLUTION["feasible"],
        "selected_defender_action_id": (
            identity["selected_defender_action_id"]
            == FROZEN_CANONICAL_SOLUTION["selected_defender_action_id"]
        ),
        "selected_sensor_position_map": bool(np.allclose(
            identity["selected_sensor_position_map"],
            FROZEN_CANONICAL_SOLUTION["selected_sensor_position_map"],
            rtol=0.0,
            atol=tolerances.sensor_position_map_abs,
        )),
        "selected_attacker_candidate_id": (
            identity["selected_attacker_candidate_id"]
            == FROZEN_CANONICAL_SOLUTION["selected_attacker_candidate_id"]
        ),
        "attacker_objective": bool(np.isclose(
            identity["attacker_objective"], FROZEN_CANONICAL_SOLUTION["attacker_objective"],
            rtol=0.0, atol=tolerances.attacker_objective_abs,
        )),
        "defender_objective_pod": bool(np.isclose(
            identity["defender_objective_pod"], FROZEN_CANONICAL_SOLUTION["defender_objective_pod"],
            rtol=0.0, atol=tolerances.defender_objective_abs,
        )),
        "mission_time_s": bool(np.isclose(
            identity["mission_time_s"], FROZEN_CANONICAL_SOLUTION["mission_time_s"],
            rtol=0.0, atol=tolerances.mission_time_s_abs,
        )),
        "cumulative_hazard": bool(np.isclose(
            identity["cumulative_hazard"], FROZEN_CANONICAL_SOLUTION["cumulative_hazard"],
            rtol=0.0, atol=tolerances.cumulative_hazard_abs,
        )),
        "detection_probability": bool(np.isclose(
            identity["detection_probability"], FROZEN_CANONICAL_SOLUTION["detection_probability"],
            rtol=0.0, atol=tolerances.detection_probability_abs,
        )),
        "independent_replay": bool(reference["independent_replay"]["passed"]),
        "replay_time": (
            float(reference["independent_replay"]["time_error_s"])
            <= tolerances.replay_time_s_abs
        ),
        "replay_hazard": (
            float(reference["independent_replay"]["hazard_error"])
            <= tolerances.replay_hazard_abs
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "tolerances": asdict(tolerances),
        "expected_pre_stage14_solution": FROZEN_CANONICAL_SOLUTION,
    }


__all__ = [
    "CANONICAL_DEFENDER_X_MAP", "CANONICAL_RESOLUTION",
    "DEFAULT_STAGE14_TOLERANCES", "FROZEN_CANONICAL_SOLUTION", "SCHEMA_VERSION",
    "SolutionRegressionTolerances", "admissible_state_count",
    "benchmark_configuration", "benchmark_result_schema", "canonical_attacker_kwargs",
    "canonical_stage14_config",
    "environment_manifest", "frozen_reference_from_run", "quantity_definitions",
    "solver_source_manifest", "validate_frozen_reference",
]

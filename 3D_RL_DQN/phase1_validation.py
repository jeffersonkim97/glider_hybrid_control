"""Phase 1 parity validation and development-only visual diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import project_paths
from P1b_Exact_Local_SSE import exact_best_response
from P1b_RL_approximation import QLearningConfig, default_sensor, switching_candidates
from P1b_condition import ComputationCondition, build_scene
from attacker_br_problem import AttackerBRProblem, AttackerBRResult, solve_attacker_br


PHASE0_DIR = project_paths.CORE_DIR / "figure" / "phase_0_baseline"
OUTPUT_DIR = project_paths.CORE_DIR / "figure" / "phase_1_attacker_br_interface"
MANIFEST_NAME = "phase1_attacker_br_interface_manifest.json"
TOLERANCE = 1.0e-12


def _relative(path: Path) -> str:
    return path.resolve().relative_to(project_paths.WORKSPACE_ROOT.resolve()).as_posix()


def _load_phase0() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(
        (PHASE0_DIR / "phase0_discovery_manifest.json").read_text(encoding="utf-8")
    )
    results = json.loads(
        (PHASE0_DIR / "phase0_baseline_results.json").read_text(encoding="utf-8")
    )
    return manifest, results


def _condition_from_phase0(manifest: dict[str, Any]) -> ComputationCondition:
    stored = manifest["canonical_cube_case"]["configuration"]["condition"]
    return ComputationCondition(
        spatial_resolution_m=float(stored["spatial_resolution_m"]),
        heading_spacing_deg=float(stored["heading_spacing_deg"]),
        r_neighbor=int(stored["r_neighbor"]),
        terrain_category=str(stored["terrain_category"]),
        hazard_weight=stored.get("hazard_weight"),
        time_weight=stored.get("time_weight"),
        defender_goal_margin_map=float(stored["defender_goal_margin_map"]),
    )


def _state_ids_for_parity(problem: AttackerBRProblem, solution: Any, count: int = 64) -> tuple[int, ...]:
    active = np.flatnonzero(np.asarray(solution.active_mask, dtype=bool))
    candidates = [
        int(state_id) for state_id in active
        if not problem.is_terminal(int(state_id)) and problem.successors(int(state_id))
    ]
    if len(candidates) <= count:
        return tuple(candidates)
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return tuple(candidates[int(index)] for index in indices)


def _edge_signature(edge: Any) -> tuple[Any, ...]:
    return (
        int(edge.source_id), int(edge.target_id), edge.source_state, edge.target_state,
        float(edge.horizontal_distance_m), float(edge.altitude_loss_m),
        float(edge.duration_s), float(edge.heading_change_rad),
    )


def run_parity_checks(
    problem: AttackerBRProblem,
    oracle: Any,
    phase0_manifest: dict[str, Any],
    phase0_results: dict[str, Any],
    bellman_result: AttackerBRResult,
    tabular_result: AttackerBRResult,
) -> dict[str, Any]:
    sampled = _state_ids_for_parity(problem, oracle.solution)
    transition_rows = 0
    transitions_checked = 0
    transition_mismatches: list[dict[str, Any]] = []
    feasibility_mismatches: list[dict[str, Any]] = []
    cost_mismatches: list[dict[str, Any]] = []
    terminal_mismatches: list[int] = []
    maximum_cost_error = 0.0

    for state_id in sampled:
        common = problem.successors(state_id)
        legacy = tuple(problem.scene.graph.adjacency[state_id])
        transition_rows += 1
        transitions_checked += len(legacy)
        if tuple(map(_edge_signature, common)) != tuple(map(_edge_signature, legacy)):
            transition_mismatches.append({
                "state_id": state_id,
                "common_targets": [int(edge.target_id) for edge in common],
                "legacy_targets": [int(edge.target_id) for edge in legacy],
            })

        raw, _, _ = problem.raw_transition_candidates(state_id)
        expected = tuple(
            edge for edge in raw
            if problem.scene.graph.node_mask[int(edge.target_id)]
        )
        if tuple(map(_edge_signature, common)) != tuple(map(_edge_signature, expected)):
            feasibility_mismatches.append({
                "state_id": state_id,
                "common_targets": [int(edge.target_id) for edge in common],
                "raw_goal_reachable_targets": [int(edge.target_id) for edge in expected],
            })

        bellman_costs = oracle.solution.edge_cost_by_source[state_id]
        for action_index, expected_cost in enumerate(bellman_costs):
            actual = problem.transition(state_id, action_index).stage_cost
            error = abs(float(actual) - float(expected_cost))
            maximum_cost_error = max(maximum_cost_error, error)
            if error > TOLERANCE:
                cost_mismatches.append({
                    "state_id": state_id,
                    "action_index": action_index,
                    "common_cost": actual,
                    "bellman_cost": float(expected_cost),
                    "absolute_error": error,
                })

        if problem.is_terminal(state_id) != bool(problem.scene.graph.terminal_mask[state_id]):
            terminal_mismatches.append(state_id)

    terminal_ids = np.flatnonzero(problem.scene.graph.terminal_mask)
    for state_id in terminal_ids[: min(32, len(terminal_ids))]:
        value = int(state_id)
        if not problem.is_terminal(value) or problem.successors(value):
            terminal_mismatches.append(value)

    exact_objective = problem.attacker_objective(
        int(oracle.switching_state_id), tuple(oracle.glide_state_ids),
    )
    phase0_bellman = phase0_results["bellman"]
    phase0_tabular = phase0_results["tabular_q_learning"]
    bellman_j_error = abs(
        float(bellman_result.attacker_objective)
        - float(phase0_bellman["fixed_sensor_J_A"])
    )
    tabular_j_error = abs(
        float(tabular_result.attacker_objective)
        - float(phase0_tabular["fixed_sensor_J_A"])
    )
    trajectory_j_error = abs(float(exact_objective) - float(oracle.attacker_objective))

    existing_candidates = switching_candidates(
        problem.scene, problem.sensor_map,
    ).state_ids
    initial_candidate_match = problem.switching_state_ids() == existing_candidates
    start_match = np.array_equal(
        problem.scene.config.start.as_array(),
        np.asarray(
            phase0_manifest["canonical_cube_case"]["configuration"]["attacker_start_map"],
            dtype=float,
        ),
    )
    bellman_trajectory_match = (
        list(bellman_result.trajectory) == phase0_bellman["trajectory_state_ids"]
    )
    tabular_trajectory_match = (
        list(tabular_result.trajectory) == phase0_tabular["trajectory_state_ids"]
    )
    bellman_local = bellman_result.as_local_defender_evaluation(
        problem.scene.seed_action_id,
    )
    tabular_local = tabular_result.as_local_defender_evaluation(
        problem.scene.seed_action_id,
    )

    mismatch_total = (
        len(transition_mismatches) + len(feasibility_mismatches)
        + len(cost_mismatches) + len(set(terminal_mismatches))
        + int(not initial_candidate_match) + int(not start_match)
        + int(trajectory_j_error > TOLERANCE)
        + int(bellman_j_error > TOLERANCE) + int(tabular_j_error > TOLERANCE)
        + int(not bellman_trajectory_match) + int(not tabular_trajectory_match)
    )
    return {
        "tolerance": TOLERANCE,
        "sampled_state_ids": list(sampled),
        "sampled_nonterminal_states": len(sampled),
        "sampled_terminal_states": min(32, len(terminal_ids)),
        "transition_rows_checked": transition_rows,
        "transitions_checked": transitions_checked,
        "initial_state_matches_phase0": start_match,
        "switching_candidate_parity": initial_candidate_match,
        "terminal_mismatch_count": len(set(terminal_mismatches)),
        "transition_mismatch_count": len(transition_mismatches),
        "feasibility_mismatch_count": len(feasibility_mismatches),
        "immediate_cost_mismatch_count": len(cost_mismatches),
        "maximum_immediate_cost_absolute_error": maximum_cost_error,
        "trajectory_J_A_absolute_error": trajectory_j_error,
        "bellman_phase0_J_A_absolute_error": bellman_j_error,
        "tabular_phase0_J_A_absolute_error": tabular_j_error,
        "bellman_phase0_trajectory_match": bellman_trajectory_match,
        "tabular_phase0_trajectory_match": tabular_trajectory_match,
        "local_sse_contract": {
            "bellman_feasible": bellman_local.feasible,
            "bellman_exact_verified": bellman_local.exact_attacker_best_response_verified,
            "tabular_feasible": tabular_local.feasible,
            "tabular_exact_verified": tabular_local.exact_attacker_best_response_verified,
        },
        "mismatch_total": mismatch_total,
        "passed": mismatch_total == 0,
        "mismatches": {
            "transition": transition_mismatches,
            "feasibility": feasibility_mismatches,
            "immediate_cost": cost_mismatches,
            "terminal_state_ids": sorted(set(terminal_mismatches)),
        },
    }


def _terrain_faces(scene: Any) -> list[np.ndarray]:
    mesh = scene.terrain.obstacle_mesh()
    return [mesh.vertices[triangle] for triangle in mesh.triangles]


def _add_terrain_3d(ax: Any, scene: Any) -> None:
    ax.add_collection3d(Poly3DCollection(
        _terrain_faces(scene), facecolor="#bdbdbd", edgecolor="#555555",
        linewidth=0.5, alpha=0.35,
    ))
    bounds = scene.config.graph_bounds
    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_zlim(0.0, scene.grid.maximum_altitude_map)
    ax.set_xlabel("x [map unit]")
    ax.set_ylabel("y [map unit]")
    ax.set_zlabel("altitude [map unit]")


def _add_terrain_top(ax: Any, scene: Any) -> None:
    for box in scene.terrain.obstacle_boxes():
        rectangle = plt.Rectangle(
            (box.x_limits[0], box.y_limits[0]), box.width_x, box.width_y,
            facecolor="#bdbdbd", edgecolor="#444444", alpha=0.55,
        )
        ax.add_patch(rectangle)
    bounds = scene.config.graph_bounds
    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [map unit]")
    ax.set_ylabel("y [map unit]")


def _save(figure: Any, output: Path) -> str:
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return _relative(output)


def _line3d(ax: Any, source: np.ndarray, target: np.ndarray, **kwargs: Any) -> None:
    ax.plot(
        [source[0], target[0]], [source[1], target[1]],
        [source[2], target[2]], **kwargs,
    )


def figure_transitions(
    problem: AttackerBRProblem, state_id: int, output: Path,
) -> str:
    raw, rejected, _ = problem.raw_transition_candidates(state_id)
    feasible_targets = {edge.target_id for edge in problem.successors(state_id)}
    source = problem.position_map(state_id)
    figure = plt.figure(figsize=(9.0, 6.6))
    ax = figure.add_subplot(111, projection="3d")
    _add_terrain_3d(ax, problem.scene)
    for edge in raw:
        target = problem.position_map(edge.target_id)
        reachable = edge.target_id in feasible_targets
        _line3d(
            ax, source, target,
            color="#238443" if reachable else "#E69F00",
            linewidth=1.5 if reachable else 0.9,
            alpha=0.9 if reachable else 0.5,
        )
    reason_colors = {
        "terrain collision": "#D7301F", "map bounds": "#969696",
        "turn constraint": "#756BB1", "insufficient altitude": "#3182BD",
    }
    for item in rejected:
        _line3d(
            ax, item.source_position_map, item.target_position_map,
            color=reason_colors.get(item.reason, "#555555"), alpha=0.35,
            linewidth=0.8, linestyle="--",
        )
    ax.scatter(*source, color="black", s=45, label=f"state {state_id}")
    handles = [
        plt.Line2D([0], [0], color="#238443", label="BR-feasible"),
        plt.Line2D([0], [0], color="#E69F00", label="raw valid, not goal-reachable"),
        plt.Line2D([0], [0], color="#D7301F", linestyle="--", label="terrain rejected"),
        plt.Line2D([0], [0], color="#969696", linestyle="--", label="other rejected"),
    ]
    ax.legend(handles=handles, loc="upper left")
    ax.set_title("Authoritative transition neighborhood")
    return _save(figure, output)


def _derived_gamma_deg(source: np.ndarray, target: np.ndarray) -> float:
    horizontal = float(np.linalg.norm(target[:2] - source[:2]))
    loss = float(source[2] - target[2])
    return float(np.degrees(np.arctan2(loss, horizontal))) if horizontal else float("nan")


def figure_angles(
    problem: AttackerBRProblem, state_id: int, output: Path,
) -> str:
    raw, rejected, _ = problem.raw_transition_candidates(state_id)
    feasible = {edge.target_id for edge in problem.successors(state_id)}
    source = problem.position_map(state_id)
    figure = plt.figure(figsize=(10.0, 4.8))
    polar = figure.add_subplot(121, projection="polar")
    angular = figure.add_subplot(122)
    for edge in raw:
        psi = float(problem.grid.heading_rad(edge.target_state.heading_bin))
        gamma = _derived_gamma_deg(source, problem.position_map(edge.target_id))
        color = "#238443" if edge.target_id in feasible else "#E69F00"
        polar.scatter(psi, 1.0, color=color, s=22)
        angular.scatter(np.degrees(psi), gamma, color=color, s=24)
    for item in rejected:
        psi = float(problem.grid.heading_rad(item.target_heading_bin))
        gamma = _derived_gamma_deg(source, np.asarray(item.target_position_map))
        polar.scatter(psi, 0.72, color="#BDBDBD", s=12, alpha=0.6)
        angular.scatter(np.degrees(psi), gamma, color="#BDBDBD", s=12, alpha=0.6)
    current = float(problem.grid.heading_rad(problem.state(state_id).heading_bin))
    polar.scatter(current, 1.25, color="black", marker="*", s=90)
    polar.set_title("Heading $\\psi$ candidates")
    polar.set_yticklabels([])
    angular.set_xlabel("target heading $\\psi$ [deg]")
    angular.set_ylabel("derived descent angle $\\gamma$ [deg]")
    angular.set_title("$\\gamma$ is edge-derived; no gamma state/bin")
    angular.grid(alpha=0.25)
    return _save(figure, output)


def figure_terrain_feasibility(
    problem: AttackerBRProblem, state_id: int, output: Path,
) -> str:
    raw, rejected, _ = problem.raw_transition_candidates(state_id)
    feasible = {edge.target_id for edge in problem.successors(state_id)}
    source = problem.position_map(state_id)
    figure, ax = plt.subplots(figsize=(8.5, 5.3))
    _add_terrain_top(ax, problem.scene)
    for edge in raw:
        target = problem.position_map(edge.target_id)
        ax.plot(
            [source[0], target[0]], [source[1], target[1]],
            color="#238443" if edge.target_id in feasible else "#E69F00",
            alpha=0.8 if edge.target_id in feasible else 0.4,
        )
    for item in rejected:
        if item.reason == "terrain collision":
            target = item.target_position_map
            ax.plot([source[0], target[0]], [source[1], target[1]],
                    color="#D7301F", linestyle="--", alpha=0.8)
    ax.scatter(source[0], source[1], color="black", s=45, zorder=5)
    ax.legend(handles=[
        plt.Line2D([0], [0], color="#238443", label="BR-feasible"),
        plt.Line2D([0], [0], color="#E69F00", label="raw valid, filtered"),
        plt.Line2D([0], [0], color="#D7301F", linestyle="--", label="terrain collision"),
    ])
    ax.set_title("Terrain feasibility from the existing collision logic")
    return _save(figure, output)


def figure_hazard_overlay(
    problem: AttackerBRProblem, state_id: int, output: Path,
) -> str:
    source = problem.position_map(state_id)
    transitions = [
        problem.transition(state_id, index)
        for index in range(len(problem.successors(state_id)))
    ]
    hazards = np.asarray([item.cumulative_hazard for item in transitions], dtype=float)
    norm = Normalize(vmin=float(hazards.min()), vmax=float(hazards.max()) or 1.0)
    cmap = plt.get_cmap("viridis")
    figure, ax = plt.subplots(figsize=(8.5, 5.3))
    _add_terrain_top(ax, problem.scene)
    for item in transitions:
        target = problem.position_map(item.target_state_id)
        ax.plot([source[0], target[0]], [source[1], target[1]],
                color=cmap(norm(item.cumulative_hazard)), linewidth=2.2)
    sensor = np.asarray(problem.sensor_map)
    ax.scatter(source[0], source[1], color="black", s=45, label="state")
    ax.scatter(sensor[0], sensor[1], color="#D7301F", marker="^", s=65, label="sensor")
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colorbar.set_label("integrated edge hazard")
    ax.legend()
    ax.set_title("Existing sensing hazard on feasible transitions")
    return _save(figure, output)


def figure_rollout(
    problem: AttackerBRProblem, result: AttackerBRResult, output: Path,
) -> str:
    positions = np.asarray([problem.position_map(value) for value in result.trajectory])
    start = np.asarray(problem.scene.config.start.as_array(), dtype=float)
    positions = np.vstack((start, positions))
    figure = plt.figure(figsize=(8.8, 6.4))
    ax = figure.add_subplot(111, projection="3d")
    _add_terrain_3d(ax, problem.scene)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            color="#1C5770", marker="o", linewidth=2.3, label="common-interface rollout")
    ax.scatter(*positions[0], color="black", marker="s", s=45, label="mission start")
    ax.scatter(*positions[-1], color="#238443", marker="*", s=70, label="terminal")
    ax.legend(loc="upper left")
    ax.set_title("Feasible rollout through the common interface")
    return _save(figure, output)


def figure_parity(parity: dict[str, Any], output: Path) -> str:
    labels = ["transition", "feasibility", "terminal", "stage cost", "trajectory $J_A$"]
    checked = [
        parity["transitions_checked"], parity["transition_rows_checked"],
        parity["sampled_terminal_states"], parity["transitions_checked"], 1,
    ]
    mismatches = [
        parity["transition_mismatch_count"], parity["feasibility_mismatch_count"],
        parity["terminal_mismatch_count"], parity["immediate_cost_mismatch_count"],
        int(parity["trajectory_J_A_absolute_error"] > parity["tolerance"]),
    ]
    y = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.8))
    axes[0].barh(y, checked, color="#1C5770")
    axes[0].set_yticks(y, labels)
    axes[0].set_title("Checks performed")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(y, mismatches, color="#B3402F")
    axes[1].set_yticks(y, labels)
    axes[1].set_title(f"Mismatches (total={parity['mismatch_total']})")
    axes[1].set_xlim(0, max(1, max(mismatches) + 1))
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle("Phase 1 legacy/common-interface parity")
    return _save(figure, output)


def run_phase1(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    phase0_manifest, phase0_results = _load_phase0()
    condition = _condition_from_phase0(phase0_manifest)
    scene = build_scene(condition)
    problem = AttackerBRProblem(scene, default_sensor(scene))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 1 condition: {condition.label}; terrain={condition.terrain_category}", flush=True)
    print("Solving Bellman through the common interface...", flush=True)
    bellman = solve_attacker_br(problem, "bellman")
    print("Solving Tabular Q through the common interface (20,000 episodes)...", flush=True)
    q_config = QLearningConfig()
    tabular = solve_attacker_br(problem, "tabular_q", q_config)
    if not bellman.success or not tabular.success:
        raise RuntimeError("both canonical solver adapters must return a feasible BR")

    oracle = exact_best_response(scene, problem.sensor_map, keep_solution=True)
    parity = run_parity_checks(
        problem, oracle, phase0_manifest, phase0_results, bellman, tabular,
    )
    if not parity["passed"]:
        print(f"Parity mismatches found: {parity['mismatch_total']}", flush=True)

    representative = int(oracle.glide_state_ids[0])
    _, _, representative_stats = problem.raw_transition_candidates(representative)
    figures = {
        "one_state_feasible_transitions": figure_transitions(
            problem, representative, output_dir / "phase1_feasible_transitions.png",
        ),
        "angular_transitions": figure_angles(
            problem, representative, output_dir / "phase1_angular_transitions.png",
        ),
        "terrain_feasibility": figure_terrain_feasibility(
            problem, representative, output_dir / "phase1_terrain_feasibility.png",
        ),
        "sensor_hazard_overlay": figure_hazard_overlay(
            problem, representative, output_dir / "phase1_sensor_hazard_overlay.png",
        ),
        "feasible_rollout": figure_rollout(
            problem, bellman, output_dir / "phase1_feasible_rollout.png",
        ),
        "transition_parity": figure_parity(
            parity, output_dir / "phase1_parity_diagnostic.png",
        ),
    }

    manifest: dict[str, Any] = {
        "phase": 1,
        "stage": "16.1",
        "purpose": "unified_attacker_br_environment_and_interface",
        "canonical_cube_case": {
            "phase0_manifest": _relative(PHASE0_DIR / "phase0_discovery_manifest.json"),
            "condition": condition.as_dict(),
            "sensor_map": list(problem.sensor_map),
            "scene_sizes": scene.sizes(),
        },
        "common_environment_source": {
            "file": "3D_RL_DQN/attacker_br_problem.py",
            "symbol": "AttackerBRProblem",
            "gymnasium_dependency": False,
            "local_sse_radius_in_attacker_problem": False,
        },
        "bellman_adapter_source": {
            "file": "3D_RL_DQN/attacker_br_problem.py",
            "symbol": "BellmanAttackerBRAdapter",
            "legacy_entry_point": "3D_0827/P1b_Exact_Local_SSE.py:exact_best_response",
        },
        "tabular_adapter_source": {
            "file": "3D_RL_DQN/attacker_br_problem.py",
            "symbol": "TabularQAttackerBRAdapter",
            "legacy_entry_point": "3D_0827/P1b_RL_approximation.py:learned_best_response",
            "configuration": q_config.as_dict(),
        },
        "common_solver_call": "3D_RL_DQN/attacker_br_problem.py:solve_attacker_br",
        "state_representation_sources": [
            "3D_0827/bellman_state.py:BellmanState",
            "3D_0827/bellman_state.py:BellmanStateGrid.encode/decode",
        ],
        "transition_logic_source": [
            "3D_0827/bellman_geometry.py:GlideTransitionModel.successors",
            "3D_0827/sparse_reachability.py:ImplicitReachableAdjacency",
        ],
        "bellman_backward_access": "3D_0827/sparse_reachability.py:build_goal_backward_reachable_graph remains unchanged behind the Bellman adapter",
        "feasibility_logic_source": "3D_0827/bellman_geometry.py:GlideTransitionModel.successors",
        "terminal_logic_source": "3D_0827/bellman_state.py:is_goal_terminal via scene.graph.terminal_mask",
        "cost_objective_source": [
            "3D_0827/P1b_Exact_Local_SSE.py:_phase_cost",
            "3D_0827/P1b_RL_approximation.py:AttackerMDP.cost_row/evaluate_trajectory",
        ],
        "J_D_ownership": "not part of AttackerBRProblem; adapters preserve the existing detection probability only so the unchanged Local-SSE contract can consume the backend result",
        "legacy_transition_logic_was_duplicated": False,
        "legacy_cost_evaluation_has_two_execution_paths": True,
        "cost_path_decision": "retained both proven-equivalent legacy paths; no solver algorithm was refactored",
        "representative_state": {
            "state_id": representative,
            "state": {
                "x_index": problem.state(representative).x_index,
                "y_index": problem.state(representative).y_index,
                "altitude_index": problem.state(representative).altitude_index,
                "heading_bin": problem.state(representative).heading_bin,
            },
            "position_map": problem.position_map(representative).tolist(),
            "transition_statistics": representative_stats,
        },
        "solver_results": {
            "bellman": bellman.as_dict(),
            "tabular_q": tabular.as_dict(),
        },
        "parity_test_summary": parity,
        "visualizations": figures,
        "existing_gym_adapter_boundary": {
            "file": "3D_0827/glider_gym_env.py",
            "used_as_phase1_core": False,
            "reason": "it contains later-phase observation, fixed-action, reward, and invalid-action-policy decisions",
            "status": "retained unchanged for compatibility; Phase 2/3 must review it against the completed common core",
        },
        "files_changed": [
            "3D_RL_DQN/README.md",
            "3D_RL_DQN/attacker_br_problem.py",
            "3D_RL_DQN/test_attacker_br_problem.py",
            "3D_RL_DQN/phase1_validation.py",
        ],
        "automated_test_source": "3D_RL_DQN/test_attacker_br_problem.py",
        "unresolved_mismatches": [] if parity["passed"] else parity["mismatches"],
        "unresolved_model_facts": [
            "The current state contains heading psi only. Gamma is derived per edge from altitude loss and horizontal distance; there is no independent gamma state or discretization to expose.",
        ],
        "restrictions_confirmed": {
            "DQN_implemented": False,
            "DQN_observation_designed": False,
            "neural_network_added": False,
            "multi_terrain_training_run": False,
            "J_A_or_J_D_changed": False,
            "Local_SSE_search_changed": False,
            "GPU_used": False,
        },
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Phase 1 summary", flush=True)
    print(f"  Bellman J_A={bellman.attacker_objective:.12f}; {bellman.runtime_sec:.3f}s", flush=True)
    print(f"  Tabular J_A={tabular.attacker_objective:.12f}; {tabular.runtime_sec:.3f}s", flush=True)
    print(f"  parity: {parity['mismatch_total']} mismatch(es), passed={parity['passed']}", flush=True)
    print(f"  manifest: {_relative(manifest_path)}", flush=True)
    for name, path in figures.items():
        print(f"  {name}: {path}", flush=True)
    return manifest


if __name__ == "__main__":
    run_phase1()

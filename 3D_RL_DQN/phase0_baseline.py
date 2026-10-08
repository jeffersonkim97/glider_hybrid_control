"""Freeze the existing Bellman-versus-tabular P1b development baseline.

This file contains orchestration, validation, serialization, and plotting only.
All dynamics, costs, terrain queries, learning, Bellman solves, and Local SSE
searches are imported from the authoritative ``3D_0827`` implementation.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import project_paths
from P1b_Exact_Local_SSE import exact_best_response, exact_local_sse
from P1b_RL_approximation import (
    AttackerMDP,
    QLearningConfig,
    candidate_start_states,
    default_sensor,
    evaluate_trajectory,
    rl_local_sse,
    solve_glide_mdp,
)
from P1b_comparison_figures import compare_solutions
from P1b_condition import ComputationCondition, Scene, build_scene


OUTPUT_DIR = project_paths.CORE_DIR / "figure" / "phase_0_baseline"
MANIFEST_NAME = "phase0_discovery_manifest.json"
RESULTS_NAME = "phase0_baseline_results.json"


def canonical_condition() -> ComputationCondition:
    """Return the defaults used by ``P1b_RL_figures.run_figures``."""

    return ComputationCondition(
        spatial_resolution_m=100.0,
        heading_spacing_deg=5.0,
        r_neighbor=1,
        terrain_category="centered_cube",
    )


def _relative(path: Path) -> str:
    return path.resolve().relative_to(project_paths.WORKSPACE_ROOT.resolve()).as_posix()


def _positions(scene: Scene, state_ids: Iterable[int]) -> np.ndarray:
    ids = tuple(int(value) for value in state_ids)
    if not ids:
        return np.empty((0, 3), dtype=float)
    return np.asarray(
        [scene.grid.position_map(scene.grid.decode(state_id)) for state_id in ids],
        dtype=float,
    )


def _full_path(scene: Scene, state_ids: Iterable[int]) -> np.ndarray:
    glide = _positions(scene, state_ids)
    if not len(glide):
        return glide
    return np.vstack((np.asarray(scene.config.start.as_array(), dtype=float), glide))


def _trajectory_is_feasible(scene: Scene, state_ids: Iterable[int]) -> bool:
    ids = tuple(int(value) for value in state_ids)
    if not ids or not scene.graph.terminal_mask[ids[-1]]:
        return False
    if any(not scene.graph.node_mask[state_id] for state_id in ids):
        return False
    return all(
        any(int(edge.target_id) == target for edge in scene.graph.adjacency[source])
        for source, target in zip(ids, ids[1:])
    )


def _add_terrain(ax: Any, scene: Scene) -> None:
    mesh = scene.terrain.obstacle_mesh()
    faces = [mesh.vertices[triangle] for triangle in mesh.triangles]
    ax.add_collection3d(Poly3DCollection(
        faces, facecolor="#b8b8b8", edgecolor="#666666", alpha=0.42,
        linewidth=0.5,
    ))
    bounds = scene.config.graph_bounds
    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_zlim(0.0, scene.grid.maximum_altitude_map)
    ax.set_xlabel("x [map unit]")
    ax.set_ylabel("y [map unit]")
    ax.set_zlabel("altitude [map unit]")


def _plot_trajectory(
    scene: Scene,
    paths: list[tuple[np.ndarray, str, str]],
    output: Path,
    title: str,
) -> str:
    figure = plt.figure(figsize=(8.5, 6.2))
    ax = figure.add_subplot(111, projection="3d")
    _add_terrain(ax, scene)
    for points, label, color in paths:
        if len(points):
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color,
                    linewidth=2.2, marker="o", markersize=3, label=label)
    start = scene.config.start.as_array()
    goal = scene.config.goal.as_array()
    ax.scatter(*start, color="#222222", s=45, marker="s", label="start")
    ax.scatter(*goal, color="#238443", s=60, marker="*", label="goal")
    ax.set_title(title)
    ax.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return _relative(output)


def _plot_metrics(results: dict[str, Any], output: Path) -> str:
    bellman = results["bellman"]
    tabular = results["tabular_q_learning"]
    labels = ["Bellman", "Tabular Q"]
    panels = [
        ("Attacker BR runtime [s]", [bellman["attacker_br_runtime_sec"],
                                     tabular["attacker_br_runtime_sec"]]),
        ("Local SSE runtime [s]", [bellman["local_sse_total_runtime_sec"],
                                    tabular["local_sse_total_runtime_sec"]]),
        ("Local SSE $J_A$ (min)", [bellman["J_A"], tabular["J_A"]]),
        ("Local SSE $J_D$ (max)", [bellman["J_D"], tabular["J_D"]]),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    colors = ["#1C5770", "#B3402F"]
    for ax, (title, values) in zip(axes.flat, panels):
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f"{value:.5g}", ha="center", va="bottom", fontsize=9)
    figure.suptitle("Phase 0 canonical cube baseline")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return _relative(output)


def build_discovery_manifest(
    scene: Scene, q_config: QLearningConfig, output_dir: Path,
) -> dict[str, Any]:
    sensor = default_sensor(scene)
    return {
        "phase": 0,
        "purpose": "existing_baseline_freeze_and_repository_discovery",
        "repository": {
            "workspace_root": _relative(project_paths.WORKSPACE_ROOT),
            "bellman_root": _relative(project_paths.CORE_DIR),
            "dqn_root": _relative(project_paths.DQN_DIR),
            "result_directory": _relative(output_dir),
        },
        "entry_points": {
            "bellman_br": {
                "file": "3D_0827/P1b_Exact_Local_SSE.py",
                "symbol": "exact_best_response",
                "status": "reused",
            },
            "tabular_q_br": {
                "file": "3D_0827/P1b_RL_approximation.py",
                "symbol": "solve_glide_mdp / learned_best_response",
                "status": "reused",
            },
            "local_sse": {
                "file": "3D_0827/P1b_Exact_Local_SSE.py; 3D_0827/P1b_RL_approximation.py",
                "symbol": "exact_local_sse / rl_local_sse",
                "status": "reused",
            },
            "bellman_vs_tabular_baseline": {
                "file": "3D_RL_DQN/phase0_baseline.py",
                "symbol": "run_phase0",
                "status": "wrapped",
                "existing_source_entry_point": "3D_0827/P1b_RL_figures.py:run_figures",
            },
        },
        "bellman_attacker_br": {
            "solver": "3D_0827/sparse_additive.py:solve_sparse_additive_bellman",
            "state": "3D_0827/bellman_state.py:BellmanState(x_index, y_index, altitude_index, heading_bin)",
            "transition_generation": "3D_0827/sparse_reachability.py:build_goal_backward_reachable_graph and ImplicitReachableAdjacency",
            "terminal_condition": "3D_0827/bellman_state.py:is_goal_terminal; exposed as scene.graph.terminal_mask",
            "stage_cost": "3D_0827/P1b_Exact_Local_SSE.py:_phase_cost",
            "trajectory_reconstruction": "3D_0827/additive_bellman.py:AdditiveBellmanSolution.path_edges",
            "J_A_path": "3D_0827/P1b_Exact_Local_SSE.py:exact_best_response",
            "runtime": "ExactBestResponse.timing[T_total_s] and external wall clock",
        },
        "tabular_q_learning": {
            "source_file": "3D_0827/P1b_RL_approximation.py",
            "training_symbol": "train",
            "state": "shared integer Bellman state_id",
            "action": "index into scene.graph.adjacency[state_id]",
            "feasible_actions": "AttackerMDP.actions reads the shared feasible adjacency",
            "transition": "AttackerMDP.step",
            "update_convention": "cost minimization: c + gamma * min Q",
            "stage_cost": "AttackerMDP.cost_row uses the same hazard/time terms as Bellman",
            "policy_extraction": "greedy_trajectory and _switching_readout",
            "J_A_path": "evaluate_trajectory plus powered switching cost",
            "hyperparameters": {
                "episodes": q_config.episodes,
                "learning_rate_alpha": q_config.alpha,
                "discount_factor_gamma": q_config.gamma,
                "epsilon_start": q_config.epsilon,
                "epsilon_min": q_config.epsilon_min,
                "epsilon_schedule": f"multiplicative decay {q_config.epsilon_decay} once per episode, clamped at epsilon_min",
                "random_seed": q_config.seed,
                "q_initialization": q_config.initial_q,
                "max_steps_per_episode": q_config.max_steps,
                "evaluation_interval_episodes": q_config.evaluation_interval,
                "stopping_rule": "fixed episode budget; no early-convergence stop",
                "local_sse_warm_start": "one shared QTable; first position uses full episode budget, later positions default to episodes // 5",
                "retry_or_restart_logic": "none",
            },
        },
        "local_sse": {
            "search": "3D_0827/local_sse_search.py:run_local_sse_search",
            "defender_strategy": "3D_0827/defender_grid_2d.py:DefenderGrid2D integer action_id mapped to (x, y, ground_z)",
            "initialization": "scene.seed_action_id, nearest defender-grid action to Stage11Config.sensor",
            "neighbors": "3D_0827/local_sse_contract.py:DefenderGridTopology.neighbors",
            "distance": "2-D Chebyshev grid-index distance <= r_neighbor, excluding center",
            "neighbor_radius_parameter": "ComputationCondition.r_neighbor -> DefenderNeighborhoodConfig.r_neighbor",
            "stopping": "no strictly improving feasible neighbor after required neighbor evaluation; exact path certifies local SSE",
            "strong_tie_breaking": "exact_best_response maximizes detection probability among attacker co-optima, then uses lowest state_id",
            "attacker_br_invocation": "exact_best_response or learned_best_response",
            "J_D_path": "cumulative glide hazard -> hazard_to_detection_probability",
        },
        "authoritative_components": {
            "dynamics": {"file": "3D_0827/bellman_geometry.py", "symbol": "GlideTransitionModel"},
            "transition_generation": {"file": "3D_0827/sparse_reachability.py", "symbol": "build_goal_backward_reachable_graph"},
            "feasibility": {"file": "3D_0827/bellman_geometry.py", "symbol": "GlideTransitionModel.successors"},
            "terrain_query": {"file": "3D_0827/map_geometry.py", "symbol": "TerrainModel / CompositeTerrainMap"},
            "sensing_hazard": {"file": "3D_0827/detection_hazard.py", "symbol": "GlideDetectionHazardModel"},
            "J_A": {
                "file": "3D_0827/P1b_Exact_Local_SSE.py",
                "symbol": "_phase_cost",
                "definition": "w_H * H / H_ref + w_T * T / T_ref",
                "weights_and_normalization": scene.config.as_dict()["attacker_objective"],
                "sign_convention": "attacker minimizes J_A",
            },
            "J_D": {
                "file": "3D_0827/detection_hazard.py",
                "symbol": "hazard_to_detection_probability",
                "definition": "1 - exp(-H)",
                "sign_convention": "defender maximizes J_D",
            },
            "strong_sse_tie_breaking": {
                "file": "3D_0827/P1b_Exact_Local_SSE.py",
                "symbol": "exact_best_response",
            },
        },
        "canonical_cube_case": {
            "source_file": "3D_0827/P1b_RL_figures.py",
            "source_symbol": "run_figures",
            "configuration": {
                "condition": scene.condition.as_dict(),
                "condition_label": scene.condition.label,
                "terrain": scene.terrain.category_id,
                "attacker_start_map": scene.config.start.as_array().tolist(),
                "attacker_goal_map": scene.config.goal.as_array().tolist(),
                "sensor_map": list(sensor),
                "graph_bounds": scene.config.as_dict()["graph_bounds"],
                "discretization": scene.config.discretization.as_dict(),
                "scene_sizes": scene.sizes(),
            },
            "scope": "development_regression_only",
        },
        "future_condition_domain": {
            "spatial_discretization_m_dx_eq_dy_eq_dh": [100, 50, 25, 12.5, 10, 5],
            "angular_discretization_deg_dpsi_eq_dgamma": [1.25, 2.5, 5, 10],
            "local_sse_neighbor_radius": [1, 2, 3, 4, 5],
            "phase0_full_grid_executed": False,
        },
        "cross_directory_import": {
            "3D_RL_DQN_can_reuse_3D_0827": True,
            "method": "import 3D_RL_DQN/project_paths.py, which prepends both sibling source directories to sys.path; the workspace .venv also has a local .pth convenience file",
            "portable_repository_file": "3D_RL_DQN/project_paths.py",
            "copied_authoritative_code": False,
            "circular_imports_introduced": False,
            "reusable_interfaces": [
                "P1b_condition.build_scene", "bellman_geometry.GlideTransitionModel",
                "sparse_reachability.build_goal_backward_reachable_graph",
                "map_geometry.TerrainModel", "detection_hazard.GlideDetectionHazardModel",
                "P1b_Exact_Local_SSE._phase_cost", "detection_hazard.hazard_to_detection_probability",
                "local_sse_search.run_local_sse_search",
            ],
            "blockers": [],
        },
        "runtime_scope": {
            "required": ["attacker_br_runtime_sec", "local_sse_total_runtime_sec", "J_A", "J_D"],
            "excluded_in_phase0": ["CPU utilization", "RAM utilization", "GPU utilization"],
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "unresolved": [
            "The current BellmanState has heading psi but no independent glide-scope-angle gamma state; therefore dgamma is future-domain metadata only and has no current canonical value to report.",
        ],
    }


def run_phase0(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    condition = canonical_condition()
    q_config = QLearningConfig()
    scene = build_scene(condition)
    sensor = default_sensor(scene)

    manifest = build_discovery_manifest(scene, q_config, output_dir)
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Phase 0 condition: {condition.label}; terrain={condition.terrain_category}", flush=True)
    print("Running fixed-sensor Bellman and Tabular Q attacker best responses...", flush=True)
    fixed_started = perf_counter()
    fixed = solve_glide_mdp(
        scene,
        sensor,
        q_config,
        start_state_ids=candidate_start_states(scene, sensor),
        read_switching=True,
    )
    fixed_total_seconds = perf_counter() - fixed_started
    if fixed.switching is None or fixed.switching.objective is None:
        raise RuntimeError("Tabular Q-learning did not produce a full best response")

    bellman_ids = fixed.bellman_trajectory
    tabular_ids = fixed.switching.trajectory
    tabular_eval = evaluate_trajectory(
        AttackerMDP(scene, sensor), tabular_ids, reached_goal=True,
    )
    tabular_br_runtime = (
        fixed.training_seconds
        + float(fixed.extra.get("switching_inference_seconds", fixed.inference_seconds))
    )

    print("Running Bellman Local SSE...", flush=True)
    exact_local_started = perf_counter()
    exact_sse = exact_local_sse(scene)
    exact_local_wall = perf_counter() - exact_local_started
    if not exact_sse.feasible:
        raise RuntimeError(f"Bellman Local SSE failed: {exact_sse.termination_status}")

    print("Running Tabular Q-learning Local SSE...", flush=True)
    tabular_local_started = perf_counter()
    learned_sse = rl_local_sse(scene, q_config, progress=True)
    tabular_local_wall = perf_counter() - tabular_local_started
    if not learned_sse.feasible:
        raise RuntimeError(f"Tabular Local SSE failed: {learned_sse.termination_status}")

    comparison = compare_solutions(scene, exact_sse, learned_sse)
    exact_sse_response = exact_best_response(
        scene, tuple(float(value) for value in exact_sse.selected_sensor_map),
        keep_solution=False,
    )

    bellman_png = _plot_trajectory(
        scene, [(_full_path(scene, bellman_ids), "Bellman", "#1C5770")],
        output_dir / "phase0_bellman_trajectory.png",
        "Bellman attacker trajectory — canonical cube",
    )
    tabular_png = _plot_trajectory(
        scene, [(_full_path(scene, tabular_ids), "Tabular Q", "#B3402F")],
        output_dir / "phase0_tabular_q_trajectory.png",
        "Tabular Q-learning attacker trajectory — canonical cube",
    )
    comparison_png = _plot_trajectory(
        scene,
        [
            (_full_path(scene, bellman_ids), "Bellman", "#1C5770"),
            (_full_path(scene, tabular_ids), "Tabular Q", "#B3402F"),
        ],
        output_dir / "phase0_bellman_vs_tabular_trajectories.png",
        "Bellman vs Tabular Q — canonical cube",
    )

    results: dict[str, Any] = {
        "phase": 0,
        "purpose": "development_regression_only",
        "scenario": {
            "name": "canonical_cube_tabular_comparison",
            "source": "3D_0827/P1b_RL_figures.py:run_figures defaults",
            "terrain": condition.terrain_category,
            "sensor_map": list(sensor),
            "spatial_discretization_m": condition.spatial_resolution_m,
            "angular_discretization_deg": condition.heading_spacing_deg,
            "neighbor_radius_r": condition.r_neighbor,
        },
        "bellman": {
            "status": "success",
            "attacker_br_runtime_sec": fixed.bellman_seconds,
            "local_sse_total_runtime_sec": exact_local_wall,
            "J_A": exact_sse.attacker_objective,
            "J_D": exact_sse.detection_probability,
            "fixed_sensor_J_A": fixed.extra["oracle_objective"],
            "fixed_sensor_J_D": fixed.extra["oracle_detection_probability"],
            "trajectory_feasible": _trajectory_is_feasible(scene, bellman_ids),
            "trajectory_state_ids": list(bellman_ids),
            "trajectory_output": bellman_png,
            "selected_local_sse_sensor_map": exact_sse.selected_sensor_map,
            "local_sse_termination_status": exact_sse.termination_status,
            "local_sse_verified": True,
            "timing_diagnostics": exact_sse.timing,
        },
        "tabular_q_learning": {
            "status": "success",
            "random_seed": q_config.seed,
            "attacker_br_runtime_sec": tabular_br_runtime,
            "local_sse_total_runtime_sec": tabular_local_wall,
            "J_A": learned_sse.attacker_objective,
            "J_D": learned_sse.detection_probability,
            "fixed_sensor_J_A": fixed.switching.objective,
            "fixed_sensor_J_D": float(-np.expm1(-tabular_eval.cumulative_hazard)),
            "trajectory_feasible": _trajectory_is_feasible(scene, tabular_ids),
            "trajectory_state_ids": list(tabular_ids),
            "trajectory_output": tabular_png,
            "selected_local_sse_sensor_map": learned_sse.selected_sensor_map,
            "local_sse_termination_status": learned_sse.termination_status,
            "local_sse_verified": learned_sse.local_sse_verified,
            "timing_diagnostics": {
                "training_sec": fixed.training_seconds,
                "single_start_query_sec": fixed.inference_seconds,
                "full_switching_policy_query_sec": fixed.extra["switching_inference_seconds"],
                "fixed_comparison_total_wall_sec": fixed_total_seconds,
                "local_sse": learned_sse.timing,
            },
            "training": learned_sse.training,
        },
        "comparison": comparison.as_dict(),
        "visualization": {
            "bellman_trajectory": bellman_png,
            "tabular_q_trajectory": tabular_png,
            "trajectory_comparison": comparison_png,
            "compact_metrics": None,
        },
        "validation": {
            "same_scene_object_for_fixed_br": True,
            "bellman_trajectory_feasible": _trajectory_is_feasible(scene, bellman_ids),
            "tabular_trajectory_feasible": _trajectory_is_feasible(scene, tabular_ids),
            "bellman_local_sse_replayed_J_A": exact_sse_response.attacker_objective,
            "bellman_local_sse_replayed_J_D": exact_sse_response.detection_probability,
        },
        "notes": [
            "Top-level J_A and J_D are each method's Local SSE outputs.",
            "Fixed-sensor values use the same canonical sensor and isolate the attacker best response.",
            "The learned Local SSE is explicitly approximate and therefore does not claim exact local-SSE certification.",
            "CPU, RAM, and GPU utilization are intentionally outside Phase 0 scope.",
        ],
    }
    metric_png = _plot_metrics(results, output_dir / "phase0_metric_comparison.png")
    results["visualization"]["compact_metrics"] = metric_png

    results_path = output_dir / RESULTS_NAME
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nPhase 0 repository discovery", flush=True)
    print("  Bellman BR: 3D_0827/P1b_Exact_Local_SSE.py:exact_best_response", flush=True)
    print("  Tabular Q BR: 3D_0827/P1b_RL_approximation.py:solve_glide_mdp", flush=True)
    print("  Local SSE: exact_local_sse / rl_local_sse", flush=True)
    print("  canonical case: 3D_0827/P1b_RL_figures.py:run_figures defaults", flush=True)
    print("  J_A: P1b_Exact_Local_SSE.py:_phase_cost", flush=True)
    print("  J_D: detection_hazard.py:hazard_to_detection_probability", flush=True)
    print(f"Q-learning configuration: {q_config.as_dict()}", flush=True)
    print("Baseline execution", flush=True)
    print(f"  Bellman success; J_A={exact_sse.attacker_objective:.9f}; J_D={exact_sse.detection_probability:.9f}; BR={fixed.bellman_seconds:.3f}s; Local SSE={exact_local_wall:.3f}s", flush=True)
    print(f"  Tabular success; J_A={learned_sse.attacker_objective:.9f}; J_D={learned_sse.detection_probability:.9f}; BR={tabular_br_runtime:.3f}s; Local SSE={tabular_local_wall:.3f}s", flush=True)
    print("Files generated", flush=True)
    for path in (manifest_path, results_path):
        print(f"  {_relative(path)}", flush=True)
    for path in results["visualization"].values():
        print(f"  {path}", flush=True)
    print("Unresolved items", flush=True)
    for item in manifest["unresolved"]:
        print(f"  {item}", flush=True)
    return results


if __name__ == "__main__":
    run_phase0()

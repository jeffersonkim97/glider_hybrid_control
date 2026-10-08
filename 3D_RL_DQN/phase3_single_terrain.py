"""Run and visualize the approved Phase 16.3 single-terrain DQN sanity study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import project_paths
from P1b_Exact_Local_SSE import exact_best_response
from P1b_RL_approximation import default_sensor
from P1b_condition import ComputationCondition, build_scene
from attacker_br_problem import AttackerBRProblem
from dqn_action_catalog import HeadingActionCatalog
from dqn_env import ObservationTensorCache
from dqn_model import DQNNetworkConfig
from dqn_training import (
    CompatibilitySignature,
    DQNTrainingConfig,
    checkpoint_payload,
    epsilon_greedy_action,
    load_compatible_checkpoint,
    save_checkpoint,
    train_one_seed,
    write_history,
)
from terrain_observation import ObservationConfig, TerrainObservationBuilder


OUTPUT_DIR = project_paths.CORE_DIR / "figure" / "phase_3_single_terrain_dqn"
CHECKPOINT_DIR = project_paths.DQN_DIR / "checkpoints" / "phase3_single_terrain"
MANIFEST_NAME = "phase3_single_terrain_dqn_manifest.json"
CONFIG_NAME = "phase3_dqn_config.json"
ACTION_CATALOG_NAME = "phase3_action_catalog.json"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(project_paths.WORKSPACE_ROOT.resolve()).as_posix()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _save(figure: Any, output: Path) -> str:
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return _relative(output)


def _rolling_mean(values: list[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return array
    width = min(int(window), len(array))
    kernel = np.ones(width, dtype=float) / width
    valid = np.convolve(array, kernel, mode="valid")
    prefix = np.full(width - 1, np.nan)
    return np.concatenate((prefix, valid))


def _load_history(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def figure_training_returns(histories: list[dict[str, Any]], output: Path) -> str:
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
    for item in histories:
        seed = item["seed"]
        history = item["history"]
        episodes = np.asarray(history["episode"])
        returns = history["episode_full_transformed_return"]
        success = np.asarray(history["episode_reached_goal"], dtype=float)
        axes[0].plot(
            episodes, _rolling_mean(returns, 200), label=f"seed {seed}", linewidth=1.5,
        )
        axes[1].plot(
            episodes, _rolling_mean(success.tolist(), 200),
            label=f"seed {seed}", linewidth=1.5,
        )
    axes[0].set_ylabel("rolling transformed return\n$-C_{powered}+\\sum r_t$")
    axes[0].set_title("Training objective history (200-episode rolling mean)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("training episode")
    axes[1].set_ylabel("goal success fraction")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    return _save(figure, output)


def figure_td_loss(histories: list[dict[str, Any]], output: Path) -> str:
    figure, ax = plt.subplots(figsize=(10.5, 5.2))
    for item in histories:
        history = item["history"]
        steps = np.asarray(history["optimization_step"])
        loss = _rolling_mean(history["td_loss"], 500)
        ax.plot(steps, loss, label=f"seed {item['seed']}", linewidth=1.3)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Huber TD loss (500-update rolling mean)")
    ax.set_yscale("log")
    ax.set_title("Masked DQN TD-loss history")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(figure, output)


def figure_evaluation_success(histories: list[dict[str, Any]], output: Path) -> str:
    figure, ax = plt.subplots(figsize=(10.5, 4.8))
    offsets = np.linspace(-45.0, 45.0, len(histories))
    for offset, item in zip(offsets, histories):
        history = item["history"]
        episodes = np.asarray(history["evaluation_episode"], dtype=float) + offset
        successes = np.asarray(history["evaluation_success"], dtype=float)
        ax.step(episodes, successes, where="mid", label=f"seed {item['seed']}")
    ax.set_xlabel("training episode")
    ax.set_ylabel("deterministic evaluation success")
    ax.set_yticks((0, 1), ("failure", "goal reached"))
    ax.set_ylim(-0.15, 1.15)
    ax.set_title("Evaluation goal success over training")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(figure, output)


def figure_J_A_gap(histories: list[dict[str, Any]], output: Path) -> str:
    figure, ax = plt.subplots(figsize=(10.5, 5.2))
    for item in histories:
        history = item["history"]
        episodes = np.asarray(history["evaluation_episode"])
        relative = np.asarray([
            np.nan if value is None else 100.0 * float(value)
            for value in history["evaluation_relative_error"]
        ])
        ax.plot(episodes, relative, marker="o", markersize=3, label=f"seed {item['seed']}")
    ax.axhline(10.0, color="#009E73", linestyle="--", label="approved median ≤10%")
    ax.axhline(20.0, color="#D55E00", linestyle=":", label="approved per-seed ≤20%")
    ax.set_xlabel("training episode")
    ax.set_ylabel("absolute Bellman-relative $J_A$ error [%]")
    ax.set_title("Authoritative Bellman-relative objective history")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    return _save(figure, output)


def _add_terrain_top(ax: Any, problem: AttackerBRProblem) -> None:
    for box in problem.scene.terrain.obstacle_boxes():
        ax.add_patch(plt.Rectangle(
            (box.x_limits[0], box.y_limits[0]), box.width_x, box.width_y,
            facecolor="#bdbdbd", edgecolor="#333333", alpha=0.65,
        ))
    bounds = problem.grid.bounds
    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [map unit]")
    ax.set_ylabel("y [map unit]")


def figure_trajectories(
    problem: AttackerBRProblem,
    bellman_trajectory: tuple[int, ...],
    dqn_trajectory: tuple[int, ...],
    seed: int,
    output: Path,
) -> str:
    bellman = np.asarray([problem.position_map(value) for value in bellman_trajectory])
    dqn = np.asarray([problem.position_map(value) for value in dqn_trajectory])
    goal = problem.scene.config.goal.as_array()
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    _add_terrain_top(axes[0], problem)
    axes[0].plot(bellman[:, 0], bellman[:, 1], "o-", color="#0072B2", label="Bellman")
    axes[0].plot(dqn[:, 0], dqn[:, 1], "s--", color="#D55E00", label=f"DQN seed {seed}")
    axes[0].scatter(goal[0], goal[1], marker="*", s=130, color="#009E73", label="goal")
    axes[0].set_title("Top view")
    axes[0].legend()
    bellman_distance = np.arange(len(bellman))
    dqn_distance = np.arange(len(dqn))
    axes[1].plot(bellman_distance, bellman[:, 2], "o-", color="#0072B2", label="Bellman")
    axes[1].plot(dqn_distance, dqn[:, 2], "s--", color="#D55E00", label=f"DQN seed {seed}")
    axes[1].axhline(goal[2], color="#009E73", linestyle=":", label="goal altitude")
    axes[1].set_xlabel("trajectory state index")
    axes[1].set_ylabel("altitude [map unit]")
    axes[1].set_title("Altitude progression")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Canonical cube: Bellman vs deterministic DQN trajectory")
    return _save(figure, output)


def figure_action_mask(
    problem: AttackerBRProblem,
    builder: TerrainObservationBuilder,
    catalog: HeadingActionCatalog,
    network: Any,
    device: torch.device,
    state_id: int,
    output: Path,
) -> str:
    cache = ObservationTensorCache(builder)
    observation = cache.get(state_id)
    spatial = torch.from_numpy(observation.spatial.copy())[None, ...].to(device)
    scalar = torch.from_numpy(observation.scalar.copy())[None, ...].to(device)
    network = network.to(device)
    network.eval()
    with torch.inference_mode():
        q_values = network(spatial, scalar)[0].detach().cpu().numpy()
    mask = catalog.feasible_mask(state_id)
    selected = epsilon_greedy_action(q_values, mask, 0.0, np.random.default_rng(0))
    action_ids = np.arange(catalog.action_count)
    colors = np.where(mask, "#009E73", "#BDBDBD")
    figure, ax = plt.subplots(figsize=(13.0, 5.0))
    ax.bar(action_ids, mask.astype(int), color=colors, width=0.85)
    ax.bar(selected, 1.0, fill=False, edgecolor="#D55E00", linewidth=2.5, width=0.9)
    ax.set_xlabel("canonical action ID = target heading bin")
    ax.set_ylabel("feasible mask")
    ax.set_yticks((0, 1), ("infeasible", "feasible"))
    ax.set_xlim(-1, catalog.action_count)
    ax.set_title(
        f"Action-mask sanity at state {state_id}: "
        f"{int(mask.sum())}/72 feasible, selected action {selected}"
    )
    ax.grid(axis="x", alpha=0.15)
    return _save(figure, output)


def figure_summary(
    condition: ComputationCondition,
    bellman_J_A: float,
    bellman_runtime: float,
    summaries: list[dict[str, Any]],
    result_label: str,
    output: Path,
) -> str:
    rows = []
    for item in summaries:
        rows.append((
            str(item["seed"]),
            "yes" if item["success"] else "no",
            "yes" if item["trajectory_feasible"] else "no",
            "—" if item["dqn_J_A"] is None else f"{item['dqn_J_A']:.6f}",
            "—" if item["relative_error"] is None else f"{100*item['relative_error']:.2f}%",
            f"{item['training_runtime_sec']:.2f}",
            f"{1000*item['inference_runtime_sec']:.2f}",
        ))
    figure, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=(
            "seed", "goal", "feasible", "$J_A$", "relative error",
            "train [s]", "inference [ms]",
        ),
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax.set_title(
        f"Phase 16.3 {result_label} — "
        f"dx={condition.spatial_resolution_m:g} m, dpsi={condition.heading_spacing_deg:g}°, "
        f"Bellman J_A={bellman_J_A:.6f}, Bellman BR={bellman_runtime:.3f} s",
        pad=18,
    )
    return _save(figure, output)


def _reward_equivalence(problem: AttackerBRProblem, catalog: HeadingActionCatalog, oracle: Any) -> dict[str, Any]:
    trajectory = tuple(int(value) for value in oracle.glide_state_ids)
    reward = 0.0
    for source, target in zip(trajectory, trajectory[1:]):
        edge = next(
            edge for edge in problem.successors(source)
            if int(edge.target_id) == target
        )
        reward -= catalog.transition(source, int(edge.target_state.heading_bin)).stage_cost
    scored = problem.evaluate_trajectory(trajectory, reached_goal=True)
    powered = problem.powered_cost(int(oracle.switching_state_id))
    glide_error = abs(reward + scored.cost)
    full_error = abs((-powered + reward) + float(oracle.attacker_objective))
    return {
        "bellman_trajectory": list(trajectory),
        "summed_reward": reward,
        "authoritative_glide_cost": scored.cost,
        "powered_cost": powered,
        "authoritative_full_J_A": float(oracle.attacker_objective),
        "glide_identity_absolute_error": glide_error,
        "full_identity_absolute_error": full_error,
        "passed": glide_error <= 1.0e-12 and full_error <= 1.0e-12,
    }


def _condition_tag(spatial_resolution_m: float, heading_spacing_deg: float) -> str:
    def clean(value: float) -> str:
        return f"{float(value):g}".replace(".", "p")

    return f"dx{clean(spatial_resolution_m)}m_dpsi{clean(heading_spacing_deg)}deg"


def _action_audit_states(
    problem: AttackerBRProblem,
    starts: tuple[int, ...],
    bellman_trajectory: tuple[int, ...],
    *,
    maximum_sampled_states: int = 10_000,
) -> np.ndarray | None:
    reachable = np.flatnonzero(problem.scene.graph.node_mask)
    if len(reachable) <= 250_000:
        return None
    sample_indices = np.linspace(
        0, len(reachable) - 1,
        min(maximum_sampled_states, len(reachable)),
        dtype=np.int64,
    )
    required = np.concatenate((
        reachable[sample_indices],
        np.asarray(starts, dtype=np.int64),
        np.asarray(bellman_trajectory, dtype=np.int64),
        np.asarray(problem.scene.graph.terminal_ids, dtype=np.int64),
    ))
    return np.unique(required)


def run_phase3(
    *,
    smoke: bool = False,
    device_override: str | None = None,
    spatial_resolution_m: float = 100.0,
    heading_spacing_deg: float = 5.0,
) -> Path:
    canonical = (
        float(spatial_resolution_m) == 100.0
        and float(heading_spacing_deg) == 5.0
    )
    condition_root = (
        OUTPUT_DIR if canonical else
        OUTPUT_DIR / _condition_tag(spatial_resolution_m, heading_spacing_deg)
    )
    checkpoint_root = (
        CHECKPOINT_DIR if canonical else
        CHECKPOINT_DIR / _condition_tag(spatial_resolution_m, heading_spacing_deg)
    )
    output_dir = condition_root if not smoke else condition_root / "smoke"
    checkpoint_dir = checkpoint_root if not smoke else checkpoint_root / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    condition = ComputationCondition(
        spatial_resolution_m=float(spatial_resolution_m),
        heading_spacing_deg=float(heading_spacing_deg),
        r_neighbor=1,
        terrain_category="centered_cube",
    )
    scene = build_scene(condition)
    problem = AttackerBRProblem(scene, default_sensor(scene))
    bellman_started = perf_counter()
    oracle = exact_best_response(scene, problem.sensor_map, keep_solution=True)
    bellman_runtime = perf_counter() - bellman_started
    if not oracle.feasible:
        raise RuntimeError("canonical Bellman attacker BR is infeasible")
    starts = problem.switching_state_ids()
    builder = TerrainObservationBuilder(
        problem,
        ObservationConfig(
            local_window_extent_m=2000.0,
            local_window_shape_cells=(21, 21),
            local_window_orientation="heading_aligned",
        ),
    )
    catalog = HeadingActionCatalog(problem)
    network_config = DQNNetworkConfig(
        spatial_channels=builder.tensor_ready_shapes["spatial"][0],
        spatial_height=builder.tensor_ready_shapes["spatial"][1],
        spatial_width=builder.tensor_ready_shapes["spatial"][2],
        scalar_features=builder.tensor_ready_shapes["scalar"][0],
        action_count=catalog.action_count,
    )
    training_config = DQNTrainingConfig()
    if smoke:
        training_config = DQNTrainingConfig(
            episodes=20,
            replay_capacity=256,
            replay_warmup=16,
            batch_size=16,
            target_update_steps=20,
            evaluation_interval_episodes=10,
            seeds=(0,),
            device="cpu" if device_override is None else device_override,
        )
    elif device_override is not None:
        values = training_config.as_dict()
        values.pop("epsilon_decay_definition", None)
        values["device"] = device_override
        training_config = DQNTrainingConfig.from_dict(values)
    compatibility = CompatibilitySignature.from_components(problem, builder, catalog)
    config_path = output_dir / CONFIG_NAME
    _write_json(config_path, {
        "condition": condition.as_dict(),
        "observation_config": builder.config.as_dict(),
        "network_config": network_config.as_dict(),
        "training_config": training_config.as_dict(),
        "compatibility": compatibility.as_dict(),
        "approved_pass_criterion": {
            "seed_count": 3,
            "every_seed_goal_reaching_and_feasible": True,
            "median_relative_J_A_error_max": 0.10,
            "every_seed_relative_J_A_error_max": 0.20,
            "formula": "abs(J_A_DQN-J_A_Bellman)/abs(J_A_Bellman)",
        },
        "smoke_run": smoke,
    })
    audit_states = _action_audit_states(
        problem,
        starts,
        tuple(int(value) for value in oracle.glide_state_ids),
    )
    action_audit = catalog.audit_reachable_graph(audit_states)
    action_catalog_path = output_dir / ACTION_CATALOG_NAME
    _write_json(action_catalog_path, {
        **catalog.as_dict(), "reachable_graph_audit": action_audit,
    })
    if not action_audit["passed"]:
        raise RuntimeError("fixed action catalog failed full reachable-graph audit")
    reward_equivalence = _reward_equivalence(problem, catalog, oracle)
    if not reward_equivalence["passed"]:
        raise RuntimeError("reward transformation failed objective equivalence")

    histories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    checkpoint_paths: list[dict[str, str]] = []
    for seed in training_config.seeds:
        print(f"Phase 16.3 training seed {seed} on {training_config.device}", flush=True)
        run = train_one_seed(
            problem, builder, catalog, starts, float(oracle.attacker_objective),
            network_config, training_config, seed,
        )
        best_path = checkpoint_dir / f"phase3_seed{seed}_best.pt"
        final_path = checkpoint_dir / f"phase3_seed{seed}_final.pt"
        save_checkpoint(best_path, checkpoint_payload(
            model_state=run.best_model_state,
            network_config=network_config,
            training_config=training_config,
            compatibility=compatibility,
            seed=seed,
            episode=run.best_episode,
            checkpoint_kind="best_authoritative_evaluation",
        ))
        save_checkpoint(final_path, checkpoint_payload(
            model_state=run.final_model_state,
            network_config=network_config,
            training_config=training_config,
            compatibility=compatibility,
            seed=seed,
            episode=training_config.episodes,
            checkpoint_kind="final_training_state",
            optimizer_state=run.final_optimizer_state,
        ))
        # Load through the strict path before accepting either checkpoint.
        _, loaded = load_compatible_checkpoint(
            best_path, network_config, compatibility,
        )
        if loaded["seed"] != seed:
            raise RuntimeError("reloaded checkpoint seed metadata changed")
        history_path = output_dir / f"phase3_seed{seed}_history.json"
        write_history(history_path, run)
        history_payload = _load_history(history_path)
        histories.append(history_payload)
        evaluation = run.best_evaluation
        dqn_J_A = evaluation.attacker_objective
        absolute = (
            None if dqn_J_A is None
            else abs(float(dqn_J_A) - float(oracle.attacker_objective))
        )
        relative = (
            None if absolute is None
            else absolute / abs(float(oracle.attacker_objective))
        )
        summaries.append({
            "seed": seed,
            "device": run.device,
            "success": evaluation.success,
            "trajectory_feasible": evaluation.trajectory_feasible,
            "switching_state_id": evaluation.switching_state_id,
            "trajectory": list(evaluation.trajectory),
            "dqn_J_A": dqn_J_A,
            "absolute_error": absolute,
            "relative_error": relative,
            "best_episode": run.best_episode,
            "training_runtime_sec": run.training_runtime_sec,
            "inference_runtime_sec": evaluation.inference_runtime_sec,
            "environment_steps": run.environment_steps,
            "optimization_steps": run.optimization_steps,
            "replay_size": run.replay_size,
            "cached_observation_states": run.cached_observation_states,
            "history_path": _relative(history_path),
        })
        checkpoint_paths.append({
            "seed": str(seed),
            "best": _relative(best_path),
            "final": _relative(final_path),
        })
        print(json.dumps(summaries[-1], indent=2), flush=True)

    relative_errors = [
        float(item["relative_error"])
        for item in summaries if item["relative_error"] is not None
    ]
    all_success = bool(
        len(summaries) == len(training_config.seeds)
        and all(item["success"] and item["trajectory_feasible"] for item in summaries)
    )
    median_relative = median(relative_errors) if relative_errors else float("inf")
    maximum_relative = max(relative_errors) if relative_errors else float("inf")
    official_pass = bool(
        not smoke
        and len(relative_errors) == 3
        and all_success
        and median_relative <= 0.10
        and maximum_relative <= 0.20
    )

    representative = min(
        summaries,
        key=lambda item: float("inf") if item["relative_error"] is None
        else abs(float(item["relative_error"]) - median_relative),
    )
    representative_seed = int(representative["seed"])
    representative_path = checkpoint_dir / f"phase3_seed{representative_seed}_best.pt"
    representative_network, _ = load_compatible_checkpoint(
        representative_path, network_config, compatibility,
    )
    representative_device = torch.device("cpu")
    visualization_outputs = [
        figure_training_returns(histories, output_dir / "phase3_training_returns.png"),
        figure_td_loss(histories, output_dir / "phase3_td_loss.png"),
        figure_evaluation_success(histories, output_dir / "phase3_evaluation_success.png"),
        figure_J_A_gap(histories, output_dir / "phase3_bellman_relative_J_A.png"),
        figure_trajectories(
            problem, tuple(int(value) for value in oracle.glide_state_ids),
            tuple(int(value) for value in representative["trajectory"]),
            representative_seed, output_dir / "phase3_bellman_vs_dqn_trajectory.png",
        ),
        figure_action_mask(
            problem, builder, catalog, representative_network,
            representative_device, int(oracle.switching_state_id),
            output_dir / "phase3_action_mask.png",
        ),
        figure_summary(
            condition, float(oracle.attacker_objective), bellman_runtime,
            summaries,
            "SMOKE CHECK" if smoke else "PASS" if official_pass else "FAIL",
            output_dir / "phase3_result_summary.png",
        ),
    ]

    manifest = {
        "phase": 3,
        "stage": "16.3",
        "purpose": "single_terrain_dqn_sanity_check",
        "status": (
            "smoke_only" if smoke else
            "passed_approved_sanity_criterion" if official_pass
            else "failed_approved_sanity_criterion"
        ),
        "terrain_case": condition.terrain_category,
        "spatial_discretization_m": condition.spatial_resolution_m,
        "angular_discretization_deg": condition.heading_spacing_deg,
        "local_sse_radius_used_by_dqn": False,
        "observation_schema_id": builder.config.schema_id,
        "observation_shapes": {
            key: list(value) for key, value in builder.tensor_ready_shapes.items()
        },
        "action_catalog_id": catalog.catalog_id,
        "action_catalog_path": _relative(action_catalog_path),
        "action_catalog_audit": action_audit,
        "network_config": network_config.as_dict(),
        "network_parameter_count": representative_network.parameter_count,
        "training_config": training_config.as_dict(),
        "random_seeds": list(training_config.seeds),
        "device": {
            "requested": training_config.device,
            "actual_per_seed": [item["device"] for item in summaries],
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "reward_cost_equivalence": reward_equivalence,
        "bellman": {
            "J_A": float(oracle.attacker_objective),
            "switching_state_id": int(oracle.switching_state_id),
            "trajectory": list(oracle.glide_state_ids),
            "attacker_br_runtime_sec": bellman_runtime,
        },
        "dqn_seed_results": summaries,
        "approved_pass_criterion": {
            "all_three_seeds_goal_reaching_and_feasible": True,
            "median_relative_J_A_error_max": 0.10,
            "every_seed_relative_J_A_error_max": 0.20,
        },
        "criterion_evaluation": {
            "all_seeds_goal_reaching_and_feasible": all_success,
            "median_relative_J_A_error": (
                None if not np.isfinite(median_relative) else median_relative
            ),
            "maximum_relative_J_A_error": (
                None if not np.isfinite(maximum_relative) else maximum_relative
            ),
            "passed": official_pass,
            "evaluated_for_official_run": not smoke,
        },
        "checkpoint_paths": checkpoint_paths,
        "configuration_path": _relative(config_path),
        "visualization_paths": visualization_outputs,
        "automated_test_commands": [
            (
                ".venv_p1b\\Scripts\\python.exe -m unittest discover "
                "-s 3D_RL_DQN -p test_*.py -v"
            ),
            (
                "cd 3D_0827; ..\\.venv_p1b\\Scripts\\python.exe -m unittest "
                "test_P1b_equivalence test_baseline_regression "
                "test_glider_gym_env -v"
            ),
        ],
        "test_results_at_implementation": {
            "phase3_tests_run": 11,
            "phase3_tests_passed": 11,
            "phase1_through_phase3_dqn_tests_run": 32,
            "phase1_through_phase3_dqn_tests_passed": 32,
            "related_legacy_regression_tests_run": 27,
            "related_legacy_regression_tests_passed": 27,
            "historical_3D_0827_full_discovery_suite": (
                "not an applicable regression gate: it discovers tests for removed "
                "Stage 6-14 modules and absent generated-result fixtures"
            ),
        },
        "scope_limits": {
            "terrain_generalization_claim": False,
            "local_sse_integration": False,
            "final_runtime_regime_claim": False,
            "training_terrain_count": 1,
        },
        "files_changed": [
            "3D_RL_DQN/dqn_action_catalog.py",
            "3D_RL_DQN/dqn_env.py",
            "3D_RL_DQN/dqn_model.py",
            "3D_RL_DQN/dqn_replay.py",
            "3D_RL_DQN/dqn_training.py",
            "3D_RL_DQN/phase3_single_terrain.py",
            "3D_RL_DQN/test_dqn_phase3.py",
            "3D_RL_DQN/README.md",
            "3D_0827/docs/stage16_dqn_terrain_generalization_plan.md",
            "SETUP.md",
            "p1b/requirements.txt",
        ],
    }
    manifest_path = output_dir / MANIFEST_NAME
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "manifest": _relative(manifest_path),
        "status": manifest["status"],
        "bellman_J_A": manifest["bellman"]["J_A"],
        "median_relative_error": manifest["criterion_evaluation"]["median_relative_J_A_error"],
        "maximum_relative_error": manifest["criterion_evaluation"]["maximum_relative_J_A_error"],
    }, indent=2), flush=True)
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="run a nonofficial 20-episode CPU pipeline check",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default=None,
        help="override the approved auto device selection",
    )
    parser.add_argument(
        "--spatial-resolution-m", type=float, default=100.0,
        help="condition-specific dx=dy=dh in meters (default: 100)",
    )
    parser.add_argument(
        "--heading-spacing-deg", type=float, default=5.0,
        help="condition-specific heading spacing in degrees (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_phase3(
        smoke=arguments.smoke,
        device_override=arguments.device,
        spatial_resolution_m=arguments.spatial_resolution_m,
        heading_spacing_deg=arguments.heading_spacing_deg,
    )

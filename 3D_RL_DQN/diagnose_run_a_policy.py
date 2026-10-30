"""Diagnose the trained Run A policy against the exact Bellman solution.

This is an evaluation-only utility.  It rebuilds the approved Run A scene,
loads a saved checkpoint, and separates the observed objective gap into the
switching-state and glide-policy components.  It also audits the learned
action ranking along the exact Bellman trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import project_paths
from dqn_action_catalog import HeadingActionCatalog, SharedActionRowCache
from dqn_env import ObservationTensorCache
from dqn_model import DQNNetworkConfig, TerrainDQN
from dqn_training import CompatibilitySignature
from multiscale_observation import (
    ExactBatchedAABBGPUObservationMultiscaleBuilder,
    MultiscaleObservationConfig,
    MultiscaleTerrainObservationBuilder,
)
from phase4_extended_dataset import (
    ExtendedPilotConfig,
    build_extended_scenario_bank,
    extended_dataset_manifest,
)
from phase4_extended_scene import build_extended_scene
from phase4_multi_terrain import (
    ScenarioRuntime,
    evaluate_policy_batched,
    load_phase4_checkpoint,
)
from phase4_v2_run_a import RunAGoalReachableProblem, _reference_for, _run_a_scenario
from sparse_additive import solve_sparse_additive_bellman


CHECKPOINT = (
    project_paths.DQN_DIR
    / "checkpoints"
    / "phase4_extended_random_boxes"
    / "training_v2"
    / "run_a_upper_bound"
    / "dx25m_dpsi5deg"
    / "double_dqn_b_v1"
    / "phase4_seed0_best.pt"
)
OUTPUT_ROOT = (
    project_paths.CORE_DIR
    / "figure"
    / "phase_4_extended_random_boxes"
    / "training_v2"
    / "run_a_upper_bound"
    / "dx25m_dpsi5deg"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _network_q(
    network: TerrainDQN,
    cache: ObservationTensorCache,
    state_id: int,
    device: torch.device,
) -> np.ndarray:
    observation = cache.get(int(state_id))
    # Cached observations are deliberately immutable.  Copy before creating a
    # tensor so PyTorch never receives a non-writable NumPy view.
    spatial = torch.from_numpy(observation.spatial[None, ...].copy()).to(
        device=device,
    )
    scalar = torch.from_numpy(observation.scalar[None, ...].copy()).to(
        device=device,
    )
    with torch.inference_mode():
        return network(spatial, scalar)[0].detach().cpu().numpy()


def _rollout_from(
    network: TerrainDQN,
    context: ScenarioRuntime,
    start_state_id: int,
    device: torch.device,
) -> dict[str, Any]:
    current = int(start_state_id)
    trajectory = [current]
    actions: list[int] = []
    reached_goal = context.problem.is_terminal(current)
    reason = "goal" if reached_goal else None
    horizon = max(1, int(context.problem.grid.altitude_count) - 1)
    for _ in range(horizon):
        if reached_goal:
            break
        row = context.catalog.row(current)
        feasible = np.flatnonzero(row.feasible_mask)
        if not len(feasible):
            reason = "dead_end"
            break
        q_values = _network_q(network, context.cache, current, device)
        action = int(feasible[np.argmax(q_values[feasible])])
        target = int(row.target_state_ids[action])
        if target < 0:
            raise RuntimeError("masked policy selected an infeasible action")
        actions.append(action)
        trajectory.append(target)
        reached_goal = bool(row.terminal[action])
        reason = "goal" if reached_goal else None
        current = target
    if reason is None:
        reason = "horizon"
    result: dict[str, Any] = {
        "start_state_id": int(start_state_id),
        "trajectory": trajectory,
        "action_ids": actions,
        "edge_count": len(actions),
        "reached_goal": bool(reached_goal),
        "termination_reason": reason,
    }
    if reached_goal:
        scored = context.problem.evaluate_trajectory(
            tuple(trajectory), reached_goal=True,
        )
        powered = context.problem.powered_cost(int(start_state_id))
        result.update({
            "powered_cost": float(powered),
            "glide_cost": float(scored.cost),
            "attacker_objective": float(powered + scored.cost),
        })
    return result


def _audit_states(
    label: str,
    state_ids: Iterable[int],
    network: TerrainDQN,
    context: ScenarioRuntime,
    solution: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    ids = tuple(int(value) for value in state_ids)
    records: list[dict[str, Any]] = []
    for step, state_id in enumerate(ids[:-1]):
        row = context.catalog.row(state_id)
        feasible = np.flatnonzero(row.feasible_mask)
        q_values = _network_q(network, context.cache, state_id, device)
        greedy_action = int(feasible[np.argmax(q_values[feasible])])
        bellman_action = int(solution.solution.policy_action_index[state_id])
        bellman_successor = int(solution.solution.policy_successor[state_id])
        if bellman_action < 0 or bellman_successor < 0:
            raise RuntimeError(f"state {state_id} lacks an exact Bellman action")
        if not row.feasible_mask[bellman_action]:
            raise RuntimeError("Bellman action is absent from the DQN action mask")

        exact_costs: dict[int, float] = {}
        for action_value in feasible:
            action = int(action_value)
            local_index = int(row.local_indices[action])
            target = int(row.target_state_ids[action])
            exact_costs[action] = float(
                solution.solution.edge_cost_by_source[state_id][local_index]
                + solution.solution.value[target]
            )
        bellman_q = float(q_values[bellman_action])
        greedy_q = float(q_values[greedy_action])
        q_rank = 1 + sum(
            float(q_values[action]) > bellman_q + 1.0e-12
            for action in feasible
        )
        greedy_exact_cost = exact_costs[greedy_action]
        exact_rank = 1 + sum(
            cost < greedy_exact_cost - 1.0e-12
            for cost in exact_costs.values()
        )
        state = context.problem.state(state_id)
        position = context.problem.position_map(state_id)
        records.append({
            "path": label,
            "step": step,
            "state_id": state_id,
            "x_map": float(position[0]),
            "y_map": float(position[1]),
            "altitude_map": float(position[2]),
            "heading_bin": int(state.heading_bin),
            "feasible_action_count": int(len(feasible)),
            "bellman_action_id": bellman_action,
            "bellman_target_state_id": bellman_successor,
            "dqn_greedy_action_id": greedy_action,
            "dqn_target_state_id": int(row.target_state_ids[greedy_action]),
            "actions_agree": greedy_action == bellman_action,
            "bellman_action_dqn_q_rank": int(q_rank),
            "dqn_action_exact_cost_rank": int(exact_rank),
            "dqn_q_greedy": greedy_q,
            "dqn_q_bellman_action": bellman_q,
            "dqn_q_margin_over_bellman": greedy_q - bellman_q,
            "dqn_predicted_greedy_cost_to_go": -greedy_q,
            "dqn_predicted_bellman_action_cost_to_go": -bellman_q,
            "exact_optimal_cost_to_go": float(solution.solution.value[state_id]),
            "exact_cost_if_dqn_action_then_bellman": greedy_exact_cost,
            "one_step_exact_regret": float(
                greedy_exact_cost - solution.solution.value[state_id]
            ),
        })
    return records


def _summarize_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    disagreements = [item for item in records if not item["actions_agree"]]
    return {
        "state_count": len(records),
        "action_agreement_count": len(records) - len(disagreements),
        "action_agreement_fraction": (
            (len(records) - len(disagreements)) / len(records) if records else None
        ),
        "first_divergence_step": (
            None if not disagreements else int(disagreements[0]["step"])
        ),
        "bellman_action_q_rank_median": (
            None if not records else float(np.median([
                item["bellman_action_dqn_q_rank"] for item in records
            ]))
        ),
        "bellman_action_q_rank_maximum": (
            None if not records else max(
                item["bellman_action_dqn_q_rank"] for item in records
            )
        ),
        "predicted_cost_to_go_mae": (
            None if not records else float(np.mean([
                abs(
                    item["dqn_predicted_greedy_cost_to_go"]
                    - item["exact_optimal_cost_to_go"]
                )
                for item in records
            ]))
        ),
        "maximum_one_step_exact_regret": (
            None if not records else max(
                item["one_step_exact_regret"] for item in records
            )
        ),
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot(path: Path, records: list[dict[str, Any]]) -> None:
    steps = np.asarray([item["step"] for item in records], dtype=int)
    exact = np.asarray([
        item["exact_optimal_cost_to_go"] for item in records
    ], dtype=float)
    predicted = np.asarray([
        item["dqn_predicted_greedy_cost_to_go"] for item in records
    ], dtype=float)
    ranks = np.asarray([
        item["bellman_action_dqn_q_rank"] for item in records
    ], dtype=int)
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(steps, exact, "o-", label="Exact Bellman cost-to-go")
    axes[0].plot(steps, predicted, "s--", label="DQN predicted greedy cost-to-go")
    axes[0].set_ylabel("Glide cost-to-go")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(steps, ranks, "o-", color="#D55E00")
    axes[1].axhline(1, color="black", linewidth=1, linestyle=":")
    axes[1].set_xlabel("Bellman trajectory step")
    axes[1].set_ylabel("Bellman action rank by DQN Q")
    axes[1].set_yticks(sorted(set([1, *ranks.tolist()])))
    axes[1].grid(alpha=0.3)
    figure.suptitle("Run A Double DQN policy diagnostic")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def diagnose(
    checkpoint_path: Path = CHECKPOINT,
    output_dir: Path | None = None,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Run A policy diagnosis requires CUDA")
    device = torch.device("cuda")
    started = perf_counter()
    pilot_config = ExtendedPilotConfig()
    manifest = extended_dataset_manifest(pilot_config)
    spec, terrain_spec = _run_a_scenario(pilot_config)
    scene = build_extended_scene(
        spec,
        terrain_spec,
        spatial_resolution_m=25.0,
        heading_spacing_deg=5.0,
        pilot_config=pilot_config,
    )
    problem = RunAGoalReachableProblem(scene, spec.sensor_map)
    starts = problem.switching_state_ids()
    reference = _reference_for(spec.scenario_id, manifest["manifest_sha256"])
    observation_config = MultiscaleObservationConfig()
    builder = MultiscaleTerrainObservationBuilder(problem, observation_config)
    batch_builder = ExactBatchedAABBGPUObservationMultiscaleBuilder(builder, device)
    cache = ObservationTensorCache(
        builder, max_entries=2_048, batch_builder=batch_builder,
    )
    catalog = HeadingActionCatalog(
        problem, row_cache=SharedActionRowCache(max_entries=50_000),
    )
    compatibility = CompatibilitySignature.from_components(
        problem, builder, catalog,
    )
    context = ScenarioRuntime(
        spec=spec,
        problem=problem,
        builder=builder,
        catalog=catalog,
        starts=starts,
        cache=cache,
        reference=reference,
    )
    shapes = builder.tensor_ready_shapes
    network_config = DQNNetworkConfig(
        spatial_channels=shapes["spatial"][0],
        spatial_height=shapes["spatial"][1],
        spatial_width=shapes["spatial"][2],
        scalar_features=shapes["scalar"][0],
        action_count=catalog.action_count,
        architecture_id="p1b-two-branch-dqn-multiscale-fused-v1",
    )
    raw_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    network, checkpoint = load_phase4_checkpoint(
        checkpoint_path,
        expected_network=network_config,
        expected_compatibility=compatibility,
        expected_model_contract=raw_checkpoint["model_contract"],
        device=device,
    )

    print("Solving the exact Run A Bellman corridor...", flush=True)
    solve_started = perf_counter()
    sparse = solve_sparse_additive_bellman(
        scene.graph,
        starts,
        problem.mdp.hazard_field,
        objective_parameters=scene.config.attacker_objective,
        parameters=scene.config.glider,
        physical_scale=scene.config.physical_scale,
        quadrature_resolution=scene.config.discretization.hazard_quadrature_resolution,
    )
    bellman_solve_sec = perf_counter() - solve_started
    bellman_start = int(reference.switching_state_id)
    exact_reference_j = float(reference.attacker_objective)
    reconstructed_reference_j = float(
        problem.powered_cost(bellman_start) + sparse.solution.value[bellman_start]
    )
    if abs(reconstructed_reference_j - exact_reference_j) > 1.0e-10:
        raise RuntimeError("reconstructed Bellman objective does not match reference")

    print("Rolling out all switching candidates with the checkpoint...", flush=True)
    evaluation = evaluate_policy_batched(network, context, device)
    if not evaluation.success or evaluation.switching_state_id is None:
        raise RuntimeError("checkpoint produced no successful Run A trajectory")
    selected_start = int(evaluation.switching_state_id)
    selected_exact_j = float(
        problem.powered_cost(selected_start) + sparse.solution.value[selected_start]
    )
    switching_gap = selected_exact_j - exact_reference_j
    glide_gap = float(evaluation.attacker_objective) - selected_exact_j
    total_gap = float(evaluation.attacker_objective) - exact_reference_j
    if abs(total_gap - switching_gap - glide_gap) > 1.0e-10:
        raise RuntimeError("objective-gap decomposition failed to close")

    print("Auditing Bellman and learned trajectories...", flush=True)
    bellman_rollout = _rollout_from(network, context, bellman_start, device)
    bellman_records = _audit_states(
        "bellman_reference",
        reference.trajectory,
        network,
        context,
        sparse,
        device,
    )
    dqn_records = _audit_states(
        "dqn_selected",
        evaluation.trajectory,
        network,
        context,
        sparse,
        device,
    )
    payload = {
        "schema": "p1b-run-a-policy-diagnostic-v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_kind": checkpoint["kind"],
        "checkpoint_episode": int(checkpoint["episode"]),
        "device": str(device),
        "scenario_id": spec.scenario_id,
        "switching_candidate_count": len(starts),
        "bellman_solve_runtime_sec": bellman_solve_sec,
        "total_diagnostic_runtime_sec": perf_counter() - started,
        "bellman_reference": {
            "switching_state_id": bellman_start,
            "trajectory": list(reference.trajectory),
            "edge_count": len(reference.trajectory) - 1,
            "powered_cost": float(problem.powered_cost(bellman_start)),
            "glide_cost": float(sparse.solution.value[bellman_start]),
            "attacker_objective": exact_reference_j,
        },
        "all_candidate_dqn_evaluation": evaluation.as_dict(),
        "dqn_rollout_forced_from_bellman_start": bellman_rollout,
        "gap_decomposition": {
            "bellman_J_A": exact_reference_j,
            "dqn_J_A": float(evaluation.attacker_objective),
            "total_absolute_gap": total_gap,
            "total_relative_gap": total_gap / abs(exact_reference_j),
            "exact_J_A_at_dqn_selected_switching_state": selected_exact_j,
            "switching_location_absolute_gap": switching_gap,
            "switching_location_fraction_of_total_gap": (
                switching_gap / total_gap if abs(total_gap) > 1.0e-15 else None
            ),
            "glide_policy_absolute_gap": glide_gap,
            "glide_policy_fraction_of_total_gap": (
                glide_gap / total_gap if abs(total_gap) > 1.0e-15 else None
            ),
        },
        "bellman_path_audit_summary": _summarize_audit(bellman_records),
        "dqn_path_audit_summary": _summarize_audit(dqn_records),
        "bellman_path_steps": bellman_records,
        "dqn_path_steps": dqn_records,
    }
    selected_output = (
        OUTPUT_ROOT / checkpoint_path.parent.name / "policy_diagnostic_v1"
        if output_dir is None else output_dir
    )
    selected_output.mkdir(parents=True, exist_ok=True)
    json_path = selected_output / "run_a_double_dqn_policy_diagnostic.json"
    csv_path = selected_output / "run_a_double_dqn_action_audit.csv"
    plot_path = selected_output / "run_a_double_dqn_bellman_path_q_diagnostic.png"
    _write_json(json_path, payload)
    _write_csv(csv_path, [*bellman_records, *dqn_records])
    _plot(plot_path, bellman_records)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "gap_decomposition": payload["gap_decomposition"],
        "bellman_path_audit_summary": payload["bellman_path_audit_summary"],
        "forced_bellman_start_J_A": bellman_rollout.get("attacker_objective"),
    }, indent=2), flush=True)
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    diagnose(arguments.checkpoint, arguments.output_dir)


if __name__ == "__main__":
    main()


__all__ = ["diagnose"]

"""Compare uniform and altitude-stratified Run A replay sampling."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import project_paths


BASE = (
    project_paths.CORE_DIR
    / "figure"
    / "phase_4_extended_random_boxes"
    / "training_v2"
    / "run_a_upper_bound"
    / "dx25m_dpsi5deg"
)
OUTPUT_DIR = BASE / "double_dqn_replay_sampling_comparison_v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _series(summary: dict[str, Any]) -> list[dict[str, float]]:
    return [
        {
            "episode": int(item["episode"]),
            "relative_error": float(
                item["VALIDATION_SIMPLE"]["median_relative_error"]
            ),
        }
        for item in summary["seed_result"]["history"]["evaluations"]
    ]


def _diagnostic_record(value: dict[str, Any]) -> dict[str, Any]:
    gap = value["gap_decomposition"]
    audit = value["bellman_path_audit_summary"]
    forced = value["dqn_rollout_forced_from_bellman_start"]
    return {
        "checkpoint_episode": int(value["checkpoint_episode"]),
        "J_A": float(gap["dqn_J_A"]),
        "relative_error": float(gap["total_relative_gap"]),
        "switching_location_absolute_gap": float(
            gap["switching_location_absolute_gap"]
        ),
        "glide_policy_absolute_gap": float(gap["glide_policy_absolute_gap"]),
        "bellman_path_action_agreement_fraction": float(
            audit["action_agreement_fraction"]
        ),
        "bellman_action_q_rank_median": float(
            audit["bellman_action_q_rank_median"]
        ),
        "cost_to_go_mae": float(audit["predicted_cost_to_go_mae"]),
        "forced_bellman_start_J_A": float(forced["attacker_objective"]),
        "forced_bellman_start_relative_error": (
            float(forced["attacker_objective"]) - float(gap["bellman_J_A"])
        ) / abs(float(gap["bellman_J_A"])),
    }


def _training_record(summary: dict[str, Any]) -> dict[str, Any]:
    seed = summary["seed_result"]
    series = _series(summary)
    errors = [item["relative_error"] for item in series]
    losses = [float(value) for value in seed["history"]["td_loss"][-500:]]
    late = errors[-3:]
    return {
        "best_episode": int(seed["best_episode"]),
        "best_relative_error": min(errors),
        "final_relative_error": errors[-1],
        "late_three_mean_relative_error": mean(late),
        "late_three_range": max(late) - min(late),
        "last_500_td_loss_mean": mean(losses),
        "last_500_td_loss_median": median(losses),
        "training_wall_runtime_sec": float(seed["training_wall_runtime_sec"]),
        "pure_training_runtime_sec": float(seed["pure_training_runtime_sec"]),
        "environment_steps": int(seed["environment_steps"]),
        "optimization_steps": int(seed["optimization_steps"]),
        "replay_sampling_stratum_counts": seed.get(
            "replay_sampling_stratum_counts", {}
        ),
        "evaluation_series": series,
    }


def compare() -> Path:
    uniform_dir = BASE / "double_dqn_b_v1"
    stratified_dir = BASE / "double_dqn_b_altitude_stratified_v1"
    uniform_summary = _read(uniform_dir / "run_a_summary.json")
    stratified_summary = _read(stratified_dir / "run_a_summary.json")
    uniform_diagnostic = _read(
        uniform_dir
        / "policy_diagnostic_v1"
        / "run_a_double_dqn_policy_diagnostic.json"
    )
    stratified_best = _read(
        stratified_dir
        / "policy_diagnostic_best_v1"
        / "run_a_double_dqn_policy_diagnostic.json"
    )
    stratified_final = _read(
        stratified_dir
        / "policy_diagnostic_final_v1"
        / "run_a_double_dqn_policy_diagnostic.json"
    )
    payload = {
        "schema": "p1b-run-a-double-dqn-replay-sampling-comparison-v1",
        "controlled_variables": {
            "seed": 0,
            "episodes": 3_000,
            "n_step_return": 1,
            "learning_rate": 3.0e-4,
            "target_update_optimization_steps": 500,
            "td_target_estimator": "double_dqn",
        },
        "changed_variable": "uniform versus altitude_stratified replay sampling",
        "uniform": {
            "training": _training_record(uniform_summary),
            "best_and_final_diagnostic": _diagnostic_record(uniform_diagnostic),
        },
        "altitude_stratified": {
            "training": _training_record(stratified_summary),
            "best_diagnostic": _diagnostic_record(stratified_best),
            "final_diagnostic": _diagnostic_record(stratified_final),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "run_a_replay_sampling_comparison.json"
    _write(json_path, payload)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for summary, label, marker in (
        (uniform_summary, "Uniform replay", "o"),
        (stratified_summary, "Altitude-stratified replay", "s"),
    ):
        series = _series(summary)
        axis.plot(
            [item["episode"] for item in series],
            [100.0 * item["relative_error"] for item in series],
            marker=marker,
            linewidth=2.0,
            label=label,
        )
    axis.axhline(10.0, color="black", linestyle=":", label="10% target")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Relative $J_A$ error [%]")
    axis.set_title("Run A Double DQN: replay sampling comparison")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "run_a_replay_sampling_error.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    return json_path


if __name__ == "__main__":
    print(compare())


__all__ = ["compare"]

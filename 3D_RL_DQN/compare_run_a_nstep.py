"""Compare the Run A one-step and three-step Double DQN experiments."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
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
OUTPUT_DIR = BASE / "double_dqn_nstep3_comparison_v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _evaluation_series(summary: dict[str, Any]) -> list[dict[str, float]]:
    return [
        {
            "episode": int(item["episode"]),
            "relative_error": float(
                item["VALIDATION_SIMPLE"]["median_relative_error"]
            ),
        }
        for item in summary["seed_result"]["history"]["evaluations"]
    ]


def _checkpoint_record(
    summary: dict[str, Any], diagnostic: dict[str, Any], *, best: bool,
) -> dict[str, Any]:
    seed = summary["seed_result"]
    records = seed["official_best_records"] if best else seed["final_records"]
    record = records[0]
    evaluation = record["dqn"]
    gap = diagnostic["gap_decomposition"]
    audit = diagnostic["bellman_path_audit_summary"]
    forced = diagnostic["dqn_rollout_forced_from_bellman_start"]
    return {
        "episode": int(seed["best_episode"] if best else seed["episodes_completed"]),
        "J_A": float(evaluation["attacker_objective"]),
        "relative_error": float(record["relative_error"]),
        "switching_state_id": int(evaluation["switching_state_id"]),
        "edge_count": len(evaluation["trajectory"]) - 1,
        "powered_cost": float(evaluation["powered_cost"]),
        "glide_cost": float(evaluation["glide_cost"]),
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
        "bellman_action_q_rank_maximum": int(
            audit["bellman_action_q_rank_maximum"]
        ),
        "bellman_path_predicted_cost_to_go_mae": float(
            audit["predicted_cost_to_go_mae"]
        ),
        "forced_bellman_start_J_A": float(forced["attacker_objective"]),
        "forced_bellman_start_relative_error": (
            float(forced["attacker_objective"]) - float(gap["bellman_J_A"])
        ) / abs(float(gap["bellman_J_A"])),
    }


def compare() -> Path:
    baseline_summary = _read(BASE / "double_dqn_b_v1" / "run_a_summary.json")
    nstep_summary = _read(
        BASE / "double_dqn_b_nstep3_v1" / "run_a_summary.json"
    )
    baseline_diagnostic = _read(
        BASE
        / "double_dqn_b_v1"
        / "policy_diagnostic_v1"
        / "run_a_double_dqn_policy_diagnostic.json"
    )
    nstep_best_diagnostic = _read(
        BASE
        / "double_dqn_b_nstep3_v1"
        / "policy_diagnostic_best_v1"
        / "run_a_double_dqn_policy_diagnostic.json"
    )
    nstep_final_diagnostic = _read(
        BASE
        / "double_dqn_b_nstep3_v1"
        / "policy_diagnostic_final_v1"
        / "run_a_double_dqn_policy_diagnostic.json"
    )
    baseline_series = _evaluation_series(baseline_summary)
    nstep_series = _evaluation_series(nstep_summary)

    def training_record(summary: dict[str, Any]) -> dict[str, Any]:
        seed = summary["seed_result"]
        errors = [item["relative_error"] for item in _evaluation_series(summary)]
        late = errors[-3:]
        return {
            "training_wall_runtime_sec": float(seed["training_wall_runtime_sec"]),
            "pure_training_runtime_sec": float(seed["pure_training_runtime_sec"]),
            "measured_seed_runtime_sec": float(seed["measured_seed_runtime_sec"]),
            "environment_steps": int(seed["environment_steps"]),
            "optimization_steps": int(seed["optimization_steps"]),
            "best_episode": int(seed["best_episode"]),
            "best_relative_error": min(errors),
            "final_relative_error": errors[-1],
            "late_three_mean_relative_error": mean(late),
            "late_three_range": max(late) - min(late),
            "evaluation_series": _evaluation_series(summary),
        }

    payload = {
        "schema": "p1b-run-a-double-dqn-nstep-comparison-v1",
        "controlled_variables": {
            "seed": 0,
            "episodes": 3_000,
            "learning_rate": 3.0e-4,
            "target_update_optimization_steps": 500,
            "td_target_estimator": "double_dqn",
            "spatial_resolution_m": 25.0,
            "heading_spacing_deg": 5.0,
        },
        "changed_variable": "n_step_return: 1 versus 3",
        "bellman_J_A": float(
            baseline_diagnostic["gap_decomposition"]["bellman_J_A"]
        ),
        "one_step": {
            "training": training_record(baseline_summary),
            "best_and_final_checkpoint": _checkpoint_record(
                baseline_summary, baseline_diagnostic, best=True,
            ),
        },
        "three_step": {
            "training": training_record(nstep_summary),
            "best_checkpoint": _checkpoint_record(
                nstep_summary, nstep_best_diagnostic, best=True,
            ),
            "final_checkpoint": _checkpoint_record(
                nstep_summary, nstep_final_diagnostic, best=False,
            ),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "run_a_double_dqn_nstep_comparison.json"
    _write(json_path, payload)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for series, label, marker in (
        (baseline_series, "1-step Double DQN", "o"),
        (nstep_series, "3-step Double DQN", "s"),
    ):
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
    axis.set_title("Run A: one-step versus three-step Double DQN")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "run_a_double_dqn_nstep_error.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    return json_path


if __name__ == "__main__":
    print(compare())


__all__ = ["compare"]

"""Freeze and summarize the completed G1 terrain-generalization pilot."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import project_paths


TAG = "dx25m_dpsi5deg"
RUN_LABEL = "seed0_ep10000_v1"
RESULT_PATH = (
    project_paths.DQN_DIR
    / "runs"
    / "phase4_extended_random_boxes"
    / "training_v2"
    / "g1_generalization"
    / TAG
    / RUN_LABEL
    / "phase4_seed0_complete.json"
)
CHECKPOINT_PATH = (
    project_paths.DQN_DIR
    / "checkpoints"
    / "phase4_extended_random_boxes"
    / "training_v2"
    / "g1_generalization"
    / TAG
    / RUN_LABEL
    / "phase4_seed0_best.pt"
)
OUTPUT_DIR = (
    project_paths.CORE_DIR
    / "figure"
    / "phase_4_extended_random_boxes"
    / "training_v2"
    / "g1_generalization"
    / TAG
    / RUN_LABEL
    / "closeout_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _aggregate(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(records)
    errors = [
        float(record["relative_error"])
        for record in selected if record["relative_error"] is not None
    ]
    return {
        "case_count": len(selected),
        "success_count": sum(bool(record["dqn"]["success"]) for record in selected),
        "success_rate": (
            sum(bool(record["dqn"]["success"]) for record in selected)
            / len(selected) if selected else 0.0
        ),
        "median_relative_error": median(errors) if errors else None,
        "maximum_relative_error": max(errors) if errors else None,
        "mean_bellman_runtime_sec": float(np.mean([
            record["bellman_runtime_sec"] for record in selected
        ])) if selected else 0.0,
        "mean_dqn_inference_runtime_sec": float(np.mean([
            record["dqn"]["inference_runtime_sec"] for record in selected
        ])) if selected else 0.0,
    }


def _records_by_split(
    records: list[dict[str, Any]], split: str,
) -> list[dict[str, Any]]:
    return [record for record in records if record["split"] == split]


def _write_evaluation_history(result: dict[str, Any], output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for snapshot in result["history"]["evaluations"]:
        rows.append({
            "episode": snapshot["episode"],
            "evaluation_runtime_sec": snapshot["runtime_sec"],
            "train_success_rate": snapshot["TRAIN_SIMPLE"]["success_rate"],
            "train_median_relative_error": snapshot["TRAIN_SIMPLE"][
                "median_relative_error"
            ],
            "train_maximum_relative_error": snapshot["TRAIN_SIMPLE"][
                "maximum_relative_error"
            ],
            "validation_success_rate": snapshot["VALIDATION_SIMPLE"][
                "success_rate"
            ],
            "validation_median_relative_error": snapshot["VALIDATION_SIMPLE"][
                "median_relative_error"
            ],
            "validation_maximum_relative_error": snapshot["VALIDATION_SIMPLE"][
                "maximum_relative_error"
            ],
        })
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_scenario_table(result: dict[str, Any], output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for checkpoint, records in (
        ("best", result["official_best_records"]),
        ("final", result["final_records"]),
    ):
        for record in records:
            dqn = record["dqn"]
            rows.append({
                "checkpoint": checkpoint,
                "scenario_id": record["scenario_id"],
                "split": record["split"],
                "terrain_category": record["terrain_category"],
                "bellman_J_A": record["bellman_J_A"],
                "dqn_J_A": dqn["attacker_objective"],
                "absolute_error": record["absolute_error"],
                "relative_error": record["relative_error"],
                "dqn_success": dqn["success"],
                "bellman_runtime_sec": record["bellman_runtime_sec"],
                "dqn_inference_runtime_sec": dqn["inference_runtime_sec"],
                "bellman_switching_state_id": (
                    record["bellman_trajectory"][0]
                    if record["bellman_trajectory"] else None
                ),
                "dqn_switching_state_id": dqn["switching_state_id"],
                "bellman_trajectory_states": len(record["bellman_trajectory"]),
                "dqn_trajectory_states": len(dqn["trajectory"]),
            })
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_learning(result: dict[str, Any], output: Path) -> None:
    evaluations = result["history"]["evaluations"]
    episodes = np.asarray([item["episode"] for item in evaluations])
    train_median = np.asarray([
        item["TRAIN_SIMPLE"]["median_relative_error"] for item in evaluations
    ]) * 100.0
    validation_median = np.asarray([
        item["VALIDATION_SIMPLE"]["median_relative_error"] for item in evaluations
    ]) * 100.0
    train_maximum = np.asarray([
        item["TRAIN_SIMPLE"]["maximum_relative_error"] for item in evaluations
    ]) * 100.0
    validation_maximum = np.asarray([
        item["VALIDATION_SIMPLE"]["maximum_relative_error"] for item in evaluations
    ]) * 100.0
    best_episode = int(result["best_episode"])

    figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    for axis, train, validation, threshold, title in (
        (axes[0], train_median, validation_median, 10.0, "Median relative error"),
        (axes[1], train_maximum, validation_maximum, 20.0, "Maximum relative error"),
    ):
        axis.plot(episodes, train, marker="o", label="Training")
        axis.plot(episodes, validation, marker="s", label="Held-out validation")
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1.2,
                     label=f"Approved threshold ({threshold:g}%)")
        axis.axvline(best_episode, color="#9467bd", linestyle=":", linewidth=1.4,
                     label=f"Best checkpoint ({best_episode:,})")
        axis.set_ylabel("Relative error [%]")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_xlabel("Training episode")
    figure.suptitle("G1 25 m terrain generalization: training vs held-out validation")
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_scenario_errors(result: dict[str, Any], output: Path) -> None:
    records = sorted(
        result["official_best_records"],
        key=lambda item: (item["split"], float(item["relative_error"])),
    )
    errors = np.asarray([float(item["relative_error"]) for item in records]) * 100.0
    colors = [
        "#1f77b4" if item["split"] == "TRAIN_SIMPLE" else "#d62728"
        for item in records
    ]
    labels = [
        ("T" if item["split"] == "TRAIN_SIMPLE" else "V") + str(index + 1)
        for index, item in enumerate(records)
    ]
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.bar(np.arange(len(records)), errors, color=colors)
    axis.axhline(20.0, color="black", linestyle="--", linewidth=1.2,
                 label="Approved maximum-error threshold (20%)")
    axis.set_xticks(np.arange(len(records)), labels, rotation=60)
    axis.set_ylabel("Relative error [%]")
    axis.set_xlabel("Scenario (T = training, V = validation)")
    axis.set_title("G1 best checkpoint: per-scenario Bellman–DQN $J_A$ error")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_runtime(result: dict[str, Any], output: Path) -> None:
    profile = result["runtime_profile_sec"]
    labels = [
        "Action observation\n+ forward",
        "Environment transition\n+ replay",
        "Optimization observation\npreparation/transfer",
        "DQN forward/backward\n+ optimizer",
        "Scheduled\nevaluation",
        "Checkpoint\nI/O",
        "Post-training\nevaluation",
    ]
    seconds = np.asarray([
        profile["action_observation_and_forward_sec"],
        profile["environment_transition_and_replay_sec"],
        profile["optimization_observation_transfer_sec"],
        profile["dqn_forward_backward_optimizer_sec"],
        result["checkpoint_evaluation_runtime_sec"],
        result["checkpoint_io_runtime_sec"] + result["final_checkpoint_io_runtime_sec"],
        result["post_training_evaluation_runtime_sec"],
    ])
    figure, axis = plt.subplots(figsize=(10.5, 5.0))
    bars = axis.bar(np.arange(len(labels)), seconds / 60.0, color="#4c78a8")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    axis.set_ylabel("Runtime [min]")
    axis.set_title("G1 measured runtime decomposition")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, seconds / 60.0):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.1f}",
            ha="center", va="bottom", fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def closeout_g1() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    best_train = _aggregate(_records_by_split(
        result["official_best_records"], "TRAIN_SIMPLE",
    ))
    best_validation_records = _records_by_split(
        result["official_best_records"], "VALIDATION_SIMPLE",
    )
    best_validation = _aggregate(best_validation_records)
    final_train = _aggregate(_records_by_split(
        result["final_records"], "TRAIN_SIMPLE",
    ))
    final_validation = _aggregate(_records_by_split(
        result["final_records"], "VALIDATION_SIMPLE",
    ))
    worst_validation = max(
        best_validation_records, key=lambda item: float(item["relative_error"]),
    )
    time_saved = (
        best_validation["mean_bellman_runtime_sec"]
        - best_validation["mean_dqn_inference_runtime_sec"]
    )
    break_even = (
        float(result["pure_training_runtime_sec"]) / time_saved
        if time_saved > 0.0 else None
    )
    closeout = {
        "schema": "p1b-phase4-training-v2-g1-closeout-v1",
        "status": "closed_accuracy_threshold_not_met",
        "decision": (
            "freeze episode-7000 best checkpoint; do not run seeds 1/2 and do "
            "not tune this 25 m pilot further"
        ),
        "scope": (
            "attacker best response J_A on held-out in-family random-box terrain; "
            "not a sealed final test and not full SSE/J_D"
        ),
        "seed": result["seed"],
        "episodes_completed": result["episodes_completed"],
        "best_episode": result["best_episode"],
        "quality_thresholds": {
            "success_rate": 1.0,
            "median_relative_error": 0.10,
            "maximum_relative_error": 0.20,
        },
        "best": {
            "training": best_train,
            "validation": best_validation,
        },
        "final": {
            "training": final_train,
            "validation": final_validation,
        },
        "worst_best_validation_case": {
            "scenario_id": worst_validation["scenario_id"],
            "bellman_J_A": worst_validation["bellman_J_A"],
            "dqn_J_A": worst_validation["dqn"]["attacker_objective"],
            "relative_error": worst_validation["relative_error"],
        },
        "runtime_sec": {
            "training_wall": result["training_wall_runtime_sec"],
            "pure_training": result["pure_training_runtime_sec"],
            "scheduled_evaluation": result["checkpoint_evaluation_runtime_sec"],
            "post_training_evaluation": result[
                "post_training_evaluation_runtime_sec"
            ],
            "measured_seed_total": result["measured_seed_runtime_sec"],
            "profile": result["runtime_profile_sec"],
        },
        "preliminary_attacker_br_amortization": {
            "mean_bellman_runtime_sec_per_validation_case": best_validation[
                "mean_bellman_runtime_sec"
            ],
            "mean_dqn_inference_runtime_sec_per_validation_case": best_validation[
                "mean_dqn_inference_runtime_sec"
            ],
            "runtime_saved_sec_per_case": time_saved,
            "pure_training_break_even_case_count": break_even,
            "limitation": (
                "preliminary J_A attacker-BR estimate only; final decision requires "
                "sealed test, J_D, and full-SSE measurements"
            ),
        },
        "provenance": {
            "dataset_manifest_sha256": result["dataset_manifest_sha256"],
            "result_path": str(RESULT_PATH),
            "result_sha256": _sha256(RESULT_PATH),
            "best_checkpoint_path": str(CHECKPOINT_PATH),
            "best_checkpoint_sha256": _sha256(CHECKPOINT_PATH),
        },
    }
    summary_path = OUTPUT_DIR / "g1_closeout_summary.json"
    _write_json(summary_path, closeout)
    _write_evaluation_history(result, OUTPUT_DIR / "g1_evaluation_history.csv")
    _write_scenario_table(result, OUTPUT_DIR / "g1_scenario_metrics.csv")
    _plot_learning(result, OUTPUT_DIR / "g1_train_validation_error.png")
    _plot_scenario_errors(result, OUTPUT_DIR / "g1_best_scenario_errors.png")
    _plot_runtime(result, OUTPUT_DIR / "g1_runtime_breakdown.png")

    best_v = closeout["best"]["validation"]
    final_v = closeout["final"]["validation"]
    amortization = closeout["preliminary_attacker_br_amortization"]
    report = f"""# G1 Terrain-Generalization Pilot Close-out

## Decision

G1 is closed as an accuracy-threshold miss. Freeze the episode {result['best_episode']:,}
checkpoint, do not run seeds 1/2, and do not tune this 25 m pilot further.

## Scope

This result covers attacker best-response $J_A$ on held-out, in-family random-box
terrain. It is not a sealed final test and does not include full SSE or $J_D$.

## Accuracy

| Checkpoint | Validation success | Median error | Maximum error |
|---|---:|---:|---:|
| Best ({result['best_episode']:,}) | {best_v['success_rate']:.0%} | {best_v['median_relative_error']:.2%} | {best_v['maximum_relative_error']:.2%} |
| Final ({result['episodes_completed']:,}) | {final_v['success_rate']:.0%} | {final_v['median_relative_error']:.2%} | {final_v['maximum_relative_error']:.2%} |
| Approved threshold | 100% | <=10% | <=20% |

The final training median error improved to {final_train['median_relative_error']:.2%},
while held-out validation degraded after episode {result['best_episode']:,}. This is
the observed G1 overfitting signal.

## Runtime

- Pure training: {result['pure_training_runtime_sec'] / 60.0:.2f} min
- Scheduled validation: {result['checkpoint_evaluation_runtime_sec'] / 60.0:.2f} min
- Post-training evaluation: {result['post_training_evaluation_runtime_sec'] / 60.0:.2f} min
- Measured seed total: {result['measured_seed_runtime_sec'] / 60.0:.2f} min

The dominant pure-training cost was optimization observation preparation/transfer
({result['runtime_profile_sec']['optimization_observation_transfer_sec'] / 60.0:.2f} min),
not neural-network forward/backward optimization
({result['runtime_profile_sec']['dqn_forward_backward_optimizer_sec'] / 60.0:.2f} min).

## Preliminary attacker-BR amortization

- Mean Bellman validation runtime: {amortization['mean_bellman_runtime_sec_per_validation_case']:.3f} s/case
- Mean DQN validation inference runtime: {amortization['mean_dqn_inference_runtime_sec_per_validation_case']:.3f} s/case
- Runtime saved: {amortization['runtime_saved_sec_per_case']:.3f} s/case
- Pure-training break-even: {amortization['pure_training_break_even_case_count']:.1f} cases

This break-even is preliminary and must not be used as the final Bellman-vs-DQN
decision boundary until sealed-test, $J_D$, and full-SSE measurements exist.

## Provenance

- Dataset manifest SHA-256: `{result['dataset_manifest_sha256']}`
- Result SHA-256: `{closeout['provenance']['result_sha256']}`
- Best checkpoint SHA-256: `{closeout['provenance']['best_checkpoint_sha256']}`
"""
    (OUTPUT_DIR / "G1_CLOSEOUT_REPORT.md").write_text(report, encoding="utf-8")
    return summary_path


if __name__ == "__main__":
    print(closeout_g1())


__all__ = ["closeout_g1"]

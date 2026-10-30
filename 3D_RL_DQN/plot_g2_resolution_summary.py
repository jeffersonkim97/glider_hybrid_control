"""Plot the measured G2 seed-0 resolution sweep summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUN_ROOT = (
    ROOT
    / "runs"
    / "phase4_extended_random_boxes"
    / "training_v2"
    / "g2_generalization"
)
OUTPUT_DIR = RUN_ROOT / "resolution_summary"
RESOLUTIONS_M = (100, 50, 25, 10)


def load_measurements() -> list[dict[str, float]]:
    measurements: list[dict[str, float]] = []
    for dx_m in RESOLUTIONS_M:
        result_path = (
            RUN_ROOT
            / f"dx{dx_m}m_dpsi5deg"
            / "seed0_ep10000_v1"
            / "phase4_seed0_complete.json"
        )
        with result_path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)

        validation_errors = sorted(
            100.0 * float(record["relative_error"])
            for record in result["final_records"]
            if record["split"] == "VALIDATION_SIMPLE"
            and record["dqn"]["success"]
        )
        if len(validation_errors) != 16:
            raise RuntimeError(
                f"Expected 16 successful validation records in {result_path}, "
                f"found {len(validation_errors)}."
            )

        measurements.append(
            {
                "dx_m": float(dx_m),
                "dqn_training_min": (
                    float(result["pure_training_runtime_sec"]) / 60.0
                ),
                "bellman_reference_min": sum(
                    float(record["bellman_runtime_sec"])
                    for record in result["final_records"]
                )
                / 60.0,
                "median_error_pct": float(np.median(validation_errors)),
            }
        )
    return measurements


def plot_computation_time(measurements: list[dict[str, float]]) -> Path:
    positions = np.arange(len(measurements))
    labels = [f"{item['dx_m']:.0f}" for item in measurements]
    dqn = [item["dqn_training_min"] for item in measurements]
    bellman = [item["bellman_reference_min"] for item in measurements]

    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    axis.plot(
        positions,
        dqn,
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="DQN pure training",
        color="#0072B2",
    )
    axis.plot(
        positions,
        bellman,
        marker="s",
        linewidth=2.2,
        markersize=7,
        label="Bellman reference computation (48 cases)",
        color="#D55E00",
    )
    for x_value, value in zip(positions, dqn, strict=True):
        axis.annotate(
            f"{value:.2f}",
            (x_value, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color="#0072B2",
            fontsize=9,
        )
    for x_value, value in zip(positions, bellman, strict=True):
        axis.annotate(
            f"{value:.2f}",
            (x_value, value),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            color="#D55E00",
            fontsize=9,
        )

    axis.set_xticks(positions, labels)
    axis.set_xlabel("Spatial discretization, dx [m]")
    axis.set_ylabel("Computation time [min, log scale]")
    axis.set_yscale("log")
    axis.set_title("DQN Training vs. Bellman Reference Computation")
    axis.grid(True, which="both", linestyle="--", alpha=0.35)
    axis.legend(frameon=False)

    output = OUTPUT_DIR / "g2_dx_vs_dqn_training_and_bellman_time_seed0.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_computation_time_linear(measurements: list[dict[str, float]]) -> Path:
    positions = np.arange(len(measurements))
    labels = [f"{item['dx_m']:.0f}" for item in measurements]
    dqn = [item["dqn_training_min"] for item in measurements]
    bellman = [item["bellman_reference_min"] for item in measurements]

    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    axis.plot(
        positions,
        dqn,
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="DQN pure training",
        color="#0072B2",
    )
    axis.plot(
        positions,
        bellman,
        marker="s",
        linewidth=2.2,
        markersize=7,
        label="Bellman reference computation (48 cases)",
        color="#D55E00",
    )
    for x_value, value in zip(positions, dqn, strict=True):
        axis.annotate(
            f"{value:.2f}",
            (x_value, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color="#0072B2",
            fontsize=9,
        )
    for x_value, value in zip(positions, bellman, strict=True):
        label_offset = (0, 8) if value < 5.0 else (0, -15)
        axis.annotate(
            f"{value:.2f}",
            (x_value, value),
            xytext=label_offset,
            textcoords="offset points",
            ha="center",
            color="#D55E00",
            fontsize=9,
        )

    axis.set_xticks(positions, labels)
    axis.set_xlabel("Spatial discretization, dx [m]")
    axis.set_ylabel("Computation time [min]")
    axis.set_ylim(bottom=0.0)
    axis.set_title("DQN Training vs. Bellman Reference Computation")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(frameon=False)

    output = (
        OUTPUT_DIR
        / "g2_dx_vs_dqn_training_and_bellman_time_seed0_linear_scale.png"
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_median_error(measurements: list[dict[str, float]]) -> Path:
    positions = np.arange(len(measurements))
    labels = [f"{item['dx_m']:.0f}" for item in measurements]
    errors = [item["median_error_pct"] for item in measurements]

    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    axis.plot(
        positions,
        errors,
        marker="o",
        linewidth=2.2,
        markersize=7,
        color="#009E73",
        label="DQN median error relative to Bellman",
    )
    axis.axhline(
        10.0,
        color="#CC79A7",
        linestyle="--",
        linewidth=1.8,
        label="Approved median-error threshold (10%)",
    )
    for x_value, value in zip(positions, errors, strict=True):
        axis.annotate(
            f"{value:.2f}%",
            (x_value, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    axis.set_xticks(positions, labels)
    axis.set_xlabel("Spatial discretization, dx [m]")
    axis.set_ylabel("Median Bellman-relative DQN $J_A$ error [%]")
    axis.set_ylim(0.0, 11.5)
    axis.set_title("DQN Validation Error Across Spatial Resolutions")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(frameon=False, loc="lower left")

    output = OUTPUT_DIR / "g2_dx_vs_dqn_median_bellman_relative_error_seed0.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    measurements = load_measurements()
    for output in (
        plot_computation_time(measurements),
        plot_computation_time_linear(measurements),
        plot_median_error(measurements),
    ):
        print(output)


if __name__ == "__main__":
    main()

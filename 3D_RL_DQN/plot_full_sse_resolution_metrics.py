"""Plot stepped-pyramid full-SSE payoffs and runtime versus spatial resolution."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import project_paths


RESOLUTIONS_M = (10.0, 25.0, 50.0, 100.0)
RUN_LABEL = "official_batched_v1"
TRAINING_SEEDS = (0, 1, 2)
OUTPUT_DIR = (
    project_paths.CORE_DIR
    / "figure"
    / "phase_4_multi_terrain_dqn"
    / "stepped_pyramid_resolution_comparison"
)


def condition_tag(resolution_m: float) -> str:
    return f"dx{float(resolution_m):g}m_dpsi5deg"


def load_rows() -> list[dict[str, float]]:
    comparison = json.loads(
        (OUTPUT_DIR / "full_sse_resolution_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    by_resolution = {
        float(row["spatial_resolution_m"]): row for row in comparison["rows"]
    }
    rows: list[dict[str, float]] = []
    for resolution in RESOLUTIONS_M:
        row = by_resolution[resolution]
        manifest_path = (
            project_paths.CORE_DIR
            / "figure"
            / "phase_4_multi_terrain_dqn"
            / condition_tag(resolution)
            / RUN_LABEL
            / "phase4_multi_terrain_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append({
            "dx": resolution,
            "bellman_J_A": float(row["bellman_J_A"]),
            "dqn_J_A": float(row["dqn_J_A"]),
            "bellman_J_D": float(row["bellman_J_D"]),
            "dqn_J_D": float(row["dqn_J_D"]),
            "bellman_full_sse_sec": float(row["bellman_full_sse_sec"]),
            "dqn_full_sse_sec": float(row["dqn_full_sse_sec"]),
            "training_mean_sec": (
                float(manifest["runtime_sec"]["training_total"])
                / len(TRAINING_SEEDS)
            ),
        })
    return rows


def style_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.set_xlabel(r"Spatial discretization $dx=dy=dh$ [m]")
    axis.set_ylabel(ylabel)
    axis.set_xticks(RESOLUTIONS_M)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best", frameon=True)


def plot_payoff(
    dx: np.ndarray,
    bellman: np.ndarray,
    dqn: np.ndarray,
    *, symbol: str,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    axis.plot(
        dx, bellman, marker="o", linewidth=2.2, markersize=7,
        color="#0072B2", label="Bellman full SSE",
    )
    axis.plot(
        dx, dqn, marker="s", linewidth=2.2, markersize=7,
        color="#E69F00", label="DQN approximate full SSE",
    )
    style_axis(axis, rf"Full-SSE ${symbol}$")
    axis.set_title(
        rf"Stepped-pyramid full Local-SSE: ${symbol}$ vs spatial discretization"
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_runtime(
    dx: np.ndarray,
    bellman: np.ndarray,
    dqn: np.ndarray,
    training: np.ndarray,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    axis.plot(
        dx, bellman, marker="o", linewidth=2.2, markersize=7,
        color="#0072B2", label="Bellman full SSE",
    )
    axis.plot(
        dx, dqn, marker="s", linewidth=2.2, markersize=7,
        color="#E69F00", label="DQN full SSE (training excluded)",
    )
    axis.plot(
        dx, training, marker="^", linewidth=2.2, markersize=8,
        color="#009E73",
        label="DQN training (3-seed mean; periodic eval included)",
    )
    axis.set_yscale("log")
    style_axis(axis, "Wall-clock time [s] (log scale)")
    axis.set_title(
        "Stepped-pyramid full Local-SSE and DQN training time vs spatial discretization"
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    dx = np.asarray([row["dx"] for row in rows], dtype=float)
    outputs = {
        "J_A": OUTPUT_DIR / "full_sse_J_A_vs_dx.png",
        "J_D": OUTPUT_DIR / "full_sse_J_D_vs_dx.png",
        "runtime": OUTPUT_DIR / "full_sse_runtime_vs_dx.png",
    }
    plot_payoff(
        dx,
        np.asarray([row["bellman_J_A"] for row in rows]),
        np.asarray([row["dqn_J_A"] for row in rows]),
        symbol="J_A",
        output=outputs["J_A"],
    )
    plot_payoff(
        dx,
        np.asarray([row["bellman_J_D"] for row in rows]),
        np.asarray([row["dqn_J_D"] for row in rows]),
        symbol="J_D",
        output=outputs["J_D"],
    )
    plot_runtime(
        dx,
        np.asarray([row["bellman_full_sse_sec"] for row in rows]),
        np.asarray([row["dqn_full_sse_sec"] for row in rows]),
        np.asarray([row["training_mean_sec"] for row in rows]),
        outputs["runtime"],
    )
    manifest = {
        "condition": {
            "terrain": "stepped_pyramid",
            "heading_spacing_deg": 5.0,
            "r_neighbor": 1,
            "spatial_resolutions_m": list(RESOLUTIONS_M),
        },
        "dpi": 300,
        "runtime_units": "seconds",
        "runtime_plot_scale": "logarithmic",
        "bellman_and_dqn_runtime_definition": (
            "full Local-SSE wall time excluding shared scene construction, "
            "checkpoint loading, audit, and plot-only exact replay"
        ),
        "training_time_definition": (
            "mean per-seed training-loop wall time across seeds 0,1,2, computed "
            "from the recorded three-seed total divided by three; "
            "includes periodic checkpoint evaluation and excludes post-training "
            "final/official evaluation"
        ),
        "rows": rows,
        "outputs": {key: str(value).replace("\\", "/") for key, value in outputs.items()},
    }
    manifest_path = OUTPUT_DIR / "full_sse_resolution_plots_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for output in (*outputs.values(), manifest_path):
        print(output)


if __name__ == "__main__":
    main()

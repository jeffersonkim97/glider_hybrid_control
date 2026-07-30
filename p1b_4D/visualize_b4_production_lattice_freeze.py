"""Decision dashboard for the B4 production-lattice freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


LEVEL_COLORS = ("#A0CBE8", "#4C78A8", "#1F4E79")
TERRAIN_LABELS = {
    "two_hill": "Two hill",
    "single_hill": "Single hill",
    "goal_in_valley": "Goal in valley",
}


def create_b4_figure(manifest_path: Path, output_path: Path) -> Path:
    """Render the B1-to-B4 evidence chain and final C-lite contract."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("B4 manifest must have complete status")

    figure = plt.figure(figsize=(16.0, 9.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(0.72, 1.55))
    flow_axis = figure.add_subplot(grid[0, :])
    resolution_axis = figure.add_subplot(grid[1, 0])
    speed_axis = figure.add_subplot(grid[1, 1])
    contract_axis = figure.add_subplot(grid[1, 2])

    _plot_flow(flow_axis)
    _plot_resolution_choice(resolution_axis, manifest)
    _plot_speed_choice(speed_axis, manifest)
    _plot_contract(contract_axis, manifest)
    figure.suptitle(
        "B4 Production Solver Freeze — Evidence to C-lite Contract",
        fontsize=18,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_flow(axis):
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    boxes = (
        (0.02, "B1", "Structure check", "Nested grids/actions\nexact endpoints"),
        (0.27, "B2", "Two-hill pilot", "Resolution sensitivity\nand ranking test"),
        (0.52, "B3", "Terrain extension", "Single hill + valley\ncommon replay"),
        (0.77, "B4", "Freeze one solver", "One immutable setup\nfor C-lite"),
    )
    for index, (x, label, title, body) in enumerate(boxes):
        selected = label == "B4"
        box = FancyBboxPatch(
            (x, 0.20), 0.205, 0.62,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=2.2 if selected else 1.5,
            edgecolor="#2F6B3C" if selected else "#4C78A8",
            facecolor="#E8F5E9" if selected else "#EDF4FA",
        )
        axis.add_patch(box)
        axis.text(
            x + 0.025, 0.68, label, fontsize=15, fontweight="bold",
            color="#2F6B3C" if selected else "#1F4E79",
        )
        axis.text(x + 0.025, 0.51, title, fontsize=11.5, fontweight="bold")
        axis.text(x + 0.025, 0.29, body, fontsize=9.8, linespacing=1.35)
        if index < len(boxes) - 1:
            axis.add_patch(FancyArrowPatch(
                (x + 0.21, 0.51), (x + 0.245, 0.51),
                arrowstyle="-|>", mutation_scale=16,
                linewidth=1.6, color="#666666",
            ))


def _plot_resolution_choice(axis, manifest):
    levels = np.arange(3)
    two_hill_nodes = np.asarray((81 * 51, 161 * 101, 321 * 201))
    action_counts = np.asarray((3 * 9, 9 * 9, 33 * 9))
    bars = axis.bar(
        levels - 0.17, two_hill_nodes, width=0.34,
        color=LEVEL_COLORS, label="Position nodes (two hill)",
    )
    axis.set_yscale("log")
    axis.set_xticks(levels, ("L0", "L1", "L2"))
    axis.set_ylabel("Position-grid nodes (log scale)")
    axis.set_title(
        "1. Use the finest tested position/action grid\n"
        "Frozen L2: 33 directions × 9 speeds = 297 choices/state",
        fontsize=12.3, fontweight="bold",
    )
    axis.grid(axis="y", which="both", alpha=0.20)
    for bar, value in zip(bars, two_hill_nodes):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.12, f"{value:,}", ha="center", fontsize=8.8,
        )
    twin = axis.twinx()
    twin.plot(
        levels + 0.17, action_counts, color="#E45756",
        marker="o", linewidth=2.2, label="Movement choices/state",
    )
    twin.set_ylabel("Movement choices per state")
    twin.set_ylim(0.0, 335.0)
    for level, value in zip(levels, action_counts):
        twin.annotate(
            str(value), (level + 0.17, value), xytext=(0, 8),
            textcoords="offset points", ha="center", fontsize=9.2,
            color="#A3332A",
        )
    axis.axvspan(1.68, 2.32, color="#54A24B", alpha=0.10)


def _plot_speed_choice(axis, manifest):
    evidence = manifest["selection_evidence"]
    terrain_names = ("two_hill", "single_hill", "goal_in_valley")
    ratios = []
    for terrain_name in terrain_names:
        item = evidence[terrain_name]
        ratios.append(
            item["maximum_speed_sensitivity"]
            / item["selection_tolerance"]
        )
    ratios = np.asarray(ratios)
    colors = ["#E45756" if ratio > 1.0 else "#54A24B" for ratio in ratios]
    bars = axis.bar(np.arange(3), ratios, color=colors, width=0.62)
    axis.axhline(
        1.0, color="black", linestyle="--", linewidth=1.5,
        label="Allowed sensitivity",
    )
    axis.set_xticks(
        np.arange(3), [TERRAIN_LABELS[name] for name in terrain_names],
        rotation=12,
    )
    axis.set_ylabel("Speed sensitivity / allowed tolerance")
    axis.set_title(
        "2. Use nine speeds because one terrain needs them",
        fontsize=12.3, fontweight="bold",
    )
    axis.set_ylim(0.0, max(4.0, float(np.max(ratios)) * 1.25))
    axis.grid(axis="y", alpha=0.20)
    axis.legend(loc="upper right", fontsize=8.8)
    for bar, ratio in zip(bars, ratios):
        label = f"{ratio:.2f}×" if ratio > 0.0 else "0×"
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(ratio, 0.04) + 0.10,
            label, ha="center", fontsize=9.5, fontweight="bold",
        )
    axis.text(
        0.62, 0.72,
        "Two hill exceeds the limit\n"
        "→ retain 9 speeds everywhere",
        transform=axis.transAxes, ha="center", va="center", fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.3", "fc": "#FFF3E0", "ec": "#E45756"},
    )


def _plot_contract(axis, manifest):
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    axis.set_title(
        "3. Freeze planning and final replay checks",
        fontsize=12.3, fontweight="bold", pad=12,
    )
    cards = (
        (0.79, "Movement-cost calculation", "9 points / segment",
         "Trying 17 points did not change any selected B policy."),
        (0.50, "Final continuous replay", "1025 points / segment",
         "All B2/B3 feasible policies passed comparison with 2049."),
        (0.21, "C-lite contract", "Re-search every sensor candidate",
         "Do not reuse P2 sensor locations as the final optimum."),
    )
    for y, title, value, note in cards:
        box = FancyBboxPatch(
            (0.05, y - 0.115), 0.90, 0.23,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.4, edgecolor="#4C78A8", facecolor="#F7FAFC",
        )
        axis.add_patch(box)
        axis.text(
            0.09, y + 0.055, title, fontsize=10.0, fontweight="bold"
        )
        axis.text(
            0.09, y - 0.005, value, fontsize=11.0, fontweight="bold",
            color="#1F4E79",
        )
        axis.text(
            0.09, y - 0.075, note, fontsize=8.7, color="#444444"
        )
    axis.text(
        0.5, 0.015,
        manifest["production_configuration_id"],
        ha="center", va="bottom", fontsize=9.8, family="monospace",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#E8F5E9", "ec": "#2F6B3C"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/direction_b/b4_production_lattice_freeze.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/direction_b/figures/b4_production_lattice_freeze.png"
        ),
    )
    arguments = parser.parse_args()
    output = create_b4_figure(arguments.manifest, arguments.output)
    print(json.dumps({"figure": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()

"""Plot measured and power-law-extrapolated runtime versus spatial resolution."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import project_paths


INPUT_NAME = "full_sse_resolution_plots_manifest.json"
OUTPUT_STEM = "full_sse_runtime_power_law_extrapolation"
PREDICTED_DX_M = np.asarray([5.0, 2.5, 1.0, 0.1], dtype=float)
OUTPUT_DIR = (
    project_paths.CORE_DIR
    / "figure"
    / "phase_4_multi_terrain_dqn"
    / "stepped_pyramid_resolution_comparison"
)


def fit_power_law(
    dx: np.ndarray, runtime_sec: np.ndarray,
) -> tuple[float, float, float]:
    """Return coefficient, exponent, and log-space R-squared for t=C*dx**p."""
    log_dx = np.log(dx)
    log_runtime = np.log(runtime_sec)
    exponent, log_coefficient = np.polyfit(log_dx, log_runtime, deg=1)
    fitted = log_coefficient + exponent * log_dx
    residual = float(np.sum((log_runtime - fitted) ** 2))
    total = float(np.sum((log_runtime - np.mean(log_runtime)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return float(np.exp(log_coefficient)), float(exponent), r_squared


def main() -> None:
    source_path = OUTPUT_DIR / INPUT_NAME
    source = json.loads(source_path.read_text(encoding="utf-8"))
    measured_rows = sorted(source["rows"], key=lambda row: float(row["dx"]))
    measured_dx = np.asarray([row["dx"] for row in measured_rows], dtype=float)

    series = {
        "bellman_full_sse_sec": {
            "label": "Bellman full SSE",
            "color": "#0072B2",
        },
        "dqn_full_sse_sec": {
            "label": "DQN full SSE (training excluded)",
            "color": "#E69F00",
        },
        "training_mean_sec": {
            "label": "DQN training (3-seed mean)",
            "color": "#009E73",
        },
    }

    estimates: dict[str, dict[str, object]] = {}
    figure, axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    curve_dx = np.geomspace(float(PREDICTED_DX_M.min()), float(measured_dx.max()), 400)

    for key, settings in series.items():
        measured_time = np.asarray(
            [row[key] for row in measured_rows], dtype=float,
        )
        coefficient, exponent, r_squared = fit_power_law(measured_dx, measured_time)
        predicted_time = coefficient * PREDICTED_DX_M ** exponent
        curve_time = coefficient * curve_dx ** exponent
        estimates[key] = {
            "label": settings["label"],
            "model": "runtime_sec = coefficient * dx_m ** exponent",
            "coefficient": coefficient,
            "exponent": exponent,
            "log_space_r_squared": r_squared,
            "measured": [
                {"dx_m": float(x), "runtime_sec": float(y)}
                for x, y in zip(measured_dx, measured_time, strict=True)
            ],
            "predicted": [
                {"dx_m": float(x), "runtime_sec": float(y)}
                for x, y in zip(PREDICTED_DX_M, predicted_time, strict=True)
            ],
        }

        color = str(settings["color"])
        label = str(settings["label"])
        axis.plot(
            curve_dx, curve_time, color=color, linewidth=1.9, linestyle="--",
            alpha=0.85, label=f"{label} power-law fit ($R^2$={r_squared:.3f})",
        )
        axis.plot(
            measured_dx, measured_time, color=color, linewidth=2.4,
            marker="o", markersize=7, label=f"{label} measured",
        )
        axis.scatter(
            PREDICTED_DX_M, predicted_time, color=color, marker="D", s=54,
            facecolors="white", linewidths=1.8, zorder=5,
            label=f"{label} extrapolated",
        )

    axis.axvspan(
        float(PREDICTED_DX_M.min()), float(measured_dx.min()),
        color="#999999", alpha=0.08,
    )
    axis.axvline(
        float(measured_dx.min()), color="#555555", linestyle=":", linewidth=1.2,
    )
    axis.text(
        0.13, 0.03, "extrapolation region", transform=axis.get_xaxis_transform(),
        color="#555555", fontsize=9,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    all_ticks = sorted(set(measured_dx.tolist() + PREDICTED_DX_M.tolist()))
    axis.set_xticks(all_ticks)
    axis.set_xticklabels([f"{value:g}" for value in all_ticks])
    axis.set_xlabel(r"Spatial discretization $dx=dy=dh$ [m] (log scale)")
    axis.set_ylabel("Wall-clock time [s] (log scale)")
    axis.set_title(
        "Stepped-pyramid runtime: measured values and power-law extrapolation"
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best", fontsize=8.2, ncol=1, frameon=True)

    output_png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(figure)

    output_json = OUTPUT_DIR / f"{OUTPUT_STEM}.json"
    payload = {
        "condition": {
            "terrain": "stepped_pyramid",
            "heading_spacing_deg": 5.0,
            "r_neighbor": 1,
        },
        "method": (
            "independent ordinary least-squares fits of log(runtime_sec) against "
            "log(dx_m), equivalent to runtime_sec=coefficient*dx_m**exponent"
        ),
        "measured_dx_m": measured_dx.tolist(),
        "extrapolated_dx_m": PREDICTED_DX_M.tolist(),
        "training_time_definition": source["training_time_definition"],
        "source": str(source_path).replace("\\", "/"),
        "dpi": 300,
        "series": estimates,
        "output_png": str(output_png).replace("\\", "/"),
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    output_csv = OUTPUT_DIR / f"{OUTPUT_STEM}.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dx_m", "value_type", "bellman_full_sse_sec",
                "dqn_full_sse_sec", "training_mean_sec",
            ],
        )
        writer.writeheader()
        for index, dx_m in enumerate(measured_dx):
            writer.writerow({
                "dx_m": float(dx_m),
                "value_type": "measured",
                **{
                    key: estimates[key]["measured"][index]["runtime_sec"]
                    for key in series
                },
            })
        for index, dx_m in enumerate(PREDICTED_DX_M):
            writer.writerow({
                "dx_m": float(dx_m),
                "value_type": "extrapolated_power_law",
                **{
                    key: estimates[key]["predicted"][index]["runtime_sec"]
                    for key in series
                },
            })

    print(output_png)
    print(output_json)
    print(output_csv)


if __name__ == "__main__":
    main()

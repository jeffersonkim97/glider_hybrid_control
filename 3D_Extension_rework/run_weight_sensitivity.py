"""Generate Stage-9 fixed-sensor and cached-candidate weight sensitivity."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .configuration import build_configuration
from .weight_sensitivity import analyze_weight_sensitivity


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "results" / "stage_8_stackelberg" / "stackelberg_solution.json"
RESULT_DIR = ROOT / "results" / "stage_9_weight_sensitivity"
FIGURE_DIR = ROOT / "figures"
JSON_PATH = RESULT_DIR / "defender_weight_sensitivity.json"
CSV_PATH = RESULT_DIR / "defender_weight_sensitivity.csv"
PNG_PATH = FIGURE_DIR / "stage_9_defender_weight_sensitivity.png"
PDF_PATH = FIGURE_DIR / "stage_9_defender_weight_sensitivity.pdf"


def _save_csv(result: dict) -> None:
    columns = (
        "w_pod", "w_coverage", "fixed_objective", "fixed_weighted_pod",
        "fixed_weighted_coverage", "selected_x_m", "selected_y_m",
        "selected_z_m", "selected_objective", "selected_mission_pod",
        "selected_pod_normalized", "selected_coverage_volume_normalized",
        "selected_attacker_objective", "switch_x_m", "switch_y_m", "switch_h_m",
    )
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in result["reported_weights"]:
            fixed = row["fixed_reference"]
            selected = row["cached_candidate_selection"]
            sensor = selected["sensor_position_m"]
            switching = selected["switching_point_m"]
            writer.writerow({
                "w_pod": fixed["w_pod"], "w_coverage": fixed["w_coverage"],
                "fixed_objective": fixed["defender_objective"],
                "fixed_weighted_pod": fixed["weighted_pod"],
                "fixed_weighted_coverage": fixed["weighted_coverage"],
                "selected_x_m": sensor[0], "selected_y_m": sensor[1],
                "selected_z_m": sensor[2],
                "selected_objective": selected["defender_objective"],
                "selected_mission_pod": selected["mission_pod"],
                "selected_pod_normalized": selected["pod_normalized"],
                "selected_coverage_volume_normalized": selected["coverage_volume_normalized"],
                "selected_attacker_objective": selected["attacker_objective"],
                "switch_x_m": switching[0], "switch_y_m": switching[1],
                "switch_h_m": switching[2],
            })


def _transition_weights(result: dict) -> list[float]:
    return [
        float(item["coverage_weight_end"])
        for item in result["selection_segments"][:-1]
    ]


def _create_figure(result: dict) -> plt.Figure:
    dense = result["dense"]
    weights = np.asarray(dense["coverage_weights"], dtype=float)
    fixed = dense["fixed_reference"]
    selected = dense["cached_candidate_selection"]
    fixed_pod = np.asarray([item["weighted_pod"] for item in fixed])
    fixed_coverage = np.asarray([item["weighted_coverage"] for item in fixed])
    fixed_total = np.asarray([item["defender_objective"] for item in fixed])
    sensor = np.asarray([item["sensor_position_m"] for item in selected])
    selected_pod = 100.0 * np.asarray([item["mission_pod"] for item in selected])
    selected_coverage = 100.0 * np.asarray([
        item["coverage_volume_normalized"] for item in selected
    ])
    selected_attacker = np.asarray([item["attacker_objective"] for item in selected])
    tradeoff = result["candidate_tradeoff"]
    tradeoff_coverage = np.asarray([
        item["coverage_volume_normalized"] for item in tradeoff
    ])
    tradeoff_pod = np.asarray([item["pod_normalized"] for item in tradeoff])
    reference = result["reference_sensor"]
    transitions = _transition_weights(result)

    figure, axes = plt.subplots(2, 2, figsize=(15.8, 10.0))
    axis_fixed, axis_sensor, axis_response, axis_tradeoff = axes.flat

    axis_fixed.plot(weights, fixed_total, color="black", linewidth=3.0,
                    label=r"Total $J_D$")
    axis_fixed.plot(weights, fixed_pod, color="#d73027", linewidth=2.2,
                    label=r"$w_{PoD}P_{norm}$")
    axis_fixed.plot(weights, fixed_coverage, color="#4575b4", linewidth=2.2,
                    label=r"$w_{cov}C_{LOS}$")
    for row in result["reported_weights"]:
        axis_fixed.scatter(
            row["w_coverage"], row["fixed_reference"]["defender_objective"],
            color="black", s=28, zorder=5,
        )
    axis_fixed.set(
        xlabel=r"Coverage weight $w_{cov}$  ($w_{PoD}=1-w_{cov}$)",
        ylabel="Weighted objective contribution",
        title=(
            "A  Fixed center sensor: objective decomposition\n"
            f"(2500, 500, 8.79 m):  $P_{{norm}}$={reference['pod_normalized']:.5f}, "
            f"$C_{{LOS}}$={reference['coverage_volume_normalized']:.5f}"
        ),
    )
    axis_fixed.grid(alpha=0.25)
    axis_fixed.legend(loc="upper left")

    axis_sensor.step(weights, sensor[:, 0], where="post", linewidth=2.5,
                     color="#1b9e77", label=r"$x_{sensor}$")
    axis_sensor.step(weights, sensor[:, 1], where="post", linewidth=2.5,
                     color="#7570b3", label=r"$y_{sensor}$")
    axis_sensor.set(
        xlabel=r"Coverage weight $w_{cov}$",
        ylabel="Selected horizontal coordinate [m]",
        title="B  Best cached sensor candidate", ylim=(-80.0, 2650.0),
    )
    axis_sensor.grid(alpha=0.25)
    axis_sensor.legend(loc="center left")

    axis_response.step(weights, selected_pod, where="post", color="#d73027",
                       linewidth=2.5, label="Mission PoD")
    axis_response.step(weights, selected_coverage, where="post", color="#4575b4",
                       linewidth=2.5, label="LOS volume")
    axis_response.set(
        xlabel=r"Coverage weight $w_{cov}$", ylabel="Selected response [%]",
        title="C  Raw outcome after candidate reselection",
    )
    axis_response.grid(alpha=0.25)
    attacker_axis = axis_response.twinx()
    attacker_axis.step(weights, selected_attacker, where="post", color="#fdae61",
                       linestyle="--", linewidth=2.0, label="Attacker objective")
    attacker_axis.set_ylabel("Attacker objective", color="#b65f00")
    handles, labels = axis_response.get_legend_handles_labels()
    extra_handles, extra_labels = attacker_axis.get_legend_handles_labels()
    axis_response.legend(handles + extra_handles, labels + extra_labels,
                         loc="center left")

    axis_tradeoff.scatter(
        tradeoff_coverage, tradeoff_pod, color="#999999", edgecolor="black",
        linewidth=0.45, s=70, alpha=0.75, label="Feasible cached candidates",
    )
    highlighted = []
    for segment in result["selection_segments"]:
        xy = segment["sensor_position_m"][:2]
        match = next(item for item in tradeoff if np.allclose(
            item["sensor_position_m"][:2], xy, rtol=0.0, atol=1.0e-9,
        ))
        identity = tuple(xy)
        if identity in highlighted:
            continue
        highlighted.append(identity)
        axis_tradeoff.scatter(
            match["coverage_volume_normalized"], match["pod_normalized"],
            marker="*", s=300, color="#ffd92f", edgecolor="black", zorder=8,
        )
        is_right_edge = match["coverage_volume_normalized"] > 0.64
        axis_tradeoff.annotate(
            f"({xy[0]:.0f}, {xy[1]:.0f})",
            (match["coverage_volume_normalized"], match["pod_normalized"]),
            xytext=(-8, 6) if is_right_edge else (8, -14),
            ha="right" if is_right_edge else "left",
            textcoords="offset points", fontsize=9.0,
        )
    axis_tradeoff.set(
        xlabel="Normalized LOS coverage volume",
        ylabel="Normalized PoD component", title="D  Cached candidate tradeoff",
    )
    axis_tradeoff.grid(alpha=0.25)
    axis_tradeoff.legend(loc="best")
    axis_tradeoff.margins(x=0.08, y=0.08)

    for transition in transitions:
        for axis in (axis_fixed, axis_sensor, axis_response):
            axis.axvline(transition, color="#7f3b08", linestyle=":", linewidth=1.8)
        axis_sensor.text(
            transition, 0.97, f" switch ≈ {transition:.3f}",
            transform=axis_sensor.get_xaxis_transform(), ha="center", va="top",
            color="#7f3b08", fontsize=9.0,
        )

    figure.suptitle(
        "Stage 9 - Defender Weight Sensitivity\n"
        "fixed-sensor scale check and cached-candidate Stackelberg screening",
        fontsize=15, fontweight="bold", y=0.985,
    )
    figure.subplots_adjust(
        left=0.065, right=0.965, bottom=0.075, top=0.865,
        wspace=0.24, hspace=0.34,
    )
    return figure


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Run Stage 8 first: {SOURCE_PATH}")
    configuration = build_configuration()
    solution = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    result = analyze_weight_sensitivity(solution, configuration["weight_sensitivity"])
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _save_csv(result)
    figure = _create_figure(result)
    figure.savefig(PNG_PATH, dpi=230, bbox_inches="tight")
    figure.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({
        "reference_sensor": result["reference_sensor"],
        "selection_segments": result["selection_segments"],
        "reported_weights": result["reported_weights"],
    }, indent=2), flush=True)
    print(JSON_PATH, flush=True)
    print(CSV_PATH, flush=True)
    print(PNG_PATH, flush=True)


if __name__ == "__main__":
    main()

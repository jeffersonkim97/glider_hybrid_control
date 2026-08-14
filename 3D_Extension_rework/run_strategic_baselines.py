"""Run the four strategic Defender baselines on the centered 3D one-hill case."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .geometry import build_geometry
from .stackelberg import (
    evaluate_defender_position,
    make_cached_evaluator,
    solve_stackelberg_game,
)
from .strategic_baselines import (
    compute_nominal_attacker_response,
    evaluate_selected_sensors,
    reconcile_stackelberg_candidate,
    select_coverage_only_sensor,
    select_fixed_sensor,
    select_nominal_path_sensor,
)
from .terrain_scenarios import build_scenario_configuration


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "stage_11_strategic_baselines" / "centered_single_hill"
CACHE_DIR = RESULT_DIR / "evaluation_cache"
CHECKPOINT_PATH = RESULT_DIR / "selection_checkpoint.json"
RESULT_PATH = RESULT_DIR / "strategic_baseline_results.json"
FIGURE_DIR = ROOT / "figures" / "stage_11_strategic_baselines"
PNG_PATH = FIGURE_DIR / "centered_single_hill_strategic_baselines.png"
PDF_PATH = FIGURE_DIR / "centered_single_hill_strategic_baselines.pdf"
DIRECT_MAXFUN = 80
STRATEGY_ORDER = ("fixed", "coverage_only", "nominal_path", "stackelberg")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8",
    )


def _load_checkpoint() -> dict[str, Any]:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {
        "scenario_id": "centered_single_hill",
        "selection_rules": {},
        "selected_sensor_xy_m": {},
    }


def _selection_metadata(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "sensor_xy_m", "selection_score", "evaluation_count",
            "optimizer_success", "optimizer_message",
        )
    }


def _progress(event: dict[str, Any]) -> None:
    print(json.dumps(event, default=_json_default), flush=True)


def _create_figure(
    configuration: dict[str, Any], result: dict[str, Any],
) -> plt.Figure:
    geometry = build_geometry(configuration)
    mesh_x, mesh_y = np.meshgrid(
        geometry["x_grid"], geometry["y_grid"], indexing="ij",
    )
    evaluations = result["evaluations"]
    positions = {
        name: np.asarray(evaluations[name]["sensor_position_m"], dtype=float)
        for name in STRATEGY_ORDER
    }
    colors = {
        "fixed": "#7f7f7f", "coverage_only": "#9467bd",
        "nominal_path": "#f28e2b", "stackelberg": "#0072b2",
    }
    labels = {
        "fixed": "Fixed", "coverage_only": "Coverage-only",
        "nominal_path": "Nominal-path", "stackelberg": "Stackelberg",
    }

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), constrained_layout=True)
    axis_map, axis_objective, axis_pod, axis_coverage = axes.flat
    terrain = geometry["terrain_height"]
    filled = axis_map.contourf(
        mesh_x, mesh_y, terrain, levels=18, cmap="terrain", alpha=0.86,
    )
    axis_map.contour(mesh_x, mesh_y, terrain, levels=10, colors="0.25", linewidths=0.45)
    for name in STRATEGY_ORDER:
        point = positions[name]
        axis_map.scatter(
            point[0], point[1], s=180 if name == "stackelberg" else 115,
            marker="*" if name == "stackelberg" else "o",
            color=colors[name], edgecolor="black", linewidth=1.0,
            label=f"{labels[name]} ({point[0]:.0f}, {point[1]:.0f})", zorder=8,
        )
    launch = geometry["launch_position"]
    goal = geometry["goal_position"]
    axis_map.scatter(launch[0], launch[1], marker="s", s=90, color="black", label="Launch")
    axis_map.scatter(goal[0], goal[1], marker="X", s=120, color="#1a9850", edgecolor="black", label="Goal")
    axis_map.set(
        xlim=configuration["environment"]["x_bounds_m"],
        ylim=configuration["environment"]["y_bounds_m"],
        xlabel="x [m]", ylabel="y [m]",
        title="A  Sensor selected by each strategic rule",
    )
    axis_map.legend(loc="upper left", fontsize=8.2, ncols=2)
    figure.colorbar(filled, ax=axis_map, label="Terrain height [m]")

    x = np.arange(len(STRATEGY_ORDER))
    feasible_by_strategy = {
        name: bool(evaluations[name].get("feasible", False)) for name in STRATEGY_ORDER
    }
    objectives = [
        evaluations[name]["defender_objective"] if feasible_by_strategy[name] else 0.0
        for name in STRATEGY_ORDER
    ]
    pods = [
        100.0 * evaluations[name]["mission_pod"] if feasible_by_strategy[name] else 0.0
        for name in STRATEGY_ORDER
    ]
    coverages = [
        evaluations[name]["coverage_volume_normalized"] if feasible_by_strategy[name] else 0.0
        for name in STRATEGY_ORDER
    ]
    bar_colors = [colors[name] for name in STRATEGY_ORDER]

    def bars(axis, values, title, ylabel, precision):
        rectangles = axis.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.7)
        axis.set_xticks(x, [labels[name] for name in STRATEGY_ORDER], rotation=12)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.25)
        visible_maximum = max((value for value in values if np.isfinite(value)), default=1.0)
        text_offset = max(0.025 * visible_maximum, 1.0e-4)
        for name, rectangle, value in zip(STRATEGY_ORDER, rectangles, values):
            if feasible_by_strategy[name]:
                text = f"{value:.{precision}f}"
                height = rectangle.get_height()
                color = "black"
            else:
                rectangle.set_facecolor("white")
                rectangle.set_hatch("///")
                rectangle.set_edgecolor("#b2182b")
                text = "INFEASIBLE"
                height = 0.0
                color = "#b2182b"
            axis.text(
                rectangle.get_x() + rectangle.get_width() / 2.0,
                height + text_offset, text,
                ha="center", va="bottom", fontsize=9, color=color,
                fontweight="bold" if not feasible_by_strategy[name] else "normal",
            )

    bars(axis_objective, objectives, "B  Common adaptive-response evaluation", "Defender objective", 4)
    bars(axis_pod, pods, "C  Realized mission detection", "Mission PoD [%]", 3)
    bars(axis_coverage, coverages, "D  Geometry component", "Normalized LOS coverage volume", 3)
    defender = configuration["cost"]["defender"]
    figure.suptitle(
        "Centered Single-Hill: 3D Strategic-Baseline Test\n"
        "all selected sensors re-evaluated against the same adaptive Bellman follower; "
        f"$w_{{PoD}}={defender['w_pod']:.1f}$, $w_{{cov}}={defender['w_coverage']:.1f}$",
        fontsize=15, fontweight="bold",
    )
    return figure


def main() -> None:
    configuration = build_scenario_configuration("centered_single_hill")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_checkpoint()
    selected = checkpoint["selected_sensor_xy_m"]
    rules = checkpoint["selection_rules"]

    if "fixed" not in selected:
        fixed = select_fixed_sensor(configuration)
        selected["fixed"] = list(fixed)
        rules["fixed"] = {"information": "preconfigured sensor only"}
        _write_json(CHECKPOINT_PATH, checkpoint)
        _progress({"event": "fixed_selected", "sensor_xy_m": fixed})

    if "coverage_only" not in selected:
        coverage = select_coverage_only_sensor(configuration, maxfun=DIRECT_MAXFUN)
        selected["coverage_only"] = list(coverage["sensor_xy_m"])
        rules["coverage_only"] = {
            "information": "geometry-only normalized LOS volume",
            **_selection_metadata(coverage),
        }
        _write_json(CHECKPOINT_PATH, checkpoint)
        _progress({"event": "coverage_only_selected", **rules["coverage_only"]})

    if "nominal_path" not in selected:
        reference = select_fixed_sensor(configuration)
        _progress({"event": "nominal_time_only_solve_started", "reference_sensor_xy_m": reference})
        nominal = compute_nominal_attacker_response(configuration, reference)
        nominal_selection = select_nominal_path_sensor(
            configuration, nominal, maxfun=DIRECT_MAXFUN,
        )
        selected["nominal_path"] = list(nominal_selection["sensor_xy_m"])
        rules["nominal_path"] = {
            "information": "hazard against one fixed time-only Bellman path",
            "reference_sensor_xy_m": list(reference),
            "nominal_mission_time_s": nominal["summary"]["mission_time_s"],
            "nominal_switching_point_m": nominal["summary"]["switching_point_m"],
            **_selection_metadata(nominal_selection),
        }
        _write_json(CHECKPOINT_PATH, checkpoint)
        _progress({"event": "nominal_path_selected", **rules["nominal_path"]})

    _progress({"event": "stackelberg_search_started"})
    stackelberg = solve_stackelberg_game(
        configuration, cache_dir=CACHE_DIR, progress_callback=_progress,
    )
    if not stackelberg["status"]["success"]:
        raise RuntimeError(stackelberg["status"]["message"])
    stack_record = {
        **stackelberg["final_evaluation"]["summary"],
        "feasible": True,
        "error": None,
    }
    selected["stackelberg"] = stack_record["sensor_position_m"][:2]
    rules["stackelberg"] = {
        "information": "adaptive physical Bellman best response",
        "search_metadata": stackelberg["search"]["metadata"],
        "search_levels": stackelberg["search"]["levels"],
    }
    _write_json(CHECKPOINT_PATH, checkpoint)

    evaluator, cache_stats = make_cached_evaluator(
        configuration, CACHE_DIR, progress_callback=_progress,
    )
    baseline_positions = {
        name: tuple(selected[name]) for name in ("fixed", "coverage_only", "nominal_path")
    }
    baseline_evaluations = evaluate_selected_sensors(baseline_positions, evaluator)
    reconciliation = reconcile_stackelberg_candidate(stack_record, baseline_evaluations)
    if reconciliation["promoted"]:
        promoted_xy = tuple(reconciliation["record"]["sensor_position_m"][:2])
        _progress({
            "event": "stackelberg_candidate_promoted",
            "source": reconciliation["source"], "sensor_xy_m": promoted_xy,
        })
        promoted_full = evaluate_defender_position(
            promoted_xy, configuration, retain_full_pipeline=True,
        )
        stack_record = {**promoted_full["summary"], "feasible": True, "error": None}
    selected["stackelberg"] = stack_record["sensor_position_m"][:2]
    evaluations = {**baseline_evaluations, "stackelberg": stack_record}
    feasible = all(evaluations[name].get("feasible", False) for name in STRATEGY_ORDER)
    maximum_baseline = max(
        evaluations[name]["defender_objective"]
        for name in ("fixed", "coverage_only", "nominal_path")
        if evaluations[name].get("feasible", False)
    )
    checks = {
        "all_four_rules_evaluated": set(evaluations) == set(STRATEGY_ORDER),
        "all_four_outcomes_classified": all(
            isinstance(evaluations[name].get("feasible"), bool)
            for name in STRATEGY_ORDER
        ),
        "all_feasible_outcomes_validated": all(
            not evaluations[name].get("feasible", False)
            or evaluations[name].get("validation_passed", False)
            for name in STRATEGY_ORDER
        ),
        "stackelberg_dominates_evaluated_baselines": (
            stack_record["defender_objective"] >= maximum_baseline - 1.0e-12
        ),
        "stackelberg_continuous_replay_passed": stack_record["validation_passed"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "scenario_id": "centered_single_hill",
        "weights": configuration["cost"]["defender"],
        "selection_rules": rules,
        "selected_sensor_xy_m": selected,
        "evaluations": evaluations,
        "strategy_ranking": sorted(
            [name for name in STRATEGY_ORDER if evaluations[name].get("feasible", False)],
            key=lambda name: -evaluations[name]["defender_objective"],
        ),
        "infeasible_strategies": [
            name for name in STRATEGY_ORDER if not evaluations[name].get("feasible", False)
        ],
        "all_selected_strategies_feasible": feasible,
        "stackelberg_reconciliation": {
            "promoted": reconciliation["promoted"],
            "source": reconciliation["source"],
        },
        "cache_stats_after_search": cache_stats,
        "validation": {"passed": not failed, "checks": checks, "failed_checks": failed},
    }
    _write_json(RESULT_PATH, result)
    _write_json(CHECKPOINT_PATH, checkpoint)
    figure = _create_figure(configuration, result)
    figure.savefig(PNG_PATH, dpi=230, bbox_inches="tight", facecolor="white")
    figure.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(RESULT_PATH, flush=True)
    print(PNG_PATH, flush=True)
    print(PDF_PATH, flush=True)
    print(json.dumps({
        "strategy_ranking": result["strategy_ranking"],
        "validation": result["validation"],
    }), flush=True)


if __name__ == "__main__":
    main()

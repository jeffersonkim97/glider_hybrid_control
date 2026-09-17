"""Stage 14.7 theoretical/empirical scaling analysis from frozen measurements."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go

from stage11_notebook_support import save_figure_png, write_json


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_7_scaling_analysis"
STAGE14_7_SCHEMA_VERSION = "stage14.7-v2-resolution-axes"
MINIMUM_FIT_POINTS = 3

INPUTS = {
    "spatial": (
        ROOT / "figure" / "stage_14_3_spatial_resolution" / "spatial_sweep_summaries.json",
        ROOT / "figure" / "stage_14_3_spatial_resolution" / "raw_spatial_repetitions.json",
    ),
    "heading": (
        ROOT / "figure" / "stage_14_4_heading_resolution" / "heading_sweep_summaries.json",
        ROOT / "figure" / "stage_14_4_heading_resolution" / "raw_heading_repetitions.json",
    ),
    "switching": (
        ROOT / "figure" / "stage_14_5_switching_candidates" / "candidate_sweep_summaries.json",
        ROOT / "figure" / "stage_14_5_switching_candidates" / "raw_candidate_repetitions.json",
    ),
    "radius": (
        ROOT / "figure" / "stage_14_6_neighbor_radius" / "radius_sweep_summaries.json",
        ROOT / "figure" / "stage_14_6_neighbor_radius" / "raw_radius_repetitions.json",
    ),
}

FIGURE_FILES = (
    "01_total_runtime_vs_spatial_resolution.png",
    "02_total_runtime_vs_heading_resolution.png",
    "03_total_runtime_vs_switching_contour_resolution.png",
    "04_total_local_runtime_vs_neighbor_radius.png",
    "05_defender_search_work_vs_neighbor_radius.png",
    "06_bellman_runtime_vs_discretization.png",
    "07_peak_memory_vs_discretization.png",
    "08_runtime_decomposition_by_sweep.png",
    "09_objective_resolution_diagnostics.png",
)

ADDITIVE_RUNTIME_COMPONENTS = (
    ("T_graph_s", "Shared graph", "#4472C4"),
    ("T_LOS_s", "LOS", "#70AD47"),
    ("T_switch_s", "Switching", "#FFC000"),
    ("T_hazard_s", "Hazard", "#ED7D31"),
    ("T_Bellman_s", "Bellman", "#A64D79"),
    ("T_attacker_residual_s", "Attacker residual", "#5B9BD5"),
    ("T_local_search_control_s", "Local-search control", "#8064A2"),
    ("T_validation_s", "Validation", "#9BBB59"),
    ("T_wrapper_residual_s", "Wrapper residual", "#C0504D"),
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stat_median(statistic: dict[str, Any] | None) -> float | None:
    return None if statistic is None else float(statistic["median"])


def _timing_statistics(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("runtime_component_statistics_s") or summary.get("runtime_statistics_s") or {}


def _normalize_summary(sweep: str, summary: dict[str, Any]) -> dict[str, Any]:
    timing = _timing_statistics(summary)
    counts = summary.get("state_and_game_size") or {}
    heading = summary.get("realized_heading_grid") or {}
    parameters = ((summary.get("configuration") or {}).get("parameters") or {})
    spatial_resolution_m = float(
        summary.get("resolution_m", parameters.get("spatial_resolution_m", 25.0))
    )
    heading_resolution_deg = float(
        heading.get(
            "requested_heading_spacing_deg",
            summary.get("heading_spacing_deg", parameters.get("heading_spacing_deg", 5.0)),
        )
    )
    switching_contour_samples = int(
        summary.get(
            "requested_contour_sample_count",
            parameters.get("switching_contour_sample_count", 1),
        )
    )
    switching_contour_spacing = 1.0 / float(switching_contour_samples)
    if sweep == "spatial":
        control = spatial_resolution_m
        label = f"Δxyz={control:g} m"
    elif sweep == "heading":
        control = heading_resolution_deg
        label = f"Δψ={control:g}°"
    elif sweep == "switching":
        control = switching_contour_spacing
        label = f"Δs={control:.4g}"
    elif sweep == "radius":
        control = float(summary["r_neighbor"])
        label = f"r={int(control)}"
    else:
        raise ValueError(f"unknown sweep {sweep!r}")
    row = {
        "sweep": sweep,
        "case_id": summary["case_id"],
        "profile": summary["profile"],
        "algorithm_variant": summary["algorithm_variant"],
        "label": label,
        "control_value": control,
        "spatial_resolution_m": spatial_resolution_m,
        "heading_resolution_deg": heading_resolution_deg,
        "switching_contour_sample_count": switching_contour_samples,
        "switching_contour_spacing": switching_contour_spacing,
        "status_counts": summary["status_counts"],
        "attempted_repetitions": int(summary["attempted_repetitions"]),
        "successful_repetitions": int(summary["successful_repetitions"]),
        "T_total_s": _stat_median(timing.get("T_SSE_s")),
        "T_Bellman_s": _stat_median(timing.get("T_Bellman_s")),
        "T_graph_s": _stat_median(timing.get("T_graph_s")),
        "T_hazard_s": _stat_median(timing.get("T_hazard_s")),
        "peak_rss_bytes": _stat_median(summary.get("peak_rss_statistics_bytes")),
        "J_A": _stat_median(summary.get("J_A_statistics")),
        "J_D": _stat_median(summary.get("J_D_statistics")),
        "N_S_cart": counts.get("N_S_cart"),
        "N_S_active": counts.get("N_S_active"),
        "N_E": counts.get("N_E"),
        "B": counts.get("B"),
        "Q": counts.get("Q"),
        "N_psi": counts.get("N_psi", heading.get("heading_bin_count")),
        "N_C": counts.get("N_C_raw", counts.get("N_C", summary.get("expected_raw_candidate_count"))),
        "r_neighbor": summary.get("r_neighbor"),
        "N_eval_unique": _stat_median(summary.get("unique_evaluation_statistics")),
        "raw_evaluation_requests": _stat_median(summary.get("raw_request_statistics")),
        "cache_hits": _stat_median(summary.get("cache_hit_statistics")),
        "local_search_iterations": _stat_median(summary.get("iteration_statistics")),
        "T_BR_sum_s": _stat_median(summary.get("summed_br_statistics_s")),
        "runtime_components_s": {
            key: _stat_median(timing.get(key)) for key, _, _ in ADDITIVE_RUNTIME_COMPONENTS
        },
    }
    row["N_SB"] = (
        None if row["N_S_active"] is None or row["B"] is None
        else int(row["N_S_active"]) * int(row["B"])
    )
    row["N_EQ"] = (
        None if row["N_E"] is None or row["Q"] is None
        else int(row["N_E"]) * int(row["Q"])
    )
    return row


def load_and_normalize() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    raw_by_sweep: dict[str, list[dict[str, Any]]] = {}
    source_manifest: dict[str, Any] = {}
    for sweep, (summary_path, raw_path) in INPUTS.items():
        summaries = _read(summary_path)
        raw_rows = _read(raw_path)
        normalized.extend(_normalize_summary(sweep, item) for item in summaries)
        raw_by_sweep[sweep] = raw_rows
        source_manifest[sweep] = {
            "summary_path": str(summary_path.relative_to(ROOT)),
            "raw_path": str(raw_path.relative_to(ROOT)),
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "summary_rows": len(summaries),
            "raw_repetitions": len(raw_rows),
        }
    normalized.sort(key=lambda row: (
        ("spatial", "heading", "switching", "radius").index(row["sweep"]),
        row["algorithm_variant"], row["control_value"],
    ))
    return normalized, raw_by_sweep, source_manifest


def crosscheck_raw_and_summaries(
    normalized: list[dict[str, Any]], raw_by_sweep: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    checks = []
    for summary in normalized:
        raw = [row for row in raw_by_sweep[summary["sweep"]] if row["case_id"] == summary["case_id"]]
        completed = [row for row in raw if row["status"] == "completed"]
        status_counts = dict(sorted(Counter(row["status"] for row in raw).items()))

        def raw_median(extractor) -> float | None:
            values = [extractor(row) for row in completed]
            values = [float(value) for value in values if value is not None]
            return None if not values else float(median(values))

        observed = {
            "attempted_repetitions": len(raw),
            "successful_repetitions": len(completed),
            "status_counts": status_counts,
            "T_total_s": raw_median(lambda row: ((row.get("timing") or {}).get("totals") or {}).get("T_SSE_s")),
            "peak_rss_bytes": raw_median(lambda row: row["memory"]["peak_rss_bytes"]),
            "J_A": raw_median(lambda row: row.get("J_A")),
            "J_D": raw_median(lambda row: row.get("J_D")),
        }

        def equal(left: Any, right: Any) -> bool:
            if left is None or right is None:
                return left is right
            if isinstance(left, (float, int)) and isinstance(right, (float, int)):
                return bool(np.isclose(float(left), float(right), rtol=1.0e-12, atol=1.0e-12))
            return left == right

        fields = {
            key: equal(summary[key], observed[key]) for key in observed
        }
        checks.append({
            "sweep": summary["sweep"], "case_id": summary["case_id"],
            "algorithm_variant": summary["algorithm_variant"],
            "field_checks": fields, "passed": all(fields.values()),
        })
    return {"passed": all(item["passed"] for item in checks), "cases": checks}


def fit_power_law(
    rows: Iterable[dict[str, Any]], *, fit_id: str, x_field: str, y_field: str,
    sweep: str, algorithm_variant: str,
) -> dict[str, Any]:
    points = sorted({
        (float(row[x_field]), float(row[y_field]))
        for row in rows
        if row.get(x_field) is not None and row.get(y_field) is not None
        and float(row[x_field]) > 0.0 and float(row[y_field]) > 0.0
    })
    base = {
        "fit_id": fit_id, "sweep": sweep, "algorithm_variant": algorithm_variant,
        "independent_variable": x_field, "dependent_variable": y_field,
        "sample_count": len(points), "measured_points": [list(point) for point in points],
        "minimum_required_points": MINIMUM_FIT_POINTS,
        "empirical_fit_only": True, "formal_big_o_claim": False,
        "computational_limit_points_excluded": True,
        "extrapolation_performed": False,
    }
    if len(points) < MINIMUM_FIT_POINTS or len({x for x, _ in points}) < MINIMUM_FIT_POINTS:
        return base | {"status": "insufficient_completed_points", "c": None, "alpha": None, "r_squared": None, "fit_range": None}
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    log_x = np.log(x)
    log_y = np.log(y)
    alpha, log_c = np.polyfit(log_x, log_y, 1)
    predicted = log_c + alpha * log_x
    residual = float(np.sum((log_y - predicted) ** 2))
    total = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 if total == 0.0 and residual <= 1.0e-24 else (
        None if total == 0.0 else 1.0 - residual / total
    )
    return base | {
        "status": "completed",
        "c": float(np.exp(log_c)), "alpha": float(alpha),
        "r_squared": None if r_squared is None else float(r_squared),
        "fit_range": {"minimum": float(np.min(x)), "maximum": float(np.max(x))},
        "prediction_endpoints": [
            [float(np.min(x)), float(np.exp(log_c) * np.min(x) ** alpha)],
            [float(np.max(x)), float(np.exp(log_c) * np.max(x) ** alpha)],
        ],
    }


def compute_fits(normalized: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specifications: list[tuple[str, str, str, str]] = []
    for variant in ("local_sse", "global_oracle"):
        specifications.extend([
            ("spatial", variant, "spatial_resolution_m", "T_total_s"),
            ("spatial", variant, "N_S_cart", "T_total_s"),
            ("spatial", variant, "N_S_active", "T_total_s"),
            ("spatial", variant, "N_S_active", "T_Bellman_s"),
            ("spatial", variant, "N_E", "T_Bellman_s"),
            ("spatial", variant, "N_SB", "T_graph_s"),
            ("spatial", variant, "N_EQ", "T_hazard_s"),
            ("heading", variant, "heading_resolution_deg", "T_total_s"),
            ("heading", variant, "N_psi", "T_total_s"),
            ("heading", variant, "N_S_active", "T_Bellman_s"),
            ("heading", variant, "N_E", "T_Bellman_s"),
            ("heading", variant, "N_SB", "T_graph_s"),
            ("heading", variant, "N_EQ", "T_hazard_s"),
            ("switching", variant, "switching_contour_spacing", "T_total_s"),
            ("switching", variant, "N_C", "T_total_s"),
        ])
    specifications.extend([
        ("radius", "local_sse", "r_neighbor", "T_total_s"),
        ("radius", "local_sse", "N_eval_unique", "T_total_s"),
        ("radius", "local_sse", "N_eval_unique", "T_BR_sum_s"),
    ])
    fits = []
    for sweep, variant, x_field, y_field in specifications:
        rows = [
            row for row in normalized if row["sweep"] == sweep
            and row["algorithm_variant"] == variant and row["successful_repetitions"] > 0
        ]
        fit_id = f"{sweep}__{variant}__{y_field}_vs_{x_field}"
        fits.append(fit_power_law(
            rows, fit_id=fit_id, x_field=x_field, y_field=y_field,
            sweep=sweep, algorithm_variant=variant,
        ))
    return fits


def computational_limit_inventory(raw_by_sweep: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    noncompleted = []
    for sweep, rows in raw_by_sweep.items():
        for row in rows:
            if row["status"] != "completed":
                message = row.get("failure_message") or ""
                noncompleted.append({
                    "sweep": sweep, "case_id": row["case_id"],
                    "repetition_id": row["repetition_id"],
                    "algorithm_variant": row.get("algorithm_variant"),
                    "declared_sweep_value": row.get("declared_sweep_value"),
                    "status": row["status"],
                    "failure_category": row.get("failure_category"),
                    "failure_message": message,
                    "classification": (
                        "model_infeasible" if "infeasible" in row["status"]
                        else "computational_or_instrumentation_failure"
                    ),
                })
    return {
        "all_noncompleted_repetitions": noncompleted,
        "noncompleted_repetition_count": len(noncompleted),
        "model_infeasible_repetition_count": sum(
            item["classification"] == "model_infeasible" for item in noncompleted
        ),
        "computational_or_instrumentation_failure_repetition_count": sum(
            item["classification"] == "computational_or_instrumentation_failure"
            for item in noncompleted
        ),
        "timeout_repetition_count": sum(item["status"] == "timeout" for item in noncompleted),
        "memory_limit_repetition_count": sum(item["status"] == "memory_limit" for item in noncompleted),
    }


def _fit_lookup(fits: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["fit_id"]: item for item in fits}


def _rows(
    normalized: list[dict[str, Any]], sweep: str, variant: str,
) -> list[dict[str, Any]]:
    return sorted([
        row for row in normalized if row["sweep"] == sweep
        and row["algorithm_variant"] == variant and row["successful_repetitions"] > 0
    ], key=lambda row: row["control_value"])


def _add_fit_trace(
    figure: go.Figure, fit: dict[str, Any], *, color: str, name: str,
) -> None:
    if fit["status"] != "completed":
        return
    endpoints = fit["prediction_endpoints"]
    figure.add_trace(go.Scatter(
        x=[item[0] for item in endpoints], y=[item[1] for item in endpoints],
        mode="lines", name=f"{name} fit α={fit['alpha']:.2f}",
        line={"color": color, "dash": "dot", "width": 2},
    ))


def _layout(title: str, x_title: str, y_title: str, *, log_x: bool = False, log_y: bool = False) -> dict[str, Any]:
    return {
        "title": title,
        "xaxis": {"title": x_title, "type": "log" if log_x else "linear"},
        "yaxis": {"title": y_title, "type": "log" if log_y else "linear"},
        "template": "plotly_white",
    }


def generate_figures(
    normalized: list[dict[str, Any]], fits: list[dict[str, Any]], output: Path,
) -> list[str]:
    lookup = _fit_lookup(fits)
    written: list[str] = []

    figure = go.Figure()
    for variant, color, marker, label in (
        ("local_sse", "#4472C4", "circle", "Local"),
        ("global_oracle", "#ED7D31", "diamond", "Global"),
    ):
        rows = _rows(normalized, "spatial", variant)
        figure.add_trace(go.Scatter(
            x=[row["spatial_resolution_m"] for row in rows],
            y=[row["T_total_s"] for row in rows],
            mode="lines+markers", name=f"{label} runtime measured",
            line={"color": color, "width": 2.5},
            marker={"color": color, "symbol": marker, "size": 9},
        ))
        fit = lookup[f"spatial__{variant}__T_total_s_vs_spatial_resolution_m"]
        _add_fit_trace(figure, fit, color=color, name=label)
    representative = _rows(normalized, "spatial", "local_sse")
    for field, label, color, symbol in (
        ("N_S_cart", "Full 4D-grid states", "#7F7F7F", "square"),
        ("N_S_active", "Active reachable-corridor states", "#70AD47", "x"),
    ):
        figure.add_trace(go.Scatter(
            x=[row["spatial_resolution_m"] for row in representative],
            y=[row[field] for row in representative], yaxis="y2",
            mode="lines+markers", name=label,
            line={"color": color, "width": 2, "dash": "dash"},
            marker={"color": color, "symbol": symbol, "size": 8},
        ))
    layout = _layout(
        "Spatial sweep: total runtime and processed-state counts versus grid spacing",
        "spatial grid spacing Δx = Δy = Δh [m]",
        "median total runtime [s]", log_x=True, log_y=True,
    )
    layout["yaxis2"] = {
        "title": "representative 4D-state count", "type": "log",
        "overlaying": "y", "side": "right", "showgrid": False,
    }
    figure.update_layout(**layout)
    written.append(FIGURE_FILES[0]); save_figure_png(figure, output, written[-1])

    figure = go.Figure()
    for variant, color, marker, label in (
        ("local_sse", "#4472C4", "circle", "Local"),
        ("global_oracle", "#ED7D31", "diamond", "Global"),
    ):
        rows = _rows(normalized, "heading", variant)
        figure.add_trace(go.Scatter(
            x=[row["heading_resolution_deg"] for row in rows], y=[row["T_total_s"] for row in rows],
            mode="lines+markers", name=f"{label} measured",
            line={"color": color, "width": 2.5}, marker={"symbol": marker, "size": 9},
        ))
        _add_fit_trace(
            figure, lookup[f"heading__{variant}__T_total_s_vs_heading_resolution_deg"],
            color=color, name=label,
        )
    figure.update_layout(**_layout(
        "Heading sweep: total runtime versus heading discretization",
        "requested heading spacing Δψ [deg]", "median total runtime [s]",
        log_x=True, log_y=True,
    ))
    written.append(FIGURE_FILES[1]); save_figure_png(figure, output, written[-1])

    figure = go.Figure()
    for variant, color, marker, label in (
        ("local_sse", "#4472C4", "circle", "Local"),
        ("global_oracle", "#ED7D31", "diamond", "Global"),
    ):
        rows = _rows(normalized, "switching", variant)
        figure.add_trace(go.Scatter(
            x=[row["switching_contour_spacing"] for row in rows],
            y=[row["T_total_s"] for row in rows],
            mode="lines+markers", name=f"{label} measured",
            line={"color": color, "width": 2.5}, marker={"symbol": marker, "size": 9},
        ))
        _add_fit_trace(
            figure, lookup[f"switching__{variant}__T_total_s_vs_switching_contour_spacing"],
            color=color, name=label,
        )
    switching_reference = _rows(normalized, "switching", "local_sse")
    figure.add_trace(go.Scatter(
        x=[row["switching_contour_spacing"] for row in switching_reference],
        y=[row["N_C"] for row in switching_reference], yaxis="y2",
        mode="lines+markers", name="Raw switching candidates (diagnostic)",
        line={"color": "#70AD47", "width": 2, "dash": "dot"},
        marker={"color": "#70AD47", "symbol": "square", "size": 8},
    ))
    layout = _layout(
        "Switching sweep: total runtime versus contour discretization",
        "normalized contour-parameter spacing Δs = 1 / N_contour",
        "median total runtime [s]", log_x=True, log_y=True,
    )
    layout["yaxis2"] = {
        "title": "raw switching-candidate count N_C", "type": "log",
        "overlaying": "y", "side": "right", "showgrid": False,
    }
    figure.update_layout(**layout)
    written.append(FIGURE_FILES[2]); save_figure_png(figure, output, written[-1])

    local_radius = _rows(normalized, "radius", "local_sse")
    global_radius = _rows(normalized, "radius", "global_oracle")
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[row["r_neighbor"] for row in local_radius], y=[row["T_total_s"] for row in local_radius],
        mode="lines+markers", name="Local SSE measured",
        line={"color": "#4472C4", "width": 3}, marker={"size": 9},
    ))
    _add_fit_trace(
        figure, lookup["radius__local_sse__T_total_s_vs_r_neighbor"],
        color="#4472C4", name="Local SSE",
    )
    if global_radius:
        figure.add_trace(go.Scatter(
            x=[global_radius[0]["r_neighbor"]], y=[global_radius[0]["T_total_s"]],
            mode="markers", name="Exact global full-radius reference",
            marker={"color": "#C00000", "symbol": "diamond", "size": 12},
        ))
    figure.update_layout(**_layout(
        "Local-SSE neighborhood sweep: total runtime", "r_neighbor [Defender grid lengths]",
        "median total runtime [s]",
    ))
    written.append(FIGURE_FILES[3]); save_figure_png(figure, output, written[-1])

    figure = go.Figure()
    for field, label, color, dash in (
        ("N_eval_unique", "Unique exact BR evaluations", "#4472C4", "solid"),
        ("raw_evaluation_requests", "Raw evaluation requests", "#ED7D31", "dash"),
        ("cache_hits", "Cache hits", "#70AD47", "dot"),
    ):
        figure.add_trace(go.Scatter(
            x=[row["r_neighbor"] for row in local_radius], y=[row[field] for row in local_radius],
            mode="lines+markers", name=label,
            line={"color": color, "dash": dash, "width": 3}, marker={"size": 9},
        ))
    figure.add_trace(go.Scatter(
        x=[row["r_neighbor"] for row in local_radius],
        y=[row["local_search_iterations"] for row in local_radius],
        mode="lines+markers", name="Local-search iterations K", yaxis="y2",
        line={"color": "#A64D79", "dash": "dashdot", "width": 3},
        marker={"symbol": "diamond", "size": 9},
    ))
    layout = _layout(
        "Local-SSE neighborhood sweep: realized Defender-search work",
        "r_neighbor [Defender grid lengths]", "median action count",
    )
    layout["yaxis2"] = {"title": "median local-search iterations K", "overlaying": "y", "side": "right", "showgrid": False}
    figure.update_layout(**layout)
    written.append(FIGURE_FILES[4]); save_figure_png(figure, output, written[-1])

    grid_rows = [
        row for row in normalized if row["sweep"] in {"spatial", "heading"}
        and row["algorithm_variant"] == "local_sse" and row["successful_repetitions"] > 0
    ]
    grid_rows.sort(key=lambda row: (
        ("spatial", "heading").index(row["sweep"]), row["control_value"],
    ))
    grid_prefix = {"spatial": "P", "heading": "H"}
    figure = go.Figure()
    for sweep, color, marker in (("spatial", "#4472C4", "circle"), ("heading", "#ED7D31", "diamond")):
        subset = [row for row in grid_rows if row["sweep"] == sweep]
        x_values = [f"{grid_prefix[sweep]}:{row['label']}" for row in subset]
        figure.add_trace(go.Scatter(
            x=x_values, y=[row["T_Bellman_s"] for row in subset],
            mode="lines+markers", name=f"{sweep}: Bellman runtime",
            line={"color": color, "width": 2.5}, marker={"symbol": marker, "size": 8},
        ))
        for field, shade, symbol, label in (
            ("N_S_active", "#70AD47", "square", "active corridor states"),
            ("N_E", "#A64D79", "x", "active edges"),
        ):
            figure.add_trace(go.Scatter(
                x=x_values, y=[row[field] for row in subset], yaxis="y2",
                mode="lines+markers", name=f"{sweep}: {label}",
                line={"color": shade, "width": 1.8, "dash": "dash" if sweep == "spatial" else "dot"},
                marker={"symbol": symbol, "size": 8},
            ))
    layout = _layout(
        "Bellman runtime and graph size versus sweep-specific discretization",
        "discretization configuration (P=spatial spacing, H=heading spacing)",
        "median Bellman runtime [s]", log_y=True,
    )
    layout["yaxis2"] = {
        "title": "active state / edge count", "type": "log",
        "overlaying": "y", "side": "right", "showgrid": False,
    }
    figure.update_layout(**layout)
    written.append(FIGURE_FILES[5]); save_figure_png(figure, output, written[-1])

    figure = go.Figure()
    for sweep, color, marker in (("spatial", "#4472C4", "circle"), ("heading", "#ED7D31", "diamond")):
        subset = [row for row in grid_rows if row["sweep"] == sweep]
        x_values = [f"{grid_prefix[sweep]}:{row['label']}" for row in subset]
        figure.add_trace(go.Scatter(
            x=x_values,
            y=[row["peak_rss_bytes"] / 1024.0**2 for row in subset],
            mode="lines+markers", name=f"{sweep}: peak RSS",
            line={"color": color, "width": 2.5}, marker={"symbol": marker, "size": 8},
        ))
        for field, shade, symbol, label in (
            ("N_S_active", "#70AD47", "square", "active corridor states"),
            ("N_E", "#A64D79", "x", "active edges"),
        ):
            figure.add_trace(go.Scatter(
                x=x_values, y=[row[field] for row in subset], yaxis="y2",
                mode="lines+markers", name=f"{sweep}: {label}",
                line={"color": shade, "width": 1.8, "dash": "dash" if sweep == "spatial" else "dot"},
                marker={"symbol": symbol, "size": 8},
            ))
    layout = _layout(
        "Peak memory and graph size versus sweep-specific discretization",
        "discretization configuration (P=spatial spacing, H=heading spacing)",
        "median peak RSS [MiB]",
    )
    layout["yaxis2"] = {
        "title": "active state / edge count", "type": "log",
        "overlaying": "y", "side": "right", "showgrid": False,
    }
    figure.update_layout(**layout)
    written.append(FIGURE_FILES[6]); save_figure_png(figure, output, written[-1])

    decomposition_rows = sorted([
        row for row in normalized if row["algorithm_variant"] == "local_sse"
        and row["successful_repetitions"] > 0
    ], key=lambda row: (("spatial", "heading", "switching", "radius").index(row["sweep"]), row["control_value"]))
    prefix = {"spatial": "P", "heading": "H", "switching": "C", "radius": "R"}
    labels = [f"{prefix[row['sweep']]}:{row['label']}" for row in decomposition_rows]
    figure = go.Figure()
    for key, label, color in ADDITIVE_RUNTIME_COMPONENTS:
        figure.add_trace(go.Bar(
            x=labels, y=[row["runtime_components_s"].get(key) or 0.0 for row in decomposition_rows],
            name=label, marker={"color": color},
        ))
    figure.update_layout(
        **_layout(
            "Local-SSE runtime decomposition across completed one-variable sweeps",
            "configuration (P=spatial, H=heading, C=switching, R=radius)",
            "median runtime component [s]",
        ), barmode="stack",
    )
    written.append(FIGURE_FILES[7]); save_figure_png(figure, output, written[-1], width_px=1900)

    figure = go.Figure()
    objective_styles = {"J_A": ("#4472C4", "circle"), "J_D": ("#70AD47", "square")}
    for sweep in ("spatial", "heading", "switching"):
        rows = _rows(normalized, sweep, "local_sse")
        for objective, (color, marker) in objective_styles.items():
            figure.add_trace(go.Scatter(
                x=[f"{prefix[sweep]}:{row['label']}" for row in rows],
                y=[row[objective] for row in rows], mode="lines+markers",
                name=f"{sweep} {objective}",
                line={"color": color, "dash": {"spatial": "solid", "heading": "dash", "switching": "dot"}[sweep], "width": 2.5},
                marker={"symbol": marker, "size": 8},
            ))
    figure.update_layout(**_layout(
        "Objective sensitivity to discretization (diagnostic, not convergence proof)",
        "completed local-SSE discretization configuration", "median objective value",
    ))
    written.append(FIGURE_FILES[8]); save_figure_png(figure, output, written[-1], width_px=1900)
    return written


def _write_fit_csv(path: Path, fits: list[dict[str, Any]]) -> None:
    fields = (
        "fit_id", "sweep", "algorithm_variant", "independent_variable",
        "dependent_variable", "status", "sample_count", "c", "alpha",
        "r_squared", "fit_min", "fit_max", "empirical_fit_only",
        "formal_big_o_claim", "extrapolation_performed",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fit in fits:
            fit_range = fit.get("fit_range") or {}
            writer.writerow({
                **{field: fit.get(field) for field in fields},
                "fit_min": fit_range.get("minimum"), "fit_max": fit_range.get("maximum"),
            })


def _interpretation_markdown(fits: list[dict[str, Any]], limits: dict[str, Any]) -> str:
    primary_ids = (
        "spatial__local_sse__T_total_s_vs_spatial_resolution_m",
        "heading__local_sse__T_total_s_vs_heading_resolution_deg",
        "switching__local_sse__T_total_s_vs_switching_contour_spacing",
        "radius__local_sse__T_total_s_vs_r_neighbor",
        "radius__local_sse__T_total_s_vs_N_eval_unique",
    )
    lookup = _fit_lookup(fits)
    lines = [
        "# Stage 14.7 theoretical-versus-empirical interpretation",
        "",
        "All exponents below are descriptive log-log fits over the measured finite range. They are not formal Big-O results and are not extrapolated.",
        "Negative exponents on spacing variables mean runtime rises as the discretization spacing becomes finer.",
        "",
    ]
    for fit_id in primary_ids:
        fit = lookup[fit_id]
        if fit["status"] == "completed":
            lines.append(
                f"- `{fit_id}`: alpha={fit['alpha']:.6g}, c={fit['c']:.6g}, "
                f"R^2={fit['r_squared']:.6g}, n={fit['sample_count']}, "
                f"range=[{fit['fit_range']['minimum']:.6g}, {fit['fit_range']['maximum']:.6g}]."
            )
        else:
            lines.append(f"- `{fit_id}`: no fit ({fit['status']}, n={fit['sample_count']}).")
    lines.extend([
        "",
        "The active-state and active-edge counts are retained separately. No artificial `N_S_active + N_E` plot variable is constructed.",
        "",
        f"Noncompleted repetitions: {limits['noncompleted_repetition_count']} total; "
        f"{limits['model_infeasible_repetition_count']} model-infeasible and "
        f"{limits['computational_or_instrumentation_failure_repetition_count']} computational/instrumentation failures.",
        "",
        "Reachability pruning, changing edge density, fixed overhead, and local-search path changes can cause departures from a simple power law.",
    ])
    return "\n".join(lines) + "\n"


def run_stage14_7_analysis(output_directory: Path = OUTPUT) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    normalized, raw_by_sweep, sources = load_and_normalize()
    crosscheck = crosscheck_raw_and_summaries(normalized, raw_by_sweep)
    fits = compute_fits(normalized)
    limits = computational_limit_inventory(raw_by_sweep)
    figures = generate_figures(normalized, fits, output_directory)
    plot_payload = {
        "schema_version": STAGE14_7_SCHEMA_VERSION,
        "normalized_measurements": normalized,
        "figure_files": figures,
        "active_state_and_edge_policy": "separate variables; never summed",
        "x_axis_policy": {
            "spatial": "spatial_resolution_m",
            "heading": "heading_resolution_deg",
            "switching": "switching_contour_spacing=1/switching_contour_sample_count",
            "neighbor_search": "r_neighbor in Defender grid lengths",
            "state_and_edge_counts": "secondary y-axis diagnostics only; never an x-axis",
        },
    }
    plot_payload_json = json.dumps(plot_payload, sort_keys=True, separators=(",", ":"))
    plot_digest = hashlib.sha256(plot_payload_json.encode("utf-8")).hexdigest()
    completed_fits = [fit for fit in fits if fit["status"] == "completed"]
    checks = {
        "all_required_source_artifacts_loaded": all(Path(ROOT / item["summary_path"]).is_file() and Path(ROOT / item["raw_path"]).is_file() for item in sources.values()),
        "raw_summary_crosscheck_passes": crosscheck["passed"],
        "all_completed_fits_have_required_metadata": bool(completed_fits) and all(
            fit["c"] is not None and fit["alpha"] is not None
            and fit["r_squared"] is not None and fit["sample_count"] >= MINIMUM_FIT_POINTS
            and fit["fit_range"] is not None for fit in completed_fits
        ),
        "empirical_and_formal_claims_distinguished": all(
            fit["empirical_fit_only"] is True and fit["formal_big_o_claim"] is False
            for fit in fits
        ),
        "no_fit_extrapolates_beyond_measured_range": all(
            fit["extrapolation_performed"] is False for fit in fits
        ),
        "computational_limit_and_infeasible_cases_retained": limits["noncompleted_repetition_count"] > 0,
        "active_states_and_edges_never_artificially_summed": all(
            "N_S_active+N_E" not in fit["independent_variable"]
            and "N_S_active + N_E" not in fit["independent_variable"] for fit in fits
        ),
        "all_figure_x_axes_use_declared_discretization_controls": all(
            row["spatial_resolution_m"] > 0.0
            and row["heading_resolution_deg"] > 0.0
            and row["switching_contour_spacing"] > 0.0
            for row in normalized
        ),
        "nine_required_static_png_figures_exist": figures == list(FIGURE_FILES) and all(
            (output_directory / name).is_file()
            and (output_directory / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
            for name in figures
        ),
        "analysis_started_no_solver_or_benchmark_worker": True,
    }
    gate_passed = all(checks.values())
    write_json(output_directory / "source_artifact_manifest.json", sources)
    write_json(output_directory / "normalized_scaling_measurements.json", normalized)
    write_json(output_directory / "raw_summary_crosscheck.json", crosscheck)
    write_json(output_directory / "fit_parameters.json", fits)
    _write_fit_csv(output_directory / "fit_parameters.csv", fits)
    write_json(output_directory / "computational_limit_cases.json", limits)
    write_json(output_directory / "plot_data.json", plot_payload | {"sha256": plot_digest})
    (output_directory / "theoretical_vs_empirical_interpretation.md").write_text(
        _interpretation_markdown(fits, limits), encoding="utf-8",
    )
    validation = {
        "schema_version": STAGE14_7_SCHEMA_VERSION,
        "gate_passed": gate_passed,
        "checks": checks,
        "theoretical_references": {
            "Bellman": "O(N_S + N_E)",
            "graph": "approximately O(N_S B)",
            "hazard": "approximately O(N_E Q)",
            "local_sse": "T_shared + sum_{d in E(r)} T_BR(d) + T_local_overhead(r)",
        },
        "claim_boundary": "empirical exponents are finite-range descriptive fits, never formal Big-O",
        "next_stage_started": False,
    }
    write_json(output_directory / "stage14_7_validation_report.json", validation)
    summary = {
        "stage": "14.7", "schema_version": STAGE14_7_SCHEMA_VERSION,
        "gate_passed": gate_passed,
        "source_sweeps": list(INPUTS),
        "normalized_configuration_count": len(normalized),
        "completed_configuration_count": sum(row["successful_repetitions"] > 0 for row in normalized),
        "fit_record_count": len(fits),
        "completed_fit_count": len(completed_fits),
        "noncompleted_repetition_count": limits["noncompleted_repetition_count"],
        "figure_files": figures,
        "x_axis_policy": plot_payload["x_axis_policy"],
        "plot_data_sha256": plot_digest,
        "solver_or_benchmark_workers_started": 0,
        "gate_checks": checks,
        "next_stage_started": False,
    }
    write_json(output_directory / "stage14_7_summary.json", summary)
    if not gate_passed:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Stage-14.7 exit gate failed: {failed}")
    print(
        "Stage 14.7 scaling analysis: PASS; "
        f"configurations={len(normalized)}, fits={len(completed_fits)}/{len(fits)}"
    )
    return summary


def regenerate_stage14_7_figures_from_saved_data(output_directory: Path = OUTPUT) -> list[str]:
    normalized = _read(output_directory / "normalized_scaling_measurements.json")
    fits = _read(output_directory / "fit_parameters.json")
    return generate_figures(normalized, fits, output_directory)


if __name__ == "__main__":
    run_stage14_7_analysis()

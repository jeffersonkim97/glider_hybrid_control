"""Shared static figure schema for Stage 14 one-variable sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import plotly.graph_objects as go

from stage11_notebook_support import save_figure_png


VARIANTS = ("local_sse", "global_oracle")
VARIANT_STYLE = {
    "local_sse": {
        "label": "Local SSE", "color": "#4472C4",
        "dash": "solid", "symbol": "circle",
    },
    "global_oracle": {
        "label": "Global oracle", "color": "#ED7D31",
        "dash": "dash", "symbol": "diamond",
    },
}
COMPONENT_SPECS = (
    ("T_LOS_s", "LOS", "#5B9BD5"),
    ("T_graph_s", "Graph/reachability", "#70AD47"),
    ("T_hazard_s", "Hazard", "#FFC000"),
    ("T_switch_s", "Switching", "#ED7D31"),
    ("T_Bellman_s", "Bellman", "#A5A5A5"),
)
FOCUSED_COMPONENT_SPECS = (
    ("T_LOS_s", "LOS", "#5B9BD5"),
    ("T_graph_s", "backward reachability", "#70AD47"),
    ("T_hazard_s", "glide segment detection hazard", "#FFC000"),
    ("T_switch_s", "switching candidate and feasibility", "#ED7D31"),
    ("T_Bellman_s", "Cost-to-go and Bellman Recursion", "#A5A5A5"),
)


def _runtime(row: dict[str, Any], key: str = "T_SSE_s") -> float | None:
    value = ((row.get("timing") or {}).get("totals") or {}).get(key)
    return None if value is None else float(value)


def _groups(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], x_key: str,
) -> dict[str, dict[float, list[dict[str, Any]]]]:
    result = {variant: {} for variant in VARIANTS}
    for row in rows:
        variant = row.get("algorithm_variant")
        if row.get("status") != "completed" or variant not in result:
            continue
        x_value = float(metadata[row["case_id"]][x_key])
        result[variant].setdefault(x_value, []).append(row)
    return result


def _median_series(
    grouped: dict[float, list[dict[str, Any]]], value: Callable[[dict[str, Any]], float],
) -> tuple[list[float], list[float]]:
    x_values = sorted(grouped)
    return x_values, [
        float(np.median([value(row) for row in grouped[x_value]]))
        for x_value in x_values
    ]


def _representatives(
    grouped: dict[str, dict[float, list[dict[str, Any]]]],
) -> dict[str, dict[float, dict[str, Any]]]:
    return {
        variant: {
            x_value: rows[0] for x_value, rows in variant_groups.items() if rows
        }
        for variant, variant_groups in grouped.items()
    }


def _layout(figure: go.Figure, *, title: str, subtitle: str, x_title: str) -> None:
    figure.update_layout(
        title=title + f"<br><sup>{subtitle}</sup>",
        legend={
            "orientation": "h", "yanchor": "top", "y": -0.22,
            "xanchor": "center", "x": 0.5,
        },
        margin={"b": 125},
    )
    figure.update_xaxes(title_text=x_title)


def _add_raw_and_median(
    figure: go.Figure,
    grouped: dict[str, dict[float, list[dict[str, Any]]]],
    value: Callable[[dict[str, Any]], float],
    *, quantity_label: str, show_individual_runs: bool = True,
    compact_legend: bool = False,
) -> None:
    raw_symbols = {"local_sse": "circle-open", "global_oracle": "diamond"}
    for variant in VARIANTS:
        style = VARIANT_STYLE[variant]
        variant_groups = grouped[variant]
        if not variant_groups:
            continue
        raw_rows = [
            row for x_value in sorted(variant_groups)
            for row in variant_groups[x_value]
        ]
        raw_x = [
            x_value for x_value in sorted(variant_groups)
            for _ in variant_groups[x_value]
        ]
        if show_individual_runs:
            figure.add_trace(go.Scatter(
                x=raw_x, y=[value(row) for row in raw_rows], mode="markers",
                name=f"{style['label']} individual isolated-process runs (n=3)",
                marker={
                    "color": style["color"], "symbol": raw_symbols[variant],
                    "size": 7, "opacity": 0.28,
                },
            ))
        x_values, medians = _median_series(variant_groups, value)
        figure.add_trace(go.Scatter(
            x=x_values, y=medians, mode="lines+markers",
            name=(f"Median {quantity_label}" if compact_legend
                  else f"{style['label']} median {quantity_label}"),
            line={"color": style["color"], "width": 2, "dash": style["dash"]},
            marker={"color": style["color"], "symbol": style["symbol"], "size": 9},
        ))


def _median_runtime_component(
    rows: list[dict[str, Any]], key: str,
) -> float:
    values = [value for row in rows if (value := _runtime(row, key)) is not None]
    return 0.0 if not values else float(np.median(values))


def _median_other_runtime(rows: list[dict[str, Any]]) -> float:
    values = []
    component_keys = tuple(key for key, _label, _color in COMPONENT_SPECS)
    for row in rows:
        total = _runtime(row)
        if total is None:
            continue
        classified = sum(_runtime(row, key) or 0.0 for key in component_keys)
        values.append(max(0.0, total - classified))
    return 0.0 if not values else float(np.median(values))


def _trajectory_code_map(
    grouped: dict[str, dict[float, list[dict[str, Any]]]],
) -> dict[str, int]:
    hashes = sorted({
        str((row.get("trajectory_identity") or {}).get("sha256"))
        for variant in VARIANTS
        for rows in grouped[variant].values()
        for row in rows
        if (row.get("trajectory_identity") or {}).get("sha256")
    })
    return {digest: index + 1 for index, digest in enumerate(hashes)}


def build_resolution_figure_set(
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    *,
    x_key: str,
    x_title: str,
    sweep_label: str,
    output: Path,
    variants: tuple[str, ...] = VARIANTS,
    clean_performance_display: bool = False,
    include_pod_figure: bool = False,
    focused_runtime_display: bool = False,
    performance_local_only: bool = False,
) -> list[str]:
    """Write six common performance figures and an optional separate PoD figure."""
    if not variants or len(set(variants)) != len(variants) or any(
        variant not in VARIANTS for variant in variants
    ):
        raise ValueError("variants must select distinct supported algorithms")
    rows = [row for row in rows if row.get("algorithm_variant") in variants]
    grouped = _groups(rows, metadata, x_key)
    representatives = _representatives(grouped)
    subtitle = "Local SSE only; completed measurements only" if variants == ("local_sse",) else (
        "local SSE primary (solid); global finite oracle/reference (dashed); "
        "completed measurements only"
    )
    names: list[str] = []
    performance_variants = ("local_sse",) if performance_local_only else variants
    performance_grouped = (
        _groups([row for row in rows if row.get("algorithm_variant") == "local_sse"],
                metadata, x_key)
        if performance_local_only else grouped
    )
    performance_clean = clean_performance_display or performance_local_only
    performance_subtitle = (
        "Local SSE only; completed measurements only" if performance_local_only else subtitle
    )

    total = go.Figure()
    _add_raw_and_median(
        total, performance_grouped, lambda row: float(_runtime(row)), quantity_label="T_SSE",
        show_individual_runs=not performance_clean,
        compact_legend=performance_clean,
    )
    _layout(
        total, title=f"{sweep_label}: total SSE runtime",
        subtitle=performance_subtitle, x_title=x_title,
    )
    total.update_layout(yaxis_title="Total SSE runtime T_SSE [s]")
    names.append("01_total_sse_runtime.png")
    save_figure_png(total, output, names[-1])

    decomposition = go.Figure()
    component_specs = FOCUSED_COMPONENT_SPECS if focused_runtime_display else (
        COMPONENT_SPECS + (("__other__", "Other solver/control", "#8064A2"),)
    )
    for variant in performance_variants:
        style = VARIANT_STYLE[variant]
        variant_groups = performance_grouped[variant]
        x_values = sorted(variant_groups)
        for key, label, color in component_specs:
            values = [
                _median_other_runtime(variant_groups[x_value])
                if key == "__other__"
                else _median_runtime_component(variant_groups[x_value], key)
                for x_value in x_values
            ]
            decomposition.add_trace(go.Scatter(
                x=x_values, y=values, mode="lines+markers",
                name=label if performance_clean else f"{style['label']} {label}",
                line={"color": color, "width": 2, "dash": style["dash"]},
                marker={"color": color, "symbol": style["symbol"], "size": 8},
            ))
    _layout(
        decomposition, title=f"{sweep_label}: runtime decomposition",
        subtitle=(
            "Local SSE only; non-overlapping runtime components"
            if performance_variants == ("local_sse",)
            else "same non-overlapping component schema; solid local SSE, dashed global oracle"
        ),
        x_title=x_title,
    )
    decomposition.update_layout(yaxis_title="Median component time [s]")
    if focused_runtime_display:
        decomposition.update_layout(meta={"png_legend_columns": 2})
    names.append("02_runtime_decomposition.png")
    save_figure_png(decomposition, output, names[-1])

    memory = go.Figure()
    _add_raw_and_median(
        memory, performance_grouped,
        lambda row: float(row["memory"]["peak_rss_bytes"]) / 1024**2,
        quantity_label="peak RSS",
        show_individual_runs=not performance_clean,
        compact_legend=performance_clean,
    )
    _layout(
        memory, title=f"{sweep_label}: peak memory",
        subtitle=performance_subtitle, x_title=x_title,
    )
    memory.update_layout(yaxis_title="Peak process-tree RSS [MiB]")
    names.append("03_peak_memory.png")
    save_figure_png(memory, output, names[-1])

    objective = go.Figure()
    for variant in variants:
        style = VARIANT_STYLE[variant]
        variant_groups = grouped[variant]
        if not variant_groups:
            continue
        for key, objective_label, color, symbol in (
            ("J_A", "Attacker J_A", style["color"], style["symbol"]),
            ("J_D", "Defender J_D", "#70AD47" if variant == "local_sse" else "#A64D79", "square"),
        ):
            x_values, medians = _median_series(
                variant_groups, lambda row, objective_key=key: float(row[objective_key])
            )
            objective.add_trace(go.Scatter(
                x=x_values, y=medians, mode="lines+markers",
                name=(objective_label if clean_performance_display
                      else f"{style['label']} {objective_label}"),
                line={"color": color, "width": 2, "dash": style["dash"]},
                marker={"color": color, "symbol": symbol, "size": 9},
            ))
    _layout(
        objective, title=f"{sweep_label}: equilibrium objectives",
        subtitle=subtitle + "; convergence not assumed", x_title=x_title,
    )
    objective.update_layout(yaxis_title="Median objective value")
    names.append("04_objectives.png")
    save_figure_png(objective, output, names[-1])

    if include_pod_figure:
        pod = go.Figure()
        for variant in variants:
            variant_groups = grouped[variant]
            if not variant_groups:
                continue
            # Both players refer to the same event: this selected Attacker is
            # detected by this Defender. J_A mixes time and hazard and must not
            # be converted to PoD. J_D is already the measured detection PoD.
            x_values, probabilities = _median_series(
                variant_groups, lambda row: float(row["J_D"])
            )
            if any(not np.isfinite(value) or not 0.0 <= value <= 1.0
                   for value in probabilities):
                raise ValueError("measured PoD must lie in [0, 1]")
            for label, color, dash, symbol in (
                ("Attacker PoD", "#4472C4", "solid", "circle-open"),
                ("Defender PoD", "#70AD47", "dash", "square-open"),
            ):
                pod.add_trace(go.Scatter(
                    x=x_values, y=probabilities, mode="lines+markers",
                    name=label if clean_performance_display else f"{VARIANT_STYLE[variant]['label']} {label}",
                    line={"color": color, "width": 2, "dash": dash},
                    marker={"color": color, "symbol": symbol, "size": 9},
                ))
        _layout(
            pod, title=f"{sweep_label}: equilibrium PoD (Figure 4-1)",
            subtitle="Attacker being detected = Defender detecting Attacker; curves coincide",
            x_title=x_title,
        )
        pod.update_layout(yaxis_title="Median probability of detection (PoD)")
        pod.update_yaxes(range=[0.0, 1.02])
        names.append("04_1_detection_probability.png")
        save_figure_png(pod, output, names[-1])

    equilibrium = go.Figure()
    for variant in variants:
        style = VARIANT_STYLE[variant]
        variant_groups = grouped[variant]
        x_values = sorted(variant_groups)
        for key, label, yaxis in (
            ("selected_defender_action_id", "Defender action", "y"),
            ("selected_attacker_candidate_id", "Attacker candidate", "y2"),
        ):
            medians = [
                float(np.median([float(row[key]) for row in variant_groups[x_value]]))
                for x_value in x_values
            ]
            equilibrium.add_trace(go.Scatter(
                x=x_values, y=medians, mode="lines+markers+text",
                text=[str(int(value)) if value.is_integer() else str(value) for value in medians],
                textposition="top center", name=f"{style['label']} {label}",
                legendgroup=variant, yaxis=yaxis,
                line={"color": style["color"], "width": 2, "dash": style["dash"]},
                marker={
                    "color": style["color"],
                    "symbol": style["symbol"] if yaxis == "y" else "square",
                    "size": 9,
                },
            ))
    _layout(
        equilibrium, title=f"{sweep_label}: selected finite equilibrium",
        subtitle="action/candidate IDs report the selected discrete equilibrium",
        x_title=x_title,
    )
    equilibrium.update_layout(
        yaxis={"title": "Defender action ID", "dtick": 1},
        yaxis2={
            "title": "Attacker candidate ID", "overlaying": "y",
            "side": "right", "showgrid": False, "dtick": 1,
        },
    )
    names.append("05_equilibrium_selection.png")
    save_figure_png(equilibrium, output, names[-1])

    trajectory_codes = _trajectory_code_map(grouped)
    trajectory = go.Figure()
    for variant in variants:
        style = VARIANT_STYLE[variant]
        variant_representatives = representatives[variant]
        x_values = sorted(variant_representatives)
        codes = [
            trajectory_codes[str(variant_representatives[x]["trajectory_identity"]["sha256"])]
            for x in x_values
        ]
        switching_points = [
            variant_representatives[x]["trajectory_identity"]["switching_position_map"]
            for x in x_values
        ]
        trajectory.add_trace(go.Scatter(
            x=x_values, y=codes, mode="lines+markers+text",
            text=[
                f"T{code}; S=({point[0]:.3g},{point[1]:.3g},{point[2]:.3g})"
                for code, point in zip(codes, switching_points)
            ],
            textposition="top center",
            name=f"{style['label']} trajectory identity", legendgroup=variant,
            line={"color": style["color"], "width": 2, "dash": style["dash"]},
            marker={"color": style["color"], "symbol": style["symbol"], "size": 9},
        ))
    _layout(
        trajectory, title=f"{sweep_label}: selected trajectory",
        subtitle="trajectory IDs are exact SHA-256 identities; S=(x,y,z) labels the switching point in map coordinates",
        x_title=x_title,
    )
    trajectory.update_yaxes(
        title_text="Trajectory identity",
        tickmode="array",
        tickvals=list(trajectory_codes.values()),
        ticktext=[
            f"T{code} ({digest[:10]}...)"
            for digest, code in trajectory_codes.items()
        ],
    )
    names.append("06_trajectory_identity.png")
    save_figure_png(trajectory, output, names[-1])
    return names


__all__ = ["build_resolution_figure_set"]

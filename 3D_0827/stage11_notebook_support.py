"""Presentation and artifact helpers for the Stage-11 notebook.

No terrain, LOS, energy, Bellman, or validation mathematics lives here.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from html import escape
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_figure(figure: Any, directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    figure.write_html(path, include_plotlyjs="directory", full_html=True)
    return path


def _plain_plotly_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?sup>", "", text, flags=re.IGNORECASE)
    return re.sub(r"<[^>]+>", "", text)


def save_figure_png(
    figure: Any,
    directory: Path,
    filename: str,
    *,
    width_px: int = 1600,
    height_px: int = 950,
    dpi: int = 200,
) -> Path:
    """Render the simple 2D Plotly diagnostics used by Stage 14 as static PNG.

    This avoids a Kaleido/browser dependency and deliberately supports the
    Stage-14 trace vocabulary: scatter/line/text, bar, stacked bar, and h-lines.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not str(filename).lower().endswith(".png"):
        raise ValueError("static figure filename must end in .png")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    fig, axis = plt.subplots(
        figsize=(width_px / dpi, height_px / dpi), dpi=dpi,
    )
    secondary_x_axis = None
    secondary_y_axis = None
    stack_bottom: dict[tuple[Any, ...], np.ndarray] = {}
    maximum_categorical_count = 0
    marker_map = {
        "circle": "o", "circle-open": "o", "star": "*", "square": "s",
        "diamond": "D", "line-ew": "_", "x": "x", "cross": "+",
    }
    for trace in figure.data:
        target = axis
        uses_secondary_x = getattr(trace, "xaxis", None) == "x2"
        uses_secondary_y = getattr(trace, "yaxis", None) == "y2"
        if uses_secondary_x and uses_secondary_y:
            raise TypeError("simultaneous Plotly x2/y2 traces are not supported")
        if uses_secondary_x:
            if secondary_x_axis is None:
                secondary_x_axis = axis.twiny()
            target = secondary_x_axis
        elif uses_secondary_y:
            if secondary_y_axis is None:
                secondary_y_axis = axis.twinx()
            target = secondary_y_axis
        trace_type = getattr(trace, "type", "")
        x = list(trace.x) if getattr(trace, "x", None) is not None else []
        if x and all(isinstance(value, str) for value in x):
            maximum_categorical_count = max(maximum_categorical_count, len(set(x)))
        y = np.asarray(list(trace.y), dtype=float) if getattr(trace, "y", None) is not None else np.asarray([])
        name = str(getattr(trace, "name", "") or "")
        if trace_type == "bar":
            colors = getattr(getattr(trace, "marker", None), "color", None)
            key = tuple(x)
            bottom = None
            if getattr(figure.layout, "barmode", None) == "stack":
                bottom = stack_bottom.setdefault(key, np.zeros(len(y), dtype=float)).copy()
                stack_bottom[key] += y
            bars = target.bar(x, y, bottom=bottom, label=name, color=colors)
            text = list(trace.text) if getattr(trace, "text", None) is not None else []
            if text:
                for bar, label in zip(bars, text):
                    target.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height(),
                        _plain_plotly_text(label), ha="center", va="bottom", fontsize=7,
                    )
            continue
        if trace_type not in {"scatter", "scattergl"}:
            raise TypeError(f"unsupported static Stage-14 trace type: {trace_type!r}")
        mode = str(getattr(trace, "mode", "") or "")
        line = getattr(trace, "line", None)
        marker = getattr(trace, "marker", None)
        color = getattr(line, "color", None) or getattr(marker, "color", None)
        if isinstance(color, (list, tuple, np.ndarray)):
            color = None
        line_style = {"dash": "--", "dot": ":", "dashdot": "-."}.get(
            str(getattr(line, "dash", "") or ""), "-"
        )
        marker_symbol = marker_map.get(
            str(getattr(marker, "symbol", "circle") or "circle"), "o"
        )
        marker_size = float(getattr(marker, "size", 8) or 8)
        if "lines" in mode:
            target.plot(
                x, y, linestyle=line_style,
                linewidth=float(getattr(line, "width", 1.8) or 1.8),
                marker=marker_symbol if "markers" in mode else None,
                markersize=max(3.0, marker_size * 0.55), color=color, label=name,
            )
        elif "markers" in mode:
            marker_color = getattr(marker, "color", None)
            scatter_kwargs: dict[str, Any] = {
                "s": max(16.0, marker_size**2), "marker": marker_symbol,
                "label": name, "alpha": float(getattr(marker, "opacity", 1.0) or 1.0),
            }
            if isinstance(marker_color, (list, tuple, np.ndarray)):
                values = list(marker_color)
                if values and all(isinstance(value, (int, float, np.number)) for value in values):
                    scatter_kwargs.update({"c": values, "cmap": "viridis"})
                else:
                    scatter_kwargs["c"] = values
            elif marker_color is not None:
                scatter_kwargs["c"] = marker_color
            if str(getattr(marker, "symbol", "")) == "circle-open":
                scatter_kwargs.update({"facecolors": "none", "edgecolors": marker_color or "black"})
                scatter_kwargs.pop("c", None)
            target.scatter(x, y, **scatter_kwargs)
        text = list(trace.text) if getattr(trace, "text", None) is not None else []
        if "text" in mode and text:
            for x_value, y_value, label in zip(x, y, text):
                target.annotate(
                    _plain_plotly_text(label), (x_value, y_value),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7,
                )
    for shape in getattr(figure.layout, "shapes", ()) or ():
        if getattr(shape, "type", None) == "line" and shape.y0 == shape.y1:
            dash = {"dash": "--", "dot": ":", "dashdot": "-."}.get(
                str(getattr(shape.line, "dash", "") or ""), "-"
            )
            axis.axhline(
                float(shape.y0), color=getattr(shape.line, "color", None) or "#777777",
                linestyle=dash, linewidth=float(getattr(shape.line, "width", 1.5) or 1.5),
            )
    layout = figure.layout
    axis.set_title(_plain_plotly_text(getattr(getattr(layout, "title", None), "text", "")))
    axis.set_xlabel(_plain_plotly_text(getattr(getattr(layout.xaxis, "title", None), "text", "")))
    axis.set_ylabel(_plain_plotly_text(getattr(getattr(layout.yaxis, "title", None), "text", "")))
    if getattr(layout.xaxis, "type", None) == "log":
        axis.set_xscale("log")
    if getattr(layout.yaxis, "range", None):
        axis.set_ylim(tuple(layout.yaxis.range))
    if getattr(layout.yaxis, "type", None) == "log":
        axis.set_yscale("log")
    if secondary_x_axis is not None:
        secondary_x_axis.set_xlabel(
            _plain_plotly_text(getattr(getattr(layout.xaxis2, "title", None), "text", ""))
        )
        if getattr(layout.xaxis2, "type", None) == "log":
            secondary_x_axis.set_xscale("log")
    if secondary_y_axis is not None:
        secondary_y_axis.set_ylabel(
            _plain_plotly_text(getattr(getattr(layout.yaxis2, "title", None), "text", ""))
        )
        if getattr(layout.yaxis2, "range", None):
            secondary_y_axis.set_ylim(tuple(layout.yaxis2.range))
        if getattr(layout.yaxis2, "type", None) == "log":
            secondary_y_axis.set_yscale("log")
    axis.grid(True, alpha=0.22)
    if maximum_categorical_count >= 4:
        axis.tick_params(axis="x", labelrotation=32)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    handles, labels = axis.get_legend_handles_labels()
    if secondary_x_axis is not None:
        second_handles, second_labels = secondary_x_axis.get_legend_handles_labels()
        handles += second_handles
        labels += second_labels
    if secondary_y_axis is not None:
        second_handles, second_labels = secondary_y_axis.get_legend_handles_labels()
        handles += second_handles
        labels += second_labels
    if handles:
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            if label and label not in unique:
                unique[label] = handle
        legend_metadata = figure.layout.meta
        requested_columns = (
            legend_metadata.get("png_legend_columns", 4)
            if isinstance(legend_metadata, dict) else 4
        )
        legend_columns = min(max(1, int(requested_columns)), len(unique))
        legend_rows = (len(unique) + legend_columns - 1) // legend_columns
        reserved_bottom = min(0.16 + 0.055 * legend_rows, 0.38)
        fig.legend(
            unique.values(), unique.keys(), loc="upper center",
            bbox_to_anchor=(0.5, reserved_bottom - 0.105), borderaxespad=0.0,
            fontsize=7, ncol=legend_columns,
        )
        fig.tight_layout(rect=(0.0, reserved_bottom, 1.0, 1.0))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def html_table(
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    title: str | None = None,
) -> str:
    heading = "" if title is None else f"<h4>{escape(title)}</h4>"
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        heading
        + "<table style='border-collapse:collapse'>"
        + f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
        + "<style>th,td{border:1px solid #bbb;padding:4px 8px;text-align:left}</style>"
    )


__all__ = ["html_table", "save_figure", "save_figure_png", "write_json"]

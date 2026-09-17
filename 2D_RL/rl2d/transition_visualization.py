"""Visual checkpoint for the exact Bellman-to-RL transition contract."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np

from .transition_contract import ExactTransitionModel


def _extent(model: ExactTransitionModel) -> tuple[float, ...]:
    dz = float(model.z_grid[1] - model.z_grid[0]) / 2.0
    dh = float(model.h_grid[1] - model.h_grid[0]) / 2.0
    return (
        float(model.z_grid[0] - dz),
        float(model.z_grid[-1] + dz),
        float(model.h_grid[0] - dh),
        float(model.h_grid[-1] + dh),
    )


def _overlay_path(axis: plt.Axes, arrays: Mapping[str, np.ndarray]) -> None:
    path = np.asarray(arrays["optimal_glide_trajectory"])
    axis.plot(path[:, 0], path[:, 1], color="#00bcd4", linewidth=2.1, zorder=20)


def _heatmap(
    axis: plt.Axes,
    model: ExactTransitionModel,
    values: np.ndarray,
    title: str,
    label: str,
    cmap: Any,
    arrays: Mapping[str, np.ndarray],
    norm: Any = None,
) -> None:
    image = axis.imshow(
        np.ma.masked_invalid(values).T,
        origin="lower",
        aspect="auto",
        extent=_extent(model),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    _overlay_path(axis, arrays)
    axis.set_title(title)
    bar = axis.figure.colorbar(image, ax=axis, pad=0.012, fraction=0.045)
    bar.set_label(label)


def _representative_state(
    model: ExactTransitionModel, arrays: Mapping[str, np.ndarray]
) -> tuple[int, int]:
    path = np.asarray(arrays["optimal_glide_trajectory"])
    for fraction in (0.55, 0.4, 0.7, 0.25):
        point = path[int(fraction * (path.shape[0] - 1))]
        zi = int(np.argmin(np.abs(model.z_grid - point[0])))
        hi = int(np.argmin(np.abs(model.h_grid - point[1])))
        if model.reference_policy[zi, hi] >= 0:
            return zi, hi
    states = np.argwhere(model.reference_policy >= 0)
    return tuple(int(value) for value in states[len(states) // 2])


def _draw_outgoing_transitions(
    axis: plt.Axes,
    model: ExactTransitionModel,
    arrays: Mapping[str, np.ndarray],
) -> tuple[int, int, int]:
    zi, hi = _representative_state(model, arrays)
    state = model.state((zi, hi))
    feasible = np.flatnonzero(model.action_mask((zi, hi)))
    greedy = int(model.reference_policy[zi, hi])
    endpoint_groups: dict[tuple[float, float], list[int]] = {}
    for ai in feasible:
        transition = model.transition((zi, hi), int(ai))
        endpoint = np.asarray(transition.next_state)
        key = (float(endpoint[0]), float(endpoint[1]))
        endpoint_groups.setdefault(key, []).append(int(ai))
    endpoints_array = np.asarray(list(endpoint_groups))
    for endpoint in endpoints_array:
        axis.plot(
            [state[0], endpoint[0]], [state[1], endpoint[1]],
            color="#90a4ae", linewidth=0.7, alpha=0.35, zorder=2,
        )
    axis.scatter(
        endpoints_array[:, 0], endpoints_array[:, 1],
        color="#546e7a", s=22, alpha=0.85, zorder=5,
    )
    greedy_transition = model.transition((zi, hi), greedy)
    greedy_endpoint = np.asarray(greedy_transition.next_state)
    axis.plot(
        [state[0], greedy_endpoint[0]], [state[1], greedy_endpoint[1]],
        color="#d32f2f", linewidth=3.0, zorder=8,
    )
    axis.scatter(state[0], state[1], color="black", marker="o", s=55, zorder=10)
    axis.scatter(
        greedy_endpoint[0], greedy_endpoint[1], color="#d32f2f",
        marker="*", s=110, edgecolor="black", linewidth=0.5, zorder=10,
    )
    margin_z = max(90.0, float(np.ptp(endpoints_array[:, 0])) * 0.25)
    margin_h = max(12.0, float(np.ptp(endpoints_array[:, 1])) * 0.35)
    axis.set_xlim(state[0] - margin_z, float(np.max(endpoints_array[:, 0])) + margin_z)
    axis.set_ylim(float(np.min(endpoints_array[:, 1])) - margin_h, state[1] + margin_h)
    axis.set_title(
        "C. Outgoing actions at one path state\n"
        f"state=({state[0]:.1f}, {state[1]:.1f}) m, "
        f"{len(endpoint_groups)} spatial endpoints × "
        f"{feasible.size // len(endpoint_groups)} speeds = {feasible.size} actions"
    )
    greedy_speed = float(model.action_speed[greedy])
    greedy_gamma = float(np.rad2deg(model.action_gamma[greedy]))
    axis.text(
        0.98, 0.04,
        f"Greedy action {greedy}\nspeed={greedy_speed:.1f} m/s\nγ={greedy_gamma:.2f}°",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#d32f2f", "alpha": 0.92},
    )
    axis.legend(
        handles=[
            Line2D([0], [0], color="#90a4ae", lw=1.5, label="Feasible action"),
            Line2D([0], [0], color="#d32f2f", lw=3.0, label="Greedy action"),
        ],
        loc="lower left",
        fontsize=8,
    )
    return zi, hi, greedy


def create_transition_verification_figure(
    model: ExactTransitionModel,
    arrays: Mapping[str, np.ndarray],
    validation: dict[str, Any],
    output_path: Path,
    dpi: int = 350,
) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = validation["metrics"]
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 9.4), constrained_layout=True)

    feasible_counts = np.count_nonzero(model.valid, axis=2).astype(float)
    feasible_counts[feasible_counts == 0] = np.nan
    _heatmap(
        axes[0, 0], model, feasible_counts,
        "A. Exact feasible-action mask", "Feasible actions per state",
        "cividis", arrays,
    )
    _heatmap(
        axes[0, 1], model,
        np.where(np.isfinite(validation["reconstructed_value"]),
                 validation["reconstructed_value"], np.nan),
        "B. Value reconstructed through RL transition API", "Vcontract(z,h)",
        "viridis", arrays,
    )
    _draw_outgoing_transitions(axes[1, 0], model, arrays)

    agreement_map = np.full(model.state_shape, np.nan)
    active = model.reference_policy >= 0
    agreement_map[active] = np.where(validation["policy_match_map"][active], 1.0, 0.0)
    agreement_map[model.goal_mask] = 2.0
    agreement_cmap = ListedColormap(["#d32f2f", "#43a047", "#fdd835"])
    agreement_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], agreement_cmap.N)
    image = axes[1, 1].imshow(
        np.ma.masked_invalid(agreement_map).T,
        origin="lower", aspect="auto", extent=_extent(model),
        cmap=agreement_cmap, norm=agreement_norm, interpolation="nearest",
    )
    _overlay_path(axes[1, 1], arrays)
    axes[1, 1].set_title("D. Greedy-policy agreement with exact Bellman")
    bar = figure.colorbar(image, ax=axes[1, 1], pad=0.012, fraction=0.045, ticks=[0, 1, 2])
    bar.ax.set_yticklabels(["Mismatch", "Match", "Goal"])

    for axis in axes.flat:
        axis.set_xlabel("Downrange z [m]")
        axis.set_ylabel("Altitude h [m]")
        axis.grid(color="white", alpha=0.18, linewidth=0.5)

    figure.suptitle(
        "Bellman → RL Transition Contract Verification\n"
        f"states={metrics['state_count']:,}  |  actions={metrics['action_count']}  |  "
        f"state-actions={metrics['state_action_count']:,}  |  feasible={metrics['feasible_transition_count']:,}\n"
        f"ΔVmax={metrics['maximum_value_error']:.1e}  |  "
        f"policy agreement={100.0 * metrics['greedy_policy_agreement']:.3f}%  |  "
        f"all checks={'PASS' if validation['passed'] else 'FAIL'}",
        fontsize=12.2,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path

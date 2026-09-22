"""Stage-4 diagnostics: is tabular Q-learning converging toward the Bellman oracle?

Five plots, each answering one question about the run:

    1  episode cost vs episode            is the agent's own experience improving?
    2  moving-average episode cost        the same, with exploration noise removed
    3  greedy evaluation cost vs episode  is the *policy* improving, not just the
                                          behaviour under exploration?
    4  optimality gap vs episode          how far from the oracle, and does it stop?
    5  both trajectories in the terrain   where the two solutions differ in space

Plots 1-4 share an x axis and read top to bottom; 3 and 4 are the ones that decide
whether the run converged, because 1 and 2 include epsilon-greedy noise and can
look unconverged when the greedy policy is already exact.

Interactive ``.html`` and static ``.png`` are written for each.

    python P1b_RL_figures.py [spatial_resolution_m] [episodes] [x,y,z sensor]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from P1b_Exact_figures import (
    EXACT, GOAL, HEIGHT, INK, INK_FAINT, INK_SOFT, SENSOR, SWITCH, WIDTH,
    _layout_3d, _save, _scene_markers, _terrain_mesh, _terrain_rectangles,
)
from P1b_RL_approximation import (
    GlideSolveResult, QLearningConfig, candidate_start_states, default_sensor,
    solve_glide_mdp,
)
from P1b_condition import ComputationCondition, Scene, build_scene


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "P1b_rl_verification"

LEARNED = "#9C6511"
ORACLE = EXACT
MOVING_AVERAGE_WINDOW = 100


def _layout_2d(x_title: str, y_title: str, **overrides: Any) -> dict[str, Any]:
    layout: dict[str, Any] = {
        "width": WIDTH, "height": int(HEIGHT * 0.62),
        "margin": {"l": 76, "r": 28, "t": 18, "b": 58},
        "paper_bgcolor": "white", "plot_bgcolor": "white",
        "font": {"color": INK, "size": 13},
        "xaxis": {
            "title": x_title, "gridcolor": INK_FAINT, "zeroline": False,
            "linecolor": INK_SOFT, "ticks": "outside", "tickcolor": INK_SOFT,
        },
        "yaxis": {
            "title": y_title, "gridcolor": INK_FAINT, "zeroline": False,
            "linecolor": INK_SOFT, "ticks": "outside", "tickcolor": INK_SOFT,
        },
        "legend": {
            "orientation": "h", "yanchor": "bottom", "y": 1.01,
            "xanchor": "right", "x": 1.0,
            "bgcolor": "rgba(255,255,255,0.75)", "borderwidth": 0,
        },
        "hovermode": "x unified",
    }
    layout.update(overrides)
    return layout


def figure_episode_cost(result: GlideSolveResult, output: Path) -> str:
    """Raw per-episode cost.  Noisy by construction: these are epsilon-greedy runs."""
    history = result.history
    figure = go.Figure()
    figure.add_trace(go.Scattergl(
        x=history.episode, y=history.episode_cost, mode="markers",
        name=f"episode cost ({len(history.episode):,} episodes)",
        marker={"size": 2.5, "color": LEARNED, "opacity": 0.35},
    ))
    figure.add_hline(
        y=result.bellman_cost, line={"color": ORACLE, "width": 2, "dash": "dash"},
    )
    figure.add_trace(go.Scattergl(
        x=[None], y=[None], mode="lines", name="Bellman optimum",
        line={"color": ORACLE, "width": 2, "dash": "dash"},
    ))
    figure.update_layout(**_layout_2d("training episode", "episode cost"))
    return _save(figure, output, "1-RL_episode_cost")


def figure_moving_average(result: GlideSolveResult, output: Path) -> str:
    """The same series smoothed, so a trend is visible through the exploration noise."""
    history = result.history
    smoothed = history.moving_average_cost(MOVING_AVERAGE_WINDOW)
    figure = go.Figure()
    if len(smoothed):
        offset = MOVING_AVERAGE_WINDOW - 1
        figure.add_trace(go.Scattergl(
            x=history.episode[offset:], y=smoothed, mode="lines",
            name=f"moving average over {MOVING_AVERAGE_WINDOW} episodes",
            line={"color": LEARNED, "width": 2},
        ))
    figure.add_hline(
        y=result.bellman_cost, line={"color": ORACLE, "width": 2, "dash": "dash"},
    )
    figure.add_trace(go.Scattergl(
        x=[None], y=[None], mode="lines", name="Bellman optimum",
        line={"color": ORACLE, "width": 2, "dash": "dash"},
    ))
    figure.update_layout(**_layout_2d("training episode", "episode cost"))
    return _save(figure, output, "2-RL_moving_average_cost")


def figure_greedy_evaluation(result: GlideSolveResult, output: Path) -> str:
    """Cost of the greedy policy, measured periodically with exploration switched off.

    Episodes that never reached the goal have no finite cost; they are drawn on the
    axis as failures rather than dropped, so a policy that regresses into not
    arriving is visible instead of leaving a gap in the line.
    """
    history = result.history
    episodes = np.asarray(history.evaluation_episode, dtype=float)
    costs = np.asarray(history.evaluation_cost, dtype=float)
    finite = np.isfinite(costs)
    figure = go.Figure()
    figure.add_trace(go.Scattergl(
        x=episodes[finite], y=costs[finite], mode="lines+markers",
        name="greedy policy cost",
        line={"color": LEARNED, "width": 2}, marker={"size": 4},
    ))
    if np.any(~finite):
        figure.add_trace(go.Scattergl(
            x=episodes[~finite],
            y=np.full(int(np.count_nonzero(~finite)), result.bellman_cost),
            mode="markers", name="greedy policy did not reach the goal",
            marker={"size": 7, "color": "#B3402F", "symbol": "x"},
        ))
    figure.add_hline(
        y=result.bellman_cost, line={"color": ORACLE, "width": 2, "dash": "dash"},
    )
    figure.add_trace(go.Scattergl(
        x=[None], y=[None], mode="lines", name="Bellman optimum",
        line={"color": ORACLE, "width": 2, "dash": "dash"},
    ))
    figure.update_layout(**_layout_2d("training episode", "greedy policy cost"))
    return _save(figure, output, "3-RL_greedy_evaluation_cost")


def figure_optimality_gap(result: GlideSolveResult, output: Path) -> str:
    """Relative gap to the oracle.  Log axis, because it spans orders of magnitude.

    An exact match is a gap of zero, which a log axis cannot place, so those points
    sit on a floor below the smallest positive gap and are marked.  The line runs
    through every evaluation including those, because a line drawn only through the
    non-exact points would join two distant excursions straight across a stretch
    where the policy was exact, and read as a trend that did not happen.
    """
    history = result.history
    episodes = np.asarray(history.evaluation_episode, dtype=float)
    gaps = np.asarray(history.evaluation_relative_gap, dtype=float) * 100.0
    known = np.isfinite(gaps)
    positive = known & (gaps > 0.0)
    exact = known & (gaps <= 0.0)
    floor = float(np.min(gaps[positive])) / 10.0 if np.any(positive) else 1.0e-3
    plotted = np.where(positive, gaps, floor)

    figure = go.Figure()
    figure.add_trace(go.Scattergl(
        x=episodes[known], y=plotted[known], mode="lines+markers",
        name=f"relative gap to Bellman ({int(np.count_nonzero(known))} evaluations)",
        line={"color": LEARNED, "width": 1.5}, marker={"size": 3.5},
        hovertemplate="episode %{x:,.0f}<extra></extra>",
    ))
    if np.any(exact):
        figure.add_trace(go.Scattergl(
            x=episodes[exact], y=np.full(int(np.count_nonzero(exact)), floor),
            mode="markers",
            name=f"exact match, gap = 0 ({int(np.count_nonzero(exact))})",
            marker={"size": 7, "color": ORACLE, "symbol": "circle"},
        ))
    figure.update_layout(**_layout_2d(
        "training episode", "relative gap to Bellman (%)",
        yaxis={
            "title": "relative gap to Bellman (%)", "type": "log",
            "gridcolor": INK_FAINT, "zeroline": False, "linecolor": INK_SOFT,
            "ticks": "outside", "tickcolor": INK_SOFT,
            "dtick": 1, "tickformat": ".3~g", "showexponent": "none",
        },
    ))
    return _save(figure, output, "4-RL_optimality_gap")


def _positions(scene: Scene, state_ids: tuple[int, ...]) -> np.ndarray:
    if not len(state_ids):
        return np.empty((0, 3))
    return np.asarray(
        [scene.grid.position_map(scene.grid.decode(int(i))) for i in state_ids],
        dtype=float,
    )


def _full_path(scene: Scene, glide_ids: tuple[int, ...]) -> np.ndarray:
    """Powered leg from the start to the switching state, then the glide."""
    glide = _positions(scene, glide_ids)
    if not len(glide):
        return glide
    start = np.asarray(scene.config.start.as_array(), dtype=float)
    return np.vstack((start[None, :], glide))


def _solution_paths(scene: Scene, result: GlideSolveResult):
    """Bellman's solution, the learner's, its cost, and whether they coincide.

    When the run picked its own switching state, that is the solution to draw - the
    path from Bellman's switching state would show the learner starting where it
    was not told to start.
    """
    readout = result.switching
    if readout is not None and readout.trajectory:
        learned_ids, learned_cost = readout.trajectory, readout.objective
    else:
        learned_ids, learned_cost = result.q_trajectory, result.q_cost
    oracle = _full_path(scene, result.bellman_trajectory)
    learned = _full_path(scene, learned_ids)
    identical = (
        oracle.shape == learned.shape and bool(np.allclose(oracle, learned))
    )
    return oracle, learned, learned_cost, identical


def _oracle_cost(result: GlideSolveResult) -> float:
    """What the learner's number must be compared against, on the same footing."""
    if result.switching is not None:
        return float(result.extra["oracle_objective"])
    return result.bellman_cost


def _learned_label(cost: float | None, identical: bool) -> str:
    if identical:
        return "Q-learning optimum  identical to Bellman"
    if cost is None:
        return "Q-learning  did not reach the goal"
    return f"Q-learning optimum  J = {cost:.6f}"


def figure_trajectories(
    scene: Scene, result: GlideSolveResult, output: Path,
) -> str:
    """Both optimal solutions in the terrain, powered leg included.

    When the learner has converged the two paths lie on top of one another, which
    is the result rather than a drawing problem - so the learned path is dashed
    with open markers, and the legend says outright that they coincide.
    """
    oracle, learned, learned_cost, identical = _solution_paths(scene, result)
    figure = go.Figure()
    for mesh in _terrain_mesh(scene.terrain):
        figure.add_trace(mesh)
    if len(oracle) > 1:
        figure.add_trace(go.Scatter3d(
            x=oracle[:, 0], y=oracle[:, 1], z=oracle[:, 2],
            mode="lines+markers",
            name=f"Bellman optimum  J = {_oracle_cost(result):.6f}",
            line={"color": ORACLE, "width": 8}, marker={"size": 3},
        ))
    if len(learned) > 1:
        figure.add_trace(go.Scatter3d(
            x=learned[:, 0], y=learned[:, 1], z=learned[:, 2],
            mode="lines+markers", name=_learned_label(learned_cost, identical),
            line={"color": LEARNED, "width": 4, "dash": "dot"},
            marker={"size": 5, "symbol": "diamond-open"},
        ))
    if len(learned) > 1:
        figure.add_trace(go.Scatter3d(
            x=[learned[1, 0]], y=[learned[1, 1]], z=[learned[1, 2]],
            mode="markers", name="switching state",
            marker={"size": 7, "color": SWITCH, "symbol": "diamond"},
        ))
    for trace in _scene_markers(scene, result.sensor_map):
        figure.add_trace(trace)
    figure.update_layout(**_layout_3d(scene))
    return _save(figure, output, "5-RL_trajectories_vs_Bellman")


def figure_trajectories_plane(
    scene: Scene, result: GlideSolveResult, output: Path, plane: str,
) -> str:
    """The same comparison flattened, where overlap and altitude read more easily."""
    axis = 2 if plane == "side" else 1
    oracle, learned, learned_cost, identical = _solution_paths(scene, result)
    config = scene.config
    bounds = scene.grid.bounds
    figure = go.Figure()
    if len(oracle) > 1:
        figure.add_trace(go.Scatter(
            x=oracle[:, 0], y=oracle[:, axis], mode="lines+markers",
            name=f"Bellman optimum  J = {_oracle_cost(result):.6f}",
            line={"color": ORACLE, "width": 4}, marker={"size": 7},
        ))
    if len(learned) > 1:
        figure.add_trace(go.Scatter(
            x=learned[:, 0], y=learned[:, axis], mode="lines+markers",
            name=_learned_label(learned_cost, identical),
            line={"color": LEARNED, "width": 2, "dash": "dot"},
            marker={"size": 11, "symbol": "diamond-open"},
        ))
    figure.add_trace(go.Scatter(
        x=[config.start.x],
        y=[config.start.z if plane == "side" else config.start.y],
        mode="markers", name="start",
        marker={"size": 11, "color": "white", "line": {"color": INK, "width": 2}},
    ))
    figure.add_trace(go.Scatter(
        x=[config.goal.x],
        y=[config.goal.z if plane == "side" else config.goal.y],
        mode="markers", name="goal",
        marker={"size": 15, "color": GOAL, "symbol": "star",
                "line": {"color": INK, "width": 1}},
    ))
    figure.add_trace(go.Scatter(
        x=[result.sensor_map[0]],
        y=[result.sensor_map[2] if plane == "side" else result.sensor_map[1]],
        mode="markers", name="sensor d",
        marker={"size": 13, "color": SENSOR, "symbol": "triangle-down"},
    ))
    y_title = "altitude [map unit]" if plane == "side" else "y [map unit]"
    y_range = (
        [0.0, scene.grid.maximum_altitude_map] if plane == "side"
        else [bounds.y_min, bounds.y_max]
    )
    figure.update_layout(
        shapes=_terrain_rectangles(scene.terrain, plane),
        xaxis={"title": "x [map unit]", "range": [bounds.x_min, bounds.x_max]},
        yaxis={"title": y_title, "range": y_range,
               "scaleanchor": "x" if plane == "top" else None},
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.75)"},
        width=WIDTH, height=560, template="plotly_white",
        margin={"l": 60, "r": 20, "t": 20, "b": 50},
    )
    stem = (
        "5-1-RL_trajectories_side_view" if plane == "side"
        else "5-2-RL_trajectories_top_view"
    )
    return _save(figure, output, stem)


def run_figures(
    resolution_m: float = 100.0,
    episodes: int = 20000,
    sensor_map: tuple[float, float, float] | None = None,
    *,
    output: Path = OUTPUT,
) -> GlideSolveResult:
    scene = build_scene(ComputationCondition(spatial_resolution_m=resolution_m))
    sensor = sensor_map or default_sensor(scene)
    # The exploration schedule is QLearningConfig's default, which is the one the
    # seed sweep in its docstring selected; overriding it here would make the
    # convergence figures a picture of some other configuration.
    config = QLearningConfig(
        episodes=episodes, evaluation_interval=max(1, episodes // 200),
    )
    # Solve over every switching candidate, so the plotted Q-learning solution is a
    # full best response the learner produced on its own - switching state included
    # - rather than one handed its start by the oracle it is compared against.
    result = solve_glide_mdp(
        scene, sensor, config,
        start_state_ids=candidate_start_states(scene, sensor),
        read_switching=True,
    )
    print(result.summary())

    output.mkdir(parents=True, exist_ok=True)
    written = [
        figure_episode_cost(result, output),
        figure_moving_average(result, output),
        figure_greedy_evaluation(result, output),
        figure_optimality_gap(result, output),
        figure_trajectories(scene, result, output),
        figure_trajectories_plane(scene, result, output, "side"),
        figure_trajectories_plane(scene, result, output, "top"),
    ]
    print("  wrote: " + ", ".join(written))
    return result


def main(argv: list[str]) -> None:
    resolution = float(argv[1]) if len(argv) > 1 else 100.0
    episodes = int(argv[2]) if len(argv) > 2 else 20000
    sensor = (
        tuple(float(value) for value in argv[3].split(",")) if len(argv) > 3 else None
    )
    run_figures(resolution, episodes, sensor)


__all__ = [
    "figure_episode_cost", "figure_greedy_evaluation", "figure_moving_average",
    "figure_optimality_gap", "figure_trajectories", "figure_trajectories_plane",
    "run_figures",
]


if __name__ == "__main__":
    main(sys.argv)

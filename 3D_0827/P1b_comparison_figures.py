"""Exact local SSE and its Q-learning approximation, in one picture.

Both solvers run the same Defender climb over the same lattice; only the Attacker
best response inside it differs - an exact Bellman sweep on one side, a tabular
Q-learning table on the other.  The figure puts their two answers in the same
scene so the comparison is spatial rather than a pair of numbers:

    blue    exact        Defender position, its search path, the full Attacker
                         trajectory it induces
    red     Q-learning   the same three, approximated

Behind them, the LOS tangent surface carries the exact objective as a heat map:
every admissible switching state coloured by ``powered(s) + V(s)``, which is the
landscape both solvers are searching.  It is drawn at the *exact* Defender
position, since that is the reference; when the two solvers disagree about where
the Defender should stand, the red trajectory is therefore crossing a landscape
computed for the blue sensor, and its own landscape is a different one.

    python P1b_comparison_figures.py [resolution_m] [episodes] [terrain]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import plotly.graph_objects as go

from P1b_Exact_Local_SSE import exact_best_response, exact_local_sse
from P1b_Exact_figures import (
    GOAL, HEIGHT, INK, INK_FAINT, SENSOR, WIDTH, _contour_for, _domain_mask,
    _layout_3d, _save, _switching_objective_by_state, _terrain_mesh,
    _terrain_rectangles,
)
from P1b_RL_approximation import QLearningConfig, rl_local_sse
from P1b_condition import ComputationCondition, Scene, build_scene


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "P1b_comparison"

EXACT_BLUE = "#1C5770"
LEARNED_RED = "#B3402F"

# How good the approximation has to be before it counts as good enough.  The
# comparison is decision-theoretic, not a contest: an approximation that is within
# the threshold on both strategies is acceptable whatever its gap, and one outside
# it is not, however small the gap looks.  Without a threshold the question has no
# answer - the exact solver is optimal by construction, so it wins by definition.
RL_THRESHOLD = 0.9


def _positions(scene: Scene, state_ids) -> np.ndarray:
    ids = tuple(int(i) for i in state_ids)
    if not ids:
        return np.empty((0, 3))
    return np.asarray(
        [scene.grid.position_map(scene.grid.decode(i)) for i in ids], dtype=float,
    )


def _mirror(points: np.ndarray, mirror_y: bool) -> np.ndarray:
    """Fold y onto its positive half.

    On a terrain symmetric about y = 0 the reflection of a solution is itself a
    solution, with the same cost to the last digit, so folding is a change of
    drawing and not of answer.  It applies to every piece of drawn geometry -
    trajectories and Defender markers alike - because folding only some of them
    would put a mirrored path next to an unmirrored sensor and invent a geometry
    that neither solver produced.  Never used on a terrain that is not symmetric.
    """
    if not mirror_y or not len(points):
        return points
    folded = np.asarray(points, dtype=float).copy()
    folded[..., 1] = np.abs(folded[..., 1])
    return folded


def _full_path(scene: Scene, glide_ids, *, mirror_y: bool = False) -> np.ndarray:
    """Powered leg from the start to the switching state, then the glide."""
    glide = _positions(scene, glide_ids)
    if not len(glide):
        return glide
    start = np.asarray(scene.config.start.as_array(), dtype=float)
    return _mirror(np.vstack((start[None, :], glide)), mirror_y)


def _terrain_is_y_symmetric(scene: Scene) -> bool:
    return all(
        abs(float(box.center_y)) < 1.0e-12
        for box in scene.terrain.obstacle_boxes()
    )


def _tangent_surface(scene: Scene, sensor_map) -> np.ndarray:
    discretization = scene.config.discretization
    contour = _contour_for(scene, sensor_map)
    origin = contour.origin.as_array()
    fractions = np.linspace(0.0, 1.0, 180, endpoint=False)
    scales = np.linspace(
        discretization.switching_radial_min, discretization.switching_radial_max, 48,
    )
    directions = np.asarray(
        [contour.tangent_vector_at(float(v)) for v in fractions], dtype=float,
    )
    surface = origin[None, None, :] + scales[None, :, None] * directions[:, None, :]
    masked = surface.copy()
    masked[~_domain_mask(surface, scene)] = np.nan
    return masked


def _defender_path(scene: Scene, action_ids) -> np.ndarray:
    ids = tuple(int(i) for i in action_ids)
    if not ids:
        return np.empty((0, 3))
    return np.asarray(
        [scene.defender_grid.position(i) for i in ids], dtype=float,
    )


@dataclass(frozen=True)
class ComparisonMetrics:
    """Like-for-like comparison of the two solvers.

    Reporting ``learned.J_A - exact.J_A`` is wrong whenever the two solvers pick
    different Defender positions, and they usually do.  Each objective is then
    measured against a different opponent, so the difference mixes two effects and
    can even come out negative - measured at weights 0.25/0.75 it read -15.4%, as
    though the learner had beaten the optimum.  It had not: its Defender had
    settled somewhere far weaker, and a weaker Defender is exactly what makes the
    Attacker's problem cheaper.

    So the two questions are separated, each answered against a fixed opponent:

    attacker_relative_error
        the learner's trajectory against the exact one **at the learner's own
        Defender position**.  Cannot be negative; if it ever is, the two solvers
        are not solving the same problem.

    defender_relative_loss
        what the Defender gives up by standing where the learner put it, both
        sides scored with the exact Attacker.  Negative means worse for the
        Defender, which maximises J_D.

    detection_overestimate
        how much more detection the learner believed it would get at its own
        position than it actually gets there.  Positive is the expected direction:
        a suboptimal Attacker is more exposed, so J_D computed against it reads
        high, and an argmax over such estimates favours wherever that error is
        largest.
    """

    exact_sensor: tuple[float, float, float]
    learned_sensor: tuple[float, float, float] | None
    exact_objective: float
    exact_detection: float
    learned_objective: float | None
    learned_detection: float | None
    exact_objective_at_learned_sensor: float | None
    true_detection_at_learned_sensor: float | None
    threshold: float = RL_THRESHOLD

    @property
    def attacker_accuracy(self) -> float | None:
        """exact / learned, in (0, 1].  J_A is minimised, so the learner is never
        below the optimum and the ratio never exceeds one."""
        if self.learned_objective is None or not self.learned_objective:
            return None
        reference = self.exact_objective_at_learned_sensor
        return None if reference is None else reference / self.learned_objective

    @property
    def defender_accuracy(self) -> float | None:
        """learner's position / exact position, in (0, 1].  J_D is maximised, so
        the Defender can only lose by standing somewhere other than the optimum."""
        actual = self.true_detection_at_learned_sensor
        if actual is None or not self.exact_detection:
            return None
        return actual / self.exact_detection

    @property
    def meets_threshold(self) -> bool | None:
        """Both strategies, not just the cheaper one to get right.

        The Attacker's trajectory and the Defender's placement are separate
        answers, and the measurements so far move in opposite directions - the
        weighting that makes the trajectory easy to learn makes the placement hard
        to find.  Accepting on the better of the two would report success from
        whichever half happened to be easy.
        """
        attacker = self.attacker_accuracy
        defender = self.defender_accuracy
        if attacker is None or defender is None:
            return None
        return attacker >= self.threshold and defender >= self.threshold

    @property
    def same_defender_position(self) -> bool:
        if self.learned_sensor is None:
            return False
        return bool(
            np.allclose(
                np.asarray(self.exact_sensor, dtype=float),
                np.asarray(self.learned_sensor, dtype=float),
            )
        )

    @property
    def attacker_relative_error(self) -> float | None:
        reference = self.exact_objective_at_learned_sensor
        if self.learned_objective is None or not reference:
            return None
        return (self.learned_objective - reference) / reference

    @property
    def defender_relative_loss(self) -> float | None:
        actual = self.true_detection_at_learned_sensor
        if actual is None or not self.exact_detection:
            return None
        return (actual - self.exact_detection) / self.exact_detection

    @property
    def detection_overestimate(self) -> float | None:
        actual = self.true_detection_at_learned_sensor
        if actual is None or self.learned_detection is None or not actual:
            return None
        return (self.learned_detection - actual) / actual

    def as_dict(self) -> dict[str, Any]:
        return {
            "exact_sensor_map": list(self.exact_sensor),
            "learned_sensor_map": (
                None if self.learned_sensor is None else list(self.learned_sensor)
            ),
            "same_defender_position": self.same_defender_position,
            "exact_objective": self.exact_objective,
            "exact_detection_probability": self.exact_detection,
            "learned_objective": self.learned_objective,
            "learned_detection_probability": self.learned_detection,
            "exact_objective_at_learned_sensor": self.exact_objective_at_learned_sensor,
            "true_detection_at_learned_sensor": self.true_detection_at_learned_sensor,
            "attacker_relative_error": self.attacker_relative_error,
            "defender_relative_loss": self.defender_relative_loss,
            "detection_overestimate": self.detection_overestimate,
            "attacker_accuracy": self.attacker_accuracy,
            "defender_accuracy": self.defender_accuracy,
            "rl_threshold": self.threshold,
            "meets_threshold": self.meets_threshold,
        }

    def summary(self) -> str:
        def shown(value, form="{:.6f}"):
            return "n/a" if value is None else form.format(value)

        lines = [
            f"  attacker   J_A at the learner's own d:"
            f"   exact {shown(self.exact_objective_at_learned_sensor)}"
            f"   learned {shown(self.learned_objective)}"
            f"   error {shown(self.attacker_relative_error, '{:+.4%}')}",
            f"  defender   true J_D:"
            f"   exact d* {shown(self.exact_detection)}"
            f"   learner's d {shown(self.true_detection_at_learned_sensor)}"
            f"   loss {shown(self.defender_relative_loss, '{:+.4%}')}"
            + ("   (same position)" if self.same_defender_position else ""),
            f"  learner believed J_D {shown(self.learned_detection)} at its own d,"
            f"  overestimate {shown(self.detection_overestimate, '{:+.4%}')}",
            f"  accuracy   attacker {shown(self.attacker_accuracy, '{:.4f}')}"
            f"   defender {shown(self.defender_accuracy, '{:.4f}')}"
            f"   threshold {self.threshold:.2f}"
            f"   -> {'ACCEPT' if self.meets_threshold else 'REJECT'}",
        ]
        return "\n".join(lines)


def compare_solutions(
    scene: Scene, exact_sse, learned, *, threshold: float = RL_THRESHOLD,
) -> ComparisonMetrics:
    """Score both answers against a fixed opponent; one extra exact best response."""
    exact_sensor = tuple(float(v) for v in exact_sse.selected_sensor_map)
    learned_sensor = (
        tuple(float(v) for v in learned.selected_sensor_map)
        if learned.selected_sensor_map else None
    )
    reference_objective = None
    reference_detection = None
    if learned_sensor is not None:
        if np.allclose(np.asarray(exact_sensor), np.asarray(learned_sensor)):
            reference_objective = exact_sse.attacker_objective
            reference_detection = exact_sse.detection_probability
        else:
            at_learned = exact_best_response(scene, learned_sensor)
            if at_learned.feasible:
                reference_objective = at_learned.attacker_objective
                reference_detection = at_learned.detection_probability
    return ComparisonMetrics(
        exact_sensor=exact_sensor,
        learned_sensor=learned_sensor,
        exact_objective=float(exact_sse.attacker_objective),
        exact_detection=float(exact_sse.detection_probability),
        learned_objective=learned.attacker_objective,
        learned_detection=learned.detection_probability,
        exact_objective_at_learned_sensor=reference_objective,
        true_detection_at_learned_sensor=reference_detection,
        threshold=float(threshold),
    )


def figure_solver_comparison(
    scene: Scene, exact_sse, exact_response, learned, output: Path, *, stem: str,
    mirror_exact: bool = True,
) -> str:
    # Both solvers landed on the same side of a symmetric terrain, so one is drawn
    # on its mirror: co-optimal, same cost, and otherwise invisible underneath.
    mirror = bool(mirror_exact) and _terrain_is_y_symmetric(scene)
    figure = go.Figure()
    for mesh in _terrain_mesh(scene.terrain):
        figure.add_trace(mesh)

    exact_sensor = tuple(float(v) for v in exact_sse.selected_sensor_map)
    surface = _tangent_surface(scene, exact_sensor)
    figure.add_trace(go.Surface(
        x=surface[:, :, 0], y=surface[:, :, 1], z=surface[:, :, 2],
        colorscale=[[0, INK_FAINT], [1, INK_FAINT]], showscale=False, opacity=0.15,
        name="LOS tangent surface", showlegend=True, hoverinfo="name",
    ))

    objective_of = _switching_objective_by_state(scene, exact_sensor, exact_response)
    best: dict[tuple[float, float, float], float] = {}
    for state_id, objective in objective_of.items():
        key = tuple(
            float(v) for v in scene.grid.position_map(scene.grid.decode(state_id))
        )
        if key not in best or objective < best[key]:
            best[key] = objective
    points = np.asarray(list(best), dtype=float)
    objectives = np.asarray([best[tuple(p)] for p in points], dtype=float)
    figure.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
        name=f"switching-point J_A, exact ({len(points):,} positions)",
        marker={
            "size": 4.0, "color": objectives, "colorscale": "Viridis_r",
            # A long tail would flatten nine points in ten to one colour; the
            # colourbar says where it was cut rather than letting that look flat.
            "cmin": float(objectives.min()),
            "cmax": float(np.percentile(objectives, 90)),
            "opacity": 0.85, "line": {"width": 0},
            "colorbar": {
                "title": "J_A<br><sub>exact, clipped at p90</sub>",
                "len": 0.62, "x": 1.02,
            },
        },
        hovertemplate="x %{x:.2f}  y %{y:.2f}  z %{z:.2f}"
                      "<br>J_A %{marker.color:.6f}<extra></extra>",
    ))

    for label, colour, sensor, trajectory, visited, objective, detection, dash in (
        (
            "exact", EXACT_BLUE, exact_sensor, exact_response.glide_state_ids,
            exact_sse.visited_action_ids, exact_sse.attacker_objective,
            exact_sse.detection_probability, "solid",
        ),
        (
            "Q-learning", LEARNED_RED,
            tuple(float(v) for v in (learned.selected_sensor_map or ())),
            learned.trajectory, learned.visited_action_ids,
            learned.attacker_objective, learned.detection_probability, "dot",
        ),
    ):
        if not sensor:
            continue
        path = _full_path(scene, trajectory, mirror_y=mirror)
        if len(path) > 1:
            figure.add_trace(go.Scatter3d(
                x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines+markers",
                name=f"{label} Attacker trajectory   J_A = {objective:.6f}",
                line={"color": colour, "width": 7 if dash == "solid" else 5,
                      "dash": dash},
                marker={"size": 4 if dash == "solid" else 5,
                        "symbol": "circle" if dash == "solid" else "diamond-open"},
            ))
        search = _mirror(_defender_path(scene, visited), mirror)
        if len(search) > 1:
            figure.add_trace(go.Scatter3d(
                x=search[:, 0], y=search[:, 1], z=search[:, 2],
                mode="lines+markers", name=f"{label} Defender search path",
                line={"color": colour, "width": 3, "dash": "dash"}, opacity=0.55,
                marker={"size": 4, "symbol": "square"},
            ))
        drawn = _mirror(np.asarray(sensor, dtype=float), mirror)
        figure.add_trace(go.Scatter3d(
            x=[drawn[0]], y=[drawn[1]], z=[drawn[2]], mode="markers",
            name=f"{label} Defender d   J_D = {detection:.6f}",
            marker={"size": 11, "color": colour, "symbol": "x",
                    "line": {"color": "white", "width": 1}},
        ))

    start = np.asarray(scene.config.start.as_array(), dtype=float)
    goal = np.asarray(scene.config.goal.as_array(), dtype=float)
    figure.add_trace(go.Scatter3d(
        x=[start[0]], y=[start[1]], z=[start[2]], mode="markers", name="start",
        marker={"size": 8, "color": "white", "line": {"color": INK, "width": 2}},
    ))
    figure.add_trace(go.Scatter3d(
        x=[goal[0]], y=[goal[1]], z=[goal[2]], mode="markers", name="goal",
        marker={"size": 12, "color": GOAL, "symbol": "diamond",
                "line": {"color": INK, "width": 1}},
    ))

    metrics = compare_solutions(scene, exact_sse, learned)
    attacker = metrics.attacker_relative_error
    defender = metrics.defender_relative_loss
    subtitle = (
        f"exact J_A = {exact_sse.attacker_objective:.6f} at "
        f"d = {[float(v) for v in exact_sensor]}"
        + (
            "" if attacker is None
            else f"   attacker error {attacker:+.2%} at the learner's own d"
        )
        + (
            "" if defender is None
            else f"   defender loss {defender:+.2%}"
        )
    )
    if mirror:
        subtitle += "   |y| folded"
    layout = _layout_3d(scene)
    layout["title"] = {
        "text": f"{scene.condition.label}   terrain {scene.condition.terrain_category}"
                f"<br><sub>{subtitle}</sub>",
        "x": 0.5, "xanchor": "center", "font": {"size": 15, "color": INK},
    }
    margin = dict(layout.get("margin") or {})
    margin["t"] = max(int(margin.get("t", 0)), 76)
    layout["margin"] = margin
    figure.update_layout(**layout)
    return _save(figure, output, stem)


def figure_solver_comparison_plane(
    scene: Scene, exact_sse, exact_response, learned, output: Path, *,
    stem: str, plane: str, mirror_exact: bool = True,
) -> str:
    """The same comparison flattened.

    The 3-D view puts both trajectories in one perspective, where the nearer one
    hides the further one; these two projections are what the numbers can actually
    be read off.  Side shows altitude against down-range, top shows the lateral
    detour, and between them nothing about either path is hidden.
    """
    axis = 2 if plane == "side" else 1
    mirror = bool(mirror_exact) and _terrain_is_y_symmetric(scene)
    config = scene.config
    bounds = scene.grid.bounds
    figure = go.Figure()

    exact_sensor = tuple(float(v) for v in exact_sse.selected_sensor_map)
    learned_sensor = tuple(float(v) for v in (learned.selected_sensor_map or ()))
    for label, colour, sensor, trajectory, objective, detection, solid in (
        (
            "exact", EXACT_BLUE, exact_sensor, exact_response.glide_state_ids,
            exact_sse.attacker_objective, exact_sse.detection_probability, True,
        ),
        (
            "Q-learning", LEARNED_RED, learned_sensor, learned.trajectory,
            learned.attacker_objective, learned.detection_probability, False,
        ),
    ):
        if not sensor:
            continue
        path = _full_path(scene, trajectory, mirror_y=mirror)
        if len(path) > 1:
            figure.add_trace(go.Scatter(
                x=path[:, 0], y=path[:, axis], mode="lines+markers",
                name=f"{label} Attacker   J_A = {objective:.6f}",
                line={"color": colour, "width": 4 if solid else 2,
                      "dash": "solid" if solid else "dot"},
                marker={"size": 7 if solid else 11,
                        "symbol": "circle" if solid else "diamond-open"},
            ))
        drawn = _mirror(np.asarray(sensor, dtype=float), mirror)
        figure.add_trace(go.Scatter(
            x=[drawn[0]], y=[drawn[axis]], mode="markers",
            name=f"{label} Defender d   J_D = {detection:.6f}",
            marker={"size": 14, "color": colour, "symbol": "x-thin",
                    "line": {"color": colour, "width": 3}},
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
    y_title = "altitude [map unit]" if plane == "side" else "y [map unit]"
    y_range = (
        [0.0, scene.grid.maximum_altitude_map] if plane == "side"
        else [bounds.y_min, bounds.y_max]
    )
    figure.update_layout(
        shapes=_terrain_rectangles(scene.terrain, plane),
        title={
            "text": f"{scene.condition.label}   {scene.condition.terrain_category}"
                    + ("   |y| folded" if mirror else ""),
            "x": 0.5, "xanchor": "center", "font": {"size": 14, "color": INK},
        },
        xaxis={"title": "x [map unit]", "range": [bounds.x_min, bounds.x_max]},
        yaxis={"title": y_title, "range": y_range,
               "scaleanchor": "x" if plane == "top" else None},
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.8)"},
        width=WIDTH, height=560, template="plotly_white",
        margin={"l": 60, "r": 20, "t": 46, "b": 50},
    )
    return _save(figure, output, stem)


def run_comparison(
    resolution_m: float = 50.0,
    episodes: int = 20000,
    terrain_category: str = "centered_cube_half_height",
    *,
    heading_spacing_deg: float = 5.0,
    r_neighbor: int = 1,
    hazard_weight: float | None = None,
    time_weight: float | None = None,
    warm_start_episodes: int | None = None,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    condition = ComputationCondition(
        spatial_resolution_m=resolution_m,
        heading_spacing_deg=heading_spacing_deg,
        r_neighbor=r_neighbor, terrain_category=terrain_category,
        hazard_weight=hazard_weight, time_weight=time_weight,
    )
    scene = build_scene(condition)
    print(f"condition {condition.label}   terrain {terrain_category}", flush=True)

    started = perf_counter()
    exact_sse = exact_local_sse(scene)
    exact_seconds = perf_counter() - started
    if not exact_sse.feasible:
        raise RuntimeError("the exact local SSE search found nothing feasible")
    exact_sensor = tuple(float(v) for v in exact_sse.selected_sensor_map)
    exact_response = exact_best_response(scene, exact_sensor, keep_solution=True)
    print(f"  exact       d* {list(exact_sensor)}   J_A {exact_sse.attacker_objective:.6f}"
          f"   J_D {exact_sse.detection_probability:.6f}"
          f"   {exact_sse.unique_evaluations} evaluations   {exact_seconds:.1f}s",
          flush=True)

    config = QLearningConfig(episodes=episodes)
    started = perf_counter()
    learned = rl_local_sse(
        scene, config, warm_start_episodes=warm_start_episodes, progress=True,
    )
    learned_seconds = perf_counter() - started
    if not learned.feasible:
        raise RuntimeError(
            f"the learned Defender search returned nothing: {learned.termination_status}"
        )
    print(f"  Q-learning  d* {learned.selected_sensor_map}"
          f"   J_A {learned.attacker_objective:.6f}"
          f"   J_D {learned.detection_probability:.6f}"
          f"   {learned.unique_evaluations} evaluations   {learned_seconds:.1f}s"
          f"   ({learned.termination_status}, certified={learned.local_sse_verified})",
          flush=True)
    metrics = compare_solutions(scene, exact_sse, learned)
    print(metrics.summary(), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    config_start = scene.config.start.as_array()
    config_goal = scene.config.goal.as_array()
    # The weight tag keeps runs at different objective weightings from overwriting
    # one another; a run at the base weights keeps the original unsuffixed name.
    suffix = "" if condition.weight_tag is None else f"_{condition.weight_tag}"
    # Heading spacing and Defender radius go in the name only when they are not the
    # defaults, so the runs already on disk keep the names they were written under.
    # Without the radius a wider search would overwrite the narrow one it is meant
    # to be compared against.
    if heading_spacing_deg != 5.0:
        suffix = f"_psi{heading_spacing_deg:g}{suffix}"
    if r_neighbor != 1:
        suffix = f"_r{r_neighbor}{suffix}"
    stem = f"1_exact_vs_qlearning_{resolution_m:g}m_{terrain_category}{suffix}"
    written = [
        figure_solver_comparison(
            scene, exact_sse, exact_response, learned, output, stem=stem,
        ),
        figure_solver_comparison_plane(
            scene, exact_sse, exact_response, learned, output,
            stem=f"1-1_exact_vs_qlearning_{resolution_m:g}m_{terrain_category}"
                 f"{suffix}_side_view",
            plane="side",
        ),
        figure_solver_comparison_plane(
            scene, exact_sse, exact_response, learned, output,
            stem=f"1-2_exact_vs_qlearning_{resolution_m:g}m_{terrain_category}"
                 f"{suffix}_top_view",
            plane="top",
        ),
    ]
    # The learned side costs twenty minutes, so the answer is written down.  A
    # redraw then needs only the exact side, which is seconds.
    record = output / f"{stem}.json"
    record.write_text(json.dumps({
        "schema": "p1b-exact-vs-qlearning-v2",
        "comparison": metrics.as_dict(),
        "condition": condition.as_dict(),
        "start_map": [float(v) for v in config_start],
        "goal_map": [float(v) for v in config_goal],
        "exact": {
            "sensor_map": list(exact_sensor),
            "selected_action_id": exact_sse.selected_action_id,
            "attacker_objective": exact_sse.attacker_objective,
            "detection_probability": exact_sse.detection_probability,
            # The Defender search's own record: what it saw at every position it
            # evaluated, and how it terminated.  Without these a finished run
            # cannot be asked why the climb stopped where it did, and the only
            # way to find out is to pay for the whole search again.
            "iterations": exact_sse.iterations,
            "termination_status": exact_sse.termination_status,
            "cached_reuses": exact_sse.cached_reuses,
            "evaluated": list(exact_sse.evaluated),
            "timing": dict(exact_sse.timing),
            "sizes": dict(exact_sse.sizes),
            "glide_state_ids": [int(i) for i in exact_response.glide_state_ids],
            "glide_positions_map": [
                [float(v) for v in scene.grid.position_map(scene.grid.decode(int(i)))]
                for i in exact_response.glide_state_ids
            ],
            "drawn_mirrored_to_positive_y": _terrain_is_y_symmetric(scene),
            "visited_action_ids": [int(i) for i in exact_sse.visited_action_ids],
            "unique_evaluations": exact_sse.unique_evaluations,
            "seconds": exact_seconds,
        },
        "learned": {
            "sensor_map": learned.selected_sensor_map,
            "attacker_objective": learned.attacker_objective,
            "detection_probability": learned.detection_probability,
            "trajectory": [int(i) for i in learned.trajectory],
            "trajectory_positions_map": [
                [float(v) for v in scene.grid.position_map(scene.grid.decode(int(i)))]
                for i in learned.trajectory
            ],
            "switching_state_id": learned.switching_state_id,
            "visited_action_ids": [int(i) for i in learned.visited_action_ids],
            "unique_evaluations": learned.unique_evaluations,
            "selected_action_id": learned.selected_action_id,
            "iterations": learned.iterations,
            "termination_status": learned.termination_status,
            "local_sse_verified": learned.local_sse_verified,
            "sizes": dict(learned.sizes),
            "timing": learned.timing,
            "training": learned.training,
            "evaluated": list(learned.evaluated),
            "seconds": learned_seconds,
        },
    }, indent=2), encoding="utf-8")
    print("  wrote: " + ", ".join(written) + f", {record.name}", flush=True)
    return {
        "scene": scene, "exact": exact_sse, "exact_response": exact_response,
        "learned": learned, "exact_seconds": exact_seconds,
        "learned_seconds": learned_seconds, "figure": written,
    }


@dataclass(frozen=True)
class _RecordedExact:
    """The exact side of a finished run, restored from its record.

    The Defender search is not repeated on a redraw: at 10 m with 2.5 degree
    headings it is twenty-three best responses and over an hour, and every number
    it produces is already in the record.  Only the one best response the figure
    actually needs - the solved value function behind the switching-point heat map
    at the chosen Defender position - is recomputed.
    """

    selected_sensor_map: list[float]
    attacker_objective: float
    detection_probability: float
    visited_action_ids: tuple[int, ...]
    unique_evaluations: int
    selected_action_id: int | None = None


@dataclass(frozen=True)
class _RecordedRun:
    """The learned side of a finished run, restored from its record.

    Only the fields the figures and the metrics read.  It exists so a record can
    be redrawn without paying for the learned side again - twenty minutes at 50 m
    - which is the whole reason the record is written.
    """

    selected_sensor_map: list[float] | None
    attacker_objective: float | None
    detection_probability: float | None
    trajectory: tuple[int, ...]
    visited_action_ids: tuple[int, ...]
    switching_state_id: int | None
    unique_evaluations: int
    termination_status: str
    local_sse_verified: bool


def redraw_from_record(record_path: Path) -> dict[str, Any]:
    """Redraw a finished comparison from its record, re-solving only the exact side.

    The exact side is cheap and deterministic, so it is recomputed rather than
    stored in full; the learned side is restored verbatim from the record, which
    is what makes this seconds instead of half an hour.  The record is rewritten
    with the corrected comparison block.
    """
    record_path = Path(record_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    stored = dict(record["condition"])
    condition = ComputationCondition(
        spatial_resolution_m=stored["spatial_resolution_m"],
        heading_spacing_deg=stored["heading_spacing_deg"],
        r_neighbor=stored["r_neighbor"],
        terrain_category=stored["terrain_category"],
        defender_goal_margin_map=stored["defender_goal_margin_map"],
        hazard_weight=stored.get("hazard_weight"),
        time_weight=stored.get("time_weight"),
    )
    scene = build_scene(condition)
    print(f"redraw {record_path.name}   {condition.label}"
          f"   terrain {condition.terrain_category}", flush=True)

    exact_block = record["exact"]
    exact_sensor = tuple(float(v) for v in exact_block["sensor_map"])
    exact_sse = _RecordedExact(
        selected_sensor_map=list(exact_sensor),
        attacker_objective=exact_block["attacker_objective"],
        detection_probability=exact_block["detection_probability"],
        visited_action_ids=tuple(
            int(i) for i in exact_block["visited_action_ids"]
        ),
        unique_evaluations=exact_block["unique_evaluations"],
    )
    exact_response = exact_best_response(scene, exact_sensor, keep_solution=True)
    if abs(exact_response.attacker_objective - exact_block["attacker_objective"]) > 1e-9:
        raise RuntimeError(
            "the exact solver no longer reproduces the recorded objective at the "
            "recorded Defender position; the record and the current code describe "
            "different problems"
        )

    learned_block = record["learned"]
    learned = _RecordedRun(
        selected_sensor_map=learned_block["sensor_map"],
        attacker_objective=learned_block["attacker_objective"],
        detection_probability=learned_block["detection_probability"],
        trajectory=tuple(int(i) for i in learned_block["trajectory"]),
        visited_action_ids=tuple(int(i) for i in learned_block["visited_action_ids"]),
        switching_state_id=learned_block.get("switching_state_id"),
        unique_evaluations=learned_block["unique_evaluations"],
        termination_status=learned_block["termination_status"],
        local_sse_verified=learned_block["local_sse_verified"],
    )

    metrics = compare_solutions(scene, exact_sse, learned)
    print(metrics.summary(), flush=True)

    output = record_path.parent
    stem = record_path.stem
    written = [
        figure_solver_comparison(
            scene, exact_sse, exact_response, learned, output, stem=stem,
        ),
        figure_solver_comparison_plane(
            scene, exact_sse, exact_response, learned, output,
            stem=stem.replace("1_", "1-1_", 1) + "_side_view", plane="side",
        ),
        figure_solver_comparison_plane(
            scene, exact_sse, exact_response, learned, output,
            stem=stem.replace("1_", "1-2_", 1) + "_top_view", plane="top",
        ),
    ]
    record["schema"] = "p1b-exact-vs-qlearning-v2"
    record["comparison"] = metrics.as_dict()
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("  wrote: " + ", ".join(written) + f", {record_path.name}", flush=True)
    return record


def main(argv: list[str]) -> None:
    if len(argv) > 1 and argv[1] == "redraw":
        for name in argv[2:] or sorted(
            str(path) for path in OUTPUT.glob("1_*.json")
        ):
            redraw_from_record(Path(name))
        return
    resolution = float(argv[1]) if len(argv) > 1 else 50.0
    episodes = int(argv[2]) if len(argv) > 2 else 20000
    terrain = argv[3] if len(argv) > 3 else "centered_cube_half_height"
    # An empty weights argument means "keep the base weights", so that a heading
    # can be given positionally without also having to name a weighting.
    raw_weights = argv[4].strip() if len(argv) > 4 else ""
    weights = (
        tuple(float(v) for v in raw_weights.split(","))
        if raw_weights else (None, None)
    )
    heading = float(argv[5]) if len(argv) > 5 else 5.0
    run_comparison(
        resolution, episodes, terrain, heading_spacing_deg=heading,
        hazard_weight=weights[0], time_weight=weights[1],
    )


__all__ = [
    "RL_THRESHOLD",
    "ComparisonMetrics", "compare_solutions", "figure_solver_comparison",
    "figure_solver_comparison_plane", "redraw_from_record", "run_comparison",
]


if __name__ == "__main__":
    main(sys.argv)

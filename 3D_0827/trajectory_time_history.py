"""Auditable time histories for the selected finite Stackelberg trajectory.

The optimizer stores scalar mission time and cumulative hazard.  This module
replays that same selected trajectory with the same segment-wise trapezoidal
quadrature so the position and detection histories can be inspected without
changing the optimization model.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from attacker_best_response import AttackerBestResponseRun
from detection_hazard import HazardField
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE, GliderParameters, PhysicalScale


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
ACOUSTIC_ASSUMPTION = (
    "acoustic hazard = 0 under the current LOS-tangent switching / "
    "post-switch acoustic-neutralization model"
)


def _readonly(values: object, *, ndim: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float).copy()
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-D array")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TrajectoryTimeHistory:
    """Time-aligned kinematic and cumulative detection diagnostics."""

    time_s: FloatArray
    position_map: FloatArray
    position_m: FloatArray
    phase: tuple[str, ...]
    visible: BoolArray
    radial_velocity_mps: FloatArray
    radar_cross_section: FloatArray
    acoustic_rate_per_s: FloatArray
    radar_rate_per_s: FloatArray
    doppler_rate_per_s: FloatArray
    cumulative_acoustic_hazard: FloatArray
    cumulative_radar_hazard: FloatArray
    cumulative_doppler_hazard: FloatArray
    cumulative_total_hazard: FloatArray
    acoustic_pod: FloatArray
    radar_equivalent_pod: FloatArray
    doppler_equivalent_pod: FloatArray
    total_pod: FloatArray
    powered_end_time_s: float
    virtual_end_time_s: float
    stored_mission_time_s: float
    stored_cumulative_hazard: float
    stored_detection_probability: float
    acoustic_assumption: str = ACOUSTIC_ASSUMPTION

    def __post_init__(self) -> None:
        time = _readonly(self.time_s, ndim=1, name="time_s")
        position_map = _readonly(self.position_map, ndim=2, name="position_map")
        position_m = _readonly(self.position_m, ndim=2, name="position_m")
        count = len(time)
        if count < 2 or position_map.shape != (count, 3) or position_m.shape != (count, 3):
            raise ValueError("time history positions must have shape (n, 3), n >= 2")
        if len(self.phase) != count:
            raise ValueError("phase must label every time-history sample")
        if np.any(np.diff(time) < -1.0e-12):
            raise ValueError("time_s must be nondecreasing")
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "position_map", position_map)
        object.__setattr__(self, "position_m", position_m)

        visible = np.asarray(self.visible, dtype=bool).copy()
        if visible.shape != (count,):
            raise ValueError("visible must have one value per sample")
        visible.setflags(write=False)
        object.__setattr__(self, "visible", visible)

        vector_names = (
            "radial_velocity_mps", "radar_cross_section",
            "acoustic_rate_per_s", "radar_rate_per_s", "doppler_rate_per_s",
            "cumulative_acoustic_hazard", "cumulative_radar_hazard",
            "cumulative_doppler_hazard", "cumulative_total_hazard",
            "acoustic_pod", "radar_equivalent_pod", "doppler_equivalent_pod",
            "total_pod",
        )
        for name in vector_names:
            vector = _readonly(getattr(self, name), ndim=1, name=name)
            if vector.shape != (count,):
                raise ValueError(f"{name} must have one value per sample")
            object.__setattr__(self, name, vector)

    @property
    def mission_time_error_s(self) -> float:
        return abs(float(self.time_s[-1]) - self.stored_mission_time_s)

    @property
    def cumulative_hazard_error(self) -> float:
        return abs(float(self.cumulative_total_hazard[-1]) - self.stored_cumulative_hazard)

    @property
    def detection_probability_error(self) -> float:
        return abs(float(self.total_pod[-1]) - self.stored_detection_probability)

    def summary(self) -> dict[str, object]:
        return {
            "sample_count": len(self.time_s),
            "mission_time_s": float(self.time_s[-1]),
            "stored_mission_time_s": self.stored_mission_time_s,
            "mission_time_error_s": self.mission_time_error_s,
            "cumulative_acoustic_hazard": float(self.cumulative_acoustic_hazard[-1]),
            "cumulative_rcs_radar_hazard": float(self.cumulative_radar_hazard[-1]),
            "cumulative_radial_velocity_doppler_hazard": float(
                self.cumulative_doppler_hazard[-1]
            ),
            "cumulative_total_hazard": float(self.cumulative_total_hazard[-1]),
            "stored_cumulative_hazard": self.stored_cumulative_hazard,
            "cumulative_hazard_error": self.cumulative_hazard_error,
            "acoustic_pod": float(self.acoustic_pod[-1]),
            "rcs_radar_equivalent_pod": float(self.radar_equivalent_pod[-1]),
            "radial_velocity_doppler_equivalent_pod": float(
                self.doppler_equivalent_pod[-1]
            ),
            "total_detection_probability": float(self.total_pod[-1]),
            "stored_detection_probability": self.stored_detection_probability,
            "detection_probability_error": self.detection_probability_error,
            "powered_end_time_s": self.powered_end_time_s,
            "virtual_end_time_s": self.virtual_end_time_s,
            "acoustic_assumption": self.acoustic_assumption,
            "component_pod_note": (
                "Component PoDs are 1-exp(-H_i) diagnostics and do not add to total PoD; "
                "total PoD is 1-exp(-sum_i H_i)."
            ),
        }


def _segment_cumulative(rate: FloatArray, duration_s: float) -> FloatArray:
    """Cumulative trapezoidal integral evaluated at uniform sample nodes."""
    if len(rate) == 1:
        return np.zeros(1, dtype=float)
    spacing = float(duration_s) / (len(rate) - 1)
    increments = 0.5 * (rate[:-1] + rate[1:]) * spacing
    return np.concatenate(([0.0], np.cumsum(increments)))


def build_trajectory_time_history(
    run: AttackerBestResponseRun,
    hazard_field: HazardField,
    *,
    quadrature_resolution: int = 8,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    validation_tolerance: float = 1.0e-10,
) -> TrajectoryTimeHistory:
    """Replay the selected response and decompose its cumulative hazard."""
    if not isinstance(run, AttackerBestResponseRun):
        raise TypeError("run must be an AttackerBestResponseRun")
    if not isinstance(hazard_field, HazardField):
        raise TypeError("hazard_field must implement HazardField")
    if not isinstance(quadrature_resolution, Integral) or isinstance(quadrature_resolution, bool):
        raise TypeError("quadrature_resolution must be an integer")
    resolution = int(quadrature_resolution)
    if resolution < 2:
        raise ValueError("quadrature_resolution must be at least two")
    selected = run.selected_result
    if selected is None or selected.selected_option is None:
        raise ValueError("Attacker run has no feasible selected trajectory")
    if selected.mission_time_s is None or selected.cumulative_hazard is None:
        raise ValueError("selected trajectory lacks stored mission metrics")
    if selected.detection_probability is None:
        raise ValueError("selected trajectory lacks stored detection probability")

    option = selected.selected_option
    switch = selected.evaluation.switching_state.position_map
    segments: list[tuple[str, FloatArray, FloatArray, float, bool]] = [
        (
            "powered", run.mission_points.start.as_array(), switch,
            option.powered_phase.duration_s, False,
        ),
        (
            "virtual", switch, option.connection.target_position_map,
            option.connection.duration_s, True,
        ),
    ]
    segments.extend(
        (
            "glide",
            run.graph.grid.position_map(edge.source_state),
            run.graph.grid.position_map(edge.target_state),
            edge.duration_s,
            True,
        )
        for edge in option.glide_edges
    )

    times: list[float] = []
    positions: list[FloatArray] = []
    phases: list[str] = []
    visible_values: list[bool] = []
    radial_values: list[float] = []
    rcs_values: list[float] = []
    acoustic_rates: list[float] = []
    radar_rates: list[float] = []
    doppler_rates: list[float] = []
    cumulative_acoustic: list[float] = []
    cumulative_radar: list[float] = []
    cumulative_doppler: list[float] = []

    elapsed = 0.0
    acoustic_offset = radar_offset = doppler_offset = 0.0
    powered_end = option.powered_phase.duration_s
    virtual_end = powered_end + option.virtual_phase.duration_s

    for phase, start, end, duration, detection_enabled in segments:
        duration = float(duration)
        if duration <= 1.0e-15:
            continue
        fractions = np.linspace(0.0, 1.0, resolution)
        segment_times = elapsed + fractions * duration
        segment_positions = start[None, :] + fractions[:, None] * (end - start)[None, :]
        velocity_mps = physical_scale.position_m(end - start) / duration
        evaluations = tuple(
            hazard_field.evaluate_rate(position, velocity_mps, float(sample_time))
            for position, sample_time in zip(segment_positions, segment_times)
        )
        segment_acoustic_rate = np.zeros(resolution, dtype=float)
        if detection_enabled:
            segment_radar_rate = np.asarray(
                [item.radar_rate_per_s for item in evaluations], dtype=float,
            )
            segment_doppler_rate = np.asarray(
                [item.doppler_rate_per_s for item in evaluations], dtype=float,
            )
        else:
            # The optimization deliberately excludes all pre-switch powered
            # detection hazard; retain kinematic diagnostics but not rates.
            segment_radar_rate = np.zeros(resolution, dtype=float)
            segment_doppler_rate = np.zeros(resolution, dtype=float)

        local_acoustic = _segment_cumulative(segment_acoustic_rate, duration)
        local_radar = _segment_cumulative(segment_radar_rate, duration)
        local_doppler = _segment_cumulative(segment_doppler_rate, duration)
        times.extend(segment_times.tolist())
        positions.extend(segment_positions)
        phases.extend([phase] * resolution)
        visible_values.extend(item.visible for item in evaluations)
        radial_values.extend(item.radial_velocity_mps for item in evaluations)
        rcs_values.extend(item.radar_cross_section for item in evaluations)
        acoustic_rates.extend(segment_acoustic_rate.tolist())
        radar_rates.extend(segment_radar_rate.tolist())
        doppler_rates.extend(segment_doppler_rate.tolist())
        cumulative_acoustic.extend((acoustic_offset + local_acoustic).tolist())
        cumulative_radar.extend((radar_offset + local_radar).tolist())
        cumulative_doppler.extend((doppler_offset + local_doppler).tolist())
        acoustic_offset += float(local_acoustic[-1])
        radar_offset += float(local_radar[-1])
        doppler_offset += float(local_doppler[-1])
        elapsed += duration

    cumulative_acoustic_array = np.asarray(cumulative_acoustic, dtype=float)
    cumulative_radar_array = np.asarray(cumulative_radar, dtype=float)
    cumulative_doppler_array = np.asarray(cumulative_doppler, dtype=float)
    cumulative_total = (
        cumulative_acoustic_array + cumulative_radar_array + cumulative_doppler_array
    )
    history = TrajectoryTimeHistory(
        time_s=np.asarray(times),
        position_map=np.asarray(positions),
        position_m=np.asarray(positions) * physical_scale.meters_per_map_unit,
        phase=tuple(phases),
        visible=np.asarray(visible_values),
        radial_velocity_mps=np.asarray(radial_values),
        radar_cross_section=np.asarray(rcs_values),
        acoustic_rate_per_s=np.asarray(acoustic_rates),
        radar_rate_per_s=np.asarray(radar_rates),
        doppler_rate_per_s=np.asarray(doppler_rates),
        cumulative_acoustic_hazard=cumulative_acoustic_array,
        cumulative_radar_hazard=cumulative_radar_array,
        cumulative_doppler_hazard=cumulative_doppler_array,
        cumulative_total_hazard=cumulative_total,
        acoustic_pod=-np.expm1(-cumulative_acoustic_array),
        radar_equivalent_pod=-np.expm1(-cumulative_radar_array),
        doppler_equivalent_pod=-np.expm1(-cumulative_doppler_array),
        total_pod=-np.expm1(-cumulative_total),
        powered_end_time_s=float(powered_end),
        virtual_end_time_s=float(virtual_end),
        stored_mission_time_s=float(selected.mission_time_s),
        stored_cumulative_hazard=float(selected.cumulative_hazard),
        stored_detection_probability=float(selected.detection_probability),
    )
    if history.mission_time_error_s > validation_tolerance:
        raise RuntimeError("time-history replay does not match stored mission time")
    if history.cumulative_hazard_error > validation_tolerance:
        raise RuntimeError("time-history replay does not match stored cumulative hazard")
    if history.detection_probability_error > validation_tolerance:
        raise RuntimeError("time-history replay does not match stored detection probability")
    return history


def plot_trajectory_time_history(history: TrajectoryTimeHistory) -> go.Figure:
    """Create position, total-PoD, component-PoD, and raw-signal panels."""
    if not isinstance(history, TrajectoryTimeHistory):
        raise TypeError("history must be a TrajectoryTimeHistory")
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Position history", "Total cumulative hazard converted to PoD",
            "Hazard-component equivalent PoD", "Raw radial velocity and RCS diagnostics",
        ),
        specs=[[{}, {}], [{}, {"secondary_y": True}]],
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
    )
    colors = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}
    for coordinate, values in zip(("x", "y", "z"), history.position_m.T):
        figure.add_trace(go.Scatter(
            x=history.time_s, y=values, mode="lines", name=f"{coordinate} position",
            line={"color": colors[coordinate], "width": 2}, legendgroup="position",
            hovertemplate=f"t=%{{x:.3f}} s<br>{coordinate}=%{{y:.3f}} m<extra></extra>",
        ), row=1, col=1)
    figure.add_trace(go.Scatter(
        x=history.time_s, y=history.total_pod, mode="lines", name="Total PoD",
        line={"color": "#111111", "width": 3}, legendgroup="pod",
        customdata=history.cumulative_total_hazard,
        hovertemplate="t=%{x:.3f} s<br>PoD=%{y:.8f}<br>H=%{customdata:.8f}<extra></extra>",
    ), row=1, col=2)
    component_traces = (
        ("Acoustic equivalent PoD", history.acoustic_pod, history.cumulative_acoustic_hazard, "#9467bd"),
        ("RCS/radar equivalent PoD", history.radar_equivalent_pod, history.cumulative_radar_hazard, "#d62728"),
        ("Radial-velocity/Doppler equivalent PoD", history.doppler_equivalent_pod, history.cumulative_doppler_hazard, "#17becf"),
    )
    for name, pod, hazard, color in component_traces:
        figure.add_trace(go.Scatter(
            x=history.time_s, y=pod, mode="lines", name=name,
            line={"color": color, "width": 2}, legendgroup="components",
            customdata=hazard,
            hovertemplate="t=%{x:.3f} s<br>equivalent PoD=%{y:.8f}<br>H_i=%{customdata:.8f}<extra></extra>",
        ), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=history.time_s, y=history.radial_velocity_mps, mode="lines",
        name="Radial velocity", line={"color": "#17becf", "width": 2},
        legendgroup="raw", hovertemplate="t=%{x:.3f} s<br>v_r=%{y:.3f} m/s<extra></extra>",
    ), row=2, col=2, secondary_y=False)
    figure.add_trace(go.Scatter(
        x=history.time_s, y=history.radar_cross_section, mode="lines",
        name="RCS", line={"color": "#d62728", "width": 2, "dash": "dash"},
        legendgroup="raw", hovertemplate="t=%{x:.3f} s<br>RCS=%{y:.5f}<extra></extra>",
    ), row=2, col=2, secondary_y=True)

    for boundary, label in (
        (history.powered_end_time_s, "switch"),
        (history.virtual_end_time_s, "glide lattice"),
    ):
        for row in (1, 2):
            for col in (1, 2):
                figure.add_vline(
                    x=boundary, line={"color": "#777", "dash": "dot", "width": 1},
                    row=row, col=col,
                )
        figure.add_annotation(
            x=boundary, y=1.045, xref="x domain", yref="paper", text=label,
            showarrow=False, textangle=-90, font={"size": 10, "color": "#666"},
        )

    figure.update_xaxes(title_text="mission time [s]", row=1, col=1)
    figure.update_xaxes(title_text="mission time [s]", row=1, col=2)
    figure.update_xaxes(title_text="mission time [s]", row=2, col=1)
    figure.update_xaxes(title_text="mission time [s]", row=2, col=2)
    figure.update_yaxes(title_text="position [m]", row=1, col=1)
    figure.update_yaxes(title_text="detection probability", range=[0.0, 1.0], row=1, col=2)
    figure.update_yaxes(title_text="equivalent detection probability", range=[0.0, 1.0], row=2, col=1)
    figure.update_yaxes(title_text="radial velocity [m/s]", row=2, col=2, secondary_y=False)
    figure.update_yaxes(title_text="RCS [model units]", row=2, col=2, secondary_y=True)
    figure.update_layout(
        title=(
            "Equilibrium Attacker trajectory: time, position, and detection histories"
            "<br><sup>Powered hazard is disabled by the current objective; acoustic hazard is zero. "
            "Component PoDs are diagnostic and are not additive.</sup>"
        ),
        template="plotly_white", height=900, width=1400,
        legend={"x": 1.02, "xanchor": "left", "y": 1.0, "yanchor": "top"},
        margin={"l": 70, "r": 280, "t": 115, "b": 65},
    )
    return figure


__all__ = [
    "ACOUSTIC_ASSUMPTION", "TrajectoryTimeHistory",
    "build_trajectory_time_history", "plot_trajectory_time_history",
]

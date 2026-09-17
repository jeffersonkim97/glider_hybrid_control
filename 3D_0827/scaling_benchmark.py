"""Stage-13 reproducible scaling records for the exact finite solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from numbers import Integral
from statistics import median
from threading import Event, Thread
from time import sleep
from typing import Any, Iterable, Sequence

import numpy as np
import psutil

from discretization_config import DiscretizationConfig
from stage11_config import Stage11Config
from stackelberg_solver import (
    FiniteStackelbergRun,
    generate_defender_line_actions,
    run_finite_stackelberg,
)
from stackelberg_validation import validate_selected_stackelberg_trajectory
from terrain_catalog import build_terrain


@dataclass(frozen=True)
class ScalingRunConfig:
    """One fully reproducible point in a one-variable-at-a-time sweep."""

    run_id: str
    sweep_variable: str
    terrain_category: str = "centered_cube"
    discretization: DiscretizationConfig = field(default_factory=DiscretizationConfig)
    defender_x_map: tuple[float, ...] = (5.0,)
    defender_y_map: float = 0.0
    defender_z_map: float = 0.0
    repeat_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        if not isinstance(self.sweep_variable, str) or not self.sweep_variable.strip():
            raise ValueError("sweep_variable must be a nonempty string")
        if not isinstance(self.discretization, DiscretizationConfig):
            raise TypeError("discretization must be a DiscretizationConfig")
        coordinates = tuple(float(value) for value in self.defender_x_map)
        if not coordinates or any(not np.isfinite(value) for value in coordinates):
            raise ValueError("defender_x_map must contain finite values")
        object.__setattr__(self, "defender_x_map", coordinates)
        for name in ("defender_y_map", "defender_z_map"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if (
            not isinstance(self.repeat_count, Integral)
            or isinstance(self.repeat_count, bool)
            or int(self.repeat_count) < 1
        ):
            raise ValueError("repeat_count must be a positive integer")
        object.__setattr__(self, "repeat_count", int(self.repeat_count))

    def stage11_config(self) -> Stage11Config:
        return Stage11Config(
            terrain_category=self.terrain_category,
            discretization=self.discretization,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sweep_variable": self.sweep_variable,
            "terrain_category": self.terrain_category,
            "discretization": self.discretization.as_dict(),
            "defender_x_map": list(self.defender_x_map),
            "defender_y_map": self.defender_y_map,
            "defender_z_map": self.defender_z_map,
            "defender_action_count": len(self.defender_x_map),
            "repeat_count": self.repeat_count,
            "mission_and_physics": self.stage11_config().as_dict(),
        }


@dataclass(frozen=True)
class RepeatMeasurement:
    repeat_index: int
    selected_defender_action_id: int
    selected_attacker_candidate_id: int
    attacker_objective: float
    defender_payoff: float
    validation_passed: bool
    exhaustive_attacker_verified: bool
    exhaustive_defender_verified: bool
    state_count: int
    edge_count: int
    switch_candidate_count: int
    feasible_defender_action_count: int
    t_graph_build_s: float
    t_los_s: float
    t_hazard_s: float
    t_bellman_s: float
    t_attacker_br_s: float
    t_sse_s: float
    process_rss_start_bytes: int
    process_rss_peak_bytes: int
    process_rss_delta_peak_bytes: int


@dataclass(frozen=True)
class ScalingBenchmarkRecord:
    """One aggregate machine-readable benchmark row required by the plan."""

    run_id: str
    sweep_variable: str
    terrain: str
    grid_dx: float
    grid_dy: float
    altitude_resolution: float
    heading_bins: int
    motion_primitive_radius: int
    motion_primitive_step_cells: int
    state_count: int
    edge_count: int
    switch_candidate_count: int
    defender_action_count: int
    repeat_count: int
    t_graph_build_s: float
    t_los_s: float
    t_hazard_s: float
    t_bellman_s: float
    t_attacker_br_s: float
    t_sse_s: float
    runtime_min_s: float
    runtime_median_s: float
    runtime_max_s: float
    runtime_spread_s: float
    peak_memory_bytes: int
    peak_memory_delta_bytes: int
    attacker_objective: float
    defender_payoff: float
    selected_defender_action_id: int
    selected_attacker_candidate_id: int
    validation_passed: bool
    deterministic: bool
    exhaustive_attacker_verified: bool
    exhaustive_defender_verified: bool
    memory_measurement_method: str
    repeats: tuple[RepeatMeasurement, ...]
    configuration: dict[str, Any]

    def as_dict(self, *, include_repeats: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if not include_repeats:
            result.pop("repeats", None)
            result.pop("configuration", None)
        return result


class ProcessPeakMemorySampler:
    """Sample actual resident process memory while a solver call is active."""

    method = "psutil sampled process RSS (10 ms interval)"

    def __init__(self, interval_s: float = 0.01) -> None:
        self.interval_s = float(interval_s)
        self.process = psutil.Process()
        self.start_rss = int(self.process.memory_info().rss)
        self.peak_rss = self.start_rss
        self._stop = Event()
        self._thread: Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, int(self.process.memory_info().rss))
            sleep(self.interval_s)

    def __enter__(self) -> "ProcessPeakMemorySampler":
        self._thread = Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 4.0 * self.interval_s))
        self.peak_rss = max(self.peak_rss, int(self.process.memory_info().rss))

    @property
    def delta_peak_bytes(self) -> int:
        return max(0, self.peak_rss - self.start_rss)


def _attacker_kwargs(config: Stage11Config) -> dict[str, object]:
    resolution = config.discretization
    return {
        "bellman_grid": resolution.build_bellman_grid(config.graph_bounds),
        "contour_sample_count": resolution.switching_contour_sample_count,
        "radial_scales": resolution.radial_scales,
        "quadrature_resolution": resolution.hazard_quadrature_resolution,
        "los_probe_grid_size": resolution.los_probe_grid_size,
        "los_boundary_refinement_steps": resolution.los_boundary_refinement_steps,
        "los_display_extension_factor": resolution.los_display_extension_factor,
        "visualization_ray_count": resolution.visualization_ray_count,
        "parameters": config.glider,
        "physical_scale": config.physical_scale,
        "detection_parameters": config.detection,
        "objective_parameters": config.attacker_objective,
    }


def _measure_once(
    scaling: ScalingRunConfig,
    repeat_index: int,
) -> tuple[RepeatMeasurement, FiniteStackelbergRun]:
    config = scaling.stage11_config()
    terrain = build_terrain(scaling.terrain_category)
    actions = generate_defender_line_actions(
        scaling.defender_x_map,
        y_map=scaling.defender_y_map,
        z_map=scaling.defender_z_map,
    )
    with ProcessPeakMemorySampler() as memory:
        run = run_finite_stackelberg(
            actions,
            config.attacker_initial_condition,
            terrain=terrain,
            attacker_kwargs=_attacker_kwargs(config),
        )
        audit = validate_selected_stackelberg_trajectory(run, config)
    selected = run.selected_evaluation
    follower = selected.sse_follower_result
    if follower is None:
        raise RuntimeError("selected scaling outcome has no follower trajectory")
    attacker_metrics = tuple(
        evaluation.attacker_run.metrics for evaluation in run.evaluations
    )
    payoffs = tuple(
        evaluation.outcome.defender_payoff
        for evaluation in run.evaluations if evaluation.outcome is not None
    )
    defender_verified = bool(
        payoffs
        and np.isclose(run.outcome.defender_payoff, max(payoffs), rtol=0.0, atol=1.0e-12)
    )
    attacker_verified = all(
        evaluation.attacker_run.selected_result is None
        or np.isclose(
            float(evaluation.attacker_run.selected_result.objective),
            min(
                float(result.objective)
                for result in evaluation.attacker_run.candidate_results
                if result.feasible
            ),
            rtol=0.0, atol=1.0e-12,
        )
        for evaluation in run.evaluations
    )
    representative_graph = run.evaluations[0].attacker_run.graph
    return RepeatMeasurement(
        repeat_index=repeat_index,
        selected_defender_action_id=selected.candidate.action_id,
        selected_attacker_candidate_id=follower.candidate_id,
        attacker_objective=run.outcome.attacker_payoff,
        defender_payoff=run.outcome.defender_payoff,
        validation_passed=audit.report.passed,
        exhaustive_attacker_verified=attacker_verified,
        exhaustive_defender_verified=defender_verified,
        state_count=representative_graph.statistics.state_count,
        edge_count=representative_graph.statistics.valid_edge_count,
        switch_candidate_count=attacker_metrics[0].number_of_candidates,
        feasible_defender_action_count=sum(item.feasible for item in run.evaluations),
        t_graph_build_s=sum(item.timing.graph_build_s for item in attacker_metrics),
        t_los_s=sum(item.timing.los_s for item in attacker_metrics),
        t_hazard_s=sum(item.timing.hazard_precompute_s for item in attacker_metrics),
        t_bellman_s=sum(item.timing.bellman_solve_s for item in attacker_metrics),
        t_attacker_br_s=sum(item.timing.total_attacker_br_s for item in attacker_metrics),
        t_sse_s=run.timing.total_s,
        process_rss_start_bytes=memory.start_rss,
        process_rss_peak_bytes=memory.peak_rss,
        process_rss_delta_peak_bytes=memory.delta_peak_bytes,
    ), run


def run_scaling_benchmark(
    config: ScalingRunConfig,
) -> tuple[ScalingBenchmarkRecord, FiniteStackelbergRun]:
    """Execute and independently validate every requested repetition."""
    measurements: list[RepeatMeasurement] = []
    final_run: FiniteStackelbergRun | None = None
    for repeat_index in range(config.repeat_count):
        measurement, final_run = _measure_once(config, repeat_index)
        measurements.append(measurement)
    if final_run is None:
        raise RuntimeError("benchmark produced no solver run")
    values = tuple(measurements)
    reference = values[0]
    deterministic = all(
        item.selected_defender_action_id == reference.selected_defender_action_id
        and item.selected_attacker_candidate_id == reference.selected_attacker_candidate_id
        and np.isclose(item.attacker_objective, reference.attacker_objective, rtol=0.0, atol=1.0e-12)
        and np.isclose(item.defender_payoff, reference.defender_payoff, rtol=0.0, atol=1.0e-12)
        for item in values
    )
    runtime_values = tuple(item.t_sse_s for item in values)

    def med(name: str) -> float:
        return float(median(getattr(item, name) for item in values))

    resolution = config.discretization
    record = ScalingBenchmarkRecord(
        run_id=config.run_id,
        sweep_variable=config.sweep_variable,
        terrain=config.terrain_category,
        grid_dx=resolution.horizontal_spacing_map,
        grid_dy=resolution.horizontal_spacing_map,
        altitude_resolution=resolution.altitude_spacing_map,
        heading_bins=resolution.heading_bin_count,
        motion_primitive_radius=resolution.motion_primitive_radius,
        motion_primitive_step_cells=resolution.motion_primitive_step_cells,
        state_count=reference.state_count,
        edge_count=reference.edge_count,
        switch_candidate_count=reference.switch_candidate_count,
        defender_action_count=len(config.defender_x_map),
        repeat_count=config.repeat_count,
        t_graph_build_s=med("t_graph_build_s"),
        t_los_s=med("t_los_s"),
        t_hazard_s=med("t_hazard_s"),
        t_bellman_s=med("t_bellman_s"),
        t_attacker_br_s=med("t_attacker_br_s"),
        t_sse_s=float(median(runtime_values)),
        runtime_min_s=min(runtime_values),
        runtime_median_s=float(median(runtime_values)),
        runtime_max_s=max(runtime_values),
        runtime_spread_s=max(runtime_values) - min(runtime_values),
        peak_memory_bytes=max(item.process_rss_peak_bytes for item in values),
        peak_memory_delta_bytes=max(item.process_rss_delta_peak_bytes for item in values),
        attacker_objective=reference.attacker_objective,
        defender_payoff=reference.defender_payoff,
        selected_defender_action_id=reference.selected_defender_action_id,
        selected_attacker_candidate_id=reference.selected_attacker_candidate_id,
        validation_passed=all(item.validation_passed for item in values),
        deterministic=deterministic,
        exhaustive_attacker_verified=all(
            item.exhaustive_attacker_verified for item in values
        ),
        exhaustive_defender_verified=all(
            item.exhaustive_defender_verified for item in values
        ),
        memory_measurement_method=ProcessPeakMemorySampler.method,
        repeats=values,
        configuration=config.as_dict(),
    )
    return record, final_run


def differing_fields(
    reference: ScalingRunConfig,
    candidate: ScalingRunConfig,
) -> tuple[str, ...]:
    """Return named model fields changed between two benchmark configs."""
    differences: list[str] = []
    if reference.terrain_category != candidate.terrain_category:
        differences.append("terrain_category")
    if reference.defender_x_map != candidate.defender_x_map:
        differences.append("defender_x_map")
    if reference.defender_y_map != candidate.defender_y_map:
        differences.append("defender_y_map")
    if reference.defender_z_map != candidate.defender_z_map:
        differences.append("defender_z_map")
    reference_resolution = asdict(reference.discretization)
    candidate_resolution = asdict(candidate.discretization)
    differences.extend(
        f"discretization.{name}"
        for name in reference_resolution
        if reference_resolution[name] != candidate_resolution[name]
    )
    return tuple(differences)


__all__ = [
    "ProcessPeakMemorySampler", "RepeatMeasurement", "ScalingBenchmarkRecord",
    "ScalingRunConfig", "differing_fields", "run_scaling_benchmark",
]

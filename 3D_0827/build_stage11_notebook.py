"""Generate the integrated Stage-1 through Stage-12 validation notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "3D_Attacker_Bellman_Validated.ipynb"


def _cell(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def build_notebook() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.14"},
    }
    notebook.cells = [
        _cell(r'''
# Cell 1 — Configuration
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
from IPython.display import HTML, display

ROOT = Path.cwd().resolve()
if ROOT.name != "3D_0827":
    ROOT = (ROOT / "3D_0827").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacker_best_response import run_attacker_best_response
from bellman_graph import build_bellman_graph, solve_unit_cost_reachability
from candidate_energy import evaluate_switching_candidates
from detection_hazard import GlideDetectionHazardModel
from los_geometry import LOSModel
from mission_response import solve_single_candidate_response
from reachability_surface import LOSSurfaceReachabilityClassifier
from stage11_config import Stage11Config
from stage11_notebook_support import html_table, save_figure, write_json
from switching_candidates import generate_switching_candidates
from terrain_catalog import build_terrain
from trajectory_validation import snapshot_selected_trajectory, validate_trajectory_replay
from visualization import (
    plot_attacker_best_response,
    plot_bellman_reachability,
    plot_candidate_energy_classification,
    plot_candidate_objective_diagnostics,
    plot_los_tangent_surface,
    plot_single_candidate_mission,
    plot_switching_candidates,
    plot_terrain_map,
    plot_trajectory_validation,
)

CONFIG = Stage11Config()
D = CONFIG.discretization
np.random.seed(CONFIG.random_seed)
FIGURE_DIRECTORY = ROOT / "figure" / "stage_11_integrated_notebook"
FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
timings = {}
workflow_started = perf_counter()
write_json(FIGURE_DIRECTORY / "configuration.json", CONFIG.as_dict())

configuration_rows = [
    ("terrain", CONFIG.terrain_category),
    ("sensor [map]", CONFIG.sensor.as_array().tolist()),
    ("start [map]", CONFIG.start.as_array().tolist()),
    ("goal [map]", CONFIG.goal.as_array().tolist()),
    ("physical scale [m/map]", CONFIG.physical_scale.meters_per_map_unit),
    ("aircraft", CONFIG.glider.aircraft_name),
    ("mass [kg]", CONFIG.glider.mass_kg),
    ("best glide speed [m/s]", CONFIG.glider.best_glide_speed_mps),
    ("L/D", CONFIG.glider.best_glide_ratio),
    ("maximum bank [deg]", CONFIG.glider.maximum_bank_deg),
    ("goal tolerance [m]", CONFIG.glider.goal_tolerance_m),
    ("Bellman x/y spacing [map]", D.horizontal_spacing_map),
    ("Bellman altitude spacing [map]", D.altitude_spacing_map),
    ("heading bins / primitive radius", f"{D.heading_bin_count} / {D.motion_primitive_radius}"),
    ("candidate contour × radial", f"{D.switching_contour_sample_count} × {D.switching_radial_sample_count}"),
    ("LOS probe / refinement", f"{D.los_probe_grid_size} / {D.los_boundary_refinement_steps}"),
    ("hazard quadrature", D.hazard_quadrature_resolution),
    ("objective hazard/time weights", f"{CONFIG.attacker_objective.hazard_weight} / {CONFIG.attacker_objective.time_weight}"),
    ("random seed", CONFIG.random_seed),
]
display(HTML("<h2>3D Attacker Bellman — Validated Fixed-Defender Workflow</h2>"))
display(HTML(html_table(("Configuration", "Value"), configuration_rows)))
CONFIG
'''),
        _cell(r'''
# Cell 2 — Terrain
display(HTML("<h3>Cell 2 — Terrain</h3>"))
started = perf_counter()
terrain = build_terrain(CONFIG.terrain_category)
mission_points = CONFIG.mission_points
mission_points.validate_against(terrain)
ground_mesh, obstacle_mesh = terrain.surface_meshes()
terrain_sanity = {
    "bounds": asdict(terrain.bounds),
    "ground_z": terrain.ground_z,
    "maximum_height": terrain.maximum_height,
    "ground_vertices": len(ground_mesh.vertices),
    "ground_triangles": len(ground_mesh.triangles),
    "obstacle_vertices": len(obstacle_mesh.vertices),
    "obstacle_triangles": len(obstacle_mesh.triangles),
}
assert terrain.maximum_height > terrain.ground_z
timings["terrain_s"] = perf_counter() - started
terrain_figure = plot_terrain_map(terrain)
save_figure(terrain_figure, FIGURE_DIRECTORY, "cell_02_terrain.html")
display(HTML(html_table(("Terrain check", "Value"), terrain_sanity.items())))
display(terrain_figure)
terrain_sanity
'''),
        _cell(r'''
# Cell 3 — LOS
display(HTML("<h3>Cell 3 — LOS tangent contour and surface</h3>"))
started = perf_counter()
los_model = LOSModel(terrain)
tangent_contour = los_model.trace_tangent_contour(
    mission_points.sensor,
    probe_grid_size=D.los_probe_grid_size,
    boundary_refinement_steps=D.los_boundary_refinement_steps,
)
visualization_rays = tangent_contour.sample_for_visualization(D.visualization_ray_count)
los_surface = los_model.build_tangent_surface(
    tangent_contour,
    display_extension_factor=D.los_display_extension_factor,
)
timings["los_s"] = perf_counter() - started
los_diagnostics = {
    "tangent_ray_count": len(tangent_contour.rays),
    "displayed_ray_count": len(visualization_rays.rays),
    "discarded_ground_candidates": tangent_contour.discarded_ground_candidate_count,
    "closed": tangent_contour.closed,
    "minimum_tangent_altitude": float(np.min(tangent_contour.tangent_points[:, 2])),
    "surface_panel_count": los_surface.panel_count,
}
assert los_diagnostics["minimum_tangent_altitude"] > terrain.ground_z
los_figure = plot_los_tangent_surface(terrain, mission_points, los_surface, visualization_rays)
save_figure(los_figure, FIGURE_DIRECTORY, "cell_03_los_surface.html")
display(HTML(html_table(("LOS diagnostic", "Value"), los_diagnostics.items())))
display(los_figure)
los_diagnostics
'''),
        _cell(r'''
# Cell 4 — Switching candidates
display(HTML("<h3>Cell 4 — Continuous-surface switching candidates</h3>"))
started = perf_counter()
candidates = generate_switching_candidates(
    tangent_contour,
    contour_sample_count=D.switching_contour_sample_count,
    radial_scales=D.radial_scales,
)
timings["candidate_generation_s"] = perf_counter() - started
residuals = np.asarray([candidate.surface_residual for candidate in candidates])
candidate_diagnostics = {
    "candidate_count": len(candidates),
    "contour_samples": D.switching_contour_sample_count,
    "radial_samples": D.switching_radial_sample_count,
    "maximum_surface_residual": float(np.max(residuals)),
    "mean_surface_residual": float(np.mean(residuals)),
}
candidate_figure = plot_switching_candidates(
    terrain, mission_points, los_surface, visualization_rays, candidates,
)
save_figure(candidate_figure, FIGURE_DIRECTORY, "cell_04_switching_candidates.html")
display(HTML(html_table(("Candidate diagnostic", "Value"), candidate_diagnostics.items())))
display(candidate_figure)
candidate_diagnostics
'''),
        _cell(r'''
# Cell 5 — Energy filtering
display(HTML("<h3>Cell 5 — Powered phase and total-energy filtering</h3>"))
started = perf_counter()
energy_classifier = LOSSurfaceReachabilityClassifier(CONFIG.glider, CONFIG.physical_scale)
energy_evaluations = evaluate_switching_candidates(
    candidates,
    tangent_contour,
    terrain,
    mission_points,
    classifier=energy_classifier,
)
timings["energy_filtering_s"] = perf_counter() - started
energy_counts = {
    "total": len(energy_evaluations),
    "acoustically_neutralized": sum(item.acoustically_neutralized for item in energy_evaluations),
    "powered_feasible": sum(item.powered_feasible for item in energy_evaluations),
    "energy_reachable": sum(item.reachable for item in energy_evaluations),
    "energy_unreachable": sum(item.powered_feasible and not item.reachable for item in energy_evaluations),
}
energy_figure = plot_candidate_energy_classification(
    terrain, mission_points, los_surface, visualization_rays, energy_evaluations,
)
save_figure(energy_figure, FIGURE_DIRECTORY, "cell_05_energy_filtering.html")
display(HTML(html_table(("Energy category", "Count"), energy_counts.items())))
display(energy_figure)
energy_counts
'''),
        _cell(r'''
# Cell 6 — Bellman reachable set
display(HTML("<h3>Cell 6 — Bellman graph and exact goal-reachable set</h3>"))
bellman_grid = D.build_bellman_grid(CONFIG.graph_bounds)
started = perf_counter()
bellman_graph = build_bellman_graph(
    bellman_grid,
    terrain,
    mission_points.goal,
    parameters=CONFIG.glider,
    physical_scale=CONFIG.physical_scale,
)
unit_reachability = solve_unit_cost_reachability(bellman_graph)
topological_order = bellman_graph.topological_sort()
timings["graph_build_s"] = perf_counter() - started
assert len(topological_order) == bellman_graph.statistics.state_count
graph_diagnostics = {
    **asdict(bellman_graph.statistics),
    "goal_reachable_states": unit_reachability.goal_reachable_state_count,
    "xy_spacing_map": bellman_grid.horizontal_spacing_map,
    "altitude_spacing_map": bellman_grid.altitude_spacing_map,
    "heading_bins": bellman_grid.heading_bin_count,
    "motion_primitive_radius": bellman_grid.motion_primitive_radius,
}
bellman_figure = plot_bellman_reachability(bellman_graph, unit_reachability)
save_figure(bellman_figure, FIGURE_DIRECTORY, "cell_06_bellman_reachability.html")
display(HTML(html_table(("Bellman diagnostic", "Value"), graph_diagnostics.items())))
display(bellman_figure)
graph_diagnostics
'''),
        _cell(r'''
# Cell 7 — Single-candidate trajectory
display(HTML("<h3>Cell 7 — Deterministic single-candidate mission</h3>"))
fixed_evaluation = next(
    item for item in energy_evaluations
    if item.candidate_id == CONFIG.deterministic_single_candidate_id
)
hazard_field = GlideDetectionHazardModel(
    terrain,
    mission_points.sensor,
    parameters=CONFIG.detection,
    physical_scale=CONFIG.physical_scale,
)
started = perf_counter()
single_response = solve_single_candidate_response(
    fixed_evaluation,
    bellman_graph,
    unit_reachability,
    mission_points,
    hazard_field,
    parameters=CONFIG.glider,
    physical_scale=CONFIG.physical_scale,
    objective_parameters=CONFIG.attacker_objective,
    quadrature_resolution=D.hazard_quadrature_resolution,
)
timings["single_candidate_s"] = perf_counter() - started
assert single_response.feasible
single_objective = single_response.selected_option.objective
single_diagnostics = {
    "candidate_id": single_response.candidate_id,
    "virtual_target_state": single_response.selected_option.connection.target_state_id,
    "glide_edges": len(single_response.selected_option.glide_edges),
    "mission_time_s": single_objective.mission_time_s,
    "mission_hazard": single_objective.mission_hazard,
    "mission_PoD": single_objective.mission_pod,
    "attacker_objective": single_objective.objective_value,
    "energy_margin_m": single_response.selected_option.energy.margin_m,
}
single_figure = plot_single_candidate_mission(
    terrain, mission_points, los_surface, visualization_rays, single_response,
)
save_figure(single_figure, FIGURE_DIRECTORY, "cell_07_single_candidate.html")
display(HTML(html_table(("Single-candidate result", "Value"), single_diagnostics.items())))
display(single_figure)
single_diagnostics
'''),
        _cell(r'''
# Cell 8 — All-candidate exact optimum
display(HTML("<h3>Cell 8 — Exact exhaustive attacker best response</h3>"))
started = perf_counter()
attacker_run = run_attacker_best_response(
    CONFIG.defender_action,
    CONFIG.attacker_initial_condition,
    terrain=terrain,
    bellman_grid=bellman_grid,
    contour_sample_count=D.switching_contour_sample_count,
    radial_scales=D.radial_scales,
    quadrature_resolution=D.hazard_quadrature_resolution,
    los_probe_grid_size=D.los_probe_grid_size,
    los_boundary_refinement_steps=D.los_boundary_refinement_steps,
    los_display_extension_factor=D.los_display_extension_factor,
    visualization_ray_count=D.visualization_ray_count,
    parameters=CONFIG.glider,
    physical_scale=CONFIG.physical_scale,
    detection_parameters=CONFIG.detection,
    objective_parameters=CONFIG.attacker_objective,
)
timings["all_candidate_br_s"] = perf_counter() - started
selected = attacker_run.selected_result
assert selected is not None and selected.feasible
finite_objectives = [item.objective for item in attacker_run.candidate_results if item.feasible]
assert np.isclose(selected.objective, min(finite_objectives), rtol=0.0, atol=1.0e-12)
candidate_rows = [
    (
        item.candidate_id,
        item.status_category,
        "" if item.objective is None else f"{item.objective:.9f}",
        "" if item.mission_time_s is None else f"{item.mission_time_s:.6f}",
        "" if item.cumulative_hazard is None else f"{item.cumulative_hazard:.9f}",
        item.infeasibility_reason or "",
    )
    for item in attacker_run.candidate_results
]
optimum_diagnostics = {
    "selected_candidate_id": selected.candidate_id,
    "cooptimal_candidate_ids": attacker_run.cooptimal_candidate_ids,
    "objective": selected.objective,
    "mission_time_s": selected.mission_time_s,
    "cumulative_hazard": selected.cumulative_hazard,
    "detection_probability": selected.detection_probability,
    "feasible_candidates": attacker_run.metrics.number_feasible,
    "total_candidates": attacker_run.metrics.number_of_candidates,
}
optimum_figure = plot_attacker_best_response(attacker_run)
objective_figure = plot_candidate_objective_diagnostics(attacker_run)
save_figure(optimum_figure, FIGURE_DIRECTORY, "cell_08_exact_attacker_best_response.html")
save_figure(objective_figure, FIGURE_DIRECTORY, "cell_08_candidate_objectives.html")
display(HTML(html_table(("Optimum diagnostic", "Value"), optimum_diagnostics.items())))
display(HTML(html_table(
    ("candidate", "status", "objective", "time_s", "hazard", "reason"),
    candidate_rows,
    title="Exhaustive candidate table",
)))
display(optimum_figure)
display(objective_figure)
optimum_diagnostics
'''),
        _cell(r'''
# Cell 9 — Independent validation report
display(HTML("<h3>Cell 9 — Independent continuous replay validation</h3>"))
validation_started = perf_counter()
trajectory_snapshot = snapshot_selected_trajectory(
    attacker_run,
    hazard_quadrature_resolution=D.hazard_quadrature_resolution,
)
validation_audit = validate_trajectory_replay(
    trajectory_snapshot,
    terrain,
    mission_points,
    tangent_contour,
    hazard_field,
    parameters=CONFIG.glider,
    physical_scale=CONFIG.physical_scale,
)
timings["validation_s"] = perf_counter() - validation_started
validation_report = validation_audit.report
assert validation_report.passed
validation_rows = [
    ("terrain clear", validation_report.terrain_clear),
    ("LOS phase valid", validation_report.los_phase_valid),
    ("turn constraints valid", validation_report.turn_constraints_valid),
    ("energy valid", validation_report.energy_valid),
    ("time consistent", validation_report.time_consistent),
    ("hazard consistent", validation_report.hazard_consistent),
    ("goal valid", validation_report.goal_valid),
    ("max turn violation [rad]", validation_report.max_turn_violation),
    ("time error [s]", validation_report.time_error_s),
    ("hazard error", validation_report.hazard_error),
    ("goal distance [m]", validation_audit.goal_distance_m),
    ("energy margin [m]", validation_audit.energy_margin_m),
    ("overall", "PASS" if validation_report.passed else "FAIL"),
]
validation_figure = plot_trajectory_validation(
    terrain,
    mission_points,
    trajectory_snapshot,
    validation_audit,
    goal_tolerance_map=(CONFIG.glider.goal_tolerance_m / CONFIG.physical_scale.meters_per_map_unit),
)
save_figure(validation_figure, FIGURE_DIRECTORY, "cell_09_independent_validation.html")
display(HTML(html_table(("Replay check", "Result"), validation_rows)))
display(validation_figure)
validation_report
'''),
        _cell(r'''
# Cell 10 — Timing and final structured report
display(HTML("<h3>Cell 10 — Timing report and Stage-11 gate</h3>"))
run_timing = attacker_run.metrics.timing
timing_report = {
    "terrain_LOS_s": timings["terrain_s"] + timings["los_s"],
    "candidate_generation_s": run_timing.candidate_generation_s,
    "energy_filtering_s": run_timing.energy_filter_s,
    "graph_build_s": run_timing.graph_build_s,
    "hazard_precompute_s": run_timing.hazard_precompute_s,
    "Bellman_solve_s": run_timing.bellman_solve_s,
    "all_candidate_BR_s": run_timing.total_attacker_br_s,
    "validation_s": timings["validation_s"],
    "notebook_total_s": perf_counter() - workflow_started,
}
summary = {
    "stage": 11,
    "gate_passed": validation_report.passed,
    "configuration": CONFIG.as_dict(),
    "terrain": terrain_sanity,
    "los": los_diagnostics,
    "candidates": candidate_diagnostics,
    "energy": energy_counts,
    "graph": graph_diagnostics,
    "single_candidate": single_diagnostics,
    "exact_attacker_best_response": optimum_diagnostics,
    "validation_report": asdict(validation_report),
    "validation_recomputed": {
        "mission_time_s": validation_audit.recomputed_mission_time_s,
        "cumulative_hazard": validation_audit.recomputed_cumulative_hazard,
        "detection_probability": validation_audit.recomputed_detection_probability,
        "goal_distance_m": validation_audit.goal_distance_m,
        "energy_margin_m": validation_audit.energy_margin_m,
    },
    "timing_s": timing_report,
}
write_json(FIGURE_DIRECTORY / "stage11_summary.json", summary)
display(HTML(html_table(
    ("Stage", "Seconds"),
    ((name, f"{seconds:.6f}") for name, seconds in timing_report.items()),
)))
display(HTML("<h3 style='color:#16803a'>Stage 11 gate: PASS</h3>"))
summary
'''),
        _cell(r'''
# Cell 11 - Fine-grid Stage 12 Stackelberg solve
display(HTML("<h3>Cell 11 - 25 m / 25 m / 5 deg sparse Stackelberg solve</h3>"))
from stackelberg_gui import run_resolution_case

STAGE12_FIGURE_DIRECTORY = ROOT / "figure" / "stage_12_integrated_notebook"
STAGE12_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
stage12_started = perf_counter()
stage12_result = run_resolution_case(
    horizontal_step_m=25.0,
    altitude_step_m=25.0,
    heading_step_deg=5.0,
)
stage12_run = stage12_result.run
stage12_summary = dict(stage12_result.summary)
stage12_summary["stage"] = 12
stage12_summary["gate_passed"] = stage12_result.validation_passed
stage12_summary["notebook_cell_runtime_s"] = perf_counter() - stage12_started
assert stage12_result.validation_passed
assert stage12_summary["cartesian_state_count"] == 3_243_240
assert stage12_summary["goal_reachable_state_count"] < stage12_summary["cartesian_state_count"]
assert stage12_summary["active_corridor_state_count"] < stage12_summary["goal_reachable_state_count"]
save_figure(
    stage12_result.figure,
    STAGE12_FIGURE_DIRECTORY,
    "cell_11_sparse_stackelberg_reachable_set.html",
)
write_json(STAGE12_FIGURE_DIRECTORY / "stage12_integrated_summary.json", stage12_summary)
display(HTML(html_table(("Stage 12 fine-grid result", "Value"), stage12_summary.items())))
display(stage12_result.figure)
stage12_summary
'''),
        _cell(r'''
# Cell 12 - Exhaustive Defender table and Stage 12 gate
display(HTML("<h3>Cell 12 - Finite Defender enumeration and regression gate</h3>"))
from visualization import plot_defender_payoff_comparison

defender_rows = []
for evaluation in stage12_run.evaluations:
    follower = evaluation.sse_follower_result
    defender_rows.append((
        evaluation.candidate.action_id,
        evaluation.candidate.action.sensor_position_map.tolist(),
        evaluation.feasible,
        None if follower is None else follower.candidate_id,
        None if evaluation.outcome is None else evaluation.outcome.attacker_payoff,
        None if evaluation.outcome is None else evaluation.outcome.defender_payoff,
        evaluation.attacker_run.metrics.goal_reachable_state_count,
        evaluation.attacker_run.metrics.active_state_count,
        evaluation.attacker_br_runtime_s,
    ))
assert len(defender_rows) == 3
assert stage12_run.outcome.defender_payoff == max(
    row[5] for row in defender_rows if row[5] is not None
)
payoff_figure = plot_defender_payoff_comparison(stage12_run)
save_figure(
    payoff_figure,
    STAGE12_FIGURE_DIRECTORY,
    "cell_12_defender_payoff_comparison.html",
)
display(HTML(html_table(
    (
        "D", "sensor [map]", "feasible", "A*", "attacker J", "Defender PoD",
        "goal-reachable states", "active corridor states", "runtime [s]",
    ),
    defender_rows,
)))
display(payoff_figure)
display(HTML("<h3 style='color:#16803a'>Stage 12 integration gate: PASS</h3>"))
defender_rows
'''),
        _cell(r'''
# Cell 13 - Standalone browser GUI
display(HTML("<h3>Cell 13 - Resolution-control GUI in a separate browser window</h3>"))
import os
from stackelberg_gui import launch_stackelberg_gui

if "STACKELBERG_GUI_SERVER" in globals():
    try:
        STACKELBERG_GUI_SERVER.shutdown()
    except Exception:
        pass

STACKELBERG_GUI_SERVER = launch_stackelberg_gui(
    initial_result=stage12_result,
    open_browser=(os.environ.get("BELLMAN_GUI_HEADLESS") != "1"),
)
display(HTML(
    "<p>The standalone interactive GUI is running at "
    f"<a href='{STACKELBERG_GUI_SERVER.url}' target='_blank'>"
    f"{STACKELBERG_GUI_SERVER.url}</a>.</p>"
    "<p>Re-run this GUI-launch cell if the notebook kernel was restarted. "
    "The three sliders control horizontal, altitude, and heading discretization; "
    "the Run button recomputes the exact finite Stackelberg game.</p>"
))
STACKELBERG_GUI_SERVER.url
'''),
        _cell(r'''
# Cell 14 - Equilibrium trajectory time histories (after GUI)
display(HTML("<h3>Cell 14 - Time histories for the selected Stackelberg trajectory</h3>"))
from detection_hazard import GlideDetectionHazardModel
from stackelberg_validation import selected_attacker_run
from trajectory_time_history import (
    build_trajectory_time_history,
    plot_trajectory_time_history,
)

equilibrium_attacker_run = selected_attacker_run(stage12_run)
equilibrium_hazard_field = GlideDetectionHazardModel(
    equilibrium_attacker_run.terrain,
    equilibrium_attacker_run.mission_points.sensor,
    parameters=CONFIG.detection,
    physical_scale=CONFIG.physical_scale,
)
trajectory_time_history = build_trajectory_time_history(
    equilibrium_attacker_run,
    equilibrium_hazard_field,
    quadrature_resolution=CONFIG.discretization.hazard_quadrature_resolution,
    parameters=CONFIG.glider,
    physical_scale=CONFIG.physical_scale,
)
trajectory_time_history_figure = plot_trajectory_time_history(
    trajectory_time_history,
)
trajectory_time_history_summary = trajectory_time_history.summary()
assert trajectory_time_history_summary["cumulative_hazard_error"] <= 1.0e-10
assert trajectory_time_history_summary["detection_probability_error"] <= 1.0e-10
assert trajectory_time_history_summary["mission_time_error_s"] <= 1.0e-10
save_figure(
    trajectory_time_history_figure,
    STAGE12_FIGURE_DIRECTORY,
    "cell_14_trajectory_time_histories.html",
)
write_json(
    STAGE12_FIGURE_DIRECTORY / "cell_14_trajectory_time_history_summary.json",
    trajectory_time_history_summary,
)
display(HTML(
    "<p><b>Model interpretation:</b> acoustic hazard is identically zero under "
    "the current LOS-tangent acoustic-neutralization assumption. RCS/radar and "
    "radial-velocity/Doppler curves are component-equivalent PoDs "
    "<code>1-exp(-H_i)</code>; they are not additive. Total PoD is computed from "
    "the sum of component hazards.</p>"
))
display(HTML(html_table(
    ("Time-history diagnostic", "Value"),
    trajectory_time_history_summary.items(),
)))
display(trajectory_time_history_figure)
trajectory_time_history_summary
'''),
    ]
    nbf.write(notebook, NOTEBOOK_PATH)
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(build_notebook())

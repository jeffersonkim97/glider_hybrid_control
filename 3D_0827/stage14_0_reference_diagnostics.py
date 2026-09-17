"""Build and validate the Stage-14.0 exact-SSE frozen-reference artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from stage11_notebook_support import write_json
from stage14_benchmark_contract import (
    CANONICAL_DEFENDER_X_MAP,
    CANONICAL_RESOLUTION,
    DEFAULT_STAGE14_TOLERANCES,
    SCHEMA_VERSION,
    benchmark_configuration,
    benchmark_result_schema,
    canonical_stage14_config,
    environment_manifest,
    frozen_reference_from_run,
    validate_frozen_reference,
)
from stackelberg_gui import run_resolution_case
from stackelberg_solver import FiniteStackelbergRun
from stackelberg_validation import validate_selected_stackelberg_trajectory
from trajectory_validation import TrajectoryReplayAudit


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIRECTORY = ROOT / "figure" / "stage_14_0_benchmark_contract"
NOTEBOOK_STAGE12_SUMMARY = (
    ROOT / "figure" / "stage_12_integrated_notebook" / "stage12_integrated_summary.json"
)
NOTEBOOK_TIME_HISTORY_SUMMARY = (
    ROOT / "figure" / "stage_12_integrated_notebook"
    / "cell_14_trajectory_time_history_summary.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    import json

    if not path.is_file():
        raise FileNotFoundError(f"required validated artifact is missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return result


def _notebook_regression(reference: dict[str, Any]) -> dict[str, Any]:
    notebook = _read_json(NOTEBOOK_STAGE12_SUMMARY)
    history = _read_json(NOTEBOOK_TIME_HISTORY_SUMMARY)
    identity = reference["solution_identity"]
    sizes = reference["state_and_game_size"]
    tolerance = DEFAULT_STAGE14_TOLERANCES
    checks = {
        "selected_defender_action_id": (
            identity["selected_defender_action_id"]
            == notebook["selected_defender_action_id"]
        ),
        "selected_attacker_candidate_id": (
            identity["selected_attacker_candidate_id"]
            == notebook["selected_attacker_candidate_id"]
        ),
        "sensor_position": bool(np.allclose(
            identity["selected_sensor_position_map"],
            notebook["selected_sensor_position_map"],
            rtol=0.0,
            atol=tolerance.sensor_position_map_abs,
        )),
        "attacker_objective": bool(np.isclose(
            identity["attacker_objective"], notebook["attacker_objective"],
            rtol=0.0, atol=tolerance.attacker_objective_abs,
        )),
        "defender_objective": bool(np.isclose(
            identity["defender_objective_pod"], notebook["defender_payoff_pod"],
            rtol=0.0, atol=tolerance.defender_objective_abs,
        )),
        "mission_time": bool(np.isclose(
            identity["mission_time_s"], history["stored_mission_time_s"],
            rtol=0.0, atol=tolerance.mission_time_s_abs,
        )),
        "cumulative_hazard": bool(np.isclose(
            identity["cumulative_hazard"], history["stored_cumulative_hazard"],
            rtol=0.0, atol=tolerance.cumulative_hazard_abs,
        )),
        "detection_probability": bool(np.isclose(
            identity["detection_probability"], history["stored_detection_probability"],
            rtol=0.0, atol=tolerance.detection_probability_abs,
        )),
        "cartesian_states": sizes["N_S_cart"] == notebook["cartesian_state_count"],
        "goal_reachable_states": (
            sizes["N_S_goal_reachable"] == notebook["goal_reachable_state_count"]
        ),
        "active_states": sizes["N_S_active"] == notebook["active_corridor_state_count"],
        "active_edges": sizes["N_E"] == notebook["active_edge_count"],
        "validated_notebook_gate": bool(
            notebook["gate_passed"] and notebook["validation_passed"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "stage12_summary": str(NOTEBOOK_STAGE12_SUMMARY.relative_to(ROOT)),
        "time_history_summary": str(NOTEBOOK_TIME_HISTORY_SUMMARY.relative_to(ROOT)),
    }


def _write_tables(output_directory: Path, reference: dict[str, Any]) -> None:
    """Write the Stage-14.0 reference table as a self-contained static image."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import textwrap

    identity = reference["solution_identity"]
    sizes = reference["state_and_game_size"]
    identity_rows = (
        ("schema version", reference["schema_version"]),
        ("feasible", identity["feasible"]),
        ("selected Defender", f"D{identity['selected_defender_action_id']}"),
        ("sensor position [map]", identity["selected_sensor_position_map"]),
        ("leader co-optimal actions", identity["leader_cooptimal_action_ids"]),
        ("Attacker objective co-optima", identity["attacker_objective_cooptimal_candidate_ids"]),
        ("SSE-selected Attacker", f"A{identity['selected_attacker_candidate_id']}"),
        ("Attacker objective J_A", identity["attacker_objective"]),
        ("Defender objective J_D = PoD", identity["defender_objective_pod"]),
        ("mission time [s]", identity["mission_time_s"]),
        ("cumulative hazard", identity["cumulative_hazard"]),
        ("trajectory SHA-256", identity["trajectory_identity"]["sha256"]),
        ("independent replay", reference["independent_replay"]["passed"]),
    )
    state_rows = tuple(
        (name, value, reference["quantity_definitions"].get(name, {}).get("definition", "derived diagnostic"))
        for name, value in sizes.items()
    )

    def wrapped(value: Any, width: int) -> str:
        return "\n".join(textwrap.wrap(str(value), width=width))

    figure, axes = plt.subplots(
        2, 1, figsize=(15, 15), dpi=200,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    figure.suptitle(
        "Stage 14.0 - Canonical exact-SSE frozen reference",
        fontsize=16, fontweight="bold",
    )
    for axis in axes:
        axis.axis("off")
    identity_table = axes[0].table(
        cellText=[(wrapped(name, 34), wrapped(value, 88)) for name, value in identity_rows],
        colLabels=("Quantity", "Frozen value"), colWidths=(0.25, 0.75),
        cellLoc="left", colLoc="left", loc="center",
    )
    identity_table.auto_set_font_size(False)
    identity_table.set_fontsize(8)
    identity_table.scale(1.0, 1.55)
    figure.text(0.02, 0.945, "Canonical solution identity", fontsize=12, fontweight="bold")
    state_table = axes[1].table(
        cellText=[
            (wrapped(name, 18), wrapped(value, 16), wrapped(definition, 74))
            for name, value, definition in state_rows
        ],
        colLabels=("Symbol", "Value", "Definition"),
        colWidths=(0.18, 0.13, 0.69), cellLoc="left", colLoc="left", loc="center",
    )
    state_table.auto_set_font_size(False)
    state_table.set_fontsize(7.5)
    state_table.scale(1.0, 1.55)
    figure.text(
        0.02, 0.485, "State, edge, candidate, and game-size definitions",
        fontsize=12, fontweight="bold",
    )
    for table in (identity_table, state_table):
        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor("#b8b8b8")
            if row == 0:
                cell.set_facecolor("#d9e6f2")
                cell.set_text_props(fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f5f7fa")
    figure.text(
        0.01, 0.01, "No runtime scaling curve is produced in Stage 14.0.",
        fontsize=8, color="#555555",
    )
    figure.tight_layout(rect=(0.01, 0.03, 0.99, 0.965))
    figure.savefig(
        output_directory / "baseline_identity_and_state_counts.png",
        dpi=200, bbox_inches="tight", facecolor="white",
    )
    plt.close(figure)


def run_stage14_0_diagnostics(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    allow_scenario_rebaseline: bool = False,
) -> tuple[FiniteStackelbergRun, TrajectoryReplayAudit, dict[str, Any]]:
    """Run the unmodified canonical solver and freeze the Stage-14 contract."""
    output_directory.mkdir(parents=True, exist_ok=True)
    config = canonical_stage14_config()
    horizontal_m, altitude_m, heading_deg = CANONICAL_RESOLUTION
    result = run_resolution_case(
        horizontal_step_m=horizontal_m,
        altitude_step_m=altitude_m,
        heading_step_deg=heading_deg,
        base_config=config,
        defender_x=CANONICAL_DEFENDER_X_MAP,
    )
    run = result.run
    audit = validate_selected_stackelberg_trajectory(run, config)
    environment = environment_manifest(ROOT.parent)
    reference = frozen_reference_from_run(run, audit, environment)
    frozen_regression = validate_frozen_reference(reference)
    notebook_regression = _notebook_regression(reference)
    source_fingerprint = reference["solver_source_fingerprint"]
    source_matches_manifest = (
        source_fingerprint
        == environment["software_revision"]["solver_source"]["aggregate_sha256"]
    )
    legacy_regression_passed = bool(
        frozen_regression["passed"] and notebook_regression["passed"]
    )
    rebaseline_authorized = bool(allow_scenario_rebaseline)
    reference["regression"] = {
        "frozen_pre_stage14": frozen_regression,
        "validated_notebook": notebook_regression,
        "solver_source_fingerprint_matches_manifest": source_matches_manifest,
        "solver_behavior_unchanged": legacy_regression_passed,
        "scenario_rebaseline_authorized": rebaseline_authorized,
        "scenario_rebaseline_reason": (
            "Defender grid changed from the legacy three-point prototype to "
            "the approved uniform six-point grid [5, 6, 7, 8, 9, 10]."
            if rebaseline_authorized else None
        ),
    }
    reference["gate_passed"] = bool(
        (legacy_regression_passed or rebaseline_authorized)
        and audit.report.passed
        and source_matches_manifest
    )
    if not reference["gate_passed"]:
        raise RuntimeError("Stage-14.0 frozen-reference regression gate failed")

    write_json(output_directory / "benchmark_configuration.json", benchmark_configuration())
    write_json(output_directory / "benchmark_result_schema.json", benchmark_result_schema())
    write_json(output_directory / "environment_manifest.json", environment)
    write_json(output_directory / "frozen_reference.json", reference)
    _write_tables(output_directory, reference)
    summary = {
        "stage": "14.0",
        "schema_version": SCHEMA_VERSION,
        "gate_passed": True,
        "solver_behavior_unchanged": legacy_regression_passed,
        "scenario_rebaseline_authorized": rebaseline_authorized,
        "independent_replay_passed": audit.report.passed,
        "solver_source_fingerprint": source_fingerprint,
        "configuration": benchmark_configuration(),
        "state_and_game_size": reference["state_and_game_size"],
        "solution_identity": reference["solution_identity"],
        "regression": reference["regression"],
        "runtime_scaling_curve_produced": False,
        "profiling_added": False,
        "sweep_runner_added": False,
        "regression_tolerances": asdict(DEFAULT_STAGE14_TOLERANCES),
    }
    write_json(output_directory / "stage14_0_summary.json", summary)

    selected = reference["solution_identity"]
    print("Stage 14.0 frozen exact-SSE reference: PASS")
    print(
        f"D{selected['selected_defender_action_id']} / "
        f"A{selected['selected_attacker_candidate_id']}; "
        f"J_A={selected['attacker_objective']:.15f}; "
        f"J_D={selected['defender_objective_pod']:.15f}"
    )
    print(f"solver source fingerprint: {source_fingerprint}")
    return run, audit, reference


if __name__ == "__main__":
    run_stage14_0_diagnostics()

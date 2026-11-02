"""Build the paper-ready Phase-6 Bellman-vs-frozen-DQN pilot artifacts.

The source Phase-6 runner recorded two deterministic DQN Local-SSE passes:

* ``warmup_wall_sec_excluded``: the first solve with empty observation,
  action-row, and scenario caches after scene/checkpoint setup;
* ``wall_runtime_sec``: a repeated solve of the identical query with the
  caches left populated.

This closeout makes the first solve the primary deployment timing and retains
the repeated solve only as a secondary latency diagnostic.  It does not rerun
or alter any Bellman/DQN solution.
"""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import project_paths
from defender_grid_2d import DefenderRegion, build_defender_grid_2d
from phase4_extended_dataset import build_random_box_terrain
from phase5_g2_final_id_test import configure_final_terrain, freeze_final_id_000
from phase6_g2_local_sse import PHASE6_SCHEMA


CLOSEOUT_SCHEMA = "p1b-phase6-paper-integration-closeout-v1"
TERRAIN_INDEX = 1
RESOLUTIONS_M = (100.0, 50.0, 25.0, 10.0)
HEADING_SPACING_DEG = 5.0
R_NEIGHBOR = 1
JA_DIAGNOSTIC_THRESHOLD = 0.10
SOURCE_ROOT = (
    project_paths.DQN_DIR / "runs" / "phase6_g2_local_sse"
    / f"final_id_{TERRAIN_INDEX:03d}"
)
OUTPUT_DIR = SOURCE_ROOT / "paper_closeout_v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_path(dx_m: float) -> Path:
    return (
        SOURCE_ROOT / f"dx{dx_m:g}m_dpsi5deg_r1"
        / "phase6_g2_local_sse_pilot.json"
    )


def _final_dqn_record(result: dict[str, Any]) -> dict[str, Any]:
    final_id = int(
        result["frozen_dqn_local_sse"]["search"]
        ["final_local_sse_action_id"]
    )
    selected = [
        item for item in result["frozen_dqn_local_sse"]["evaluated"]
        if int(item["action_id"]) == final_id
    ]
    if len(selected) != 1:
        raise RuntimeError(f"expected one DQN final record for action {final_id}")
    return selected[0]


def _row(dx_m: float, result: dict[str, Any]) -> dict[str, Any]:
    if result.get("schema") != PHASE6_SCHEMA or result.get("status") != "complete":
        raise RuntimeError(f"incomplete or incompatible Phase-6 result at dx={dx_m:g}")
    condition = result["condition"]
    if (
        float(condition["spatial_resolution_m"]) != float(dx_m)
        or float(condition["heading_spacing_deg"]) != HEADING_SPACING_DEG
        or int(condition["r_neighbor"]) != R_NEIGHBOR
        or int(condition["terrain_index"]) != TERRAIN_INDEX
    ):
        raise RuntimeError(f"condition mismatch at dx={dx_m:g}")

    exact = result["exact_bellman_local_sse"]
    dqn = result["frozen_dqn_local_sse"]
    comparison = result["comparison"]
    dqn_final = _final_dqn_record(result)
    exact_at_dqn = comparison["dqn_selected_sensor_exact_response"]
    exact_ja = float(exact_at_dqn["attacker_objective"])
    dqn_ja = float(dqn_final["attacker_objective"])
    exact_jd = float(exact["final_response"]["detection_probability"])
    exact_jd_at_dqn = float(exact_at_dqn["detection_probability"])
    dqn_jd = float(dqn_final["detection_probability"])
    ja_error = abs(dqn_ja - exact_ja) / abs(exact_ja)
    jd_regret = (exact_jd - exact_jd_at_dqn) / abs(exact_jd)
    jd_response_error = abs(dqn_jd - exact_jd_at_dqn) / abs(exact_jd_at_dqn)
    local_certificate = bool(
        comparison["dqn_selected_sensor_exact_neighbor_audit"]
        ["exact_local_sse_verified"]
    )
    first_solve = float(dqn["warmup_wall_sec_excluded"])
    repeated_solve = float(dqn["wall_runtime_sec"])
    bellman_time = float(exact["wall_runtime_sec"])
    ja_diagnostic_pass = ja_error <= JA_DIAGNOSTIC_THRESHOLD
    quality_admissible = ja_diagnostic_pass and local_certificate
    if not quality_admissible:
        pilot_label = "BELLMAN_QUALITY_REQUIRED"
    elif first_solve < bellman_time:
        pilot_label = "DQN_FIRST_SOLVE_FASTER"
    else:
        pilot_label = "BELLMAN_FIRST_SOLVE_FASTER"

    return {
        "spatial_resolution_m": float(dx_m),
        "heading_spacing_deg": HEADING_SPACING_DEG,
        "r_neighbor": R_NEIGHBOR,
        "bellman_final_action_id": int(
            exact["search"]["final_local_sse_action_id"]
        ),
        "dqn_final_action_id": int(
            dqn["search"]["final_local_sse_action_id"]
        ),
        "bellman_final_sensor_map": list(exact["final_response"]["sensor_map"]),
        "dqn_final_sensor_map": list(dqn_final["sensor_map"]),
        "selected_sensor_distance_m": float(
            comparison["selected_sensor_distance_m"]
        ),
        "same_sensor_exact_J_A": exact_ja,
        "same_sensor_dqn_J_A": dqn_ja,
        "same_sensor_J_A_relative_error": ja_error,
        "bellman_final_J_D": exact_jd,
        "exact_J_D_at_dqn_sensor": exact_jd_at_dqn,
        "dqn_response_J_D_at_dqn_sensor": dqn_jd,
        "dqn_response_J_D_relative_error": jd_response_error,
        "defender_J_D_relative_regret": jd_regret,
        "exact_local_optimality_margin_J_D": float(
            comparison["dqn_selected_sensor_exact_neighbor_audit"]
            ["exact_local_optimality_margin_J_D"]
        ),
        "dqn_selected_sensor_exact_local_sse_verified": local_certificate,
        "same_sensor_J_A_10pct_diagnostic_pass": ja_diagnostic_pass,
        "phase6_pilot_quality_admissible": quality_admissible,
        "phase6_pilot_label": pilot_label,
        "bellman_first_solve_sec": bellman_time,
        "dqn_first_solve_sec": first_solve,
        "dqn_repeated_query_sec": repeated_solve,
        "bellman_over_dqn_first_solve_ratio": bellman_time / first_solve,
        "bellman_over_dqn_repeated_query_ratio": bellman_time / repeated_solve,
        "bellman_defender_evaluations": int(
            exact["search"]["unique_defender_evaluations"]
        ),
        "dqn_defender_evaluations": int(
            dqn["search"]["unique_defender_evaluations"]
        ),
        "bellman_visited_defender_actions": list(
            exact["search"]["visited_defender_actions"]
        ),
        "dqn_visited_defender_actions": list(
            dqn["search"]["visited_defender_actions"]
        ),
        "bellman_evaluated_defender_actions": list(
            exact["search"]["evaluated_defender_actions"]
        ),
        "dqn_evaluated_defender_actions": list(
            dqn["search"]["evaluated_defender_actions"]
        ),
    }


def _plot_runtime(rows: list[dict[str, Any]], output: Path) -> None:
    ordered = sorted(rows, key=lambda item: item["spatial_resolution_m"])
    dx = np.asarray([item["spatial_resolution_m"] for item in ordered])
    figure, axis = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    axis.plot(dx, [item["bellman_first_solve_sec"] for item in ordered],
              "o-", linewidth=2.3, label="Bellman first solve", color="#D55E00")
    axis.plot(dx, [item["dqn_first_solve_sec"] for item in ordered],
              "s-", linewidth=2.3, label="DQN first solve", color="#0072B2")
    axis.plot(dx, [item["dqn_repeated_query_sec"] for item in ordered],
              "^-", linewidth=2.0, label="DQN repeated query", color="#009E73")
    axis.set_yscale("log")
    axis.set_xticks(dx)
    axis.set_xlabel(r"Spatial discretization $dx=dy=dh$ [m]")
    axis.set_ylabel("Local-SSE wall time [s] (log scale)")
    axis.set_title("Phase 6: first-solve and repeated-query timing")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(frameon=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_quality(rows: list[dict[str, Any]], output: Path) -> None:
    ordered = sorted(rows, key=lambda item: item["spatial_resolution_m"])
    dx = np.asarray([item["spatial_resolution_m"] for item in ordered])
    ja = 100.0 * np.asarray([
        item["same_sensor_J_A_relative_error"] for item in ordered
    ])
    jd = 100.0 * np.asarray([
        item["defender_J_D_relative_regret"] for item in ordered
    ])
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    axes[0].plot(dx, ja, "o-", linewidth=2.3, color="#0072B2")
    axes[0].axhline(10.0, color="#D55E00", linestyle="--", linewidth=1.8,
                    label="Phase-5 J_A diagnostic (10%)")
    axes[0].set_ylabel(r"Same-sensor $J_A$ relative error [%]")
    axes[0].legend(frameon=True)
    axes[1].plot(dx, jd, "s-", linewidth=2.3, color="#CC79A7")
    axes[1].set_ylabel(r"Exact defender $J_D$ regret [%]")
    for axis in axes:
        axis.set_xticks(dx)
        axis.set_xlabel(r"Spatial discretization $dx=dy=dh$ [m]")
        axis.grid(True, alpha=0.3)
    figure.suptitle("Follower-objective accuracy does not ensure leader-decision quality")
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_search_trace(
    rows: list[dict[str, Any]], results: dict[float, dict[str, Any]], output: Path,
) -> None:
    configure_final_terrain(TERRAIN_INDEX)
    _, _, terrain_spec, config = freeze_final_id_000()
    terrain = build_random_box_terrain(terrain_spec, config.pilot)
    scale = float(config.pilot.meters_per_map_unit)
    figures, axes = plt.subplots(2, 2, figsize=(12.0, 10.0), constrained_layout=True)
    by_dx = {float(row["spatial_resolution_m"]): row for row in rows}
    for axis, dx in zip(axes.flat, RESOLUTIONS_M):
        result = results[dx]
        row = by_dx[dx]
        region_data = result["condition"]["defender_region_map"]
        region = DefenderRegion(
            x_min=float(region_data["x_min"]), x_max=float(region_data["x_max"]),
            y_min=float(region_data["y_min"]), y_max=float(region_data["y_max"]),
        )
        grid = build_defender_grid_2d(
            region, dx / scale, terrain=terrain,
        )
        for box in terrain.obstacle_boxes():
            x_low, x_high = box.x_limits
            y_low, y_high = box.y_limits
            axis.add_patch(plt.Rectangle(
                (x_low * scale, y_low * scale),
                (x_high - x_low) * scale, (y_high - y_low) * scale,
                facecolor="#999999", edgecolor="#555555", alpha=0.45,
            ))

        exact_search = result["exact_bellman_local_sse"]["search"]
        dqn_search = result["frozen_dqn_local_sse"]["search"]
        exact_evaluated = np.asarray([
            grid.position(int(action_id))[:2]
            for action_id in exact_search["evaluated_defender_actions"]
        ]) * scale
        dqn_evaluated = np.asarray([
            grid.position(int(action_id))[:2]
            for action_id in dqn_search["evaluated_defender_actions"]
        ]) * scale
        exact_path = np.asarray([
            grid.position(int(action_id))[:2]
            for action_id in exact_search["visited_defender_actions"]
        ]) * scale
        dqn_path = np.asarray([
            grid.position(int(action_id))[:2]
            for action_id in dqn_search["visited_defender_actions"]
        ]) * scale
        axis.scatter(exact_evaluated[:, 0], exact_evaluated[:, 1], s=20,
                     color="#D55E00", alpha=0.25, label="Bellman evaluated")
        axis.scatter(dqn_evaluated[:, 0], dqn_evaluated[:, 1], s=20,
                     color="#0072B2", alpha=0.25, label="DQN evaluated")
        axis.plot(exact_path[:, 0], exact_path[:, 1], "o-", color="#D55E00",
                  linewidth=2.0, markersize=4, label="Bellman accepted path")
        axis.plot(dqn_path[:, 0], dqn_path[:, 1], "s-", color="#0072B2",
                  linewidth=2.0, markersize=4, label="DQN accepted path")
        axis.scatter(*exact_path[-1], s=180, marker="*", color="#D55E00",
                     edgecolor="black", linewidth=0.6, zorder=6)
        axis.scatter(*dqn_path[-1], s=150, marker="X", color="#0072B2",
                     edgecolor="black", linewidth=0.6, zorder=6)
        axis.set_title(
            f"dx={dx:g} m | distance={row['selected_sensor_distance_m']:.1f} m | "
            f"certificate={'PASS' if row['dqn_selected_sensor_exact_local_sse_verified'] else 'FAIL'}"
        )
        axis.set_xlabel("sensor x [m]")
        axis.set_ylabel("sensor y [m]")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figures.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
    figures.suptitle("Phase 6 defender Local-SSE search traces", fontsize=15)
    figures.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figures)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    scalar_keys = [
        key for key, value in rows[0].items()
        if not isinstance(value, (dict, list))
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in scalar_keys})


def _markdown(rows: list[dict[str, Any]], outputs: dict[str, Path]) -> str:
    lines = [
        "# Phase 6 Paper-Ready Integration Closeout",
        "",
        "Frozen G2 DQN versus exact Bellman Local-SSE on final-ID Terrain 001. ",
        "No training, fine-tuning, or checkpoint selection was performed.",
        "",
        "## Primary timing definition",
        "",
        "The primary DQN time is the first Local-SSE solve with empty scenario, ",
        "observation, and action-row caches after shared scene construction and ",
        "checkpoint loading. The repeated identical-query time is secondary only.",
        "",
        "| dx | Bellman first solve | DQN first solve | DQN repeated | "
        "Bellman/DQN first | J_A error | J_D regret | exact local certificate |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['spatial_resolution_m']:g} m | "
            f"{row['bellman_first_solve_sec']:.3f} s | "
            f"{row['dqn_first_solve_sec']:.3f} s | "
            f"{row['dqn_repeated_query_sec']:.3f} s | "
            f"{row['bellman_over_dqn_first_solve_ratio']:.3f}x | "
            f"{100.0 * row['same_sensor_J_A_relative_error']:.2f}% | "
            f"{100.0 * row['defender_J_D_relative_regret']:.2f}% | "
            f"{'PASS' if row['dqn_selected_sensor_exact_local_sse_verified'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Phase-6 finding",
        "",
        "All four DQN responses satisfy the Phase-5 same-sensor J_A 10% diagnostic, ",
        "but only dx=100 m preserves the exact Local-SSE sensor decision. At that ",
        "coarse condition Bellman is faster for a first unseen solve. Therefore no ",
        "observed Phase-6 condition supports DQN as both first-solve faster and ",
        "leader-decision admissible. This is a single-terrain integration result, ",
        "not the Phase-7 Monte Carlo regime boundary.",
        "",
        "## Artifacts",
        "",
        *[f"- {name}: `{path.as_posix()}`" for name, path in outputs.items()],
        "",
        "## Scope limitations",
        "",
        "- Terrain 001 only; no Monte Carlo uncertainty statement.",
        "- dpsi=5 deg and r=1 only.",
        "- Bellman versus generalized DQN is the primary paper comparison; the ",
        "  case-specific Tabular baseline remains deferred to the Phase-7 secondary analysis.",
        "- The 100 m defender region was symmetrically center-aligned to the lattice.",
        "- J_A closeness alone is not an end-to-end Stackelberg certificate.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_results: dict[float, dict[str, Any]] = {}
    source_files: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for dx in RESOLUTIONS_M:
        path = _source_path(dx)
        result = json.loads(path.read_text(encoding="utf-8"))
        source_results[dx] = result
        source_files.append({
            "path": str(path.relative_to(project_paths.WORKSPACE_ROOT)).replace("\\", "/"),
            "sha256": _sha256(path),
        })
        rows.append(_row(dx, result))

    outputs = {
        "aggregate_json": OUTPUT_DIR / "phase6_paper_closeout.json",
        "aggregate_csv": OUTPUT_DIR / "phase6_paper_closeout.csv",
        "report": OUTPUT_DIR / "phase6_paper_closeout.md",
        "runtime_figure": OUTPUT_DIR / "phase6_runtime_first_vs_repeated.png",
        "quality_figure": OUTPUT_DIR / "phase6_quality_error_vs_regret.png",
        "search_trace_figure": OUTPUT_DIR / "phase6_defender_search_traces.png",
        **{
            f"solution_3d_dx{dx:g}m": (
                OUTPUT_DIR / f"phase6_final_solutions_dx{dx:g}m_3d.html"
            )
            for dx in RESOLUTIONS_M
        },
    }
    _plot_runtime(rows, outputs["runtime_figure"])
    _plot_quality(rows, outputs["quality_figure"])
    _plot_search_trace(rows, source_results, outputs["search_trace_figure"])
    _write_csv(rows, outputs["aggregate_csv"])
    outputs["report"].write_text(
        _markdown(rows, outputs), encoding="utf-8",
    )
    missing_outputs = [
        str(path) for name, path in outputs.items()
        if name != "aggregate_json" and not path.is_file()
    ]
    if missing_outputs:
        raise RuntimeError(
            "paper artifact generation is incomplete: " + ", ".join(missing_outputs)
        )
    manifest_without_hash = {
        "schema": CLOSEOUT_SCHEMA,
        "status": "complete",
        "scope": {
            "terrain_index": TERRAIN_INDEX,
            "spatial_resolutions_m": list(RESOLUTIONS_M),
            "heading_spacing_deg": HEADING_SPACING_DEG,
            "r_neighbor": R_NEIGHBOR,
            "bellman_vs_generalized_dqn_primary": True,
            "tabular_secondary_deferred_to_phase7": True,
            "phase7_monte_carlo_claim": False,
        },
        "timing_contract": {
            "primary": "first Local-SSE solve with empty DQN problem caches",
            "secondary": "repeated identical query with retained DQN caches",
            "excluded": [
                "shared scene construction", "checkpoint load",
                "offline DQN training", "post-hoc exact neighbor audit",
            ],
            "legacy_field_mapping": {
                "dqn_first_solve_sec": "warmup_wall_sec_excluded",
                "dqn_repeated_query_sec": "wall_runtime_sec",
            },
        },
        "quality_contract": {
            "same_sensor_J_A_10pct": "Phase-5 follower diagnostic only",
            "end_to_end_primary": "exact Bellman audit of DQN-selected sensor and r=1 neighbors",
            "leader_quality_outputs": [
                "exact local certificate", "exact J_D regret",
                "exact local optimality margin",
            ],
        },
        "source_files": source_files,
        "rows": rows,
        "outputs": {
            name: str(path.relative_to(project_paths.WORKSPACE_ROOT)).replace("\\", "/")
            for name, path in outputs.items()
            if name != "aggregate_json"
        },
        "output_sha256": {
            name: _sha256(path)
            for name, path in outputs.items()
            if name != "aggregate_json"
        },
    }
    manifest = {
        **manifest_without_hash,
        "manifest_sha256": sha256(
            _canonical_json(manifest_without_hash).encode("utf-8")
        ).hexdigest(),
    }
    _write_json(outputs["aggregate_json"], manifest)
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()


__all__ = ["CLOSEOUT_SCHEMA", "main"]

"""Controlled Phase-7 Chebyshev-neighborhood sweep at dx=25 m, dpsi=10 deg."""

from __future__ import annotations

import argparse
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
from phase4_g2_training import OUTPUT_ROOT as G2_OUTPUT_ROOT, _condition_tag
from phase6_g2_local_sse import run_phase6_pilot


PHASE7_NEIGHBOR_SCHEMA = "p1b-phase7-neighbor-sweep-v1"
LOCAL_SSE_RESULT_SCHEMA = "p1b-phase7-neighbor-local-sse-v1"
DX_M = 25.0
HEADING_DEG = 10.0
RADII = (1, 2, 3, 4, 5)
TERRAIN_INDEX = 1
PHASE7_ROOT = project_paths.DQN_DIR / "runs" / "phase7_neighbor_sweep"
CLOSEOUT_DIR = PHASE7_ROOT / f"final_id_{TERRAIN_INDEX:03d}" / "closeout_v1"


def configure_heading(heading_deg: float) -> None:
    """Select the controlled angular condition without mixing output roots."""

    global HEADING_DEG, PHASE7_ROOT, CLOSEOUT_DIR
    heading_deg = float(heading_deg)
    if heading_deg not in (5.0, 10.0):
        raise ValueError("neighbor sweep supports dpsi=5 or 10 deg")
    HEADING_DEG = heading_deg
    PHASE7_ROOT = (
        project_paths.DQN_DIR / "runs" / "phase7_neighbor_sweep"
        if heading_deg == 10.0
        else project_paths.DQN_DIR / "runs" / "phase7_neighbor_sweep_dpsi5deg"
    )
    CLOSEOUT_DIR = (
        PHASE7_ROOT / f"final_id_{TERRAIN_INDEX:03d}" / "closeout_v1"
    )


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


def _training_summary_path() -> Path:
    return (
        G2_OUTPUT_ROOT / _condition_tag(DX_M, HEADING_DEG)
        / "seed0_ep10000_v1" / "g2_training_summary.json"
    )


def _local_sse_path(radius: int) -> Path:
    condition = f"dx{DX_M:g}m_dpsi{HEADING_DEG:g}deg_r{int(radius)}"
    if int(radius) == 1:
        if HEADING_DEG == 5.0:
            return (
                project_paths.DQN_DIR / "runs" / "phase6_g2_local_sse"
                / f"final_id_{TERRAIN_INDEX:03d}" / condition
                / "phase6_g2_local_sse_pilot.json"
            )
        return (
            project_paths.DQN_DIR / "runs" / "phase7_angular_sweep"
            / f"final_id_{TERRAIN_INDEX:03d}" / condition
            / "phase7_angular_local_sse.json"
        )
    return (
        PHASE7_ROOT / f"final_id_{TERRAIN_INDEX:03d}" / condition
        / "phase7_neighbor_local_sse.json"
    )


def run_local_sse(radius: int) -> Path:
    radius = int(radius)
    if radius not in RADII:
        raise ValueError(f"unsupported Chebyshev radius: {radius}")
    if radius == 1:
        path = _local_sse_path(radius)
        if not path.is_file():
            raise FileNotFoundError(f"missing angular-sweep r=1 anchor: {path}")
        return path
    return run_phase6_pilot(
        terrain_index=TERRAIN_INDEX,
        spatial_resolution_m=DX_M,
        heading_spacing_deg=HEADING_DEG,
        r_neighbor=radius,
        device_name="cuda",
        output_root=PHASE7_ROOT,
        result_schema=LOCAL_SSE_RESULT_SCHEMA,
        experiment_label="Phase-7 neighborhood sweep",
        output_stem="phase7_neighbor_local_sse",
    )


def _final_dqn_record(result: dict[str, Any]) -> dict[str, Any]:
    action_id = int(
        result["frozen_dqn_local_sse"]["search"]["final_local_sse_action_id"]
    )
    selected = [
        item for item in result["frozen_dqn_local_sse"]["evaluated"]
        if int(item["action_id"]) == action_id
    ]
    if len(selected) != 1:
        raise RuntimeError(f"expected one final DQN record for action {action_id}")
    return selected[0]


def _row(radius: int) -> dict[str, Any]:
    path = _local_sse_path(radius)
    if not path.is_file():
        raise FileNotFoundError(f"missing Local-SSE result: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    condition = result["condition"]
    if (
        float(condition["spatial_resolution_m"]) != DX_M
        or float(condition["heading_spacing_deg"]) != HEADING_DEG
        or int(condition["r_neighbor"]) != int(radius)
        or int(condition["terrain_index"]) != TERRAIN_INDEX
    ):
        raise RuntimeError(f"condition mismatch at r={radius}")
    exact = result["exact_bellman_local_sse"]
    dqn = result["frozen_dqn_local_sse"]
    comparison = result["comparison"]
    dqn_final = _final_dqn_record(result)
    exact_at_dqn = comparison["dqn_selected_sensor_exact_response"]
    audit = comparison["dqn_selected_sensor_exact_neighbor_audit"]
    exact_ja = float(exact_at_dqn["attacker_objective"])
    dqn_ja = float(dqn_final["attacker_objective"])
    exact_jd = float(exact["final_response"]["detection_probability"])
    exact_at_dqn_jd = float(exact_at_dqn["detection_probability"])
    bellman_time = float(exact["wall_runtime_sec"])
    dqn_time = float(dqn["warmup_wall_sec_excluded"])
    certificate = bool(audit["exact_local_sse_verified"])
    ja_error = abs(dqn_ja - exact_ja) / abs(exact_ja)
    jd_signed_gap = (exact_jd - exact_at_dqn_jd) / abs(exact_jd)
    bellman_action = int(exact["search"]["final_local_sse_action_id"])
    dqn_action = int(dqn["search"]["final_local_sse_action_id"])
    action_match = bellman_action == dqn_action
    return {
        "spatial_resolution_m": DX_M,
        "heading_spacing_deg": HEADING_DEG,
        "r_neighbor": int(radius),
        "maximum_interior_neighbor_count": (2 * int(radius) + 1) ** 2 - 1,
        "bellman_defender_evaluations": int(
            exact["search"]["unique_defender_evaluations"]
        ),
        "dqn_defender_evaluations": int(
            dqn["search"]["unique_defender_evaluations"]
        ),
        "bellman_first_solve_sec": bellman_time,
        "dqn_first_solve_sec": dqn_time,
        "dqn_repeated_query_sec": float(dqn["wall_runtime_sec"]),
        "bellman_over_dqn_first_solve_ratio": bellman_time / dqn_time,
        "bellman_final_action_id": bellman_action,
        "dqn_final_action_id": dqn_action,
        "selected_defender_action_match": action_match,
        "selected_sensor_distance_m": float(comparison["selected_sensor_distance_m"]),
        "same_sensor_exact_J_A": exact_ja,
        "same_sensor_dqn_J_A": dqn_ja,
        "same_sensor_J_A_relative_error": ja_error,
        "bellman_final_J_D": exact_jd,
        "exact_J_D_at_dqn_sensor": exact_at_dqn_jd,
        "defender_J_D_signed_relative_gap": jd_signed_gap,
        "defender_J_D_relative_regret_clipped": max(0.0, jd_signed_gap),
        "exact_local_optimality_margin_J_D": float(
            audit["exact_local_optimality_margin_J_D"]
        ),
        "exact_local_sse_verified": certificate,
        "same_sensor_J_A_10pct_diagnostic_pass": ja_error <= 0.10,
        "dqn_first_solve_faster": dqn_time < bellman_time,
        "local_sse_result": path.relative_to(
            project_paths.WORKSPACE_ROOT
        ).as_posix(),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(rows: list[dict[str, Any]]) -> dict[str, Path]:
    radii = np.asarray([row["r_neighbor"] for row in rows])
    runtime_path = CLOSEOUT_DIR / "phase7_neighbor_runtime.png"
    evaluation_path = CLOSEOUT_DIR / "phase7_neighbor_evaluations.png"
    quality_path = CLOSEOUT_DIR / "phase7_neighbor_quality.png"

    figure, axis = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    axis.plot(radii, [row["bellman_first_solve_sec"] for row in rows],
              "o-", linewidth=2.2, label="Bellman first solve")
    axis.plot(radii, [row["dqn_first_solve_sec"] for row in rows],
              "s-", linewidth=2.2, label="DQN first solve")
    axis.plot(radii, [row["dqn_repeated_query_sec"] for row in rows],
              "^-", linewidth=2.0, label="DQN repeated query")
    axis.set_yscale("log")
    axis.set_xticks(radii)
    axis.set_xlabel("Chebyshev neighborhood radius r")
    axis.set_ylabel("Local-SSE wall time [s] (log scale)")
    axis.set_title(
        rf"Neighborhood sweep at $dx=25$ m and $\Delta\psi={HEADING_DEG:g}^\circ$"
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(runtime_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    axis.plot(radii, [row["bellman_defender_evaluations"] for row in rows],
              "o-", linewidth=2.2, label="Bellman")
    axis.plot(radii, [row["dqn_defender_evaluations"] for row in rows],
              "s-", linewidth=2.2, label="DQN")
    axis.plot(radii, [row["maximum_interior_neighbor_count"] for row in rows],
              "--", linewidth=1.8, label=r"maximum $(2r+1)^2-1$")
    axis.set_xticks(radii)
    axis.set_xlabel("Chebyshev neighborhood radius r")
    axis.set_ylabel("Unique defender positions")
    axis.set_title("Actual Local-SSE BR evaluation counts")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(evaluation_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    axis.plot(radii, 100.0 * np.asarray([
        row["same_sensor_J_A_relative_error"] for row in rows
    ]), "o-", linewidth=2.2, label=r"same-sensor $J_A$ error")
    axis.plot(radii, 100.0 * np.asarray([
        row["defender_J_D_signed_relative_gap"] for row in rows
    ]), "s-", linewidth=2.2, label=r"exact defender $J_D$ signed gap")
    axis.axhline(0.0, linestyle=":", color="#666666", linewidth=1.2)
    axis.axhline(10.0, linestyle="--", color="black", linewidth=1.4,
                 label=r"$J_A$ diagnostic threshold")
    axis.set_xticks(radii)
    axis.set_xlabel("Chebyshev neighborhood radius r")
    axis.set_ylabel("Relative difference [%]")
    axis.set_title("Neighborhood-size quality propagation")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(quality_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {
        "runtime_figure": runtime_path,
        "evaluation_figure": evaluation_path,
        "quality_figure": quality_path,
    }


def aggregate() -> Path:
    CLOSEOUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_row(radius) for radius in RADII]
    outputs = {
        "json": CLOSEOUT_DIR / "phase7_neighbor_sweep.json",
        "csv": CLOSEOUT_DIR / "phase7_neighbor_sweep.csv",
        "report": CLOSEOUT_DIR / "phase7_neighbor_sweep.md",
        **_plots(rows),
    }
    _write_csv(rows, outputs["csv"])
    lines = [
        "# Phase 7 Chebyshev-Neighborhood Sweep",
        "",
        (
            f"Controlled comparison at `dx=25 m`, `dpsi={HEADING_DEG:g} deg`, "
            "unseen final-ID Terrain 001."
        ),
        "The same frozen seed-0, episode-10,000 G2 checkpoint is used for every radius.",
        "",
        "Negative signed J_D gap means the DQN-selected sensor has a higher exact defender payoff than the Bellman local-search outcome.",
        "",
        "| r | max neighbors | Bellman evals | DQN evals | Bellman first | DQN first | DQN faster | J_A error | signed J_D gap | certificate | action match |",
        "|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['r_neighbor']} | {row['maximum_interior_neighbor_count']} | "
            f"{row['bellman_defender_evaluations']} | "
            f"{row['dqn_defender_evaluations']} | "
            f"{row['bellman_first_solve_sec']:.3f} s | "
            f"{row['dqn_first_solve_sec']:.3f} s | "
            f"{'yes' if row['dqn_first_solve_faster'] else 'no'} | "
            f"{100.0 * row['same_sensor_J_A_relative_error']:.2f}% | "
            f"{100.0 * row['defender_J_D_signed_relative_gap']:.2f}% | "
            f"{'PASS' if row['exact_local_sse_verified'] else 'FAIL'} | "
            f"{'yes' if row['selected_defender_action_match'] else 'no'} |"
        )
    outputs["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    training_path = _training_summary_path()
    source_paths = {"training": training_path}
    source_paths.update({
        f"local_sse_r{radius}": _local_sse_path(radius) for radius in RADII
    })
    payload_without_hash = {
        "schema": PHASE7_NEIGHBOR_SCHEMA,
        "status": "complete",
        "condition_contract": {
            "spatial_resolution_m": DX_M,
            "heading_spacing_deg": HEADING_DEG,
            "r_neighbors": list(RADII),
            "terrain_index": TERRAIN_INDEX,
            "same_frozen_checkpoint_for_all_radii": True,
        },
        "timing_contract": {
            "primary_dqn": "first Local-SSE solve with empty problem caches",
            "secondary_dqn": "repeated identical query with retained caches",
            "offline_training_reported_separately": True,
        },
        "sources": {
            name: {
                "path": path.relative_to(project_paths.WORKSPACE_ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for name, path in source_paths.items()
        },
        "rows": rows,
        "outputs": {
            name: path.relative_to(project_paths.WORKSPACE_ROOT).as_posix()
            for name, path in outputs.items() if name != "json"
        },
    }
    payload = {
        **payload_without_hash,
        "manifest_sha256": sha256(
            _canonical_json(payload_without_hash).encode("utf-8")
        ).hexdigest(),
    }
    _write_json(outputs["json"], payload)
    return outputs["json"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-local-sse", type=int)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--heading-spacing-deg", type=float, choices=(5.0, 10.0), default=10.0,
    )
    arguments = parser.parse_args()
    configure_heading(arguments.heading_spacing_deg)
    if arguments.run_local_sse is not None:
        print(run_local_sse(arguments.run_local_sse), flush=True)
    elif arguments.aggregate:
        print(aggregate(), flush=True)
    else:
        parser.error("select --run-local-sse R or --aggregate")


if __name__ == "__main__":
    main()


__all__ = ["aggregate", "configure_heading", "run_local_sse"]

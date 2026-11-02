"""Controlled Phase-7 angular sweep at dx=25 m and Local-SSE radius r=1.

The 5-degree condition reuses the frozen Phase-6 result.  The 10- and
2.5-degree conditions require their own seed-0, episode-10,000 G2 checkpoints
because angular spacing changes the DQN action catalog and compatibility
signature.  This module never trains or selects a checkpoint.
"""

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


PHASE7_ANGULAR_SCHEMA = "p1b-phase7-angular-sweep-v1"
LOCAL_SSE_RESULT_SCHEMA = "p1b-phase7-angular-local-sse-v1"
DX_M = 25.0
R_NEIGHBOR = 1
TERRAIN_INDEX = 1
HEADING_SPACINGS_DEG = (10.0, 5.0, 2.5)
PHASE7_ROOT = project_paths.DQN_DIR / "runs" / "phase7_angular_sweep"
CLOSEOUT_DIR = PHASE7_ROOT / f"final_id_{TERRAIN_INDEX:03d}" / "closeout_v1"


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


def _training_summary_path(heading_deg: float) -> Path:
    return (
        G2_OUTPUT_ROOT
        / _condition_tag(DX_M, heading_deg)
        / "seed0_ep10000_v1"
        / "g2_training_summary.json"
    )


def _local_sse_path(heading_deg: float) -> Path:
    condition = f"dx{DX_M:g}m_dpsi{heading_deg:g}deg_r{R_NEIGHBOR}"
    if float(heading_deg) == 5.0:
        return (
            project_paths.DQN_DIR / "runs" / "phase6_g2_local_sse"
            / f"final_id_{TERRAIN_INDEX:03d}" / condition
            / "phase6_g2_local_sse_pilot.json"
        )
    return (
        PHASE7_ROOT / f"final_id_{TERRAIN_INDEX:03d}" / condition
        / "phase7_angular_local_sse.json"
    )


def run_local_sse(heading_deg: float) -> Path:
    heading_deg = float(heading_deg)
    if heading_deg not in HEADING_SPACINGS_DEG:
        raise ValueError(f"unsupported angular condition: {heading_deg:g} deg")
    if heading_deg == 5.0:
        path = _local_sse_path(heading_deg)
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen Phase-6 anchor: {path}")
        return path
    return run_phase6_pilot(
        terrain_index=TERRAIN_INDEX,
        spatial_resolution_m=DX_M,
        heading_spacing_deg=heading_deg,
        r_neighbor=R_NEIGHBOR,
        device_name="cuda",
        output_root=PHASE7_ROOT,
        result_schema=LOCAL_SSE_RESULT_SCHEMA,
        experiment_label="Phase-7 angular sweep",
        output_stem="phase7_angular_local_sse",
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


def _row(heading_deg: float) -> dict[str, Any]:
    local_path = _local_sse_path(heading_deg)
    training_path = _training_summary_path(heading_deg)
    if not local_path.is_file():
        raise FileNotFoundError(f"missing Local-SSE result: {local_path}")
    if not training_path.is_file():
        raise FileNotFoundError(f"missing G2 training summary: {training_path}")
    result = json.loads(local_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    condition = result["condition"]
    if (
        float(condition["spatial_resolution_m"]) != DX_M
        or float(condition["heading_spacing_deg"]) != float(heading_deg)
        or int(condition["r_neighbor"]) != R_NEIGHBOR
        or int(condition["terrain_index"]) != TERRAIN_INDEX
    ):
        raise RuntimeError(f"Local-SSE condition mismatch at {heading_deg:g} deg")
    compatibility = training["seed_result"]["compatibility"]
    if (
        float(compatibility["spatial_discretization_m"]) != DX_M
        or float(compatibility["angular_discretization_deg"]) != float(heading_deg)
    ):
        raise RuntimeError(f"training condition mismatch at {heading_deg:g} deg")

    exact = result["exact_bellman_local_sse"]
    dqn = result["frozen_dqn_local_sse"]
    comparison = result["comparison"]
    dqn_final = _final_dqn_record(result)
    exact_at_dqn = comparison["dqn_selected_sensor_exact_response"]
    exact_ja = float(exact_at_dqn["attacker_objective"])
    dqn_ja = float(dqn_final["attacker_objective"])
    exact_jd = float(exact["final_response"]["detection_probability"])
    exact_at_dqn_jd = float(exact_at_dqn["detection_probability"])
    audit = comparison["dqn_selected_sensor_exact_neighbor_audit"]
    seed = training["seed_result"]
    best_snapshot = seed["best_snapshot"]
    validation_summary = best_snapshot["VALIDATION_SIMPLE"]
    validation_errors = np.asarray([
        float(item["relative_error"])
        for item in best_snapshot["cases"]
        if item["split"] == "VALIDATION_SIMPLE"
    ])
    if validation_errors.size != int(validation_summary["case_count"]):
        raise RuntimeError(
            f"validation case accounting mismatch at {heading_deg:g} deg"
        )
    validation_p90 = float(np.percentile(
        validation_errors, 90.0, method="higher",
    ))
    validation_gate = bool(
        float(validation_summary["success_rate"]) == 1.0
        and float(validation_summary["median_relative_error"]) <= 0.10
        and validation_p90 <= 0.30
    )
    ja_error = abs(dqn_ja - exact_ja) / abs(exact_ja)
    jd_regret = (exact_jd - exact_at_dqn_jd) / abs(exact_jd)
    certificate = bool(audit["exact_local_sse_verified"])
    first_solve = float(dqn["warmup_wall_sec_excluded"])
    bellman_solve = float(exact["wall_runtime_sec"])
    return {
        "spatial_resolution_m": DX_M,
        "heading_spacing_deg": float(heading_deg),
        "r_neighbor": R_NEIGHBOR,
        "action_count": int(training["network_config"]["action_count"]),
        "training_episodes": int(seed["episodes_completed"]),
        "training_measured_seed_runtime_sec": float(seed["measured_seed_runtime_sec"]),
        "training_pure_runtime_sec": float(seed["pure_training_runtime_sec"]),
        "training_evaluation_runtime_sec": float(seed["evaluation_runtime_sec"]),
        "training_best_episode": int(seed["best_episode"]),
        "training_early_stopping_success": bool(
            seed["early_stopping_decision"]["success"]
        ),
        "training_early_stopping_reason": str(
            seed["early_stopping_decision"]["reason"]
        ),
        "validation_success_rate": float(validation_summary["success_rate"]),
        "validation_median_relative_error": float(
            validation_summary["median_relative_error"]
        ),
        "validation_p90_relative_error_higher": validation_p90,
        "validation_maximum_relative_error": float(
            validation_summary["maximum_relative_error"]
        ),
        "frozen_model_validation_gate_pass": validation_gate,
        "bellman_first_solve_sec": bellman_solve,
        "dqn_first_solve_sec": first_solve,
        "dqn_repeated_query_sec": float(dqn["wall_runtime_sec"]),
        "bellman_over_dqn_first_solve_ratio": bellman_solve / first_solve,
        "bellman_final_action_id": int(exact["search"]["final_local_sse_action_id"]),
        "dqn_final_action_id": int(dqn["search"]["final_local_sse_action_id"]),
        "selected_sensor_distance_m": float(comparison["selected_sensor_distance_m"]),
        "same_sensor_exact_J_A": exact_ja,
        "same_sensor_dqn_J_A": dqn_ja,
        "same_sensor_J_A_relative_error": ja_error,
        "bellman_final_J_D": exact_jd,
        "exact_J_D_at_dqn_sensor": exact_at_dqn_jd,
        "defender_J_D_relative_regret": jd_regret,
        "exact_local_optimality_margin_J_D": float(
            audit["exact_local_optimality_margin_J_D"]
        ),
        "exact_local_sse_verified": certificate,
        "same_sensor_J_A_10pct_diagnostic_pass": ja_error <= 0.10,
        "primary_admissible_and_faster": (
            certificate and first_solve < bellman_solve
        ),
        "training_summary": training_path.relative_to(
            project_paths.WORKSPACE_ROOT
        ).as_posix(),
        "local_sse_result": local_path.relative_to(
            project_paths.WORKSPACE_ROOT
        ).as_posix(),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]]) -> dict[str, Path]:
    ordered = sorted(rows, key=lambda item: item["heading_spacing_deg"])
    psi = np.asarray([item["heading_spacing_deg"] for item in ordered])
    runtime_path = CLOSEOUT_DIR / "phase7_angular_runtime.png"
    quality_path = CLOSEOUT_DIR / "phase7_angular_quality.png"
    training_path = CLOSEOUT_DIR / "phase7_angular_training_time.png"

    figure, axis = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    axis.plot(psi, [item["bellman_first_solve_sec"] for item in ordered],
              "o-", linewidth=2.2, label="Bellman first solve")
    axis.plot(psi, [item["dqn_first_solve_sec"] for item in ordered],
              "s-", linewidth=2.2, label="DQN first solve")
    axis.plot(psi, [item["dqn_repeated_query_sec"] for item in ordered],
              "^-", linewidth=2.0, label="DQN repeated query")
    axis.set_yscale("log")
    axis.set_xticks(psi)
    axis.set_xlabel(r"Heading discretization $\Delta\psi$ [deg]")
    axis.set_ylabel("Local-SSE wall time [s] (log scale)")
    axis.set_title(r"Angular sweep at $dx=25$ m and $r=1$")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(runtime_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    axis.plot(psi, 100.0 * np.asarray([
        item["same_sensor_J_A_relative_error"] for item in ordered
    ]), "o-", linewidth=2.2, label=r"same-sensor $J_A$ error")
    axis.plot(psi, 100.0 * np.asarray([
        item["defender_J_D_relative_regret"] for item in ordered
    ]), "s-", linewidth=2.2, label=r"exact defender $J_D$ regret")
    axis.axhline(10.0, linestyle="--", color="black", linewidth=1.4,
                 label=r"$J_A$ diagnostic threshold")
    axis.set_xticks(psi)
    axis.set_xlabel(r"Heading discretization $\Delta\psi$ [deg]")
    axis.set_ylabel("Relative difference [%]")
    axis.set_title(r"Angular-discretization quality propagation")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(quality_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    axis.plot(psi, np.asarray([
        item["training_measured_seed_runtime_sec"] for item in ordered
    ]) / 60.0, "o-", linewidth=2.2, color="#009E73")
    axis.set_xticks(psi)
    axis.set_xlabel(r"Heading discretization $\Delta\psi$ [deg]")
    axis.set_ylabel("Seed-0 measured training runtime [min]")
    axis.set_title(r"Condition-specific DQN training cost")
    axis.grid(True, alpha=0.3)
    figure.savefig(training_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {
        "runtime_figure": runtime_path,
        "quality_figure": quality_path,
        "training_figure": training_path,
    }


def aggregate() -> Path:
    CLOSEOUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_row(heading) for heading in HEADING_SPACINGS_DEG]
    outputs = {
        "json": CLOSEOUT_DIR / "phase7_angular_sweep.json",
        "csv": CLOSEOUT_DIR / "phase7_angular_sweep.csv",
        "report": CLOSEOUT_DIR / "phase7_angular_sweep.md",
        **_plot(rows),
    }
    _write_csv(rows, outputs["csv"])
    lines = [
        "# Phase 7 Angular-Discretization Sweep",
        "",
        "Controlled comparison at `dx=25 m`, `r=1`, unseen final-ID Terrain 001.",
        "Each angular condition uses its own seed-0, episode-10,000 G2 model.",
        "",
        "## Training and validation",
        "",
        "| dpsi | actions | training | validation median | validation P90 | frozen-model gate |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['heading_spacing_deg']:g} deg | {row['action_count']} | "
            f"{row['training_measured_seed_runtime_sec'] / 60.0:.2f} min | "
            f"{100.0 * row['validation_median_relative_error']:.2f}% | "
            f"{100.0 * row['validation_p90_relative_error_higher']:.2f}% | "
            f"{'PASS' if row['frozen_model_validation_gate_pass'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Unseen-terrain Local-SSE",
        "",
        "| dpsi | Bellman first | DQN first | J_A error | J_D regret | exact certificate |",
        "|---:|---:|---:|---:|---:|:---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['heading_spacing_deg']:g} deg | "
            f"{row['bellman_first_solve_sec']:.3f} s | "
            f"{row['dqn_first_solve_sec']:.3f} s | "
            f"{100.0 * row['same_sensor_J_A_relative_error']:.2f}% | "
            f"{100.0 * row['defender_J_D_relative_regret']:.2f}% | "
            f"{'PASS' if row['exact_local_sse_verified'] else 'FAIL'} |"
        )
    outputs["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    source_paths = {
        f"training_dpsi{heading:g}": _training_summary_path(heading)
        for heading in HEADING_SPACINGS_DEG
    }
    source_paths.update({
        f"local_sse_dpsi{heading:g}": _local_sse_path(heading)
        for heading in HEADING_SPACINGS_DEG
    })
    payload_without_hash = {
        "schema": PHASE7_ANGULAR_SCHEMA,
        "status": "complete",
        "condition_contract": {
            "spatial_resolution_m": DX_M,
            "heading_spacings_deg": list(HEADING_SPACINGS_DEG),
            "r_neighbor": R_NEIGHBOR,
            "terrain_index": TERRAIN_INDEX,
            "training_seed": 0,
            "training_episode_budget": 10_000,
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
    parser.add_argument("--run-local-sse", type=float)
    parser.add_argument("--aggregate", action="store_true")
    arguments = parser.parse_args()
    if arguments.run_local_sse is not None:
        print(run_local_sse(arguments.run_local_sse), flush=True)
    elif arguments.aggregate:
        print(aggregate(), flush=True)
    else:
        parser.error("select --run-local-sse DPSI or --aggregate")


if __name__ == "__main__":
    main()


__all__ = ["aggregate", "run_local_sse"]

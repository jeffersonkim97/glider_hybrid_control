"""Aggregate the four-terrain zero-sum spatial-discretization experiment."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import project_paths


SCHEMA = "p1b-zero-sum-spatial-sweep-v1"
SOURCE_SCHEMA = "p1b-zero-sum-g2-local-search-v1"
TERRAINS = (0, 1, 2, 3)
RESOLUTIONS_M = (100.0, 50.0, 25.0, 10.0)
DPSI_DEG = 5.0
R_NEIGHBOR = 1
SOURCE_ROOT = project_paths.DQN_DIR / "runs" / "phase6_g2_local_sse"
OUTPUT_DIR = (
    project_paths.DQN_DIR / "runs" / "zero_sum_spatial_sweep" / "closeout_v1"
)


def _tag(value: float) -> str:
    return f"{value:g}"


def _source_path(terrain: int, dx_m: float) -> Path:
    return (
        SOURCE_ROOT / f"final_id_{terrain:03d}"
        / f"dx{_tag(dx_m)}m_dpsi{_tag(DPSI_DEG)}deg_r{R_NEIGHBOR}"
        / "zero_sum_g2_local_search.json"
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _p90(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values), 90, method="higher"))


def _row(terrain: int, dx_m: float) -> dict[str, Any]:
    path = _source_path(terrain, dx_m)
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema") != SOURCE_SCHEMA or result.get("status") != "complete":
        raise RuntimeError(f"invalid zero-sum result in {path}")
    condition = result["condition"]
    actual = (
        int(condition["terrain_index"]),
        float(condition["spatial_resolution_m"]),
        float(condition["heading_spacing_deg"]),
        int(condition["r_neighbor"]),
        condition["leader_objective"],
    )
    expected = (terrain, dx_m, DPSI_DEG, R_NEIGHBOR, "attacker_objective")
    if actual != expected:
        raise RuntimeError(f"condition mismatch in {path}: {actual} != {expected}")

    exact = result["exact_bellman_local_sse"]
    dqn = result["frozen_dqn_local_sse"]
    comparison = result["comparison"]
    certificate = comparison["zero_sum_single_point_certificate"]
    if not certificate["one_sided_inequality_verified"]:
        raise RuntimeError(f"one-sided certificate failed in {path}")
    exact_action = int(exact["search"]["final_local_sse_action_id"])
    dqn_action = int(dqn["search"]["final_local_sse_action_id"])
    exact_final_u = float(exact["search"]["final_leader_value"])
    dqn_approx_u = float(dqn["search"]["final_leader_value"])
    exact_u_at_dqn = float(certificate["exact_value_U"])
    delta = float(certificate["certified_delta"])
    return {
        "terrain_index": terrain,
        "terrain_id": condition["terrain_id"],
        "dx_m": dx_m,
        "dpsi_deg": DPSI_DEG,
        "r_neighbor": R_NEIGHBOR,
        "state_count": int(result["shared_setup"]["scene_sizes"]["N_S_cart"]),
        "defender_action_count": int(condition["defender_action_count"]),
        "bellman_defender_evaluations": int(
            exact["search"]["unique_defender_evaluations"]
        ),
        "dqn_defender_evaluations": int(
            dqn["search"]["unique_defender_evaluations"]
        ),
        "bellman_first_sec": float(exact["wall_runtime_sec"]),
        "dqn_first_sec": float(dqn["first_query_wall_sec"]),
        "dqn_repeated_sec": float(dqn["repeated_query_wall_sec"]),
        "bellman_over_dqn_first_ratio": float(
            comparison["bellman_over_dqn_first_query_ratio"]
        ),
        "bellman_over_dqn_repeated_ratio": float(
            comparison["bellman_over_dqn_repeated_query_ratio"]
        ),
        "exact_search_final_U": exact_final_u,
        "dqn_search_approximate_U_tilde": dqn_approx_u,
        "exact_U_at_dqn_sensor": exact_u_at_dqn,
        "certificate_delta": delta,
        "certificate_delta_relative": delta / abs(exact_u_at_dqn),
        "exact_search_final_minus_exact_U_at_dqn": (
            exact_final_u - exact_u_at_dqn
        ),
        "selected_action_match": exact_action == dqn_action,
        "bellman_action_id": exact_action,
        "dqn_action_id": dqn_action,
        "selected_sensor_distance_m": float(
            comparison["selected_sensor_distance_m"]
        ),
        "one_sided_inequality_verified": True,
        "source_path": path.relative_to(project_paths.WORKSPACE_ROOT).as_posix(),
        "source_sha256": _sha256(path),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dx_m in RESOLUTIONS_M:
        group = [row for row in rows if row["dx_m"] == dx_m]
        if len(group) != len(TERRAINS):
            raise RuntimeError(f"incomplete terrain group at dx={dx_m:g}")

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in group]

        output.append({
            "dx_m": dx_m,
            "terrain_count": len(group),
            "bellman_first_median_sec": median(values("bellman_first_sec")),
            "bellman_first_p90_sec": _p90(values("bellman_first_sec")),
            "dqn_first_median_sec": median(values("dqn_first_sec")),
            "dqn_first_p90_sec": _p90(values("dqn_first_sec")),
            "dqn_repeated_median_sec": median(values("dqn_repeated_sec")),
            "dqn_repeated_p90_sec": _p90(values("dqn_repeated_sec")),
            "delta_median": median(values("certificate_delta")),
            "delta_p90": _p90(values("certificate_delta")),
            "relative_delta_median": median(
                values("certificate_delta_relative")
            ),
            "relative_delta_p90": _p90(values("certificate_delta_relative")),
            "action_match_count": sum(
                bool(row["selected_action_match"]) for row in group
            ),
            "one_sided_pass_count": sum(
                bool(row["one_sided_inequality_verified"]) for row in group
            ),
        })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(aggregate: list[dict[str, Any]]) -> list[Path]:
    runtime_path = OUTPUT_DIR / "zero_sum_spatial_runtime.png"
    certificate_path = OUTPUT_DIR / "zero_sum_spatial_certificate.png"
    x = np.arange(len(aggregate), dtype=float)
    labels = [f"{row['dx_m']:g}" for row in aggregate]

    figure, axis = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    for key, label, marker in (
        ("bellman_first_median_sec", "Bellman first", "o"),
        ("dqn_first_median_sec", "DQN first", "s"),
        ("dqn_repeated_median_sec", "DQN repeated", "^"),
    ):
        axis.plot(x, [row[key] for row in aggregate], marker=marker, label=label)
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_xlabel("Spatial discretization dx [m]")
    axis.set_ylabel("Median Local-SSE wall time [s] (log scale)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(runtime_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    axis.plot(
        x,
        [100.0 * row["relative_delta_median"] for row in aggregate],
        "o-", label="median",
    )
    axis.plot(
        x,
        [100.0 * row["relative_delta_p90"] for row in aggregate],
        "s--", label="P90 (higher)",
    )
    axis.set_xticks(x, labels)
    axis.set_xlabel("Spatial discretization dx [m]")
    axis.set_ylabel(r"Single-point certificate $\delta/|U|$ [%]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(certificate_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return [runtime_path, certificate_path]


def build() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        _row(terrain, dx_m)
        for dx_m in RESOLUTIONS_M
        for terrain in TERRAINS
    ]
    aggregate = _aggregate(rows)
    cases_csv = OUTPUT_DIR / "zero_sum_spatial_cases.csv"
    aggregate_csv = OUTPUT_DIR / "zero_sum_spatial_aggregate.csv"
    _write_csv(cases_csv, rows)
    _write_csv(aggregate_csv, aggregate)
    plots = _write_plots(aggregate)

    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "contract": {
            "terrain_indices": list(TERRAINS),
            "spatial_resolutions_m": list(RESOLUTIONS_M),
            "heading_spacing_deg": DPSI_DEG,
            "r_neighbor": R_NEIGHBOR,
            "leader_objective": "attacker_objective",
            "new_training_performed": False,
        },
        "cases": rows,
        "aggregate": aggregate,
        "artifacts": {
            "cases_csv": str(cases_csv.relative_to(project_paths.WORKSPACE_ROOT)),
            "aggregate_csv": str(
                aggregate_csv.relative_to(project_paths.WORKSPACE_ROOT)
            ),
            "plots": [
                str(path.relative_to(project_paths.WORKSPACE_ROOT)) for path in plots
            ],
        },
    }
    output = OUTPUT_DIR / "zero_sum_spatial_sweep.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Zero-Sum Multi-Terrain Spatial Sweep",
        "",
        "Frozen DQN checkpoints; no training or fine-tuning. Median and P90 are "
        "over final-ID terrains 000--003.",
        "",
        "| dx | Bellman first med / P90 | DQN first med / P90 | DQN repeated med / P90 | delta med / P90 | relative delta med / P90 | action match |",
        "|---:|:---|:---|:---|:---|:---|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['dx_m']:g} m | "
            f"{row['bellman_first_median_sec']:.2f} / {row['bellman_first_p90_sec']:.2f} s | "
            f"{row['dqn_first_median_sec']:.2f} / {row['dqn_first_p90_sec']:.2f} s | "
            f"{row['dqn_repeated_median_sec']:.3f} / {row['dqn_repeated_p90_sec']:.3f} s | "
            f"{row['delta_median']:.6f} / {row['delta_p90']:.6f} | "
            f"{100.0 * row['relative_delta_median']:.2f}% / "
            f"{100.0 * row['relative_delta_p90']:.2f}% | "
            f"{row['action_match_count']}/{row['terrain_count']} |"
        )
    lines.extend([
        "",
        "The delta certificate is local and instance-specific. Action mismatch "
        "and the exact-search outcome gap are diagnostics, not violations of "
        "the approximate local-solution theorem.",
    ])
    (OUTPUT_DIR / "zero_sum_spatial_sweep.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    print(output, flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if not args.aggregate:
        parser.error("pass --aggregate after all 16 zero-sum cases complete")
    build()


if __name__ == "__main__":
    main()


"""Run and aggregate zero-sum angular and neighborhood main-effect sweeps."""

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
from phase6_g2_local_sse import (
    LEADER_OBJECTIVE_ATTACKER_COST,
    run_phase6_pilot,
)


SCHEMA = "p1b-zero-sum-multi-axis-sweep-v1"
ANGULAR_SCHEMA = "p1b-zero-sum-angular-local-search-v1"
NEIGHBOR_SCHEMA = "p1b-zero-sum-neighbor-local-search-v1"
TERRAINS = (0, 1, 2, 3)
DX_M = 25.0
ANGULAR_VALUES = (10.0, 5.0, 2.5)
ANGULAR_R = 1
NEIGHBOR_DPSI = 5.0
NEIGHBOR_VALUES = (1, 2, 3, 4, 5)

RUN_ROOT = project_paths.DQN_DIR / "runs"
PHASE6_ROOT = RUN_ROOT / "phase6_g2_local_sse"
PHASE7_ANGULAR_ROOT = RUN_ROOT / "phase7_angular_sweep"
PHASE7_NEIGHBOR_ROOT = RUN_ROOT / "phase7_neighbor_sweep_dpsi5deg"
PHASE9_ROOT = RUN_ROOT / "phase9_multi_axis_generalization"
ANGULAR_OTHER_ROOT = PHASE9_ROOT / "angular"
NEIGHBOR_OTHER_ROOT = PHASE9_ROOT / "neighbor_dpsi5deg"
OUTPUT_DIR = RUN_ROOT / "zero_sum_multi_axis_sweep" / "closeout_v1"


def _tag(value: float) -> str:
    return f"{float(value):g}"


def _condition(dpsi: float, radius: int) -> str:
    return f"dx25m_dpsi{_tag(dpsi)}deg_r{int(radius)}"


def _angular_root(terrain: int) -> Path:
    return PHASE7_ANGULAR_ROOT if int(terrain) == 1 else ANGULAR_OTHER_ROOT


def _neighbor_root(terrain: int) -> Path:
    return PHASE7_NEIGHBOR_ROOT if int(terrain) == 1 else NEIGHBOR_OTHER_ROOT


def angular_path(terrain: int, dpsi: float) -> Path:
    condition = _condition(dpsi, ANGULAR_R)
    if float(dpsi) == 5.0:
        return (
            PHASE6_ROOT / f"final_id_{terrain:03d}" / condition
            / "zero_sum_g2_local_search.json"
        )
    return (
        _angular_root(terrain) / f"final_id_{terrain:03d}" / condition
        / "zero_sum_angular_local_search.json"
    )


def neighbor_path(terrain: int, radius: int) -> Path:
    condition = _condition(NEIGHBOR_DPSI, radius)
    if int(radius) == 1:
        return (
            PHASE6_ROOT / f"final_id_{terrain:03d}" / condition
            / "zero_sum_g2_local_search.json"
        )
    return (
        _neighbor_root(terrain) / f"final_id_{terrain:03d}" / condition
        / "zero_sum_neighbor_local_search.json"
    )


def run_angular(terrain: int, dpsi: float) -> Path:
    terrain = int(terrain)
    dpsi = float(dpsi)
    if terrain not in TERRAINS or dpsi not in (10.0, 2.5):
        raise ValueError("angular run requires terrain 0..3 and dpsi 10 or 2.5")
    return run_phase6_pilot(
        terrain_index=terrain,
        spatial_resolution_m=DX_M,
        heading_spacing_deg=dpsi,
        r_neighbor=ANGULAR_R,
        device_name="cuda",
        output_root=_angular_root(terrain),
        result_schema=ANGULAR_SCHEMA,
        experiment_label="Zero-sum angular main effect",
        output_stem="zero_sum_angular_local_search",
        leader_objective=LEADER_OBJECTIVE_ATTACKER_COST,
    )


def run_neighbor(terrain: int, radius: int) -> Path:
    terrain = int(terrain)
    radius = int(radius)
    if terrain not in TERRAINS or radius not in (2, 3, 4, 5):
        raise ValueError("neighbor run requires terrain 0..3 and r 2, 3, 4, or 5")
    return run_phase6_pilot(
        terrain_index=terrain,
        spatial_resolution_m=DX_M,
        heading_spacing_deg=NEIGHBOR_DPSI,
        r_neighbor=radius,
        device_name="cuda",
        output_root=_neighbor_root(terrain),
        result_schema=NEIGHBOR_SCHEMA,
        experiment_label="Zero-sum neighborhood main effect",
        output_stem="zero_sum_neighbor_local_search",
        leader_objective=LEADER_OBJECTIVE_ATTACKER_COST,
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _p90(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values), 90, method="higher"))


def _row(
    *, axis: str, value: float, terrain: int, path: Path,
    dpsi: float, radius: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise RuntimeError(f"incomplete result: {path}")
    condition = result["condition"]
    actual = (
        int(condition["terrain_index"]),
        float(condition["spatial_resolution_m"]),
        float(condition["heading_spacing_deg"]),
        int(condition["r_neighbor"]),
        condition["leader_objective"],
    )
    expected = (terrain, DX_M, dpsi, radius, "attacker_objective")
    if actual != expected:
        raise RuntimeError(f"condition mismatch in {path}: {actual} != {expected}")
    exact = result["exact_bellman_local_sse"]
    dqn = result["frozen_dqn_local_sse"]
    comparison = result["comparison"]
    certificate = comparison["zero_sum_single_point_certificate"]
    if not certificate["one_sided_inequality_verified"]:
        raise RuntimeError(f"one-sided certificate failed: {path}")
    exact_action = int(exact["search"]["final_local_sse_action_id"])
    dqn_action = int(dqn["search"]["final_local_sse_action_id"])
    exact_u_at_dqn = float(certificate["exact_value_U"])
    delta = float(certificate["certified_delta"])
    return {
        "axis": axis,
        "axis_value": value,
        "terrain_index": terrain,
        "dx_m": DX_M,
        "dpsi_deg": dpsi,
        "r_neighbor": radius,
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
        "exact_search_final_U": float(exact["search"]["final_leader_value"]),
        "dqn_search_approximate_U_tilde": float(
            dqn["search"]["final_leader_value"]
        ),
        "exact_U_at_dqn_sensor": exact_u_at_dqn,
        "certificate_delta": delta,
        "certificate_delta_relative": delta / abs(exact_u_at_dqn),
        "selected_action_match": exact_action == dqn_action,
        "bellman_action_id": exact_action,
        "dqn_action_id": dqn_action,
        "source_path": path.relative_to(project_paths.WORKSPACE_ROOT).as_posix(),
        "source_sha256": _sha256(path),
    }


def _aggregate(rows: list[dict[str, Any]], axis: str, values: tuple[float, ...]) -> list[dict[str, Any]]:
    output = []
    for value in values:
        group = [
            row for row in rows
            if row["axis"] == axis and float(row["axis_value"]) == float(value)
        ]
        if len(group) != len(TERRAINS):
            raise RuntimeError(f"incomplete {axis} group at {value:g}")

        def numbers(key: str) -> list[float]:
            return [float(row[key]) for row in group]

        output.append({
            "axis": axis,
            "axis_value": value,
            "terrain_count": len(group),
            "bellman_first_median_sec": median(numbers("bellman_first_sec")),
            "bellman_first_p90_sec": _p90(numbers("bellman_first_sec")),
            "dqn_first_median_sec": median(numbers("dqn_first_sec")),
            "dqn_first_p90_sec": _p90(numbers("dqn_first_sec")),
            "dqn_repeated_median_sec": median(numbers("dqn_repeated_sec")),
            "dqn_repeated_p90_sec": _p90(numbers("dqn_repeated_sec")),
            "delta_median": median(numbers("certificate_delta")),
            "delta_p90": _p90(numbers("certificate_delta")),
            "relative_delta_median": median(numbers("certificate_delta_relative")),
            "relative_delta_p90": _p90(numbers("certificate_delta_relative")),
            "action_match_count": sum(
                bool(row["selected_action_match"]) for row in group
            ),
        })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(aggregate: list[dict[str, Any]], axis: str, values: tuple[float, ...], label: str, stem: str) -> list[Path]:
    subset = [row for row in aggregate if row["axis"] == axis]
    runtime = OUTPUT_DIR / f"{stem}_runtime.png"
    quality = OUTPUT_DIR / f"{stem}_certificate.png"
    x = np.arange(len(values), dtype=float)
    figure, plot = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    for key, name, marker in (
        ("bellman_first_median_sec", "Bellman first", "o"),
        ("dqn_first_median_sec", "DQN first", "s"),
        ("dqn_repeated_median_sec", "DQN repeated", "^"),
    ):
        plot.plot(x, [row[key] for row in subset], marker=marker, label=name)
    plot.set_yscale("log")
    plot.set_xticks(x, [f"{value:g}" for value in values])
    plot.set_xlabel(label)
    plot.set_ylabel("Median Local-SSE time [s] (log scale)")
    plot.grid(True, which="both", alpha=0.3)
    plot.legend()
    figure.savefig(runtime, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, plot = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    plot.plot(x, [100 * row["relative_delta_median"] for row in subset], "o-", label="median")
    plot.plot(x, [100 * row["relative_delta_p90"] for row in subset], "s--", label="P90 (higher)")
    plot.set_xticks(x, [f"{value:g}" for value in values])
    plot.set_xlabel(label)
    plot.set_ylabel(r"Certificate $\delta/|U|$ [%]")
    plot.grid(True, alpha=0.3)
    plot.legend()
    figure.savefig(quality, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return [runtime, quality]


def aggregate() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    angular = [
        _row(axis="dpsi_deg", value=dpsi, terrain=terrain,
             path=angular_path(terrain, dpsi), dpsi=dpsi, radius=ANGULAR_R)
        for dpsi in ANGULAR_VALUES for terrain in TERRAINS
    ]
    neighbor = [
        _row(axis="r_neighbor", value=float(radius), terrain=terrain,
             path=neighbor_path(terrain, radius), dpsi=NEIGHBOR_DPSI, radius=radius)
        for radius in NEIGHBOR_VALUES for terrain in TERRAINS
    ]
    rows = angular + neighbor
    summary = (
        _aggregate(rows, "dpsi_deg", ANGULAR_VALUES)
        + _aggregate(rows, "r_neighbor", tuple(float(v) for v in NEIGHBOR_VALUES))
    )
    cases_csv = OUTPUT_DIR / "zero_sum_multi_axis_cases.csv"
    aggregate_csv = OUTPUT_DIR / "zero_sum_multi_axis_aggregate.csv"
    _write_csv(cases_csv, rows)
    _write_csv(aggregate_csv, summary)
    plots = (
        _plots(summary, "dpsi_deg", ANGULAR_VALUES, "Heading spacing dpsi [deg]", "angular")
        + _plots(summary, "r_neighbor", tuple(float(v) for v in NEIGHBOR_VALUES), "Chebyshev radius r", "neighbor")
    )
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "contract": {
            "terrain_indices": list(TERRAINS), "dx_m": DX_M,
            "angular_dpsi_deg": list(ANGULAR_VALUES), "angular_r": ANGULAR_R,
            "neighbor_dpsi_deg": NEIGHBOR_DPSI,
            "neighbor_radii": list(NEIGHBOR_VALUES),
            "leader_objective": "attacker_objective",
            "new_training_performed": False, "full_factorial": False,
        },
        "cases": rows, "aggregate": summary,
        "plots": [str(path.relative_to(project_paths.WORKSPACE_ROOT)) for path in plots],
    }
    output = OUTPUT_DIR / "zero_sum_multi_axis_sweep.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Zero-Sum Angular and Neighborhood Main Effects", "",
        "Four final-ID terrains; frozen checkpoints; no training or fine-tuning.", "",
        "| Axis | Value | Bellman first med/P90 | DQN first med/P90 | DQN repeated med/P90 | Relative delta med/P90 | Action match |",
        "|:---|---:|:---|:---|:---|:---|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['axis']} | {row['axis_value']:g} | "
            f"{row['bellman_first_median_sec']:.2f}/{row['bellman_first_p90_sec']:.2f} s | "
            f"{row['dqn_first_median_sec']:.2f}/{row['dqn_first_p90_sec']:.2f} s | "
            f"{row['dqn_repeated_median_sec']:.3f}/{row['dqn_repeated_p90_sec']:.3f} s | "
            f"{100*row['relative_delta_median']:.2f}%/{100*row['relative_delta_p90']:.2f}% | "
            f"{row['action_match_count']}/{row['terrain_count']} |"
        )
    (OUTPUT_DIR / "zero_sum_multi_axis_sweep.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    print(output, flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-angular", action="store_true")
    group.add_argument("--run-neighbor", action="store_true")
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--terrain-index", type=int)
    parser.add_argument("--heading-spacing-deg", type=float)
    parser.add_argument("--r-neighbor", type=int)
    args = parser.parse_args()
    if args.run_angular:
        if args.terrain_index is None or args.heading_spacing_deg is None:
            parser.error("angular run needs terrain and heading spacing")
        print(run_angular(args.terrain_index, args.heading_spacing_deg), flush=True)
    elif args.run_neighbor:
        if args.terrain_index is None or args.r_neighbor is None:
            parser.error("neighbor run needs terrain and radius")
        print(run_neighbor(args.terrain_index, args.r_neighbor), flush=True)
    else:
        aggregate()


if __name__ == "__main__":
    main()


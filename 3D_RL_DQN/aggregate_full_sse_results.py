"""Aggregate per-terrain Bellman/DQN full Local-SSE comparison outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TERRAINS = (
    "centered_cube",
    "centered_cube_half_height",
    "offset_cube_left",
    "offset_cube_right",
    "stepped_pyramid",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-sse-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    output_dir = args.full_sse_dir.resolve()

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for terrain in TERRAINS:
        path = output_dir / f"full_sse_{terrain}_comparison.json"
        if not path.is_file():
            missing.append(terrain)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        exact = payload["exact_bellman"]
        dqn = payload["trained_dqn"]
        comparison = payload["comparison"]
        rows.append({
            "terrain": terrain,
            "bellman_J_A": exact["J_A"],
            "bellman_J_D": exact["J_D"],
            "bellman_wall_sec": exact["wall_runtime_sec"],
            "bellman_defender_evaluations": exact["unique_defender_evaluations"],
            "dqn_J_A": dqn["predicted_J_A"],
            "dqn_J_D": dqn["predicted_J_D"],
            "dqn_wall_sec": dqn["wall_runtime_sec"],
            "dqn_defender_evaluations": dqn["unique_defender_evaluations"],
            "bellman_over_dqn_speedup": comparison[
                "full_sse_runtime_speedup_bellman_over_dqn"
            ],
            "selected_defender_action_match": comparison[
                "selected_defender_action_match"
            ],
            "bellman_sensor": exact["selected_sensor_map"],
            "dqn_sensor": dqn["selected_sensor_map"],
            "dqn_selected_sensor_exact_local_sse_certified": comparison[
                "dqn_selected_sensor_exact_local_sse_certified"
            ],
        })

    if missing and not args.allow_partial:
        raise FileNotFoundError(f"missing full-SSE results: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": "10 m Bellman versus trained-DQN full Local-SSE by terrain",
        "scope": "full Local-SSE; Chebyshev r=1",
        "completed_terrains": [row["terrain"] for row in rows],
        "missing_terrains": missing,
        "complete": not missing,
        "rows": rows,
    }
    (output_dir / "full_sse_all_terrains_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = output_dir / "full_sse_all_terrains_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# 10 m full Local-SSE comparison by terrain",
        "",
        "Times include the complete Local-SSE search for each backend and exclude the "
        "shared scene build and DQN checkpoint load, matching the per-terrain benchmark.",
        "",
        "| Terrain | Bellman J_A | Bellman J_D | Bellman time (s) | DQN J_A | DQN J_D | DQN time (s) | Speedup | Action match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(
        f"| `{row['terrain']}` | {row['bellman_J_A']:.8f} | "
        f"{row['bellman_J_D']:.8f} | {row['bellman_wall_sec']:.3f} | "
        f"{row['dqn_J_A']:.8f} | {row['dqn_J_D']:.8f} | "
        f"{row['dqn_wall_sec']:.3f} | {row['bellman_over_dqn_speedup']:.3f}x | "
        f"{row['selected_defender_action_match']} |"
        for row in rows
    )
    if missing:
        lines.extend(["", "Pending terrains: " + ", ".join(f"`{x}`" for x in missing)])
    lines.append("")
    (output_dir / "full_sse_all_terrains_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

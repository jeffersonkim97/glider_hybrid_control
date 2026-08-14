"""Assemble the resumable staged Defender search into one audited summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .experiment_staged_defender_optimization import OUTPUT_DIR


HOMOTOPY_DIR = (
    OUTPUT_DIR.parent
    / "defender_sensor_continuation_x2600_y0"
    / "detection_homotopy"
)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    summary_path = OUTPUT_DIR / "optimization_summary.json"
    payload = _load(summary_path)
    medium_directories = (
        OUTPUT_DIR / "medium_2_x1200_y0",
        OUTPUT_DIR / "medium_3_x1000_y-400",
        OUTPUT_DIR / "medium_replacement_x2600_y0",
    )
    payload["medium_records"] = [
        _load(directory / "summary.json") for directory in medium_directories
    ]
    prior_verified_directory = OUTPUT_DIR / "medium_3_x1000_y-400"
    prior_verified = _load(prior_verified_directory / "continuous_summary.json")
    prior_verified["source_directory"] = str(prior_verified_directory)
    homotopy = _load(HOMOTOPY_DIR / "homotopy_summary.json")
    selected_continuous = dict(homotopy["selected_record"])
    selected_continuous.update({
        "source_directory": str(HOMOTOPY_DIR / "scale_1.00"),
        "continuation_method": (
            "sensor-position continuation followed by detection-hazard "
            "homotopy with limited-memory BFGS"
        ),
        "symmetry_equivalent_topology_count": 2,
    })
    payload["continuous_medium_records"] = [prior_verified, selected_continuous]
    payload["continuous_candidate_audits"] = {
        "x1200_y0": _load(
            OUTPUT_DIR / "medium_2_x1200_y0" / "continuous_attempts.json"
        ),
        "x1000_y-400": _load(
            prior_verified_directory / "continuous_attempts.json"
        ),
        "x2600_y0_direct": _load(
            OUTPUT_DIR / "medium_replacement_x2600_y0" / "continuous_attempts.json"
        ),
        "x2600_y0_homotopy": homotopy,
    }
    fine_directory = OUTPUT_DIR / "fine_selected_x2600_y0"
    payload["fine_record"] = _load(fine_directory / "summary.json")
    payload["continuous_fine_record"] = selected_continuous
    payload["selected_fine_directory"] = str(fine_directory)

    # Materialize the selected homotopy solution beside the fine result so
    # plotting and downstream reporting have one canonical result directory.
    scale_directory = HOMOTOPY_DIR / "scale_1.00"
    with np.load(scale_directory / "trajectory_data.npz") as handle:
        dense_time = np.asarray(handle["dense_time"])
        dense_states = np.asarray(handle["dense_states"])
        controls = np.asarray(handle["controls"])
        switch_state = np.asarray(handle["switch_state"])
    with np.load(fine_directory / "trajectory_data.npz") as handle:
        launch = np.asarray(handle["powered_path"][0])
    powered_path = launch[None, :] + np.linspace(0.0, 1.0, 301)[:, None] * (
        switch_state[:3] - launch
    )[None, :]
    with (fine_directory / "continuous_summary.json").open(
        "w", encoding="utf-8",
    ) as handle:
        json.dump(selected_continuous, handle, indent=2)
    np.savez_compressed(
        fine_directory / "continuous_trajectory.npz",
        dense_time=dense_time,
        dense_states=dense_states,
        powered_path=powered_path,
        controls=controls,
    )
    payload["status_success"] = bool(
        payload["continuous_fine_record"]["status_success"]
    )
    payload["completion_status"] = "completed_strictly_validated_budgeted_search"
    payload["selected_sensor_xy"] = [2600.0, 0.0]
    payload["selected_medium_continuous_record"] = selected_continuous
    payload["selection_basis"] = (
        "highest Defender objective among candidates passing exact medium, "
        "continuous 3-DOF dense validation, exact fine, and repeated "
        "continuous dense validation"
    )
    payload["global_optimality_claimed"] = False
    payload["best_unverified_candidate"] = None
    payload["prior_strictly_validated_candidate"] = prior_verified
    payload["homotopy_intermediate_scales_are_physical_results"] = False
    payload["homotopy_final_scale_uses_original_detection_model"] = True
    payload["projection_6d_to_3d_modified"] = False
    payload["projection_used"] = False
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({
        "status_success": payload["status_success"],
        "completion_status": payload["completion_status"],
        "selected_sensor_xy": payload["selected_sensor_xy"],
        "fine_defender_objective": payload["continuous_fine_record"][
            "defender_objective"
        ],
        "fine_mission_pod": payload["continuous_fine_record"]["mission_pod"],
        "global_optimality_claimed": payload["global_optimality_claimed"],
    }, indent=2))


if __name__ == "__main__":
    main()

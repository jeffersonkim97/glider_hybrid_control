"""Detection-weight homotopy for the difficult (2600, 0) sensor response."""

from __future__ import annotations

import json

from .continuous_trajectory_refinement import _dense_validate, solve_continuous_refinement
from .experiment_defender_sensor_continuation import (
    OUTPUT_DIR as CONTINUATION_DIR,
    _load_stage,
    _record,
    _save_stage,
)


TARGET_SENSOR_XY = (2600.0, 0.0)
SCALES = (
    0.25, 0.35, 0.45, 0.48, 0.51, 0.54, 0.57, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
)
OUTPUT_DIR = CONTINUATION_DIR / "detection_homotopy"
START_DIR = CONTINUATION_DIR / "stage_4_x2412.5_y67.5"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, previous_result = _load_stage(START_DIR)
    records = []
    for index, scale in enumerate(SCALES, start=1):
        stage_dir = OUTPUT_DIR / f"scale_{scale:.2f}"
        if (stage_dir / "summary.json").exists():
            existing_record, existing_result = _load_stage(stage_dir)
            if existing_record["status_success"]:
                print(f"SKIP validated hazard scale={scale:.2f}", flush=True)
                records.append(existing_record)
                previous_result = existing_result
                continue
        successful = False
        for attempt in range(1, 3):
            print(
                f"START hazard homotopy scale={scale:.2f} attempt={attempt}",
                flush=True,
            )
            result = solve_continuous_refinement(
                interval_count=50,
                initial_topology="south",
                initialization_source="result_mapping",
                initial_result=previous_result,
                sensor_xy=TARGET_SENSOR_XY,
                maximum_cpu_time_s=120.0,
                maximum_iterations=7000,
                accept_limited_solution=True,
                nlp_speed_buffer_m_s=0.02,
                detection_hazard_scale=scale,
                use_limited_memory_hessian=True,
            )
            validation = _dense_validate(result)
            record = _record(result, validation, TARGET_SENSOR_XY, index)
            record["detection_hazard_scale"] = scale
            record["physical_result"] = bool(scale == 1.0)
            attempt_dir = OUTPUT_DIR / f"scale_{scale:.2f}_attempt_{attempt}"
            _save_stage(attempt_dir, record, result, validation)
            print(
                f"DONE scale={scale:.2f} valid={record['status_success']} "
                f"hazard={record['mission_hazard']:.6f} "
                f"solver={record['solver_return_status']}",
                flush=True,
            )
            previous_result = result
            if validation["passed"]:
                records.append(record)
                _save_stage(stage_dir, record, result, validation)
                successful = True
                break
        if not successful:
            raise RuntimeError(
                f"Detection homotopy failed at scale={scale:.2f}"
            )
    payload = {
        "status_success": True,
        "target_sensor_xy": list(TARGET_SENSOR_XY),
        "records": records,
        "selected_record": records[-1],
        "intermediate_scales_are_physical_results": False,
        "final_scale_is_original_detection_model": True,
        "global_optimality_claimed": False,
        "projection_6d_to_3d_modified": False,
        "projection_used": False,
    }
    with (OUTPUT_DIR / "homotopy_summary.json").open(
        "w", encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["selected_record"], indent=2), flush=True)


if __name__ == "__main__":
    main()

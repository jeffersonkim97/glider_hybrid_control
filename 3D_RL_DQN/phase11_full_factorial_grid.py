"""Complete the 4 x 3 x 5 zero-sum grid over four unseen terrains.

The runner is resumable.  It evaluates only the 40 conditions not already
present in the Phase-9/10 main-effect and interaction experiments.  Exact
attacker best responses are independent of the local-search radius, so a
guarded shared cache is reused across r values with the same
(terrain, dx, dpsi) physical contract.  Reported benchmark runtimes remain
the sum of the original per-response runtimes, not the shortened resume wall
time.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any

import project_paths
from phase4_g2_training import (
    CHECKPOINT_ROOT,
    _condition_tag as _training_condition_tag,
    run_g2,
)
from phase6_g2_local_sse import (
    EXACT_CACHE_SCHEMA,
    run_zero_sum_local_sse,
)


SCHEMA = "p1b-phase11-full-factorial-grid-v1"
DX_VALUES = (10.0, 25.0, 50.0, 100.0)
DPSI_VALUES = (2.5, 5.0, 10.0)
R_VALUES = (1, 2, 3, 4, 5)
TERRAINS = (0, 1, 2, 3)

# Conditions already represented by four-terrain aggregate results.
EXISTING_CONDITIONS = frozenset(
    {(dx, 5.0, 1) for dx in DX_VALUES}
    | {(25.0, dpsi, 1) for dpsi in DPSI_VALUES}
    | {(25.0, 5.0, radius) for radius in R_VALUES}
    | {
        (dx, dpsi, radius)
        for dx in (50.0, 100.0)
        for dpsi in DPSI_VALUES
        for radius in (1, 3)
    }
)
ALL_CONDITIONS = frozenset(
    (dx, dpsi, radius)
    for dx in DX_VALUES
    for dpsi in DPSI_VALUES
    for radius in R_VALUES
)
MISSING_CONDITIONS = ALL_CONDITIONS - EXISTING_CONDITIONS

# Coarse conditions finish first; within a physical condition the largest
# radius runs first to populate most of the reusable response cache.
DX_ORDER = (100.0, 50.0, 25.0, 10.0)
DPSI_ORDER = (10.0, 5.0, 2.5)
R_ORDER = (5, 4, 3, 2, 1)

OUTPUT_ROOT = project_paths.DQN_DIR / "runs" / "phase11_full_factorial_grid"
SHARED_CACHE_ROOT = OUTPUT_ROOT / "_shared_exact_cache"
STATUS_PATH = OUTPUT_ROOT / "phase11_full_factorial_status.json"
STDOUT_PATH = OUTPUT_ROOT / "phase11_full_factorial_stdout.log"
STDERR_PATH = OUTPUT_ROOT / "phase11_full_factorial_stderr.log"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tag(value: float) -> str:
    return f"{float(value):g}"


def _condition_tag(dx: float, dpsi: float, radius: int) -> str:
    return f"dx{_tag(dx)}m_dpsi{_tag(dpsi)}deg_r{int(radius)}"


def _checkpoint_path(dx: float, dpsi: float) -> Path:
    return (
        CHECKPOINT_ROOT
        / _training_condition_tag(dx, dpsi)
        / "seed0_ep10000_v1"
        / "phase4_seed0_final.pt"
    )


def _result_path(
    terrain: int, dx: float, dpsi: float, radius: int,
) -> Path:
    return (
        OUTPUT_ROOT
        / f"final_id_{terrain:03d}"
        / _condition_tag(dx, dpsi, radius)
        / "zero_sum_g2_local_search.json"
    )


def _condition_cache_path(
    terrain: int, dx: float, dpsi: float, radius: int,
) -> Path:
    return _result_path(terrain, dx, dpsi, radius).with_name(
        "exact_response_cache.json"
    )


def _shared_cache_path(terrain: int, dx: float, dpsi: float) -> Path:
    return (
        SHARED_CACHE_ROOT
        / f"final_id_{terrain:03d}"
        / f"dx{_tag(dx)}m_dpsi{_tag(dpsi)}deg"
        / "exact_response_cache.json"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_status() -> dict[str, Any]:
    if STATUS_PATH.is_file():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {
        "schema": SCHEMA,
        "status": "not_started",
        "created_at_utc": _utc_now(),
        "completed_runs": [],
    }


def _update_status(**updates: Any) -> None:
    status = _load_status()
    if updates.get("status") != "failed":
        status.pop("error", None)
        status.pop("traceback", None)
    status.update(updates)
    status["updated_at_utc"] = _utc_now()
    _write_json(STATUS_PATH, status)


def _plan() -> dict[str, Any]:
    ordered = [
        (dx, dpsi, radius)
        for dx in DX_ORDER
        for dpsi in DPSI_ORDER
        for radius in R_ORDER
        if (dx, dpsi, radius) in MISSING_CONDITIONS
    ]
    evaluations = [
        {
            "terrain_index": terrain,
            "dx_m": dx,
            "dpsi_deg": dpsi,
            "r_neighbor": radius,
            "exists": _result_path(
                terrain, dx, dpsi, radius
            ).is_file(),
        }
        for dx, dpsi, radius in ordered
        for terrain in TERRAINS
    ]
    checkpoints = [
        {
            "dx_m": dx,
            "dpsi_deg": dpsi,
            "path": _checkpoint_path(dx, dpsi).relative_to(
                project_paths.WORKSPACE_ROOT
            ).as_posix(),
            "exists": _checkpoint_path(dx, dpsi).is_file(),
        }
        for dx in DX_VALUES
        for dpsi in DPSI_VALUES
    ]
    return {
        "schema": SCHEMA,
        "target_grid": {
            "dx_m": list(DX_VALUES),
            "dpsi_deg": list(DPSI_VALUES),
            "r_neighbor": list(R_VALUES),
            "terrain_indices": list(TERRAINS),
        },
        "full_condition_count": len(ALL_CONDITIONS),
        "preexisting_condition_count": len(EXISTING_CONDITIONS),
        "missing_condition_count": len(MISSING_CONDITIONS),
        "pending_evaluation_count": sum(
            not item["exists"] for item in evaluations
        ),
        "missing_checkpoint_count": sum(
            not item["exists"] for item in checkpoints
        ),
        "checkpoints": checkpoints,
        "evaluations": evaluations,
    }


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != EXACT_CACHE_SCHEMA:
        raise RuntimeError(f"unexpected exact cache schema: {path}")
    if not isinstance(payload.get("contract"), dict):
        raise RuntimeError(f"exact cache lacks contract: {path}")
    if not isinstance(payload.get("responses"), dict):
        raise RuntimeError(f"exact cache lacks responses: {path}")
    return payload


def _physical_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in contract.items()
        if key not in {"r_neighbor", "result_schema"}
    }


def _merge_cache_payloads(
    base: dict[str, Any] | None,
    addition: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if base is None:
        return addition
    if addition is None:
        return base
    if _physical_contract(base["contract"]) != _physical_contract(
        addition["contract"]
    ):
        raise RuntimeError("refusing to merge different exact-response contracts")
    responses = dict(base["responses"])
    for action_id, response in addition["responses"].items():
        responses.setdefault(action_id, response)
    merged = dict(base)
    merged["responses"] = responses
    merged["completed_action_count"] = len(responses)
    merged["last_completed_action_id"] = max(
        (int(action_id) for action_id in responses), default=-1
    )
    merged["status"] = "in_progress"
    return merged


def _seed_condition_cache(
    terrain: int, dx: float, dpsi: float, radius: int,
) -> None:
    shared_path = _shared_cache_path(terrain, dx, dpsi)
    target_path = _condition_cache_path(terrain, dx, dpsi, radius)
    shared = _read_cache(shared_path)
    target = _read_cache(target_path)
    merged = _merge_cache_payloads(target, shared)
    if merged is None:
        return
    merged = dict(merged)
    merged["contract"] = dict(merged["contract"])
    merged["contract"]["r_neighbor"] = int(radius)
    merged["contract"].pop("result_schema", None)
    _write_json(target_path, merged)


def _harvest_condition_cache(
    terrain: int, dx: float, dpsi: float, radius: int,
) -> None:
    shared_path = _shared_cache_path(terrain, dx, dpsi)
    target_path = _condition_cache_path(terrain, dx, dpsi, radius)
    merged = _merge_cache_payloads(
        _read_cache(shared_path), _read_cache(target_path)
    )
    if merged is None:
        raise FileNotFoundError(target_path)
    _write_json(shared_path, merged)


def _ensure_checkpoint(dx: float, dpsi: float) -> None:
    checkpoint = _checkpoint_path(dx, dpsi)
    if checkpoint.is_file():
        return
    _update_status(
        status="training",
        current={"dx_m": dx, "dpsi_deg": dpsi, "episodes": 10_000},
    )
    print(f"Training missing checkpoint dx={dx:g}, dpsi={dpsi:g}", flush=True)
    run_g2(
        episodes=10_000,
        evaluation_interval_episodes=1_000,
        spatial_resolution_m=dx,
        heading_spacing_deg=dpsi,
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"training completed without final checkpoint: {checkpoint}"
        )


def run() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plan = _plan()
    completed = list(_load_status().get("completed_runs", []))
    _update_status(
        status="running",
        started_at_utc=_load_status().get("started_at_utc", _utc_now()),
        plan=plan,
        completed_runs=completed,
    )
    try:
        for dx in DX_ORDER:
            required_dpsi = {
                dpsi
                for condition_dx, dpsi, _ in MISSING_CONDITIONS
                if condition_dx == dx
            }
            for dpsi in DPSI_ORDER:
                if dpsi in required_dpsi:
                    _ensure_checkpoint(dx, dpsi)

            for dpsi in DPSI_ORDER:
                radii = [
                    radius for radius in R_ORDER
                    if (dx, dpsi, radius) in MISSING_CONDITIONS
                ]
                if not radii:
                    continue
                for terrain in TERRAINS:
                    for radius in radii:
                        result_path = _result_path(
                            terrain, dx, dpsi, radius
                        )
                        run_id = (
                            f"terrain{terrain:03d}_dx{dx:g}_"
                            f"dpsi{dpsi:g}_r{radius}"
                        )
                        if result_path.is_file():
                            print(f"Result exists: {result_path}", flush=True)
                            _harvest_condition_cache(
                                terrain, dx, dpsi, radius
                            )
                            if run_id not in completed:
                                completed.append(run_id)
                            continue

                        _seed_condition_cache(
                            terrain, dx, dpsi, radius
                        )
                        _update_status(
                            status="evaluating",
                            current={
                                "run_id": run_id,
                                "terrain_index": terrain,
                                "dx_m": dx,
                                "dpsi_deg": dpsi,
                                "r_neighbor": radius,
                            },
                            completed_runs=completed,
                        )
                        print(f"Running {run_id}", flush=True)
                        produced = run_zero_sum_local_sse(
                            terrain_index=terrain,
                            spatial_resolution_m=dx,
                            heading_spacing_deg=dpsi,
                            r_neighbor=radius,
                            device_name="cuda",
                            output_root=OUTPUT_ROOT,
                        )
                        if produced != result_path or not result_path.is_file():
                            raise RuntimeError(
                                "unexpected result path: "
                                f"produced={produced}, expected={result_path}"
                            )
                        _harvest_condition_cache(
                            terrain, dx, dpsi, radius
                        )
                        completed.append(run_id)
                        _update_status(
                            status="evaluating",
                            current=None,
                            completed_runs=completed,
                        )

        _update_status(
            status="completed",
            current=None,
            completed_runs=completed,
            completed_at_utc=_utc_now(),
            plan=_plan(),
        )
        print("Completed full 60-condition grid.", flush=True)
        return STATUS_PATH
    except Exception as error:
        _update_status(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            completed_runs=completed,
        )
        raise


def main() -> None:
    global DX_ORDER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--coarse-only",
        action="store_true",
        help="evaluate only dx=100, 50, and 25; delegate dx=10 remotely",
    )
    arguments = parser.parse_args()
    if arguments.coarse_only:
        DX_ORDER = (100.0, 50.0, 25.0)
    if arguments.plan:
        print(json.dumps(_plan(), indent=2))
    else:
        print(run())


if __name__ == "__main__":
    main()

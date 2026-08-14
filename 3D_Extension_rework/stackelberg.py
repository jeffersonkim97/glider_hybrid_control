"""Terrain-following 2D Defender search around the 3D Attacker follower.

For every horizontal sensor decision ``(x_s, y_s)``, sensor altitude is
derived from terrain and the complete Stage 1--7 follower pipeline is solved
fresh.  The Defender objective is unchanged from p1b_4D; normalized LOS area
is replaced only by its direct 3D counterpart, normalized LOS volume.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .bellman import build_cost_to_go_bundle
from .continuous_replay import replay_glide_continuous_3d
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry, terrain_height
from .stage_cost import construct_stage_cost_6d
from .switching import select_switching_point
from .trajectory import extract_optimal_trajectory


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = ROOT / "results" / "stage_8_stackelberg" / "evaluation_cache"


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return sha256(("3d-stackelberg-v1|" + payload).encode("utf-8")).hexdigest()[:16]


def configuration_for_sensor(
    base_configuration: dict[str, Any], sensor_xy: tuple[float, float],
) -> dict[str, Any]:
    configuration = deepcopy(base_configuration)
    bounds = configuration["defender_search"]
    x, y = (float(sensor_xy[0]), float(sensor_xy[1]))
    if not (
        bounds["x_bounds_m"][0] <= x <= bounds["x_bounds_m"][1]
        and bounds["y_bounds_m"][0] <= y <= bounds["y_bounds_m"][1]
    ):
        raise ValueError("sensor position lies outside defender search bounds")
    configuration["environment"]["sensor_xy_m"] = (x, y)
    return configuration


def evaluate_defender_position(
    sensor_xy: tuple[float, float],
    base_configuration: dict[str, Any],
    *,
    retain_full_pipeline: bool = True,
) -> dict[str, Any]:
    """Run one fresh nested Attacker solve and evaluate the Defender."""
    configuration = configuration_for_sensor(base_configuration, sensor_xy)
    geometry = build_geometry(configuration)
    detection = build_symbolic_detection_bundle(configuration, geometry)
    stage = construct_stage_cost_6d(configuration, geometry, detection)
    cost_to_go = build_cost_to_go_bundle(
        configuration, geometry, detection, stage,
    )
    switching = select_switching_point(configuration, geometry, cost_to_go)
    trajectory = extract_optimal_trajectory(
        configuration, geometry, cost_to_go, switching,
    )
    replay = replay_glide_continuous_3d(
        configuration, geometry, detection, trajectory,
    )
    if not replay["status"]["success"]:
        raise RuntimeError(replay["status"]["message"])

    glide_hazard = float(np.sum(trajectory["hazard_profile"]))
    powered_hazard = float(trajectory["mission"]["hazard"] - glide_hazard)
    coverage = geometry["coverage"]
    outputs = detection["functions"]["defender_objective"](
        powered_hazard, glide_hazard, coverage["normalized_los_volume"],
    )
    values = outputs if isinstance(outputs, tuple) else (outputs,)
    pod_normalized, coverage_normalized, defender_objective = (
        float(value) for value in values
    )
    sensor = np.asarray(geometry["sensor_position"], dtype=float)
    expected_height = float(terrain_height(
        geometry["terrain_model"], sensor[0], sensor[1],
    )) + float(configuration["environment"]["sensor_mount_height_m"])
    checks = {
        "nested_follower_passed": bool(
            cost_to_go["status"]["success"]
            and switching["status"]["success"]
            and trajectory["status"]["success"]
            and replay["status"]["success"]
        ),
        "sensor_height_is_terrain_following": abs(sensor[2] - expected_height) <= 1.0e-10,
        "coverage_normalized": 0.0 <= coverage_normalized <= 1.0,
        "defender_objective_finite": bool(np.isfinite(defender_objective)),
        "attacker_replay_feasible": bool(replay["feasible"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    summary = {
        "sensor_position_m": sensor.tolist(),
        "defender_objective": defender_objective,
        "defender_pod_normalized": pod_normalized,
        "coverage_volume_normalized": coverage_normalized,
        "coverage_volume_m3": float(coverage["los_volume_m3"]),
        "mission_pod": float(replay["continuous_mission_pod"]),
        "mission_hazard": float(replay["continuous_mission_hazard"]),
        "attacker_objective": float(replay["continuous_mission_cost"]),
        "mission_time_s": float(replay["continuous_mission_time_s"]),
        "switching_point_m": np.asarray(trajectory["switching_point"]).tolist(),
        "goal_distance_m": float(replay["validation"]["metrics"]["goal_distance_m"]),
        "validation_passed": not failed,
    }
    result = {
        "summary": summary,
        "objective_breakdown": {
            "pod_normalized": pod_normalized,
            "coverage_volume_normalized": coverage_normalized,
            "weighted_pod": configuration["cost"]["defender"]["w_pod"] * pod_normalized,
            "weighted_coverage": configuration["cost"]["defender"]["w_coverage"] * coverage_normalized,
            "total": defender_objective,
        },
        "validation": {"passed": not failed, "checks": checks, "failed_checks": failed},
        "metadata": {
            "fresh_nested_follower_solve": True,
            "sensor_decision_axes": ("x_sensor", "y_sensor"),
            "sensor_height_rule": configuration["defender_search"]["sensor_height_rule"],
            "defender_objective_id": configuration["cost"]["defender"]["objective_id"],
            "coverage_extension": "2D LOS area fraction -> 3D LOS volume fraction",
        },
        "status": {"success": not failed, "message": "defender evaluation passed" if not failed else f"failed: {failed}"},
    }
    if retain_full_pipeline:
        result["pipeline"] = {
            "configuration": configuration,
            "geometry": geometry,
            "detection": detection,
            "stage_cost": stage,
            "cost_to_go": cost_to_go,
            "switching": switching,
            "trajectory": trajectory,
            "continuous_replay": replay,
        }
    return result


def _cache_key(configuration: dict[str, Any], x: float, y: float) -> str:
    decimals = int(configuration["defender_search"]["cache_round_decimals"])
    return f"{configuration_fingerprint(configuration)}_x{x:.{decimals}f}_y{y:.{decimals}f}"


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def make_cached_evaluator(
    configuration: dict[str, Any],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Callable[[float, float], dict[str, Any]], dict[str, int]]:
    stats = {"fresh_evaluations": 0, "cache_hits": 0}

    def evaluate(x: float, y: float) -> dict[str, Any]:
        x, y = float(x), float(y)
        key = _cache_key(configuration, x, y)
        path = cache_dir / f"{key}.json"
        if path.exists():
            stats["cache_hits"] += 1
            if progress_callback is not None:
                progress_callback({"event": "cache_hit", "x_sensor": x, "y_sensor": y})
            return json.loads(path.read_text(encoding="utf-8"))
        if progress_callback is not None:
            progress_callback({"event": "evaluation_started", "x_sensor": x, "y_sensor": y})
        try:
            result = evaluate_defender_position(
                (x, y), configuration, retain_full_pipeline=False,
            )
            record = {**result["summary"], "feasible": True, "error": None}
        except (ValueError, RuntimeError) as error:
            record = {
                "sensor_position_m": [x, y, float("nan")],
                "defender_objective": float("-inf"),
                "defender_pod_normalized": float("nan"),
                "coverage_volume_normalized": float("nan"),
                "coverage_volume_m3": float("nan"),
                "mission_pod": float("nan"),
                "mission_hazard": float("nan"),
                "attacker_objective": float("nan"),
                "mission_time_s": float("nan"),
                "switching_point_m": [float("nan")] * 3,
                "goal_distance_m": float("nan"),
                "validation_passed": False,
                "feasible": False,
                "error": f"{type(error).__name__}: {error}",
            }
        stats["fresh_evaluations"] += 1
        _write_summary(path, record)
        if progress_callback is not None:
            progress_callback({
                "event": "evaluation_completed", "x_sensor": x, "y_sensor": y,
                "feasible": record["feasible"],
                "defender_objective": record["defender_objective"],
                "fresh_evaluation_count": stats["fresh_evaluations"],
            })
        return record

    return evaluate, stats


def hierarchical_candidate_search(
    evaluate: Callable[[float, float], dict[str, Any]],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic coarse-to-fine 2D candidate search."""
    x_bounds = tuple(float(value) for value in options["x_bounds_m"])
    y_bounds = tuple(float(value) for value in options["y_bounds_m"])
    center_y = float(options["symmetry_center_y_m"])
    symmetry_requested = bool(options["use_y_reflection_symmetry"])
    history: dict[tuple[float, float], dict[str, Any]] = {}
    symmetry_verified = not symmetry_requested

    def raw(x: float, y: float) -> dict[str, Any]:
        key = (round(float(x), 10), round(float(y), 10))
        if key not in history:
            history[key] = evaluate(float(x), float(y))
        return history[key]

    if symmetry_requested and bool(options["verify_y_reflection_symmetry"]):
        x_mid = 0.5 * (x_bounds[0] + x_bounds[1])
        lower = raw(x_mid, y_bounds[0])
        upper = raw(x_mid, y_bounds[1])
        tolerance = float(options["symmetry_tolerance"])
        symmetry_verified = bool(
            lower["feasible"] == upper["feasible"]
            and (
                not lower["feasible"]
                or abs(lower["defender_objective"] - upper["defender_objective"]) <= tolerance
            )
        )
        if not symmetry_verified:
            raise RuntimeError("configured y-reflection symmetry failed explicit verification")

    def score(x: float, y: float) -> dict[str, Any]:
        if symmetry_requested and symmetry_verified and y > center_y:
            mirror_y = 2.0 * center_y - y
            source = raw(x, mirror_y)
            record = deepcopy(source)
            record["sensor_position_m"][1] = float(y)
            if np.isfinite(record["switching_point_m"][1]):
                record["switching_point_m"][1] = 2.0 * center_y - record["switching_point_m"][1]
            record["symmetry_source_y_m"] = float(mirror_y)
            history[(round(float(x), 10), round(float(y), 10))] = record
            return record
        return raw(x, y)

    x_values = np.linspace(*x_bounds, int(options["coarse_x_count"]))
    y_values = np.linspace(*y_bounds, int(options["coarse_y_count"]))
    levels: list[dict[str, Any]] = []
    for level in range(int(options["refinement_levels"]) + 1):
        before = set(history)
        for x in x_values:
            for y in y_values:
                score(float(x), float(y))
        finite = [
            record for record in history.values()
            if record["feasible"] and np.isfinite(record["defender_objective"])
        ]
        if not finite:
            raise RuntimeError("no feasible Defender candidate has an Attacker response")
        best = max(
            finite,
            key=lambda item: (
                item["defender_objective"],
                -item["sensor_position_m"][0],
                -item["sensor_position_m"][1],
            ),
        )
        levels.append({
            "level": level,
            "x_values_m": x_values.tolist(),
            "y_values_m": y_values.tolist(),
            "new_candidate_count": len(set(history) - before),
            "best_sensor_position_m": best["sensor_position_m"],
            "best_defender_objective": best["defender_objective"],
        })
        if level == int(options["refinement_levels"]):
            break
        x_step = float(x_values[1] - x_values[0]) / float(options["refinement_factor"])
        y_step = float(y_values[1] - y_values[0]) / float(options["refinement_factor"])
        count = int(options["local_stencil_count"])
        offsets = np.arange(count) - (count - 1) / 2.0
        x_values = np.unique(np.clip(
            best["sensor_position_m"][0] + offsets * x_step, *x_bounds,
        ))
        y_values = np.unique(np.clip(
            best["sensor_position_m"][1] + offsets * y_step, *y_bounds,
        ))

    feasible = [
        record for record in history.values()
        if record["feasible"] and np.isfinite(record["defender_objective"])
    ]
    best = max(
        feasible,
        key=lambda item: (
            item["defender_objective"],
            -item["sensor_position_m"][0],
            -item["sensor_position_m"][1],
        ),
    )
    records = sorted(
        history.values(),
        key=lambda item: (item["sensor_position_m"][0], item["sensor_position_m"][1]),
    )
    return {
        "best_summary": best,
        "evaluations": tuple(records),
        "levels": tuple(levels),
        "metadata": {
            "algorithm": "deterministic_2d_hierarchical_candidate_search",
            "symmetry_requested": symmetry_requested,
            "symmetry_verified": symmetry_verified,
            "candidate_count": len(records),
            "fresh_unique_geometry_count": len([
                key for key, value in history.items()
                if "symmetry_source_y_m" not in value
            ]),
        },
    }


def solve_stackelberg_game(
    configuration: dict[str, Any],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Search the 2D sensor region, then rerun the selected full pipeline."""
    evaluator, cache_stats = make_cached_evaluator(
        configuration, cache_dir, progress_callback,
    )
    search = hierarchical_candidate_search(
        evaluator, configuration["defender_search"],
    )
    best_xy = tuple(search["best_summary"]["sensor_position_m"][:2])
    if progress_callback is not None:
        progress_callback({
            "event": "final_evaluation_started",
            "x_sensor": best_xy[0], "y_sensor": best_xy[1],
        })
    final = evaluate_defender_position(
        best_xy, configuration, retain_full_pipeline=True,
    )
    if progress_callback is not None:
        progress_callback({
            "event": "final_evaluation_completed",
            "x_sensor": best_xy[0], "y_sensor": best_xy[1],
            "defender_objective": final["summary"]["defender_objective"],
        })
    evaluated_maximum = max(
        item["defender_objective"] for item in search["evaluations"] if item["feasible"]
    )
    checks = {
        "final_nested_pipeline_passed": bool(final["status"]["success"]),
        "selected_candidate_matches_search_maximum": abs(
            final["summary"]["defender_objective"] - evaluated_maximum
        ) <= 1.0e-9,
        "sensor_within_configured_region": bool(
            configuration["defender_search"]["x_bounds_m"][0] <= best_xy[0]
            <= configuration["defender_search"]["x_bounds_m"][1]
            and configuration["defender_search"]["y_bounds_m"][0] <= best_xy[1]
            <= configuration["defender_search"]["y_bounds_m"][1]
        ),
        "sensor_height_terrain_following": final["validation"]["checks"][
            "sensor_height_is_terrain_following"
        ],
        "final_attacker_continuous_replay_passed": final["pipeline"][
            "continuous_replay"
        ]["status"]["success"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "optimal_sensor_position_m": final["summary"]["sensor_position_m"],
        "defender_objective": final["summary"]["defender_objective"],
        "final_evaluation": final,
        "search": search,
        "cache_stats": cache_stats,
        "metadata": {
            "game": "3D terrain-following sensor Stackelberg",
            "leader_decision_dimension": 2,
            "leader_decision_axes": ("x_sensor", "y_sensor"),
            "follower": "authoritative physical Bellman Attacker response",
            "continuous_nlp_applied": False,
        },
        "validation": {"passed": not failed, "checks": checks, "failed_checks": failed},
        "status": {"success": not failed, "message": "3D Stackelberg solve passed" if not failed else f"failed: {failed}"},
    }

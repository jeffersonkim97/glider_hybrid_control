"""Measure exact Bellman and trained-DQN full Local-SSE on one terrain.

The shared Defender search, topology, initial action, neighborhood radius, and
payoff definitions come directly from ``3D_0827``.  Only the attacker best
response evaluator changes.  The DQN run is explicitly approximate: it can find
a local optimum under its supplied rollouts but cannot certify an exact SSE.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
import gc
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

import project_paths
from P1b_Exact_Local_SSE import exact_best_response, exact_local_sse
from P1b_condition import ComputationCondition, build_scene
from aabb_gpu_observation import ExactBatchedAABBGPUObservationBuilder
from attacker_br_problem import AttackerBRProblem
from detection_hazard import hazard_to_detection_probability
from dqn_action_catalog import HeadingActionCatalog, SharedActionRowCache
from dqn_env import ObservationTensorCache
from dqn_model import DQNNetworkConfig, TerrainDQN
from dqn_training import CompatibilitySignature
from local_sse_contract import DefenderNeighborhoodConfig, LocalDefenderEvaluation
from local_sse_search import run_local_sse_search
from phase4_dataset import Phase4ScenarioSpec, terrain_definition_hash
from phase4_multi_terrain import (
    PHASE4_ACTION_ROW_CACHE_ENTRIES_PER_TERRAIN,
    PHASE4_OBSERVATION_CACHE_ENTRIES,
    PHASE4_SURFACE_CACHE_ENTRIES_PER_TERRAIN,
    BellmanReference,
    ScenarioRuntime,
    evaluate_policy_batched,
)
from terrain_observation import ObservationConfig, TerrainObservationBuilder


def _condition_tag(spatial_resolution_m: float, heading_spacing_deg: float) -> str:
    def clean(value: float) -> str:
        return f"{float(value):g}".replace(".", "p")

    return f"dx{clean(spatial_resolution_m)}m_dpsi{clean(heading_spacing_deg)}deg"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_network(
    checkpoint: Path, device: torch.device,
) -> tuple[TerrainDQN, dict[str, Any]]:
    started = perf_counter()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    network_config = DQNNetworkConfig.from_dict(payload["network_config"])
    network = TerrainDQN(network_config)
    network.load_state_dict(payload["model_state_dict"])
    network.to(device)
    network.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return network, payload | {"_load_runtime_sec": perf_counter() - started}


def _exact_summary(result: Any, wall_runtime_sec: float) -> dict[str, Any]:
    return {
        "method": "exact_bellman",
        "wall_runtime_sec": wall_runtime_sec,
        "selected_action_id": result.selected_action_id,
        "selected_sensor_map": result.selected_sensor_map,
        "J_A": result.attacker_objective,
        "J_D": result.detection_probability,
        "unique_defender_evaluations": result.unique_evaluations,
        "cached_evaluation_reuses": result.cached_reuses,
        "local_search_iterations": result.iterations,
        "visited_action_ids": list(result.visited_action_ids),
        "termination_status": result.termination_status,
        "local_sse_verified": bool(result.feasible),
        "evaluated": list(result.evaluated),
        "timing": result.timing,
        "sizes": result.sizes,
    }


def _audit_dqn_selected_action_with_exact_br(
    scene: Any,
    selected_action_id: int,
    configuration: DefenderNeighborhoodConfig,
    known_records: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Check the learned selection against exact BR values at every neighbor."""

    topology = scene.defender_grid.topology()
    required_ids = (int(selected_action_id),) + topology.neighbors(
        int(selected_action_id), configuration,
    )
    records: dict[int, dict[str, Any]] = {}
    added_runtime = 0.0
    for action_id in required_ids:
        existing = known_records.get(action_id)
        if existing is not None:
            records[action_id] = dict(existing) | {"source": "exact_full_sse_cache"}
            continue
        sensor = tuple(
            float(value) for value in scene.defender_grid.position(action_id)
        )
        started = perf_counter()
        response = exact_best_response(scene, sensor)
        runtime = perf_counter() - started
        added_runtime += runtime
        records[action_id] = {
            "action_id": int(action_id),
            "x_map": sensor[0],
            "y_map": sensor[1],
            "feasible": bool(response.feasible),
            "J_A": response.attacker_objective,
            "J_D": response.detection_probability,
            "source": "added_exact_audit",
            "runtime_sec": runtime,
        }
        print(
            f"  Exact audit action={action_id} sensor={sensor} "
            f"J_D={response.detection_probability} wall={runtime:.3f}s",
            flush=True,
        )
    current = records[int(selected_action_id)]
    feasible_neighbors = [
        records[action_id] for action_id in required_ids[1:]
        if records[action_id]["feasible"]
    ]
    maximum_neighbor = max(float(item["J_D"]) for item in feasible_neighbors)
    margin = float(current["J_D"]) - maximum_neighbor
    return {
        "selected_action_id": int(selected_action_id),
        "required_action_ids": list(required_ids),
        "all_required_exact_values_available": len(records) == len(required_ids),
        "exact_local_sse_verified": bool(current["feasible"] and margin >= -1.0e-12),
        "exact_local_optimality_margin": margin,
        "maximum_neighbor_J_D": maximum_neighbor,
        "added_exact_evaluations": sum(
            item["source"] == "added_exact_audit" for item in records.values()
        ),
        "added_exact_runtime_sec_excluded_from_benchmarks": added_runtime,
        "evaluated": [records[action_id] for action_id in required_ids],
    }


def run_comparison(
    *, terrain_category: str = "stepped_pyramid",
    spatial_resolution_m: float = 25.0,
    heading_spacing_deg: float = 5.0,
    r_neighbor: int = 1,
    run_label: str = "official_batched_v1",
    device_name: str = "cuda",
) -> dict[str, Any]:
    condition = ComputationCondition(
        spatial_resolution_m=spatial_resolution_m,
        heading_spacing_deg=heading_spacing_deg,
        r_neighbor=r_neighbor,
        terrain_category=terrain_category,
    )
    scene_started = perf_counter()
    scene = build_scene(condition)
    scene_wall = perf_counter() - scene_started
    initial_sensor = tuple(
        float(value) for value in scene.defender_grid.position(scene.seed_action_id)
    )
    print(
        f"Shared scene ready in {scene_wall:.3f}s; initial defender "
        f"action={scene.seed_action_id}, sensor={initial_sensor}",
        flush=True,
    )

    condition_tag = _condition_tag(spatial_resolution_m, heading_spacing_deg)
    checkpoint = (
        project_paths.DQN_DIR / "checkpoints" / "phase4_multi_terrain"
        / condition_tag / run_label / "phase4_official_generalized.pt"
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    network, checkpoint_payload = _load_network(checkpoint, device)
    print(
        f"Loaded trained DQN seed={checkpoint_payload['selected_from_seed']} "
        f"episode={checkpoint_payload['episode']} on {device}",
        flush=True,
    )

    print("Running exact Bellman full Local-SSE...", flush=True)
    exact_started = perf_counter()
    exact_result = exact_local_sse(scene)
    exact_wall = perf_counter() - exact_started
    print(
        f"Exact complete: {exact_wall:.3f}s, evaluations="
        f"{exact_result.unique_evaluations}, sensor={exact_result.selected_sensor_map}",
        flush=True,
    )

    topology = scene.defender_grid.topology()
    neighborhood = DefenderNeighborhoodConfig(r_neighbor=r_neighbor)
    surface_cache: OrderedDict[tuple[float, float], float] = OrderedDict()
    row_cache = SharedActionRowCache(
        max_entries=PHASE4_ACTION_ROW_CACHE_ENTRIES_PER_TERRAIN,
    )
    terrain_hash = terrain_definition_hash(terrain_category)
    expected_compatibility = checkpoint_payload["compatibility"]
    responses: dict[int, dict[str, Any]] = {}
    dqn_evaluator_wall = 0.0
    dqn_inference_wall = 0.0
    compatibility_checked = False

    def dqn_evaluator(action_id: int) -> LocalDefenderEvaluation:
        nonlocal dqn_evaluator_wall, dqn_inference_wall, compatibility_checked
        evaluation_started = perf_counter()
        sensor = tuple(
            float(value) for value in scene.defender_grid.position(action_id)
        )
        problem = AttackerBRProblem(scene, sensor)
        builder = TerrainObservationBuilder(
            problem,
            ObservationConfig(),
            surface_height_cache=surface_cache,
            max_surface_cache_entries=PHASE4_SURFACE_CACHE_ENTRIES_PER_TERRAIN,
        )
        catalog = HeadingActionCatalog(problem, row_cache=row_cache)
        starts = problem.switching_state_ids()
        if not starts:
            elapsed = perf_counter() - evaluation_started
            dqn_evaluator_wall += elapsed
            responses[int(action_id)] = {
                "action_id": int(action_id), "sensor_map": list(sensor),
                "success": False, "status": "no_admissible_switching_state",
                "evaluator_wall_sec": elapsed,
            }
            return LocalDefenderEvaluation(
                action_id=action_id, status="model_infeasible",
                defender_value=None, attacker_objective=None,
                selected_attacker_response_id=None,
                exact_attacker_best_response_verified=False,
                strong_tie_break_verified=False,
                diagnostic="no admissible lattice switching state",
            )

        signature = CompatibilitySignature.from_components(problem, builder, catalog)
        if signature.as_dict() != expected_compatibility:
            raise ValueError("trained DQN is incompatible with this Local-SSE context")
        compatibility_checked = True
        batch_builder = (
            ExactBatchedAABBGPUObservationBuilder(builder, device)
            if device.type == "cuda" else None
        )
        spec = Phase4ScenarioSpec(
            scenario_id=f"full_sse__{terrain_category}__action_{int(action_id)}",
            split="FULL_SSE",
            terrain_category=terrain_category,
            sensor_map=sensor,
            terrain_hash=terrain_hash,
        )
        context = ScenarioRuntime(
            spec=spec,
            problem=problem,
            builder=builder,
            catalog=catalog,
            starts=starts,
            cache=ObservationTensorCache(
                builder,
                max_entries=PHASE4_OBSERVATION_CACHE_ENTRIES,
                batch_builder=batch_builder,
            ),
            reference=BellmanReference(
                scenario_id=spec.scenario_id,
                feasible=False,
                attacker_objective=None,
                switching_state_id=None,
                trajectory=(),
                runtime_sec=0.0,
                timing={},
            ),
        )
        evaluation = evaluate_policy_batched(network, context, device)
        dqn_inference_wall += float(evaluation.inference_runtime_sec)
        if not evaluation.success or not evaluation.trajectory:
            elapsed = perf_counter() - evaluation_started
            dqn_evaluator_wall += elapsed
            responses[int(action_id)] = {
                "action_id": int(action_id), "sensor_map": list(sensor),
                **evaluation.as_dict(), "evaluator_wall_sec": elapsed,
            }
            print(
                f"  DQN action={action_id} sensor={sensor} infeasible "
                f"wall={elapsed:.3f}s",
                flush=True,
            )
            return LocalDefenderEvaluation(
                action_id=action_id, status="model_infeasible",
                defender_value=None, attacker_objective=None,
                selected_attacker_response_id=None,
                exact_attacker_best_response_verified=False,
                strong_tie_break_verified=False,
                diagnostic=evaluation.status,
            )

        scored = problem.evaluate_trajectory(
            evaluation.trajectory, reached_goal=True,
        )
        detection_probability = hazard_to_detection_probability(
            scored.cumulative_hazard
        )
        elapsed = perf_counter() - evaluation_started
        dqn_evaluator_wall += elapsed
        responses[int(action_id)] = {
            "action_id": int(action_id),
            "sensor_map": list(sensor),
            **evaluation.as_dict(),
            "detection_probability": float(detection_probability),
            "cumulative_hazard": float(scored.cumulative_hazard),
            "mission_glide_duration_s": float(scored.duration_s),
            "evaluator_wall_sec": elapsed,
            "context_and_candidate_overhead_sec": max(
                0.0, elapsed - float(evaluation.inference_runtime_sec)
            ),
        }
        print(
            f"  DQN action={action_id} sensor={sensor} J_A="
            f"{evaluation.attacker_objective:.6f} J_D={detection_probability:.6f} "
            f"candidates={evaluation.candidate_count} wall={elapsed:.3f}s "
            f"inference={evaluation.inference_runtime_sec:.3f}s",
            flush=True,
        )
        problem.clear_runtime_caches()
        del context, batch_builder, catalog, builder, problem
        gc.collect()
        return LocalDefenderEvaluation(
            action_id=action_id,
            status="feasible",
            defender_value=float(detection_probability),
            attacker_objective=float(evaluation.attacker_objective),
            selected_attacker_response_id=int(evaluation.switching_state_id),
            exact_attacker_best_response_verified=False,
            strong_tie_break_verified=False,
            diagnostic=None,
        )

    print("Running trained-DQN approximate full Local-SSE...", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dqn_started = perf_counter()
    dqn_search = run_local_sse_search(
        scene.seed_action_id,
        topology,
        dqn_evaluator,
        neighborhood,
        require_exact_attacker_verification=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dqn_wall = perf_counter() - dqn_started
    print(
        f"DQN Local-SSE complete: {dqn_wall:.3f}s, evaluations="
        f"{dqn_search.unique_defender_evaluations}, action="
        f"{dqn_search.final_local_sse_action_id}",
        flush=True,
    )

    dqn_final_id = dqn_search.final_local_sse_action_id
    dqn_final_sensor = (
        None if dqn_final_id is None else [
            float(value) for value in scene.defender_grid.position(dqn_final_id)
        ]
    )
    same_action = bool(
        dqn_final_id is not None
        and exact_result.selected_action_id is not None
        and int(dqn_final_id) == int(exact_result.selected_action_id)
    )
    known_exact_records = {
        int(item["action_id"]): dict(item) for item in exact_result.evaluated
    }
    exact_replay_runtime = 0.0
    if dqn_final_sensor is None:
        exact_at_dqn = None
    elif same_action:
        exact_at_dqn = {
            "source": "exact_full_sse_final_record",
            "J_A": exact_result.attacker_objective,
            "J_D": exact_result.detection_probability,
            "switching_state_id": None,
        }
    elif int(dqn_final_id) in known_exact_records:
        known = known_exact_records[int(dqn_final_id)]
        exact_at_dqn = {
            "source": "exact_full_sse_evaluated_record",
            "J_A": known["J_A"],
            "J_D": known["J_D"],
            "switching_state_id": None,
        }
    else:
        replay_started = perf_counter()
        replay = exact_best_response(scene, tuple(dqn_final_sensor))
        exact_replay_runtime = perf_counter() - replay_started
        exact_at_dqn = {
            "source": "separate_exact_best_response_replay",
            "J_A": replay.attacker_objective,
            "J_D": replay.detection_probability,
            "switching_state_id": replay.switching_state_id,
            "runtime_sec": exact_replay_runtime,
        }

    if dqn_final_id is None:
        exact_local_audit = None
    elif same_action and exact_result.feasible:
        exact_local_audit = {
            "selected_action_id": int(dqn_final_id),
            "exact_local_sse_verified": True,
            "source": "exact_full_sse_certification",
            "added_exact_evaluations": 0,
            "added_exact_runtime_sec_excluded_from_benchmarks": 0.0,
        }
    else:
        print(
            "Auditing the DQN-selected Defender action with exact neighboring BRs...",
            flush=True,
        )
        exact_local_audit = _audit_dqn_selected_action_with_exact_br(
            scene, int(dqn_final_id), neighborhood, known_exact_records,
        )

    dqn_summary = {
        "method": "trained_dqn_approximation",
        "wall_runtime_sec": dqn_wall,
        "selected_action_id": dqn_final_id,
        "selected_sensor_map": dqn_final_sensor,
        "predicted_J_A": dqn_search.final_J_A,
        "predicted_J_D": dqn_search.final_J_D,
        "unique_defender_evaluations": dqn_search.unique_defender_evaluations,
        "cached_evaluation_reuses": dqn_search.cached_evaluation_reuses,
        "local_search_iterations": dqn_search.local_search_iterations,
        "visited_action_ids": list(dqn_search.visited_defender_actions),
        "termination_status": dqn_search.termination_status,
        "local_sse_verified": dqn_search.local_sse_verified,
        "approximate_local_optimum_under_supplied_evaluations": (
            dqn_final_id is not None
        ),
        "evaluator_wall_sum_sec": dqn_evaluator_wall,
        "network_rollout_and_scoring_sum_sec": dqn_inference_wall,
        "context_candidate_and_control_sec": max(
            0.0, dqn_wall - dqn_inference_wall
        ),
        "evaluated": [responses[action_id] for action_id in dqn_search.evaluated_defender_actions],
        "search": dqn_search.as_dict(),
    }

    exact_summary = _exact_summary(exact_result, exact_wall)
    comparison = {
        "selected_defender_action_match": same_action,
        "selected_sensor_distance_m": (
            None if dqn_final_sensor is None or exact_result.selected_sensor_map is None
            else float(np.linalg.norm(
                np.asarray(dqn_final_sensor[:2])
                - np.asarray(exact_result.selected_sensor_map[:2])
            ) * scene.config.physical_scale.meters_per_map_unit)
        ),
        "full_sse_runtime_speedup_bellman_over_dqn": exact_wall / dqn_wall,
        "unique_evaluation_difference": (
            dqn_search.unique_defender_evaluations - exact_result.unique_evaluations
        ),
        "exact_payoff_at_dqn_selected_sensor": exact_at_dqn,
        "dqn_J_A_relative_error_at_selected_sensor": (
            None if exact_at_dqn is None or dqn_search.final_J_A is None
            else abs(float(dqn_search.final_J_A) - float(exact_at_dqn["J_A"]))
            / abs(float(exact_at_dqn["J_A"]))
        ),
        "dqn_J_D_absolute_error_at_selected_sensor": (
            None if exact_at_dqn is None or dqn_search.final_J_D is None
            else abs(float(dqn_search.final_J_D) - float(exact_at_dqn["J_D"]))
        ),
        "dqn_selected_sensor_exact_local_sse_certified": (
            None if exact_local_audit is None
            else bool(exact_local_audit["exact_local_sse_verified"])
        ),
        "dqn_selected_sensor_exact_local_audit": exact_local_audit,
        "extra_exact_replay_runtime_sec_excluded_from_benchmarks": exact_replay_runtime,
    }

    output_dir = (
        project_paths.CORE_DIR / "figure" / "phase_4_multi_terrain_dqn"
        / condition_tag / run_label / "full_sse"
    )
    result = {
        "experiment": "Bellman versus trained-DQN full Local-SSE",
        "condition": condition.as_dict(),
        "initial_defender_action_id": scene.seed_action_id,
        "initial_sensor_map": list(initial_sensor),
        "device": {
            "requested": device_name,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(project_paths.WORKSPACE_ROOT)).replace("\\", "/"),
            "seed": checkpoint_payload["selected_from_seed"],
            "episode": checkpoint_payload["episode"],
            "load_runtime_sec_excluded_from_benchmarks": checkpoint_payload["_load_runtime_sec"],
            "compatibility_checked": compatibility_checked,
        },
        "shared_scene": {
            "build_wall_sec_excluded_from_benchmarks": scene_wall,
            "build_timing": scene.build_timing,
            "sizes": scene.sizes(),
        },
        "exact_bellman": exact_summary,
        "trained_dqn": dqn_summary,
        "comparison": comparison,
        "scope": {
            "full_local_sse": True,
            "global_sse": False,
            "same_defender_search_and_neighborhood": True,
            "exact_run_certifies_local_sse": True,
            "dqn_run_is_approximate_and_cannot_self_certify": True,
            "benchmark_excludes_shared_scene_build_and_model_load": True,
        },
    }
    output_path = output_dir / f"full_sse_{terrain_category}_comparison.json"
    _write_json(output_path, result)

    markdown = [
        f"# Full Local-SSE comparison — `{terrain_category}`",
        "",
        f"Condition: `{spatial_resolution_m:g} m`, `{heading_spacing_deg:g} deg`, "
        f"Chebyshev `r={r_neighbor}`; initial sensor `{initial_sensor}`.",
        "",
        "| Method | Full SSE time (s) | Defender evaluations | Iterations | Final sensor | J_A | J_D | Status |",
        "|---|---:|---:|---:|---|---:|---:|---|",
        f"| Exact Bellman | {exact_wall:.3f} | {exact_result.unique_evaluations} | "
        f"{exact_result.iterations} | `{exact_result.selected_sensor_map}` | "
        f"{exact_result.attacker_objective:.8f} | {exact_result.detection_probability:.8f} | "
        f"`{exact_result.termination_status}` |",
        f"| Trained DQN | {dqn_wall:.3f} | {dqn_search.unique_defender_evaluations} | "
        f"{dqn_search.local_search_iterations} | `{dqn_final_sensor}` | "
        f"{dqn_search.final_J_A:.8f} | {dqn_search.final_J_D:.8f} | "
        f"`{dqn_search.termination_status}` |",
        "",
        f"Bellman/DQN runtime ratio: **{exact_wall / dqn_wall:.3f}x**.",
        f"Selected Defender action match: **{same_action}**.",
        "",
        "The shared scene/graph build and checkpoint load are excluded from both "
        "full-SSE timings. The exact result is certified; the DQN result is an "
        "approximate local optimum under learned attacker responses.",
        "",
    ]
    (output_dir / f"full_sse_{terrain_category}_comparison.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(f"Wrote {output_path}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", default="stepped_pyramid")
    parser.add_argument("--spatial-resolution-m", type=float, default=25.0)
    parser.add_argument("--heading-spacing-deg", type=float, default=5.0)
    parser.add_argument("--r-neighbor", type=int, default=1)
    parser.add_argument("--run-label", default="official_batched_v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run_comparison(
        terrain_category=args.terrain,
        spatial_resolution_m=args.spatial_resolution_m,
        heading_spacing_deg=args.heading_spacing_deg,
        r_neighbor=args.r_neighbor,
        run_label=args.run_label,
        device_name=args.device,
    )
    print(json.dumps({
        "exact_bellman": result["exact_bellman"],
        "trained_dqn": {
            key: result["trained_dqn"][key] for key in (
                "wall_runtime_sec", "selected_action_id", "selected_sensor_map",
                "predicted_J_A", "predicted_J_D", "unique_defender_evaluations",
                "local_search_iterations", "termination_status",
            )
        },
        "comparison": result["comparison"],
    }, indent=2))


if __name__ == "__main__":
    main()

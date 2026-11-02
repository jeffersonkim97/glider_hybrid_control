"""Phase-6 pilot: frozen G2 DQN inside the existing Local-SSE search.

The outer Defender topology, Chebyshev neighborhood, deterministic movement
rule, and stopping rule come from ``3D_0827/local_sse_search.py``.  The exact
and approximate runs differ only in the attacker best-response evaluator.
No training, fine-tuning, replay mutation, or checkpoint selection occurs.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import gc
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch

import project_paths
from defender_grid_2d import DefenderRegion, build_defender_grid_2d
from detection_hazard import hazard_to_detection_probability
from dqn_action_catalog import HeadingActionCatalog, SharedActionRowCache
from dqn_env import ObservationTensorCache
from dqn_model import DQNNetworkConfig, TerrainDQN
from dqn_training import CompatibilitySignature
from local_sse_contract import DefenderNeighborhoodConfig, LocalDefenderEvaluation
from local_sse_search import run_local_sse_search
from multiscale_observation import (
    ExactBatchedAABBGPUObservationMultiscaleBuilder,
    MultiscaleObservationConfig,
    MultiscaleTerrainObservationBuilder,
)
from phase4_extended_scene import build_extended_scene
from phase4_g2_training import (
    CHECKPOINT_ROOT as G2_CHECKPOINT_ROOT,
    _condition_tag as _g2_training_condition_tag,
)
from phase4_multi_terrain import (
    BellmanReference,
    ScenarioRuntime,
    evaluate_policy_batched,
)
from phase4_v2_run_a import RunAGoalReachableProblem
from phase5_g2_final_id_test import (
    FINAL_ID_SCHEMA,
    configure_final_terrain,
    freeze_final_id_000,
)
from sparse_additive import solve_sparse_additive_bellman


PHASE6_SCHEMA = "p1b-phase6-g2-local-sse-pilot-v1"
EXACT_CACHE_SCHEMA = "p1b-phase6-g2-exact-response-cache-v1"
DEFAULT_TERRAIN_INDEX = 1
DEFAULT_SPATIAL_RESOLUTION_M = 25.0
DEFAULT_HEADING_SPACING_DEG = 5.0
DEFAULT_R_NEIGHBOR = 1
SUPPORTED_SPATIAL_RESOLUTIONS_M = (100.0, 50.0, 25.0, 10.0)
OBSERVATION_CACHE_ENTRIES = 4_096
ACTION_ROW_CACHE_ENTRIES = 50_000


@dataclass(frozen=True)
class ExactG2Response:
    feasible: bool
    sensor_map: tuple[float, float, float]
    attacker_objective: float | None
    detection_probability: float | None
    switching_state_id: int | None
    trajectory: tuple[int, ...]
    candidate_count: int
    cooptimal_candidate_count: int
    strong_tie_break_verified: bool
    runtime_sec: float
    active_state_count: int | None
    active_edge_count: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExactG2Response":
        return cls(
            feasible=bool(value["feasible"]),
            sensor_map=tuple(float(item) for item in value["sensor_map"]),
            attacker_objective=(
                None if value["attacker_objective"] is None
                else float(value["attacker_objective"])
            ),
            detection_probability=(
                None if value["detection_probability"] is None
                else float(value["detection_probability"])
            ),
            switching_state_id=(
                None if value["switching_state_id"] is None
                else int(value["switching_state_id"])
            ),
            trajectory=tuple(int(item) for item in value["trajectory"]),
            candidate_count=int(value["candidate_count"]),
            cooptimal_candidate_count=int(value["cooptimal_candidate_count"]),
            strong_tie_break_verified=bool(
                value["strong_tie_break_verified"]
            ),
            runtime_sec=float(value["runtime_sec"]),
            active_state_count=(
                None if value["active_state_count"] is None
                else int(value["active_state_count"])
            ),
            active_edge_count=(
                None if value["active_edge_count"] is None
                else int(value["active_edge_count"])
            ),
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_exact_response_cache(
    path: Path, expected_contract: dict[str, Any],
) -> dict[int, ExactG2Response]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != EXACT_CACHE_SCHEMA:
        raise RuntimeError(f"unexpected exact cache schema in {path}")
    if payload.get("contract") != expected_contract:
        raise RuntimeError(
            "exact response cache contract mismatch; refusing stale results"
        )
    responses = payload.get("responses")
    if not isinstance(responses, dict):
        raise RuntimeError("exact response cache has no response mapping")
    return {
        int(action_id): ExactG2Response.from_dict(response)
        for action_id, response in responses.items()
    }


def _save_exact_response_cache(
    path: Path,
    contract: dict[str, Any],
    responses: dict[int, ExactG2Response],
    *,
    last_completed_action_id: int,
    status: str = "in_progress",
) -> None:
    if status not in {"in_progress", "complete"}:
        raise ValueError("exact cache status must be in_progress or complete")
    _write_json(path, {
        "schema": EXACT_CACHE_SCHEMA,
        "status": status,
        "contract": contract,
        "completed_action_count": len(responses),
        "last_completed_action_id": int(last_completed_action_id),
        "responses": {
            str(action_id): responses[action_id].as_dict()
            for action_id in sorted(responses)
        },
    })


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _condition_tag(dx_m: float, heading_deg: float, radius: int) -> str:
    return f"dx{dx_m:g}m_dpsi{heading_deg:g}deg_r{int(radius)}"


def _center_aligned_defender_region(
    region: DefenderRegion, spacing: float,
) -> DefenderRegion:
    """Shrink symmetrically so each span is an integer lattice multiple."""

    def aligned_bounds(lower: float, upper: float) -> tuple[float, float]:
        span = float(upper) - float(lower)
        interval_count = int(np.floor(span / float(spacing) + 1.0e-12))
        if interval_count < 1:
            raise ValueError("defender region is smaller than one grid interval")
        aligned_span = interval_count * float(spacing)
        padding = 0.5 * (span - aligned_span)
        return float(lower) + padding, float(upper) - padding

    x_min, x_max = aligned_bounds(region.x_min, region.x_max)
    y_min, y_max = aligned_bounds(region.y_min, region.y_max)
    return DefenderRegion(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
    )


def _exact_g2_best_response(
    scene: Any, sensor_map: tuple[float, float, float],
) -> ExactG2Response:
    """Exact powered-occluded BR with the existing Strong follower tie rule."""

    started = perf_counter()
    problem = RunAGoalReachableProblem(scene, sensor_map)
    starts = problem.switching_state_ids()
    if not starts:
        return ExactG2Response(
            feasible=False, sensor_map=sensor_map, attacker_objective=None,
            detection_probability=None, switching_state_id=None, trajectory=(),
            candidate_count=0, cooptimal_candidate_count=0,
            strong_tie_break_verified=True,
            runtime_sec=perf_counter() - started,
            active_state_count=None, active_edge_count=None,
        )

    sparse = solve_sparse_additive_bellman(
        scene.graph,
        starts,
        problem.mdp.hazard_field,
        objective_parameters=scene.config.attacker_objective,
        parameters=scene.config.glider,
        physical_scale=scene.config.physical_scale,
        quadrature_resolution=(
            scene.config.discretization.hazard_quadrature_resolution
        ),
    )
    candidates: list[dict[str, Any]] = []
    for state_id in starts:
        if not sparse.solution.goal_reachable[state_id]:
            continue
        edges = sparse.solution.path_edges(state_id)
        trajectory = (int(state_id),) + tuple(
            int(edge.target_id) for edge in edges
        )
        scored = problem.evaluate_trajectory(trajectory, reached_goal=True)
        objective = problem.powered_cost(state_id) + float(
            sparse.solution.value[state_id]
        )
        candidates.append({
            "state_id": int(state_id),
            "trajectory": trajectory,
            "objective": float(objective),
            "detection_probability": float(
                hazard_to_detection_probability(scored.cumulative_hazard)
            ),
        })
    if not candidates:
        return ExactG2Response(
            feasible=False, sensor_map=sensor_map, attacker_objective=None,
            detection_probability=None, switching_state_id=None, trajectory=(),
            candidate_count=len(starts), cooptimal_candidate_count=0,
            strong_tie_break_verified=True,
            runtime_sec=perf_counter() - started,
            active_state_count=int(sparse.metrics.active_state_count),
            active_edge_count=int(sparse.metrics.active_edge_count),
        )

    minimum = min(item["objective"] for item in candidates)
    cooptimal = [
        item for item in candidates
        if item["objective"] <= minimum + 1.0e-12
    ]
    selected = max(
        cooptimal,
        key=lambda item: (
            item["detection_probability"], -item["state_id"],
        ),
    )
    runtime = perf_counter() - started
    problem.clear_runtime_caches()
    return ExactG2Response(
        feasible=True,
        sensor_map=sensor_map,
        attacker_objective=float(selected["objective"]),
        detection_probability=float(selected["detection_probability"]),
        switching_state_id=int(selected["state_id"]),
        trajectory=tuple(selected["trajectory"]),
        candidate_count=len(starts),
        cooptimal_candidate_count=len(cooptimal),
        strong_tie_break_verified=True,
        runtime_sec=runtime,
        active_state_count=int(sparse.metrics.active_state_count),
        active_edge_count=int(sparse.metrics.active_edge_count),
    )


def _local_evaluation_from_exact(
    action_id: int, response: ExactG2Response,
) -> LocalDefenderEvaluation:
    if not response.feasible:
        return LocalDefenderEvaluation(
            action_id=action_id, status="model_infeasible",
            defender_value=None, attacker_objective=None,
            selected_attacker_response_id=None,
            exact_attacker_best_response_verified=True,
            strong_tie_break_verified=True,
            diagnostic="no powered-occluded goal-reachable switching state",
        )
    return LocalDefenderEvaluation(
        action_id=action_id, status="feasible",
        defender_value=float(response.detection_probability),
        attacker_objective=float(response.attacker_objective),
        selected_attacker_response_id=int(response.switching_state_id),
        exact_attacker_best_response_verified=True,
        strong_tie_break_verified=True,
    )


def _load_frozen_network(
    checkpoint_path: Path, device: torch.device,
) -> tuple[TerrainDQN, dict[str, Any], float]:
    started = perf_counter()
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    if payload.get("kind") != "final_training_state":
        raise RuntimeError("Phase-6 requires a frozen final-training checkpoint")
    if int(payload.get("episode", -1)) != 10_000:
        raise RuntimeError("Phase-6 requires the episode-10,000 checkpoint")
    network = TerrainDQN(DQNNetworkConfig.from_dict(payload["network_config"]))
    network.load_state_dict(payload["model_state_dict"])
    metadata = {
        "kind": payload["kind"],
        "seed": int(payload["seed"]),
        "episode": int(payload["episode"]),
        "dataset_manifest_sha256": payload["dataset_manifest_sha256"],
        "compatibility": dict(payload["compatibility"]),
    }
    del payload
    gc.collect()
    network.requires_grad_(False)
    network.to(device)
    network.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return network, metadata, perf_counter() - started


def _exact_neighbor_audit(
    *, selected_action_id: int, scene: Any,
    configuration: DefenderNeighborhoodConfig,
    exact_response: Callable[[int], ExactG2Response],
) -> dict[str, Any]:
    topology = scene.defender_grid.topology()
    required = (int(selected_action_id),) + topology.neighbors(
        int(selected_action_id), configuration,
    )
    records = [(action_id, exact_response(action_id)) for action_id in required]
    current = records[0][1]
    feasible_neighbors = [
        response for _, response in records[1:] if response.feasible
    ]
    maximum_neighbor = (
        None if not feasible_neighbors
        else max(float(item.detection_probability) for item in feasible_neighbors)
    )
    margin = (
        None if not current.feasible
        else (
            float("inf") if maximum_neighbor is None
            else float(current.detection_probability) - maximum_neighbor
        )
    )
    return {
        "selected_action_id": int(selected_action_id),
        "required_action_ids": list(required),
        "exact_local_sse_verified": bool(
            current.feasible and (margin is not None and margin >= -1.0e-12)
        ),
        "exact_local_optimality_margin_J_D": margin,
        "maximum_feasible_neighbor_J_D": maximum_neighbor,
        "evaluated": [
            {"action_id": action_id, **response.as_dict()}
            for action_id, response in records
        ],
    }


def run_phase6_pilot(
    *, terrain_index: int = DEFAULT_TERRAIN_INDEX,
    spatial_resolution_m: float = DEFAULT_SPATIAL_RESOLUTION_M,
    heading_spacing_deg: float = DEFAULT_HEADING_SPACING_DEG,
    r_neighbor: int = DEFAULT_R_NEIGHBOR,
    device_name: str = "cuda",
    output_root: Path | None = None,
    result_schema: str = PHASE6_SCHEMA,
    experiment_label: str = "Phase-6",
    output_stem: str = "phase6_g2_local_sse_pilot",
) -> Path:
    if int(terrain_index) != 1:
        raise ValueError("the frozen Phase-6 pilot contract uses terrain index 1")
    if float(spatial_resolution_m) not in SUPPORTED_SPATIAL_RESOLUTIONS_M:
        raise ValueError(
            "the frozen Phase-6 comparison uses dx in "
            f"{SUPPORTED_SPATIAL_RESOLUTIONS_M} m"
        )
    if not np.isfinite(heading_spacing_deg) or float(heading_spacing_deg) <= 0.0:
        raise ValueError("heading spacing must be finite and positive")
    if int(r_neighbor) not in range(1, 6):
        raise ValueError("the controlled comparison supports integer r in [1, 5]")
    if not result_schema or not experiment_label or not output_stem:
        raise ValueError("result schema, experiment label, and output stem are required")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the G2 Phase-6 pilot requires CUDA")

    configure_final_terrain(terrain_index)
    manifest, scenarios, terrain_spec, benchmark_config = freeze_final_id_000()
    if manifest.get("schema") != FINAL_ID_SCHEMA:
        raise RuntimeError("unexpected final-ID manifest schema")
    mixed = [item for item in scenarios if item.visibility_class == "mixed"]
    if len(mixed) != 2:
        raise RuntimeError("Phase-6 pilot requires Mixed A and Mixed B")
    seed_scenario = mixed[1]

    scene_started = perf_counter()
    scene = build_extended_scene(
        seed_scenario,
        terrain_spec,
        spatial_resolution_m=spatial_resolution_m,
        heading_spacing_deg=heading_spacing_deg,
        r_neighbor=r_neighbor,
        pilot_config=benchmark_config.pilot,
    )
    nominal_defender_region = DefenderRegion(
        x_min=float(benchmark_config.pilot.sensor_x_range_map[0]),
        x_max=float(benchmark_config.pilot.sensor_x_range_map[1]),
        y_min=float(benchmark_config.pilot.sensor_y_range_map[0]),
        y_max=float(benchmark_config.pilot.sensor_y_range_map[1]),
    )
    defender_spacing_map = (
        spatial_resolution_m
        / benchmark_config.pilot.meters_per_map_unit
    )
    defender_region = _center_aligned_defender_region(
        nominal_defender_region, defender_spacing_map,
    )
    scene.defender_grid = build_defender_grid_2d(
        defender_region,
        defender_spacing_map,
        terrain=scene.terrain,
    )
    scene.seed_action_id = scene.defender_grid.nearest_action_id(
        seed_scenario.sensor_map[0], seed_scenario.sensor_map[1],
    )
    scene_wall = perf_counter() - scene_started
    initial_sensor = tuple(
        float(value)
        for value in scene.defender_grid.position(scene.seed_action_id)
    )
    topology = scene.defender_grid.topology()
    neighborhood = DefenderNeighborhoodConfig(r_neighbor=r_neighbor)
    print(
        f"{experiment_label} G2 scene ready: terrain={terrain_index:03d}, "
        f"Defender actions={scene.defender_grid.action_count}, "
        f"seed={scene.seed_action_id} at {initial_sensor}",
        flush=True,
    )

    condition_tag = _condition_tag(
        spatial_resolution_m, heading_spacing_deg, r_neighbor,
    )
    if output_root is None:
        output_root = project_paths.DQN_DIR / "runs" / "phase6_g2_local_sse"
    output_dir = output_root / f"final_id_{terrain_index:03d}" / condition_tag
    exact_cache_path = output_dir / "exact_response_cache.json"
    checkpoint_key = (
        f"dx{spatial_resolution_m:g}m_dpsi{heading_spacing_deg:g}deg"
    )
    frozen_checkpoint = manifest["checkpoints"].get(checkpoint_key)
    if frozen_checkpoint is not None:
        checkpoint_relative = frozen_checkpoint["path"]
        expected_checkpoint_hash = frozen_checkpoint["sha256"]
        checkpoint_source = "frozen_phase5_manifest"
    else:
        checkpoint_path = (
            G2_CHECKPOINT_ROOT
            / _g2_training_condition_tag(
                spatial_resolution_m, heading_spacing_deg,
            )
            / "seed0_ep10000_v1"
            / "phase4_seed0_final.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "missing condition-specific final G2 checkpoint: "
                f"{checkpoint_path}"
            )
        checkpoint_relative = checkpoint_path.relative_to(
            project_paths.WORKSPACE_ROOT
        ).as_posix()
        expected_checkpoint_hash = _file_sha256(checkpoint_path)
        checkpoint_source = "condition_specific_g2_training_output"
    checkpoint_path = project_paths.WORKSPACE_ROOT / checkpoint_relative
    if _file_sha256(checkpoint_path) != expected_checkpoint_hash:
        raise RuntimeError("frozen checkpoint hash changed")

    defender_grid_payload = [
        [float(value) for value in scene.defender_grid.position(action_id)]
        for action_id in range(scene.defender_grid.action_count)
    ]
    defender_grid_sha256 = sha256(json.dumps(
        defender_grid_payload, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    exact_cache_contract = {
        "phase6_schema": PHASE6_SCHEMA,
        "terrain_index": int(terrain_index),
        "terrain_hash": terrain_spec.terrain_hash,
        "spatial_resolution_m": float(spatial_resolution_m),
        "heading_spacing_deg": float(heading_spacing_deg),
        "r_neighbor": int(r_neighbor),
        "defender_grid_sha256": defender_grid_sha256,
        "initial_action_id": int(scene.seed_action_id),
    }
    if result_schema != PHASE6_SCHEMA:
        exact_cache_contract["result_schema"] = result_schema
    exact_cache = _load_exact_response_cache(
        exact_cache_path, exact_cache_contract,
    )
    initial_cached_action_ids = frozenset(exact_cache)
    exact_computed_calls = 0
    exact_computed_runtime = 0.0
    if exact_cache:
        print(
            f"Resuming exact cache: {len(exact_cache)} completed actions "
            f"from {exact_cache_path}",
            flush=True,
        )

    def exact_response(action_id: int) -> ExactG2Response:
        nonlocal exact_computed_calls, exact_computed_runtime
        action_id = int(action_id)
        if action_id not in exact_cache:
            sensor = tuple(
                float(value) for value in scene.defender_grid.position(action_id)
            )
            print(
                f"  Exact BR action={action_id} sensor={sensor}", flush=True,
            )
            response = _exact_g2_best_response(scene, sensor)
            exact_cache[action_id] = response
            exact_computed_calls += 1
            exact_computed_runtime += response.runtime_sec
            _save_exact_response_cache(
                exact_cache_path,
                exact_cache_contract,
                exact_cache,
                last_completed_action_id=action_id,
            )
            gc.collect()
            print(
                f"    feasible={response.feasible} J_A="
                f"{response.attacker_objective} J_D="
                f"{response.detection_probability} "
                f"wall={response.runtime_sec:.3f}s "
                f"checkpoint={len(exact_cache)}",
                flush=True,
            )
        return exact_cache[action_id]

    def exact_evaluator(action_id: int) -> LocalDefenderEvaluation:
        return _local_evaluation_from_exact(
            int(action_id), exact_response(int(action_id)),
        )

    print("Running exact powered-occluded Local-SSE...", flush=True)
    exact_started = perf_counter()
    exact_search = run_local_sse_search(
        scene.seed_action_id,
        topology,
        exact_evaluator,
        neighborhood,
        require_exact_attacker_verification=True,
    )
    exact_current_process_wall = perf_counter() - exact_started
    exact_search_action_ids = tuple(
        int(action_id) for action_id in exact_search.evaluated_defender_actions
    )
    exact_search_calls = len(exact_search_action_ids)
    exact_search_runtime = sum(
        exact_cache[action_id].runtime_sec
        for action_id in exact_search_action_ids
    )
    exact_search_reused_calls = sum(
        action_id in initial_cached_action_ids
        for action_id in exact_search_action_ids
    )
    exact_search_computed_calls = (
        exact_search_calls - exact_search_reused_calls
    )
    if exact_search_reused_calls:
        exact_benchmark_runtime = exact_search_runtime
        exact_timing_basis = "sum_of_per_response_runtime_across_resumed_runs"
    else:
        exact_benchmark_runtime = exact_current_process_wall
        exact_timing_basis = "single_process_wall_runtime"
    print(
        f"Exact Local-SSE complete: action="
        f"{exact_search.final_local_sse_action_id}, "
        f"evaluations={exact_search.unique_defender_evaluations}, "
        f"benchmark={exact_benchmark_runtime:.3f}s, "
        f"current_process_wall={exact_current_process_wall:.3f}s",
        flush=True,
    )

    print("Loading frozen DQN only after exact search...", flush=True)
    network, checkpoint_payload, checkpoint_load_sec = _load_frozen_network(
        checkpoint_path, device,
    )
    if checkpoint_payload.get("dataset_manifest_sha256") != manifest.get(
        "training_dataset_manifest_sha256"
    ):
        raise RuntimeError("G2 checkpoint training dataset hash drifted")

    surface_cache: OrderedDict[tuple[float, float], float] = OrderedDict()
    row_cache = SharedActionRowCache(max_entries=ACTION_ROW_CACHE_ENTRIES)
    context_cache: dict[int, ScenarioRuntime | None] = {}
    expected_compatibility = checkpoint_payload["compatibility"]
    compatibility_checked = False
    dqn_records: dict[int, dict[str, Any]] = {}
    dqn_evaluator_wall = 0.0
    dqn_inference_wall = 0.0

    def context_for(action_id: int) -> ScenarioRuntime | None:
        nonlocal compatibility_checked
        action_id = int(action_id)
        if action_id in context_cache:
            return context_cache[action_id]
        sensor = tuple(
            float(value) for value in scene.defender_grid.position(action_id)
        )
        problem = RunAGoalReachableProblem(scene, sensor)
        starts = problem.switching_state_ids()
        if not starts:
            context_cache[action_id] = None
            return None
        builder = MultiscaleTerrainObservationBuilder(
            problem,
            MultiscaleObservationConfig(),
            surface_height_cache=surface_cache,
            max_surface_cache_entries=200_000,
        )
        catalog = HeadingActionCatalog(problem, row_cache=row_cache)
        signature = CompatibilitySignature.from_components(
            problem, builder, catalog,
        )
        if signature.as_dict() != expected_compatibility:
            raise RuntimeError("G2 checkpoint/Local-SSE compatibility mismatch")
        compatibility_checked = True
        batch_builder = ExactBatchedAABBGPUObservationMultiscaleBuilder(
            builder, device,
        )
        spec = replace(
            seed_scenario,
            scenario_id=(
                f"{output_stem}__terrain_{terrain_index:03d}__action_{action_id}"
            ),
            sensor_map=sensor,
        )
        context = ScenarioRuntime(
            spec=spec,
            problem=problem,
            builder=builder,
            catalog=catalog,
            starts=starts,
            cache=ObservationTensorCache(
                builder,
                max_entries=OBSERVATION_CACHE_ENTRIES,
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
            failure_terminal_cost=0.0,
        )
        context_cache[action_id] = context
        return context

    def dqn_evaluator(action_id: int) -> LocalDefenderEvaluation:
        nonlocal dqn_evaluator_wall, dqn_inference_wall
        action_id = int(action_id)
        started = perf_counter()
        context = context_for(action_id)
        sensor = tuple(
            float(value) for value in scene.defender_grid.position(action_id)
        )
        if context is None:
            elapsed = perf_counter() - started
            dqn_evaluator_wall += elapsed
            dqn_records[action_id] = {
                "action_id": action_id, "sensor_map": list(sensor),
                "success": False, "status": "mission_infeasible",
                "evaluator_wall_sec": elapsed,
            }
            return LocalDefenderEvaluation(
                action_id=action_id, status="model_infeasible",
                defender_value=None, attacker_objective=None,
                selected_attacker_response_id=None,
                exact_attacker_best_response_verified=False,
                strong_tie_break_verified=False,
                diagnostic="no powered-occluded goal-reachable switching state",
            )
        evaluation = evaluate_policy_batched(network, context, device)
        dqn_inference_wall += float(evaluation.inference_runtime_sec)
        if not evaluation.success or not evaluation.trajectory:
            elapsed = perf_counter() - started
            dqn_evaluator_wall += elapsed
            dqn_records[action_id] = {
                "action_id": action_id, "sensor_map": list(sensor),
                **evaluation.as_dict(), "evaluator_wall_sec": elapsed,
            }
            return LocalDefenderEvaluation(
                action_id=action_id, status="model_infeasible",
                defender_value=None, attacker_objective=None,
                selected_attacker_response_id=None,
                exact_attacker_best_response_verified=False,
                strong_tie_break_verified=False,
                diagnostic=evaluation.status,
            )
        scored = context.problem.evaluate_trajectory(
            evaluation.trajectory, reached_goal=True,
        )
        defender_value = float(
            hazard_to_detection_probability(scored.cumulative_hazard)
        )
        elapsed = perf_counter() - started
        dqn_evaluator_wall += elapsed
        dqn_records[action_id] = {
            "action_id": action_id, "sensor_map": list(sensor),
            **evaluation.as_dict(),
            "detection_probability": defender_value,
            "cumulative_hazard": float(scored.cumulative_hazard),
            "evaluator_wall_sec": elapsed,
        }
        print(
            f"  DQN BR action={action_id} sensor={sensor} "
            f"J_A={evaluation.attacker_objective:.6f} "
            f"J_D={defender_value:.6f} "
            f"candidates={evaluation.candidate_count} wall={elapsed:.3f}s",
            flush=True,
        )
        return LocalDefenderEvaluation(
            action_id=action_id, status="feasible",
            defender_value=defender_value,
            attacker_objective=float(evaluation.attacker_objective),
            selected_attacker_response_id=int(evaluation.switching_state_id),
            exact_attacker_best_response_verified=False,
            strong_tie_break_verified=False,
        )

    print(
        "Warming the complete approximate Local-SSE path (excluded)...",
        flush=True,
    )
    warm_started = perf_counter()
    warm_search = run_local_sse_search(
        scene.seed_action_id,
        topology,
        dqn_evaluator,
        neighborhood,
        require_exact_attacker_verification=False,
    )
    torch.cuda.synchronize(device)
    warm_wall = perf_counter() - warm_started
    warm_records = dict(dqn_records)

    dqn_records.clear()
    dqn_evaluator_wall = 0.0
    dqn_inference_wall = 0.0
    print("Running warm frozen-DQN approximate Local-SSE...", flush=True)
    torch.cuda.synchronize(device)
    dqn_started = perf_counter()
    dqn_search = run_local_sse_search(
        scene.seed_action_id,
        topology,
        dqn_evaluator,
        neighborhood,
        require_exact_attacker_verification=False,
    )
    torch.cuda.synchronize(device)
    dqn_wall = perf_counter() - dqn_started
    if (
        dqn_search.final_local_sse_action_id
        != warm_search.final_local_sse_action_id
        or tuple(dqn_search.visited_defender_actions)
        != tuple(warm_search.visited_defender_actions)
    ):
        raise RuntimeError("warm timing changed the approximate Local-SSE path")
    print(
        f"DQN Local-SSE complete: action="
        f"{dqn_search.final_local_sse_action_id}, "
        f"evaluations={dqn_search.unique_defender_evaluations}, "
        f"wall={dqn_wall:.3f}s (warm-up {warm_wall:.3f}s excluded)",
        flush=True,
    )

    dqn_final_id = dqn_search.final_local_sse_action_id
    if dqn_final_id is None:
        raise RuntimeError("DQN Local-SSE returned no feasible defender action")
    for context in context_cache.values():
        if context is not None:
            context.problem.clear_runtime_caches()
    context_cache.clear()
    surface_cache.clear()
    network.to("cpu")
    del network
    del row_cache
    gc.collect()
    torch.cuda.empty_cache()

    exact_cache_ids_before_audit = frozenset(exact_cache)
    exact_computed_calls_before_audit = exact_computed_calls
    exact_computed_runtime_before_audit = exact_computed_runtime
    exact_at_dqn = exact_response(int(dqn_final_id))
    if not exact_at_dqn.feasible:
        raise RuntimeError("DQN-selected action is exact-model infeasible")
    exact_audit = _exact_neighbor_audit(
        selected_action_id=int(dqn_final_id),
        scene=scene,
        configuration=neighborhood,
        exact_response=exact_response,
    )

    exact_final_id = exact_search.final_local_sse_action_id
    if exact_final_id is None:
        raise RuntimeError("exact Local-SSE returned no feasible defender action")
    exact_final = exact_response(int(exact_final_id))
    dqn_final_record = dqn_records[int(dqn_final_id)]
    selected_match = int(exact_final_id) == int(dqn_final_id)
    selected_distance_m = float(np.linalg.norm(
        np.asarray(scene.defender_grid.position(int(exact_final_id))[:2])
        - np.asarray(scene.defender_grid.position(int(dqn_final_id))[:2])
    ) * scene.config.physical_scale.meters_per_map_unit)
    dqn_j_a_gap = float(dqn_final_record["attacker_objective"]) - float(
        exact_at_dqn.attacker_objective
    )
    exact_payoff_difference = float(exact_final.detection_probability) - float(
        exact_at_dqn.detection_probability
    )

    exact_audit_new_action_ids = sorted(
        set(exact_cache) - set(exact_cache_ids_before_audit)
    )
    _save_exact_response_cache(
        exact_cache_path,
        exact_cache_contract,
        exact_cache,
        last_completed_action_id=int(exact_final_id),
        status="complete",
    )
    result = {
        "schema": result_schema,
        "status": "complete",
        "scope": {
            "full_local_sse": True,
            "global_sse": False,
            "new_training_performed": False,
            "checkpoint_frozen": True,
            "same_outer_search_and_neighborhood": True,
            "only_attacker_br_backend_changes": True,
            "single_point_J_A_gap_is_not_a_nonzero_sum_local_certificate": True,
            "exact_neighbor_audit_is_the_local_certificate": True,
        },
        "condition": {
            "terrain_index": terrain_index,
            "terrain_id": terrain_spec.terrain_instance_id,
            "terrain_hash": terrain_spec.terrain_hash,
            "spatial_resolution_m": spatial_resolution_m,
            "heading_spacing_deg": heading_spacing_deg,
            "r_neighbor": r_neighbor,
            "nominal_defender_region_map": nominal_defender_region.as_dict(),
            "defender_region_map": defender_region.as_dict(),
            "defender_region_center_aligned_to_spacing": True,
            "defender_action_count": scene.defender_grid.action_count,
            "initial_action_id": scene.seed_action_id,
            "initial_sensor_map": list(initial_sensor),
            "initial_sensor_source": "frozen terrain-001 Mixed-B scenario",
        },
        "checkpoint": {
            "path": checkpoint_relative,
            "sha256": expected_checkpoint_hash,
            "source": checkpoint_source,
            "seed": int(checkpoint_payload["seed"]),
            "episode": int(checkpoint_payload["episode"]),
            "training_dataset_manifest_sha256": checkpoint_payload[
                "dataset_manifest_sha256"
            ],
            "load_runtime_sec_excluded": checkpoint_load_sec,
            "gradient_updates": 0,
        },
        "shared_setup": {
            "scene_build_wall_sec_excluded": scene_wall,
            "scene_build_timing": scene.build_timing,
            "scene_sizes": scene.sizes(),
            "checkpoint_compatibility_checked": compatibility_checked,
        },
        "exact_bellman_local_sse": {
            "wall_runtime_sec": exact_benchmark_runtime,
            "timing_basis": exact_timing_basis,
            "current_process_search_wall_sec": exact_current_process_wall,
            "best_response_runtime_sum_sec": exact_search_runtime,
            "best_response_calls": exact_search_calls,
            "best_response_calls_reused_from_disk": (
                exact_search_reused_calls
            ),
            "best_response_calls_computed_this_run": (
                exact_search_computed_calls
            ),
            "cache_path": str(exact_cache_path.relative_to(
                project_paths.WORKSPACE_ROOT
            )),
            "cache_entries_loaded_at_start": len(initial_cached_action_ids),
            "search": exact_search.as_dict(),
            "final_response": exact_final.as_dict(),
        },
        "frozen_dqn_local_sse": {
            "warmup_wall_sec_excluded": warm_wall,
            "wall_runtime_sec": dqn_wall,
            "evaluator_wall_sum_sec": dqn_evaluator_wall,
            "network_rollout_and_scoring_sum_sec": dqn_inference_wall,
            "search": dqn_search.as_dict(),
            "evaluated": [
                dqn_records[action_id]
                for action_id in dqn_search.evaluated_defender_actions
            ],
            "warmup_evaluated_action_ids": sorted(warm_records),
        },
        "comparison": {
            "selected_defender_action_match": selected_match,
            "selected_sensor_distance_m": selected_distance_m,
            "bellman_over_dqn_runtime_speedup": (
                exact_benchmark_runtime / dqn_wall
            ),
            "dqn_selected_sensor_exact_response": exact_at_dqn.as_dict(),
            "dqn_selected_sensor_J_A_absolute_gap": dqn_j_a_gap,
            "dqn_selected_sensor_J_A_relative_gap": (
                dqn_j_a_gap / abs(float(exact_at_dqn.attacker_objective))
            ),
            "exact_baseline_minus_dqn_selected_exact_J_D": (
                exact_payoff_difference
            ),
            "dqn_selected_sensor_exact_neighbor_audit": exact_audit,
            "exact_audit_additional_best_response_calls": (
                exact_computed_calls - exact_computed_calls_before_audit
            ),
            "exact_audit_additional_best_response_runtime_sec": (
                exact_computed_runtime - exact_computed_runtime_before_audit
            ),
            "exact_audit_new_action_ids": exact_audit_new_action_ids,
        },
    }
    output_path = output_dir / f"{output_stem}.json"
    _write_json(output_path, result)

    markdown = [
        f"# {experiment_label} G2 Local-SSE Pilot",
        "",
        f"Terrain `{terrain_index:03d}`, dx `{spatial_resolution_m:g} m`, "
        f"heading `{heading_spacing_deg:g} deg`, Chebyshev `r={r_neighbor}`.",
        "No training or fine-tuning was performed.",
        "",
        "| Method | Time | Evaluations | Final action | Final sensor | J_A | J_D |",
        "|:---|---:|---:|---:|:---|---:|---:|",
        f"| Exact Bellman | {exact_benchmark_runtime:.3f} s | "
        f"{exact_search.unique_defender_evaluations} | {exact_final_id} | "
        f"`{scene.defender_grid.position(int(exact_final_id))}` | "
        f"{exact_final.attacker_objective:.8f} | "
        f"{exact_final.detection_probability:.8f} |",
        f"| Frozen DQN | {dqn_wall:.3f} s | "
        f"{dqn_search.unique_defender_evaluations} | {dqn_final_id} | "
        f"`{scene.defender_grid.position(int(dqn_final_id))}` | "
        f"{dqn_final_record['attacker_objective']:.8f} | "
        f"{dqn_final_record['detection_probability']:.8f} |",
        "",
        f"- Selected action match: `{selected_match}`",
        f"- Selected sensor distance: `{selected_distance_m:.3f} m`",
        f"- Bellman/DQN runtime ratio: "
        f"`{exact_benchmark_runtime / dqn_wall:.3f}x`",
        f"- Exact timing basis: `{exact_timing_basis}`",
        f"- Exact responses reused from disk: `{exact_search_reused_calls}`",
        f"- DQN-selected sensor exact `J_A` gap: `{dqn_j_a_gap:.8f}`",
        f"- Exact-baseline minus DQN-selected exact `J_D`: "
        f"`{exact_payoff_difference:.8f}`",
        f"- DQN-selected sensor exact local certificate: "
        f"`{exact_audit['exact_local_sse_verified']}`",
        "",
        "The DQN warm-up pass, shared scene construction, and checkpoint load "
        "are excluded from benchmark timings. Because the implemented game is "
        "non-zero-sum, the single-point J_A gap is diagnostic only; the exact "
        "selected-plus-neighbor audit supplies the local Stackelberg certificate.",
    ]
    (output_dir / f"{output_stem}.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8",
    )
    print(f"Wrote {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain-index", type=int, default=1)
    parser.add_argument("--spatial-resolution-m", type=float, default=25.0)
    parser.add_argument("--heading-spacing-deg", type=float, default=5.0)
    parser.add_argument("--r-neighbor", type=int, default=1)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    arguments = parser.parse_args()
    output = run_phase6_pilot(
        terrain_index=arguments.terrain_index,
        spatial_resolution_m=arguments.spatial_resolution_m,
        heading_spacing_deg=arguments.heading_spacing_deg,
        r_neighbor=arguments.r_neighbor,
        device_name=arguments.device,
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "ExactG2Response",
    "SUPPORTED_SPATIAL_RESOLUTIONS_M",
    "run_phase6_pilot",
]

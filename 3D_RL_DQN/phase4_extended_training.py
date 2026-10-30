"""Runnable extended Phase 16.4 training entry point.

This runner is intentionally separate from the frozen simple-terrain experiment.
It consumes the deterministic random-box bank, multiscale observation, expanded
failure action space, powered-occlusion filter, and approved early stopping.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch

import project_paths
from dqn_action_catalog import SharedActionRowCache
from dqn_env import ObservationTensorCache
from dqn_model import DQNNetworkConfig
from dqn_training import CompatibilitySignature, DQNTrainingConfig
from extended_failure_env import (
    ExpandedFailureAttackerBRProblem,
    ExpandedFailureHeadingActionCatalog,
    FailureCostConfig,
)
from multiscale_observation import (
    ExactBatchedAABBGPUObservationMultiscaleBuilder,
    MultiscaleObservationConfig,
    MultiscaleTerrainObservationBuilder,
)
from phase4_early_stopping import EarlyStoppingConfig
from phase4_extended_dataset import (
    TRAIN_EXTENDED,
    VALIDATION_EXTENDED,
    ExtendedPilotConfig,
    ExtendedScenarioSpec,
    build_extended_scenario_bank,
    extended_dataset_manifest,
)
from phase4_extended_scene import build_extended_scene
from phase4_multi_terrain import (
    BellmanReference,
    Phase4ExecutionConfig,
    ScenarioRuntime,
    train_phase4_seed,
)
from sparse_additive import solve_sparse_additive_bellman


OUTPUT_ROOT = (
    project_paths.CORE_DIR / "figure" / "phase_4_extended_random_boxes" / "training_v1"
)
CHECKPOINT_ROOT = (
    project_paths.DQN_DIR / "checkpoints" / "phase4_extended_random_boxes" / "training_v1"
)
RUN_ROOT = (
    project_paths.DQN_DIR / "runs" / "phase4_extended_random_boxes" / "training_v1"
)
REFERENCE_SCHEMA = "p1b-phase4-extended-filtered-bellman-reference-v1"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _source_contract_hash() -> str:
    digest = hashlib.sha256()
    for directory, names in (
        (project_paths.CORE_DIR, (
            "sparse_additive.py", "sparse_reachability.py", "bellman_geometry.py",
            "edge_hazard.py", "P1b_RL_approximation.py",
        )),
        (project_paths.DQN_DIR, (
            "phase4_extended_dataset.py", "phase4_extended_scene.py",
            "extended_failure_env.py", "multiscale_observation.py",
        )),
    ):
        for name in names:
            digest.update(name.encode("utf-8"))
            digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def filtered_bellman_reference(
    problem: ExpandedFailureAttackerBRProblem,
    scenario_id: str,
) -> BellmanReference:
    """Exact Bellman BR over switching candidates passing powered occlusion."""

    started = perf_counter()
    starts = problem.switching_state_ids()
    if not starts:
        return BellmanReference(
            scenario_id=scenario_id,
            feasible=False,
            attacker_objective=None,
            switching_state_id=None,
            trajectory=(),
            runtime_sec=perf_counter() - started,
            timing={
                "T_total_s": perf_counter() - started,
                "mission_failure_no_powered_occluded_switch": 1.0,
            },
        )
    scene = problem.scene
    solve_started = perf_counter()
    sparse = solve_sparse_additive_bellman(
        scene.graph,
        starts,
        problem.mdp.hazard_field,
        objective_parameters=scene.config.attacker_objective,
        parameters=scene.config.glider,
        physical_scale=scene.config.physical_scale,
        quadrature_resolution=scene.config.discretization.hazard_quadrature_resolution,
    )
    solve_runtime = perf_counter() - solve_started
    candidates: list[tuple[float, int, tuple[int, ...]]] = []
    for start in starts:
        if not sparse.solution.goal_reachable[start]:
            continue
        edges = sparse.solution.path_edges(start)
        trajectory = (int(start),) + tuple(int(edge.target_id) for edge in edges)
        objective = problem.powered_cost(start) + float(sparse.solution.value[start])
        candidates.append((float(objective), int(start), trajectory))
    runtime = perf_counter() - started
    if not candidates:
        return BellmanReference(
            scenario_id=scenario_id,
            feasible=False,
            attacker_objective=None,
            switching_state_id=None,
            trajectory=(),
            runtime_sec=runtime,
            timing={"T_total_s": runtime, "T_sparse_bellman_s": solve_runtime},
        )
    objective, switching, trajectory = min(candidates, key=lambda item: (item[0], item[1]))
    return BellmanReference(
        scenario_id=scenario_id,
        feasible=True,
        attacker_objective=objective,
        switching_state_id=switching,
        trajectory=trajectory,
        runtime_sec=runtime,
        timing={
            "T_total_s": runtime,
            "T_sparse_bellman_s": solve_runtime,
            "active_state_count": float(sparse.metrics.active_state_count),
            "active_edge_count": float(sparse.metrics.active_edge_count),
        },
    )


def _load_reference_cache(
    path: Path,
    *,
    dataset_hash: str,
    condition: dict[str, float],
    source_hash: str,
) -> dict[str, BellmanReference]:
    if not path.exists():
        return {}
    values = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": REFERENCE_SCHEMA,
        "dataset_manifest_sha256": dataset_hash,
        "condition": condition,
        "source_contract_sha256": source_hash,
    }
    if any(values.get(key) != value for key, value in expected.items()):
        return {}
    return {
        item["scenario_id"]: BellmanReference.from_dict(item)
        for item in values.get("references", [])
    }


def _save_reference_cache(
    path: Path,
    references: dict[str, BellmanReference],
    *,
    dataset_hash: str,
    condition: dict[str, float],
    source_hash: str,
) -> None:
    _write_json(path, {
        "schema": REFERENCE_SCHEMA,
        "dataset_manifest_sha256": dataset_hash,
        "condition": condition,
        "source_contract_sha256": source_hash,
        "references": [references[key].as_dict() for key in sorted(references)],
    })


def build_extended_runtimes(
    specs: Iterable[ExtendedScenarioSpec],
    *,
    pilot_config: ExtendedPilotConfig,
    spatial_resolution_m: float,
    heading_spacing_deg: float,
    dataset_hash: str,
    reference_path: Path,
    observation_device: str | torch.device | None = None,
) -> tuple[list[ScenarioRuntime], list[dict[str, Any]], dict[str, Any]]:
    """Build shared terrain-goal graphs, references, and trainable contexts."""

    specs = tuple(specs)
    _, terrain_specs = build_extended_scenario_bank(pilot_config)
    condition = {
        "spatial_resolution_m": float(spatial_resolution_m),
        "heading_spacing_deg": float(heading_spacing_deg),
    }
    source_hash = _source_contract_hash()
    references = _load_reference_cache(
        reference_path,
        dataset_hash=dataset_hash,
        condition=condition,
        source_hash=source_hash,
    )
    scenes: dict[str, Any] = {}
    surface_caches: dict[str, OrderedDict[tuple[float, float], float]] = {}
    row_caches: dict[str, SharedActionRowCache] = {}
    contexts: list[ScenarioRuntime] = []
    mission_failures: list[dict[str, Any]] = []
    compatibility: CompatibilitySignature | None = None
    observation_config = MultiscaleObservationConfig()
    for ordinal, spec in enumerate(specs, start=1):
        key = spec.graph_cache_key
        scene = scenes.get(key)
        if scene is None:
            scene = build_extended_scene(
                spec,
                terrain_specs[spec.terrain_category],
                spatial_resolution_m=spatial_resolution_m,
                heading_spacing_deg=heading_spacing_deg,
                pilot_config=pilot_config,
            )
            scenes[key] = scene
            surface_caches[key] = OrderedDict()
            row_caches[key] = SharedActionRowCache(max_entries=50_000)
        problem = ExpandedFailureAttackerBRProblem(scene, spec.sensor_map)
        starts = problem.switching_state_ids()
        reference = references.get(spec.scenario_id)
        if reference is None:
            print(
                f"Extended Bellman reference {ordinal}/{len(specs)}: {spec.scenario_id}",
                flush=True,
            )
            reference = filtered_bellman_reference(problem, spec.scenario_id)
            references[spec.scenario_id] = reference
            _save_reference_cache(
                reference_path,
                references,
                dataset_hash=dataset_hash,
                condition=condition,
                source_hash=source_hash,
            )
        if not starts or not reference.feasible:
            mission_failures.append({
                "scenario_id": spec.scenario_id,
                "split": spec.split,
                "terrain_category": spec.terrain_category,
                "visibility_class": spec.visibility_class,
                "reason": "no_powered_occluded_goal_reachable_switching_candidate",
            })
            continue
        builder = MultiscaleTerrainObservationBuilder(
            problem,
            observation_config,
            surface_height_cache=surface_caches[key],
            max_surface_cache_entries=200_000,
        )
        catalog = ExpandedFailureHeadingActionCatalog(
            problem, row_cache=row_caches[key],
        )
        signature = CompatibilitySignature.from_components(problem, builder, catalog)
        if compatibility is None:
            compatibility = signature
        elif signature != compatibility:
            raise RuntimeError("extended observation/action compatibility drifted")
        batch_builder = None
        if observation_device is not None and torch.device(observation_device).type == "cuda":
            batch_builder = ExactBatchedAABBGPUObservationMultiscaleBuilder(
                builder, observation_device,
            )
        contexts.append(ScenarioRuntime(
            spec=spec,  # type: ignore[arg-type]
            problem=problem,
            builder=builder,  # type: ignore[arg-type]
            catalog=catalog,
            starts=starts,
            cache=ObservationTensorCache(
                builder,  # type: ignore[arg-type]
                max_entries=2_048,
                batch_builder=batch_builder,
            ),
            reference=reference,
        ))
    if compatibility is None:
        raise RuntimeError("extended bank contains no trainable scenario")
    metadata = {
        "condition": condition,
        "compatibility": compatibility.as_dict(),
        "graph_cache_count": len(scenes),
        "scenario_count_requested": len(specs),
        "scenario_count_trainable": len(contexts),
        "mission_failure_count": len(mission_failures),
        "mission_failures": mission_failures,
        "reference_cache": str(reference_path),
        "source_contract_sha256": source_hash,
    }
    return contexts, mission_failures, metadata


def run_extended_training(
    *,
    smoke: bool = False,
    resume: bool = True,
    device: str = "cuda",
    require_cuda: bool = True,
    spatial_resolution_m: float = 25.0,
    heading_spacing_deg: float = 5.0,
    prepare_only: bool = False,
    selected_seeds: tuple[int, ...] | None = None,
) -> Path:
    requested_device = torch.device(device).type
    if not prepare_only and requested_device != "cuda":
        raise ValueError(
            "extended DQN smoke and official training require device='cuda'; "
            "CPU is supported only for Bellman preparation"
        )
    if not prepare_only and not torch.cuda.is_available():
        raise RuntimeError("extended DQN smoke and official training require CUDA")
    if (
        prepare_only and require_cuda and requested_device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA preparation was requested but CUDA is unavailable")
    pilot_config = ExtendedPilotConfig(
        train_terrain_count=1 if smoke else 4,
        validation_terrain_count=1 if smoke else 2,
    )
    manifest = extended_dataset_manifest(pilot_config)
    splits, _ = build_extended_scenario_bank(pilot_config)
    tag = f"dx{float(spatial_resolution_m):g}m_dpsi{float(heading_spacing_deg):g}deg"
    run_label = "smoke_v1" if smoke else "official_v1"
    output_dir = OUTPUT_ROOT / tag / run_label
    checkpoint_dir = CHECKPOINT_ROOT / tag / run_label
    run_dir = RUN_ROOT / tag / run_label
    for path in (output_dir, checkpoint_dir, run_dir):
        path.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "extended_dataset_manifest.json", manifest)
    reference_path = output_dir / "extended_bellman_references.json"
    all_specs = (*splits[TRAIN_EXTENDED], *splits[VALIDATION_EXTENDED])
    contexts, mission_failures, runtime_metadata = build_extended_runtimes(
        all_specs,
        pilot_config=pilot_config,
        spatial_resolution_m=spatial_resolution_m,
        heading_spacing_deg=heading_spacing_deg,
        dataset_hash=manifest["manifest_sha256"],
        reference_path=reference_path,
        observation_device=(device if device == "cuda" else None),
    )
    valid_references = [
        context.reference.attacker_objective
        for context in contexts
        if context.reference.attacker_objective is not None
    ]
    failure_terminal_cost = FailureCostConfig().from_reference_objectives(
        valid_references,
    )
    for context in contexts:
        context.failure_terminal_cost = failure_terminal_cost
    preparation = {
        "schema": "p1b-phase4-extended-training-preparation-v1",
        "status": "prepared" if prepare_only else "training_ready",
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "runtime_metadata": runtime_metadata,
        "mission_failures": mission_failures,
        "failure_terminal_cost": failure_terminal_cost,
        "failure_terminal_cost_definition": (
            "max finite filtered Bellman J_A plus max(1.0, 10% of its magnitude)"
        ),
        "early_stopping": EarlyStoppingConfig().as_dict(),
        "scope": "no training run" if prepare_only else "training requested",
    }
    preparation_path = output_dir / "extended_training_preparation.json"
    _write_json(preparation_path, preparation)
    if prepare_only:
        return preparation_path

    training_contexts = [
        context for context in contexts if context.spec.split == TRAIN_EXTENDED
    ]
    if not training_contexts:
        raise RuntimeError("no trainable extended training scenarios remain")
    evaluation_contexts = list(contexts)
    shapes = training_contexts[0].builder.tensor_ready_shapes
    network_config = DQNNetworkConfig(
        spatial_channels=shapes["spatial"][0],
        spatial_height=shapes["spatial"][1],
        spatial_width=shapes["spatial"][2],
        scalar_features=shapes["scalar"][0],
        action_count=training_contexts[0].catalog.action_count,
        architecture_id="p1b-two-branch-dqn-multiscale-fused-v1",
    )
    training_config = DQNTrainingConfig(
        episodes=6 if smoke else 60_000,
        replay_capacity=128 if smoke else 50_000,
        replay_warmup=4 if smoke else 2_000,
        batch_size=4 if smoke else 128,
        target_update_steps=8 if smoke else 1_000,
        evaluation_interval_episodes=3 if smoke else 1_500,
        seeds=(0,) if smoke else (0, 1, 2),
        device=device,
        config_id=(
            "p1b-phase4-extended-smoke-v1"
            if smoke else "p1b-phase4-extended-training-v1"
        ),
    )
    run_seeds = (
        training_config.seeds
        if selected_seeds is None
        else tuple(int(seed) for seed in selected_seeds)
    )
    if not run_seeds or len(set(run_seeds)) != len(run_seeds):
        raise ValueError("selected seeds must be a nonempty unique sequence")
    unsupported_seeds = sorted(set(run_seeds) - set(training_config.seeds))
    if unsupported_seeds:
        raise ValueError(
            f"selected seeds are outside the approved configuration: {unsupported_seeds}"
        )
    execution_config = Phase4ExecutionConfig(
        episode_batch_size=2 if smoke else 8,
        optimization_prefetch_batches=2 if smoke else 8,
        optimizer_updates_per_vector_step=1,
        checkpoint_interval_episodes=3 if smoke else 1_500,
        evaluation_interval_episodes=3 if smoke else 1_500,
        execution_id=(
            "p1b-phase4-extended-smoke-execution-v1"
            if smoke else "p1b-phase4-extended-execution-v1"
        ),
    )
    early_stopping = (
        None if smoke else EarlyStoppingConfig()
    )
    model_contract = {
        "dataset_schema": manifest["schema"],
        "observation": MultiscaleObservationConfig().as_dict(),
        "failure_terminal_cost": failure_terminal_cost,
        "failure_action_space": "all physically feasible glide edges",
        "powered_occlusion": "all 5 m-spaced interior powered-segment samples blocked",
        "goal_terminal_radius_m": pilot_config.goal_terminal_radius_m,
    }
    results = []
    compatibility = CompatibilitySignature.from_dict(
        runtime_metadata["compatibility"]
    )
    for seed in run_seeds:
        results.append(train_phase4_seed(
            training_contexts,
            evaluation_contexts,
            network_config,
            training_config,
            compatibility,
            manifest["manifest_sha256"],
            seed,
            checkpoint_dir=checkpoint_dir,
            run_dir=run_dir,
            resume=resume,
            model_contract=model_contract,
            execution_config=execution_config,
            early_stopping_config=early_stopping,
        ))
    summary_suffix = (
        "" if run_seeds == training_config.seeds
        else "_seeds_" + "_".join(str(seed) for seed in run_seeds)
    )
    summary_path = output_dir / f"extended_training_summary{summary_suffix}.json"
    _write_json(summary_path, {
        **preparation,
        "status": "completed",
        "network_config": network_config.as_dict(),
        "training_config": training_config.as_dict(),
        "executed_seeds": list(run_seeds),
        "execution_config": execution_config.as_dict(),
        "seed_results": results,
    })
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda",
        help="CPU is accepted only with --prepare-only; DQN runs require CUDA",
    )
    parser.add_argument("--spatial-resolution-m", type=float, default=25.0)
    parser.add_argument("--heading-spacing-deg", type=float, default=5.0)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        help="approved seeds to execute; omit to run every configured seed",
    )
    arguments = parser.parse_args()
    path = run_extended_training(
        smoke=arguments.smoke,
        prepare_only=arguments.prepare_only,
        resume=not arguments.no_resume,
        device=arguments.device,
        require_cuda=True,
        spatial_resolution_m=arguments.spatial_resolution_m,
        heading_spacing_deg=arguments.heading_spacing_deg,
        selected_seeds=(None if arguments.seeds is None else tuple(arguments.seeds)),
    )
    print(path)


if __name__ == "__main__":
    main()


__all__ = [
    "build_extended_runtimes", "filtered_bellman_reference",
    "run_extended_training",
]

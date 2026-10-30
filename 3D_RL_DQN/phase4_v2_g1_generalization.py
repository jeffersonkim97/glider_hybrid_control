"""G1 terrain-generalization pilot for the extended Phase 16.4 dataset.

G1 freezes the stable Run-A learner while restoring the actual experiment:
one policy is trained across several random-box terrains and selected only by
performance on held-out random-box terrains.  The final out-of-family mesh or
height-map test remains sealed for the later test stage.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

import torch

import project_paths
from dqn_action_catalog import HeadingActionCatalog, SharedActionRowCache
from dqn_env import ObservationTensorCache
from dqn_model import DQNNetworkConfig
from dqn_training import CompatibilitySignature, DQNTrainingConfig
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
from phase4_extended_training import (
    _load_reference_cache,
    _save_reference_cache,
    _source_contract_hash,
    filtered_bellman_reference,
)
from phase4_multi_terrain import (
    BellmanReference,
    Phase4ExecutionConfig,
    ScenarioRuntime,
    train_phase4_seed,
)
from phase4_v2_run_a import RunAGoalReachableProblem


G1_SCHEMA = "p1b-phase4-training-v2-g1-generalization-v1"
CANONICAL_REFERENCE_PATH = (
    project_paths.CORE_DIR
    / "figure"
    / "phase_4_extended_random_boxes"
    / "training_v1"
    / "dx25m_dpsi5deg"
    / "official_v1"
    / "extended_bellman_references.json"
)
OUTPUT_ROOT = (
    project_paths.CORE_DIR
    / "figure"
    / "phase_4_extended_random_boxes"
    / "training_v2"
    / "g1_generalization"
)
CHECKPOINT_ROOT = (
    project_paths.DQN_DIR
    / "checkpoints"
    / "phase4_extended_random_boxes"
    / "training_v2"
    / "g1_generalization"
)
RUN_ROOT = (
    project_paths.DQN_DIR
    / "runs"
    / "phase4_extended_random_boxes"
    / "training_v2"
    / "g1_generalization"
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_g1_runtimes(
    specs: Iterable[ExtendedScenarioSpec],
    *,
    pilot_config: ExtendedPilotConfig,
    spatial_resolution_m: float,
    heading_spacing_deg: float,
    dataset_hash: str,
    reference_path: Path,
    observation_device: str | torch.device = "cuda",
) -> tuple[list[ScenarioRuntime], list[dict[str, Any]], dict[str, Any]]:
    """Build G1 contexts with powered occlusion and goal-reachable actions."""

    specs = tuple(specs)
    if torch.device(observation_device).type != "cuda":
        raise ValueError("G1 observation generation must use CUDA")
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
        graph_key = spec.graph_cache_key
        scene = scenes.get(graph_key)
        if scene is None:
            scene = build_extended_scene(
                spec,
                terrain_specs[spec.terrain_category],
                spatial_resolution_m=spatial_resolution_m,
                heading_spacing_deg=heading_spacing_deg,
                pilot_config=pilot_config,
            )
            scenes[graph_key] = scene
            surface_caches[graph_key] = OrderedDict()
            row_caches[graph_key] = SharedActionRowCache(max_entries=50_000)

        problem = RunAGoalReachableProblem(scene, spec.sensor_map)
        starts = problem.switching_state_ids()
        reference = references.get(spec.scenario_id)
        if reference is None:
            print(
                f"G1 Bellman reference {ordinal}/{len(specs)}: {spec.scenario_id}",
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
            surface_height_cache=surface_caches[graph_key],
            max_surface_cache_entries=200_000,
        )
        catalog = HeadingActionCatalog(
            problem, row_cache=row_caches[graph_key],
        )
        signature = CompatibilitySignature.from_components(
            problem, builder, catalog,
        )
        if compatibility is None:
            compatibility = signature
        elif signature != compatibility:
            raise RuntimeError("G1 observation/action compatibility drifted")
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
                max_entries=4_096,
                batch_builder=batch_builder,
            ),
            reference=reference,
            failure_terminal_cost=0.0,
        ))

    if compatibility is None:
        raise RuntimeError("G1 bank contains no trainable scenario")
    metadata = {
        "schema": G1_SCHEMA,
        "condition": condition,
        "compatibility": compatibility.as_dict(),
        "graph_cache_count": len(scenes),
        "scenario_count_requested": len(specs),
        "scenario_count_trainable": len(contexts),
        "training_scenario_count": sum(
            context.spec.split == TRAIN_EXTENDED for context in contexts
        ),
        "validation_scenario_count": sum(
            context.spec.split == VALIDATION_EXTENDED for context in contexts
        ),
        "mission_failure_count": len(mission_failures),
        "mission_failures": mission_failures,
        "reference_cache": str(reference_path),
        "source_contract_sha256": source_hash,
        "action_space": "exact_goal_backward_reachable_adjacency",
        "powered_occlusion_required": True,
        "observation_backend": "exact_batched_aabb_cuda",
    }
    return contexts, mission_failures, metadata


def run_g1(
    *,
    smoke: bool = False,
    prepare_only: bool = False,
    resume: bool = True,
    episodes: int = 10_000,
    evaluation_interval_episodes: int = 1_000,
) -> Path:
    """Prepare or execute the 25 m, seed-0 G1 generalization pilot."""

    if not torch.cuda.is_available():
        raise RuntimeError("G1 requires CUDA for observation generation and DQN")
    episode_count = 16 if smoke else int(episodes)
    evaluation_interval = 8 if smoke else int(evaluation_interval_episodes)
    if episode_count < 1 or episode_count % 8 != 0:
        raise ValueError("G1 episodes must be positive and divisible by batch size 8")
    if evaluation_interval < 1 or evaluation_interval % 8 != 0:
        raise ValueError("G1 evaluation interval must be divisible by batch size 8")
    if episode_count % evaluation_interval != 0:
        raise ValueError("G1 episodes must be divisible by evaluation interval")
    if not smoke and episode_count < 10_000:
        raise ValueError("official G1 retains the approved 10,000-episode minimum")

    spatial_resolution_m = 25.0
    heading_spacing_deg = 5.0
    pilot_config = ExtendedPilotConfig(
        train_terrain_count=4,
        validation_terrain_count=2,
    )
    manifest = extended_dataset_manifest(pilot_config)
    splits, _ = build_extended_scenario_bank(pilot_config)
    tag = "dx25m_dpsi5deg"
    run_label = "smoke_v1" if smoke else f"seed0_ep{episode_count}_v1"
    output_dir = OUTPUT_ROOT / tag / run_label
    checkpoint_dir = CHECKPOINT_ROOT / tag / run_label
    run_dir = RUN_ROOT / tag / run_label
    for path in (output_dir, checkpoint_dir, run_dir):
        path.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "extended_dataset_manifest.json", manifest)

    all_specs = (*splits[TRAIN_EXTENDED], *splits[VALIDATION_EXTENDED])
    contexts, mission_failures, runtime_metadata = build_g1_runtimes(
        all_specs,
        pilot_config=pilot_config,
        spatial_resolution_m=spatial_resolution_m,
        heading_spacing_deg=heading_spacing_deg,
        dataset_hash=manifest["manifest_sha256"],
        reference_path=CANONICAL_REFERENCE_PATH,
        observation_device="cuda",
    )
    training_contexts = [
        context for context in contexts if context.spec.split == TRAIN_EXTENDED
    ]
    validation_contexts = [
        context for context in contexts if context.spec.split == VALIDATION_EXTENDED
    ]
    if not training_contexts or not validation_contexts:
        raise RuntimeError("G1 requires nonempty train and held-out validation splits")

    preparation = {
        "schema": G1_SCHEMA,
        "status": "prepared" if prepare_only else "training_ready",
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "runtime_metadata": runtime_metadata,
        "mission_failures": mission_failures,
        "split_contract": {
            "training": "four random-box terrain seeds; policy optimization only",
            "validation": (
                "two disjoint random-box terrain seeds; checkpoint selection and "
                "early stopping only"
            ),
            "final_test": (
                "sealed and not instantiated in G1; reserved for unseen mesh or "
                "height-map evaluation"
            ),
        },
        "algorithm": {
            "name": "1-step Double DQN",
            "learning_rate": 3.0e-4,
            "target_update_optimizer_steps": 500,
            "replay_sampling": "uniform",
            "n_step_return": 1,
        },
        "episodes": episode_count,
        "evaluation_interval_episodes": evaluation_interval,
        "seed": 0,
        "device": "cuda",
        "scope": "configuration only" if prepare_only else "seed-0 G1 pilot",
    }
    preparation_path = output_dir / "g1_preparation.json"
    _write_json(preparation_path, preparation)
    if prepare_only:
        return preparation_path

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
        episodes=episode_count,
        replay_capacity=256 if smoke else 50_000,
        replay_warmup=32 if smoke else 2_000,
        batch_size=16 if smoke else 128,
        learning_rate=3.0e-4,
        target_update_steps=16 if smoke else 500,
        evaluation_interval_episodes=evaluation_interval,
        seeds=(0,),
        device="cuda",
        config_id=(
            "p1b-phase4-training-v2-g1-smoke-v1"
            if smoke else "p1b-phase4-training-v2-g1-seed0-v1"
        ),
    )
    execution_config = Phase4ExecutionConfig(
        episode_batch_size=8,
        optimization_prefetch_batches=2 if smoke else 8,
        optimizer_updates_per_vector_step=1,
        checkpoint_interval_episodes=evaluation_interval,
        evaluation_interval_episodes=evaluation_interval,
        target_sync_basis="optimization_steps",
        td_target_estimator="double_dqn",
        n_step_return=1,
        replay_sampling="uniform",
        execution_id=(
            "p1b-phase4-training-v2-g1-smoke-execution-v1"
            if smoke else "p1b-phase4-training-v2-g1-seed0-execution-v1"
        ),
    )
    early_stopping = None if smoke else EarlyStoppingConfig(
        evaluation_interval_episodes=evaluation_interval,
        minimum_episodes=10_000,
        stability_evaluations=3,
        median_error_range_threshold=0.10,
        median_relative_error_threshold=0.10,
        maximum_relative_error_threshold=0.20,
        maximum_episodes=episode_count,
    )
    compatibility = CompatibilitySignature.from_dict(
        runtime_metadata["compatibility"]
    )
    model_contract = {
        **preparation,
        "status": "training_contract",
        "observation": MultiscaleObservationConfig().as_dict(),
        "powered_occlusion": "required at 5 m-spaced interior samples",
        "glide_action_space": "exact_goal_backward_reachable_adjacency",
        "failure_terminal_cost": 0.0,
        "early_stopping": (
            None if early_stopping is None else early_stopping.as_dict()
        ),
    }
    result = train_phase4_seed(
        training_contexts,
        contexts,
        network_config,
        training_config,
        compatibility,
        manifest["manifest_sha256"],
        0,
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        resume=resume,
        model_contract=model_contract,
        execution_config=execution_config,
        early_stopping_config=early_stopping,
    )
    summary_path = output_dir / "g1_training_summary.json"
    _write_json(summary_path, {
        **preparation,
        "status": "completed",
        "network_config": network_config.as_dict(),
        "training_config": training_config.as_dict(),
        "execution_config": execution_config.as_dict(),
        "early_stopping": (
            None if early_stopping is None else early_stopping.as_dict()
        ),
        "seed_result": result,
    })
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--evaluation-interval-episodes", type=int, default=1_000)
    arguments = parser.parse_args()
    path = run_g1(
        smoke=arguments.smoke,
        prepare_only=arguments.prepare_only,
        resume=not arguments.no_resume,
        episodes=arguments.episodes,
        evaluation_interval_episodes=arguments.evaluation_interval_episodes,
    )
    print(path)


if __name__ == "__main__":
    main()


__all__ = ["build_g1_runtimes", "run_g1"]

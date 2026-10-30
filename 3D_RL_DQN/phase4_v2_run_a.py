"""Phase 16.4 training-v2 Run A: one-scenario constrained upper bound.

Run A is a diagnostic curriculum stage, not the final generalized policy.  It
keeps the extended random-box mission, multiscale observation, powered-leg
occlusion requirement, and Bellman objective, while exposing only the existing
goal-backward-reachable glide adjacency.  This establishes whether the DQN can
learn J_A before failure branches and scenario diversity are introduced.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import project_paths
from attacker_br_problem import AttackerBRProblem
from dqn_action_catalog import HeadingActionCatalog, SharedActionRowCache
from dqn_env import ObservationTensorCache
from dqn_model import DQNNetworkConfig
from dqn_training import CompatibilitySignature, DQNTrainingConfig
from extended_failure_env import ExpandedFailureAttackerBRProblem
from multiscale_observation import (
    ExactBatchedAABBGPUObservationMultiscaleBuilder,
    MultiscaleObservationConfig,
    MultiscaleTerrainObservationBuilder,
)
from phase4_extended_dataset import (
    TRAIN_EXTENDED,
    VALIDATION_EXTENDED,
    ExtendedPilotConfig,
    build_extended_scenario_bank,
    extended_dataset_manifest,
)
from phase4_extended_scene import build_extended_scene
from phase4_early_stopping import EarlyStoppingConfig
from phase4_multi_terrain import (
    BellmanReference,
    Phase4ExecutionConfig,
    ScenarioRuntime,
    train_phase4_seed,
)


RUN_A_SCHEMA = "p1b-phase4-training-v2-run-a-target-sync-v2"
REFERENCE_PATH = (
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
    / "run_a_upper_bound"
)
CHECKPOINT_ROOT = (
    project_paths.DQN_DIR
    / "checkpoints"
    / "phase4_extended_random_boxes"
    / "training_v2"
    / "run_a_upper_bound"
)
RUN_ROOT = (
    project_paths.DQN_DIR
    / "runs"
    / "phase4_extended_random_boxes"
    / "training_v2"
    / "run_a_upper_bound"
)


class RunAGoalReachableProblem(ExpandedFailureAttackerBRProblem):
    """Keep powered occlusion but restore the exact goal-reachable glide graph."""

    def successors(self, state_id: int) -> tuple[Any, ...]:
        return AttackerBRProblem.successors(self, int(state_id))

    def transition(self, state_id: int, action_index: int) -> Any:
        return AttackerBRProblem.transition(self, int(state_id), int(action_index))

    def evaluate_trajectory(
        self, state_ids: tuple[int, ...], *, reached_goal: bool | None = None,
    ) -> Any:
        return AttackerBRProblem.evaluate_trajectory(
            self, state_ids, reached_goal=reached_goal,
        )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _reference_for(scenario_id: str, manifest_hash: str) -> BellmanReference:
    values = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    if values.get("dataset_manifest_sha256") != manifest_hash:
        raise ValueError("Run A Bellman reference has an incompatible dataset hash")
    if values.get("condition") != {
        "spatial_resolution_m": 25.0,
        "heading_spacing_deg": 5.0,
    }:
        raise ValueError("Run A Bellman reference has an incompatible discretization")
    selected = next(
        (
            item for item in values.get("references", [])
            if item.get("scenario_id") == scenario_id
        ),
        None,
    )
    if selected is None or not selected.get("feasible"):
        raise ValueError("Run A scenario lacks a finite Bellman reference")
    return BellmanReference.from_dict(selected)


def _run_a_scenario(config: ExtendedPilotConfig) -> tuple[Any, Any]:
    splits, terrains = build_extended_scenario_bank(config)
    selected = next(
        spec for spec in splits[TRAIN_EXTENDED]
        if spec.terrain_category == "random_boxes_train_simple_000"
        and spec.visibility_class == "mixed"
    )
    return selected, terrains[selected.terrain_category]


def _contract_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_a(
    *,
    smoke: bool = False,
    stop_rule_test: bool = False,
    tuning_variant: str | None = None,
    double_dqn: bool = False,
    n_step_return: int = 1,
    altitude_stratified_replay: bool = False,
    resume: bool = True,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Run A requires CUDA")
    if tuning_variant not in {None, "b", "c", "d"}:
        raise ValueError("tuning_variant must be one of None, 'b', 'c', or 'd'")
    if int(n_step_return) < 1:
        raise ValueError("n_step_return must be positive")
    if int(n_step_return) != 1 and not double_dqn:
        raise ValueError("Run A n-step comparison requires Double DQN")
    if altitude_stratified_replay and not double_dqn:
        raise ValueError("Run A stratified replay comparison requires Double DQN")
    if altitude_stratified_replay and int(n_step_return) != 1:
        raise ValueError(
            "Run A comparison changes n-step return or replay sampling, not both"
        )
    if sum((
        bool(smoke), bool(stop_rule_test), tuning_variant is not None,
        bool(double_dqn),
    )) > 1:
        raise ValueError(
            "smoke, stop-rule test, tuning, and Double DQN modes are mutually "
            "exclusive"
        )

    pilot_config = ExtendedPilotConfig()
    full_manifest = extended_dataset_manifest(pilot_config)
    spec, terrain_spec = _run_a_scenario(pilot_config)
    scene = build_extended_scene(
        spec,
        terrain_spec,
        spatial_resolution_m=25.0,
        heading_spacing_deg=5.0,
        pilot_config=pilot_config,
    )
    problem = RunAGoalReachableProblem(scene, spec.sensor_map)
    starts = problem.switching_state_ids()
    if not starts:
        raise RuntimeError("Run A scenario has no powered-occluded switching state")

    reference = _reference_for(
        spec.scenario_id, full_manifest["manifest_sha256"],
    )
    observation_config = MultiscaleObservationConfig()
    builder = MultiscaleTerrainObservationBuilder(problem, observation_config)
    batch_builder = ExactBatchedAABBGPUObservationMultiscaleBuilder(
        builder, "cuda",
    )
    cache = ObservationTensorCache(
        builder, max_entries=2_048, batch_builder=batch_builder,
    )
    catalog = HeadingActionCatalog(
        problem, row_cache=SharedActionRowCache(max_entries=50_000),
    )
    compatibility = CompatibilitySignature.from_components(
        problem, builder, catalog,
    )
    training_context = ScenarioRuntime(
        spec=spec,
        problem=problem,
        builder=builder,
        catalog=catalog,
        starts=starts,
        cache=cache,
        reference=reference,
    )
    validation_spec = replace(
        spec,
        scenario_id=f"run_a_validation__{spec.scenario_id}",
        split=VALIDATION_EXTENDED,
    )
    validation_context = ScenarioRuntime(
        spec=validation_spec,
        problem=problem,
        builder=builder,
        catalog=catalog,
        starts=starts,
        cache=cache,
        reference=reference,
    )

    shapes = builder.tensor_ready_shapes
    network_config = DQNNetworkConfig(
        spatial_channels=shapes["spatial"][0],
        spatial_height=shapes["spatial"][1],
        spatial_width=shapes["spatial"][2],
        scalar_features=shapes["scalar"][0],
        action_count=catalog.action_count,
        architecture_id="p1b-two-branch-dqn-multiscale-fused-v1",
    )
    tuning = {
        "b": {"learning_rate": 3.0e-4, "target_update_steps": 500},
        "c": {"learning_rate": 1.0e-4, "target_update_steps": 1_000},
        "d": {"learning_rate": 1.0e-4, "target_update_steps": 500},
    }.get(tuning_variant)
    if double_dqn:
        tuning = {"learning_rate": 3.0e-4, "target_update_steps": 500}
    episodes = (
        16 if smoke else (3_000 if stop_rule_test or tuning else 2_000)
    )
    evaluation_interval = (
        8 if smoke else (1_000 if stop_rule_test else (500 if tuning else 250))
    )
    mode_id = (
        "smoke-target-sync-v2"
        if smoke
        else (
            "stop-rule-3000-v1"
            if stop_rule_test
            else (
                (
                    "double-dqn-b-altitude-stratified-v1"
                    if altitude_stratified_replay
                    else (
                        f"double-dqn-b-nstep{int(n_step_return)}-v1"
                        if int(n_step_return) != 1
                        else "double-dqn-b-v1"
                    )
                )
                if double_dqn
                else (f"tuning-{tuning_variant}-v1" if tuning else "target-sync-v2")
            )
        )
    )
    training_config = DQNTrainingConfig(
        episodes=episodes,
        replay_capacity=256 if smoke else 50_000,
        replay_warmup=32 if smoke else 2_000,
        batch_size=16 if smoke else 128,
        learning_rate=(
            3.0e-4 if tuning is None else float(tuning["learning_rate"])
        ),
        target_update_steps=(
            16 if smoke else (
                1_000 if tuning is None else int(tuning["target_update_steps"])
            )
        ),
        evaluation_interval_episodes=evaluation_interval,
        seeds=(0,),
        device="cuda",
        config_id=f"p1b-phase4-training-v2-run-a-{mode_id}",
    )
    execution_config = Phase4ExecutionConfig(
        episode_batch_size=8,
        optimization_prefetch_batches=8,
        optimizer_updates_per_vector_step=1,
        checkpoint_interval_episodes=evaluation_interval,
        evaluation_interval_episodes=evaluation_interval,
        target_sync_basis="optimization_steps",
        td_target_estimator="double_dqn" if double_dqn else "dqn",
        n_step_return=int(n_step_return),
        replay_sampling=(
            "altitude_stratified"
            if altitude_stratified_replay else "uniform"
        ),
        execution_id=f"p1b-phase4-training-v2-run-a-{mode_id}-execution",
    )
    early_stopping = (
        EarlyStoppingConfig(
            evaluation_interval_episodes=1_000,
            minimum_episodes=1,
            stability_evaluations=3,
            median_error_range_threshold=0.10,
            median_relative_error_threshold=0.10,
            maximum_relative_error_threshold=0.20,
            maximum_episodes=3_000,
        )
        if stop_rule_test else None
    )
    run_contract = {
        "schema": RUN_A_SCHEMA,
        "role": "constrained_upper_bound_diagnostic",
        "source_dataset_manifest_sha256": full_manifest["manifest_sha256"],
        "scenario": spec.as_dict(),
        "spatial_resolution_m": 25.0,
        "heading_spacing_deg": 5.0,
        "powered_occlusion_required": True,
        "glide_action_space": "exact_goal_backward_reachable_adjacency",
        "target_network_sync_basis": "optimization_steps",
        "observation": observation_config.as_dict(),
        "bellman_reference": reference.as_dict(),
        "switching_candidate_count": len(starts),
        "smoke": smoke,
    }
    if stop_rule_test:
        run_contract.update({
            "stop_rule_test": True,
            "early_stopping": early_stopping.as_dict(),
        })
    if tuning is not None and not double_dqn:
        run_contract.update({
            "tuning_variant": tuning_variant,
            "tuning_hyperparameters": tuning,
            "tuning_evaluation_interval_episodes": evaluation_interval,
            "tuning_early_stopping_enabled": False,
        })
    if double_dqn:
        run_contract.update({
            "algorithm": "double_dqn",
            "comparison_baseline": "tuning_b_v1",
            "comparison_hyperparameters": tuning,
            "comparison_evaluation_interval_episodes": evaluation_interval,
            "comparison_early_stopping_enabled": False,
            "n_step_return": int(n_step_return),
            "replay_sampling": (
                "altitude_stratified"
                if altitude_stratified_replay else "uniform"
            ),
        })
    dataset_hash = _contract_hash(run_contract)
    model_contract = {
        **run_contract,
        "run_contract_sha256": dataset_hash,
    }

    tag = "dx25m_dpsi5deg"
    label = (
        "smoke_target_sync_v2"
        if smoke
        else (
            "stop_rule_3000_v1"
            if stop_rule_test
            else (
                (
                    "double_dqn_b_altitude_stratified_v1"
                    if altitude_stratified_replay
                    else (
                        f"double_dqn_b_nstep{int(n_step_return)}_v1"
                        if int(n_step_return) != 1
                        else "double_dqn_b_v1"
                    )
                )
                if double_dqn
                else (f"tuning_{tuning_variant}_v1" if tuning else "target_sync_v2")
            )
        )
    )
    output_dir = OUTPUT_ROOT / tag / label
    checkpoint_dir = CHECKPOINT_ROOT / tag / label
    run_dir = RUN_ROOT / tag / label
    for path in (output_dir, checkpoint_dir, run_dir):
        path.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_a_manifest.json", model_contract)

    result = train_phase4_seed(
        [training_context],
        [training_context, validation_context],
        network_config,
        training_config,
        compatibility,
        dataset_hash,
        0,
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        resume=resume,
        model_contract=model_contract,
        execution_config=execution_config,
        early_stopping_config=early_stopping,
    )
    summary_path = output_dir / "run_a_summary.json"
    _write_json(summary_path, {
        "schema": RUN_A_SCHEMA,
        "status": "completed",
        "run_contract": model_contract,
        "seed_result": result,
    })
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-rule-test", action="store_true")
    parser.add_argument("--tuning-variant", choices=("b", "c", "d"))
    parser.add_argument("--double-dqn", action="store_true")
    parser.add_argument("--n-step-return", type=int, default=1)
    parser.add_argument("--altitude-stratified-replay", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()
    print(run_a(
        smoke=arguments.smoke,
        stop_rule_test=arguments.stop_rule_test,
        tuning_variant=arguments.tuning_variant,
        double_dqn=arguments.double_dqn,
        n_step_return=arguments.n_step_return,
        altitude_stratified_replay=arguments.altitude_stratified_replay,
        resume=not arguments.no_resume,
    ))


if __name__ == "__main__":
    main()


__all__ = ["RUN_A_SCHEMA", "RunAGoalReachableProblem", "run_a"]

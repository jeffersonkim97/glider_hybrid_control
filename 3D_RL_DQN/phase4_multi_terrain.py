"""Phase 16.4 multi-terrain training for one shared condition-specific DQN.

This module is an executable backend for the master notebook.  Importing it has
no training side effects.  The official run uses CUDA, three approved simple
training terrains, two held-out simple validation terrains, and the fixed
Phase 16.2 observation contract.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import gc
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

import project_paths
from P1b_Exact_Local_SSE import exact_best_response
from P1b_condition import ComputationCondition, build_scene
from aabb_gpu_observation import ExactBatchedAABBGPUObservationBuilder
from attacker_br_problem import AttackerBRProblem
from dqn_action_catalog import HeadingActionCatalog, SharedActionRowCache
from dqn_env import DQNAttackerEnv, ObservationTensorCache
from dqn_model import DQNNetworkConfig, TerrainDQN, masked_bootstrap_values
from dqn_training import (
    CompatibilitySignature,
    DQNTrainingConfig,
    PolicyEvaluation,
    epsilon_at_step,
    epsilon_greedy_action,
    evaluate_policy,
    seed_everything,
    select_device,
)
from phase4_dataset import (
    RESERVED_PHASE5_TEST,
    TRAIN_SIMPLE,
    VALIDATION_SIMPLE,
    Phase4ScenarioSpec,
    approved_dataset_manifest,
    specs_from_manifest,
)
from phase4_replay import MultiScenarioBatch, MultiScenarioReplayBuffer
from terrain_observation import ObservationConfig, TerrainObservationBuilder


PHASE4_CHECKPOINT_FORMAT = "p1b-phase4-generalized-dqn-v1"
PHASE4_OBSERVATION_CACHE_ENTRIES = 2_048
PHASE4_SURFACE_CACHE_ENTRIES_PER_TERRAIN = 200_000
PHASE4_ACTION_ROW_CACHE_ENTRIES_PER_TERRAIN = 50_000
PHASE4_EVALUATION_BATCH_SIZE = 1_024
PHASE4_TRAINING_EPISODE_BATCH_SIZE = 8
PHASE4_OPTIMIZATION_PREFETCH_BATCHES = 8
PHASE4_DEFERRED_EVALUATION_INTERVAL_EPISODES = 6_000
OUTPUT_ROOT = project_paths.CORE_DIR / "figure" / "phase_4_multi_terrain_dqn"
CHECKPOINT_ROOT = project_paths.DQN_DIR / "checkpoints" / "phase4_multi_terrain"
RUN_ROOT = project_paths.DQN_DIR / "runs" / "phase4_multi_terrain"
MANIFEST_NAME = "phase4_multi_terrain_manifest.json"


def bellman_contract_sha256() -> str:
    """Hash authoritative sources that make a cached Bellman reference valid."""

    source_names = (
        "P1b_Exact_Local_SSE.py", "P1b_condition.py", "sparse_additive.py",
        "bellman_graph.py", "detection_hazard.py", "edge_hazard.py",
    )
    digest = hashlib.sha256()
    for name in source_names:
        path = project_paths.CORE_DIR / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _condition_tag(spatial_resolution_m: float, heading_spacing_deg: float) -> str:
    def clean(value: float) -> str:
        return f"{float(value):g}".replace(".", "p")
    return f"dx{clean(spatial_resolution_m)}m_dpsi{clean(heading_spacing_deg)}deg"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(project_paths.WORKSPACE_ROOT.resolve()).as_posix()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_phase4_checkpoint(
    path: Path, *, expected_network: DQNNetworkConfig,
    expected_compatibility: CompatibilitySignature,
    expected_model_contract: dict[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[TerrainDQN, dict[str, Any]]:
    """Load a generalized checkpoint only when every scientific signature matches."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "format": PHASE4_CHECKPOINT_FORMAT,
        "network_config": expected_network.as_dict(),
        "compatibility": expected_compatibility.as_dict(),
        "model_contract": expected_model_contract,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Phase 16.4 checkpoint has incompatible {key}")
    network = TerrainDQN(expected_network)
    network.load_state_dict(payload["model_state_dict"])
    network.to(torch.device(device))
    network.eval()
    return network, payload


@dataclass(frozen=True)
class BellmanReference:
    scenario_id: str
    feasible: bool
    attacker_objective: float | None
    switching_state_id: int | None
    trajectory: tuple[int, ...]
    runtime_sec: float
    timing: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["trajectory"] = list(self.trajectory)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BellmanReference":
        copied = dict(value)
        copied["trajectory"] = tuple(int(item) for item in copied["trajectory"])
        return cls(**copied)


@dataclass
class ScenarioRuntime:
    spec: Phase4ScenarioSpec
    problem: AttackerBRProblem
    builder: TerrainObservationBuilder
    catalog: HeadingActionCatalog
    starts: tuple[int, ...]
    cache: ObservationTensorCache
    reference: BellmanReference


@dataclass
class Phase4History:
    episode: list[int] = field(default_factory=list)
    scenario_id: list[str] = field(default_factory=list)
    terrain_category: list[str] = field(default_factory=list)
    sensor_map: list[list[float]] = field(default_factory=list)
    episode_glide_return: list[float] = field(default_factory=list)
    episode_full_transformed_return: list[float] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)
    episode_reached_goal: list[bool] = field(default_factory=list)
    epsilon: list[float] = field(default_factory=list)
    optimization_step: list[int] = field(default_factory=list)
    td_loss: list[float] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Phase4History":
        return cls(**value)


@dataclass(frozen=True)
class Phase4ExecutionConfig:
    """Runtime scheduling knobs that do not change the Bellman/DQN contract."""

    episode_batch_size: int = PHASE4_TRAINING_EPISODE_BATCH_SIZE
    optimization_prefetch_batches: int = PHASE4_OPTIMIZATION_PREFETCH_BATCHES
    optimizer_updates_per_vector_step: int = 1
    checkpoint_interval_episodes: int = 1_500
    evaluation_interval_episodes: int = (
        PHASE4_DEFERRED_EVALUATION_INTERVAL_EPISODES
    )
    execution_id: str = "p1b-phase4-vectorized-episodes-v1"

    def __post_init__(self) -> None:
        if min(
            int(self.episode_batch_size),
            int(self.optimization_prefetch_batches),
            int(self.optimizer_updates_per_vector_step),
            int(self.checkpoint_interval_episodes),
            int(self.evaluation_interval_episodes),
        ) < 1:
            raise ValueError("Phase 16.4 execution counts must be positive")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["maximum_optimizer_updates_per_environment_step"] = (
            float(self.optimizer_updates_per_vector_step)
            / float(self.episode_batch_size)
        )
        result["target_network_sync_basis"] = (
            "environment steps, preserving the approved 1000-step cadence"
        )
        return result


@dataclass
class Phase4RuntimeProfile:
    """Cumulative wall-clock components, synchronized at CUDA boundaries."""

    action_observation_and_forward_sec: float = 0.0
    environment_transition_and_replay_sec: float = 0.0
    optimization_observation_transfer_sec: float = 0.0
    dqn_forward_backward_optimizer_sec: float = 0.0
    checkpoint_evaluation_sec: float = 0.0
    checkpoint_save_sec: float = 0.0
    post_training_evaluation_sec: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            key: float(value) for key, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "Phase4RuntimeProfile":
        if not value:
            return cls()
        allowed = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{
            key: float(item) for key, item in value.items() if key in allowed
        })


@dataclass
class _EpisodeSlot:
    episode_number: int
    scenario_index: int
    start_state_id: int
    state_id: int
    epsilon_at_start: float
    glide_return: float = 0.0
    steps: int = 0
    reached_goal: bool = False
    ended: bool = False


@dataclass(frozen=True)
class _PreparedOptimizationBatches:
    spatial: Tensor
    scalar: Tensor
    next_spatial: Tensor
    next_scalar: Tensor
    actions: Tensor
    rewards: Tensor
    terminal: Tensor
    next_masks: Tensor

    @property
    def count(self) -> int:
        return int(self.actions.shape[0])


def approved_phase4_training_config(
    *, device: str = "cuda", smoke: bool = False,
) -> DQNTrainingConfig:
    if smoke:
        return DQNTrainingConfig(
            episodes=6, replay_capacity=128, replay_warmup=4, batch_size=4,
            target_update_steps=8, evaluation_interval_episodes=3,
            seeds=(0,), device=device,
            config_id="p1b-phase4-smoke-v1",
        )
    return DQNTrainingConfig(
        episodes=60_000,
        evaluation_interval_episodes=1_500,
        seeds=(0, 1, 2),
        device=device,
        config_id="p1b-phase4-training-v1",
    )


def approved_phase4_execution_config(
    *, smoke: bool = False,
) -> Phase4ExecutionConfig:
    if smoke:
        return Phase4ExecutionConfig(
            episode_batch_size=2,
            optimization_prefetch_batches=2,
            optimizer_updates_per_vector_step=1,
            checkpoint_interval_episodes=3,
            evaluation_interval_episodes=3,
            execution_id="p1b-phase4-vectorized-episodes-smoke-v1",
        )
    return Phase4ExecutionConfig()


def _reference_cache(
    path: Path, dataset_hash: str, condition: ComputationCondition,
    solver_contract_hash: str,
) -> dict[str, BellmanReference]:
    if not path.exists():
        return {}
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("dataset_manifest_sha256") != dataset_hash:
        return {}
    if stored.get("condition") != condition.as_dict():
        return {}
    if stored.get("bellman_contract_sha256") != solver_contract_hash:
        return {}
    return {
        item["scenario_id"]: BellmanReference.from_dict(item)
        for item in stored.get("references", [])
    }


def _save_reference_cache(
    path: Path, dataset_hash: str, condition: ComputationCondition,
    solver_contract_hash: str, references: dict[str, BellmanReference],
) -> None:
    _write_json(path, {
        "schema": "p1b-phase4-bellman-reference-v1",
        "dataset_manifest_sha256": dataset_hash,
        "condition": condition.as_dict(),
        "bellman_contract_sha256": solver_contract_hash,
        "references": [
            references[key].as_dict() for key in sorted(references)
        ],
    })


def build_scenario_runtimes(
    specs: Iterable[Phase4ScenarioSpec],
    condition_template: ComputationCondition,
    *, reference_path: Path,
    dataset_hash: str,
    observation_cache_entries: int | None = None,
    surface_cache_entries_per_terrain: int | None = None,
    observation_device: str | torch.device | None = None,
) -> tuple[list[ScenarioRuntime], dict[str, Any]]:
    """Build one graph per terrain and one hazard-conditioned problem per scenario."""

    specs = tuple(specs)
    scenes: dict[str, Any] = {}
    surface_caches: dict[
        str, OrderedDict[tuple[float, float], float]
    ] = {}
    action_row_caches: dict[str, SharedActionRowCache] = {}
    solver_contract_hash = bellman_contract_sha256()
    references = _reference_cache(
        reference_path, dataset_hash, condition_template, solver_contract_hash,
    )
    runtimes: list[ScenarioRuntime] = []
    compatibility: CompatibilitySignature | None = None
    catalog_id: str | None = None
    for ordinal, spec in enumerate(specs, start=1):
        scene = scenes.get(spec.terrain_category)
        if scene is None:
            condition = ComputationCondition(
                spatial_resolution_m=condition_template.spatial_resolution_m,
                heading_spacing_deg=condition_template.heading_spacing_deg,
                r_neighbor=condition_template.r_neighbor,
                terrain_category=spec.terrain_category,
                hazard_weight=condition_template.hazard_weight,
                time_weight=condition_template.time_weight,
                defender_goal_margin_map=condition_template.defender_goal_margin_map,
            )
            scene = build_scene(condition)
            scenes[spec.terrain_category] = scene
            surface_caches[spec.terrain_category] = OrderedDict()
            action_row_caches[spec.terrain_category] = SharedActionRowCache(
                max_entries=PHASE4_ACTION_ROW_CACHE_ENTRIES_PER_TERRAIN,
            )
        problem = AttackerBRProblem(scene, spec.sensor_map)
        builder = TerrainObservationBuilder(
            problem, ObservationConfig(),
            surface_height_cache=surface_caches[spec.terrain_category],
            max_surface_cache_entries=surface_cache_entries_per_terrain,
        )
        catalog = HeadingActionCatalog(
            problem, row_cache=action_row_caches[spec.terrain_category],
        )
        starts = problem.switching_state_ids()
        if not starts:
            raise RuntimeError(f"scenario has no admissible switching states: {spec.scenario_id}")
        signature = CompatibilitySignature.from_components(problem, builder, catalog)
        if compatibility is None:
            compatibility = signature
            catalog_id = catalog.catalog_id
        elif signature != compatibility or catalog.catalog_id != catalog_id:
            raise RuntimeError(
                f"observation/action/discretization contract drifted at {spec.scenario_id}"
            )

        reference = references.get(spec.scenario_id)
        if reference is None:
            print(
                f"Phase 16.4 Bellman reference {ordinal}/{len(specs)}: "
                f"{spec.scenario_id}", flush=True,
            )
            started = perf_counter()
            oracle = exact_best_response(scene, spec.sensor_map, keep_solution=False)
            runtime = perf_counter() - started
            reference = BellmanReference(
                scenario_id=spec.scenario_id,
                feasible=bool(oracle.feasible),
                attacker_objective=(
                    None if oracle.attacker_objective is None
                    else float(oracle.attacker_objective)
                ),
                switching_state_id=(
                    None if oracle.switching_state_id is None
                    else int(oracle.switching_state_id)
                ),
                trajectory=tuple(int(value) for value in oracle.glide_state_ids),
                runtime_sec=runtime,
                timing={key: float(value) for key, value in oracle.timing.items()},
            )
            references[spec.scenario_id] = reference
            _save_reference_cache(
                reference_path, dataset_hash, condition_template,
                solver_contract_hash, references,
            )
        if not reference.feasible or reference.attacker_objective is None:
            raise RuntimeError(f"Bellman reference is infeasible: {spec.scenario_id}")
        batch_builder = None
        if observation_device is not None:
            selected_observation_device = torch.device(observation_device)
            if selected_observation_device.type == "cuda":
                batch_builder = ExactBatchedAABBGPUObservationBuilder(
                    builder, selected_observation_device,
                )
        runtimes.append(ScenarioRuntime(
            spec=spec, problem=problem, builder=builder, catalog=catalog,
            starts=starts,
            cache=ObservationTensorCache(
                builder, max_entries=observation_cache_entries,
                batch_builder=batch_builder,
            ),
            reference=reference,
        ))

    if compatibility is None:
        raise ValueError("at least one scenario is required")
    metadata = {
        "compatibility": compatibility.as_dict(),
        "action_catalog_id": catalog_id,
        "terrain_scene_count": len(scenes),
        "scenario_count": len(runtimes),
        "reference_cache_path": _relative(reference_path),
        "bellman_contract_sha256": solver_contract_hash,
        "observation_cache_entries_per_scenario": observation_cache_entries,
        "surface_cache_entries_per_terrain": surface_cache_entries_per_terrain,
        "action_row_cache_entries_per_terrain": (
            PHASE4_ACTION_ROW_CACHE_ENTRIES_PER_TERRAIN
        ),
        "observation_backend": (
            "exact_batched_aabb_cuda"
            if observation_device is not None
            and torch.device(observation_device).type == "cuda"
            else "scalar_cpu_reference"
        ),
    }
    return runtimes, metadata


def _observation_batch(
    contexts: list[ScenarioRuntime], scenario_indices: Iterable[int],
    state_ids: Iterable[int], device: torch.device,
) -> tuple[Tensor, Tensor]:
    context_indices = [int(value) for value in scenario_indices]
    ids = [int(value) for value in state_ids]
    observations: list[Any] = [None] * len(ids)
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for position, (context_index, state_id) in enumerate(zip(context_indices, ids)):
        grouped[context_index].append((position, state_id))
    for context_index, entries in grouped.items():
        built = contexts[context_index].cache.get_many(
            state_id for _, state_id in entries
        )
        for (position, _), observation in zip(entries, built):
            observations[position] = observation
    spatial = torch.from_numpy(np.stack(
        [item.spatial for item in observations], axis=0,
    )).to(device=device)
    scalar = torch.from_numpy(np.stack(
        [item.scalar for item in observations], axis=0,
    )).to(device=device)
    return spatial, scalar


def _single_q_values(
    network: TerrainDQN, context: ScenarioRuntime, state_id: int,
    device: torch.device,
) -> np.ndarray:
    observation = context.cache.get(int(state_id))
    spatial = torch.from_numpy(observation.spatial.copy())[None, ...].to(device)
    scalar = torch.from_numpy(observation.scalar.copy())[None, ...].to(device)
    with torch.inference_mode():
        return network(spatial, scalar)[0].detach().cpu().numpy()


def _q_values_batch(
    network: TerrainDQN, contexts: list[ScenarioRuntime],
    scenario_indices: Iterable[int], state_ids: Iterable[int],
    device: torch.device,
) -> np.ndarray:
    spatial, scalar = _observation_batch(
        contexts, scenario_indices, state_ids, device,
    )
    with torch.inference_mode():
        return network(spatial, scalar).detach().cpu().numpy()


def _feasible_masks_many(
    contexts: list[ScenarioRuntime], scenario_indices: Iterable[int],
    state_ids: Iterable[int],
) -> list[np.ndarray]:
    context_indices = [int(value) for value in scenario_indices]
    ids = [int(value) for value in state_ids]
    if len(context_indices) != len(ids):
        raise ValueError("scenario_indices and state_ids must have equal lengths")
    masks: list[np.ndarray | None] = [None] * len(ids)
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for position, (context_index, state_id) in enumerate(zip(context_indices, ids)):
        grouped[context_index].append((position, state_id))
    for context_index, entries in grouped.items():
        rows = contexts[context_index].catalog.rows_many(
            state_id for _, state_id in entries
        )
        for (position, _), row in zip(entries, rows):
            masks[position] = row.feasible_mask
    if any(mask is None for mask in masks):
        raise RuntimeError("failed to resolve a batched feasible-action mask")
    return [np.asarray(mask, dtype=np.bool_) for mask in masks]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_policy_batched(
    network: TerrainDQN, context: ScenarioRuntime, device: torch.device,
    *, batch_size: int = PHASE4_EVALUATION_BATCH_SIZE,
) -> PolicyEvaluation:
    """Evaluate every switching start in synchronized neural-network batches."""

    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    starts = tuple(int(value) for value in context.starts)
    states = list(starts)
    trajectories = [[state_id] for state_id in starts]
    reached = [context.problem.is_terminal(state_id) for state_id in starts]
    stopped = reached.copy()
    was_training = network.training
    network.eval()
    started = perf_counter()
    try:
        horizon = max(1, int(context.problem.grid.altitude_count) - 1)
        for _ in range(horizon):
            pending = [
                candidate_index
                for candidate_index in range(len(states))
                if not stopped[candidate_index]
            ]
            pending_rows = context.catalog.rows_many(
                states[candidate_index] for candidate_index in pending
            )
            active: list[int] = []
            masks: list[np.ndarray] = []
            active_rows: list[Any] = []
            for candidate_index, row in zip(pending, pending_rows):
                mask = row.feasible_mask
                if not mask.any():
                    stopped[candidate_index] = True
                    continue
                active.append(candidate_index)
                active_rows.append(row)
                masks.append(mask)
            if not active:
                break

            q_parts: list[np.ndarray] = []
            for offset in range(0, len(active), int(batch_size)):
                indices = active[offset:offset + int(batch_size)]
                observations = context.cache.get_many(
                    states[index] for index in indices
                )
                spatial = torch.from_numpy(np.stack(
                    [item.spatial for item in observations], axis=0,
                )).to(device=device)
                scalar = torch.from_numpy(np.stack(
                    [item.scalar for item in observations], axis=0,
                )).to(device=device)
                with torch.inference_mode():
                    q_parts.append(network(spatial, scalar).detach().cpu().numpy())
            q_values = np.concatenate(q_parts, axis=0)
            action_ids = np.argmax(
                np.where(np.stack(masks, axis=0), q_values, -np.inf), axis=1,
            )
            for row, candidate_index in enumerate(active):
                selected_action = int(action_ids[row])
                structural = active_rows[row]
                next_state_id = int(structural.target_state_ids[selected_action])
                if next_state_id < 0:
                    raise RuntimeError("masked DQN selected an infeasible action")
                states[candidate_index] = next_state_id
                trajectories[candidate_index].append(next_state_id)
                if bool(structural.terminal[selected_action]):
                    reached[candidate_index] = True
                    stopped[candidate_index] = True
    finally:
        network.train(was_training)

    candidates: list[tuple[float, int, tuple[int, ...], float, float]] = []
    for index, start_id in enumerate(starts):
        if not reached[index]:
            continue
        path = tuple(trajectories[index])
        scored = context.problem.evaluate_trajectory(path, reached_goal=True)
        powered = context.problem.powered_cost(start_id)
        candidates.append((powered + scored.cost, start_id, path, scored.cost, powered))
    runtime = perf_counter() - started
    if not candidates:
        return PolicyEvaluation(
            success=False, switching_state_id=None, trajectory=(),
            reached_goal=False, trajectory_feasible=False,
            attacker_objective=None, glide_cost=None, powered_cost=None,
            candidate_successes=0, candidate_count=len(starts),
            inference_runtime_sec=runtime, status="no_goal_reaching_candidate",
        )
    objective, start_id, trajectory, glide_cost, powered = min(
        candidates, key=lambda item: (item[0], item[1]),
    )
    return PolicyEvaluation(
        success=True, switching_state_id=start_id, trajectory=trajectory,
        reached_goal=True, trajectory_feasible=True,
        attacker_objective=float(objective), glide_cost=float(glide_cost),
        powered_cost=float(powered), candidate_successes=len(candidates),
        candidate_count=len(starts), inference_runtime_sec=runtime,
        status="success",
    )


def _optimize(
    policy: TerrainDQN, target: TerrainDQN,
    optimizer: torch.optim.Optimizer, batch: MultiScenarioBatch,
    contexts: list[ScenarioRuntime], device: torch.device,
    config: DQNTrainingConfig,
) -> float:
    spatial, scalar = _observation_batch(
        contexts, batch.scenario_indices, batch.state_ids, device,
    )
    next_spatial, next_scalar = _observation_batch(
        contexts, batch.scenario_indices, batch.next_state_ids, device,
    )
    actions = torch.from_numpy(batch.action_ids).to(device)
    rewards = torch.from_numpy(batch.rewards).to(device)
    terminal = torch.from_numpy(batch.terminal).to(device)
    next_masks = torch.from_numpy(batch.next_feasible_masks).to(device)
    predicted = policy(spatial, scalar).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_q = target(next_spatial, next_scalar)
        expected = rewards + config.gamma * masked_bootstrap_values(
            next_q, next_masks, terminal,
        )
    loss = F.smooth_l1_loss(predicted, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip_norm)
    optimizer.step()
    return float(loss.detach().cpu())


def _prepare_optimization_batches(
    batches: list[MultiScenarioBatch], contexts: list[ScenarioRuntime],
    device: torch.device,
) -> _PreparedOptimizationBatches:
    if not batches:
        raise ValueError("at least one replay batch is required")
    sizes = {len(batch.state_ids) for batch in batches}
    if len(sizes) != 1:
        raise ValueError("prefetched replay batches must have a common size")
    count = len(batches)
    batch_size = sizes.pop()
    scenario_indices = np.concatenate(
        [batch.scenario_indices for batch in batches], axis=0,
    )
    state_ids = np.concatenate([batch.state_ids for batch in batches], axis=0)
    next_state_ids = np.concatenate(
        [batch.next_state_ids for batch in batches], axis=0,
    )
    spatial, scalar = _observation_batch(
        contexts, scenario_indices, state_ids, device,
    )
    next_spatial, next_scalar = _observation_batch(
        contexts, scenario_indices, next_state_ids, device,
    )
    spatial_shape = tuple(spatial.shape[1:])
    scalar_shape = tuple(scalar.shape[1:])
    return _PreparedOptimizationBatches(
        spatial=spatial.reshape(count, batch_size, *spatial_shape),
        scalar=scalar.reshape(count, batch_size, *scalar_shape),
        next_spatial=next_spatial.reshape(count, batch_size, *spatial_shape),
        next_scalar=next_scalar.reshape(count, batch_size, *scalar_shape),
        actions=torch.from_numpy(np.stack(
            [batch.action_ids for batch in batches], axis=0,
        )).to(device),
        rewards=torch.from_numpy(np.stack(
            [batch.rewards for batch in batches], axis=0,
        )).to(device),
        terminal=torch.from_numpy(np.stack(
            [batch.terminal for batch in batches], axis=0,
        )).to(device),
        next_masks=torch.from_numpy(np.stack(
            [batch.next_feasible_masks for batch in batches], axis=0,
        )).to(device),
    )


def _optimize_prepared(
    policy: TerrainDQN, target: TerrainDQN,
    optimizer: torch.optim.Optimizer, prepared: _PreparedOptimizationBatches,
    index: int, config: DQNTrainingConfig,
) -> float:
    selected = int(index)
    predicted = policy(
        prepared.spatial[selected], prepared.scalar[selected],
    ).gather(1, prepared.actions[selected, :, None]).squeeze(1)
    with torch.no_grad():
        next_q = target(
            prepared.next_spatial[selected], prepared.next_scalar[selected],
        )
        expected = prepared.rewards[selected] + config.gamma * masked_bootstrap_values(
            next_q, prepared.next_masks[selected], prepared.terminal[selected],
        )
    loss = F.smooth_l1_loss(predicted, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip_norm)
    optimizer.step()
    return float(loss.detach().cpu())


def _cpu_state_dict(network: TerrainDQN) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in network.state_dict().items()
    }


def evaluate_scenarios(
    network: TerrainDQN, contexts: Iterable[ScenarioRuntime],
    device: torch.device, *, clear_after_each: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for context in contexts:
        evaluation = evaluate_policy_batched(network, context, device)
        bellman = float(context.reference.attacker_objective)
        dqn = evaluation.attacker_objective
        absolute = None if dqn is None else abs(float(dqn) - bellman)
        relative = (
            None if absolute is None or abs(bellman) <= 1.0e-15
            else absolute / abs(bellman)
        )
        records.append({
            "scenario_id": context.spec.scenario_id,
            "split": context.spec.split,
            "terrain_category": context.spec.terrain_category,
            "sensor_map": list(context.spec.sensor_map),
            "bellman_J_A": bellman,
            "bellman_trajectory": list(context.reference.trajectory),
            "bellman_runtime_sec": context.reference.runtime_sec,
            "dqn": evaluation.as_dict(),
            "dqn_edge_hazards": _trajectory_edge_hazards(
                context, evaluation.trajectory,
            ),
            "absolute_error": absolute,
            "relative_error": relative,
        })
        if clear_after_each:
            context.problem.clear_runtime_caches()
    return records


def _trajectory_edge_hazards(
    context: ScenarioRuntime, trajectory: Iterable[int],
) -> list[float]:
    state_ids = tuple(int(value) for value in trajectory)
    values: list[float] = []
    for source, target in zip(state_ids, state_ids[1:]):
        matching = [
            index for index, edge in enumerate(context.problem.successors(source))
            if int(edge.target_id) == target
        ]
        if len(matching) != 1:
            raise RuntimeError("trajectory edge is absent from the authoritative graph")
        values.append(float(
            context.problem.transition(source, matching[0]).cumulative_hazard
        ))
    return values + ([values[-1]] if values else ([0.0] if state_ids else []))


def _clear_surface_caches(contexts: Iterable[ScenarioRuntime]) -> None:
    seen: set[int] = set()
    for context in contexts:
        identity = id(context.builder._surface_height_cache)
        if identity not in seen:
            context.builder.clear_runtime_caches()
            seen.add(identity)


def evaluate_phase4_suite(
    network: TerrainDQN, training_contexts: list[ScenarioRuntime],
    validation_specs: Iterable[Phase4ScenarioSpec],
    condition_template: ComputationCondition, *, reference_path: Path,
    dataset_hash: str, device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate retained training cases and one validation terrain at a time."""

    records = evaluate_scenarios(
        network, training_contexts, device, clear_after_each=True,
    )
    _clear_surface_caches(training_contexts)
    by_terrain: dict[str, list[Phase4ScenarioSpec]] = defaultdict(list)
    for spec in validation_specs:
        by_terrain[spec.terrain_category].append(spec)
    for terrain_category in sorted(by_terrain):
        validation_contexts, _ = build_scenario_runtimes(
            by_terrain[terrain_category], condition_template,
            reference_path=reference_path, dataset_hash=dataset_hash,
            observation_cache_entries=PHASE4_OBSERVATION_CACHE_ENTRIES,
            surface_cache_entries_per_terrain=(
                PHASE4_SURFACE_CACHE_ENTRIES_PER_TERRAIN
            ),
            observation_device=device,
        )
        records.extend(evaluate_scenarios(
            network, validation_contexts, device, clear_after_each=True,
        ))
        _clear_surface_caches(validation_contexts)
        del validation_contexts
        gc.collect()
    return records


def _aggregate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    successful = [item for item in records if item["dqn"]["success"]]
    errors = [
        float(item["relative_error"])
        for item in records if item["relative_error"] is not None
    ]
    return {
        "case_count": len(records),
        "success_count": len(successful),
        "success_rate": len(successful) / len(records) if records else 0.0,
        "median_relative_error": median(errors) if errors else None,
        "maximum_relative_error": max(errors) if errors else None,
        "mean_inference_runtime_sec": (
            float(np.mean([item["dqn"]["inference_runtime_sec"] for item in records]))
            if records else 0.0
        ),
    }


def _evaluation_snapshot(
    episode: int, records: list[dict[str, Any]], runtime_sec: float,
) -> dict[str, Any]:
    return {
        "episode": int(episode),
        "runtime_sec": float(runtime_sec),
        "TRAIN_SIMPLE": _aggregate_records(
            item for item in records if item["split"] == TRAIN_SIMPLE
        ),
        "VALIDATION_SIMPLE": _aggregate_records(
            item for item in records if item["split"] == VALIDATION_SIMPLE
        ),
        "cases": records,
    }


def _selection_key(snapshot: dict[str, Any]) -> tuple[float, float, float]:
    validation = snapshot["VALIDATION_SIMPLE"]
    median_error = validation["median_relative_error"]
    maximum_error = validation["maximum_relative_error"]
    return (
        float(validation["success_rate"]),
        -float("inf") if median_error is None else -float(median_error),
        -float("inf") if maximum_error is None else -float(maximum_error),
    )


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _resume_payload(
    *, seed: int, episode: int, dataset_hash: str,
    training_config: DQNTrainingConfig, network_config: DQNNetworkConfig,
    compatibility: CompatibilitySignature, policy: TerrainDQN,
    target: TerrainDQN, optimizer: torch.optim.Optimizer,
    replay: MultiScenarioReplayBuffer, scenario_rng: np.random.Generator,
    action_rng: np.random.Generator, history: Phase4History,
    environment_steps: int, optimization_steps: int,
    evaluation_runtime_sec: float, training_runtime_sec: float,
    best_episode: int, best_key: tuple[float, float, float] | None,
    best_model_state: dict[str, Tensor] | None,
    best_snapshot: dict[str, Any] | None,
    context_ids: list[str], model_contract: dict[str, Any],
    execution_config: Phase4ExecutionConfig,
    runtime_profile: Phase4RuntimeProfile,
) -> dict[str, Any]:
    return {
        "format": PHASE4_CHECKPOINT_FORMAT,
        "kind": "resumable_training_state",
        "seed": int(seed), "episode": int(episode),
        "dataset_manifest_sha256": dataset_hash,
        "training_config": training_config.as_dict(),
        "network_config": network_config.as_dict(),
        "compatibility": compatibility.as_dict(),
        "model_contract": model_contract,
        "context_ids": context_ids,
        "policy_state_dict": _cpu_state_dict(policy),
        "target_state_dict": _cpu_state_dict(target),
        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
        "replay_state_dict": replay.state_dict(),
        "scenario_rng_state": scenario_rng.bit_generator.state,
        "action_rng_state": action_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "history": history.as_dict(),
        "environment_steps": int(environment_steps),
        "optimization_steps": int(optimization_steps),
        "evaluation_runtime_sec": float(evaluation_runtime_sec),
        "training_runtime_sec": float(training_runtime_sec),
        "best_episode": int(best_episode),
        "best_key": None if best_key is None else list(best_key),
        "best_model_state": best_model_state,
        "best_snapshot": best_snapshot,
        "execution_config": execution_config.as_dict(),
        "runtime_profile_sec": runtime_profile.as_dict(),
    }


def _validate_resume(
    payload: dict[str, Any], *, seed: int, dataset_hash: str,
    training_config: DQNTrainingConfig, network_config: DQNNetworkConfig,
    compatibility: CompatibilitySignature, context_ids: list[str],
    model_contract: dict[str, Any],
) -> None:
    expected = {
        "format": PHASE4_CHECKPOINT_FORMAT,
        "seed": int(seed),
        "dataset_manifest_sha256": dataset_hash,
        "training_config": training_config.as_dict(),
        "network_config": network_config.as_dict(),
        "compatibility": compatibility.as_dict(),
        "context_ids": context_ids,
        "model_contract": model_contract,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"resume checkpoint has incompatible {key}")


def train_phase4_seed(
    training_contexts: list[ScenarioRuntime],
    evaluation_contexts: list[ScenarioRuntime],
    network_config: DQNNetworkConfig,
    training_config: DQNTrainingConfig,
    compatibility: CompatibilitySignature,
    dataset_hash: str,
    seed: int,
    *, checkpoint_dir: Path, run_dir: Path, resume: bool,
    model_contract: dict[str, Any],
    validation_specs: Iterable[Phase4ScenarioSpec] = (),
    condition_template: ComputationCondition | None = None,
    reference_path: Path | None = None,
    execution_config: Phase4ExecutionConfig | None = None,
) -> dict[str, Any]:
    execution_config = execution_config or approved_phase4_execution_config()
    seed_everything(seed)
    device = select_device(training_config.device)
    policy = TerrainDQN(network_config).to(device)
    target = TerrainDQN(network_config).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=training_config.learning_rate)
    replay = MultiScenarioReplayBuffer(
        training_config.replay_capacity, network_config.action_count, seed + 20_000,
    )
    scenario_rng = np.random.default_rng(seed + 30_000)
    action_rng = np.random.default_rng(seed + 10_000)
    history = Phase4History()
    environment_steps = 0
    optimization_steps = 0
    evaluation_runtime_sec = 0.0
    prior_training_runtime_sec = 0.0
    best_episode = 0
    best_key: tuple[float, float, float] | None = None
    best_model_state: dict[str, Tensor] | None = None
    best_snapshot: dict[str, Any] | None = None
    start_episode = 0
    runtime_profile = Phase4RuntimeProfile()
    context_ids = [context.spec.scenario_id for context in training_contexts]
    validation_specs = tuple(validation_specs)
    use_lazy_validation = bool(validation_specs)
    if use_lazy_validation and (condition_template is None or reference_path is None):
        raise ValueError(
            "lazy validation requires condition_template and reference_path"
        )

    def evaluate_all(selected_network: TerrainDQN) -> list[dict[str, Any]]:
        if not use_lazy_validation:
            return evaluate_scenarios(selected_network, evaluation_contexts, device)
        return evaluate_phase4_suite(
            selected_network, training_contexts, validation_specs,
            condition_template, reference_path=reference_path,
            dataset_hash=dataset_hash, device=device,
        )
    resume_path = checkpoint_dir / f"phase4_seed{seed}_resume.pt"
    complete_path = run_dir / f"phase4_seed{seed}_complete.json"
    if resume and complete_path.exists():
        completed = json.loads(complete_path.read_text(encoding="utf-8"))
        completed_contract = {
            "seed": int(seed),
            "dataset_manifest_sha256": dataset_hash,
            "training_config": training_config.as_dict(),
            "network_config": network_config.as_dict(),
            "compatibility": compatibility.as_dict(),
            "model_contract": model_contract,
        }
        for key, value in completed_contract.items():
            if completed.get(key) != value:
                raise ValueError(f"completed seed artifact has incompatible {key}")
        return completed
    if resume and resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        _validate_resume(
            payload, seed=seed, dataset_hash=dataset_hash,
            training_config=training_config, network_config=network_config,
            compatibility=compatibility, context_ids=context_ids,
            model_contract=model_contract,
        )
        policy.load_state_dict(payload["policy_state_dict"])
        target.load_state_dict(payload["target_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
        replay = MultiScenarioReplayBuffer.from_state_dict(payload["replay_state_dict"])
        scenario_rng.bit_generator.state = payload["scenario_rng_state"]
        action_rng.bit_generator.state = payload["action_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        history = Phase4History.from_dict(payload["history"])
        environment_steps = int(payload["environment_steps"])
        optimization_steps = int(payload["optimization_steps"])
        evaluation_runtime_sec = float(payload["evaluation_runtime_sec"])
        prior_training_runtime_sec = float(payload["training_runtime_sec"])
        best_episode = int(payload["best_episode"])
        best_key = (
            None if payload["best_key"] is None
            else tuple(float(value) for value in payload["best_key"])
        )
        best_model_state = payload["best_model_state"]
        best_snapshot = payload["best_snapshot"]
        start_episode = int(payload["episode"])
        runtime_profile = Phase4RuntimeProfile.from_dict(
            payload.get("runtime_profile_sec")
        )
        print(f"Resuming Phase 16.4 seed {seed} at episode {start_episode}", flush=True)

    by_terrain: dict[str, list[int]] = defaultdict(list)
    for index, context in enumerate(training_contexts):
        by_terrain[context.spec.terrain_category].append(index)
    terrain_names = sorted(by_terrain)
    max_episode_steps = max(
        max(1, int(context.problem.grid.altitude_count) - 1)
        for context in training_contexts
    )
    decay_steps = max(1, round(
        training_config.episodes * max_episode_steps
        * training_config.epsilon_decay_fraction
    ))
    last_target_sync_bucket = (
        environment_steps // training_config.target_update_steps
    )
    started = perf_counter()
    completed_episodes = start_episode
    while completed_episodes < training_config.episodes:
        next_checkpoint = min(
            training_config.episodes,
            (
                completed_episodes
                // execution_config.checkpoint_interval_episodes + 1
            ) * execution_config.checkpoint_interval_episodes,
        )
        next_evaluation = min(
            training_config.episodes,
            (
                completed_episodes
                // execution_config.evaluation_interval_episodes + 1
            ) * execution_config.evaluation_interval_episodes,
        )
        next_boundary = min(next_checkpoint, next_evaluation)
        episode_batch_size = min(
            execution_config.episode_batch_size,
            training_config.episodes - completed_episodes,
            next_boundary - completed_episodes,
        )
        slots: list[_EpisodeSlot] = []
        for offset in range(episode_batch_size):
            terrain = terrain_names[int(scenario_rng.integers(len(terrain_names)))]
            candidates = by_terrain[terrain]
            scenario_index = candidates[int(scenario_rng.integers(len(candidates)))]
            context = training_contexts[scenario_index]
            start_id = context.starts[int(scenario_rng.integers(len(context.starts)))]
            slots.append(_EpisodeSlot(
                episode_number=completed_episodes + offset + 1,
                scenario_index=scenario_index,
                start_state_id=int(start_id),
                state_id=int(start_id),
                epsilon_at_start=epsilon_at_step(
                    environment_steps + offset,
                    start=training_config.epsilon_start,
                    end=training_config.epsilon_end, decay_steps=decay_steps,
                ),
            ))

        while any(not slot.ended for slot in slots):
            active = [slot for slot in slots if not slot.ended]
            scenario_indices = [slot.scenario_index for slot in active]
            state_ids = [slot.state_id for slot in active]

            _synchronize(device)
            action_started = perf_counter()
            masks = _feasible_masks_many(
                training_contexts, scenario_indices, state_ids,
            )
            q_values = _q_values_batch(
                policy, training_contexts, scenario_indices, state_ids, device,
            )
            action_ids = []
            for offset, (values, mask) in enumerate(zip(q_values, masks)):
                epsilon = epsilon_at_step(
                    environment_steps + offset,
                    start=training_config.epsilon_start,
                    end=training_config.epsilon_end, decay_steps=decay_steps,
                )
                action_ids.append(epsilon_greedy_action(
                    values, mask, epsilon, action_rng,
                ))
            _synchronize(device)
            runtime_profile.action_observation_and_forward_sec += (
                perf_counter() - action_started
            )

            transition_started = perf_counter()
            transitions = [
                training_contexts[slot.scenario_index].catalog.transition(
                    slot.state_id, action_id,
                )
                for slot, action_id in zip(active, action_ids)
            ]
            next_state_ids = [
                int(transition.target_state_id) for transition in transitions
            ]
            next_masks = _feasible_masks_many(
                training_contexts, scenario_indices, next_state_ids,
            )
            pending_replay_batches: list[MultiScenarioBatch] = []
            for slot, action_id, transition, next_state_id, next_mask in zip(
                active, action_ids, transitions, next_state_ids, next_masks,
            ):
                slot.steps += 1
                reached_goal = bool(transition.terminal)
                dead_end = bool(not reached_goal and not next_mask.any())
                terminated = bool(reached_goal or dead_end)
                truncated = bool(
                    slot.steps >= max_episode_steps and not terminated
                )
                if truncated:
                    raise RuntimeError(
                        "authoritative DAG exceeded its derived horizon"
                    )
                ended = bool(terminated or truncated)
                reward = -float(transition.stage_cost)
                replay.add(
                    scenario_index=slot.scenario_index,
                    state_id=slot.state_id, action_id=action_id,
                    reward=reward, next_state_id=next_state_id,
                    terminal=ended, next_feasible_mask=next_mask,
                )
                environment_steps += 1
                slot.glide_return += reward
                slot.state_id = next_state_id
                slot.reached_goal = reached_goal
                slot.ended = ended
            if len(replay) >= training_config.replay_warmup:
                for _ in range(
                    execution_config.optimizer_updates_per_vector_step
                ):
                    pending_replay_batches.append(
                        replay.sample(training_config.batch_size)
                    )
            runtime_profile.environment_transition_and_replay_sec += (
                perf_counter() - transition_started
            )

            prefetch_size = execution_config.optimization_prefetch_batches
            for offset in range(0, len(pending_replay_batches), prefetch_size):
                selected_batches = pending_replay_batches[offset:offset + prefetch_size]
                _synchronize(device)
                preparation_started = perf_counter()
                prepared = _prepare_optimization_batches(
                    selected_batches, training_contexts, device,
                )
                _synchronize(device)
                runtime_profile.optimization_observation_transfer_sec += (
                    perf_counter() - preparation_started
                )

                optimization_started = perf_counter()
                for batch_index in range(prepared.count):
                    loss = _optimize_prepared(
                        policy, target, optimizer, prepared, batch_index,
                        training_config,
                    )
                    optimization_steps += 1
                    history.optimization_step.append(optimization_steps)
                    history.td_loss.append(loss)
                _synchronize(device)
                runtime_profile.dqn_forward_backward_optimizer_sec += (
                    perf_counter() - optimization_started
                )
            target_sync_bucket = (
                environment_steps // training_config.target_update_steps
            )
            if target_sync_bucket > last_target_sync_bucket:
                target.load_state_dict(policy.state_dict())
                last_target_sync_bucket = target_sync_bucket

        for slot in sorted(slots, key=lambda item: item.episode_number):
            context = training_contexts[slot.scenario_index]
            history.episode.append(slot.episode_number)
            history.scenario_id.append(context.spec.scenario_id)
            history.terrain_category.append(context.spec.terrain_category)
            history.sensor_map.append(list(context.spec.sensor_map))
            history.episode_glide_return.append(slot.glide_return)
            history.episode_full_transformed_return.append(
                slot.glide_return - context.problem.powered_cost(
                    slot.start_state_id
                )
            )
            history.episode_steps.append(slot.steps)
            history.episode_reached_goal.append(slot.reached_goal)
            history.epsilon.append(slot.epsilon_at_start)
        completed_episodes += episode_batch_size

        evaluate_now = (
            completed_episodes % execution_config.evaluation_interval_episodes == 0
            or completed_episodes == training_config.episodes
        )
        checkpoint_now = (
            completed_episodes % execution_config.checkpoint_interval_episodes == 0
            or completed_episodes == training_config.episodes
        )
        snapshot = None
        if evaluate_now:
            evaluation_started = perf_counter()
            records = evaluate_all(policy)
            evaluation_runtime = perf_counter() - evaluation_started
            evaluation_runtime_sec += evaluation_runtime
            runtime_profile.checkpoint_evaluation_sec += evaluation_runtime
            snapshot = _evaluation_snapshot(
                completed_episodes, records, evaluation_runtime,
            )
            history.evaluations.append(snapshot)
            key = _selection_key(snapshot)
            if best_key is None or key > best_key:
                best_key = key
                best_episode = completed_episodes
                best_model_state = _cpu_state_dict(policy)
                best_snapshot = snapshot

        if checkpoint_now:
            elapsed = prior_training_runtime_sec + perf_counter() - started
            save_started = perf_counter()
            payload = _resume_payload(
                seed=seed, episode=completed_episodes,
                dataset_hash=dataset_hash,
                training_config=training_config, network_config=network_config,
                compatibility=compatibility, policy=policy, target=target,
                optimizer=optimizer, replay=replay, scenario_rng=scenario_rng,
                action_rng=action_rng, history=history,
                environment_steps=environment_steps,
                optimization_steps=optimization_steps,
                evaluation_runtime_sec=evaluation_runtime_sec,
                training_runtime_sec=elapsed, best_episode=best_episode,
                best_key=best_key, best_model_state=best_model_state,
                best_snapshot=best_snapshot, context_ids=context_ids,
                model_contract=model_contract,
                execution_config=execution_config,
                runtime_profile=runtime_profile,
            )
            _atomic_torch_save(resume_path, payload)
            runtime_profile.checkpoint_save_sec += perf_counter() - save_started
            latest_validation = (
                None if snapshot is None
                else snapshot[VALIDATION_SIMPLE]
            )
            progress = {
                "seed": seed, "episode": completed_episodes,
                "episodes_total": training_config.episodes,
                "device": str(device), "elapsed_sec": elapsed,
                "best_episode": best_episode,
                "evaluation_performed": evaluate_now,
                "latest_validation": latest_validation,
                "execution_config": execution_config.as_dict(),
                "runtime_profile_sec": runtime_profile.as_dict(),
                "resume_checkpoint": _relative(resume_path),
            }
            _write_json(run_dir / f"phase4_seed{seed}_progress.json", progress)
            print(json.dumps(progress, indent=2), flush=True)

    training_runtime_sec = prior_training_runtime_sec + perf_counter() - started
    final_model_state = _cpu_state_dict(policy)
    if best_model_state is None or best_snapshot is None:
        raise RuntimeError("Phase 16.4 produced no checkpoint evaluation")
    post_training_evaluation_started = perf_counter()
    final_records = evaluate_all(policy)
    policy.load_state_dict(best_model_state)
    official_records = evaluate_all(policy)
    runtime_profile.post_training_evaluation_sec += (
        perf_counter() - post_training_evaluation_started
    )
    best_path = checkpoint_dir / f"phase4_seed{seed}_best.pt"
    final_path = checkpoint_dir / f"phase4_seed{seed}_final.pt"
    common = {
        "format": PHASE4_CHECKPOINT_FORMAT,
        "seed": seed,
        "dataset_manifest_sha256": dataset_hash,
        "network_config": network_config.as_dict(),
        "training_config": training_config.as_dict(),
        "compatibility": compatibility.as_dict(),
        "model_contract": model_contract,
        "execution_config": execution_config.as_dict(),
        "runtime_profile_sec": runtime_profile.as_dict(),
    }
    _atomic_torch_save(best_path, {
        **common, "kind": "best_validation_checkpoint",
        "episode": best_episode, "model_state_dict": best_model_state,
        "validation_selection": best_snapshot["VALIDATION_SIMPLE"],
    })
    _atomic_torch_save(final_path, {
        **common, "kind": "final_training_state",
        "episode": training_config.episodes,
        "model_state_dict": final_model_state,
        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
    })
    result = {
        "seed": seed, "device": str(device),
        "dataset_manifest_sha256": dataset_hash,
        "training_config": training_config.as_dict(),
        "network_config": network_config.as_dict(),
        "compatibility": compatibility.as_dict(),
        "model_contract": model_contract,
        "best_episode": best_episode,
        "best_checkpoint": _relative(best_path),
        "final_checkpoint": _relative(final_path),
        "training_runtime_sec": training_runtime_sec,
        "evaluation_runtime_sec": evaluation_runtime_sec,
        "environment_steps": environment_steps,
        "optimization_steps": optimization_steps,
        "execution_config": execution_config.as_dict(),
        "runtime_profile_sec": runtime_profile.as_dict(),
        "replay_size": len(replay),
        "replay_training_scenario_counts": {
            training_contexts[index].spec.scenario_id: count
            for index, count in replay.scenario_counts().items()
        },
        "episode_training_scenario_counts": dict(sorted(
            (name, history.scenario_id.count(name)) for name in set(history.scenario_id)
        )),
        "cached_observation_states": {
            context.spec.scenario_id: context.cache.cached_states
            for context in training_contexts
        },
        "best_snapshot": best_snapshot,
        "official_best_records": official_records,
        "final_records": final_records,
        "history": history.as_dict(),
    }
    _write_json(complete_path, result)
    return result


def _rolling(values: list[float], window: int = 500) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return array
    width = min(window, len(array))
    kernel = np.ones(width) / width
    prefix = np.full(width - 1, np.nan)
    return np.concatenate((prefix, np.convolve(array, kernel, mode="valid")))


def _terrain_axes(ax: Any, terrain_category: str) -> None:
    from terrain_catalog import build_terrain
    terrain = build_terrain(terrain_category)
    for box in terrain.obstacle_boxes():
        ax.add_patch(plt.Rectangle(
            (box.x_limits[0], box.y_limits[0]), box.width_x, box.width_y,
            facecolor="#bdbdbd", edgecolor="#333333", alpha=0.75,
        ))
        ax.text(
            box.center_x, box.center_y,
            f"base={box.base_z:g}\nh={box.height:g}",
            ha="center", va="center", fontsize=7,
        )
    ax.set_xlim(-8.0, 8.0)
    ax.set_ylim(-4.0, 4.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(terrain_category)
    ax.set_xlabel("x [map unit]")
    ax.set_ylabel("y [map unit]")


def _save(figure: Any, path: Path) -> str:
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return _relative(path)


def figure_terrain_gallery(
    terrain_categories: Iterable[str], title: str, output: Path,
) -> str:
    categories = list(terrain_categories)
    figure, axes = plt.subplots(1, len(categories), figsize=(5 * len(categories), 4.2))
    axes = np.atleast_1d(axes)
    for ax, category in zip(axes, categories):
        _terrain_axes(ax, category)
    figure.suptitle(title)
    return _save(figure, output)


def figure_sampling(seed_results: list[dict[str, Any]], output: Path) -> str:
    counts: dict[str, int] = defaultdict(int)
    for result in seed_results:
        for scenario_id, count in result["episode_training_scenario_counts"].items():
            counts[scenario_id] += int(count)
    labels = sorted(counts)
    figure, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(np.arange(len(labels)), [counts[label] for label in labels], color="#0072B2")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_ylabel("episodes across seeds")
    ax.set_title("Phase 16.4 uniform terrain/sensor sampling audit")
    return _save(figure, output)


def figure_learning(seed_results: list[dict[str, Any]], output: Path) -> str:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for result in seed_results:
        history = result["history"]
        episode = np.asarray(history["episode"])
        axes[0].plot(
            episode, _rolling(history["episode_full_transformed_return"]),
            label=f"seed {result['seed']}",
        )
        axes[1].plot(
            episode, _rolling([float(value) for value in history["episode_reached_goal"]]),
            label=f"seed {result['seed']}",
        )
    axes[0].set_ylabel("rolling transformed return")
    axes[1].set_ylabel("rolling goal rate")
    axes[1].set_xlabel("episode")
    axes[0].legend()
    axes[1].set_ylim(-0.02, 1.02)
    axes[0].set_title("Shared-DQN multi-terrain learning")
    return _save(figure, output)


def figure_train_validation_gap(
    seed_results: list[dict[str, Any]], output: Path,
) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for result in seed_results:
        evaluations = result["history"]["evaluations"]
        episodes = [item["episode"] for item in evaluations]
        for ax, split in zip(axes, (TRAIN_SIMPLE, VALIDATION_SIMPLE)):
            values = [item[split]["median_relative_error"] for item in evaluations]
            ax.plot(episodes, np.asarray(values, dtype=float) * 100.0,
                    label=f"seed {result['seed']}")
            ax.axhline(10.0, color="#D55E00", linestyle=":")
            ax.set_title(split)
            ax.set_xlabel("episode")
            ax.grid(alpha=0.25)
    axes[0].set_ylabel("median Bellman-relative J_A error [%]")
    axes[0].legend()
    return _save(figure, output)


def figure_per_terrain(seed_results: list[dict[str, Any]], output: Path) -> str:
    terrain_names = sorted({
        item["terrain_category"]
        for result in seed_results
        for item in result["official_best_records"]
    })
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    axes_flat = axes.ravel()
    for ax, terrain in zip(axes_flat, terrain_names):
        error_axis = ax.twinx()
        for result in seed_results:
            episodes: list[int] = []
            dqn_values: list[float] = []
            bellman_values: list[float] = []
            error_values: list[float] = []
            failure_episodes: list[int] = []
            for snapshot in result["history"]["evaluations"]:
                cases = [
                    item for item in snapshot["cases"]
                    if item["terrain_category"] == terrain
                ]
                successful = [item for item in cases if item["relative_error"] is not None]
                episodes.append(int(snapshot["episode"]))
                bellman_values.append(median([
                    float(item["bellman_J_A"]) for item in cases
                ]))
                dqn_values.append(
                    median([float(item["dqn"]["attacker_objective"]) for item in successful])
                    if successful else np.nan
                )
                error_values.append(
                    100.0 * median([float(item["relative_error"]) for item in successful])
                    if successful else np.nan
                )
                if len(successful) != len(cases):
                    failure_episodes.append(int(snapshot["episode"]))
            line = ax.plot(
                episodes, dqn_values, label=f"DQN seed {result['seed']}", linewidth=1.4,
            )[0]
            ax.plot(
                episodes, bellman_values, color=line.get_color(), linestyle=":",
                linewidth=1.0, alpha=0.75,
            )
            error_axis.plot(
                episodes, error_values, color=line.get_color(), linestyle="--",
                linewidth=0.9, alpha=0.45,
            )
            if failure_episodes:
                ax.scatter(
                    failure_episodes, np.full(len(failure_episodes), np.nanmin(bellman_values)),
                    marker="x", color=line.get_color(), s=24,
                )
        ax.set_title(terrain)
        ax.set_xlabel("episode")
        ax.set_ylabel("median J_A (solid DQN, dotted Bellman)")
        error_axis.set_ylabel("median relative error [%], dashed", color="#666666")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    for ax in axes_flat[len(terrain_names):]:
        ax.axis("off")
    figure.suptitle("Per-terrain checkpoint evaluation; x markers denote any failed case")
    return _save(figure, output)


def figure_representative_trajectories(
    records: list[dict[str, Any]], position_problem: AttackerBRProblem, output: Path,
) -> str:
    selected = [
        item for item in records
        if np.allclose(item["sensor_map"], [5.0, 0.0, 0.0])
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes_flat = axes.ravel()
    for ax, item in zip(axes_flat, selected):
        _terrain_axes(ax, item["terrain_category"])
        bellman = np.asarray([
            position_problem.position_map(state_id)
            for state_id in item["bellman_trajectory"]
        ])
        ax.plot(bellman[:, 0], bellman[:, 1], "o-", label="Bellman")
        trajectory = item["dqn"]["trajectory"]
        if trajectory:
            dqn = np.asarray([
                position_problem.position_map(state_id) for state_id in trajectory
            ])
            ax.plot(dqn[:, 0], dqn[:, 1], "s--", label="DQN")
            hazard = np.asarray(item["dqn_edge_hazards"], dtype=float)
            colored = ax.scatter(
                dqn[:, 0], dqn[:, 1], c=hazard, cmap="magma", s=32,
                edgecolors="white", linewidths=0.35, zorder=5,
            )
            figure.colorbar(
                colored, ax=ax, fraction=0.046, pad=0.03,
                label="DQN edge hazard",
            )
        ax.scatter(5.0, 0.0, marker="^", color="#CC79A7", s=55, label="sensor")
        error = item["relative_error"]
        ax.set_title(
            f"{item['terrain_category']}\nerror="
            f"{'failure' if error is None else f'{100.0 * error:.2f}%'}"
        )
        ax.legend(fontsize=7)
    for ax in axes_flat[len(selected):]:
        ax.axis("off")
    figure.suptitle("Bellman and generalized-DQN trajectories at sensor (5,0,0)")
    return _save(figure, output)


def figure_overfitting(seed_results: list[dict[str, Any]], output: Path) -> str:
    figure, ax = plt.subplots(figsize=(6.5, 5.5))
    for result in seed_results:
        best = result["best_snapshot"]
        train = best[TRAIN_SIMPLE]["median_relative_error"]
        validation = best[VALIDATION_SIMPLE]["median_relative_error"]
        ax.scatter(100.0 * float(train), 100.0 * float(validation), s=80,
                   label=f"seed {result['seed']}")
    limit = max(20.0, *ax.get_xlim(), *ax.get_ylim())
    ax.plot([0.0, limit], [0.0, limit], "k--", alpha=0.5)
    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_xlabel("training median error [%]")
    ax.set_ylabel("validation median error [%]")
    ax.set_title("Phase 16.4 overfitting diagnostic")
    ax.legend()
    return _save(figure, output)


def _official_quality(seed_results: list[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    validation_records = [
        item for result in seed_results for item in result["official_best_records"]
        if item["split"] == VALIDATION_SIMPLE
    ]
    errors = [
        float(item["relative_error"])
        for item in validation_records if item["relative_error"] is not None
    ]
    all_success = bool(validation_records) and all(
        item["dqn"]["success"] and item["dqn"]["trajectory_feasible"]
        for item in validation_records
    )
    median_error = median(errors) if errors else float("inf")
    maximum_error = max(errors) if errors else float("inf")
    passed = bool(
        not smoke and len(seed_results) == 3 and all_success
        and len(errors) == len(validation_records)
        and median_error <= 0.10 and maximum_error <= 0.20
    )
    return {
        "all_validation_cases_goal_reaching_and_feasible": all_success,
        "validation_record_count": len(validation_records),
        "median_relative_J_A_error": (
            None if not np.isfinite(median_error) else median_error
        ),
        "maximum_relative_J_A_error": (
            None if not np.isfinite(maximum_error) else maximum_error
        ),
        "median_threshold": 0.10,
        "every_case_threshold": 0.20,
        "passed": passed,
        "evaluated_for_official_run": not smoke,
    }


def run_phase4(
    *, smoke: bool = False, resume: bool = True,
    device: str = "cuda", require_cuda: bool = True,
    spatial_resolution_m: float = 25.0,
    heading_spacing_deg: float = 5.0,
    episode_batch_size: int | None = None,
    optimization_prefetch_batches: int | None = None,
    evaluation_interval_episodes: int | None = None,
    run_label: str | None = None,
) -> Path:
    """Run or resume the approved Phase 16.4 experiment."""

    if require_cuda and not smoke and not torch.cuda.is_available():
        raise RuntimeError(
            "Phase 16.4 official training requires CUDA; no CPU fallback is allowed"
        )
    if require_cuda and not smoke and device != "cuda":
        raise ValueError("Phase 16.4 official training must request device='cuda'")
    tag = _condition_tag(spatial_resolution_m, heading_spacing_deg)
    approved_execution = approved_phase4_execution_config(smoke=smoke)
    execution_config = Phase4ExecutionConfig(
        episode_batch_size=(
            approved_execution.episode_batch_size
            if episode_batch_size is None else int(episode_batch_size)
        ),
        optimization_prefetch_batches=(
            approved_execution.optimization_prefetch_batches
            if optimization_prefetch_batches is None
            else int(optimization_prefetch_batches)
        ),
        optimizer_updates_per_vector_step=(
            approved_execution.optimizer_updates_per_vector_step
        ),
        checkpoint_interval_episodes=(
            approved_execution.checkpoint_interval_episodes
        ),
        evaluation_interval_episodes=(
            approved_execution.evaluation_interval_episodes
            if evaluation_interval_episodes is None
            else int(evaluation_interval_episodes)
        ),
        execution_id=approved_execution.execution_id,
    )
    run_label = run_label or (
        "smoke_batched_v1" if smoke else "official_batched_v1"
    )
    if not run_label or any(character in run_label for character in "\\/:"):
        raise ValueError("run_label must be one safe path component")
    output_dir = OUTPUT_ROOT / tag / run_label
    checkpoint_dir = CHECKPOINT_ROOT / tag / run_label
    run_dir = RUN_ROOT / tag / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = approved_dataset_manifest()
    condition = ComputationCondition(
        spatial_resolution_m=float(spatial_resolution_m),
        heading_spacing_deg=float(heading_spacing_deg),
        r_neighbor=1,
    )
    training_config = approved_phase4_training_config(device=device, smoke=smoke)
    final_manifest_path = output_dir / MANIFEST_NAME
    if resume and final_manifest_path.exists():
        completed = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        if (
            completed.get("dataset_manifest_sha256")
            != dataset_manifest["manifest_sha256"]
            or completed.get("condition") != condition.as_dict()
        ):
            raise ValueError("completed Phase 16.4 manifest is incompatible")
        print(f"Phase 16.4 is already complete: {_relative(final_manifest_path)}", flush=True)
        return final_manifest_path

    dataset_path = output_dir / "phase4_dataset_manifest.json"
    _write_json(dataset_path, dataset_manifest)
    config_path = output_dir / "phase4_training_config.json"
    _write_json(config_path, {
        "condition": condition.as_dict(),
        "training_config": training_config.as_dict(),
        "execution_config": execution_config.as_dict(),
        "observation_config": ObservationConfig().as_dict(),
        "require_cuda_for_official_training": require_cuda,
        "checkpoint_selection": (
            "maximize validation success rate, then minimize validation median "
            "relative J_A error, then minimize validation maximum error; earliest wins ties"
        ),
    })
    training_specs = specs_from_manifest(dataset_manifest, TRAIN_SIMPLE)
    validation_specs = specs_from_manifest(dataset_manifest, VALIDATION_SIMPLE)
    canonical_reference = (
        OUTPUT_ROOT / tag / "official" / "phase4_bellman_reference_manifest.json"
    )
    reference_path = (
        canonical_reference
        if not smoke and canonical_reference.exists()
        else output_dir / "phase4_bellman_reference_manifest.json"
    )
    training_contexts, runtime_metadata = build_scenario_runtimes(
        training_specs, condition, reference_path=reference_path,
        dataset_hash=dataset_manifest["manifest_sha256"],
        observation_cache_entries=PHASE4_OBSERVATION_CACHE_ENTRIES,
        surface_cache_entries_per_terrain=(
            PHASE4_SURFACE_CACHE_ENTRIES_PER_TERRAIN
        ),
        observation_device=device,
    )
    runtime_metadata.update({
        "retained_training_terrain_scene_count": runtime_metadata["terrain_scene_count"],
        "retained_training_scenario_count": len(training_contexts),
        "lazy_validation_terrain_scene_count": len({
            spec.terrain_category for spec in validation_specs
        }),
        "lazy_validation_scenario_count": len(validation_specs),
        "evaluation_batch_size": PHASE4_EVALUATION_BATCH_SIZE,
    })
    if any(spec.split == RESERVED_PHASE5_TEST for spec in (*training_specs, *validation_specs)):
        raise RuntimeError("Phase 5 scenario was loaded into Phase 16.4")
    compatibility = CompatibilitySignature.from_components(
        training_contexts[0].problem, training_contexts[0].builder,
        training_contexts[0].catalog,
    )
    shapes = training_contexts[0].builder.tensor_ready_shapes
    network_config = DQNNetworkConfig(
        spatial_channels=shapes["spatial"][0],
        spatial_height=shapes["spatial"][1],
        spatial_width=shapes["spatial"][2],
        scalar_features=shapes["scalar"][0],
        action_count=training_contexts[0].catalog.action_count,
    )
    model_contract = {
        "condition": condition.as_dict(),
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "observation_config": training_contexts[0].builder.config.as_dict(),
        "normalization": "fixed deterministic physical scaling; no fitted statistics",
        "action_catalog_id": training_contexts[0].catalog.catalog_id,
        "attacker_objective": asdict(
            training_contexts[0].problem.scene.config.attacker_objective
        ),
        "glider": asdict(training_contexts[0].problem.scene.config.glider),
        "detection": asdict(training_contexts[0].problem.scene.config.detection),
        "physical_scale": asdict(
            training_contexts[0].problem.scene.config.physical_scale
        ),
        "bellman_contract_sha256": runtime_metadata["bellman_contract_sha256"],
    }

    seed_results: list[dict[str, Any]] = []
    for seed in training_config.seeds:
        print(f"Phase 16.4 shared-DQN training seed {seed} on {device}", flush=True)
        seed_results.append(train_phase4_seed(
            training_contexts, training_contexts, network_config, training_config,
            compatibility, dataset_manifest["manifest_sha256"], seed,
            checkpoint_dir=checkpoint_dir, run_dir=run_dir, resume=resume,
            model_contract=model_contract,
            validation_specs=validation_specs, condition_template=condition,
            reference_path=reference_path,
            execution_config=execution_config,
        ))

    best_result = max(
        seed_results,
        key=lambda result: (
            _selection_key(result["best_snapshot"]),
            -int(result["best_episode"]), -int(result["seed"]),
        ),
    )
    source_checkpoint = project_paths.WORKSPACE_ROOT / best_result["best_checkpoint"]
    official_checkpoint = checkpoint_dir / "phase4_official_generalized.pt"
    official_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    official_payload["kind"] = "official_generalized_checkpoint"
    official_payload["selected_from_seed"] = int(best_result["seed"])
    _atomic_torch_save(official_checkpoint, official_payload)
    official_records = best_result["official_best_records"]
    quality = _official_quality(seed_results, smoke)

    training_history_path = output_dir / "phase4_training_history.json"
    evaluation_history_path = output_dir / "phase4_evaluation_history.json"
    _write_json(training_history_path, {
        "seeds": [
            {key: result[key] for key in (
                "seed", "device", "best_episode", "training_runtime_sec",
                "evaluation_runtime_sec", "environment_steps", "optimization_steps",
                "execution_config", "runtime_profile_sec",
                "replay_size", "replay_training_scenario_counts",
                "episode_training_scenario_counts", "cached_observation_states", "history",
            )}
            for result in seed_results
        ]
    })
    _write_json(evaluation_history_path, {
        "seeds": [
            {
                "seed": result["seed"],
                "best_snapshot": result["best_snapshot"],
                "official_best_records": result["official_best_records"],
                "final_records": result["final_records"],
            }
            for result in seed_results
        ]
    })

    train_names = sorted({item.spec.terrain_category for item in training_contexts})
    validation_names = sorted({item.terrain_category for item in validation_specs})
    visualizations = [
        figure_terrain_gallery(
            train_names, "TRAIN_SIMPLE terrain gallery",
            output_dir / "phase4_training_terrain_gallery.png",
        ),
        figure_terrain_gallery(
            validation_names, "VALIDATION_SIMPLE terrain gallery",
            output_dir / "phase4_validation_terrain_gallery.png",
        ),
        figure_sampling(seed_results, output_dir / "phase4_sampling_distribution.png"),
        figure_learning(seed_results, output_dir / "phase4_learning_curve.png"),
        figure_train_validation_gap(
            seed_results, output_dir / "phase4_train_validation_gap.png",
        ),
        figure_per_terrain(
            seed_results, output_dir / "phase4_per_terrain_evaluation.png",
        ),
        figure_representative_trajectories(
            official_records, training_contexts[0].problem,
            output_dir / "phase4_representative_trajectories.png",
        ),
        figure_overfitting(seed_results, output_dir / "phase4_overfitting_diagnostic.png"),
    ]
    model_manifest_path = output_dir / "phase4_generalized_model_manifest.json"
    _write_json(model_manifest_path, {
        "format": PHASE4_CHECKPOINT_FORMAT,
        "official_checkpoint": _relative(official_checkpoint),
        "selected_seed": best_result["seed"],
        "selected_episode": best_result["best_episode"],
        "selection_validation": best_result["best_snapshot"][VALIDATION_SIMPLE],
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "condition": condition.as_dict(),
        "compatibility": compatibility.as_dict(),
        "network_config": network_config.as_dict(),
        "training_config": training_config.as_dict(),
        "execution_config": execution_config.as_dict(),
        "model_contract": model_contract,
    })
    manifest = {
        "phase": 4, "stage": "16.4",
        "purpose": "one shared condition-specific DQN across simple terrains",
        "status": (
            "smoke_only" if smoke else
            "passed_approved_generalization_criterion" if quality["passed"]
            else "failed_approved_generalization_criterion"
        ),
        "condition": condition.as_dict(),
        "dataset_manifest": _relative(dataset_path),
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "training_config_path": _relative(config_path),
        "execution_config": execution_config.as_dict(),
        "bellman_reference_manifest": _relative(reference_path),
        "training_history": _relative(training_history_path),
        "evaluation_history": _relative(evaluation_history_path),
        "generalized_model_manifest": _relative(model_manifest_path),
        "official_checkpoint": _relative(official_checkpoint),
        "runtime_metadata": runtime_metadata,
        "quality": quality,
        "device": {
            "requested": device,
            "require_cuda": require_cuda and not smoke,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "runtime_sec": {
            "training_total": sum(result["training_runtime_sec"] for result in seed_results),
            "evaluation_total": sum(result["evaluation_runtime_sec"] for result in seed_results),
        },
        "visualizations": visualizations,
        "scope_limits": {
            "phase5_complex_unseen_test_performed": False,
            "local_sse_integration_performed": False,
            "cross_discretization_model": False,
            "final_computational_regime_claim": False,
        },
    }
    _write_json(final_manifest_path, manifest)
    print(json.dumps({
        "manifest": _relative(final_manifest_path),
        "status": manifest["status"],
        "official_checkpoint": manifest["official_checkpoint"],
        "quality": quality,
    }, indent=2), flush=True)
    return final_manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--spatial-resolution-m", type=float, default=25.0)
    parser.add_argument("--heading-spacing-deg", type=float, default=5.0)
    parser.add_argument("--episode-batch-size", type=int)
    parser.add_argument("--optimization-prefetch-batches", type=int)
    parser.add_argument("--evaluation-interval-episodes", type=int)
    parser.add_argument("--run-label")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_phase4(
        smoke=arguments.smoke,
        resume=not arguments.no_resume,
        device=arguments.device,
        require_cuda=not arguments.allow_cpu,
        spatial_resolution_m=arguments.spatial_resolution_m,
        heading_spacing_deg=arguments.heading_spacing_deg,
        episode_batch_size=arguments.episode_batch_size,
        optimization_prefetch_batches=arguments.optimization_prefetch_batches,
        evaluation_interval_episodes=arguments.evaluation_interval_episodes,
        run_label=arguments.run_label,
    )


__all__ = [
    "BellmanReference", "PHASE4_CHECKPOINT_FORMAT", "Phase4History",
    "ScenarioRuntime", "Phase4ExecutionConfig", "Phase4RuntimeProfile",
    "approved_phase4_execution_config", "approved_phase4_training_config",
    "build_scenario_runtimes", "evaluate_scenarios", "run_phase4",
    "train_phase4_seed", "load_phase4_checkpoint",
]

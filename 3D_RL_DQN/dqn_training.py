"""Masked, reproducible Phase 16.3 DQN training and evaluation core."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn import functional as F

from attacker_br_problem import AttackerBRProblem
from dqn_action_catalog import HeadingActionCatalog
from dqn_env import DQNAttackerEnv, ObservationTensorCache
from dqn_model import (
    DQNNetworkConfig,
    TerrainDQN,
    masked_bootstrap_values,
    masked_greedy_actions,
)
from dqn_replay import ReplayBatch, ReplayBuffer
from terrain_observation import TerrainObservationBuilder


CHECKPOINT_FORMAT = "p1b-phase3-dqn-checkpoint-v1"


@dataclass(frozen=True)
class DQNTrainingConfig:
    episodes: int = 20_000
    gamma: float = 1.0
    replay_capacity: int = 50_000
    replay_warmup: int = 2_000
    batch_size: int = 128
    optimizer: str = "AdamW"
    learning_rate: float = 3.0e-4
    loss: str = "huber"
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.60
    target_update_steps: int = 1_000
    updates_per_environment_step: int = 1
    gradient_clip_norm: float = 10.0
    evaluation_interval_episodes: int = 500
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "auto"
    config_id: str = "p1b-phase3-training-v1"

    def __post_init__(self) -> None:
        integer_positive = (
            self.episodes, self.replay_capacity, self.replay_warmup,
            self.batch_size, self.target_update_steps,
            self.updates_per_environment_step, self.evaluation_interval_episodes,
        )
        if any(int(value) < 1 for value in integer_positive):
            raise ValueError("training counts must be positive")
        if self.replay_warmup > self.replay_capacity:
            raise ValueError("replay_warmup cannot exceed replay_capacity")
        if self.batch_size > self.replay_warmup:
            raise ValueError("batch_size cannot exceed replay_warmup")
        if self.gamma != 1.0:
            raise ValueError("Phase 16.3 objective equivalence requires gamma=1.0")
        if self.optimizer != "AdamW" or self.loss != "huber":
            raise ValueError("optimizer/loss differ from the approved Phase 16.3 bundle")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not 0.0 < self.epsilon_decay_fraction <= 1.0:
            raise ValueError("epsilon_decay_fraction must lie in (0,1]")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a nonempty unique sequence")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        result["epsilon_decay_definition"] = (
            "linear by environment step over episodes * maximum DAG steps * "
            "epsilon_decay_fraction"
        )
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DQNTrainingConfig":
        copied = dict(values)
        copied.pop("epsilon_decay_definition", None)
        copied["seeds"] = tuple(copied["seeds"])
        return cls(**copied)


@dataclass(frozen=True)
class CompatibilitySignature:
    spatial_discretization_m: float
    angular_discretization_deg: float
    observation_schema_id: str
    scalar_shape: tuple[int, ...]
    spatial_shape: tuple[int, ...]
    action_catalog_id: str

    @classmethod
    def from_components(
        cls,
        problem: AttackerBRProblem,
        builder: TerrainObservationBuilder,
        catalog: HeadingActionCatalog,
    ) -> "CompatibilitySignature":
        shapes = builder.tensor_ready_shapes
        return cls(
            spatial_discretization_m=float(
                problem.scene.condition.spatial_resolution_m
            ),
            angular_discretization_deg=float(
                problem.scene.condition.heading_spacing_deg
            ),
            observation_schema_id=builder.config.schema_id,
            scalar_shape=tuple(shapes["scalar"]),
            spatial_shape=tuple(shapes["spatial"]),
            action_catalog_id=catalog.catalog_id,
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["scalar_shape"] = list(self.scalar_shape)
        result["spatial_shape"] = list(self.spatial_shape)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CompatibilitySignature":
        copied = dict(values)
        copied["scalar_shape"] = tuple(copied["scalar_shape"])
        copied["spatial_shape"] = tuple(copied["spatial_shape"])
        return cls(**copied)


@dataclass(frozen=True)
class PolicyEvaluation:
    success: bool
    switching_state_id: int | None
    trajectory: tuple[int, ...]
    reached_goal: bool
    trajectory_feasible: bool
    attacker_objective: float | None
    glide_cost: float | None
    powered_cost: float | None
    candidate_successes: int
    candidate_count: int
    inference_runtime_sec: float
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "switching_state_id": self.switching_state_id,
            "trajectory": list(self.trajectory),
            "reached_goal": self.reached_goal,
            "trajectory_feasible": self.trajectory_feasible,
            "attacker_objective": self.attacker_objective,
            "glide_cost": self.glide_cost,
            "powered_cost": self.powered_cost,
            "candidate_successes": self.candidate_successes,
            "candidate_count": self.candidate_count,
            "inference_runtime_sec": self.inference_runtime_sec,
            "status": self.status,
        }


@dataclass
class DQNTrainingHistory:
    episode: list[int] = field(default_factory=list)
    episode_glide_return: list[float] = field(default_factory=list)
    episode_full_transformed_return: list[float] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)
    episode_reached_goal: list[bool] = field(default_factory=list)
    epsilon: list[float] = field(default_factory=list)
    optimization_step: list[int] = field(default_factory=list)
    td_loss: list[float] = field(default_factory=list)
    evaluation_episode: list[int] = field(default_factory=list)
    evaluation_success: list[bool] = field(default_factory=list)
    evaluation_J_A: list[float | None] = field(default_factory=list)
    evaluation_absolute_error: list[float | None] = field(default_factory=list)
    evaluation_relative_error: list[float | None] = field(default_factory=list)
    evaluation_inference_runtime_sec: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DQNTrainingRun:
    seed: int
    device: str
    history: DQNTrainingHistory
    best_episode: int
    best_evaluation: PolicyEvaluation
    final_evaluation: PolicyEvaluation
    training_runtime_sec: float
    environment_steps: int
    optimization_steps: int
    replay_size: int
    cached_observation_states: int
    best_model_state: dict[str, Tensor]
    final_model_state: dict[str, Tensor]
    final_optimizer_state: dict[str, Any]


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def epsilon_at_step(
    step: int, *, start: float, end: float, decay_steps: int,
) -> float:
    if decay_steps < 1:
        raise ValueError("decay_steps must be positive")
    fraction = min(max(int(step), 0) / decay_steps, 1.0)
    return float(start + fraction * (end - start))


def epsilon_greedy_action(
    q_values: NDArray[np.float32] | NDArray[np.float64],
    feasible_mask: NDArray[np.bool_],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    values = np.asarray(q_values)
    mask = np.asarray(feasible_mask, dtype=np.bool_)
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError("q_values and feasible_mask must be matching vectors")
    feasible = np.flatnonzero(mask)
    if not len(feasible):
        raise ValueError("cannot select an action without a feasible action")
    if rng.random() < float(epsilon):
        return int(feasible[int(rng.integers(len(feasible)))])
    masked = np.where(mask, values, -np.inf)
    return int(np.argmax(masked))


def _observation_batch(
    cache: ObservationTensorCache,
    state_ids: Iterable[int],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    observations = [cache.get(int(state_id)) for state_id in state_ids]
    spatial = torch.from_numpy(np.stack(
        [item.spatial for item in observations], axis=0,
    )).to(device=device)
    scalar = torch.from_numpy(np.stack(
        [item.scalar for item in observations], axis=0,
    )).to(device=device)
    return spatial, scalar


def _single_q_values(
    network: TerrainDQN,
    cache: ObservationTensorCache,
    state_id: int,
    device: torch.device,
) -> NDArray[np.float32]:
    spatial, scalar = _observation_batch(cache, (state_id,), device)
    with torch.inference_mode():
        values = network(spatial, scalar)[0].detach().cpu().numpy()
    return values.astype(np.float32, copy=False)


def _trajectory_is_feasible(
    problem: AttackerBRProblem, trajectory: tuple[int, ...],
) -> bool:
    if not trajectory:
        return False
    for source, target in zip(trajectory, trajectory[1:]):
        if not any(
            int(edge.target_id) == int(target)
            for edge in problem.successors(int(source))
        ):
            return False
    return True


def evaluate_policy(
    network: TerrainDQN,
    problem: AttackerBRProblem,
    catalog: HeadingActionCatalog,
    cache: ObservationTensorCache,
    start_state_ids: Iterable[int],
    device: torch.device,
) -> PolicyEvaluation:
    """Greedy evaluation with no learning, replay, normalization, or RNG updates."""

    starts = tuple(int(value) for value in start_state_ids)
    was_training = network.training
    network.eval()
    candidates: list[tuple[float, int, tuple[int, ...], float, float]] = []
    started = perf_counter()
    try:
        for start_id in starts:
            state_id = start_id
            trajectory = [state_id]
            reached = problem.is_terminal(state_id)
            for _ in range(max(1, int(problem.grid.altitude_count) - 1)):
                if reached:
                    break
                mask = catalog.feasible_mask(state_id)
                if not mask.any():
                    break
                q_values = _single_q_values(network, cache, state_id, device)
                action_id = epsilon_greedy_action(
                    q_values, mask, 0.0, np.random.default_rng(0),
                )
                transition = catalog.transition(state_id, action_id)
                state_id = transition.target_state_id
                trajectory.append(state_id)
                reached = transition.terminal
            path = tuple(trajectory)
            if not reached or not _trajectory_is_feasible(problem, path):
                continue
            scored = problem.evaluate_trajectory(path, reached_goal=True)
            powered = problem.powered_cost(start_id)
            objective = powered + scored.cost
            candidates.append((objective, start_id, path, scored.cost, powered))
    finally:
        network.train(was_training)
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
    policy: TerrainDQN,
    target: TerrainDQN,
    optimizer: torch.optim.Optimizer,
    batch: ReplayBatch,
    cache: ObservationTensorCache,
    device: torch.device,
    config: DQNTrainingConfig,
) -> float:
    spatial, scalar = _observation_batch(cache, batch.state_ids, device)
    next_spatial, next_scalar = _observation_batch(
        cache, batch.next_state_ids, device,
    )
    actions = torch.from_numpy(batch.action_ids).to(device=device)
    rewards = torch.from_numpy(batch.rewards).to(device=device)
    terminal = torch.from_numpy(batch.terminal).to(device=device)
    next_masks = torch.from_numpy(batch.next_feasible_masks).to(device=device)

    predicted = policy(spatial, scalar).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_q = target(next_spatial, next_scalar)
        bootstrap = masked_bootstrap_values(next_q, next_masks, terminal)
        expected = rewards + config.gamma * bootstrap
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


def _evaluation_errors(
    evaluation: PolicyEvaluation, bellman_J_A: float,
) -> tuple[float | None, float | None]:
    if evaluation.attacker_objective is None:
        return None, None
    absolute = abs(float(evaluation.attacker_objective) - float(bellman_J_A))
    relative = absolute / abs(float(bellman_J_A)) if abs(bellman_J_A) > 1.0e-15 else None
    return absolute, relative


def train_one_seed(
    problem: AttackerBRProblem,
    builder: TerrainObservationBuilder,
    catalog: HeadingActionCatalog,
    start_state_ids: Iterable[int],
    bellman_J_A: float,
    network_config: DQNNetworkConfig,
    training_config: DQNTrainingConfig,
    seed: int,
) -> DQNTrainingRun:
    seed_everything(seed)
    device = select_device(training_config.device)
    cache = ObservationTensorCache(builder)
    starts = tuple(int(value) for value in start_state_ids)
    env = DQNAttackerEnv(
        problem, builder, starts, action_catalog=catalog, observation_cache=cache,
    )
    policy = TerrainDQN(network_config).to(device)
    target = TerrainDQN(network_config).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=training_config.learning_rate,
    )
    replay = ReplayBuffer(
        training_config.replay_capacity, catalog.action_count, seed + 20_000,
    )
    action_rng = np.random.default_rng(seed + 10_000)
    history = DQNTrainingHistory()
    max_budget_steps = training_config.episodes * env.max_episode_steps
    epsilon_decay_steps = max(
        1, round(max_budget_steps * training_config.epsilon_decay_fraction),
    )
    environment_steps = 0
    optimization_steps = 0
    best_episode = 0
    best_evaluation: PolicyEvaluation | None = None
    best_model_state: dict[str, Tensor] | None = None

    started = perf_counter()
    for episode_index in range(training_config.episodes):
        reset_seed = seed if episode_index == 0 else None
        _, info = env.reset(seed=reset_seed)
        state_id = int(info["state_id"])
        start_id = state_id
        glide_return = 0.0
        reached_goal = False
        episode_steps = 0
        episode_epsilon = epsilon_at_step(
            environment_steps,
            start=training_config.epsilon_start,
            end=training_config.epsilon_end,
            decay_steps=epsilon_decay_steps,
        )
        while True:
            mask = np.asarray(info["action_mask"], dtype=np.bool_)
            q_values = _single_q_values(policy, cache, state_id, device)
            epsilon = epsilon_at_step(
                environment_steps,
                start=training_config.epsilon_start,
                end=training_config.epsilon_end,
                decay_steps=epsilon_decay_steps,
            )
            action_id = epsilon_greedy_action(
                q_values, mask, epsilon, action_rng,
            )
            next_observation, reward, terminated, truncated, next_info = env.step(
                action_id,
            )
            del next_observation
            next_state_id = int(next_info["state_id"])
            episode_ended = bool(terminated or truncated)
            if truncated:
                raise RuntimeError(
                    "authoritative altitude-decreasing graph exceeded its derived horizon"
                )
            replay.add(
                state_id=state_id,
                action_id=action_id,
                reward=reward,
                next_state_id=next_state_id,
                terminal=episode_ended,
                next_feasible_mask=np.asarray(
                    next_info["action_mask"], dtype=np.bool_,
                ),
            )
            environment_steps += 1
            episode_steps += 1
            glide_return += float(reward)
            if len(replay) >= training_config.replay_warmup:
                for _ in range(training_config.updates_per_environment_step):
                    loss = _optimize(
                        policy, target, optimizer,
                        replay.sample(training_config.batch_size),
                        cache, device, training_config,
                    )
                    optimization_steps += 1
                    history.optimization_step.append(optimization_steps)
                    history.td_loss.append(loss)
                    if optimization_steps % training_config.target_update_steps == 0:
                        target.load_state_dict(policy.state_dict())
            state_id = next_state_id
            info = next_info
            if episode_ended:
                reached_goal = bool(next_info["reached_goal"])
                break

        history.episode.append(episode_index + 1)
        history.episode_glide_return.append(glide_return)
        history.episode_full_transformed_return.append(
            glide_return - problem.powered_cost(start_id)
        )
        history.episode_steps.append(episode_steps)
        history.episode_reached_goal.append(reached_goal)
        history.epsilon.append(episode_epsilon)

        evaluate_now = (
            (episode_index + 1) % training_config.evaluation_interval_episodes == 0
            or episode_index + 1 == training_config.episodes
        )
        if evaluate_now:
            evaluation = evaluate_policy(
                policy, problem, catalog, cache, starts, device,
            )
            absolute, relative = _evaluation_errors(evaluation, bellman_J_A)
            history.evaluation_episode.append(episode_index + 1)
            history.evaluation_success.append(evaluation.success)
            history.evaluation_J_A.append(evaluation.attacker_objective)
            history.evaluation_absolute_error.append(absolute)
            history.evaluation_relative_error.append(relative)
            history.evaluation_inference_runtime_sec.append(
                evaluation.inference_runtime_sec,
            )
            if evaluation.success and (
                best_evaluation is None
                or float(evaluation.attacker_objective)
                < float(best_evaluation.attacker_objective) - 1.0e-15
            ):
                best_evaluation = evaluation
                best_episode = episode_index + 1
                best_model_state = _cpu_state_dict(policy)

    training_runtime = perf_counter() - started
    final_model_state = _cpu_state_dict(policy)
    final_evaluation = evaluate_policy(
        policy, problem, catalog, cache, starts, device,
    )
    if best_evaluation is None or best_model_state is None:
        best_evaluation = final_evaluation
        best_episode = training_config.episodes
        best_model_state = final_model_state
    policy.load_state_dict(best_model_state)
    best_evaluation = evaluate_policy(
        policy, problem, catalog, cache, starts, device,
    )
    return DQNTrainingRun(
        seed=int(seed), device=str(device), history=history,
        best_episode=best_episode, best_evaluation=best_evaluation,
        final_evaluation=final_evaluation,
        training_runtime_sec=training_runtime,
        environment_steps=environment_steps,
        optimization_steps=optimization_steps,
        replay_size=len(replay), cached_observation_states=cache.cached_states,
        best_model_state=best_model_state,
        final_model_state=final_model_state,
        final_optimizer_state=deepcopy(optimizer.state_dict()),
    )


def checkpoint_payload(
    *,
    model_state: dict[str, Tensor],
    network_config: DQNNetworkConfig,
    training_config: DQNTrainingConfig,
    compatibility: CompatibilitySignature,
    seed: int,
    episode: int,
    checkpoint_kind: str,
    optimizer_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_kind": str(checkpoint_kind),
        "seed": int(seed),
        "episode": int(episode),
        "network_config": network_config.as_dict(),
        "training_config": training_config.as_dict(),
        "compatibility": compatibility.as_dict(),
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_compatible_checkpoint(
    path: Path,
    expected_network: DQNNetworkConfig,
    expected_compatibility: CompatibilitySignature,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TerrainDQN, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported DQN checkpoint format")
    actual_network = DQNNetworkConfig.from_dict(payload["network_config"])
    if actual_network != expected_network:
        raise ValueError("checkpoint network configuration is incompatible")
    actual_signature = CompatibilitySignature.from_dict(payload["compatibility"])
    if actual_signature != expected_compatibility:
        raise ValueError("checkpoint observation/action/discretization signature is incompatible")
    network = TerrainDQN(expected_network)
    network.load_state_dict(payload["model_state_dict"])
    network.to(map_location)
    return network, payload


def write_history(path: Path, run: DQNTrainingRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "seed": run.seed,
            "device": run.device,
            "best_episode": run.best_episode,
            "best_evaluation": run.best_evaluation.as_dict(),
            "final_evaluation": run.final_evaluation.as_dict(),
            "training_runtime_sec": run.training_runtime_sec,
            "environment_steps": run.environment_steps,
            "optimization_steps": run.optimization_steps,
            "replay_size": run.replay_size,
            "cached_observation_states": run.cached_observation_states,
            "history": run.history.as_dict(),
        }, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CompatibilitySignature",
    "DQNTrainingConfig",
    "DQNTrainingHistory",
    "DQNTrainingRun",
    "PolicyEvaluation",
    "checkpoint_payload",
    "epsilon_at_step",
    "epsilon_greedy_action",
    "evaluate_policy",
    "load_compatible_checkpoint",
    "save_checkpoint",
    "seed_everything",
    "select_device",
    "train_one_seed",
    "write_history",
]

"""Replay the frozen G1 workload to benchmark observation-cache capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

import project_paths
from phase4_extended_dataset import (
    TRAIN_EXTENDED,
    ExtendedPilotConfig,
    build_extended_scenario_bank,
    extended_dataset_manifest,
)
from phase4_multi_terrain import _prepare_optimization_batches
from phase4_replay import MultiScenarioBatch, MultiScenarioReplayBuffer
from phase4_v2_g1_generalization import (
    CANONICAL_REFERENCE_PATH,
    build_g1_runtimes,
)


CHECKPOINT_PATH = (
    project_paths.DQN_DIR
    / "checkpoints"
    / "phase4_extended_random_boxes"
    / "training_v2"
    / "g1_generalization"
    / "dx25m_dpsi5deg"
    / "seed0_ep10000_v1"
    / "phase4_seed0_resume.pt"
)
OUTPUT_PATH = (
    project_paths.CORE_DIR
    / "figure"
    / "phase_4_extended_random_boxes"
    / "training_v2"
    / "g1_generalization"
    / "dx25m_dpsi5deg"
    / "seed0_ep10000_v1"
    / "closeout_v1"
    / "g1_observation_cache_benchmark.json"
)


def _tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _prepared_digest(prepared: Any) -> dict[str, str]:
    return {
        name: _tensor_digest(getattr(prepared, name))
        for name in (
            "spatial", "scalar", "next_spatial", "next_scalar", "actions",
            "rewards", "terminal", "next_masks", "bootstrap_steps",
        )
    }


def _reset_caches(contexts: list[Any], capacity: int) -> None:
    for context in contexts:
        context.cache.clear()
        context.cache.max_entries = int(capacity)
        context.cache.reset_statistics()
        context.builder.clear_runtime_caches()


def _cache_totals(contexts: list[Any]) -> dict[str, Any]:
    values = [context.cache.statistics() for context in contexts]
    summed = {
        key: sum(int(value[key]) for value in values)
        for key in (
            "cached_states", "requests", "cache_hits", "batch_reuses",
            "cache_misses", "builds", "evictions",
        )
    }
    summed["reuse_rate"] = (
        (summed["cache_hits"] + summed["batch_reuses"]) / summed["requests"]
        if summed["requests"] else 0.0
    )
    summed["per_scenario"] = values
    return summed


def _run_workload(
    contexts: list[Any],
    workload: list[list[MultiScenarioBatch]],
    *, capacity: int, passes: int, device: torch.device,
) -> dict[str, Any]:
    _reset_caches(contexts, capacity)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = perf_counter()
    prepared_groups = 0
    for _ in range(int(passes)):
        for batches in workload:
            prepared = _prepare_optimization_batches(batches, contexts, device)
            prepared_groups += 1
            del prepared
    torch.cuda.synchronize(device)
    runtime = perf_counter() - started
    return {
        "capacity_per_scenario": int(capacity),
        "passes": int(passes),
        "prepared_groups": prepared_groups,
        "optimizer_batches_represented": prepared_groups * len(workload[0]),
        "runtime_sec": runtime,
        "sec_per_optimizer_batch": (
            runtime / (prepared_groups * len(workload[0]))
        ),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cache": _cache_totals(contexts),
    }


def benchmark_cache(
    *, groups: int = 24, prefetch_batches: int = 8, passes: int = 2,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("the G1 observation-cache benchmark requires CUDA")
    if min(int(groups), int(prefetch_batches), int(passes)) < 1:
        raise ValueError("benchmark sizes must be positive")
    device = torch.device("cuda")
    pilot_config = ExtendedPilotConfig(
        train_terrain_count=4, validation_terrain_count=2,
    )
    manifest = extended_dataset_manifest(pilot_config)
    splits, _ = build_extended_scenario_bank(pilot_config)
    contexts, failures, metadata = build_g1_runtimes(
        splits[TRAIN_EXTENDED],
        pilot_config=pilot_config,
        spatial_resolution_m=25.0,
        heading_spacing_deg=5.0,
        dataset_hash=manifest["manifest_sha256"],
        reference_path=CANONICAL_REFERENCE_PATH,
        observation_device=device,
    )
    payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    replay = MultiScenarioReplayBuffer.from_state_dict(
        payload["replay_state_dict"]
    )
    workload = [
        [replay.sample(128) for _ in range(int(prefetch_batches))]
        for _ in range(int(groups))
    ]

    # A full-tensor digest proves that capacity changes only reuse, not values.
    equivalence_batch = workload[0]
    equivalence: dict[str, dict[str, str]] = {}
    for label, capacity in (("baseline_2048", 2_048), ("optimized_4096", 4_096)):
        _reset_caches(contexts, capacity)
        prepared = _prepare_optimization_batches(
            equivalence_batch, contexts, device,
        )
        equivalence[label] = _prepared_digest(prepared)
        del prepared
    bitwise_equal = equivalence["baseline_2048"] == equivalence["optimized_4096"]
    if not bitwise_equal:
        raise RuntimeError("cache-capacity change altered prepared observations")

    baseline = _run_workload(
        contexts, workload, capacity=2_048, passes=passes, device=device,
    )
    optimized = _run_workload(
        contexts, workload, capacity=4_096, passes=passes, device=device,
    )
    result = {
        "schema": "p1b-g1-observation-cache-benchmark-v1",
        "device": torch.cuda.get_device_name(device),
        "checkpoint": str(CHECKPOINT_PATH),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "training_context_count": len(contexts),
        "excluded_training_missions": failures,
        "runtime_metadata": metadata,
        "workload": {
            "groups": int(groups),
            "prefetch_batches_per_group": int(prefetch_batches),
            "batch_size": 128,
            "passes": int(passes),
            "transition_samples_per_pass": (
                int(groups) * int(prefetch_batches) * 128
            ),
            "current_and_next_observation_requests_per_pass": (
                int(groups) * int(prefetch_batches) * 128 * 2
            ),
        },
        "numerical_equivalence": {
            "bitwise_equal": bitwise_equal,
            "tensor_sha256": equivalence["baseline_2048"],
        },
        "baseline_2048": baseline,
        "optimized_4096": optimized,
        "comparison": {
            "speedup": baseline["runtime_sec"] / optimized["runtime_sec"],
            "runtime_reduction_fraction": (
                1.0 - optimized["runtime_sec"] / baseline["runtime_sec"]
            ),
            "build_reduction_fraction": (
                1.0
                - optimized["cache"]["builds"] / baseline["cache"]["builds"]
            ),
            "eviction_reduction_fraction": (
                1.0
                - optimized["cache"]["evictions"]
                / baseline["cache"]["evictions"]
                if baseline["cache"]["evictions"] else 0.0
            ),
        },
        "scope": (
            "frozen replay observation preparation only; no optimizer update and "
            "no policy training"
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return OUTPUT_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=24)
    parser.add_argument("--prefetch-batches", type=int, default=8)
    parser.add_argument("--passes", type=int, default=2)
    arguments = parser.parse_args()
    print(benchmark_cache(
        groups=arguments.groups,
        prefetch_batches=arguments.prefetch_batches,
        passes=arguments.passes,
    ))


if __name__ == "__main__":
    main()


__all__ = ["benchmark_cache"]

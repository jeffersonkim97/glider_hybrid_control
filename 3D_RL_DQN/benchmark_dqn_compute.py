"""Measure Phase 16.4 neural-network compute without environment work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from dqn_model import DQNNetworkConfig, TerrainDQN, masked_bootstrap_values


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(iterations: int, device_name: str) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(20261006)
    config = DQNNetworkConfig()
    policy = TerrainDQN(config).to(device)
    target = TerrainDQN(config).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3.0e-4)

    def inputs(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.randn(
                batch_size, config.spatial_channels,
                config.spatial_height, config.spatial_width, device=device,
            ),
            torch.randn(batch_size, config.scalar_features, device=device),
        )

    def time_inference(batch_size: int, count: int) -> float:
        spatial, scalar = inputs(batch_size)
        for _ in range(50):
            with torch.inference_mode():
                policy(spatial, scalar)
        _synchronize(device)
        started = perf_counter()
        for _ in range(count):
            with torch.inference_mode():
                policy(spatial, scalar)
        _synchronize(device)
        return (perf_counter() - started) / count

    batch_size = 128
    spatial, scalar = inputs(batch_size)
    next_spatial, next_scalar = inputs(batch_size)
    actions = torch.randint(config.action_count, (batch_size,), device=device)
    rewards = torch.randn(batch_size, device=device)
    terminal = torch.zeros(batch_size, dtype=torch.bool, device=device)
    masks = torch.ones(
        batch_size, config.action_count, dtype=torch.bool, device=device,
    )

    def update() -> None:
        predicted = policy(spatial, scalar).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_q = target(next_spatial, next_scalar)
            expected = rewards + masked_bootstrap_values(next_q, masks, terminal)
        loss = F.smooth_l1_loss(predicted, expected)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()

    for _ in range(50):
        update()
    _synchronize(device)
    started = perf_counter()
    for _ in range(iterations):
        update()
    _synchronize(device)
    update_sec = (perf_counter() - started) / iterations

    measured_steps = 803_088
    measured_updates = 801_089
    forward_one_sec = time_inference(1, max(1000, iterations * 4))
    forward_eight_sec = time_inference(8, max(1000, iterations * 2))
    projected_neural_sec = (
        measured_updates * update_sec + measured_steps * forward_one_sec
    )
    return {
        "schema": "p1b-phase4-dqn-kernel-benchmark-v1",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "network_parameter_count": policy.parameter_count,
        "iterations": iterations,
        "inference_batch1_ms": 1000.0 * forward_one_sec,
        "inference_batch8_ms": 1000.0 * forward_eight_sec,
        "inference_batch8_per_state_ms": 1000.0 * forward_eight_sec / 8.0,
        "training_update_batch128_ms": 1000.0 * update_sec,
        "seed0_measured_environment_steps": measured_steps,
        "seed0_measured_optimizer_updates": measured_updates,
        "projected_seed0_neural_compute_sec": projected_neural_sec,
        "projected_seed0_neural_compute_hours": projected_neural_sec / 3600.0,
        "scope": (
            "network inference plus forward/backward/AdamW only; excludes "
            "observation generation, replay sampling, host-device transfer, "
            "environment transitions, evaluation, and checkpoint I/O"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = benchmark(arguments.iterations, arguments.device)
    rendered = json.dumps(result, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

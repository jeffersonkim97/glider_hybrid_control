"""Approved two-branch Phase 16.3 DQN and masked Q-value operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DQNNetworkConfig:
    spatial_channels: int = 7
    spatial_height: int = 21
    spatial_width: int = 21
    scalar_features: int = 8
    action_count: int = 72
    convolution_channels: tuple[int, int, int] = (32, 64, 64)
    spatial_embedding: int = 256
    scalar_hidden: tuple[int, int] = (64, 64)
    fusion_hidden: int = 256
    activation: str = "relu"
    architecture_id: str = "p1b-two-branch-dqn-v1"

    def __post_init__(self) -> None:
        positive = (
            self.spatial_channels, self.spatial_height, self.spatial_width,
            self.scalar_features, self.action_count, self.spatial_embedding,
            self.fusion_hidden, *self.convolution_channels, *self.scalar_hidden,
        )
        if any(int(value) < 1 for value in positive):
            raise ValueError("all network dimensions must be positive")
        if len(self.convolution_channels) != 3:
            raise ValueError("the approved architecture uses three convolution layers")
        if len(self.scalar_hidden) != 2:
            raise ValueError("the approved scalar branch uses two hidden layers")
        if self.activation != "relu":
            raise ValueError("the approved Phase 16.3 activation is relu")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["convolution_channels"] = list(self.convolution_channels)
        result["scalar_hidden"] = list(self.scalar_hidden)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DQNNetworkConfig":
        copied = dict(values)
        copied["convolution_channels"] = tuple(copied["convolution_channels"])
        copied["scalar_hidden"] = tuple(copied["scalar_hidden"])
        return cls(**copied)


class TerrainDQN(nn.Module):
    """Keep spatial perception and scalar navigation features separate."""

    def __init__(self, config: DQNNetworkConfig = DQNNetworkConfig()) -> None:
        super().__init__()
        self.config = config
        c1, c2, c3 = config.convolution_channels
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(config.spatial_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            probe = torch.zeros(
                1, config.spatial_channels,
                config.spatial_height, config.spatial_width,
            )
            flattened = int(self.spatial_branch(probe).shape[1])
        self.spatial_projection = nn.Sequential(
            nn.Linear(flattened, config.spatial_embedding),
            nn.ReLU(),
        )
        scalar_first, scalar_second = config.scalar_hidden
        self.scalar_branch = nn.Sequential(
            nn.Linear(config.scalar_features, scalar_first),
            nn.ReLU(),
            nn.Linear(scalar_first, scalar_second),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(
                config.spatial_embedding + scalar_second,
                config.fusion_hidden,
            ),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden, config.action_count),
        )

    def forward(self, spatial: Tensor, scalar: Tensor) -> Tensor:
        if spatial.ndim != 4:
            raise ValueError("spatial input must have shape (B,C,H,W)")
        if scalar.ndim != 2:
            raise ValueError("scalar input must have shape (B,F)")
        spatial_embedding = self.spatial_projection(self.spatial_branch(spatial))
        scalar_embedding = self.scalar_branch(scalar)
        return self.fusion(torch.cat((spatial_embedding, scalar_embedding), dim=1))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def masked_greedy_actions(q_values: Tensor, feasible_mask: Tensor) -> Tensor:
    """Argmax over feasible actions only."""

    if q_values.shape != feasible_mask.shape:
        raise ValueError("q_values and feasible_mask must have identical shapes")
    mask = feasible_mask.to(dtype=torch.bool)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every greedy-selection row needs a feasible action")
    masked = q_values.masked_fill(~mask, torch.finfo(q_values.dtype).min)
    return masked.argmax(dim=1)


def masked_bootstrap_values(
    q_values: Tensor,
    feasible_mask: Tensor,
    terminal: Tensor,
) -> Tensor:
    """Maximum next Q with no bootstrap at terminal transitions."""

    if q_values.shape != feasible_mask.shape:
        raise ValueError("q_values and feasible_mask must have identical shapes")
    done = terminal.to(dtype=torch.bool).reshape(-1)
    if done.shape[0] != q_values.shape[0]:
        raise ValueError("terminal batch length must match Q-value rows")
    mask = feasible_mask.to(dtype=torch.bool)
    invalid_nonterminal = (~done) & (~mask.any(dim=1))
    if torch.any(invalid_nonterminal):
        raise ValueError("a nonterminal bootstrap row has no feasible action")
    safe_mask = mask.clone()
    safe_mask[done, 0] = True
    masked = q_values.masked_fill(~safe_mask, torch.finfo(q_values.dtype).min)
    values = masked.max(dim=1).values
    return torch.where(done, torch.zeros_like(values), values)


__all__ = [
    "DQNNetworkConfig",
    "TerrainDQN",
    "masked_bootstrap_values",
    "masked_greedy_actions",
]

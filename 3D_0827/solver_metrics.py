"""Structured runtime instrumentation for exact Bellman solves."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class SolverTiming:
    """Wall-clock durations recorded by a complete solver run."""

    graph_build_s: float
    solve_s: float
    backtrack_s: float

    def __post_init__(self) -> None:
        for name in ("graph_build_s", "solve_s", "backtrack_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SolverMetrics:
    """Programmatic timing and graph-size report."""

    timing: SolverTiming
    state_count: int
    edge_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.timing, SolverTiming):
            raise TypeError("timing must be a SolverTiming")
        for name in ("state_count", "edge_count"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class HazardTiming:
    """Runtime isolated from graph construction and Bellman solution time."""

    precompute_s: float
    per_edge_s: float
    edge_count: int
    quadrature_resolution: int

    def __post_init__(self) -> None:
        for name in ("precompute_s", "per_edge_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        for name in ("edge_count", "quadrature_resolution"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))
        if self.quadrature_resolution < 2:
            raise ValueError("quadrature_resolution must be at least two")

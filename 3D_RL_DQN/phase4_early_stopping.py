"""Approved validation-quality stopping rule for extended Phase 16.4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EarlyStoppingConfig:
    evaluation_interval_episodes: int = 1_500
    minimum_episodes: int = 10_000
    stability_evaluations: int = 3
    median_error_range_threshold: float = 0.10
    median_relative_error_threshold: float = 0.10
    # Retained as a reporting reference for backward-compatible manifests.
    # Maximum error is no longer an early-stopping quality gate.
    maximum_relative_error_threshold: float = 0.20
    maximum_episodes: int = 60_000

    def __post_init__(self) -> None:
        counts = (
            self.evaluation_interval_episodes,
            self.minimum_episodes,
            self.stability_evaluations,
            self.maximum_episodes,
        )
        if any(int(value) < 1 for value in counts):
            raise ValueError("early-stopping counts must be positive")
        if self.minimum_episodes > self.maximum_episodes:
            raise ValueError("minimum episodes cannot exceed maximum episodes")
        thresholds = (
            self.median_error_range_threshold,
            self.median_relative_error_threshold,
            self.maximum_relative_error_threshold,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in thresholds):
            raise ValueError("early-stopping thresholds must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["parameter_origin"] = "user_approved_2026-10-30"
        result["median_error_range_unit"] = (
            "absolute proportion (0.10 = 10 percentage points)"
        )
        result["quality_window_policy"] = (
            "all recent evaluations must have 100% success; the median of their "
            "median relative errors and the latest median relative error must be "
            "at most 10%; their median-error range must be at most 10 percentage "
            "points"
        )
        result["maximum_error_policy"] = "report_only_not_a_stopping_gate"
        result["final_model_acceptance"] = {
            "success_rate_minimum": 1.0,
            "median_relative_error_maximum": 0.10,
            "p90_relative_error_maximum": 0.30,
            "p90_method": "higher",
            "maximum_relative_error_policy": "report_only",
        }
        result["failure_plateau_policy"] = "continue_to_maximum_episodes"
        return result


@dataclass(frozen=True)
class ValidationPoint:
    episode: int
    success_rate: float
    median_relative_error: float | None
    maximum_relative_error: float | None


@dataclass(frozen=True)
class EarlyStoppingDecision:
    stop: bool
    success: bool
    reason: str
    quality_passed: bool
    stability_passed: bool


@dataclass(frozen=True)
class FinalAcceptanceDecision:
    accepted: bool
    success_rate: float
    median_relative_error: float | None
    p90_relative_error: float | None
    maximum_relative_error: float | None


def evaluate_final_acceptance(
    *, success_count: int, case_count: int,
    successful_relative_errors: list[float] | tuple[float, ...],
) -> FinalAcceptanceDecision:
    """Apply the approved 2026-10-30 final-model acceptance rule."""

    if case_count < 1:
        raise ValueError("final acceptance requires at least one case")
    if success_count < 0 or success_count > case_count:
        raise ValueError("success count must be within the case count")
    errors = np.asarray(successful_relative_errors, dtype=float)
    if len(errors) != success_count:
        raise ValueError("one relative error is required per successful case")
    if len(errors) and (not np.all(np.isfinite(errors)) or np.any(errors < 0.0)):
        raise ValueError("relative errors must be finite and nonnegative")

    success_rate = float(success_count / case_count)
    median_error = None if not len(errors) else float(np.median(errors))
    p90_error = (
        None
        if not len(errors)
        else float(np.percentile(errors, 90.0, method="higher"))
    )
    maximum_error = None if not len(errors) else float(np.max(errors))
    accepted = bool(
        success_rate >= 1.0
        and median_error is not None
        and median_error <= 0.10
        and p90_error is not None
        and p90_error <= 0.30
    )
    return FinalAcceptanceDecision(
        accepted=accepted,
        success_rate=success_rate,
        median_relative_error=median_error,
        p90_relative_error=p90_error,
        maximum_relative_error=maximum_error,
    )


@dataclass
class EarlyStoppingTracker:
    config: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    history: list[ValidationPoint] = field(default_factory=list)

    def update(self, point: ValidationPoint) -> EarlyStoppingDecision:
        if self.history and point.episode <= self.history[-1].episode:
            raise ValueError("validation episodes must increase strictly")
        if point.episode % self.config.evaluation_interval_episodes != 0:
            raise ValueError("validation point is off the approved evaluation cadence")
        self.history.append(point)
        recent = self.history[-self.config.stability_evaluations:]
        complete_window = len(recent) == self.config.stability_evaluations
        median_values = np.asarray([
            item.median_relative_error for item in recent
            if item.median_relative_error is not None
        ], dtype=float)
        finite_median_window = bool(
            complete_window
            and len(median_values) == self.config.stability_evaluations
            and np.all(np.isfinite(median_values))
        )
        quality = bool(
            finite_median_window
            and all(item.success_rate >= 1.0 for item in recent)
            and float(np.median(median_values))
            <= self.config.median_relative_error_threshold
            and float(median_values[-1])
            <= self.config.median_relative_error_threshold
        )
        stability = bool(
            finite_median_window
            and float(np.max(median_values) - np.min(median_values))
            <= self.config.median_error_range_threshold
        )
        if point.episode >= self.config.minimum_episodes and quality and stability:
            return EarlyStoppingDecision(
                stop=True,
                success=True,
                reason="approved_accuracy_reached_and_stable",
                quality_passed=True,
                stability_passed=True,
            )
        if point.episode >= self.config.maximum_episodes:
            return EarlyStoppingDecision(
                stop=True,
                success=False,
                reason="maximum_episode_budget_reached_without_approved_accuracy",
                quality_passed=quality,
                stability_passed=stability,
            )
        return EarlyStoppingDecision(
            stop=False,
            success=False,
            reason=(
                "minimum_episode_budget_not_reached"
                if point.episode < self.config.minimum_episodes
                else "quality_or_stability_gate_not_met"
            ),
            quality_passed=quality,
            stability_passed=stability,
        )


__all__ = [
    "EarlyStoppingConfig", "EarlyStoppingDecision", "EarlyStoppingTracker",
    "FinalAcceptanceDecision", "ValidationPoint", "evaluate_final_acceptance",
]

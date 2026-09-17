"""Independent validation adapter for a selected finite Stackelberg outcome."""

from __future__ import annotations

from dataclasses import replace

from attacker_best_response import AttackerBestResponseRun, attacker_response_from_result
from detection_hazard import GlideDetectionHazardModel
from stage11_config import Stage11Config
from stackelberg_solver import FiniteStackelbergRun
from trajectory_validation import (
    TrajectoryReplayAudit,
    snapshot_selected_trajectory,
    validate_trajectory_replay,
)


def selected_attacker_run(run: FiniteStackelbergRun) -> AttackerBestResponseRun:
    """Return a view whose selected result is the SSE follower, not ID tie-break."""
    evaluation = run.selected_evaluation
    follower = evaluation.sse_follower_result
    if follower is None:
        raise ValueError("selected Defender action has no feasible follower response")
    return replace(
        evaluation.attacker_run,
        selected_result=follower,
        response=attacker_response_from_result(follower),
    )


def validate_selected_stackelberg_trajectory(
    run: FiniteStackelbergRun,
    config: Stage11Config,
) -> TrajectoryReplayAudit:
    """Run the Stage-10 independent replay on the equilibrium trajectory."""
    selected = selected_attacker_run(run)
    snapshot = snapshot_selected_trajectory(
        selected,
        hazard_quadrature_resolution=(
            config.discretization.hazard_quadrature_resolution
        ),
    )
    hazard = GlideDetectionHazardModel(
        selected.terrain,
        selected.mission_points.sensor,
        parameters=config.detection,
        physical_scale=config.physical_scale,
    )
    return validate_trajectory_replay(
        snapshot,
        selected.terrain,
        selected.mission_points,
        selected.tangent_contour,
        hazard,
        parameters=config.glider,
        physical_scale=config.physical_scale,
    )


__all__ = [
    "selected_attacker_run", "validate_selected_stackelberg_trajectory",
]

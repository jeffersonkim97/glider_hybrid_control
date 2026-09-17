"""Game-level conventions that remain outside the future Bellman solver."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from attacker_best_response import attacker_best_response
from game_types import GameOutcome


def select_sse_follower_outcome(
    outcomes: Sequence[GameOutcome],
    *,
    objective_tolerance: float = 0.0,
) -> GameOutcome:
    """Apply the Strong Stackelberg follower tie-breaking convention.

    All outcomes must belong to one fixed Defender commitment. First select
    the minimum Attacker cost. Among responses tied within the requested
    numerical tolerance, select maximum Defender payoff. A remaining exact
    tie is resolved by stable input order.
    """
    tolerance = float(objective_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("objective_tolerance must be finite and nonnegative")
    if not outcomes:
        raise ValueError("at least one GameOutcome is required")
    if any(not isinstance(outcome, GameOutcome) for outcome in outcomes):
        raise TypeError("outcomes must contain only GameOutcome values")

    committed_position = outcomes[0].defender_action.sensor_position_map
    if any(
        not np.array_equal(
            outcome.defender_action.sensor_position_map,
            committed_position,
        )
        for outcome in outcomes[1:]
    ):
        raise ValueError("SSE follower tie-breaking requires one Defender action")

    minimum_attacker_cost = min(outcome.attacker_payoff for outcome in outcomes)
    follower_best_responses = [
        outcome
        for outcome in outcomes
        if outcome.attacker_payoff <= minimum_attacker_cost + tolerance
    ]
    return max(follower_best_responses, key=lambda outcome: outcome.defender_payoff)


__all__ = ["attacker_best_response", "select_sse_follower_outcome"]

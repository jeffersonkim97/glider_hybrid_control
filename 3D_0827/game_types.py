"""Typed Stage-2.5 data boundary for the future Stackelberg game.

Sign convention
---------------
``AttackerResponse.objective`` and ``GameOutcome.attacker_payoff`` are costs
that the Attacker minimizes. ``GameOutcome.defender_payoff`` is a utility that
the Defender maximizes. The two values are not assumed to be zero-sum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scenario import Point3D


FloatArray = NDArray[np.float64]
DiscreteState = tuple[int, ...]

ATTACKER_OBJECTIVE_SENSE = "minimize"
DEFENDER_PAYOFF_SENSE = "maximize"
ATTACKER_OBJECTIVE_COMPONENTS = ("mission_time_s", "detection_probability")
DEFENDER_OBJECTIVE_COMPONENTS = ("detection_probability",)
ZERO_SUM_ASSUMED = False
SSE_TIE_BREAK_CONVENTION = (
    "Among equal Attacker minimum-cost responses, select the response with "
    "maximum Defender payoff."
)


def _immutable_vector3(values: FloatArray, name: str) -> FloatArray:
    vector = np.array(values, dtype=float, copy=True)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain exactly three finite coordinates")
    vector.setflags(write=False)
    return vector


def _finite_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validated_discrete_states(
    states: tuple[DiscreteState, ...],
) -> tuple[DiscreteState, ...]:
    if not isinstance(states, tuple):
        raise TypeError("discrete_states must be a tuple")
    normalized: list[DiscreteState] = []
    for state in states:
        if not isinstance(state, tuple) or not state:
            raise ValueError("each discrete state must be a nonempty tuple of integers")
        if any(not isinstance(index, int) or isinstance(index, bool) for index in state):
            raise ValueError("each discrete state index must be an integer")
        normalized.append(tuple(state))
    return tuple(normalized)


@dataclass(frozen=True)
class DefenderAction:
    """One committed Defender action expressed as a sensor map position."""

    sensor_position_map: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sensor_position_map",
            _immutable_vector3(self.sensor_position_map, "sensor_position_map"),
        )


@dataclass(frozen=True)
class AttackerInitialCondition:
    """Game-neutral Attacker mission endpoints."""

    start: Point3D
    goal: Point3D

    def __post_init__(self) -> None:
        if not isinstance(self.start, Point3D) or not isinstance(self.goal, Point3D):
            raise TypeError("start and goal must be Point3D values")
        if np.linalg.norm(self.start.as_array() - self.goal.as_array()) <= 0.0:
            raise ValueError("start and goal must be distinct")


@dataclass(frozen=True)
class AttackerResponse:
    """Stable result returned by a future Attacker best-response solver."""

    feasible: bool
    switching_point_map: FloatArray | None
    discrete_states: tuple[DiscreteState, ...]
    objective: float
    mission_time_s: float
    cumulative_hazard: float
    detection_probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.feasible, (bool, np.bool_)):
            raise TypeError("feasible must be boolean")
        object.__setattr__(self, "feasible", bool(self.feasible))

        switching_point = self.switching_point_map
        if self.feasible and switching_point is None:
            raise ValueError("a feasible response requires a switching point")
        if not self.feasible and switching_point is not None:
            raise ValueError("an infeasible response cannot report a switching point")
        if switching_point is not None:
            object.__setattr__(
                self,
                "switching_point_map",
                _immutable_vector3(switching_point, "switching_point_map"),
            )

        states = _validated_discrete_states(self.discrete_states)
        if self.feasible and not states:
            raise ValueError("a feasible response requires at least one discrete state")
        object.__setattr__(self, "discrete_states", states)

        objective = float(self.objective)
        mission_time = float(self.mission_time_s)
        if self.feasible:
            objective = _finite_scalar(objective, "objective")
            mission_time = _finite_scalar(mission_time, "mission_time_s")
        else:
            if np.isnan(objective) or np.isneginf(objective):
                raise ValueError("infeasible objective must be finite or positive infinity")
            if np.isnan(mission_time) or np.isneginf(mission_time):
                raise ValueError(
                    "infeasible mission_time_s must be finite or positive infinity"
                )
        cumulative_hazard = _finite_scalar(
            self.cumulative_hazard, "cumulative_hazard",
        )
        detection_probability = _finite_scalar(
            self.detection_probability, "detection_probability",
        )
        if mission_time < 0.0:
            raise ValueError("mission_time_s cannot be negative")
        if cumulative_hazard < 0.0:
            raise ValueError("cumulative_hazard cannot be negative")
        if not 0.0 <= detection_probability <= 1.0:
            raise ValueError("detection_probability must lie in [0, 1]")
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "mission_time_s", mission_time)
        object.__setattr__(self, "cumulative_hazard", cumulative_hazard)
        object.__setattr__(self, "detection_probability", detection_probability)


@dataclass(frozen=True)
class GameOutcome:
    """Payoffs for one Defender commitment and one Attacker response.

    Despite the field name retained from the implementation plan,
    ``attacker_payoff`` follows the Attacker cost convention and is minimized.
    ``defender_payoff`` is independently maximized; zero-sum is not assumed.
    """

    defender_action: DefenderAction
    attacker_response: AttackerResponse
    defender_payoff: float
    attacker_payoff: float

    def __post_init__(self) -> None:
        if not isinstance(self.defender_action, DefenderAction):
            raise TypeError("defender_action must be a DefenderAction")
        if not isinstance(self.attacker_response, AttackerResponse):
            raise TypeError("attacker_response must be an AttackerResponse")
        object.__setattr__(
            self,
            "defender_payoff",
            _finite_scalar(self.defender_payoff, "defender_payoff"),
        )
        object.__setattr__(
            self,
            "attacker_payoff",
            _finite_scalar(self.attacker_payoff, "attacker_payoff"),
        )

    @property
    def attacker_objective(self) -> float:
        """Explicit cost-oriented alias for ``attacker_payoff``."""
        return self.attacker_payoff

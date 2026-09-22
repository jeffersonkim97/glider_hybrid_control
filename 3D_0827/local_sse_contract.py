"""Mathematical and data contract for a finite-discretized local SSE.

For every feasible Defender action ``d``, the induced value is
``V_D(d) = J_D(a_SSE(d), d)`` where ``a_SSE(d)`` is still the exact finite
Attacker best response with the existing Strong Stackelberg follower
tie-breaking rule.  A candidate ``d*`` is a local SSE leader action iff
``V_D(d*) >= V_D(d)`` for every explicitly configured feasible grid neighbor.

This module certifies only neighborhood optimality.  It neither searches the
Defender space nor changes the existing exact Attacker/global-SSE solvers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np


LOCAL_SSE_SCHEMA_VERSION = "stage14.2A-v1"
GRID_INDEX_METRIC = "chebyshev_grid_index"
DEFAULT_R_NEIGHBOR = 1
LOCAL_SSE_SCOPE = (
    "local Strong Stackelberg equilibrium over an explicit finite Defender "
    "grid neighborhood; not a global or continuous-space optimality claim"
)
GLOBAL_SSE_SCOPE = "compare the induced Defender value against every feasible d in D"
LOCAL_SSE_CONDITION = (
    "V_D(d*) >= V_D(d) for every d in N_r(d*) intersect D_feas, within the "
    "configured leader numerical tolerance"
)
DISCRETIZED_EXISTENCE_CONDITION = (
    "A local SSE exists on every nonempty finite connected component of the "
    "feasible Defender-neighborhood graph when exact finite Attacker best "
    "responses, Strong tie-breaking, and finite J_A/J_D values are well-defined. "
    "This is not a global-optimality or continuous-game existence claim."
)


def _nonnegative_tolerance(value: float) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("leader_payoff_tolerance must be finite and nonnegative")
    return tolerance


@dataclass(frozen=True)
class DefenderNeighborhoodConfig:
    """Configurable grid-index radius for the local leader comparison."""

    r_neighbor: int = DEFAULT_R_NEIGHBOR
    metric: str = GRID_INDEX_METRIC

    def __post_init__(self) -> None:
        if (
            not isinstance(self.r_neighbor, Integral)
            or isinstance(self.r_neighbor, bool)
            or int(self.r_neighbor) < 1
        ):
            raise ValueError("r_neighbor must be an integer greater than or equal to one")
        if self.metric != GRID_INDEX_METRIC:
            raise ValueError(f"metric must be {GRID_INDEX_METRIC!r}")
        object.__setattr__(self, "r_neighbor", int(self.r_neighbor))

    def as_metadata(self) -> dict[str, Any]:
        return {
            "r_neighbor": self.r_neighbor,
            "metric": self.metric,
            "radius_units": "integer Defender grid-index lengths",
            "one_dimensional_rule": "0 < abs(j-i) <= r_neighbor",
            "two_dimensional_rule": (
                "0 < max(abs(p-i), abs(q-j)) <= r_neighbor"
            ),
            "boundary_rule": "clip to explicitly configured grid actions",
        }


@dataclass(frozen=True)
class DefenderGridTopology:
    """Deterministic action-ID to integer grid-index mapping."""

    action_grid_indices: tuple[tuple[int, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        if not self.action_grid_indices:
            raise ValueError("Defender grid topology cannot be empty")
        normalized: list[tuple[int, tuple[int, ...]]] = []
        dimensions: set[int] = set()
        for action_id, coordinate in self.action_grid_indices:
            if (
                not isinstance(action_id, Integral)
                or isinstance(action_id, bool)
                or int(action_id) < 0
            ):
                raise ValueError("Defender action IDs must be nonnegative integers")
            if (
                not isinstance(coordinate, tuple)
                or not coordinate
                or any(
                    not isinstance(index, Integral) or isinstance(index, bool)
                    for index in coordinate
                )
            ):
                raise ValueError("grid indices must be nonempty integer tuples")
            normalized_coordinate = tuple(int(index) for index in coordinate)
            dimensions.add(len(normalized_coordinate))
            normalized.append((int(action_id), normalized_coordinate))
        if len(dimensions) != 1:
            raise ValueError("all Defender grid indices must have one common dimension")
        action_ids = tuple(item[0] for item in normalized)
        coordinates = tuple(item[1] for item in normalized)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("Defender action IDs must be unique")
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("Defender grid indices must be unique")
        object.__setattr__(self, "action_grid_indices", tuple(normalized))

    @classmethod
    def ordered_line(cls, action_ids: Sequence[int]) -> "DefenderGridTopology":
        return cls(tuple((int(action_id), (index,)) for index, action_id in enumerate(action_ids)))

    @property
    def dimension(self) -> int:
        return len(self.action_grid_indices[0][1])

    @property
    def action_ids(self) -> tuple[int, ...]:
        return tuple(action_id for action_id, _ in self.action_grid_indices)

    def coordinate(self, action_id: int) -> tuple[int, ...]:
        for candidate_id, coordinate in self.action_grid_indices:
            if candidate_id == action_id:
                return coordinate
        raise KeyError(f"Defender action {action_id} is not in the grid topology")

    def neighbors(
        self,
        action_id: int,
        configuration: DefenderNeighborhoodConfig,
    ) -> tuple[int, ...]:
        if not isinstance(configuration, DefenderNeighborhoodConfig):
            raise TypeError("configuration must be DefenderNeighborhoodConfig")
        center = self.coordinate(action_id)
        neighbors: list[int] = []
        for candidate_id, coordinate in self.action_grid_indices:
            if candidate_id == action_id:
                continue
            distance = max(abs(value - origin) for value, origin in zip(coordinate, center))
            if 0 < distance <= configuration.r_neighbor:
                neighbors.append(candidate_id)
        return tuple(neighbors)

    def radius_covers_all_actions(
        self,
        configuration: DefenderNeighborhoodConfig,
    ) -> bool:
        expected = len(self.action_ids) - 1
        return all(
            len(self.neighbors(action_id, configuration)) == expected
            for action_id in self.action_ids
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "ordered_action_grid_indices": [
                {"action_id": action_id, "grid_index": list(coordinate)}
                for action_id, coordinate in self.action_grid_indices
            ],
        }


@dataclass(frozen=True)
class LocalDefenderEvaluation:
    """Minimal exact-evaluation view consumed by the pure local verifier."""

    action_id: int
    status: str
    defender_value: float | None
    attacker_objective: float | None
    selected_attacker_response_id: int | None
    exact_attacker_best_response_verified: bool
    strong_tie_break_verified: bool
    diagnostic: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.action_id, Integral)
            or isinstance(self.action_id, bool)
            or int(self.action_id) < 0
        ):
            raise ValueError("action_id must be a nonnegative integer")
        object.__setattr__(self, "action_id", int(self.action_id))
        allowed = {
            "feasible", "model_infeasible", "numerical_failure", "timeout",
            "memory_limit",
        }
        if self.status not in allowed:
            raise ValueError(f"unsupported Defender evaluation status: {self.status!r}")
        if self.status == "feasible":
            if self.defender_value is None or self.attacker_objective is None:
                raise ValueError("a feasible evaluation requires finite J_D and J_A")
            if not np.isfinite(self.defender_value) or not np.isfinite(self.attacker_objective):
                raise ValueError("a feasible evaluation requires finite J_D and J_A")
            if self.selected_attacker_response_id is None:
                raise ValueError("a feasible evaluation requires a selected exact response")
        elif any(
            value is not None
            for value in (
                self.defender_value,
                self.attacker_objective,
                self.selected_attacker_response_id,
            )
        ):
            raise ValueError("a non-feasible evaluation cannot contain fabricated payoffs")
        object.__setattr__(
            self, "exact_attacker_best_response_verified",
            bool(self.exact_attacker_best_response_verified),
        )
        object.__setattr__(
            self, "strong_tie_break_verified", bool(self.strong_tie_break_verified),
        )

    @property
    def feasible(self) -> bool:
        return self.status == "feasible"


@dataclass(frozen=True)
class FollowerPayoffRecord:
    """One feasible response record used only to audit Strong tie-breaking."""

    response_id: int
    attacker_objective: float
    defender_payoff: float

    def __post_init__(self) -> None:
        if not isinstance(self.response_id, Integral) or isinstance(self.response_id, bool):
            raise ValueError("response_id must be an integer")
        if not np.isfinite(self.attacker_objective) or not np.isfinite(self.defender_payoff):
            raise ValueError("follower payoff records must be finite")
        object.__setattr__(self, "response_id", int(self.response_id))


@dataclass(frozen=True)
class StrongFollowerTieAudit:
    selected_response_id: int
    attacker_best_response_ids: tuple[int, ...]
    defender_best_ids_within_attacker_tie: tuple[int, ...]
    expected_selected_response_id: int
    passed: bool


def audit_strong_follower_selection(
    responses: Sequence[FollowerPayoffRecord],
    selected_response_id: int,
    *,
    objective_tolerance: float = 1.0e-12,
) -> StrongFollowerTieAudit:
    """Audit, but do not replace, the existing Strong follower selection."""
    records = tuple(responses)
    if not records:
        raise ValueError("at least one feasible follower response is required")
    if len({record.response_id for record in records}) != len(records):
        raise ValueError("follower response IDs must be unique")
    tolerance = _nonnegative_tolerance(objective_tolerance)
    minimum = min(record.attacker_objective for record in records)
    attacker_best = tuple(
        record for record in records
        if record.attacker_objective <= minimum + tolerance
    )
    maximum_defender = max(record.defender_payoff for record in attacker_best)
    defender_best = tuple(
        record for record in attacker_best
        if record.defender_payoff == maximum_defender
    )
    expected = min(record.response_id for record in defender_best)
    selected = int(selected_response_id)
    return StrongFollowerTieAudit(
        selected_response_id=selected,
        attacker_best_response_ids=tuple(sorted(record.response_id for record in attacker_best)),
        defender_best_ids_within_attacker_tie=tuple(
            sorted(record.response_id for record in defender_best)
        ),
        expected_selected_response_id=expected,
        passed=selected == expected,
    )


@dataclass(frozen=True)
class LocalSSEVerification:
    candidate_action_id: int
    candidate_defender_value: float | None
    neighbor_action_ids: tuple[int, ...]
    feasible_neighbor_ids: tuple[int, ...]
    infeasible_neighbor_diagnostics: tuple[tuple[int, str | None], ...]
    unknown_neighbor_diagnostics: tuple[tuple[int, str], ...]
    improving_neighbor_ids: tuple[int, ...]
    equal_payoff_neighbor_ids: tuple[int, ...]
    best_feasible_neighbor_value: float | None
    local_optimality_margin: float | None
    isolated_feasible_local_solution: bool
    exact_attacker_responses_verified: bool
    strong_tie_breaking_verified: bool
    local_sse_verified: bool
    global_condition_equivalent: bool
    comparison_scope: str
    neighborhood_metadata: dict[str, Any]
    # Whether exactness of the Attacker responses was *required* to certify.  An
    # approximate Attacker solver reports its responses honestly as inexact; it
    # simply is not held to a standard it cannot meet.  The two flags above stay
    # truthful either way, so an approximate run is never mistaken for an exact one.
    exact_attacker_verification_required: bool = True
    # Local optimality with respect to the evaluations actually supplied, which is
    # the strongest statement available when those evaluations are approximate.
    local_optimum_under_supplied_evaluations: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_local_sse(
    candidate_action_id: int,
    evaluations: Mapping[int, LocalDefenderEvaluation],
    topology: DefenderGridTopology,
    configuration: DefenderNeighborhoodConfig = DefenderNeighborhoodConfig(),
    *,
    leader_payoff_tolerance: float = 1.0e-12,
    require_exact_attacker_verification: bool = True,
) -> LocalSSEVerification:
    """Purely verify a candidate against all required configured neighbors."""
    tolerance = _nonnegative_tolerance(leader_payoff_tolerance)
    require_exact = bool(require_exact_attacker_verification)
    normalized = dict(evaluations)
    if any(action_id != evaluation.action_id for action_id, evaluation in normalized.items()):
        raise ValueError("evaluation mapping keys must equal evaluation action IDs")
    if candidate_action_id not in topology.action_ids:
        raise KeyError("candidate action is absent from the Defender grid topology")
    candidate = normalized.get(candidate_action_id)
    neighbor_ids = topology.neighbors(candidate_action_id, configuration)
    infeasible: list[tuple[int, str | None]] = []
    unknown: list[tuple[int, str]] = []
    feasible_neighbors: list[LocalDefenderEvaluation] = []
    for neighbor_id in neighbor_ids:
        evaluation = normalized.get(neighbor_id)
        if evaluation is None:
            unknown.append((neighbor_id, "evaluation_missing"))
        elif evaluation.status == "model_infeasible":
            if evaluation.exact_attacker_best_response_verified or not require_exact:
                infeasible.append((neighbor_id, evaluation.diagnostic))
            else:
                unknown.append((neighbor_id, "infeasibility_not_exactly_verified"))
        elif evaluation.feasible:
            feasible_neighbors.append(evaluation)
        else:
            unknown.append((neighbor_id, evaluation.status))
    # Strict readiness always reports the honest exactness picture; the relaxed
    # one is what an approximate search is allowed to climb on.
    strict_candidate_ready = bool(
        candidate is not None
        and candidate.feasible
        and candidate.exact_attacker_best_response_verified
        and candidate.strong_tie_break_verified
    )
    candidate_ready = strict_candidate_ready if require_exact else bool(
        candidate is not None and candidate.feasible
    )
    all_exact = bool(
        strict_candidate_ready
        and all(item.exact_attacker_best_response_verified for item in feasible_neighbors)
        and not unknown
    )
    all_strong = bool(
        strict_candidate_ready
        and all(item.strong_tie_break_verified for item in feasible_neighbors)
        and not unknown
    )
    candidate_value = None if candidate is None else candidate.defender_value
    if candidate_value is not None and feasible_neighbors:
        best_neighbor = max(float(item.defender_value) for item in feasible_neighbors)
        margin = float(candidate_value) - best_neighbor
        improving = tuple(
            item.action_id for item in feasible_neighbors
            if float(item.defender_value) > float(candidate_value) + tolerance
        )
        equal = tuple(
            item.action_id for item in feasible_neighbors
            if abs(float(item.defender_value) - float(candidate_value)) <= tolerance
        )
    else:
        best_neighbor = None
        margin = None
        improving = ()
        equal = ()
    isolated = bool(candidate_ready and not feasible_neighbors and not unknown)
    local_verified = bool(
        strict_candidate_ready
        and all_exact
        and all_strong
        and not improving
        and not unknown
    )
    local_optimum_supplied = bool(candidate_ready and not improving and not unknown)
    return LocalSSEVerification(
        candidate_action_id=int(candidate_action_id),
        candidate_defender_value=candidate_value,
        neighbor_action_ids=neighbor_ids,
        feasible_neighbor_ids=tuple(item.action_id for item in feasible_neighbors),
        infeasible_neighbor_diagnostics=tuple(infeasible),
        unknown_neighbor_diagnostics=tuple(unknown),
        improving_neighbor_ids=improving,
        equal_payoff_neighbor_ids=equal,
        best_feasible_neighbor_value=best_neighbor,
        local_optimality_margin=margin,
        isolated_feasible_local_solution=isolated,
        exact_attacker_responses_verified=all_exact,
        strong_tie_breaking_verified=all_strong,
        local_sse_verified=local_verified,
        global_condition_equivalent=topology.radius_covers_all_actions(configuration),
        comparison_scope=LOCAL_SSE_SCOPE,
        neighborhood_metadata={
            "configuration": configuration.as_metadata(),
            "topology": topology.as_metadata(),
        },
        exact_attacker_verification_required=require_exact,
        local_optimum_under_supplied_evaluations=local_optimum_supplied,
    )


def global_sse_action_ids(
    evaluations: Mapping[int, LocalDefenderEvaluation],
    *,
    leader_payoff_tolerance: float = 1.0e-12,
) -> tuple[int, ...]:
    """Return all exact feasible global leader maximizers for tiny-game audits."""
    tolerance = _nonnegative_tolerance(leader_payoff_tolerance)
    feasible = tuple(evaluation for evaluation in evaluations.values() if evaluation.feasible)
    if not feasible:
        return ()
    if any(
        not evaluation.exact_attacker_best_response_verified
        or not evaluation.strong_tie_break_verified
        for evaluation in feasible
    ):
        raise ValueError("global audit requires exact Strong follower evaluations")
    maximum = max(float(evaluation.defender_value) for evaluation in feasible)
    return tuple(sorted(
        evaluation.action_id for evaluation in feasible
        if float(evaluation.defender_value) >= maximum - tolerance
    ))


__all__ = [
    "DEFAULT_R_NEIGHBOR", "DISCRETIZED_EXISTENCE_CONDITION", "GLOBAL_SSE_SCOPE",
    "GRID_INDEX_METRIC", "LOCAL_SSE_CONDITION", "LOCAL_SSE_SCHEMA_VERSION",
    "LOCAL_SSE_SCOPE", "DefenderGridTopology", "DefenderNeighborhoodConfig",
    "FollowerPayoffRecord", "LocalDefenderEvaluation", "LocalSSEVerification",
    "StrongFollowerTieAudit", "audit_strong_follower_selection",
    "global_sse_action_ids", "verify_local_sse",
]

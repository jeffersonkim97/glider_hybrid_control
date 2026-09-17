"""Stage 14.2A contract, existence, and local/global distinction diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from local_sse_contract import (
    DEFAULT_R_NEIGHBOR,
    DISCRETIZED_EXISTENCE_CONDITION,
    GLOBAL_SSE_SCOPE,
    GRID_INDEX_METRIC,
    LOCAL_SSE_CONDITION,
    LOCAL_SSE_SCHEMA_VERSION,
    LOCAL_SSE_SCOPE,
    DefenderGridTopology,
    DefenderNeighborhoodConfig,
    FollowerPayoffRecord,
    LocalDefenderEvaluation,
    audit_strong_follower_selection,
    global_sse_action_ids,
    verify_local_sse,
)
from stage11_notebook_support import save_figure_png, write_json
from stage14_benchmark_contract import environment_manifest


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_2a_local_sse_contract"
STAGE12_SUMMARY = ROOT / "figure" / "stage_12_finite_stackelberg" / "stackelberg_summary.json"
FROZEN_STAGE14 = ROOT / "figure" / "stage_14_0_benchmark_contract" / "frozen_reference.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation(
    action_id: int,
    defender_value: float,
    *,
    attacker_objective: float | None = None,
    response_id: int | None = None,
) -> LocalDefenderEvaluation:
    return LocalDefenderEvaluation(
        action_id=action_id,
        status="feasible",
        defender_value=defender_value,
        attacker_objective=(10.0 - defender_value if attacker_objective is None else attacker_objective),
        selected_attacker_response_id=(100 + action_id if response_id is None else response_id),
        exact_attacker_best_response_verified=True,
        strong_tie_break_verified=True,
    )


def _tiny_game_report() -> dict[str, Any]:
    values = (1.0, 4.0, 3.0, 7.0, 6.0)
    evaluations = {
        action_id: _evaluation(action_id, value)
        for action_id, value in enumerate(values)
    }
    topology = DefenderGridTopology.ordered_line(tuple(evaluations))
    radius_one = DefenderNeighborhoodConfig(r_neighbor=1)
    verifications = {
        action_id: verify_local_sse(action_id, evaluations, topology, radius_one)
        for action_id in evaluations
    }
    local_ids = tuple(
        action_id for action_id, result in verifications.items()
        if result.local_sse_verified
    )
    global_ids = global_sse_action_ids(evaluations)
    full_radius = DefenderNeighborhoodConfig(r_neighbor=len(values) - 1)
    full_radius_local_ids = tuple(
        action_id for action_id in evaluations
        if verify_local_sse(
            action_id, evaluations, topology, full_radius,
        ).local_sse_verified
    )
    infeasible_evaluations = dict(evaluations)
    infeasible_evaluations[0] = LocalDefenderEvaluation(
        action_id=0,
        status="model_infeasible",
        defender_value=None,
        attacker_objective=None,
        selected_attacker_response_id=None,
        exact_attacker_best_response_verified=True,
        strong_tie_break_verified=False,
        diagnostic="no feasible exact finite Attacker response",
    )
    infeasible_neighbor = verify_local_sse(
        1, infeasible_evaluations, topology, radius_one,
    )
    strong_audit = audit_strong_follower_selection(
        (
            FollowerPayoffRecord(10, 1.0, 0.2),
            FollowerPayoffRecord(11, 1.0, 0.7),
            FollowerPayoffRecord(12, 2.0, 0.9),
        ),
        selected_response_id=11,
    )
    return {
        "payoff_line": [
            {"action_id": action_id, "V_D": value}
            for action_id, value in enumerate(values)
        ],
        "r_neighbor": 1,
        "local_sse_action_ids": list(local_ids),
        "global_sse_action_ids": list(global_ids),
        "local_but_not_global_action_ids": sorted(set(local_ids) - set(global_ids)),
        "verification_by_action": {
            str(action_id): result.as_dict()
            for action_id, result in verifications.items()
        },
        "full_radius": full_radius.r_neighbor,
        "full_radius_local_action_ids": list(full_radius_local_ids),
        "full_radius_equals_global": full_radius_local_ids == global_ids,
        "infeasible_neighbor_diagnostic": infeasible_neighbor.as_dict(),
        "strong_follower_tie_audit": {
            "selected_response_id": strong_audit.selected_response_id,
            "attacker_best_response_ids": list(strong_audit.attacker_best_response_ids),
            "defender_best_ids_within_attacker_tie": list(
                strong_audit.defender_best_ids_within_attacker_tie
            ),
            "expected_selected_response_id": strong_audit.expected_selected_response_id,
            "passed": strong_audit.passed,
        },
    }


def _current_discretized_report() -> dict[str, Any]:
    """Reverify the stored exact centered-cube global run under local semantics."""
    source = _read_json(STAGE12_SUMMARY)
    evaluations = {
        int(row["defender_action_id"]): _evaluation(
            int(row["defender_action_id"]),
            float(row["detection_probability"]),
            attacker_objective=float(row["attacker_objective"]),
            response_id=int(row["attacker_candidate_id"]),
        )
        for row in source["defender_actions"] if row["feasible"]
    }
    topology = DefenderGridTopology.ordered_line(tuple(
        int(row["defender_action_id"]) for row in source["defender_actions"]
    ))
    configuration = DefenderNeighborhoodConfig()
    global_id = int(source["selected"]["defender_action_id"])
    verification = verify_local_sse(
        global_id, evaluations, topology, configuration,
    )
    component_nonempty = bool(evaluations)
    finite_values = all(
        evaluation.defender_value is not None
        and evaluation.attacker_objective is not None
        for evaluation in evaluations.values()
    )
    assumptions = {
        "finite_nonempty_feasible_component": component_nonempty,
        "each_recorded_action_has_exact_finite_attacker_response": bool(
            source["literal_exhaustive_check"]["all_defender_actions_evaluated"]
        ),
        "finite_J_A_and_J_D": finite_values,
        "strong_tie_breaking_well_defined": "maximize Defender PoD" in source[
            "payoff_convention"
        ]["follower_tie"],
        "independent_validation_passed": bool(
            source["validation"]["report"]["passed"]
        ),
    }
    return {
        "source_artifact": str(STAGE12_SUMMARY.relative_to(ROOT)),
        "terrain_category": source["configuration"]["terrain_category"],
        "configured_neighborhood": configuration.as_metadata(),
        "topology": topology.as_metadata(),
        "frozen_global_action_id": global_id,
        "global_action_local_verification": verification.as_dict(),
        "existence_assumptions": assumptions,
        "existence_condition_satisfied": all(assumptions.values()),
        "interpretation": DISCRETIZED_EXISTENCE_CONDITION,
        "not_claimed": [
            "global optimality of an arbitrary local SSE",
            "continuous-game local or global optimality",
        ],
    }


def _payoff_figure(tiny: dict[str, Any]) -> go.Figure:
    x_values = [item["action_id"] for item in tiny["payoff_line"]]
    y_values = [item["V_D"] for item in tiny["payoff_line"]]
    local_ids = set(tiny["local_sse_action_ids"])
    global_ids = set(tiny["global_sse_action_ids"])
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=x_values, y=y_values, mode="lines+markers+text",
        text=[f"D{action_id}: {value:g}" for action_id, value in zip(x_values, y_values)],
        textposition="top center", name="V_D on ordered Defender grid",
        line={"color": "#7F7F7F"}, marker={"size": 9},
    ))
    figure.add_trace(go.Scatter(
        x=sorted(local_ids), y=[y_values[action_id] for action_id in sorted(local_ids)],
        mode="markers", name="r_neighbor=1 local SSE",
        marker={"size": 16, "color": "#70AD47", "symbol": "circle-open"},
    ))
    figure.add_trace(go.Scatter(
        x=sorted(global_ids), y=[y_values[action_id] for action_id in sorted(global_ids)],
        mode="markers", name="global SSE",
        marker={"size": 16, "color": "#C00000", "symbol": "star"},
    ))
    figure.update_layout(
        title="Stage 14.2A: local versus global leader condition",
        xaxis={"title": "Ordered Defender grid action", "dtick": 1},
        yaxis={"title": "Induced Defender value V_D"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"t": 110},
    )
    return figure


def run_stage14_2a_diagnostics(
    output_directory: Path = OUTPUT,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    configuration = DefenderNeighborhoodConfig()
    one_dimensional = DefenderGridTopology.ordered_line((0, 1, 2, 3, 4))
    two_dimensional = DefenderGridTopology(tuple(
        (row * 5 + column, (row, column))
        for row in range(5) for column in range(5)
    ))
    neighborhood_metadata = {
        "default_configuration": configuration.as_metadata(),
        "one_dimensional_examples": {
            "center_action_id": 2,
            "r1_neighbors": list(one_dimensional.neighbors(2, configuration)),
            "r2_neighbors": list(one_dimensional.neighbors(
                2, DefenderNeighborhoodConfig(2)
            )),
        },
        "two_dimensional_examples": {
            "center_action_id": 12,
            "r1_neighbor_count": len(two_dimensional.neighbors(
                12, DefenderNeighborhoodConfig(1)
            )),
            "r2_neighbor_count": len(two_dimensional.neighbors(
                12, DefenderNeighborhoodConfig(2)
            )),
        },
    }
    tiny = _tiny_game_report()
    current = _current_discretized_report()
    frozen = _read_json(FROZEN_STAGE14)
    current_environment = environment_manifest(ROOT.parent)
    solver_unchanged = bool(
        current_environment["software_revision"]["solver_source"]["aggregate_sha256"]
        == frozen["solver_source_fingerprint"]
    )
    contract = {
        "schema_version": LOCAL_SSE_SCHEMA_VERSION,
        "local_scope": LOCAL_SSE_SCOPE,
        "global_scope": GLOBAL_SSE_SCOPE,
        "local_condition": LOCAL_SSE_CONDITION,
        "existence_condition": DISCRETIZED_EXISTENCE_CONDITION,
        "attacker_response_requirement": "existing exact finite Attacker best response",
        "follower_tie_requirement": (
            "among Attacker-objective co-optima, maximize Defender payoff; "
            "remaining exact tie uses deterministic lowest response ID"
        ),
        "default_r_neighbor": DEFAULT_R_NEIGHBOR,
        "neighbor_metric": GRID_INDEX_METRIC,
        "global_equivalence": (
            "when the configured radius covers every other feasible Defender "
            "action, the local condition equals the global finite-SSE condition"
        ),
    }
    gates = {
        "local_not_mislabeled_global": "not a global" in LOCAL_SSE_SCOPE,
        "neighborhood_explicit_deterministic_exported": (
            neighborhood_metadata["one_dimensional_examples"]["r1_neighbors"] == [1, 3]
            and neighborhood_metadata["two_dimensional_examples"]["r1_neighbor_count"] == 8
            and neighborhood_metadata["two_dimensional_examples"]["r2_neighbor_count"] == 24
        ),
        "attacker_response_remains_exact": current["existence_assumptions"][
            "each_recorded_action_has_exact_finite_attacker_response"
        ],
        "strong_tie_breaking_preserved": tiny["strong_follower_tie_audit"]["passed"],
        "tiny_games_distinguish_local_global": (
            tiny["local_sse_action_ids"] == [1, 3]
            and tiny["global_sse_action_ids"] == [3]
            and tiny["full_radius_equals_global"]
        ),
        "global_solver_source_unchanged": solver_unchanged,
    }
    gate_passed = all(gates.values()) and bool(
        current["existence_condition_satisfied"]
        and current["global_action_local_verification"]["local_sse_verified"]
    )
    if not gate_passed:
        raise RuntimeError("Stage-14.2A local-SSE contract gate failed")
    write_json(output_directory / "local_sse_mathematical_contract.json", contract)
    write_json(output_directory / "neighborhood_metadata.json", neighborhood_metadata)
    write_json(output_directory / "tiny_game_verification.json", tiny)
    write_json(output_directory / "current_discretized_existence_report.json", current)
    save_figure_png(
        _payoff_figure(tiny), output_directory, "local_vs_global_payoff_diagnostic.png"
    )
    summary = {
        "stage": "14.2A",
        "schema_version": LOCAL_SSE_SCHEMA_VERSION,
        "gate_passed": True,
        "default_r_neighbor": configuration.r_neighbor,
        "neighbor_metric": configuration.metric,
        "tiny_local_sse_action_ids": tiny["local_sse_action_ids"],
        "tiny_global_sse_action_ids": tiny["global_sse_action_ids"],
        "full_radius_equals_global": tiny["full_radius_equals_global"],
        "current_global_action_id": current["frozen_global_action_id"],
        "current_global_action_passes_local_verifier": current[
            "global_action_local_verification"
        ]["local_sse_verified"],
        "current_discretized_existence_condition_satisfied": current[
            "existence_condition_satisfied"
        ],
        "solver_source_fingerprint": frozen["solver_source_fingerprint"],
        "solver_source_unchanged": solver_unchanged,
        "gate_checks": gates,
        "local_search_implemented": False,
        "global_solver_modified": False,
    }
    write_json(output_directory / "stage14_2a_summary.json", summary)
    print("Stage 14.2A local-SSE mathematical contract: PASS")
    print(
        f"r_neighbor={configuration.r_neighbor}; tiny local={tiny['local_sse_action_ids']}; "
        f"global={tiny['global_sse_action_ids']}; full-radius equivalence=PASS"
    )
    return summary


if __name__ == "__main__":
    run_stage14_2a_diagnostics()

"""Fresh-process Stage-14.6 neighborhood-radius instrumentation worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage11_notebook_support import write_json
from stage14_2c_worker import run_worker_case
from stage14_6_contract import (
    DEFENDER_ACTION_COUNT,
    DEFENDER_X_MAP,
    FULL_RADIUS,
    STAGE14_6_SCHEMA_VERSION,
)


def _expected_neighbors(action_id: int, radius: int) -> list[int]:
    return [
        other for other in range(DEFENDER_ACTION_COUNT)
        if 0 < abs(other - action_id) <= radius
    ]


def _local_audit(payload: dict[str, Any], radius: int) -> dict[str, Any]:
    search = payload["local_search"]
    seen = {int(search["initial_defender_action_id"])}
    per_iteration = []
    raw_requests = 1
    cache_hits = 0
    neighborhood_rule_passed = True
    for iteration in search["iterations"]:
        current = int(iteration["current_action_id"])
        actual = [int(value) for value in iteration["neighbor_action_ids"]]
        expected = _expected_neighbors(current, radius)
        neighborhood_rule_passed &= actual == expected
        new_ids = [action_id for action_id in actual if action_id not in seen]
        hit_ids = [action_id for action_id in actual if action_id in seen]
        raw_requests += len(actual)
        cache_hits += len(hit_ids)
        seen.update(actual)
        chosen = iteration["chosen_next_action_id"]
        if chosen is not None:
            raw_requests += 1
            cache_hits += 1
        per_iteration.append({
            "iteration": int(iteration["iteration"]),
            "current_action_id": current,
            "current_sensor_x_map": DEFENDER_X_MAP[current],
            "expected_neighbor_action_ids": expected,
            "actual_neighbor_action_ids": actual,
            "neighborhood_size": len(actual),
            "new_exact_evaluation_action_ids": new_ids,
            "cached_neighbor_action_ids": hit_ids,
            "chosen_next_action_id": chosen,
            "payoff_improvement": float(iteration["payoff_improvement"]),
        })
    unique = int(search["unique_defender_evaluations"])
    reported_hits = int(search["cached_evaluation_reuses"])
    records = list(search["exact_evaluation_records"])
    sum_br = sum(float(record["attacker_br_runtime_s"]) for record in records)
    totals = payload["timing"]["totals"]
    shared = float(totals["T_graph_s"])
    return {
        "r_neighbor": radius,
        "r_full": FULL_RADIUS,
        "fixed_defender_action_count": DEFENDER_ACTION_COUNT,
        "fixed_defender_x_map": list(DEFENDER_X_MAP),
        "neighborhood_rule": "0 < abs(j - i) <= r_neighbor",
        "raw_evaluation_requests": raw_requests,
        "cache_hits": cache_hits,
        "unique_exact_attacker_br_evaluations": unique,
        "evaluated_action_ids": list(search["evaluated_defender_actions"]),
        "visited_action_ids": list(search["visited_defender_actions"]),
        "local_search_iterations": int(search["local_search_iterations"]),
        "per_iteration": per_iteration,
        "per_action_exact_evaluations": records,
        "summed_per_action_attacker_br_runtime_s": sum_br,
        "shared_graph_runtime_s": shared,
        "local_search_overhead_s": max(
            0.0, float(totals["T_SSE_s"]) - shared - sum_br
        ),
        "neighborhood_rule_passed": neighborhood_rule_passed,
        "request_accounting_passed": raw_requests == unique + cache_hits,
        "reported_cache_accounting_passed": reported_hits == cache_hits,
        "unique_record_accounting_passed": unique == len(records) == len(seen),
        "full_grid_covered": set(seen) == set(range(DEFENDER_ACTION_COUNT)),
    }


def run_worker(configuration: dict[str, Any]) -> dict[str, Any]:
    payload = run_worker_case(configuration)
    payload["stage14_6_schema_version"] = STAGE14_6_SCHEMA_VERSION
    radius = int(configuration["parameters"]["r_neighbor"])
    payload["stage14_6_fixed_grid"] = {
        "defender_action_count": DEFENDER_ACTION_COUNT,
        "defender_x_map": list(DEFENDER_X_MAP),
        "initial_defender_action_id": int(
            configuration["parameters"]["initial_defender_action_id"]
        ),
        "r_neighbor": radius,
    }
    if payload.get("status") != "completed":
        return payload
    if payload["algorithm_variant"] == "local_sse":
        audit = _local_audit(payload, radius)
        payload["local_search"]["radius_sweep_audit"] = audit
        payload.setdefault("exactness", {}).update({
            "radius_neighborhood_audit_verified": audit["neighborhood_rule_passed"],
            "evaluation_accounting_reconciled": bool(
                audit["request_accounting_passed"]
                and audit["reported_cache_accounting_passed"]
                and audit["unique_record_accounting_passed"]
            ),
        })
    else:
        metadata = payload.setdefault("oracle_metadata", {})
        records = metadata.get("per_action_evaluation_records", [])
        metadata["stage14_6_full_radius_reference"] = {
            "r_neighbor": radius,
            "r_full": FULL_RADIUS,
            "evaluated_action_ids": [record["action_id"] for record in records],
            "complete_action_grid_covered": (
                {record["action_id"] for record in records}
                == set(range(DEFENDER_ACTION_COUNT))
            ),
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.configuration.read_text(encoding="utf-8"))
    payload = run_worker(configuration)
    write_json(arguments.output, payload)
    print(f"{configuration['repetition_id']}: {payload['status']}")


if __name__ == "__main__":
    main()

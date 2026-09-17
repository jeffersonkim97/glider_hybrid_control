"""Stage-14.1 timing, memory, and instrumentation regression tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

from attacker_best_response import AttackerBestResponseTiming
from stage14_benchmark_contract import solver_source_manifest
from stage14_profiling import (
    ProcessTreeMemoryMeasurement,
    decompose_exact_sse_timing,
    run_profiled_subprocess,
)
from stackelberg_solver import FiniteStackelbergRun, StackelbergTiming


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_1_runtime_profiling"
STAGE14_0 = ROOT / "figure" / "stage_14_0_benchmark_contract" / "frozen_reference.json"


def _fake_run(*, bellman_backend: str = "dense") -> FiniteStackelbergRun:
    rows = []
    for action_id, scale in enumerate((1.0, 1.5)):
        timing = AttackerBestResponseTiming(
            los_s=0.10 * scale,
            candidate_generation_s=0.05 * scale,
            energy_filter_s=0.05 * scale,
            virtual_connection_s=0.05 * scale,
            graph_build_s=0.10 * scale,
            hazard_precompute_s=0.20 * scale,
            bellman_solve_s=(
                0.30 * scale
                if bellman_backend == "goal_backward_sparse"
                else 0.10 * scale
            ),
            attacker_bellman_s=0.50 * scale,
            total_attacker_br_s=1.00 * scale,
        )
        rows.append(SimpleNamespace(
            candidate=SimpleNamespace(action_id=action_id),
            feasible=True,
            attacker_run=SimpleNamespace(metrics=SimpleNamespace(
                timing=timing,
                bellman_backend=bellman_backend,
            )),
        ))
    run = object.__new__(FiniteStackelbergRun)
    object.__setattr__(run, "evaluations", tuple(rows))
    object.__setattr__(run, "timing", StackelbergTiming(
        per_action_s=(1.01, 1.51),
        total_s=2.75,
        shared_graph_build_s=0.20,
    ))
    return run


class Stage141ProfilingUnitTests(unittest.TestCase):
    def test_nonoverlapping_timing_reconciles_exactly(self) -> None:
        result = decompose_exact_sse_timing(_fake_run(), validation_time_s=0.25)
        totals = result["totals"]
        self.assertAlmostEqual(totals["T_SSE_s"], 2.75)
        self.assertAlmostEqual(totals["T_graph_s"], 0.45)
        self.assertAlmostEqual(totals["T_switch_s"], 0.375)
        self.assertAlmostEqual(totals["T_defender_overhead_s"], 0.05)
        self.assertAlmostEqual(result["T_SSE_accounted_s"], 2.75)
        self.assertLessEqual(abs(result["T_SSE_reconciliation_error_s"]), 1.0e-12)
        self.assertFalse(result["validation_included_in_T_SSE"])
        self.assertEqual(
            result["sparse_bellman_timer_semantics"], "hazard-exclusive-v1"
        )

    def test_sparse_inclusive_solve_timer_does_not_double_count_hazard(self) -> None:
        result = decompose_exact_sse_timing(
            _fake_run(bellman_backend="goal_backward_sparse"),
            validation_time_s=0.25,
        )
        totals = result["totals"]
        self.assertAlmostEqual(totals["T_hazard_s"], 0.50)
        self.assertAlmostEqual(totals["T_Bellman_s"], 0.25)
        self.assertAlmostEqual(result["T_SSE_accounted_s"], 2.75)
        self.assertLessEqual(abs(result["T_SSE_reconciliation_error_s"]), 1.0e-12)

    def test_memory_contract_requires_peak_minus_start_delta(self) -> None:
        valid = ProcessTreeMemoryMeasurement(10, 25, 15, 0.01)
        self.assertTrue(valid.native_allocations_included)
        self.assertTrue(valid.child_processes_included)
        with self.assertRaises(ValueError):
            ProcessTreeMemoryMeasurement(10, 25, 14, 0.01)

    def test_fresh_subprocess_sampler_observes_native_numpy_allocation(self) -> None:
        command = (
            sys.executable,
            "-c",
            "import numpy as np,time; x=np.ones(2000000); time.sleep(0.08); print(x.size)",
        )
        result = run_profiled_subprocess(command, cwd=ROOT, sampling_interval_s=0.005)
        self.assertEqual(result.returncode, 0)
        self.assertIn("2000000", result.stdout)
        self.assertGreater(result.memory.start_rss_bytes, 0)
        self.assertGreaterEqual(result.memory.peak_rss_bytes, result.memory.start_rss_bytes)
        self.assertGreater(result.memory.delta_peak_rss_bytes, 0)


class Stage141ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = json.loads(
            (OUTPUT / "canonical_profile.json").read_text(encoding="utf-8")
        )
        cls.infeasible = json.loads(
            (OUTPUT / "infeasible_profile.json").read_text(encoding="utf-8")
        )
        cls.summary = json.loads(
            (OUTPUT / "stage14_1_summary.json").read_text(encoding="utf-8")
        )
        cls.frozen = json.loads(STAGE14_0.read_text(encoding="utf-8"))

    def test_required_machine_readable_and_visual_artifacts_exist(self) -> None:
        expected = (
            "canonical_worker_raw.json", "infeasible_worker_raw.json",
            "canonical_profile.json", "infeasible_profile.json",
            "canonical_worker_monitor.json", "infeasible_worker_monitor.json",
            "profiling_contract.json", "canonical_timing.csv",
            "canonical_runtime_decomposition.png", "canonical_memory_summary.png",
            "stage14_1_summary.json",
        )
        for filename in expected:
            path = OUTPUT / filename
            self.assertTrue(path.is_file(), filename)
            self.assertGreater(path.stat().st_size, 50, filename)

    def test_every_required_timing_is_finite_nonnegative_and_reconciled(self) -> None:
        timing = self.profile["timing"]
        totals = timing["totals"]
        for name in (
            "T_SSE_s", "T_graph_s", "T_LOS_s", "T_hazard_s", "T_switch_s",
            "T_Bellman_s", "T_defender_overhead_s", "T_attacker_unclassified_s",
            "T_validation_s",
        ):
            self.assertIn(name, totals)
            self.assertGreaterEqual(totals[name], 0.0)
        self.assertEqual(len(timing["per_defender_action"]), 3)
        self.assertLessEqual(abs(timing["T_SSE_reconciliation_error_s"]), 1.0e-9)
        self.assertAlmostEqual(
            totals["T_switch_s"],
            totals["T_switch_candidate_generation_s"]
            + totals["T_switch_energy_filter_s"]
            + totals["T_switch_virtual_connection_s"],
            places=9,
        )

    def test_primary_memory_is_process_tree_rss_including_native_allocations(self) -> None:
        memory = self.profile["process_tree_memory"]
        self.assertGreater(memory["start_rss_bytes"], 0)
        self.assertGreaterEqual(memory["peak_rss_bytes"], memory["start_rss_bytes"])
        self.assertEqual(
            memory["delta_peak_rss_bytes"],
            memory["peak_rss_bytes"] - memory["start_rss_bytes"],
        )
        self.assertTrue(memory["native_allocations_included"])
        self.assertTrue(memory["child_processes_included"])
        self.assertEqual(memory["sampling_interval_s"], 0.01)

    def test_profiled_solution_is_identical_to_unprofiled_frozen_reference(self) -> None:
        regression = self.profile["profiled_vs_unprofiled_regression"]
        self.assertTrue(regression["passed"])
        self.assertTrue(all(regression["checks"].values()))
        self.assertEqual(
            self.profile["solution_identity"], self.frozen["solution_identity"],
        )
        self.assertEqual(
            self.profile["state_and_game_size"], self.frozen["state_and_game_size"],
        )
        self.assertTrue(self.profile["independent_replay"]["passed"])

    def test_infeasible_case_is_explicit_and_not_silently_skipped(self) -> None:
        result = self.infeasible
        self.assertEqual(result["status"], "model_infeasible")
        self.assertEqual(result["terrain_category"], "centered_cube")
        self.assertEqual(
            result["failure_message"],
            "no Defender action has a feasible Attacker response",
        )
        self.assertGreaterEqual(result["T_SSE_attempt_s"], 0.0)
        self.assertFalse(result["component_timing_available"])
        self.assertTrue(result["component_timing_unavailable_reason"])
        self.assertTrue(result["configuration"]["vehicle_and_objectives_unchanged"])

    def test_csv_contains_canonical_counts_objectives_and_validation(self) -> None:
        with (OUTPUT / "canonical_timing.csv").open(newline="", encoding="utf-8") as handle:
            rows = tuple(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for name in (
            "T_SSE_s", "T_graph_s", "T_LOS_s", "T_hazard_s", "T_switch_s",
            "T_Bellman_s", "T_defender_overhead_s", "peak_rss_bytes",
            "N_S_cart", "N_E", "N_C_raw", "N_D", "attacker_objective",
            "defender_objective_pod", "validation_passed",
        ):
            self.assertIn(name, row)
        self.assertEqual(int(row["N_S_cart"]), 3_243_240)
        self.assertEqual(int(row["N_E"]), 29_035_732)
        self.assertEqual(row["validation_passed"], "True")

    def test_solver_source_and_stage_gate_remain_frozen(self) -> None:
        fingerprint = solver_source_manifest(ROOT)["aggregate_sha256"]
        self.assertEqual(fingerprint, self.frozen["solver_source_fingerprint"])
        self.assertEqual(fingerprint, self.profile["solver_source_fingerprint"])
        self.assertTrue(self.summary["gate_passed"])
        self.assertTrue(self.summary["profiled_vs_unprofiled_match"])
        self.assertFalse(self.summary["scaling_claim_made"])
        self.assertFalse(self.summary["sweep_runner_added"])


if __name__ == "__main__":
    unittest.main()

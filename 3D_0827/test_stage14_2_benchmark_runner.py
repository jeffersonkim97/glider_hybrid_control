"""Unit and exit-integration gates for Stage 14.2."""

from __future__ import annotations

import json
import math
import statistics
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from stage14_2_benchmark_diagnostics import OUTPUT, stage14_2_smoke_cases
from stage14_benchmark_runner import (
    BenchmarkCase,
    numeric_statistics,
    read_jsonl,
    run_benchmark_cases,
    run_limited_subprocess,
    summarize_repetitions,
)


ROOT = Path(__file__).resolve().parent


class Stage142RunnerUnitTests(unittest.TestCase):
    def test_case_is_immutable_and_id_is_deterministic(self) -> None:
        case = BenchmarkCase(
            experiment_type="unit", name="same case", worker_mode="controlled_timeout",
            sweep_variable="delay_s", parameters=(("delay_s", 0.01),),
            repetitions=1, timeout_s=1.0,
        )
        duplicate = BenchmarkCase(**{
            "experiment_type": "unit", "name": "same case",
            "worker_mode": "controlled_timeout", "sweep_variable": "delay_s",
            "parameters": (("delay_s", 0.01),), "repetitions": 1,
            "timeout_s": 1.0,
        })
        self.assertEqual(case.case_id, duplicate.case_id)
        self.assertEqual(case.repetition_id(0), duplicate.repetition_id(0))
        with self.assertRaises(FrozenInstanceError):
            case.name = "changed"  # type: ignore[misc]

    def test_case_requires_exactly_one_declared_parameter_name(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkCase(
                experiment_type="unit", name="bad", worker_mode="canonical",
                sweep_variable="missing", parameters=(("x", 1),), repetitions=1,
                timeout_s=1.0,
            )

    def test_statistics_definition_matches_python_reference(self) -> None:
        values = [3.0, 1.0, 2.0]
        result = numeric_statistics(values)
        self.assertEqual(result["individual"], values)
        self.assertEqual(result["median"], statistics.median(values))
        self.assertEqual(result["mean"], statistics.mean(values))
        self.assertEqual(
            result["standard_deviation_sample"], statistics.stdev(values)
        )

    def test_parent_enforces_controlled_memory_limit(self) -> None:
        result = run_limited_subprocess(
            (sys.executable, "-c", "import time; time.sleep(2)"),
            cwd=ROOT, timeout_s=1.0, memory_limit_bytes=1,
            sampling_interval_s=0.01,
        )
        self.assertEqual(result.limit_status, "memory_limit")
        self.assertGreaterEqual(result.peak_rss_bytes, result.start_rss_bytes)

    def test_append_safe_resume_skips_terminal_repetition(self) -> None:
        case = BenchmarkCase(
            experiment_type="unit", name="quick timeout",
            worker_mode="controlled_timeout", sweep_variable="delay_s",
            parameters=(("delay_s", 1.0),), repetitions=1, timeout_s=0.05,
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary)
            first = run_benchmark_cases(
                (case,), output_directory=output, environment_manifest_id="unit-manifest"
            )
            second = run_benchmark_cases(
                (case,), output_directory=output, environment_manifest_id="unit-manifest"
            )
            rows = read_jsonl(output / "raw_repetitions.jsonl")
        self.assertEqual(first["attempted_repetition_ids"], [case.repetition_id(0)])
        self.assertEqual(second["attempted_repetition_ids"], [])
        self.assertEqual(second["skipped_repetition_ids"], [case.repetition_id(0)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "timeout")

    def test_force_replaces_requested_repetition_without_duplicates(self) -> None:
        case = BenchmarkCase(
            experiment_type="unit", name="forced quick timeout",
            worker_mode="controlled_timeout", sweep_variable="delay_s",
            parameters=(("delay_s", 1.0),), repetitions=1, timeout_s=0.05,
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary)
            run_benchmark_cases(
                (case,), output_directory=output, environment_manifest_id="unit-manifest"
            )
            forced = run_benchmark_cases(
                (case,), output_directory=output, environment_manifest_id="unit-manifest",
                force=True,
            )
            rows = read_jsonl(output / "raw_repetitions.jsonl")
        self.assertEqual(forced["attempted_repetition_ids"], [case.repetition_id(0)])
        self.assertEqual(forced["skipped_repetition_ids"], [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["repetition_id"], case.repetition_id(0))


class Stage142ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            (OUTPUT / "stage14_2_summary.json").read_text(encoding="utf-8")
        )
        cls.rows = read_jsonl(OUTPUT / "raw_repetitions.jsonl")
        cls.summaries = json.loads(
            (OUTPUT / "configuration_summaries.json").read_text(encoding="utf-8")
        )

    def test_every_attempt_and_failure_is_recorded(self) -> None:
        self.assertEqual(len(self.rows), 5)
        self.assertEqual(len({row["repetition_id"] for row in self.rows}), 5)
        statuses = {status: sum(row["status"] == status for row in self.rows) for status in {
            "completed", "model_infeasible", "timeout"
        }}
        self.assertEqual(statuses, {"completed": 3, "model_infeasible": 1, "timeout": 1})
        for row in self.rows:
            self.assertTrue(row["started_at_utc"])
            self.assertTrue(row["ended_at_utc"])
            self.assertTrue(row["environment_manifest_id"])
            self.assertEqual(row["process_semantics"], "fresh cold process")
        failures = [row for row in self.rows if row["status"] != "completed"]
        self.assertTrue(all(row["failure_category"] for row in failures))
        self.assertTrue(all(row["failure_message"] for row in failures))

    def test_repeated_canonical_solution_is_identical(self) -> None:
        canonical_id = self.summary["canonical_case_id"]
        rows = [row for row in self.rows if row["case_id"] == canonical_id]
        self.assertEqual(len(rows), 3)
        signatures = {
            (
                row["J_A"], row["J_D"], row["selected_defender_action_id"],
                row["selected_attacker_candidate_id"],
                row["trajectory_identity"]["sha256"],
                json.dumps(row["tie_breaking_result"], sort_keys=True),
            )
            for row in rows
        }
        self.assertEqual(len(signatures), 1)
        self.assertTrue(all(row["independent_replay_passed"] for row in rows))

    def test_summary_statistics_reproduce_raw_repetitions(self) -> None:
        reproduced = summarize_repetitions(self.rows)
        self.assertEqual(reproduced, self.summaries)
        stats = self.summary["canonical_repetition_statistics"]
        canonical_times = [
            row["timing"]["totals"]["T_SSE_s"] for row in self.rows
            if row["case_id"] == self.summary["canonical_case_id"]
        ]
        self.assertTrue(math.isclose(stats["median"], statistics.median(canonical_times)))
        self.assertTrue(math.isclose(stats["mean"], statistics.mean(canonical_times)))
        self.assertTrue(math.isclose(
            stats["standard_deviation_sample"], statistics.stdev(canonical_times)
        ))

    def test_resume_is_deterministic_and_duplicates_are_absent(self) -> None:
        report = self.summary["resume_report"]
        self.assertTrue(report["deterministic_resume_passed"])
        self.assertEqual(report["second_invocation_attempted"], [])
        self.assertEqual(len(report["second_invocation_skipped"]), 5)

    def test_case_contract_has_one_sweep_variable_and_process_limits(self) -> None:
        for case in stage14_2_smoke_cases():
            self.assertIn(case.sweep_variable, case.parameter_dict)
            self.assertGreater(case.timeout_s, 0.0)
            self.assertEqual(case.case_id, case.case_id)

    def test_machine_readable_and_visual_artifacts_exist(self) -> None:
        required = (
            "smoke_case_configurations.json", "raw_repetitions.jsonl",
            "raw_repetitions.json", "raw_repetitions.csv",
            "configuration_summaries.json", "configuration_summaries.csv",
            "resume_report.json", "canonical_repetition_stability.png",
            "stage14_2_summary.json",
        )
        for name in required:
            path = OUTPUT / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 0, name)

    def test_stage14_notebook_preserves_runner_and_substage_order(self) -> None:
        notebook = json.loads(
            (ROOT / "3D_Stage14_Computation_Load.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("run_stage14_2_diagnostics", source)
        self.assertIn("RECOMPUTE_STAGE14_2 = False", source)
        self.assertIn(
            "run_stage14_2_diagnostics(STAGE14_2_DIR, force=True)", source
        )
        self.assertIn(
            "run_stage14_3_diagnostics(STAGE14_3_DIR, force=True)", source
        )
        self.assertIn(
            "run_stage14_4_diagnostics(STAGE14_4_DIR, force=True)", source
        )
        self.assertIn(
            "run_stage14_5_diagnostics(STAGE14_5_DIR, force=True)", source
        )
        self.assertIn(
            "run_stage14_6_diagnostics(STAGE14_6_DIR, force=True)", source
        )
        self.assertIn("run_stage14_3_diagnostics", source)
        self.assertIn("run_stage14_4_diagnostics", source)
        self.assertIn("run_stage14_5_diagnostics", source)
        self.assertIn("run_stage14_6_diagnostics", source)
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertLess(ids.index("stage14-2-figure"), ids.index("stage14-3-heading"))
        self.assertLess(ids.index("stage14-3-heading"), ids.index("stage14-4-heading"))
        self.assertLess(ids.index("stage14-4-heading"), ids.index("stage14-5-heading"))
        self.assertLess(ids.index("stage14-5-heading"), ids.index("stage14-6-heading"))


if __name__ == "__main__":
    unittest.main()

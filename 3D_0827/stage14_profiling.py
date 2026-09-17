"""Stage-14.1 non-overlapping timing and process-tree RSS instrumentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
from time import perf_counter, sleep
from typing import Any, Sequence

import numpy as np
import psutil

from stackelberg_solver import FiniteStackelbergRun


PROFILE_SCHEMA_VERSION = "stage14.1-v1"
SPARSE_BELLMAN_TIMER_SEMANTICS = "hazard-exclusive-v1"


@dataclass(frozen=True)
class ProcessTreeMemoryMeasurement:
    """Sampled RSS for one fresh worker and all of its descendants."""

    start_rss_bytes: int
    peak_rss_bytes: int
    delta_peak_rss_bytes: int
    sampling_interval_s: float
    child_processes_included: bool = True
    native_allocations_included: bool = True
    method: str = "psutil sampled worker process-tree RSS"

    def __post_init__(self) -> None:
        integer_values = (
            self.start_rss_bytes,
            self.peak_rss_bytes,
            self.delta_peak_rss_bytes,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in integer_values):
            raise ValueError("RSS values must be nonnegative integers")
        if self.peak_rss_bytes < self.start_rss_bytes:
            raise ValueError("peak RSS must be at least start RSS")
        if self.delta_peak_rss_bytes != self.peak_rss_bytes - self.start_rss_bytes:
            raise ValueError("delta peak RSS must equal peak minus start")
        if not np.isfinite(self.sampling_interval_s) or self.sampling_interval_s <= 0.0:
            raise ValueError("sampling_interval_s must be finite and positive")


@dataclass(frozen=True)
class ProfiledSubprocessResult:
    returncode: int
    stdout: str
    stderr: str
    monitor_wall_s: float
    memory: ProcessTreeMemoryMeasurement


TIMING_DEFINITIONS = {
    "T_SSE_s": "wall time inside run_finite_stackelberg for the complete finite leader solve",
    "T_graph_s": "shared goal-backward graph construction plus per-action graph access/build time",
    "T_LOS_s": "sum of LOS/tangent geometry time over Defender actions",
    "T_hazard_s": "sum of active-edge detection-hazard precomputation time",
    "T_switch_s": "candidate generation + powered/energy filtering + virtual-connection processing",
    "T_Bellman_s": "sum of additive Bellman numerical solve time",
    "T_defender_overhead_s": "leader/follower tie selection and loop overhead outside shared graph and Attacker-BR calls",
    "T_attacker_unclassified_s": "exact Attacker-BR residual: corridor setup, option replay/objective evaluation, selection, and other glue",
    "T_validation_s": "independent replay after T_SSE; excluded from T_SSE accounting",
}


COUPLING_NOTES = {
    "T_graph_s": (
        "The canonical solver reuses one exact goal-backward graph. The shared build "
        "is timed once; tiny per-action prebuilt-graph access checks remain in the "
        "Attacker graph timer and are retained rather than reassigned."
    ),
    "T_switch_s": (
        "Reported both as a reliable aggregate and as candidate-generation, "
        "energy-filter, and virtual-connection subcomponents."
    ),
    "T_attacker_unclassified_s": (
        "The current solver couples forward-corridor restriction, objective-edge "
        "assembly, candidate replay, and final selection. Stage 14.1 does not invent "
        "a finer split."
    ),
    "T_defender_overhead_s": (
        "Derived from T_SSE minus shared graph and exact Attacker-BR wall times; it "
        "includes SSE follower tie-breaking and final leader comparison."
    ),
}


def _nonnegative_residual(total: float, parts: Sequence[float], name: str) -> float:
    residual = float(total) - float(sum(parts))
    if residual < -1.0e-9:
        raise RuntimeError(f"{name} timing components exceed their enclosing wall time")
    return max(0.0, residual)


def decompose_exact_sse_timing(
    run: FiniteStackelbergRun,
    *,
    validation_time_s: float,
) -> dict[str, Any]:
    """Normalize existing exact-solver timers into one non-overlapping record."""
    if not isinstance(run, FiniteStackelbergRun):
        raise TypeError("run must be a FiniteStackelbergRun")
    validation = float(validation_time_s)
    if not np.isfinite(validation) or validation < 0.0:
        raise ValueError("validation_time_s must be finite and nonnegative")

    per_defender: list[dict[str, Any]] = []
    for evaluation in run.evaluations:
        timing = evaluation.attacker_run.metrics.timing
        bellman_s = float(timing.bellman_solve_s)
        if getattr(
            evaluation.attacker_run.metrics, "bellman_backend", "dense"
        ) == "goal_backward_sparse":
            # SparseAdditiveMetrics.solve_s encloses the timed hazard batches.
            # Remove that nested interval so the Stage-14 decomposition remains
            # non-overlapping and reconciles to the enclosing Attacker-BR wall time.
            bellman_s = _nonnegative_residual(
                bellman_s,
                (timing.hazard_precompute_s,),
                "Sparse Bellman solve",
            )
        switch_parts = (
            timing.candidate_generation_s,
            timing.energy_filter_s,
            timing.virtual_connection_s,
        )
        named_parts = (
            timing.los_s,
            timing.graph_build_s,
            timing.hazard_precompute_s,
            bellman_s,
            *switch_parts,
        )
        residual = _nonnegative_residual(
            timing.total_attacker_br_s,
            named_parts,
            "Attacker best-response",
        )
        per_defender.append({
            "defender_action_id": evaluation.candidate.action_id,
            "feasible": evaluation.feasible,
            "T_BR_s": timing.total_attacker_br_s,
            "T_graph_access_or_build_s": timing.graph_build_s,
            "T_LOS_s": timing.los_s,
            "T_switch_s": sum(switch_parts),
            "T_switch_candidate_generation_s": timing.candidate_generation_s,
            "T_switch_energy_filter_s": timing.energy_filter_s,
            "T_switch_virtual_connection_s": timing.virtual_connection_s,
            "T_hazard_s": timing.hazard_precompute_s,
            "T_Bellman_s": bellman_s,
            "T_attacker_unclassified_s": residual,
            "T_BR_accounted_s": sum(named_parts) + residual,
        })

    shared_graph = run.timing.shared_graph_build_s
    attacker_total = sum(row["T_BR_s"] for row in per_defender)
    defender_overhead = _nonnegative_residual(
        run.timing.total_s,
        (shared_graph, attacker_total),
        "Stackelberg",
    )
    totals = {
        "T_SSE_s": run.timing.total_s,
        "T_graph_s": shared_graph + sum(
            row["T_graph_access_or_build_s"] for row in per_defender
        ),
        "T_graph_shared_s": shared_graph,
        "T_graph_per_action_s": sum(
            row["T_graph_access_or_build_s"] for row in per_defender
        ),
        "T_LOS_s": sum(row["T_LOS_s"] for row in per_defender),
        "T_switch_s": sum(row["T_switch_s"] for row in per_defender),
        "T_switch_candidate_generation_s": sum(
            row["T_switch_candidate_generation_s"] for row in per_defender
        ),
        "T_switch_energy_filter_s": sum(
            row["T_switch_energy_filter_s"] for row in per_defender
        ),
        "T_switch_virtual_connection_s": sum(
            row["T_switch_virtual_connection_s"] for row in per_defender
        ),
        "T_hazard_s": sum(row["T_hazard_s"] for row in per_defender),
        "T_Bellman_s": sum(row["T_Bellman_s"] for row in per_defender),
        "T_attacker_unclassified_s": sum(
            row["T_attacker_unclassified_s"] for row in per_defender
        ),
        "T_defender_overhead_s": defender_overhead,
        "T_validation_s": validation,
    }
    nonoverlapping_keys = (
        "T_graph_s", "T_LOS_s", "T_switch_s", "T_hazard_s", "T_Bellman_s",
        "T_attacker_unclassified_s", "T_defender_overhead_s",
    )
    accounted = float(sum(totals[key] for key in nonoverlapping_keys))
    reconciliation_error = accounted - totals["T_SSE_s"]
    if abs(reconciliation_error) > 1.0e-9:
        raise RuntimeError("normalized timing does not reconcile with T_SSE")
    if any(not np.isfinite(value) or value < 0.0 for value in totals.values()):
        raise RuntimeError("normalized timing contains invalid values")
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "sparse_bellman_timer_semantics": SPARSE_BELLMAN_TIMER_SEMANTICS,
        "timing_definitions": TIMING_DEFINITIONS,
        "coupling_notes": COUPLING_NOTES,
        "per_defender_action": per_defender,
        "totals": totals,
        "nonoverlapping_T_SSE_component_keys": list(nonoverlapping_keys),
        "T_SSE_accounted_s": accounted,
        "T_SSE_reconciliation_error_s": reconciliation_error,
        "validation_included_in_T_SSE": False,
    }


def _rss_process_tree_bytes(process: psutil.Process) -> int:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    total = 0
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def run_profiled_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    sampling_interval_s: float = 0.01,
) -> ProfiledSubprocessResult:
    """Run one fresh worker and sample process-tree RSS including native memory."""
    interval = float(sampling_interval_s)
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("sampling_interval_s must be finite and positive")
    started = perf_counter()
    child = subprocess.Popen(
        tuple(command), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    )
    process = psutil.Process(child.pid)
    start_rss = 0
    peak_rss = 0
    while child.poll() is None:
        rss = _rss_process_tree_bytes(process)
        if rss > 0 and start_rss == 0:
            start_rss = rss
        peak_rss = max(peak_rss, rss)
        sleep(interval)
    rss = _rss_process_tree_bytes(process)
    if rss > 0 and start_rss == 0:
        start_rss = rss
    peak_rss = max(peak_rss, rss, start_rss)
    stdout, stderr = child.communicate()
    measurement = ProcessTreeMemoryMeasurement(
        start_rss_bytes=start_rss,
        peak_rss_bytes=peak_rss,
        delta_peak_rss_bytes=peak_rss - start_rss,
        sampling_interval_s=interval,
    )
    return ProfiledSubprocessResult(
        returncode=int(child.returncode),
        stdout=stdout,
        stderr=stderr,
        monitor_wall_s=perf_counter() - started,
        memory=measurement,
    )


def subprocess_result_dict(result: ProfiledSubprocessResult) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "monitor_wall_s": result.monitor_wall_s,
        "memory": asdict(result.memory),
    }


__all__ = [
    "COUPLING_NOTES", "PROFILE_SCHEMA_VERSION", "TIMING_DEFINITIONS",
    "SPARSE_BELLMAN_TIMER_SEMANTICS",
    "ProcessTreeMemoryMeasurement", "ProfiledSubprocessResult",
    "decompose_exact_sse_timing", "run_profiled_subprocess", "subprocess_result_dict",
]

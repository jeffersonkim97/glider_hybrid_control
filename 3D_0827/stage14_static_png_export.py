"""Regenerate every Stage-14 visual as PNG from already-saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from stage11_notebook_support import save_figure_png
from stage14_0_reference_diagnostics import _write_tables
from stage14_1_profiling_diagnostics import _memory_figure, _runtime_figure
from stage14_2_benchmark_diagnostics import run_stage14_2_diagnostics
from stage14_2a_local_sse_diagnostics import run_stage14_2a_diagnostics
from stage14_2b_diagnostics import run_stage14_2b_diagnostics
from stage14_2c_diagnostics import run_stage14_2c_diagnostics


ROOT = Path(__file__).resolve().parent
FIGURE_ROOT = ROOT / "figure"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def export_all() -> list[Path]:
    stage0 = FIGURE_ROOT / "stage_14_0_benchmark_contract"
    stage1 = FIGURE_ROOT / "stage_14_1_runtime_profiling"
    reference = _read_json(stage0 / "frozen_reference.json")
    profile = _read_json(stage1 / "canonical_profile.json")
    _write_tables(stage0, reference)
    save_figure_png(
        _runtime_figure(profile), stage1, "canonical_runtime_decomposition.png"
    )
    save_figure_png(_memory_figure(profile), stage1, "canonical_memory_summary.png")
    run_stage14_2_diagnostics(FIGURE_ROOT / "stage_14_2_benchmark_runner")
    run_stage14_2a_diagnostics(FIGURE_ROOT / "stage_14_2a_local_sse_contract")
    run_stage14_2b_diagnostics(
        FIGURE_ROOT / "stage_14_2b_local_sse_search", recompute=False
    )
    run_stage14_2c_diagnostics(
        FIGURE_ROOT / "stage_14_2c_local_scaling", force=False
    )
    return sorted(FIGURE_ROOT.glob("stage_14*/*.png"))


if __name__ == "__main__":
    for artifact in export_all():
        print(artifact.relative_to(ROOT))

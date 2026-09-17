"""Re-render every Stage-14 graph from saved artifacts without benchmark work."""

from __future__ import annotations

import json
from pathlib import Path

from stage11_notebook_support import save_figure_png
from stage14_1_profiling_diagnostics import _memory_figure as profile_memory_figure
from stage14_1_profiling_diagnostics import _runtime_figure as profile_runtime_figure
from stage14_2_benchmark_diagnostics import _repetition_figure
from stage14_2a_local_sse_diagnostics import _payoff_figure
from stage14_2b_diagnostics import _diagnostic_figure
from stage14_2c_contract import stage14_2c_cases
from stage14_2c_diagnostics import _case_metadata, _generate_figures
from stage14_3_diagnostics import regenerate_stage14_3_figures_from_saved_data
from stage14_4_diagnostics import regenerate_stage14_4_figures_from_saved_data
from stage14_5_diagnostics import regenerate_stage14_5_figures_from_saved_data
from stage14_6_diagnostics import regenerate_stage14_6_figures_from_saved_data
from stage14_7_scaling_analysis import regenerate_stage14_7_figures_from_saved_data


ROOT = Path(__file__).resolve().parent
FIGURE_ROOT = ROOT / "figure"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def regenerate_all() -> list[Path]:
    written: list[Path] = []

    # Stage 14.0 contains a table-only PNG and therefore has no legend.
    stage1 = FIGURE_ROOT / "stage_14_1_runtime_profiling"
    profile = _read(stage1 / "canonical_profile.json")
    written.append(save_figure_png(
        profile_runtime_figure(profile), stage1, "canonical_runtime_decomposition.png"
    ))
    written.append(save_figure_png(
        profile_memory_figure(profile), stage1, "canonical_memory_summary.png"
    ))

    stage2 = FIGURE_ROOT / "stage_14_2_benchmark_runner"
    rows = _read(stage2 / "raw_repetitions.json")
    summaries = _read(stage2 / "configuration_summaries.json")
    stage2_summary = _read(stage2 / "stage14_2_summary.json")
    canonical_id = stage2_summary["canonical_case_id"]
    canonical_rows = sorted(
        [row for row in rows if row["case_id"] == canonical_id],
        key=lambda row: row["repetition_index"],
    )
    canonical_summary = next(
        summary for summary in summaries if summary["case_id"] == canonical_id
    )
    written.append(save_figure_png(
        _repetition_figure(canonical_rows, canonical_summary), stage2,
        "canonical_repetition_stability.png",
    ))

    stage2a = FIGURE_ROOT / "stage_14_2a_local_sse_contract"
    tiny = _read(stage2a / "tiny_game_verification.json")
    written.append(save_figure_png(
        _payoff_figure(tiny), stage2a, "local_vs_global_payoff_diagnostic.png"
    ))

    stage2b = FIGURE_ROOT / "stage_14_2b_local_sse_search"
    payload = _read(stage2b / "canonical_worker_result.json")
    written.append(save_figure_png(
        _diagnostic_figure(payload), stage2b, "local_search_diagnostic.png"
    ))

    stage2c = FIGURE_ROOT / "stage_14_2c_local_scaling"
    stage2c_rows = _read(stage2c / "raw_repetitions.json")
    comparisons = _read(stage2c / "local_vs_global_oracle.json")
    stage2c_names = _generate_figures(
        stage2c_rows, comparisons, _case_metadata(stage14_2c_cases()), stage2c,
    )
    written.extend(stage2c / name for name in stage2c_names)

    stage3 = FIGURE_ROOT / "stage_14_3_spatial_resolution"
    written.extend(
        stage3 / name
        for name in regenerate_stage14_3_figures_from_saved_data(stage3)
    )
    stage4 = FIGURE_ROOT / "stage_14_4_heading_resolution"
    written.extend(
        stage4 / name
        for name in regenerate_stage14_4_figures_from_saved_data(stage4)
    )

    stage5 = FIGURE_ROOT / "stage_14_5_switching_candidates"
    written.extend(
        stage5 / name
        for name in regenerate_stage14_5_figures_from_saved_data(stage5)
    )

    stage6 = FIGURE_ROOT / "stage_14_6_neighbor_radius"
    written.extend(
        stage6 / name
        for name in regenerate_stage14_6_figures_from_saved_data(stage6)
    )

    stage7 = FIGURE_ROOT / "stage_14_7_scaling_analysis"
    written.extend(
        stage7 / name
        for name in regenerate_stage14_7_figures_from_saved_data(stage7)
    )

    print(
        f"Re-rendered {len(written)} Stage-14 graphs from saved artifacts; "
        "no benchmark worker invoked. Stage 14.0 table-only PNG unchanged."
    )
    return written


if __name__ == "__main__":
    regenerate_all()

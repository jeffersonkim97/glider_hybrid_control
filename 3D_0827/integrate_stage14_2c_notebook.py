"""Idempotently append the approved Stage 14.2C report to the Stage-14 notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "3D_Stage14_Computation_Load.ipynb"


def _replace_once(source: str, old: str, new: str) -> str:
    if new in source:
        return source
    if old not in source:
        raise ValueError(f"notebook integration anchor is absent: {old!r}")
    return source.replace(old, new, 1)


def integrate() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells[0].source = _replace_once(
        notebook.cells[0].source,
        "14.2A, and 14.2B are integrated here",
        "14.2A, 14.2B, and 14.2C are integrated here",
    )
    imports = notebook.cells[1].source
    imports = _replace_once(
        imports,
        "from IPython.display import FileLink, HTML, display",
        "from IPython.display import FileLink, HTML, Image, display",
    )
    imports = _replace_once(
        imports,
        "from stage14_2b_diagnostics import run_stage14_2b_diagnostics",
        "from stage14_2b_diagnostics import run_stage14_2b_diagnostics\n"
        "from stage14_2c_diagnostics import run_stage14_2c_diagnostics",
    )
    imports = _replace_once(
        imports,
        "STAGE14_2B_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_2b_local_sse_search'",
        "STAGE14_2B_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_2b_local_sse_search'\n"
        "STAGE14_2C_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_2c_local_scaling'",
    )
    notebook.cells[1].source = imports
    notebook.cells[2].source = _replace_once(
        notebook.cells[2].source,
        "Stage 14.2B runs the canonical exact local search in a fresh measured process.",
        "Stage 14.2B runs the canonical exact local search in a fresh measured process. "
        "Stage 14.2C runs/resumes the approved local-scaling and matched global-oracle matrix.",
    )
    notebook.cells[3].source = _replace_once(
        notebook.cells[3].source,
        "RECOMPUTE_STAGE14_2B = False",
        "RECOMPUTE_STAGE14_2B = False\nRECOMPUTE_STAGE14_2C = False",
    )
    if any(cell.id == "stage14-2c-heading" for cell in notebook.cells):
        nbformat.write(notebook, NOTEBOOK)
        return
    insert_at = next(
        index for index, cell in enumerate(notebook.cells)
        if cell.id == "future-substages"
    )
    new_cells = [
        nbformat.v4.new_markdown_cell(
            """## Stage 14.2C — local-SSE scaling and global-oracle gap

This section reports 60 fresh-process measurements over four controlled
one-factor-at-a-time sweeps. Terrain remains the single centered cube. The
single-start local SSE is the primary object; the exact global finite SSE is
used only as a matched tractable oracle. Reported gaps are empirical values on
the tested finite grids, not approximation-error bounds.""",
        ),
        nbformat.v4.new_code_cell(
            """if RECOMPUTE_STAGE14_2C:
    stage14_2c_summary = run_stage14_2c_diagnostics(STAGE14_2C_DIR)
else:
    summary_path = STAGE14_2C_DIR / 'stage14_2c_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError('Stage 14.2C artifact is absent; set RECOMPUTE_STAGE14_2C=True.')
    stage14_2c_summary = load_json(summary_path)

display_record({
    'gate_passed': stage14_2c_summary['gate_passed'],
    'local_cases': stage14_2c_summary['local_case_count'],
    'global_oracle_cases': stage14_2c_summary['global_oracle_case_count'],
    'repetitions_per_case': stage14_2c_summary['repetitions_per_case'],
    'raw_rows': stage14_2c_summary['total_repetition_rows'],
    'completed_local_repetitions': stage14_2c_summary['local_completed_repetitions'],
    'completed_global_oracle_repetitions': stage14_2c_summary['global_oracle_completed_repetitions'],
    'oracle_comparisons': stage14_2c_summary['successful_oracle_comparisons'],
    'multi_start_performed': stage14_2c_summary['interpretation']['multi_start_performed'],
    'gap_is_empirical_not_bounded': stage14_2c_summary['interpretation']['gap_is_empirical_not_bounded'],
})""",
        ),
        nbformat.v4.new_code_cell(
            """stage14_2c_local = load_json(STAGE14_2C_DIR / 'local_scaling_summaries.json')
stage14_2c_oracle = load_json(STAGE14_2C_DIR / 'local_vs_global_oracle.json')
local_rows = [[
    row['profile'],
    round(row['runtime_statistics_s']['median'], 3),
    row['state_and_game_size']['N_S_active'],
    row['state_and_game_size']['N_E'],
    row['state_and_game_size']['N_C'],
    row['state_and_game_size']['N_D'],
    row['local_search_statistics']['unique_defender_evaluations']['median'],
    row['local_search_statistics']['local_search_iterations']['median'],
    row['local_search_outcomes']['isolated_feasible_local_sse_repetitions'],
    round(row['J_D_statistics']['median'], 9),
] for row in stage14_2c_local]
display(HTML(
    '<table><tr><th>profile</th><th>T_local median [s]</th><th>N_S active</th>'
    '<th>N_E</th><th>N_C</th><th>N_D</th><th>N_eval</th><th>K</th>'
    '<th>isolated reps</th><th>J_D</th></tr>'
    + ''.join('<tr>' + ''.join(f'<td>{value}</td>' for value in row) + '</tr>' for row in local_rows)
    + '</table><style>th,td{border:1px solid #bbb;padding:4px 7px}</style>'
))

oracle_rows = [row for row in stage14_2c_oracle if row['status'] == 'completed']
display(HTML(
    '<h4>Tractable exact-global oracle comparisons</h4>'
    '<table><tr><th>profile</th><th>T_local</th><th>T_global</th><th>ratio</th><th>ΔJ_D</th></tr>'
    + ''.join(
        f"<tr><td>{row['profile']}</td><td>{row['T_local_median_s']:.3f}</td>"
        f"<td>{row['T_global_median_s']:.3f}</td>"
        f"<td>{row['runtime_ratio_global_over_local']:.3f}</td>"
        f"<td>{row['delta_J_D']:.9f}</td></tr>" for row in oracle_rows
    )
    + '</table><style>th,td{border:1px solid #bbb;padding:4px 7px}</style>'
))""",
        ),
        nbformat.v4.new_code_cell(
            """stage14_2c_figures = stage14_2c_summary['figure_files']
display(HTML('<h4>Static Stage 14.2C figures</h4>'))
for figure_name in stage14_2c_figures:
    display(HTML(f'<h5>{figure_name}</h5>'))
    display(Image(filename=str(STAGE14_2C_DIR / figure_name)))

display(FileLink(STAGE14_2C_DIR / 'raw_repetitions.csv'))
display(FileLink(STAGE14_2C_DIR / 'raw_repetitions.jsonl'))
display(FileLink(STAGE14_2C_DIR / 'regression_validation_report.json'))""",
        ),
        nbformat.v4.new_markdown_cell(
            """### Interpretation boundary

- Local certification, computational scaling, and global-oracle gap are separate claims.
- A small measured gap is not a proven error bound.
- Multi-start is not performed and cannot be called global optimization.
- No RL, approximate planner, or terrain-complexity sweep is included.""",
        ),
    ]
    ids = (
        "stage14-2c-heading", "stage14-2c-summary", "stage14-2c-tables",
        "stage14-2c-figures", "stage14-2c-interpretation",
    )
    for cell, cell_id in zip(new_cells, ids):
        cell.id = cell_id
    notebook.cells[insert_at:insert_at] = new_cells
    notebook.cells[-1].source = notebook.cells[-1].source.replace(
        "Stage 14.2C or any later sweep starts only after a separate review and GO decision.",
        "Any post-14.2C extension starts only after a separate review and GO decision.",
    )
    nbformat.write(notebook, NOTEBOOK)


if __name__ == "__main__":
    integrate()

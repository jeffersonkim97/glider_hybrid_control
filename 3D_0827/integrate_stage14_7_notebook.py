"""Idempotently append the approved Stage 14.7 static scaling report."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "3D_Stage14_Computation_Load.ipynb"


FIGURE_GUIDE = """### Stage 14.7 consolidated scaling figure guide

All points come from previously saved Stage 14.3–14.6 measurements; this
analysis starts no solver or benchmark worker. Dotted power-law curves are
finite-range empirical fits only. Their `alpha` values are not formal Big-O.

1. Total runtime versus spatial spacing `Delta x = Delta y = Delta h` in metres;
   full-grid and active-corridor state counts are secondary-y diagnostics.
2. Total runtime versus requested heading spacing `Delta psi` in degrees.
3. Total runtime versus normalized switching-contour spacing
   `Delta s = 1 / N_contour`; raw candidate count is a secondary-y diagnostic.
4. Total local-SSE runtime versus `r_neighbor`.
5. Unique evaluations and local-search iterations versus `r_neighbor`.
6. Bellman runtime versus sweep-specific spatial/heading discretization;
   `N_S_active` and `N_E` remain separate secondary-y diagnostics.
7. Peak memory versus sweep-specific spatial/heading discretization;
   `N_S_active` and `N_E` remain separate secondary-y diagnostics.
8. Additive local-SSE runtime decomposition across the four sweeps.
9. Objective sensitivity to spatial, heading, and switching discretization;
   this is a diagnostic and not a convergence proof.

Every legend is outside the axes at bottom center. Every output is static PNG."""


INTERPRETATION = """### Stage 14.7 claim boundary

- Theoretical reference relations and measured empirical fits are separate.
- Fits use one median point per completed configuration, not three repetitions
  treated as independent scaling samples.
- Fit curves stop at the measured minimum and maximum; no extrapolation is
  performed.
- Model-infeasible and computational/instrumentation failures are retained in
  the failure inventory and excluded from fits.
- `N_S_active` and `N_E` are never added into an artificial combined plot axis.
- State, edge, candidate, and heading-bin counts are not used as figure
  x-axes. Each figure uses the user-controlled discretization scale.
- No Stage 14.8 work is started here."""


def integrate() -> Path:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    by_id = {cell.get("id"): cell for cell in notebook.cells}
    notebook.cells[0].source = notebook.cells[0].source.replace(
        "14.5, and 14.6 are integrated here",
        "14.5, 14.6, and 14.7 are integrated here",
    )
    imports = notebook.cells[1].source
    if "run_stage14_7_analysis" not in imports:
        imports = imports.replace(
            "from stage14_6_diagnostics import run_stage14_6_diagnostics",
            "from stage14_6_diagnostics import run_stage14_6_diagnostics\n"
            "from stage14_7_scaling_analysis import run_stage14_7_analysis",
        )
    if "STAGE14_7_DIR" not in imports:
        imports = imports.replace(
            "STAGE14_6_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_6_neighbor_radius'",
            "STAGE14_6_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_6_neighbor_radius'\n"
            "STAGE14_7_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_7_scaling_analysis'",
        )
    notebook.cells[1].source = imports
    if "RECOMPUTE_STAGE14_7" not in notebook.cells[3].source:
        notebook.cells[3].source = notebook.cells[3].source.replace(
            "RECOMPUTE_STAGE14_6 = False",
            "RECOMPUTE_STAGE14_6 = False\nRECOMPUTE_STAGE14_7 = False",
        )

    if "stage14-7-heading" not in by_id:
        insert_at = next(
            index for index, cell in enumerate(notebook.cells)
            if cell.get("id") == "future-substages"
        )
        cells = [
            nbformat.v4.new_markdown_cell(
                """## Stage 14.7 — theoretical and empirical scaling analysis

This section consolidates completed Stage 14.3–14.6 measurements. It performs
analysis and static rendering only; it does not launch Bellman, Attacker-BR,
local-SSE, or global-SSE workers."""
            ),
            nbformat.v4.new_code_cell(
                """if RECOMPUTE_STAGE14_7:
    stage14_7_summary = run_stage14_7_analysis(STAGE14_7_DIR)
else:
    summary_path = STAGE14_7_DIR / 'stage14_7_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError('Stage 14.7 artifact is absent; set RECOMPUTE_STAGE14_7=True.')
    stage14_7_summary = load_json(summary_path)

stage14_7_validation = load_json(STAGE14_7_DIR / 'stage14_7_validation_report.json')
display_record({
    'gate_passed': stage14_7_summary['gate_passed'],
    'source_sweeps': stage14_7_summary['source_sweeps'],
    'normalized_configurations': stage14_7_summary['normalized_configuration_count'],
    'completed_configurations': stage14_7_summary['completed_configuration_count'],
    'completed_fits': f"{stage14_7_summary['completed_fit_count']}/{stage14_7_summary['fit_record_count']}",
    'noncompleted_repetitions_retained': stage14_7_summary['noncompleted_repetition_count'],
    'solver_or_benchmark_workers_started': stage14_7_summary['solver_or_benchmark_workers_started'],
    'next_stage_started': stage14_7_summary['next_stage_started'],
})"""
            ),
            nbformat.v4.new_code_cell(
                """stage14_7_fits = load_json(STAGE14_7_DIR / 'fit_parameters.json')
primary_fit_ids = [
    'spatial__local_sse__T_total_s_vs_spatial_resolution_m',
    'heading__local_sse__T_total_s_vs_heading_resolution_deg',
    'switching__local_sse__T_total_s_vs_switching_contour_spacing',
    'radius__local_sse__T_total_s_vs_r_neighbor',
    'radius__local_sse__T_total_s_vs_N_eval_unique',
]
fit_by_id = {row['fit_id']: row for row in stage14_7_fits}
table_rows = []
for fit_id in primary_fit_ids:
    row = fit_by_id[fit_id]
    table_rows.append([
        fit_id, row['sample_count'], row['fit_range'],
        round(row['c'], 6), round(row['alpha'], 6), round(row['r_squared'], 6),
        row['empirical_fit_only'], row['formal_big_o_claim'], row['extrapolation_performed'],
    ])
display(HTML(
    '<table><tr><th>fit</th><th>n</th><th>range</th><th>c</th><th>alpha</th>'
    '<th>R^2</th><th>empirical only</th><th>formal Big-O</th><th>extrapolated</th></tr>'
    + ''.join('<tr>' + ''.join(f'<td>{value}</td>' for value in row) + '</tr>' for row in table_rows)
    + '</table><style>th,td{border:1px solid #bbb;padding:4px 7px}</style>'
))"""
            ),
            nbformat.v4.new_markdown_cell(FIGURE_GUIDE),
            nbformat.v4.new_code_cell(
                """display(HTML('<h4>Static Stage 14.7 figures</h4>'))
for figure_name in stage14_7_summary['figure_files']:
    display(HTML(f'<h5>{figure_name}</h5>'))
    display(Image(filename=str(STAGE14_7_DIR / figure_name)))

display(FileLink(STAGE14_7_DIR / 'fit_parameters.csv'))
display(FileLink(STAGE14_7_DIR / 'fit_parameters.json'))
display(FileLink(STAGE14_7_DIR / 'computational_limit_cases.json'))
display(FileLink(STAGE14_7_DIR / 'raw_summary_crosscheck.json'))
display(FileLink(STAGE14_7_DIR / 'theoretical_vs_empirical_interpretation.md'))
display(FileLink(STAGE14_7_DIR / 'stage14_7_validation_report.json'))"""
            ),
            nbformat.v4.new_markdown_cell(INTERPRETATION),
        ]
        ids = (
            "stage14-7-heading", "stage14-7-summary", "stage14-7-fit-table",
            "stage14-7-figure-guide", "stage14-7-figures", "stage14-7-interpretation",
        )
        for cell, cell_id in zip(cells, ids):
            cell.id = cell_id
        notebook.cells[insert_at:insert_at] = cells
    else:
        by_id["stage14-7-figure-guide"].source = FIGURE_GUIDE
        by_id["stage14-7-interpretation"].source = INTERPRETATION
        fit_source = by_id["stage14-7-fit-table"].source
        fit_source = fit_source.replace(
            "spatial__local_sse__T_total_s_vs_N_S_active",
            "spatial__local_sse__T_total_s_vs_spatial_resolution_m",
        ).replace(
            "heading__local_sse__T_total_s_vs_N_psi",
            "heading__local_sse__T_total_s_vs_heading_resolution_deg",
        ).replace(
            "switching__local_sse__T_total_s_vs_N_C",
            "switching__local_sse__T_total_s_vs_switching_contour_spacing",
        )
        by_id["stage14-7-fit-table"].source = fit_source

    notebook.cells[-1].source = notebook.cells[-1].source.replace(
        "Stage 14.7 starts only after a separate review and GO decision.",
        "Stage 14.8 starts only after a separate review and GO decision.",
    )
    nbformat.write(notebook, NOTEBOOK)
    return NOTEBOOK


if __name__ == "__main__":
    print(integrate())

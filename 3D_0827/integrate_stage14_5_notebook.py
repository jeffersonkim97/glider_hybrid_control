"""Idempotently append the approved Stage 14.5 static report."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "3D_Stage14_Computation_Load.ipynb"


FIGURE_GUIDE = """### Stage 14.5 switching-candidate-density figure guide

The common x-axis is normalized contour spacing `Delta s = 1/N_contour`. Only
the contour sampling density changes; radial sampling remains fixed at eight.
The certified local SSE is the primary solid series and the frozen exact global
solver is a dashed tractable oracle/reference series.

Stages 14.3--14.6 use the same six-figure schema:

1. total SSE runtime;
2. the common runtime-component decomposition for both solvers;
3. peak process-tree memory;
4. `J_A` and `J_D`;
5. selected Defender/Attacker equilibrium IDs;
6. selected trajectory identity and switching-point coordinates.

Candidate-filter counts remain available in raw/tabular validation artifacts,
but are not a primary performance figure.

Every repetition uses a fresh isolated Python worker. This does not claim cold
operating-system or hardware caches. `N_C_unique` records distinct physical
positions but does not change or deduplicate the solver candidate set."""


TABLE_SOURCE = """stage14_5_rows = load_json(STAGE14_5_DIR / 'candidate_sweep_summaries.json')
table_rows = []
for row in stage14_5_rows:
    components = row['runtime_statistics_s']
    med = lambda key: None if components.get(key) is None else round(components[key]['median'], 4)
    identity = row['solution_identities'][0] if row['solution_identities'] else (None,) * 5
    table_rows.append([
        row['algorithm_variant'], round(1.0 / row['requested_contour_sample_count'], 6),
        med('T_SSE_s'), med('T_LOS_s'), med('T_graph_s'), med('T_hazard_s'), med('T_switch_s'), med('T_Bellman_s'),
        None if row['peak_rss_statistics_bytes'] is None else round(row['peak_rss_statistics_bytes']['median'] / 1024**2, 2),
        None if row['J_A_statistics'] is None else row['J_A_statistics']['median'],
        None if row['J_D_statistics'] is None else row['J_D_statistics']['median'],
        identity[0], identity[1], None if identity[4] is None else identity[4][:10],
    ])
display(HTML(
    '<table><tr><th>variant</th><th>Delta s</th><th>T_SSE</th><th>T_LOS</th>'
    '<th>T_graph</th><th>T_hazard</th><th>T_switch</th><th>T_Bellman</th>'
    '<th>peak RSS [MiB]</th><th>J_A</th><th>J_D</th><th>D ID</th><th>A ID</th><th>trajectory</th></tr>'
    + ''.join('<tr>' + ''.join(f'<td>{value}</td>' for value in values) + '</tr>' for values in table_rows)
    + '</table><style>th,td{border:1px solid #bbb;padding:4px 7px}</style>'
))"""


INTERPRETATION = """### Stage 14.5 interpretation boundary

- Only switching-contour sample count varies; spatial/altitude/heading grids,
  radial sampling, quadrature, Defender actions, terrain, physics, objectives,
  and tie rules remain fixed.
- Every plotted timing point is a median of three isolated-process repetitions.
- The candidate funnel contains cumulative survivor counts. The unique-position
  curve is diagnostic and is not an additional solver filter.
- Exact exhaustive candidate minimization, Strong follower tie handling,
  local-SSE certification, and independent trajectory replay are exit-gate
  requirements.
- Objective convergence is not declared from three candidate-density points.
- No Stage 14.6 work is started here."""


def integrate() -> Path:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    by_id = {cell.get("id"): cell for cell in notebook.cells}
    notebook.cells[0].source = notebook.cells[0].source.replace(
        "14.2C, 14.3, and 14.4 are integrated here",
        "14.2C, 14.3, 14.4, and 14.5 are integrated here",
    )
    imports = notebook.cells[1].source
    if "run_stage14_5_diagnostics" not in imports:
        imports = imports.replace(
            "from stage14_4_diagnostics import run_stage14_4_diagnostics",
            "from stage14_4_diagnostics import run_stage14_4_diagnostics\n"
            "from stage14_5_diagnostics import run_stage14_5_diagnostics",
        )
    if "STAGE14_5_DIR" not in imports:
        imports = imports.replace(
            "STAGE14_4_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_4_heading_resolution'",
            "STAGE14_4_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_4_heading_resolution'\n"
            "STAGE14_5_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_5_switching_candidates'",
        )
    notebook.cells[1].source = imports
    if "RECOMPUTE_STAGE14_5" not in notebook.cells[3].source:
        notebook.cells[3].source = notebook.cells[3].source.replace(
            "RECOMPUTE_STAGE14_4 = False",
            "RECOMPUTE_STAGE14_4 = False\nRECOMPUTE_STAGE14_5 = False",
        )

    if "stage14-5-heading" not in by_id:
        insert_at = next(
            index for index, cell in enumerate(notebook.cells)
            if cell.get("id") == "future-substages"
        )
        cells = [
            nbformat.v4.new_markdown_cell(
                """## Stage 14.5 — one-variable switching-candidate count sweep

This section varies only LOS-tangent-surface contour sampling over 6, 9, and
12 samples while holding the radial count at 8. The resulting raw candidate
counts are 48, 72, and 96. The physical mission and continuous terrain are
unchanged."""
            ),
            nbformat.v4.new_code_cell(
                """if RECOMPUTE_STAGE14_5:
    stage14_5_summary = run_stage14_5_diagnostics(STAGE14_5_DIR)
else:
    summary_path = STAGE14_5_DIR / 'stage14_5_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError('Stage 14.5 artifact is absent; set RECOMPUTE_STAGE14_5=True.')
    stage14_5_summary = load_json(summary_path)

stage14_5_validation = load_json(STAGE14_5_DIR / 'stage14_5_validation_report.json')
display_record({
    'gate_passed': stage14_5_summary['gate_passed'],
    'contour_sample_counts': stage14_5_summary['contour_sample_counts'],
    'fixed_radial_sample_count': stage14_5_summary['fixed_radial_sample_count'],
    'raw_candidate_counts': stage14_5_summary['raw_candidate_counts'],
    'total_repetitions': stage14_5_summary['total_repetition_count'],
    'Bellman architecture measurement': stage14_5_summary['bellman_runtime_architecture_measurement'],
    'next_stage_started': stage14_5_summary['next_stage_started'],
})"""
            ),
            nbformat.v4.new_code_cell(TABLE_SOURCE),
            nbformat.v4.new_markdown_cell(FIGURE_GUIDE),
            nbformat.v4.new_code_cell(
                """display(HTML('<h4>Static Stage 14.5 figures</h4>'))
for figure_name in stage14_5_summary['figure_files']:
    display(HTML(f'<h5>{figure_name}</h5>'))
    display(Image(filename=str(STAGE14_5_DIR / figure_name)))

display(FileLink(STAGE14_5_DIR / 'raw_candidate_repetitions.csv'))
display(FileLink(STAGE14_5_DIR / 'raw_candidate_repetitions.jsonl'))
display(FileLink(STAGE14_5_DIR / 'candidate_sweep_summaries.csv'))
display(FileLink(STAGE14_5_DIR / 'stage14_5_validation_report.json'))"""
            ),
            nbformat.v4.new_markdown_cell(INTERPRETATION),
        ]
        ids = (
            "stage14-5-heading", "stage14-5-summary", "stage14-5-table",
            "stage14-5-figure-guide", "stage14-5-figures", "stage14-5-interpretation",
        )
        for cell, cell_id in zip(cells, ids):
            cell.id = cell_id
        notebook.cells[insert_at:insert_at] = cells
    else:
        by_id["stage14-5-table"].source = TABLE_SOURCE
        by_id["stage14-5-figure-guide"].source = FIGURE_GUIDE
        by_id["stage14-5-interpretation"].source = INTERPRETATION

    notebook.cells[-1].source = notebook.cells[-1].source.replace(
        "Stage 14.5 starts only after a separate review and GO decision.",
        "Stage 14.6 starts only after a separate review and GO decision.",
    )
    nbformat.write(notebook, NOTEBOOK)
    return NOTEBOOK


if __name__ == "__main__":
    print(integrate())

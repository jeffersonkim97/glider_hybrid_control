"""Idempotently append the approved Stage 14.6 static report."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "3D_Stage14_Computation_Load.ipynb"


FIGURE_GUIDE = """### Stage 14.6 neighborhood-radius figure guide

The common x-axis is `r_neighbor`, measured in Defender grid-index lengths.
The complete six-action Defender grid, physical sensor coordinates, initial
action, Attacker discretization, terrain, physics, objectives, and numerical
tolerances remain fixed. The exact global solver appears only as the full-radius
reference endpoint.

Stages 14.3--14.6 use the same six-figure schema; only the controlled x-axis
changes:

1. total SSE runtime;
2. the common runtime-component decomposition for both solvers;
3. peak process-tree memory;
4. `J_A` and `J_D`;
5. selected Defender/Attacker equilibrium IDs;
6. selected trajectory identity and switching-point coordinates.

Evaluation counts, cache hits, local-search iterations, and neighborhood-width
records remain in raw/tabular validation artifacts, not in the primary
performance figures.

All figures are static PNGs. Legends are outside the axes at bottom center."""


TABLE_SOURCE = """stage14_6_rows = load_json(STAGE14_6_DIR / 'radius_sweep_summaries.json')
table_rows = []
for row in stage14_6_rows:
    components = row['runtime_statistics_s']
    med = lambda key: None if components.get(key) is None else round(components[key]['median'], 4)
    identity = row['solution_identities'][0] if row['solution_identities'] else (None,) * 5
    table_rows.append([
        row['algorithm_variant'], row['r_neighbor'],
        med('T_SSE_s'), med('T_LOS_s'), med('T_graph_s'), med('T_hazard_s'), med('T_switch_s'), med('T_Bellman_s'),
        None if row['peak_rss_statistics_bytes'] is None else round(row['peak_rss_statistics_bytes']['median'] / 1024**2, 2),
        None if row['J_A_statistics'] is None else row['J_A_statistics']['median'],
        None if row['J_D_statistics'] is None else row['J_D_statistics']['median'],
        identity[0], identity[1], None if identity[4] is None else identity[4][:10],
    ])
display(HTML(
    '<table><tr><th>variant</th><th>r_neighbor</th><th>T_SSE</th><th>T_LOS</th>'
    '<th>T_graph</th><th>T_hazard</th><th>T_switch</th><th>T_Bellman</th>'
    '<th>peak RSS [MiB]</th><th>J_A</th><th>J_D</th><th>D ID</th><th>A ID</th><th>trajectory</th></tr>'
    + ''.join('<tr>' + ''.join(f'<td>{value}</td>' for value in values) + '</tr>' for values in table_rows)
    + '</table><style>th,td{border:1px solid #bbb;padding:4px 7px}</style>'
))"""


INTERPRETATION = """### Stage 14.6 interpretation boundary

- This is a one-variable local-neighborhood-radius experiment; it is not a
  Defender-action-count sweep.
- Runtime is interpreted using realized unique exact BR evaluations and measured
  overhead. Increasing radius need not imply monotonic runtime because search
  iteration count can change.
- `r_neighbor=8=N_D-1` covers the complete fixed action grid and must match the
  unchanged exact global finite-SSE result within frozen tolerances.
- Smaller-radius results certify local SSE only and are initialization- and
  neighborhood-dependent; they are not global-equilibrium claims.
- No Stage 14.7 work is started here."""


def integrate() -> Path:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    by_id = {cell.get("id"): cell for cell in notebook.cells}
    notebook.cells[0].source = notebook.cells[0].source.replace(
        "14.4, and 14.5 are integrated here",
        "14.4, 14.5, and 14.6 are integrated here",
    )
    imports = notebook.cells[1].source
    if "run_stage14_6_diagnostics" not in imports:
        imports = imports.replace(
            "from stage14_5_diagnostics import run_stage14_5_diagnostics",
            "from stage14_5_diagnostics import run_stage14_5_diagnostics\n"
            "from stage14_6_diagnostics import run_stage14_6_diagnostics",
        )
    if "STAGE14_6_DIR" not in imports:
        imports = imports.replace(
            "STAGE14_5_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_5_switching_candidates'",
            "STAGE14_5_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_5_switching_candidates'\n"
            "STAGE14_6_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_6_neighbor_radius'",
        )
    notebook.cells[1].source = imports
    if "RECOMPUTE_STAGE14_6" not in notebook.cells[3].source:
        notebook.cells[3].source = notebook.cells[3].source.replace(
            "RECOMPUTE_STAGE14_5 = False",
            "RECOMPUTE_STAGE14_5 = False\nRECOMPUTE_STAGE14_6 = False",
        )

    if "stage14-6-heading" not in by_id:
        insert_at = next(
            index for index, cell in enumerate(notebook.cells)
            if cell.get("id") == "future-substages"
        )
        cells = [
            nbformat.v4.new_markdown_cell(
                """## Stage 14.6 — one-variable local-neighborhood radius sweep

This section holds a deterministic six-action Defender grid fixed and varies
only `r_neighbor` over `{1, 2, 4, 8}`. Action ID 0 is the fixed start, and
`r_neighbor=8` is the complete-grid/global-reference endpoint."""
            ),
            nbformat.v4.new_code_cell(
                """if RECOMPUTE_STAGE14_6:
    stage14_6_summary = run_stage14_6_diagnostics(STAGE14_6_DIR)
else:
    summary_path = STAGE14_6_DIR / 'stage14_6_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError('Stage 14.6 artifact is absent; set RECOMPUTE_STAGE14_6=True.')
    stage14_6_summary = load_json(summary_path)

stage14_6_validation = load_json(STAGE14_6_DIR / 'stage14_6_validation_report.json')
display_record({
    'gate_passed': stage14_6_summary['gate_passed'],
    'r_neighbor_values': stage14_6_summary['r_neighbor_values'],
    'fixed_defender_action_count': stage14_6_summary['defender_action_count'],
    'fixed_defender_x_map': stage14_6_summary['defender_x_map'],
    'full_radius': stage14_6_summary['full_radius'],
    'full_radius_equals_global': stage14_6_summary['full_radius_global_equivalence_passed'],
    'total_repetitions': stage14_6_summary['total_repetition_count'],
    'next_stage_started': stage14_6_summary['next_stage_started'],
})"""
            ),
            nbformat.v4.new_code_cell(
                """stage14_6_rows = load_json(STAGE14_6_DIR / 'radius_sweep_summaries.json')
table_rows = []
for row in stage14_6_rows:
    timing = row['runtime_statistics_s']
    median_t = None if timing.get('T_SSE_s') is None else round(timing['T_SSE_s']['median'], 4)
    metric = lambda key: None if row.get(key) is None else round(row[key]['median'], 4)
    table_rows.append([
        row['algorithm_variant'], row['r_neighbor'], median_t,
        metric('summed_br_statistics_s'), metric('shared_graph_statistics_s'),
        metric('local_overhead_statistics_s'), metric('unique_evaluation_statistics'),
        metric('raw_request_statistics'), metric('cache_hit_statistics'),
        metric('iteration_statistics'), row['selected_defender_action_ids'][0],
        None if row['J_D_statistics'] is None else row['J_D_statistics']['median'],
        row['all_completed_exactness_checks_pass'], row['all_completed_replays_pass'],
    ])
display(HTML(
    '<table><tr><th>variant</th><th>r_neighbor</th><th>T_total</th><th>ΣT_BR</th>'
    '<th>T_shared</th><th>T_overhead</th><th>unique eval</th><th>raw requests</th>'
    '<th>cache hits</th><th>K</th><th>selected d</th><th>J_D</th><th>exact</th><th>replay</th></tr>'
    + ''.join('<tr>' + ''.join(f'<td>{value}</td>' for value in row) + '</tr>' for row in table_rows)
    + '</table><style>th,td{border:1px solid #bbb;padding:4px 7px}</style>'
))"""
            ),
            nbformat.v4.new_markdown_cell(FIGURE_GUIDE),
            nbformat.v4.new_code_cell(
                """display(HTML('<h4>Static Stage 14.6 figures</h4>'))
for figure_name in stage14_6_summary['figure_files']:
    display(HTML(f'<h5>{figure_name}</h5>'))
    display(Image(filename=str(STAGE14_6_DIR / figure_name)))

display(FileLink(STAGE14_6_DIR / 'raw_radius_repetitions.csv'))
display(FileLink(STAGE14_6_DIR / 'raw_radius_repetitions.jsonl'))
display(FileLink(STAGE14_6_DIR / 'radius_sweep_summaries.json'))
display(FileLink(STAGE14_6_DIR / 'full_radius_global_equivalence.json'))
display(FileLink(STAGE14_6_DIR / 'stage14_6_validation_report.json'))"""
            ),
            nbformat.v4.new_markdown_cell(INTERPRETATION),
        ]
        ids = (
            "stage14-6-heading", "stage14-6-summary", "stage14-6-table",
            "stage14-6-figure-guide", "stage14-6-figures", "stage14-6-interpretation",
        )
        for cell, cell_id in zip(cells, ids):
            cell.id = cell_id
        notebook.cells[insert_at:insert_at] = cells
    else:
        by_id["stage14-6-figure-guide"].source = FIGURE_GUIDE
        by_id["stage14-6-interpretation"].source = INTERPRETATION

    next(
        cell for cell in notebook.cells if cell.get("id") == "stage14-6-table"
    ).source = TABLE_SOURCE

    notebook.cells[-1].source = notebook.cells[-1].source.replace(
        "Stage 14.6 starts only after a separate review and GO decision.",
        "Stage 14.7 starts only after a separate review and GO decision.",
    )
    nbformat.write(notebook, NOTEBOOK)
    return NOTEBOOK


if __name__ == "__main__":
    print(integrate())

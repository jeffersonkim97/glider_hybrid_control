"""Idempotently append the approved Stage 14.3 static report."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "3D_Stage14_Computation_Load.ipynb"


FIGURE_GUIDE = """### Stage 14.3 shared resolution-sweep figure guide

The spatial grid scale `Delta x = Delta y = Delta h` is the controlled x-axis.
Stages 14.3--14.6 use the same six-figure performance/solution schema; only the
controlled x-axis changes. Certified local SSE is solid and the completed exact
global finite solver is a dashed oracle/reference.

1. **Total SSE runtime** (`T_SSE`).
2. **Runtime decomposition** using the same LOS, graph/reachability, hazard,
   switching, Bellman, and residual components for both solver variants.
3. **Peak process-tree memory** (RSS).
4. **Equilibrium objectives** (`J_A`, `J_D`).
5. **Selected finite equilibrium** (Defender action and Attacker candidate).
6. **Selected trajectory identity** (exact hash) and switching-point coordinates.

State/edge growth is retained only in raw/tabular audit artifacts; it is not a
primary performance figure.

`Individual isolated-process runs (n=3)` means a newly launched Python worker
for each repetition, not guaranteed-cold operating-system or hardware caches.
Any infeasible, timeout, memory-limit, or worker-failure result remains explicit
in the table/validation artifacts; missing points are not relabeled as successful
measurements."""


TABLE_SOURCE = """stage14_3_rows = load_json(STAGE14_3_DIR / 'spatial_sweep_summaries.json')
table_rows = []
for row in stage14_3_rows:
    components = row['runtime_component_statistics_s']
    med = lambda key: None if components.get(key) is None else round(components[key]['median'], 4)
    identity = row['solution_identities'][0] if row['solution_identities'] else (None,) * 5
    table_rows.append([
        row['algorithm_variant'], row['resolution_m'],
        None if row['T_SSE_statistics_s'] is None else round(row['T_SSE_statistics_s']['median'], 4),
        med('T_LOS_s'), med('T_graph_s'), med('T_hazard_s'), med('T_switch_s'), med('T_Bellman_s'),
        None if row['peak_rss_statistics_bytes'] is None else round(row['peak_rss_statistics_bytes']['median'] / 1024**2, 2),
        None if row['J_A_statistics'] is None else row['J_A_statistics']['median'],
        None if row['J_D_statistics'] is None else row['J_D_statistics']['median'],
        identity[0], identity[1], None if identity[4] is None else identity[4][:10],
    ])
display(HTML(
    '<table><tr><th>variant</th><th>Delta xyz [m]</th><th>T_SSE</th><th>T_LOS</th>'
    '<th>T_graph</th><th>T_hazard</th><th>T_switch</th><th>T_Bellman</th>'
    '<th>peak RSS [MiB]</th><th>J_A</th><th>J_D</th><th>D ID</th><th>A ID</th><th>trajectory</th></tr>'
    + ''.join('<tr>' + ''.join(f'<td>{value}</td>' for value in values) + '</tr>' for values in table_rows)
    + '</table><style>th,td{border:1px solid #bbb;padding:4px 7px}</style>'
))"""


def integrate() -> Path:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    by_id = {cell.get("id"): cell for cell in notebook.cells}
    if "stage14-3-heading" in by_id:
        notebook.cells[2].source = notebook.cells[2].source.replace(
            "coarse/baseline cold-process rows",
            "coarse/baseline isolated-process rows",
        )
        guide = by_id.get("stage14-3-figure-guide")
        if guide is None:
            figure_index = next(
                index for index, cell in enumerate(notebook.cells)
                if cell.get("id") == "stage14-3-figures"
            )
            guide = nbformat.v4.new_markdown_cell(FIGURE_GUIDE)
            guide.id = "stage14-3-figure-guide"
            notebook.cells.insert(figure_index, guide)
        else:
            guide.source = FIGURE_GUIDE
        by_id["stage14-3-table"].source = TABLE_SOURCE
        interpretation = by_id["stage14-3-interpretation"]
        warning = (
            "\n- The 20 m Cartesian grid is larger than the 25 m grid, but its "
            "reachable/active graph is much smaller; runtime and objective trends "
            "are therefore non-monotonic and do not establish convergence."
        )
        if warning.strip() not in interpretation.source:
            interpretation.source = interpretation.source.rstrip() + warning
        nbformat.write(notebook, NOTEBOOK)
        return NOTEBOOK
    notebook.cells[0].source = notebook.cells[0].source.replace(
        "and 14.2C are integrated here",
        "14.2C, and 14.3 are integrated here",
    )
    imports = notebook.cells[1].source
    imports = imports.replace(
        "from stage14_2c_diagnostics import run_stage14_2c_diagnostics",
        "from stage14_2c_diagnostics import run_stage14_2c_diagnostics\n"
        "from stage14_3_diagnostics import run_stage14_3_diagnostics",
    )
    imports = imports.replace(
        "STAGE14_2C_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_2c_local_scaling'",
        "STAGE14_2C_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_2c_local_scaling'\n"
        "STAGE14_3_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_3_spatial_resolution'",
    )
    notebook.cells[1].source = imports
    notebook.cells[2].source += (
        " Stage 14.3 reuses identical 14.2C coarse/baseline isolated-process rows "
        "and runs only missing progressively finer spatial cases."
    )
    notebook.cells[3].source = notebook.cells[3].source.replace(
        "RECOMPUTE_STAGE14_2C = False",
        "RECOMPUTE_STAGE14_2C = False\nRECOMPUTE_STAGE14_3 = False",
    )
    insert_at = next(index for index, cell in enumerate(notebook.cells) if cell.id == "future-substages")
    cells = [
        nbformat.v4.new_markdown_cell(
            """## Stage 14.3 — one-variable spatial-resolution sweep

This section varies only `Delta x = Delta y = Delta h` over the 20 exact
isotropic spacings `100/n` m for `n=1,...,20` (5--100 m). The centered-cube
mission, 5-degree heading grid, switching density, quadrature, Defender set,
and exact tie rules remain fixed. The preserved global finite-SSE solver is
shown only as an exact oracle/reference path. Local-SSE infeasibility at finer
single-start cases remains explicitly recorded rather than relabeled."""
        ),
        nbformat.v4.new_code_cell(
            """if RECOMPUTE_STAGE14_3:
    stage14_3_summary = run_stage14_3_diagnostics(STAGE14_3_DIR)
else:
    summary_path = STAGE14_3_DIR / 'stage14_3_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError('Stage 14.3 artifact is absent; set RECOMPUTE_STAGE14_3=True.')
    stage14_3_summary = load_json(summary_path)

stage14_3_validation = load_json(STAGE14_3_DIR / 'stage14_3_validation_report.json')
display_record({
    'gate_passed': stage14_3_summary['gate_passed'],
    'resolutions_m': stage14_3_summary['resolutions_m'],
    'reused_repetitions': stage14_3_summary['reused_repetition_count'],
    'new_repetitions': stage14_3_summary['new_repetition_count'],
    'total_repetitions': stage14_3_summary['total_repetition_count'],
    'finer_outcome': stage14_3_summary['finer_resolution_outcome'],
    'next_stage_started': stage14_3_summary['next_stage_started'],
})"""
        ),
        nbformat.v4.new_code_cell(TABLE_SOURCE),
        nbformat.v4.new_markdown_cell(FIGURE_GUIDE),
        nbformat.v4.new_code_cell(
            """display(HTML('<h4>Static Stage 14.3 figures</h4>'))
for figure_name in stage14_3_summary['figure_files']:
    display(HTML(f'<h5>{figure_name}</h5>'))
    display(Image(filename=str(STAGE14_3_DIR / figure_name)))

display(FileLink(STAGE14_3_DIR / 'raw_spatial_repetitions.csv'))
display(FileLink(STAGE14_3_DIR / 'raw_spatial_repetitions.jsonl'))
display(FileLink(STAGE14_3_DIR / 'stage14_3_validation_report.json'))"""
        ),
        nbformat.v4.new_markdown_cell(
            """### Stage 14.3 interpretation boundary

- Completed exact global-SSE results are oracle/reference measurements, not a
  reversal of the Local-SSE research pivot.
- The 20 m oracle completed; finer Local-SSE `D0` failures are model
  infeasibility, not runtime limits.
- The 12.5 m oracle timing-accounting failure is retained as an explicit
  worker failure and is not plotted as a completed measurement.
- The 20 m Cartesian grid is larger than the 25 m grid, but its
  reachable/active graph is much smaller; runtime and objective trends are
  therefore non-monotonic and do not establish convergence.
- No empirical curve is presented as formal asymptotic Big-O."""
        ),
    ]
    ids = (
        "stage14-3-heading", "stage14-3-summary", "stage14-3-table",
        "stage14-3-figure-guide", "stage14-3-figures", "stage14-3-interpretation",
    )
    for cell, cell_id in zip(cells, ids):
        cell.id = cell_id
    notebook.cells[insert_at:insert_at] = cells
    notebook.cells[-1].source = notebook.cells[-1].source.replace(
        "Any post-14.2C extension starts only after a separate review and GO decision.",
        "Stage 14.4 starts only after a separate review and GO decision.",
    )
    nbformat.write(notebook, NOTEBOOK)
    return NOTEBOOK


if __name__ == "__main__":
    print(integrate())

"""Idempotently append the approved Stage 14.4 static report."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "3D_Stage14_Computation_Load.ipynb"


FIGURE_GUIDE = """### Stage 14.4 shared resolution-sweep figure guide

The requested heading resolution `Delta psi [deg]` is the controlled x-axis.
Realized `N_psi` remains in the summary table and machine-readable metadata
for reproducibility, but is not repeated on the plots because
`N_psi = 360 deg / realized Delta psi`. Stages 14.3--14.6 use the same
six-figure performance/solution schema; only the controlled x-axis changes.
Certified local SSE is solid and the exact global finite solver is the dashed
oracle/reference.

1. **Total SSE runtime** (`T_SSE`).
2. **Runtime decomposition** using the common component schema for both solvers.
3. **Peak process-tree memory** (RSS).
4. **Equilibrium objectives** (`J_A`, `J_D`).
5. **Selected finite equilibrium** (Defender action and Attacker candidate).
6. **Selected trajectory identity** and switching-point coordinates.

Heading-bin/quantization and state/edge diagnostics remain in machine-readable
audit artifacts, not in the primary performance figures.

`Individual isolated-process runs (n=3)` means a fresh Python worker for every
repetition, not guaranteed-cold operating-system or hardware caches. The 15,
10, and 5 degree rows reuse identical Stage-14.2C measurements; all other
declared spacings are Stage-14.4 measurements."""


TABLE_SOURCE = """stage14_4_rows = load_json(STAGE14_4_DIR / 'heading_sweep_summaries.json')
table_rows = []
for row in stage14_4_rows:
    components = row['runtime_component_statistics_s']
    med = lambda key: None if components.get(key) is None else round(components[key]['median'], 4)
    identity = row['solution_identities'][0] if row['solution_identities'] else (None,) * 5
    table_rows.append([
        row['algorithm_variant'], row['heading_spacing_deg'],
        None if row['T_SSE_statistics_s'] is None else round(row['T_SSE_statistics_s']['median'], 4),
        med('T_LOS_s'), med('T_graph_s'), med('T_hazard_s'), med('T_switch_s'), med('T_Bellman_s'),
        None if row['peak_rss_statistics_bytes'] is None else round(row['peak_rss_statistics_bytes']['median'] / 1024**2, 2),
        None if row['J_A_statistics'] is None else row['J_A_statistics']['median'],
        None if row['J_D_statistics'] is None else row['J_D_statistics']['median'],
        identity[0], identity[1], None if identity[4] is None else identity[4][:10],
    ])
display(HTML(
    '<table><tr><th>variant</th><th>Delta psi [deg]</th><th>T_SSE</th><th>T_LOS</th>'
    '<th>T_graph</th><th>T_hazard</th><th>T_switch</th><th>T_Bellman</th>'
    '<th>peak RSS [MiB]</th><th>J_A</th><th>J_D</th><th>D ID</th><th>A ID</th><th>trajectory</th></tr>'
    + ''.join('<tr>' + ''.join(f'<td>{value}</td>' for value in values) + '</tr>' for values in table_rows)
    + '</table><style>th,td{border:1px solid #bbb;padding:4px 7px}</style>'
))"""


INTERPRETATION = """### Stage 14.4 interpretation boundary

- The primary plotted algorithm is the certified local SSE with exact finite
  Attacker responses. The frozen global solver remains a tractable oracle only.
- Completed repetitions are reported together with explicit fine-grid
  computational limits under the declared time and memory limits.
- Twenty requested heading spacings from 1 to 45 degrees are declared. Runtime
  behavior is measured only for completed points and is not a formal Big-O claim.
- Local and global outcomes coincide for this three-action Defender grid at all
  measured heading resolutions. This does not redefine local SSE as global SSE
  for other grids or initializations.
- The objective curves remain discretization-dependent. Twenty requested points
  improve resolution coverage but do not by themselves prove convergence.
- No Stage 14.5 work is started here."""


def integrate() -> Path:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    by_id = {cell.get("id"): cell for cell in notebook.cells}
    notebook.cells[0].source = notebook.cells[0].source.replace(
        "14.2C, and 14.3 are integrated here",
        "14.2C, 14.3, and 14.4 are integrated here",
    )
    imports = notebook.cells[1].source
    if "run_stage14_4_diagnostics" not in imports:
        imports = imports.replace(
            "from stage14_3_diagnostics import run_stage14_3_diagnostics",
            "from stage14_3_diagnostics import run_stage14_3_diagnostics\n"
            "from stage14_4_diagnostics import run_stage14_4_diagnostics",
        )
    if "STAGE14_4_DIR" not in imports:
        imports = imports.replace(
            "STAGE14_3_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_3_spatial_resolution'",
            "STAGE14_3_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_3_spatial_resolution'\n"
            "STAGE14_4_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_4_heading_resolution'",
        )
    notebook.cells[1].source = imports
    if "RECOMPUTE_STAGE14_4" not in notebook.cells[3].source:
        notebook.cells[3].source = notebook.cells[3].source.replace(
            "RECOMPUTE_STAGE14_3 = False",
            "RECOMPUTE_STAGE14_3 = False\nRECOMPUTE_STAGE14_4 = False",
        )

    if "stage14-4-heading" not in by_id:
        insert_at = next(
            index for index, cell in enumerate(notebook.cells)
            if cell.get("id") == "future-substages"
        )
        cells = [
            nbformat.v4.new_markdown_cell(
                """## Stage 14.4 — one-variable heading-resolution sweep

This section varies only requested heading spacing over 20 declared values from
1 through 45 degrees. Spatial spacing remains 25 m; terrain, switching density, quadrature,
Defender actions, local-neighborhood radius, physics, objectives, and tie rules
remain fixed. The local SSE is primary and every evaluated Attacker response is
exact over the corresponding finite model."""
            ),
            nbformat.v4.new_code_cell(
                """if RECOMPUTE_STAGE14_4:
    stage14_4_summary = run_stage14_4_diagnostics(STAGE14_4_DIR)
else:
    summary_path = STAGE14_4_DIR / 'stage14_4_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError('Stage 14.4 artifact is absent; set RECOMPUTE_STAGE14_4=True.')
    stage14_4_summary = load_json(summary_path)

stage14_4_validation = load_json(STAGE14_4_DIR / 'stage14_4_validation_report.json')
display_record({
    'gate_passed': stage14_4_summary['gate_passed'],
    'heading_spacings_deg': stage14_4_summary['heading_spacings_deg'],
    'reused_repetitions': stage14_4_summary['reused_repetition_count'],
    'new_repetitions': stage14_4_summary['new_repetition_count'],
    'total_repetitions': stage14_4_summary['total_repetition_count'],
    'fine_heading_outcome': stage14_4_summary['fine_heading_outcome'],
    'next_stage_started': stage14_4_summary['next_stage_started'],
})"""
            ),
            nbformat.v4.new_code_cell(TABLE_SOURCE),
            nbformat.v4.new_markdown_cell(FIGURE_GUIDE),
            nbformat.v4.new_code_cell(
                """display(HTML('<h4>Static Stage 14.4 figures</h4>'))
for figure_name in stage14_4_summary['figure_files']:
    display(HTML(f'<h5>{figure_name}</h5>'))
    display(Image(filename=str(STAGE14_4_DIR / figure_name)))

display(FileLink(STAGE14_4_DIR / 'raw_heading_repetitions.csv'))
display(FileLink(STAGE14_4_DIR / 'raw_heading_repetitions.jsonl'))
display(FileLink(STAGE14_4_DIR / 'stage14_4_validation_report.json'))"""
            ),
            nbformat.v4.new_markdown_cell(INTERPRETATION),
        ]
        ids = (
            "stage14-4-heading", "stage14-4-summary", "stage14-4-table",
            "stage14-4-figure-guide", "stage14-4-figures", "stage14-4-interpretation",
        )
        for cell, cell_id in zip(cells, ids):
            cell.id = cell_id
        notebook.cells[insert_at:insert_at] = cells
    else:
        by_id["stage14-4-table"].source = TABLE_SOURCE
        by_id["stage14-4-figure-guide"].source = FIGURE_GUIDE
        by_id["stage14-4-interpretation"].source = INTERPRETATION

    notebook.cells[-1].source = notebook.cells[-1].source.replace(
        "Stage 14.4 starts only after a separate review and GO decision.",
        "Stage 14.5 starts only after a separate review and GO decision.",
    )
    nbformat.write(notebook, NOTEBOOK)
    return NOTEBOOK


if __name__ == "__main__":
    print(integrate())

"""Idempotently append the approved Stage 14.2B artifact view to its notebook."""

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
        "Stage 14.0, 14.1, 14.2, and 14.2A are integrated here",
        "Stage 14.0, 14.1, 14.2, 14.2A, and 14.2B are integrated here",
    )
    imports = notebook.cells[1].source
    if "Image" not in imports:
        imports = _replace_once(
            imports,
            "from IPython.display import FileLink, HTML, display",
            "from IPython.display import FileLink, HTML, Image, display",
        )
    imports = _replace_once(
        imports,
        "from stage14_2a_local_sse_diagnostics import run_stage14_2a_diagnostics",
        "from stage14_2a_local_sse_diagnostics import run_stage14_2a_diagnostics\n"
        "from stage14_2b_diagnostics import run_stage14_2b_diagnostics",
    )
    imports = _replace_once(
        imports,
        "STAGE14_2A_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_2a_local_sse_contract'",
        "STAGE14_2A_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_2a_local_sse_contract'\n"
        "STAGE14_2B_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_2b_local_sse_search'",
    )
    notebook.cells[1].source = imports
    notebook.cells[2].source = _replace_once(
        notebook.cells[2].source,
        "Stage 14.2A regenerates only the local-SSE contract diagnostics.",
        "Stage 14.2A regenerates only the local-SSE contract diagnostics. "
        "Stage 14.2B runs the canonical exact local search in a fresh measured process.",
    )
    notebook.cells[3].source = _replace_once(
        notebook.cells[3].source,
        "RECOMPUTE_STAGE14_2A = False",
        "RECOMPUTE_STAGE14_2A = False\nRECOMPUTE_STAGE14_2B = False",
    )

    marker = "stage14_2b_heading"
    if any(cell.metadata.get("stage14_id") == marker for cell in notebook.cells):
        nbformat.write(notebook, NOTEBOOK)
        return
    insert_at = next(
        index for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and cell.source.startswith("## Later Stage 14")
    )
    new_cells = [
        nbformat.v4.new_markdown_cell(
            """## Stage 14.2B — exact local-SSE search and certification

Starting from canonical Defender action `D0`, this section displays the saved
single-start, strict-improvement local search with `r_neighbor=1`. Every action
that appears with a payoff was evaluated using the existing exhaustive finite
Attacker best response and Strong follower tie-break. This is a **local finite-grid
SSE certificate**, not a global or continuous-space optimality claim.""",
            metadata={"stage14_id": marker},
        ),
        nbformat.v4.new_code_cell(
            """if RECOMPUTE_STAGE14_2B:
    stage14_2b_summary = run_stage14_2b_diagnostics(STAGE14_2B_DIR, recompute=True)
else:
    summary_path = STAGE14_2B_DIR / 'stage14_2b_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError('Stage 14.2B artifact is absent; set RECOMPUTE_STAGE14_2B=True.')
    stage14_2b_summary = load_json(summary_path)

display_record({
    'gate_passed': stage14_2b_summary['gate_passed'],
    'termination': stage14_2b_summary['termination_status'],
    'scope': stage14_2b_summary['solution_scope'],
    'visited': stage14_2b_summary['visited_defender_actions'],
    'evaluated': stage14_2b_summary['evaluated_defender_actions'],
    'final_defender_action': stage14_2b_summary['final_local_sse_action_id'],
    'final_attacker_response': stage14_2b_summary['final_attacker_response_id'],
    'J_A': stage14_2b_summary['final_J_A'],
    'J_D_PoD': stage14_2b_summary['final_J_D'],
    'local_margin': stage14_2b_summary['local_optimality_margin'],
    'T_local_s': stage14_2b_summary['runtime_decomposition_s']['T_local_s'],
    'peak_RSS_MiB': stage14_2b_summary['peak_memory']['peak_rss_bytes'] / 1024**2,
    'global_optimality_evaluated': stage14_2b_summary['global_optimality_evaluated'],
})""",
            metadata={"stage14_id": "stage14_2b_summary"},
        ),
        nbformat.v4.new_code_cell(
            """stage14_2b_history = load_json(STAGE14_2B_DIR / 'local_search_history.json')
stage14_2b_result = load_json(STAGE14_2B_DIR / 'canonical_local_sse_result.json')
records = stage14_2b_history['exact_evaluation_records']
value_by_id = {int(row['action_id']): row['local_evaluation']['defender_value'] for row in records}
x_by_id = {int(row['action_id']): row['sensor_position_map'][0] for row in records}
evaluated = [int(value) for value in stage14_2b_result['evaluated_defender_actions']]
visited = [int(value) for value in stage14_2b_result['visited_defender_actions']]
final_id = int(stage14_2b_result['final_local_sse_action_id'])
final_neighbors = [int(value) for value in stage14_2b_result['iterations'][-1]['neighbor_action_ids']]

local_search_fig = go.Figure()
local_search_fig.add_trace(go.Scatter(
    x=[x_by_id[action] for action in evaluated], y=[value_by_id[action] for action in evaluated],
    mode='markers+text', text=[f'D{action}' for action in evaluated], textposition='top center',
    name='Actually evaluated', marker={'size': 11, 'color': '#4472C4'},
))
local_search_fig.add_trace(go.Scatter(
    x=[x_by_id[action] for action in visited], y=[value_by_id[action] for action in visited],
    mode='lines+markers', name='Visited sequence',
    line={'color': '#ED7D31', 'width': 3},
))
local_search_fig.add_trace(go.Scatter(
    x=[x_by_id[action] for action in final_neighbors], y=[value_by_id[action] for action in final_neighbors],
    mode='markers', name='Final required neighborhood',
    marker={'size': 18, 'symbol': 'circle-open', 'color': '#A5A5A5'},
))
local_search_fig.add_trace(go.Scatter(
    x=[x_by_id[final_id]], y=[value_by_id[final_id]], mode='markers', name='Certified local SSE',
    marker={'size': 20, 'symbol': 'star', 'color': '#70AD47'},
))
local_search_fig.update_layout(
    title='Stage 14.2B canonical local-SSE search — evaluated actions only',
    xaxis_title='Defender sensor x position [map units]',
    yaxis={'title': 'Induced Defender value V_D (PoD)', 'range': [0, 1]},
    legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02},
)
display(Image(filename=str(STAGE14_2B_DIR / 'local_search_diagnostic.png')))
display(FileLink(STAGE14_2B_DIR / 'local_search_history.json'))
display(FileLink(STAGE14_2B_DIR / 'runtime_memory_summary.json'))""",
            metadata={"stage14_id": "stage14_2b_visualization"},
        ),
    ]
    for cell, cell_id in zip(
        new_cells,
        ("stage14-2b-heading", "stage14-2b-summary", "stage14-2b-figure"),
    ):
        cell.id = cell_id
    notebook.cells[insert_at:insert_at] = new_cells
    notebook.cells[-1].source = notebook.cells[-1].source.replace(
        "No Stage 14.2 sweep or scaling claim is executed by the current notebook.",
        "Stage 14.2C or any later sweep starts only after a separate review and GO decision.",
    )
    nbformat.write(notebook, NOTEBOOK)


if __name__ == "__main__":
    integrate()

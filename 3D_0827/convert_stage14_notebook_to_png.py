"""Convert every Stage-14 notebook visualization to an inline static PNG."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "3D_Stage14_Computation_Load.ipynb"


STATIC_CELLS = {
    "profiling-figures": """display(Image(filename=str(STAGE14_1_DIR / 'canonical_runtime_decomposition.png')))
display(Image(filename=str(STAGE14_1_DIR / 'canonical_memory_summary.png')))""",
    "stage14-2-figure": """display(Image(filename=str(STAGE14_2_DIR / 'canonical_repetition_stability.png')))
display(FileLink(STAGE14_2_DIR / 'raw_repetitions.jsonl'))
display(FileLink(STAGE14_2_DIR / 'raw_repetitions.csv'))
display(FileLink(STAGE14_2_DIR / 'configuration_summaries.csv'))""",
    "stage14-2a-figure": """display(Image(filename=str(STAGE14_2A_DIR / 'local_vs_global_payoff_diagnostic.png')))
display(FileLink(STAGE14_2A_DIR / 'local_sse_mathematical_contract.json'))
display(FileLink(STAGE14_2A_DIR / 'current_discretized_existence_report.json'))""",
    "stage14-2b-figure": """display(Image(filename=str(STAGE14_2B_DIR / 'local_search_diagnostic.png')))
display(FileLink(STAGE14_2B_DIR / 'local_search_history.json'))
display(FileLink(STAGE14_2B_DIR / 'runtime_memory_summary.json'))""",
    "stage14-2c-figures": """stage14_2c_figures = stage14_2c_summary['figure_files']
display(HTML('<h4>Static Stage 14.2C figures</h4>'))
for figure_name in stage14_2c_figures:
    display(HTML(f'<h5>{figure_name}</h5>'))
    display(Image(filename=str(STAGE14_2C_DIR / figure_name)))

display(FileLink(STAGE14_2C_DIR / 'raw_repetitions.csv'))
display(FileLink(STAGE14_2C_DIR / 'raw_repetitions.jsonl'))
display(FileLink(STAGE14_2C_DIR / 'regression_validation_report.json'))""",
}


def convert_notebook(path: Path = NOTEBOOK) -> Path:
    notebook = nbformat.read(path, as_version=4)
    by_id = {cell.get("id"): cell for cell in notebook.cells}
    imports = by_id["imports-and-paths"].source
    imports = imports.replace("import plotly.graph_objects as go\n", "")
    imports = imports.replace(
        "from IPython.display import FileLink, HTML, IFrame, display",
        "from IPython.display import FileLink, HTML, Image, display",
    )
    by_id["imports-and-paths"].source = imports
    stage0 = by_id["stage14-0-control"].source
    static_line = "display(Image(filename=str(STAGE14_0_DIR / 'baseline_identity_and_state_counts.png')))"
    if static_line not in stage0:
        stage0 = stage0.rstrip() + "\n\n" + static_line
    by_id["stage14-0-control"].source = stage0
    for cell_id, source in STATIC_CELLS.items():
        by_id[cell_id].source = source
    nbformat.write(notebook, path)
    return path


if __name__ == "__main__":
    print(convert_notebook())

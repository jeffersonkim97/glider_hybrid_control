"""Idempotently add the Stage 14.8 final-gate report to the Stage-14 notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "3D_Stage14_Computation_Load.ipynb"


def integrate() -> Path:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    by_id = {cell.get("id"): cell for cell in notebook.cells}
    notebook.cells[0].source = notebook.cells[0].source.replace(
        "14.5, 14.6, and 14.7 are integrated here",
        "14.5, 14.6, 14.7, and 14.8 are integrated here",
    )
    imports = notebook.cells[1].source
    if "STAGE14_8_DIR" not in imports:
        imports = imports.replace(
            "STAGE14_7_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_7_scaling_analysis'",
            "STAGE14_7_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_7_scaling_analysis'\n"
            "STAGE14_8_DIR = NOTEBOOK_DIR / 'figure' / 'stage_14_8_final_gate'",
        )
    notebook.cells[1].source = imports

    if "stage14-8-heading" not in by_id:
        insert_at = next(
            index for index, cell in enumerate(notebook.cells)
            if cell.get("id") == "future-substages"
        )
        cells = [
            nbformat.v4.new_markdown_cell(
                """## Stage 14.8 — final integration and regression gate

This is a report-only notebook section. The external Stage-14.8 harness ran
the single canonical fresh-process smoke, all tests, artifact regeneration,
and this notebook execution. It did not rerun the Stage-14.3–14.6 sweeps and
did not start work beyond Stage 14."""
            ),
            nbformat.v4.new_code_cell(
                """stage14_8_summary_path = STAGE14_8_DIR / 'stage14_8_summary.json'
if not stage14_8_summary_path.exists():
    raise FileNotFoundError(
        'Stage 14.8 report is absent. Run stage14_8_final_gate.py outside the notebook.'
    )
stage14_8_summary = load_json(stage14_8_summary_path)
display_record({
    'gate_passed': stage14_8_summary['gate_passed'],
    'test_count': stage14_8_summary['test_count'],
    'figure_count': stage14_8_summary['figure_count'],
    'raw_repetition_count': stage14_8_summary['raw_repetition_count'],
    'completed_repetitions': stage14_8_summary['completed_repetition_count'],
    'noncompleted_repetitions': stage14_8_summary['noncompleted_repetition_count'],
    'canonical_clean_smoke_recorded': stage14_8_summary['canonical_clean_smoke_recorded'],
    'full_scaling_sweeps_rerun': stage14_8_summary['full_scaling_sweeps_rerun'],
    'next_stage_started': stage14_8_summary['next_stage_started'],
})
assert stage14_8_summary['gate_passed']
assert stage14_8_summary['full_scaling_sweeps_rerun'] is False
assert stage14_8_summary['next_stage_started'] is False"""
            ),
            nbformat.v4.new_code_cell(
                """stage14_8_validation = load_json(STAGE14_8_DIR / 'stage14_8_validation_report.json')
check_rows = [[name, passed] for name, passed in stage14_8_validation['checks'].items()]
display(HTML(
    '<table><tr><th>Final integration check</th><th>Passed</th></tr>'
    + ''.join('<tr>' + ''.join(f'<td>{value}</td>' for value in row) + '</tr>' for row in check_rows)
    + '</table><style>th,td{border:1px solid #bbb;padding:4px 7px}</style>'
))
assert stage14_8_validation['gate_passed']
assert all(stage14_8_validation['stage14_pass_checklist'].values())"""
            ),
            nbformat.v4.new_code_cell(
                """display(FileLink(STAGE14_8_DIR / 'README.md'))
display(FileLink(STAGE14_8_DIR / 'all_raw_repetitions.csv'))
display(FileLink(STAGE14_8_DIR / 'all_raw_repetitions.jsonl'))
display(FileLink(STAGE14_8_DIR / 'all_configuration_summaries.csv'))
display(FileLink(STAGE14_8_DIR / 'frozen_benchmark_configuration_set.json'))
display(FileLink(STAGE14_8_DIR / 'fit_parameters.csv'))
display(FileLink(STAGE14_8_DIR / 'figure_manifest.json'))
display(FileLink(STAGE14_8_DIR / 'test_validation_report.json'))
display(FileLink(STAGE14_8_DIR / 'practical_boundary_observations.md'))
display(FileLink(STAGE14_8_DIR / 'stage14_8_validation_report.json'))"""
            ),
            nbformat.v4.new_markdown_cell(
                """### Stage 14 final claim boundary

The certified result is an exact Strong Stackelberg equilibrium of the
configured finite Defender-action, switching-candidate, and Bellman-lattice
model. The empirical scaling observations are limited to measured
configurations. This gate makes no continuous-space equilibrium claim and no
unmeasured-resolution extrapolation. Work beyond Stage 14 remains stopped."""
            ),
        ]
        ids = (
            "stage14-8-heading", "stage14-8-summary", "stage14-8-checks",
            "stage14-8-artifacts", "stage14-8-boundary",
        )
        for cell, cell_id in zip(cells, ids):
            cell.id = cell_id
        notebook.cells[insert_at:insert_at] = cells
    else:
        by_id["stage14-8-summary"].source = by_id["stage14-8-summary"].source.replace(
            "'canonical_smoke_attempted_now': stage14_8_summary['canonical_smoke_attempted_now']",
            "'canonical_clean_smoke_recorded': stage14_8_summary['canonical_clean_smoke_recorded']",
        )

    future = next(cell for cell in notebook.cells if cell.get("id") == "future-substages")
    future.source = """## Beyond Stage 14 — not started

The practical tractability-boundary study and any exact-SSE-versus-RL
comparison require a separate updated plan and explicit user approval."""
    nbformat.write(notebook, NOTEBOOK)
    return NOTEBOOK


if __name__ == "__main__":
    print(integrate())

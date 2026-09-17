"""Execute the Stage-11 notebook top-to-bottom and persist its outputs."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "3D_Attacker_Bellman_Validated.ipynb"


def execute_notebook() -> Path:
    with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    previous_headless = os.environ.get("BELLMAN_GUI_HEADLESS")
    os.environ["BELLMAN_GUI_HEADLESS"] = "1"
    try:
        client.execute(cwd=str(ROOT))
    finally:
        if previous_headless is None:
            os.environ.pop("BELLMAN_GUI_HEADLESS", None)
        else:
            os.environ["BELLMAN_GUI_HEADLESS"] = previous_headless
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return NOTEBOOK_PATH


if __name__ == "__main__":
    print(execute_notebook())

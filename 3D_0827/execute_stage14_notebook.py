"""Execute the integrated Stage-14 notebook top-to-bottom and save outputs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "3D_Stage14_Computation_Load.ipynb"


def execute_notebook() -> Path:
    with tempfile.TemporaryDirectory(prefix=".tmp_stage14_", dir=ROOT) as temporary_root:
        temporary_root_path = Path(temporary_root)
        jupyter_runtime = temporary_root_path / "jupyter_runtime"
        ipython_directory = temporary_root_path / "ipython"
        jupyter_runtime.mkdir()
        ipython_directory.mkdir()

        previous_jupyter_runtime = os.environ.get("JUPYTER_RUNTIME_DIR")
        previous_ipython_directory = os.environ.get("IPYTHONDIR")
        os.environ["JUPYTER_RUNTIME_DIR"] = str(jupyter_runtime)
        os.environ["IPYTHONDIR"] = str(ipython_directory)
        try:
            notebook = nbformat.read(NOTEBOOK, as_version=4)
            client = NotebookClient(
                notebook,
                timeout=300,
                kernel_name="venv_p1b",
                resources={"metadata": {"path": str(ROOT)}},
                allow_errors=False,
            )
            client.execute(cwd=str(ROOT))
            nbformat.write(notebook, NOTEBOOK)
        finally:
            if previous_jupyter_runtime is None:
                os.environ.pop("JUPYTER_RUNTIME_DIR", None)
            else:
                os.environ["JUPYTER_RUNTIME_DIR"] = previous_jupyter_runtime
            if previous_ipython_directory is None:
                os.environ.pop("IPYTHONDIR", None)
            else:
                os.environ["IPYTHONDIR"] = previous_ipython_directory
    return NOTEBOOK


if __name__ == "__main__":
    print(execute_notebook())

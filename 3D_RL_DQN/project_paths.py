"""Make the repository's core and DQN modules importable from either folder.

The historical directories start with digits, so they cannot be referenced with
normal package syntax such as ``from 3D_0827 import ...``.  Import this module once
at the top of a DQN notebook or script, then import the existing core modules by
their filenames:

    import project_paths  # noqa: F401
    from P1b_condition import ComputationCondition, build_scene
    from glider_gym_env import TerrainGlideEnv

The workspace virtual environment also contains a matching ``.pth`` file, so new
Python processes and restarted notebook kernels normally receive both paths
automatically.  This module keeps scripts portable when that environment setup is
not present.
"""

from __future__ import annotations

import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = WORKSPACE_ROOT / "3D_0827"
DQN_DIR = WORKSPACE_ROOT / "3D_RL_DQN"


def configure_import_paths() -> tuple[Path, Path]:
    """Add both source directories to ``sys.path`` and return them."""

    for directory in (CORE_DIR, DQN_DIR):
        if not directory.is_dir():
            raise RuntimeError(f"required project directory does not exist: {directory}")
        text = str(directory)
        if text not in sys.path:
            sys.path.insert(0, text)
    return CORE_DIR, DQN_DIR


configure_import_paths()


__all__ = [
    "CORE_DIR",
    "DQN_DIR",
    "WORKSPACE_ROOT",
    "configure_import_paths",
]

"""Centralized project paths for the Stackelberg security workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Immutable absolute paths used by every future project phase."""

    project_root: Path
    results_dir: Path
    json_dir: Path
    npz_dir: Path
    figure_dir: Path
    log_dir: Path

    def as_dict(self) -> dict[str, str]:
        """Return string path values for configuration and metadata."""
        return {
            "project_root": str(self.project_root),
            "results_dir": str(self.results_dir),
            "json_dir": str(self.json_dir),
            "npz_dir": str(self.npz_dir),
            "figure_dir": str(self.figure_dir),
            "log_dir": str(self.log_dir),
        }


def create_project_paths(project_root: Path | None = None) -> ProjectPaths:
    """Resolve the project root and create all standard output directories.

    Inputs
    ------
    project_root:
        Optional repository root. It is derived from this module when omitted.

    Outputs
    -------
    ProjectPaths
        Immutable absolute paths whose output directories exist.

    Assumptions
    -----------
    This module resides in the p1b_4D package under the repository root.

    Notes
    -----
    Directory creation is this function's primary and only side effect.
    """
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    results = root / "results"
    paths = ProjectPaths(
        project_root=root,
        results_dir=results,
        json_dir=results / "json",
        npz_dir=results / "npz",
        figure_dir=results / "figures",
        log_dir=results / "logs",
    )
    for directory in (
        paths.results_dir,
        paths.json_dir,
        paths.npz_dir,
        paths.figure_dir,
        paths.log_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths

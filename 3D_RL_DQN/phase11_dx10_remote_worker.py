"""Train and evaluate the delegated dx=10 m full-factorial block.

One command trains the missing dpsi={2.5, 10} checkpoints, reuses the tracked
dpsi=5 checkpoint, evaluates exactly the 14 non-overlapping (dpsi, r)
conditions on four unseen terrains, and creates a Git-LFS result bundle.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import project_paths
import phase11_full_factorial_grid as grid


REMOTE_OUTPUT_ROOT = (
    project_paths.DQN_DIR / "runs" / "phase11_dx10_remote"
)
ARTIFACT_DIR = (
    project_paths.DQN_DIR / "artifacts" / "phase11_dx10_remote"
)
ARTIFACT_PATH = ARTIFACT_DIR / "phase11_dx10_remote_results.zip"
MANIFEST_PATH = ARTIFACT_DIR / "phase11_dx10_remote_manifest.json"
EXPECTED_RESULT_COUNT = 14 * 4


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_remote_grid() -> None:
    grid.DX_VALUES = (10.0,)
    grid.DPSI_VALUES = (2.5, 5.0, 10.0)
    grid.R_VALUES = (1, 2, 3, 4, 5)
    grid.TERRAINS = (0, 1, 2, 3)
    grid.DX_ORDER = (10.0,)
    grid.DPSI_ORDER = (10.0, 5.0, 2.5)
    grid.R_ORDER = (5, 4, 3, 2, 1)
    grid.EXISTING_CONDITIONS = frozenset({(10.0, 5.0, 1)})
    grid.ALL_CONDITIONS = frozenset(
        (10.0, dpsi, radius)
        for dpsi in grid.DPSI_VALUES
        for radius in grid.R_VALUES
    )
    grid.MISSING_CONDITIONS = (
        grid.ALL_CONDITIONS - grid.EXISTING_CONDITIONS
    )
    grid.OUTPUT_ROOT = REMOTE_OUTPUT_ROOT
    grid.SHARED_CACHE_ROOT = REMOTE_OUTPUT_ROOT / "_shared_exact_cache"
    grid.STATUS_PATH = (
        REMOTE_OUTPUT_ROOT / "phase11_dx10_remote_status.json"
    )
    grid.STDOUT_PATH = (
        REMOTE_OUTPUT_ROOT / "phase11_dx10_remote_stdout.log"
    )
    grid.STDERR_PATH = (
        REMOTE_OUTPUT_ROOT / "phase11_dx10_remote_stderr.log"
    )


def _result_paths() -> list[Path]:
    return sorted(REMOTE_OUTPUT_ROOT.glob(
        "final_id_*/dx10m_dpsi*deg_r*/zero_sum_g2_local_search.json"
    ))


def package_results() -> Path:
    results = _result_paths()
    if len(results) != EXPECTED_RESULT_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_RESULT_COUNT} raw results, found {len(results)}"
        )
    status_path = REMOTE_OUTPUT_ROOT / "phase11_dx10_remote_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise RuntimeError("remote worker status is not completed")

    included = list(results)
    included.extend(
        path.with_suffix(".md")
        for path in results
        if path.with_suffix(".md").is_file()
    )
    included.append(status_path)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT_PATH.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
    ) as archive:
        for path in included:
            archive.write(path, path.relative_to(REMOTE_OUTPUT_ROOT).as_posix())
    temporary.replace(ARTIFACT_PATH)

    manifest = {
        "schema": "p1b-phase11-dx10-remote-artifact-v1",
        "artifact": ARTIFACT_PATH.relative_to(
            project_paths.WORKSPACE_ROOT
        ).as_posix(),
        "artifact_sha256": _sha256(ARTIFACT_PATH),
        "raw_result_count": len(results),
        "condition_count": 14,
        "terrain_count": 4,
        "delegated_conditions": [
            {"dx_m": 10.0, "dpsi_deg": dpsi, "r_neighbor": radius}
            for dpsi in grid.DPSI_VALUES
            for radius in grid.R_VALUES
            if (10.0, dpsi, radius) not in grid.EXISTING_CONDITIONS
        ],
        "files": [
            {
                "path": path.relative_to(REMOTE_OUTPUT_ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for path in results
        ],
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"Packaged: {ARTIFACT_PATH}", flush=True)
    return ARTIFACT_PATH


def import_artifact() -> Path:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_hash = str(manifest["artifact_sha256"])
    actual_hash = _sha256(ARTIFACT_PATH)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"artifact SHA-256 mismatch: {actual_hash} != {expected_hash}"
        )
    root = REMOTE_OUTPUT_ROOT.resolve()
    with zipfile.ZipFile(ARTIFACT_PATH, "r") as archive:
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root):
                raise RuntimeError(f"unsafe artifact member: {member.filename}")
        archive.extractall(root)
    results = _result_paths()
    if len(results) != int(manifest["raw_result_count"]):
        raise RuntimeError("imported raw-result count does not match manifest")
    for item in manifest["files"]:
        path = REMOTE_OUTPUT_ROOT / item["path"]
        if _sha256(path) != item["sha256"]:
            raise RuntimeError(f"imported result hash mismatch: {path}")
    print(f"Imported {len(results)} raw results into {REMOTE_OUTPUT_ROOT}")
    return REMOTE_OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--import-artifact", action="store_true")
    arguments = parser.parse_args()
    _configure_remote_grid()
    if arguments.plan:
        print(json.dumps(grid._plan(), indent=2))
    elif arguments.import_artifact:
        import_artifact()
    elif arguments.package_only:
        package_results()
    else:
        grid.run()
        package_results()


if __name__ == "__main__":
    main()

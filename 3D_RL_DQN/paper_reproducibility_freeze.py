"""Build the versioned paper reproducibility and artifact freeze manifest.

The manifest is derived from the immutable training summaries and the exact
final checkpoints used by the Phase 5--7 evaluations.  Current source-code
defaults are deliberately not treated as evidence for a historical run.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import gymnasium
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DQN_ROOT = ROOT / "3D_RL_DQN"
TRAINING_ROOT = (
    ROOT / "3D_0827" / "figure" / "phase_4_extended_random_boxes"
    / "training_v2" / "g2_generalization"
)
CHECKPOINT_ROOT = (
    DQN_ROOT / "checkpoints" / "phase4_extended_random_boxes"
    / "training_v2" / "g2_generalization"
)
OUTPUT_DIR = DQN_ROOT / "paper_freeze_v1"

CONDITIONS = (
    (100.0, 5.0, "dx100m_dpsi5deg"),
    (50.0, 5.0, "dx50m_dpsi5deg"),
    (25.0, 10.0, "dx25m_dpsi10deg"),
    (25.0, 5.0, "dx25m_dpsi5deg"),
    (25.0, 2.5, "dx25m_dpsi2p5deg"),
    (10.0, 5.0, "dx10m_dpsi5deg"),
)

ARTIFACT_ROOTS = (
    DQN_ROOT / "runs" / "phase5_g2_final_id_test",
    DQN_ROOT / "runs" / "phase6_g2_local_sse",
    DQN_ROOT / "runs" / "phase7_angular_sweep",
    DQN_ROOT / "runs" / "phase7_neighbor_sweep",
    DQN_ROOT / "runs" / "phase7_neighbor_sweep_dpsi5deg",
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=False,
        text=True, capture_output=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _gpu_driver() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"available": False}
    name, driver, memory_mib = (
        item.strip() for item in completed.stdout.strip().splitlines()[0].split(",")
    )
    return {
        "available": True,
        "name": name,
        "driver_version": driver,
        "memory_total_mib": int(memory_mib),
    }


def _serializable_optimizer_group(group: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in group.items():
        if key == "params":
            result["parameter_tensor_count"] = len(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def _model_record(dx_m: float, dpsi_deg: float, tag: str) -> dict[str, Any]:
    run_dir = TRAINING_ROOT / tag / "seed0_ep10000_v1"
    checkpoint = CHECKPOINT_ROOT / tag / "seed0_ep10000_v1" / "phase4_seed0_final.pt"
    summary_path = run_dir / "g2_training_summary.json"
    preparation_path = run_dir / "g2_preparation.json"
    dataset_path = run_dir / "g2_dataset_manifest.json"
    required = (checkpoint, summary_path, preparation_path, dataset_path)
    missing = [_relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing freeze inputs for {tag}: {missing}")

    summary = _read_json(summary_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    parameter_count = int(sum(value.numel() for value in state.values()))
    optimizer = payload.get("optimizer_state_dict") or {}
    optimizer_groups = [
        _serializable_optimizer_group(group)
        for group in optimizer.get("param_groups", [])
    ]
    seed_result = summary["seed_result"]
    return {
        "condition": {
            "spatial_resolution_m": dx_m,
            "heading_spacing_deg": dpsi_deg,
            "tag": tag,
        },
        "checkpoint": {
            "path": _relative(checkpoint),
            "sha256": _sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "kind": "final checkpoint used by Phase 5--7 evaluation",
        },
        "source_manifests": {
            "training_summary": {
                "path": _relative(summary_path),
                "sha256": _sha256(summary_path),
            },
            "preparation": {
                "path": _relative(preparation_path),
                "sha256": _sha256(preparation_path),
            },
            "dataset": {
                "path": _relative(dataset_path),
                "sha256": _sha256(dataset_path),
            },
        },
        "algorithm": summary["algorithm"],
        "training_config": summary["training_config"],
        "execution_config": summary["execution_config"],
        "early_stopping": summary["early_stopping"],
        "network_config": {
            **summary["network_config"],
            "trainable_parameter_count": parameter_count,
        },
        "optimizer_param_groups_from_checkpoint": optimizer_groups,
        "measured_training": {
            key: seed_result.get(key)
            for key in (
                "seed", "device", "episodes_completed", "best_episode",
                "training_runtime_sec", "training_wall_runtime_sec",
                "pure_training_runtime_sec", "evaluation_runtime_sec",
                "checkpoint_evaluation_runtime_sec", "checkpoint_io_runtime_sec",
                "post_training_evaluation_runtime_sec",
                "final_checkpoint_io_runtime_sec", "measured_seed_runtime_sec",
                "environment_steps", "optimization_steps", "replay_size",
            )
        },
        "configuration_source": (
            "immutable run summary plus optimizer state stored in the final "
            "checkpoint; not current source defaults"
        ),
    }


def _include_artifact(path: Path) -> bool:
    # Phase-5 contains 84 regenerable interactive HTML files (~389 MiB).  They
    # remain local and are intentionally omitted from Git; their JSON sources
    # and generation code are frozen.  Phase-6 paper 3-D HTML is compact enough
    # to retain as the representative end-to-end visualization set.
    phase5_root = DQN_ROOT / "runs" / "phase5_g2_final_id_test"
    if phase5_root in path.parents and path.suffix.lower() == ".html":
        return False
    return path.is_file()


def _artifact_records(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for root in ARTIFACT_ROOTS:
        paths.update(path for path in root.rglob("*") if _include_artifact(path))
    for model in models:
        paths.add(ROOT / model["checkpoint"]["path"])
        for source in model["source_manifests"].values():
            paths.add(ROOT / source["path"])
    return [
        {
            "path": _relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(paths, key=_relative)
    ]


def _software_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_from_torch": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "nvidia_smi": _gpu_driver(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_hyperparameters(models: list[dict[str, Any]]) -> Path:
    path = OUTPUT_DIR / "hyperparameters.csv"
    rows = []
    for model in models:
        condition = model["condition"]
        training = model["training_config"]
        network = model["network_config"]
        execution = model["execution_config"]
        measured = model["measured_training"]
        optimizer = (
            model["optimizer_param_groups_from_checkpoint"][0]
            if model["optimizer_param_groups_from_checkpoint"] else {}
        )
        rows.append({
            "dx_m": condition["spatial_resolution_m"],
            "dpsi_deg": condition["heading_spacing_deg"],
            "seed": measured["seed"],
            "algorithm": model["algorithm"]["name"],
            "episodes": measured["episodes_completed"],
            "environment_steps": measured["environment_steps"],
            "optimization_steps": measured["optimization_steps"],
            "gamma": training["gamma"],
            "learning_rate": training["learning_rate"],
            "optimizer": training["optimizer"],
            "adamw_weight_decay": optimizer.get("weight_decay"),
            "adamw_betas": optimizer.get("betas"),
            "adamw_eps": optimizer.get("eps"),
            "loss": training["loss"],
            "batch_size": training["batch_size"],
            "replay_capacity": training["replay_capacity"],
            "replay_warmup": training["replay_warmup"],
            "replay_sampling": model["algorithm"]["replay_sampling"],
            "epsilon_start": training["epsilon_start"],
            "epsilon_end": training["epsilon_end"],
            "epsilon_decay_fraction": training["epsilon_decay_fraction"],
            "target_update_optimizer_steps": training["target_update_steps"],
            "n_step_return": model["algorithm"]["n_step_return"],
            "gradient_clip_norm": training["gradient_clip_norm"],
            "episode_batch_size": execution["episode_batch_size"],
            "effective_max_updates_per_env_step": (
                execution["maximum_optimizer_updates_per_environment_step"]
            ),
            "action_count": network["action_count"],
            "parameter_count": network["trainable_parameter_count"],
            "checkpoint_sha256": model["checkpoint"]["sha256"],
        })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_artifacts(records: list[dict[str, Any]]) -> Path:
    path = OUTPUT_DIR / "artifact_checksums.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "size_bytes", "sha256"))
        writer.writeheader()
        writer.writerows(records)
    return path


def _write_report(manifest: dict[str, Any]) -> Path:
    path = OUTPUT_DIR / "README.md"
    rows = []
    for model in manifest["models"]:
        c = model["condition"]
        n = model["network_config"]
        t = model["training_config"]
        m = model["measured_training"]
        rows.append(
            f"| {c['spatial_resolution_m']:g} | {c['heading_spacing_deg']:g} | "
            f"{n['action_count']} | {n['trainable_parameter_count']:,} | "
            f"{m['episodes_completed']:,} | {m['environment_steps']:,} | "
            f"{m['optimization_steps']:,} | {t['learning_rate']:.1e} | "
            f"`{model['checkpoint']['sha256'][:12]}` |"
        )
    text = "\n".join([
        "# Phase 5–7 Paper Reproducibility Freeze v1",
        "",
        "This directory freezes the exact model configurations and artifact "
        "checksums used by the completed Phase 5–7 controlled experiments.",
        "Historical run summaries and checkpoint payloads are the configuration "
        "source of truth; current code defaults are not substituted.",
        "",
        "## Frozen models",
        "",
        "| dx [m] | dpsi [deg] | actions | parameters | episodes | environment steps | optimization steps | learning rate | checkpoint SHA-256 prefix |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
        *rows,
        "",
        "## Timing contract",
        "",
        "- Bellman first solve: main exact Local-SSE wall time after shared scene setup.",
        "- DQN first solve: first Local-SSE query with empty problem caches after scene/checkpoint setup.",
        "- DQN repeated query: secondary identical-query latency with retained caches.",
        "- Offline training, scene construction, checkpoint loading, CUDA initialization, and post-hoc exact certification are excluded from online first-solve time.",
        "",
        "## Artifact policy",
        "",
        "- Final evaluation checkpoints are stored through Git LFS.",
        "- Phase 5–7 JSON/CSV/Markdown, logs, and static PNG outputs are frozen.",
        "- Phase-6 representative 3-D HTML outputs are frozen.",
        "- The 84 regenerable Phase-5 HTML files (~389 MiB) are intentionally omitted; their raw JSON inputs and generation code are frozen.",
        "",
        "## Files",
        "",
        "- `reproducibility_manifest.json`: full configurations, optimizer parameter groups, measured training counts, software/hardware environment, and source hashes.",
        "- `hyperparameters.csv`: paper/appendix-ready per-condition hyperparameter table.",
        "- `artifact_checksums.csv`: SHA-256 index of every frozen input and output artifact.",
        "",
        "## Scope note",
        "",
        "`dx` and `dpsi` are computational/MDP discretization conditions. "
        "Chebyshev radius `r` is an outer Defender Local-SSE search condition and "
        "is not a DQN training hyperparameter.",
        "",
    ])
    path.write_text(text, encoding="utf-8")
    return path


def build() -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = [_model_record(*condition) for condition in CONDITIONS]
    artifacts = _artifact_records(models)
    manifest = {
        "schema": "p1b-paper-reproducibility-freeze-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_parent_commit": _git("rev-parse", "HEAD"),
        "source_branch": _git("branch", "--show-current"),
        "configuration_authority": (
            "immutable training summary and final checkpoint payload"
        ),
        "software_hardware": _software_environment(),
        "timing_contract": {
            "bellman_first": "main exact Local-SSE after shared setup",
            "dqn_first": "empty problem caches after shared setup/checkpoint load",
            "dqn_repeated": "identical query with retained problem caches",
            "excluded_online": [
                "offline training", "scene and graph construction",
                "checkpoint loading", "CUDA initialization",
                "post-hoc exact DQN-neighborhood audit",
            ],
        },
        "models": models,
        "artifact_policy": {
            "included_records": len(artifacts),
            "included_size_bytes": sum(item["size_bytes"] for item in artifacts),
            "phase5_html": "omitted as regenerable; raw JSON and code retained",
        },
    }
    manifest_path = OUTPUT_DIR / "reproducibility_manifest.json"
    _write_json(manifest_path, manifest)
    hyperparameters_path = _write_hyperparameters(models)
    artifacts_path = _write_artifacts(artifacts)
    report_path = _write_report(manifest)
    print(report_path)
    print(manifest_path)
    print(hyperparameters_path)
    print(artifacts_path)
    return report_path, manifest_path, hyperparameters_path


if __name__ == "__main__":
    build()

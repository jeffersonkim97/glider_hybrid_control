"""Plot per-episode Phase 16.4 attacker cost and its trailing mean.

The Phase 16.4 training history stores ``episode_full_transformed_return``.
Each environment reward is the negative transition cost and the stored value
also subtracts the powered-flight cost, so the physical attacker objective is

    J_A = -episode_full_transformed_return.

Only goal-reaching episodes have a complete J_A.  Failed episodes, if any, are
left out of both the raw trace and any rolling window containing them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import project_paths


DEFAULT_RUN_DIR = (
    project_paths.WORKSPACE_ROOT
    / "3D_0827"
    / "figure"
    / "phase_4_multi_terrain_dqn"
    / "dx25m_dpsi5deg"
    / "official_batched_v1"
)


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return a strict trailing mean, with NaN before the first full window."""

    if window <= 0:
        raise ValueError("window must be positive")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if values.size < window:
        return result

    finite = np.isfinite(values)
    sums = np.convolve(np.where(finite, values, 0.0), np.ones(window), mode="valid")
    counts = np.convolve(finite.astype(np.int64), np.ones(window, dtype=np.int64), mode="valid")
    valid = counts == window
    result[window - 1 :][valid] = sums[valid] / float(window)
    return result


def load_seed_series(history_path: Path, required_seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    by_seed = {int(item["seed"]): item for item in payload["seeds"]}
    missing = sorted(set(required_seeds) - set(by_seed))
    if missing:
        raise ValueError(f"training history is missing seeds: {missing}")

    result: list[dict[str, Any]] = []
    for seed in required_seeds:
        history = by_seed[seed]["history"]
        episode = np.asarray(history["episode"], dtype=np.int64)
        transformed_return = np.asarray(
            history["episode_full_transformed_return"], dtype=np.float64
        )
        reached_goal = np.asarray(history["episode_reached_goal"], dtype=bool)
        if not (episode.size == transformed_return.size == reached_goal.size):
            raise ValueError(f"seed {seed} history arrays have unequal lengths")
        if episode.size and np.any(np.diff(episode) <= 0):
            raise ValueError(f"seed {seed} episode numbers are not strictly increasing")

        attacker_cost = -transformed_return
        attacker_cost[~reached_goal] = np.nan
        result.append(
            {
                "seed": seed,
                "episode": episode,
                "J_A": attacker_cost,
                "reached_goal": reached_goal,
            }
        )
    return result


def style_axis(ax: Any, seed: int, window: int) -> None:
    ax.set_title(f"Seed {seed}")
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"Attacker objective $J_A$ (lower is better)")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")
    ax.ticklabel_format(axis="x", style="plain")


def plot_one(
    series: dict[str, Any], output: Path, window: int, condition_label: str,
) -> None:
    episodes = series["episode"]
    values = series["J_A"]
    average = trailing_mean(values, window)
    figure, ax = plt.subplots(figsize=(12.0, 5.6), constrained_layout=True)
    ax.plot(
        episodes,
        values,
        color="#56B4E9",
        linewidth=0.45,
        alpha=0.22,
        label=r"Episode $J_A$",
        rasterized=True,
    )
    ax.plot(
        episodes,
        average,
        color="#0072B2",
        linewidth=2.0,
        label=f"{window}-episode trailing mean",
    )
    style_axis(ax, int(series["seed"]), window)
    figure.suptitle(
        f"Phase 16.4 DQN training cost ({condition_label})", fontsize=14
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_combined(
    series_by_seed: list[dict[str, Any]], output: Path, window: int,
    condition_label: str,
) -> None:
    figure, axes = plt.subplots(
        len(series_by_seed),
        1,
        figsize=(12.0, 4.25 * len(series_by_seed)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, series in zip(axes, series_by_seed):
        episodes = series["episode"]
        values = series["J_A"]
        ax.plot(
            episodes,
            values,
            color="#56B4E9",
            linewidth=0.4,
            alpha=0.18,
            label=r"Episode $J_A$",
            rasterized=True,
        )
        ax.plot(
            episodes,
            trailing_mean(values, window),
            color="#0072B2",
            linewidth=1.8,
            label=f"{window}-episode trailing mean",
        )
        style_axis(ax, int(series["seed"]), window)
    figure.suptitle(
        f"Phase 16.4 DQN training cost by seed ({condition_label})", fontsize=14
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_summary(series_by_seed: list[dict[str, Any]], window: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for series in series_by_seed:
        values = series["J_A"]
        reached_goal = series["reached_goal"]
        average = trailing_mean(values, window)
        finite_average = average[np.isfinite(average)]
        records.append(
            {
                "seed": int(series["seed"]),
                "episode_count": int(values.size),
                "goal_reaching_episode_count": int(reached_goal.sum()),
                "goal_reaching_rate": float(reached_goal.mean()),
                "episode_J_A_mean": float(np.nanmean(values)),
                "episode_J_A_min": float(np.nanmin(values)),
                "episode_J_A_max": float(np.nanmax(values)),
                f"initial_{window}_episode_mean_J_A": (
                    None if not finite_average.size else float(finite_average[0])
                ),
                f"final_{window}_episode_mean_J_A": (
                    None if not finite_average.size else float(finite_average[-1])
                ),
                f"minimum_{window}_episode_mean_J_A": (
                    None if not finite_average.size else float(finite_average.min())
                ),
            }
        )
    return {
        "source_field": "episode_full_transformed_return",
        "conversion": "J_A = -episode_full_transformed_return",
        "rolling_window_episodes": window,
        "rolling_definition": "trailing arithmetic mean over complete goal-reaching episodes",
        "lower_J_A_is_better": True,
        "seeds": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--window", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    history_path = run_dir / "phase4_training_history.json"
    if not history_path.is_file():
        raise FileNotFoundError(history_path)

    model_manifest_path = run_dir / "phase4_generalized_model_manifest.json"
    if not model_manifest_path.is_file():
        raise FileNotFoundError(model_manifest_path)
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    condition = model_manifest["condition"]
    condition_label = (
        f"{float(condition['spatial_resolution_m']):g} m, "
        f"{float(condition['heading_spacing_deg']):g} deg"
    )

    series_by_seed = load_seed_series(history_path, (0, 1, 2))
    outputs: list[Path] = []
    for series in series_by_seed:
        output = run_dir / f"phase4_seed_{series['seed']}_episode_J_A.png"
        plot_one(series, output, args.window, condition_label)
        outputs.append(output)

    combined = run_dir / "phase4_episode_J_A_by_seed.png"
    plot_combined(series_by_seed, combined, args.window, condition_label)
    outputs.append(combined)

    summary_path = run_dir / "phase4_episode_J_A_by_seed_summary.json"
    summary_path.write_text(
        json.dumps(make_summary(series_by_seed, args.window), indent=2) + "\n",
        encoding="utf-8",
    )
    outputs.append(summary_path)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

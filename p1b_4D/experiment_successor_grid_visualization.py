"""Generate a diagnostic comparison of snapped and physical-edge paths."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.stackelberg_solver import solve_attacker_best_response


def main() -> Path:
    root = Path(__file__).resolve().parent.parent
    configuration = build_configuration_bundle(root)
    try:
        results = {}
        for model in ("snapped_fixed_time_step", "successor_grid_physical_edge"):
            selected = deepcopy(configuration)
            selected["primary_result"]["attacker_solver_config"][
                "transition_model"
            ] = model
            results[model] = solve_attacker_best_response(
                4000.0, selected, f"visualization-{model}"
            )["primary_result"]

        geometry = results["successor_grid_physical_edge"]["geometry_bundle"][
            "primary_result"
        ]
        terrain_arrays = geometry["terrain_arrays"]
        terrain = np.column_stack(
            (terrain_arrays["z"], terrain_arrays["height"])
        )
        los = geometry["los_geometry"]["los_boundary"]
        sensor = geometry["sensor_position"]
        goal = geometry["goal_position"]

        figure, axes = plt.subplots(1, 2, figsize=(15, 5.7), sharex=True, sharey=True)
        specifications = (
            ("snapped_fixed_time_step", "Legacy snapped transition"),
            ("successor_grid_physical_edge", "Physical successor-grid transition"),
        )
        for axis, (model, title) in zip(axes, specifications, strict=True):
            best = results[model]["best_found_attacker_response"]
            planned = np.asarray(best["trajectory"])
            replay = np.asarray(best["continuous_replay_validation"]["trajectory"])
            axis.fill_between(
                terrain[:, 0], 0.0, terrain[:, 1], color="0.82", label="Terrain"
            )
            axis.plot(terrain[:, 0], terrain[:, 1], color="black", linewidth=1.4)
            axis.plot(
                los[:, 0], los[:, 1], color="tab:green", linestyle="--",
                linewidth=1.5, label="LOS boundary",
            )
            axis.plot(
                planned[:, 0], planned[:, 1], color="tab:blue", linewidth=2.0,
                marker=".", markersize=2.8, label="Bellman reported path",
            )
            axis.plot(
                replay[:, 0], replay[:, 1], color="tab:red", linewidth=2.0,
                linestyle=":", label="Unsnapped continuous replay",
            )
            axis.scatter(*best["switching_point"], color="tab:orange", s=55,
                         marker="o", zorder=5, label="Virtual/switch state")
            axis.scatter(*sensor, color="black", s=70, marker="^", zorder=5,
                         label="Sensor")
            axis.scatter(*goal, color="gold", edgecolor="black", s=90, marker="*",
                         zorder=5, label="Goal")
            replay_result = best["continuous_replay_validation"]
            residual = best["constraint_residuals"].get(
                "maximum_edge_endpoint_residual", np.nan
            )
            axis.set_title(
                f"{title}\ncontinuous feasible={replay_result['feasible']}, "
                f"node residual={residual:.3g} m"
            )
            axis.set_xlabel("Along-track position z [m]")
            axis.grid(alpha=0.22)
        axes[0].set_ylabel("Altitude h [m]")
        handles, labels = axes[1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        figure.suptitle(
            "Bellman path vs. unsnapped continuous execution (z_sensor = 4000 m)",
            fontsize=14,
        )
        figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.93))
        output = root / "results" / "figures" / "successor_grid_transition_comparison.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(output)
        return output
    finally:
        close_phase_logger(
            configuration["primary_result"]["logging_utilities"]["logger"]
        )


if __name__ == "__main__":
    main()

"""Computation time against each discretization axis, exact versus Q-learning.

One figure per axis on log-log axes.  Measurements only: no fitted curve.  The
learned side is split into training and query because they behave differently -
training is where the whole of its growth lives, while query stays nearly flat,
and a single combined curve would hide that.

The Cartesian axis is the whole Defender search.  One best response is not what
a solve costs: at 5 m a single response is 445.8 s, but the search runs 48 of
them and takes 21,221 s.  The heading axis is one best response at a fixed
Defender position, which is the only heading sweep on disk.

    python P1b_sweep_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (
    FuncFormatter, LogLocator, MaxNLocator, NullFormatter,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "r2_discretization_sweeps" / "data"
OUTPUT = ROOT / "r2_discretization_sweeps" / "figure"
RADIUS_DATA = ROOT / "r3_neighbourhood_radius" / "data"
RADIUS_OUTPUT = ROOT / "r3_neighbourhood_radius" / "figure"

SERIES = (
    ("Exact, total", "exact_s", "#1C5770", "o", "-"),
    ("Q-learning, training", "rl_train_s", "#B3402F", "s", "--"),
    ("Q-learning, query", "rl_query_s", "#9C6511", "^", "-."),
)


def _power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least squares on log(y) = log(a) + b log(x); returns a, b and R^2.

    Reported on the console only - the figures carry no fitted curve.
    """
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(np.count_nonzero(keep)) < 2:
        return float("nan"), float("nan"), float("nan")
    log_x, log_y = np.log(x[keep]), np.log(y[keep])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    predicted = intercept + slope * log_x
    residual = float(np.sum((log_y - predicted) ** 2))
    total = float(np.sum((log_y - log_y.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else float("nan")
    return float(np.exp(intercept)), float(slope), r_squared


def figure_axis(
    records: list[dict[str, Any]], key: str, title: str, x_title: str, stem: str,
    *, log: bool = True, output: Path = OUTPUT, integer_x: bool = False,
    legend_loc: str = "upper right",
) -> str:
    rows = sorted(records, key=lambda r: r[key])
    x = np.asarray([r[key] for r in rows], dtype=float)

    figure, axes = plt.subplots(figsize=(7.2, 5.0), dpi=200)
    for name, field, colour, marker, dash in SERIES:
        y = np.asarray([r[field] for r in rows], dtype=float)
        axes.plot(x, y, marker=marker, linestyle=dash, color=colour,
                  markersize=6, linewidth=1.6, label=name)

    if log:
        axes.set_xscale("log")
        axes.set_yscale("log")
        # The measured lattices, labelled by name: there are only a handful and
        # the reader wants to find them, not read them off decade ticks.
        axes.set_xticks(x)
        axes.set_xticklabels([f"{v:g}" for v in x])
        axes.xaxis.set_minor_locator(LogLocator(subs="all"))
        axes.xaxis.set_minor_formatter(NullFormatter())
        # One label per decade on y; matplotlib's minor log labels print bare
        # mantissas that read as values.
        axes.yaxis.set_major_locator(LogLocator(base=10.0))
        axes.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        axes.yaxis.set_minor_formatter(NullFormatter())
    else:
        # Linear axes start at the origin, so the reader can read a ratio off
        # the plot rather than off the tick labels.
        axes.set_xlim(0.0, float(x.max()) * 1.05)
        axes.set_ylim(bottom=0.0)
        # Even ticks, not the measured steps: on a linear axis 10 and 12.5 sit
        # close enough that their labels collide.  The markers still show where
        # the measurements are.
        if integer_x:
            # r is a count of lattice steps; every value it can take is measured,
            # so label them all rather than letting the locator invent halves.
            axes.set_xlim(0.0, float(x.max()) + 0.5)
            axes.set_xticks(x)
            axes.set_xticklabels([f"{v:g}" for v in x])
        else:
            axes.xaxis.set_major_locator(
                MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10])
            )
            axes.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        axes.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _pos: f"{v:,.0f}")
        )

    axes.set_xlabel(x_title)
    axes.set_ylabel("computation time [s]")
    axes.set_title(title, fontsize=11)
    axes.grid(True, which="major", color="#DDDDDD", linewidth=0.7)
    axes.grid(True, which="minor", color="#F0F0F0", linewidth=0.5)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    # Whichever corner the series leave free: they fall to the right on the two
    # discretization axes and rise to the right on the radius axis.
    axes.legend(loc=legend_loc, frameon=True, framealpha=0.9,
                edgecolor="none", fontsize=9)

    figure.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{stem}.png", bbox_inches="tight")
    plt.close(figure)
    return stem


def report(records: list[dict[str, Any]], key: str, label: str) -> None:
    rows = sorted(records, key=lambda r: r[key])
    x = np.asarray([r[key] for r in rows], dtype=float)
    print(f"\n{label}")
    print(f"  {'series':<28} {'amplitude':>12} {'exponent':>9} {'R^2':>8}")
    for name, field, _colour, _marker, _dash in SERIES:
        amplitude, exponent, r_squared = _power_law(
            x, np.asarray([r[field] for r in rows], dtype=float),
        )
        print(f"  {name:<28} {amplitude:>12.4g} {exponent:>9.3f} {r_squared:>8.4f}")


def main() -> None:
    spatial = json.loads(
        (DATA / "spatial_sweep_full_sse.json").read_text(encoding="utf-8")
    )
    heading = json.loads(
        (DATA / "heading_sweep_fixed_defender.json").read_text(encoding="utf-8")
    )
    radius = json.loads(
        (RADIUS_DATA / "r_sweep_50m.json").read_text(encoding="utf-8")
    )
    written = []
    # Both scalings of the same measurements: log-log separates the coarse end,
    # where the series sit within a factor of two of one another, while the
    # linear pair shows how much of the total the 5 m point actually is.
    for suffix, log in (("", True), ("_linear", False)):
        written.append(figure_axis(
            spatial, "dx_m",
            "Cartesian discretization against computation time\n"
            "full local SSE, r = 1; heading 5 deg; 20,000 episodes",
            "spatial step dx [m]  (finer to the left)",
            f"1_cartesian_discretization_vs_time{suffix}", log=log,
        ))
        written.append(figure_axis(
            heading, "dpsi_deg",
            "Heading discretization against computation time\n"
            "one best response at a fixed Defender position (5, 0, 0);"
            " dx = 12.5 m; 20,000 episodes",
            "heading step [deg]  (finer to the left)",
            f"2_heading_discretization_vs_time{suffix}", log=log,
        ))
        # r rises to the right, unlike the two discretization axes, so the free
        # corner is the upper left; and every value r takes is measured, so the
        # linear panel labels all seven rather than inventing half-steps.
        written.append(figure_axis(
            radius, "r_neighbor",
            "Defender neighbourhood radius against computation time\n"
            "full local SSE; dx = 50 m; heading 5 deg; 20,000 episodes",
            "Chebyshev neighbourhood radius r  (wider to the right)",
            f"3_neighbourhood_radius_vs_time{suffix}", log=log,
            output=RADIUS_OUTPUT, integer_x=True, legend_loc="upper left",
        ))
    report(spatial, "dx_m", "Cartesian axis, full local SSE (console only)")
    report(heading, "dpsi_deg", "Heading axis, fixed Defender")
    report(radius, "r_neighbor", "Neighbourhood radius axis, full local SSE")
    print("\nwrote: " + ", ".join(written))


__all__ = ["figure_axis", "main"]


if __name__ == "__main__":
    main()

"""Generate Stage-3 geometric switching-candidate diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from los_explorer_gui import compute_los_case
from switching_candidates import (
    INITIAL_CONTOUR_SAMPLE_COUNT,
    INITIAL_RADIAL_SCALES,
    generate_switching_candidates,
)
from terrain_catalog import TERRAIN_LABELS
from visualization import plot_switching_candidates


DEFAULT_STAGE3_FIGURE_DIRECTORY = (
    Path(__file__).resolve().parent / "figure" / "stage_3_switching_candidates"
)


def generate_stage3_artifacts(
    output_directory: Path = DEFAULT_STAGE3_FIGURE_DIRECTORY,
) -> dict[str, dict[str, Any]]:
    """Write one 96-candidate interactive figure per terrain category."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}

    for terrain_name in TERRAIN_LABELS:
        los_result = compute_los_case(terrain_name, 5.0, 0.0)
        contour = los_result.tangent_contour
        candidates = generate_switching_candidates(contour)
        positions = np.vstack([candidate.position_map for candidate in candidates])
        residuals = np.fromiter(
            (candidate.surface_residual for candidate in candidates),
            dtype=float,
        )
        positive_direction_products = np.fromiter(
            (
                float(np.dot(
                    candidate.position_map - contour.origin.as_array(),
                    contour.tangent_vector_at(candidate.contour_fraction),
                ))
                for candidate in candidates
            ),
            dtype=float,
        )
        summaries[terrain_name] = {
            "candidate_count": len(candidates),
            "contour_sample_count": INITIAL_CONTOUR_SAMPLE_COUNT,
            "radial_sample_count": len(INITIAL_RADIAL_SCALES),
            "radial_scales": list(INITIAL_RADIAL_SCALES),
            "candidate_id_range": [
                candidates[0].candidate_id,
                candidates[-1].candidate_id,
            ],
            "position_min": np.min(positions, axis=0).tolist(),
            "position_max": np.max(positions, axis=0).tolist(),
            "max_surface_residual": float(np.max(residuals)),
            "min_positive_direction_product": float(
                np.min(positive_direction_products)
            ),
            "all_coordinates_finite": bool(np.all(np.isfinite(positions))),
        }

        figure = plot_switching_candidates(
            los_result.terrain_map,
            los_result.mission_points,
            los_result.los_surface,
            los_result.visualization_rays,
            candidates,
        )
        figure.update_layout(title={
            "text": (
                f"Stage 3 switching candidates: {TERRAIN_LABELS[terrain_name]}"
                "<br><sup>12 contour × 8 positive radial samples; "
                "geometry only, no energy classes</sup>"
            ),
            "x": 0.5,
        })
        figure.write_html(
            output_directory / f"{terrain_name}_switching_candidates.html",
            include_plotlyjs="directory",
            full_html=True,
            auto_open=False,
        )

    (output_directory / "switching_candidate_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summaries


if __name__ == "__main__":
    generated = generate_stage3_artifacts()
    print(
        f"Generated {len(generated)} interactive switching-candidate figures in "
        f"{DEFAULT_STAGE3_FIGURE_DIRECTORY}"
    )

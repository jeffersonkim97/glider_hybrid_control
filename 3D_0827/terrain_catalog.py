"""Selectable terrain cases for the interactive LOS explorer."""

from __future__ import annotations

from collections.abc import Callable

from map_geometry import BoxObstacle, CompositeTerrainMap, MapBounds


DEFAULT_BOUNDS = MapBounds(x_min=-10.0, x_max=10.0, y_min=-8.0, y_max=8.0)


def _centered_cube(bounds: MapBounds) -> CompositeTerrainMap:
    return CompositeTerrainMap(
        bounds=bounds,
        boxes=(
            BoxObstacle(
                center_x=0.0,
                center_y=0.0,
                width_x=4.0,
                width_y=4.0,
                height=4.0,
            ),
        ),
        category_id="centered_cube",
    )


def _centered_cube_half_height(bounds: MapBounds) -> CompositeTerrainMap:
    """The centred cube at half height, same footprint.

    The full-height cube tops out at 4.0 against a lattice ceiling of 5.0, so the
    tangent surface it casts runs out of headroom: measured at 50 m, half of the
    traced surface sits above the ceiling and the admissible switching set is cut
    off flat at exactly 5.00.  Halving the height leaves three map units of
    clearance instead of one, so the surface ends where the geometry ends rather
    than where the box does.  The footprint is unchanged, so the horizontal
    occlusion - which is what separates the admissible arc from the start-shadowed
    one - stays comparable to the full-height case.
    """
    return CompositeTerrainMap(
        bounds=bounds,
        boxes=(
            BoxObstacle(
                center_x=0.0,
                center_y=0.0,
                width_x=4.0,
                width_y=4.0,
                height=2.0,
            ),
        ),
        category_id="centered_cube_half_height",
    )


def _offset_cube(bounds: MapBounds, *, center_y: float, category_id: str) -> CompositeTerrainMap:
    return CompositeTerrainMap(
        bounds=bounds,
        boxes=(
            BoxObstacle(
                center_x=0.0,
                center_y=center_y,
                width_x=4.0,
                width_y=4.0,
                height=4.0,
            ),
        ),
        category_id=category_id,
    )


def _stepped_pyramid(bounds: MapBounds) -> CompositeTerrainMap:
    return CompositeTerrainMap(
        bounds=bounds,
        boxes=(
            BoxObstacle(
                center_x=0.0,
                center_y=0.0,
                width_x=6.0,
                width_y=6.0,
                height=1.25,
                base_z=0.0,
            ),
            BoxObstacle(
                center_x=0.0,
                center_y=0.0,
                width_x=4.5,
                width_y=4.5,
                height=1.25,
                base_z=1.25,
            ),
            BoxObstacle(
                center_x=0.0,
                center_y=0.0,
                width_x=3.0,
                width_y=3.0,
                height=1.50,
                base_z=2.50,
            ),
        ),
        category_id="stepped_pyramid",
    )


TERRAIN_LABELS: dict[str, str] = {
    "centered_cube": "Centered cube",
    "centered_cube_half_height": "Centered cube - half height",
    "offset_cube_left": "Offset cube - left",
    "offset_cube_right": "Offset cube - right",
    "stepped_pyramid": "Stepped pyramid (box approximation)",
}


_TERRAIN_BUILDERS: dict[str, Callable[[MapBounds], CompositeTerrainMap]] = {
    "centered_cube": _centered_cube,
    "centered_cube_half_height": _centered_cube_half_height,
    "offset_cube_left": lambda bounds: _offset_cube(
        bounds,
        center_y=-3.0,
        category_id="offset_cube_left",
    ),
    "offset_cube_right": lambda bounds: _offset_cube(
        bounds,
        center_y=3.0,
        category_id="offset_cube_right",
    ),
    "stepped_pyramid": _stepped_pyramid,
}


def build_terrain(
    category_id: str,
    *,
    bounds: MapBounds = DEFAULT_BOUNDS,
) -> CompositeTerrainMap:
    """Build one fresh terrain object from its GUI category identifier."""
    try:
        builder = _TERRAIN_BUILDERS[category_id]
    except KeyError as error:
        valid_ids = ", ".join(TERRAIN_LABELS)
        raise ValueError(f"unknown terrain category {category_id!r}; choose {valid_ids}") from error
    return builder(bounds)


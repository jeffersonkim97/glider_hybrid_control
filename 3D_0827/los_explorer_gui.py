"""Notebook GUI for interactive terrain and defender-position LOS exploration."""

from __future__ import annotations

from dataclasses import dataclass

import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import clear_output, display

from los_geometry import (
    LOSModel,
    LOSTangentSurface,
    TangentContour,
    VisualizationRaySet,
)
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE
from map_geometry import TerrainModel
from reachability_surface import (
    EnergyReachabilitySurface,
    LOSSurfaceReachabilityClassifier,
)
from scenario import MissionPoints, Point3D
from terrain_catalog import DEFAULT_BOUNDS, TERRAIN_LABELS, build_terrain
from visualization import plot_energy_reachability_surface, plot_los_tangent_surface


@dataclass(frozen=True)
class LOSExplorerResult:
    """Complete result from one terrain/sensor GUI evaluation."""

    terrain_map: TerrainModel
    los_model: LOSModel
    mission_points: MissionPoints
    tangent_contour: TangentContour
    visualization_rays: VisualizationRaySet
    los_surface: LOSTangentSurface
    figure: go.Figure


@dataclass(frozen=True)
class EnergyExplorerResult:
    """LOS geometry plus the total-energy reachability classification."""

    los_result: LOSExplorerResult
    energy_surface: EnergyReachabilitySurface
    figure: go.Figure


def compute_los_case(
    terrain_category: str,
    sensor_x: float,
    sensor_y: float,
    *,
    probe_grid_size: int = 101,
    displayed_ray_count: int = 10,
    display_extension_factor: float = 4.0,
) -> LOSExplorerResult:
    """Compute one GUI state without any widget dependencies or side effects."""
    if not 5.0 <= sensor_x <= 10.0:
        raise ValueError("sensor_x must lie in [5, 10]")
    if not DEFAULT_BOUNDS.y_min <= sensor_y <= DEFAULT_BOUNDS.y_max:
        raise ValueError("sensor_y must lie inside the complete map y range")

    terrain_map = build_terrain(terrain_category)
    mission_points = MissionPoints(
        sensor=Point3D(x=float(sensor_x), y=float(sensor_y), z=terrain_map.ground_z),
        start=Point3D(x=-8.0, y=0.0, z=terrain_map.ground_z),
        goal=Point3D(x=8.0, y=0.0, z=terrain_map.ground_z),
    )
    mission_points.validate_against(terrain_map)

    los_model = LOSModel(terrain_map)
    tangent_contour = los_model.trace_tangent_contour(
        mission_points.sensor,
        probe_grid_size=probe_grid_size,
        boundary_refinement_steps=24,
    )
    visualization_rays = tangent_contour.sample_for_visualization(displayed_ray_count)
    los_surface = los_model.build_tangent_surface(
        tangent_contour,
        display_extension_factor=display_extension_factor,
    )
    figure = plot_los_tangent_surface(
        terrain_map,
        mission_points,
        los_surface,
        visualization_rays,
    )
    terrain_label = TERRAIN_LABELS[terrain_category]
    figure.update_layout(
        title={
            "text": (
                f"LOS Explorer - {terrain_label}"
                f"<br><sup>Sensor = ({sensor_x:.2f}, {sensor_y:.2f}, 0.00) map unit</sup>"
            ),
            "x": 0.5,
        }
    )
    return LOSExplorerResult(
        terrain_map=terrain_map,
        los_model=los_model,
        mission_points=mission_points,
        tangent_contour=tangent_contour,
        visualization_rays=visualization_rays,
        los_surface=los_surface,
        figure=figure,
    )


def compute_energy_case(
    terrain_category: str,
    sensor_x: float,
    sensor_y: float,
    *,
    probe_grid_size: int = 101,
    displayed_ray_count: int = 10,
    display_extension_factor: float = 4.0,
    radial_section_count: int = 25,
    contour_section_count: int = 96,
) -> EnergyExplorerResult:
    """Compute LOS geometry and classify its display clip by goal reachability."""
    los_result = compute_los_case(
        terrain_category,
        sensor_x,
        sensor_y,
        probe_grid_size=probe_grid_size,
        displayed_ray_count=displayed_ray_count,
        display_extension_factor=display_extension_factor,
    )
    classifier = LOSSurfaceReachabilityClassifier(
        DEFAULT_GLIDER,
        DEFAULT_PHYSICAL_SCALE,
    )
    energy_surface = classifier.classify_display_surface(
        los_result.los_surface,
        los_result.terrain_map,
        los_result.mission_points,
        radial_section_count=radial_section_count,
        contour_section_count=contour_section_count,
    )
    figure = plot_energy_reachability_surface(
        los_result.terrain_map,
        los_result.mission_points,
        los_result.los_surface,
        los_result.visualization_rays,
        energy_surface,
        aircraft_name=DEFAULT_GLIDER.aircraft_name,
        meters_per_map_unit=DEFAULT_PHYSICAL_SCALE.meters_per_map_unit,
        goal_tolerance_m=DEFAULT_GLIDER.goal_tolerance_m,
    )
    terrain_label = TERRAIN_LABELS[terrain_category]
    figure.update_layout(
        title={
            "text": (
                f"Total-Energy LOS Explorer - {terrain_label}"
                f"<br><sup>Sensor=({sensor_x:.2f}, {sensor_y:.2f}, 0.00); "
                f"{DEFAULT_GLIDER.aircraft_name}; "
                f"1 map unit={DEFAULT_PHYSICAL_SCALE.meters_per_map_unit:g} m; "
                f"goal tolerance={DEFAULT_GLIDER.goal_tolerance_m:g} m</sup>"
            ),
            "x": 0.5,
        }
    )
    return EnergyExplorerResult(
        los_result=los_result,
        energy_surface=energy_surface,
        figure=figure,
    )


class LOSExplorer:
    """ipywidgets controller for live terrain and sensor-position updates."""

    def __init__(
        self,
        *,
        probe_grid_size: int = 101,
        displayed_ray_count: int = 10,
    ) -> None:
        self.probe_grid_size = probe_grid_size
        self.displayed_ray_count = displayed_ray_count
        self.latest_result: LOSExplorerResult | None = None
        self.latest_error: Exception | None = None

        self.terrain_toggle = widgets.ToggleButtons(
            options=[(label, category_id) for category_id, label in TERRAIN_LABELS.items()],
            value="centered_cube",
            description="Terrain:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="100%"),
        )
        self.sensor_x_slider = widgets.FloatSlider(
            value=5.0,
            min=5.0,
            max=10.0,
            step=0.25,
            description="Sensor x:",
            readout_format=".2f",
            continuous_update=False,
            style={"description_width": "80px"},
            layout=widgets.Layout(width="48%"),
        )
        self.sensor_y_slider = widgets.FloatSlider(
            value=0.0,
            min=DEFAULT_BOUNDS.y_min,
            max=DEFAULT_BOUNDS.y_max,
            step=0.25,
            description="Sensor y:",
            readout_format=".2f",
            continuous_update=False,
            style={"description_width": "80px"},
            layout=widgets.Layout(width="48%"),
        )
        self.status = widgets.HTML()
        self.output = widgets.Output(
            layout=widgets.Layout(width="100%", min_height="720px")
        )
        self.widget = widgets.VBox(
            [
                widgets.HTML("<h3>3D Terrain / Defender LOS Explorer</h3>"),
                self.terrain_toggle,
                widgets.HBox(
                    [self.sensor_x_slider, self.sensor_y_slider],
                    layout=widgets.Layout(justify_content="space-between"),
                ),
                self.status,
                self.output,
            ],
            layout=widgets.Layout(width="100%"),
        )

        for control in (
            self.terrain_toggle,
            self.sensor_x_slider,
            self.sensor_y_slider,
        ):
            control.observe(self._on_control_change, names="value")
        self.refresh()

    def _set_controls_disabled(self, disabled: bool) -> None:
        self.terrain_toggle.disabled = disabled
        self.sensor_x_slider.disabled = disabled
        self.sensor_y_slider.disabled = disabled

    def _on_control_change(self, _change: dict[str, object]) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Recompute the complete LOS result from the current GUI state."""
        self._set_controls_disabled(True)
        terrain_category = str(self.terrain_toggle.value)
        sensor_x = float(self.sensor_x_slider.value)
        sensor_y = float(self.sensor_y_slider.value)
        terrain_label = TERRAIN_LABELS[terrain_category]
        self.status.value = (
            f"<span style='color:#555'>Computing {terrain_label} at "
            f"sensor ({sensor_x:.2f}, {sensor_y:.2f}, 0.00) m...</span>"
        )
        try:
            result = compute_los_case(
                terrain_category,
                sensor_x,
                sensor_y,
                probe_grid_size=self.probe_grid_size,
                displayed_ray_count=self.displayed_ray_count,
            )
            self.latest_result = result
            self.latest_error = None
            with self.output:
                clear_output(wait=True)
                display(result.figure)
            self.status.value = (
                "<span style='color:#176b3a'><b>Ready.</b> "
                f"Dense upper horizon: {len(result.tangent_contour.rays)} rays; "
                f"discarded ground candidates: "
                f"{result.tangent_contour.discarded_ground_candidate_count}; "
                f"surface: {result.los_surface.panel_count} panels; "
                f"displayed samples: {len(result.visualization_rays.rays)}.</span>"
            )
        except Exception as error:
            self.latest_result = None
            self.latest_error = error
            with self.output:
                clear_output(wait=True)
                print(f"LOS computation failed: {type(error).__name__}: {error}")
            self.status.value = (
                "<span style='color:#b00020'><b>Error:</b> "
                f"{type(error).__name__}: {error}</span>"
            )
        finally:
            self._set_controls_disabled(False)

    def display(self) -> None:
        display(self.widget)


def create_los_explorer(
    *,
    probe_grid_size: int = 101,
    displayed_ray_count: int = 10,
) -> LOSExplorer:
    """Create a ready-to-display notebook LOS explorer."""
    return LOSExplorer(
        probe_grid_size=probe_grid_size,
        displayed_ray_count=displayed_ray_count,
    )


class EnergyReachabilityExplorer:
    """Notebook GUI for the red/green energy classification of the LOS surface."""

    def __init__(
        self,
        *,
        probe_grid_size: int = 101,
        displayed_ray_count: int = 10,
        radial_section_count: int = 25,
        contour_section_count: int = 96,
    ) -> None:
        self.probe_grid_size = probe_grid_size
        self.displayed_ray_count = displayed_ray_count
        self.radial_section_count = radial_section_count
        self.contour_section_count = contour_section_count
        self.latest_result: EnergyExplorerResult | None = None
        self.latest_error: Exception | None = None

        self.terrain_toggle = widgets.ToggleButtons(
            options=[(label, category_id) for category_id, label in TERRAIN_LABELS.items()],
            value="centered_cube",
            description="Terrain:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="100%"),
        )
        self.sensor_x_slider = widgets.FloatSlider(
            value=5.0,
            min=5.0,
            max=10.0,
            step=0.25,
            description="Sensor x:",
            readout_format=".2f",
            continuous_update=False,
            style={"description_width": "80px"},
            layout=widgets.Layout(width="48%"),
        )
        self.sensor_y_slider = widgets.FloatSlider(
            value=0.0,
            min=DEFAULT_BOUNDS.y_min,
            max=DEFAULT_BOUNDS.y_max,
            step=0.25,
            description="Sensor y:",
            readout_format=".2f",
            continuous_update=False,
            style={"description_width": "80px"},
            layout=widgets.Layout(width="48%"),
        )
        parameter_note = (
            f"<b>Prototype:</b> {DEFAULT_GLIDER.aircraft_name}, "
            f"V<sub>powered</sub>={DEFAULT_GLIDER.powered_speed_mps:.2f} m/s, "
            f"V<sub>glide</sub>={DEFAULT_GLIDER.best_glide_speed_mps:.2f} m/s, "
            f"straight L/D={DEFAULT_GLIDER.best_glide_ratio:g}, "
            f"30°-turn L/D={DEFAULT_GLIDER.turn_glide_ratio:.1f}, "
            f"bank≤{DEFAULT_GLIDER.maximum_bank_deg:g}°, "
            f"R<sub>min</sub>={DEFAULT_GLIDER.minimum_turn_radius_m:.1f} m, "
            f"switch loss={DEFAULT_GLIDER.switch_energy_loss_height_m:g} m, "
            f"goal tolerance={DEFAULT_GLIDER.goal_tolerance_m:g} m, "
            f"scale={DEFAULT_PHYSICAL_SCALE.meters_per_map_unit:g} m/map unit."
        )
        self.status = widgets.HTML()
        self.output = widgets.Output(
            layout=widgets.Layout(width="100%", min_height="720px")
        )
        self.widget = widgets.VBox(
            [
                widgets.HTML("<h3>3D Total-Energy / Goal-Reachability Explorer</h3>"),
                widgets.HTML(parameter_note),
                self.terrain_toggle,
                widgets.HBox(
                    [self.sensor_x_slider, self.sensor_y_slider],
                    layout=widgets.Layout(justify_content="space-between"),
                ),
                self.status,
                self.output,
            ],
            layout=widgets.Layout(width="100%"),
        )

        for control in (
            self.terrain_toggle,
            self.sensor_x_slider,
            self.sensor_y_slider,
        ):
            control.observe(self._on_control_change, names="value")
        self.refresh()

    def _set_controls_disabled(self, disabled: bool) -> None:
        self.terrain_toggle.disabled = disabled
        self.sensor_x_slider.disabled = disabled
        self.sensor_y_slider.disabled = disabled

    def _on_control_change(self, _change: dict[str, object]) -> None:
        self.refresh()

    def refresh(self) -> None:
        self._set_controls_disabled(True)
        terrain_category = str(self.terrain_toggle.value)
        sensor_x = float(self.sensor_x_slider.value)
        sensor_y = float(self.sensor_y_slider.value)
        terrain_label = TERRAIN_LABELS[terrain_category]
        self.status.value = (
            f"<span style='color:#555'>Computing energy classification for "
            f"{terrain_label} at sensor ({sensor_x:.2f}, {sensor_y:.2f})...</span>"
        )
        try:
            result = compute_energy_case(
                terrain_category,
                sensor_x,
                sensor_y,
                probe_grid_size=self.probe_grid_size,
                displayed_ray_count=self.displayed_ray_count,
                radial_section_count=self.radial_section_count,
                contour_section_count=self.contour_section_count,
            )
            self.latest_result = result
            self.latest_error = None
            with self.output:
                clear_output(wait=True)
                display(result.figure)
            surface = result.energy_surface
            contour = result.los_result.tangent_contour
            self.status.value = (
                "<span style='color:#176b3a'><b>Ready.</b> "
                f"Upper-horizon rays: {len(contour.rays)}; "
                f"discarded ground candidates: "
                f"{contour.discarded_ground_candidate_count}; "
                f"Clipped green triangles: {surface.reachable_face_count:,}; "
                f"clipped red triangles: {surface.unreachable_face_count:,}; "
                f"green display area: {100.0 * surface.reachable_fraction:.1f}%; "
                f"powered-infeasible vertices: "
                f"{surface.powered_infeasible_vertex_count:,}.</span>"
            )
        except Exception as error:
            self.latest_result = None
            self.latest_error = error
            with self.output:
                clear_output(wait=True)
                print(f"Energy computation failed: {type(error).__name__}: {error}")
            self.status.value = (
                "<span style='color:#b00020'><b>Error:</b> "
                f"{type(error).__name__}: {error}</span>"
            )
        finally:
            self._set_controls_disabled(False)

    def display(self) -> None:
        display(self.widget)


def create_energy_explorer(
    *,
    probe_grid_size: int = 101,
    displayed_ray_count: int = 10,
    radial_section_count: int = 25,
    contour_section_count: int = 96,
) -> EnergyReachabilityExplorer:
    """Create the Stage-6 energy/reachability notebook GUI."""
    return EnergyReachabilityExplorer(
        probe_grid_size=probe_grid_size,
        displayed_ray_count=displayed_ray_count,
        radial_section_count=radial_section_count,
        contour_section_count=contour_section_count,
    )

"""Single configuration source for the Stage-11 validated notebook."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from detection_hazard import (
    DEFAULT_ATTACKER_HAZARD_TIME,
    DEFAULT_DETECTION_HAZARD,
    AttackerHazardTimeParameters,
    DetectionHazardParameters,
)
from discretization_config import DiscretizationConfig
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE, GliderParameters, PhysicalScale
from game_types import AttackerInitialCondition, DefenderAction
from map_geometry import MapBounds
from scenario import MissionPoints, Point3D


@dataclass(frozen=True)
class Stage11Config:
    """Mission, physics, objective, and discretization in one explicit object."""

    terrain_category: str = "centered_cube"
    sensor: Point3D = field(default_factory=lambda: Point3D(5.0, 0.0, 0.0))
    start: Point3D = field(default_factory=lambda: Point3D(-8.0, 0.0, 0.0))
    goal: Point3D = field(default_factory=lambda: Point3D(8.0, 0.0, 0.0))
    graph_bounds: MapBounds = field(
        default_factory=lambda: MapBounds(-8.0, 8.0, -4.0, 4.0),
    )
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE
    glider: GliderParameters = DEFAULT_GLIDER
    detection: DetectionHazardParameters = DEFAULT_DETECTION_HAZARD
    attacker_objective: AttackerHazardTimeParameters = DEFAULT_ATTACKER_HAZARD_TIME
    discretization: DiscretizationConfig = field(default_factory=DiscretizationConfig)
    deterministic_single_candidate_id: int = 25
    random_seed: int = 0

    @property
    def mission_points(self) -> MissionPoints:
        return MissionPoints(sensor=self.sensor, start=self.start, goal=self.goal)

    @property
    def defender_action(self) -> DefenderAction:
        return DefenderAction(self.sensor.as_array())

    @property
    def attacker_initial_condition(self) -> AttackerInitialCondition:
        return AttackerInitialCondition(self.start, self.goal)

    def as_dict(self) -> dict[str, Any]:
        return {
            "terrain_category": self.terrain_category,
            "sensor_map": self.sensor.as_array().tolist(),
            "start_map": self.start.as_array().tolist(),
            "goal_map": self.goal.as_array().tolist(),
            "graph_bounds": asdict(self.graph_bounds),
            "physical_scale": asdict(self.physical_scale),
            "glider": asdict(self.glider),
            "detection": asdict(self.detection),
            "attacker_objective": asdict(self.attacker_objective),
            "discretization": self.discretization.as_dict(),
            "deterministic_single_candidate_id": self.deterministic_single_candidate_id,
            "random_seed": self.random_seed,
        }


DEFAULT_STAGE11_CONFIG = Stage11Config()


__all__ = ["DEFAULT_STAGE11_CONFIG", "Stage11Config"]

# 3D Extension Rework

This directory restarts the 3D implementation from the completed `p1b_4D`
process.  The implemented stages currently contain configuration, geometry,
the symbolic detection model, authoritative local stage cost, and projected
Bellman cost-to-go:

1. one Gaussian terrain surface;
2. terrain-following sensor and goal positions;
3. sensor-ray LOS and occlusion volumes;
4. the terrain contact manifold of tangent sensor rays; and
5. validated visualization/export.
6. 6D-state detection using 3D range, heading, aspect, and radial velocity.
7. dense local `J6D(x,y,h,v,gamma,psi)` with the original 2D phase masks.
8. exact physical grid-node successors with full-segment terrain/LOS checks.
9. heading-state `V4D(x,y,h,psi_in)` and projected `V3D(x,y,h)` cost-to-go.
10. LOS tangent-boundary-surface switching candidates connected to `V4D`
    through an exact physical virtual glide edge, with no endpoint snapping.
11. authoritative physical trajectory extraction by following the selected
    virtual edge and heading-state Bellman policy to the goal intersection.
12. unsnapped 3D continuous replay of the fixed action sequence, including
    state-drift, terrain, LOS, goal, hazard, time, and objective validation.
13. terrain-following two-dimensional Defender search over configurable
    `(x_sensor, y_sensor)` bounds with a fresh Stage 1--7 follower solve at
    every candidate and the unchanged Defender PoD-plus-coverage objective.
14. fixed-sensor objective decomposition and cached-candidate sensitivity to
    complementary Defender PoD/coverage weights.

The detection state order is `(x, y, h, v, gamma, psi)`. Detection
coefficients, additive-hazard PoD, and objective normalization are unchanged
from `p1b_4D`; only the spatial formulas are extended to 3D.

`J6D` is a local stage cost, not a value function. Bellman minimizes its
physical action contract to obtain heading-state cost-to-go, then minimizes
only over incoming heading for the 3D visualization. Switching candidates and
trajectory extraction remain later stages.

The terrain tangent-contact curve and the switching locus are deliberately
distinct, exactly as in 2D: contact points generate the tangent rays, while
switching candidates lie on the resulting `H_LOS(x,y)` boundary surface.

Run from the repository root:

```powershell
.\.venv_p1b\Scripts\python.exe -m 3D_Extension_rework.visualize_geometry
.\.venv_p1b\Scripts\python.exe -m 3D_Extension_rework.visualize_detection
.\.venv_p1b\Scripts\python.exe -m 3D_Extension_rework.visualize_stage_cost
.\.venv_p1b\Scripts\python.exe -m 3D_Extension_rework.visualize_cost_to_go
.\.venv_p1b\Scripts\python.exe -m 3D_Extension_rework.create_bellman_cost_to_go_heatmap
.\.venv_p1b\Scripts\python.exe -m 3D_Extension_rework.visualize_switching
.\.venv_p1b\Scripts\python.exe -m 3D_Extension_rework.visualize_trajectory
.\.venv_p1b\Scripts\python.exe -m 3D_Extension_rework.visualize_continuous_replay
.\.venv_p1b\Scripts\python.exe -m 3D_Extension_rework.create_continuous_replay_animation
.\.venv_p1b\Scripts\python.exe -m 3D_Extension_rework.run_stackelberg
.\.venv_p1b\Scripts\python.exe -m 3D_Extension_rework.run_weight_sensitivity
.\.venv_p1b\Scripts\python.exe -m 3D_Extension_rework.create_stackelberg_3d_outputs
```

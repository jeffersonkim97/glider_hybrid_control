# Representative Final-ID Trajectory Visualizations

One feasible case is selected for each spatial resolution by minimizing the
absolute distance between its Bellman-relative `J_A` error and the aggregate
median error at that resolution. This median-nearest rule avoids selecting only
the best-looking or worst-looking trajectory.

| dx | Aggregate median | Selected terrain / case | Case error | Bellman J_A | DQN J_A | Interactive 3D plot |
|---:|---:|:---|---:|---:|---:|:---|
| 100 m | 4.77% | Terrain 001 / Mixed B | 4.31% | 0.177034 | 0.184657 | [Open HTML](../final_id_001/dx100m_dpsi5deg/trajectory_mixed_b_3d.html) |
| 50 m | 9.64% | Terrain 000 / Mixed B | 9.26% | 0.191454 | 0.209175 | [Open HTML](../final_id_000/dx50m_dpsi5deg/trajectory_mixed_b_3d.html) |
| 25 m | 8.00% | Terrain 000 / Mixed B | 8.19% | 0.189162 | 0.204655 | [Open HTML](../final_id_000/dx25m_dpsi5deg/trajectory_mixed_b_3d.html) |
| 10 m | 7.61% | Terrain 003 / High | 6.54% | 0.164929 | 0.175719 | [Open HTML](../final_id_003/dx10m_dpsi5deg/trajectory_high_3d.html) |

Each plot contains the terrain, start and goal, the selected sensor, its LOS
tangent surface, Bellman and DQN powered/glide segments, and both switching
points.

# Stepped-pyramid full Local-SSE resolution comparison

All rows use heading spacing 5 deg and Chebyshev r=1. Full-SSE times exclude shared scene construction, checkpoint loading, and the exact trajectory replay used only to build the 3D HTML.

| dx | Bellman J_A | DQN J_A | J_A diff | Bellman J_D | DQN J_D | J_D diff | Bellman (s) | DQN (s) | Speedup | 3D HTML |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 m | 0.19055793 | 0.28840276 | 51.35% | 0.01798391 | 0.18465034 | 926.75% | 1128.526 | 324.113 | 3.482x | [open](D:/git/git_research/glider_hybrid_control/3D_0827/figure/phase_4_multi_terrain_dqn/dx10m_dpsi5deg/official_batched_v1/full_sse/full_sse_stepped_pyramid_3d.html) |
| 25 m | 0.18853346 | 0.19725083 | 4.62% | 0.01603268 | 0.02334913 | 45.63% | 98.063 | 56.397 | 1.739x | [open](D:/git/git_research/glider_hybrid_control/3D_0827/figure/phase_4_multi_terrain_dqn/dx25m_dpsi5deg/official_batched_v1/full_sse/full_sse_stepped_pyramid_3d.html) |
| 50 m | 0.30798946 | 0.33822749 | 9.82% | 0.21658091 | 0.23313359 | 7.64% | 70.454 | 71.683 | 0.983x | [open](D:/git/git_research/glider_hybrid_control/3D_0827/figure/phase_4_multi_terrain_dqn/dx50m_dpsi5deg/official_batched_v1/full_sse/full_sse_stepped_pyramid_3d.html) |
| 100 m | 0.33204714 | 0.20227569 | 39.08% | 0.24364011 | 0.03762721 | 84.56% | 29.950 | 37.414 | 0.801x | [open](D:/git/git_research/glider_hybrid_control/3D_0827/figure/phase_4_multi_terrain_dqn/dx100m_dpsi5deg/official_batched_v1/full_sse/full_sse_stepped_pyramid_3d.html) |

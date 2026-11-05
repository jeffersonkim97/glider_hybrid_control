# Zero-sum neighborhood main effect G2 Local-SSE Pilot

Terrain `003`, dx `25 m`, heading `5 deg`, Chebyshev `r=5`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 4478.209 s | n/a | 94 | 449 | `(6.75, 0.0, 0.0)` | 0.48122608 | 0.45905155 |
| Frozen DQN | 1216.410 s | 2023.858 s | 94 | 449 | `(6.75, 0.0, 0.0)` | 0.56283545 | 0.50095435 |

- Selected action match: `True`
- Selected sensor distance: `0.000 m`
- Bellman/DQN first-query ratio: `3.681x`
- Bellman/DQN repeated-query ratio: `2.213x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.08160937`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.08160937`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.

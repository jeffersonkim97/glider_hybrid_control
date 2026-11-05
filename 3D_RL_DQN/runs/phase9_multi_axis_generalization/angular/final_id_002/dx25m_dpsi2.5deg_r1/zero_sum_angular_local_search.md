# Zero-sum angular main effect G2 Local-SSE Pilot

Terrain `002`, dx `25 m`, heading `2.5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 687.659 s | n/a | 14 | 449 | `(6.75, 0.0, 0.0)` | 0.49004300 | 0.46192239 |
| Frozen DQN | 123.863 s | 49.282 s | 14 | 449 | `(6.75, 0.0, 0.0)` | 0.58655494 | 0.51075566 |

- Selected action match: `True`
- Selected sensor distance: `0.000 m`
- Bellman/DQN first-query ratio: `5.552x`
- Bellman/DQN repeated-query ratio: `13.954x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.09651194`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.09651194`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.

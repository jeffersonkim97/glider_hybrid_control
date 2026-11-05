# Zero-sum G2 Local-SSE Pilot

Terrain `001`, dx `50 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 136.966 s | n/a | 22 | 119 | `(6.75, -0.25, 0.0)` | 0.21476679 | 0.07615009 |
| Frozen DQN | 92.913 s | 2.894 s | 24 | 120 | `(6.75, 0.25, 0.0)` | 0.23833251 | 0.09150122 |

- Selected action match: `False`
- Selected sensor distance: `50.000 m`
- Bellman/DQN first-query ratio: `1.474x`
- Bellman/DQN repeated-query ratio: `47.322x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.02606107`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.02606107`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.

# Zero-sum angular main effect G2 Local-SSE Pilot

Terrain `003`, dx `25 m`, heading `10 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 263.762 s | n/a | 14 | 449 | `(6.75, 0.0, 0.0)` | 0.46732044 | 0.44398177 |
| Frozen DQN | 137.449 s | 62.211 s | 14 | 449 | `(6.75, 0.0, 0.0)` | 0.52395156 | 0.48237809 |

- Selected action match: `True`
- Selected sensor distance: `0.000 m`
- Bellman/DQN first-query ratio: `1.919x`
- Bellman/DQN repeated-query ratio: `4.240x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.05663111`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.05663111`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.

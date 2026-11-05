# Zero-sum neighborhood main effect G2 Local-SSE Pilot

Terrain `000`, dx `25 m`, heading `5 deg`, Chebyshev `r=4`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 1305.964 s | n/a | 96 | 445 | `(6.75, -1.0, 0.0)` | 0.22017506 | 0.09169841 |
| Frozen DQN | 586.720 s | 8.608 s | 132 | 235 | `(5.0, 0.75, 0.0)` | 0.34496173 | 0.01204028 |

- Selected action match: `False`
- Selected sensor distance: `247.487 m`
- Bellman/DQN first-query ratio: `2.226x`
- Bellman/DQN repeated-query ratio: `151.717x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.15419748`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.15419748`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.

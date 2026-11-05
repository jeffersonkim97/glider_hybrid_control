# Zero-sum angular main effect G2 Local-SSE Pilot

Terrain `003`, dx `25 m`, heading `2.5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 1614.309 s | n/a | 14 | 449 | `(6.75, 0.0, 0.0)` | 0.48829991 | 0.46187572 |
| Frozen DQN | 223.876 s | 91.229 s | 14 | 449 | `(6.75, 0.0, 0.0)` | 0.53609928 | 0.49287883 |

- Selected action match: `True`
- Selected sensor distance: `0.000 m`
- Bellman/DQN first-query ratio: `7.211x`
- Bellman/DQN repeated-query ratio: `17.695x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.04779938`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.04779938`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.

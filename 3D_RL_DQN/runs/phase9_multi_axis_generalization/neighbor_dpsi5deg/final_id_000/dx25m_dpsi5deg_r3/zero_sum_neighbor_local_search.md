# Zero-sum neighborhood main effect G2 Local-SSE Pilot

Terrain `000`, dx `25 m`, heading `5 deg`, Chebyshev `r=3`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 988.533 s | n/a | 68 | 445 | `(6.75, -1.0, 0.0)` | 0.22017506 | 0.09169841 |
| Frozen DQN | 291.268 s | 5.116 s | 60 | 449 | `(6.75, 0.0, 0.0)` | 0.23827701 | 0.05919956 |

- Selected action match: `False`
- Selected sensor distance: `100.000 m`
- Bellman/DQN first-query ratio: `3.394x`
- Bellman/DQN repeated-query ratio: `193.237x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.04694912`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.04694912`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.

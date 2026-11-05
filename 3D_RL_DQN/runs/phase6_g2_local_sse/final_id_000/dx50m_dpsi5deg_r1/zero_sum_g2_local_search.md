# Zero-sum G2 Local-SSE Pilot

Terrain `000`, dx `50 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 79.051 s | n/a | 14 | 118 | `(6.75, -0.75, 0.0)` | 0.22651359 | 0.07232373 |
| Frozen DQN | 67.211 s | 0.644 s | 14 | 120 | `(6.75, 0.25, 0.0)` | 0.32119401 | 0.17755038 |

- Selected action match: `False`
- Selected sensor distance: `100.000 m`
- Bellman/DQN first-query ratio: `1.176x`
- Bellman/DQN repeated-query ratio: `122.712x`
- Exact timing basis: `sum_of_per_response_runtime_across_resumed_runs`
- Exact responses reused from disk: `13`
- DQN-selected sensor exact `J_A` gap: `0.12696084`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.12696084`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.

# Zero-sum G2 Local-SSE Pilot

Terrain `000`, dx `25 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 329.676 s | n/a | 24 | 445 | `(6.75, -1.0, 0.0)` | 0.22017506 | 0.09169841 |
| Frozen DQN | 72.619 s | 0.967 s | 12 | 325 | `(5.75, 0.0, 0.0)` | 0.21756703 | 0.03080222 |

- Selected action match: `False`
- Selected sensor distance: `141.421 m`
- Bellman/DQN first-query ratio: `4.540x`
- Bellman/DQN repeated-query ratio: `340.873x`
- Exact timing basis: `sum_of_per_response_runtime_across_resumed_runs`
- Exact responses reused from disk: `22`
- DQN-selected sensor exact `J_A` gap: `0.03155182`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.03155182`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.

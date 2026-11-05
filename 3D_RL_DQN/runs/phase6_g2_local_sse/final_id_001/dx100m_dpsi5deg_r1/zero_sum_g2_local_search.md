# Zero-sum G2 Local-SSE Pilot

Terrain `001`, dx `100 m`, heading `5 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 37.532 s | n/a | 14 | 27 | `(6.5, -0.5, 0.0)` | 0.22017439 | 0.09131489 |
| Frozen DQN | 46.073 s | 0.265 s | 13 | 4 | `(3.5, 0.5, 0.0)` | 0.19615335 | 0.05290142 |

- Selected action match: `False`
- Selected sensor distance: `316.228 m`
- Bellman/DQN first-query ratio: `0.815x`
- Bellman/DQN repeated-query ratio: `141.414x`
- Exact timing basis: `sum_of_per_response_runtime_across_resumed_runs`
- Exact responses reused from disk: `14`
- DQN-selected sensor exact `J_A` gap: `0.01538865`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.01538865`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.

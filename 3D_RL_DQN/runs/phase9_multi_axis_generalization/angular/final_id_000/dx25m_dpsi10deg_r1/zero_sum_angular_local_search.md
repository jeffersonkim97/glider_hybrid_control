# Zero-sum angular main effect G2 Local-SSE Pilot

Terrain `000`, dx `25 m`, heading `10 deg`, Chebyshev `r=1`.
No training or fine-tuning was performed.

| Method | First query | Repeated query | Evaluations | Final action | Final sensor | J_A | J_D |
|:---|---:|---:|---:|---:|:---|---:|---:|
| Exact Bellman | 148.531 s | n/a | 24 | 445 | `(6.75, -1.0, 0.0)` | 0.22619274 | 0.09199275 |
| Frozen DQN | 103.842 s | 1.711 s | 22 | 446 | `(6.75, -0.75, 0.0)` | 0.25565835 | 0.12332028 |

- Selected action match: `False`
- Selected sensor distance: `25.000 m`
- Bellman/DQN first-query ratio: `1.430x`
- Bellman/DQN repeated-query ratio: `86.785x`
- Exact timing basis: `single_process_wall_runtime`
- Exact responses reused from disk: `0`
- DQN-selected sensor exact `J_A` gap: `0.03191338`
- Primary leader objective: `attacker_objective` (zero-sum)
- Certified delta at the DQN-selected sensor: `0.03191338`
- One-sided feasible-response inequality verified: `True`

The returned frozen-DQN action is a delta-approximate r-local Stackelberg solution for the finite zero-sum game. The certificate uses one exact best response at the returned action; J_D is reported only as a physical diagnostic.

The DQN warm-up pass, shared scene construction, and checkpoint load are excluded from benchmark timings.

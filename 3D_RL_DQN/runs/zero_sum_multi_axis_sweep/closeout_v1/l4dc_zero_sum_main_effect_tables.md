# L4DC Zero-Sum Main-Effect Tables

## Experiment contract

- Finite zero-sum local Stackelberg game: the Attacker minimizes `J_A`, and
  the Defender maximizes the same value `J_A`.
- Leader-objective field: `attacker_objective`.
- Evaluation terrains: final-ID random-box terrains 000--003 (`n=4`).
- Frozen condition-specific DQN checkpoints; no new training was performed.
- Times are single-process wall-clock seconds. Training time is not included.
- `DQN first` includes the first measured Local-SSE query after excluded setup
  and warm-up; `DQN repeated` is the repeated measured query.
- Relative gap is the zero-sum single-point certificate gap
  `(U_tilde - U) / abs(U)` at the DQN-returned Defender action.
- P90 uses the `higher` percentile convention. With four terrains, P90 is the
  observed maximum.
- Every reported source case passed the one-sided inequality check.
- This is a main-effect design, not a full factorial experiment.

## Spatial discretization main effect

Fixed conditions: `dpsi=5 deg`, Chebyshev neighborhood radius `r=1`.

| dx [m] | Bellman median / P90 [s] | DQN first median / P90 [s] | DQN repeated median / P90 [s] | Relative gap median / P90 | Action match | One-sided pass |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 30.50 / 37.53 | 43.56 / 46.23 | 0.25 / 0.41 | 5.26% / 17.04% | 3/4 | 4/4 |
| 50 | 108.01 / 165.80 | 70.65 / 92.91 | 2.06 / 3.37 | 15.60% / 65.37% | 1/4 | 4/4 |
| 25 | 570.35 / 835.66 | 162.10 / 217.01 | 54.38 / 107.82 | 15.93% / 16.96% | 2/4 | 4/4 |
| 10 | 7,960.17 / 14,503.43 | 2,021.07 / 8,166.32 | 3,783.07 / 19,628.02 | 22.91% / 28.15% | 0/4 | 4/4 |

## Angular discretization main effect

Fixed conditions: `dx=25 m`, Chebyshev neighborhood radius `r=1`.

| dpsi [deg] | Cartesian-heading states | Bellman median / P90 [s] | DQN first median / P90 [s] | DQN repeated median / P90 [s] | Relative gap median / P90 | Action match | One-sided pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1,621,620 | 206.15 / 277.38 | 120.65 / 204.12 | 12.12 / 62.21 | 13.19% / 16.08% | 3/4 | 4/4 |
| 5 | 3,243,240 | 570.35 / 835.66 | 162.10 / 217.01 | 54.38 / 107.82 | 15.93% / 16.96% | 2/4 | 4/4 |
| 2.5 | 6,486,480 | 1,150.98 / 1,865.05 | 127.44 / 223.88 | 42.96 / 91.23 | 19.96% / 44.48% | 2/4 | 4/4 |

## Defender-neighborhood main effect

Fixed conditions: `dx=25 m`, `dpsi=5 deg`.

| Chebyshev r | Bellman median / P90 [s] | DQN first median / P90 [s] | DQN repeated median / P90 [s] | Relative gap median / P90 | Action match | One-sided pass |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 570.35 / 835.66 | 162.10 / 217.01 | 54.38 / 107.82 | 15.93% / 16.96% | 2/4 | 4/4 |
| 2 | 1,024.09 / 1,663.10 | 349.01 / 628.12 | 166.57 / 846.52 | 15.93% / 80.83% | 3/4 | 4/4 |
| 3 | 1,565.51 / 2,573.06 | 487.38 / 904.28 | 403.49 / 1,268.18 | 15.93% / 24.54% | 3/4 | 4/4 |
| 4 | 1,980.44 / 3,634.95 | 756.31 / 1,245.51 | 778.83 / 1,835.33 | 15.93% / 80.83% | 3/4 | 4/4 |
| 5 | 2,625.10 / 4,478.21 | 995.82 / 1,681.98 | 1,350.96 / 2,541.21 | 15.93% / 80.83% | 3/4 | 4/4 |

## Source artifacts

- Spatial source: `zero_sum_spatial_sweep/closeout_v1/zero_sum_spatial_sweep.json`.
- Angular and neighborhood source:
  `zero_sum_multi_axis_sweep/closeout_v1/zero_sum_multi_axis_sweep.json`.
- Case-level source hashes are stored in the corresponding aggregate JSON and
  CSV files.

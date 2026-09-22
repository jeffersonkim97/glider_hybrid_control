# R1 - Cartesian discretization sweep

Exact Bellman local SSE against its tabular Q-learning approximation, with the
spatial step `dx` as the only independent variable.

## Conditions

| | |
|---|---|
| terrain | `centered_cube_half_height` - one box, footprint x,y in [-2, 2], height 2.0, base 0 |
| lattice bounds | x [-8, 8], y [-4, 4], altitude [0, 5] map units (1 map unit = 100 m) |
| start / goal | (-8, 0, 0) / (8, 0, 0) |
| heading spacing | 5 deg (72 bins), held fixed |
| Defender neighbourhood | Chebyshev radius r = 1 (8 neighbours) |
| Defender seed | nearest lattice point to (5, 0) |
| Defender region | x from the terrain's far edge to goal_x - 1, full y span, z = 0 |
| attacker objective | `J_A = 0.5*(H/1.0) + 0.5*(T/221.23893805309734 s)`, minimised |
| defender objective | `J_D = 1 - exp(-H)`, maximised |
| hazard | radar form, rate proportional to 1/R^4, integrated along each edge |
| Q-learning | tabular, cost form, `Q <- Q + a[c + g min Q' - Q]`, gamma = 1 |
| | alpha 0.5, epsilon 1.0 -> 0.05 decaying 0.999 per episode, max_steps 400, seed 0 |
| episodes | 20,000 at the first Defender position, 4,000 warm-started at each later one |
| switching-point ranking | by rollout, scored with the original evaluator, not by learned value |
| acceptance | `RL_threshold = 0.9`, required of **both** strategies |
| repetitions | 1 per point, single seed. No variance estimate. |

Accuracy is measured against a fixed opponent, never across different Defender
positions:

```
attacker_accuracy = J_A_exact(at the learner's own d) / J_A_learned     in (0, 1]
defender_accuracy = true J_D(at the learner's d)      / J_D_exact(d*)   in (0, 1]
```

## Result 1 - full local SSE, both solvers, Defender search included

| dx | exact s | ev | exact d* | exact J_D | RL s | ev | RL d* | true J_D at RL d* | RL/exact | attacker | defender | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 m | 21.2 | 9 | (5.000, 0.000) | 0.034293 | 38.1 | 9 | (5.000, 0.000) | 0.034293 | 1.80x | 1.0000 | 1.0000 | ACCEPT |
| 50 m | 27.8 | 9 | (5.000, 0.000) | 0.025019 | 236.7 | 14 | (5.500, -0.500) | 0.018154 | 8.53x | 0.9274 | 0.7256 | REJECT |
| 25 m | 75.1 | 12 | (4.750, 0.000) | 0.025687 | 582.0 | 12 | (4.750, 0.000) | 0.025687 | 7.75x | 0.9465 | 1.0000 | ACCEPT |
| 12.5 m | 374.5 | 12 | (4.875, 0.000) | 0.028877 | 1638.8 | 14 | (5.125, 0.125) | 0.019394 | 4.38x | 0.9305 | 0.6716 | REJECT |
| 10 m | 985.4 | 18 | (5.300, 0.000) | 0.024689 | 1851.9 | 14 | (5.100, -0.100) | 0.017969 | 1.88x | 0.9344 | 0.7278 | REJECT |
| 5 m | 22984.7 | 48 | (5.500, -0.100) | 0.019531 | 5045.3 | 14 | (5.050, -0.050) | 0.013930 | 0.22x | 0.9579 | 0.7132 | REJECT |

Measured readings:

- RL/exact falls 1.80 -> 8.53 -> 7.75 -> 4.38 -> 1.88 -> 0.22; the two cross between 10 m and 5 m.
- Exact's evaluation count rises with refinement (9, 9, 12, 12, 18, 48); RL's does not (9, 14, 12, 14, 14, 14).
- Exact's chosen d* moves 0, 0, 1, 1, 3, 10 lattice steps from the seed; RL's moves 0, 1, 1, 1, 1, 1.
- attacker_accuracy stays in [0.9274, 1.0000] and clears the threshold at every dx.
- defender_accuracy is 1.0000 exactly when RL lands on the same d* as exact (100 m, 25 m) and
  0.6716-0.7278 when it does not. It does not vary monotonically with dx.
- Every d* found lies between x = 4.75 and x = 5.50. Measured separately at 50 m, d = (7, 0)
  scores J_D = 0.216615 against 0.025019 at (5, 0), a factor of 8.7, so the J_D values reported
  here are those of a neighbourhood the r = 1 climb did not leave, not of the region's best sensor.

## Result 2 - one best response at one fixed Defender position (5, 0, 0)

Search length removed, so this isolates the cost of a single solve.

| dx | N_S | N_S*A | candidates | peak RSS MiB | exact s | LOS | filter | Bellman | powered | RL s | train | query | RL/exact | attacker acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 m | 66,096 | 4,758,912 | 189 | 23 | 2.21 | 0.86 | 1.13 | 0.19 | 0.03 | 4.63 | 2.66 | 1.98 | 2.1x | 1.0000 |
| 50 m | 444,312 | 31,990,464 | 465 | 216 | 2.98 | 0.79 | 1.10 | 0.89 | 0.20 | 20.99 | 19.03 | 1.96 | 7.0x | 0.9481 |
| 25 m | 3,243,240 | 233,513,280 | 1,231 | 904 | 6.45 | 0.80 | 1.16 | 4.21 | 0.28 | 72.95 | 70.81 | 2.14 | 11.3x | 0.9560 |
| 20 m | 6,216,912 | 447,617,664 | 1,392 | 1215 | 8.74 | 0.77 | 1.14 | 6.46 | 0.36 | 97.92 | 95.57 | 2.35 | 11.2x | 0.9303 |
| 12.5 m | 24,752,520 | 1,782,181,440 | 3,208 | 2102 | 30.46 | 0.79 | 1.19 | 28.14 | 0.32 | 165.72 | 162.96 | 2.76 | 5.4x | 0.9351 |
| 10 m | 47,886,552 | 3,447,831,744 | 3,963 | 2814 | 54.57 | 0.79 | 1.18 | 52.19 | 0.33 | 211.08 | 207.84 | 3.24 | 3.9x | 0.9450 |
| 5 m | 375,824,232 | 27,059,344,704 | 7,319 | 9898 | 445.82 | 0.79 | 1.33 | 442.72 | 0.20 | 428.32 | 423.31 | 5.01 | 1.0x | 0.9596 |

Measured readings:

- Stages 1, 2 and 4 are flat in dx (LOS 0.77-0.86 s, filter 1.10-1.33 s, powered 0.03-0.36 s).
  All of exact's growth is the Bellman sweep, which goes from 8.7% to 99.3% of the total.
- RL's query is likewise flat (1.96-5.01 s); all of its growth is training.
- Exact's cost per state-action pair settles at 15.8-17.1 ns below 12.5 m.
- At this fixed position RL/exact reaches 1.0 at 5 m, against 0.22 for the full SSE at the same
  dx; the difference is the evaluation count, which this measurement holds at one.
- Peak RSS at 5 m is 9,898 MiB with N_S = 375,824,232.

## Files

```
data/sse_sweep_summary.json                6-point summary of Result 1
data/spatial_sweep_fixed_defender.json     7-point record of Result 2 (includes 20 m)
data/1_exact_vs_qlearning_<dx>_*.json      per-resolution record: conditions, both
                                           solutions with trajectories in state ids and
                                           map coordinates, timings, per-position
                                           Defender evaluations, comparison block
figure/1_*.png .html                       3-D scene: terrain, LOS tangent surface,
                                           switching-point J_A heat map, both solutions
figure/1-1_*_side_view.*                   x-z projection
figure/1-2_*_top_view.*                    x-y projection
```

Figures fold y onto its positive half (`|y| folded` in the title). The terrain is
symmetric about y = 0, so a reflected solution is itself a solution with the same cost;
without folding, the two solvers' answers overlap and cannot both be read.

## Not measured

- Variance over seeds. One run per point.
- J_D at x = 7 for any dx other than 50 m.
- Any dx below 5 m; the extrapolated memory wall for the dense value array was not reached.
- Heading spacing as an independent variable, held at 5 deg throughout.
- Why one exact 50 m rerun took 4.21 s per evaluation against 3.08-3.19 s in three others.

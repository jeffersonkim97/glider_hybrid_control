# 3D weight-tradeoff example — meeting notes

## One-sentence claim

Extending the spatial state from `(z, h)` to `(x, y, h)` allows the attacker
to trade mission time for a larger lateral terrain-shadow detour, a strategy
that the original vertical-plane model cannot represent.

## What the figure shows

All three paths use the same Gaussian hill and the same sensor position. Only
the attacker's PoD/time objective weights change.

| Policy | Minimum cross-range | Mission PoD | Mission time |
|---|---:|---:|---:|
| Detection priority (`wPoD=0.75`) | `-850 m` | `2.97%` | `114.8 s` |
| Balanced (`wPoD=0.50`) | `-800 m` | `3.67%` | `113.4 s` |
| Time priority (`wPoD=0.25`) | `-750 m` | `5.57%` | `111.7 s` |

Increasing the detection weight therefore produces a wider lateral detour,
lower final PoD, and a modest time penalty. The top-down panel makes the
route separation visible; the PoD-history panel verifies the resulting
detection trade-off.

## Suggested 20-second narration

> In the vertical-plane model the vehicle cannot move around terrain; it can
> only pass above it. In the 3D extension, heading becomes an additional
> control and the Bellman state is `(x, y, h)`. With the same terrain and
> sensor, increasing the detection weight moves the optimal route farther
> around the hill—from 750 to 850 metres cross-range—reducing mission PoD
> from 5.57% to 2.97% at a roughly three-second time cost.

## Scope statement

This is a coarse-grid qualitative prototype based on the saved legacy 3D
Bellman results. It demonstrates the strategic value of the added lateral
dimension, but it is not yet synchronized with the current physical
successor-grid and exact segment-certification implementation in `p1b_4D`.

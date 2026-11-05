# Zero-Sum Multi-Terrain Spatial Sweep

Frozen DQN checkpoints; no training or fine-tuning. Median and P90 are over final-ID terrains 000--003.

| dx | Bellman first med / P90 | DQN first med / P90 | DQN repeated med / P90 | delta med / P90 | relative delta med / P90 | action match |
|---:|:---|:---|:---|:---|:---|---:|
| 100 m | 30.49 / 37.53 s | 43.56 / 46.23 s | 0.252 / 0.414 s | 0.013605 / 0.038497 | 5.26% / 17.04% | 3/4 |
| 50 m | 108.01 / 165.80 s | 70.65 / 92.91 s | 2.061 / 3.373 s | 0.068466 / 0.126961 | 15.60% / 65.37% | 1/4 |
| 25 m | 570.35 / 835.66 s | 162.10 / 217.01 s | 54.383 / 107.824 s | 0.051679 / 0.081609 | 15.93% / 16.96% | 2/4 |
| 10 m | 7960.17 / 14503.43 s | 2021.06 / 8166.32 s | 3783.067 / 19628.022 s | 0.073114 / 0.114150 | 22.91% / 28.15% | 0/4 |

The delta certificate is local and instance-specific. Action mismatch and the exact-search outcome gap are diagnostics, not violations of the approximate local-solution theorem.

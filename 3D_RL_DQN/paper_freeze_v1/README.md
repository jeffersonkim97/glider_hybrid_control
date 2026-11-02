# Phase 5–7 Paper Reproducibility Freeze v1

This directory freezes the exact model configurations and artifact checksums used by the completed Phase 5–7 controlled experiments.
Historical run summaries and checkpoint payloads are the configuration source of truth; current code defaults are not substituted.

## Frozen models

| dx [m] | dpsi [deg] | actions | parameters | episodes | environment steps | optimization steps | learning rate | checkpoint SHA-256 prefix |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 100 | 5 | 72 | 2,147,624 | 10,000 | 40,732 | 5,935 | 3.0e-04 | `030b9d165d60` |
| 50 | 5 | 72 | 2,147,624 | 10,000 | 74,917 | 12,002 | 3.0e-04 | `c2f0898cf75d` |
| 25 | 10 | 36 | 2,138,372 | 10,000 | 157,040 | 24,068 | 3.0e-04 | `39bd586763a5` |
| 25 | 5 | 72 | 2,147,624 | 10,000 | 143,804 | 23,887 | 3.0e-04 | `cea13352ab86` |
| 25 | 2.5 | 144 | 2,166,128 | 10,000 | 93,346 | 15,919 | 3.0e-04 | `45e877518aa7` |
| 10 | 5 | 72 | 2,147,624 | 10,000 | 355,793 | 59,641 | 3.0e-04 | `1d3dad6004a3` |

## Timing contract

- Bellman first solve: main exact Local-SSE wall time after shared scene setup.
- DQN first solve: first Local-SSE query with empty problem caches after scene/checkpoint setup.
- DQN repeated query: secondary identical-query latency with retained caches.
- Offline training, scene construction, checkpoint loading, CUDA initialization, and post-hoc exact certification are excluded from online first-solve time.

## Artifact policy

- Final evaluation checkpoints are stored through Git LFS.
- Phase 5–7 JSON/CSV/Markdown, logs, and static PNG outputs are frozen.
- Phase-6 representative 3-D HTML outputs are frozen.
- The 84 regenerable Phase-5 HTML files (~389 MiB) are intentionally omitted; their raw JSON inputs and generation code are frozen.

## Files

- `reproducibility_manifest.json`: full configurations, optimizer parameter groups, measured training counts, software/hardware environment, and source hashes.
- `hyperparameters.csv`: paper/appendix-ready per-condition hyperparameter table.
- `artifact_checksums.csv`: SHA-256 index of every frozen input and output artifact.

## Scope note

`dx` and `dpsi` are computational/MDP discretization conditions. Chebyshev radius `r` is an outer Defender Local-SSE search condition and is not a DQN training hyperparameter.

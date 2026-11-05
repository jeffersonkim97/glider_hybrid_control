# L4DC Zero-Sum Result Sanity Audit

## Outcome

- Integrity, source-hash, zero-sum contract, certificate, and aggregate checks: `PASS`
- Referenced cases: `48`; unique source files: `40`
- Aggregate recomputation failures: `0`
- Timing assessment: `CAUTION` (10 identical-workload repeated queries were slower than first queries)

## Metric distinction required in the paper

The existing `relative gap` is the zero-sum single-point certificate error at the DQN-returned Defender action. It is not the direct loss relative to the exact Bellman-selected action. The latter is reported below as exact decision regret. Both metrics should be retained.

### Spatial resolution

| dx [m] | Certificate median | Certificate P90 | Decision-regret median | Decision-regret P90 |
|---:|---:|---:|---:|---:|
| 10 | 22.91% | 28.15% | 6.99% | 21.14% |
| 25 | 15.93% | 16.96% | 7.76% | 18.28% |
| 50 | 15.60% | 65.37% | 0.64% | 14.25% |
| 100 | 5.26% | 17.04% | 0.00% | 17.90% |

### Angular resolution

| dpsi [deg] | Certificate median | Certificate P90 | Decision-regret median | Decision-regret P90 |
|---:|---:|---:|---:|---:|
| 2.5 | 19.96% | 44.48% | 7.94% | 24.20% |
| 5 | 15.93% | 16.96% | 7.76% | 18.28% |
| 10 | 13.19% | 16.08% | 0.00% | 1.08% |

### Neighborhood radius

| r | Certificate median | Certificate P90 | Decision-regret median | Decision-regret P90 |
|---:|---:|---:|---:|---:|
| 1 | 15.93% | 16.96% | 7.76% | 18.28% |
| 2 | 15.93% | 80.83% | 0.00% | 13.36% |
| 3 | 15.93% | 24.54% | 0.00% | 13.10% |
| 4 | 15.93% | 80.83% | 0.00% | 13.36% |
| 5 | 15.93% | 80.83% | 0.00% | 13.36% |

## Certificate-gap outliers

| Axis | Value | Terrain | Certificate gap | Decision regret | Bellman action | DQN action |
|:---|---:|---:|---:|---:|---:|---:|
| r_neighbor | 2 | 0 | 80.83% | 13.36% | 445 | 235 |
| r_neighbor | 4 | 0 | 80.83% | 13.36% | 445 | 235 |
| r_neighbor | 5 | 0 | 80.83% | 13.36% | 445 | 235 |
| dx_m | 50 | 0 | 65.37% | 14.25% | 118 | 120 |
| dpsi_deg | 2.5 | 0 | 44.48% | 15.88% | 445 | 449 |
| dx_m | 10 | 0 | 28.15% | 13.42% | 2688 | 1859 |
| dx_m | 10 | 2 | 25.34% | 0.30% | 2698 | 2697 |
| r_neighbor | 3 | 0 | 24.54% | 13.10% | 445 | 449 |

The repeated `80.83%` radius P90 is not an aggregation error. Terrain 000 returns DQN action 235 at r=2, r=4, and r=5, producing the same certificate gap. At r=3 it returns action 449 and the gap is 24.54%. The radius median remains 15.93% because terrains 001--003 supply the same middle two values (14.90% and 16.96%) for every radius.

The dx=50 m P90 65.37% is also a real Terrain-000 source value, not a table error. Its direct exact decision regret is 14.25%.

## Timing audit

- Bellman timing basis: `{'sum_of_per_response_runtime_across_resumed_runs': 21, 'single_process_wall_runtime': 19}`
- The resumed-run Bellman value is the sum of stored per-response runtimes, whereas uncached runs use single-process wall time.
- Maximum wall-time versus best-response-kernel-sum difference: `3.41%`.
- All first/repeated DQN pairs evaluated the same Defender action set. Warm-up order is not stored, so exact order equality cannot be audited.
- Repeated query slower than first query: `10` of `40` unique cases.

| dx | dpsi | r | Terrain | First [s] | Repeated [s] | Ratio | Evaluations |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | 1 | 3 | 8166.32 | 19628.02 | 2.40 | 32 |
| 10 | 5 | 1 | 2 | 2352.21 | 4733.30 | 2.01 | 28 |
| 10 | 5 | 1 | 1 | 1689.92 | 2832.84 | 1.68 | 17 |
| 25 | 5 | 5 | 3 | 1216.41 | 2023.86 | 1.66 | 94 |
| 25 | 5 | 5 | 1 | 1681.98 | 2541.21 | 1.51 | 190 |
| 25 | 5 | 4 | 1 | 1245.51 | 1835.33 | 1.47 | 159 |
| 25 | 5 | 3 | 1 | 904.28 | 1268.18 | 1.40 | 117 |
| 25 | 5 | 2 | 1 | 628.12 | 846.52 | 1.35 | 88 |
| 25 | 5 | 4 | 3 | 925.91 | 1211.45 | 1.31 | 68 |
| 25 | 5 | 5 | 2 | 636.68 | 678.06 | 1.06 | 94 |

Because the evaluated action sets are identical, the repeated-time inflation is not caused by additional Local-SSE action evaluations. The stored data do not contain sufficient system telemetry to attribute it conclusively to memory pressure, thermal behavior, or another runtime-state effect.

## Required treatment before a central paper claim

1. Rename the current gap column to `certificate gap`.
2. Add exact `decision regret` as a separate performance column.
3. Normalize Bellman timing to best-response kernel-sum time for all cases, or disclose the mixed wall/resumed timing basis. The observed basis difference is at most 3.42%.
4. Do not describe the current repeated DQN timing as a stable steady-state estimate. Use both first/repeated observations, or run a controlled timing-only replication for the flagged conditions.
5. State that P90 with four terrains is the observed maximum and is therefore controlled by one terrain.

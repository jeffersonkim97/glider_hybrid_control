# DQN workspace

All new DQN environments, models, replay buffers, training code, evaluation code,
tests, notebooks, checkpoints, and DQN-specific figures belong in this directory.
`3D_0827` remains the source of truth for the existing terrain, transition,
hazard, objective, Bellman, and Stackelberg implementations.

The directory names begin with digits and therefore cannot be used as normal
Python package names.  In a notebook or script in this directory, use:

```python
import project_paths  # configures both sibling source directories

from P1b_condition import ComputationCondition, build_scene
from P1b_RL_approximation import AttackerMDP
from glider_gym_env import TerrainGlideEnv
```

The workspace virtual environment also has a `glider_hybrid_control_paths.pth`
file that exposes both directories to restarted kernels and Python processes,
regardless of their working directory.  `project_paths` is the portable fallback
for a new environment or clone.

Imports in the other direction use the future DQN module's filename directly.
Module filenames must therefore be unique across `3D_0827` and `3D_RL_DQN`.

## Phase 16.1 attacker best-response boundary

`attacker_br_problem.py` is the solver-independent entry point for new work.  It
delegates transitions, terminals, hazards, and costs to the authoritative
`3D_0827` implementation and exposes the existing Bellman and Tabular-Q solvers
through one call:

```python
import project_paths

from attacker_br_problem import AttackerBRProblem, solve_attacker_br
from P1b_condition import ComputationCondition, build_scene
from P1b_RL_approximation import default_sensor

scene = build_scene(ComputationCondition(spatial_resolution_m=100.0))
problem = AttackerBRProblem(scene, default_sensor(scene))
result = solve_attacker_br(problem, "bellman")
```

The common problem deliberately has no Local-SSE neighbor radius and no
Gymnasium dependency.  `glider_gym_env.py` predates this boundary and contains
later-phase observation, reward, and fixed-action decisions; review it against
the common problem during the observation/DQN phases rather than treating it as
the Phase 16.1 core.

## Phase 16.2 terrain-aware observation

`terrain_observation.py` builds the deterministic structured observation used
by later DQN phases.  It sits on `AttackerBRProblem` and does not alter the
authoritative transition, feasibility, terminal, hazard, or objective logic.
The approved condition-specific schema uses a heading-aligned 2000 m × 2000 m
window sampled on a fixed 21 × 21 grid:

```python
from terrain_observation import ObservationConfig, TerrainObservationBuilder

config = ObservationConfig(
    local_window_extent_m=2000.0,
    local_window_shape_cells=(21, 21),
    local_window_orientation="heading_aligned",
)
builder = TerrainObservationBuilder(problem, config)
raw = builder.build(state_id)
tensor = builder.to_tensor_ready(raw)

assert tensor.scalar.shape == (8,)
assert tensor.spatial.shape == (7, 21, 21)
```

The seven spatial channels are terrain clearance, five hazard sufficient
statistics, and an explicit domain-validity mask.  Terrain/scenario IDs,
discretization parameters, and Local-SSE radius are not neural inputs.  Run the
full Phase 16.2 validation and regenerate its manifest/figures with:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase2_validation.py
& .\.venv_p1b\Scripts\python.exe -m unittest discover -s .\3D_RL_DQN -p test_terrain_observation.py -v
```

## Phase 16.3 single-terrain DQN

`phase3_single_terrain.py` trains the approved condition-specific DQN on the
canonical centered-cube case. The fixed 72-action catalog is derived from the
authoritative transition graph, and feasible-action masks are applied during
exploration, greedy evaluation, and target maximization. Rewards are the
negative authoritative glide-stage costs with `gamma=1.0`; final policies are
scored through the existing authoritative `J_A` evaluator.

Run the short CUDA pipeline check without making an official pass/fail claim:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase3_single_terrain.py --smoke --device cuda
```

Run the approved three-seed, 20,000-episode experiment on CUDA:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase3_single_terrain.py
```

To repeat the same single-terrain comparison at another condition while
preserving the canonical output, pass the condition explicitly. For example:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase3_single_terrain.py --spatial-resolution-m 25 --heading-spacing-deg 5
```

Noncanonical outputs use a condition-tagged subdirectory such as
`dx25m_dpsi5deg/` for both figures and checkpoints.

The official manifest and plots are written under
`3D_0827/figure/phase_3_single_terrain_dqn/`; compatible best/final checkpoints
are written under `3D_RL_DQN/checkpoints/phase3_single_terrain/`. The `smoke/`
subdirectories keep pipeline-check outputs separate from the official run.

Run the Phase 16.3 unit tests with:

```powershell
& .\.venv_p1b\Scripts\python.exe -m unittest discover -s .\3D_RL_DQN -p test_dqn_phase3.py -v
```

## Master Phase 16 notebook

`Phase16_DQN_Terrain_Generalization.ipynb` is the user-facing entry point for
Phases 16.1 through 16.4. Its first code cell exposes four independent Boolean
switches:

```python
RUN_PHASE_16_1 = False
RUN_PHASE_16_2 = False
RUN_PHASE_16_3 = False
RUN_PHASE_16_4 = False
```

Set only the phases to run to `True`, then use **Run All**. The notebook imports
the tested Python modules rather than duplicating solver or training logic. A
CUDA-enabled `.venv_p1b` kernel is required for both smoke and official DQN
runs. Phase 16.4 resume is enabled independently with:

```python
REQUIRE_CUDA_FOR_TRAINING = True
RESUME_PHASE_16_4 = True
```

## Phase 16.4 multi-terrain generalized DQN

`phase4_dataset.py` freezes the approved development dataset. One shared DQN is
trained on three simple terrain families and nine sensing locations per terrain
(27 training scenarios). Two terrain families are held out from replay and
weight updates and evaluated at the same nine sensing locations (18 validation
scenarios). Terrain and scenario IDs remain metadata only.

The official condition and training budget are:

```text
dx = dy = dh = 25 m
dpsi = dgamma = 5 degrees
60,000 episodes per seed
seeds = 0, 1, 2
checkpoint interval = 1,500 episodes
full 45-case evaluation interval = 6,000 episodes
parallel episode batch = 8
optimizer updates per vector step = 1 (maximum UTD = 1/8)
device = CUDA (required)
```

The trainer samples a terrain uniformly and then a sensing location uniformly.
`phase4_replay.py` qualifies each state ID by scenario so identical lattice IDs
from different terrain/hazard problems cannot collide. A resumable checkpoint
contains the online and target networks, optimizer, replay contents, RNG states,
episode counters, evaluation history, and the current validation-selected best
model.

The vectorized trainer advances eight independent episodes from one policy
snapshot, batches their action observations/forward pass, and performs one
replay update per vector step. The target network remains synchronized by
environment-step count every 1,000 transitions. This changes the optimizer
update-to-data ratio from the earlier serial pilot, so batched seeds are stored
as a separate experiment and are never mixed with the serial seed artifacts.
The full evaluation is deferred to every 6,000 episodes, while lightweight
resumable checkpoints remain at every 1,500 episodes.

Phase 16.4 evaluation advances all switching-state candidates in neural-network
batches of 1,024. Training contexts retain bounded 2,048-entry observation LRU
caches, while validation terrains are built and released one terrain at a time.
The terrain-height query cache is shared by scenarios on the same terrain and
bounded at 200,000 entries. Deterministic MDP cost/action rows are cleared after
each evaluation case. These controls preserve the policy and authoritative
trajectory scoring while preventing evaluation caches from growing without a
bound. The scalar CPU reference backend remains available for parity checks and
non-CUDA runs.

For the approved box-based Phase 16.4 terrains, `aabb_gpu_observation.py`
provides an exact batched CUDA backend. It uses float64 strict-interior AABB
slab tests with the same tolerance as the authoritative terrain queries and
retains the existing observation schema. CPU/CUDA parity is tested across all
five Phase 16.4 terrain categories. At 25 m, the measured full 45-case
checkpoint evaluation fell from 2,133 s to about 396 s; a 3,000-episode
continuation from episode 3,000 to 6,000 completed its two checkpoint cycles in
1,590 s. These measurements concern implementation performance and do not
change the approved quality criterion.

A warmed steady-state profile after the CUDA change measured 5.62 s for one
representative stepped-pyramid case: CPU feasible-action masks used 3.92 s,
CPU transitions used 1.11 s, GPU observation construction used 0.445 s, and
network forward passes used 0.078 s. CUDA observation construction is therefore
no longer the steady-state evaluation bottleneck. The next optimization target
is the authoritative graph lookup used by action masking and transitions.

The graph lookup now uses bounded terrain-level structural action-row caches.
For cache misses, all active states in an altitude layer are generated in
authoritative NumPy action slabs; sensor scenarios on the same terrain share
the resulting feasible masks, target IDs, terminals, and exact `GlideEdge`
rows. Hazard and stage-cost rows remain sensor-specific. Evaluation rollouts
read only structural targets and terminals, then use the unchanged trajectory
evaluator for final `J_A`. The episode-6,000 model reproduced all 45 stored
objectives, trajectories, switching states, and successful-candidate counts.
Its full checkpoint evaluation fell from 391.93 s to 213.09 s (1.84x); nine
stepped-pyramid sensor cases fell from 74.97 s to 31.20 s (2.40x).

## Extended Phase 16.4 random-box pilot

The frozen simple-terrain experiment remains unchanged.  The extension approved
on 2026-10-19 uses a separate dataset and checkpoint namespace and adds:

- a deterministic finite random-box terrain bank,
- horizontal start sampling inside `r_sp = 100 m`,
- common-lattice goal sampling inside `r_gp = 100 m`,
- a physical spherical terminal radius of `50 m`,
- geometry-ranked single-sensor sampling at 25% low, 50% mixed, and 25% high
  visibility,
- fused multiscale observation channels (`1,000 m / 41x41 / 25 m` local and
  `2,000 m / 21x21 / 100 m` global),
- physically feasible goal-unreachable actions for learnable failure branches,
- a Bellman-reference-derived terminal failure cost,
- a 5 m sampled-interior powered-segment occlusion requirement, and
- validation-quality early stopping after at least 10,000 episodes, using the
  latest three 100%-successful evaluations, a window-median and latest-median
  Bellman-relative error no greater than 10%, and a median-error range no
  greater than 10 percentage points. Final-model acceptance additionally uses
  median error no greater than 10% and P90 error no greater than 30%; maximum
  error is reported but is not a hard gate.

## G2 final-ID evaluation interpretation

This section records the interpretation frozen during the first unseen G2
final-ID test on 2026-10-30. It distinguishes the Phase-5 attacker-BR
generalization test from the later Stackelberg leader search.

### Performance metrics

For each mission-feasible sensor case, both Bellman and DQN trajectories are
rescored by the same authoritative attacker-objective evaluator. The per-case
Bellman-relative error is

```text
e_i = abs(J_A,DQN(i) - J_A,Bellman(i)) / abs(J_A,Bellman(i)).
```

Mission feasibility, policy success, and approximation accuracy are separate:

1. **Mission feasibility:** at least one powered-occluded, goal-reachable
   switching state exists.
2. **DQN policy success:** on a feasible mission, the DQN rollout reaches the
   goal and passes the authoritative trajectory checks.
3. **Approximation accuracy:** only successful, feasible cases contribute an
   `e_i` value.

The median is the middle `e_i` after sorting and describes typical-case
accuracy without allowing one extreme case to dominate the statistic. P90 is
the 90th percentile and describes tail reliability: approximately 90% of
evaluated cases should have an error no greater than this value. The project
uses NumPy's conservative `method="higher"`. Consequently, when only three
cases are feasible, P90 equals the maximum error; with sixteen feasible cases,
it is approximately the second-worst ordered error.

The frozen final-model gate is:

```text
policy success on feasible missions = 100%
median Bellman-relative J_A error   <= 10%
P90 Bellman-relative J_A error      <= 30%
maximum error                        report only
```

These are follower-BR quality metrics. They do not by themselves certify a
Stackelberg solution or measure the defender objective `J_D`.

### Meaning of Low, Mixed A, Mixed B, and High

The four cases on one terrain share the same terrain, start, and goal. Only the
fixed single-sensor position changes. The generator samples 96 candidate
sensors in the approved sensor region. For each candidate it measures the
fraction of valid terrain probes with an unblocked segment to the sensor. It
then ranks the 96 candidates on that terrain and defines:

- **Low:** bottom visibility quartile;
- **Mixed:** middle two visibility quartiles;
- **High:** top visibility quartile.

One Low, two Mixed, and one High candidate are sampled, giving the approved
25%/50%/25% mixture. Mixed A and Mixed B are merely the two independently
selected members of the same middle pool; A/B does not encode sensor type,
difficulty order, or visibility order. Low and High are also relative to the
candidate distribution on that terrain, not fixed absolute visibility
thresholds.

For final-ID terrain 000 the frozen cases are:

| Case | Sensor position [m] | Visibility fraction |
|---|---:|---:|
| Low | `(501.2, -170.4, 0)` | 0.512 |
| Mixed A | `(508.1, 133.8, 0)` | 0.600 |
| Mixed B | `(571.4, -26.7, 0)` | 0.592 |
| High | `(613.2, 299.1, 0)` | 0.656 |

The High case is mission-infeasible at 100/50/25/10 m. A valid switching state
must originate on the sensor-specific LOS tangent surface, survive lattice and
physical-feasibility filtering, have a goal-reachable glide continuation, and
have an entirely terrain-occluded straight powered segment from launch to
switch. No candidate satisfies that conjunction for this sensor geometry.
Bellman and DQN therefore both lack a trajectory or switching point. This is a
mission-level feasibility result, not a DQN policy failure. High visibility is
not universally infeasible; this result is specific to this terrain, mission,
and sensor geometry.

### Why visibility strata are used before Stackelberg integration

Low/Mixed/High stratification was introduced to prevent random training and
validation banks from being dominated by one common visibility regime. It
forces the generalized attacker-BR model to see, and later be diagnosed on, a
range of sensing geometries. The four cases are therefore balanced follower-BR
training/evaluation slices. They are **not** four defender actions that replace
the Stackelberg optimization, and the best of the four must not be reported as
the Stackelberg solution.

The intended hierarchy remains:

```text
Defender leader selects/evaluates sensor position d
    -> Bellman or frozen DQN computes attacker BR(d)
    -> the common evaluator computes J_A and J_D
    -> the outer local/global defender search selects its sensor placement
```

Phase 5 validates whether the frozen DQN can serve as the inner attacker-BR
backend on unseen terrain and sensor positions. Phase 6 integrates that backend
into the same Local-SSE/Stackelberg search used by Bellman. Phase 7 performs the
computational-condition sweep. Once integrated, every sensor position visited
by the defender search must receive a BR evaluation; visibility class is only a
diagnostic label and does not select the final defender strategy.

Under the current finite-game formulation, the defender feasible set `D`
contains only sensor placements with a nonempty switching-state set. The
final-ID-000 High case would therefore lie outside `D` and cannot be assigned a
Stackelberg `J_D` under the current contract. Before claiming an end-to-end
Stackelberg result on unfiltered sensor regions, the project must retain one of
the following explicit interpretations:

1. restrict `D` to sensor placements for which the attacker mission is
   feasible, as in the current formulation;
2. define a defender/attacker payoff for no-launch or mission-denial cases; or
3. relax powered occlusion from a hard constraint and model powered exposure
   and its detection cost.

Until that modeling decision changes, infeasible cases remain useful for
reporting mission-feasibility rate, but they are excluded from Bellman-relative
`J_A` accuracy and from the Stackelberg objective comparison.

## Phase 6 paper-ready Local-SSE closeout

Phase 6 uses the frozen G2 checkpoint without retraining, fine-tuning, or
checkpoint selection. Exact Bellman and frozen-DQN attacker-BR backends were
integrated into the same deterministic `r=1` defender Local-SSE search on the
unseen final-ID Terrain 001 at `dx=100/50/25/10 m` and `dpsi=5 deg`.

The primary DQN deployment time is the first Local-SSE solve after scene and
checkpoint setup with empty scenario, observation, and structural action-row
caches. The repeated identical-query time with retained caches is reported only
as a secondary latency diagnostic. Offline training, scene construction,
checkpoint loading, and the post-hoc exact-neighbor certificate are excluded
from both Bellman and DQN solve timings.

| dx | Bellman first solve | DQN first solve | same-sensor J_A error | exact J_D regret | exact local certificate |
|---:|---:|---:|---:|---:|:---:|
| 100 m | 50.889 s | 65.014 s | 0.00% | 0.00% | PASS |
| 50 m | 131.597 s | 51.062 s | 6.72% | 79.70% | FAIL |
| 25 m | 961.422 s | 116.917 s | 8.23% | 90.42% | FAIL |
| 10 m | 5999.106 s | 2022.959 s | 4.52% | 42.38% | FAIL |

All four DQN responses pass the Phase-5 same-sensor `J_A <= 10%` diagnostic,
but only `dx=100 m` preserves the exact Bellman Local-SSE sensor decision. At
that condition Bellman is faster for the first solve. Therefore this pilot has
no observed condition at which DQN is both first-solve faster and admissible as
an exact leader-decision replacement. The separation between low follower
objective error and high defender regret is the principal Phase-6 integration
finding.

This is one held-out terrain at one angular spacing and neighbor radius, not a
break-even theorem or Monte Carlo regime boundary. Phase 7 must estimate that
boundary over terrain/mission/sensor repetitions and additional computational
conditions. Bellman versus generalized DQN is the primary paper comparison;
the legacy case-specific Tabular Q-learning backend is explicitly deferred to
the Phase-7 secondary baseline rather than being represented as a completed
Phase-6 three-backend comparison.

Rebuild the paper-ready tables and static figures from the immutable raw Phase-6
JSON files, then rebuild the interactive 3D solutions, with:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase6_paper_3d.py
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase6_paper_closeout.py
```

The versioned closeout manifest, CSV, report, 300-dpi figures, and self-contained
3D HTML files are written under
`3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/`.

## Phase 7 controlled computational-condition sweeps

Phase 7 keeps the frozen G2 checkpoints and the common Local-SSE contract while
varying one computational condition at a time:

- spatial: `dx=100/50/25/10 m`, `dpsi=5 deg`, `r=1`;
- angular: `dpsi=10/5/2.5 deg`, `dx=25 m`, `r=1`;
- neighborhood A: `r=1..5`, `dx=25 m`, `dpsi=10 deg`;
- neighborhood B: `r=1..5`, `dx=25 m`, `dpsi=5 deg`.

Each closeout reports Bellman and DQN timings, realized defender-evaluation
counts, same-sensor `J_A` error, signed exact `J_D` gap, exact discrete Local-SSE
certificate, and selected-action match as separate facts.  It does not assign
an automatic Bellman/DQN adoption label.

The paper reproducibility freeze is generated from the immutable run summaries
and final checkpoint payloads, rather than from current source defaults:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\paper_reproducibility_freeze.py
```

The resulting full manifest, paper-ready hyperparameter CSV, artifact checksum
index, software/hardware record, and freeze policy are written under
`3D_RL_DQN/paper_freeze_v1/`.  Final evaluation checkpoints are versioned with
Git LFS.  Regenerable Phase-5 interactive HTML files are excluded from the
freeze; their raw JSON inputs and generation code are retained.

Generate the finite bank, CPU smoke audit, and interactive 3D bank visualization
without importing PyTorch:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase4_extended_pilot.py
```

The outputs are written under
`3D_0827/figure/phase_4_extended_random_boxes/pilot_v1/`.

The master notebook exposes the same extension through three independent
switches: `RUN_EXTENDED_PILOT_VISUALIZATION`, `RUN_EXTENDED_PREPARATION`, and
`RUN_EXTENDED_TRAINING`. Leave all three `False` unless that operation should
run. If preparation and training are both `True`, training runs once and
includes preparation. `EXTENDED_TRAINING_SMOKE = True` selects the short CUDA
pipeline check; official training uses the regular 60,000-episode
maximum with early stopping.

Prepare all condition-specific scenes and filtered Bellman references without
starting DQN training:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase4_extended_training.py `
  --prepare-only --device cuda --spatial-resolution-m 25
```

Run the short CUDA end-to-end training smoke:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase4_extended_training.py `
  --smoke --device cuda
```

Run or resume the official CUDA experiment:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase4_extended_training.py
```

Run only seed 0 for the first official check with `--seeds 0`. Later runs can
use `--seeds 1 2` without changing the approved three-seed training contract.
Subset summaries use seed-specific filenames so they do not overwrite the
eventual three-seed summary.

The result separates pure episode/replay/optimizer time, scheduled evaluation,
resume-checkpoint I/O, post-training evaluation, final-checkpoint I/O, and total
measured seed runtime. The legacy `training_runtime_sec` field is retained as an
alias for training-loop wall time and includes scheduled evaluation and resume
checkpoint writes.

The official runner uses seeds 0, 1, and 2 and a maximum of 60,000 episodes per
seed.  It evaluates every 1,500 episodes and stops a seed only after 12,000
episodes when four consecutive validations vary by no more than 1%, median
Bellman-relative `J_A` error is at most 10%, every-case error is at most 20%, and
all validation missions reach the goal.  A low-quality plateau continues to the
maximum budget and is recorded as a failed seed.

Scenarios with no powered-occluded, goal-reachable switching candidate are kept
in the manifest as mission-level failures and excluded from replay because no
glide episode can begin.  Policy-selected dead ends, altitude exhaustion, and
horizon exhaustion remain learnable terminal failures.

The notebook is the recommended runner. The same backend can also be invoked
directly:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase4_multi_terrain.py
```

Re-running the command resumes incomplete seeds and skips completed seeds. Use
`--no-resume` only when intentionally starting a new run. The batched official
outputs are written below. The earlier `official/` directories retain the
serial pilot and its episode-18,000 seed-1 checkpoint.

```text
3D_0827/figure/phase_4_multi_terrain_dqn/dx25m_dpsi5deg/official_batched_v1/
3D_RL_DQN/checkpoints/phase4_multi_terrain/dx25m_dpsi5deg/official_batched_v1/
3D_RL_DQN/runs/phase4_multi_terrain/dx25m_dpsi5deg/official_batched_v1/
```

After an official run completes, create the per-terrain 3D switching audit
without retraining:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase4_representative_3d.py --html-only
```

The command writes self-contained interactive HTML plots while preserving the
existing PNG files. It overlays each representative Bellman and DQN trajectory
on the continuous LOS tangent surface, verifies that both initial states belong
to the authoritative switching-candidate set, and reports the surface gap
introduced by snapping the continuous surface to the Bellman lattice. It also
writes CSV and Markdown tables comparing exact Bellman and trained-DQN attacker
trajectory query times for the representative cases.

Measure the exact Bellman and trained-DQN full Local-SSE searches on the held-out
stepped-pyramid terrain with the same Defender topology and neighborhood:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\full_sse_dqn_comparison.py `
    --terrain stepped_pyramid --spatial-resolution-m 25 `
    --heading-spacing-deg 5 --r-neighbor 1 --device cuda
```

The full-SSE timings exclude the common scene/graph build and checkpoint load.
The output also replays the learned selection with exact Bellman responses and
checks every required neighbor, outside the benchmark timing, so an approximate
Defender selection is never reported as a certified local SSE.

The optimized pipeline was checked with the full DQN test suite and a 100 m
vectorized Phase 16.4 smoke run. Runtime profiles split batched action
observation/forward work, authoritative transitions/replay, replay-observation
transfer, DQN forward/backward/AdamW, checkpoint evaluation, and checkpoint I/O.
Current process state and logs are under the run directory above.
Run the contract tests with:

```powershell
& .\.venv_p1b\Scripts\python.exe -m unittest discover -s .\3D_RL_DQN -p test_dqn_phase4.py -v
```

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

Run the short CPU pipeline check without making an official pass/fail claim:

```powershell
& .\.venv_p1b\Scripts\python.exe .\3D_RL_DQN\phase3_single_terrain.py --smoke --device cpu
```

Run the approved three-seed, 20,000-episode experiment with automatic CUDA/CPU
selection:

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
the tested Python modules rather than duplicating solver or training logic. The
official training guard requires a CUDA-enabled `.venv_p1b` kernel; smoke runs
are allowed on CPU. Phase 16.4 resume is enabled independently with:

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

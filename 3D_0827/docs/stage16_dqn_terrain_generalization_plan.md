# Stage 16 — Terrain-Generalized DQN and Bellman Selection Boundary

## 1. Research objective

Stage 15 measured exact Bellman against tabular Q-learning on one fixed terrain.
Stage 16 changes the approximation question: train a terrain-conditioned DQN once
on a distribution of terrains and reuse it on held-out terrains and defender
positions.

For computational condition

```
c = (delta_xyz, delta_heading, delta_glide, r_neighbor)
```

determine when exact Bellman local SSE or a pretrained DQN attacker best-response
approximation should be used.  The decision uses measured solution quality, wall
time, peak memory, CPU/GPU utilization, and failure status.  DQN is eligible only
after reaching fixed attacker and defender quality thresholds on held-out terrain.

The outer local Stackelberg search remains identical.  Only the inner attacker
best-response solver changes.

## 2. Claims and non-claims

The primary claim is an empirical solver-selection boundary over the tested
condition and terrain distributions.  Stage 16 does not claim global optimality,
universal terrain generalization, or an accuracy guarantee outside the measured
test distribution.

The key comparison is:

```
exact: same local defender search -> Bellman attacker BR -> selected placement
DQN:   same local defender search -> DQN attacker BR     -> selected placement
```

Tabular Q-learning remains a baseline showing the cost of per-terrain learning.
It is not the proposed generalized approximation.

## 3. Unit of model reuse

The first experiment trains one DQN per inner discretization condition:

```
(delta_xyz, delta_heading, delta_glide) -> one shared DQN across terrains/sensors
```

`r_neighbor` is an outer-search workload knob and is not a policy input.  It
changes the number of defender positions queried, approximately `(2r+1)^2`.

A single network spanning different action resolutions is out of scope for the
first experiment because the output dimension and motion-primitive meaning change.
That requires a variable-action or graph policy and would confound terrain
generalization with architecture generalization.

## 4. Environment contract

`glider_gym_env.TerrainGlideEnv` is an adapter over the shared `AttackerMDP`.  It
must not reimplement dynamics or costs.

### Observation

The observation is a dictionary:

```
terrain:     normalized height map, shape (1, Ny, Nx)
state:       13 normalized values
action_mask: one validity flag per motion primitive
```

The state vector contains absolute aircraft position, sine/cosine heading,
goal-relative position, sensor-relative position, and attacker objective weights.
The terrain map is encoded by a CNN; the state vector is encoded by an MLP; the
embeddings are concatenated before the Q head.

The replay implementation must deduplicate the terrain tensor by terrain ID or
cache its CNN embedding.  Copying a full map into every transition is acceptable
for the environment contract but not for the scaling benchmark.

### Action

Action ID is the fixed target heading bin / motion-primitive index.  It has the
same physical meaning in every state for a fixed condition.  State-dependent
invalid primitives are excluded by `action_mask` during both behavior selection
and target calculation:

```
Q(next_state, invalid_action) = -infinity before max
```

An invalid action selected despite the mask terminates the episode with an
explicit diagnostic penalty.  It must never be silently remapped to the local
adjacency-list order.

### Reward and termination

For a valid edge:

```
reward = -shared_stage_cost
```

so reward maximization is exactly attacker cost minimization.  Goal arrival is a
successful terminal state.  A dead end or invalid action is a failed terminal
state.  `max_steps` is an external truncation only.

### Hidden preprocessing prohibition

The current adapter consumes a `Scene` whose goal-backward reachable graph has
already been built.  Benchmark accounting must therefore report:

```
T_env_setup, T_terrain_encode, T_train, T_query
```

and include `T_env_setup` in cold-start DQN cost.  A later online-transition
environment should generate only local successors from `GlideTransitionModel`
and avoid full backward reachability on a new terrain.  It may replace the first
adapter only after valid-action transition and cost equivalence tests pass.

## 5. Terrain distribution and data split

Terrain factories must use deterministic seeds and immutable specifications.
Split specifications before training:

```
train:      policy optimization only
validation: architecture, hyperparameters, stopping budget, threshold selection
test-ID:    unseen instances from training families
test-OOD:   unseen structural families or parameter ranges
```

Initial families should cover translated/rescaled boxes, multiple boxes, stepped
pyramids, ridges, and valleys where supported by the terrain contract.  The five
catalog entries alone are smoke cases, not a sufficient generalization dataset.

No test terrain may influence normalization statistics, early stopping, model
selection, or reward design.  Exact Bellman results on test terrains are evaluation
oracles, not DQN labels unless a separately named imitation-learning ablation is
run.

## 6. Quality metrics

Do not collapse non-zero-sum quality into one ambiguous accuracy percentage.

For a fixed defender position:

```
gap_A = (J_A_DQN - J_A_exact) / J_A_exact
```

For the placement selected by the outer search:

```
regret_D = J_D_true(a_exact) - J_D_true(a_DQN)
```

Both terms in `regret_D` use exact attacker reevaluation where feasible.  The DQN
attacker path's own reported `J_D` is diagnostic because a suboptimal attacker can
make defender performance look optimistically high.

Also report goal success rate, infeasible/invalid rate, path-duration error,
detection-probability error, terrain-wise quantiles, and seed variance.

Before the full sweep, freeze eligibility thresholds.  Initial sensitivity values:

```
tau_A_median in {0.02, 0.05, 0.10}
tau_A_p95    in {0.05, 0.10, 0.20}
tau_D        in {0.01, 0.02, 0.05} PoD points
goal success >= 0.99
```

The main result must show threshold sensitivity rather than selecting a favorable
threshold after observing test results.

## 7. Cost accounting and break-even rule

For `K(r)` defender positions and `N` deployments sharing a pretrained model:

```
K(r) = number of unique positions actually evaluated by local SSE

T_DQN(c, r, N)
  = T_train(c) / N
  + T_env_setup(c)
  + T_terrain_encode(c)
  + K(r) * T_query(c)

T_exact(c, r)
  = T_exact_setup(c)
  + K(r) * T_BR(c)
```

Report both cold-start (`N=1`, full training charged) and pretrained deployment
cost.  Also report the measured break-even deployment count.  Wall time, CPU time,
peak RSS, peak GPU memory, mean/peak CPU utilization, model size, and status must
be recorded separately; CPU Bellman and GPU DQN results must name their hardware
and must not be presented as hardware-independent speedups.

## 8. Solver selection rule

For threshold set `tau`, DQN is eligible only if the held-out test distribution
meets every frozen quality and feasibility constraint.  Then:

```
choose DQN
  iff DQN is eligible
  and measured/amortized DQN cost is below exact cost
  and deployment resource limits are met

otherwise choose Bellman when Bellman completes within limits
```

The decision map has four outcomes:

1. `bellman_preferred`
2. `dqn_preferred`
3. `bellman_unavailable_dqn_eligible`
4. `neither_verified`

## 9. Implementation phases

### Phase 16.0 — contract and equivalence

- Add the fixed-action Gym adapter and action mask.
- Test every valid Gym transition against `AttackerMDP.step` and shared cost.
- Record current full-graph setup as explicit benchmark cost.
- Specify terrain datasets and immutable split manifests.

Exit gate: no duplicated dynamics; fixed action semantics and reward sign are
tested; test terrains are frozen before training.

### Phase 16.1 — terrain dataset and encoder

- Add seeded parameterized terrain factories.
- Add height-map rasterization checks and split manifests.
- Implement CNN terrain encoder plus MLP state encoder.
- Store terrain IDs in replay rather than duplicated maps.

Exit gate: embeddings differ for geometrically different terrains; train/validation/
test leakage checks pass; one batch runs end to end.

### Phase 16.2 — masked DQN

- Implement policy/target networks and replay buffer.
- Apply masks in epsilon-greedy action choice and Bellman target.
- Train across sampled training terrains and defender positions.
- Save checkpoints, optimizer state, seeds, condition, and dataset fingerprints.

Exit gate: coarse single-terrain DQN learns a feasible policy, then one shared
model succeeds on held-out instances without retraining.

### Phase 16.3 — oracle quality evaluation

- Run exact Bellman on feasible held-out conditions.
- Measure `gap_A`, `regret_D`, success rate, and distribution quantiles.
- Separate ID and OOD generalization.
- Replay and visualize exact and DQN trajectories on identical scenes.

Exit gate: eligibility can be decided without using DQN self-reported `J_D`.

### Phase 16.4 — scaling and selection boundary

- Sweep spatial, heading, glide-direction, and Chebyshev-radius conditions.
- Measure cold-start and amortized resource costs.
- Include timeout, memory limit, DQN budget exhaustion, and neither-verified cells.
- Produce decision maps for each frozen threshold set and deployment count.

Exit gate: every cell is reproducible from a condition, terrain split fingerprint,
model checkpoint, seed set, and hardware record.

## 10. Immediate next implementation

The next code milestone is a masked DQN smoke run on the coarse 100 m condition
using several training terrain instances and at least one held-out instance.  It is
not yet a scaling result.  Its purpose is to validate the observation encoder,
fixed action mapping, replay representation, reward direction, and no-retraining
evaluation path before expensive sweeps begin.

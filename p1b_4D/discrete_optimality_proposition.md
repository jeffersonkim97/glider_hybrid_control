# Discrete Optimality of the Coarse Bellman Attacker Response

Paper-section draft for the ACC/L-CSS submission (plan item 6,
`p1b_roadmap_0727.md`). Establishes that `select_authoritative_bellman_
response` returns the exact minimum-cost Attacker response over the full
discretized decision space — not a local or greedy approximation — given
the transition structure `p1b_4D` actually constructs. Existing residual/
telescoping-sum unit tests (`test_bellman.py`, `test_stage_cost.py`, etc.)
are cited as *implementation validation* that the code satisfies this
proposition's hypotheses; they are not offered as a substitute for the
proof itself, and the two should stay visibly distinct in the writeup.

## Setup

Fix a configuration (terrain, sensor position, cost weights, grid
resolution). Let:

- $z_1 < z_2 < \dots < z_{N_z}$ and $h_1, \dots, h_{N_h}$ be the along-track
  and altitude grids (`grids["z"]`, `grids["h"]`).
- $\mathcal{X} = \{1,\dots,N_z\} \times \{1,\dots,N_h\}$ be the discrete
  spatial state space (glide phase); a state is a grid index pair
  $(i,j)$, not a continuous position.
- $\mathcal{A} = \{1,\dots,N_v\} \times \{1,\dots,N_\gamma\}$ be the
  finite discretized action space (speed, flight-path angle).
- $G \subset \mathcal{X}$ be the goal mask (`goal_mask`): states within
  `goal_radius` of the goal position.
- $\Sigma \subset \{1,\dots,N_z\}$ be the discretized admissible switching
  set — **the set the code actually enumerates**
  (`generate_switching_point_seeds`): every $z$-grid index strictly
  between launch and the sensor whose LOS-boundary height lies inside the
  configured airspace band. This is *not* claimed to equal the continuous
  LOS-tangent boundary; it is the boundary's value at each admissible grid
  node, which is what the algorithm can distinguish at this resolution.

For $(i,j)\in\mathcal{X}$ and $(k,l)\in\mathcal{A}$, `construct_coarse_
transitions` computes a kinematic step $\delta z = v_k\cos\gamma_l\,\Delta
t$, $\delta h = v_k\sin\gamma_l\,\Delta t$ and maps it to a successor
index via
$$
\text{next}_z(i,k,l) = \left\lceil \frac{(z_i+\delta z) - z_1}{\Delta z}
\right\rceil, \qquad
\text{next}_h(i,j,k,l) = \operatorname{round}\!\left(\frac{(h_j+\delta h)
- h_1}{\Delta h}\right),
$$
marked **valid** only if both indices are in-bounds, every intermediate
sub-sample (`segment_check_count` fractions of the segment) stays clear of
terrain and inside the LOS-visible region, and the transition either
reaches $G$ or lands on a spatially-valid, non-goal successor state.

## Proposition 1 (finite DAG)

*The set of valid transitions, viewed as a directed graph on $\mathcal{X}$,
is a finite DAG in which every edge strictly increases the $z$-index.*

**Proof.** $\mathcal{X}$ is finite by construction. Fix a valid transition
from $(i,j)$ under action $(k,l)$. The vehicle's configured flight-path-
angle bounds satisfy $\gamma_{\min} \le \gamma_l \le \gamma_{\max} <
0^\circ$ (glide is always descending, never level or climbing), so
$\cos\gamma_l > 0$ for every $\gamma_l$ in this range, and since $v_k > 0$,
$\delta z = v_k\cos\gamma_l\,\Delta t > 0$ strictly. Writing $x =
i + \delta z/\Delta z > i$ with $i$ an integer, $\lceil x\rceil \ge i+1$
for any $x>i$. Hence $\text{next}_z(i,k,l) \ge i+1 > i$ for *every* valid
transition, with no exception. A graph in which every edge strictly
increases an integer-valued node label cannot contain a cycle, so the
graph is acyclic; sorting nodes by decreasing $z$-index is therefore a
valid topological order. $\blacksquare$

This is exactly the property `solve_coarse_bellman` relies on structurally
(`"acyclic_forward_transition": True` in its own diagnostics) and exactly
why a *single* backward sweep — not iterate-to-convergence value
iteration — suffices (Proposition 3).

## Proposition 2 (exhaustive action and switching-seed enumeration)

*For every state, all $N_v \times N_\gamma$ actions are evaluated. For the
switching decision, every element of $\Sigma$ is evaluated as a candidate.*

**Proof.** `_ordered_actions` constructs `actions` as the full Python
list comprehension over `range(v_grid.size) × range(gamma_grid.size)` —
i.e. the complete Cartesian product — and only *sorts* it into one of four
tie-breaking orders; no action is filtered out before `solve_coarse_
bellman`'s inner loop `for velocity_index, gamma_index in actions` visits
it. Separately, `generate_switching_point_seeds` builds `domain_z` as
every $z$-grid node with `z_start <= z < z_sensor`, restricted to
`within_airspace`, and returns one seed per surviving node — i.e. all of
$\Sigma$, not a sample of it. `generate_bellman_candidates` then calls
`evaluate_powered_segment` and `extract_coarse_candidate` once per seed in
this full set, with no early termination. $\blacksquare$

## Proposition 3 (backward induction computes exact cost-to-go)

*After one sweep of `solve_coarse_bellman`, `value[i,j]` equals the exact
minimum discretized cost-to-go from $(i,j)$ to the goal, for every
$(i,j)\in\mathcal{X}$ reachable from some switching seed.*

**Proof.** By Proposition 1 the transition graph is a DAG with edges only
increasing the $z$-index; processing $z$-index from $N_z{-}1$ down to $0$
(as the code does: `for z_index in range(z_grid.size - 1, -1, -1)`) is a
valid reverse topological order, so whenever state $(i,j)$ is processed,
`value[next_z, next_h]` for every action's successor has *already* been
finalized (either it is a goal cell, pinned to $0$ before the sweep
starts, or it was visited at a strictly larger $z$-index earlier in the
same sweep). The update computed for $(i,j)$,
$$
\text{value}[i,j] = \min_{(k,l)\in\mathcal{A},\ \text{valid}}\ \Big(
f(i,j,k,l)\cdot j_{4D}[i,j,k,l] + \text{value}[\text{next}(i,j,k,l)]
\Big),
$$
(where $f$ is $1$ for a non-terminal transition or the terminal fraction
for one that reaches $G$ mid-segment, and $j_{4D}$ is the local stage cost
from `construct_stage_cost_4d` — itself already validated elsewhere to
equal the additive hazard/time increment for that one coarse step) is
exactly the Bellman optimality equation restricted to this DAG. By
induction on topological distance to $G$ — base case $\text{value}[G]=0$
by definition, inductive step is the minimization above using only
already-correct successor values — `value[i,j]` equals the true minimum
sum of local costs over all valid discrete paths from $(i,j)$ to $G$. This
is the standard single-source-shortest-path-in-a-DAG argument (one
relaxation pass in topological order is sufficient and exact whenever the
graph has no cycles), applied here to a DAG whose acyclicity is *forced*
by vehicle dynamics (Proposition 1) rather than assumed. No further sweep
can change any value once computed, which is exactly why the
implementation performs `"sweep_count": 1` rather than an
iterate-to-convergence loop — this is not an approximation or an
early-exit; on this DAG, one sweep already is the fixed point.
$\blacksquare$

## Proposition 4 (overall discrete Attacker optimum)

*Combining Propositions 1–3, `select_authoritative_bellman_response`'s
`mission_cost` is the exact global minimum of the Attacker's total mission
cost (powered-segment cost plus glide-phase cost-to-go) over the entire
discretized decision space — every switching seed in $\Sigma$ crossed with
every DAG-optimal glide policy from that seed.*

**Proof.** For each seed $\sigma\in\Sigma$ (Proposition 2), `evaluate_
powered_segment` computes the exact powered-phase cost for the one
admissible straight-line path from launch to $\sigma$ (closed form, no
discretization choice involved beyond the already-fixed seed), and
`extract_coarse_candidate` replays the policy fixed by Proposition 3 from
$\sigma$'s grid cell forward, so the resulting `mission_cost` for that seed
is exactly (powered cost) + (exact glide cost-to-go from $\sigma$).
`select_authoritative_bellman_response` takes the minimum of this
quantity over *all* $\sigma\in\Sigma$ (`sorted(candidates, key=...
mission_cost...)`, no filtering applied beforehand). Since every seed was
evaluated exactly (not approximately) and every seed in the admissible set
was tried, the returned minimum is the global minimum over the full
discretized joint space. $\blacksquare$

## Scope and what this does *not* claim

- This is optimality **over the discretization the code defines** — grid
  resolution, action grid, and the switching-seed set $\Sigma$ are all
  finite approximations of the true continuous problem. No claim is made
  about the continuous-problem optimum; item 1's resolution-convergence
  study is the empirical evidence for how close the discretized optimum
  tracks it.
- This says nothing about the **Defender's** continuous outer search
  (DIRECT) — that remains "best-found," not certified (see item 7's
  terminology cleanup). Propositions 1–4 apply strictly to the inner
  (Attacker) problem for one *fixed* sensor position.
- The empirical scaling exponent of 1.043 (item 5) is consistent with, but
  not itself proof of, Proposition 3's $O(1)$-work-per-cell claim — the
  proposition is proved from the code's control flow, the measurement is
  corroborating evidence, and the writeup should present them as two
  independent lines of evidence for the same underlying structural fact,
  not conflate them.

## Independent cross-check

Proof by inspection of control flow is not a substitute for testing the
actual numbers it predicts. `test_discrete_optimality_crosscheck.py`
(new, see below) constructs a small toy grid, builds the *same* transition
graph and stage costs via the real `construct_coarse_transitions`/
`construct_stage_cost_4d` code, then computes shortest-cost-to-go via an
independent library (`networkx.single_source_dijkstra`) over that graph
and asserts it matches `solve_coarse_bellman`'s output exactly. This
tests the Proposition 3 argument's *conclusion* against a trusted external
implementation, using the project's own (separately validated) transition
construction as the shared input — it does not re-derive the physics.

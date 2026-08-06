# Discrete Optimality of the Physical Successor-Grid Bellman Follower

This document states the formal result implemented by
`successor_grid_solver.py`. It supersedes the legacy proposition based on
`solve_coarse_bellman`, fixed-duration actions, and endpoint snapping. The
result concerns the Attacker follower problem for one fixed terrain, sensor
configuration, cost-weight selection, and finite discretization.

The central claim is deliberately scoped:

> Bellman computes the exact optimal value on the finite physical-successor
> graph. It does not compute or certify the optimum of a continuous
> trajectory space.

## 1. Finite follower problem

Let the spatial grids be

$$
\mathcal Z_\Delta=\{z_0,\ldots,z_{N_z-1}\},
\qquad
\mathcal H_\Delta=\{h_0,\ldots,h_{N_h-1}\},
$$

with uniform spacings $\Delta z>0$ and $\Delta h>0$. A regular glide state is

$$
x_{i,j}=(z_i,h_j)\in\mathcal X_\Delta
=\mathcal Z_\Delta\times\mathcal H_\Delta.
$$

The goal set is the finite-grid mask

$$
\mathcal G_\Delta
=
\left\{
x_{i,j}\in\mathcal X_\Delta:
\|x_{i,j}-x_g\|_2\le r_g
\right\},
$$

together with a terminal sink $g$ representing an edge that intersects the
goal ball before reaching its full grid-to-grid endpoint.

### 1.1 Regular physical-successor actions

A regular action is defined by a positive integer successor offset and a
speed,

$$
a=(p,q,v),
\qquad p\ge1,\quad q\ge1,\quad v\in\mathcal V_\Delta.
$$

Its physical displacement, flight-path angle, length, and duration are

$$
d_a=(p\Delta z,-q\Delta h),
$$

$$
\gamma_a
=
\operatorname{atan2}(-q\Delta h,p\Delta z),
\qquad
\ell_a=\|d_a\|_2,
\qquad
\tau_a=\frac{\ell_a}{v}.
$$

Thus a nonterminal edge from $x_{i,j}$ ends at the exact successor node

$$
T(x_{i,j},a)=x_{i+p,j-q}.
$$

No endpoint is rounded or snapped. The implementation admits the edge only
when:

1. the speed, angle, lift coefficient, and drag checks pass;
2. the endpoint is in bounds and spatially admissible, unless the edge is
   terminal; and
3. the P1 all-segment geometry certificate proves terrain and LOS feasibility
   over the complete straight segment under the implemented piecewise-cubic
   terrain and piecewise-linear swept LOS boundary.

If the segment first intersects the goal ball at fraction
$\rho\in(0,1]$, it becomes a terminal edge to $g$ with displacement
$\rho d_a$ and duration $\rho\tau_a$.

### 1.2 Edge cost

For a regular or terminal edge $e$, let $H_e^{(Q)}$ be the cumulative hazard
computed by the configured $Q$-sample trapezoidal rule, and let $T_e$ be its
physical duration. The finite problem defines the edge cost as

$$
c_\Delta(e)
=
w_{\mathrm{PoD}}
\frac{H_e^{(Q)}}{H_{\mathrm{ref}}}
+
w_{\mathrm{time}}
\frac{T_e}{T_{\mathrm{ref}}}.
$$

The quadrature is part of the finite objective definition. The proposition
does not call this integral closed form and does not claim that $Q$ samples
equal an exact continuous-time integral.

### 1.3 Virtual switching states

`generate_switching_point_seeds` defines a finite switching set
$\Sigma_\Delta$. In the normal case it evaluates the swept, potentially
multi-hill LOS boundary at every eligible $z$-grid coordinate strictly left
of the sensor and retains the points inside the configured altitude band.
Accordingly, a switching seed

$$
\sigma=(z_i,h_{\mathrm{LOS}}(z_i))\in\Sigma_\Delta
$$

can have an off-grid altitude and is treated as a virtual state rather than
being snapped onto $\mathcal X_\Delta$.

For each $\sigma$, `virtual_switch_target_indices` enumerates the complete
finite target set $\mathcal Y_\Delta(\sigma)$ specified by the configured
physical forward/descent box. For every target $y\in\mathcal Y_\Delta(\sigma)$
and speed $v\in\mathcal V_\Delta$, the virtual edge uses

$$
\gamma(\sigma,y)=\operatorname{atan2}(y_h-\sigma_h,y_z-\sigma_z),
$$

$$
\tau(\sigma,y,v)=\frac{\|y-\sigma\|_2}{v}.
$$

The edge therefore terminates exactly at $y$. It is admitted only if its
control, P1 all-segment terrain/LOS certificate, and downstream finite-value
checks pass.

The powered phase is the implementation-defined straight segment from the
launch state $x_0$ to $\sigma$. Its terrain and powered-occlusion feasibility
use the P1 all-segment certificate; its hazard uses the configured finite
quadrature. Let its resulting finite cost be $C_p^\Delta(\sigma)$.

## 2. Proposition 1: the regular successor graph is a finite DAG

**Statement.** The directed graph produced by `build_successor_grid_graph`
is finite and acyclic. Every nonterminal regular edge strictly increases the
$z$-index.

**Proof.** The graph has at most $N_zN_h$ regular nodes and one terminal sink,
so it is finite. Every regular action has $p\ge1$ and maps

$$
(i,j)\longmapsto(i+p,j-q).
$$

Hence the $z$-index strictly increases along every nonterminal edge. A
directed cycle would have to return to its original $z$-index, which is
impossible under a strict increase. Terminal edges end at $g$, which has no
outgoing edge. Therefore the graph is a DAG, and descending $z$-index is a
valid reverse-topological processing order. $\blacksquare$

## 3. Proposition 2: every admitted edge is execution-consistent

**Statement.** Every regular and virtual edge represents the straight
constant-speed physical segment defined by its two stored endpoints. Its
kinematic reconstruction reaches the stored endpoint without a state reset
or endpoint snapping.

**Proof.** Consider any admitted edge from $x$ to $y$ with

$$
\gamma=\operatorname{atan2}(y_h-x_h,y_z-x_z),
\qquad
\tau=\frac{\|y-x\|_2}{v}.
$$

Since $y_z>x_z$ and $y_h<x_h$ for every enumerated glide edge,

$$
v\tau
\begin{bmatrix}
\cos\gamma\\
\sin\gamma
\end{bmatrix}
=y-x.
$$

Therefore constant-speed execution from $x$ for duration $\tau$ ends at $y$
exactly, up to floating-point roundoff. For a terminal edge, replacing
$y-x$ by $\rho(y-x)$ and $\tau$ by $\rho\tau$ gives the same identity at the
goal-ball intersection. The virtual switching edge uses the identical
endpoint-derived construction with $x=\sigma$. No transition applies a
post-execution grid reset. $\blacksquare$

This proposition establishes endpoint consistency. P1 additionally computes
the global terrain and LOS margin minima for each straight edge under the
implemented geometry representation: cubic-spline knots and stationary points
for terrain, and every crossed breakpoint for the piecewise-linear swept LOS
boundary. The high-fidelity replay remains an independent sampled objective
reevaluation and implementation diagnostic; it is no longer the formal basis
for terrain/LOS edge feasibility.

## 4. Proposition 3: one backward Bellman sweep is exact on the graph

Let $\mathcal E_\Delta(x)$ denote all valid regular or terminal edges leaving
regular node $x$. Define

$$
V_\Delta(g)=0
$$

and

$$
V_\Delta(x)
=
\min_{e\in\mathcal E_\Delta(x)}
\left[c_\Delta(e)+V_\Delta(T(e))\right],
$$

where $T(e)=g$ for a terminal edge and $V_\Delta(x)=+\infty$ if no path from
$x$ reaches $g$.

**Statement.** `solve_successor_grid_bellman` computes $V_\Delta(x)$ exactly
for every regular grid node, relative to the stored finite edge costs.

**Proof.** By Proposition 1, descending $z$-index is a reverse-topological
order. When the algorithm processes $x_{i,j}$, the value of every
nonterminal successor $x_{i+p,j-q}$ has already been finalized because
$i+p>i$; the terminal sink has value zero. The update enumerates every action
stored in `graph["actions"]`, discards exactly those marked invalid, and takes
the minimum of the stored edge cost plus the already-final successor value.
This is the shortest-path recursion on a finite DAG. Induction in
reverse-topological order therefore proves that the stored value equals the
minimum total edge cost over every finite graph path from that node to $g$.
No fixed-point iteration is required. $\blacksquare$

If two regular actions have exactly equal floating-point candidate costs,
the first action in the frozen action enumeration is retained. This changes
only the representative policy, not $V_\Delta$.

## 5. Proposition 4: exhaustive virtual-switch evaluation gives the finite follower optimum

For a seed $\sigma\in\Sigma_\Delta$, let
$\mathcal E_\Delta^V(\sigma)$ be every admissible virtual edge produced by
the Cartesian enumeration of its configured target indices and speed grid.
Define

$$
J_\Delta(\sigma)
=
C_p^\Delta(\sigma)
+
\min_{e\in\mathcal E_\Delta^V(\sigma)}
\left[c_\Delta(e)+V_\Delta(T(e))\right],
$$

with $J_\Delta(\sigma)=+\infty$ when the powered segment or every virtual edge
is infeasible. The finite follower optimal value is

$$
\boxed{
J_\Delta^*
=
\min_{\sigma\in\Sigma_\Delta}J_\Delta(\sigma)
}.
$$

**Statement.** `solve_successor_grid_attacker` evaluates every member of
$\Sigma_\Delta$, every configured virtual target for that seed, and every
configured speed. Its reported `minimum_mission_cost` equals $J_\Delta^*$,
up to ordinary floating-point evaluation of the finite costs.

**Proof.** Proposition 3 supplies the exact regular-graph value at every
finite virtual-edge target. `_best_virtual_switch_edge` enumerates the full
finite target/speed Cartesian product, checks the finite admissibility rules,
and minimizes virtual-edge cost plus downstream Bellman value. The outer
loop evaluates every generated switching seed and forms the additive powered
plus glide mission cost. Finally, sorting all feasible candidates by mission
cost places the smallest finite value first. These nested exhaustive minima
are exactly the definition of $J_\Delta^*$. $\blacksquare$

## 6. Exact minimum selection and deterministic tie rule

After computing $J_\Delta^*$, the implementation restricts tie-breaking to
the exact finite-cost minimizers

$$
\mathcal T_\Delta
=
\left\{
\sigma\in\Sigma_\Delta:
J_\Delta(\sigma)=J_\Delta^*
\right\}.
$$

It returns the member of $\mathcal T_\Delta$ with the smallest switching
$z$, followed by seed index. A candidate whose cost is merely close to the
minimum cannot displace an absolute minimizer. Consequently,

$$
\boxed{
J_\Delta^{\mathrm{selected}}=J_\Delta^*
}.
$$

The switching-$z$ rule therefore chooses a deterministic representative from
the exact best-response set without weakening finite follower optimality.

## 7. Complexity

Let

- $N=N_zN_h$ be the number of regular spatial nodes;
- $M$ be the number of dynamically admissible regular offset-speed actions;
- $Q$ be the planning quadrature count;
- $S=|\Sigma_\Delta|$ be the number of switching seeds;
- $B$ be the maximum number of virtual target-speed pairs per seed; and
- $L$ be the maximum extracted policy length.

Then graph construction costs

$$
O(QNM)
$$

time and $O(NM)$ stored edge data. The Bellman sweep costs $O(NM)$ time and
$O(N)$ value/policy storage in addition to the graph. Switching evaluation
and path extraction cost

$$
O\!\left(S(BQ+L)\right)
$$

plus the configured powered-segment evaluation for each seed. For a fixed
finite action family and quadrature count, the Bellman portion scales
linearly with the number of grid states.

## 8. B4 production instance

Direction B4 freezes one instance of this finite family for C-lite:

- the L2 spatial grid for each terrain;
- the enriched physical-offset family;
- nine speed candidates;
- nine planning samples per edge; and
- 1025 samples per selected edge in the common high-fidelity replay.

The configuration identifier is
`direction_b_l2_enriched_v9_q9_e1025`. The 1025-point evaluator does not
alter the Bellman optimum; it independently reevaluates the selected finite
policy at a common higher sampling resolution.

## 9. Scope of the result

The proposition proves:

1. exact shortest-path values on the implementation-defined finite regular
   successor graph;
2. exhaustive optimization over the implementation-defined finite virtual
   switching decisions;
3. endpoint-consistent physical execution of every selected edge; and
4. an exact finite-cost-minimizing returned representative, with deterministic
   switching-$z$ selection only among exact cost ties.

The proposition does **not** prove:

- a continuous state-action or continuous trajectory-space optimum;
- a continuous defender-position optimum;
- a certificate relative to an unknown real terrain surface beyond the
  implemented spline/LOS representation;
- exact continuous integration of detection hazard; or
- resolution-independent optimality.

Those are intentionally outside the ACC core. Nested L0/L1/L2 experiments
measure sensitivity to finite spatial/action resolution, and the common
high-fidelity replay independently checks execution and objective integration
at higher sampling resolution. Neither is relabeled as continuous global
optimality.

## 10. Implementation validation

The proof and tests have distinct roles. The propositions follow from the
finite graph definition and exhaustive control flow. The tests verify that
the implementation satisfies those hypotheses:

- `test_successor_grid_solver.py` compares Bellman values with independent
  NetworkX shortest-path distances on the same finite graph;
- `test_direction_b_discretization.py` checks nested physical actions,
  virtual-switch targets, and machine-precision endpoint reconstruction;
- `test_continuous_replay_evaluation.py` and
  `test_successor_grid_solver.py` check executable replay without endpoint
  snapping; and
- `test_segment_feasibility.py` checks exact cubic stationary points, LOS
  breakpoints, multi-hill dense-reference agreement, and degenerate powered
  segments; and
- the B2, B3, and B4 regression tests check that the frozen production
  configuration and evaluator contract remain unchanged.

Passing these tests supports implementation fidelity; it is not a substitute
for the finite-DAG proof above.

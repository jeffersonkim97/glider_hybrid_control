# ACC Claim and Scope Freeze — 2026-07-30

## Status

This is the normative claim contract for the current ACC paper plan. It is
frozen for P1--P7 execution but may be revised explicitly when mathematical
analysis, verified literature, numerical evidence, or advisor feedback changes
the defensible scope. A revision must record what changed and why; experiment
results must not silently broaden the claims below.

## Central technical statement

The paper studies a terrain-dependent side-view Stackelberg surveillance game
with an endogenous powered-to-glide attacker switch. Its technical contribution
is an LOS-derived virtual-switch and physical-successor construction that
reduces the stated hybrid follower problem to an execution-consistent finite
acyclic graph, enabling exhaustive finite follower optimization by backward
dynamic programming.

The novelty is attributed to the reduction and its demonstrated strategic
consequences, not to Bellman recursion by itself or to a first combination of
otherwise known components.

## Frozen contribution claims

1. **Hybrid game formulation.** The defender commits to a terrain-mounted
   sensor position, and the follower jointly selects a powered-to-glide switch
   and a terrain/LOS-admissible glide response in a vertical terrain section.
2. **Execution-consistent finite reduction.** Off-grid LOS switching states
   connect to exact physical grid successors without endpoint snapping or a
   post-transition state reset.
3. **Formal finite solution property.** Subject to the P1 geometry certificate,
   every admitted finite path is sound under the implemented geometry model;
   every trajectory in the explicitly declared finite admissible class is
   represented; and Bellman recursion returns its exact finite optimum.
4. **Strategic mechanism.** Controlled results test whether terrain-induced
   changes in LOS topology and the hybrid reachable set propagate to attacker
   response and finite optimal sensor placement.

## Exactness vocabulary

Allowed:

> exact follower response over the declared execution-consistent finite graph

Not allowed:

> exact continuous follower response

> globally optimal continuous hybrid trajectory

> exact continuous Stackelberg equilibrium

Bellman is described as the exact finite shortest-path solver enabled by the
new graph structure, not as a new dynamic-programming algorithm.

## Claims explicitly outside scope

- continuous state/action or trajectory-space global optimality;
- a continuous defender-position optimum;
- resolution-independent optimality or asymptotic convergence;
- exact continuous integration of detection hazard;
- a new Bellman recursion or general-purpose Stackelberg algorithm;
- formal 3D feasibility, optimality, convergence, or equilibrium guarantees;
- literature priority claims until primary sources have been verified.

Detection hazard quadrature remains part of the finite objective definition.
The high-fidelity evaluator independently reevaluates selected policies; it is
not a continuous optimality certificate.

## Formal-result contract

P1 and P2 must distinguish three results:

1. all-segment terrain/LOS feasibility under the implemented piecewise-cubic
   terrain and piecewise-linear swept LOS geometry;
2. soundness and completeness relative to the declared finite switching,
   target, action, speed, terminal, and feasibility rules; and
3. exact minimum cost over that finite graph.

Completeness never quantifies over arbitrary continuous trajectories.

## Numerical-evidence contract

- Direction B is a frozen nested-grid sensitivity and production-lattice
  study, not a proof of continuous convergence.
- P3 must freeze randomized-terrain generation and heuristic rules before the
  confirmation solve and must report failures only when reproduced.
- P4/C-lite is exact only over its stated finite defender set and the B4 finite
  follower graph.
- P5 may claim a terrain-induced structural transition only if the complete
  recorded mechanism chain is observed and reproduced.
- P7 is limited to one qualitative 3D figure.

## Evidence-to-claim map

| Claim | Required evidence |
|---|---|
| Finite physical reduction | graph definition, exact endpoint construction, P1 geometry certificate |
| Soundness/completeness | P2 mathematical proofs plus hypothesis regressions |
| Finite follower exactness | DAG proof, exhaustive enumeration proof, independent shortest-path regression |
| Heuristic insufficiency | frozen P3 counterexample/ablation and common evaluation |
| Finite Stackelberg solution | exhaustive P4 sensor enumeration and deterministic tie rule |
| Terrain strategic mechanism | controlled P5 parameter continuation and mechanism diagnostics |
| Broader 3D applicability | one explicitly qualitative P7 figure |

## Change control

Any later broadening of the central claims requires all three of:

1. an explicit edit to this document with a dated reason;
2. a matching mathematical or numerical acceptance criterion; and
3. an update to `p1b_roadmap_0729.md`.


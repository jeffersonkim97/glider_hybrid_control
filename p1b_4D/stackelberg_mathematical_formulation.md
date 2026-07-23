# Hierarchical Stackelberg Security Problem

## Phase 0–2: Authoritative Mathematical Formulation

**Status:** mathematical specification only. This document contains no implementation, algorithm selection, or notebook-cell design.

> **SUPERSEDED (2026-07-23):** Following advisor review, the Attacker best response is computed exclusively by Bellman dynamic programming (`p1b_4D.bellman.select_authoritative_bellman_response`) — an optimal solution of the discretized dynamic-programming formulation on the switching-point and state-action grid, not a continuous global optimum. The "CasADi NLP role" section and every reference to continuous NLP refinement of a selected Bellman candidate are **deprecated and disconnected**; `attacker_nlp.py` is retained only as an offline discretization-error comparison, never called by the authoritative solver. Bellman and NLP no longer need to "use the same normalized objective" for correctness — Bellman alone defines \(J_A\) for the reported response.

This document defines the mathematical objects and relationships that every future implementation shall preserve.

## 1. Stackelberg game

The game has one Defender and one Attacker:

- Leader: Defender
- Follower: Attacker

Let the Defender strategy space be \(\mathcal D\), the Attacker feasible-strategy correspondence be \(\mathcal A(d)\), and the game data be fixed. The Defender first chooses \(d\in\mathcal D\). The Attacker observes \(d\) and then chooses \(a\in\mathcal A(d)\).

The game is therefore sequential and nested. It shall not be reformulated as a simultaneous or reversed-order game.

## 2. Defender strategy

The Defender strategy is the continuous scalar

\[
d := z_{\mathrm{sensor}},
\qquad
\mathcal D=[z_{\mathrm{sensor}}^{\min},z_{\mathrm{sensor}}^{\max}]
\subset\mathbb R.
\]

For terrain elevation \(h_{\mathrm{terrain}}(z)\) and nonnegative mount height \(h_{\mathrm{mount}}\), the sensor position is

\[
p_{\mathrm{sensor}}(d)
=
\begin{bmatrix}
d\\
h_{\mathrm{sensor}}(d)
\end{bmatrix},
\qquad
h_{\mathrm{sensor}}(d)
=h_{\mathrm{terrain}}(d)+h_{\mathrm{mount}}.
\]

The Defender decision variable shall remain continuous. Any numerical sampling used to evaluate or initialize a future optimizer does not redefine \(d\) as a discrete strategy.

## 3. Attacker strategy

For a fixed Defender decision \(d\), the Attacker strategy is

\[
a=(s,\chi_g)\in\mathcal A(d),
\]

where the switching point is

\[
s=(z_{\mathrm{switch}},h_{\mathrm{switch}})
\]

and the glide trajectory is

\[
\chi_g
=
\bigl(p_g(\cdot),v_g(\cdot),\gamma_g(\cdot)\bigr).
\]

Here

\[
p_g(t)=
\begin{bmatrix}
z_g(t)\\
h_g(t)
\end{bmatrix},
\qquad
t\in[0,T_g],
\]

\(v_g(t)\) is the speed profile, and \(\gamma_g(t)\) is the flight-path-angle profile.

The glide boundary conditions are mandatory:

\[
p_g(0)=
\begin{bmatrix}
z_{\mathrm{switch}}\\
h_{\mathrm{switch}}
\end{bmatrix},
\qquad
p_g(T_g)=
\begin{bmatrix}
z_{\mathrm{goal}}\\
0
\end{bmatrix}.
\]

Thus the glide trajectory begins exactly at the switching point and terminates at \((z_{\mathrm{goal}},h_{\mathrm{goal}})\), with

\[
h_{\mathrm{goal}}=0.
\]

The switching point, position trajectory, speed profile, and \(\gamma\) profile are continuous decision objects whenever the mathematical and physical constraints permit.

## 4. Detection model decomposition

Let

\[
P_{A,p}(a,d)
\]

denote the cumulative acoustic probability of detection during the powered phase. Let

\[
P_{R,g}(a,d),\qquad
P_{V_r,g}(a,d),\qquad
P_{\mathrm{RCS},g}(a,d)
\]

denote the cumulative radar, radial-velocity, and radar-cross-section detection contributions during the glide phase.

The project-level detection aggregation operator is denoted by \(\mathcal F_{\mathrm{det}}\). The total probability of detection is

\[
P_{\mathrm{det}}(a,d)
=
\mathcal F_{\mathrm{det}}
\left(
P_{A,p},
P_{R,g},
P_{V_r,g},
P_{\mathrm{RCS},g}
\right).
\]

The aggregation operator shall preserve the following phase allocation:

- powered phase: acoustic detection;
- glide phase: radar, radial-velocity, and RCS detection.

No future implementation may silently move a detection contribution to a different phase. The exact fusion law shall be supplied by the detection-model contract while producing a valid probability:

\[
0\le P_{\mathrm{det}}(a,d)\le 1.
\]

## 5. Mission time

Let \(T_p(a,d)\ge 0\) be powered time and \(T_g(a,d)\ge 0\) be glide time. Total mission time is

\[
T_{\mathrm{mission}}(a,d)
=T_p(a,d)+T_g(a,d).
\]

Both phase durations contribute to the Attacker objective.

## 6. Normalized quantities

Define fixed normalization maps

\[
\mathcal N_{\mathrm{PoD}},\qquad
\mathcal N_T,\qquad
\mathcal N_C,
\]

shared by all solvers and all Defender evaluations. Then

\[
\mathrm{PoD}_{\mathrm{Normalized}}(a,d)
=
\mathcal N_{\mathrm{PoD}}
\bigl(P_{\mathrm{det}}(a,d)\bigr),
\]

\[
\mathrm{Time}_{\mathrm{Normalized}}(a,d)
=
\mathcal N_T
\bigl(T_{\mathrm{mission}}(a,d)\bigr),
\]

and

\[
\mathrm{CoverageArea}_{\mathrm{Normalized}}(d)
=
\mathcal N_C
\bigl(A_{\mathrm{LOS}}(d)\bigr).
\]

The normalization maps, reference values, and bounds are part of the fixed game definition. They shall be identical in Bellman evaluation, CasADi NLP refinement, response selection, Defender evaluation, and final reporting.

## 7. Attacker objective

For nonnegative fixed weights \(w^{A}_{\mathrm{pod}}\) and \(w_t\), the Attacker minimizes

\[
J_A(a,d)
=
w^{A}_{\mathrm{pod}}
\,\mathrm{PoD}_{\mathrm{Normalized}}(a,d)
+
w_t
\,\mathrm{Time}_{\mathrm{Normalized}}(a,d).
\]

The weights and normalized component definitions are invariant across the complete Attacker solution procedure. In particular, multi-start diversity shall not be created by changing objective weights.

## 8. LOS coverage and Defender objective

Let \(\Omega\) be the coverage domain and let

\[
\mathbf 1_{\mathrm{LOS}}(x;d)
\]

indicate that point \(x\in\Omega\) has line of sight to the sensor placed at \(d\). The LOS coverage area is

\[
A_{\mathrm{LOS}}(d)
=
\int_{\Omega}
\mathbf 1_{\mathrm{LOS}}(x;d)\,\mathrm dx.
\]

Coverage area always means LOS coverage area; it shall not be replaced by geometric range area, detection-threshold area, or another coverage definition.

For nonnegative fixed weights \(w^{D}_{\mathrm{pod}}\) and \(w_{\mathrm{cover}}\), the Defender objective is

\[
J_D(a,d)
=
w^{D}_{\mathrm{pod}}
\,\mathrm{PoD}_{\mathrm{Normalized}}(a,d)
+
w_{\mathrm{cover}}
\,\mathrm{CoverageArea}_{\mathrm{Normalized}}(d).
\]

The Defender objective shall always be evaluated using the refined Best-found Attacker Response returned by the full Attacker solver.

## 9. Nested Stackelberg optimization

For each fixed \(d\in\mathcal D\), the ideal Attacker best-response correspondence is

\[
\operatorname{BR}_A(d)
=
\arg\min_{a\in\mathcal A(d)}
J_A(a,d).
\]

Writing a selected ideal response as \(a^*(d)\in\operatorname{BR}_A(d)\), the ideal Stackelberg problem is

\[
d^*
\in
\arg\max_{d\in\mathcal D}
J_D\bigl(a^*(d),d\bigr).
\]

Operationally, the specified numerical pipeline returns a Best-found Attacker Response rather than a proven global minimizer. Denote this solver mapping by

\[
\widehat a(d)
:=
\mathcal S_A(d).
\]

The implemented Defender problem shall therefore be reported as

\[
\widehat d
\in
\arg\max_{d\in\mathcal D}
J_D\bigl(\widehat a(d),d\bigr),
\]

with final reported pair

\[
(\widehat d,\widehat a(\widehat d)).
\]

This notation preserves the ideal mathematical game while avoiding an unsupported global-optimality claim for the numerical Attacker response.

## 10. Mandatory Attacker solver mapping

For every Defender decision \(d\), the Attacker solver is the composition

\[
\mathcal S_A
=
\mathcal S_{\mathrm{select}}
\circ
\mathcal S_{\mathrm{NLP}}^{\mathrm{multi}}
\circ
\mathcal S_{\mathrm{warm}}
\circ
\mathcal S_{\mathrm{top}K}
\circ
\mathcal S_{\mathrm{filter}}
\circ
\mathcal S_{\mathrm{Bellman}}^{\mathrm{multi}}.
\]

Its mandatory stages are:

1. multi-start coarse Bellman;
2. candidate filtering;
3. Top-\(K\) candidate selection;
4. Bellman-to-NLP warm-start construction;
5. multi-start CasADi NLP;
6. feasible refined-response selection.

The resulting value

\[
\widehat a(d)=\mathcal S_A(d)
\]

is the Best-found Attacker Response.

### Bellman role

Bellman performs global topology discovery and generates candidate switching points, candidate glide paths, and warm starts. Its candidates may be coarse or discretized, but they are not final Attacker responses. Bellman performs no continuous refinement.

### CasADi NLP role

For every selected Bellman candidate, CasADi NLP refines the switching point, trajectory, controls, and objective over continuous decision variables. The response selector chooses the best feasible refined result according to the unchanged \(J_A\).

Neither an individual NLP result nor the selected result shall be called a global optimum.

## 11. Defender-optimizer interface

The Defender optimizer treats the complete Attacker solver as a nested black box. Define the evaluation map

\[
\mathcal E_D(d)
=
\left(
\widehat a(d),
J_D\bigl(\widehat a(d),d\bigr),
\eta(d)
\right),
\]

where \(\eta(d)\) contains feasibility, solver-status, objective-component, and provenance information.

For every requested continuous Defender decision \(d\), evaluation of \(\mathcal E_D(d)\) requires a complete execution of \(\mathcal S_A(d)\). Cached evaluation is mathematically admissible only when it represents the same \(d\), game data, solver contract, tolerances, and configuration.

The Defender-optimizer contract is:

\[
\mathcal S_D:
\bigl(\mathcal D,\mathcal E_D,\text{termination contract}\bigr)
\longmapsto
\bigl(\widehat d,\widehat a(\widehat d),J_D,\text{provenance}\bigr).
\]

No specific outer optimization algorithm is prescribed. In particular, this formulation does not commit to gradient descent, Bayesian optimization, Brent search, golden-section search, or any other named method.

## 12. Mathematical invariants

Every future implementation shall preserve all of the following:

1. The Defender moves first and the Attacker observes the Defender decision.
2. \(z_{\mathrm{sensor}}\) is a continuous Defender variable.
3. Sensor altitude is determined by terrain elevation and mount height.
4. The Attacker strategy contains both the switching point and the complete glide trajectory.
5. The glide begins at the switching point and ends at \((z_{\mathrm{goal}},0)\).
6. Powered detection is acoustic; glide detection contains radar, radial-velocity, and RCS contributions.
7. Mission time is powered time plus glide time.
8. Bellman and NLP use the same normalized Attacker objective and weights.
9. Bellman generates topology and warm starts but never constitutes the final response.
10. CasADi NLP performs continuous refinement.
11. The selected refined result is called the Best-found Attacker Response, not the global optimum.
12. Every Defender evaluation executes or retrieves an equivalent complete nested Attacker solve.
13. The Defender objective uses only the refined response and LOS coverage area.
14. The outer Defender optimizer remains implementation-independent.

This Phase 0–2 formulation is the mathematical source of truth for all subsequent phases.

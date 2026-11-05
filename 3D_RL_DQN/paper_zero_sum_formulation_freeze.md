# Paper Zero-Sum Formulation Freeze

Frozen: 2026-11-03

## Purpose

This document freezes the primary paper formulation before the Bellman--DQN
experiments are re-run with the zero-sum Defender objective. It consolidates
the formulation in the IE690 midterm report with the earlier P1b formulation
decision log and separates the primary theorem-bearing experiment from the
existing non-zero-sum diagnostic experiments.

This freeze changes neither the physical attacker model nor the learned
Attacker policy. It changes only the scalar value maximized by the Defender's
outer search.

## Source of truth

- Author report: `IE690_Midterm_Report.pdf`, SHA-256
  `71c5aebf5d78f7bb4aed007150858289e0f0cc3ed59e9d16129689e7ef1143a2`.
- Formal decisions: `3D_0827/docs/p1b_formulation/decision_log.md`, especially
  D1, D2, D19, D21, D25, D26, and D27.
- Formal draft: `3D_0827/docs/p1b_formulation/formulation_draft.md`.
- Authoritative exact Attacker model: `3D_0827`.
- Frozen learned Attacker implementation and checkpoints: `3D_RL_DQN`.

The PDF is an external source document and is not copied into the repository
by this freeze.

## Primary paper game

For each feasible Defender placement `d`, let `R(d)` be the finite nonempty set
of powered-switching-state and feasible-glide-trajectory responses. The common
mission cost is

```text
J(response; d) = normalized hazard cost + normalized mission-time cost.
```

The Attacker minimizes this cost and the Defender maximizes the resulting
best-response value:

```text
U(d) = min_{response in R(d)} J(response; d),
d* in argmax_{d in D} U(d).
```

This is the primary zero-sum abstraction. Detection probability `P_D` remains
a reported physical diagnostic but is not the value climbed by the primary
Defender search.

## Learned response and certificate

The frozen masked policy produces a feasible response from every admissible
switching state. The exact evaluator scores all policy trajectories and keeps
the minimum:

```text
U_tilde(d) = minimum J among the frozen policy's feasible responses at d.
delta(d) = U_tilde(d) - U(d).
```

Because `U(d)` is the minimum over every feasible Attacker response and the
masked policy returns a feasible response,

```text
delta(d) >= 0.
```

Let `d_tilde` be the placement at which strict best-improvement local search
using `U_tilde` stops over the configured Chebyshev `r`-neighborhood. Then

```text
U(d') <= U(d_tilde) + delta(d_tilde)
for every d' in N_r(d_tilde).
```

Thus `d_tilde` is a `delta(d_tilde)`-approximate `r`-local Stackelberg
solution. One exact Bellman response at `d_tilde` measures the certificate.
No exact Bellman sweep over all neighbors of `d_tilde` is required for this
zero-sum certificate.

## Frozen implementation contract

- Attacker objective: unchanged `J_A`, minimized by Bellman and DQN.
- Defender outer-search value in the primary experiment: `J_A`, maximized.
- Detection probability `J_D = P_D`: recorded only as a diagnostic.
- DQN checkpoints: unchanged frozen condition-specific checkpoints.
- DQN training, fine-tuning, and checkpoint selection during evaluation:
  forbidden.
- Feasible-action mask, switching candidates, powered occlusion, transition
  graph, terminal set, and exact trajectory scorer: unchanged.
- Defender grid, initial action, Chebyshev neighborhood, strict-improvement
  rule, deterministic tie order, and stopping rule: unchanged.
- Exact Attacker-response caches may be reused because an exact response does
  not depend on which scalar the Defender subsequently maximizes.
- Each result must state its leader objective explicitly. A zero-sum result
  must never overwrite a legacy detection-probability result.
- The one-sided inequality must be checked numerically. A material negative
  `U_tilde(d) - U(d)` is a correctness failure, not an accuracy result.

## Relationship to existing experiments

Phase 6--9 results generated with Defender value `P_D` remain valid
non-zero-sum diagnostic data. They are not evidence for the single-point
zero-sum certificate and will not be relabeled.

The non-zero-sum experiments show how a small Attacker-objective error can
change the Defender action when the leader maximizes a different payoff. Their
selected-plus-neighbor exact audit remains the correct certificate for that
reported discrete non-zero-sum local solution.

## Primary paper claim boundary

The primary claim is limited to the finite discretized game and the tested
terrain distribution. It does not claim continuous-space optimality,
discretization-error bounds, global optimality of a local search, or universal
terrain generalization.

The empirical study will report the computational conditions under which the
frozen learned response offers a time advantage, together with its measured
`delta` certificate. The final Bellman-versus-RL method choice remains a
research decision based on the reported runtime, amortized training cost, and
certificate size.

## Immediate execution sequence

1. Add an explicit zero-sum leader-objective mode without changing the default
   legacy non-zero-sum path.
2. Unit-test payoff routing and the one-sided certificate calculation.
3. Run one CUDA smoke case after the currently active simulation finishes.
4. Re-run the controlled spatial experiment on final-ID terrains 000--003.
5. Extend the angular and neighborhood main-effect experiment only after the
   spatial zero-sum results pass the contract checks.


# Slide 1 - Final Goal and Current Progress

## Final Goal - Target: November 13

Determine the computational conditions under which we should use:

- **Exact Bellman dynamic programming**, or
- **A DQN-based approximation with an approved accuracy level**

The decision boundary will be based on spatial/angular discretization, neighboring radius, computation time, memory use, and errors in J_A and J_D.

## Current Position

**Framework complete** -> **Generalized DQN dataset complete** -> **Resolution sweep in progress** -> Unseen-terrain tests -> Full-SSE comparison -> Bellman/DQN decision boundary

- Completed: 100 m, 50 m, and 25 m experiments
- In progress: 10 m experiment - 9,000/10,000 episodes completed
- Next: sealed ID/OOD terrain evaluation and selected full-SSE cases

**Suggested visual:** A horizontal milestone arrow showing the current position at the resolution-sweep stage.

---

# Slide 2 - Current Work and Preliminary Results

## Multi-Terrain Generalization Experiment

- One DQN is trained across **8 random-box terrains / 32 scenarios**.
- Validation uses **4 separate terrains / 16 scenarios**.
- Start, goal, sensor location, and sensor visibility vary across scenarios.
- The same feasible scenario bank is used at 100, 50, 25, and 10 m.

## Measured Results - Seed 0

| Spatial resolution | Bellman time for 48 BR cases | Pure DQN training time | Validation median J_A error | Validation max J_A error |
|---:|---:|---:|---:|---:|
| 100 m | 0.34 min | 3.51 min | 6.36% | 13.01% |
| 50 m | 2.38 min | 9.08 min | 8.16% | 28.48% |
| 25 m | 10.47 min | 20.45 min | 7.42% | 12.16% |
| 10 m | 154.35 min | >=66.9 min at 9k episodes | 13.17% at 9k | 73.33% at 9k |

## Preliminary Observation

- Bellman reference computation increased by approximately **450x from 100 m to 10 m**.
- DQN produced feasible trajectories with **100% validation success**, but worst-case accuracy is not yet stable.
- The 10 m experiment is still running; its values are not final.

**Suggested visual:** Two-panel plot: (1) log-scale computation time versus resolution and (2) median/max J_A error versus resolution.

---

# Slide 3 - Current Roadblocks

## 1. Fine-Resolution Evaluation Cost

- At 10 m, validation rollout - not neural-network training - is the dominant cost.
- Through 9,000 episodes:
  - Pure DQN training: **66.9 min**
  - Scheduled evaluation: **630.8 min**
- Approximately **90% of DQN wall time** is currently spent on evaluation.
- Mean validation inference time reached **177 s per case** at 9,000 episodes.

## 2. Unstable Worst-Case Error

- Trajectory success remains 100%, but max J_A error varies substantially between checkpoints.
- No resolution has yet satisfied the approved accuracy gates for three consecutive evaluations.
- The current median-first checkpoint ranking can select a model with low median error but unacceptable max error.

## 3. Remaining Scope Gap

- Current measurements are attacker best-response J_A results.
- Unseen-terrain testing, full SSE, and J_D comparison are not yet complete.

**Suggested visual:** Stacked bars comparing pure training time and evaluation time at each resolution, highlighting the 10 m evaluation bottleneck.

---

# Slide 4 - Plan for the Coming Week

1. **Complete and consolidate the 10 m experiment.**
   - Finalize accuracy, Bellman time, training time, and evaluation time.

2. **Correct the model-selection and evaluation protocol.**
   - Select checkpoints by success and accuracy-gate compliance before median error.
   - Reduce redundant fine-resolution evaluation while preserving a fair comparison.

3. **Freeze the training and checkpoint-selection contract.**
   - No further tuning after the test protocol is opened.

4. **Run sealed generalization tests.**
   - Unseen random-box terrains for in-distribution testing
   - Corridor/wall and stepped-pyramid terrains for out-of-distribution testing
   - Mesh or height-map terrain as the final extension

5. **Perform the final Bellman-DQN comparison.**
   - Compare J_A, runtime, and resource use on unseen terrains.
   - Run selected full-SSE cases to compare J_D.
   - Produce the Bellman-versus-DQN computational decision boundary by November 13.

**Expected deliverable:** A decision chart specifying when Bellman remains computationally practical and when a validated DQN approximation is preferable.

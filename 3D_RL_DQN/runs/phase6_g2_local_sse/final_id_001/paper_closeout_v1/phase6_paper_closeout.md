# Phase 6 Paper-Ready Integration Closeout

Frozen G2 DQN versus exact Bellman Local-SSE on final-ID Terrain 001. 
No training, fine-tuning, or checkpoint selection was performed.

## Primary timing definition

The primary DQN time is the first Local-SSE solve with empty scenario, 
observation, and action-row caches after shared scene construction and 
checkpoint loading. The repeated identical-query time is secondary only.

| dx | Bellman first solve | DQN first solve | DQN repeated | Bellman/DQN first | J_A error | J_D regret | exact local certificate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 100 m | 50.889 s | 65.014 s | 0.339 s | 0.783x | 0.00% | 0.00% | PASS |
| 50 m | 131.597 s | 51.062 s | 0.974 s | 2.577x | 6.72% | 79.70% | FAIL |
| 25 m | 961.422 s | 116.917 s | 19.529 s | 8.223x | 8.23% | 90.42% | FAIL |
| 10 m | 5999.106 s | 2022.959 s | 3787.477 s | 2.966x | 4.52% | 42.38% | FAIL |

## Phase-6 finding

All four DQN responses satisfy the Phase-5 same-sensor J_A 10% diagnostic, 
but only dx=100 m preserves the exact Local-SSE sensor decision. At that 
coarse condition Bellman is faster for a first unseen solve. Therefore no 
observed Phase-6 condition supports DQN as both first-solve faster and 
leader-decision admissible. This is a single-terrain integration result, 
not the Phase-7 Monte Carlo regime boundary.

## Artifacts

- aggregate_json: `D:/git/git_research/glider_hybrid_control/3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_paper_closeout.json`
- aggregate_csv: `D:/git/git_research/glider_hybrid_control/3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_paper_closeout.csv`
- report: `D:/git/git_research/glider_hybrid_control/3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_paper_closeout.md`
- runtime_figure: `D:/git/git_research/glider_hybrid_control/3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_runtime_first_vs_repeated.png`
- quality_figure: `D:/git/git_research/glider_hybrid_control/3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_quality_error_vs_regret.png`
- search_trace_figure: `D:/git/git_research/glider_hybrid_control/3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_defender_search_traces.png`
- solution_3d_dx100m: `D:/git/git_research/glider_hybrid_control/3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_final_solutions_dx100m_3d.html`
- solution_3d_dx50m: `D:/git/git_research/glider_hybrid_control/3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_final_solutions_dx50m_3d.html`
- solution_3d_dx25m: `D:/git/git_research/glider_hybrid_control/3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_final_solutions_dx25m_3d.html`
- solution_3d_dx10m: `D:/git/git_research/glider_hybrid_control/3D_RL_DQN/runs/phase6_g2_local_sse/final_id_001/paper_closeout_v1/phase6_final_solutions_dx10m_3d.html`

## Scope limitations

- Terrain 001 only; no Monte Carlo uncertainty statement.
- dpsi=5 deg and r=1 only.
- Bellman versus generalized DQN is the primary paper comparison; the 
  case-specific Tabular baseline remains deferred to the Phase-7 secondary analysis.
- The 100 m defender region was symmetrically center-aligned to the lattice.
- J_A closeness alone is not an end-to-end Stackelberg certificate.

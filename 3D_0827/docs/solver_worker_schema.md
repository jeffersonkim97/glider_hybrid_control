# Solver-agnostic 워커 스키마

**Phase 1 산출물** · Stage 15
관련: [`stage15_plan.md`](stage15_plan.md) · [`exact_vs_rl_decision_framework.md`](exact_vs_rl_decision_framework.md)

RL 워커가 기존 벤치마크 하네스를 **수정 없이** 통과하기 위한 입출력 규약.

---

## 1. 결론부터 — 러너 수정은 불필요하다

`stage14_benchmark_runner.py`의 하네스는 이미 solver-agnostic이다. 확인한 근거:

- `run_benchmark_cases(..., worker_path: Path | None = None)` — 실행할 워커 스크립트를 인자로 받는다.
  기본값이 `stage14_2_worker.py`일 뿐 하드코딩이 아니다.
- 워커는 별도 프로세스로 실행된다:
  `python <worker> --configuration <config.json> --output <result.json>`
- `run_limited_subprocess()`가 부모에서 wall-time과 RSS 상한을 강제하므로,
  **계측 방식이 solver 종류와 무관하다.** Q-table 메모리도 프로세스 RSS에 그대로 잡힌다.
- `stage14_2_worker.py`는 얇은 디스패처다: config JSON을 읽고, 프로파일 함수를 부르고,
  payload를 `write_json`으로 쓰고, 한 줄 출력한다.

따라서 Stage 15는 **같은 CLI 규약을 지키는 새 워커 스크립트**만 작성하면 된다.
`run_benchmark_cases`, `run_limited_subprocess`, `numeric_statistics`,
`summarize_repetitions`는 그대로 재사용한다.

---

## 2. CLI 규약

```
python <worker>.py --configuration <path> --output <path>
```

| 항목 | 규약 |
|---|---|
| 입력 | `BenchmarkCase.as_configuration()` + `case_id` + `repetition_id`가 담긴 JSON |
| 출력 | payload JSON을 `--output` 경로에 기록 |
| 종료 코드 | 정상 0. 0이 아니면 러너가 `worker_failure`로 기록 |
| stdout | 한 줄 요약 (러너가 `worker_stdout`에 보존) |
| 프로세스 | 반복마다 새 cold process |

**주의**: stdout은 Windows에서 cp1252로 인코딩될 수 있다. 요약 줄에는 ASCII만 쓴다
(Phase 0에서 `Δ`·`×` 때문에 실제로 깨진 적 있음).

---

## 3. Payload 필드

`stage14_benchmark_runner.py`의 `_row_from_attempt`(246-330행)가 실제로 읽는 키를 기준으로 분류했다.
`payload.get(...)`로 읽으므로 **없는 키는 `None`이 되고 러너는 죽지 않는다.**
즉 exact 전용 필드를 RL 워커가 생략해도 안전하다.

### 3.1 양쪽 공통 — 필수

| 키 | 타입 | 의미 |
|---|---|---|
| `status` | str | §4의 어휘 중 하나. 이것만 진짜 필수다 |
| `timing.totals.T_SSE_s` | float | 러너가 주 런타임 통계로 집계 |
| `solution_identity.attacker_objective` | float | → 행의 `J_A` |
| `solution_identity.defender_objective_pod` | float | → 행의 `J_D` |
| `solution_identity.feasible` | bool | |
| `solution_identity.selected_defender_action_id` | int | 선택된 배치 |
| `state_and_game_size` | dict | `N_S_*`, `N_E`, `N_D`, `B`, `Q` 등 |
| `algorithm_variant` | str | `exact_local_sse` / `rl_local_sse` / `global_oracle` |
| `complete_configuration` | dict | → 행의 `worker_configuration` |

메모리는 워커가 보고하지 않는다. 부모가 psutil로 샘플링해 `memory.peak_rss_bytes`에 채운다.

### 3.2 Stage 15 신규 — timing 분해

프레임워크 §5의 비용 회계를 표현하려면 기존 `T_SSE_s` 하나로는 부족하다.

```json
"timing": {
  "totals": {
    "T_SSE_s":   0.0,     // 기존 키. 총 벽시계 시간 (호환 유지)
    "T_train_s": 0.0,     // 신규. RL 학습 시간. exact 워커는 0.0
    "T_query_s": 0.0      // 신규. 학습된 정책으로 defender 1개를 평가한 시간
  },
  "components": { "T_graph_s": 0.0, "T_hazard_s": 0.0, "T_Bellman_s": 0.0, "...": 0.0 }
}
```

규칙:
- `T_SSE_s`는 항상 채운다. 기존 집계 코드가 이 키를 본다.
- exact 워커는 `T_train_s = 0.0`, `T_query_s`는 BR 1회 시간.
- RL 워커는 `T_train_s`에 학습 시간, `T_query_s`에 질의 1회 시간을 넣는다.
- 손익분기 `T_train(B*) < (2r+1)² · (T_BR − T_query)` 계산이 이 세 값으로 가능해야 한다.

### 3.3 Stage 15 신규 — 정확도

```json
"exactness": {
  "exact": false,                    // RL이면 false
  "reference": "exact_local_sse_same_conditions",
  "regret_pod": 0.0,                 // 프레임워크 3c. 영역 II에서는 null
  "J_D_true_selected": 0.0,          // RL 선택 배치를 exact BR로 재평가한 진짜 PoD
  "measurement_region": "I"          // "I" | "II" | "III" (프레임워크 4)
}
```

**`solution_identity.defender_objective_pod`를 정확도 판정에 쓰지 말 것.**
RL의 attacker 응답이 최적보다 나쁘면 그 값은 체계적으로 부풀려진다(프레임워크 §3b).
판정에는 반드시 `exactness.J_D_true_selected`와 `regret_pod`를 쓴다.

### 3.4 RL 전용

```json
"rl": {
  "training_config": { "...": "RLTrainingConfig as_dict()" },
  "seed": 0,
  "episodes_completed": 0,
  "budget_reached": false,
  "final_bellman_residual": 0.0,
  "q_table_bytes": 0
}
```

### 3.5 Exact 전용 — RL 워커는 생략

`trajectory_identity`, `independent_replay`, `leader_cooptimal_action_ids`,
`attacker_objective_cooptimal_candidate_ids`, `tie_break_convention`,
`solver_source_fingerprint`, `oracle_metadata`.

RL 해에는 exact tie-break 개념이 적용되지 않으므로 co-optimal 집합을 만들지 않는다.
단 **`TrajectoryReplayAudit`에는 RL 정책도 통과시킨다** — 이때는 exact 최적성 인증이 아니라
"이 궤적이 물리적으로 재현 가능한가"의 검증이며, 결과는 `rl.replay_audit`에 따로 기록한다.

---

## 4. 상태 어휘

`stage14_benchmark_runner.TERMINAL_STATUSES`를 그대로 계승하고 두 개만 추가한다
(`stage15_0_contract.RL_TERMINAL_STATUSES`).

| 상태 | 출처 | 결정 지도에서 |
|---|---|---|
| `completed` | 기존 | 값으로 판정 |
| `timeout`, `memory_limit` | 기존, 부모 강제 | exact 쪽이면 "exact 불가" |
| `model_infeasible` | 기존 | 조건에서 제외 |
| `worker_failure` | 기존 | 진단 필요 |
| **`rl_budget_exhausted`** | 신규 | **RL 부적합** — 실패가 아니라 정당한 결과 |
| **`rl_non_convergent`** | 신규 | RL 부적합, 진단 필요 |

`rl_budget_exhausted`를 crash와 구분하는 것이 핵심이다. 예산을 다 쓰고도 τ에 못 미친 것은
"이 조건에서는 RL을 쓰지 말라"는 **답**이지 버그가 아니다.

---

## 5. 구현 시 확인 사항

- [ ] 새 워커가 `--configuration` / `--output` 규약을 지키는가
- [ ] `status`가 §4 어휘 안에 있는가
- [ ] `timing.totals.T_SSE_s`가 항상 채워지는가 (기존 집계 호환)
- [ ] `T_train_s` / `T_query_s`가 분리 기록되는가
- [ ] 정확도 판정에 `defender_objective_pod` 대신 `exactness.*`를 쓰는가
- [ ] stdout 요약이 ASCII 전용인가
- [ ] `run_benchmark_cases(worker_path=...)`로 실제 1회 통과하는가

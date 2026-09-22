# SSE–RL 결정 경계 — Stage 15 연구 계획

**범위**: `3D_0827` / 지형 `centered_cube` 고정 / Stage 14 frozen contract 불변
**Knob**: `Δ_state` · `N_C` · `r`
**Phase**: 0 → 4

어떤 computational condition에서 exact Strong Stackelberg equilibrium을 계산하는 것이 유리하고,
어떤 조건에서 RL approximation이 유리한가. 새 알고리즘을 제안하지 않고, 기존 exact 솔버와
tabular Q-learning의 비용·정확도를 같은 문제 위에서 측정해 선택 기준을 empirical하게 도출한다.

---

## 01 · 결정 규칙

조건 `c = (Δ_state, N_C, r)`, 정확도 임계값 `τ`, RL 계산 예산 `B`에 대해 다음을 평가한다.
RL의 예산–정확도 곡선을 조건마다 하나의 숫자 `B*`로 축약하는 것이 핵심이다.

```
B*(c, τ) = min { B : accuracy(c, B) ≥ τ }

RL 선택  ⟺  B*(c, τ) 가 존재  ∧  T_train(B*) + T_query(c) < T_exact(c)
그 외에는 exact 선택
```

`τ`는 고정 상수가 아니라 **파라미터**다. τ를 올릴수록 RL 유리 영역이 어떻게 줄어드는지가
이 연구의 민감도 결과이며, 단일 교차점보다 강한 주장이 된다.

### 결과 지도의 네 가지 셀

| 셀 | 의미 |
|---|---|
| **exact 유리** | exact가 충분히 빠르다. RL을 쓸 이유가 없다. |
| **RL 유리** | exact는 비싸고, RL이 τ를 통과하면서 더 빠르다. |
| **RL 부적합** | exact가 비싸지만 RL이 τ에 못 미치거나 Q-table이 메모리 한도를 넘는다. "RL을 쓰지 말라"도 결과다. |
| **exact 불가 · RL 가용** | **이 연구의 핵심 주장 영역.** exact가 timeout·memory로 완주하지 못하는 구간에서, 오차가 다소 크더라도 RL은 답을 낸다. 가용성 자체가 결과다. |

> **Phase 0 확정 사항 2건이 이 절을 수정한다** — 상세는
> [`exact_vs_rl_decision_framework.md`](exact_vs_rl_decision_framework.md) §3c · §4.
>
> 1. **exact 기준선은 전역 최적이 아니라 동일 조건의 exact local SSE다.** 바깥 탐색(반경 r)을
>    양쪽 고정하고 안쪽 best response만 exact ↔ RL로 바꾼다. 2D defender 격자에서 전역 열거는
>    25 m에 825회, 5 m에 19,481회 full BR을 요구해 불가능하다. Global oracle은 100 m · 50 m의
>    sanity check로만 쓴다.
> 2. **"ground truth가 없다"는 서술은 너무 비관적이었다.** 단일 BR 평가는 훨씬 오래 감당 가능하므로,
>    exact local SSE가 완주하지 못하는 구간에서도 RL 배치의 **진짜 PoD는 정확히 측정된다**
>    (모르는 것은 "그보다 나은 배치가 있는가"뿐). 측정 영역을 셋으로 나눠 서술한다.

---

## 02 · 고정 설정

`stage11_config.py` 기준. 1 map unit = 100 m, 도메인은 1600 × 800 × 500 m.
terrain 변형(`stepped_pyramid` 등)은 이번 stage 대상이 아니며 후속 stage에서 추가한다.

| 항목 | 값 | 비고 |
|---|---|---|
| Graph bounds | x ∈ [−8, 8], y ∈ [−4, 4] | map unit |
| Start / Goal | (−8, 0, 0) / (8, 0, 0) | — |
| Terrain | `centered_cube` | 중심 (0,0), 4×4 footprint → x ∈ [−2, 2] |
| Defender 영역 | **x ∈ [2, 8], y ∈ [−4, 4], z = 0** | terrain max x → goal x, y는 전체. 600 × 800 m |
| Defender 격자 | state의 x, y 격자와 동일 | 별도 간격 파라미터 없음 |
| 이웃 형태 | Chebyshev 정사각 | 이웃 수 = (2r+1)² |

### 해상도별 Defender action 수 N_D

기존 1D 6개(`{5,…,10}`)를 폐기한 결과. canonical 25 m에서 약 140배가 되며,
global oracle은 이만큼의 full exact solve를 요구한다 — "정확하지만 5시간" 상황이 여기서 실제로 재현된다.

*(아래 수치는 `defender_grid_2d.summarize_resolutions()`가 실제 장면 기하에서 계산한 값이다.)*

| 공간 해상도 | Defender 격자 (x × y) | N_D | 포화 시작 r (중심 출발) | r = 1…10 스윕 |
|---|---|---|---|---|
| 100 m | 7 × 9 | 63 | 4 | r ≥ 5 포화 |
| 50 m | 13 × 17 | 221 | 8 | r = 10 만 포화 |
| 25 m *(canonical)* | 25 × 33 | **825** | 16 | **전 구간 비포화** |
| 12.5 m | 49 × 65 | 3,185 | 32 | 전 구간 비포화 |
| 10 m | 61 × 81 | 4,941 | 40 | 전 구간 비포화 |
| 6.25 m | 97 × 129 | 12,513 | 64 | 전 구간 비포화 |
| 5 m | 121 × 161 | 19,481 | 80 | 전 구간 비포화 |

**포화는 거친 격자에서만 문제가 된다.** canonical 25 m부터는 r = 1…10 전 구간이 비포화이므로
r 축 실험에 제약이 없다. 100 m와 50 m의 포화 구간은 "local SSE가 사실상 global 열거"임을 뜻하므로
측정값으로 기록하되 local-SSE 추세선에는 포함하지 않는다.

포화 판정은 `defender_grid_2d`가 단일 정의를 소유한다 — `centre_saturating_radius()`(중심 출발,
실질 기준)와 `saturating_radius()`(임의 출발 보장 기준), 그리고 특정 seed에 대한 정확한 검사
`saturates_from()`. 후자는 기존 `DefenderGridTopology.radius_covers_all_actions()`와 같은 규칙이다.

### 세 knob의 스윕 범위

| Knob | 범위 | 기존 인프라 |
|---|---|---|
| 상태 해상도 `Δ_state` | 100 m → 5 m (100/n, n=1…20) / heading 45° → 1° | `stage14_3_contract.py`, `stage14_4_contract.py` |
| 스위칭 후보 밀도 `N_C` | 현재 (6, 9, 12) — 확장 필요 | `stage14_5_contract.py` |
| 이웃 반경 `r` | **1 → 10 (격자 단위)** | `stage14_6_contract.py` 재정의 |

r을 **격자 단위**로 재는 것이 설계상 중요하다. 그래야 local SSE가 한 번에 평가하는 이웃 수
`(2r+1)²`가 해상도와 무관하게 고정되어, 해상도 축(→ `T_BR(N_S)`만 변함)과
defender 탐색 축(→ 평가 횟수만 9 → 441로 약 49배 변함)이 깨끗하게 분리된다.
결합이 남는 것은 global oracle뿐이며, 그것은 "언제 무너지는가"를 보여주는 기준선 역할이다.

---

## 03 · 가설 — 검증 대상이지 결론이 아니다

아래는 코드 구조에서 유도한 **반증 가능한 예측**이다. 이 연구가 하는 일은 이것을 주장하는 것이 아니라
측정으로 판정하는 것이며, Phase 4에서 각 가설의 예측 대 관측을 명시적으로 대조한다.

| | Knob | exact 비용 | RL 비용 | 예측 | 반증 조건 |
|---|---|---|---|---|---|
| **H1** | 해상도 ↑ | N_S, N_E ↑ → T_graph + T_hazard + T_Bellman 상승 | Q-table이 dense `N_S×A`로 증가, 상태 커버리지 요구로 sample complexity가 더 가파르게 상승. exact의 `N_S_active ≪ N_S_cart` sparse 축소 혜택이 없음 | **exact 유리** | RL 정확도가 해상도에 둔감하거나 Q-table sparse화로 완화되는 경우 |
| **H2** | N_C ↑ | Bellman 값함수를 한 번 계산해 모든 후보에 재사용 → feasibility/connection 비용만 | 학습된 Q도 동일하게 재사용 → 후보당 질의만 | **중립 — 결정 knob 아님** | 어느 한쪽만 가파르게 상승하는 경우 |
| **H3** | r ↑ | defender 위치마다 hazard field가 바뀌어 `precompute_edge_hazards` + Bellman 재해결 → (2r+1)²에 선형 | defender action을 context로 학습했다면 1회 학습 + 이웃당 질의 | **RL 유리** | defender context 포함으로 커진 학습 비용이 (2r+1)² 이득을 상쇄하는 경우 |

---

## 04 · 실행 계획

각 Phase는 **목표 · 작업 · 시각화 · 산출물** 네 칸으로 구성된다. 시각화는 장식이 아니라
다음 Phase로 넘어가기 전의 검증 게이트다. 모든 그림은 저장된 JSON만으로 솔버 재실행 없이
재생성 가능해야 한다 (`stage14_7_scaling_analysis.py`의
`regenerate_stage14_7_figures_from_saved_data()` 관례).

### Phase 0 — 설계 확정 *(문서)*

**목표**: 코드를 한 줄도 쓰기 전에 결정 규칙·정확도 지표·비용 회계를 확정한다.
여기서 모호하게 남긴 항목은 Phase 3에서 반드시 재작업을 부른다.

**작업**
1. 결정 규칙 `B*(c, τ)` 정식화, τ 후보값 범위 결정.
2. 정확도 지표 확정. `mission_response.py`와 `bellman_objectives.py`에서
   `defender_objective_pod`의 최대화/최소화 방향을 먼저 확인한 뒤 relative gap 공식의 부호를 정한다.
3. RL 비용 회계: `T_train`과 `T_query`를 분리 기록. 재사용 단위는 defender action이며,
   이를 context로 포함해 학습한다는 결정을 명시한다.
4. exact-infeasible 구간 서술 규칙.
5. 실패 상태 어휘: `stage14_benchmark_runner.py`의 `TERMINAL_STATUSES`에
   RL 고유 상태(`rl_non_convergent` 등)를 어떻게 정합시킬지.

**시각화**
- `0-1` 비교축 2×2 사분면. 가로축 탐색 폭(Local↔Global), 세로축 해 정확도(Exact↔RL).
  기존 Stage 14 작업물이 전부 위쪽 행에만 찍혀 있고 이번 연구가 아래 행으로 내려간다는 것을 확인.
- `0-2` **최종 목표 그림 목업.** 가짜 숫자로 Phase 4의 4종 셀 결정 지도를 미리 그린다.
  실제 데이터가 하나도 없는 단계에서 "내가 원하는 최종 그림이 이게 맞나"를 먼저 확정하면,
  이후 Phase들이 생산해야 할 데이터가 역으로 정해진다.

**산출물**: `docs/exact_vs_rl_decision_framework.md`

---

### Phase 1 — Defender 격자 2D화 · 계약 · 스키마 *(기반 재정의)*

**목표**: defender action set을 x–y 평면 격자로 재정의하고, RL이 들어갈 계약과 워커 자리를 만든다.
Stage 14의 재현성 보증은 건드리지 않는다.

**작업**
1. `defender_grid_config.py`의 2D 재정의. 현재 19줄짜리 1D 정의를
   x ∈ [terrain_max_x, goal_x] × y ∈ [y_min, y_max] 격자 생성 함수로 대체한다.
2. `stage15_0_contract.py` 신규 생성. RL 허용 정책과 별도 `RL_SOURCE_FILES` fingerprint를 선언한다.
3. 워커 스키마 확장: timing에 `T_train_s` / `T_query_s` 분리, `exactness.objective_gap` 추가,
   RL 실패 상태 반영. `stage14_2_worker.py`의 실제 payload를 먼저 읽고
   `_row_from_attempt`가 요구하는 키와 대조할 것.
4. r 이웃 생성기를 2D Chebyshev로 확장 (`local_sse_contract.py`의 `DefenderGridTopology`).

> **파손 위험 2곳**
> 1. `stage14_3/4/5/6_contract.py`가 모두 `DEFENDER_ACTION_COUNT`를 import한다 —
>    기존 심볼을 바꾸면 Stage 14 게이트가 깨진다. 기존 심볼은 그대로 두고 새 모듈·새 함수를 추가한다.
> 2. `stage14_benchmark_contract.py`의 `SOLVER_SOURCE_FILES`에 RL 파일을 추가하면
>    `stage14_8_final_gate.py`의 fingerprint 회귀 검사가 통과 불가능해진다.

**시각화**
- `1-1` **Defender 격자 지도.** x–y 평면에 terrain, start, goal, defender 후보 격자점을 함께 표시하고
  해상도별 N_D를 주석으로. 영역 정의가 의도대로인지 눈으로 즉시 검증된다.
- `1-2` r 이웃 중첩도. 같은 지도 위에 r = 1, 3, 5, 10 이웃을 겹쳐 그려 어느 해상도에서 포화하는지 확인.
- `1-3` Fingerprint 격리 다이어그램. 변경 전후의 Stage 14 aggregate 해시를 나란히 출력해
  **값이 바뀌지 않았음**을 확인. 이 Phase의 핵심 검증.

**산출물**: `defender_grid_2d.py`, `stage15_0_contract.py`, `docs/solver_worker_schema.md`

---

### Phase 2 — exact 비용 곡선과 실행 가능 경계 *(exact 기준선)*

**목표**: 세 knob에 대한 exact 쪽 비용과 "어디서 무너지는가"를 확보한다.
RL 코드 없이 진행하며, 여기서 나온 `T_exact(c)`가 Phase 4 결정 규칙의 좌변이 된다.

**작업**
1. 12.5 m 근방에서 이미 보고된 model infeasibility / worker failure를 단독 재현해 원인을
   `map_geometry.py` · `sparse_reachability.py` · `switching_candidates.py` 중 하나로 분류한다.
   전체 스윕 재실행은 하지 않는다.
2. r = 1…10 스윕 측정 (해상도 고정) — H3의 exact 쪽 예측 검증.
3. 해상도 스윕 측정 (r 고정) — H1의 exact 쪽 예측 검증.
4. N_C 스윕 측정 — H2 검증. 현재 (6, 9, 12)는 점이 3개뿐이라 확장이 필요하다.
5. 이론 복잡도 재도출. `bellman_solver.py`가 `heapq`를 쓰므로 기존에 적힌 `O(N_S + N_E)`가
   실제로는 `O((N_S + N_E) log N_S)`여야 하는지 확인하고, 실측 지수와 대조한다.
6. `T_exact(N_S, |E(r)|)` 예측식 도출. `N_E ≈ B_effective × N_S_active`이므로 다중공선성을 점검할 것.

**시각화**
- `2-1` **실행 가능 경계 지도.** x축 해상도, y축 r, 셀 색 = status
  (completed / model_infeasible / timeout / memory_limit). exact가 어디서 무너지는지가 한 장에 담기며,
  이것이 곧 Phase 4 결정 지도의 "exact 불가" 영역이 된다.
- `2-2` r 대 총 런타임, 해상도별 곡선 여러 개. 기울기가 (2r+1)²에 비례하는지가 H3의 직접 검증.
- `2-3` 이론 대 실측 지수 비교. log–log 산점도에 실측 fit 직선과 이론 기울기 직선을 겹쳐 그림.
- `2-4` 예측력 parity plot. x축 예측 시간, y축 실제 시간, y = x 대각선.
  점이 대각선에 붙어야 이 예측식을 Phase 4에서 외삽에 쓸 수 있다.

**산출물**: `figure/stage_15_2_exact_baseline/`, `docs/exact_cost_model.md`

---

### Phase 3 — Tabular Q-learning 구현과 예산–정확도 곡선 *(RL 측정)*

**목표**: defender action을 context로 받는 tabular Q-learning을 **같은 MDP** 위에 구현하고,
조건별 `B*(c, τ)`를 산출한다.

**작업**
1. 환경은 새로 짜지 않는다. `bellman_graph.py`의 전이 생성과
   `edge_hazard.py` · `bellman_objectives.py`의 비용 함수를 얇게 감싼다.
   별도 시뮬레이터를 만들면 성능 차이가 알고리즘 때문인지 환경 구현 차이 때문인지 구분할 수 없어
   비교 자체가 무효가 된다.
2. **defender action을 state 또는 context에 포함한다.** 이것이 r 축 이점의 전제이며,
   포함하지 않으면 defender마다 재학습이 필요해 RL이 이길 수 있는 유일한 축이 사라진다.
   상태공간이 커지는 trade-off 자체가 측정 대상이다.
3. 보상은 exact가 최소화하는 edge 비용의 음수로 둔다. 그래야 "학습 → ∞일 때 gap → 0"이 성립해
   정확도 축이 해석 가능해진다.
4. `RLTrainingConfig`를 `DiscretizationConfig`와 같은 frozen dataclass 패턴으로 버전 관리.
5. seed 10개 이상 실행, median과 95% 구간 기록 (`numeric_statistics` 관례 준수).
6. 예산 B를 스윕해 accuracy(B) 곡선을 얻고 τ 교차점에서 `B*`를 산출.
7. 학습된 정책을 `trajectory_validation.py`의 `TrajectoryReplayAudit`에 통과시켜
   **exact 없이도 측정 가능한 절대 달성값**을 확보.

**시각화**
- `3-1` **학습 곡선 + exact 기준선.** x축 episode, y축 J_D. seed별 곡선은 연하게, median은 진하게,
  exact 최적값을 수평 점선으로 overlay. RL이 제대로 학습되고 있는지가 이 한 장으로 판정된다.
- `3-2` **예산–정확도 곡선.** x축 예산, y축 정확도, τ를 수평선으로.
  교차점이 곧 `B*`이며 Phase 4의 직접 입력이다.
- `3-3` 궤적 중첩. 동일 지형 위에 exact 최적 궤적과 RL 정책 궤적을 겹쳐 그림(`visualization.py` 재사용).
- `3-4` Q-table 메모리 대 exact peak RSS. RL이 exact보다 **먼저** 메모리 벽에 부딪히는지 확인.

**산출물**: `stage15_1_rl_worker.py`, `RLTrainingConfig`, `figure/stage_15_3_rl/`

---

### Phase 4 — 결정 지도와 가설 판정 *(종합)*

**목표**: 세 knob 공간에서 4종 셀 지도를 만들고 H1 · H2 · H3를 판정한다. 이것이 논문의 주 결과 그림이다.

**작업**
1. Phase 2의 `T_exact(c)` 예측식과 Phase 3의 `B*(c, τ)`를 결합해 결정 규칙을 평가한다.
2. τ를 스윕해 RL 유리 영역이 어떻게 변하는지 측정한다.
3. exact 불가 구간을 별도 셀로 표시하고 "RL 가용" 주장을 명시한다.
   오차는 외삽이며 불확실성을 함께 표기한다.
4. H1 · H2 · H3의 예측 대 관측을 대조해 서술한다. 가설이 기각되어도 그것이 결과다.

**시각화**
- `4-1` **메인 결정 지도.** x축 해상도, y축 r, 4색 셀. τ마다 한 장씩.
  Phase 0의 목업(`0-2`)과 나란히 놓고 비교하면 목표 달성 여부가 바로 보인다.
- `4-2` Pareto 비교. x축 시간, y축 정확도. exact는 (긴 시간, 100%) 단일 점, RL은 곡선.
  시간 예산을 수직선으로 그으면 그 선 위에서 누가 더 높은지가 곧 답이다.
- `4-3` τ 민감도. τ 변화에 따른 RL 유리 영역의 면적 변화.
- `4-4` 가설 판정 요약. H1 · H2 · H3별 예측과 관측을 나란히.

**산출물**: `stage15_2_decision_map.py`, `figure/stage_15_4_decision_map/`

---

## 05 · 검증 게이트 순서

1. `0-2` **목업**으로 최종 그림 형태를 먼저 확정 — 이후 모든 데이터 요구사항이 여기서 역산된다.
2. `1-1` **격자 지도**와 `1-3` **해시 불변**으로 기반 재정의가 의도대로인지 확인.
3. `2-1` **실행 가능 경계**로 exact가 무너지는 지점을 확정,
   `2-4` **parity plot**으로 예측식을 신뢰해도 되는지 판정.
4. `3-1` **학습 곡선**으로 RL이 정상 학습되는지, `3-2` **예산–정확도**로 `B*`를 확보.
5. `4-1` **결정 지도**로 최종 답. `0-2` 목업과 대조해 목표 달성 확인.

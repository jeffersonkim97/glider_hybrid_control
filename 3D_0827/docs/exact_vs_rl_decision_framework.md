# Exact SSE vs RL Approximation — 결정 프레임워크

**Phase 0 산출물** · Stage 15 · 지형 `centered_cube` 고정
상위 계획: [`stage15_plan.md`](stage15_plan.md)

이 문서는 코드 작성 전에 확정해야 하는 정의를 고정한다. 여기서 모호하게 남긴 항목은
Phase 3에서 반드시 재작업을 부른다. 모든 규약은 실제 소스에서 확인한 값이며,
§2의 공식은 frozen reference 값으로 수치 검증했다.

---

## 1. 비교축 분리

두 개의 직교하는 축이 있고, 이 연구는 **축 B만** 다룬다.

| | 축 A — 탐색 폭 | 축 B — 해 정확도 |
|---|---|---|
| **양 끝** | Local SSE (r 제한) ↔ Global oracle (전체 열거) | Exact Bellman ↔ RL approximation |
| **공통점** | 둘 다 exact — 정확도 손실 없음 | 둘 다 같은 defender 탐색 전략 사용 |
| **기존 측정** | `stage_14_6_neighbor_radius` | 없음 (Stage 14는 exact 전용) |

`local_sse_search.py`의 Local SSE는 defender action 탐색 폭을 `r_neighbor`로 줄일 뿐,
각 이웃 평가는 여전히 `bellman_solver.py`의 exact backward induction을 호출한다.
따라서 **Stage 14.6의 local vs global 측정치를 RL 비교의 근거로 재사용하지 않는다.**
두 축을 섞으면 "무엇 때문에 빨라졌는지"가 구분되지 않는다.

r은 축 B의 실험에서 **조건 변수**로 등장한다. 즉 "축 A의 어느 지점에서 축 B의 승패가 갈리는가"를 묻는다.

---

## 2. 부호 규약 — 코드에서 확인 완료

`game_types.py:23-31`에 명시적으로 선언되어 있다. 추정하지 않았다.

```python
ATTACKER_OBJECTIVE_SENSE      = "minimize"
DEFENDER_PAYOFF_SENSE         = "maximize"
ATTACKER_OBJECTIVE_COMPONENTS = ("mission_time_s", "detection_probability")
DEFENDER_OBJECTIVE_COMPONENTS = ("detection_probability",)
ZERO_SUM_ASSUMED              = False
```

### Attacker 목적함수 — `attacker_hazard_time_v2`

`detection_hazard.py:292-362`, 파라미터는 `AttackerHazardTimeParameters`:

```
J_A = w_h · (H / H_ref) + w_t · (T / T_ref)          ← 최소화

w_h = w_t = 0.5,  H_ref = 1.0,  T_ref = 5000 / 22.6 ≈ 221.2389 s
```

### Defender 목적함수 — 탐지 확률

`detection_hazard.py:280-289`, `hazard_to_detection_probability`:

```
J_D = P_D = 1 − exp(−H)                              ← 최대화, [0, 1] 유계
```

### 수치 검증

`stage14_benchmark_contract.py:36-46`의 `FROZEN_CANONICAL_SOLUTION` 값으로 두 공식을 확인했다.
`H = 1.520905739469603`, `T = 81.25366155073556` 을 대입하면:

| 양 | 공식으로 계산 | frozen 기록값 | 일치 |
|---|---|---|---|
| `J_A` | 0.5·1.5209057 + 0.5·(81.2537 / 221.2389) = **0.944086145** | 0.944086144839464 | ✓ |
| `J_D` | 1 − exp(−1.5209057) = **0.781486119** | 0.7814861193516704 | ✓ |

### 비영합(non zero-sum)의 결과

`ZERO_SUM_ASSUMED = False`이므로 **attacker 쪽 오차가 defender 쪽 오차로 단조 변환되지 않는다.**
J_A는 hazard와 time의 가중합이지만 J_D는 hazard만 본다. 따라서 attacker 오차를
"정확도"의 대리 지표로 쓸 수 없고, §3에서 세 가지를 분리해 정의한다.

---

## 3. 정확도 지표 — 세 가지를 구분한다

RL은 **attacker의 best response**를 대체한다. 바깥의 Stackelberg 탐색 구조는 그대로 둔다.
그러면 서로 다른 세 가지 오차가 생기고, 의사결정에 쓰이는 것은 세 번째다.

### (a) Follower 응답 품질 — 내부 구동 지표

고정된 defender action `d`에 대해:

```
gap_A(d) = ( J_A^RL(d) − J_A^exact(d) ) / J_A^exact(d)     ≥ 0
```

exact가 진짜 최소값을 주므로 항상 비음수. RL 학습이 진행되는지 보는 데 쓰고,
**최종 판정 기준으로는 쓰지 않는다.**

### (b) RL이 보고하는 J_D — 낙관 편향이 있으므로 직접 쓰지 않는다

RL의 attacker 응답이 최적보다 나쁘면, 그 attacker는 더 탐지되기 쉬운 경로를 택한다.
그러면 defender의 측정 PoD가 **실제보다 높게** 나온다. 즉 RL 실행이 스스로 보고하는 J_D는
체계적으로 부풀려져 있다. 이 값을 그대로 정확도로 쓰면 RL이 실제보다 좋아 보인다.

> **규칙**: RL 실행이 내부적으로 보고한 J_D는 진단용으로만 기록하고,
> 판정에는 아래 (c)의 재평가 값을 쓴다.

### (c) 배치 후회(regret) — 판정 기준 ★

의사결정자가 실제로 신경 쓰는 것은 "RL이 알려준 위치에 센서를 놓으면 진짜 탐지확률이 얼마냐"다.

```
J_D_true(a) ≡ exact attacker BR로 평가한 defender action a 의 진짜 PoD

regret(r) = J_D_true( a*_local(r) ) − J_D_true( a_RL(r) )     [PoD 포인트, 0 이상]
```

**두 값 모두 exact BR로 재평가한다.** `a_RL(r)`은 RL을 inner BR로 쓴 local SSE가 선택한 배치,
`a*_local(r)`은 **동일한 조건**(같은 반경 r, 같은 초기 action, 같은 이웃 규칙)에서
exact BR을 쓴 local SSE가 선택한 배치다.
PoD가 이미 [0,1]의 확률이므로 차이를 그대로 "탐지확률 몇 퍼센트포인트 손해"로 읽을 수 있다.

> **기준선은 전역 최적이 아니라 동일 조건의 exact local SSE다.** 두 가지 이유가 있다.
>
> 1. **비교의 청결성.** 바깥 탐색(반경 r의 local SSE)이 양쪽 동일하고 **안쪽 best response만**
>    exact ↔ RL로 달라진다. 축 A(탐색 폭)를 고정한 채 축 B(해 정확도)만 움직이는 설계와 정확히 일치한다.
> 2. **측정 가능성.** 전역 최적 `a*_global`을 구하려면 `N_D`회 full BR이 필요한데, 2D defender 격자에서는
>    25 m에 825회, 5 m에 19,481회다. 반면 기준선을 local SSE로 두면 `(2r+1)² × 반복횟수`회로 끝나며
>    이 값은 **해상도와 무관**하다(r이 격자 단위이므로). 세밀한 격자에서도 기준값을 만들 수 있다.
>
> 대가로 이 연구는 전역 최적성을 주장하지 않는다. 주장 범위는 "동일한 local 탐색 프로토콜 아래에서
> inner BR을 RL로 바꿨을 때의 손해"이며, 이는 원래 목표와 일치한다.

---

## 4. 세 가지 측정 영역 — 기준값 부재 문제는 생각보다 약하다

Phase 0 검토 중 발견한 사항이다. 계획서에 적었던 "exact가 못 도는 구간에는 ground truth가 없다"는
서술은 **너무 비관적**이었다.

exact가 비싸지는 주된 이유는 **defender action마다 full BR 재계산**이 필요해
`(2r+1)² × 반복횟수`회를 풀어야 하기 때문이다.
반면 **단 한 번의 BR 풀이는 훨씬 오래 감당 가능하다.** 따라서 영역이 셋으로 갈린다.

| 영역 | 조건 | 얻을 수 있는 것 | 못 얻는 것 |
|---|---|---|---|
| **I. 완전 측정** | exact local SSE가 반경 r에서 완주 | regret 전체 (§3c) | 전역 최적성 (애초에 주장 않음) |
| **II. 부분 측정** | exact local SSE는 불가, 단일 BR은 가능 | `J_D_true(a_RL)` — RL 배치의 **진짜 PoD를 정확히** 측정 | `a*_local(r)`을 모르므로 regret은 불가 |
| **III. 미측정** | 단일 BR조차 불가 | RL 내부 추정 + `TrajectoryReplayAudit`의 재생 검증값 | 절대 품질 보증 |

기준선을 local SSE로 바꾼 결과, 영역 I의 경계는 `N_D`가 아니라 `(2r+1)²`가 결정한다.
`(2r+1)²`는 해상도와 무관하므로 **영역 I이 세밀한 격자까지 훨씬 넓게 유지된다.**
영역 I을 벗어나는 경로는 이제 두 가지다: r이 커져 이웃 수가 폭증하거나,
격자가 세밀해져 BR 1회 자체가 비싸지거나.

영역 II가 핵심이다. 여기서는 "RL이 고른 배치의 탐지확률은 정확히 0.74다.
다만 그보다 나은 배치가 있는지는 알 수 없다"라고 **정량적으로** 말할 수 있다.
"아무것도 모른다"와는 전혀 다른 위치다.

### 주장 서술 규칙

- 영역 I: regret 기반 판정. 그림에서 **실선**.
- 영역 II: 절대 PoD 보고 + regret은 영역 I 추세의 외삽으로 상한만 제시. 그림에서 **점선**.
- 영역 III: 가용성만 주장. 그림에서 **빈 마커 + 음영**.
- 어느 영역에서도 "오차가 τ 이내임이 보장된다"고 쓰지 않는다. 영역 I에서만 "측정된 regret ≤ τ"라고 쓴다.

---

## 5. RL 비용 회계

### 분리 기록

| 항목 | 정의 | 재사용 단위 |
|---|---|---|
| `T_train(B)` | 예산 B까지의 학습 시간 | (지형, `Δ_state`, `N_C`) 당 1회 |
| `T_query(c)` | 학습된 정책으로 defender action 1개를 평가하는 시간 | defender action 당 |

### 손익분기

반경 r의 local SSE 1회 실행을 기준으로 (이웃 수 `(2r+1)²`):

```
RL 총비용    = T_train(B*) + (2r+1)² · T_query
exact 총비용 = (2r+1)² · T_BR

RL 유리 ⟺ T_train(B*) < (2r+1)² · ( T_BR − T_query )
```

`(2r+1)²`가 r=1에서 9, r=10에서 441로 약 49배 변하므로,
**이 부등식이 r에 따라 뒤집히는 지점이 있는지가 H3의 정량적 형태다.**
Phase 2에서 `T_BR`을, Phase 3에서 `T_train`과 `T_query`를 측정하면 바로 판정된다.

### Defender action을 어떻게 반영할 것인가 — Phase 3의 핵심 결정

hazard field가 defender 위치에 의존하므로(`edge_hazard.py`의 `precompute_edge_hazards`),
정책은 defender action에 따라 달라져야 한다. 두 가지 방법이 있다.

**방법 1 — Warm-start 전이 (권장).**
Q-table 크기는 `N_S × A`로 유지하고, 이웃 defender action `d'`의 학습을
이미 수렴한 `d`의 Q에서 시작한다. 첫 해결은 비싸지만 이후 이웃들은 훨씬 싸진다.
r 축의 상각 효과를 그대로 만들면서 테이블 크기가 폭발하지 않는다.

**방법 2 — Context 증강.**
defender action을 상태에 포함해 `N_S × N_D × A` 테이블을 만든다. 개념적으로 깨끗하지만
canonical 25 m에서 `N_D = 825`이므로 테이블이 825배가 된다. **대부분의 조건에서 메모리로 불가능할 것으로 예상**되며,
이것이 현실화되면 그 자체가 H3의 반증(“context 비용이 이득을 상쇄”)으로 기록된다.

> **결정**: 방법 1을 기본으로 구현한다. 방법 2는 메모리 한계를 측정해 H3 반증 조건의
> 정량적 근거로 남기되, 전 조건에 걸쳐 돌리지 않는다.

---

## 6. 정확도 임계값 τ

regret이 PoD 포인트 단위이므로 τ도 같은 단위로 둔다. 운영자가 바로 해석할 수 있다.

| τ | 해석 |
|---|---|
| 0.01 | RL 배치가 최적 대비 탐지확률 1%p 이내 손해 — 엄격 |
| 0.02 | 2%p 이내 — 기본값 |
| 0.05 | 5%p 이내 — 관대 |

세 값 모두에 대해 결정 지도를 생성하고, τ 민감도(그림 `4-3`)로 결론의 견고성을 보인다.

비율형(`J_D_RL / J_D_exact ≥ 0.8` 같은)도 계산해 부록에 남기되 **주 지표로 쓰지 않는다.**
확률의 비율은 해석이 모호하고(0.4→0.32와 0.9→0.72가 같은 80%로 취급됨),
포인트 차이가 배치 의사결정에 직접 대응한다.

---

## 7. 실패 상태 어휘

`stage14_benchmark_runner.py:24-28`의 `TERMINAL_STATUSES`를 그대로 계승하고 두 개만 추가한다.

| 상태 | 의미 | 결정 지도에서의 셀 |
|---|---|---|
| `completed` | 정상 종료 | 값에 따라 판정 |
| `timeout` / `memory_limit` | 부모 프로세스가 강제 종료 | exact 쪽이면 "exact 불가" |
| `model_infeasible` | 모델 자체가 해를 갖지 않음 | 조건에서 제외 (계산 한계 아님) |
| **`rl_budget_exhausted`** *(신규)* | 예산 B를 다 썼으나 τ 미도달 | **RL 부적합** — 실패가 아니라 정당한 결과 |
| **`rl_non_convergent`** *(신규)* | Bellman residual이 감소하지 않음 (학습 발산) | RL 부적합, 단 진단 필요 |

`rl_budget_exhausted`를 crash와 구분하는 것이 중요하다. 이것은 "RL을 쓰지 말라"는 답이지
버그가 아니다.

---

## 8. 스윕 값 확정

| Knob | Phase 2/3에서 사용할 값 | 근거 |
|---|---|---|
| `Δ_state` (공간) | 100, 50, 25, 12.5, 10, 6.25, 5 m | `stage14_3_contract.py`의 `100/n` 격자 중 선택 |
| `Δ_state` (heading) | 5° 고정 (canonical) | 1차 스윕에서는 공간만 변화 |
| `N_C` | 6, 9, 12, 18, 24 | 기존 (6,9,12)에서 확장 — 점 3개로는 추세 판정 불가 |
| `r` | 1, 2, 3, 5, 7, 10 | 격자 단위 Chebyshev, 이웃 수 9 → 441 |
| `τ` | 0.01, 0.02, 0.05 | §6 |
| seed | ≥ 10 | RL 확률성, median + 95% 구간 |
| global oracle | 100 m, 50 m 에서만 | 아래 참조 |

**Global oracle의 용도 축소.** 2D defender 격자에서 전역 열거는 25 m에 825회, 5 m에 19,481회
full BR을 요구해 대부분의 조건에서 불가능하다. 따라서 global oracle은 **주 비교 대상이 아니라**
100 m · 50 m(`N_D` = 63 · 221)에서 "local SSE가 전역해를 놓치지 않았는지" 확인하는
sanity check로만 쓴다. 이 연구의 exact 기준선은 §3c에 따라 동일 조건의 exact local SSE다.

---

## 9. Phase 1로 넘어가기 전 확인 사항

- [x] 비교축 분리 확정 (§1)
- [x] 부호 규약 코드 확인 및 수치 검증 (§2)
- [x] 정확도 지표 = 배치 후회, exact BR 재평가 기반 (§3c)
- [x] 세 측정 영역 정의 및 서술 규칙 (§4)
- [x] `T_train` / `T_query` 분리, 손익분기 부등식 (§5)
- [x] Warm-start 방식 채택, context 증강은 반증 근거용 (§5)
- [x] τ 후보 및 단위 (§6)
- [x] 실패 상태 어휘 (§7)
- [x] 스윕 값 (§8)

### 계획서 대비 변경점 2건 (모두 승인 완료)

**변경 1 — 기준값 부재 문제의 완화.** `stage15_plan.md` §1의 "exact 불가 구간에는 ground truth가 없다"는
서술을 §4의 3영역 구분으로 대체한다. 단일 BR 평가가 가능한 영역 II에서는
RL 배치의 진짜 PoD를 정확히 측정할 수 있으므로, 핵심 주장 영역이 예상보다 넓다.

**변경 2 — exact 기준선을 전역 최적에서 동일 조건 local SSE로 교체.**
Phase 0 그림 작업 중 2D defender 격자의 `N_D`를 실제로 계산한 결과
(25 m에서 825, 5 m에서 19,481) 전역 열거가 대부분의 조건에서 불가능함이 드러났다.
기준선을 동일 조건 exact local SSE로 바꾸면 비교가 더 청결해지고(축 A 고정, 축 B만 변화)
기준값 생성 비용이 해상도와 무관해진다. §3c·§4·§8에 반영 완료.
대가로 전역 최적성은 주장하지 않으며, 이는 원래 연구 목표와 일치한다.

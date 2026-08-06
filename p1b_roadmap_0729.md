# P1B ACC Positioning and Solution-Novelty Roadmap — 2026-07-29

이번 연구는 이전의 top-down surveillance-evasion 연구를 단순히
side-view로 옮긴 것이 아니라, terrain-coupled hybrid regime으로 확장한
후속 연구로 잡는 것이 가장 자연스럽다.

## Current execution status — 2026-07-29

- **Direction B technical core: COMPLETED.** Finite physical-successor
  Bellman optimality, exact minimum-response selection, B2/B3의 54-case
  nested-consistency study, common high-fidelity replay 및 B4 production
  lattice freeze를 완료했다.
- Continuous optimal lower bound는 검토 후 제외했으며, B의 scope는
  discretized follower space에 한정한다.
- 다음 technical stage는 C-lite finite defender-set exhaustive solve이다.
- ACC manuscript로 theorem, table 및 figure를 옮기는 작업은 이후 전체
  paper-writing 단계에서 수행한다.

예비 판단은 다음과 같다.

- Problem novelty: 충분히 설득력 있을 가능성이 높음
- 현재 solution novelty: 개별 알고리즘 요소만 보면 강하지 않음
- 가장 유망한 방향: 문제 특유의 monotone hybrid structure를 이용한
  finite-grid best-response oracle
- 더 강하게 만들려면: convergence/error control 또는 local NLP 대비 보장 중
  하나를 추가해야 함

## 1. 이전 논문과 현재 연구의 관계

이전 논문은 경쟁 선행연구가 아니라 이번 논문의 출발점이다.

### 이전 논문

> Continuous sensor redeployment and surveillance-evasion trajectories in a
> top-down planar environment.

주요 특징:

- top-down 2D 공간
- building boundary를 따라 움직이는 continuous sensor placement
- directional 및 omnidirectional sensor
- 복수 sensor
- STP-RRT* attacker initialization
- attacker/defender nonlinear optimization
- alternating best response
- local Nash equilibrium 또는 local stationarity

### 현재 연구

> Terrain-aware sensor placement against a hybrid powered-to-glide evader in a
> vertical terrain cross-section.

주요 특징:

- side-view 2D, 즉 \((z,h)\) 공간
- terrain-mounted omnidirectional sensor
- terrain masking과 LOS
- powered-to-glide hybrid attacker
- endogenous switching point
- glide reachable set
- path-integrated detection probability
- physical-successor Bellman follower
- defender commitment 후 attacker best response

| 항목 | 이전 논문 | 현재 연구 |
|---|---|---|
| 공간 | 수평면 top-down | 수직 단면 side-view |
| 주요 geometry | 건물 경계와 sensor FOV | 지형, 고도, terrain masking |
| Sensor | directional + omnidirectional, 복수 | omnidirectional, 단일 terrain-mounted sensor |
| Defender decision | 경계를 따라 continuous redeployment | terrain 위 downrange 위치 |
| Attacker | waypoint UAS | powered-to-glide hybrid vehicle |
| Attacker decision | planar path | switching state + glide path |
| Visibility | smooth detectability 중심 | terrain-dependent LOS 전환 |
| Solution | STP-RRT* + NLP | physical-successor DAG + Bellman |
| Game computation | alternating local best responses | leader commitment + follower oracle |
| 보장 | local stationarity/LNE | finite-grid follower minimum |

## 2. Side-view가 단순한 좌표 변경이 아닌 이유

단순히 top-down의 \((x,y)\)를 side-view의 \((z,h)\)로 바꾼 것으로
설명하면 약하다.

현재 문제에서는

\[
\dot z=v\cos\gamma,\qquad
\dot h=v\sin\gamma
\]

이고, 다음 요소들이 결합된다.

- 고도 감소와 중력
- admissible glide angle
- speed selection
- terrain clearance
- powered-to-glide switching
- switching 이후의 reachable set
- sensor와 vehicle 사이의 LOS
- path-dependent detection probability

가시성도 단순한 거리 함수가 아니라 대략

\[
\operatorname{visible}(z,h;s)
=
\mathbf 1\!\left[
h\ge h_{\mathrm{LOS}}(z;s)
\right]
\]

와 같은 terrain-dependent 조건을 갖는다.

Sensor 위치 \(s\)가 바뀌면 동시에:

- LOS tangent가 바뀌고
- visible region이 바뀌고
- optimal switching state가 바뀌고
- reachable glide set이 바뀌며
- attacker의 masking corridor가 바뀐다.

따라서 side-view는 visualization 차이가 아니라 problem structure의
변화이다.

## 3. Sensor model이 단순해진 것은 어떻게 설명할까

이전 논문보다 sensor 수가 줄었고, directional sensor가 없어졌으며,
omnidirectional sensor 하나만 사용한다. 그러므로 sensor-placement
generality를 novelty로 내세우면 안 된다.

대신 의도적인 model isolation으로 설명하는 것이 좋다.

> We deliberately consider a single omnidirectional terrain-mounted sensor to
> isolate the strategic coupling among terrain occlusion, hybrid vehicle
> dynamics, and adaptive evasion.

즉, 이번 연구의 목적은 heterogeneous sensor allocation, directional
scheduling, multi-sensor coordination이 아니라 다음을 분석하는 것이다.

- terrain이 attacker response를 어떻게 바꾸는지
- hybrid switching decision이 placement에 어떻게 결합되는지
- coverage-only placement와 strategic placement가 언제 달라지는지

Sensor model은 단순해졌지만 attacker/environment coupling은 훨씬
복잡해졌다는 구도이다.

## 4. 현재 solution의 개별 요소는 얼마나 새로운가

냉정하게 보면 다음 요소들은 각각 선행연구가 있다.

| 구성요소 | Novelty 판단 |
|---|---|
| Stackelberg sensor placement | 기존 연구 존재 |
| Intruder minimum-exposure path | 기존 연구 존재 |
| Terrain-masked UAV path planning | 기존 연구 존재 |
| Grid node를 정확히 잇는 physical edge | state-lattice와 유사 |
| DAG의 Bellman backward sweep | 표준 |
| Continuous outer placement | 기존 연구 존재 |
| Hybrid switch + terrain LOS + finite-grid follower oracle | 문제 특화 기여 가능 |
| Execution-consistent best response | 조합과 formal guarantee에 따라 기여 가능 |

Defender가 sensor 위치를 정하고 intruder가 minimum-exposure path로
대응하는 bilevel 문제는 이미 연구됐다.
[Karabulut et al., 2017](https://www.sciencedirect.com/science/article/abs/pii/S0377221716307524)

Terrain masking을 이용한 UAV detection-risk path planning도 기존 연구가
있다.
[Pelosi et al., 2012](https://www.tandfonline.com/doi/abs/10.1080/08839514.2012.713308)

Dynamics를 만족하면서 discrete node에 정확히 끝나는 motion primitive도
state-lattice planning의 핵심 아이디어다.
[Pivtoraiko et al., 2009](https://onlinelibrary.wiley.com/doi/pdf/10.1002/rob.20285)

따라서 다음과 같은 주장은 피해야 한다.

> We propose a novel Bellman algorithm.

> We introduce motion primitives that terminate exactly at grid nodes.

Bellman과 physical lattice edge 자체가 novelty는 아니다.

## 5. 현재 가장 유망한 solution contribution

가장 좋은 positioning은 다음이다.

> A structure-exploiting best-response oracle for a terrain-coupled hybrid
> surveillance-evasion Stackelberg game.

핵심은 Bellman을 사용했다는 것이 아니라, hybrid follower problem을 다음과
같이 환원했다는 점이다.

\[
\operatorname{BR}_\Delta(s)
=
\min_{\sigma\in\Sigma_\Delta}
\left[
C_{\mathrm{powered}}(\sigma;s)
+
d_{\mathcal G_\Delta(s)}
\left(\sigma,\mathcal X_{\mathrm{goal}}\right)
\right].
\]

여기서:

- \(s\): defender sensor 위치
- \(\sigma\): powered-to-glide switching state
- \(\Sigma_\Delta\): admissible switching-state set
- \(\mathcal G_\Delta(s)\): physical-successor graph
- \(d_{\mathcal G_\Delta}\): goal까지의 minimum accumulated glide cost

잠재적인 technical contribution은 네 단계로 정리할 수 있다.

### 1. Hybrid decomposition

- powered phase와 glide phase 분리
- continuous switching point를 유지
- switching point를 억지로 grid에 snapping하지 않음

### 2. Virtual switching state

- off-grid switching state를 virtual source로 사용
- virtual source에서 admissible successor grid node로 physical edge 연결
- powered phase와 glide lattice를 일관되게 결합

### 3. Physical-successor DAG

- successor node를 먼저 선택
- 해당 node에 정확히 도달하는 \(\gamma\)와 duration 계산
- terrain, LOS, flight-envelope 조건을 physical edge에서 평가
- monotone downrange motion으로 graph가 acyclic

### 4. Finite-grid follower best response

- DAG에서 한 번의 backward Bellman sweep
- 모든 admissible switching state 비교
- finite discretization에서 minimum-cost follower response 선택

## 6. 이전 논문과 가장 중요한 method 차이

이전 top-down 문제에서는 attacker가 여러 방향으로 움직일 수 있기 때문에
state graph에 cycle이 생길 수 있고, 일반적인 local NLP/RRT* 접근이
자연스럽다.

현재 side-view glide에서는 일반적으로

\[
z_{k+1}>z_k
\]

가 성립한다. 이 monotonicity가 graph를 DAG로 바꾸고:

\[
V(z_i,h_j)
=
\min_{(z_{i'},h_{j'})\in\mathcal S(z_i,h_j)}
\left[
c_{ij,i'j'}+V(z_{i'},h_{j'})
\right]
\]

를 한 번의 backward sweep으로 계산할 수 있게 한다.

따라서 가장 중요한 문장은 다음이다.

> The side-view hybrid dynamics are not merely a geometric restriction; their
> monotone downrange structure converts the follower problem into an acyclic
> physical-successor graph.

이전 논문의 local attacker NLP를 현재 문제의 structure-exploiting
finite-grid oracle로 교체했다는 흐름이다.

## 7. 현재 상태에서 부족한 점

### 7.1 Proposition과 실제 successor solver의 정합화 — RESOLVED (2026-07-29)

`discrete_optimality_proposition.md`를 현재
`successor_grid_solver.py`와 B4 production contract 기준으로 다시 작성했다.
Legacy `solve_coarse_bellman`, fixed-duration action 및 endpoint snapping
가정은 제거했다.

현재 formal result에 포함된 명제:

1. successor graph acyclicity
2. virtual switching edge admissibility
3. graph path와 piecewise-physical trajectory의 대응
4. backward Bellman의 exact finite-graph optimal value
5. switching-state와 virtual target/speed enumeration을 포함한 finite-grid
   follower optimal value
6. 계산복잡도

Scope와 구현상의 tie rule도 명시했다.

- inner follower에 대한 finite-grid optimum
- continuous trajectory-space optimum은 아님
- outer DIRECT는 best-found
- absolute minimum-cost candidate를 항상 선택하고, exact-equal cost에서만
  smallest switching-z와 seed index로 tie-break

### 7.2 Continuous feasibility는 아직 analytic certificate가 아님

현재 continuous replay는 finite sampling으로 terrain/LOS를 확인한다.

현재 확실히 말할 수 있는 것은:

- planned endpoint와 replay endpoint 일치
- accumulated snapping drift 제거
- 지정된 validation resolution에서 feasible
- 고해상도 continuous replay 통과

하지만 다음을 자동으로 증명한 것은 아니다.

> 모든 연속시간에서 terrain 및 LOS constraint가 만족된다.

더 강한 보장이 필요하다면 adaptive edge subdivision, analytic
maximum-violation check, interval bound, sampling-error bound 중 하나가
필요하다.

### 7.3 P4는 순수 action convergence가 아님

현재 `3x8 -> 6x16` stencil 변경은 successor 방향 후보, maximum cell
offset, maximum physical edge reach를 동시에 변경한다. 따라서 현재 P4를
순수 angular/action refinement evidence로 쓰면 안 된다.

새 refinement family에서는 가능한 한 다음을 분리해야 한다.

- spatial spacing
- speed resolution
- directional resolution
- maximum physical edge length
- quadrature resolution

## 8. Solution novelty를 강화하는 세 가지 방향

### 방향 A — 최소 추가 작업

> Novel problem + tailored finite-grid oracle

필요한 작업:

- successor solver 기준 theorem 재작성
- complexity proposition
- snapped Bellman과 physical Bellman 비교
- 이전 local NLP solver와 비교
- runtime scaling
- replay feasibility와 endpoint drift 비교

장점:

- 현재 구현을 대부분 활용
- ACC 분량에 적합
- 작업량이 비교적 작음

단점:

- 알고리즘 자체의 novelty는 제한적
- problem novelty와 tailored construction에 많이 의존

### 방향 B — 추천

> Error-controlled or consistent hybrid best-response oracle

예를 들어:

\[
J_\Delta^*(s)\rightarrow J^*(s)
\]

또는:

\[
J_{\mathrm{LB},\Delta}(s)
\le J^*(s)
\le J_{\mathrm{UB},\Delta}(s)
\]

와 같은 결과를 목표로 한다.

가능한 방법:

- relaxed DP로 lower bound
- continuous-feasible lattice path로 upper bound
- 두 값의 차이를 optimality gap으로 사용
- LOS와 switching boundary 주변 adaptive refinement
- nested spatial/action lattice 정의

완전한 continuous convergence theorem이 어렵더라도 일관된 nested lattice,
physical edge length 통제, 독립적인 action refinement, a posteriori
resolution gap 정도만 있어도 현재보다 solution contribution이 강해진다.

### 방향 C — 야심찬 방향

> Certified bilevel or Stackelberg solver

Outer problem:

\[
\max_{s\in\mathcal S}
J_D(s,\operatorname{BR}(s))
\]

에 대해 sensor-position interval bound와 branch-and-bound를 구성한다.

목표:

- inner follower gap
- outer leader gap
- 최종 Stackelberg optimality gap

이렇게 되면 solution novelty가 매우 명확하지만 작업량이 크다.

## 9. 이전 solver를 baseline으로 쓰는 방법

이전 논문의 STP-RRT* + NLP 접근을 현재 side-view 문제에 맞춰 제한하면 가장
자연스러운 baseline이 된다.

| 방법 | 초기화 의존성 | 결과 | 연속성 | 보장 |
|---|---|---|---|---|
| STP-RRT* + NLP | 있음 | local trajectory | continuous | local stationarity |
| Snapped Bellman | 없음 | finite-grid minimum | replay drift 가능 | discrete model |
| Physical-successor Bellman | 없음 | finite-lattice minimum | endpoint consistent | finite-lattice exactness |

핵심 비교 질문은 다음이다.

> Does the local NLP approach remain reliable when terrain masking creates
> discontinuous visibility and multiple path topologies?

비교할 지표:

- attacker mission cost
- initialization sensitivity
- 선택된 masking corridor
- switching point
- continuous violation
- endpoint drift
- computation time

Local NLP가 초기화에 따라 서로 다른 terrain corridor에 갇히고 physical
Bellman이 더 낮은 cost path를 찾는다면 좋은 method result가 된다.

## 10. 추천 contribution hierarchy

### Contribution 1 — 문제 formulation

> Terrain-aware Stackelberg sensor placement against a hybrid powered-to-glide
> aerial evader.

### Contribution 2 — 구조를 이용한 solver

> A virtual-switch physical-successor DAG formulation that exploits monotone
> downrange dynamics.

### Contribution 3 — follower optimality와 execution consistency

> The method exhaustively minimizes over the defined switching-state and
> successor lattice while eliminating accumulated snapping drift.

### Contribution 4 — terrain-induced strategic mechanism

> Numerical results characterize when terrain topology causes strategic
> placement to differ from coverage-only placement.

## 11. Intro 연결 문장 초안

> Our prior work considered continuous sensor redeployment and
> surveillance-evasion trajectories in a top-down planar environment using
> alternating bilevel optimization. The present work studies a terrain-coupled
> regime in which an energy-limited aerial evader chooses both a powered-to-glide
> switching state and a dynamically feasible terrain-masked trajectory.

> Although the defender model is deliberately simplified to a terrain-mounted
> omnidirectional sensor, the follower problem becomes hybrid and nonsmooth
> because sensor placement changes the line-of-sight boundary, reachable glide
> set, switching decision, and accumulated detection risk simultaneously.

> Exploiting monotone downrange motion, we represent the follower problem as a
> directed acyclic physical-successor graph augmented by virtual switching
> states, replacing the locally initialized attacker NLP used in the planar
> setting with an exhaustive finite-lattice best-response oracle.

## 12. 종합 결론

이번 연구를 다음처럼 표현하면 약하다.

> 이전 연구의 side-view 버전이며 sensor를 하나로 단순화했다.

대신 다음처럼 잡는 것이 좋다.

> 이전 연구는 continuous sensor geometry와 local equilibrium computation을
> 다뤘고, 현재 연구는 terrain-coupled hybrid evasion과
> structure-exploiting follower computation을 다룬다.

현재 solver를 완전히 버리거나 새 알고리즘을 처음부터 만들 필요는 없어
보인다. 가장 현실적인 방향은:

1. 현재 problem formulation을 명확히 하고
2. successor-grid solver를 기준으로 theorem을 다시 작성하고
3. 이전 local NLP 및 snapped Bellman과 비교하고
4. 올바른 nested refinement 또는 error-control 결과를 하나 추가하는 것

이다.

이 정도면 problem novelty에만 의존하지 않고, solution 쪽에서도
`hybrid structure를 이용한 finite-grid best-response oracle`이라는 독립적인
contribution을 만들 수 있다.

## 13. 예상 작업량, 3D 필요성, 8월 중순 일정 평가

아래 평가는 한 사람이 사실상 풀타임으로 이 프로젝트에 집중하고,
예상하지 못한 수학적 반례나 대규모 재설계 없이 진행한다는 가정이다.
논문 작성과 지도교수 피드백을 포함하면 실제 calendar time은 더 길어질 수
있다.

### 13.1 방향 B-lite: empirical consistency/error control

범위:

- successor solver에 맞는 theorem 재작성
- nested spatial/action discretization 정의
- physical edge reach와 angular resolution 분리
- 여러 resolution에서 objective, switching point, path, feasibility 측정
- 공통 high-fidelity evaluator에서 cross-resolution policy 재평가
- a posteriori numerical gap 보고
- analytic continuous optimality bound는 주장하지 않음

예상 작업량:

| 작업 | 예상 |
|---|---:|
| 수학적 formulation 및 theorem 정리 | 3–5일 |
| 올바른 nested lattice 설계 | 3–5일 |
| solver/configuration 수정 | 3–6일 |
| regression 및 convergence 실험 | 4–7일 |
| 결과 분석 및 논문 정리 | 3–5일 |
| 합계 | 약 2.5–4주 |

현재 코드를 상당 부분 재사용할 수 있지만, 기존 P4는 action refinement와
physical edge reach가 섞여 있어 그대로 사용할 수 없다.

### 13.2 방향 B-full: rigorous lower/upper bound

강한 방향 B는 다음과 같은 bound를 요구한다.

\[
J_{\mathrm{LB},\Delta}(s)
\le J^*(s)
\le J_{\mathrm{UB},\Delta}(s).
\]

필요한 작업:

- relaxed follower problem 정의
- continuous-feasible path로 upper bound 구성
- lower bound validity 증명
- terrain/LOS discontinuity 처리
- switching-state discretization error 처리
- quadrature error 처리
- bound-gap 기반 adaptive refinement
- toy problem과 production terrain에서 검증

예상 작업량은 약 5–8주이며, LOS와 terrain constraint에 대한 analytic
certification까지 포함하면 그 이상도 가능하다. 가장 큰 불확실성은 코딩이
아니라 lower bound의 validity와 tightness이다.

### 13.3 방향 C-full: continuous leader까지 certified bilevel solution

원래 방향 C는

\[
\max_{s\in\mathcal S}J_D(s,\operatorname{BR}(s))
\]

에 대해 continuous sensor interval 전체의 upper/lower bound를 만들고,
branch-and-bound로 최종 optimality gap을 제공하는 것이다.

C는 B-full의 inner follower bound가 먼저 필요하다. 이후에도
sensor-position interval bound, LOS topology change, outer branch-and-bound,
toy certification, production experiment가 필요하다. B-full 이후 추가로
약 5–9주, B부터 합하면 총 약 10–16주를 보수적인 범위로 본다.

### 13.4 C-lite: fully discretized Stackelberg certificate

C를 다음과 같이 축소하면 작업량이 크게 줄어든다.

- finite sensor set \(\mathcal S_\Delta\) 정의
- 모든 sensor candidate exhaustive evaluation
- 각 sensor에서 finite-grid follower minimum 계산
- finite game에서 exact leader solution 선택

\[
s_\Delta^*
=
\arg\max_{s\in\mathcal S_\Delta}
J_D(s,\operatorname{BR}_\Delta(s)).
\]

이 경우 보장 범위는 다음과 같다.

> Exact Stackelberg solution of the fully discretized leader–follower game.

구현과 검증은 약 1–2주 내 가능성이 있으나 continuous sensor placement에
대한 C-full과는 다른 결과이다.

### 13.5 ACC에 3D가 필수인가

3D는 필수가 아니다. ACC에서 중요한 것은 차원 자체보다 명확한
control/game problem, 문제에서 유도되는 구조, formal result, 검증된
algorithm, numerical insight이다.

2D vertical cross-section은 terrain masking, powered-to-glide switching,
LOS topology, strategic sensor placement, structure-exploiting follower
computation을 연구하는 reduced-order model로 정당화할 수 있다. 다만 다음
제한은 명시해야 한다.

- single omnidirectional sensor
- vertical plane으로 제한된 attacker motion
- lateral bypass 제외
- 실제 3D terrain 우회 경로 제외

현재 `p1b_3DExtension`에는 3D geometry, detection, stage cost, Bellman,
2D defender search prototype이 있고, 190번 outer evaluation을 포함한 약
5.42시간의 baseline run도 존재한다. 그러나 paper-ready 상태는 아니다.

- 3D 전용 regression test가 없음
- 2D의 `successor_grid_physical_edge`와 같은 정식 수정이 없음
- 2D 수준의 continuous replay validation이 없음
- resolution experiment 6개 중 1개 실패
- resolution에 따른 switching point 및 objective 변화가 큼
- full-result serialization 실패
- single Gaussian hill만 검증
- heading turn-rate constraint 없음

따라서 3D를 core contribution으로 만들려면 최소 4–8주 이상의 별도 작업이
필요할 수 있다. 반면 single-hill qualitative demonstration은 제한된 sanity
check와 정확한 scope statement를 전제로 약 1–2주 범위에서 검토할 수 있다.

### 13.6 8월 중순까지 가능한 범위

7월 29일부터 8월 15일까지는 약 17일, 약 12 working days이다.

| 목표 | 8월 중순 가능성 |
|---|---|
| 방향 A 완료 | 높음 |
| B-lite | 가능하지만 일정 위험 있음 |
| B-full | 낮음 |
| C-lite: finite leader-grid certificate | 조건부 가능 |
| C-full: continuous bilevel certificate | 현실적으로 불가능 |
| validated 3D + B/C | 불가능에 가까움 |

원래 정의한 C-full을 8월 중순까지 완료하는 계획은 신뢰성 있는 일정이
아니다. 현실적인 최대 범위는 A + B-lite이며, 공격적으로 진행할 경우
scope를 명확히 제한한 C-lite를 추가할 수 있다.

## 14. 선택된 ACC 목표 — B + C-lite + 3D qualitative demonstration

### 14.1 최종 선택

이번 ACC submission의 목표를 다음으로 정한다.

1. **Direction B:** execution-consistent successor-grid follower에 대해
   올바른 nested discretization family를 구성하고, 공통 continuous evaluator
   및 cross-resolution error indicator로 numerical consistency를 평가한다.
   Continuous optimal lower bound는 검토 후 제외한다. 본 연구는 finite
   discretized state-action space의 exact Bellman value, physical-edge replay,
   nested-grid consistency에 집중하며 analytic continuous certificate를
   주장하지 않는다.
2. **Direction C-lite:** finite defender sensor set과 finite physical-successor
   follower lattice로 정의된 fully discretized Stackelberg game을 exhaustive
   leader enumeration과 exact finite follower oracle로 해결한다.
3. **3D qualitative demonstration:** 기존 3D prototype을 paper의 formal
   guarantee에 포함하지 않고, single-hill terrain에서 확장 가능성을 보여주는
   qualitative demonstration으로 제한한다.

### 14.2 명시적으로 제외하는 범위

- continuous leader-position global certificate
- continuous trajectory-space global optimum
- fully validated 3D Stackelberg solver
- multi-sensor 또는 directional-sensor extension
- 3D resolution-convergence claim

### 14.3 목표 contribution 구조

1. Terrain-aware powered-to-glide Stackelberg formulation
2. Virtual-switch physical-successor DAG follower oracle
3. Nested-discretization consistency/error study
4. Exact solution of the fully discretized leader–follower game
5. Limited 3D qualitative extension demonstrating broader geometric applicability

### 14.4 성공 조건

- successor solver 기준 proposition과 구현이 일치할 것
- spatial, speed, direction, edge reach, quadrature refinement가 구분될 것
- 모든 paper-facing trajectory가 동일한 high-fidelity continuous evaluator를
  통과할 것
- C-lite의 defender set, follower lattice, tie-breaking rule이 명시될 것
- C-lite 결과를 continuous Stackelberg certificate라고 부르지 않을 것
- 3D 결과를 qualitative demonstration 이상으로 해석하지 않을 것

## 15. Direction B execution plan

### 15.1 B의 claim target

현재 일정에서 현실적인 B는 다음으로 정의한다.

> Nested-discretization consistency study with a common high-fidelity
> continuous evaluator for an execution-consistent finite follower family.

필수 B는 올바른 nested discretization, 동일한 physical action envelope,
공통 high-fidelity evaluator, objective/path/switching/ranking 변화 측정 및
numerical error indicator를 포함한다. Continuous optimal lower bound는
검토 후 제외했으므로 결과를 `continuous-optimal`, `error-controlled` 또는
`certified`라고 부르지 않고 `nested-discretization consistency study`라고
부른다.

### 15.2 Nested spatial grid

Resolution level \(\ell=0,1,2\)에 대해

\[
\Delta z_{\ell+1}=\frac{\Delta z_\ell}{2},\qquad
\Delta h_{\ell+1}=\frac{\Delta h_\ell}{2},
\]

\[
N_{\ell+1}=2N_\ell-1
\]

로 정의하여 coarse node가 refined grid에 모두 포함되게 한다. 초기 후보는
`81x51`, `161x101`, `321x201`이며 terrain domain에 따라 정확한 count를
B0에서 확정한다.

### 15.3 Fixed physical edge reach

Grid interval을 절반으로 줄일 때 cell offset을 두 배로 늘려 동일한
physical displacement를 보존한다.

\[
(p,q)_\ell\longrightarrow(2p,2q)_{\ell+1},
\]

\[
p\Delta z_\ell=(2p)\Delta z_{\ell+1},\qquad
q\Delta h_\ell=(2q)\Delta h_{\ell+1}.
\]

따라서 기존 P4처럼 stencil 확대와 maximum physical edge reach 확대를
혼합하지 않는다.

### 15.4 Transported and enriched action families

두 action family를 구분한다.

1. **Transported family:** 이전 level의 physical edge를 refined grid에
   그대로 embedding하여 spatial-state discretization effect를 측정한다.
2. **Enriched family:** 동일한 maximum physical envelope 안에서 새로운
   integer-offset direction을 추가하여 directional-action discretization
   effect를 측정한다.

\[
\mathcal A_\ell^{\mathrm{transport}}
\subset
\mathcal A_{\ell+1}^{\mathrm{transport}},
\qquad
\mathcal A_\ell^{\mathrm{transport}}
\subset
\mathcal A_\ell^{\mathrm{enriched}}.
\]

### 15.5 Nested speed and quadrature levels

Speed grid는

\[
\mathcal V_5\subset\mathcal V_9,
\qquad
\mathcal V_9[::2]=\mathcal V_5
\]

로 둔다. 먼저 v5에서 spatial/direction family를 검증한 뒤 최종 두 level에서
v5/v9를 비교한다.

Solver edge quadrature는 9/17 sample tier로 분리하고, selected policy는
모든 resolution에서 동일한 65 또는 129 sample high-fidelity evaluator로
재평가한다. 정확한 수치는 B0에서 고정한다.

### 15.6 Nested virtual switching states

Switching state는 continuous virtual state를 유지한다.

\[
\Sigma_\ell
=
\left\{
\left(z_i,h_{\mathrm{LOS}}(z_i;s)\right):
z_i\in Z_\ell^{\mathrm{switch}}
\right\},
\qquad
\Sigma_\ell\subset\Sigma_{\ell+1}.
\]

Coarse switching-z sample은 refined set에 포함되고, altitude는 동일한
continuous LOS formula로 계산한다. Virtual source에서 successor까지
physical edge를 사용한다.

### 15.7 Common high-fidelity evaluator

각 resolution의 내부 objective를 직접 비교하지 않고, selected policy
\(\pi_\ell\)를 동일한 evaluator \(\mathcal E_{\mathrm{HF}}\)에 넣는다.

\[
J_\ell^{\mathrm{HF}}=\mathcal E_{\mathrm{HF}}(\pi_\ell).
\]

공통 evaluator는 grid snapping 없음, physical duration, 동일한 continuous
terrain/LOS formula, 높은 segment check 및 quadrature resolution, 동일한 goal
radius와 feasibility tolerance를 사용한다.

### 15.8 B error indicators

Objective change:

\[
e_\ell^J=left|J_{\ell+1}^{\mathrm{HF}}-J_\ell^{\mathrm{HF}}\right|.
\]

Switching displacement:

\[
e_\ell^\sigma=\left\|\sigma_{\ell+1}-\sigma_\ell\right\|_2.
\]

Path comparison은 common-z altitude RMSE, maximum altitude difference,
Hausdorff distance, path-node count 및 masking-corridor/topology change를
포함한다. Feasibility는 terrain/LOS violation, goal reach, minimum clearance,
maximum endpoint residual을 포함한다.

고정된 두 sensor candidate \(s_a,s_b\)의 ranking indicator는

\[
M_\ell
=
J_D^{\mathrm{HF}}(s_a)-J_D^{\mathrm{HF}}(s_b)
\]

로 정의하고 sign, magnitude, resolution shift, reversal을 기록한다.

### 15.9 Continuous optimal lower bound — REVIEWED AND EXCLUDED (2026-07-29)

Continuous optimal lower bound는 검토 후 ACC core에서 제외한다.

제외 사유:

1. 본 연구의 follower problem과 optimality claim은 구현이 정의한 finite
   discretized state-action space를 대상으로 한다.
2. Bellman은 해당 finite physical-successor DAG의 optimal value를 직접
   계산하므로, discrete optimality를 증명하기 위한 별도 lower bound가
   필요하지 않다.
3. Continuous lower bound를 도입하려면 continuous feasible set, objective
   integration, switching set 및 interval/global minimization에 대한 별도
   certification이 필요하며, 이는 현재 논문의 범위를 실질적으로 확장한다.
4. Grid 밖의 물리적 타당성은 continuous global optimality claim이 아니라
   high-fidelity replay와 L0/L1/L2 nested-consistency evidence로 다룬다.

따라서 Direction B의 최종 구조는 다음으로 고정한다.

\[
\boxed{
\text{finite-grid Bellman optimality}
+
\text{physical-edge replay}
+
\text{nested-grid consistency}
}
\]

`p1b_4D/optimal_lower_bound_analytic_equation.md`는 수학적 검토 기록으로만
보존하며, ACC core claim, acceptance gate 또는 production experiment에는
사용하지 않는다.

### 15.10 B execution sequence

1. **B0 — protocol freeze:** level별 state grid, physical edge envelope,
   action family, speed, switching set, quadrature, common evaluator, metrics를
   수식과 숫자로 고정한다.
2. **B1 — toy regression:** node/edge/action/speed/switching nestedness와
   machine-precision endpoint consistency를 검증한다.
3. **B2 — two-hill pilot:** P2의 coverage `1966.4609 m`와 Stackelberg
   `1982.9218 m` 위치만 고정해 3개 spatial level과 transported/enriched
   family를 평가한다. Outer optimization은 반복하지 않는다.
4. **B3 — terrain extension:** two-hill protocol이 안정된 뒤 single hill과
   goal-in-valley로 확장한다.
5. **B4 — production lattice freeze:** B 결과로 C-lite가 사용할 하나의
   production discretization을 고정한다.

### 15.11 Required formal results

최소 proposition chain:

1. Monotone successor graph is a DAG.
2. Every admitted graph edge represents the implementation-defined physical
   segment.
3. Coarse physical actions are embedded in the refined lattice.
4. Bellman computes the exact minimum on each finite graph.
5. Exhaustive switching-state comparison returns the exact finite follower
   value and an exact minimum-cost representative path; exact cost ties use
   the stated switching-z/seed-index rule.
6. High-fidelity replay supplies a feasible-policy upper value.

Continuous lower-bound proposition은 추가하지 않는다. Formal result는
finite physical-successor graph의 optimality와 high-fidelity replay의 역할을
분리해 기술한다.

### 15.12 Link from B to C-lite

B 완료 후 production follower lattice, evaluator, tie-breaking, switching set,
feasibility definition을 고정한다. C-lite는 finite defender set
\(\mathcal S_D=\{s_1,\ldots,s_N\}\)의 모든 위치를 exhaustive evaluation하고

\[
s^*=\arg\max_{s_i\in\mathcal S_D}
J_D(s_i,\operatorname{BR}_\Delta(s_i))
\]

를 선택한다. DIRECT는 comparison 또는 candidate-generation 용도로만
사용하며 exactness claim에는 포함하지 않는다.

### 15.13 Overall priority

1. B protocol 수식 확정
2. nestedness regression
3. two-hill B pilot
4. continuous lower-bound 검토 후 제외 — 완료
5. B main experiment
6. production lattice freeze
7. discrete optimality proposition 현 구현 기준 재작성 — 완료
8. C-lite exhaustive leader solve
9. 3D qualitative demo
10. ACC writing 및 figure 정리

3D는 마지막에 수행한다. B와 C-lite가 formal core이고, 3D는 일정이 남을 때
추가하는 demonstration이다.

## 16. B0 protocol freeze — COMPLETED (2026-07-29)

The normative B0 specification is
`p1b_4D/b0_nested_discretization_protocol.md`.

B0 fixed the following before solver implementation:

- exactly nested L0/L1/L2 spatial grids for all three terrains;
- a corrected `117 -> 233 -> 465` valley z-grid family, replacing the
  non-nestable 467-point B grid;
- fixed physical successor envelopes implemented by cell offsets
  `1x2 -> 2x4 -> 4x8`;
- physically nested enriched and transported action families;
- verified V5 enriched action counts `10 -> 40 -> 160` before terrain/LOS
  state filtering;
- physical-coordinate virtual-switch target boxes instead of floor-index
  windows;
- nested V5/V9 speed grids;
- 9-point main and 17-point sensitivity planning quadrature;
- a 129-point common powered-plus-glide evaluator qualified against 257 points,
  with a 257/513 fallback;
- fixed feasibility, objective, path, switching, ranking, and endpoint metrics;
- deterministic follower tie-breaking;
- the two-hill B2 pilot matrix and the later terrain-extension rule;
- a deterministic production-lattice rule for C-lite; and
- a three-working-day go/no-go gate for an optional relaxed lower bound.

The primary B claim remains numerical consistency over the frozen nested
family. It becomes a lower/upper-bound result only if the optional lower-bound
proof and tightness gates pass. No continuous optimality or analytic
continuous-time feasibility claim is made by B0.

### B1 entry gates

B1 may begin only by implementing tests for:

1. spatial-node subset relations;
2. physical-edge embedding under `(p,q) -> (2p,2q)`;
3. enriched/transported action nestedness;
4. V5-in-V9 nestedness;
5. virtual-switch seed and physical-target nestedness;
6. machine-precision physical endpoint alignment; and
7. common-evaluator 129/257 qualification behavior.

## 17. B1 implementation and pre-B2 audit — COMPLETED WITH B2 BLOCKER (2026-07-29)

### 17.1 Implemented foundation

B1 now has an executable, machine-checkable version of the B0 protocol:

- `p1b_4D/direction_b_discretization.py` constructs the frozen L0/L1/L2,
  enriched/transported, V5/V9, and 9/17-point planning configurations;
- `p1b_4D/successor_grid_solver.py` supports both regular action families,
  physical-box virtual switching targets, and the frozen exact-minimum
  candidate tie rule;
- `p1b_4D/high_fidelity_policy_evaluation.py` independently reevaluates
  powered and glide edges with endpoint-inclusive trapezoidal quadrature and
  no state snapping;
- `p1b_4D/geometry.py` can use a fixed physical-geometry reference z-grid so
  changing the planning grid no longer changes the terrain spline or LOS
  boundary being solved; and
- `p1b_4D/test_direction_b_discretization.py` implements all seven B1 entry
  gates.

The fixed geometry reference is necessary for a valid nested comparison.
Previously the terrain spline and LOS sweep were rebuilt on each planning
grid, so L0/L1/L2 did not represent exactly the same physical geometry.

### 17.2 Verification completed

- All 7 B1 entry-gate tests pass.
- 38 existing configuration, geometry, successor-grid, continuous-replay,
  and Stackelberg regression tests pass.
- The two-hill L2 enriched V5/Q9 policy is continuously feasible under the
  common evaluator, with maximum physical endpoint residual
  `1.11e-16 m`.
- For that L2 pilot, 129 versus 257 samples did **not** meet the frozen
  objective threshold: `|ΔJ_A| = 3.84864e-6 > 1e-6`.
- The prescribed 257 versus 513 fallback passed:
  `|ΔJ_A| = 9.62142e-7` and `|ΔPoD| = 5.02816e-9`, with matching feasibility
  and goal classifications. This is a preliminary pilot; the final common
  sample count is still qualified over every selected B2 policy.

### 17.3 Blocking feasibility result for the frozen three-level family

At the fixed two-hill Stackelberg sensor candidate
`z_sensor = 1982.9218106995881 m`, the enriched V5/Q9 audit found:

| Level | Result | Minimum available descent angle |
|---|---|---:|
| L0 | no switching response reaches the goal | `6.6373 deg` |
| L1 | no switching response reaches the goal | `3.3298 deg` |
| L2 | feasible | `1.6663 deg` |

The regular DAG itself has goal-reachable states at L0/L1, but none lies in
the frozen physical virtual-switch target box of any LOS switching seed. This
is caused by action-lattice resolution: with fixed maximum forward reach
`34.375 m`, the smallest positive descent is `4 m`, `2 m`, and `1 m` at
L0/L1/L2. Only L2 supplies a sufficiently shallow edge direction to connect
the LOS switching boundary to the goal-reachable portion of the DAG.

This is not repaired silently in B1 because changing the envelope, spatial
levels, or admissible offsets would change the frozen B0 experiment family.
B1 infrastructure and regression gates are complete, but B2 cannot produce
the planned three-level objective-convergence table until the protocol is
revised. The next decision must preserve exact nesting while ensuring at
least two, preferably three, feasible enriched levels.

## 18. B1.5 feasible-family revision — COMPLETED (2026-07-29)

The B2 precondition was repaired with one nested shallow backbone added only
to the enriched family:

\[
(4,1)_{L0}\rightarrow(8,2)_{L1}\rightarrow(16,4)_{L2}.
\]

For two hill this represents `(137.5,-4.0) m` and `1.6663 deg` at every
level. Candidate preflight results were:

- no backbone: L0/L1 infeasible at both fixed sensors;
- L0 `(2,1)` backbone: L0/L1 still infeasible;
- L0 `(3,1)` backbone: feasible, but its `2.22 deg` minimum direction does
  not match L2's existing `1.6663 deg` minimum;
- L0 `(4,1)` backbone: feasible at both sensors and both L0/L1, while exactly
  matching L2's shallowest local direction.

The revised enriched offset counts are `3 -> 9 -> 33`, or
`15 -> 45 -> 165` regular actions with V5. The transported ablation is not
augmented. Nine B1/B1.5 tests now pass, including explicit L0/L1 feasibility
at the coverage and Stackelberg B2 sensor candidates.

B2 proceeds with this revised family. The outer defender optimization remains
fixed; only the two recorded P2 sensor candidates are evaluated.

## 19. B2 two-hill nested-consistency run — COMPLETED (2026-07-29)

The full 18-case matrix completed in `148.78 s`:

- 12 enriched main/sensitivity cases were feasible;
- all 6 transported ablation cases were infeasible and retained as such;
- all feasible paths passed terrain, LOS, goal, and physical endpoint checks;
- maximum reported endpoint residual was below `3e-14 m`.

### 19.1 Common evaluator

The initially planned sample counts were insufficient for the `1e-6`
objective gate. Feasibility and goal classifications agreed at every tested
count, but objective quadrature required further refinement. An explicit
doubling sequence was applied:

`129 -> 257 -> 513 -> 1025 -> 2049`.

All 12 feasible policies passed the common 1025/2049 qualification pair. The
worst objective difference was `8.74272e-7`, so B2 uses 1025 samples per edge.

### 19.2 Main enriched V5/Q9 values

| Sensor | L0 HF objective | L1 HF objective | L2 HF objective |
|---|---:|---:|---:|
| coverage | 2.409797807 | 2.398538133 | 2.254588383 |
| Stackelberg | 3.469798211 | 3.171011091 | 3.001366624 |

Objective shifts:

- coverage: L0->L1 `0.011259673`, L1->L2 `0.143949750`;
- Stackelberg: L0->L1 `0.298787120`, L1->L2 `0.169644467`.

The coverage sequence does not show monotonically shrinking shifts. The
Stackelberg shift decreases but remains substantial. B2 therefore records the
actual numerical behavior rather than labeling the three-level sequence as
asymptotically converged.

### 19.3 Path changes

L1-to-L2 common-z altitude RMSE was `10.0736 m` for coverage and `5.2216 m`
for Stackelberg. The corresponding symmetric Hausdorff distances were
`45.3453 m` and `28.5652 m`. Switching-point displacement was about `8.61 m`
for both. All paths retained the same switching-region topology signature.

### 19.4 Ranking and production-lattice decision

The common-reference defender margins were:

- L0: `0.023949998`;
- L1: `0.018420070`;
- L2: `0.019997874`.

Their signs were stable. With L1-to-L2 defender resolution shift
`R = 0.005443445`, the diagnostic condition
`|M_L2| > 2R` passed.

Speed sensitivity exceeded `tau_B = 0.000544344` because the maximum
V9-minus-V5 objective change was `0.001869241`. The production speed lattice
is therefore V9. Q9/Q17 high-fidelity policy values were identical at both
sensors, so production planning quadrature remains Q9.

Machine-readable output:
`results/direction_b/b2_two_hill_nested_consistency.json`.

## 20. B3 multi-terrain nested-consistency extension — COMPLETED (2026-07-29)

B3 repeated the frozen B2 nine-case-per-sensor matrix without rerunning the
outer defender optimization. The fixed final-P2 coverage/Stackelberg sensor
candidates were:

- single hill: `4499.866636 m` / `4499.955545 m`;
- goal in valley: `2199.657064 m` / `2197.828075 m`.

The complete 36-case run finished in `1034.48 s`. Thirty cases were feasible.
For single hill, all 12 enriched main/sensitivity cases were feasible and all
6 transported cases were infeasible. For goal in valley, all 18 cases were
feasible, including every transported ablation. Transported-family
feasibility is therefore terrain/goal dependent rather than a universal
property of the lattice.

### 20.1 Common evaluator and execution consistency

B3 alone would satisfy the objective gate at a smaller sample count, but B2
already required 1025 samples per edge. To preserve one common evaluator for
all Direction-B results, B3 inherited the B2 minimum and was evaluated with
the 1025/2049 qualification pair.

- all 30 feasible policies passed continuous replay;
- maximum 1025/2049 attacker-objective difference: `5.18120e-8`;
- maximum 1025/2049 PoD difference: `2.86160e-8`;
- maximum physical endpoint residual: `4.54748e-13 m`.

### 20.2 Main enriched V5/Q9 values

| Terrain | Sensor | L0 HF objective | L1 HF objective | L2 HF objective |
|---|---|---:|---:|---:|
| single hill | coverage | 0.674388957 | 0.544707020 | 0.535539594 |
| single hill | Stackelberg | 0.674465576 | 0.544718126 | 0.535540156 |
| goal in valley | coverage | 0.913991916 | 0.369200570 | 0.323638614 |
| goal in valley | Stackelberg | 0.903246377 | 0.367479675 | 0.323403740 |

Successive objective shifts were:

- single-hill coverage: `0.129681938`, then `0.009167425`;
- single-hill Stackelberg: `0.129747450`, then `0.009177970`;
- valley coverage: `0.544791345`, then `0.045561956`;
- valley Stackelberg: `0.535766702`, then `0.044075934`.

The L1-to-L2 path changes remain material. Single-hill altitude RMSE was
`29.7271 m` at both candidates and switching displacement was about
`328.26 m`. Valley altitude RMSE was `14.9617 m` for coverage and `14.4851 m`
for Stackelberg, with switching displacement about `8.71 m`. No categorical
switching-region change occurred.

### 20.3 Fixed-candidate ranking diagnostics

Single-hill defender margins remained positive but were extremely small:

- L0: `+5.07757e-5`;
- L1: `+1.67218e-5`;
- L2: `+7.36917e-6`.

The L1-to-L2 defender resolution shift was `R = 0.011579167`, so the frozen
`|M_L2| > 2R` diagnostic did not pass.

Goal-in-valley margins were negative at all three levels:

- L0: `-0.002088802`;
- L1: `-0.001230536`;
- L2: `-0.000224653`.

This means the previously recorded P2 Stackelberg candidate is worse than the
P2 coverage candidate under the B3 finite problems. However, the diagnostic
is unresolved because `R = 0.034359843` is much larger than `|M_L2|`. The
sign reversal must therefore be reported as a grid-family sensitivity of the
fixed candidates, not as a resolved continuous-game ranking.

V5/V9 and Q9/Q17 selected-policy high-fidelity objectives were identical for
both added terrains. Terrain-only decisions would therefore use V5/Q9, but
B4 must combine this result with B2, which required V9/Q9.

### 20.4 Artifacts and verification

- machine-readable result:
  `results/direction_b/b3_multiterrain_nested_consistency.json`;
- trajectories:
  `results/direction_b/figures/b3_multiterrain_trajectories.png`;
- objective/ranking diagnostics:
  `results/direction_b/figures/b3_multiterrain_consistency.png`;
- evaluator qualification:
  `results/direction_b/figures/b3_common_evaluator_qualification.png`.

Forty-two configuration, geometry, successor-grid, continuous-replay,
Direction-B, B2/B3-driver, and visualization regression tests passed after
the B3 implementation.

## 21. B4 production-lattice freeze — COMPLETED (2026-07-29)

B4 did not run another outer optimization. It converted the B1–B3 evidence
into one immutable follower configuration for the finite C-lite game.

### 21.1 Frozen solver in intuitive terms

Every C-lite sensor candidate must now use:

- the finest tested position grid for its terrain;
- 33 physical movement directions and 9 speed candidates, giving 297 regular
  movement choices per state before terrain/LOS filtering;
- 9 samples along each movement segment when planning its cost;
- 1025 samples along each selected segment in the final continuous replay;
- physical successor edges with no endpoint snapping.

The corresponding configuration ID is
`direction_b_l2_enriched_v9_q9_e1025`.

Terrain position-grid sizes are:

| Terrain | Production position grid | Position nodes |
|---|---:|---:|
| single hill | `641x401` | 257,041 |
| two hill | `321x201` | 64,521 |
| goal in valley | `465x201` | 93,465 |

### 21.2 Why these settings were selected

- **Finest grid/action set:** required by the frozen B protocol after the
  L0/L1/L2 consistency study.
- **Nine speeds:** two-hill maximum speed sensitivity was `0.001869241`,
  which exceeded its selection tolerance `0.000544344` by a factor of about
  `3.43`. The other two terrains showed zero selected-policy speed
  sensitivity, but the global configuration uses the most conservative
  terrain requirement.
- **Nine planning samples:** Q9 and Q17 selected the same policy and produced
  the same common-evaluator objective at every tested candidate.
- **1025 replay samples:** B2 required 1025/2049 qualification, and all B3
  policies also passed at this shared count.

### 21.3 C-lite contract

The P2 coverage and Stackelberg locations are not carried forward as C-lite
optima. C-lite must enumerate its stated finite sensor set and solve every
candidate with the B4 production factory. The freeze is an exact contract for
the fully discretized leader–follower game; it does not claim a continuous
leader or continuous follower optimum.

Artifacts:

- manifest: `results/direction_b/b4_production_lattice_freeze.json`;
- decision visualization:
  `results/direction_b/figures/b4_production_lattice_freeze.png`;
- production factory: `build_direction_b_production_configuration` in
  `p1b_4D/direction_b_discretization.py`.

The manifest passed every acceptance gate and has SHA-256
`62b573c6a029067fb68d52dde471bd93cf2c990e5f539665ec797f5e3685da44`.
Forty-five related configuration, geometry, successor-grid, replay,
Direction-B, B2/B3/B4, and visualization regression tests passed.

## 22. Direction B technical core — COMPLETED (2026-07-29)

Direction B의 technical core를 다음 범위로 종료한다.

1. B0/B1은 nested spatial/action/speed/switching family와 common evaluator를
   고정하고 regression으로 검증했다.
2. B2/B3는 two-hill, single-hill, goal-in-valley에서 계획된 54개 case를
   모두 실행했다.
3. B4는 C-lite가 사용할
   `direction_b_l2_enriched_v9_q9_e1025` production lattice를 고정했다.
4. `discrete_optimality_proposition.md`는 current physical-successor DAG,
   virtual switching edge, finite Bellman optimality 및 계산복잡도를 기준으로
   재작성했다.
5. Final response selector는 absolute minimum planning cost를 반드시
   선택하며, exact-equal cost에서만 smallest switching-z와 seed index로
   tie-break하도록 수정했다.
6. Near-tie가 absolute minimum을 대체하지 못하는 regression, exact-tie
   deterministic selection regression 및 final response cost equality check를
   추가했다.

Tie-rule 수정 후 B2의 18개 case와 B3의 36개 case를 모두 재실행했다.
수정 전후의 planning cost, switching point, path-node count 및 high-fidelity
objective를 case별로 비교한 결과 변경된 case는 `0/54`였다. B4 manifest도
모든 acceptance gate를 다시 통과했고 SHA-256은 기존과 동일한
`62b573c6a029067fb68d52dde471bd93cf2c990e5f539665ec797f5e3685da44`였다.

Continuous optimal lower bound는 Section 15.9의 결정대로 제외한다. 따라서
B의 최종 claim은 finite discretized follower optimality, physical-edge
execution consistency 및 nested-grid numerical consistency이며, continuous
trajectory-space global optimality는 포함하지 않는다.

이 시점 이후 다음 technical stage는 C-lite finite defender-set exhaustive
solve이다. ACC manuscript에 B theorem, tables 및 figures를 옮기는 작업은
전체 paper-writing 단계에서 별도로 수행한다.

## 23. 2026-07-30 Update — Current ACC Reinforcement Plan

> **STATUS: CURRENT LATEST PLAN AS OF 2026-07-30.** This is the present
> working plan for the ACC submission and may be revised when formal analysis,
> numerical evidence, verified literature, or advisor feedback changes the
> appropriate scope.

Direction B remains complete in the finite-discretized scope stated in
Section 22. The next work does not repeat B2--B4. It strengthens the paper by
adding a continuous edge-feasibility certificate, graph-level mathematical
results, counterexamples to intuitive follower heuristics, an exhaustive
finite leader solve, and a terrain-to-strategy mechanism study. The 3D result
is limited to one qualitative figure and is not part of the formal guarantee.

### P0 — Freeze the paper claims — COMPLETED (2026-07-30)

The paper is organized around four claims:

1. a terrain-dependent powered-to-glide hybrid Stackelberg formulation;
2. an LOS-derived virtual-switch and physical-successor finite reduction;
3. soundness/completeness properties of that reduction and exact Bellman
   optimality over the explicitly defined finite follower graph; and
4. a strategic mechanism in which terrain changes the hybrid reachable
   structure and thereby changes defender placement.

The paper does not claim a continuous trajectory-space global optimum, a
continuous defender optimum, a new Bellman algorithm, or a formal 3D result.
The strongest technical framing is the execution-consistent graph structure
that makes exact finite dynamic programming possible, not the novelty of the
Bellman recursion itself.

The normative claim contract is
`p1b_4D/acc_claim_scope_0730.md`. It fixes allowed exactness terminology,
explicit non-claims, the formal-result contract, numerical-evidence gates, and
change control for P1--P7. P0 is complete; subsequent evidence may narrow or
explicitly revise the contract but cannot silently broaden it.

### P1 — Continuous segment-feasibility certificate — COMPLETED (2026-07-30)

Replace sampled terrain/LOS admissibility as the formal feasibility basis for
each straight physical edge with an all-segment geometry check under the
implemented geometry model.

- Terrain clearance is minimized over segment endpoints, crossed cubic-spline
  knots, and real stationary points inside every crossed spline interval.
- LOS clearance is checked at segment endpoints and every crossed breakpoint
  of the piecewise-linear swept LOS boundary.
- Constant edge controls and exact endpoint reconstruction retain their current
  checks.
- Detection hazard remains the configured finite quadrature objective; an
  exact continuous hazard integral is not claimed.
- Dense high-fidelity replay remains an independent numerical validation and
  objective reevaluation, rather than the formal feasibility certificate.

Acceptance requires multi-hill support, near-contact regression cases,
agreement with independent dense replay, and no endpoint snapping or state
reset.

P1 is implemented in `p1b_4D/segment_feasibility.py` and documented in
`p1b_4D/p1_exact_segment_feasibility.md`. The checker enumerates cubic-spline
stationary points and LOS-boundary breakpoints, handles exact terminal-edge
truncation, and is connected to powered, virtual, and regular physical edges.
The full `p1b_4D` suite passed 101 tests. A stored-policy audit certified all
42 feasible B2/B3 selected policies; 0 failed, while the other 12 of 54 cases
were already infeasible and had no selected path. Because admissibility was
strengthened without changing finite costs and every stored optimum remains
admitted, the prior B2/B3 optimum values are preserved. P1 is complete and P2
is the next stage.

### P2 — Mathematical proof package

P2 means a mathematical proof, not another numerical convergence experiment.
The finite admissible trajectory class must be defined before proving:

1. **path soundness:** every admitted DAG path concatenates into a
   kinematically continuous powered-to-switch-to-glide trajectory satisfying
   the P1 all-segment terrain/LOS conditions and terminating in the finite goal
   set;
2. **completeness relative to the stated discretization:** every admissible
   trajectory constructed from the declared switching seeds, physical target
   set, action offsets, speed set, terminal rule, and feasibility rules is
   represented in the DAG; and
3. **finite exactness:** exhaustive switching/virtual-edge evaluation and one
   backward Bellman sweep return the global minimum over that finite graph.

Completeness is not claimed for arbitrary continuous trajectories. Regression
tests separately verify that the implementation satisfies the definitions and
hypotheses used by the proof; tests are supporting evidence, not a substitute
for the proof.

### P3 — Intuitive-heuristic counterexample study

Before running the optimal solver, the researcher and Codex will brainstorm
physically plausible terrain intuitions and use them to define randomized
terrain families. To avoid selecting only favorable examples, the terrain
generator, parameter ranges, random seeds, sample count, baseline rules, and
failure metrics must be frozen before comparing solutions. A discovery set may
locate informative regimes, followed by a separately seeded confirmation set.

The planned follower comparisons are:

- nearest LOS-tangent switching;
- highest feasible switching;
- sequential switch-then-glide optimization;
- the legacy snapped/fixed-time transition as an execution-consistency
  ablation; and
- the full virtual-switch physical-successor solver.

The existing legacy case in which the discretely accepted snapped path
violates continuous LOS at step 11 is a candidate controlled ablation, but it
must be regenerated under a frozen configuration. Primary reporting uses a
two-hill or randomized multi-hill counterexample, with a single-hill case as a
control where simple intuition may succeed.

Reported outcomes include goal reachability, all-segment feasibility, attacker
objective gap, switching displacement, and whether an incorrect follower model
changes the selected finite defender position. A strong result requires at
least one reproducible infeasibility, goal failure, material objective gap, or
defender-choice change; otherwise the paper must not claim heuristic failure.

### P4 — C-lite exhaustive finite leader solve

**C-lite is embedded in P4.** It is not a separate stage outside this update.
For each declared finite sensor set, every candidate is evaluated with the B4
production follower configuration
`direction_b_l2_enriched_v9_q9_e1025`, the common evaluator, and deterministic
tie rules. The finite leader solution is

$$
s_\Delta^* = \arg\max_{s\in\mathcal S_{D,\Delta}}
J_D\!\left(s,\operatorname{BR}_\Delta(s)\right).
$$

The sensor candidates, spacing, bounds, endpoint treatment, feasibility
handling, and defender tie rule are fixed before execution. If computationally
practical, nested defender sets are used to report leader-grid sensitivity.
The primary exact finite-game case is two hill; single hill and goal in valley
serve as supporting cases unless their ranking sensitivity becomes sufficiently
resolved. P4 does not produce a continuous Stackelberg certificate.

### P5 — Terrain-induced strategic-mechanism study

Use a controlled terrain-parameter continuation, initially the second-hill
height or spacing in a two-hill family, to measure the chain

$$
\text{terrain parameter}
\rightarrow \text{LOS-boundary topology}
\rightarrow \text{feasible switching/successor set}
\rightarrow \text{follower cost-to-go and response}
\rightarrow \text{finite optimal sensor placement}.
$$

At each frozen parameter value, record active LOS-boundary structure,
switching-seed and admissible-virtual-edge counts, goal-reachable structure,
attacker switching/cost, and the P4-style finite defender optimum. A placement
jump or structural transition is reported only if observed and reproducible;
otherwise the result is presented as a sensitivity study without a phase-change
claim.

### P6 — Verified literature positioning

Verify primary sources for terrain/LOS surveillance or pursuit-evasion games,
hybrid powered-to-glide optimization, finite graph reductions, and exact finite
Stackelberg follower computation. Build a comparison table covering geometry,
endogenous switching, execution consistency, and the precise optimality scope.
No `first`, novelty, or literature-gap claim is used until the cited sources
have been opened and checked against the stated comparison.

### P7 — One-figure qualitative 3D example

The 3D extension is limited to one paper-facing qualitative figure showing a
representative terrain, sensor geometry, and attacker trajectory. It is used
only to illustrate broader geometric applicability. It carries no 3D
optimality, convergence, execution-certificate, or Stackelberg-equilibrium
claim. If schedule or page limits require reduction, P7 is reduced or removed
before P1--P6.

### Execution priority after the 2026-07-30 update

1. P1 continuous edge-feasibility certificate;
2. P2 mathematical proof package;
3. P3 frozen randomized-terrain and heuristic-counterexample study;
4. P4 C-lite exhaustive finite leader solve;
5. P5 terrain-induced strategic mechanism;
6. P6 verified literature positioning and manuscript framing; and
7. P7 one-figure qualitative 3D example.

### Stage review notebook index — 2026-07-30

B1–B4와 P1–P7은 각각 한 파일에서 목적, 실행 진입점, 표·그림 output을 검토할 수 있도록
`p1b_4D/stage_notebooks/` 아래의 독립 노트북으로 정리했다. 전체 사용법과 상태표는
`p1b_4D/stage_notebooks/README.md`에 둔다.

- B1–B4와 P1은 `COMPLETED`이며 저장된 output을 포함한다.
- B2/B3/B4는 기본적으로 기존 검증 결과를 즉시 표시한다. 원자료부터 다시 계산하려면
  각 노트북의 `RERUN = True`로 변경한다.
- P1은 Run All 시 exact geometry audit와 핵심 regression을 다시 수행한다.
- P2–P7은 아직 구현 전이므로 `PENDING`을 명시하고, 완료 조건과 향후 module/result
  entry point만 출력한다. 완료되지 않은 결과를 notebook이 대신 주장하지 않는다.
- 생성 원본은 `p1b_4D/build_stage_review_notebooks.py`이며, 생성기를 실행하면 11개
  노트북의 설명과 실행 구조를 일관되게 재생성할 수 있다.

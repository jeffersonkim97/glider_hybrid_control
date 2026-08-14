# 2D Discrete SAC Extension Plan

## 목표

- `p1b_4D`의 exact physical Bellman 해를 기준군(ground truth)으로 고정한다.
- 동일한 2D MDP에서 Discrete Soft Actor-Critic(SAC)을 학습한다.
- SAC가 Bellman 경로와 cost-to-go를 어느 정확도로 복원하는지 정량 검증한다.
- 검증 후에만 3D, multi-sensor, continuous-action SAC로 확장한다.

## 변경하지 않을 기준 계약

- `p1b_4D`는 authoritative reference이며 직접 수정하지 않는다.
- 상태는 기존 4D 상태를 유지한다.
  - `s = (z, h, v, gamma)`
- 행동은 기존 physical successor action의 discrete index를 사용한다.
- terrain, LOS, detection, edge quadrature, attacker cost와 goal 조건을 그대로 사용한다.
- powered phase와 LOS tangent-line switching 후보 생성은 기존 방식을 유지한다.
- 첫 실험에서는 SAC가 switching 이후 glide policy만 학습한다.
- 4D→2D projected cost-to-go는 시각화용이며 SAC observation으로 사용하지 않는다.

## 1. Exact Bellman 기준군 제작

- `p1b_4D` 모듈을 복사하지 않고 import하여 동일한 계산을 호출한다.
- 첫 기준 시나리오를 하나로 고정한다.
  - canonical single-hill
  - 고정 sensor 위치
  - 고정 grid, cost weight, random seed
- 다음 기준 데이터를 저장한다.
  - 4D state grids와 state-validity mask
  - physical successor/action table
  - edge cost와 feasibility
  - exact Bellman value `V*(s)`
  - exact greedy policy `pi*(s)`
  - switching candidates와 선택된 switching point
  - optimal trajectory, mission cost, PoD, mission time
  - continuous replay validation 결과
- 모든 결과에 configuration hash와 grid metadata를 기록한다.

## 2. RL Transition 계약

- Bellman과 SAC가 동일한 transition을 사용하도록 다음 tuple을 정의한다.
  - `(state, action_index, edge_cost, next_state, done, feasible)`
- reward는 기존 cost의 음수만 사용한다.
  - `reward = -edge_cost`
- 초기 비교에서는 별도의 reward shaping을 사용하지 않는다.
- infeasible action은 action mask로 선택을 차단한다.
- goal 도달은 terminal success로 처리한다.
- collision, domain 이탈, LOS 위반은 허용하지 않는다.
- state와 reward normalization parameter를 별도 파일로 저장한다.

## 3. Discrete SAC 구현

- Actor
  - 4D state를 입력받아 discrete action별 categorical probability 출력
- Critic
  - twin Q-network `Q1(s,a)`, `Q2(s,a)` 사용
- Target critic
  - Polyak averaging으로 갱신
- Replay buffer
  - off-policy transition 저장 및 mini-batch sampling
- Entropy temperature
  - `alpha` 고정 실험 후 automatic temperature tuning 비교
- Action mask
  - actor sampling과 target-value 계산 모두에 동일하게 적용
- Evaluation policy
  - 학습 중에는 stochastic action
  - 최종 평가는 maximum-probability deterministic action

## 4. Switching과 SAC 연결

- 기존 LOS tangent-line에서 switching 후보를 생성한다.
- 각 switching state에 대해 SAC critic의 downstream value를 평가한다.
- powered cost와 SAC downstream cost의 합이 최소인 후보를 선택한다.
- 선택된 switching state부터 deterministic SAC policy를 rollout한다.
- exact Bellman과 동일한 continuous replay validator로 최종 경로를 검사한다.

## 5. 학습 순서

- 작은 grid에서 pipeline smoke test
- 단일 fixed sensor에서 학습 안정성 확인
- random seed 반복으로 평균과 분산 측정
- exact Bellman transition을 이용한 offline prefill 유무 비교
- full reference grid에서 최종 학습
- 학습에 사용하지 않은 초기 state와 sensor 위치에서 generalization 평가

## 6. Bellman 대비 평가항목

- value-function error
  - MAE, RMSE, maximum error on reachable states
- policy agreement
  - exact Bellman action과 SAC greedy action의 일치율
- trajectory performance
  - mission-cost optimality gap
  - mission PoD error
  - mission-time error
  - switching-point distance
- feasibility
  - goal success rate
  - terrain violation rate
  - LOS violation rate
  - invalid-action selection rate
- computation
  - Bellman solve time
  - SAC training time
  - 학습 후 1회 policy inference/rollout time

## 7. 초기 성공 기준

- continuous replay 기준 terrain/LOS violation `0`
- deterministic evaluation goal success rate `>= 99%`
- reachable test states에서 mean mission-cost gap `<= 5%`
- 최종 대표 경로의 PoD와 time을 exact Bellman 결과와 함께 보고
- 여러 random seed에서 결과의 평균과 표준편차 보고

## 8. 예정 폴더 구조

```text
2D_RL/
├── DISCRETE_SAC_PLAN.md
├── configuration.py          # RL 실험 설정과 reference hash
├── bellman_reference.py      # p1b_4D 기준군 실행/내보내기
├── transition_dataset.py     # 공통 (s,a,c,s',done,mask) 계약
├── environment.py            # 2D discrete-action RL environment
├── networks.py               # categorical actor, twin critics
├── discrete_sac.py           # 학습 및 target update
├── train.py                  # reproducible training runner
├── evaluate.py               # Bellman-SAC 정량 비교
├── visualize.py              # value, policy, trajectory 비교
├── tests/
├── results/
└── figures/
```

## 9. 구현 순서

- [ ] `p1b_4D` exact Bellman 기준군 생성 및 hash 저장
- [ ] Bellman transition/action mask exporter 작성
- [ ] RL environment 작성 및 Bellman transition과 일치 검증
- [ ] Discrete SAC actor/critic/replay buffer 구현
- [ ] 작은 grid smoke training
- [ ] full 2D single-hill 학습
- [ ] Bellman-SAC value/policy/trajectory 비교 figure 생성
- [ ] continuous replay와 성공 기준 검증
- [ ] sensor-conditioned policy 설계
- [ ] 검증 완료 후 multi-sensor 및 3D 확장 판단

## 이번 단계의 비범위

- 3D SAC
- continuous-action SAC
- multi-sensor optimization
- Defender RL
- mixed-strategy learning
- RL을 이용한 switching action 학습

이 항목들은 2D Discrete SAC가 exact Bellman 기준군을 충분히 재현한 뒤 진행한다.

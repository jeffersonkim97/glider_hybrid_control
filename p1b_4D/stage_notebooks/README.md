# B1–B4 / P1–P7 stage review notebooks

각 노트북은 해당 단계의 목적, 실행 코드, 표·그림 출력을 한 파일 안에서 검토하기 위한 진입점이다. 커널은 프로젝트의 `Python (.venv_p1b)`를 사용한다.

| 단계 | 노트북 | 현재 상태 | Run All 동작 |
|---|---|---|---|
| B1 | `B1_Nested_Discretization_Verification.ipynb` | 완료 | nested grid/action regression을 다시 실행하고 구조 표를 출력 |
| B2 | `B2_Two_Hill_Nested_Consistency.ipynb` | 완료 | 저장된 18-case 결과를 읽고 표와 그림을 즉시 출력 |
| B3 | `B3_Multi_Terrain_Extension.ipynb` | 완료 | 저장된 36-case 결과를 읽고 표와 그림을 즉시 출력 |
| B4 | `B4_Production_Lattice_Freeze.ipynb` | 완료 | frozen production manifest를 읽고 표와 그림을 즉시 출력 |
| P1 | `P1_Exact_Segment_Feasibility.ipynb` | 완료 | exact segment audit와 핵심 regression을 다시 실행 |
| P2 | `P2_Mathematical_Proof_Package.ipynb` | 대기 | 현재 proof 초안과 남은 proof 조건을 출력 |
| P3 | `P3_Randomized_Terrain_Counterexamples.ipynb` | 대기 | 사전 고정할 실험 계약과 결과 경로를 출력 |
| P4 | `P4_C_Lite_Exhaustive_Leader_Solve.ipynb` | 대기 | C-Lite 조건과 B4 frozen follower 설정을 출력 |
| P5 | `P5_Terrain_Strategic_Mechanism.ipynb` | 대기 | terrain-mechanism 실험 계약을 출력 |
| P6 | `P6_Verified_Literature_Positioning.ipynb` | 대기 | literature verification 완료 조건을 출력 |
| P7 | `P7_3D_Qualitative_Demo.ipynb` | 대기 | 3D figure 조건과 기존 exploratory artifact 링크를 출력 |

## 재계산 방법

B2, B3, B4는 기본값이 `RERUN = False`다. Run All만 누르면 이미 검증된 결과를 빠르게 검토할 수 있다. 원자료부터 다시 계산하려면 각 노트북 첫 실행 셀의 `RERUN = True`로 바꾼다. B2/B3는 계산량이 크다.

P2–P7은 아직 구현되지 않았으므로 현재는 완료 결과를 가장하지 않고 `PENDING`을 표시한다. 해당 단계가 구현되면 노트북에 고정된 module/result entry point가 같은 파일 안에서 계산과 시각화를 수행하게 된다.

노트북을 코드에서 다시 생성하고 모든 output을 저장하려면 repository root에서 다음을 실행한다.

```powershell
.\.venv_p1b\Scripts\python.exe -m p1b_4D.build_stage_review_notebooks --execute --timeout 900
```


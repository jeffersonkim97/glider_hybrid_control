"""Generate the B1--B4 and P1--P7 single-file review notebooks.

The notebooks are intentionally thin orchestration/review layers. Numerical
implementations remain in importable Python modules, while each notebook gives
one Korean, intuitive narrative and renders the stage artifacts inline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "p1b_4D" / "stage_notebooks"


def _markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def _code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def _setup(stage: str, status: str):
    return _code(f"""
    from pathlib import Path
    import json
    from html import escape
    import subprocess
    import sys
    from IPython.display import display, Markdown, Image, HTML, FileLink

    def locate_repo_root():
        current = Path.cwd().resolve()
        for candidate in (current, *current.parents):
            if (candidate / "p1b_4D").is_dir() and (candidate / "p1b_roadmap_0729.md").exists():
                return candidate
        raise RuntimeError("glider_hybrid_control repository root를 찾지 못했습니다.")

    ROOT = locate_repo_root()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    RESULTS = ROOT / "results"
    STAGE = "{stage}"
    STATUS = "{status}"

    def run_module(module, *arguments, timeout=None):
        command = [sys.executable, "-m", module, *map(str, arguments)]
        completed = subprocess.run(
            command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout,
        )
        print(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(f"{{module}} 실행 실패: exit={{completed.returncode}}")
        return completed.stdout

    def load_json(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"결과 파일이 없습니다: {{path}}")
        return json.loads(path.read_text(encoding="utf-8"))

    def show_png(path, width=1050):
        path = Path(path)
        if path.exists():
            display(Image(filename=str(path), width=width))
        else:
            display(Markdown(f"> ⚠️ 그림이 없습니다: `{{path}}`"))

    def display_table(rows, columns=None):
        if isinstance(rows, dict):
            rows = [rows]
        rows = list(rows)
        if columns is None:
            columns = []
            for row in rows:
                for key in row:
                    if key not in columns:
                        columns.append(key)
        if not rows:
            display(Markdown("_(표시할 행이 없습니다.)_"))
            return
        def cell(value):
            if isinstance(value, float):
                value = f"{{value:.8g}}"
            elif isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False)
            return escape(str(value))
        header = "".join(f"<th>{{cell(name)}}</th>" for name in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{{cell(row.get(name, ''))}}</td>" for name in columns) + "</tr>"
            for row in rows
        )
        display(HTML(
            "<div style='overflow-x:auto'><table>"
            f"<thead><tr>{{header}}</tr></thead><tbody>{{body}}</tbody></table></div>"
        ))

    display(Markdown(f"**{{STAGE}} 상태:** `{{STATUS}}`  \\nRepository: `{{ROOT}}`"))
    """)


def _notebook(stage: str, title: str, status: str, intro: str, cells: list):
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        _markdown(f"# {stage} — {title}\n\n{intro}"),
        _setup(stage, status),
        *cells,
    ]
    notebook.metadata.update({
        "kernelspec": {
            "display_name": "Python (.venv_p1b)",
            "language": "python",
            "name": "venv_p1b",
        },
        "language_info": {"name": "python", "version": "3"},
        "stage_review": {
            "stage": stage,
            "status": status,
            "generated_by": "p1b_4D.build_stage_review_notebooks",
        },
    })
    return notebook


def _b1():
    return _notebook(
        "B1", "Nested Discretization 구조 검증", "COMPLETED",
        """
        B1은 **격자를 촘촘하게 만들 때 이전 단계의 위치와 물리적 움직임이
        그대로 포함되는지** 확인한다. 쉽게 말해 L0에서 가능했던 움직임을
        L1/L2가 잃어버리지 않고, virtual switching edge가 실제 grid node에
        정확히 연결되는지 검사하는 단계다.
        """,
        [
            _markdown("""
            ## 이 노트북에서 확인하는 것

            - L0 ⊂ L1 ⊂ L2 spatial grid
            - physical edge와 speed/action family의 nesting
            - virtual switching target의 nesting
            - endpoint snapping 없이 machine-precision endpoint 도달
            - common high-fidelity evaluator qualification 계약
            """),
            _code("""
            from p1b_4D.direction_b_discretization import (
                DIRECTION_B_GRID_COUNTS, DIRECTION_B_SPEEDS,
                DIRECTION_B_PRODUCTION_CONFIGURATION_ID,
            )

            rows = []
            enriched_offsets = {0: 3, 1: 9, 2: 33}
            for terrain, levels in DIRECTION_B_GRID_COUNTS.items():
                for level, (nz, nh) in enumerate(levels):
                    rows.append({
                        "terrain": terrain, "level": f"L{level}",
                        "z nodes": nz, "h nodes": nh,
                        "position nodes": nz * nh,
                        "enriched directions": enriched_offsets[level],
                        "V5 actions/state": enriched_offsets[level] * len(DIRECTION_B_SPEEDS["V5"]),
                    })
            display_table(rows)
            print("B4 production configuration:", DIRECTION_B_PRODUCTION_CONFIGURATION_ID)
            """),
            _markdown("## Regression 실행"),
            _code("""
            RUN_TESTS = True
            if RUN_TESTS:
                output = run_module(
                    "unittest", "p1b_4D.test_direction_b_discretization"
                )
                display(Markdown("✅ **B1 entry-gate regression 통과**"))
            else:
                display(Markdown("검사를 건너뛰었습니다. `RUN_TESTS=True`로 변경하세요."))
            """),
            _markdown("""
            ## 직관적 결론

            L0→L1→L2는 서로 무관한 세 solver가 아니다. 동일한 물리 문제를
            더 세밀하게 보는 nested family다. 따라서 뒤 단계의 차이를
            grid/action resolution 변화로 해석할 수 있다.
            """),
        ],
    )


def _b2():
    return _notebook(
        "B2", "Two-Hill Nested Consistency", "COMPLETED",
        """
        B2는 two-hill 환경에서 coverage 위치와 Stackelberg 위치를 고정한 뒤,
        L0/L1/L2와 action/speed/quadrature 변화를 비교한다. 핵심 질문은
        **격자를 바꿔도 경로와 두 sensor 후보의 순위가 얼마나 유지되는가**다.
        """,
        [
            _code("""
            RERUN = False  # True이면 18-case 전체를 다시 계산하므로 오래 걸립니다.
            RESULT_PATH = RESULTS / "direction_b" / "b2_two_hill_nested_consistency.json"
            FIGURE_DIR = RESULTS / "direction_b" / "figures"
            if RERUN:
                run_module(
                    "p1b_4D.experiment_b2_two_hill_nested_consistency",
                    "--output", RESULT_PATH,
                )
            result = load_json(RESULT_PATH)
            display(Markdown(
                f"**상태:** `{result['status']}` · cases={result['case_count']} · "
                f"feasible={result['feasible_case_count']} · "
                f"common evaluator={result['global_common_evaluator_sample_count']}"
            ))
            """),
            _code("""
            rows = []
            for case in result["cases"]:
                if (
                    case["action_family"] == "enriched"
                    and case["speed_family"] == "V5"
                    and case["planning_quadrature_count"] == 9
                ):
                    rows.append({
                        "sensor": case["sensor_name"],
                        "level": f"L{case['level']}",
                        "status": case["status"],
                        "HF attacker objective": (
                            case.get("selected_high_fidelity", {}).get("attacker_objective")
                        ),
                        "switch z": (
                            case.get("planning", {}).get("switching_point", [None])[0]
                        ),
                        "path nodes": case.get("planning", {}).get("path_node_count"),
                    })
            display_table(rows)
            display_table([
                {"metric": "ranking sign stable", "value": result["analysis"]["ranking_sign_stable"]},
                {"metric": "ranking diagnostically resolved", "value": result["analysis"]["ranking_diagnostically_resolved"]},
                {"metric": "production speed", "value": result["analysis"]["production_speed_family"]},
                {"metric": "production planning quadrature", "value": result["analysis"]["production_planning_quadrature_count"]},
            ])
            """),
            _code("""
            from p1b_4D.visualize_b2_two_hill_nested_consistency import create_b2_figures
            figure_paths = create_b2_figures(RESULT_PATH, FIGURE_DIR, ROOT)
            for figure_path in figure_paths:
                show_png(figure_path)
            """),
            _markdown("""
            ## 직관적 결론

            모든 feasible 경로는 물리적으로 replay되지만 L1→L2 경로 변화는
            아직 작지 않다. 다만 two-hill의 두 sensor 후보 순위 부호는
            유지됐고, speed sensitivity 때문에 최종 production speed는 V9로
            결정됐다.
            """),
        ],
    )


def _b3():
    return _notebook(
        "B3", "Multi-Terrain Nested Consistency", "COMPLETED",
        """
        B3는 B2에서 고정한 동일한 비교 절차를 single hill과 goal-in-valley에
        적용한다. 새로운 outer optimum을 찾는 단계가 아니라, **같은 solver
        family가 다른 terrain에서도 어떻게 달라지는지** 보는 확장 실험이다.
        """,
        [
            _code("""
            RERUN = False  # True이면 36-case 전체를 다시 계산하므로 오래 걸립니다.
            RESULT_PATH = RESULTS / "direction_b" / "b3_multiterrain_nested_consistency.json"
            FIGURE_DIR = RESULTS / "direction_b" / "figures"
            if RERUN:
                run_module(
                    "p1b_4D.experiment_b3_multiterrain_nested_consistency",
                    "--output", RESULT_PATH,
                )
            result = load_json(RESULT_PATH)
            display(Markdown(
                f"**상태:** `{result['status']}` · cases={result['case_count']} · "
                f"feasible={result['feasible_case_count']} · "
                f"common evaluator={result['global_common_evaluator_sample_count']}"
            ))
            """),
            _code("""
            rows = []
            for case in result["cases"]:
                if (
                    case["action_family"] == "enriched"
                    and case["speed_family"] == "V5"
                    and case["planning_quadrature_count"] == 9
                ):
                    rows.append({
                        "terrain": case["terrain_name"],
                        "sensor": case["sensor_name"],
                        "level": f"L{case['level']}",
                        "status": case["status"],
                        "HF attacker objective": (
                            case.get("selected_high_fidelity", {}).get("attacker_objective")
                        ),
                        "switch z": case.get("planning", {}).get("switching_point", [None])[0],
                    })
            display_table(rows)
            ranking_rows = []
            for terrain, analysis in result["analysis"].items():
                ranking_rows.append({
                    "terrain": terrain,
                    "ranking sign stable": analysis["ranking_sign_stable"],
                    "ranking resolved": analysis["ranking_diagnostically_resolved"],
                    "production speed": analysis["production_speed_family_if_terrain_only"],
                    "planning quadrature": analysis["production_planning_quadrature_count_if_terrain_only"],
                })
            display_table(ranking_rows)
            """),
            _code("""
            from p1b_4D.visualize_b3_multiterrain_nested_consistency import create_b3_figures
            figure_paths = create_b3_figures(RESULT_PATH, FIGURE_DIR, ROOT)
            for figure_path in figure_paths:
                show_png(figure_path)
            """),
            _markdown("""
            ## 직관적 결론

            동일한 action family라도 terrain/goal 구조에 따라 feasibility와
            ranking sensitivity가 달라진다. 특히 single-hill과 valley의 두
            고정 sensor 후보 순위는 resolution noise보다 작으므로, 이를
            연속-game의 확정적 순위로 해석하지 않는다.
            """),
        ],
    )


def _b4():
    return _notebook(
        "B4", "Production Lattice Freeze", "COMPLETED",
        """
        B4는 새 최적화를 수행하는 단계가 아니다. B1–B3의 결과를 근거로
        이후 C-lite가 모든 sensor candidate에 사용할 **하나의 고정 follower
        solver 설정**을 선택하고 manifest로 잠그는 단계다.
        """,
        [
            _code("""
            RERUN = False
            B2_PATH = RESULTS / "direction_b" / "b2_two_hill_nested_consistency.json"
            B3_PATH = RESULTS / "direction_b" / "b3_multiterrain_nested_consistency.json"
            MANIFEST_PATH = RESULTS / "direction_b" / "b4_production_lattice_freeze.json"
            FIGURE_PATH = RESULTS / "direction_b" / "figures" / "b4_production_lattice_freeze.png"
            if RERUN:
                run_module(
                    "p1b_4D.experiment_b4_production_lattice_freeze",
                    "--b2-result", B2_PATH, "--b3-result", B3_PATH,
                    "--output", MANIFEST_PATH,
                )
            manifest = load_json(MANIFEST_PATH)
            display(Markdown(
                f"**상태:** `{manifest['status']}`  \\n"
                f"**Configuration ID:** `{manifest['production_configuration_id']}`  \\n"
                f"**SHA-256:** `{manifest['manifest_sha256']}`"
            ))
            """),
            _code("""
            display_table([
                {"setting": key, "selected value": value}
                for key, value in manifest["selected_settings"].items()
            ])
            terrain_rows = []
            for terrain, settings in manifest["terrain_configurations"].items():
                terrain_rows.append({"terrain": terrain, **settings})
            display_table(terrain_rows)
            display_table([
                {"acceptance gate": key, "passed": value}
                for key, value in manifest["acceptance_gates"].items()
            ])
            """),
            _code("""
            from p1b_4D.visualize_b4_production_lattice_freeze import create_b4_figure
            create_b4_figure(MANIFEST_PATH, FIGURE_PATH)
            show_png(FIGURE_PATH)
            """),
            _markdown("""
            ## 직관적 결론

            C-lite는 L2 위치 격자, enriched 방향, 9개 속도, Q9 planning cost,
            1025-point replay를 모든 leader 후보에 동일하게 사용해야 한다.
            이 설정은 finite game contract이며 continuous optimum 주장이 아니다.
            """),
        ],
    )


def _p1():
    return _notebook(
        "P1", "Exact Straight-Segment Geometry Feasibility", "COMPLETED",
        """
        P1은 edge 위 몇 점만 검사하던 방식을 바꿔, **edge 전체에서 terrain과
        LOS를 실제로 침범하지 않는지** 확인한다. Cubic terrain은 knot와
        stationary point를, piecewise-linear LOS는 모든 breakpoint를 검사한다.
        """,
        [
            _code("""
            RERUN_AUDIT = True
            RUN_REGRESSION_TESTS = True
            RUN_FULL_SUITE = False  # True이면 약 7분 이상 걸릴 수 있습니다.
            AUDIT_PATH = RESULTS / "direction_b" / "p1_exact_geometry_audit.json"

            if RERUN_AUDIT:
                run_module(
                    "p1b_4D.audit_p1_direction_b_geometry_certificates",
                    "--project-root", ROOT, "--output", AUDIT_PATH,
                )
            audit = load_json(AUDIT_PATH)
            display_table(audit["summary"])
            """),
            _code("""
            applicable = [
                record for record in audit["records"] if record.get("passed") is not None
            ]
            margin_columns = [
                "source", "terrain_name", "sensor_name", "case_id", "passed",
                "minimum_powered_terrain_margin", "minimum_powered_occlusion_margin",
                "minimum_glide_terrain_margin", "minimum_glide_los_margin",
            ]
            display_table(applicable, margin_columns)
            numeric_margin_columns = [
                "minimum_powered_terrain_margin", "minimum_powered_occlusion_margin",
                "minimum_glide_terrain_margin", "minimum_glide_los_margin",
            ]
            terrain_minima = []
            for terrain_name in sorted({record["terrain_name"] for record in applicable}):
                subset = [record for record in applicable if record["terrain_name"] == terrain_name]
                terrain_minima.append({
                    "terrain_name": terrain_name,
                    **{
                        name: min(record[name] for record in subset)
                        for name in numeric_margin_columns
                    },
                })
            display_table(terrain_minima)
            """),
            _code("""
            if RUN_REGRESSION_TESTS:
                run_module(
                    "unittest", "p1b_4D.test_segment_feasibility",
                    "p1b_4D.test_successor_grid_solver",
                )
                display(Markdown("✅ **P1 exact-geometry와 successor regression 통과**"))
            if RUN_FULL_SUITE:
                run_module("unittest", "discover", "-s", "p1b_4D", "-p", "test_*.py")
            """),
            _markdown("""
            ## 직관적 결론

            저장된 B2/B3 사례 54개 중 경로가 존재하는 42개 모두가 exact
            geometry certificate를 통과했다. 따라서 P1은 기존 선택 경로와
            finite cost를 유지하면서, 다른 sampled-valid edge가 중간에서
            장애물을 통과할 가능성을 제거한다.
            """),
        ],
    )


def _pending_notebook(stage, title, intuitive, required, expected_module, result_rel, extra_cells=None):
    cells = [
        _markdown(f"""
        ## 현재 상태

        이 단계는 아직 구현 완료되지 않았다. 이 노트북은 완료된 척하지 않고,
        구현 목표와 향후 실행 진입점을 한 파일에 고정한다.

        **완료 조건**

        {required}
        """),
        _code(f"""
        RERUN = False
        EXPECTED_MODULE = ROOT / "p1b_4D" / "{expected_module}.py"
        RESULT_PATH = ROOT / "{result_rel}"
        status = {{
            "stage": "{stage}",
            "implementation module exists": EXPECTED_MODULE.exists(),
            "result exists": RESULT_PATH.exists(),
            "rerun requested": RERUN,
        }}
        display_table(status)

        if RERUN:
            if not EXPECTED_MODULE.exists():
                raise RuntimeError(
                    f"{stage} 구현이 아직 없습니다: {{EXPECTED_MODULE.name}}"
                )
            run_module("p1b_4D.{expected_module}")

        if RESULT_PATH.exists():
            payload = load_json(RESULT_PATH)
            display(Markdown("✅ 저장된 결과를 발견했습니다."))
            display(Markdown("```json\\n" + json.dumps(payload, ensure_ascii=False, indent=2)[:12000] + "\\n```"))
        else:
            display(Markdown(
                "> ⏳ **PENDING:** 구현·실험이 완료되면 이 셀이 결과 표와 그림을 바로 표시합니다."
            ))
        """),
    ]
    if extra_cells:
        cells.extend(extra_cells)
    cells.append(_markdown(f"## 직관적 목적\n\n{intuitive}"))
    return _notebook(stage, title, "PENDING", intuitive, cells)


def _p2():
    return _pending_notebook(
        "P2", "Mathematical Proof Package",
        "P2는 실험이 아니라, finite graph의 path가 실제 hybrid trajectory가 되고 그 graph 안에서는 Bellman 해가 전역 최적이라는 것을 수학적으로 연결하는 단계다.",
        """
        - graph-path soundness proof
        - declared discretization에 대한 relative completeness proof
        - finite Bellman/exhaustive-switch exactness proof
        - continuous trajectory completeness를 주장하지 않는 명확한 scope
        """,
        "p2_mathematical_proof_package",
        "results/p2/p2_proof_validation.json",
        extra_cells=[
            _code("""
            PROPOSITION = ROOT / "p1b_4D" / "discrete_optimality_proposition.md"
            text = PROPOSITION.read_text(encoding="utf-8")
            headings = [line for line in text.splitlines() if line.startswith("##")]
            display(Markdown("### 현재 proof 초안의 구조\\n" + "\\n".join(f"- {h}" for h in headings)))
            display(FileLink(str(PROPOSITION)))
            display(Markdown(
                "> 현재 문서는 finite edge/DAG/Bellman proof를 포함하지만, P2의 graph-level soundness와 relative completeness 패키지는 아직 미완료입니다."
            ))
            """),
        ],
    )


def _p3():
    return _pending_notebook(
        "P3", "Randomized-Terrain Heuristic Counterexamples",
        "직관적 tangent/highest/sequential heuristic이 언제 실패하는지, 미리 고정한 randomized terrain generator와 별도 confirmation seed에서 검증한다.",
        """
        - terrain generator, 범위, seed, sample 수 사전 고정
        - discovery set과 confirmation set 분리
        - tangent/highest/sequential/legacy/full solver 비교
        - feasibility, goal, cost gap, defender-choice 변화 출력
        """,
        "experiment_p3_randomized_terrain_counterexamples",
        "results/p3/p3_randomized_terrain_counterexamples.json",
    )


def _p4():
    return _pending_notebook(
        "P4", "C-Lite Exhaustive Finite Leader Solve",
        "P4가 C-lite다. 유한 sensor 후보를 하나도 건너뛰지 않고 B4 follower로 모두 풀어 finite Stackelberg optimum을 선택한다.",
        """
        - finite sensor set/bounds/spacing/tie rule 사전 고정
        - 모든 leader 후보 exhaustive evaluation
        - 동일한 B4 follower와 evaluator 사용
        - defender objective curve, argmax, leader-grid sensitivity 출력
        """,
        "experiment_p4_c_lite_exhaustive_leader",
        "results/p4/p4_c_lite_exhaustive_leader.json",
        extra_cells=[
            _code("""
            manifest_path = RESULTS / "direction_b" / "b4_production_lattice_freeze.json"
            manifest = load_json(manifest_path)
            display(Markdown(
                f"P4가 사용할 frozen follower: `{manifest['production_configuration_id']}`"
            ))
            display_table([
                {"setting": key, "value": value}
                for key, value in manifest["selected_settings"].items()
            ])
            """),
        ],
    )


def _p5():
    return _pending_notebook(
        "P5", "Terrain-Induced Strategic Mechanism",
        "두 번째 hill의 높이/간격을 바꾸면서 LOS topology→successor set→attacker response→sensor optimum의 연결을 한 화면에서 추적한다.",
        """
        - terrain parameter continuation 사전 고정
        - LOS active structure와 feasible successor count 기록
        - attacker switch/cost와 P4-style sensor optimum 기록
        - 관찰된 경우에만 structural transition 주장
        """,
        "experiment_p5_terrain_strategic_mechanism",
        "results/p5/p5_terrain_strategic_mechanism.json",
    )


def _p6():
    return _pending_notebook(
        "P6", "Verified Literature Positioning",
        "실제 원문을 확인해 terrain/LOS game, hybrid switching, execution-consistent graph, finite Stackelberg exactness의 차이를 표로 정리한다.",
        """
        - 모든 논문의 존재와 원문 claim 확인
        - method/geometry/switching/guarantee 비교표
        - 검증 전에는 first/novelty claim 금지
        - BibTeX와 source URL/DOI 저장
        """,
        "p6_verified_literature_positioning",
        "results/p6/p6_verified_literature_positioning.json",
    )


def _p7():
    return _pending_notebook(
        "P7", "One-Figure Qualitative 3D Example",
        "3D가 정식 theorem을 확장하는 단계는 아니다. 대표 terrain/sensor/trajectory 한 장으로 2D 구조의 확장 가능성만 보여준다.",
        """
        - paper-facing static figure 1장
        - representative terrain, sensor, trajectory 표시
        - 3D optimality/convergence/equilibrium claim 없음
        """,
        "experiment_p7_3d_qualitative_demo",
        "results/p7/p7_3d_qualitative_demo.json",
        extra_cells=[
            _code("""
            exploratory_dir = ROOT / "result_3D_visualization"
            summary_path = exploratory_dir / "summary.json"
            if summary_path.exists():
                display(Markdown("### 기존 exploratory 3D artifact (P7 최종 결과 아님)"))
                display_table(load_json(summary_path))
                for name in ("trajectory_3d.html", "occlusion_topdown.html"):
                    path = exploratory_dir / name
                    if path.exists():
                        display(FileLink(str(path)))
            else:
                display(Markdown("기존 exploratory 3D artifact가 없습니다."))
            """),
        ],
    )


NOTEBOOKS = {
    "B1_Nested_Discretization_Verification.ipynb": _b1,
    "B2_Two_Hill_Nested_Consistency.ipynb": _b2,
    "B3_Multi_Terrain_Extension.ipynb": _b3,
    "B4_Production_Lattice_Freeze.ipynb": _b4,
    "P1_Exact_Segment_Feasibility.ipynb": _p1,
    "P2_Mathematical_Proof_Package.ipynb": _p2,
    "P3_Randomized_Terrain_Counterexamples.ipynb": _p3,
    "P4_C_Lite_Exhaustive_Leader_Solve.ipynb": _p4,
    "P5_Terrain_Strategic_Mechanism.ipynb": _p5,
    "P6_Verified_Literature_Positioning.ipynb": _p6,
    "P7_3D_Qualitative_Demo.ipynb": _p7,
}


def build_all() -> tuple[Path, ...]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for filename, factory in NOTEBOOKS.items():
        path = OUTPUT_DIR / filename
        nbf.write(factory(), path)
        outputs.append(path)
    return tuple(outputs)


def execute_notebook(path: Path, timeout: int = 900) -> Path:
    """Execute one generated review notebook and persist every cell output."""
    notebook = nbf.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="venv_p1b",
        resources={"metadata": {"path": str(OUTPUT_DIR)}},
        allow_errors=False,
    )
    client.execute()
    nbf.write(notebook, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute", action="store_true",
        help="execute every generated notebook and save its cell outputs",
    )
    parser.add_argument(
        "--only", nargs="*", default=(),
        help="optional filename or stage prefix subset, for example B2 P1",
    )
    parser.add_argument("--timeout", type=int, default=900)
    arguments = parser.parse_args()
    paths = build_all()
    if arguments.only:
        requested = tuple(value.lower() for value in arguments.only)
        paths = tuple(
            path for path in paths
            if any(
                path.name.lower() == value
                or path.stem.lower().startswith(value)
                for value in requested
            )
        )
    for path in paths:
        if arguments.execute:
            execute_notebook(path, arguments.timeout)
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()

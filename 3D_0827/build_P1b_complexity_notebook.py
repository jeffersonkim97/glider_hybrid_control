"""Generate P1b_complexity.ipynb.

The notebook reads the measurements already on disk rather than re-running the
solvers: the 5 m Cartesian point alone is 5.9 hours, so a notebook that recomputed
its own figures could not be opened and run.  The one live section is pinned to the
100 m lattice, where a full local SSE is about 20 s for exact and 40 s for the
learned side, and it is there to show the API rather than to produce a figure.

Regenerate with ``python build_P1b_complexity_notebook.py``; edit the cells here,
not in the .ipynb.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "P1b_complexity.ipynb"


def _cells() -> list:
    cells = []

    cells.append(new_markdown_cell(
        "# P1b — exact local SSE 대 tabular Q-learning\n"
        "\n"
        "묻는 것: **어떤 계산 조건에서 exact local SSE 계산이 RL 근사보다 나은가.**\n"
        "알고리즘 우열이 아니라 조건별 선택 문제다.\n"
        "\n"
        "이 노트북은 디스크에 저장된 실측치를 읽어 표와 그림으로 정리한다. 솔버를\n"
        "다시 돌리지 않는다 — Cartesian 축의 5 m 한 점만 5.9시간이다.\n"
        "\n"
        "| 축 | 측정 대상 | 점 개수 |\n"
        "|---|---|---|\n"
        "| Cartesian `dx` | full local SSE (방어자 탐색 전체) | 6 |\n"
        "| heading `dpsi` | 고정 `d`에서의 best response 1회 | 4 |\n"
        "| Cartesian `dx` (참고) | 고정 `d`에서의 best response 1회 | 7 |\n"
        "\n"
        "두 축의 측정 대상이 다르다. 헤딩을 full SSE로 훑은 데이터가 없어서지,\n"
        "의도한 설계가 아니다. 조건 전체는 `r2_discretization_sweeps/README.md` 참조.\n"
    ))

    cells.append(new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "from matplotlib.ticker import (\n"
        "    FuncFormatter, LogLocator, MaxNLocator, NullFormatter,\n"
        ")\n"
        "\n"
        "from P1b_sweep_figures import SERIES, _power_law\n"
        "\n"
        "# P1b_sweep_figures pins the Agg backend so it can write PNGs headless;\n"
        "# undo that here or every figure below renders to nothing.\n"
        "%matplotlib inline\n"
        "\n"
        "DATA = Path('r2_discretization_sweeps/data')\n"
        "\n"
        "spatial_sse = json.loads((DATA / 'spatial_sweep_full_sse.json').read_text('utf-8'))\n"
        "heading = json.loads((DATA / 'heading_sweep_fixed_defender.json').read_text('utf-8'))\n"
        "spatial_fixed = json.loads(\n"
        "    (DATA / 'spatial_sweep_fixed_defender.json').read_text('utf-8')\n"
        ")\n"
        "\n"
        "print(f'{len(spatial_sse)} full-SSE points, {len(heading)} heading points, '\n"
        "      f'{len(spatial_fixed)} fixed-Defender points')\n"
    ))

    cells.append(new_markdown_cell(
        "## 1. 그림 — 이산화 축 대 계산 시간\n"
        "\n"
        "세 계열만 그린다. 적합 곡선은 넣지 않는다.\n"
        "\n"
        "- **Exact, total** — 네 단계 전부와 탐색 제어\n"
        "- **Q-learning, training** / **Q-learning, query** — 둘을 나누는 이유는\n"
        "  증가분이 거의 전부 training 쪽에 있고 query는 평평하기 때문이다.\n"
        "  합쳐 그리면 그 차이가 가려진다.\n"
        "\n"
        "exact 쪽은 Bellman 단계만이 아니라 **총합**이다. RL의 query가 같은 LOS·후보\n"
        "기하를 이미 포함하므로, 그 고정 비용을 한쪽에만 물리면 exact가 과소 표시된다.\n"
    ))

    cells.append(new_code_cell(
        "def axis_figure(records, key, title, x_title, *, log=True, ax=None):\n"
        "    \"\"\"Redraw one sweep axis inline, on the same terms as the saved PNG.\"\"\"\n"
        "    rows = sorted(records, key=lambda r: r[key])\n"
        "    x = np.asarray([r[key] for r in rows], dtype=float)\n"
        "    if ax is None:\n"
        "        _, ax = plt.subplots(figsize=(7.2, 4.6), dpi=110)\n"
        "    for name, field, colour, marker, dash in SERIES:\n"
        "        y = np.asarray([r[field] for r in rows], dtype=float)\n"
        "        ax.plot(x, y, marker=marker, linestyle=dash, color=colour,\n"
        "                markersize=6, linewidth=1.6, label=name)\n"
        "    if log:\n"
        "        ax.set_xscale('log'); ax.set_yscale('log')\n"
        "        ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in x])\n"
        "        ax.xaxis.set_minor_locator(LogLocator(subs='all'))\n"
        "        ax.xaxis.set_minor_formatter(NullFormatter())\n"
        "        ax.yaxis.set_major_locator(LogLocator(base=10.0))\n"
        "        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{v:g}'))\n"
        "        ax.yaxis.set_minor_formatter(NullFormatter())\n"
        "    else:\n"
        "        ax.set_xlim(0.0, float(x.max()) * 1.05); ax.set_ylim(bottom=0.0)\n"
        "        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))\n"
        "        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{v:g}'))\n"
        "        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{v:,.0f}'))\n"
        "    ax.set_xlabel(x_title); ax.set_ylabel('computation time [s]')\n"
        "    ax.set_title(title, fontsize=11)\n"
        "    ax.grid(True, which='major', color='#DDDDDD', linewidth=0.7)\n"
        "    ax.grid(True, which='minor', color='#F0F0F0', linewidth=0.5)\n"
        "    ax.set_axisbelow(True)\n"
        "    for side in ('top', 'right'):\n"
        "        ax.spines[side].set_visible(False)\n"
        "    ax.legend(loc='upper right', frameon=True, framealpha=0.9,\n"
        "              edgecolor='none', fontsize=9)\n"
        "    return ax\n"
    ))

    cells.append(new_code_cell(
        "figure, axes = plt.subplots(1, 2, figsize=(14.4, 4.6), dpi=110)\n"
        "axis_figure(spatial_sse, 'dx_m',\n"
        "            'Cartesian — full local SSE, r = 1',\n"
        "            'spatial step dx [m]  (finer to the left)', ax=axes[0])\n"
        "axis_figure(spatial_sse, 'dx_m',\n"
        "            'Cartesian — linear axes',\n"
        "            'spatial step dx [m]  (finer to the left)', log=False, ax=axes[1])\n"
        "figure.tight_layout()\n"
        "plt.show()\n"
    ))

    cells.append(new_code_cell(
        "figure, axes = plt.subplots(1, 2, figsize=(14.4, 4.6), dpi=110)\n"
        "axis_figure(heading, 'dpsi_deg',\n"
        "            'Heading — one best response at d = (5, 0, 0), dx = 12.5 m',\n"
        "            'heading step [deg]  (finer to the left)', ax=axes[0])\n"
        "axis_figure(heading, 'dpsi_deg',\n"
        "            'Heading — linear axes',\n"
        "            'heading step [deg]  (finer to the left)', log=False, ax=axes[1])\n"
        "figure.tight_layout()\n"
        "plt.show()\n"
    ))

    cells.append(new_markdown_cell(
        "## 2. Cartesian 축 — full local SSE 실측표\n"
    ))

    cells.append(new_code_cell(
        "head = (f\"{'dx':>7}{'exact':>10}{'Bellman':>10}{'ev':>4}{'it':>4}\"\n"
        "        f\"{'RL':>10}{'train':>9}{'query':>9}{'ev':>4}{'RL/exact':>10}\")\n"
        "print(head)\n"
        "print('-' * len(head))\n"
        "for r in sorted(spatial_sse, key=lambda r: -r['dx_m']):\n"
        "    print(f\"{r['dx_m']:>6g}m{r['exact_s']:>10.1f}{r['exact_stage3_bellman_s']:>10.1f}\"\n"
        "          f\"{r['exact_evaluations']:>4}{r['exact_iterations']:>4}\"\n"
        "          f\"{r['rl_s']:>10.1f}{r['rl_train_s']:>9.1f}{r['rl_query_s']:>9.1f}\"\n"
        "          f\"{r['rl_evaluations']:>4}{r['rl_s'] / r['exact_s']:>9.2f}x\")\n"
    ))

    cells.append(new_markdown_cell(
        "### 교차점\n"
        "\n"
        "exact가 RL보다 비싸지는 지점이 이 실험의 답이다. 아래 셀이 그 위치를\n"
        "실측 두 점 사이로 좁힌다.\n"
    ))

    cells.append(new_code_cell(
        "rows = sorted(spatial_sse, key=lambda r: -r['dx_m'])\n"
        "ratios = [(r['dx_m'], r['rl_s'] / r['exact_s']) for r in rows]\n"
        "for (coarse, a), (fine, b) in zip(ratios, ratios[1:]):\n"
        "    if (a - 1.0) * (b - 1.0) < 0:\n"
        "        print(f'crossing between dx = {coarse:g} m (RL/exact {a:.2f}x) '\n"
        "              f'and dx = {fine:g} m (RL/exact {b:.2f}x)')\n"
        "print()\n"
        "print('Bellman share of exact total:')\n"
        "for r in rows:\n"
        "    share = r['exact_stage3_bellman_s'] / r['exact_s']\n"
        "    print(f\"  dx = {r['dx_m']:>5g} m   {share:>6.1%}\"\n"
        "          f\"   fixed-cost stages {r['exact_s'] - r['exact_stage3_bellman_s']:>7.1f} s\")\n"
    ))

    cells.append(new_markdown_cell(
        "### 탐색 길이가 곱해지는 부분\n"
        "\n"
        "full SSE의 비용은 best response 1회 비용에 방어자 평가 횟수를 곱한 것이다.\n"
        "두 데이터셋을 나란히 두면 그 곱이 보인다 — 그리고 exact의 평가 횟수만\n"
        "해상도에 따라 늘어난다.\n"
    ))

    cells.append(new_code_cell(
        "fixed = {r['dx_m']: r for r in spatial_fixed}\n"
        "head = (f\"{'dx':>7}{'1 solve':>10}{'x ev':>6}{'= expected':>12}\"\n"
        "        f\"{'measured':>11}{'exact ev':>10}{'RL ev':>7}\")\n"
        "print(head)\n"
        "print('-' * len(head))\n"
        "for r in sorted(spatial_sse, key=lambda x: -x['dx_m']):\n"
        "    one = fixed.get(r['dx_m'])\n"
        "    if one is None:\n"
        "        continue\n"
        "    expected = one['exact_stage3_bellman_s'] * r['exact_evaluations']\n"
        "    print(f\"{r['dx_m']:>6g}m{one['exact_stage3_bellman_s']:>10.1f}\"\n"
        "          f\"{r['exact_evaluations']:>6}{expected:>12.1f}\"\n"
        "          f\"{r['exact_stage3_bellman_s']:>11.1f}\"\n"
        "          f\"{r['exact_evaluations']:>10}{r['rl_evaluations']:>7}\")\n"
    ))

    cells.append(new_markdown_cell(
        "## 3. Heading 축 — 고정 `d`, best response 1회\n"
        "\n"
        "`dpsi`를 절반으로 줄이면 `A`가 2배, `N_S`가 2배이므로 `N_S*A`는 4배가 된다.\n"
    ))

    cells.append(new_code_cell(
        "head = (f\"{'dpsi':>8}{'A':>5}{'N_S':>14}{'N_S*A':>17}{'RSS MiB':>9}\"\n"
        "        f\"{'exact':>9}{'Bellman':>9}{'RL':>9}{'train':>9}{'query':>8}{'acc':>8}\")\n"
        "print(head)\n"
        "print('-' * len(head))\n"
        "for r in sorted(heading, key=lambda r: -r['dpsi_deg']):\n"
        "    print(f\"{r['dpsi_deg']:>6g}dg{r['A']:>5}{r['N_S']:>14,}{r['N_S_times_A']:>17,}\"\n"
        "          f\"{r['rss_peak_MiB']:>9.0f}{r['exact_s']:>9.2f}\"\n"
        "          f\"{r['exact_stage3_bellman_s']:>9.2f}{r['rl_s']:>9.2f}\"\n"
        "          f\"{r['rl_train_s']:>9.2f}{r['rl_query_s']:>8.2f}\"\n"
        "          f\"{r['attacker_accuracy']:>8.4f}\")\n"
        "print()\n"
        "print('step-to-step growth, coarse -> fine:')\n"
        "rows = sorted(heading, key=lambda r: -r['dpsi_deg'])\n"
        "for a, b in zip(rows, rows[1:]):\n"
        "    print(f\"  {a['dpsi_deg']:>5g} -> {b['dpsi_deg']:<5g} deg\"\n"
        "          f\"   N_S*A x{b['N_S_times_A'] / a['N_S_times_A']:.2f}\"\n"
        "          f\"   Bellman x{b['exact_stage3_bellman_s'] / a['exact_stage3_bellman_s']:.2f}\"\n"
        "          f\"   RL train x{b['rl_train_s'] / a['rl_train_s']:.2f}\")\n"
    ))

    cells.append(new_markdown_cell(
        "### 정확도\n"
        "\n"
        "`attacker_accuracy`는 **같은 상대**에 대해서만 잰다. 서로 다른 `d`에서 나온\n"
        "`J_A`를 비교하면 값이 뒤집혀 의미를 잃는다.\n"
        "\n"
        "```\n"
        "attacker_accuracy = J_A_exact(학습자가 고른 d에서) / J_A_learned   in (0, 1]\n"
        "defender_accuracy = true J_D(학습자의 d에서)      / J_D_exact(d*)  in (0, 1]\n"
        "```\n"
        "\n"
        "`RL_threshold = 0.9`이고 두 값 모두 넘겨야 통과다.\n"
    ))

    cells.append(new_code_cell(
        "RL_THRESHOLD = 0.9\n"
        "values = [r['attacker_accuracy'] for r in heading]\n"
        "print(f'heading axis attacker_accuracy: min {min(values):.4f}  max {max(values):.4f}'\n"
        "      f'  -> threshold {RL_THRESHOLD} ' + ('met at every step' if min(values) >= RL_THRESHOLD\n"
        "                                          else 'MISSED somewhere'))\n"
        "print()\n"
        "print('full SSE — did the learner land on the same d* as exact?')\n"
        "for r in sorted(spatial_sse, key=lambda r: -r['dx_m']):\n"
        "    same = r['exact_sensor_map'] == r['rl_sensor_map']\n"
        "    ratio = r['rl_J_D'] / r['exact_J_D']\n"
        "    print(f\"  dx = {r['dx_m']:>5g} m   exact d* {tuple(round(v, 3) for v in r['exact_sensor_map'][:2])}\"\n"
        "          f\"   RL d* {tuple(round(v, 3) for v in r['rl_sensor_map'][:2])}\"\n"
        "          f\"   {'same' if same else 'differs'}   believed J_D ratio {ratio:.4f}\")\n"
    ))

    cells.append(new_markdown_cell(
        "## 4. 살아 있는 실행 — 100 m 격자 한 번\n"
        "\n"
        "API 확인용이다. 100 m에서 full local SSE는 exact 약 20 s, RL 약 40 s다.\n"
        "**더 세밀한 격자로 바꾸지 말 것** — 5 m는 exact 5.9시간, RL 1.4시간이다.\n"
    ))

    cells.append(new_code_cell(
        "from time import perf_counter\n"
        "\n"
        "from P1b_condition import ComputationCondition, build_scene\n"
        "from P1b_Exact_Local_SSE import exact_local_sse\n"
        "from P1b_RL_approximation import QLearningConfig, rl_local_sse\n"
        "\n"
        "CONDITION = ComputationCondition(\n"
        "    spatial_resolution_m=100.0,   # dx = dy = dh\n"
        "    heading_spacing_deg=5.0,\n"
        "    r_neighbor=1,                 # Chebyshev radius on the Defender grid\n"
        "    terrain_category='centered_cube_half_height',\n"
        ")\n"
        "CONFIG = QLearningConfig(episodes=20000, seed=0)\n"
        "\n"
        "scene = build_scene(CONDITION)\n"
        "print(CONDITION.label)\n"
        "for key, value in scene.sizes().items():\n"
        "    print(f'  {key:<24} {value:,}' if isinstance(value, int)\n"
        "          else f'  {key:<24} {value}')\n"
    ))

    cells.append(new_code_cell(
        "started = perf_counter()\n"
        "exact = exact_local_sse(scene)\n"
        "exact_wall = perf_counter() - started\n"
        "\n"
        "started = perf_counter()\n"
        "learned = rl_local_sse(scene, CONFIG, warm_start_episodes=4000)\n"
        "learned_wall = perf_counter() - started\n"
        "\n"
        "print(f\"{'':<26}{'exact':>26}{'Q-learning':>26}\")\n"
        "print('-' * 78)\n"
        "rows = [\n"
        "    ('wall [s]', exact_wall, learned_wall),\n"
        "    ('unique evaluations', exact.unique_evaluations, learned.unique_evaluations),\n"
        "    ('search iterations', exact.iterations, learned.iterations),\n"
        "    ('selected action id', exact.selected_action_id, learned.selected_action_id),\n"
        "    ('J_D', exact.detection_probability, learned.detection_probability),\n"
        "    ('J_A', exact.attacker_objective, learned.attacker_objective),\n"
        "    ('termination', exact.termination_status, learned.termination_status),\n"
        "]\n"
        "for label, a, b in rows:\n"
        "    fa = f'{a:.6f}' if isinstance(a, float) else str(a)\n"
        "    fb = f'{b:.6f}' if isinstance(b, float) else str(b)\n"
        "    print(f'{label:<26}{fa:>26}{fb:>26}')\n"
        "print('-' * 78)\n"
        "print('same d*:', exact.selected_action_id == learned.selected_action_id)\n"
        "print()\n"
        "print('exact timing :', {k: round(v, 2) for k, v in exact.timing.items()})\n"
        "print('learned timing:', {k: round(v, 2) for k, v in learned.timing.items()})\n"
        "print('local_sse_verified:', learned.local_sse_verified)\n"
    ))

    cells.append(new_markdown_cell(
        "## 해석 경계\n"
        "\n"
        "- **두 축의 측정 대상이 다르다.** Cartesian은 방어자 탐색 전체, heading은\n"
        "  고정 `d`에서의 1회 해다. 두 그림의 y값을 서로 비교하면 안 된다.\n"
        "- **반복 1회, 시드 1개.** 분산 추정치가 없다. 두 계열이 가까운 구간\n"
        "  (5 m 부근, 1.25 deg)의 순서는 재현성이 확인되지 않았다.\n"
        "- **Cartesian의 exact 시간은 backfill 재실행분**이고 RL 시간은 원 실행분이다.\n"
        "  exact 솔버는 결정론적이고 목적값이 일치함을 확인한 뒤 기록했지만, 벽시계\n"
        "  시간은 실행마다 -7.7% ~ +5.7% 차이가 났다.\n"
        "- **`r` 축 그림은 없다.** 부분 데이터만 있고 고정 `dx`에서의 계열이 없다.\n"
        "- **RL 결과에 exactness 인증을 붙이지 않는다** (`local_sse_verified=False`).\n"
        "  근사해이고, 그렇게 보고한다.\n"
        "- **지형 1개에서만 측정했다** (`centered_cube_half_height`).\n"
        "- 찾은 `d*`는 전부 x = 4.75 ~ 5.50 구간에 있다. 50 m에서 따로 재 본\n"
        "  d = (7, 0)의 `J_D`는 0.216615로 (5, 0)의 0.025019보다 8.7배 크다. 즉 여기\n"
        "  보고된 `J_D`는 r = 1 등반이 벗어나지 못한 이웃의 값이지, 영역 최적이 아니다.\n"
    ))

    return cells


def resolve_kernel_name(preferred: str = "venv_p1b") -> str:
    """Pick an installed kernel rather than assuming one is registered.

    The project venv ships its own ``python3`` kernelspec, so fall back to that
    instead of writing a new kernelspec into the user's Jupyter configuration.
    """
    try:
        from jupyter_client.kernelspec import KernelSpecManager
    except ImportError:
        return "python3"

    installed = set(KernelSpecManager().find_kernel_specs())
    for candidate in (preferred, "python3"):
        if candidate in installed:
            return candidate
    return "python3"


def build_notebook(path: Path = NOTEBOOK) -> Path:
    kernel = resolve_kernel_name()
    notebook = new_notebook(cells=_cells())
    notebook.metadata.update({
        "kernelspec": {"display_name": kernel, "language": "python", "name": kernel},
        "language_info": {"name": "python"},
    })
    nbformat.write(notebook, path)
    return path


if __name__ == "__main__":
    written = build_notebook()
    print(f"wrote {written.name} ({len(_cells())} cells)")

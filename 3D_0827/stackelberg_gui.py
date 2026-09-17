"""Standalone browser GUI for resolution-controlled 3D Stackelberg runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock, Thread
from typing import Any
import webbrowser

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from defender_grid_config import CANONICAL_DEFENDER_X_MAP
from discretization_config import (
    DiscretizationConfig,
    discretization_from_physical_steps,
)
from stage11_config import Stage11Config
from stackelberg_solver import (
    FiniteStackelbergRun,
    generate_defender_line_actions,
    run_finite_stackelberg,
)
from stackelberg_validation import validate_selected_stackelberg_trajectory
from terrain_catalog import build_terrain
from visualization import plot_finite_stackelberg_solution


HORIZONTAL_OPTIONS_M = (10, 20, 25, 40, 50, 80, 100)
ALTITUDE_OPTIONS_M = (10, 20, 25, 50, 100)
HEADING_OPTIONS_DEG = (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 45)
DEFAULT_GUI_RESOLUTION = (25.0, 25.0, 5.0)
DEFAULT_DEFENDER_X = CANONICAL_DEFENDER_X_MAP
DEFAULT_MAX_CARTESIAN_STATES = 8_000_000


@dataclass(frozen=True)
class ResolutionEstimate:
    horizontal_step_m: float
    altitude_step_m: float
    heading_step_deg: float
    heading_bins: int
    motion_primitive_radius: int
    cartesian_state_count: int


@dataclass(frozen=True)
class InteractiveStackelbergResult:
    config: Stage11Config
    run: FiniteStackelbergRun
    validation_passed: bool
    figure: go.Figure
    summary: dict[str, Any]


def resolution_estimate(
    horizontal_step_m: float,
    altitude_step_m: float,
    heading_step_deg: float,
    *,
    base_config: Stage11Config | None = None,
) -> tuple[DiscretizationConfig, ResolutionEstimate]:
    config = base_config or Stage11Config()
    discretization = discretization_from_physical_steps(
        horizontal_step_m,
        altitude_step_m,
        heading_step_deg,
        meters_per_map_unit=config.physical_scale.meters_per_map_unit,
        template=config.discretization,
    )
    grid = discretization.build_bellman_grid(config.graph_bounds)
    return discretization, ResolutionEstimate(
        horizontal_step_m=float(horizontal_step_m),
        altitude_step_m=float(altitude_step_m),
        heading_step_deg=float(heading_step_deg),
        heading_bins=grid.heading_bin_count,
        motion_primitive_radius=grid.motion_primitive_radius,
        cartesian_state_count=grid.state_count,
    )


def _attacker_kwargs(config: Stage11Config) -> dict[str, object]:
    discretization = config.discretization
    return {
        "bellman_grid": discretization.build_bellman_grid(config.graph_bounds),
        "contour_sample_count": discretization.switching_contour_sample_count,
        "radial_scales": discretization.radial_scales,
        "quadrature_resolution": discretization.hazard_quadrature_resolution,
        "los_probe_grid_size": discretization.los_probe_grid_size,
        "los_boundary_refinement_steps": (
            discretization.los_boundary_refinement_steps
        ),
        "los_display_extension_factor": (
            discretization.los_display_extension_factor
        ),
        "visualization_ray_count": discretization.visualization_ray_count,
        "parameters": config.glider,
        "physical_scale": config.physical_scale,
        "detection_parameters": config.detection,
        "objective_parameters": config.attacker_objective,
        "bellman_backend": "goal_backward_sparse",
    }


def _add_reachable_set_trace(
    figure: go.Figure,
    run: FiniteStackelbergRun,
    *,
    maximum_points: int = 60_000,
) -> None:
    attacker_run = run.selected_evaluation.attacker_run
    grid = attacker_run.graph.grid
    counts = attacker_run.unit_reachability.goal_reachable.reshape((
        grid.altitude_count,
        grid.y_count,
        grid.x_count,
        grid.heading_bin_count,
    )).sum(axis=3)
    indices = np.argwhere(counts > 0)
    if not len(indices):
        return
    stride = max(1, int(np.ceil(len(indices) / maximum_points)))
    selected = indices[::stride]
    selected_counts = counts[
        selected[:, 0], selected[:, 1], selected[:, 2]
    ]
    figure.add_trace(go.Scatter3d(
        x=grid.bounds.x_min + selected[:, 2] * grid.horizontal_spacing_map,
        y=grid.bounds.y_min + selected[:, 1] * grid.horizontal_spacing_map,
        z=grid.minimum_altitude_map + selected[:, 0] * grid.altitude_spacing_map,
        mode="markers",
        name=f"Goal-backward reachable set ({len(indices):,} positions)",
        marker={
            "size": 2.1,
            "color": selected_counts,
            "colorscale": "Viridis",
            "opacity": 0.18,
            "colorbar": {"title": "reachable<br>headings", "x": 1.08},
        },
        customdata=selected_counts[:, None],
        hovertemplate=(
            "Goal-reachable lattice position<br>x=%{x:.2f}<br>y=%{y:.2f}"
            "<br>z=%{z:.2f}<br>reachable headings=%{customdata[0]}"
            "<extra></extra>"
        ),
    ))


def run_resolution_case(
    horizontal_step_m: float = DEFAULT_GUI_RESOLUTION[0],
    altitude_step_m: float = DEFAULT_GUI_RESOLUTION[1],
    heading_step_deg: float = DEFAULT_GUI_RESOLUTION[2],
    *,
    base_config: Stage11Config | None = None,
    defender_x: tuple[float, ...] = DEFAULT_DEFENDER_X,
    maximum_cartesian_states: int = DEFAULT_MAX_CARTESIAN_STATES,
) -> InteractiveStackelbergResult:
    base = base_config or Stage11Config()
    discretization, estimate = resolution_estimate(
        horizontal_step_m,
        altitude_step_m,
        heading_step_deg,
        base_config=base,
    )
    if estimate.cartesian_state_count > int(maximum_cartesian_states):
        raise ValueError(
            f"requested grid has {estimate.cartesian_state_count:,} Cartesian "
            f"states; interactive safety limit is {int(maximum_cartesian_states):,}. "
            "Choose a coarser combination or raise the limit explicitly."
        )
    config = replace(base, discretization=discretization)
    terrain = build_terrain(config.terrain_category)
    run = run_finite_stackelberg(
        generate_defender_line_actions(defender_x),
        config.attacker_initial_condition,
        terrain=terrain,
        attacker_kwargs=_attacker_kwargs(config),
        reuse_reachability_graph=True,
    )
    audit = validate_selected_stackelberg_trajectory(run, config)
    if not audit.report.passed:
        raise RuntimeError("selected Stackelberg trajectory failed independent replay")
    figure = plot_finite_stackelberg_solution(run)
    _add_reachable_set_trace(figure, run)
    selected = run.selected_evaluation
    follower = selected.sse_follower_result
    if follower is None:
        raise RuntimeError("selected defender action has no follower result")
    grid = selected.attacker_run.graph.grid
    target_angles = np.arange(grid.heading_bin_count) * 360.0 / grid.heading_bin_count
    realized = np.degrees(np.mod([
        grid.heading_rad(index) for index in range(grid.heading_bin_count)
    ], 2.0 * np.pi))
    angle_error = np.abs((realized - target_angles + 180.0) % 360.0 - 180.0)
    summary = {
        "horizontal_step_m": estimate.horizontal_step_m,
        "altitude_step_m": estimate.altitude_step_m,
        "requested_heading_step_deg": estimate.heading_step_deg,
        "heading_bins": estimate.heading_bins,
        "maximum_realized_heading_error_deg": float(np.max(angle_error)),
        "motion_primitive_radius_cells": estimate.motion_primitive_radius,
        "cartesian_state_count": estimate.cartesian_state_count,
        "goal_reachable_state_count": selected.attacker_run.metrics.goal_reachable_state_count,
        "active_corridor_state_count": selected.attacker_run.metrics.active_state_count,
        "active_edge_count": selected.attacker_run.metrics.hazard_edge_count,
        "selected_defender_action_id": selected.candidate.action_id,
        "selected_sensor_position_map": (
            selected.candidate.action.sensor_position_map.tolist()
        ),
        "selected_attacker_candidate_id": follower.candidate_id,
        "attacker_objective": run.outcome.attacker_payoff,
        "defender_payoff_pod": run.outcome.defender_payoff,
        "validation_passed": True,
        "total_stackelberg_s": run.timing.total_s,
        "shared_graph_build_s": run.timing.shared_graph_build_s,
    }
    figure.update_layout(title={
        "text": (
            "Interactive 3D Stackelberg / Reachable-Set Result"
            f"<br><sup>{horizontal_step_m:g} m horizontal, "
            f"{altitude_step_m:g} m altitude, {heading_step_deg:g} deg heading; "
            f"D{selected.candidate.action_id}, A{follower.candidate_id}</sup>"
        ),
        "x": 0.5,
    }, legend={
        "orientation": "h",
        "x": 0.5,
        "xanchor": "center",
        "y": -0.16,
        "yanchor": "top",
        "font": {"size": 10},
        "bgcolor": "rgba(255,255,255,0.88)",
    }, margin={"l": 0, "r": 0, "t": 80, "b": 150})
    return InteractiveStackelbergResult(
        config=config,
        run=run,
        validation_passed=True,
        figure=figure,
        summary=summary,
    )


def _html() -> str:
    horizontal = json.dumps(HORIZONTAL_OPTIONS_M)
    altitude = json.dumps(ALTITUDE_OPTIONS_M)
    heading = json.dumps(HEADING_OPTIONS_DEG)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>3D Stackelberg Explorer</title>
<script src="/plotly.js"></script>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f7fb;color:#18212f}}
header{{padding:14px 22px;background:#172a46;color:white}}
.controls{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr)) auto;gap:18px;align-items:end;padding:14px 22px;background:white;box-shadow:0 2px 8px #0002}}
.controls label{{font-weight:650}} .controls input{{width:100%}} .value{{color:#2563eb;font-weight:700}}
button{{padding:11px 22px;border:0;border-radius:7px;background:#1769aa;color:white;font-weight:700;cursor:pointer}}
#status{{padding:10px 22px;background:#eaf1fb}} #summary{{padding:10px 22px;white-space:pre-wrap;font-family:Consolas,monospace}}
.workspace{{display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:0;align-items:stretch;background:white}}
#plot{{width:100%;height:72vh;min-width:0}}
.layers{{height:72vh;box-sizing:border-box;border-left:1px solid #d8e0ea;padding:12px;background:#fbfcfe;overflow:auto}}
.layers h3{{margin:0 0 8px}} .layer-actions{{display:flex;gap:7px;margin-bottom:10px}}
.layer-actions button{{padding:6px 9px;font-size:12px;background:#52677f}}
.layer-item{{display:flex;align-items:flex-start;gap:8px;padding:6px 3px;border-bottom:1px solid #edf1f5;font-size:13px;line-height:1.25}}
.layer-item input{{width:auto;margin-top:2px}} .layer-count{{color:#718096;font-size:11px}}
.error{{color:#a61b1b;background:#feecec}}
@media(max-width:900px){{.controls{{grid-template-columns:1fr 1fr}}.workspace{{grid-template-columns:1fr}}.layers{{height:auto;max-height:280px;border-left:0;border-top:1px solid #d8e0ea}}}}
</style></head><body>
<header><h2 style="margin:0">3D Total-Energy / Stackelberg Explorer</h2></header>
<div class="controls">
<label>Horizontal discretization: <span id="hv" class="value"></span><input id="h" type="range"></label>
<label>Altitude discretization: <span id="zv" class="value"></span><input id="z" type="range"></label>
<label>Heading discretization: <span id="av" class="value"></span><input id="a" type="range"></label>
<button id="run">Run exact finite game</button></div>
<div id="status">Ready.</div><div id="summary"></div>
<div class="workspace"><div id="plot"></div><aside class="layers">
<h3>Layers</h3><div class="layer-actions"><button id="show-all">Show all</button><button id="hide-all">Hide all</button></div>
<div id="layer-list"></div></aside></div>
<script>
const opts={{h:{horizontal},z:{altitude},a:{heading}}};
function setup(id, values, initial, suffix){{const e=document.getElementById(id);e.min=0;e.max=values.length-1;e.step=1;e.value=values.indexOf(initial);const update=()=>document.getElementById(id+'v').textContent=values[+e.value]+suffix;e.oninput=update;update();}}
setup('h',opts.h,25,' m');setup('z',opts.z,25,' m');setup('a',opts.a,5,' deg');
let layerGroups=[];
function buildLayerControls(data){{
  const grouped=new Map();
  data.forEach((trace,index)=>{{
    const name=trace.name||`${{trace.type||'trace'}} ${{index+1}}`;
    if(!grouped.has(name))grouped.set(name,[]);
    grouped.get(name).push(index);
  }});
  layerGroups=[...grouped.entries()].map(([name,indices])=>({{name,indices}}));
  const list=document.getElementById('layer-list');list.replaceChildren();
  layerGroups.forEach((group,groupIndex)=>{{
    const row=document.createElement('label');row.className='layer-item';
    const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.dataset.group=groupIndex;
    checkbox.checked=group.indices.some(index=>data[index].visible!==false&&data[index].visible!=='legendonly');
    const text=document.createElement('span');text.textContent=group.name;
    if(group.indices.length>1){{const count=document.createElement('span');count.className='layer-count';count.textContent=` (${{group.indices.length}} traces)`;text.appendChild(count);}}
    checkbox.onchange=()=>Plotly.restyle('plot',{{visible:checkbox.checked}},group.indices);
    row.append(checkbox,text);list.appendChild(row);
  }});
}}
function setAllLayers(visible){{
  document.querySelectorAll('#layer-list input').forEach(item=>item.checked=visible);
  const indices=layerGroups.flatMap(group=>group.indices);
  if(indices.length)Plotly.restyle('plot',{{visible}},indices);
}}
document.getElementById('show-all').onclick=()=>setAllLayers(true);
document.getElementById('hide-all').onclick=()=>setAllLayers(false);
async function solve(){{const status=document.getElementById('status');status.className='';status.textContent='Computing exact goal-backward set and Stackelberg equilibrium...';document.getElementById('run').disabled=true;
try{{const payload={{horizontal_step_m:opts.h[+h.value],altitude_step_m:opts.z[+z.value],heading_step_deg:opts.a[+a.value]}};const response=await fetch('/solve',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});const result=await response.json();if(!response.ok)throw new Error(result.error||'solve failed');result.figure.layout.showlegend=false;result.figure.layout.margin=Object.assign({{}},result.figure.layout.margin||{{}},{{r:15,b:15}});await Plotly.react('plot',result.figure.data,result.figure.layout,{{responsive:true,displaylogo:false}});buildLayerControls(result.figure.data);document.getElementById('summary').textContent=JSON.stringify(result.summary,null,2);status.textContent='PASS - independent replay validated.';}}
catch(error){{status.className='error';status.textContent=error.message;}}finally{{document.getElementById('run').disabled=false;}}}}
document.getElementById('run').onclick=solve;solve();
</script></body></html>"""


class StackelbergGUIServer:
    """Handle for the daemon HTTP server opened by the notebook's last cell."""

    def __init__(
        self,
        *,
        initial_result: InteractiveStackelbergResult | None = None,
        maximum_cartesian_states: int = DEFAULT_MAX_CARTESIAN_STATES,
    ) -> None:
        self._lock = Lock()
        self._last_key: tuple[float, float, float] | None = None
        self._last_result = initial_result
        if initial_result is not None:
            summary = initial_result.summary
            self._last_key = (
                float(summary["horizontal_step_m"]),
                float(summary["altitude_step_m"]),
                float(summary["requested_heading_step_deg"]),
            )
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/":
                    self._send(200, "text/html; charset=utf-8", _html().encode())
                elif self.path == "/plotly.js":
                    self._send(
                        200,
                        "application/javascript; charset=utf-8",
                        get_plotlyjs().encode(),
                    )
                elif self.path == "/health":
                    self._send(200, "application/json", b'{"status":"ok"}')
                else:
                    self._send(404, "text/plain", b"not found")

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/solve":
                    self._send(404, "text/plain", b"not found")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    request = json.loads(self.rfile.read(length))
                    key = (
                        float(request["horizontal_step_m"]),
                        float(request["altitude_step_m"]),
                        float(request["heading_step_deg"]),
                    )
                    with owner._lock:
                        if owner._last_result is None or owner._last_key != key:
                            owner._last_result = run_resolution_case(
                                *key,
                                maximum_cartesian_states=maximum_cartesian_states,
                            )
                            owner._last_key = key
                        result = owner._last_result
                    payload = json.dumps({
                        "summary": result.summary,
                        "figure": json.loads(result.figure.to_json()),
                    }, allow_nan=False).encode()
                    self._send(200, "application/json", payload)
                except Exception as error:  # browser needs a structured failure
                    payload = json.dumps({"error": str(error)}).encode()
                    self._send(400, "application/json", payload)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/"

    def open_new_window(self) -> bool:
        return bool(webbrowser.open_new(self.url))

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


def launch_stackelberg_gui(
    *,
    initial_result: InteractiveStackelbergResult | None = None,
    open_browser: bool = True,
    maximum_cartesian_states: int = DEFAULT_MAX_CARTESIAN_STATES,
) -> StackelbergGUIServer:
    server = StackelbergGUIServer(
        initial_result=initial_result,
        maximum_cartesian_states=maximum_cartesian_states,
    )
    if open_browser:
        server.open_new_window()
    return server


__all__ = [
    "ALTITUDE_OPTIONS_M",
    "DEFAULT_GUI_RESOLUTION",
    "HEADING_OPTIONS_DEG",
    "HORIZONTAL_OPTIONS_M",
    "InteractiveStackelbergResult",
    "ResolutionEstimate",
    "StackelbergGUIServer",
    "launch_stackelberg_gui",
    "resolution_estimate",
    "run_resolution_case",
]

# P1b CasADi Environment Setup

Sets up everything needed to open and run `p1b/p1b_casadi_symbolic.ipynb`
(and the rest of the `p1b_4d_dp.py` / `p1b_main.ipynb` pipeline) on a fresh
clone of this repo. Run this once per machine, from the repo root.

**Instructions for Claude Code**: detect the current OS from your own
environment info (already stated in your system prompt — no need to probe
for it) and run only the matching block below. Do not run both.

## What this does

1. Checks whether `uv` (the Python package/env manager this project uses)
   is installed; installs it if missing.
2. Creates a virtual environment at `.venv_p1b`. `uv venv` reads
   `.python-version` (pinned to `3.14.6`) automatically and downloads that
   exact Python interpreter if it isn't already available on the machine —
   no separate "install Python" step is needed.
3. Installs the pinned dependencies from `p1b/requirements.txt` (casadi,
   numpy, scipy, matplotlib, nbformat/nbclient/ipykernel) into that venv.
4. Prints a final check importing `casadi`/`numpy`/`scipy` to confirm the
   environment is actually usable before declaring success.

Safe to re-run: each step checks whether it's already done before acting.

---

## Windows (PowerShell)

```powershell
# 1. Install uv if missing
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv --version

# 2. Create the venv (auto-downloads Python 3.14.6 per .python-version if needed)
uv venv .venv_p1b

# 3. Install pinned dependencies
uv pip install --python .venv_p1b\Scripts\python.exe -r p1b\requirements.txt

# 4. Sanity check
.venv_p1b\Scripts\python.exe -c "import casadi, numpy, scipy, matplotlib, nbformat, nbclient; print('OK:', casadi.__version__, numpy.__version__, scipy.__version__)"
```

To run the notebook headlessly afterward: `.venv_p1b\Scripts\python.exe execute_casadi_nb.py`
To open it interactively: open `p1b/p1b_casadi_symbolic.ipynb` in VS Code, then
**Select Kernel → Python Environments → `.venv_p1b`** (the notebook's saved
kernel name won't exist on a new machine — pick the interpreter directly).

---

## Linux / macOS (bash)

```bash
# 1. Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# 2. Create the venv (auto-downloads Python 3.14.6 per .python-version if needed)
uv venv .venv_p1b

# 3. Install pinned dependencies
uv pip install --python .venv_p1b/bin/python -r p1b/requirements.txt

# 4. Sanity check
.venv_p1b/bin/python -c "import casadi, numpy, scipy, matplotlib, nbformat, nbclient; print('OK:', casadi.__version__, numpy.__version__, scipy.__version__)"
```

To run the notebook headlessly afterward: `.venv_p1b/bin/python execute_casadi_nb.py`
To open it interactively: open `p1b/p1b_casadi_symbolic.ipynb` in VS Code, then
**Select Kernel → Python Environments → `.venv_p1b`** (the notebook's saved
kernel name won't exist on a new machine — pick the interpreter directly).

---

## Data dependencies

Already committed to the repo, no separate download needed:
`p1b/data/params.json`, `p1b/data/costmap_2d.json`, `p1b/data/terrain.json`,
`p1b/data/p1b_4d_dp_results.npz`. These are frozen numeric baselines the
notebook validates its symbolic reconstruction against — if they ever need
regenerating, `params.json`/`costmap_2d.json`/`terrain.json` come from
running `p1b_main.ipynb` Steps 1-6, and `p1b_4d_dp_results.npz` comes from
running `p1b/p1b_4d_dp.py`.

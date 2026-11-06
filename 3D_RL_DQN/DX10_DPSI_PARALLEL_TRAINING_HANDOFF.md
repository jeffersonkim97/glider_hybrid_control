# DX=10 m Remote Training and Evaluation Handoff

## Purpose

Delegate the entire non-overlapping `dx=10 m` block to another CUDA machine.
The primary machine continues only `dx={100,50,25}`. The remote machine:

1. trains the missing `dpsi={2.5,10}` DQN checkpoints;
2. reuses the existing `dpsi=5` checkpoint;
3. evaluates the 14 missing `(dpsi,r)` conditions on four unseen terrains;
4. packages all 56 raw result files;
5. commits the checkpoints and result bundle with Git LFS.

## Exact work split

| dpsi [deg] | r values evaluated remotely | Conditions | Terrain runs |
|---:|---|---:|---:|
| 2.5 | 1, 2, 3, 4, 5 | 5 | 20 |
| 5 | 2, 3, 4, 5 | 4 | 16 |
| 10 | 1, 2, 3, 4, 5 | 5 | 20 |
| **Total** |  | **14** | **56** |

`(dx=10,dpsi=5,r=1)` already exists and is not repeated.

## 1. Pull the prepared worker

```powershell
git clone https://github.com/jeffersonkim97/glider_hybrid_control.git
cd glider_hybrid_control
git lfs install
git pull origin main
git lfs pull
```

Recreate the CUDA-enabled `.venv_p1b` environment used by this repository.
Verify CUDA before starting:

```powershell
.\.venv_p1b\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The second printed value must be `True`.

## 2. Create a remote experiment branch

```powershell
git switch -c exp/dx10-remote-training-evaluation
```

## 3. Verify the delegated plan

```powershell
.\.venv_p1b\Scripts\python.exe `
  .\3D_RL_DQN\phase11_dx10_remote_worker.py --plan
```

The plan must report:

- full condition count: `15`
- pre-existing condition count: `1`
- missing condition count: `14`
- pending evaluation count: `56`
- missing checkpoint count: `2`

The missing checkpoints must be exactly:

- `dx10m_dpsi2p5deg/seed0_ep10000_v1/phase4_seed0_final.pt`
- `dx10m_dpsi10deg/seed0_ep10000_v1/phase4_seed0_final.pt`

## 4. Run training, evaluation, and packaging

```powershell
.\.venv_p1b\Scripts\python.exe `
  .\3D_RL_DQN\phase11_dx10_remote_worker.py
```

This single resumable command performs all remote work. Execution order:

1. train `dx=10,dpsi=10` if its checkpoint is missing;
2. reuse the tracked `dx=10,dpsi=5` checkpoint;
3. train `dx=10,dpsi=2.5` if its checkpoint is missing;
4. for each `(terrain,dpsi)`, evaluate `r=5 -> 4 -> 3 -> 2 -> 1`;
5. reuse exact attacker-BR cache across `r` values;
6. package the 56 completed raw results.

Interrupted runs can be resumed by executing the same command again. Completed
checkpoints, result files, and exact-response caches are reused.

## 5. Expected outputs

New checkpoints:

```text
3D_RL_DQN/checkpoints/phase4_extended_random_boxes/training_v2/g2_generalization/
  dx10m_dpsi2p5deg/seed0_ep10000_v1/phase4_seed0_final.pt
  dx10m_dpsi10deg/seed0_ep10000_v1/phase4_seed0_final.pt
```

Git-LFS result artifact:

```text
3D_RL_DQN/artifacts/phase11_dx10_remote/
  phase11_dx10_remote_results.zip
  phase11_dx10_remote_manifest.json
```

The ZIP contains only completed `zero_sum_g2_local_search.json`/Markdown
results and the completion status. Large duplicate exact-response caches and
console logs are not included. The manifest records the ZIP SHA-256 and every
raw-result SHA-256.

## 6. Validate before committing

```powershell
.\.venv_p1b\Scripts\python.exe `
  .\3D_RL_DQN\phase11_dx10_remote_worker.py --package-only

git lfs ls-files | Select-String 'dx10m_dpsi(2p5|10)deg'
```

`--package-only` must report 56 raw results. Both new `.pt` files must be
present. The artifact ZIP is covered by the repository's Git LFS rules.

## 7. Commit and push the remote outputs

`3D_RL_DQN/checkpoints/` is ignored by default, so the two new directories
must be force-added. Do not force-add the ignored `runs/` directory.

```powershell
git add -f 3D_RL_DQN/checkpoints/phase4_extended_random_boxes/training_v2/g2_generalization/dx10m_dpsi2p5deg
git add -f 3D_RL_DQN/checkpoints/phase4_extended_random_boxes/training_v2/g2_generalization/dx10m_dpsi10deg
git add 3D_RL_DQN/artifacts/phase11_dx10_remote

git status --short
git commit -m "Add remote dx10 training and full r-sweep results"
git push -u origin exp/dx10-remote-training-evaluation
```

## 8. Import on the primary machine

After the remote branch has been pushed, merge it into a clean primary-machine
worktree:

```powershell
git fetch origin
git merge --no-ff origin/exp/dx10-remote-training-evaluation
git lfs pull
```

Then verify and restore the raw results:

```powershell
.\.venv_p1b\Scripts\python.exe `
  .\3D_RL_DQN\phase11_dx10_remote_worker.py --import-artifact
```

The importer verifies the artifact SHA-256, blocks unsafe archive paths,
extracts 56 raw results into `runs/phase11_dx10_remote`, and verifies each
result hash. These results can then be merged with the 46 existing/local
conditions to produce the complete 60-condition table.

## 9. Reproducibility contract

The remote checkpoint payload records:

- seed and completed episode;
- training/network/execution configuration;
- spatial and angular discretization;
- action-catalog compatibility signature;
- dataset manifest SHA-256;
- model and optimizer state.

Use the same Git commit, dataset, Python/PyTorch/CUDA environment, seed, and
episode count. Identical GPU hardware improves timing comparability but does
not guarantee bitwise-identical CUDA training.

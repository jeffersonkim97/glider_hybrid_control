# Stage 14 reproducibility record

Run from the repository root with the project virtual environment.

```powershell
.\.venv_p1b\Scripts\python.exe 3D_0827\stage14_8_final_gate.py
```

The command performs or resumes exactly one canonical fresh-process smoke run, rebuilds Stage-14 figures from saved machine-readable artifacts, runs all `test_stage14*.py` tests, and executes the integrated notebook. It does not run the full Stage-14.3--14.6 sweep again.

Primary outputs are `stage14_8_summary.json`, `stage14_8_validation_report.json`, `all_raw_repetitions.{json,jsonl,csv}`, `all_configuration_summaries.{json,csv}`, `frozen_benchmark_configuration_set.json`, `figure_manifest.json`, `artifact_manifest.json`, and `command_record.json`.

Recorded final gate: PASS.
The certified equilibrium remains an exact equilibrium of the configured finite discretized model, not a continuous-space global-equilibrium claim.

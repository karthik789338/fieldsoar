# FieldSOAR Reproducibility Package

FieldSOAR studies how to schedule network and compute resources for agricultural UAV fleets while meeting task deadlines. This package contains the results and replay code for experiments with 4, 8, 16, 32, 64, and 128 UAVs.

## Check the paper results

Run these commands from the repository root:

```sh
sha256sum -c --quiet SHA256SUMS
python tools/verify_published_tables.py --export reproduced/tables
python tools/verify_mission_duration_results.py
```

The checksum command verifies the release files. The two Python scripts use the standard library. They compare the saved CSVs with the values in Tables II, VI, and VIII and recompute all 480 rows of Table VIII from the recorded episodes.

The first Python script also writes editable CSVs for Tables II–VIII to `reproduced/tables`. The exports of Tables III–V and VII come from the final Word tables. The repository includes the available archived experiment outputs for those tables.

## Set up the replay environment

Use Python 3.12. For a CPU replay on Linux:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-replay.txt
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

We ran the original experiments with PyTorch 2.7.1 and CUDA 11.8. GPU replay requires a compatible CUDA build of PyTorch. The broader research dependencies are in `requirements-core.txt`.

## Recompute the IPPO and QMIX comparisons

To regenerate the IPPO and QMIX results for Tables II and VI from the saved held-out evaluations:

```sh
RUN=results/noms_baselines_all_fleets_ep10000/20261120T051535Z
python run_baseline_experiments.py report "$RUN"
```

The script prints the directory containing the regenerated scores and paired bootstrap confidence intervals. The corresponding paper CSVs are in `results/published/`.

## Replay the mission-duration experiments

```sh
RUN=results/noms_baselines_all_fleets_ep10000/20261120T051535Z
python mission_duration/replay_mission_durations.py --baseline-10k-root "$RUN" --device cpu
```

This command reruns IPPO and QMIX at 3, 5, 10, 20, and 30 seconds. It reports SLA over the full mission and for tasks arriving in the final three seconds. Each baseline uses the checkpoint selected on validation from its 10,000-episode run, with training seeds 11, 23, and 37.

The combined output uses the saved FieldSOAR and Critical Local-first results. The FieldSOAR results come from the original 900-episode checkpoints. The replay scripts resolve archived paths relative to this repository and preserve the original configurations and checkpoint manifests.

## Train the baselines again

Use a fresh output directory. The preparation step loads the hyperparameters selected in the original validation runs.

```sh
NEW=results/noms_baselines_all_fleets_ep10000/reproduced_run
python run_baseline_experiments.py prepare "$NEW"
python run_baseline_experiments.py run "$NEW" --device cpu
python run_baseline_experiments.py report "$NEW"
```

Training takes substantially longer than checkpoint replay. Results may vary with hardware and software versions.

## Find the inputs and saved results

| Location | Contents |
|---|---|
| `results/published/` | Final Word tables and the benchmark CSVs |
| `results/noms_baselines_all_fleets_ep10000/` | IPPO and QMIX checkpoints, tuning files, held-out episode metrics, and reports |
| `results/noms_horizon_check/` | Mission-duration episode records and twelve 30-second radio traces, with trajectories and manifests |
| `data/` | Workload sample manifest and twelve three-second radio traces |
| `.fieldvision_audit_v14_2/` | Frozen code used for the published checkpoints |

The selected FieldSOAR weights are in the original evaluation directories. The archived `fieldvision_*.py` drivers and input generators retain the filenames referenced by the original runs.

To reconstruct the three-second radio traces, run `python tools/reconstruct_three_second_traces.py`. The twelve reconstructed CSVs match the archived three-second prefixes byte for byte. The checks are recorded in `data/traces/reconstruction_provenance.json`.

The workload sample manifest matches the checksums in both profiling records. Replay uses the saved CPU and GPU profiles. Repeating the original profiling measurements requires the original imagery and hardware.

The bundled UavNetSim license applies to the upstream UavNetSim code.

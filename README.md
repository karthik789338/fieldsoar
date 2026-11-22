# FieldSOAR Reproducibility Package

FieldSOAR evaluates joint network and compute orchestration under service-level objectives for agricultural UAV fleets of 4, 8, 16, 32, 64, and 128 UAVs.

## Main scripts

| Task | File |
|---|---|
| Verify and export the paper tables | `tools/verify_published_tables.py` |
| Recompute Table VIII from episode records | `tools/verify_mission_duration_results.py` |
| Train or report the 10,000-episode IPPO and QMIX baselines | `run_baseline_experiments.py` |
| Replay mission durations with selected checkpoints | `mission_duration/replay_mission_durations.py` |
| Reconstruct three-second analytical radio traces | `tools/reconstruct_three_second_traces.py` |

The archived `fieldvision_*.py` experiment drivers, the original input generators, and the files under `.fieldvision_audit_v14_2/` retain their original names for compatibility with historical runs and provenance records.

## Verify the reported results

Extract the ZIP and run from its root directory:

```sh
sha256sum -c --quiet SHA256SUMS
python tools/verify_published_tables.py --export reproduced/tables
python tools/verify_mission_duration_results.py
```

These commands use the standard library. They check the reported CSV values against Tables II, VI, and VIII and independently recompute all 480 mission-duration results from recorded episodes. The export writes editable CSVs for Tables II–VIII. Tables III–V and VII are exported from the final Word tables. Available experiment outputs for those tables are also included.

## Install the replay environment

Use Python 3.12. For the CPU replay:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-replay.txt
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

The original experiment environment used PyTorch 2.7.1 with CUDA 11.8. To replay with CUDA, use a compatible GPU environment and a CUDA build of PyTorch. The included `requirements-core.txt` describes the broader research environment.

## Recompute Tables II and VI from saved held-out evaluations

```sh
RUN=results/noms_baselines_all_fleets_ep10000/20261120T051535Z
python run_baseline_experiments.py report "$RUN"
```

The command prints the report directory and writes its baseline point estimates and paired bootstrap intervals there. Compare its two CSVs with `results/published/`.

## Replay the mission-duration baselines

```sh
RUN=results/noms_baselines_all_fleets_ep10000/20261120T051535Z
python mission_duration/replay_mission_durations.py --baseline-10k-root "$RUN" --device cpu
```

The default input selection uses the bundled original completed horizon runs for the smaller fleets and for 128 UAVs. The evaluator validates the 30-second input manifests and the 10,000-episode IPPO/QMIX checkpoints, then evaluates the whole-mission and final-3-second windows at 3, 5, 10, 20, and 30 seconds. The combined CSV incorporates the archived FieldSOAR and Critical Local-first results. FieldSOAR uses the original 900-episode checkpoints. IPPO and QMIX use the validation-selected checkpoint from each 10,000-episode run and seeds 11, 23, and 37. The package provides these weights, traces, and profiling inputs together. The replay launchers resolve historical project-root paths inside the extracted tree without rewriting frozen configuration files or checkpoint manifests.

## Repeat baseline training

Select a new run directory. The launcher uses the archived validation-selected hyperparameters.

```sh
NEW=results/noms_baselines_all_fleets_ep10000/reproduced_run
python run_baseline_experiments.py prepare "$NEW"
python run_baseline_experiments.py run "$NEW" --device cpu
python run_baseline_experiments.py report "$NEW"
```

Training is substantially longer than checkpoint replay. Training and numerical outputs can vary with the hardware and software environment.

## Archive contents and provenance

- `results/published/` contains the final Word tables and benchmark report CSVs.
- `results/noms_baselines_all_fleets_ep10000/` contains the selected IPPO and QMIX weights, tuning files, held-out episode metrics, and reports. The selected original FieldSOAR weights are in the original evaluation directories.
- `results/noms_horizon_check/` contains the original and updated episode records and the twelve 30-second radio traces with their input trajectories and manifests.
- `data/` contains the workload sample manifest and the twelve three-second radio inputs. The sample manifest matches both measurement profile digests. The radio inputs were reconstructed from the archived trajectories and assumptions. All twelve CSVs byte-match their archived three-second prefixes. The reconstruction evidence is in `data/traces/reconstruction_provenance.json`. The sample manifest and CPU/GPU profile records are sufficient for replay. Regenerating the measurement profiles would require the original imagery and profiling hardware.
- `.fieldvision_audit_v14_2/` contains the frozen code used for the published checkpoints. Supporting scripts and configurations are retained for the historical experiments.

The release excludes logs, environments, caches, and earlier archives. Extract the release to commit browsable source and results to GitHub, or attach the single ZIP to a release. The original project license was not provided. The included UavNetSim license applies to that upstream project.

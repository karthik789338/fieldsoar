#!/usr/bin/env bash
# Run from the FieldVision repository root after activating .venv.
set -euo pipefail

device="${1:-cpu}"
if [[ $# -gt 1 || ( "$device" != cpu && "$device" != cuda ) ]]; then
  printf 'Usage: bash run_128uav_mission_durations.sh [cpu|cuda]\n' >&2
  exit 2
fi

root="$(pwd -P)"
native_python="$root/.venv_uavnetsim/bin/python"
project_python="$root/.venv/bin/python"
repo="$root/third_party/UavNetSim"
expected_commit=c2703f24eb58cdb557a5f67ded5c7b0368a964ed
if [[ "$(git -C "$repo" rev-parse HEAD)" != "$expected_commit" ]]; then
  printf 'Expected UavNetSim commit %s at %s\n' "$expected_commit" "$repo" >&2
  exit 1
fi
if [[ "$device" == cuda ]]; then
  "$project_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not visible in .venv"; print(torch.cuda.get_device_name(0))'
fi

run="$root/results/noms_horizon_check/mobility_${device}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$(dirname "$run")"
mkdir "$run"
printf 'Output directory: %s\n' "$run"

for seed in 27 41; do
  out="$run/traces/seed${seed}"
  mkdir -p "$out/capture"
  original="$root/results/uavnetsim_native_telemetry_128uav_seed${seed}"
  "$native_python" -u - "$repo" "$expected_commit" "$seed" "$original" "$out/capture" <<'PY' 2>&1 | tee "$out/mobility.log"
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

repo, commit, seed, original, destination = sys.argv[1:]
repo, original, destination = map(Path, (repo, original, destination))
seed = int(seed)
sys.path.insert(0, str(repo))

from mobility.trajectory import generate_trajectory
from scene.airspace import Airspace
from utils import config


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


nodes = 128
duration_s = 30
source = original / "uavnetsim_uav_trajectories.csv"
source_manifest = original / "uavnetsim_capture_manifest.json"
prior = json.loads(source_manifest.read_text())
if prior["trajectories_sha256"] != sha256(source):
    raise ValueError(f"Original trajectory checksum mismatch: {source}")

config.MOBILITY_MODEL = "GaussMarkov3D"
config.NUMBER_OF_DRONES = nodes
scene_xml = Path(config.SIONNA_SCENE_PATH)
scene_json = scene_xml.with_suffix(".json")
for path, key in ((scene_json, "scene_json_sha256"), (scene_xml, "scene_xml_sha256")):
    if key in prior and prior[key] != sha256(path):
        raise ValueError(f"Original scene checksum mismatch: {path}")

airspace = Airspace.from_file(
    scene_json,
    max_height=config.MAP_HEIGHT,
    building_clearance=config.UAV_BUILDING_CLEARANCE,
    boundary_clearance=config.UAV_BOUNDARY_CLEARANCE,
    min_flight_height=config.UAV_MIN_ALTITUDE,
    max_flight_height=config.UAV_MAX_ALTITUDE,
)
trajectory = generate_trajectory(seed, nodes, duration_s * 1e6, config.UAV_SPEED, airspace)
ticks = duration_s * 10
if trajectory.times_us[:ticks].tolist() != [i * 100000.0 for i in range(ticks)]:
    raise ValueError("Expected a complete 0.1-second trajectory grid")

seen = set()
with source.open(newline="") as handle:
    for row in csv.DictReader(handle):
        time_s = float(row["sim_time_s"])
        if time_s >= 3:
            continue
        tick, node = round(time_s * 10), int(row["uav_id"])
        key = (tick, node)
        if (not 0 <= tick < 30 or not 0 <= node < nodes or key in seen
                or not math.isclose(tick / 10, time_s, abs_tol=1e-9)):
            raise ValueError(f"Invalid position key in {source}: {time_s}, {node}")
        if trajectory.positions[tick, node].tolist() != [float(row[f"{axis}_m"]) for axis in "xyz"]:
            raise ValueError(f"Original position differs for seed {seed}, tick {tick}, UAV {node}")
        if row["sleeping"].strip().lower() not in ("false", "0"):
            raise ValueError(f"Original UAV was asleep for seed {seed}, tick {tick}, UAV {node}")
        seen.add(key)
if len(seen) != 30 * nodes:
    raise ValueError(f"Expected {30 * nodes} original positions, got {len(seen)}")

output = destination / "uavnetsim_uav_trajectories.csv"
with output.open("x", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(("sim_time_s", "uav_id", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "sleeping"))
    for tick in range(ticks):
        for node in range(nodes):
            writer.writerow((tick / 10, node, *trajectory.positions[tick, node], *trajectory.velocities[tick, node], False))

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "uavnetsim_git": {
        "commit": commit,
        "dirty": bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"])),
    },
    "command_parameters": {"nodes": nodes, "duration_s": duration_s, "seed": seed, "mobility": config.MOBILITY_MODEL},
    "trajectory_generation": "uavnetsim_generate_trajectory_mobility_only",
    "native_packet_telemetry_captured": False,
    "sionna_rt_used": False,
    "sleep_model": "UAVs remain active in the mobility-only generator",
    "scene_xml_sha256": sha256(scene_xml),
    "scene_json_sha256": sha256(scene_json),
    "trajectories_sha256": sha256(output),
    "capacity_trace_created": False,
    "original_3s_trajectory_sha256": prior["trajectories_sha256"],
    "original_3s_capture_manifest_sha256": sha256(source_manifest),
    "original_3s_positions_matched": len(seen),
}
(destination / "uavnetsim_capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Seed {seed}: matched {len(seen)} original positions, wrote {ticks * nodes} positions.")
PY

  "$project_python" - "$seed" "$out/radio_config.yaml" <<'PY'
import sys
from pathlib import Path

import yaml

seed, dest = sys.argv[1:]
config = yaml.safe_load(Path(f"configs/noms_scale_v10/radio_128uav_seed{seed}.yaml").read_text())
if config["n_uavs"] != 128 or config["tick_s"] != 0.1:
    raise ValueError("Expected the original 128-UAV 0.1-second radio configuration")
config["duration_s"] = 30.0
Path(dest).write_text(yaml.safe_dump(config, sort_keys=False))
PY
  "$project_python" -u prepare_research_inputs.py radio-trace \
    --trajectory-csv "$out/capture/uavnetsim_uav_trajectories.csv" \
    --radio-config "$out/radio_config.yaml" \
    --output-trace "$out/radio_30s_seed${seed}.csv" 2>&1 | tee "$out/radio_trace.log"
  test -s "$out/radio_30s_seed${seed}.csv"
  test -s "$out/radio_30s_seed${seed}.manifest.json"
done

args=(
  --nominal-trace "$run/traces/seed27/radio_30s_seed27.csv"
  --stress-trace "$run/traces/seed41/radio_30s_seed41.csv"
  --device "$device"
)
"$project_python" -u fieldsoar_experiments/evaluate_128uav_mission_durations.py \
  "${args[@]}" --pilot --output "$run/pilot" 2>&1 | tee "$run/pilot.log"
"$project_python" -u fieldsoar_experiments/evaluate_128uav_mission_durations.py \
  "${args[@]}" --output "$run/full" 2>&1 | tee "$run/full.log"
test -s "$run/full/summary.csv"
printf 'Horizon results: %s\n' "$run/full/summary.csv"

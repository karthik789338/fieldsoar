#!/usr/bin/env bash
# Run from the FieldVision repository root after activating .venv.
set -euo pipefail

device="${1:-cpu}"
if [[ "$device" != cpu && "$device" != cuda ]]; then
  printf 'Usage: bash fieldsoar_fleet_horizons/run_smaller_fleet_mission_durations.sh [cpu|cuda] [4 8 16 32 64]\n' >&2
  exit 2
fi
if [[ $# -gt 0 ]]; then shift; fi
fleets=("$@")
if [[ ${#fleets[@]} -eq 0 ]]; then fleets=(4 8 16 32 64); fi
declare -A used=()
for fleet in "${fleets[@]}"; do
  if [[ ! "$fleet" =~ ^(4|8|16|32|64)$ || "${used[$fleet]:-}" == yes ]]; then
    printf 'Expected a unique fleet size from 4, 8, 16, 32, 64. Got %s\n' "$fleet" >&2
    exit 2
  fi
  used[$fleet]=yes
done

root="$(pwd -P)"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
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

run="$root/results/noms_horizon_check/smaller_fleets_${device}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$(dirname "$run")"
mkdir "$run"
printf 'Output directory: %s\n' "$run"

prepare_trace() {
  local fleet="$1" seed="$2" out="$3" source trace
  mkdir -p "$out/capture"
  source=$(PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}" "$project_python" -P -c \
    'import sys; from evaluate_fleet_mission_durations import original_radio_path; print(original_radio_path(int(sys.argv[1]), int(sys.argv[2])).resolve(strict=True))' \
    "$fleet" "$seed")
  "$project_python" -u - "$fleet" "$source" "$out" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

fleet, source, out = sys.argv[1:]
source, out = map(Path, (source, out))
meta = json.loads(source.with_suffix(".manifest.json").read_text())
if hashlib.sha256(source.read_bytes()).hexdigest() != meta["generated_trace_sha256"]:
    raise ValueError(f"Original radio checksum mismatch: {source}")
config = meta["assumptions"]
if (int(config["n_uavs"]), float(config["tick_s"]), float(config["duration_s"])) != (int(fleet), 0.1, 3.0):
    raise ValueError(f"Unexpected original radio configuration: {source}")
config["duration_s"] = 30.0
(out / "radio_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY
  "$native_python" -u - "$repo" "$expected_commit" "$fleet" "$seed" "$source" "$out/capture" <<'PY' 2>&1 | tee "$out/mobility.log"
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

repo, commit, nodes, seed, radio, destination = sys.argv[1:]
repo, radio, destination = map(Path, (repo, radio, destination))
nodes, seed = int(nodes), int(seed)
sys.path.insert(0, str(repo))

from mobility.trajectory import generate_trajectory
from scene.airspace import Airspace
from utils import config


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


duration_s = 30
radio_manifest = json.loads(radio.with_suffix(".manifest.json").read_text())
source = Path(radio_manifest["source_trajectory_csv"])
source_manifest = Path(radio_manifest["capture_manifest_path"])
if sha256(source_manifest) != radio_manifest["capture_manifest_sha256"]:
    raise ValueError(f"Original capture manifest checksum mismatch: {source_manifest}")
prior = json.loads(source_manifest.read_text())
if prior["trajectories_sha256"] != sha256(source) or prior["trajectories_sha256"] != radio_manifest["source_trajectory_sha256"]:
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
    "original_3s_radio_sha256": radio_manifest["generated_trace_sha256"],
    "original_3s_positions_matched": len(seen),
}
(destination / "uavnetsim_capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"{nodes} UAVs, seed {seed}: matched {len(seen)} original positions, wrote {ticks * nodes} positions.")
PY

  trace="$out/radio_30s_seed${seed}.csv"
  "$project_python" -u prepare_research_inputs.py radio-trace \
    --trajectory-csv "$out/capture/uavnetsim_uav_trajectories.csv" \
    --radio-config "$out/radio_config.yaml" \
    --output-trace "$trace" 2>&1 | tee "$out/radio_trace.log"
  "$project_python" - "$source" "$trace" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as old_file, open(sys.argv[2], newline="") as new_file:
    original = list(csv.DictReader(old_file))
    generated = csv.DictReader(new_file)
    if len(original) != 30 or any(row != next(generated, None) for row in original):
        raise ValueError(f"Radio prefix differs from the 3-second reference: {sys.argv[1]}")
print("Radio prefix matches all 30 original ticks.")
PY
}

for fleet in "${fleets[@]}"; do
  out="$run/fleet${fleet}"
  for seed in 27 41; do
    prepare_trace "$fleet" "$seed" "$out/traces/seed${seed}"
  done
  args=(
    --fleet "$fleet"
    --nominal-trace "$out/traces/seed27/radio_30s_seed27.csv"
    --stress-trace "$out/traces/seed41/radio_30s_seed41.csv"
    --device "$device"
  )
  "$project_python" -u "$here/evaluate_fleet_mission_durations.py" "${args[@]}" --pilot --output "$out/pilot" \
    2>&1 | tee "$out/pilot.log"
  "$project_python" -u "$here/evaluate_fleet_mission_durations.py" "${args[@]}" --output "$out/full" \
    2>&1 | tee "$out/full.log"
done

"$project_python" - "$run" "${fleets[@]}" <<'PY'
import csv
import itertools
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = set(itertools.product(
    ("nominal", "stress"), ("3", "5", "10", "20", "30"),
    ("fieldsoar", "ippo", "qmix", "critical_local_first"), ("whole", "final_3s"),
))
combined = []
for fleet in sys.argv[2:]:
    with (root / f"fleet{fleet}" / "full" / "summary.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    keys = {(r["protocol"], r["horizon_s"], r["method"], r["window"]) for r in rows}
    if len(rows) != len(expected) or keys != expected or {row["fleet_uavs"] for row in rows} != {fleet}:
        raise ValueError(f"Incomplete summary for {fleet} UAVs")
    combined.extend(rows)
with (root / "summary.csv").open("x", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(combined[0]))
    writer.writeheader()
    writer.writerows(combined)
print(f"Horizon results: {root / 'summary.csv'}")
PY

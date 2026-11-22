#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 || ( "$2" != 27 && "$2" != 41 ) ]]; then
  echo "Usage from the FieldVision root: bash fieldsoar_experiments/prepare_128uav_radio_trace.sh OUTPUT_ROOT 27|41" >&2
  exit 2
fi

seed="$2"
out="$1/seed${seed}"
mkdir -p "$1"
mkdir "$out"
native_python=.venv_uavnetsim/bin/python

"$native_python" -u capture_uavnetsim_native_telemetry.py \
  --repo third_party/UavNetSim --nodes 128 --duration 30 \
  --seed "$seed" --routing Greedy --mac CSMA_CA --mobility GaussMarkov3D \
  --samples "${FIELDSOAR_NATIVE_SAMPLES:-250000}" --poll-ms 1 \
  --output "$out/capture" 2>&1 | tee "$out/capture.log"

python - "$seed" "$out/radio_config.yaml" <<'PY'
import sys
from pathlib import Path

import yaml

seed, out = sys.argv[1:]
source = Path(f"configs/noms_scale_v10/radio_128uav_seed{seed}.yaml")
config = yaml.safe_load(source.read_text())
assert config["n_uavs"] == 128 and config["tick_s"] == 0.1
config["duration_s"] = 30.0
Path(out).write_text(yaml.safe_dump(config, sort_keys=False))
PY

trace="$out/radio_30s_seed${seed}.csv"
python -u prepare_research_inputs.py radio-trace \
  --trajectory-csv "$out/capture/uavnetsim_uav_trajectories.csv" \
  --radio-config "$out/radio_config.yaml" --output-trace "$trace" \
  2>&1 | tee "$out/radio_trace.log"
test -s "$trace"
test -s "${trace%.csv}.manifest.json"
printf 'Ready: %s\n' "$trace"

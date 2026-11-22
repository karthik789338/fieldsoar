"""Rerun Table VIII with 10,000-episode IPPO and QMIX checkpoints."""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from evaluate_mission_duration import FLEETS, HORIZONS, write_csv
from portable_paths import project_path

ARCHIVE = Path("results/noms_horizon_check")
BASELINES = ("ippo", "qmix")
REFERENCES = ("fieldsoar", "critical_local_first")
PROTOCOLS = ("nominal", "stress")
WINDOWS = ("whole", "final_3s")


def read_rows(path, fleet, methods):
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] in methods]
    expected = set(product(PROTOCOLS, map(str, HORIZONS), methods, WINDOWS))
    actual = set()
    for row in rows:
        if "fleet_uavs" not in row:
            if fleet != 128:
                raise ValueError(f"Fleet column missing in {path}")
            row = {"fleet_uavs": str(fleet), **row}
        if int(row["fleet_uavs"]) != fleet:
            raise ValueError(f"Unexpected fleet in {path}")
        cases = 15 if row["protocol"] == "nominal" else 25
        replicas = 1 if row["method"] == "critical_local_first" else 3
        if int(row["episodes"]) != cases * replicas:
            raise ValueError(f"Incomplete evaluation coverage in {path}")
        actual.add((row["protocol"], row["horizon_s"], row["method"], row["window"]))
    if actual != expected or len(rows) != len(expected):
        raise ValueError(f"Incomplete or duplicate result rows in {path}")
    return [{"fleet_uavs": str(fleet), **row} for row in rows]


def archive_inputs(path, fleet):
    manifest = path / "manifest.json"
    data = json.loads(manifest.read_text())
    if data["pilot"] or tuple(data["horizons_s"]) != HORIZONS or data.get("fleet_uavs", 128) != fleet:
        raise ValueError(f"Incorrect archive manifest: {manifest}")
    traces = {protocol: project_path(data["trace_paths"][protocol]).resolve(strict=True) for protocol in PROTOCOLS}
    references = read_rows(path / "summary.csv", fleet, REFERENCES)
    return manifest, traces, references


def find_archive(explicit, fleets):
    candidates = [explicit] if explicit else sorted(
        ARCHIVE.glob("mobility_*" if fleets == (128,) else "smaller_fleets_*"),
        key=lambda p: p.name.rsplit("_", 1)[-1], reverse=True,
    )
    for root in candidates:
        try:
            inputs = {fleet: archive_inputs(root / ("full" if fleet == 128 else f"fleet{fleet}/full"), fleet)
                      for fleet in fleets}
        except (ValueError, OSError, KeyError) as error:
            if explicit:
                raise
            print(f"Skipping {root}: {error}", file=sys.stderr)
        else:
            return inputs
    raise FileNotFoundError(f"No complete original horizon run found for fleets {fleets} in {ARCHIVE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-10k-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smaller-reference", type=Path, help="Original smaller_fleets_* run folder")
    parser.add_argument("--reference-128", type=Path, help="Original mobility_* run folder")
    args = parser.parse_args()
    root = args.baseline_10k_root.resolve(strict=True)
    inputs = find_archive(args.smaller_reference, FLEETS[:-1])
    inputs.update(find_archive(args.reference_128, (128,)))
    for fleet in FLEETS:
        for method in BASELINES:
            case = root / f"fleet{fleet}" / method
            (case / "tuning.yaml").resolve(strict=True)
            (case / "training/selected_model.json").resolve(strict=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ARCHIVE / f"all_fleets_ippo_qmix_10k_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"Writing horizon results to {output}", flush=True)
    all_rows = []
    for fleet in FLEETS:
        reference, traces, old_rows = inputs[fleet]
        dest = output / f"fleet{fleet}"
        print(f"Running {fleet} UAVs with reference {reference}", flush=True)
        subprocess.run([
            sys.executable, "-u", str(Path(__file__).with_name("evaluate_mission_duration.py")),
            "--fleet", str(fleet), "--baseline-10k-root", str(root),
            "--nominal-trace", str(traces["nominal"]), "--stress-trace", str(traces["stress"]),
            "--reference-manifest", str(reference), "--output", str(dest), "--device", args.device,
        ], check=True)
        rows = old_rows + read_rows(dest / "summary.csv", fleet, BASELINES)
        all_rows.extend(sorted(rows, key=lambda row: (row["protocol"], int(row["horizon_s"]), row["method"], row["window"])))
    write_csv(output / "table_VIII_summary.csv", all_rows)
    print(f"Upload {output / 'table_VIII_summary.csv'} to update Table VIII.")


if __name__ == "__main__":
    main()

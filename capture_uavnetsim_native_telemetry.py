#!/usr/bin/env python3
"""Capture *raw* UavNetSim events and trajectories, without treating throughput as capacity.

Pinned to UavNetSim c2703f24eb58cdb557a5f67ded5c7b0368a964ed.
No modifications of the upstream repository; no PHY-capacity CSV is produced.
Run with the separately installed .venv_uavnetsim environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_COMMIT = "c2703f24eb58cdb557a5f67ded5c7b0368a964ed"
POSITION_COLUMNS = [
    "sim_time_s", "uav_id", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
    "energy_j", "queue_size", "sleeping",
]


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_info(repo: Path) -> dict:
    def git(*params: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *params], check=True,
                              capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def mobility_rows(event: dict) -> list[dict]:
    """Expose simulator-reported coordinates as trajectories (not link capacities)."""
    if event.get("event_type") != "simulation_state":
        return []
    payload = event.get("data", {})
    rows = []
    for node in payload.get("nodes", []):
        pos = node.get("position", [])
        vel = node.get("velocity", [])
        if len(pos) != 3 or len(vel) != 3:
            raise ValueError("Expected 3D simulator position and velocity")
        rows.append({
            "sim_time_s": float(event["sim_time_us"]) / 1e6,
            "uav_id": int(node["id"]),
            "x_m": float(pos[0]), "y_m": float(pos[1]), "z_m": float(pos[2]),
            "vx_mps": float(vel[0]), "vy_mps": float(vel[1]), "vz_mps": float(vel[2]),
            "energy_j": float(node["energy_j"]),
            "queue_size": int(node["queue_size"]),
            "sleeping": bool(node["sleeping"]),
        })
    return rows


def check_event_sequences(events: list[dict], last_sequence: int) -> int:
    """Detect event-ring overflow or missing records between polling calls."""
    for event in events:
        current = int(event["sequence"])
        if current != last_sequence + 1:
            raise RuntimeError(f"Event history sequence gap: expected {last_sequence + 1}, got {current}. "
                               "Decrease --poll-ms and rerun.")
        last_sequence = current
    return last_sequence


def normalize(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("third_party/UavNetSim"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--duration", type=float, default=3.0, help="simulated seconds")
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--routing", default="Greedy")
    parser.add_argument("--mac", default="CSMA_CA")
    parser.add_argument("--mobility", default="GaussMarkov3D")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--poll-ms", type=int, default=100,
                        help="poll event bus every simulated N milliseconds")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if args.nodes < 2 or args.duration <= 0 or not math.isfinite(args.duration):
        parser.error("--nodes must be >=2 and --duration must be finite and positive")
    if args.samples < 1 or args.poll_ms < 1:
        parser.error("--samples and --poll-ms must be positive")
    if not (repo / "simulator" / "simulator.py").is_file():
        parser.error(f"UavNetSim source not found under {repo}")
    origin = git_info(repo)
    if origin["commit"] != EXPECTED_COMMIT:
        parser.error("Unexpected UavNetSim commit. Refusing an untracked version; check out the pinned version.")
    if destination.exists() and any(destination.iterdir()):
        parser.error(f"Output directory must be new or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    # Use the *native* simulator, not a duplicate PHY approximation.
    previous_cwd = Path.cwd()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sim = None
    try:
        import simpy
        from simulator.simulator import Simulator
        from utils import config

        config.ROUTING_PROTOCOL = args.routing
        config.MAC_PROTOCOL = args.mac
        config.MOBILITY_MODEL = args.mobility
        config.NUMBER_OF_DRONES = args.nodes
        config.MAX_TTL = args.nodes + 1
        config.SIONNA_SAMPLES_PER_SOURCE = args.samples
        duration_us = args.duration * 1e6
        env = simpy.Environment()
        sim = Simulator(seed=args.seed, env=env, n_drones=args.nodes,
                        total_simulation_time=duration_us,
                        drone_speed=config.UAV_SPEED)
        if not all(hasattr(sim.event_bus, attr) for attr in ("since", "sequence")):
            raise RuntimeError("Pinned source lacks expected EventBus telemetry API")

        event_path = destination / "uavnetsim_native_events.jsonl"
        mobility_path = destination / "uavnetsim_uav_trajectories.csv"
        counters = Counter()
        last_sequence = 0
        steps = 0
        mobility_count = 0
        with event_path.open("w", encoding="utf-8") as events_out, \
             mobility_path.open("w", newline="", encoding="utf-8") as mobility_out:
            writer = csv.DictWriter(mobility_out, fieldnames=POSITION_COLUMNS)
            writer.writeheader()
            while True:
                events = sim.event_bus.since(last_sequence)
                last_sequence = check_event_sequences(events, last_sequence)
                for event in events:
                    counters[event["event_type"]] += 1
                    events_out.write(json.dumps(event, sort_keys=True, default=normalize) + "\n")
                    for row in mobility_rows(event):
                        writer.writerow(row)
                        mobility_count += 1
                if float(env.now) >= duration_us - 1e-6:
                    break
                next_us = min(duration_us, float(env.now) + args.poll_ms * 1e3)
                env.run(until=next_us)
                steps += 1

        if counters["simulation_state"] == 0 or mobility_count == 0:
            raise RuntimeError("No time-indexed UAV state snapshots were captured")
        metrics = sim.metrics.snapshot()
        report = {
            "validation_status": "RAW_TELEMETRY_CAPTURED_CAPACITY_NOT_CALIBRATED",
            "scope": "native_uav_to_uav_network_not_ground_edge_link",
            "duration_s": args.duration,
            "n_uavs": args.nodes,
            "simulator_metrics": metrics,
            "event_counts": dict(counters),
            "events_captured": sum(counters.values()),
            "trajectory_rows": mobility_count,
            "last_event_sequence": last_sequence,
            "event_poll_cycles": steps,
            "note": "Per-packet throughput is not available link capacity. No PHY-capacity trace inferred.",
        }
        (destination / "uavnetsim_native_telemetry_summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=normalize) + "\n", encoding="utf-8")
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "uavnetsim_git": origin,
            "python": platform.python_version(),
            "hostname": platform.node(),
            "command_parameters": {"nodes": args.nodes, "duration_s": args.duration,
                                   "seed": args.seed, "routing": args.routing,
                                   "mac": args.mac, "mobility": args.mobility,
                                   "raytrace_samples": args.samples, "poll_ms": args.poll_ms},
            "scene_xml_sha256": sha256(repo / "artifacts" / "scene" / "scene.xml"),
            "scene_json_sha256": sha256(repo / "artifacts" / "scene" / "scene.json"),
            "telemetry_script_sha256": sha256(Path(__file__)),
            "native_events_sha256": sha256(event_path),
            "trajectories_sha256": sha256(mobility_path),
            "capacity_trace_created": False,
        }
        (destination / "uavnetsim_capture_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("UAVNETSIM_NATIVE_TELEMETRY: PASS")
        print(json.dumps(report, indent=2, sort_keys=True, default=normalize))
        print(f"TELEMETRY_WRITTEN: {destination}")
    finally:
        if sim is not None:
            sim.close()
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()

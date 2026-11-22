#!/usr/bin/env python3
"""FieldVision-NOMS service simulator, Phase 3A + 3C trace-interface contract.

Phase 3C adds replay of explicitly labeled per-UAV link-capacity envelopes
with action-dependent weighted allocation at a shared bottleneck.
The included Phase 3C network CSV is SYNTHETIC ONLY. No packet-level
retransmissions, Sionna calibration, empirical AI profiles or MARL yet.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

TIERS = {"low": 1, "standard": 2, "high": 4}
PLACES = {"local", "edge", "cloud"}
EPS = 1e-9


def weighted_capped_share(total: float, weights: dict[Any, float], caps: dict[Any, float]) -> dict[Any, float]:
    """Work-conserving weighted sharing under per-recipient rate caps.

    A recipient's cap may be its PHY rate, its remaining data demand, or a
    CPU-independent limit. Reclaimed shares are redistributed to recipients
    that have residual demand. This uses ideal fluid scheduling per tick;
    protocol overhead is NOT modeled here.
    """
    if not math.isfinite(total) or total < 0:
        raise ValueError("Invalid total shared capacity")
    if weights.keys() != caps.keys():
        raise ValueError("Allocation weights and caps must refer to the same recipients")
    if any(not math.isfinite(v) or v < 0 for v in [*weights.values(), *caps.values()]):
        raise ValueError("Invalid allocation weight/cap")
    allocation = {key: 0.0 for key in weights}
    remaining = min(total, sum(caps.values()))
    active = {key for key in weights if caps[key] > EPS and weights[key] > EPS}
    while active and remaining > EPS:
        denominator = sum(weights[key] for key in active)
        shares = {key: remaining * weights[key] / denominator for key in active}
        saturated = {key for key in active if shares[key] >= caps[key] - allocation[key] - EPS}
        if not saturated:
            for key in active:
                allocation[key] += shares[key]
            break
        for key in saturated:
            amount = max(0.0, caps[key] - allocation[key])
            allocation[key] += amount
            remaining = max(0.0, remaining - amount)
        active.difference_update(saturated)
    return allocation


def read_network_trace(csv_path: Path, n_uavs: int, dt: float, horizon: float) -> list[dict[str, Any]]:
    """Read a complete, tick-aligned channel-capacity input envelope.

    These fields represent capacity bounds, never throughput already
    consumed by another scheduling policy. Measured achieved throughput
    must NOT be used as an interchangeable PHY capacity measure.
    """
    expected_fields = ["start_time_s", "bottleneck_mbps", *[f"uav_{i}_link_cap_mbps" for i in range(n_uavs)]]
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_fields:
            raise ValueError(f"Expected trace columns {expected_fields}; got {reader.fieldnames}")
        rows = []
        for index, row in enumerate(reader):
            try:
                values = {k: float(v) for k, v in row.items()}
            except (TypeError, ValueError) as err:
                raise ValueError(f"Non-numeric network trace at row {index}") from err
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f"Non-finite network trace at row {index}")
            if abs(values["start_time_s"] - index * dt) > 1e-7:
                raise ValueError(f"Network trace must cover every tick; unexpected timestamp at row {index}")
            if any(v < 0 for k, v in values.items() if k != "start_time_s"):
                raise ValueError(f"Negative capacity in network trace row {index}")
            rows.append(values)
    expected_rows = round(horizon / dt)
    if len(rows) != expected_rows:
        raise ValueError(f"Network trace has {len(rows)} ticks; expected {expected_rows}")
    return rows



def sha256_file(path: Path) -> str:
    # Hash files without loading gigabytes into RAM.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_profiled_jobs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reproducible task arrivals with per-image, paired CPU/GPU reference timings.

    CPU is a measured Hellbender proxy, NOT actual onboard UAV compute.
    V100 GPU is measured reference edge hardware. Cloud is an assumption.
    Service arrivals and deadlines are synthetic experimental parameters.
    """
    prof = config["profiled_workloads"]
    profile_csv = Path(prof["profiles_csv"]).expanduser().resolve(strict=True)
    profile_manifest = Path(prof["profiles_manifest"]).expanduser().resolve(strict=True)
    sample_manifest = Path(prof["samples_manifest"]).expanduser().resolve(strict=True)
    details = json.loads(profile_manifest.read_text(encoding="utf-8"))
    samples = json.loads(sample_manifest.read_text(encoding="utf-8"))
    if sha256_file(profile_csv) != details.get("profiles_sha256"):
        raise ValueError("Profile CSV SHA-256 mismatch against benchmark manifest")
    if sha256_file(sample_manifest) != details.get("sample_manifest_sha256"):
        raise ValueError("Selected samples SHA-256 mismatch against benchmark manifest")
    if (details.get("device") != "cuda" or details.get("weights_mode") != "pretrained" or
            details.get("training_or_finetuning_performed") is not False or
            details.get("profile_kind") != "frozen_reference_architecture_forward_pass"):
        raise ValueError("Expected frozen pretrained CUDA reference-model profile")
    if details.get("energy_measured") is not False:
        raise ValueError("Unexpected energy provenance in reference profile")

    model_details = details.get("model_details", {})
    cpu_by_key: dict[tuple[str, str], dict[str, float]] = {}
    cpu_manifest_sha = None
    cpu_csv_sha = None
    use_paired_cpu = "local_cpu_profiles_csv" in prof
    if use_paired_cpu:
        cpu_csv = Path(prof["local_cpu_profiles_csv"]).expanduser().resolve(strict=True)
        cpu_meta_path = Path(prof["local_cpu_profiles_manifest"]).expanduser().resolve(strict=True)
        cpu_meta = json.loads(cpu_meta_path.read_text(encoding="utf-8"))
        cpu_csv_sha = sha256_file(cpu_csv)
        cpu_manifest_sha = sha256_file(cpu_meta_path)
        if cpu_csv_sha != cpu_meta.get("profiles_sha256") or sha256_file(sample_manifest) != cpu_meta.get("sample_manifest_sha256"):
            raise ValueError("CPU benchmark provenance/checksum mismatch")
        if cpu_meta.get("device") != "cpu" or cpu_meta.get("weights_mode") != "pretrained":
            raise ValueError("Expected pretrained CPU reference profile")
        if cpu_meta.get("training_or_finetuning_performed") is not False or cpu_meta.get("energy_measured") is not False:
            raise ValueError("Invalid CPU training/energy provenance")
        if cpu_meta.get("profile_kind") != details.get("profile_kind"):
            raise ValueError("CPU and GPU reference model families differ")
        for ds, gpu_details in model_details.items():
            cpu_details = cpu_meta.get("model_details", {}).get(ds, {})
            if (cpu_details.get("architecture") != gpu_details.get("architecture") or
                cpu_details.get("local_weights_sha256") != gpu_details.get("local_weights_sha256")):
                raise ValueError("CPU/GPU model checkpoint mismatch for " + ds)
        with cpu_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["dataset_id"], row["image_path"])
                if key in cpu_by_key:
                    raise ValueError("Duplicate CPU profiled image identity")
                if row["device"] != "cpu" or row["weights_mode"] != "pretrained":
                    raise ValueError("CPU profiling row incorrectly labeled")
                vals = (float(row["preprocess_ms"]), float(row["inference_mean_ms"]))
                if not all(math.isfinite(v) and v > 0 for v in vals):
                    raise ValueError("Invalid CPU runtime measurement")
                cpu_by_key[key] = {"total_runtime_s": sum(vals)/1000.,
                                   "input_bytes": int(row["input_bytes"]),
                                   "model": row["model"]}
    sample_index = {(row["dataset_id"], row["relative_image_path"]): row
                    for row in samples["samples"]}
    if len(sample_index) != len(samples["samples"]):
        raise ValueError("Duplicate dataset/image entries in sample manifest")
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    observed_gpu_keys: set[tuple[str, str]] = set()
    with profile_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset_id"]
            key = (dataset, row["image_path"])
            if key in observed_gpu_keys:
                raise ValueError("Duplicate GPU profiled image identity")
            observed_gpu_keys.add(key)
            sample = sample_index.get(key)
            if sample is None or int(row["input_bytes"]) != int(sample["stored_bytes"]):
                raise ValueError("Profile row does not match recorded sample/image byte size")
            if row["weights_mode"] != "pretrained" or row["device"] != "cuda":
                raise ValueError("Unsupported profile device/weight provenance")
            if model_details.get(dataset, {}).get("architecture") != row["model"]:
                raise ValueError("Profile model identity mismatch")
            if model_details[dataset].get("agriculture_finetuned") is not False:
                raise ValueError("Expected non-agricultural reference checkpoint")
            values = {key: float(row[key]) for key in ("preprocess_ms", "inference_mean_ms")}
            if any(not math.isfinite(x) or x <= 0 for x in values.values()):
                raise ValueError("Invalid measured latency in benchmark profile")
            if use_paired_cpu:
                cpu = cpu_by_key.get(key)
                if (cpu is None or cpu["input_bytes"] != int(row["input_bytes"]) or
                        cpu["model"] != row["model"]):
                    raise ValueError("CPU/GPU per-image identity or model mismatch")
            by_dataset.setdefault(dataset, []).append({
                "dataset": dataset,
                "image_path": row["image_path"],
                "model": row["model"],
                "input_mb": int(row["input_bytes"]) / 1_000_000.0,  # decimal MB (8 Mbps / MB/s)
                "edge_runtime_s": (values["preprocess_ms"] + values["inference_mean_ms"]) / 1000.0,
                "local_cpu_runtime_s": cpu_by_key[key]["total_runtime_s"] if use_paired_cpu else None,
            })
    if use_paired_cpu and observed_gpu_keys != set(cpu_by_key):
        raise ValueError("CPU/GPU benchmark sample sets differ")
    names = tuple(prof["datasets"])
    min_images = int(prof.get("min_samples_per_dataset", 16))
    if not names or len(names) != len(set(names)):
        raise ValueError("Expected nonempty unique dataset identifiers")
    for name in names:
        if len(by_dataset.get(name, ())) < min_images:
            raise ValueError(f"Insufficient empirical profile inputs for {name}")
    n_uavs = int(config["n_uavs"])
    n_jobs = int(prof["job_count"])
    per_burst = int(prof["jobs_per_burst"])
    burst_period = float(prof["burst_period_s"])
    dt = float(config["tick_s"])
    local_factor = float(prof.get("assumed_local_runtime_multiplier", 1.0))
    cloud_factor = float(prof["assumed_cloud_runtime_multiplier"])
    # Explicit sensitivity on a measured Hellbender CPU reference, NOT a
    # hardware-derived Jetson/UAV runtime. 1.0 preserves the measured CPU.
    onboard_slowdown = float(prof.get("assumed_onboard_cpu_slowdown_factor", 1.0))
    if (not all(math.isfinite(x) for x in (burst_period, dt, local_factor,
                                           cloud_factor, onboard_slowdown)) or
            min(n_jobs, per_burst, n_uavs) <= 0 or
            min(burst_period, dt, local_factor, cloud_factor, onboard_slowdown) <= 0):
        raise ValueError("Invalid job arrival or runtime assumption")
    if abs(round(burst_period / dt) * dt - burst_period) > 1e-8:
        raise ValueError("Burst period must be an integer multiple of environment tick")
    dataset_cycle = tuple(prof.get('dataset_cycle', names))
    if not dataset_cycle or any(name not in names for name in dataset_cycle):
        raise ValueError('dataset_cycle must contain existing profiled dataset names')
    service_classes = prof["service_class_by_dataset"]
    deadlines = prof["assumed_relative_deadlines_s"]
    if any(n not in service_classes for n in names):
        raise ValueError("Missing dataset service-class mapping")
    if any(float(deadlines[service_classes[n]]) <= 0 for n in names):
        raise ValueError("Invalid experimental relative SLA deadline")
    if ((n_jobs - 1) // per_burst) * burst_period >= float(config["duration_s"]):
        raise ValueError("Job arrival would occur at or after the simulation horizon")
    rng = random.Random(int(config["seed"]))
    jobs = []
    for i in range(n_jobs):
        dataset = dataset_cycle[i % len(dataset_cycle)]
        observation = rng.choice(by_dataset[dataset])
        arrival = round((i // per_burst) * burst_period, 9)
        cls = service_classes[dataset]
        measured_edge_s = observation["edge_runtime_s"]
        jobs.append({
            "job_id": f"profiled_{i:05d}_{dataset.lower()}",
            "uav_id": i % n_uavs,
            "dataset": dataset,
            "workload": observation["model"],
            "arrival_s": arrival,
            "deadline_s": round(arrival + float(deadlines[cls]), 9),
            "input_mb": observation["input_mb"],
            "local_runtime_s": (observation["local_cpu_runtime_s"] if use_paired_cpu
                                else measured_edge_s * local_factor) * onboard_slowdown,
            "edge_runtime_s": measured_edge_s,
            "cloud_runtime_s": measured_edge_s * cloud_factor,
            "service_class": cls,
            "profile_image_path": observation["image_path"],
            "reference_model": observation["model"],
            "compute_profile_source": (("measured_hellbender_cpu_proxy_and_v100_edge_cloud_assumed"
                                        if onboard_slowdown == 1 else
                                        "hellbender_cpu_proxy_analytically_scaled_edge_v100_cloud_assumed")
                                       if use_paired_cpu else "measured_reference_edge_and_assumed_other_tiers"),
        })
    provenance = {
        "source_class": ("PAIRED_HELLBENDER_CPU_PROXY_AND_V100_GPU_CLOUD_ASSUMED"
                         if use_paired_cpu else "REFERENCE_GPU_MEASURED_ON_V100_OTHER_TIERS_ASSUMED"),
        "profile_csv_path": str(profile_csv),
        "profile_csv_sha256": sha256_file(profile_csv),
        "profile_manifest_sha256": sha256_file(profile_manifest),
        "sample_manifest_sha256": sha256_file(sample_manifest),
        "profiled_gpu": details.get("gpu"),
        "profiled_torch": details.get("torch_version"),
        "reference_checkpoint_weights_sha256": {
            name: model_details[name].get("local_weights_sha256") for name in names
        },
        "cpu_profiles_csv_sha256": cpu_csv_sha,
        "cpu_profiles_manifest_sha256": cpu_manifest_sha,
        "assumed_local_multiplier": (None if use_paired_cpu else local_factor),
        "assumed_cloud_multiplier": cloud_factor,
        "assumed_onboard_cpu_slowdown_factor": onboard_slowdown,
        "onboard_runtime_scale_provenance": (
            "measured_hellbender_cpu_reference_unscaled" if onboard_slowdown == 1
            else "EXPERIMENTAL_SENSITIVITY_NOT_ACTUAL_UAV_HARDWARE"),
        "deadline_source": "experimental_config_assumption_not_field_validated",
        "arrival_source": "synthetic_burst_schedule_not_measured",
        "agriculture_accuracy_evaluated": False,
        "local_cloud_executions_measured": False,
        "local_hardware_proxy_measured": bool(use_paired_cpu),
        "actual_uav_hardware_measured": False,
        "cloud_hardware_measured": False,
        "input_size_units": "decimal_MB_from_compressed_file_bytes",
        "profile_runtime_basis": "mean_preprocess_plus_mean_forward_per_real_image",
    }
    return jobs, provenance


@dataclass(frozen=True)
class Action:
    task_id: str | None = None  # None means defer
    placement: str = "local"
    network_tier: str = "standard"
    compute_tier: str = "standard"


@dataclass
class Job:
    job_id: str
    uav_id: int
    dataset: str
    workload: str
    arrival_s: float
    deadline_s: float  # Absolute mission time, NOT relative deadline
    input_mb: float
    local_runtime_s: float
    edge_runtime_s: float
    cloud_runtime_s: float
    service_class: str
    stage: str = "not_arrived"
    placement: str | None = None
    network_tier: str = "standard"
    compute_tier: str = "standard"
    remaining_mb: float = 0.0
    remaining_compute_s: float = 0.0
    started_s: float | None = None
    completed_s: float | None = None
    profile_image_path: str | None = None
    reference_model: str | None = None
    compute_profile_source: str = "synthetic_smoke_only"


class NOMSSim:
    """Step-based simulation with one scheduling actor per UAV.

    Each `step` consumes one tick. Remote upload is bandwidth-shared across
    all UAVs, edge/cloud run on finite compute slots, and local compute has
    one per-UAV slot. Transfer completes before remote compute may start on
    the next tick. No packet-level model or energy measurement is implied.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        allowed_profiles = {"synthetic_smoke_only", "measured_reference_edge_assumed_local_cloud",
                            "measured_reference_cpu_proxy_edge_gpu_assumed_cloud"}
        self.profile_source = config.get("workload_profile_source")
        if self.profile_source not in allowed_profiles:
            raise ValueError("Unsupported or unlabeled workload profile source")
        self.profile_provenance = {"source_class": "SYNTHETIC_SMOKE_ONLY_NOT_MEASURED"}
        if self.profile_source != "synthetic_smoke_only":
            if self.profile_source == "measured_reference_cpu_proxy_edge_gpu_assumed_cloud" and "local_cpu_profiles_csv" not in config.get("profiled_workloads", {}):
                raise ValueError("Paired-profile mode requires local_cpu_profiles_csv")
            if "jobs" in config:
                raise ValueError("Profiled mode cannot also define manually specified jobs")
            self.job_definitions, self.profile_provenance = load_profiled_jobs(config)
        else:
            self.job_definitions = config["jobs"]
        self.dt = float(config.get("tick_s", 1.0))
        self.horizon = float(config["duration_s"])
        self.n_uavs = int(config["n_uavs"])
        if self.dt <= 0 or self.horizon <= 0 or self.n_uavs < 1:
            raise ValueError("Invalid simulation time or UAV count")
        if abs(round(self.horizon / self.dt) * self.dt - self.horizon) > EPS:
            raise ValueError("duration_s must be divisible by tick_s")
        self.edge_slots = int(config["resources"]["edge_slots"])
        self.cloud_slots = int(config["resources"]["cloud_slots"])
        if min(self.edge_slots, self.cloud_slots) <= 0:
            raise ValueError("Compute slots must be positive")
        self.outage_windows = [tuple(map(float, w)) for w in config['network'].get('outage_windows_s',[])]
        if any(len(w)!=2 or not (0 <= w[0] < w[1] <= self.horizon) for w in self.outage_windows):
            raise ValueError('Outage windows must lie within the scenario horizon')
        self.network_capacity_scale = float(config["network"].get("capacity_scale", 1.0))
        if not math.isfinite(self.network_capacity_scale) or self.network_capacity_scale <= 0:
            raise ValueError("Network capacity scale must be positive and finite")
        self.base_mbps = float(config["network"]["bottleneck_mbps"])
        if self.base_mbps < 0:
            raise ValueError("Network capacity cannot be negative")
        self.capacity_schedule = sorted(
            (float(k), float(v)) for k, v in config["network"].get("capacity_schedule_mbps", {}).items()
        )
        if any(v < 0 for _, v in self.capacity_schedule):
            raise ValueError("Negative network capacity")
        network_config = config["network"]
        self.trace_path: Path | None = None
        self.trace_rows: list[dict[str, Any]] | None = None
        self.trace_provenance = "not_applicable_legacy_synthetic_schedule"
        if "trace_csv" in network_config:
            if network_config.get("capacity_schedule_mbps"):
                raise ValueError("Do not combine capacity_schedule_mbps and trace_csv")
            if network_config.get("trace_provenance") not in {
                "synthetic_contract_only", "measured", "physics_simulated",
                "trajectory_driven_analytical_uncalibrated"
            }:
                raise ValueError("Invalid trace_provenance value")
            self.trace_provenance = network_config["trace_provenance"]
            self.trace_path = Path(network_config["trace_csv"]).expanduser().resolve()
            if not self.trace_path.is_file():
                raise FileNotFoundError(f"Network trace not found: {self.trace_path}")
            self.trace_rows = read_network_trace(self.trace_path, self.n_uavs, self.dt, self.horizon)
            if self.trace_provenance == "trajectory_driven_analytical_uncalibrated":
                sidecar = self.trace_path.with_suffix(".manifest.json")
                if not sidecar.is_file():
                    raise FileNotFoundError("Analytical trace requires an adjacent generation manifest")
                generation = json.loads(sidecar.read_text(encoding="utf-8"))
                if generation.get("provenance") != self.trace_provenance:
                    raise ValueError("Trace manifest provenance mismatch")
                if generation.get("generated_trace_sha256") != hashlib.sha256(self.trace_path.read_bytes()).hexdigest():
                    raise ValueError("Analytical trace checksum mismatch")
        self.time_s = 0.0
        self.jobs: dict[str, Job] = {}
        for entry in self.job_definitions:
            job = Job(**entry)
            if job.job_id in self.jobs:
                raise ValueError(f"Duplicate job ID: {job.job_id}")
            if not 0 <= job.uav_id < self.n_uavs:
                raise ValueError(f"Invalid UAV ID: {job.uav_id}")
            vals = (job.arrival_s, job.deadline_s, job.input_mb,
                    job.local_runtime_s, job.edge_runtime_s, job.cloud_runtime_s)
            if any(v < 0 for v in vals) or job.deadline_s <= job.arrival_s:
                raise ValueError(f"Invalid arrival/deadline/resource values for {job.job_id}")
            self.jobs[job.job_id] = job
        self.waiting: dict[int, list[str]] = {i: [] for i in range(self.n_uavs)}
        self.local_running: dict[int, str] = {}
        self.uploading: list[str] = []
        self.remote_queues: dict[str, list[str]] = {"edge": [], "cloud": []}
        self.remote_running: dict[str, list[str]] = {"edge": [], "cloud": []}
        self.events: list[dict[str, Any]] = []
        self.network_history: list[dict[str, Any]] = []
        self.per_uav_network_history: list[dict[str, Any]] = []
        self.compute_history: list[dict[str, Any]] = []
        self.compute_busy_slot_seconds = {"local": 0.0, "edge": 0.0, "cloud": 0.0}
        self.peak_compute_concurrency = {"edge": 0, "cloud": 0}
        self._admit()

    def _event(self, event: str, task: Job, time_s: float, **kwargs: Any) -> None:
        self.events.append({"time_s": round(time_s, 6), "event": event,
                            "task_id": task.job_id, "uav_id": task.uav_id, **kwargs})

    def _admit(self) -> None:
        for job in self.jobs.values():
            if job.stage == "not_arrived" and job.arrival_s <= self.time_s + EPS:
                job.stage = "waiting"
                self.waiting[job.uav_id].append(job.job_id)
                self._event("arrived", job, self.time_s)

    def _is_outage_tick(self) -> bool:
        # Exogenous, predeclared outage windows; no future-link information
        # is provided to decentralized actors.
        return any(start-EPS <= self.time_s < end-EPS for start,end in self.outage_windows)

    def link_capacity_mbps(self) -> float:
        if self._is_outage_tick():
            return 0.0
        if self.trace_rows is not None:
            index = min(int(round(self.time_s / self.dt)), len(self.trace_rows) - 1)
            return self.trace_rows[index]["bottleneck_mbps"] * self.network_capacity_scale
        capacity = self.base_mbps
        for t, value in self.capacity_schedule:
            if t <= self.time_s + EPS:
                capacity = value
        return capacity * self.network_capacity_scale

    def uav_link_cap_mbps(self, uav_id: int) -> float:
        if self._is_outage_tick():
            return 0.0
        if self.trace_rows is None:
            return self.link_capacity_mbps()  # legacy: only common bottleneck
        index = min(int(round(self.time_s / self.dt)), len(self.trace_rows) - 1)
        return self.trace_rows[index][f"uav_{uav_id}_link_cap_mbps"] * self.network_capacity_scale

    def observations(self) -> dict[int, dict[str, Any]]:
        """Runtime-local information. Does NOT expose future channel states."""
        return {
            i: {
                "time_s": self.time_s,
                "uav_id": i,
                "waiting_tasks": [
                    {"task_id": self.jobs[j].job_id,
                     "workload": self.jobs[j].workload,
                     "service_class": self.jobs[j].service_class,
                     "remaining_deadline_s": self.jobs[j].deadline_s - self.time_s,
                     "input_mb": self.jobs[j].input_mb,
                     # Workload metadata is available to a UAV before scheduling.
                     # These runtimes are measured at the edge and assumed on
                     # other tiers in the current exploratory scenario.
                     "estimated_local_runtime_s": self.jobs[j].local_runtime_s,
                     "estimated_edge_runtime_s": self.jobs[j].edge_runtime_s,
                     "estimated_cloud_runtime_s": self.jobs[j].cloud_runtime_s}
                    for j in self.waiting[i]
                ],
                "local_busy": i in self.local_running,
                "observed_shared_capacity_mbps": self.link_capacity_mbps(),
                "observed_uav_link_cap_mbps": self.uav_link_cap_mbps(i),
                "edge_queue_length": len(self.remote_queues["edge"]),
                "cloud_queue_length": len(self.remote_queues["cloud"]),
                "observed_edge_busy_slots": len(self.remote_running["edge"]),
                "observed_cloud_busy_slots": len(self.remote_running["cloud"]),
                "edge_slot_capacity": self.edge_slots,
                "cloud_slot_capacity": self.cloud_slots,
            }
            for i in range(self.n_uavs)
        }

    def scripted_actions(self, policy: str = "edf") -> dict[int, Action]:
        """Simple debug policies, NOT research-quality baselines."""
        if policy not in {"edf", "fifo", "local_only", "edge_only", "cloud_only",
                          "network_aware_edf", "critical_first"}:
            raise ValueError(f"Unknown smoke policy: {policy}")
        actions = {}
        for i, queue in self.waiting.items():
            if not queue:
                actions[i] = Action()
                continue
            if policy in {"edf", "network_aware_edf"}:
                job_id = min(queue, key=lambda j: (self.jobs[j].deadline_s, j))
            elif policy == "critical_first":
                job_id = min(queue, key=lambda j: (
                    {"critical":0,"standard":1,"batch":2}.get(self.jobs[j].service_class, 3),
                    self.jobs[j].deadline_s, j))
            else:
                job_id = queue[0]
            job = self.jobs[job_id]
            if policy == "local_only":
                destination = "local"
            elif policy == "edge_only":
                destination = "edge"
            elif policy == "cloud_only":
                destination = "cloud"
            elif policy == "network_aware_edf":
                # Current-state feasible completion estimates; cloud RTT and
                # energy are not characterized yet, so only local/edge here.
                local_wait = (self.jobs[self.local_running[i]].remaining_compute_s
                              if i in self.local_running else 0.0)
                t_local = local_wait + job.local_runtime_s
                link = min(self.link_capacity_mbps(), self.uav_link_cap_mbps(i))
                t_upload = job.input_mb * 8.0 / link if link > EPS else float("inf")
                # Conservative 1-tick upload->compute dispatch delay and
                # approximate shared GPU slot congestion from current queues.
                edge_queue_cost = sum(self.jobs[j].edge_runtime_s for j in self.remote_queues["edge"])
                edge_occupancy = sum(self.jobs[j].remaining_compute_s for j in self.remote_running["edge"])
                t_edge = (t_upload + self.dt + job.edge_runtime_s +
                          (edge_queue_cost + edge_occupancy)/self.edge_slots)
                destination = "local" if t_local <= t_edge else "edge"
            else:
                # Original EDF/FIFO smoke heuristic preserved for regression.
                destination = "edge" if job.edge_runtime_s + job.input_mb * 8.0 / max(self.link_capacity_mbps(), EPS) < job.local_runtime_s else "local"
            if destination == "local" and i in self.local_running:
                actions[i] = Action()  # defer until local CPU free
            else:
                tier = "high" if job.service_class == "critical" else "low" if job.service_class == "batch" else "standard"
                actions[i] = Action(job_id, destination, tier, tier)
        return actions

    def _schedule(self, actions: dict[int, Action]) -> None:
        for i in range(self.n_uavs):
            action = actions.get(i, Action())
            if action.task_id is None:
                continue
            if action.placement not in PLACES:
                raise ValueError("Invalid placement")
            if action.network_tier not in TIERS or action.compute_tier not in TIERS:
                raise ValueError("Invalid resource tier")
            if action.task_id not in self.waiting[i]:
                raise ValueError(f"UAV {i} tried to schedule unavailable task: {action.task_id}")
            if action.placement == "local" and i in self.local_running:
                raise ValueError(f"UAV {i} local processor already busy")
            job = self.jobs[action.task_id]
            self.waiting[i].remove(action.task_id)
            job.started_s = self.time_s
            job.placement = action.placement
            job.network_tier = action.network_tier
            job.compute_tier = action.compute_tier
            self._event("scheduled", job, self.time_s, placement=action.placement,
                        network_tier=action.network_tier, compute_tier=action.compute_tier,
                        runtime_provenance=job.compute_profile_source)
            if action.placement == "local":
                job.remaining_compute_s = job.local_runtime_s
                job.stage = "computing"
                self.local_running[i] = job.job_id
                self._event("compute_started", job, self.time_s, placement="local")
            else:
                job.remaining_mb = job.input_mb
                job.stage = "uploading"
                self.uploading.append(job.job_id)
                self._event("upload_started", job, self.time_s)

    def _dispatch(self, place: str, at_time_s: float | None = None) -> None:
        slots = self.edge_slots if place == "edge" else self.cloud_slots
        queue = self.remote_queues[place]
        running = self.remote_running[place]
        while queue and len(running) < slots:
            queue.sort(key=lambda j: (-TIERS[self.jobs[j].compute_tier], self.jobs[j].deadline_s, j))
            jid = queue.pop(0)
            job = self.jobs[jid]
            job.stage = "computing"
            job.remaining_compute_s = job.edge_runtime_s if place == "edge" else job.cloud_runtime_s
            running.append(jid)
            self.peak_compute_concurrency[place] = max(self.peak_compute_concurrency[place], len(running))
            self._event("compute_started", job, self.time_s if at_time_s is None else at_time_s,
                        placement=place)

    def _completed(self, jid: str, completion_time_s: float) -> None:
        job = self.jobs[jid]
        job.completed_s = round(completion_time_s, 9)
        job.stage = "completed"
        self._event("completed", job, job.completed_s,
                    on_time=job.completed_s <= job.deadline_s + EPS)

    def _progress_compute(self) -> None:
        """Continuous-time completion events within one coarse network decision tick.

        Remote compute slots can refill during a tick, but network transfers
        are fluid-updated at tick boundaries (upload completion <= one tick
        conservatively rounded). Local tasks require a new actor decision
        before a second task can be started; therefore local slots do not
        autonomously refill within the tick.
        """
        tick_end = self.time_s + self.dt
        for i, jid in list(self.local_running.items()):
            job = self.jobs[jid]
            self.compute_busy_slot_seconds["local"] += min(self.dt, max(0.0, job.remaining_compute_s))
            if job.remaining_compute_s <= self.dt + EPS:
                self._completed(jid, self.time_s + max(0.0, job.remaining_compute_s))
                del self.local_running[i]
            else:
                job.remaining_compute_s -= self.dt
        for place in ("edge", "cloud"):
            cursor = self.time_s
            while cursor < tick_end - EPS:
                self._dispatch(place, at_time_s=cursor)
                running = self.remote_running[place]
                if not running:
                    break
                next_finish = min(self.jobs[jid].remaining_compute_s for jid in running)
                advance = min(tick_end - cursor, max(next_finish, 0.0))
                self.compute_busy_slot_seconds[place] += advance * len(running)
                for jid in running:
                    self.jobs[jid].remaining_compute_s = max(
                        0.0, self.jobs[jid].remaining_compute_s - advance)
                cursor += advance
                done = [jid for jid in running if self.jobs[jid].remaining_compute_s <= EPS]
                for jid in done:
                    self._completed(jid, cursor)
                    running.remove(jid)
                if not done and advance <= EPS:
                    raise RuntimeError("Compute scheduler cannot make time progress")

    def _progress_network(self) -> None:
        capacity = self.link_capacity_mbps()
        active = list(self.uploading)
        by_uav: dict[int, list[str]] = {}
        for jid in active:
            by_uav.setdefault(self.jobs[jid].uav_id, []).append(jid)

        uav_weights = {
            i: float(sum(TIERS[self.jobs[j].network_tier] for j in jobs))
            for i, jobs in by_uav.items()
        }
        # Per-UAV link caps and available payload demand both limit actual
        # schedulable Mbps. This avoids double-counting link capacity when
        # one UAV uploads multiple independent jobs.
        uav_caps = {
            i: min(self.uav_link_cap_mbps(i),
                   sum(self.jobs[j].remaining_mb for j in jobs) * 8.0 / self.dt)
            for i, jobs in by_uav.items()
        }
        uav_shares = weighted_capped_share(capacity, uav_weights, uav_caps)
        job_rates: dict[str, float] = {}
        for i, jobs in by_uav.items():
            weights = {jid: float(TIERS[self.jobs[jid].network_tier]) for jid in jobs}
            caps = {jid: self.jobs[jid].remaining_mb * 8.0 / self.dt for jid in jobs}
            job_rates.update(weighted_capped_share(uav_shares[i], weights, caps))

        transmitted_mb = 0.0
        delivered_by_uav: dict[int, float] = {i: 0.0 for i in range(self.n_uavs)}
        for jid in active:
            job = self.jobs[jid]
            sent_mb = min(job.remaining_mb, job_rates[jid] * self.dt / 8.0)
            transmitted_mb += sent_mb
            delivered_by_uav[job.uav_id] += sent_mb
            job.remaining_mb -= sent_mb
            if job.remaining_mb <= EPS:
                job.remaining_mb = 0.0
                job.stage = "remote_queued"
                self.uploading.remove(jid)
                self.remote_queues[job.placement].append(jid)
                self._event("upload_completed", job, self.time_s + self.dt,
                            placement=job.placement)

        for i in range(self.n_uavs):
            actual_rate = delivered_by_uav[i] * 8.0 / self.dt
            bound = self.uav_link_cap_mbps(i)
            if actual_rate > bound + 1e-7:
                raise AssertionError(f"UAV {i} link cap exceeded")
            self.per_uav_network_history.append({
                "start_time_s": round(self.time_s, 6),
                "uav_id": i,
                "link_cap_mbps": bound,
                "allocated_mbps": round(uav_shares.get(i, 0.0), 8),
                "used_mbps": round(actual_rate, 8),
                "active_upload_count": len(by_uav.get(i, [])),
            })
        self.network_history.append({
            "start_time_s": round(self.time_s, 6),
            "capacity_mbps": capacity,
            "transmitted_mb": round(transmitted_mb, 8),
            "used_mbps": round(8.0 * transmitted_mb / self.dt, 8),
            "active_upload_count": len(active),
        })
        if transmitted_mb * 8.0 > capacity * self.dt + 1e-7:
            raise AssertionError("Shared network capacity exceeded")

    def step(self, actions: dict[int, Action] | None = None, policy: str = "edf") -> None:
        if self.time_s >= self.horizon - EPS:
            raise RuntimeError("Simulation horizon reached")
        # Only external arrivals become visible before scheduling decisions.
        self._admit()
        self._schedule(self.scripted_actions(policy) if actions is None else actions)
        # Start jobs that were already in remote compute queues. Jobs finishing
        # uploads during this tick enter the queues for the NEXT tick.
        self._dispatch("edge")
        self._dispatch("cloud")
        self.compute_history.append({
            "start_time_s": round(self.time_s, 6),
            "edge_running": len(self.remote_running["edge"]),
            "cloud_running": len(self.remote_running["cloud"]),
            "local_running": len(self.local_running),
            "edge_queued": len(self.remote_queues["edge"]),
            "cloud_queued": len(self.remote_queues["cloud"]),
        })
        if (len(self.remote_running["edge"]) > self.edge_slots or
                len(self.remote_running["cloud"]) > self.cloud_slots):
            raise AssertionError("Compute capacity exceeded")
        self._progress_compute()
        self._progress_network()
        self.time_s = round(self.time_s + self.dt, 9)
        self._admit()

    def run(self, policy: str = "edf") -> dict[str, Any]:
        while self.time_s < self.horizon - EPS:
            self.step(policy=policy)
        return self.summary(policy)

    def summary(self, policy: str) -> dict[str, Any]:
        arrived = [j for j in self.jobs.values() if j.arrival_s < self.horizon]
        completed = [j for j in arrived if j.stage == "completed"]
        on_time = [j for j in completed if j.completed_s <= j.deadline_s + EPS]
        # Don't mark a not-yet-due, unfinished task as an SLA violation.
        evaluable = [j for j in arrived if j.stage == "completed" or j.deadline_s <= self.horizon]
        late = [j for j in completed if j.completed_s > j.deadline_s + EPS]
        network_peak = max((r["used_mbps"] for r in self.network_history), default=0.0)
        completed_latencies = sorted(j.completed_s-j.arrival_s for j in completed)
        def p95(values):
            if not values: return None
            index = .95 * (len(values)-1)
            lo = int(index)
            return round(values[lo] + (values[min(lo+1,len(values)-1)]-values[lo])*(index-lo), 6)
        per_uav_ontime = [sum(j.uav_id == i for j in on_time) for i in range(self.n_uavs)]
        fairness = ((sum(per_uav_ontime)**2)/(self.n_uavs*sum(n*n for n in per_uav_ontime))
                    if any(per_uav_ontime) else None)
        def div(a: int, b: int) -> float | None:
            return round(a / b, 6) if b else None
        classes = {}
        for cls in sorted({j.service_class for j in arrived}):
            eligible = [j for j in evaluable if j.service_class == cls]
            eligible_ontime = [j for j in eligible if j.stage == "completed" and j.completed_s <= j.deadline_s + EPS]
            classes[cls] = {"evaluable": len(eligible), "on_time": len(eligible_ontime),
                            "sla_satisfaction_rate": div(len(eligible_ontime), len(eligible))}
        return {
            "phase": ("profiled_workload_integration" if self.profile_source != "synthetic_smoke_only"
                      else "3C_trace_contract" if self.trace_rows is not None else "3A_contract_smoke"),
            "profile_provenance": self.profile_provenance["source_class"],
            "network_trace_provenance": self.trace_provenance,
            "policy": policy,
            "uavs": self.n_uavs,
            "horizon_s": self.horizon,
            "generated_tasks": len(arrived),
            "completed_tasks": len(completed),
            "completed_on_time": len(on_time),
            "completed_late": len(late),
            "unfinished_tasks": len(arrived) - len(completed),
            "sla_evaluable_tasks": len(evaluable),
            "sla_satisfaction_rate": div(len(on_time), len(evaluable)),
            "completion_rate": div(len(completed), len(arrived)),
            "mean_completed_latency_s": (round(sum(completed_latencies)/len(completed_latencies),6)
                                          if completed_latencies else None),
            "p95_completed_latency_s": p95(completed_latencies),
            "jain_on_time_completion_fairness": round(fairness, 6) if fairness is not None else None,
            "class_sla": classes,
            "peak_network_use_mbps": network_peak,
            "max_edge_concurrency": self.peak_compute_concurrency["edge"],
            "max_cloud_concurrency": self.peak_compute_concurrency["cloud"],
            "edge_compute_slot_utilization": round(
                self.compute_busy_slot_seconds["edge"] / (self.horizon * self.edge_slots), 6),
            "cloud_compute_slot_utilization": round(
                self.compute_busy_slot_seconds["cloud"] / (self.horizon * self.cloud_slots), 6),
            "local_compute_slot_utilization": round(
                self.compute_busy_slot_seconds["local"] / (self.horizon * self.n_uavs), 6),
        }


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected Phase 3A config schema_version: 1")
    return config


def _write_csv(path: Path, data: list[dict[str, Any]]) -> None:
    if not data:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def execute(config_path: Path, output: Path, policy: str, overwrite: bool = False) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = load_config(config_path)
    sim = NOMSSim(config)
    summary = sim.run(policy)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite nonempty results directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "service_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in sim.events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    _write_csv(output / "shared_network_allocation.csv", sim.network_history)
    _write_csv(output / "per_uav_network_allocation.csv", sim.per_uav_network_history)
    _write_csv(output / "compute_resource_usage.csv", sim.compute_history)
    _write_csv(output / "generated_task_workloads.csv", sim.job_definitions)
    (output / "sla_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "network_trace_sha256": (hashlib.sha256(sim.trace_path.read_bytes()).hexdigest()
                                  if sim.trace_path is not None else None),
        "network_trace_path": str(sim.trace_path) if sim.trace_path is not None else None,
        "network_trace_provenance": sim.trace_provenance,
        "network_capacity_scale": sim.network_capacity_scale,
        "python": platform.python_version(),
        "scenario_seed": config.get("seed"),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": _git_commit(),
        "profile_provenance": summary["profile_provenance"],
        "profile_details": sim.profile_provenance,
        "generated_task_workloads_sha256": sha256_file(output / "generated_task_workloads.csv"),
        "note": ("Network provenance: " + sim.trace_provenance
                 + ". Uncalibrated radio inputs and unmeasured onboard/cloud tiers: "
                 "NOT validated end-to-end agricultural service results."),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PROFILED_WORKLOAD_INTEGRATION: PASS" if sim.profile_source != "synthetic_smoke_only"
          else ("PHASE3C_TRACE_CONTRACT: PASS" if sim.trace_rows is not None
                else "PHASE3A_SIMULATION: PASS"))
    print(json.dumps(summary, indent=2))
    print(f"RESULTS_WRITTEN: {output.resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke"])
    parser.add_argument("--config", type=Path, default=Path("configs/orchestration_contract_smoke.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/orchestration_contract_smoke_edf"))
    parser.add_argument("--policy", choices=["edf", "fifo", "local_only", "edge_only", "cloud_only",
                                             "network_aware_edf", "critical_first"], default="edf")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    execute(args.config, args.output, args.policy, overwrite=args.overwrite)


if __name__ == "__main__":
    main()


# ---- Phase 4: decentralized UAV actions, centralized-training observations ----
# This wrapper changes no transport/compute mechanics in NOMSSim.
# The reference workload/radio scenarios remain uncalibrated research scaffolding.
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None  # Raw simulator/tests can run without optional Gymnasium import.
    spaces = None

import copy
import numpy as np


class NOMSOrchestrationEnv(gym.Env if gym is not None else object):
    """Fixed-shape multi-UAV joint-action Gymnasium environment.

    All actors see ONLY own UAV task metadata, current link observations and
    observable infrastructure summaries. A centralized critic additionally
    sees aggregated current-state resource occupancy. Future radio trace
    samples, simulator RNG state, ground-truth completion, and global job
    definitions are NOT in actor observations.

    Joint action: N x 4 [task (0=defer, 1..K=deadline candidates),
    placement (local/edge/cloud), network tier (low/standard/high),
    compute tier (low/standard/high)]. Conditional action masks depend on
    the task selection. Unused heads must use the canonical standard action.
    """

    metadata = {"render_modes": []}
    SERVICE_CLASSES = ("critical", "standard", "batch")
    PLACEMENTS = ("local", "edge", "cloud")
    LEVELS = ("low", "standard", "high")
    BASE_WIDTH = 10
    TASK_WIDTH = 10
    GLOBAL_WIDTH = 7

    def __init__(self, config: dict[str, Any], top_k: int = 3,
                 sla_targets: dict[str, float] | None = None,
                 multipliers: dict[str, float] | None = None):
        self.base_config = copy.deepcopy(config)
        self.n_uavs = int(config["n_uavs"])
        self.top_k = int(top_k)
        if self.top_k < 1 or self.n_uavs < 1:
            raise ValueError("top_k and n_uavs must be positive")
        self.local_width = self.BASE_WIDTH + self.top_k * self.TASK_WIDTH
        self.central_width = self.n_uavs*self.local_width + self.GLOBAL_WIDTH
        self.sla_targets = dict(sla_targets or {"critical": .95, "standard": .8, "batch": .6})
        self.multipliers = dict(multipliers or {"critical": 1., "standard": 1., "batch": 1.})
        # Research v5: training-only reward contract, defaulting to EXACT v3 reward.
        design = config.get("reward_design", {"mode": "legacy"})
        if not isinstance(design, dict) or design.get("mode", "legacy") not in ("legacy", "class_weighted", "constrained_balanced"):
            raise ValueError("reward_design.mode must be legacy, class_weighted or constrained_balanced")
        self.reward_design = copy.deepcopy(design)
        if self.reward_design.get("mode", "legacy") == "class_weighted":
            import math
            for name in ("on_time_weights", "miss_weights"):
                values = self.reward_design.get(name)
                if not isinstance(values, dict) or set(values) != set(self.SERVICE_CLASSES):
                    raise ValueError(f"reward_design.{name} must cover critical/standard/batch")
                if any(not math.isfinite(float(x)) or float(x) < 0 for x in values.values()):
                    raise ValueError(f"reward_design.{name} weights must be finite and nonnegative")
        if self.reward_design.get("mode") == "constrained_balanced":
            import math
            credit = self.reward_design.get("on_time_weights")
            if not isinstance(credit, dict) or set(credit) != set(self.SERVICE_CLASSES):
                raise ValueError("constrained_balanced requires on_time_weights for each service class")
            if any(not math.isfinite(float(v)) or float(v) <= 0 for v in credit.values()):
                raise ValueError("constrained_balanced on-time weights must be positive and finite")
            if any(float(self.multipliers.get(s, 0.)) != 0. for s in ("standard", "batch")):
                raise ValueError("constrained_balanced cannot impose hidden noncritical dual constraints")
            if self.sla_targets.get("critical") is None:
                raise ValueError("constrained_balanced requires critical SLA target")
        if any(not 0 <= v <= 1 for v in self.sla_targets.values()):
            raise ValueError("SLA targets must be between zero and one")
        if any(v < 0 for v in self.multipliers.values()):
            raise ValueError("Constraint multipliers must be nonnegative")
        self.sim: NOMSSim | None = None
        self._candidate_ids: dict[int, list[str]] = {}
        self._last_observation: dict[str, np.ndarray] | None = None
        if spaces is not None:
            self.action_space = spaces.MultiDiscrete(
                np.tile(np.asarray([self.top_k+1, 3, 3, 3], dtype=np.int64), (self.n_uavs, 1)))
            self.observation_space = spaces.Dict({
                "local_features": spaces.Box(-5., 5., shape=(self.n_uavs, self.local_width), dtype=np.float32),
                "global_state": spaces.Box(-5., 5., shape=(self.central_width,), dtype=np.float32),
                "task_mask": spaces.MultiBinary((self.n_uavs, self.top_k+1)),
                "placement_mask": spaces.MultiBinary((self.n_uavs, self.top_k+1, 3)),
            })

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            super().reset(seed=seed)
        cfg = copy.deepcopy(self.base_config)
        if seed is not None:
            cfg["seed"] = int(seed)
        self.sim = NOMSSim(cfg)
        obs = self._observe()
        return obs, {"profile_provenance": self.sim.profile_provenance["source_class"],
                     "network_provenance": self.sim.trace_provenance}

    @staticmethod
    def _bounded(x: float, scale: float = 1.) -> float:
        return float(max(-5., min(5., x/scale)))

    def _observe(self) -> dict[str, np.ndarray]:
        assert self.sim is not None
        sim = self.sim
        obs = sim.observations()
        local = np.zeros((self.n_uavs, self.local_width), dtype=np.float32)
        task_mask = np.zeros((self.n_uavs, self.top_k+1), dtype=np.int8)
        placement_mask = np.zeros((self.n_uavs, self.top_k+1, 3), dtype=np.int8)
        self._candidate_ids = {}
        task_mask[:, 0] = 1  # defer always valid
        placement_mask[:, 0, 0] = 1
        for i, o in obs.items():
            waiting = sorted(o["waiting_tasks"], key=lambda j:
                             (j["remaining_deadline_s"], j["task_id"]))[:self.top_k]
            self._candidate_ids[i] = [j["task_id"] for j in waiting]
            local[i, :self.BASE_WIDTH] = [
                self._bounded(sim.time_s, sim.horizon),
                self._bounded(i, max(1, self.n_uavs)),
                float(o["local_busy"]),
                self._bounded(o["observed_uav_link_cap_mbps"], 25.),
                self._bounded(o["observed_shared_capacity_mbps"], 25.),
                self._bounded(o["observed_edge_busy_slots"], sim.edge_slots),
                self._bounded(o["observed_cloud_busy_slots"], sim.cloud_slots),
                self._bounded(o["edge_queue_length"], 10.),
                self._bounded(o["cloud_queue_length"], 10.),
                self._bounded(len(o["waiting_tasks"]), 10.),
            ]
            reachable = o["observed_shared_capacity_mbps"] > EPS and o["observed_uav_link_cap_mbps"] > EPS
            for rank, j in enumerate(waiting):
                k = 1+rank
                job = sim.jobs[j["task_id"]]
                cls = j["service_class"]
                category = [float(cls == s) for s in self.SERVICE_CLASSES]
                start = self.BASE_WIDTH + rank*self.TASK_WIDTH
                local[i,start:start+self.TASK_WIDTH] = [
                    1.,
                    self._bounded(j["remaining_deadline_s"], sim.horizon),
                    self._bounded(j["input_mb"], 10.),
                    self._bounded(j["estimated_local_runtime_s"], 2.),
                    self._bounded(j["estimated_edge_runtime_s"], .5),
                    self._bounded(j["estimated_cloud_runtime_s"], .5),
                    *category,
                    self._bounded(sim.time_s-job.arrival_s, sim.horizon),
                ]
                placement_mask[i,k] = [int(not o["local_busy"]),int(reachable),int(reachable)]
                if placement_mask[i,k].any():
                    task_mask[i,k] = 1
        # Centralized critic gets current global coordination signals only.
        extras = np.asarray([
            self._bounded(len(sim.uploading), max(1, self.n_uavs)),
            self._bounded(len(sim.remote_queues["edge"]), 10.),
            self._bounded(len(sim.remote_queues["cloud"]), 10.),
            self._bounded(len(sim.remote_running["edge"]), sim.edge_slots),
            self._bounded(len(sim.remote_running["cloud"]), sim.cloud_slots),
            self._bounded(sum(map(len,sim.waiting.values())), 20.),
            self._bounded(sim.link_capacity_mbps(), 25.),
        ], dtype=np.float32)
        central = np.concatenate([local.reshape(-1), extras]).astype(np.float32)
        answer = {"local_features": local, "global_state": central,
                  "task_mask": task_mask, "placement_mask": placement_mask}
        self._last_observation = answer
        return answer

    def step(self, action):
        if self.sim is None or self._last_observation is None:
            raise RuntimeError("Call reset() before step()")
        sim = self.sim
        ints = np.asarray(action)
        if ints.shape != (self.n_uavs,4) or not np.issubdtype(ints.dtype, np.integer):
            raise ValueError("Expected integer joint action with shape (n_uavs,4)")
        mask = self._last_observation
        actions = {}
        for i, (task_index, placement, network, compute) in enumerate(ints):
            if not 0 <= task_index <= self.top_k or not mask["task_mask"][i,task_index]:
                raise ValueError(f"Invalid task action for UAV {i}")
            if not 0 <= placement < 3 or not mask["placement_mask"][i,task_index,placement]:
                raise ValueError(f"Invalid placement for UAV {i}")
            if not 0 <= network < 3 or not 0 <= compute < 3:
                raise ValueError("Invalid network or compute tier")
            if (task_index == 0 or placement == 0) and (network != 1 or compute != 1):
                raise ValueError("Inactive network/compute heads must select standard tier")
            jid = self._candidate_ids[i][task_index-1] if task_index else None
            actions[i] = Action(jid, self.PLACEMENTS[placement],
                                self.LEVELS[network], self.LEVELS[compute])
        beginning = sim.time_s
        start_idx = len(sim.events)
        sim.step(actions)
        new_events = sim.events[start_idx:]
        successes = {s:0 for s in self.SERVICE_CLASSES}
        violations = {s:0 for s in self.SERVICE_CLASSES}
        for event in new_events:
            if event["event"] == "completed" and event.get("on_time", False):
                cls = sim.jobs[event["task_id"]].service_class
                if cls in successes:
                    successes[cls] += 1
        for job in sim.jobs.values():
            if (job.arrival_s < sim.horizon and
                beginning < job.deadline_s <= sim.time_s+EPS and
                (job.completed_s is None or job.completed_s > job.deadline_s+EPS) and
                job.service_class in violations):
                violations[job.service_class] += 1
        n = max(1, len(sim.jobs))
        # Compact constrained objective: positive on-time completions,
        # adaptive SLA-violation multipliers, small transfer-cost proxy.
        # ENERGY HAS NOT BEEN MEASURED and is not used in the reward.
        if self.reward_design.get("mode", "legacy") == "class_weighted":
            good = self.reward_design["on_time_weights"]
            bad = self.reward_design["miss_weights"]
            reward = (sum(float(good[s])*count for s,count in successes.items())
                      - sum((float(bad[s])+self.multipliers.get(s, 0.))*count
                            for s,count in violations.items())) / n
        elif self.reward_design.get("mode") == "constrained_balanced":
            # Lagrangian for maximum class-balanced on-time throughput subject
            # to critical-class SLA. The critical-only dual applies to its
            # per-class miss fraction: E[S_critical] >= target, not guaranteed
            # per episode. The constant -lambda*(1-target) is omitted because
            # it has no policy gradient and varies independently of actions.
            credits = self.reward_design["on_time_weights"]
            n_critical = max(1, sum(j.arrival_s < sim.horizon and
                                    j.service_class == "critical" for j in sim.jobs.values()))
            reward = (sum(float(credits[s])*count for s,count in successes.items()) / n
                      - self.multipliers.get("critical",0.) * violations["critical"] / n_critical)
        else:
            # Keep the historical reward byte-for-byte equivalent in legacy mode.
            reward = ((sum(successes.values()) - sum(self.multipliers.get(s,0.)*count
                       for s,count in violations.items())) / n)
        delta_mb = sim.network_history[-1]["transmitted_mb"]
        reward -= .005 * delta_mb / n  # explicit communication load penalty
        obs = self._observe()
        terminated = sim.time_s >= sim.horizon-EPS
        info = {"on_time_new":successes,"deadline_violations_new":violations,
                "transmitted_mb":delta_mb}
        if terminated:
            info["episode_metrics"] = sim.summary("mappo")
        return obs, float(reward), terminated, False, info

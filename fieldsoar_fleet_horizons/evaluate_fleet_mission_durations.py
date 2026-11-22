"""Matched frozen-policy duration evaluation at 4, 8, 16, 32, and 64 UAVs."""
import argparse
import csv
import functools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import yaml

HORIZONS = (3, 5, 10, 20, 30)
FLEETS = (4, 8, 16, 32, 64)
FROZEN = Path(".fieldvision_audit_v14_2/0a8ac5415374654c")
FIELDSOAR_ROOT = Path("results/noms_h123_lr3_v8")
BASELINES_ROOT = Path("results/noms_learning_baselines_v9")
MAPPO_TUNE = Path("configs/mappo_h123_lr3_weighted_v8.yaml")
BASELINES_TUNE = Path("configs/learning_baselines_v9.yaml")
STRESS_PLAN = Path("configs/h3_independent_mobility_seed41_lr3_v8.yaml")
CLASSES = ("overall", "critical", "standard", "batch")


def base_config_path(fleet):
    if fleet == 64:
        return Path("configs/noms_fleet_expansion_v4/fleet_64uav_base.yaml")
    return Path(f"results/noms_fleet_scaling_baselines_v1/fleet_configs/fleet_{fleet}uav_base.yaml")


def original_radio_path(fleet, seed):
    if seed == 27:
        base = yaml.safe_load(base_config_path(fleet).read_text())
        return Path(base["network"]["trace_csv"])
    plan = yaml.safe_load(STRESS_PLAN.read_text())
    paths = {case["network_trace_template"].format(uavs=fleet) for case in plan["heldout_scenarios"]}
    if len(paths) != 1:
        raise ValueError("Expected one independent mobility trace for all stress scenarios")
    return Path(paths.pop())

def periodic_workload(loader):
    @functools.lru_cache(maxsize=8)
    def load_period(key):
        return loader(json.loads(key))

    def load(config):
        horizon = float(config["duration_s"])
        short = {**config, "duration_s": 3.0}
        key = {k: short[k] for k in ("seed","n_uavs","tick_s","duration_s","profiled_workloads")}
        original, provenance = load_period(json.dumps(key, sort_keys=True))
        jobs = []
        for period in range(math.ceil(horizon / 3.0)):
            offset = 3 * period
            for job in original:
                arrival = round(job["arrival_s"] + offset, 9)
                if arrival >= horizon:
                    continue
                jobs.append({
                    **job, "arrival_s": arrival,
                    "deadline_s": round(job["deadline_s"] + offset, 9),
                    "job_id": job["job_id"] if period == 0 else f"p{period:02d}_{job['job_id']}",
                })
        return jobs, {**provenance, "arrival_source": "repeated_seeded_3s_profiled_workload"}
    return load

def make_prefix(trace, horizon, tick, output, sha):
    out = output / f"{trace.stem}_prefix_{horizon}s.csv"
    with trace.open(newline="") as inp, out.open("x", newline="") as dest:
        reader = csv.reader(inp)
        writer = csv.writer(dest)
        writer.writerow(next(reader))
        for _ in range(round(horizon / tick)):
            writer.writerow(next(reader))
    manifest = json.loads(trace.with_suffix(".manifest.json").read_text())
    manifest["assumptions"]["duration_s"] = horizon
    manifest["parent_trace_sha256"] = sha(trace)
    manifest["generated_trace_sha256"] = sha(out)
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return out

def metrics(sim, start=0.0):
    due = [j for j in sim.jobs.values()
           if start <= j.arrival_s < sim.horizon and j.deadline_s <= sim.horizon + 1e-9]
    row = {}
    for cls in CLASSES:
        subset = due if cls == "overall" else [j for j in due if j.service_class == cls]
        on_time = sum(j.completed_s is not None and j.completed_s <= j.deadline_s + 1e-9 for j in subset)
        row[f"{cls}_due"] = len(subset)
        row[f"{cls}_on_time"] = on_time
        row[f"{cls}_sla"] = on_time / len(subset) if subset else None
    return row

def write_csv(path, rows):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet", type=int, choices=FLEETS, required=True)
    parser.add_argument("--nominal-trace", type=Path, required=True)
    parser.add_argument("--stress-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--pilot", action="store_true", help="First case and first model seed for each protocol")
    args = parser.parse_args()

    sys.path.insert(0, str(FROZEN.resolve(strict=True)))
    import fieldvision_noms_environment as env_module
    import run_orchestration_experiments as mappo
    import run_learning_baselines_v9 as baselines

    class HorizonEnv(env_module.NOMSOrchestrationEnv):
        def _observe(self):
            obs = super()._observe()
            sim = self.sim
            local = obs["local_features"]
            if sim.horizon != 3.0:
                # Repeat the training-time phase feature, while simulator time remains continuous.
                local[:, 0] = self._bounded((round(sim.time_s / sim.dt) % round(3 / sim.dt)) * sim.dt, 3.0)
                for uav, candidates in self._candidate_ids.items():
                    for rank, job_id in enumerate(candidates):
                        job = sim.jobs[job_id]
                        pos = self.BASE_WIDTH + rank * self.TASK_WIDTH
                        local[uav, pos + 1] = self._bounded(job.deadline_s - sim.time_s, 3.0)
                        local[uav, pos + 9] = self._bounded(sim.time_s - job.arrival_s, 3.0)
                obs["global_state"][:local.size] = local.reshape(-1)
            return obs

    class CriticalLocalFirst(env_module.NOMSSim):
        def scripted_actions(self, policy="critical_local_first"):
            actions = {}
            for uav, queue in self.waiting.items():
                if not queue or uav in self.local_running:
                    continue
                critical = [jid for jid in queue if self.jobs[jid].service_class == "critical"]
                job = min(critical, key=lambda j: (self.jobs[j].deadline_s,j)) if critical else queue[0]
                actions[uav] = env_module.Action(job, "local")
            return actions

    base_path = base_config_path(args.fleet)
    base = env_module.load_config(base_path)
    if base["duration_s"] != 3.0 or base["n_uavs"] != args.fleet:
        raise ValueError(f"Unexpected mission duration or fleet size in {base_path}")
    tick = float(base["tick_s"])
    sha = env_module.sha256_file
    full_traces = {"nominal": args.nominal_trace.resolve(strict=True),
                   "stress": args.stress_trace.resolve(strict=True)}
    provenance = {p: mappo._fleet_trace_preflight(
        trace, args.fleet, 30.0, tick, expected_shared_bottleneck=base["network"]["bottleneck_mbps"]
    ) for p, trace in full_traces.items()}
    if provenance["nominal"]["mobility_sha256"] == provenance["stress"]["mobility_sha256"]:
        raise ValueError("Nominal and stress require distinct native mobility captures")
    nominal_plan = yaml.safe_load(MAPPO_TUNE.read_text())
    stress_plan = yaml.safe_load(STRESS_PLAN.read_text())

    policies = {"critical_local_first": [(None, None)]}
    selections = {}
    _, torch, _, _, _ = mappo._import_rl()
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but is unavailable")
    for method, variant, runner, tuning_path, archive in (
        ("fieldsoar", "full", mappo, MAPPO_TUNE, FIELDSOAR_ROOT),
        ("ippo", "ippo", baselines, BASELINES_TUNE, BASELINES_ROOT),
        ("qmix", "qmix", baselines, BASELINES_TUNE, BASELINES_ROOT),
    ):
        training_dir = archive / f"fleet{args.fleet}" / f"{variant}_training"
        tuning = yaml.safe_load(tuning_path.read_text())
        selection_path = training_dir / "selected_model.json"
        selection = json.loads(selection_path.read_text())
        for field, path in (
            ("base_config_sha256", base_path), ("tuning_config_sha256", tuning_path),
            ("runner_sha256", Path(runner.__file__)),
            ("environment_sha256", Path(env_module.__file__)),
        ):
            if selection[field] != sha(path):
                raise ValueError(f"{selection_path} does not match {path}")
        if selection["policy_variant"] != variant or selection["chosen_config"]["episodes"] != 900:
            raise ValueError(f"Expected an original 900-episode {variant} selection in {selection_path}")
        trials = sorted(selection["selected_training_trials"], key=lambda row: int(row["train_seed"]))
        if [int(row["train_seed"]) for row in trials] != [11, 23, 37]:
            raise ValueError(f"Incomplete training-seed coverage in {selection_path}")
        policies[method] = []
        selections[method] = []
        for trial in trials[:1] if args.pilot else trials:
            seed = int(trial["train_seed"])
            checkpoint = Path(trial["path"]) / "best_validation_checkpoint.pt"
            if not checkpoint.is_file():
                checkpoint = training_dir / Path(trial["path"]).name / checkpoint.name
            policy = runner._load_selected_policy(base, tuning, checkpoint, device=args.device)
            policies[method].append((seed, policy))
            selections[method].append({"training_seed":seed, "checkpoint":str(checkpoint),
                                       "sha256":sha(checkpoint), "trial":trial})

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"manifest.json").write_text(json.dumps({
        "traces":provenance, "trace_paths":{p:str(t) for p,t in full_traces.items()},
        "fleet_uavs":args.fleet, "base_sha256":sha(base_path), "nominal_plan_sha256":sha(MAPPO_TUNE),
        "stress_plan_sha256":sha(STRESS_PLAN), "evaluation_script_sha256":sha(Path(__file__)),
        "selections":selections, "horizons_s":HORIZONS, "pilot":args.pilot,
        "arrivals":"repeat the original seed-specific three-second workload with continuous queues",
        "outages":"repeat original three-second outage windows",
        "time_features":"three-second phase and original deadline/age scales",
        "sla_denominator":"tasks with deadlines due by mission end",
    },indent=2))
    rows = []
    env_module.load_profiled_jobs = periodic_workload(env_module.load_profiled_jobs)
    for protocol, plan in (("nominal",nominal_plan),("stress",stress_plan)):
        prefix_dir = args.output / f"{protocol}_trace_prefixes"
        prefix_dir.mkdir()
        prefixes = {h: make_prefix(full_traces[protocol],h,tick,prefix_dir,sha)
                    for h in HORIZONS if h < 30}
        prefixes[30] = full_traces[protocol]
        cases = [(s,int(seed)) for s in plan["heldout_scenarios"] for seed in plan["heldout"]["seeds"]]
        if args.pilot:
            cases = cases[:1]
        for scenario, seed in cases:
            # Original scenario config applies slowdown, workload mix and resource stress.
            cfg = mappo.scenario_config(base,scenario)
            cfg["seed"] = seed
            windows = cfg["network"].get("outage_windows_s",[])
            for h in HORIZONS:
                cfg["duration_s"] = float(h)
                cfg["network"]["trace_csv"] = str(prefixes[h].resolve())
                cfg["network"]["trace_provenance"] = provenance[protocol]["network_provenance"]
                cfg["network"]["outage_windows_s"] = [
                    [round(3*i+a,9), round(min(h,3*i+b),9)]
                    for i in range(math.ceil(h/3)) for a,b in windows if 3*i+a < h
                ]
                for method, models in policies.items():
                    for train_seed, policy in models:
                        if policy is None:
                            sim = CriticalLocalFirst(cfg)
                            sim.run("critical_local_first")
                        else:
                            env = HorizonEnv(cfg, top_k=policy.k)
                            obs,_ = env.reset(seed=seed)
                            terminated = False
                            while not terminated:
                                action,*_ = policy.predict(obs,training=False)
                                obs,_,terminated,truncated,_ = env.step(action)
                                if truncated:
                                    raise RuntimeError("Unexpected truncation")
                            sim = env.sim
                        for window,start in (("whole",0.0),("final_3s",h-3.0)):
                            rows.append({"fleet_uavs":args.fleet,"protocol":protocol,"scenario":scenario["name"],"task_seed":seed,
                                         "horizon_s":h,"method":method,"training_seed":train_seed,
                                         "window":window,"generated_jobs":len(sim.jobs),**metrics(sim,start)})
            print(f"Completed {protocol} {scenario['name']} seed {seed}",flush=True)
    write_csv(args.output/"episodes.csv",rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in ("protocol","horizon_s","method","window"))].append(row)
    summary = []
    for (protocol,h,method,window), values in sorted(grouped.items()):
        record = {"fleet_uavs":args.fleet,"protocol":protocol,"horizon_s":h,"method":method,"window":window,"episodes":len(values)}
        for cls in CLASSES:
            scores = [row[f"{cls}_sla"] for row in values if row[f"{cls}_sla"] is not None]
            record[f"{cls}_sla_pct"] = round(100*mean(scores),4) if scores else None
        summary.append(record)
    write_csv(args.output/"summary.csv",summary)
    print(f"Saved {args.output/'summary.csv'}")

if __name__ == "__main__":
    main()

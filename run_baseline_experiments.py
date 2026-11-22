"""Train and evaluate the 10,000-episode baselines for FieldSOAR Tables II and VI."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tools.run_archived_experiment import load_runner

FLEETS = (4, 8, 16, 32, 64, 128)
METHODS = ("ippo", "qmix")
PROTOCOLS = ("nominal", "independent")
SEEDS = (11, 23, 37)
EPISODES = 10000
FROZEN = Path(".fieldvision_audit_v14_2/0a8ac5415374654c")
RUNNER = FROZEN / "run_learning_baselines_v9.py"
ORIGINAL_TUNING = Path("configs/learning_baselines_v9.yaml")
H3 = Path("configs/h3_independent_mobility_seed41_lr3_v8.yaml")
DEFAULT_128 = Path("results/noms_optuna_128_ep3000/20261120T005059Z")
METRICS = ("overall_sla", "critical_sla", "standard_sla", "batch_sla", "fairness")
SCOPES = {
    "nominal": "H1_H2_HELDOUT_UNCALIBRATED_NETWORK_SENSITIVITY",
    "independent": "H3_INDEPENDENT_MOBILITY_EVALUATION",
}


def read_json(path):
    return json.loads(path.read_text())


def read_yaml(path):
    return yaml.safe_load(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def base_path(fleet):
    if fleet == 128:
        return Path("configs/noms_scale_v10/fleet_128uav_base.yaml")
    if fleet == 64:
        return Path("configs/noms_fleet_expansion_v4/fleet_64uav_base.yaml")
    return Path(f"results/noms_fleet_scaling_baselines_v1/fleet_configs/fleet_{fleet}uav_base.yaml")


def eval_dir(root, fleet, protocol):
    archive = root if fleet < 128 else Path("results/noms_scale_v10")
    return archive / f"fleet{fleet}/full_{protocol}_evaluation"


def evaluation_keys(path, protocol):
    plan = read_yaml(H3 if protocol == "independent" else path)
    return {
        (case["name"], int(seed), train)
        for case in plan["heldout_scenarios"]
        for seed in plan["heldout"]["seeds"]
        for train in SEEDS
    }


def load_eval(directory, method, protocol, base_hash, expected):
    summary = read_json(directory / "heldout_summary.json")
    selection = summary["selected_model"]
    ensure(summary["policy_variant"] == method and summary["scope"] == SCOPES[protocol]
           and selection["base_config_sha256"] == base_hash and summary["training_seed_count"] == len(SEEDS),
           f"Held-out provenance mismatch in {directory}")
    if protocol == "independent":
        ensure(summary["external_evaluation_plan_sha256"] == sha(H3), f"H3 plan mismatch in {directory}")
    with (directory / "heldout_mappo_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(row["scenario"], int(row["seed"]), int(row["training_seed"])) for row in rows}
    ensure(keys == expected and len(keys) == len(rows) == summary["sample_size"],
           f"Missing, extra, or duplicated held-out evaluations in {directory}")
    return summary, rows


def runner_module():
    return load_runner(RUNNER.stem)


def prepare(out, reference_root, baseline_root, anchor128):
    runner = runner_module()
    h3 = read_yaml(H3)
    original_tuning = read_yaml(ORIGINAL_TUNING)
    reference_tuning = read_yaml(Path("configs/mappo_h123_lr3_weighted_v8.yaml"))
    for key in ("heldout", "heldout_scenarios"):
        ensure(original_tuning[key] == reference_tuning[key], f"Nominal comparison differs in {key}")
    plan = {
        "runner_sha256": sha(RUNNER), "environment_sha256": sha(FROZEN / "fieldvision_noms_environment.py"),
        "h3_sha256": sha(H3), "episodes_per_training_seed": EPISODES, "cases": {},
    }
    tunings = {}
    for fleet in FLEETS:
        config_path = base_path(fleet)
        base = runner.load_config(config_path.resolve(strict=True))
        ensure(int(base["n_uavs"]) == fleet and float(base["duration_s"]) == 3.,
               f"Unexpected fleet or horizon in {config_path}")
        for scenario in h3["heldout_scenarios"]:
            runner.scenario_config(base, scenario)
        config_hash = sha(config_path)
        reference = {}
        for protocol in PROTOCOLS:
            directory = eval_dir(reference_root, fleet, protocol)
            archived, _ = load_eval(directory, "full", protocol, config_hash,
                                    evaluation_keys(ORIGINAL_TUNING, protocol))
            ensure(archived["selected_model"]["chosen_config"]["episodes"] == 900
                   and archived["selected_model"]["tuning_config_sha256"]
                   == sha(Path("configs/mappo_h123_lr3_weighted_v8.yaml")),
                   f"Expected the original 900-episode FieldSOAR comparison in {directory}")
            reference[protocol] = {
                "directory": str(directory),
                "summary_sha256": sha(directory / "heldout_summary.json"),
                "rows_sha256": sha(directory / "heldout_mappo_results.csv"),
            }
        for method in METHODS:
            name = f"fleet{fleet}/{method}"
            if fleet == 128:
                tuning_path = anchor128 / method / "tuning_ep3000.yaml"
                selection_path = anchor128 / method / "training/selected_model.json"
                anchor_episodes = 3000
            else:
                tuning_path = ORIGINAL_TUNING
                selection_path = baseline_root / f"fleet{fleet}/{method}_training/selected_model.json"
                anchor_episodes = 900
            original = read_json(selection_path)
            chosen = original["chosen_config"]
            ensure(original["policy_variant"] == method and chosen["episodes"] == anchor_episodes
                   and original["base_config_sha256"] == config_hash
                   and original["tuning_config_sha256"] == sha(tuning_path)
                   and sorted(row["train_seed"] for row in original["selected_training_trials"]) == list(SEEDS),
                   f"Wrong archived hyperparameter selection in {selection_path}")
            tuning = read_yaml(tuning_path)
            ensure(tuple(tuning["training"]["seeds"]) == SEEDS, f"Wrong training seeds in {tuning_path}")
            for key in ("heldout", "heldout_scenarios"):
                ensure(tuning[key] == original_tuning[key], f"Nominal data changed in {tuning_path}: {key}")
            tuning["grid"].update({
                "episodes": [EPISODES], "ppo_epochs": [chosen["ppo_epochs"]],
                "dropout": [chosen["dropout"]], "learning_rates": [chosen["learning_rate"]],
            })
            tuning_text = yaml.safe_dump(tuning, sort_keys=False)
            tunings[name] = tuning_text
            plan["cases"][name] = {
                "config": str(config_path), "config_sha256": config_hash,
                "tuning_sha256": hashlib.sha256(tuning_text.encode()).hexdigest(),
                "source_selection": str(selection_path), "source_tuning": str(tuning_path),
                "source_selection_sha256": sha(selection_path), "reference": reference,
            }
    out.mkdir(parents=True, exist_ok=False)
    for name, tuning_text in tunings.items():
        directory = out / name
        directory.mkdir(parents=True)
        (directory / "tuning.yaml").write_text(tuning_text)
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Prepared {len(tunings)} fleet/method combinations in {out}", flush=True)


def checked_case(out, plan, fleet, method):
    ensure(plan["episodes_per_training_seed"] == EPISODES
           and plan["runner_sha256"] == sha(RUNNER)
           and plan["environment_sha256"] == sha(FROZEN / "fieldvision_noms_environment.py")
           and plan["h3_sha256"] == sha(H3), "Runner, budget, or evaluation plan differs from the launch plan")
    name = f"fleet{fleet}/{method}"
    case = plan["cases"][name]
    directory = out / name
    tuning = directory / "tuning.yaml"
    config = Path(case["config"])
    ensure(sha(tuning) == case["tuning_sha256"] and sha(config) == case["config_sha256"],
           f"Base or tuning changed since preparation for {name}")
    return case, directory, tuning, config


def selected_model(directory, method, case):
    selection = read_json(directory / "training/selected_model.json")
    trials = selection["selected_training_trials"]
    ensure(selection["policy_variant"] == method and selection["chosen_config"]["episodes"] == EPISODES
           and selection["base_config_sha256"] == case["config_sha256"]
           and selection["tuning_config_sha256"] == case["tuning_sha256"]
           and sorted(row["train_seed"] for row in trials) == list(SEEDS)
           and all(row["episodes"] == EPISODES for row in trials),
           f"Incomplete or mismatched 10,000-episode sweep in {directory}")
    return selection


def run_stage(directory, name, command):
    # Never truncate a prior attempt. Official outputs are also non-overwriting.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with (directory / f"{name}_{stamp}.log").open("x", buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise


def run(out, fleets, methods, device):
    if device == "cuda":
        import torch
        ensure(torch.cuda.is_available(), "No CUDA device visible in the current allocation")
        print(f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    plan = read_json(out / "plan.json")
    for fleet in fleets:
        for method in methods:
            case, directory, tuning, config = checked_case(out, plan, fleet, method)
            training = directory / "training"
            common = [sys.executable, "-u", "tools/run_archived_experiment.py", RUNNER.stem, "--config", str(config),
                      "--tuning-config", str(tuning), "--device", device]
            if not (training / "selected_model.json").is_file():
                ensure(not training.exists(), f"Prior training attempt is incomplete at {training}; move it aside to retry")
                run_stage(directory, "training", common + [
                    "--mode", "mappo-sweep", "--variant", method, "--output-root", str(training)])
            selection = selected_model(directory, method, case)
            for protocol in PROTOCOLS:
                target = directory / protocol
                if not (target / "heldout_summary.json").is_file():
                    ensure(not target.exists(), f"Prior evaluation is incomplete at {target}; move it aside to retry")
                    command = common + ["--mode", "mappo-heldout", "--sweep-root", str(training),
                                        "--output-root", str(target)]
                    if protocol == "independent":
                        command.extend(["--evaluation-plan", str(H3)])
                    run_stage(directory, protocol, command)
                payload, _ = load_eval(target, method, protocol, case["config_sha256"], evaluation_keys(tuning, protocol))
                ensure(payload["selected_model"] == selection, f"Evaluation uses a different selection in {target}")
                print(f"Verified {fleet} UAVs / {method} / {protocol}", flush=True)


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(out):
    runner = runner_module()
    plan = read_json(out / "plan.json")
    points, gains = [], []
    for fleet in FLEETS:
        for method in METHODS:
            case, directory, tuning, _ = checked_case(out, plan, fleet, method)
            selection = selected_model(directory, method, case)
            for protocol in PROTOCOLS:
                expected = evaluation_keys(tuning, protocol)
                payload, rows = load_eval(directory / protocol, method, protocol, case["config_sha256"], expected)
                ensure(payload["selected_model"] == selection, f"Evaluation uses a different selection in {directory}")
                archived = case["reference"][protocol]
                reference_dir = Path(archived["directory"])
                for filename, key in (("heldout_summary.json", "summary_sha256"), ("heldout_mappo_results.csv", "rows_sha256")):
                    ensure(sha(reference_dir / filename) == archived[key], f"Original FieldSOAR reference changed: {reference_dir}")
                _, reference = load_eval(reference_dir, "full", protocol, case["config_sha256"], expected)
                summary = payload["summary"]
                for metric in METRICS:
                    ensure(math.isfinite(float(summary[metric])), f"Undefined {metric} in {directory}/{protocol}")
                points.append({
                    "uavs": fleet, "method": method, "condition": "stress" if protocol == "independent" else "nominal",
                    **{f"{m}_pct": f"{100 * float(summary[m]):.2f}" for m in METRICS[:-1]},
                    "jain": f"{float(summary['fairness']):.4f}", "evaluation_episodes": len(rows),
                    **{f"best_episode_seed_{row['train_seed']}": row["best_episode"]
                       for row in sorted(selection["selected_training_trials"], key=lambda row: row["train_seed"])},
                })
                for metric in ("overall_sla", "critical_sla"):
                    paired = runner._bootstrap_matched_deltas(reference, rows, metric)
                    gains.append({"uavs": fleet, "comparator": method, "condition": points[-1]["condition"],
                                  "metric": metric, **paired})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = out / f"report_{stamp}"
    dest.mkdir()
    write_csv(dest / "table_II_ippo_qmix.csv", points)
    write_csv(dest / "table_VI_ippo_qmix.csv", gains)
    print(f"Verified reports: {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("output", type=Path)
    prep.add_argument("--fieldsoar-root", type=Path, default=Path("results/noms_h123_lr3_v8"))
    prep.add_argument("--baseline-root", type=Path, default=Path("results/noms_learning_baselines_v9"))
    prep.add_argument("--anchor-128", type=Path, default=DEFAULT_128)
    train = subparsers.add_parser("run")
    train.add_argument("output", type=Path)
    train.add_argument("--fleet", type=int, nargs="+", choices=FLEETS, default=FLEETS)
    train.add_argument("--method", nargs="+", choices=METHODS, default=METHODS)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    summarize = subparsers.add_parser("report")
    summarize.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args.output, args.fieldsoar_root, args.baseline_root, args.anchor_128)
    elif args.operation == "run":
        run(args.output, args.fleet, args.method, args.device)
    else:
        report(args.output)


if __name__ == "__main__":
    main()

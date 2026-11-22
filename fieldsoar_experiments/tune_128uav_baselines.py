"""Validation-only IPPO/QMIX hyperparameter search and three-seed confirmation."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
from statistics import mean

import yaml

FROZEN = Path(".fieldvision_audit_v14_2/0a8ac5415374654c")
BASE = Path("configs/noms_scale_v10/fleet_128uav_base.yaml")
TUNING = Path("configs/learning_baselines_v9.yaml")
H3 = Path("configs/h3_independent_mobility_seed41_lr3_v8.yaml")

def configuration(template, algo, knobs):
    tune = copy.deepcopy(template)
    tune["learning_rate"] = knobs["learning_rate"]
    tune["grid"]["learning_rates"] = [knobs["learning_rate"]]
    if algo == "ippo":
        tune["grid"]["ppo_epochs"] = [knobs["ppo_epochs"]]
        tune["minibatch_steps"] = knobs["minibatch_steps"]
    else:
        for key in ("qmix_batch_size", "qmix_target_update_steps", "qmix_epsilon_final"):
            tune[key] = knobs[key]
    return tune

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--algorithms", nargs="+", choices=("ippo","qmix"), default=["ippo","qmix"])
    p.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--screen-episodes", type=int, default=600)
    p.add_argument("--confirm-episodes", type=int, default=1800)
    p.add_argument("--screen-only", action="store_true")
    args = p.parse_args()
    if min(args.trials,args.screen_episodes,args.confirm_episodes) < 1 or len(set(args.algorithms)) != len(args.algorithms):
        p.error("Use positive episode/trial counts and distinct algorithms")
    sys.path.insert(0,str(FROZEN.resolve(strict=True)))
    import optuna
    import torch
    import run_learning_baselines_v9 as runner

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested. Run inside a Slurm GPU allocation and check torch.cuda.is_available().")
    if args.device == "cuda":
        print("CUDA device:",torch.cuda.get_device_name(0),flush=True)
    template = yaml.safe_load(TUNING.read_text())
    base = runner.load_config(BASE)
    assert base["n_uavs"] == 128 and base["duration_s"] == 3.0
    args.output.mkdir(parents=True,exist_ok=False)
    for algo in args.algorithms:
        folder = args.output/algo
        folder.mkdir()
        study = optuna.create_study(
            direction="maximize", study_name=f"fieldsoar_{algo}_128",
            storage=f"sqlite:///{(folder/'optuna.db').resolve()}",
            sampler=optuna.samplers.TPESampler(seed=2027, n_startup_trials=3),
        )
        # The currently selected reference learning rate is an explicit search control.
        reference = (
            {"learning_rate":3e-4,"ppo_epochs":12,"minibatch_steps":64} if algo == "ippo"
            else {"learning_rate":5e-5,"qmix_batch_size":64,
                  "qmix_target_update_steps":250,"qmix_epsilon_final":0.05}
        )
        study.enqueue_trial(reference)
        train_once = runner._train_qmix_one if algo == "qmix" else runner.train_mappo_one

        def objective(trial):
            knobs = {"learning_rate":trial.suggest_float("learning_rate",3e-5,1e-3,log=True)}
            if algo == "ippo":
                knobs["ppo_epochs"] = trial.suggest_categorical("ppo_epochs",[6,12,18])
                knobs["minibatch_steps"] = trial.suggest_categorical("minibatch_steps",[32,64,128])
            else:
                knobs["qmix_batch_size"] = trial.suggest_categorical("qmix_batch_size",[32,64,128])
                knobs["qmix_target_update_steps"] = trial.suggest_categorical("qmix_target_update_steps",[125,250,500])
                knobs["qmix_epsilon_final"] = trial.suggest_categorical("qmix_epsilon_final",[0.02,0.05,0.10])
            tune = configuration(template,algo,knobs)
            tune["grid"]["episodes"] = [args.screen_episodes]
            trial_root = folder/f"trial_{trial.number:03d}"
            trial_root.mkdir()
            (trial_root/"tuning.yaml").write_text(yaml.safe_dump(tune,sort_keys=False))
            params = {"episodes":args.screen_episodes,"ppo_epochs":tune["grid"]["ppo_epochs"][0],
                      "dropout":tune["grid"]["dropout"][0],"hidden_size":tune.get("hidden_size",128),
                      "learning_rate":knobs["learning_rate"],"variant":algo,
                      "critic_arch":tune.get("critic_arch","flat")}
            start = time.perf_counter()
            scores = []
            for seed in (11,23):
                result = train_once(base,tune,trial_root/f"seed{seed}",params,seed,args.device)
                scores.append(float(result["best_validation_score"]))
            trial.set_user_attr("validation_scores_seed11_23", scores)
            trial.set_user_attr("elapsed_seconds", time.perf_counter()-start)
            return mean(scores)

        study.optimize(objective,n_trials=args.trials)
        best = study.best_trial
        (folder/"optuna_winner.json").write_text(json.dumps({
            "algorithm":algo,"trial":best.number,"validation_score":best.value,
            "screen_training_seeds":[11,23],"screen_episodes":args.screen_episodes,"params":best.params,
        },indent=2))
        final_tune = configuration(template,algo,best.params)
        final_tune["grid"]["episodes"] = [args.confirm_episodes]
        final_path = folder/"confirmation_tuning.yaml"
        final_path.write_text(yaml.safe_dump(final_tune,sort_keys=False))
        if args.screen_only:
            continue
        training = folder/"confirmation_training"
        runner.run_mappo_sweep(BASE,final_path,training,device=args.device,variant=algo)
        runner.evaluate_mappo_heldout(BASE,final_path,training,folder/"nominal",device=args.device)
        runner.evaluate_mappo_heldout(BASE,final_path,training,folder/"stress",
                                     device=args.device,evaluation_plan_path=H3)

if __name__ == "__main__":
    main()

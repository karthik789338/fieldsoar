#!/usr/bin/env python3
"""Route a diagnostic QMIX optimizer smoke test to the *unchanged* v9 QMIX trainer.

The v9 `--mode mappo-train` CLI only dispatches to PPO's train_mappo_one.
QMIX full sweeps already use `_train_qmix_one` correctly.  This adapter
preserves the byte-identical v9 runner used by existing scientific checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import yaml

from fieldvision_noms_environment import load_config
from run_learning_baselines_v9 import _train_qmix_one, _validate_experiment_setup


def diagnostic_qmix(config: Path, tuning_path: Path, output: Path,
                    device='cpu', episodes=12, seed=11):
    """Perform actual replay/TD optimization, not a dummy checkpoint."""
    if episodes < 1:
        raise ValueError('Diagnostic episodes must be positive')
    tuning = yaml.safe_load(tuning_path.read_text(encoding='utf-8'))
    _validate_experiment_setup(tuning)
    params = {
        'episodes':int(episodes),
        'ppo_epochs':1,  # schema-only field; QMIX never runs PPO epochs
        'dropout':0.0,
        'hidden_size':int(tuning.get('hidden_size',128)),
        'learning_rate':float(tuning.get('learning_rate',3e-4)),
        'variant':'qmix',
        'critic_arch':tuning.get('critic_arch','pooled'),
    }
    record = _train_qmix_one(load_config(config), tuning, output, params,
                             int(seed), device)
    with (output/'training_history.csv').open(newline='', encoding='utf-8') as f:
        history=list(csv.DictReader(f))
    losses=[float(row['td_loss']) for row in history if row.get('td_loss') not in (None,'')]
    if not losses or not all(math.isfinite(x) for x in losses):
        raise RuntimeError('QMIX smoke did not complete finite replay/TD optimizer updates')
    for filename in ('best_validation_checkpoint.pt','training_manifest.json','last_checkpoint.pt'):
        if not (output/filename).is_file():
            raise RuntimeError('QMIX smoke artifact missing: '+str(output/filename))
    print('QMIX_NATIVE_TD_SMOKE_PASS: '
          f'episodes={episodes} td_loss_updates={len(losses)} '
          f'last_td_loss={losses[-1]:.6g} output={output}', flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--tuning-config',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--episodes',type=int,default=12)
    parser.add_argument('--seed',type=int,default=11)
    args=parser.parse_args()
    diagnostic_qmix(args.config,args.tuning_config,args.output_root,
                    device=args.device,episodes=args.episodes,seed=args.seed)

if __name__=='__main__':
    main()

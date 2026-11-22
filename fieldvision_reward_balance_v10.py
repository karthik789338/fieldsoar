#!/usr/bin/env python3
"""Controlled 3-arm reward ablation on top of frozen FieldVision v8.

Requires the existing v8 runner/environment/configs on Hellbender. Intentionally
refuses unknown reward schemas or changed source files rather than assuming a
new reward was actually applied. Does not replace or patch any existing file.
"""
from __future__ import annotations
import argparse, copy, csv, datetime as dt, hashlib, json, math, os, pathlib, re, subprocess, sys

P = pathlib.Path
ROOT = P('results/noms_reward_balance_v10')
TEMPLATE = P('configs/mappo_h123_lr3_weighted_v8.yaml')
RUNNER = P('run_orchestration_experiments.py')
ENVIRONMENT = P('fieldvision_noms_environment.py')
BASE_OLD_RESULTS = P('results/noms_h123_lr3_v8')
FLEETS = (4, 8, 16, 32, 64)
ARMS = {
    'baseline':  {'success': (3.,1.,0.75), 'miss': (4.,1.,0.75)},
    'balanced_v1': {'success': (3.,1.5,1.), 'miss': (4.,1.5,1.)},
    'balanced_v2': {'success': (3.,2.,1.25), 'miss': (4.,2.,1.25)},
}
CLASSES = ('critical','standard','batch')
# NEW reserved heldout seeds, never v8/v9 known test seeds.
FRESH_NOMINAL_SEEDS = (12401,12413,12431,12451,12463)


def sha(path):
    return hashlib.sha256(P(path).read_bytes()).hexdigest()


def load_yaml(path):
    import yaml
    x=yaml.safe_load(P(path).read_text())
    if not isinstance(x,dict):
        raise ValueError(f'Expected YAML dictionary: {path}')
    return x


def dump_yaml(obj,path):
    import yaml
    path=P(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(obj,sort_keys=False),encoding='utf-8')


def fingerprint():
    for f in (TEMPLATE, RUNNER, ENVIRONMENT):
        if not f.is_file():
            raise FileNotFoundError(f'Expected existing v8 source/config at {f}; do not replace your v8 installation')
    src=ENVIRONMENT.read_text()
    runner=RUNNER.read_text()
    if 'reward_design' not in src or 'class_weighted' not in src or 'reward_design' not in runner:
        raise RuntimeError('Frozen v8 reward integration not detected in source; refusing to run an unverified reward experiment')
    return {'template_sha256':sha(TEMPLATE),'runner_sha256':sha(RUNNER),
            'environment_sha256':sha(ENVIRONMENT)}


def reward_section(config):
    r=config.get('reward_design')
    if not isinstance(r,dict):
        raise ValueError('Expected top-level reward_design mapping in v8 tuning YAML')
    return r


def walk_class_maps(obj,path=()):
    if isinstance(obj,dict):
        if set(CLASSES).issubset(obj.keys()) and all(isinstance(obj.get(k),(int,float)) and not isinstance(obj.get(k),bool)
                     for k in CLASSES):
            yield path,obj
        for key,v in obj.items():
            if isinstance(v,dict):
                yield from walk_class_maps(v,path+(str(key),))


def identify_maps(reward):
    """Infer by exact frozen numerical baseline; fail closed on schema differences.

    Success (3,1,.75) vs miss (4,1,.75) are different fingerprints,
    so keys need not be guessed and unrelated maps are left alone.
    """
    found={'success':[],'miss':[]}
    for path, m in walk_class_maps(reward):
        values=tuple(float(m[k]) for k in CLASSES)
        for category,expected in (('success',ARMS['baseline']['success']),('miss',ARMS['baseline']['miss'])):
            if all(math.isclose(a,b,rel_tol=0,abs_tol=1e-10) for a,b in zip(values,expected)):
                found[category].append((path,m))
    for category,candidates in found.items():
        if len(candidates)!=1:
            raise ValueError(f'Expected exactly one reward_design {category} class-weight mapping matching v8 coefficients; found {len(candidates)}. Run the audit command and inspect the actual YAML. No edits made.')
    return found['success'][0], found['miss'][0]


def make_config(template,arm):
    if arm not in ARMS:raise ValueError(f'Unknown reward arm {arm}')
    new=copy.deepcopy(template)
    reward=reward_section(new)
    (s_path, success),(m_path, miss)=identify_maps(reward)
    for cls,v in zip(CLASSES,ARMS[arm]['success']):success[cls]=v
    for cls,v in zip(CLASSES,ARMS[arm]['miss']):miss[cls]=v
    # Freeze the previously validated full-MAPPO LR at 3e-4 across all arms.
    # Reward design is the ONLY treatment variable between the new arms.
    if not isinstance(new.get('grid'),dict) or 3e-4 not in [float(x) for x in new['grid'].get('learning_rates',[])]:
        raise ValueError('Expected v8 LR3 grid with 0.0003 present')
    new['grid']['learning_rates']=[0.0003]
    if 'learning_rate' in new:new['learning_rate']=0.0003
    # All training/validation cases remain frozen. Only HELDOUT task seeds change
    # to keep new reward study separate from already analyzed v8/v9 tests.
    if not isinstance(new.get('heldout'),dict) or 'seeds' not in new['heldout']:
        raise ValueError('Unexpected v8 heldout seed schema')
    old_seeds=tuple(int(x) for x in new['heldout']['seeds'])
    assert not set(FRESH_NOMINAL_SEEDS).intersection(old_seeds)
    new['heldout']['seeds']=list(FRESH_NOMINAL_SEEDS)
    if 'heldout_scenarios' not in new or not isinstance(new['heldout_scenarios'],list):
        raise ValueError('Missing v8 heldout_scenarios')
    for scenario in new['heldout_scenarios']:
        if 'name' not in scenario:raise ValueError('Missing scenario name')
        scenario['name']='reward_v10_'+str(scenario['name'])
    # v8 sees the same reward design in train/eval; annotate safely.
    new['experiment_note']=('V10 controlled prospectively fixed reward-design experiment: '
      +arm+'; ONLY standard and batch success/miss weights differ from frozen baseline; '
      +'critical=3/4; v8 full-MAPPO LR fixed at 0.0003 for all arms; '
      +'training seeds/validation scenarios unchanged; nominal heldout task seeds reserved 12401,12413,12431,12451,12463. '
      +'No H3 seed41 retesting to select a new reward.')
    return new,{'success_class_weights_key':'.'.join(s_path),
                'miss_class_weights_key':'.'.join(m_path),'old_heldout_seeds':old_seeds}


def paths(arm,fleet):
    return (ROOT/'configs'/f'{arm}_v10.yaml',
            ROOT/f'fleet{fleet}'/f'{arm}_training',
            ROOT/f'fleet{fleet}'/f'{arm}_nominal_evaluation')


def bases(fleet):
    if int(fleet)==64:
        return P('configs/noms_fleet_expansion_v4/fleet_64uav_base.yaml')
    return P(f'results/noms_fleet_scaling_baselines_v1/fleet_configs/fleet_{fleet}uav_base.yaml')


def prepare():
    info=fingerprint()
    template=load_yaml(TEMPLATE)
    cfg_paths={}
    maps={}
    for arm in ARMS:
        cfg, audit=make_config(template,arm)
        cfg_path=paths(arm,4)[0]
        if cfg_path.exists():
            if load_yaml(cfg_path)!=cfg:
                raise FileExistsError(f'Existing v10 config differs: {cfg_path}; preserve previous experiment, do not overwrite')
        else:dump_yaml(cfg,cfg_path)
        cfg_paths[arm]=str(cfg_path)
        maps[arm]=audit
    manifest=ROOT/'reward_study_manifest.json'
    # Ensure the v8 environment actually mentions both resolved reward keys.
    env_src=ENVIRONMENT.read_text()
    for category in ('success','miss'):
        leaf=maps['baseline'][category+'_class_weights_key'].split('.')[-1]
        if leaf not in env_src:
            raise RuntimeError('Reward mapping '+leaf+' not found in frozen environment source. Refusing silent no-op modification')
    payload={'schema':'FieldVision-NOMS-v10-controlled-reward-ablation', 'provenance':info,
             'arms':ARMS,'configs':{a:{'path':p,'sha256':sha(p)} for a,p in cfg_paths.items()},
             'reward_mapping':maps,'training_seeds':[11,23,37],
             'frozen_learning_rate':0.0003,
             'new_nominal_heldout_seeds':list(FRESH_NOMINAL_SEEDS),
             'do_not_use_h3_seed41_for_selection':True,
             'note':'Critical SLA >=95% is a validation constraint only, not a promise or full feasibility guarantee.'}
    payload=json.loads(json.dumps(payload))  # canonicalize tuples to JSON arrays for stable provenance
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old!=payload:raise RuntimeError('Experiment provenance has changed since prepare; restore original source/config or start a NEW study root.')
    else:
        manifest.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print('REWARD_V10_PREPARE_PASS')
    print('Critical unchanged (success/miss):',ARMS['baseline']['success'][0],ARMS['baseline']['miss'][0])
    for a in ARMS:print('ARM',a, 'success',ARMS[a]['success'],'miss',ARMS[a]['miss'],'config',cfg_paths[a])
    print('MANIFEST:',manifest)
    return payload


def check_prepared():
    info=prepare()
    return info


def runner_command(mode,base,tuning,out,sweep=None,device='cpu'):
    cmd=[sys.executable,str(RUNNER),'--mode',mode,'--config',str(base),
         '--tuning-config',str(tuning),'--output-root',str(out),'--device',str(device)]
    if sweep is None:cmd+=['--variant','full']
    else:cmd+=['--sweep-root',str(sweep)]
    return cmd


def do_train(arm,fleet,device):
    if arm not in ARMS or fleet not in FLEETS:raise ValueError('Invalid arm or fleet size')
    prepared=check_prepared()
    cfg,training,_=paths(arm,fleet)
    base=bases(fleet)
    if not base.is_file():raise FileNotFoundError(f'Missing existing frozen fleet base config: {base}')
    if training.exists():
        # Reject collisions; do not silently trust incomplete/past data.
        raise FileExistsError(f'Training folder already exists: {training}; inspect existing results or restore from a verified checkpoint before rerunning')
    training.parent.mkdir(parents=True,exist_ok=True)
    print('TRAIN_ARM:',arm,'FLEET:',fleet,'REWARD:',ARMS[arm],flush=True)
    subprocess.run(runner_command('mappo-sweep',base,cfg,training,device=device),check=True)
    selected=training/'selected_model.json'
    if not selected.is_file():raise RuntimeError('Training completed but no selected_model.json')
    x=json.loads(selected.read_text())
    if x.get('policy_variant')!='full':raise RuntimeError('Training was not full four-head MAPPO')
    if x.get('tuning_config_sha256')!=sha(cfg):raise RuntimeError('Training tuning-config provenance mismatch')
    if x.get('base_config_sha256')!=sha(base):raise RuntimeError('Base configuration provenance mismatch')
    if len(x.get('selected_training_trials',[]))!=3:raise RuntimeError('Three selected training seeds expected')
    print('TRAIN_VERIFIED:',selected)


def do_evaluate(arm,fleet,device):
    check_prepared()
    cfg,training,out=paths(arm,fleet)
    if not (training/'selected_model.json').is_file():raise FileNotFoundError(f'Missing trained model {training}')
    if out.exists():raise FileExistsError(f'Existing evaluation {out}; no overwrite allowed')
    base=bases(fleet)
    subprocess.run(runner_command('mappo-heldout',base,cfg,out,sweep=training,device=device),check=True)
    if not (out/'heldout_summary.json').is_file():raise RuntimeError('Missing heldout summary')
    print('FRESH_HELDOUT_EVALUATION_VERIFIED:',out)




def do_validation_screen(fleet):
    """VALIDATION ONLY: check critical protection before opening fresh heldout.

    Each trained arm uses the SAME frozen 3e-4 LR as full MAPPO.
    This screen re-evaluates those selected seed checkpoints on the SAME v8
    validation scenarios. Never sees nominal heldout or H3 seed41 records.
    """
    from collections import defaultdict
    import statistics
    check_prepared()
    if fleet not in FLEETS:raise ValueError(fleet)
    import run_orchestration_experiments as runner
    if not hasattr(runner,'_load_selected_policy') or not hasattr(runner,'evaluate_policy'):
        raise RuntimeError('Expected frozen v8 checkpoint loader / validation evaluator not available')
    base=load_yaml(bases(fleet))
    records=[]
    for arm in ARMS:
        config_path,train,_=paths(arm,fleet)
        if not (train/'selected_model.json').is_file():
            raise FileNotFoundError('Run training before validation screen: '+str(train))
        tune=load_yaml(config_path)
        selection=json.loads((train/'selected_model.json').read_text())
        scores=[];scenario_samples=defaultdict(list)
        for chosen in selection['selected_training_trials']:
            ckpt=P(chosen['path'])/'best_validation_checkpoint.pt'
            policy=runner._load_selected_policy(base,tune,ckpt)
            validation_rows,validation_scores=runner.evaluate_policy(
                policy,base,tune['validation_scenarios'],tune['validation']['seeds'],
                top_k=int(tune.get('top_k',3)))
            scores.append(validation_scores)
            for row in validation_rows:
                scenario_samples[str(row['scenario'])].append(float(row['critical_sla']))
        if not scores or not scenario_samples:
            raise RuntimeError('No validation results: '+arm)
        worst=min(statistics.mean(vals) for vals in scenario_samples.values())
        metric_means={key:statistics.mean(float(x[key]) for x in scores)
                      for key in ('overall_sla','critical_sla','standard_sla','batch_sla')}
        balance_score=(.5*metric_means['overall_sla']+.25*metric_means['standard_sla']+
                       .25*metric_means['batch_sla'])
        records.append({'fleet_uavs':fleet,'reward_arm':arm,
                        'selected_learning_rate':selection['chosen_config']['learning_rate'],
                        'worst_validation_scenario_critical_sla':worst,
                        'critical_target_95pct_met':worst>=.95,
                        **metric_means,'balanced_validation_score':balance_score})
    outfile=ROOT/f'fleet{fleet}'/'reward_validation_screen.csv'
    if outfile.exists():raise FileExistsError(f'Reward validation screen already exists: {outfile}')
    outfile.parent.mkdir(parents=True,exist_ok=True)
    write_csv(outfile,records)
    eligible=[r for r in records if r['critical_target_95pct_met']]
    winner=max(eligible,key=lambda r:r['balanced_validation_score'])['reward_arm'] if eligible else None
    selection=ROOT/f'fleet{fleet}'/'reward_validation_selection.json'
    selection.write_text(json.dumps({'fleet':fleet,'rule':'worst validation-scenario critical SLA >=0.95; then 0.5*overall+0.25*standard+0.25*batch on the original validation set',
       'selected_if_feasible':winner,'no_eligible_arm':winner is None,'arms':records,
       'holdout_seen':False,'existing_h3_seed41_excluded':True},indent=2)+'\n')
    print('REWARD_VALIDATION_SCREEN:',outfile)
    for r in records:
        print(r['reward_arm'],'worst critical:',round(r['worst_validation_scenario_critical_sla'],4),
              'overall/std/batch:',*[round(r[x],4) for x in ('overall_sla','standard_sla','batch_sla')],
              'eligible:',r['critical_target_95pct_met'])
    print('PREDECLARED_VALIDATION_CHOICE:',winner if winner else 'NONE_MET_95PCT_GATE')
    return records

def do_status():
    for n in FLEETS:
        for arm in ARMS:
            cfg,tr,ev=paths(arm,n)
            print(n,arm,'training='+('COMPLETE' if (tr/'selected_model.json').is_file() else 'PENDING'),
                  'fresh_nominal='+('COMPLETE' if (ev/'heldout_summary.json').is_file() else 'PENDING'))


def do_report():
    """New prospective nominal results, including paired reward-arm bootstrap CIs."""
    check_prepared()
    rows=[]; pending=[]; paired=[]
    available={}
    for n in FLEETS:
        for arm in ARMS:
            _,tr,ev=paths(arm,n)
            fp=ev/'heldout_summary.json'
            results_path=ev/'heldout_policy_comparison.csv'
            if not fp.is_file() or not results_path.is_file():
                pending.append({'fleet_uavs':n,'reward_arm':arm,'missing_path':str(fp if not fp.is_file() else results_path)})
                continue
            payload=json.loads(fp.read_text())
            selected=json.loads((tr/'selected_model.json').read_text())
            if payload.get('scope')!='H1_H2_HELDOUT_UNCALIBRATED_NETWORK_SENSITIVITY':
                raise ValueError('Expected nominal heldout protocol: '+str(fp))
            if payload['selected_model']['tuning_config_sha256']!=sha(paths(arm,n)[0]):
                raise ValueError('Tuning provenance mismatch at '+str(fp))
            with results_path.open(newline='',encoding='utf-8') as f:
                raw=[r for r in csv.DictReader(f) if r.get('policy')=='mappo']
            if len(raw)!=payload['sample_size']:
                raise ValueError(f'Raw paired rows count mismatch: {results_path}')
            if len({(r['scenario'],r['seed'],r['training_seed']) for r in raw})!=len(raw):
                raise ValueError('Duplicate matched scenario/seed/training_seed in '+str(results_path))
            available[(n,arm)]=raw
            metrics=payload.get('summary',{})
            rows.append({'fleet_uavs':n,'reward_arm':arm,'selected_learning_rate':selected['chosen_config']['learning_rate'],
                         'validation_score':selected['chosen_config']['mean_validation_score'],
                         **{metric:metrics.get(metric) for metric in ('overall_sla','critical_sla','standard_sla','batch_sla','fairness')},
                         'critical_95pct_observed':float(metrics['critical_sla'])>=0.95})
        # Same new nominal task/scenario seeds, same training seeds, matched per-pair bootstrap.
        if (n,'baseline') in available:
            # Reuse the exact paired bootstrap implementation in the frozen v8 runner.
            from run_orchestration_experiments import _bootstrap_matched_deltas
            for arm in ('balanced_v1','balanced_v2'):
                if (n,arm) not in available:continue
                for metric in ('overall_sla','critical_sla','standard_sla','batch_sla','fairness'):
                    stats=_bootstrap_matched_deltas(available[(n,arm)],available[(n,'baseline')],metric)
                    paired.append({'fleet_uavs':n,'comparison':arm+'_minus_baseline',
                        'metric':metric,'protocol':'fresh_nominal',**stats})
    now=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=ROOT/'reports'/f'nominal_{now}'
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    write_csv(output/'reward_comparison.csv',rows)
    write_csv(output/'paired_reward_gains.csv',paired)
    write_csv(output/'pending.csv',pending)
    (output/'summary.json').write_text(json.dumps({'complete':len(rows),'pending':len(pending),
       'study':'controlled 3-arm reward design; all methods are full joint MAPPO',
       'new_nominal_heldout_seeds':FRESH_NOMINAL_SEEDS,
       'warning':'Evaluate every reward arm; do not choose a reward after seeing H3 old seeds. For confirmation, use fresh independently captured H3 mobility and task seeds.'},indent=2)+'\n')
    print('REWARD_COMPARISON_REPORT','COMPLETE' if not pending else 'PARTIAL',len(rows),'completed',len(pending),'pending')
    print('RESULTS:',output/'reward_comparison.csv')
    print('PAIRED:',output/'paired_reward_gains.csv')
    return rows


def write_csv(file,rows):
    file=P(file)
    if not rows:
        file.write_text('fleet_uavs,reward_arm\n')
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with file.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['inspect','prepare','train','validate','eval-nominal','screen','status','report'])
    ap.add_argument('--fleet',type=int,choices=FLEETS)
    ap.add_argument('--arm',choices=list(ARMS))
    ap.add_argument('--fleets',type=int,nargs='+',choices=FLEETS,default=[16,32,64])
    ap.add_argument('--device',default=os.environ.get('NOMS_DEVICE','cpu'))
    args=ap.parse_args()
    if args.command=='inspect':
        print(json.dumps(reward_section(load_yaml(TEMPLATE)),indent=2))
        print('REWARD_WEIGHT_PATHS:',[(k,'.'.join(v[0])) for k,v in zip(('success','miss'),identify_maps(reward_section(load_yaml(TEMPLATE))))])
    elif args.command=='prepare':prepare()
    elif args.command=='status':do_status()
    elif args.command=='validate':
        if args.fleet is None:ap.error('Need --fleet')
        do_validation_screen(args.fleet)
    elif args.command=='report':do_report()
    elif args.command in ('train','eval-nominal'):
        if args.fleet is None or not args.arm:ap.error('Need --fleet and --arm')
        if args.command=='train':do_train(args.arm,args.fleet,args.device)
        else:do_evaluate(args.arm,args.fleet,args.device)
    elif args.command=='screen':
        for n in args.fleets:
            for arm in ARMS:
                do_train(arm,n,args.device)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""FieldVision NOMS v14.1 reviewer experiment recovery: complete replay accounting and strict profile selection.

Reuses the installed v13 task-event hook and frozen v8/v9/v10/v12 checkpoints.
Never edits a frozen runner, checkpoint, old result, or an existing output.
External evaluations/captures run only on explicit commands (not silently).
"""
from __future__ import annotations
import argparse
import collections
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import shutil

try:
    import yaml
except ImportError:
    yaml = None

CLASSES=('critical','standard','batch')
DATASETS=('CWD30','RumexWeeds','LeafySpurge')
HEURISTICS=('local_only','edge_only','cloud_only','edf','network_aware_edf','critical_first')
V13='fieldvision_noms_revision_v13.py'
RUNNER_V8='run_orchestration_experiments.py'
RUNNER_V9='run_learning_baselines_v9.py'
INTENTS=('critical_first','balanced','recovery')


def require(ok,message):
    if not ok:raise RuntimeError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def js(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze(path,data):
    p=Path(path)
    if p.exists():raise FileExistsError(f'Will not overwrite {p}')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')


def rows(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def write_csv(path,records,fields):
    p=Path(path)
    if p.exists():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)


def mean(vals):
    x=[float(a) for a in vals if a is not None and a!='']
    return sum(x)/len(x) if x else None


def jain(vals):
    x=[float(v) for v in vals if v is not None]
    return sum(x)**2/(len(x)*sum(v*v for v in x)) if x and sum(v*v for v in x)>0 else (0.0 if x else None)


def rate_metrics(rates,targets,weights):
    require(set(rates)==set(CLASSES) and all(0<=v<=1 for v in rates.values()),'Invalid class SLA')
    require(set(targets)==set(CLASSES) and all(0<=v<=1 for v in targets.values()),'Invalid targets')
    require(set(weights)==set(CLASSES) and all(v>0 for v in weights.values()),'Invalid positive class weights')
    return {'minimum_class_sla':min(rates.values()),
            'class_jain_fairness':jain(list(rates.values())),
            'weighted_sla':sum(rates[k]*weights[k] for k in CLASSES)/sum(weights.values()),
            'max_target_deficit':max(max(0.,targets[k]-rates[k]) for k in CLASSES),
            'target_feasible_in_observed_run':all(rates[k]>=targets[k] for k in CLASSES)}


def read_intent(path):
    require(yaml is not None,'PyYAML required')
    x=yaml.safe_load(Path(path).read_text())
    targets=x.get('targets');weights=x.get('weights')
    require(isinstance(targets,dict) and isinstance(weights,dict),'Need targets and weights')
    _=rate_metrics({c:0. for c in CLASSES},targets,weights)
    return targets,weights


def v13_module(root):
    root=Path(root).resolve()
    require((root/V13).is_file(),f'Missing installed {V13}')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    import fieldvision_noms_revision_v13 as v13
    return v13


def origin_summary(root):
    root=Path(root)
    paths=[RUNNER_V8,RUNNER_V9,'fieldvision_noms_environment.py',V13,
           'fieldvision_mappo.py','fieldvision_scale_v10.py']
    return {p:{'exists':(root/p).is_file(),'sha256':digest(root/p) if (root/p).is_file() else None} for p in paths}


def check_source(root,need_trace=False):
    root=Path(root).resolve()
    a=origin_summary(root)
    require(a[V13]['exists'] and a['fieldvision_noms_environment.py']['exists'],
            'Install v13 first; missing environment or v13 source')
    installed='FIELDVISION_V13_INSTRUMENTATION_BEGIN' in (root/'fieldvision_noms_environment.py').read_text()
    if need_trace:require(installed,'Instrument the existing environment with v13 install --apply first')
    return {'root':str(root),'source_hashes':a,'v13_hook_installed':installed}


def selected_model(root,fleet,policy):
    root=Path(root)
    if fleet==128:
        d=root/f'results/noms_scale_v10/fleet128/{policy}_training'
    elif policy in ('ippo','qmix'):
        d=root/f'results/noms_learning_baselines_v9/fleet{fleet}/{policy}_training'
    else:
        d=root/f'results/noms_h123_lr3_v8/fleet{fleet}/{policy}_training'
    p=d/'selected_model.json'
    require(p.is_file(),f'Missing checkpoint-selection file {p}; no retraining assumed')
    doc=js(p)
    require(doc.get('selected_training_trials'),'Selected model has no frozen training seeds')
    for t in doc['selected_training_trials']:
        require((root/t['path']/'best_validation_checkpoint.pt').is_file(),
                'Missing frozen checkpoint: '+str(t['path']))
    return d,doc


def candidate_yamls(root):
    root=Path(root)
    paths=list(root.glob('configs/**/*.yaml'))+list(root.glob('results/noms_fleet_scaling_baselines_v1/fleet_configs/*.yaml'))
    return [p for p in paths if p.is_file()]


def identify_yaml(root,expected_sha,explicit=None):
    if explicit:
        p=Path(explicit)
        if not p.is_absolute():p=Path(root)/p
        require(p.is_file() and digest(p)==expected_sha,'Explicit config hash mismatch; refusing incompatible experiment')
        return p
    found=[p for p in candidate_yamls(root) if digest(p)==expected_sha]
    require(len(found)>0,f'Cannot locate YAML with SHA256 {expected_sha}; supply --config/--tuning-config')
    return found[0]


def command_run(cmd,cwd,log,execute,env=None):
    if not execute:
        print('DRY_RUN_COMMAND:', ' '.join(map(str,cmd)))
        return None
    p=Path(log)
    require(not p.exists(),f'Existing log: {p}; preserve it and use a fresh run ID')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        t=time.monotonic()
        completed=subprocess.run(list(map(str,cmd)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
    require(completed.returncode==0,f'Experiment failed exit={completed.returncode}; examine {p}')
    return round(time.monotonic()-t,2)



def audit_workspace(root,expected_env_hash,runner_hash=None,runner=None,create=False):
    """Copy frozen (unhooked) source into isolated workspace; never edit original.

    When a v13 hook modified the source SHA, a runner's provenance check can
    reject an otherwise valid checkpoint. Runtime-only v13 monkeypatching in
    this clean workspace preserves the exact original source bytes/hash.
    """
    root=Path(root).resolve()
    cur=root/'fieldvision_noms_environment.py'
    if cur.is_file() and digest(cur)==expected_env_hash:source=cur
    else:
        backups=list((root/'results/noms_revision_v13/source_backups').glob('*_fieldvision_noms_environment.py'))
        valid=[p for p in backups if digest(p)==expected_env_hash]
        require(len(valid)==1,'Missing exact original environment source; expected SHA256 '+str(expected_env_hash))
        source=valid[0]
    if runner and runner_hash:
        require(digest(root/runner)==runner_hash, 'Frozen runner hash changed: '+runner)
    workspace=root/'.fieldvision_audit_v14_2'/expected_env_hash[:16]
    if not create:return workspace,source
    if workspace.exists():
        require((workspace/'fieldvision_noms_environment.py').is_file() and
                digest(workspace/'fieldvision_noms_environment.py')==expected_env_hash,
                'Existing audit workspace has mismatched environment source')
    else:workspace.mkdir(parents=True)
    # Sources are copies, not symlinks, so all __file__-based hashes are stable.
    for name in ('run_orchestration_experiments.py','run_learning_baselines_v9.py',
                 'fieldvision_mappo.py','fieldvision_noms_revision_v13.py',
                 'fieldvision_noms_review_v14_1.py'):
        sp=root/name
        if not sp.is_file():continue
        dest=workspace/name
        if dest.exists():require(digest(sp)==digest(dest),'Audit workspace source drift: '+name)
        else:shutil.copy2(sp,dest)
    dest=workspace/'fieldvision_noms_environment.py'
    if not dest.exists():shutil.copy2(source,dest)
    for name in ('results','configs','data','third_party'):
        target=root/name
        if not target.exists():continue
        link=workspace/name
        if link.exists() or link.is_symlink():require(link.is_symlink() and link.resolve()==target.resolve(),'Audit workspace symlink drift: '+name)
        else:link.symlink_to(target,target_is_directory=True)
    return workspace,source



def install_selective_runtime_hooks(sim_class, expected_policy):
    """Record learned action-dictionary episodes only; baselines remain untraced.

    The frozen v8/v9 heldout runners create separate NOMSSim objects for
    scripted baseline evaluations and the learned evaluation. The v13 hook
    eagerly records all constructor instances, but does not know their policy
    until `step`. Here, attach the *unchanged* v13 EpisodeRecorder lazily on
    the first learned action-dictionary `step`.  No original simulator or
    training/checkpoint source is edited.
    """
    import atexit
    import fieldvision_noms_revision_v13 as v13

    trace=os.environ.get('FIELDVISION_V13_TRACE_ROOT')
    require(trace, 'Missing audited trace root')
    maximum=int(os.environ['FIELDVISION_V13_MAX_EPISODES'])
    require(maximum>0,'Need positive expected learned episode count')
    require(expected_policy==os.environ.get('FIELDVISION_V13_POLICY'),
            'Expected learned policy does not match capture provenance')
    require(not getattr(sim_class,'_fieldvision_v13_installed',False),
            'Cannot combine eager v13 and selective v14.2 hooks')

    mapping=v13._read_mapping(os.environ.get('FIELDVISION_V13_CLASS_MAPPING'))
    original_init,original_step=sim_class.__init__,sim_class.step
    counts={'constructed_simulators':0,'learned_episodes_started':0,
            'learned_steps':0,'scripted_baseline_steps':0,
            'unclassified_steps':0,'skipped_scripted_policy_names':{}}

    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        v13._apply_mapping(self,mapping)
        counts['constructed_simulators']+=1
        self._fieldvision_v14_2_recorder=None
        self._fieldvision_v14_2_checked_first_step=False

    def step(self,*args,**kwargs):
        action=kwargs.get('actions', args[0] if args else None)
        scripted=kwargs.get('policy')
        # Frozen scripted controls call sim.step(policy='edf'/etc.).
        # Learned MAPPO supplies a dict of per-UAV action decisions.
        if isinstance(scripted,str) and scripted!=expected_policy:
            require(scripted in HEURISTICS or scripted=='fifo',
                    f'Unexpected other policy in frozen audit: {scripted}')
            require(action is None, 'Ambiguous step: baseline label plus learned action dict')
            counts['scripted_baseline_steps']+=1
            names=counts['skipped_scripted_policy_names']
            names[scripted]=names.get(scripted,0)+1
            return original_step(self,*args,**kwargs)
        if action is None or not isinstance(action,dict):
            counts['unclassified_steps']+=1
            return original_step(self,*args,**kwargs)
        rec=getattr(self,'_fieldvision_v14_2_recorder',None)
        if rec is None:
            require(not self._fieldvision_v14_2_checked_first_step,
                    'Learned action dict appeared after unclassified early steps')
            require(counts['learned_episodes_started']<maximum,
                    f'Found more than {maximum} learned episodes; do not truncate')
            counts['learned_episodes_started']+=1
            rec=v13.EpisodeRecorder(self,Path(trace),counts['learned_episodes_started'],mapping)
            rec.episode_meta['policy_label']=expected_policy
            rec.episode_meta['recorder_protocol']='V14_2_LEARNED_ACTIONS_ONLY'
            self._fieldvision_v14_2_recorder=rec
        self._fieldvision_v14_2_checked_first_step=True
        before,waiting=rec.start_tick()
        try:
            result=original_step(self,*args,**kwargs)
        except BaseException:
            rec.flush_events()
            for stream in (rec.events,rec.actions,rec.resources):
                stream.flush()
            raise
        counts['learned_steps']+=1
        rec.record(before,waiting,action)
        return result

    sim_class.__init__=init
    sim_class.step=step
    sim_class._fieldvision_v13_installed=True  # Prevent double hooks.
    sim_class._fieldvision_v14_2_installed=True

    def audit_hook_exit():
        path=Path(trace)/'selective_hook_counts.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        # Never replace an existing attempt's data.
        if path.exists():return
        path.write_text(json.dumps(counts,indent=2,sort_keys=True)+'\n')
    atexit.register(audit_hook_exit)
    return counts

def trace_frozen(root,fleet,policy,protocol,run_id,max_episodes,plan,config,tuning,mapping,execute,mobility_seed=None):
    root=Path(root).resolve();check_source(root,need_trace=False)
    d,sel=selected_model(root,fleet,policy)
    cf=identify_yaml(root,sel['base_config_sha256'],config)
    tf=identify_yaml(root,sel['tuning_config_sha256'],tuning)
    if protocol=='independent':
        require(plan and Path(plan).is_file(),'Independent H3 requires explicit --evaluation-plan')
        seed=41 if mobility_seed is None else int(mobility_seed)
        if seed!=41:
            plan_text=Path(plan).read_text()
            require(f'seed{seed}.csv' in plan_text,
                    'New mobility seed must be explicitly referenced by the H3 plan')
            check_capture(root,fleet,seed)
            trace_path=root/f'data/traces/uavnetsim_trajectory_analytical_uplink_{fleet}uav_3s_seed{seed}.csv'
            require(trace_path.is_file() and trace_path.with_suffix('.manifest.json').is_file(),
                    'New independent mobility seed missing analytical uplink trace')
    else:
        require(not plan,'Nominal protocol must not accept an H3 evaluation plan')
        seed=27
        require(mobility_seed is None or int(mobility_seed)==27,
                'Nominal protocol uses seed 27')
    require(max_episodes>0,'Set positive maximum episodes')
    require(re.fullmatch(r'[A-Za-z0-9_.-]+',run_id) is not None,'Run ID must be safe')
    out=f'results/noms_revision_v14/traces/{run_id}'
    evalout=f'results/noms_revision_v14/evaluations/{run_id}'
    require(not (root/out).exists() and not (root/evalout).exists(),'Existing outputs must not be overwritten')
    learner=RUNNER_V9 if policy in ('ippo','qmix') else RUNNER_V8
    workspace,unhooked_source=audit_workspace(root,sel['environment_sha256'],
        sel.get('runner_sha256'),learner,create=execute)
    raw=[learner,'--mode','mappo-heldout','--config',cf,
         '--tuning-config',tf,'--sweep-root',d,'--output-root',evalout,'--device','cpu']
    if plan:raw+=['--evaluation-plan',plan]
    # Runtime hook executes after importing the byte-identical original source.
    # No physical source patch needed; recorded selected-model environment hash
    # therefore remains exactly the same as the trained checkpoint's.
    injected=("import sys,runpy; import fieldvision_noms_environment as env; "
              "import fieldvision_noms_review_v14_1 as audit; "
              "audit.install_selective_runtime_hooks(env.NOMSSim, sys.argv[1]); "
              "sys.argv=[sys.argv[2]]+sys.argv[3:]; "
              "runpy.run_path(sys.argv[0],run_name='__main__')")
    command=[sys.executable,'-c',injected,policy]+raw
    environment=dict(os.environ)
    environment.update(FIELDVISION_V13_TRACE_ROOT=str(root/out),
        FIELDVISION_V13_RUN_ID=run_id,FIELDVISION_V13_POLICY=policy,
        FIELDVISION_V13_MOBILITY_SEED=str(seed),
        FIELDVISION_V13_TASK_SEED=f'heldout_{protocol}',
        FIELDVISION_V13_MAX_EPISODES=str(max_episodes))
    if mapping:
        mp=Path(mapping)
        if not mp.is_absolute():mp=root/mp
        require(mp.is_file(),f'Missing class mapping {mapping}')
        environment['FIELDVISION_V13_CLASS_MAPPING']=str(mp)
        mapping=str(mp)
    fullmeta={'fleet_uavs':fleet,'policy':policy,'protocol':protocol,
              'training_root':str(d.relative_to(root)),
              'selected_model_sha256':digest(d/'selected_model.json'),
              'config_path':str(cf),'config_sha256':digest(cf),
              'tuning_path':str(tf),'tuning_sha256':digest(tf),
              'mapping_sha256':digest(mapping) if mapping else None,
              'H3_evaluation_plan_sha256':digest(plan) if plan else None,
              'max_episodes':max_episodes,'recorder_learned_episode_cap':max_episodes,
              'recorder_selection':'lazy_action_dict_only; scripted_baselines_excluded',
              'trace_dir':out,'evaluation_dir':evalout,
              'mobility_seed':seed,
              'frozen_original_env_sha256':digest(unhooked_source),
              'isolated_audit_workspace':str(workspace)}
    print('FROZEN_POLICY_PREFLIGHT_PASS:',json.dumps(fullmeta,indent=2))
    elapsed=command_run(command,workspace,root/f'results/noms_revision_v14/logs/{run_id}.log',execute,environment)
    if not execute:return
    v13dir=root/out
    captured=list(v13dir.rglob('manifest.json'))
    require(captured,'No captured episodes in frozen learner evaluation')
    observed={js(p).get('policy_label') for p in captured}
    require(observed=={policy},f'Learned-only recorder contamination {observed}; cannot certify')
    counters=js(v13dir/'selective_hook_counts.json')
    require(counters['learned_episodes_started']==max_episodes,
            'Learned episode count differs from requested maximum: '+str(counters))
    require(all(js(p).get('status')=='CAPTURE_COMPLETE' for p in captured),
            'Some captured episodes are incomplete; do not use audit')
    reportcmd=[sys.executable,V13,'report','--trace-root',out,'--output',
               f'results/noms_revision_v14/reports/{run_id}']
    command_run(reportcmd,root,root/f'results/noms_revision_v14/logs/{run_id}_report.log',True)
    # Validate exact learned-evaluator count; no constructor-offset workaround,
    # no scripted-baseline episodes in the recorder, and no truncation.
    eval_summary=root/evalout/'heldout_summary.json'
    require(eval_summary.is_file(), 'Missing evaluator heldout_summary.json')
    original=js(eval_summary)
    expect=int(original.get('sample_size',0))
    require(expect>0, 'Evaluator lacks a positive sample size')
    require(max_episodes>=expect,
            f'Full-replay mode needs max-episodes >= {expect}; do not report a partial capture')
    recs=rows(root/f'results/noms_revision_v14/reports/{run_id}/episode_management_metrics.csv')
    require(len(recs)==expect,
            f'Incomplete/contaminated replay: captured {len(recs)} completed episodes; '+
            f'evaluator requires {expect}. Preserve this run and diagnose initialization counts')
    if mapping:
        crosscheck={'status':'MAPPED_FULL_EPISODE_COUNT_VERIFIED',
                    'episode_count':len(recs),'mapping_hash':digest(mapping),
                    'SLA_comparison_to_original':'NOT_APPLICABLE_MAPPING_CHANGED'}
    else:
        differences={}
        for k in ('overall_sla','critical_sla'):
            if original.get('summary',{}).get(k) is not None:
                old=float(original['summary'][k])
                observed_value=mean([r[k] for r in recs])
                differences[k]=observed_value-old
                require(abs(observed_value-old)<=.02,
                        f'Learned-policy trace mismatch ({k}: {observed_value} vs {old}). '+
                        'Investigate runner/recorder before publication')
        crosscheck={'status':'MATCHED_WITHIN_2PP','differences':differences,
                    'episode_count':len(recs),'sample_size':expect}
    freeze(root/f'results/noms_revision_v14/manifests/{run_id}.json',
           dict(fullmeta,elapsed_s=elapsed,learned_policy_SLA_crosscheck=crosscheck))
    print('FROZEN_TRACE_COMPLETE:',run_id,'crosscheck=',crosscheck['status'])


def check_capture(root,fleet,seed):
    root=Path(root)
    d=root/f'results/uavnetsim_native_telemetry_{fleet}uav_seed{seed}'
    tr=d/'uavnetsim_uav_trajectories.csv'; man=d/'uavnetsim_capture_manifest.json'
    require(tr.is_file() and man.is_file(),f'Incomplete native capture: {d}')
    info=js(man)
    with tr.open(newline='') as f:
        reader=csv.DictReader(f); recs=list(reader)
    require(len(recs)==30*fleet,f'Expected 30×{fleet} positions, got {len(recs)}')
    require(set(int(r['uav_id']) for r in recs)==set(range(fleet)),'UAV positions incomplete')
    times=collections.Counter(int(r['uav_id']) for r in recs)
    require(all(k==30 for k in times.values()),'One UAV missing some timestamps')
    require(str(info.get('command_parameters',{}).get('seed'))==str(seed),'Native seed mismatch')
    return d,info,digest(tr)


def native_capture(root,fleet,seed,samples,poll_ms,execute):
    root=Path(root).resolve()
    require(fleet in (16,32,64,128),'Fleet must be one of 16 32 64 128')
    require(seed not in (27,41) and seed>0,'Use a new positive mobility seed')
    cap=root/f'results/uavnetsim_native_telemetry_{fleet}uav_seed{seed}'
    if cap.exists():
        d,info,sha=check_capture(root,fleet,seed)
        print(f'CAPTURE_ALREADY_VALIDATED: fleet={fleet} seed={seed} sha256={sha}')
        return
    script=root/'capture_uavnetsim_native_telemetry.py'
    require(script.is_file(),'Missing original native capture script')
    uvpy=root/'.venv_uavnetsim/bin/python'
    require(uvpy.is_file(),'Missing UavNetSim Python interpreter')
    require(samples>=512 and 1<=poll_ms<=100,'Invalid native capture settings')
    cmd=[uvpy,script,'--repo','third_party/UavNetSim','--nodes',fleet,'--duration',3,
         '--seed',seed,'--routing','Greedy','--mac','CSMA_CA','--mobility','GaussMarkov3D',
         '--samples',samples,'--poll-ms',poll_ms,'--output',cap]
    elapsed=command_run(cmd,root,root/f'results/noms_revision_v14/logs/capture_n{fleet}_seed{seed}.log',execute)
    if execute:
        d,info,sha=check_capture(root,fleet,seed)
        freeze(root/f'results/noms_revision_v14/manifests/capture_n{fleet}_seed{seed}.json',
               {'fleet_uavs':fleet,'seed':seed,'trajectories_sha256':sha,'capture_manifest_sha256':digest(d/'uavnetsim_capture_manifest.json'),
                'capture_script_sha256':digest(script),'source':'native_uavnetsim','elapsed_s':elapsed})
        print('NATIVE_CAPTURE_VERIFIED:',fleet,seed,sha)


def native_radio(root,fleet,seed,execute):
    root=Path(root).resolve()
    d,info,sha=check_capture(root,fleet,seed)
    config=root/f'configs/uav_to_ground_edge_radio_assumptions_{fleet}uav.yaml'
    require(config.is_file(),f'Missing existing radio assumptions {config}')
    script=root/'prepare_research_inputs.py'
    require(script.is_file(),'Missing radio trace generator')
    out=root/f'data/traces/uavnetsim_trajectory_analytical_uplink_{fleet}uav_3s_seed{seed}.csv'
    if out.exists():
        manifest=out.with_suffix('.manifest.json')
        require(manifest.is_file() and js(manifest).get('generated_trace_sha256')==digest(out),
                'Previously created trace missing valid provenance')
        print('EXISTING_RADIO_VERIFIED:',out);return
    cmd=[sys.executable,script,'radio-trace','--trajectory-csv',d/'uavnetsim_uav_trajectories.csv',
         '--radio-config',config,'--output-trace',out]
    elapsed=command_run(cmd,root,root/f'results/noms_revision_v14/logs/radio_n{fleet}_seed{seed}.log',execute)
    if execute:
        require(out.is_file() and out.with_suffix('.manifest.json').is_file(), 'Missing generated trace or manifest')
        require(js(out.with_suffix('.manifest.json')).get('generated_trace_sha256')==digest(out),
                'Trace checksum/provenance failure')
        print('NEW_ANALYTICAL_RADIO_VERIFIED:',out,'native_mobility=',sha)


def permutation_compare(original,other,out,expected_episodes=None):
    a=Path(original);b=Path(other);out=Path(out)
    require(not out.exists(),'Output exists; do not overwrite')
    def indexed(root):
        epis=sorted(root.rglob('manifest.json'))
        require(epis,'No complete v13 episodes under '+str(root))
        selected={}
        for p in epis:
            manifest=js(p); key=(int(manifest['episode_serial']),str(manifest['scenario_config_seed']),
                                 str(manifest['mobility_seed_label']))
            require(key not in selected,'Duplicate seeded episode in permutation capture')
            jobs=rows(p.parent/'jobs.csv')
            selected[key]={str(j['job_id']):j for j in jobs}
        return selected
    aa=indexed(a);bb=indexed(b)
    if expected_episodes is not None:
        require(len(aa)==expected_episodes and len(bb)==expected_episodes,
                f'Incomplete class-mapping comparison: original={len(aa)}, '+
                f'permuted={len(bb)}, expected={expected_episodes}. Preserve and rerun full trace')
    require(aa.keys()==bb.keys(),'Permutation evaluations must use identical episode/task/mobility seeds')
    phys=('job_id','uav_id','dataset','dataset_id','workload','arrival_s','deadline_s',
          'input_mb','local_runtime_s','edge_runtime_s','cloud_runtime_s')
    comparisons=[]
    for key in aa:
        require(aa[key].keys()==bb[key].keys(),'Matched episodes have different job IDs')
        for jid in aa[key]:
            j0,j1=aa[key][jid],bb[key][jid]
            for field in phys:
                require(j0.get(field)==j1.get(field),f'Physical/deadline mismatch {key} job={jid} field={field}')
            ds=j0.get('dataset') or j0.get('dataset_id')
            comparisons.append({'episode_serial':key[0],'seed':key[1],'mobility':key[2],
                                'job_id':jid,'dataset':ds,'old_class':j0['service_class'],
                                'new_class':j1['service_class']})
    require(any(x['old_class']!=x['new_class'] for x in comparisons),'No class labels changed')
    out.mkdir(parents=True)
    write_csv(out/'job_identity_verified.csv',comparisons,['episode_serial','seed','mobility','job_id','dataset','old_class','new_class'])
    freeze(out/'manifest.json',{'physically_identical_matched_jobs':len(comparisons),
         'episode_count':len(aa),'required_complete_episodes':expected_episodes,'label_changes':sum(x['old_class']!=x['new_class'] for x in comparisons),
         'original_root':str(a),'permutation_root':str(b),
         'deadline_policy':'preserve_absolute',
         'interpretation':'Frozen-policy label transfer only; does not isolate deadline urgency or retrained learned prioritization'})
    print('PERMUTATION_PAIRED_JOBS_PASS:',len(comparisons),'episodes=',len(aa))


def summarize_regimes(report_paths,output):
    # Reads multiple existing complete policy tables, never retunes or pools incompatible protocols.
    allrec=[]
    for p in report_paths:
        p=Path(p);require(p.is_file(),str(p))
        for r in rows(p):
            if not r.get('overall_sla') or not r.get('critical_sla'):continue
            classes={k:float(r[k+'_sla']) for k in CLASSES if r.get(k+'_sla') not in (None,'')}
            fair=rate_metrics(classes,
                    {'critical':.95,'standard':.8,'batch':.6},
                    {'critical':3.,'standard':2.,'batch':1.}) if len(classes)==3 else {}
            allrec.append({'source':str(p),'fleet':int(r['fleet_uavs']),
                           'protocol':r.get('protocol') or ('independent' if 'independent' in str(p) else 'nominal'),
                           'policy':r.get('variant',r.get('policy','')),
                           'overall':float(r['overall_sla']),'critical':float(r['critical_sla']),
                           'minimum_class_sla':fair.get('minimum_class_sla'),
                           'class_jain_fairness':fair.get('class_jain_fairness'),
                           'weighted_sla':fair.get('weighted_sla'),
                           'max_target_deficit':fair.get('max_target_deficit')})
    require(allrec,'No rows from policy tables')
    groups=collections.defaultdict(list)
    for r in allrec:groups[(r['source'],r['fleet'],r['protocol'])].append(r)
    result=[]
    for key,group in sorted(groups.items()):
        eligible={r['policy']:r for r in group}
        require(len(eligible)==len(group),'Duplicate policy rows in same source/fleet/protocol')
        highest_overall=max(group,key=lambda r:r['overall'])
        highest_critical=max(group,key=lambda r:r['critical'])
        full=eligible.get('full')
        with_classes=[r for r in group if r['minimum_class_sla'] is not None]
        best_min=max(with_classes,key=lambda r:r['minimum_class_sla']) if with_classes else None
        best_weighted=max(with_classes,key=lambda r:r['weighted_sla']) if with_classes else None
        result.append({'source':key[0],'fleet_uavs':key[1],'protocol':key[2],
                       'best_overall_policy':highest_overall['policy'],'best_overall_sla':highest_overall['overall'],
                       'best_critical_policy':highest_critical['policy'],'best_critical_sla':highest_critical['critical'],
                       'best_minimum_class_policy':best_min['policy'] if best_min else None,
                       'best_minimum_class_sla':best_min['minimum_class_sla'] if best_min else None,
                       'best_weighted_sla_policy':best_weighted['policy'] if best_weighted else None,
                       'best_weighted_sla':best_weighted['weighted_sla'] if best_weighted else None,
                       'full_overall_sla':full['overall'] if full else None,
                       'full_critical_sla':full['critical'] if full else None,
                       'full_minimum_class_sla':full['minimum_class_sla'] if full else None,
                       'full_class_jain_fairness':full['class_jain_fairness'] if full else None,
                       'descriptive_only':True})
    write_csv(output,result,list(result[0]))
    print('REGIME_RESULTS_WRITTEN:',output,len(result))


def measured_profile_config(root,fleet,profile_config=None):
    """Reject synthetic workloads; prefer exact measured fleet configs.

    No model-derived or inference-time profiles are invented here. An explicit
    alternate configuration is acceptable only if it has the original measured
    provenance and exactly matches the intended fleet and mobility source.
    """
    root=Path(root).resolve()
    candidates=[root/f'configs/noms_scale_v10/fleet_{fleet}uav_base.yaml',
                root/f'configs/noms_fleet_expansion_v4/fleet_{fleet}uav_base.yaml',
                root/f'results/noms_fleet_scaling_baselines_v1/fleet_configs/fleet_{fleet}uav_base.yaml']
    if profile_config:
        cp=Path(profile_config)
        if not cp.is_absolute():cp=root/cp
        candidates=[cp]
    records=[];valid=[]
    for cp in candidates:
        if not cp.is_file():
            records.append({'path':str(cp),'exists':False})
            continue
        cfg=yaml.safe_load(cp.read_text())
        source=cfg.get('workload_profile_source')
        trace=(cfg.get('network') or {}).get('trace_provenance')
        actual_fleet=cfg.get('n_uavs')
        prof=cfg.get('profiled_workloads') or {}
        known_provenance={'measured_reference_edge_assumed_local_cloud',
                          'measured_reference_cpu_proxy_edge_gpu_assumed_cloud'}
        okay=(source in known_provenance and
              trace=='trajectory_driven_analytical_uncalibrated' and
              int(actual_fleet or -1)==int(fleet) and
              isinstance(prof,dict) and
              'assumed_local_runtime_multiplier' in prof and
              (cfg.get('network') or {}).get('trace_csv'))
        records.append({'path':str(cp),'exists':True,'workload_profile_source':source,
                        'network_trace_provenance':trace,'n_uavs':actual_fleet,
                        'has_profiled_workloads':bool(prof),'verified_eligible':bool(okay),
                        'sha256':digest(cp),
                        'runtime_provenance_note':('server_CPU_proxy_not_UAV_measurement'
                             if source=='measured_reference_cpu_proxy_edge_gpu_assumed_cloud' else None)})
        if okay:valid.append((cp,cfg))
    return records,valid[0] if valid else None


def profile_audit(root,fleet,profile_config=None):
    candidates,chosen=measured_profile_config(root,fleet,profile_config)
    report={'fleet_uavs':fleet,'candidate_profiles':candidates,
            'selected_measured_configuration':str(chosen[0]) if chosen else None,
            'status':'VERIFIED_MEASURED_PROFILE_FOUND' if chosen else 'NO_COMPATIBLE_MEASURED_PROFILE'}
    print('V14_1_PROFILE_AUDIT:',json.dumps(report,indent=2))
    return chosen


def fairness_grid(root,fleet,multipliers,slots,intents,out,profile_config=None):
    """Diagnostics on fixed-native-radio stress; not scientifically matched to MAPPO until explicit replay."""
    root=Path(root).resolve();out=Path(out)
    require(not out.exists(),f'Refusing overwrite: {out}')
    require(yaml is not None,'PyYAML needed')
    v13=v13_module(root)
    import fieldvision_noms_environment as env
    chosen=profile_audit(root,fleet,profile_config)
    require(chosen is not None,
            'No eligible measured-profile configuration found. Do not bypass the provenance gate; '+
            'check candidate paths/source tags or provide --profile-config from verified profiling')
    cf,base=chosen
    require(all(x>0 for x in multipliers) and all(x>=1 for x in slots),'Bad stress grid')
    for name in intents:
        require((root/f'configs/noms_revision_v14/intent_{name}.yaml').is_file(),
                'Missing v14 intent file '+name)
    out.mkdir(parents=True)
    summary=[]
    divergence=[]
    for slowdown in multipliers:
        for edge_slots in slots:
            input_config=copy.deepcopy(base)
            require('assumed_local_runtime_multiplier' in input_config['profiled_workloads'],
                    'Unknown local slowdown key')
            input_config['profiled_workloads']['assumed_local_runtime_multiplier']=float(slowdown)
            input_config['resources']['edge_slots']=int(edge_slots)
            # Reuse exact source, same seed/job sample/radio across all intents.
            baseline_hashes=None
            baseline_signature=None
            for name in intents:
                targets,weights=read_intent(root/f'configs/noms_revision_v14/intent_{name}.yaml')
                sim=env.NOMSSim(copy.deepcopy(input_config))
                jobs={jid:(j.uav_id,j.dataset,j.arrival_s,j.deadline_s,j.input_mb,
                           j.local_runtime_s,j.edge_runtime_s,j.cloud_runtime_s) for jid,j in sim.jobs.items()}
                if baseline_hashes is None:baseline_hashes=jobs
                require(jobs==baseline_hashes,'Intent changed physical task generation')
                run_id=f'n{fleet}_localx{slowdown:g}_edge{edge_slots}_intent_{name}'
                base_dir=out/'trace'/run_id
                rec=v13.EpisodeRecorder(sim,base_dir,1,None)
                rec.episode_meta.update({'policy_label':'fair_capacity_heuristic',
                    'mobility_seed_label':'27','task_seed_label':str(input_config['seed']),
                    'intent_name':name,'stress_local_factor':float(slowdown),'edge_slots':edge_slots})
                while sim.time_s<sim.horizon-1e-7:
                    before,waiting=rec.start_tick()
                    actions=v13.capacity_fair_decisions(sim,targets,weights)
                    sim.step(actions=actions)
                    rec.record(before,waiting,actions)
                rec.finalize()
                episode_dir=next(base_dir.rglob('manifest.json')).parent
                signature={}
                for action in v13.read_jsonl(episode_dir/'decisions.jsonl'):
                    key=(action['time_s'],action['uav_id'])
                    signature[key]=(action.get('task_id'),action.get('actual_placement'),
                        action.get('actual_network_tier'),action.get('actual_compute_tier'))
                if baseline_signature is None:
                    baseline_signature=signature
                else:
                    require(signature.keys()==baseline_signature.keys(),'Intent grid has unmatched decision ticks')
                    changed=sum(signature[k]!=baseline_signature[k] for k in signature)
                    divergence.append({'fleet_uavs':fleet,'local_slowdown':slowdown,
                        'edge_slots':edge_slots,'comparison':intents[0]+'_vs_'+name,
                        'changed_decision_ticks':changed,'total_decision_ticks':len(signature),
                        'change_fraction':changed/max(1,len(signature))})
                stats,classes,placements,tiers,comp,intervals=v13.summarise_episode(episode_dir)
                rates={k:float(stats[k+'_sla']) for k in CLASSES}
                obj={'run_id':run_id,'fleet_uavs':fleet,'policy':'fair_capacity_heuristic',
                     'intent':name,'local_slowdown':slowdown,'edge_slots':edge_slots,
                     'overall_sla':stats['overall_sla'],**{k+'_sla':rates[k] for k in CLASSES},
                     **rate_metrics(rates,targets,weights),
                     'mean_network_utilization':stats['mean_network_utilization'],
                     'mean_edge_slot_utilization':stats['mean_edge_slot_utilization'],
                     'scheduled_offloads':sum(x['scheduled_offloads'] for x in tiers),
                     'scheduled_jobs':sum(x['scheduled_jobs'] for x in placements),
                     'source_config_sha256':digest(cf),'intent_source_sha256':digest(root/f'configs/noms_revision_v14/intent_{name}.yaml')}
                summary.append(obj)
    write_csv(out/'stress_intent_metrics.csv',summary,list(summary[0]))
    write_csv(out/'intent_action_divergence.csv',divergence,
              ['fleet_uavs','local_slowdown','edge_slots','comparison',
               'changed_decision_ticks','total_decision_ticks','change_fraction'])
    alltraces=out/'trace'
    v13.report(alltraces,out/'trace_report')
    if divergence and not any(x['changed_decision_ticks']>0 for x in divergence):
        print('INTENT_NO_RESPONSE_WARNING: all tested policy actions identical across intent profiles')
    if all(r['overall_sla']>=.999 for r in summary):
        warn='All tested scenarios are easy: 100% SLA. This does not validate the fairness claim.'
        print('FAIRNESS_SATURATION_WARNING:',warn)
    freeze(out/'manifest.json',{'source':'Existing v13 heuristic and original NOMSSim',
        'evaluation_type':'diagnostic_not_full_MAPPO_comparison',
        'config_sha256':digest(cf),'n_scenarios':len(summary),
        'local_multipliers':multipliers,'edge_slot_choices':slots,'intents':intents,
        'intent_action_changes_observed':bool(any(x['changed_decision_ticks']>0 for x in divergence)),
        'limits':['Do not compare with H3 MAPPO means unless exact input configs/episodes are matched',
                  'Intent differences are heuristic configuration, not runtime intent-conditioned MAPPO',
                  'Analytical radio uncalibrated']})
    print('FAIR_GRID_COMPLETE:',out,len(summary))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('preflight');s.add_argument('--root',default='.')
    s=sub.add_parser('trace-frozen');s.add_argument('--root',default='.')
    s.add_argument('--fleet',type=int,required=True);s.add_argument('--policy',required=True)
    s.add_argument('--protocol',choices=['nominal','independent'],required=True)
    s.add_argument('--run-id',required=True);s.add_argument('--max-episodes',type=int,default=75)
    s.add_argument('--evaluation-plan');s.add_argument('--config');s.add_argument('--tuning-config');s.add_argument('--mapping')
    s.add_argument('--mobility-seed',type=int);s.add_argument('--execute',action='store_true')
    s=sub.add_parser('profile-audit');s.add_argument('--root',default='.');s.add_argument('--fleet',type=int,required=True)
    s.add_argument('--profile-config')
    s=sub.add_parser('fair-grid');s.add_argument('--root',default='.')
    s.add_argument('--fleet',type=int,required=True);s.add_argument('--multipliers',nargs='+',type=float,default=[18.75,24.50])
    s.add_argument('--edge-slots',nargs='+',type=int,default=[1,2]);s.add_argument('--intents',nargs='+',choices=INTENTS,default=list(INTENTS))
    s.add_argument('--output',required=True);s.add_argument('--profile-config')
    s=sub.add_parser('permutation-check');s.add_argument('--original',required=True);s.add_argument('--permuted',required=True);s.add_argument('--output',required=True)
    s.add_argument('--expected-episodes',type=int)
    s=sub.add_parser('capture-native');s.add_argument('--root',default='.')
    s.add_argument('--fleet',type=int,required=True);s.add_argument('--seed',type=int,required=True)
    s.add_argument('--samples',type=int,default=250000);s.add_argument('--poll-ms',type=int,default=1);s.add_argument('--execute',action='store_true')
    s=sub.add_parser('radio-native');s.add_argument('--root',default='.')
    s.add_argument('--fleet',type=int,required=True);s.add_argument('--seed',type=int,required=True);s.add_argument('--execute',action='store_true')
    s=sub.add_parser('regime-report');s.add_argument('--tables',nargs='+',required=True);s.add_argument('--output',required=True)
    s=sub.add_parser('derive-h3-plan');s.add_argument('--original-plan',required=True);s.add_argument('--fleet',type=int,required=True)
    s.add_argument('--mobility-seed',type=int,required=True);s.add_argument('--output',required=True)
    args=p.parse_args()
    if args.cmd=='preflight':print(json.dumps(check_source(args.root),indent=2))
    elif args.cmd=='trace-frozen':
        trace_frozen(args.root,args.fleet,args.policy,args.protocol,args.run_id,args.max_episodes,
                     args.evaluation_plan,args.config,args.tuning_config,args.mapping,args.execute,args.mobility_seed)
    elif args.cmd=='profile-audit':profile_audit(args.root,args.fleet,args.profile_config)
    elif args.cmd=='fair-grid':fairness_grid(args.root,args.fleet,args.multipliers,args.edge_slots,args.intents,args.output,args.profile_config)
    elif args.cmd=='permutation-check':permutation_compare(args.original,args.permuted,args.output,args.expected_episodes)
    elif args.cmd=='capture-native':native_capture(args.root,args.fleet,args.seed,args.samples,args.poll_ms,args.execute)
    elif args.cmd=='radio-native':native_radio(args.root,args.fleet,args.seed,args.execute)
    elif args.cmd=='regime-report':summarize_regimes(args.tables,args.output)
    elif args.cmd=='derive-h3-plan':derive_h3_plan(args.original_plan,args.fleet,args.mobility_seed,args.output)


def derive_h3_plan(original,fleet,seed,out):
    # A frozen H3 YAML can name its external radio trace directly. Replace only
    # the trace path, never the task seeds/stress magnitudes/labels/checkpoints.
    require(yaml is not None,'PyYAML required')
    original=Path(original);out=Path(out)
    require(original.is_file(),'Missing original H3 plan')
    obj=yaml.safe_load(original.read_text())
    def replace(x):
        if isinstance(x,dict):
            return {k:replace(v) for k,v in x.items()}
        if isinstance(x,list):return [replace(v) for v in x]
        if isinstance(x,str) and re.search(r'uavnetsim_trajectory_analytical_uplink_\d+uav_3s_seed41\.csv',x):
            match=re.search(r'uavnetsim_trajectory_analytical_uplink_(\d+)uav_3s_seed41\.csv',x)
            require(int(match.group(1))==fleet,'Source plan fleet mismatch')
            replacements.append(x)
            return x.replace('seed41.csv',f'seed{seed}.csv')
        return x
    replacements=[]
    new=replace(obj)
    require(replacements,'H3 source plan has no explicit seed-41 trace reference; do not guess keys')
    require(not out.exists(),'Refusing overwrite')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(yaml.safe_dump(new,sort_keys=False))
    freeze(out.with_suffix('.manifest.json'),{
        'base_plan_sha256':digest(original),'new_plan_sha256':digest(out),
        'fleet_uavs':fleet,'mobility_seed':seed,'modified_trace_paths':replacements,
        'warning':'Plan is not an evaluation result; run frozen-policy replay with independent trace verification'})
    print('DERIVED_H3_PLAN_READY:',out)


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError,KeyError) as err:
        print('V14_GATE_FAILED:',err,file=sys.stderr)
        raise SystemExit(1)

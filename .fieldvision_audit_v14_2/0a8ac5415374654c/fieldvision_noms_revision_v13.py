#!/usr/bin/env python3
"""FieldVision NOMS v13: opt-in task/network/compute audit and controlled experiments.

Works with NOMSSim-style FieldVision environments. NEVER overwrites previous results.
Run `audit` and `install --dry-run` before applying hooks. Uses Python stdlib,
plus the existing PyYAML dependency for YAML scenarios.

Main commands: audit, install, trace, report, permute, fair-baseline, seed-plan.
"""
from __future__ import annotations
import argparse
import collections
import csv
import dataclasses
import datetime as dt
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

STAMP = 'FIELDVISION_V13_INSTRUMENTATION_BEGIN'
ENDSTAMP = 'FIELDVISION_V13_INSTRUMENTATION_END'
CLASSES = ('critical','standard','batch')
DATASETS = ('CWD30','RumexWeeds','LeafySpurge')
ALLOWED_TIERS = ('low','standard','high')
PLACEMENTS = ('local','edge','cloud')
DEFAULT_INTENT = {'critical':0.95,'standard':0.80,'batch':0.60}
_RUN_COUNT = 0

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def dump(path, obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(f'Refusing overwrite: {p}')
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')

def dictify(obj):
    if isinstance(obj,dict):return dict(obj)
    if dataclasses.is_dataclass(obj):return dataclasses.asdict(obj)
    if hasattr(obj,'__dict__'):return vars(obj).copy()
    return {}

def asfloat(x, default=None):
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except (TypeError,ValueError):return default

def safe_name(s):
    s=str(s)
    if not s or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for c in s):
        raise ValueError('Only alphanumeric, underscore, hyphen and period allowed in names')
    return s

def write_jsonl(stream,obj):stream.write(json.dumps(obj,sort_keys=True,default=str,allow_nan=False)+'\n')

def read_jsonl(path):
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def csvout(path,rows):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    rows=list(rows)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        if columns:
            w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
        else:f.write('no_rows\n')

def validate_sim_class(sim_class):
    for attr in ('__init__','step'):
        if not hasattr(sim_class,attr):raise RuntimeError(f'NOMSSim missing {attr}')

class EpisodeRecorder:
    """Streaming audit of actual simulator events; no inferred radio calibration."""
    def __init__(self, sim, parent, serial, mapping=None):
        if not isinstance(getattr(sim,'jobs',None),dict):
            raise RuntimeError('v13 expects sim.jobs as a job_id->object dictionary')
        run=safe_name(os.environ.get('FIELDVISION_V13_RUN_ID','unlabeled'))
        # One invocation can create many episodes; never reuse a directory.
        self.path=parent/run/f'episode_{os.getpid()}_{serial:06d}'
        self.path.mkdir(parents=True,exist_ok=False)
        self.s=sim
        self.mapping=mapping
        self.events=self.path.joinpath('events.jsonl').open('w',encoding='utf-8')
        self.actions=self.path.joinpath('decisions.jsonl').open('w',encoding='utf-8')
        self.resources=self.path.joinpath('resources.jsonl').open('w',encoding='utf-8')
        self.eidx=0;self.nidx=0;self.uidx=0;self.cidx=0
        self.finalized=False
        self.episode_meta={
            'run_id':run,'episode_serial':serial,'fleet_uavs':getattr(sim,'n_uavs',None),
            'horizon_s':getattr(sim,'horizon',None),'tick_s':getattr(sim,'dt',None),
            'policy_label':os.environ.get('FIELDVISION_V13_POLICY','unspecified'),
            'mobility_seed_label':os.environ.get('FIELDVISION_V13_MOBILITY_SEED','unknown'),
            'task_seed_label':os.environ.get('FIELDVISION_V13_TASK_SEED','unknown'),
            'scenario_config_seed':getattr(sim,'config',{}).get('seed') if isinstance(getattr(sim,'config',{}),dict) else None,
            'network_provenance':getattr(sim,'trace_provenance','unknown'),
            'mapping':mapping,'source_file_sha256':sha(Path(__file__)),
            'created_utc':now(), 'status':'CAPTURE_STARTED',
            'measurement_note':'Time/resource values from simulator state, not measured UAV radio power.'}
        self._write_jobs()
        self.flush_events()

    def _write_jobs(self):
        output=[]
        for jid,job in sorted(self.s.jobs.items()):
            d=dictify(job)
            rec={k:d.get(k) for k in ('job_id','uav_id','dataset','dataset_id','workload','service_class',
                'arrival_s','deadline_s','input_mb','local_runtime_s','edge_runtime_s','cloud_runtime_s',
                'compute_profile_source','profile_image_path')}
            rec['job_id']=str(rec['job_id'] or jid)
            output.append(rec)
        csvout(self.path/'jobs.csv',output)

    def flush_events(self):
        events=getattr(self.s,'events',[])
        for e in events[self.eidx:]:
            write_jsonl(self.events,dictify(e))
        self.eidx=len(events)

    def record(self, before_time, before_waiting, actions):
        self.flush_events()
        # decisions inferred from ACTUAL scheduled event, not from requested action where absent.
        events=getattr(self.s,'events',[])
        new=[dictify(e) for e in events[self._decision_eidx:]] if hasattr(self,'_decision_eidx') else []
        scheduled={int(e['uav_id']):e for e in new if e.get('event')=='scheduled' and e.get('uav_id') is not None}
        actionmap=actions if isinstance(actions,dict) else {}
        for i in range(int(getattr(self.s,'n_uavs',0))):
            event=scheduled.get(i)
            obj=dictify(actionmap.get(i))
            rec={'time_s':before_time,'uav_id':i,'waiting_before':before_waiting.get(i,0),
                 'scheduled':bool(event),'task_id':(event or {}).get('task_id') or obj.get('task_id'),
                 'requested_placement':obj.get('placement'),
                 'requested_network_tier':obj.get('network_tier'),
                 'requested_compute_tier':obj.get('compute_tier'),
                 'actual_placement':(event or {}).get('placement'),
                 'actual_network_tier':(event or {}).get('network_tier'),
                 'actual_compute_tier':(event or {}).get('compute_tier'),
                 'defer_with_pending_tasks':int(event is None and before_waiting.get(i,0)>0),
                 'task_selected_from_observed_event':event is not None}
            write_jsonl(self.actions,rec)
        net=getattr(self.s,'network_history',[])
        links=getattr(self.s,'per_uav_network_history',[])
        compute=getattr(self.s,'compute_history',[])
        for e in net[self.nidx:]:write_jsonl(self.resources,{'kind':'network',**dictify(e)})
        for e in links[self.uidx:]:write_jsonl(self.resources,{'kind':'uav_network',**dictify(e)})
        for e in compute[self.cidx:]:write_jsonl(self.resources,{'kind':'compute',**dictify(e),
            'edge_slots':getattr(self.s,'edge_slots',None),'cloud_slots':getattr(self.s,'cloud_slots',None),
            'n_uavs':getattr(self.s,'n_uavs',None)})
        self.nidx=len(net);self.uidx=len(links);self.cidx=len(compute)
        if getattr(self.s,'time_s',0)>=getattr(self.s,'horizon',float('inf'))-1e-7:self.finalize()

    def start_tick(self):
        self._decision_eidx=len(getattr(self.s,'events',[]))
        return getattr(self.s,'time_s',None),{int(i):len(q) for i,q in getattr(self.s,'waiting',{}).items()}

    def finalize(self):
        if self.finalized:return
        self.finalized=True
        self.flush_events()
        for f in [self.events,self.actions,self.resources]:f.flush();f.close()
        self.episode_meta['status']='CAPTURE_COMPLETE'
        self.episode_meta['end_time_s']=getattr(self.s,'time_s',None)
        self.episode_meta['compute_busy_slot_seconds']=getattr(self.s,'compute_busy_slot_seconds',None)
        self.episode_meta['compute_slot_capacity']={'edge':getattr(self.s,'edge_slots',None),
            'cloud':getattr(self.s,'cloud_slots',None)}
        files=('jobs.csv','events.jsonl','decisions.jsonl','resources.jsonl')
        self.episode_meta['file_sha256']={p:sha(self.path/p) for p in files}
        dump(self.path/'manifest.json',self.episode_meta)

def _read_mapping(path):
    if not path:return None
    x=json.loads(Path(path).read_text())
    mapping=x.get('dataset_to_class')
    if not isinstance(mapping,dict) or set(mapping)!=set(DATASETS) or set(mapping.values())!=set(CLASSES):
        raise ValueError('Mapping must permute all three datasets over all three classes')
    if x.get('deadline_policy')!='preserve_absolute':
        raise ValueError('This controlled experiment ONLY supports fixed absolute deadlines')
    return mapping

def _apply_mapping(sim,mapping):
    if mapping is None:return
    updated=0
    for job in sim.jobs.values():
        dataset=getattr(job,'dataset',getattr(job,'dataset_id',None))
        if dataset not in mapping:raise RuntimeError(f'Unrecognized dataset in mapping: {dataset}')
        job.service_class=mapping[dataset];updated+=1
    for entry in getattr(sim,'job_definitions',[]):
        if isinstance(entry,dict):
            ds=entry.get('dataset',entry.get('dataset_id'))
            if ds in mapping:entry['service_class']=mapping[ds]
    if not updated:raise RuntimeError('No jobs updated')

def install_runtime_hooks(namespace):
    """Called at module import when hook is installed; inert unless v13 env vars set."""
    global _RUN_COUNT
    trace=os.environ.get('FIELDVISION_V13_TRACE_ROOT')
    mapping_file=os.environ.get('FIELDVISION_V13_CLASS_MAPPING')
    if not trace and not mapping_file:return
    cls=namespace.get('NOMSSim')
    if cls is None:raise RuntimeError('v13 hook installed into module without NOMSSim')
    if getattr(cls,'_fieldvision_v13_installed',False):return
    validate_sim_class(cls)
    mapping=_read_mapping(mapping_file)
    maximum=int(os.environ.get('FIELDVISION_V13_MAX_EPISODES','100'))
    if maximum<0:raise ValueError('FIELDVISION_V13_MAX_EPISODES must be >=0')
    original_init,original_step=cls.__init__,cls.step
    def init(self,*args,**kwargs):
        global _RUN_COUNT
        original_init(self,*args,**kwargs)
        _apply_mapping(self,mapping)
        _RUN_COUNT+=1
        self._fieldvision_v13_recorder=None
        if trace and _RUN_COUNT<=maximum:
            self._fieldvision_v13_recorder=EpisodeRecorder(self,Path(trace),_RUN_COUNT,mapping)
    def step(self,*args,**kwargs):
        rec=getattr(self,'_fieldvision_v13_recorder',None)
        if rec is None:return original_step(self,*args,**kwargs)
        before,waiting=rec.start_tick()
        action=kwargs.get('actions',args[0] if args else None)
        if action is None and 'policy' in kwargs:
            rec.episode_meta['policy_label']=str(kwargs['policy'])
        try:
            result=original_step(self,*args,**kwargs)
        except BaseException:
            # Preserve incomplete record for diagnosis; never mark successful.
            rec.flush_events()
            for f in (rec.events,rec.actions,rec.resources):f.flush()
            raise
        rec.record(before,waiting,action)
        return result
    cls.__init__=init;cls.step=step;cls._fieldvision_v13_installed=True

def audit(root):
    root=Path(root).resolve()
    paths={k:root/k for k in ('fieldvision_noms_environment.py','run_learning_baselines_v9.py',
        'fieldvision_mappo.py','run_orchestration_experiments.py','run_fieldvision_scale_v10.sh')}
    env=paths['fieldvision_noms_environment.py']
    info={'root':str(root),'source':{},'results':{},'checked_utc':now()}
    for key,p in paths.items():
        info['source'][key]={'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None}
    if env.is_file():
        import ast
        src=env.read_text()
        tree=ast.parse(src)
        info['environment_contract']={'has_NOMSSim':any(isinstance(n,ast.ClassDef) and n.name=='NOMSSim' for n in tree.body),
          'has_step':any(isinstance(n,ast.ClassDef) and n.name=='NOMSSim' and any(isinstance(x,ast.FunctionDef) and x.name=='step' for x in n.body) for n in tree.body),
          'v13_hook_installed':STAMP in src}
    for path in ('results/noms_h123_lr3_v8','results/noms_learning_baselines_v9','results/noms_scale_v10/fleet128'):
        p=root/path
        info['results'][path]={'exists':p.is_dir(),'selected_models':len(list(p.rglob('selected_model.json'))) if p.is_dir() else 0,
            'prior_action_csv':len(list(p.rglob('*action*.csv'))) if p.is_dir() else 0}
    return info

def install(root,apply):
    root=Path(root).resolve();p=root/'fieldvision_noms_environment.py'
    if not p.is_file():raise FileNotFoundError(p)
    info=audit(root)
    contract=info.get('environment_contract',{})
    if not contract.get('has_NOMSSim') or not contract.get('has_step'):
        raise RuntimeError('NOMSSim/step contract not confirmed; do not modify current environment')
    if contract.get('v13_hook_installed'):
        print('V13_HOOK_ALREADY_PRESENT');return
    patch=(f'\n# {STAMP}\n'
           'from fieldvision_noms_revision_v13 import install_runtime_hooks as _fv13_install_hooks\n'
           '_fv13_install_hooks(globals())\n'
           f'# {ENDSTAMP}\n')
    print('SOURCE_FILE:',p);print('SOURCE_SHA256_BEFORE:',sha(p))
    print('APPEND_ONLY_PATCH:',patch)
    if not apply:
        print('DRY_RUN_ONLY: pass --apply after checking the output');return
    if not (root/'fieldvision_noms_revision_v13.py').is_file():
        raise RuntimeError('Copy fieldvision_noms_revision_v13.py into the project root before applying')
    backup=root/'results/noms_revision_v13/source_backups'/f'{sha(p)[:16]}_fieldvision_noms_environment.py'
    backup.parent.mkdir(parents=True,exist_ok=True)
    if backup.exists():raise FileExistsError(f'Backup exists; verify it first: {backup}')
    shutil.copyfile(p,backup)
    source=p.read_text()
    match=re.search(r'^if __name__\s*==\s*[\"\']__main__[\"\']\s*:',source,re.M)
    if match:
        modified=source[:match.start()]+patch+'\n'+source[match.start():]
        placement='before_main_entrypoint'
    else:
        modified=source+patch
        placement='end_of_importable_module'
    p.write_text(modified)
    print('HOOK_PLACEMENT:',placement)
    print('V13_HOOK_APPLIED:',p);print('BACKUP:',backup);print('SOURCE_SHA256_AFTER:',sha(p))

def trace(args):
    root=Path(args.root).resolve()
    a=audit(root)
    if not a.get('environment_contract',{}).get('v13_hook_installed'):
        raise RuntimeError('Run install --apply first; evaluator cannot be instrumented otherwise')
    command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command:raise ValueError('Supply a fresh-output evaluator command after --')
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(f'Refusing to replace trace run: {out}')
    out.mkdir(parents=True)
    env=dict(os.environ)
    env['FIELDVISION_V13_TRACE_ROOT']=str(out)
    env['FIELDVISION_V13_RUN_ID']=safe_name(args.run_id)
    env['FIELDVISION_V13_POLICY']=args.policy
    env['FIELDVISION_V13_MAX_EPISODES']=str(args.max_episodes)
    env['FIELDVISION_V13_MOBILITY_SEED']=args.mobility_seed
    env['FIELDVISION_V13_TASK_SEED']=args.task_seed
    if args.mapping:env['FIELDVISION_V13_CLASS_MAPPING']=str(Path(args.mapping).resolve())
    metadata={'run_id':args.run_id,'policy':args.policy,'created_utc':now(),
              'command':command,'mapping':args.mapping,'mobility_seed':args.mobility_seed,
              'task_seed':args.task_seed,'max_episodes':args.max_episodes,
              'environment_sha256':a['source']['fieldvision_noms_environment.py']['sha256'],
              'v13_runner_sha256':sha(Path(__file__))}
    dump(out/'invocation.json',metadata)
    with (out/'runner.log').open('w') as logfile:
        result=subprocess.run(command,cwd=root,env=env,stdout=logfile,stderr=subprocess.STDOUT)
    dump(out/'exit_status.json',{'returncode':result.returncode,'finished_utc':now()})
    print('V13_TRACE_EXIT:',result.returncode,'DIR:',out)
    if result.returncode:raise RuntimeError(f'Evaluator failed; preserve {out}/runner.log')
    count=len(list(out.rglob('manifest.json')))
    if count==0:raise RuntimeError('Evaluator finished but no NOMSSim instances were captured; inspect hook and runner')
    print('V13_CAPTURED_EPISODES:',count)

def weighted_sla(rates,weights):
    valid=[q for q in CLASSES if rates.get(q) is not None]
    den=sum(weights[q] for q in valid)
    return sum(weights[q]*rates[q] for q in valid)/den if den>0 else None

def jain(values):
    vals=[float(x) for x in values if x is not None]
    if not vals:return None
    total=sum(vals)
    return (total*total/(len(vals)*sum(x*x for x in vals))) if sum(x*x for x in vals)>0 else None

def summarise_episode(path):
    import csv as csvmod
    manifest=json.loads((path/'manifest.json').read_text())
    with (path/'jobs.csv').open(newline='') as f:jobs=list(csvmod.DictReader(f))
    if not jobs:raise RuntimeError(f'No jobs in {path}')
    byid={j['job_id']:j for j in jobs}
    timeline=collections.defaultdict(dict)
    for event in read_jsonl(path/'events.jsonl'):
        jid=str(event.get('task_id',event.get('job_id','')))
        if jid not in byid:continue
        name=event.get('event')
        if name in ('arrived','scheduled','upload_started','upload_completed','compute_started','completed'):
            if name not in timeline[jid]:timeline[jid][name]=asfloat(event.get('time_s'))
    decisionrows=list(read_jsonl(path/'decisions.jsonl'))
    allocation=[];resources=[];compute=[]
    for resource in read_jsonl(path/'resources.jsonl'):
        if resource['kind']=='network':resources.append(resource)
        elif resource['kind']=='uav_network':allocation.append(resource)
        elif resource['kind']=='compute':compute.append(resource)
    class_data=collections.defaultdict(lambda:collections.Counter())
    placement=collections.Counter()
    network_tier=collections.Counter()
    compute_tier=collections.Counter()
    latencies=collections.defaultdict(list)
    scheduled_by_job={str(rec.get('task_id')):rec for rec in decisionrows if rec.get('scheduled')}
    for j in jobs:
        ident=j['job_id']; cls=j['service_class'];t=timeline[ident]
        arrive=asfloat(j.get('arrival_s'))
        deadline=asfloat(j.get('deadline_s'))
        complete=t.get('completed')
        if arrive is None or deadline is None:raise RuntimeError(f'Job lacks numeric times {ident}')
        if arrive >= asfloat(manifest['horizon_s'],float('inf')):continue
        class_data[cls]['generated']+=1
        if complete is not None or deadline<=asfloat(manifest['horizon_s'],0)+1e-7:
            class_data[cls]['evaluable']+=1
        if t.get('scheduled') is not None:
            class_data[cls]['scheduled']+=1
        if complete is not None:
            class_data[cls]['completed']+=1
            if complete<=deadline+1e-7:class_data[cls]['on_time']+=1
            else:class_data[cls]['late']+=1
            latencies[('end_to_end',cls)].append(complete-arrive)
        else:class_data[cls]['unfinished']+=1
        if t.get('scheduled') is not None:
            placement[(cls,j.get('dataset') or j.get('dataset_id') or 'unknown',
                       (scheduled_by_job.get(ident) or {}).get('actual_placement') or 'unknown')]+=1
        intervals=[('scheduling_wait',arrive,t.get('scheduled')),
                   ('uplink_start_wait',t.get('scheduled'),t.get('upload_started')),
                   ('uplink_duration',t.get('upload_started'),t.get('upload_completed')),
                   ('compute_queue_wait',t.get('upload_completed'),t.get('compute_started')),
                   ('compute_duration',t.get('compute_started'),complete)]
        for label,left,right in intervals:
            if left is not None and right is not None:
                val=right-left
                if val< -1e-6:raise RuntimeError(f'Negative duration {label} job {ident}: {val}')
                latencies[(label,cls)].append(max(val,0))
    if sum(c['scheduled'] for c in class_data.values())==0 and sum(c['completed'] for c in class_data.values())>0:
        raise RuntimeError(f'Completed jobs but no scheduled events in {path}; unsupported event schema')
    class_rows=[];rates={}
    for cls in CLASSES:
        c=class_data[cls]
        total=c['evaluable'];rates[cls]=c['on_time']/total if total else None
        class_rows.append({'class':cls,'generated':c['generated'],'evaluable':total,'scheduled':c['scheduled'],
            'completed':c['completed'],'on_time':c['on_time'],'late':c['late'],
            'unfinished':c['unfinished'],'sla':rates[cls]})
    for rec in decisionrows:
        if not rec.get('scheduled'):continue
        jid=str(rec.get('task_id'))
        if jid not in byid:continue
        cls=byid[jid]['service_class']
        if rec.get('actual_placement') in ('edge','cloud'):
            network_tier[(cls,rec.get('actual_network_tier') or 'unknown')]+=1
        compute_tier[(cls,rec.get('actual_compute_tier') or 'unknown')]+=1
    sampled_util=[]
    for e in resources:
        capacity=asfloat(e.get('capacity_mbps'))
        used=asfloat(e.get('used_mbps'))
        if capacity is not None and used is not None and capacity>0:
            sampled_util.append(used/capacity)
    slot_util=[]
    for e in compute:
        es=asfloat(e.get('edge_slots'))
        cs=asfloat(e.get('cloud_slots'))
        if es and cs:
            slot_util.append({'edge':asfloat(e.get('edge_running'),0)/es,
                              'cloud':asfloat(e.get('cloud_running'),0)/cs})
    def mean(v):return sum(v)/len(v) if v else None
    busy=manifest.get('compute_busy_slot_seconds') or {}
    slotcap=manifest.get('compute_slot_capacity') or {}
    horizon=asfloat(manifest.get('horizon_s'))
    def actual_compute_util(place):
        seconds=asfloat(busy.get(place))
        capacity=asfloat(slotcap.get(place))
        if seconds is not None and horizon and capacity:
            return seconds/(horizon*capacity)
        return mean([x[place] for x in slot_util])
    summary={'fleet_uavs':manifest['fleet_uavs'],
             'run_id':manifest['run_id'],'policy':manifest['policy_label'],
             'mobility_seed':manifest['mobility_seed_label'],
             'task_seed':manifest['task_seed_label'],
             'episode_id':path.name,'network_provenance':manifest['network_provenance'],
             'generated':sum(x['generated'] for x in class_rows),
             'evaluable':sum(x['evaluable'] for x in class_rows),
             'overall_sla':sum(x['on_time'] for x in class_rows)/max(1,sum(x['evaluable'] for x in class_rows)),
             'critical_sla':rates['critical'],'standard_sla':rates['standard'],
             'batch_sla':rates['batch'],
             'minimum_class_sla':min(x for x in rates.values() if x is not None),
             'weighted_sla':weighted_sla(rates,{'critical':3.0,'standard':2.0,'batch':1.0}),
             'class_jain_fairness':jain(list(rates.values())),
             'total_network_mb':sum(asfloat(x.get('transmitted_mb'),0) for x in resources),
             'mean_network_utilization':mean(sampled_util),
             'p95_network_utilization':percentile(sampled_util,.95),
             'peak_network_utilization':max(sampled_util) if sampled_util else None,
             'mean_uav_allocated_mbps':mean([asfloat(x.get('allocated_mbps'),0) for x in allocation]),
             'mean_uav_used_mbps':mean([asfloat(x.get('used_mbps'),0) for x in allocation]),
             'mean_edge_queue_length':mean([asfloat(x.get('edge_queued'),0) for x in compute]),
             'mean_cloud_queue_length':mean([asfloat(x.get('cloud_queued'),0) for x in compute]),
             'max_edge_queue_length':max([asfloat(x.get('edge_queued'),0) for x in compute],default=None),
             'max_cloud_queue_length':max([asfloat(x.get('cloud_queued'),0) for x in compute],default=None),
             'mean_edge_slot_utilization':actual_compute_util('edge'),
             'mean_cloud_slot_utilization':actual_compute_util('cloud'),
             'defer_with_waiting_decision_ticks':sum(int(x.get('defer_with_pending_tasks',0)) for x in decisionrows),
             'decision_ticks':len(decisionrows)}
    intervals=[]
    for (key,cls),vals in sorted(latencies.items()):
        intervals.append({'run_id':manifest['run_id'],'episode_id':path.name,'class':cls,
            'metric':key,'observations':len(vals),'mean_s':mean(vals),
            'p95_s':percentile(vals,.95)})
    placements=[{'run_id':manifest['run_id'],'episode_id':path.name,
                 'class':c,'dataset':d,'placement':p,'scheduled_jobs':count}
                for (c,d,p),count in sorted(placement.items())]
    priorities=[{'run_id':manifest['run_id'],'episode_id':path.name,'class':c,
                 'network_tier':tier,'scheduled_offloads':cnt}
                for (c,tier),cnt in sorted(network_tier.items())]
    compute_priorities=[{'run_id':manifest['run_id'],'episode_id':path.name,'class':c,
            'compute_tier':tier,'scheduled_jobs':cnt}
            for (c,tier),cnt in sorted(compute_tier.items())]
    return summary,class_rows,placements,priorities,compute_priorities,intervals

def percentile(v,q):
    if not v:return None
    a=sorted(v);t=(len(a)-1)*q;l=int(t);h=min(l+1,len(a)-1)
    return a[l]+(a[h]-a[l])*(t-l)

def report(trace_root,output):
    root=Path(trace_root).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError(f'Refusing overwrite: {out}')
    paths=sorted(root.rglob('manifest.json'))
    if not paths:raise RuntimeError(f'No complete captured episodes under {root}; inspect runner.log')
    out.mkdir(parents=True)
    summaries=[];classes=[];placements=[];tiers=[];compute_tiers=[];intervals=[]
    for p in paths:
        summary,cl,pl,ti,ct,iv=summarise_episode(p.parent)
        summaries.append(summary)
        for x in cl:classes.append({'run_id':summary['run_id'],'episode_id':summary['episode_id'],
                                    'policy':summary['policy'],'fleet_uavs':summary['fleet_uavs'],**x})
        for x in pl:placements.append({'policy':summary['policy'],'fleet_uavs':summary['fleet_uavs'],**x})
        for x in ti:tiers.append({'policy':summary['policy'],'fleet_uavs':summary['fleet_uavs'],**x})
        for x in ct:compute_tiers.append({'policy':summary['policy'],'fleet_uavs':summary['fleet_uavs'],**x})
        for x in iv:intervals.append({'policy':summary['policy'],'fleet_uavs':summary['fleet_uavs'],**x})
    csvout(out/'episode_management_metrics.csv',summaries)
    csvout(out/'class_sla.csv',classes)
    csvout(out/'placement_by_class.csv',placements)
    csvout(out/'network_priority_by_class.csv',tiers)
    csvout(out/'compute_tier_by_class.csv',compute_tiers)
    csvout(out/'latency_components.csv',intervals)
    # Fractions are computed from actual scheduled decisions, not available actions.
    distributions=collections.defaultdict(collections.Counter)
    for rec in placements:
        key=(rec['fleet_uavs'],rec['policy'],rec['class'],'placement')
        distributions[key][rec['placement']]+=int(rec['scheduled_jobs'])
    for rec in tiers:
        key=(rec['fleet_uavs'],rec['policy'],rec['class'],'network_priority_offloads')
        distributions[key][rec['network_tier']]+=int(rec['scheduled_offloads'])
    for rec in compute_tiers:
        key=(rec['fleet_uavs'],rec['policy'],rec['class'],'compute_priority_scheduled')
        distributions[key][rec['compute_tier']]+=int(rec['scheduled_jobs'])
    action_fractions=[]
    for (fleet,policy,cls,kind),choices in sorted(distributions.items(),key=lambda kv:str(kv[0])):
        total=sum(choices.values())
        for choice,count in sorted(choices.items()):
            action_fractions.append({'fleet_uavs':fleet,'policy':policy,'class':cls,
                'decision_dimension':kind,'choice':choice,'count':count,
                'total_actions':total,'fraction':count/total if total else None})
    csvout(out/'action_distribution_by_fleet.csv',action_fractions)
    # Aggregate by fleet, policy, mobility seed rather than treating task episodes as independent channels.
    groups=collections.defaultdict(list)
    for r in summaries:groups[(r['fleet_uavs'],r['policy'],r['mobility_seed'])].append(r)
    aggregates=[]
    quantitative=('overall_sla','critical_sla','standard_sla','batch_sla','minimum_class_sla',
        'weighted_sla','class_jain_fairness','total_network_mb','mean_network_utilization',
        'p95_network_utilization','mean_uav_allocated_mbps','mean_uav_used_mbps',
        'mean_edge_queue_length','mean_cloud_queue_length',
        'mean_edge_slot_utilization','mean_cloud_slot_utilization')
    for (fleet,policy,seed),rows in sorted(groups.items(),key=lambda kv:str(kv[0])):
        rec={'fleet_uavs':fleet,'policy':policy,'mobility_seed':seed,'episode_count':len(rows)}
        for col in quantitative:
            vals=[asfloat(x.get(col)) for x in rows]
            vals=[x for x in vals if x is not None]
            rec[col]=sum(vals)/len(vals) if vals else None
        aggregates.append(rec)
    csvout(out/'fleet_policy_mobility_summary.csv',aggregates)
    doc={'created_utc':now(),'source_root':str(root),'episode_count':len(paths),
        'distinct_mobility_seeds':sorted(set(str(r['mobility_seed']) for r in summaries)),
        'files_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
        'limitations':['Fleet-policy mobility summaries are descriptive until >=3 independent mobility captures are present.',
        'Network capacity is analytical if trace provenance says uncalibrated.',
        'Queue/latency statistics condition on observed events; unfinished tasks are counted as SLA failures.',
        'Network requested tier and realized Mbps are distinct; a granted discrete tier is not inferred.']}
    dump(out/'report_manifest.json',doc)
    print(f'V13_REPORT_PASS episodes={len(paths)} cohorts={len(aggregates)} path={out}')

def permute(output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    scenarios={
      'original':{'CWD30':'critical','RumexWeeds':'standard','LeafySpurge':'batch'},
      'rumex_critical':{'CWD30':'standard','RumexWeeds':'critical','LeafySpurge':'batch'},
      'leafy_critical':{'CWD30':'batch','RumexWeeds':'standard','LeafySpurge':'critical'},
    }
    out.mkdir(parents=True)
    for name,mapping in scenarios.items():
        dump(out/f'{name}.json',{'schema_version':1,'name':name,
             'dataset_to_class':mapping,'deadline_policy':'preserve_absolute',
             'note':'Controlled relabeling of the same job inputs. Original map is a candidate assumption; verify against existing job records.',
             'generated_utc':now()})
    print('V13_PERMUTATIONS_READY:',out)

def seedplan(output,fleets,seeds):
    obj={'purpose':'Independent native UavNetSim mobility robustness evaluation',
         'network_model_provenance':'trajectory_driven_analytical_uncalibrated',
         'warnings':['The existing v10 capture wrapper may restrict seed IDs or require cleanup of unsuccessful runs.',
                     'New trajectories are distinct mobility captures, not measured wireless traces.',
                     'Do not bootstrap individual episodes as independent mobility realizations.'],
         'captures':[{'fleet_uavs':n,'seed':s,
                      'expected_capture_dir':f'results/uavnetsim_native_telemetry_{n}uav_seed{s}',
                      'suggested_capture_command':f'bash run_fieldvision_scale_v10.sh capture {n} --seed {s} --samples 250000 --poll-ms 1'}
                     for n in fleets for s in seeds], 'generated_utc':now()}
    dump(output,obj)
    print('V13_SEED_PLAN_READY:',output,'CAPTURES:',len(obj['captures']))

def capacity_fair_decisions(sim,target,weight):
    import fieldvision_noms_environment as env
    from dataclasses import dataclass
    obs=sim.observations()
    cls_seen=collections.Counter()
    cls_ontime=collections.Counter()
    for j in sim.jobs.values():
        if j.arrival_s>sim.time_s:continue
        if j.deadline_s<=sim.time_s+1e-8:
            cls_seen[j.service_class]+=1
            cls_ontime[j.service_class]+=int(j.completed_s is not None and j.completed_s<=j.deadline_s+1e-8)
    used_local=getattr(sim,'local_running',{})
    active=len(getattr(sim,'uploading',[]))
    decisions={}
    for uav in range(sim.n_uavs):
        observed=obs[uav]
        best=None
        for jobinfo in observed['waiting_tasks']:
            j=sim.jobs[jobinfo['task_id']]
            cls=j.service_class
            deadline_remaining=j.deadline_s-sim.time_s
            age=max(0.,sim.time_s-j.arrival_s)
            remaining=max(0.,j.deadline_s-j.arrival_s)
            current_sla=cls_ontime[cls]/cls_seen[cls] if cls_seen[cls] else 0.
            deficit=max(0.,target[cls]-current_sla)
            # Service deficit + urgency + age. Explicit heuristic, no learning claim.
            urgency=1./(max(deadline_remaining,0.05)+0.5)
            score=weight[cls]*deficit + 0.9*urgency + 0.7*age/max(.1,remaining)
            options=[]
            if uav not in used_local:options.append((j.local_runtime_s,'local'))
            cap=max(0.,min(observed['observed_uav_link_cap_mbps'],
                           observed['observed_shared_capacity_mbps']/max(1,active+1)))
            if cap>1e-9:
                tx=j.input_mb*8./cap
                for place in ('edge','cloud'):
                    busy=observed[f'observed_{place}_busy_slots']
                    slots=max(1,observed[f'{place}_slot_capacity'])
                    queued=observed[f'{place}_queue_length']
                    runtime=getattr(j,f'{place}_runtime_s')
                    options.append((tx+runtime*(1+queued/max(1,slots))+0.05*busy/max(1,slots),place))
            if not options:continue
            completion,place=min(options)
            feasible=(completion<=deadline_remaining+1e-9)
            rank=(int(feasible),score,-completion,str(j.job_id))
            if best is None or rank>best[0]:best=(rank,j,place)
        if best is None:
            decisions[uav]=env.Action()
            continue
        j=best[1];place=best[2]
        tier='high' if j.service_class=='critical' else ('standard' if j.service_class=='standard' else 'low')
        decisions[uav]=env.Action(j.job_id,place,tier,tier)
    return decisions

def fair_baseline(config,out,intent_path):
    import yaml
    import fieldvision_noms_environment as env
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    cfg=env.load_config(Path(config)) if hasattr(env,'load_config') else yaml.safe_load(Path(config).read_text())
    intent=yaml.safe_load(Path(intent_path).read_text())
    targets=intent['targets'];weights=intent['weights']
    if set(targets)!=set(CLASSES) or set(weights)!=set(CLASSES):raise ValueError('Intent must specify all three classes')
    if any(not 0<=float(v)<=1 for v in targets.values()):raise ValueError('SLA targets must be fractions [0,1]')
    if any(float(v)<=0 for v in weights.values()):raise ValueError('All class weights must be positive')
    sim=env.NOMSSim(cfg)
    while sim.time_s<sim.horizon-1e-7:
        sim.step(actions=capacity_fair_decisions(sim,targets,weights))
    result=sim.summary('v13_fair_capacity_edf')
    out.mkdir(parents=True)
    dump(out/'sla_summary.json',result)
    dump(out/'run_manifest.json',{'created_utc':now(),'config':str(Path(config).resolve()),
         'config_sha256':sha(config),'intent_sha256':sha(intent_path),
         'v13_source_sha256':sha(Path(__file__)),'policy':'v13_fair_capacity_edf',
         'claim':'Explicit heuristic baseline, not trained MAPPO.',
         'network_provenance':getattr(sim,'trace_provenance','unknown')})
    print('V13_FAIR_BASELINE_COMPLETE:',out)
    print(json.dumps(result,indent=2,sort_keys=True))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit');a.add_argument('--root',default='.')
    a=sub.add_parser('install');a.add_argument('--root',default='.');a.add_argument('--dry-run',action='store_true');a.add_argument('--apply',action='store_true')
    a=sub.add_parser('trace');a.add_argument('--root',default='.');a.add_argument('--output',required=True)
    a.add_argument('--run-id',required=True);a.add_argument('--policy',required=True)
    a.add_argument('--mobility-seed',default='unknown');a.add_argument('--task-seed',default='unknown')
    a.add_argument('--max-episodes',type=int,default=100);a.add_argument('--mapping')
    a.add_argument('external_command',nargs=argparse.REMAINDER)
    a=sub.add_parser('report');a.add_argument('--trace-root',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('permute');a.add_argument('--output',default='configs/noms_revision_v13/permutations')
    a=sub.add_parser('seed-plan');a.add_argument('--output',default='configs/noms_revision_v13/mobility_seed_plan.json')
    a.add_argument('--fleets',type=int,nargs='+',default=[16,32,64,128]);a.add_argument('--seeds',type=int,nargs='+',default=[53,67,79])
    a=sub.add_parser('fair-baseline');a.add_argument('--config',required=True)
    a.add_argument('--output',required=True);a.add_argument('--intent',default='configs/noms_revision_v13/intent_balanced.yaml')
    args=p.parse_args()
    if args.command=='audit':print(json.dumps(audit(args.root),indent=2,sort_keys=True))
    elif args.command=='install':
        if args.apply and args.dry_run:raise ValueError('Use either --dry-run or --apply')
        install(args.root,args.apply)
    elif args.command=='trace':
        args.command=args.external_command
        trace(args)
    elif args.command=='report':report(args.trace_root,args.output)
    elif args.command=='permute':permute(args.output)
    elif args.command=='seed-plan':seedplan(args.output,args.fleets,args.seeds)
    elif args.command=='fair-baseline':fair_baseline(args.config,args.output,args.intent)

if __name__=='__main__':main()

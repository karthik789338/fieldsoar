#!/usr/bin/env python3
"""FieldVision-NOMS v15: verified counterfactual replay and capacity/fairness baselines.

Reads frozen v14/v13 trace episode artifacts, validates policy-action replay,
then tests NEW deterministic scheduling/admission controls on physically identical
jobs, observed (input) link-capacity envelopes and compute-slot limits.

This does not train MAPPO, rerun actor inference, patch the frozen environment,
or claim field-calibrated radio data. Capacity scaling is counterfactual.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, random, sys, tempfile
from pathlib import Path

CLASSES=('critical','standard','batch')
SOURCENAMES=('jobs.csv','events.jsonl','decisions.jsonl','resources.jsonl')
POLS=('capacity_guard','capacity_fair','local_edf')
DEFAULT_TARGETS={'critical':.95,'standard':.80,'batch':.60}
DEFAULT_WEIGHTS={'critical':3.,'standard':2.,'batch':1.}
TOL=1e-7


def check(cond,message):
    if not cond:raise RuntimeError(message)

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def readcsv(path):
    with open(path,newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def readjsonl(path):
    with open(path,encoding='utf8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def writecsv(path,rows,fields=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    check(not p.exists(),f'Refusing overwrite {p}')
    columns=fields or list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)

def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    check(not p.exists(),f'Refusing overwrite {p}')
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def fval(x):
    y=float(x);check(math.isfinite(y),'Nonfinite input');return y

def manifest_index(root,episodes=None,policy='full',mobility=None):
    root=Path(root)
    check(root.is_dir(),f'Missing captured trace: {root}')
    docs=[];ids=set()
    for p in sorted(root.rglob('manifest.json')):
        doc=json.loads(p.read_text())
        check(doc.get('status')=='CAPTURE_COMPLETE',f'Incomplete episode: {p}')
        check(doc.get('policy_label')==policy,f'Mixed policy in source trace: {p}')
        if mobility is not None:
            check(str(doc.get('mobility_seed_label'))==str(mobility),f'Incorrect mobility seed in {p}')
        for name in SOURCENAMES:
            ff=p.parent/name
            check(ff.is_file() and file_sha(ff)==doc.get('file_sha256',{}).get(name),
                  f'Captured source hash mismatch {ff}')
        serial=int(doc['episode_serial'])
        check(serial not in ids,'Duplicate episode serial')
        ids.add(serial);docs.append((serial,p,doc))
    docs.sort(key=lambda x:x[0])
    check(docs,'No captured episodes')
    if episodes is not None:check(len(docs)==episodes,f'Episodes {len(docs)} != required {episodes}')
    return docs


def parse_jobs(rows):
    allowed=['job_id','uav_id','dataset','workload','arrival_s','deadline_s','input_mb',
             'local_runtime_s','edge_runtime_s','cloud_runtime_s','service_class',
             'profile_image_path','reference_model','compute_profile_source']
    output=[];seen=set()
    for row in rows:
        r={k:row[k] for k in allowed if k in row and row[k] not in ('',None,'None')}
        check(set(('job_id','uav_id','dataset','workload','arrival_s','deadline_s','input_mb',
                   'local_runtime_s','edge_runtime_s','cloud_runtime_s','service_class')).issubset(r),
              'Captured job missing required physical field')
        r['uav_id']=int(r['uav_id'])
        for k in ('arrival_s','deadline_s','input_mb','local_runtime_s','edge_runtime_s','cloud_runtime_s'):
            r[k]=fval(r[k])
        check(r['job_id'] not in seen,'Duplicate physical task IDs')
        check(r['deadline_s']>r['arrival_s'] and r['input_mb']>0,'Invalid physical task')
        check(r['service_class'] in CLASSES,'Unexpected class in recorded jobs')
        seen.add(r['job_id']);output.append(r)
    return output


def build_episode_source(folder,meta):
    horizon=fval(meta['horizon_s']);dt=fval(meta['tick_s'])
    n=int(meta['fleet_uavs']);nticks=round(horizon/dt)
    check(abs(nticks*dt-horizon)<1e-6 and n>=1,'Invalid time/fleet')
    jobs=parse_jobs(readcsv(folder/'jobs.csv'))
    check(all(0<=j['uav_id']<n for j in jobs),'Invalid UAV job')
    resources=list(readjsonl(folder/'resources.jsonl'))
    net={}; caps={}; slots={}
    for r in resources:
        kind=r.get('kind'); tick=round(fval(r.get('start_time_s',-999))/dt)
        if kind not in ('network','uav_network','compute'):continue
        check(0<=tick<nticks and abs(tick*dt-fval(r['start_time_s']))<1e-6,
              f'Misaligned resource timestamp {r}')
        if kind=='network':
            check(tick not in net,'Duplicated shared capacity input')
            net[tick]=fval(r['capacity_mbps'])
        elif kind=='uav_network':
            u=int(r['uav_id']);check(0<=u<n,'Incorrect UAV in radio trace')
            check((tick,u) not in caps,'Duplicated UAV capacity input')
            caps[tick,u]=fval(r['link_cap_mbps'])
        else:
            es=int(r['edge_slots']);cs=int(r['cloud_slots'])
            check(es>0 and cs>0,'Invalid compute slots')
            slots[tick]=(es,cs)
    check(len(net)==nticks and len(caps)==nticks*n and len(slots)==nticks,
          f'Incomplete resource timeline shared={len(net)}/{nticks} uav={len(caps)}/{nticks*n} slots={len(slots)}/{nticks}')
    check(len(set(slots.values()))==1, 'Dynamic compute slot limits not supported for this counterfactual')
    check(all(v>=0 for v in net.values()) and all(v>=0 for v in caps.values()),'Negative capacity')
    return {'n':n,'dt':dt,'horizon':horizon,'nticks':nticks,'jobs':jobs,'net':net,'caps':caps,
            'slots':next(iter(slots.values())),'episode_serial':int(meta['episode_serial']),
            'mobility':meta.get('mobility_seed_label'),
            'task_seed':meta.get('scenario_config_seed'),
            'provenance':meta.get('network_provenance')}


def make_sim(env,source,folder,scale=1.0):
    """Use captured physical jobs and capacity *inputs*, never achieved tx rates.
    A schema-adapter smoke label is used internally ONLY to load captured jobs;
    external research provenance is taken from verified v13 source manifests.
    """
    check(scale>0 and math.isfinite(scale),'Invalid capacity scale')
    csvpath=Path(folder)/'replayed_input_radio.csv'
    fields=['start_time_s','bottleneck_mbps']+[f'uav_{i}_link_cap_mbps' for i in range(source['n'])]
    rows=[]
    for tick in range(source['nticks']):
        r={'start_time_s':round(tick*source['dt'],9),
           'bottleneck_mbps':round(source['net'][tick]*scale,9)}
        r.update({f'uav_{i}_link_cap_mbps':source['caps'][tick,i] for i in range(source['n'])})
        rows.append(r)
    writecsv(csvpath,rows,fields)
    dump(csvpath.with_suffix('.manifest.json'),{
         'provenance':'trajectory_driven_analytical_uncalibrated',
         'generated_trace_sha256':file_sha(csvpath),
         'source':'v13 frozen MAPPO recorded capacity bounds only; policy-transmitted bytes are excluded',
         'bottleneck_multiplier':scale,
         'original_analytical_radio_uncalibrated':True})
    cfg={'schema_version':1,'seed':int(source.get('task_seed') or 0),
         'n_uavs':source['n'],'duration_s':source['horizon'],'tick_s':source['dt'],
         'workload_profile_source':'synthetic_smoke_only',
         'jobs':source['jobs'],
         'resources':{'edge_slots':source['slots'][0],'cloud_slots':source['slots'][1]},
         'network':{'bottleneck_mbps':source['net'][0]*scale,
                    'trace_csv':str(csvpath.resolve()),
                    'trace_provenance':'trajectory_driven_analytical_uncalibrated'}}
    # The smoke tag is an adapter to the simulator's explicit-job schema, not
    # a claim that the captured jobs are synthetic. See output provenance.
    sim=env.NOMSSim(cfg)
    check(len(sim.jobs)==len(source['jobs']),'Simulator altered job count')
    return sim


def original_actions(env,folder,source):
    rows=list(readjsonl(Path(folder)/'decisions.jsonl'))
    nt=source['nticks'];n=source['n'];dt=source['dt']
    check(len(rows)==nt*n,f'Incomplete learned actions {len(rows)}/{nt*n}')
    decisions=collections.defaultdict(dict)
    for d in rows:
        tick=round(fval(d['time_s'])/dt);u=int(d['uav_id'])
        check(0<=tick<nt and 0<=u<n,'Invalid learned action tick or UAV')
        check(u not in decisions[tick],'Duplicate learned action')
        scheduled=d.get('scheduled') is True
        if scheduled:
            check(d.get('task_selected_from_observed_event') is True,
                  'Captured learned action lacks scheduled-event confirmation')
            actions=(d.get('task_id'),d.get('actual_placement'),
                     d.get('actual_network_tier'),d.get('actual_compute_tier'))
            check(all(actions),'Incomplete scheduled learned action')
            a=env.Action(*actions)
        else:a=env.Action()
        decisions[tick][u]=a
    check(len(decisions)==nt and all(len(v)==n for v in decisions.values()),'Incomplete learned steps')
    return decisions


def timed_outcomes_from_events(events,source):
    completed={}
    for e in events:
        if e.get('event')=='completed':
            jid=e.get('task_id');check(jid not in completed,'Duplicate completion')
            completed[jid]=fval(e['time_s'])
    byid={j['job_id']:j for j in source['jobs']}
    check(set(completed).issubset(byid),'Completion unknown task')
    return {j['job_id']:int(j['job_id'] in completed and
            completed[j['job_id']]<=j['deadline_s']+TOL)
            for j in source['jobs'] if j['arrival_s']<source['horizon'] and
            (j['deadline_s']<=source['horizon']+TOL or j['job_id'] in completed)}


def validate_replay(env,source,folder,scratch):
    sim=make_sim(env,source,scratch)
    actions=original_actions(env,folder,source)
    for tick in range(source['nticks']):sim.step(actions=actions[tick])
    expected=timed_outcomes_from_events(list(readjsonl(Path(folder)/'events.jsonl')),source)
    replay=timed_outcomes_from_events(sim.events,source)
    check(expected==replay,f'CRITICAL REPLAY VALIDATION FAILED: on-time outcomes differ '
                             f'expected success={sum(expected.values())}/{len(expected)} '
                             f'replay={sum(replay.values())}/{len(replay)}')
    # Also verify transmissions and completed-task state, not just aggregate SLA.
    original_events=list(readjsonl(Path(folder)/'events.jsonl'))
    orig_comp={e['task_id'] for e in original_events if e.get('event')=='completed'}
    replay_comp={e['task_id'] for e in sim.events if e.get('event')=='completed'}
    check(orig_comp==replay_comp,'CRITICAL REPLAY VALIDATION FAILED: completed-task identities differ')
    return sim,expected


def jain(values):
    vals=[float(v) for v in values]
    den=len(vals)*sum(v*v for v in vals)
    return sum(vals)**2/den if den>0 else 0.


def summarize(sim,source,policy,scale,weights,targets,admissions):
    result=sim.summary(policy)
    cls={c:result['class_sla'].get(c,{}).get('sla_satisfaction_rate') for c in CLASSES}
    check(all(v is not None for v in cls.values()),'At least one class has no evaluable jobs')
    # Keep exactly the simulator's correct denominator for expired unfinished jobs.
    weighted=sum(float(cls[c])*weights[c] for c in CLASSES)/sum(weights.values())
    history=sim.network_history
    utilization=[r['used_mbps']/r['capacity_mbps'] if r['capacity_mbps']>0 else 0. for r in history]
    events=sim.events
    event_counts=collections.Counter(e['event'] for e in events)
    scheduled=collections.Counter(j.placement for j in sim.jobs.values() if j.placement)
    remote_queue_wait=[];tx_duration=[]
    uploaded={};upload_start={}
    for e in events:
        if e['event']=='upload_started':upload_start[e['task_id']]=e['time_s']
        if e['event']=='upload_completed':uploaded[e['task_id']]=e['time_s']
        if e['event']=='compute_started' and e.get('placement') in ('edge','cloud'):
            when=uploaded.get(e['task_id'])
            if when is not None:remote_queue_wait.append(max(0.,e['time_s']-when))
    for jid,at in uploaded.items():
        if jid in upload_start:tx_duration.append(max(0.,at-upload_start[jid]))
    summary={'episode_serial':source['episode_serial'],'task_seed':source['task_seed'],
      'fleet_uavs':source['n'],'mobility_seed':source['mobility'],'policy':policy,
      'bottleneck_multiplier':scale,'generated':result['generated_tasks'],
      'evaluable':result['sla_evaluable_tasks'],'overall_sla':result['sla_satisfaction_rate'],
      **{c+'_evaluable':int(result['class_sla'][c]['evaluable']) for c in CLASSES},
      **{c+'_on_time':int(result['class_sla'][c]['on_time']) for c in CLASSES},
      **{c+'_sla':float(cls[c]) for c in CLASSES},
      'minimum_class_sla':min(float(v) for v in cls.values()),
      'weighted_sla':weighted,'class_jain_fairness':jain(cls.values()),
      'max_target_deficit':max(max(0.,targets[c]-float(cls[c])) for c in CLASSES),
      'completed':result['completed_tasks'],'upload_completed':event_counts['upload_completed'],
      'scheduled_local':scheduled['local'],'scheduled_edge':scheduled['edge'],'scheduled_cloud':scheduled['cloud'],
      'mean_uplink_utilization':sum(utilization)/len(utilization) if utilization else 0.,
      'p95_uplink_utilization':sorted(utilization)[min(len(utilization)-1,int(math.ceil(.95*len(utilization)))-1)] if utilization else 0.,
      'mean_edge_queue_length':sum(r['edge_queued'] for r in sim.compute_history)/len(sim.compute_history),
      'mean_cloud_queue_length':sum(r['cloud_queued'] for r in sim.compute_history)/len(sim.compute_history),
      'edge_slot_utilization':result['edge_compute_slot_utilization'],
      'cloud_slot_utilization':result['cloud_compute_slot_utilization'],
      'mean_remote_queue_wait_s':(sum(remote_queue_wait)/len(remote_queue_wait) if remote_queue_wait else None),
      'mean_completed_upload_duration_s':(sum(tx_duration)/len(tx_duration) if tx_duration else None),
      'capacity_guard_candidate_rejections':admissions['capacity_reject'],
      'deadline_guard_candidate_rejections':admissions['deadline_reject'],
      'noncritical_scheduled_decisions':admissions['noncritical_scheduled']}
    return summary


class CapacityController:
    """Infrastructure-assisted heuristic with no future channel knowledge.

    The action policy uses *current* task metadata, shared current cap,
    per-UAV cap, active uploads and remote queues. Not a learned MARL policy.
    'capacity_fair' explicitly shifts local scheduling toward classes with
    unmet SLA targets. No positive performance claim is assumed.
    """
    def __init__(self,mode,targets,weights,min_rate=2.,reserve_critical=.85):
        self.mode=mode;self.targets=targets;self.weights=weights
        self.min_rate=min_rate;self.reserve_critical=reserve_critical
        self.counts=collections.Counter()

    def actions(self,sim,env):
        now=sim.time_s;obs=sim.observations();n=sim.n_uavs
        expired=collections.Counter();passed=collections.Counter()
        for job in sim.jobs.values():
            if job.arrival_s>=sim.horizon or job.arrival_s>now+TOL:continue
            if job.deadline_s<=now+TOL:
                expired[job.service_class]+=1
                if job.completed_s is not None and job.completed_s<=job.deadline_s+TOL:
                    passed[job.service_class]+=1
        deficit={c:max(0.,self.targets[c]-(passed[c]/expired[c] if expired[c] else 0.))
                 for c in CLASSES}
        active_uploads=len(sim.uploading)
        shared=sim.link_capacity_mbps()
        expected_uploads=active_uploads
        decisions={}
        for u in range(n):
            o=obs[u];options=[]
            for task in o['waiting_tasks']:
                job=sim.jobs[task['task_id']]
                rem=job.deadline_s-now
                if rem<=TOL:continue
                local_free=u not in sim.local_running
                local_est=math.ceil((job.local_runtime_s-TOL)/sim.dt)*sim.dt if local_free else math.inf
                local_feasible=local_est<=rem+TOL
                remote_places=[]
                if self.mode!='local_edf':
                    # Explicit cap for concurrent uploads; no fictional guaranteed throughput.
                    max_uploads=max(1,int(shared/max(self.min_rate,TOL))) if shared>0 else 0
                    if expected_uploads>=max_uploads:
                        self.counts['capacity_reject']+=1
                    else:
                        uav_cap=max(0.,sim.uav_link_cap_mbps(u))
                        fairshare=min(uav_cap,shared/max(1,expected_uploads+1))
                        for place in ('edge','cloud'):
                            runtime=job.edge_runtime_s if place=='edge' else job.cloud_runtime_s
                            cap=sim.edge_slots if place=='edge' else sim.cloud_slots
                            queued=len(sim.remote_queues[place]);busy=len(sim.remote_running[place])
                            # Coarse lower-bound estimate, cannot guarantee actual scheduling.
                            compute_wait=runtime*(queued+busy)/max(1,cap)
                            delay=job.input_mb*8./fairshare+runtime+compute_wait+sim.dt if fairshare>TOL else math.inf
                            if delay<=rem+TOL:remote_places.append((delay,place))
                            else:self.counts['deadline_reject']+=1
                candidates=[]
                if local_free:candidates.append((local_est,'local',local_feasible))
                for dur,place in remote_places:candidates.append((dur,place,True))
                if not candidates:continue
                feasible=[x for x in candidates if x[2]]
                if feasible:choice=min(feasible,key=lambda x:(x[0],x[1]))
                else:
                    # Only launch non-deadline-feasible local tasks if there is
                    # no feasible candidate; that may recover eventual completion
                    # but never falsely count them as on-time.
                    choice=min(candidates,key=lambda x:(x[0],x[1]))
                # Favor work actually able to finish on time before expired jobs.
                urgency=1./max(.10,rem)
                class_score=(2. if job.service_class=='critical' else 1.)*urgency
                if self.mode=='capacity_fair':
                    class_score+=2.5*self.weights[job.service_class]*deficit[job.service_class]
                    if job.service_class!='critical' and not local_feasible:
                        class_score-=.5
                    if (job.service_class=='critical' and expired['critical'] and
                        passed['critical']/expired['critical']<self.reserve_critical):
                        class_score+=3.
                else:class_score+=self.weights[job.service_class]
                rank=(int(bool(feasible)),class_score,-choice[0],job.job_id)
                options.append((rank,job,choice[1]))
            if not options:
                decisions[u]=env.Action();continue
            rank,job,place=max(options,key=lambda x:x[0])
            if self.mode=='capacity_fair' and job.service_class!='critical':
                self.counts['noncritical_scheduled']+=1
            if place!='local':expected_uploads+=1
            tier=('high' if job.service_class=='critical' else
                  'standard' if job.service_class=='standard' else 'low')
            if self.mode=='capacity_fair' and place!='local' and deficit[job.service_class]>.4:
                tier='high'
            decisions[u]=env.Action(job.job_id,place,
                                    'standard' if place=='local' else tier,
                                    'standard' if place=='local' else tier)
        return decisions


def run_baseline(env,src,work,policy,targets,weights,min_rate,scale):
    sim=make_sim(env,src,work,scale=scale)
    ctr=CapacityController(policy,targets,weights,min_rate)
    while sim.time_s<sim.horizon-TOL:
        if policy=='local_edf':
            acts={}
            for u,q in sim.waiting.items():
                if not q or u in sim.local_running:acts[u]=env.Action();continue
                jid=min(q,key=lambda k:(sim.jobs[k].deadline_s,k))
                acts[u]=env.Action(jid,'local','standard','standard')
        else:acts=ctr.actions(sim,env)
        sim.step(actions=acts)
    return summarize(sim,src,policy,scale,weights,targets,ctr.counts)


def ci_by_episode(rows,ref='frozen_MAPPO_action_replay',conf=.95,replicates=1200):
    grouped=collections.defaultdict(dict)
    for r in rows:
        if r['bottleneck_multiplier']==1.:
            check(r['policy'] not in grouped[r['episode_serial']],'Duplicate episode policy')
            grouped[r['episode_serial']][r['policy']]=r
    results=[]
    for policy in POLS:
        diffs={metric:[] for metric in ('overall_sla','critical_sla','standard_sla','batch_sla',
                                        'minimum_class_sla','weighted_sla','class_jain_fairness')}
        for episode,group in grouped.items():
            if policy not in group or ref not in group:continue
            for metric in diffs:
                diffs[metric].append(group[policy][metric]-group[ref][metric])
        for metric,vals in diffs.items():
            if not vals:continue
            rng=random.Random(5105)
            means=[sum(rng.choice(vals) for _ in vals)/len(vals) for _ in range(replicates)]
            means.sort()
            lo=means[int((1-conf)/2*replicates)]
            hi=means[min(replicates-1,int((1+conf)/2*replicates))]
            results.append({'policy':policy,'versus':ref,'metric':metric,'n_episodes':len(vals),
                            'gain_pp':100*sum(vals)/len(vals),'CI95_low_pp':100*lo,'CI95_high_pp':100*hi,
                            'inference_scope':'CONDITIONAL_ON_ONE_RECORDED_MOBILITY_SEED_NOT_INDEPENDENT_RADIO_CI'})
    return results


def summarize_campaign(rows):
    by=collections.defaultdict(list)
    for r in rows:by[(r['policy'],r['bottleneck_multiplier'])].append(r)
    out=[]
    for (policy,scale),group in sorted(by.items()):
        counts={c:(sum(r[c+'_on_time'] for r in group),sum(r[c+'_evaluable'] for r in group)) for c in CLASSES}
        slas={c:(counts[c][0]/counts[c][1] if counts[c][1] else None) for c in CLASSES}
        check(all(x is not None for x in slas.values()),'Aggregate class denominator missing')
        out.append({'policy':policy,'bottleneck_multiplier':scale,'n_episodes':len(group),
                    'overall_sla':sum(sum(r[c+'_on_time'] for c in CLASSES) for r in group)/
                                    sum(sum(r[c+'_evaluable'] for c in CLASSES) for r in group),
                    **{c+'_sla':slas[c] for c in CLASSES},
                    'minimum_class_sla':min(slas.values()),'class_jain_fairness':jain(slas.values()),
                    **{c+'_evaluable':counts[c][1] for c in CLASSES},
                    **{c+'_on_time':counts[c][0] for c in CLASSES},
                    'mean_uplink_utilization':sum(r['mean_uplink_utilization'] for r in group)/len(group),
                    'mean_edge_slot_utilization':sum(r['edge_slot_utilization'] for r in group)/len(group),
                    'mean_cloud_slot_utilization':sum(r['cloud_slot_utilization'] for r in group)/len(group),
                    'mean_scheduled_local':sum(r['scheduled_local'] for r in group)/len(group),
                    'mean_scheduled_edge':sum(r['scheduled_edge'] for r in group)/len(group),
                    'mean_scheduled_cloud':sum(r['scheduled_cloud'] for r in group)/len(group),
                    'scope':'MEASURED_REFERENCE_JOBS_REPLAYED_ANALYTICAL_RADIO; PAIRED_WITH_FROZEN_REPLAY_ONLY_AT_SCALE_1'})
    return out


def optimistic_isolated_feasibility(src):
    """Offline diagnostic bound: each task can use its own best FUTURE available
    radio rate and an otherwise idle processor. This relaxes all shared-resource
    contention and is not a feasible joint schedule or a runtime policy oracle.
    """
    totals=collections.Counter();feasible=collections.Counter();reason=collections.Counter()
    for job in src['jobs']:
        if job['arrival_s']>=src['horizon']:continue
        if job['deadline_s']>src['horizon']+TOL:continue  # unknown if not completed; bound due tasks only
        cl=job['service_class'];totals[cl]+=1
        start=max(0,min(src['nticks']-1,round(job['arrival_s']/src['dt'])))
        best_cap=max((min(src['net'][tick],src['caps'][tick,job['uav_id']])
                     for tick in range(start,src['nticks'])),default=0.)
        radio_lb=job['input_mb']*8./best_cap if best_cap>TOL else math.inf
        remote_lb=radio_lb+min(job['edge_runtime_s'],job['cloud_runtime_s'])
        best_lb=min(job['local_runtime_s'],remote_lb)
        slack=job['deadline_s']-job['arrival_s']
        possible=best_lb<=slack+TOL
        feasible[cl]+=int(possible)
        if not possible:reason[cl]+=1
    return [{'episode_serial':src['episode_serial'],'mobility_seed':src['mobility'],
             'class':cl,'due_jobs':totals[cl],
             'optimistically_isolated_feasible_jobs':feasible[cl],
             'optimistic_individual_feasible_upper_bound':feasible[cl]/totals[cl] if totals[cl] else None,
             'optimistic_individually_infeasible_jobs':reason[cl],
             'interpretation':'OFFLINE_PER_JOB_BEST_FUTURE_CAPACITY_LOWER_BOUND; NOT_JOINT_FEASIBILITY_OR_POLICY_INFORMATION'}
            for cl in CLASSES]


def load_historical_context(path,fleet,protocol='independent'):
    rows=readcsv(path)
    check(rows,'Empty historical policy table')
    output=[]
    for r in rows:
        if int(r.get('fleet_uavs',0))!=fleet or r.get('protocol')!=protocol:continue
        if r.get('variant') not in ('full','full_constrained','no_class_metadata'):continue
        output.append({'policy':r['variant'], 'fleet_uavs':fleet,'protocol':protocol,
            **{key:fval(r[key]) for key in ('overall_sla','critical_sla','standard_sla','batch_sla')},
            'comparison_warning':'REFERENCE_ONLY_UNPAIRED_AGGREGATES_TO_NEW_HEURISTIC_UNLESS_IDENTICAL_SCENARIOS_VERIFIED'})
    check(output,'No matching historical policy rows in context table')
    return output


def run_campaign(a):
    import fieldvision_noms_environment as env
    check(hasattr(env,'NOMSSim') and hasattr(env,'Action'),'Current environment lacks NOMSSim/Action')
    root=Path(a.trace_root)
    docs=manifest_index(root,a.episodes,a.policy,a.mobility_seed)
    check(all(x>0 for x in a.capacity_scales),'Invalid scale')
    check(len(set(a.capacity_scales))==len(a.capacity_scales),'Duplicate scales')
    check(1.0 in a.capacity_scales,'Required original capacity scale 1.0 to validate paired MAPPO')
    check(a.min_remote_rate>0,'Invalid minimum admission rate')
    check(a.limit_episodes is None or 0<a.limit_episodes<=len(docs),'Invalid pilot episode count')
    targets,weights=load_intent(Path(a.intent))
    out=Path(a.output)
    check(not out.exists(),'Output exists: '+str(out))
    allrows=[];validations=[];input_hashes=[];feasibility=[]
    chosen=docs[:a.limit_episodes] if a.limit_episodes is not None else docs
    check(len(chosen)>0,'No episodes selected for evaluation')
    with tempfile.TemporaryDirectory(prefix='fieldvision_v15_inputs_') as tmp:
        for serial,manifest,meta in chosen:
            folder=manifest.parent
            src=build_episode_source(folder,meta)
            check(src['provenance']=='trajectory_driven_analytical_uncalibrated',
                  'Unapproved wireless provenance')
            feasibility.extend(optimistic_isolated_feasibility(src))
            with tempfile.TemporaryDirectory(dir=tmp) as eptmp:
                orig,truth=validate_replay(env,src,folder,Path(eptmp)/'original')
                check(len(truth)>0,'No evaluable original tasks')
                validations.append({'serial':serial,'n_evaluable':len(truth),
                                    'matched_ontime_jobs':sum(truth.values()),
                                    'original_manifest_sha256':file_sha(manifest),
                                    'replay_validation':'EXACT_JOB_LEVEL_ONTIME_AND_COMPLETION_MATCH'})
                origrow=summarize(orig,src,'frozen_MAPPO_action_replay',1.,weights,targets,collections.Counter())
                allrows.append(origrow)
                for scale in a.capacity_scales:
                    for pol in a.policies:
                        work=Path(eptmp)/f'{pol}_scale_{scale:g}'
                        row=run_baseline(env,src,work,pol,targets,weights,a.min_remote_rate,scale)
                        allrows.append(row)
            input_hashes.append(file_sha(manifest))
            print(f'VERIFIED_PAIRED_EPISODE serial={serial} frozen_ontime={sum(truth.values())}/{len(truth)}',flush=True)
    # Output only after validation of all requested episodes, never overwrite.
    out.mkdir(parents=True,exist_ok=False)
    writecsv(out/'paired_episode_metrics.csv',allrows)
    writecsv(out/'policy_aggregate_summary.csv',summarize_campaign(allrows))
    if a.context_table:
        proto=a.context_protocol
        writecsv(out/'historical_frozen_policy_context_UNPAIRED.csv',
                 load_historical_context(a.context_table,allrows[0]['fleet_uavs'],proto))
    writecsv(out/'paired_episode_differences_vs_frozen_MAPPO.csv',ci_by_episode(allrows))
    report={'command':'fieldvision_v15_capacity_admission_and_class_fairness',
        'input_trace_root':str(root.resolve()),'verified_source_episodes_available':len(docs),'replayed_episodes':len(chosen),
        'n_uavs':allrows[0]['fleet_uavs'],'mobility_seed':allrows[0]['mobility_seed'],
        'policy_controls':a.policies,'capacity_multipliers':a.capacity_scales,
        'min_remote_rate_mbps':a.min_remote_rate,'intent_path':str(Path(a.intent).resolve()),
        'intent_sha256':file_sha(a.intent),'source_episode_manifest_hashes':input_hashes,
        'environment_source_sha256':file_sha(Path(env.__file__)),
        'validation':'ALL_EPISODES_MATCH_RECORDED_MAPPO_JOB_COMPLETIONS_AND_ON_TIME_OUTCOMES',
        'replay_note':'Reference-job replay uses the simulator explicit-job adapter. The actual physical jobs come verbatim from immutable measured-reference frozen traces; no new model inference is performed.',
        'radio_note':'Historical analytical radio capacity bounds (not observed used Mbps) replayed. Not independently calibrated field radio.',
        'comparisons':'Paired at original recorded capacity, episode bootstrap conditional on one mobility seed. Provisioning scales are COUNTERFACTUAL; learned MAPPO is NOT reevaluated at multiplied capacity.',
        'method_limitation':'The new controller is an explicit infrastructure-assisted heuristic, not retrained fairness-aware MAPPO.',
        'feasibility_bound_note':'Optimistic per-task bound assumes uninterrupted use of best future single-link capacity and no contention; not a jointly schedulable upper bound.'}
    dump(out/'v15_provenance.json',report)
    writecsv(out/'replay_integrity.csv',validations)
    writecsv(out/'optimistic_isolated_feasibility_by_class.csv',feasibility)
    print('V15_PAIRED_CAPACITY_FAIR_CAMPAIGN_PASS episodes=',len(chosen),'output=',out)


def load_intent(path):
    import yaml
    doc=yaml.safe_load(path.read_text())
    target=doc['targets'];weights=doc['weights']
    check(set(target)==set(CLASSES) and set(weights)==set(CLASSES),'Intent missing class')
    target={k:fval(v) for k,v in target.items()};weights={k:fval(v) for k,v in weights.items()}
    check(all(0<=v<=1 for v in target.values()) and all(v>0 for v in weights.values()),'Invalid intent')
    return target,weights


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['preflight','run'])
    ap.add_argument('--trace-root',required=True)
    ap.add_argument('--episodes',type=int,default=None)
    ap.add_argument('--limit-episodes',type=int,default=None)
    ap.add_argument('--context-table',default=None)
    ap.add_argument('--context-protocol',choices=['nominal','independent'],default='independent')
    ap.add_argument('--policy',default='full')
    ap.add_argument('--mobility-seed',type=int,default=None)
    ap.add_argument('--intent',default='configs/noms_revision_v13/intent_balanced.yaml')
    ap.add_argument('--capacity-scales',type=float,nargs='+',default=[1.0])
    ap.add_argument('--policies',nargs='+',choices=POLS,default=list(POLS))
    ap.add_argument('--min-remote-rate',type=float,default=2.)
    ap.add_argument('--output',default=None)
    args=ap.parse_args(argv)
    try:
        if args.command=='preflight':
            import fieldvision_noms_environment as env
            docs=manifest_index(args.trace_root,args.episodes,args.policy,args.mobility_seed)
            check(args.intent and Path(args.intent).is_file(),'Missing intent YAML')
            _=load_intent(Path(args.intent))
            serial,m,meta=docs[0]
            source=build_episode_source(m.parent,meta)
            with tempfile.TemporaryDirectory(prefix='fieldvision_v15_preflight_') as tmp:
                _,truth=validate_replay(env,source,m.parent,Path(tmp)/'original')
            print(f'V15_PREFLIGHT_PASS episodes_available={len(docs)} sample_serial={serial} '
                  f'matched_on_time={sum(truth.values())}/{len(truth)} original_source_unchanged=true')
        else:
            check(args.output,'--output is required for run')
            run_campaign(args)
    except (RuntimeError,ValueError,KeyError,TypeError,OSError,AssertionError) as ex:
        print('V15_REPLAY_GATE_FAILED:',ex,file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""FieldVision NOMS v15.1: fixed-cohort SLA audit and admission-control recovery.

Additive experiment on immutable v14 traces. Reuses already computed v15 MAPPO
replay metrics (and verifies them against source events), does not train MAPPO,
change checkpoints, or call any UavNetSim capture. Heuristics only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import sys
import tempfile
from pathlib import Path

import fieldvision_noms_capacity_fair_v15 as v15

CLASSES=v15.CLASSES
POLICIES=('critical_reserve','standard_rescue','joint_recovery')
METRICS=('overall_due_sla','critical_due_sla','standard_due_sla','batch_due_sla',
         'minimum_class_due_sla','class_due_jain','weighted_due_sla')


def require(condition,message):
    v15.check(condition,message)


def nonempty_records(path):
    p=Path(path)
    require(p.is_file(),f'Missing file: {p}')
    r=v15.readcsv(p)
    require(bool(r),f'Empty CSV: {p}')
    return r


def fixed_due_scores(jobs, completed_times, horizon, targets, weights):
    """Same predeclared cohort across all policies: arrived by horizon AND due by horizon.

    Unlike NOMSSim.summary, excludes early-completed tasks with a deadline beyond
    horizon from both numerator and denominator. Such completions are reported
    separately. This protects paired comparisons from policy-induced censoring.
    """
    completed={str(k):float(v) for k,v in completed_times.items()}
    counts=collections.Counter();success=collections.Counter();early=collections.Counter()
    for job in jobs:
        if isinstance(job,dict):
            jid=str(job['job_id']);cl=job['service_class'];arrival=job['arrival_s'];deadline=job['deadline_s']
        else:
            jid=str(job.job_id);cl=job.service_class;arrival=job.arrival_s;deadline=job.deadline_s
        if arrival>=horizon-v15.TOL: continue
        if deadline>horizon+v15.TOL:
            if jid in completed and completed[jid]<=deadline+v15.TOL:early[cl]+=1
            continue
        counts[cl]+=1
        if jid in completed and completed[jid]<=deadline+v15.TOL:success[cl]+=1
    require(all(counts[c] for c in CLASSES),'Missing one or more due-job cohorts')
    sla={c:success[c]/counts[c] for c in CLASSES}
    total=sum(counts.values());good=sum(success.values())
    return {'due_total':total,'due_on_time':good,'overall_due_sla':good/total,
        **{f'{c}_due_count':counts[c] for c in CLASSES},
        **{f'{c}_due_on_time':success[c] for c in CLASSES},
        **{f'{c}_due_sla':sla[c] for c in CLASSES},
        **{f'{c}_completed_before_horizon_but_not_due':early[c] for c in CLASSES},
        'minimum_class_due_sla':min(sla.values()),
        'class_due_jain':v15.jain(sla.values()),
        'weighted_due_sla':sum(weights[c]*sla[c] for c in CLASSES)/sum(weights.values()),
        'max_due_target_deficit':max(max(0.,targets[c]-sla[c]) for c in CLASSES)}


def completed_times_from_events(events):
    output={}
    for e in events:
        if e.get('event')=='completed':
            jid=str(e['task_id'])
            require(jid not in output,'Duplicate completion event')
            output[jid]=float(e['time_s'])
    return output


def audit_existing(v15_root, output):
    """Reconcile v15 pooled SLA with equal-episode paired statistics without reruns."""
    root=Path(v15_root)
    arr=nonempty_records(root/'paired_episode_metrics.csv')
    summary=nonempty_records(root/'policy_aggregate_summary.csv')
    ref='frozen_MAPPO_action_replay'
    group=collections.defaultdict(dict)
    for r in arr:
        if float(r['bottleneck_multiplier'])!=1.0:continue
        episode=int(r['episode_serial']);p=r['policy']
        require(p not in group[episode],f'Duplicate {episode} {p}')
        group[episode][p]=r
    require(group,'No original 1x paired episodes')
    policies=set.intersection(*(set(g.keys()) for g in group.values()))
    require(ref in policies,'Missing frozen reference in cached v15 results')
    output_rows=[]
    agg_index={(r['policy'],float(r['bottleneck_multiplier'])):r for r in summary}
    for pol in sorted(policies):
        for c in ('overall','critical','standard','batch'):
            metric=f'{c}_sla'
            rpp=[];total_num=0;total_den=0;change=[];den_mismatches=0
            for ep in sorted(group):
                s=group[ep][pol];m=group[ep][ref]
                rpp.append(float(s[metric]));change.append(float(s[metric])-float(m[metric]))
                if c=='overall':
                    n=sum(int(s[f'{cl}_on_time']) for cl in CLASSES)
                    d=sum(int(s[f'{cl}_evaluable']) for cl in CLASSES)
                    d0=sum(int(m[f'{cl}_evaluable']) for cl in CLASSES)
                else:
                    n=int(s[f'{c}_on_time']);d=int(s[f'{c}_evaluable']);d0=int(m[f'{c}_evaluable'])
                total_num+=n;total_den+=d;den_mismatches+=int(d!=d0)
            reported=float(agg_index[pol,1.][metric])
            pooled=total_num/total_den
            require(abs(reported-pooled)<1e-5,f'Aggregate-v15 mismatch: {pol}/{metric}')
            output_rows.append({'policy':pol,'metric':metric,'episodes':len(group),
                'pooled_sla':pooled,'equal_episode_mean_sla':sum(rpp)/len(rpp),
                'pooled_gain_vs_frozen_pp':100*(pooled-float(agg_index[ref,1.][metric])),
                'equal_episode_gain_vs_frozen_pp':100*sum(change)/len(change),
                'episode_count_with_policy_dependent_denominator':den_mismatches,
                'denominator_contract':'ORIGINAL_V15_COMPLETED_OR_DUE; NOT_PREDECLARED_DUE_ONLY'})
    out=Path(output)
    require(not out.exists(),f'Refusing to overwrite {out}')
    v15.writecsv(out,output_rows)
    print('V15_1_ACCOUNTING_AUDIT_PASS episodes=',len(group),'rows=',len(output_rows),'file=',out)
    return output_rows


class RecoveryController:
    """Explicit non-learning intervention; only current observations and queues.

    critical_reserve: give feasible waiting critical work a local CPU before any
      noncritical action; hold noncritical local starts during a declared startup
      guard window (no future-arrival information is accessed).
    standard_rescue: prefer locally feasible standard work when there is no
      waiting critical task and the standard target deficit remains nonzero.
    joint_recovery: both controls; standard recovery begins after hold window.

    This is an infrastructure-assisted heuristic, not learned MAPPO or a
    clairvoyant future-arrival scheduler. Guard window is declared ex ante.
    """
    def __init__(self,mode,targets,weights,min_rate,hold_s):
        require(mode in POLICIES,'Unknown recovery controller')
        require(min_rate>0 and hold_s>=0,'Invalid control parameters')
        self.mode=mode;self.targets=targets;self.weights=weights
        self.hold_s=hold_s
        self.core=v15.CapacityController('capacity_guard',targets,weights,min_rate)
        self.counts=collections.Counter()

    @staticmethod
    def local_fit(job,sim):
        return job.local_runtime_s<=job.deadline_s-sim.time_s+v15.TOL

    def actions(self,sim,env):
        base=self.core.actions(sim,env)
        now=sim.time_s
        due=collections.Counter();passed=collections.Counter()
        for job in sim.jobs.values():
            if job.arrival_s>=sim.horizon or job.arrival_s>now+v15.TOL:continue
            if job.deadline_s<=now+v15.TOL:
                due[job.service_class]+=1
                if job.completed_s is not None and job.completed_s<=job.deadline_s+v15.TOL:
                    passed[job.service_class]+=1
        std_observed=(passed['standard']/due['standard'] if due['standard'] else 0.0)
        deficit=max(0.,self.targets['standard']-std_observed)
        for u in range(sim.n_uavs):
            if u in sim.local_running:continue
            available=[sim.jobs[jid] for jid in sim.waiting[u]]
            critical=[j for j in available if j.service_class=='critical' and self.local_fit(j,sim)]
            if self.mode in ('critical_reserve','joint_recovery') and critical:
                j=min(critical,key=lambda z:(z.deadline_s,z.local_runtime_s,z.job_id))
                base[u]=env.Action(j.job_id,'local','standard','standard')
                self.counts['forced_critical_local']+=1
                continue
            candidate=base.get(u,env.Action())
            if self.mode in ('critical_reserve','joint_recovery') and now<self.hold_s:
                if candidate.task_id and candidate.placement=='local':
                    job=sim.jobs[candidate.task_id]
                    if job.service_class!='critical':
                        base[u]=env.Action()
                        self.counts['startup_noncritical_local_deferrals']+=1
                        candidate=base[u]
            # Respect observed critical backlog: do not substitute a standard
            # job when even one critical task is waiting, including infeasible
            # critical tasks (the underlying capacity guard handles those).
            if self.mode in ('standard_rescue','joint_recovery') and not any(
                j.service_class=='critical' for j in available):
                if self.mode=='joint_recovery' and now<self.hold_s:continue
                std=[j for j in available if j.service_class=='standard' and self.local_fit(j,sim)]
                if std and deficit>0:
                    j=min(std,key=lambda z:(z.deadline_s,z.local_runtime_s,z.job_id))
                    base[u]=env.Action(j.job_id,'local','standard','standard')
                    self.counts['forced_standard_local']+=1
        for u,a in base.items():
            if a.task_id:
                cl=sim.jobs[a.task_id].service_class
                self.counts[f'scheduled_{cl}']+=1
                self.counts[f'scheduled_{cl}_{a.placement}']+=1
        return base


def run_new_policy(env,source,scratch,mode,targets,weights,min_rate,hold_s):
    sim=v15.make_sim(env,source,scratch,scale=1.)
    ctl=RecoveryController(mode,targets,weights,min_rate,hold_s)
    while sim.time_s<sim.horizon-v15.TOL:
        acts=ctl.actions(sim,env)
        sim.step(actions=acts)
    original=v15.summarize(sim,source,mode,1.,weights,targets,ctl.core.counts)
    new=fixed_due_scores(list(sim.jobs.values()),
                         {j.job_id:j.completed_s for j in sim.jobs.values() if j.completed_s is not None},
                         source['horizon'],targets,weights)
    stages=collections.Counter((j.service_class,j.stage) for j in sim.jobs.values() if j.arrival_s<sim.horizon)
    original.update(new)
    for cl in CLASSES:
        original[f'{cl}_not_scheduled_at_horizon']=stages[cl,'waiting']
        original[f'{cl}_still_uploading_at_horizon']=stages[cl,'uploading']
        original[f'{cl}_remote_queued_at_horizon']=stages[cl,'remote_queued']
        original[f'{cl}_completed_any_time']=stages[cl,'completed']
        original[f'{cl}_scheduled_local']=ctl.counts[f'scheduled_{cl}_local']
        original[f'{cl}_scheduled_edge']=ctl.counts[f'scheduled_{cl}_edge']
        original[f'{cl}_scheduled_cloud']=ctl.counts[f'scheduled_{cl}_cloud']
    original.update({'forced_critical_local':ctl.counts['forced_critical_local'],
                     'forced_standard_local':ctl.counts['forced_standard_local'],
                     'startup_noncritical_local_deferrals':ctl.counts['startup_noncritical_local_deferrals'],
                     'hold_s':hold_s,'comparison_scope':'ACTUAL_RECORDED_RADIO_1X_ONLY'})
    return original


def due_summary(rows):
    grouped=collections.defaultdict(list)
    for r in rows:grouped[(r['policy'],r.get('intent_label','balanced'))].append(r)
    output=[]
    for (pol,intent),g in sorted(grouped.items()):
        c={cl:(sum(int(r[f'{cl}_due_on_time']) for r in g),
               sum(int(r[f'{cl}_due_count']) for r in g)) for cl in CLASSES}
        slas={cl:c[cl][0]/c[cl][1] for cl in CLASSES}
        total=sum(c[cl][1] for cl in CLASSES);passed=sum(c[cl][0] for cl in CLASSES)
        base={'policy':pol,'intent':intent,'episodes':len(g),'overall_due_sla':passed/total,
              'minimum_class_due_sla':min(slas.values()),'class_due_jain':v15.jain(slas.values())}
        base.update({f'{cl}_due_sla':slas[cl] for cl in CLASSES})
        base.update({f'{cl}_due_count':c[cl][1] for cl in CLASSES})
        base.update({f'{cl}_due_on_time':c[cl][0] for cl in CLASSES})
        base.update({'mean_weighted_due_sla_equal_episode':sum(float(r['weighted_due_sla']) for r in g)/len(g),
                     'mean_max_due_target_deficit':sum(float(r['max_due_target_deficit']) for r in g)/len(g),
                     'mean_uplink_utilization':sum(float(r['mean_uplink_utilization']) for r in g)/len(g),
                     'mean_edge_slot_utilization':sum(float(r['edge_slot_utilization']) for r in g)/len(g),
                     'mean_cloud_slot_utilization':sum(float(r['cloud_slot_utilization']) for r in g)/len(g)})
        output.append(base)
    return output


def paired_due_ci(rows,seed=151151,replicates=3000):
    by=collections.defaultdict(dict)
    for r in rows:
        k=(int(r['episode_serial']),r.get('intent_label','balanced'))
        require(r['policy'] not in by[k],'Duplicate episode/policy record')
        by[k][r['policy']]=r
    out=[]
    for intent in sorted({i for _,i in by}):
        subset=[v for (e,i),v in by.items() if i==intent]
        for pol in POLICIES:
            selected=[v for v in subset if pol in v and 'frozen_MAPPO_action_replay' in v]
            if not selected:continue
            for metric in METRICS:
                delta=[float(g[pol][metric])-float(g['frozen_MAPPO_action_replay'][metric]) for g in selected]
                rng=random.Random(seed)
                boots=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(replicates))
                out.append({'intent':intent,'policy':pol,'metric':metric,
                    'gain_pp_equal_episode_mean':100*sum(delta)/len(delta),
                    'CI95_low_pp':100*boots[int(.025*replicates)],
                    'CI95_high_pp':100*boots[int(.975*replicates)-1],
                    'episodes':len(delta),'CI_scope':'ONE_MOBILITY_SEED_EPISODE_BOOTSTRAP_NOT_INDEPENDENT_RADIO'})
    return out


def run(args):
    import fieldvision_noms_environment as env
    require(args.episodes>0,'Specify a positive episode count')
    require(args.min_remote_rate>0 and args.hold_s>=0,'Invalid baseline parameter')
    require(args.output,'--output required')
    out=Path(args.output)
    require(not out.exists(),f'Output already exists {out}')
    frozen_root=Path(args.v15_results)
    cached=nonempty_records(frozen_root/'paired_episode_metrics.csv')
    integrity=nonempty_records(frozen_root/'replay_integrity.csv')
    cached_by_serial={}
    for row in cached:
        if row['policy']=='frozen_MAPPO_action_replay' and float(row['bottleneck_multiplier'])==1.:
            ser=int(row['episode_serial'])
            require(ser not in cached_by_serial,'Duplicate cached frozen record')
            cached_by_serial[ser]=row
    hash_by_serial={int(r['serial']):r for r in integrity}
    docs=v15.manifest_index(args.trace_root,args.episodes,args.policy,args.mobility_seed)
    require(args.limit_episodes is None or 0<args.limit_episodes<=len(docs),'Invalid pilot limit')
    require(len(cached_by_serial)==len(docs)==len(hash_by_serial),
            'Existing v15 replay records are incomplete or do not match requested episode count')
    intents=[]
    for path in args.intents:
        p=Path(path);require(p.is_file(),f'Missing intent: {p}')
        targets,weights=v15.load_intent(p)
        name=p.stem;require(name not in [i[0] for i in intents],'Duplicate intent labels')
        intents.append((name,p,targets,weights))
    require(intents,'Specify one or more intent configurations')
    docs=docs[:args.limit_episodes] if args.limit_episodes is not None else docs
    allrows=[];validation=[]
    with tempfile.TemporaryDirectory(prefix='fieldvision_v151_') as root:
        for serial,manifest,meta in docs:
            require(serial in cached_by_serial and serial in hash_by_serial,
                    f'Unmatched serial {serial}')
            require(v15.file_sha(manifest)==hash_by_serial[serial]['original_manifest_sha256'],
                    f'Version mismatch against verified v15 input, episode {serial}')
            src=v15.build_episode_source(manifest.parent,meta)
            require(src['provenance']=='trajectory_driven_analytical_uncalibrated',
                    'Unexpected radio provenance')
            events=list(v15.readjsonl(manifest.parent/'events.jsonl'))
            completion=completed_times_from_events(events)
            truth=v15.timed_outcomes_from_events(events,src)
            cache=cached_by_serial[serial]
            for cl in CLASSES:
                recorded=sum(truth[j['job_id']] for j in src['jobs'] if j['service_class']==cl and j['job_id'] in truth)
                require(recorded==int(cache[f'{cl}_on_time']),
                        f'Cached frozen and original v14 job outcomes disagree in episode {serial}/{cl}')
            validation.append({'episode_serial':serial,'source_manifest_sha256':v15.file_sha(manifest),
                               'cached_v15_frozen_outcomes':'MATCHED_EXACT_CLASS_JOB_COUNTS'})
            for intent,p,targets,weights in intents:
                frozen_fixed=fixed_due_scores(src['jobs'],completion,src['horizon'],targets,weights)
                ref={k:(int(v) if k in ('episode_serial','fleet_uavs') else v) for k,v in cache.items()}
                ref.update(frozen_fixed)
                ref['intent_label']=intent
                ref['comparison_scope']='EXACT_RECORDED_FROZEN_ACTIONS_AT_1X'
                allrows.append(ref)
                for policy in POLICIES:
                    scratch=Path(root)/f'ep{serial}_{intent}_{policy}'
                    new=run_new_policy(env,src,scratch,policy,targets,weights,
                                       args.min_remote_rate,args.hold_s)
                    new['intent_label']=intent
                    allrows.append(new)
            print(f'V15_1_FIXED_DUE_EPISODE_VERIFIED serial={serial} source_cached=true',flush=True)
    out.mkdir(parents=True,exist_ok=False)
    v15.writecsv(out/'paired_episode_fixed_due_metrics.csv',allrows)
    v15.writecsv(out/'fixed_due_policy_summary.csv',due_summary(allrows))
    v15.writecsv(out/'paired_due_differences_vs_frozen.csv',paired_due_ci(allrows))
    v15.writecsv(out/'input_integrity.csv',validation)
    dump={'campaign':'v15.1 admission and service recovery',
          'source_trace':str(Path(args.trace_root).resolve()),
          'reused_v15_results':str(frozen_root.resolve()),
          'reused_original_reference_without_rerunning_learned_policy':True,
          'episode_count':len(docs),'mobility_seed':args.mobility_seed,
          'intents':{name:{'path':str(path.resolve()),'sha256':v15.file_sha(path)} for name,path,_,_ in intents},
          'policies':list(POLICIES),'fixed_due_cohort':'arrived before horizon and deadline at/before horizon; same jobs for all policies',
          'hold_s':args.hold_s,'min_remote_rate_mbps':args.min_remote_rate,
          'source_env_sha256':v15.file_sha(env.__file__),
          'research_limitation':'New controllers are heuristics; no MAPPO retraining, no future radio lookahead, one independent mobility seed.',
          'recorded_actions_not_reexecuted':True}
    v15.dump(out/'v15_1_provenance.json',dump)
    print('V15_1_RECOVERY_CAMPAIGN_PASS episodes=',len(docs),'intents=',len(intents),'output=',out)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    audit=sub.add_parser('audit-existing',help='Read-only reconcile saved v15 means and denominators')
    audit.add_argument('--v15-results',required=True)
    audit.add_argument('--output',required=True)
    runsub=sub.add_parser('run',help='Run only NEW heuristic policies on already captured episodes')
    runsub.add_argument('--trace-root',required=True)
    runsub.add_argument('--v15-results',required=True)
    runsub.add_argument('--episodes',type=int,default=75)
    runsub.add_argument('--limit-episodes',type=int,default=None,
                        help='Optional smoke pilot; source and cached results must still contain --episodes')
    runsub.add_argument('--policy',default='full')
    runsub.add_argument('--mobility-seed',type=int,default=41)
    runsub.add_argument('--intents',nargs='+',default=['configs/noms_revision_v13/intent_balanced.yaml'])
    runsub.add_argument('--hold-s',type=float,default=.6)
    runsub.add_argument('--min-remote-rate',type=float,default=2.)
    runsub.add_argument('--output',required=True)
    a=ap.parse_args(argv)
    try:
        if a.command=='audit-existing':audit_existing(a.v15_results,a.output)
        else:run(a)
    except (RuntimeError,ValueError,TypeError,KeyError,FileNotFoundError,OSError) as e:
        print('V15_1_GATE_FAILED:',e,file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':raise SystemExit(main())

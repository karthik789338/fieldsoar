#!/usr/bin/env python3
"""FieldVision-NOMS v16 — independent 128-UAV native-mobility robustness.

Stages: prepare new *native UavNetSim trajectory* seed -> frozen MAPPO heldout
-> integrity-gated paired heuristic replay -> seed-clustered results.

Scientific provenance: radio link capacities derived analytically from native
mobility trajectories, NOT UavNetSim PHY traces and NOT measured field radio.
The heuristic is NOT shielded MAPPO, and all MAPPO checkpoints remain frozen.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import sys
import tempfile
from pathlib import Path

import fieldvision_noms_capacity_fair_v15 as v15
import fieldvision_noms_fair_recovery_v15_1 as v151

CLASSES=v15.CLASSES
REFERENCE='frozen_MAPPO_action_replay'
SELECTED='critical_reserve'
OPTIONAL='capacity_guard'
METRICS=('overall_due_sla','critical_due_sla','standard_due_sla','batch_due_sla',
         'minimum_class_due_sla','class_due_jain','weighted_due_sla')
DEFAULT_SEEDS=(53,67,79)
EXISTING_SEED=41
FLEET=128
BASE_PLAN='configs/h3_independent_mobility_seed41_lr3_v8.yaml'
ANCHOR_MODEL_MANIFEST='results/noms_revision_v14/manifests/decision_full_n128_h3_seed41_v142.json'
PLAN_DIR='configs/noms_wireless_v16'
TRACE_DIR='results/noms_revision_v14/traces'
RESULTS_DIR='results/noms_wireless_v16'


def require(test,description):
    if not test:raise RuntimeError(description)


def filehash(path):
    return v15.file_sha(path)


def path_for_seed(seed,root=Path('.')):
    return Path(root)/TRACE_DIR/f'wireless_full_n128_h3_seed{seed}_v16'


def plan_for_seed(seed,root=Path('.')):
    return Path(root)/PLAN_DIR/f'h3_independent_mobility_seed{seed}_128uav_v16.yaml'


def radio_for_seed(seed,root=Path('.')):
    return Path(root)/f'data/traces/uavnetsim_trajectory_analytical_uplink_128uav_3s_seed{seed}.csv'


def native_for_seed(seed,root=Path('.')):
    return Path(root)/f'results/uavnetsim_native_telemetry_128uav_seed{seed}'


def require_seeds(seeds,include41=False):
    require(bool(seeds) and len(set(seeds))==len(seeds),'Specify distinct mobility seeds')
    require(all(s>0 and s!=27 for s in seeds),'Invalid mobility seed')
    require(include41 or all(s!=41 for s in seeds),'Seed 41 is historical: never recapture it')


def _load_legacy():
    import fieldvision_noms_review_v14_1 as old
    return old


def verify_plan(plan,original,seed):
    import yaml
    from re import search
    plan=Path(plan);original=Path(original)
    require(plan.is_file() and original.is_file(),'Missing H3 derived/base YAML')
    text=plan.read_text()
    expected=f'uavnetsim_trajectory_analytical_uplink_128uav_3s_seed{seed}.csv'
    require(expected in text,f'H3 plan does not name seed {seed} radio CSV')
    meta=plan.with_suffix('.manifest.json')
    require(meta.is_file(),'Derived H3 plan manifest missing')
    doc=json.loads(meta.read_text())
    require(doc.get('base_plan_sha256')==filehash(original),'H3 source plan changed')
    require(doc.get('new_plan_sha256')==filehash(plan),'Derived H3 plan has changed')
    require(int(doc.get('fleet_uavs',-1))==FLEET and int(doc.get('mobility_seed',-1))==seed,
            'Wrong fleet/mobility in derived plan')
    return True


def verify_native_inputs(root,seed,prior=None):
    require_seeds([seed])
    legacy=prior or _load_legacy()
    native,meta,trajhash=legacy.check_capture(root,FLEET,seed)
    cap=radio_for_seed(seed,root);man=cap.with_suffix('.manifest.json')
    require(cap.is_file() and man.is_file(),f'Missing analytical radio trace {cap}')
    m=json.loads(man.read_text())
    require(m.get('generated_trace_sha256')==filehash(cap),'Radio trace hash mismatch')
    require(trajhash!=filehash(cap),'Radio file unexpectedly identical to mobility CSV')
    return {'mobility_seed':seed,'fleet_uavs':FLEET,'native_trajectory_sha256':trajhash,
            'native_capture_manifest_sha256':filehash(native/'uavnetsim_capture_manifest.json'),
            'analytical_radio_csv_sha256':filehash(cap),
            'radio_trace_provenance':'trajectory_driven_analytical_uncalibrated',
            'native_trajectory':str((native/'uavnetsim_uav_trajectories.csv').resolve()),
            'analytical_radio_trace':str(cap.resolve())}


def prepare(root,seeds,execute,samples=1024,poll_ms=20):
    root=Path(root).resolve(); require_seeds(seeds)
    legacy=_load_legacy();original=root/BASE_PLAN
    require(original.is_file(),f'Missing historical H3 source plan: {original}')
    prepared=[]
    for seed in seeds:
        native=native_for_seed(seed,root);radio=radio_for_seed(seed,root)
        plan=plan_for_seed(seed,root)
        print(f'V16_PREPARE_SEED seed={seed} native={native} radio={radio} plan={plan}',flush=True)
        if not execute:
            print(f'DRY_RUN: v14.2 capture-native --fleet 128 --seed {seed} --samples {samples} --poll-ms {poll_ms} --execute')
            print(f'DRY_RUN: v14.2 radio-native --fleet 128 --seed {seed} --execute')
            print(f'DRY_RUN: derive-h3-plan from {original} into {plan}')
            continue
        legacy.native_capture(root,FLEET,seed,samples,poll_ms,True)
        legacy.native_radio(root,FLEET,seed,True)
        info=verify_native_inputs(root,seed,legacy)
        if plan.exists() or plan.with_suffix('.manifest.json').exists():verify_plan(plan,original,seed)
        else:legacy.derive_h3_plan(original,FLEET,seed,plan)
        verify_plan(plan,original,seed)
        prepared.append(info)
        print(f'V16_NATIVE_SEED_READY seed={seed} native_sha={info["native_trajectory_sha256"][:12]} radio_sha={info["analytical_radio_csv_sha256"][:12]}',flush=True)
    if execute:
        hashes=[x['native_trajectory_sha256'] for x in prepared]
        require(len(hashes)==len(set(hashes)),'Distinct mobility seeds produced identical native trajectories')
        print('V16_NATIVE_PREPARE_PASS seeds=',','.join(map(str,seeds)),flush=True)


def frozen(root,seed,episodes,execute):
    require_seeds([seed]);root=Path(root).resolve()
    legacy=_load_legacy()
    info=verify_native_inputs(root,seed,legacy)
    plan=plan_for_seed(seed,root);verify_plan(plan,root/BASE_PLAN,seed)
    anchor=root/ANCHOR_MODEL_MANIFEST
    require(anchor.is_file(),f'Missing verified original seed-41 model manifest: {anchor}')
    anchor_hash=json.loads(anchor.read_text()).get('selected_model_sha256')
    selection_dir,_=legacy.selected_model(root,FLEET,'full')
    current_hash=filehash(selection_dir/'selected_model.json')
    require(anchor_hash and current_hash==anchor_hash,'Frozen model selection differs from successful seed-41 reference')
    trace=path_for_seed(seed,root)
    if trace.is_dir():
        docs=v15.manifest_index(trace,episodes,'full',seed)
        require(int(docs[0][2]['fleet_uavs'])==FLEET,'Existing capture has wrong fleet size')
        hashes={filehash(p) for _,p,_ in docs}
        require(len(hashes)==len(docs),'Repeated source manifests in cached capture')
        # The v14.2 wrapper also emits an evaluator crosscheck manifest.
        runid=trace.name
        summary=root/f'results/noms_revision_v14/manifests/{runid}.json'
        require(summary.is_file(),'Trace exists without v14.2 frozen-evaluator provenance; do not silently rerun')
        j=json.loads(summary.read_text())
        require(int(j['mobility_seed'])==seed and j['protocol']=='independent' and
                j['learned_policy_SLA_crosscheck']['episode_count']==episodes,
                'Existing frozen evaluator is incomplete or uses the wrong seed')
        print(f'V16_FROZEN_EXISTING_VERIFIED seed={seed} episodes={len(docs)} trace={trace}',flush=True)
        return
    require(execute,'Frozen trace missing; rerun command with --execute to perform NEW seed evaluation')
    legacy.trace_frozen(root,FLEET,'full','independent',trace.name,episodes,
                        str(plan),None,None,None,True,seed)
    docs=v15.manifest_index(trace,episodes,'full',seed)
    print(f'V16_NEW_FROZEN_SEED_PASS seed={seed} episodes={len(docs)}',flush=True)


def physical_fingerprint(source):
    """Identity of physical tasks, intentionally excluding class mapping and radio."""
    rows=[]
    keys=('job_id','uav_id','dataset','arrival_s','deadline_s','input_mb',
          'local_runtime_s','edge_runtime_s','cloud_runtime_s')
    for j in sorted(source['jobs'],key=lambda z:str(z['job_id'])):
        rows.append([j.get(k) for k in keys])
    content=json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def cached_reference(env,src,folder,scratch,targets,weights):
    sim,truth=v15.validate_replay(env,src,folder,scratch)
    summary=v15.summarize(sim,src,REFERENCE,1.,weights,targets,collections.Counter())
    completion=v151.completed_times_from_events(list(v15.readjsonl(folder/'events.jsonl')))
    summary.update(v151.fixed_due_scores(src['jobs'],completion,src['horizon'],targets,weights))
    summary.update({'comparison_scope':'RECORDED_FROZEN_MAPPO_1X'});return summary,truth


def run_guard(env,src,scratch,targets,weights,min_rate):
    sim=v15.make_sim(env,src,scratch,scale=1.)
    ctl=v15.CapacityController('capacity_guard',targets,weights,min_rate)
    while sim.time_s<sim.horizon-v15.TOL:
        sim.step(actions=ctl.actions(sim,env))
    out=v15.summarize(sim,src,OPTIONAL,1.,weights,targets,ctl.counts)
    completion={j.job_id:j.completed_s for j in sim.jobs.values() if j.completed_s is not None}
    out.update(v151.fixed_due_scores(list(sim.jobs.values()),completion,src['horizon'],targets,weights))
    out.update({'comparison_scope':'NEW_HEURISTIC_REPLAY_1X'})
    return out


def evaluate(root,seed,episodes,limit,output,intent,hold_s,min_remote_rate,with_guard=False):
    import fieldvision_noms_environment as env
    require_seeds([seed]);root=Path(root).resolve()
    info=verify_native_inputs(root,seed)
    plan=plan_for_seed(seed,root);verify_plan(plan,root/BASE_PLAN,seed)
    anchor=root/ANCHOR_MODEL_MANIFEST
    require(anchor.is_file(),f'Missing original seed-41 frozen model manifest: {anchor}')
    anchor_hash=json.loads(anchor.read_text()).get('selected_model_sha256')
    trace=path_for_seed(seed,root)
    runid=trace.name
    evaluator=root/f'results/noms_revision_v14/manifests/{runid}.json'
    require(evaluator.is_file(),'Missing successful frozen heldout evaluation manifest')
    frozen_meta=json.loads(evaluator.read_text())
    require(anchor_hash and frozen_meta.get('selected_model_sha256')==anchor_hash,
            'New frozen evaluation did not use same MAPPO checkpoint selection as seed 41')
    require(frozen_meta.get('H3_evaluation_plan_sha256')==filehash(plan),'Frozen evaluation plan checksum mismatch')
    docs=v15.manifest_index(trace,episodes,'full',seed)
    require(all(int(meta['fleet_uavs'])==FLEET for _,_,meta in docs),'Trace has unexpected fleet size')
    require(limit is None or 1<=limit<=len(docs),'Invalid pilot episode count')
    chosen=docs[:limit] if limit else docs
    out=Path(output)
    require(hold_s>=0 and min_remote_rate>0,'Invalid controller parameters')
    if out.exists():
        require(out.is_dir(),f'Replay path exists but is not a directory: {out}')
        provenance_path=out/'v16_seed_provenance.json'
        require(provenance_path.is_file(),f'Partial existing replay: {out}; preserve and investigate')
        prev=json.loads(provenance_path.read_text())
        require(int(prev['mobility_seed'])==seed and int(prev['episode_count'])==len(chosen),
                'Existing replay seed/episode count differs from requested')
        require(abs(float(prev['controller_hold_s'])-hold_s)<1e-10 and
                abs(float(prev['min_remote_rate_mbps'])-min_remote_rate)<1e-10 and
                prev['intent_sha256']==filehash(intent) and
                prev['selected_frozen_model_sha256']==anchor_hash,
                'Existing replay parameters or source checkpoint have changed')
        require(bool(prev['includes_capacity_guard_new_heuristic_run'])==bool(with_guard),
                'Existing replay selected policies differ')
        needed={REFERENCE,SELECTED}|({OPTIONAL} if with_guard else set())
        saved=v15.readcsv(out/'paired_episode_fixed_due_metrics.csv')
        checks=v15.readcsv(out/'input_integrity.csv')
        require(len(saved)==len(chosen)*len(needed) and len(checks)==len(chosen),
                'Partial or duplicate existing replay metrics')
        for serial,manifest,_ in chosen:
            r=[x for x in checks if int(x['episode_serial'])==serial]
            require(len(r)==1 and r[0]['source_manifest_sha256']==filehash(manifest),
                    f'Source trace changed since existing replay, episode {serial}')
            pp={x['policy'] for x in saved if int(x['episode_serial'])==serial}
            require(pp==needed,f'Existing replay has missing episode policies {serial}')
        print(f'V16_SEED_EXISTING_REPLAY_VERIFIED seed={seed} episodes={len(chosen)} output={out}',flush=True)
        return
    targets,weights=v15.load_intent(Path(intent))
    rows=[];integrity=[]
    with tempfile.TemporaryDirectory(prefix='fieldvision_v16_') as work:
        for serial,manifest,meta in chosen:
            src=v15.build_episode_source(manifest.parent,meta)
            require(src['n']==FLEET,'Incorrect number of UAVs in source')
            require(src['provenance']=='trajectory_driven_analytical_uncalibrated','Radio trace provenance changed')
            scratch=Path(work)/f'{serial}_reference'
            ref,truth=cached_reference(env,src,manifest.parent,scratch,targets,weights)
            ref['mobility_seed']=seed
            rows.append(ref)
            critical=v151.run_new_policy(env,src,Path(work)/f'{serial}_reserve',SELECTED,
                                         targets,weights,min_remote_rate,hold_s)
            critical['mobility_seed']=seed;critical['comparison_scope']='NEW_HEURISTIC_REPLAY_1X'
            rows.append(critical)
            if with_guard:
                guard=run_guard(env,src,Path(work)/f'{serial}_guard',targets,weights,min_remote_rate)
                guard['mobility_seed']=seed;rows.append(guard)
            # Same due-job cohort across every policy within each episode.
            for c in CLASSES:
                expected=ref[f'{c}_due_count']
                require(critical[f'{c}_due_count']==expected,'Changed deadline-due cohort under critical reserve')
                if with_guard:require(guard[f'{c}_due_count']==expected,'Changed deadline-due cohort under capacity guard')
            integrity.append({'episode_serial':serial,'mobility_seed':seed,'source_manifest_sha256':filehash(manifest),
                              'physical_task_fingerprint':physical_fingerprint(src),
                              'verified_original_on_time':sum(truth.values()),
                              'verified_completed_task_ids':'EXACT_MATCH',
                              'verified_fleet_uavs':FLEET})
            print(f'V16_PAIRED_REPLAY_VERIFIED seed={seed} episode={serial} original_ontime={sum(truth.values())}',flush=True)
    summary=v151.due_summary(rows)
    for r in summary:r['mobility_seed']=seed
    out.mkdir(parents=True,exist_ok=False)
    v15.writecsv(out/'paired_episode_fixed_due_metrics.csv',rows)
    v15.writecsv(out/'fixed_due_policy_summary.csv',summary)
    v15.writecsv(out/'input_integrity.csv',integrity)
    v15.dump(out/'v16_seed_provenance.json',{
        'campaign':'FieldVision NOMS concern5 independent wireless generalization',
        'mobility_seed':seed,'fleet_uavs':FLEET,'episode_count':len(chosen),'source_episodes':episodes,
        'native_provenance':info,'frozen_evaluation_manifest_sha256':filehash(evaluator),
        'selected_frozen_model_sha256':frozen_meta['selected_model_sha256'],
        'original_h3_plan_sha256':filehash(root/BASE_PLAN),
        'derived_h3_plan_sha256':filehash(plan),
        'frozen_source_trace':str(trace.resolve()),
        'intent_sha256':filehash(intent),'intent_path':str(Path(intent).resolve()),
        'controller':'critical_reserve','controller_hold_s':hold_s,'min_remote_rate_mbps':min_remote_rate,
        'includes_capacity_guard_new_heuristic_run':with_guard,
        'environment_sha256':filehash(env.__file__),
        'source_validation':'MATCHED_RECORDED_FROZEN_TASK_COMPLETION_IDENTITIES_AND_ON_TIME_OUTCOMES',
        'primary_cohort':'deadline-due jobs, held fixed between frozen and new controllers',
        'radio_provenance_warning':'Native UavNetSim positions; analytical uncalibrated uplink; no field radio measurements',
        'method_note':'Frozen MAPPO actions replayed; no retraining or closed-loop actor/shield. New controller is a heuristic.'})
    print(f'V16_SEED_EVALUATION_PASS seed={seed} episodes={len(chosen)} output={out}',flush=True)


def load_seed_rows(directory,seed,expected_episodes=75):
    path=Path(directory)/'paired_episode_fixed_due_metrics.csv'
    require(path.is_file(),f'Missing {path}')
    allrows=v15.readcsv(path)
    by=collections.defaultdict(dict)
    for row in allrows:
        p=row.get('policy')
        if p not in (REFERENCE,SELECTED):continue
        ep=int(row['episode_serial']);key=(ep,p)
        require(p not in by[ep],f'Duplicate policy in seed {seed} episode {ep}')
        if row.get('mobility_seed') not in (None,'',str(seed)):
            require(False,f'Mobility seed mislabeled {path}')
        by[ep][p]=row
    require(len(by)==expected_episodes,f'Seed {seed}: got {len(by)} episodes, expected {expected_episodes}')
    require(all(set(pair)=={REFERENCE,SELECTED} for pair in by.values()),f'Seed {seed}: unpaired episodes')
    for ep,pair in by.items():
        for c in CLASSES:
            a=int(pair[REFERENCE][f'{c}_due_count']); b=int(pair[SELECTED][f'{c}_due_count'])
            require(a>0 and a==b,f'Deadline-due cohort not fixed seed {seed} ep {ep} class {c}')
    return by


def paired_seed_rates(by,seed):
    output=[]
    for policy in (REFERENCE,SELECTED):
        total=sum(int(rows[policy]['due_on_time']) for rows in by.values())
        den=sum(int(rows[policy]['due_total']) for rows in by.values())
        line={'mobility_seed':seed,'policy':policy,'episodes':len(by),'due_jobs':den,
              'overall_due_sla':total/den}
        cls=[]
        for c in CLASSES:
            num=sum(int(rows[policy][f'{c}_due_on_time']) for rows in by.values())
            d=sum(int(rows[policy][f'{c}_due_count']) for rows in by.values())
            line[f'{c}_due_sla']=num/d
            line[f'{c}_due_count']=d
            cls.append(num/d)
        line['minimum_class_due_sla']=min(cls)
        line['class_due_jain']=v15.jain(cls)
        line['mean_uplink_utilization']=sum(float(rows[policy]['mean_uplink_utilization']) for rows in by.values())/len(by)
        output.append(line)
    return output


def cluster_bootstrap(deltas,replicates=10000,seed=16005):
    """Unit of resampling = independent mobility trajectory seed (not episode)."""
    require(len(deltas)>=3,'At least three independent mobility seeds needed for clustered uncertainty')
    require(all(math.isfinite(float(x)) for x in deltas),'Non-finite seed-level contrast')
    v=list(map(float,deltas));rng=random.Random(seed)
    boots=sorted(sum(rng.choice(v) for _ in v)/len(v) for _ in range(replicates))
    lo=boots[max(0,int(.025*replicates))]
    hi=boots[min(replicates-1,int(.975*replicates)-1)]
    return sum(v)/len(v),lo,hi


def analyze(seed41,seed_root,new_seeds,output,episodes=75,anchor_model_manifest=ANCHOR_MODEL_MANIFEST):
    require_seeds(new_seeds)
    require(len(new_seeds)>=3,'Use >=3 fresh 128-UAV mobility seeds for concern 5')
    out=Path(output);require(not out.exists(),f'Output already exists {out}')
    anchor=Path(anchor_model_manifest)
    require(anchor.is_file(),f'Missing historical seed-41 checkpoint manifest: {anchor}')
    anchor_frozen=json.loads(anchor.read_text())
    anchor_model_hash=anchor_frozen.get('selected_model_sha256')
    require(anchor_model_hash and anchor_frozen.get('protocol')=='independent' and
            int(anchor_frozen.get('fleet_uavs',-1))==FLEET and int(anchor_frozen.get('mobility_seed',-1))==41,
            'Historical seed-41 checkpoint manifest is not the correct independent H3 reference')
    per=[];paired=[];provenance=[];source_fingerprints={}
    all_seeds=[EXISTING_SEED,*new_seeds]
    for seed in all_seeds:
        if seed==EXISTING_SEED:
            # Existing v15.1 policy outcomes were already audited against cached v15.
            directory=Path(seed41)
            artifact=directory/'v15_1_provenance.json'
            require(artifact.is_file(),'Seed 41 v15.1 source provenance missing')
            p=json.loads(artifact.read_text())
            require(p.get('reused_original_reference_without_rerunning_learned_policy') is True,
                    'Historical seed 41 is not from verified frozen results')
            require(int(p.get('mobility_seed',-1))==EXISTING_SEED,'Historical seed 41 mislabeled')
            require(abs(float(p.get('hold_s',-999))-.6)<1e-10,'Historical reserve hold differs from selected 0.6 seconds')
            integrity_path=directory/'input_integrity.csv'
            require(integrity_path.is_file() and len(v15.readcsv(integrity_path))==episodes,
                    'Historical seed 41 episode hash audit incomplete')
            provenance.append({'mobility_seed':seed,'origin':'CACHED_V15_1_SEED41',
                               'native_trajectory_sha256':'LEGACY_SEED41_NATIVE_NOT_INDEPENDENTLY_REHASHED_HERE',
                               'analytical_radio_csv_sha256':'LEGACY_SEED41_H3',
                               'evidence_provenance_sha256':filehash(artifact)})
        else:
            directory=Path(seed_root)/f'seed{seed}'
            artifact=directory/'v16_seed_provenance.json'
            require(artifact.is_file(),f'Missing seed {seed} independent radio provenance')
            p=json.loads(artifact.read_text())
            require(int(p.get('mobility_seed',-1))==seed and int(p['fleet_uavs'])==FLEET,
                    f'Incorrect new seed {seed} provenance')
            require(p.get('selected_frozen_model_sha256')==anchor_model_hash,
                    f'Seed {seed} was evaluated using different frozen checkpoint selection')
            require(p['source_validation']=='MATCHED_RECORDED_FROZEN_TASK_COMPLETION_IDENTITIES_AND_ON_TIME_OUTCOMES',
                    'New seed lacks exact replay verification')
            require(int(p['episode_count'])==episodes,'Only full seed evaluations may be included')
            require(abs(float(p['controller_hold_s'])-.6)<1e-10 and abs(float(p['min_remote_rate_mbps'])-2.)<1e-10,
                    'New seed controller differs from registered baseline settings')
            native=p['native_provenance']
            require(native['radio_trace_provenance']=='trajectory_driven_analytical_uncalibrated',
                    'Unexpected radio source')
            capture_manifest=Path(native['native_trajectory']).parent/'uavnetsim_capture_manifest.json'
            require(capture_manifest.is_file() and filehash(capture_manifest)==
                    native['native_capture_manifest_sha256'],
                    'Native UavNetSim capture manifest changed at aggregation')
            require(Path(native['native_trajectory']).is_file() and
                    filehash(native['native_trajectory'])==native['native_trajectory_sha256'],
                    'Native trajectory hash mismatch at aggregation')
            require(Path(native['analytical_radio_trace']).is_file() and
                    filehash(native['analytical_radio_trace'])==native['analytical_radio_csv_sha256'],
                    'Analytical radio hash mismatch at aggregation')
            checks=v15.readcsv(directory/'input_integrity.csv')
            require(len(checks)==episodes,'Missing exact frozen replay episode verification')
            source_fingerprints[seed]=[r['physical_task_fingerprint'] for r in sorted(checks,key=lambda x:int(x['episode_serial']))]
            provenance.append({'mobility_seed':seed,'origin':'NEW_128UAV_NATIVE_UAVNETSIM',
                'native_trajectory_sha256':native['native_trajectory_sha256'],
                'analytical_radio_csv_sha256':native['analytical_radio_csv_sha256'],
                'evidence_provenance_sha256':filehash(artifact)})
        by=load_seed_rows(directory,seed,episodes)
        per.extend(paired_seed_rates(by,seed))
        for ep,group in sorted(by.items()):
            r={'mobility_seed':seed,'episode_serial':ep}
            for metric in METRICS:
                r[f'{metric}_gain_pp']=100*(float(group[SELECTED][metric])-float(group[REFERENCE][metric]))
            paired.append(r)
    nativehashes=[x['native_trajectory_sha256'] for x in provenance if x['origin']=='NEW_128UAV_NATIVE_UAVNETSIM']
    radiohashes=[x['analytical_radio_csv_sha256'] for x in provenance if x['origin']=='NEW_128UAV_NATIVE_UAVNETSIM']
    require(len(nativehashes)==len(set(nativehashes)),'Independent native trajectories have identical SHA256')
    require(len(radiohashes)==len(set(radiohashes)),'Analytical radio inputs identical across new seeds')
    # Compare physical workload fingerprints where available; do not silently
    # call this cross-seed task-matched if workload identities differ.
    physical_identical=all(x==next(iter(source_fingerprints.values())) for x in source_fingerprints.values())
    result=[];leave=[];byseed={}
    for metric in METRICS:
        deltas=[]
        for seed in all_seeds:
            v=[x[f'{metric}_gain_pp'] for x in paired if x['mobility_seed']==seed]
            require(len(v)==episodes,f'Incomplete per-seed metric values {metric}/{seed}')
            val=sum(v)/len(v);deltas.append(val);byseed.setdefault(seed,{})[metric]=val
        center,lo,hi=cluster_bootstrap(deltas)
        result.append({'metric':metric,'policy':SELECTED,'reference':REFERENCE,'n_independent_mobility_seeds':len(deltas),
            'new_native_seed_count':len(new_seeds),'episodes_per_seed':episodes,
            'mean_seed_equal_episode_gain_pp':center,'cluster_bootstrap_CI95_low_pp':lo,
            'cluster_bootstrap_CI95_high_pp':hi,
            'seeds_positive':sum(d>0 for d in deltas),'seeds_negative':sum(d<0 for d in deltas),
            'seeds_zero':sum(d==0 for d in deltas),'minimum_seed_gain_pp':min(deltas),
            'maximum_seed_gain_pp':max(deltas),'uncertainty_unit':'MOBILITY_SEED_NOT_EPISODE',
            'caution':'Only 4 mobility seeds; cluster bootstrap exploratory and precision limited'})
        for seed in all_seeds:
            remaining=[byseed[s][metric] for s in all_seeds if s!=seed]
            leave.append({'metric':metric,'excluded_mobility_seed':seed,
                'remaining_seed_count':len(remaining),'mean_gain_pp':sum(remaining)/len(remaining)})
    out.mkdir(parents=True,exist_ok=False)
    v15.writecsv(out/'per_seed_policy_summary.csv',per)
    v15.writecsv(out/'paired_episode_seed_differences.csv',paired)
    v15.writecsv(out/'mobility_seed_cluster_CI.csv',result)
    v15.writecsv(out/'leave_one_mobility_seed_out.csv',leave)
    v15.writecsv(out/'mobility_provenance_audit.csv',provenance)
    v15.dump(out/'analysis_provenance.json',{
        'historical_seed':EXISTING_SEED,'new_native_uavnetsim_seeds':new_seeds,
        'selected_frozen_model_sha256':anchor_model_hash,
        'fleet_uavs':FLEET,'n_seeds':len(all_seeds),'episodes_per_seed':episodes,
        'physical_jobs_identical_across_new_seeds':physical_identical,
        'radio_provenance':'native_UavNetSim_mobility_PLUS_analytical_uncalibrated_radio',
        'no_existing_frozen_runs_repeated':True,
        'no_retrained_mappo':True,'original_policy':'frozen_MAPPO_action_replay',
        'selected_heuristic':SELECTED,'hold_s':.6,'min_remote_rate_mbps':2.,
        'seed_cluster_CI_scope':'resamples trajectory seeds, n=4; limited small-cluster reliability',
        'seed41_note':'Legacy seed 41 frozen H3 reference imported from verified v15.1 report; 128-UAV new native traces must be unique.',
        'cross_seed_task_matching':'identical' if physical_identical else 'not_identical; within-seed pairing only'})
    print(f'V16_MULTISEED_WIRELESS_ANALYSIS_PASS mobility_seeds={all_seeds} episodes_per_seed={episodes} output={out}',flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare',help='Capture NEW 128-UAV mobility and analytical radio traces; idempotent verify')
    a.add_argument('--root',default='.');a.add_argument('--seeds',type=int,nargs='+',default=list(DEFAULT_SEEDS))
    a.add_argument('--samples',type=int,default=1024);a.add_argument('--poll-ms',type=int,default=20)
    a.add_argument('--execute',action='store_true')
    a=sub.add_parser('frozen',help='Evaluate frozen MAPPO once per NEW independent mobility seed')
    a.add_argument('--root',default='.');a.add_argument('--seed',type=int,required=True)
    a.add_argument('--episodes',type=int,default=75);a.add_argument('--execute',action='store_true')
    a=sub.add_parser('replay',help='Verified paired fixed-cohort heuristic replay on one completed new seed')
    a.add_argument('--root',default='.');a.add_argument('--seed',type=int,required=True)
    a.add_argument('--episodes',type=int,default=75);a.add_argument('--limit-episodes',type=int)
    a.add_argument('--intent',default='configs/noms_revision_v13/intent_balanced.yaml')
    a.add_argument('--hold-s',type=float,default=.6);a.add_argument('--min-remote-rate',type=float,default=2.)
    a.add_argument('--with-capacity-guard',action='store_true')
    a.add_argument('--output',required=True)
    a=sub.add_parser('analyze',help='Seed-cluster analysis: historical 41 plus fresh native 128-UAV seeds')
    a.add_argument('--seed41',default='results/noms_capacity_fair_v15_1/full_128_h3_seed41_balanced')
    a.add_argument('--seed-root',default=RESULTS_DIR)
    a.add_argument('--anchor-model-manifest',default=ANCHOR_MODEL_MANIFEST)
    a.add_argument('--new-seeds',type=int,nargs='+',default=list(DEFAULT_SEEDS))
    a.add_argument('--episodes',type=int,default=75)
    a.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':prepare(args.root,args.seeds,args.execute,args.samples,args.poll_ms)
        elif args.command=='frozen':frozen(args.root,args.seed,args.episodes,args.execute)
        elif args.command=='replay':evaluate(args.root,args.seed,args.episodes,args.limit_episodes,
             args.output,args.intent,args.hold_s,args.min_remote_rate,args.with_capacity_guard)
        elif args.command=='analyze':analyze(args.seed41,args.seed_root,args.new_seeds,args.output,args.episodes,args.anchor_model_manifest)
    except (RuntimeError,ValueError,KeyError,FileNotFoundError,FileExistsError) as exc:
        print('V16_GATE_FAILED:',exc,file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':raise SystemExit(main())

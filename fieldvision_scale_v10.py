#!/usr/bin/env python3
"""Immutable, native-capture-gated FieldVision-NOMS 128/256 scaling extension.

Does NOT patch frozen v8/v9 runners, invent mobility, or silently resume partial runs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

FLEETS = (128, 256)
SEEDS = (27, 41)
ALGORITHMS = ('full', 'ippo', 'qmix')
METRICS = ('overall_sla', 'critical_sla', 'standard_sla', 'batch_sla', 'fairness')
ROOT = Path('results/noms_scale_v10')
H3_PLAN = Path('configs/h3_independent_mobility_seed41_lr3_v8.yaml')
FROZEN_V8 = Path('run_orchestration_experiments.py')
FROZEN_V9 = Path('run_learning_baselines_v9.py')
ENV_SOURCE = Path('fieldvision_noms_environment.py')
TUNING = {
    'full': Path('configs/mappo_h123_lr3_weighted_v8.yaml'),
    'ippo': Path('configs/learning_baselines_v9.yaml'),
    'qmix': Path('configs/learning_baselines_v9.yaml'),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as inp:
        for block in iter(lambda: inp.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def ensure(cond, msg):
    if not cond:
        raise ValueError(msg)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write('\n')


def capture_dir(n, seed):
    return Path(f'results/uavnetsim_native_telemetry_{n}uav_seed{seed}')


def trajectory(n, seed):
    return capture_dir(n, seed) / 'uavnetsim_uav_trajectories.csv'


def trace_path(n, seed):
    return Path(f'data/traces/uavnetsim_trajectory_analytical_uplink_{n}uav_3s_seed{seed}.csv')


def config_path(n):
    return Path(f'configs/noms_scale_v10/fleet_{n}uav_base.yaml')


def assumption_path(n, seed):
    return Path(f'configs/noms_scale_v10/radio_{n}uav_seed{seed}.yaml')


def run_logged(command, log, **kwargs):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x', encoding='utf-8') as out:
        out.write('COMMAND: ' + json.dumps([str(x) for x in command]) + '\n')
        out.flush()
        started = time.perf_counter()
        result = subprocess.run(list(map(str, command)), stdout=out, stderr=subprocess.STDOUT, **kwargs)
        elapsed = time.perf_counter() - started
    if result.returncode:
        tail = log.read_text(encoding='utf-8', errors='replace')[-5500:]
        raise RuntimeError(f'COMMAND_FAILED rc={result.returncode}; log={log}; elapsed={elapsed:.1f}s\n{tail}')
    print(f'COMMAND_PASS elapsed_s={elapsed:.2f} log={log}')
    return elapsed


def capture_preflight(n, seed):
    """Native generator's own SHA manifest + complete 30 x N geometry, no reconstruction."""
    from prepare_research_inputs import read_uavnetsim_trajectory
    t = trajectory(n, seed)
    m = t.with_name('uavnetsim_capture_manifest.json')
    if not t.is_file() or not m.is_file():
        raise FileNotFoundError(f'Missing successful native trajectory or capture manifest: {t}; {m}')
    cap = read_json(m)
    ensure(cap.get('trajectories_sha256') == sha(t), 'Native trajectory SHA mismatch')
    params = cap.get('command_parameters', {})
    ensure(int(params.get('nodes', -1)) == n, 'Native capture fleet count mismatch')
    if 'seed' in params:
        ensure(int(params['seed']) == seed, 'Native capture seed mismatch')
    if 'duration' in params:
        ensure(abs(float(params['duration'])-3.) < 1e-8, 'Native capture duration mismatch')
    complete = read_uavnetsim_trajectory(t, n, 3., .1)
    ensure(len(complete) == 30*n, 'Incomplete native trajectory')
    result = {'native_trajectory_sha256': sha(t), 'native_manifest_sha256': sha(m),
              'native_trajectory_rows': len(complete), 'native_capture_git':cap.get('uavnetsim_git',{}).get('commit')}
    print(f'NATIVE_PREFLIGHT_PASS n={n} seed={seed} positions={len(complete)} sha256={sha(t)}')
    return result


def attempt_capture(n, seed, samples, poll_ms):
    """Bounded diagnostic capture attempt using EXISTING native capture adapter.

    A dropped event sequence is treated as a real failure. This script does not
    claim to repair the native event-history backend.
    """
    d = capture_dir(n, seed)
    source = Path('capture_uavnetsim_native_telemetry.py')
    native_py = Path('.venv_uavnetsim/bin/python')
    ensure(source.is_file(), 'Missing existing native capture adapter: '+str(source))
    ensure(native_py.is_file(), 'Missing UavNetSim virtualenv: '+str(native_py))
    if d.exists():
        try:
            capture_preflight(n, seed)
            print('NATIVE_CAPTURE_ALREADY_VERIFIED: preserving existing capture')
            return
        except Exception as exc:
            raise RuntimeError(f'Existing capture directory is unverified: {d}; preserve and inspect it. {exc}') from exc
    ensure(samples >= 512 and 1 <= poll_ms <= 1000, 'Capture poll/samples out of bounds')
    logfile = ROOT / 'capture_logs' / f'n{n}_seed{seed}_samples{samples}_poll{poll_ms}.log'
    ensure(not logfile.exists(), f'Capture log already exists: {logfile}; do not overwrite')
    command = [str(native_py.resolve()), str(source.resolve()), '--repo', 'third_party/UavNetSim',
               '--nodes', n, '--duration', 3, '--seed', seed, '--routing', 'Greedy',
               '--mac', 'CSMA_CA', '--mobility', 'GaussMarkov3D',
               '--samples', samples, '--poll-ms', poll_ms, '--output', str(d)]
    try:
        run_logged(command, logfile)
        provenance = capture_preflight(n, seed)
    except Exception:
        print('CAPTURE_NOT_VERIFIED; preserve log and any partial files. No radio trace will be built.', file=sys.stderr)
        raise
    attempt = {'nodes':n, 'seed':seed, 'samples':samples, 'poll_ms':poll_ms,
               'command_sha256':hashlib.sha256(json.dumps(command).encode()).hexdigest(),
               'capture_adapter_sha256':sha(source), 'log_sha256':sha(logfile), **provenance}
    write_json_new(ROOT / 'capture_logs' / f'n{n}_seed{seed}_success.json', attempt)


def radio_preflight(n, seed):
    # Source module enforces source capture checksum, provenance and fleet count.
    from run_orchestration_experiments import _fleet_trace_preflight
    t = trace_path(n, seed)
    info = _fleet_trace_preflight(t, n, 3., .1, expected_shared_bottleneck=24.)
    native = capture_preflight(n, seed)
    ensure(info['mobility_sha256'] == native['native_trajectory_sha256'],
           'Generated radio trace derives from a different native trajectory')
    print(f'RADIO_PREFLIGHT_PASS n={n} seed={seed} trace_sha256={info["trace_sha256"]}')
    return info



def diagnose_capture(n, seed):
    """Classify failures without altering or pretending to repair source telemetry."""
    import re
    logs=sorted((ROOT/'capture_logs').glob(f'n{n}_seed{seed}_samples*_poll*.log'))
    if not logs:
        print(f'NO_V10_CAPTURE_LOG n={n} seed={seed}. Existing earlier capture logs may be in results/.')
        return {'failure':'no_v10_log'}
    p=logs[-1]
    body=p.read_text(encoding='utf-8',errors='replace')
    pattern=re.search(r'expected\s+(\d+)\s*[,;:]?\s*got\s+(\d+)',body,re.I)
    if pattern:
        expected,actual=map(int,pattern.groups())
        print(f'NATIVE_EVENT_SEQUENCE_GAP n={n} seed={seed} expected={expected} got={actual}')
        print('The telemetry event-history window lost records. This capture is invalid;')
        print('do not build radio input from it. A native backend snapshot/stream fix')
        print('may be needed. Increasing history/polling is not a demonstrated fix.')
        return {'failure':'event_gap','expected':expected,'actual':actual,'log':str(p)}
    if 'MemoryError' in body or 'out of memory' in body.lower() or 'oom' in body.lower():
        print('NATIVE_CAPTURE_MEMORY_FAILURE:',p)
        return {'failure':'memory','log':str(p)}
    print('NATIVE_CAPTURE_LOG:',p)
    print('LAST_LINES:\n'+'\n'.join(body.splitlines()[-20:]))
    return {'failure':'unclassified','log':str(p)}

def prepare_radio(n, seed):
    capture_preflight(n, seed)
    src = Path('configs/uav_to_ground_edge_radio_assumptions.yaml')
    ensure(src.is_file(), 'Missing frozen base radio assumptions YAML')
    assumptions = yaml.safe_load(src.read_text(encoding='utf-8'))
    ensure(int(assumptions['n_uavs']) > 0 and float(assumptions['duration_s']) == 3.,
           'Invalid original radio assumptions')
    ensure(float(assumptions['tick_s']) == .1 and float(assumptions['shared_bottleneck_mbps']) == 24.,
           'Radio assumptions differ from frozen 4–64 baseline')
    assumptions['n_uavs'] = n
    dest_cfg = assumption_path(n, seed)
    dest_cfg.parent.mkdir(parents=True,exist_ok=True)
    if dest_cfg.exists():
        ensure(yaml.safe_load(dest_cfg.read_text()) == assumptions,
               'Existing radio assumptions differ: '+str(dest_cfg))
    else:
        with dest_cfg.open('x', encoding='utf-8') as f:
            yaml.safe_dump(assumptions,f,sort_keys=False)
    t = trace_path(n, seed)
    if t.exists():
        radio_preflight(n, seed)
        return
    ensure(not t.with_suffix('.manifest.json').exists() and not t.with_suffix('.diagnostics.csv').exists(),
           'Partial radio output detected; preserve and inspect before retry')
    run_logged([sys.executable, 'prepare_research_inputs.py', 'radio-trace',
                '--trajectory-csv', trajectory(n, seed), '--radio-config', dest_cfg,
                '--output-trace', t], ROOT/'capture_logs'/f'n{n}_seed{seed}_radio.log')
    radio_preflight(n, seed)


def prepare_config(n):
    radio_preflight(n, 27)
    from run_orchestration_experiments import _fleet_scaling_config
    from fieldvision_noms_environment import load_config
    src = Path('configs/noms_fleet_expansion_v4/fleet_64uav_base.yaml')
    ensure(src.is_file(), 'Missing existing validated 64-UAV reference fleet configuration')
    base = load_config(src)
    ensure(int(base['n_uavs']) == 64, 'Reference config is not for 64 UAVs')
    c = _fleet_scaling_config(base, n, trace_path(n, 27), 1., 1.)
    ensure(int(c['profiled_workloads']['job_count']) == n*6,
           'Six jobs per UAV were not preserved')
    ensure(int(c['n_uavs']) == n, 'Fleet size not applied')
    # Must use same slots and assumed aggregate shared bottleneck across fleets.
    ensure(c['network']['bottleneck_mbps'] == base['network']['bottleneck_mbps'],
           'Changed shared bottleneck')
    candidate = yaml.safe_dump(c, sort_keys=False)
    destination = config_path(n)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        ensure(yaml.safe_load(destination.read_text()) == yaml.safe_load(candidate),
               'Existing fleet configuration differs: '+str(destination))
        print('CONFIG_ALREADY_VERIFIED:', destination)
    else:
        with destination.open('x', encoding='utf-8') as f:
            f.write(candidate)
        print('FLEET_CONFIG_CREATED:',destination)
    return destination


def preflight(n, seed41=False):
    from fieldvision_noms_environment import load_config
    from run_orchestration_experiments import _fleet_trace_preflight
    path = config_path(n)
    ensure(path.is_file(), f'Prepare a native-backed fleet config first: {path}')
    cfg = load_config(path)
    ensure(int(cfg['n_uavs']) == n and int(cfg['profiled_workloads']['job_count']) == 6*n,
           'Fleet/task count mismatch')
    target = trace_path(n, 27).resolve()
    ensure(Path(cfg['network']['trace_csv']).resolve() == target, 'Base fleet not tied to seed27 radio')
    first = radio_preflight(n, 27)
    other = radio_preflight(n,41) if seed41 else None
    source_ref = Path('configs/noms_fleet_expansion_v4/fleet_64uav_base.yaml')
    reference = load_config(source_ref)
    for key in ('edge','cloud'):
        if key in reference and key in cfg:
            ensure(reference[key] == cfg[key], f'{key} resources were unexpectedly scaled')
    return {'fleet_uavs':n, 'task_count':6*n, 'base_config_sha256':sha(path),
            'training_trace':first,'h3_trace':other,
            'v8_runner_sha256':sha(FROZEN_V8), 'v9_runner_sha256':sha(FROZEN_V9),
            'environment_sha256':sha(ENV_SOURCE)}


def benchmark(n):
    from fieldvision_noms_environment import NOMSSim, load_config
    evidence = preflight(n)
    cfg = load_config(config_path(n))
    started = time.perf_counter()
    sim = NOMSSim(cfg)
    setup_s = time.perf_counter()-started
    ensure(len(sim.job_definitions) == 6*n, 'Actual workload definitions != 6 per UAV')
    started = time.perf_counter()
    metric = sim.run('critical_first')
    exec_s = time.perf_counter()-started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {**evidence, 'heuristic':'critical_first','sim_construct_s':setup_s,
              'sim_heuristic_s':exec_s,'peak_rss_kib_linux':rss,
              'generated_tasks':metric.get('generated_tasks'),
              'heuristic_overall_sla':metric.get('sla_satisfaction_rate'),
              'benchmark_not_learning_result':True}
    dest=ROOT/f'fleet{n}'/'preflight_benchmark.json'
    if dest.exists():
        ensure(read_json(dest)['base_config_sha256'] == result['base_config_sha256'],
               'Existing benchmark refers to other base config')
        print('BENCHMARK_EXISTS_VERIFIED:',dest)
    else:
        write_json_new(dest,result)
    print(f'SCALING_SMOKE_PASS n={n} jobs={6*n} setup_s={setup_s:.3f} sim_s={exec_s:.3f} rss_kib={rss}')
    return result



def policy_smoke(n, algorithm, device):
    """Real optimizer/validation smoke; not a result in any H1–H3 comparison."""
    preflight(n)
    ensure((ROOT/f'fleet{n}'/'preflight_benchmark.json').is_file(),
           'Run benchmark before attempting model memory smoke')
    out=ROOT/f'fleet{n}'/f'{algorithm}_optimizer_smoke'
    if out.exists():
        manifest=out/'training_manifest.json'
        history=out/'training_history.csv'
        checkpoint=out/'best_validation_checkpoint.pt'
        ensure(manifest.is_file() and history.is_file() and checkpoint.is_file(),
               'Existing partial optimizer smoke must be preserved/inspected')
        print('POLICY_SMOKE_ALREADY_COMPLETE:',out)
        return
    tune=TUNING[algorithm]
    runner=FROZEN_V8 if algorithm=='full' else FROZEN_V9
    episodes=12 if algorithm=='qmix' else 4
    if algorithm=='qmix':
        # The frozen v9 mappo-train CLI always invokes PPO rollout and fails
        # on QMIXPolicy. Its mappo-sweep path correctly calls _train_qmix_one.
        # Use a diagnostic adapter instead of changing source-hashed v9.
        cmd=[sys.executable,Path('qmix_optimizer_smoke_v11.py'),
             '--config',config_path(n),'--tuning-config',tune,
             '--output-root',out,'--device',device,
             '--episodes',episodes,'--seed',11]
    else:
        cmd=[sys.executable,runner,'--mode','mappo-train',
             '--variant',algorithm,'--config',config_path(n),
             '--tuning-config',tune,'--output-root',out,'--device',device,
             '--episodes',episodes,'--ppo-epochs',1,'--dropout',0.0,'--seed',11]
    log=ROOT/'logs'/f'fleet{n}_{algorithm}_optimizer_smoke.log'
    elapsed=run_logged(cmd,log)
    for name in ('training_manifest.json','training_history.csv','best_validation_checkpoint.pt'):
        ensure((out/name).is_file(),f'Optimizer smoke incomplete: {out/name}')
    print(f'POLICY_OPTIMIZER_SMOKE_PASS n={n} algorithm={algorithm} episodes={episodes} elapsed_s={elapsed:.2f}; NOT A SCIENTIFIC RESULT')

def model_paths(n, algorithm):
    root = ROOT/f'fleet{n}'
    return root/f'{algorithm}_training', TUNING[algorithm], (
        FROZEN_V8 if algorithm=='full' else FROZEN_V9)


def verify_selection(n, algorithm):
    directory, tuning, runner = model_paths(n, algorithm)
    f = directory/'selected_model.json'
    ensure(f.is_file(), 'Missing selected_model.json: '+str(f))
    model = read_json(f)
    expected = {'runner_sha256':sha(runner), 'environment_sha256':sha(ENV_SOURCE),
                'base_config_sha256':sha(config_path(n)),
                'tuning_config_sha256':sha(tuning)}
    for key, value in expected.items():
        ensure(model.get(key) == value, f'Checkpoint provenance mismatch: {key} {f}')
    ensure(model.get('policy_variant') == algorithm, 'Checkpoint algorithm differs')
    tuning_yaml=yaml.safe_load(tuning.read_text())
    expected_rates={float(x) for x in tuning_yaml['grid']['learning_rates']}
    actual_rates={float(x['learning_rate']) for x in model.get('grid_tested',[])}
    ensure(expected_rates==actual_rates and len(model['grid_tested'])==len(expected_rates),
           'Not all three LR grid points evaluated')
    chosen=float(model['chosen_config']['learning_rate'])
    ensure(chosen in expected_rates, 'Selected LR is out of the predeclared grid')
    winners=model.get('selected_training_trials',[])
    ensure(sorted(int(x['train_seed']) for x in winners)==[11,23,37], 'Incomplete training seed coverage')
    for r in winners:
        ensure(float(r['learning_rate'])==chosen, 'Selected seed checkpoint has different LR')
        ensure((Path(r['path'])/'best_validation_checkpoint.pt').is_file(), 'Missing trained checkpoint '+str(r['path']))
    ensure((directory/'validation_ranking.csv').is_file(), 'Missing validation ranking')
    return model


def do_train(n, algorithm, device):
    preflight(n)
    benchmark_file = ROOT/f'fleet{n}'/'preflight_benchmark.json'
    ensure(benchmark_file.is_file(), 'Run benchmark first to establish resource scaling baseline')
    ensure((ROOT/f'fleet{n}'/f'{algorithm}_optimizer_smoke'/'best_validation_checkpoint.pt').is_file(),
           f'Run policy-smoke {n} {algorithm} before full training')
    out,tune,runner=model_paths(n,algorithm)
    if out.exists():
        verify_selection(n,algorithm)
        print(f'TRAINING_ALREADY_VERIFIED n={n} algorithm={algorithm}')
        return
    command = [sys.executable, runner, '--mode','mappo-sweep', '--variant',algorithm,
               '--config',config_path(n),'--tuning-config',tune,
               '--output-root',out, '--device',device]
    log=ROOT/'logs'/f'fleet{n}_{algorithm}_train.log'
    elapsed=run_logged(command, log)
    verify_selection(n,algorithm)
    write_json_new(ROOT/f'fleet{n}'/f'{algorithm}_training_runtime.json',
                   {'seconds':elapsed,'log':str(log),'log_sha256':sha(log),'runner_sha256':sha(runner),
                    'selection_sha256':sha(out/'selected_model.json')})
    print('TRAINING_SWEEP_VERIFIED:',out)


def verify_eval(n, algorithm, protocol):
    selected = verify_selection(n,algorithm)
    output=ROOT/f'fleet{n}'/f'{algorithm}_{protocol}_evaluation'
    p=output/'heldout_summary.json'
    d=output/'heldout_mappo_results.csv'
    ensure(p.is_file() and d.is_file(), 'Incomplete eval output directory: '+str(output))
    x=read_json(p)
    ensure(x['selected_model']==selected, 'Evaluation/selection provenance differs')
    ensure(x.get('policy_variant')==algorithm, 'Evaluation algorithm mismatch')
    ensure(int(x.get('training_seed_count',-1))==3, 'Not all selected seeds were evaluated')
    ensure(x['scope']==('H3_INDEPENDENT_MOBILITY_EVALUATION' if protocol=='independent'
                        else 'H1_H2_HELDOUT_UNCALIBRATED_NETWORK_SENSITIVITY'), 'Evaluation scope mismatch')
    if protocol=='independent':
        ensure(x.get('external_evaluation_plan_sha256')==sha(H3_PLAN), 'H3 evaluation plan differs')
    with d.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    expected_scenarios=(yaml.safe_load(H3_PLAN.read_text())['heldout_scenarios'] if protocol=='independent'
                        else yaml.safe_load(TUNING[algorithm].read_text())['heldout_scenarios'])
    seeds=(yaml.safe_load(H3_PLAN.read_text())['heldout']['seeds'] if protocol=='independent'
           else yaml.safe_load(TUNING[algorithm].read_text())['heldout']['seeds'])
    wanted={(s['name'],int(e),int(t)) for s in expected_scenarios for e in seeds for t in (11,23,37)}
    observed={(x['scenario'],int(x['seed']),int(x['training_seed'])) for x in rows}
    ensure(len(rows)==len(wanted) and observed==wanted and x['sample_size']==len(rows),
           'Missing/duplicated paired evaluation rows')
    return rows,x


def do_eval(n, algorithm, protocol, device):
    preflight(n, seed41=(protocol=='independent'))
    verify_selection(n,algorithm)
    dest=ROOT/f'fleet{n}'/f'{algorithm}_{protocol}_evaluation'
    if dest.exists():
        verify_eval(n,algorithm,protocol)
        print(f'EVALUATION_ALREADY_VERIFIED n={n} algorithm={algorithm} protocol={protocol}')
        return
    src,tune,runner=model_paths(n,algorithm)
    cmd=[sys.executable,runner,'--mode','mappo-heldout','--config',config_path(n),
         '--tuning-config',tune,'--sweep-root',src,'--output-root',dest,'--device',device]
    if protocol=='independent':
        cmd += ['--evaluation-plan',H3_PLAN]
    log=ROOT/'logs'/f'fleet{n}_{algorithm}_{protocol}_eval.log'
    run_logged(cmd,log)
    verify_eval(n,algorithm,protocol)
    print('EVALUATION_VERIFIED:',dest)


def report(n_list):
    from run_learning_baselines_v9 import _bootstrap_matched_deltas
    summary=[]; comparisons=[]; pending=[]
    for n in n_list:
        for protocol in ('nominal','independent'):
            raw={}
            for algo in ALGORITHMS:
                dest=ROOT/f'fleet{n}'/f'{algo}_{protocol}_evaluation'
                if not dest.exists():
                    pending.append({'fleet_uavs':n,'algorithm':algo,'protocol':protocol,
                                    'missing_path':str(dest/'heldout_summary.json')})
                    continue
                data, payload=verify_eval(n,algo,protocol)
                raw[algo]=data
                sm=payload['summary']
                summary.append({'fleet_uavs':n,'variant':algo,'protocol':protocol,
                                'selected_learning_rate':payload['selected_model']['chosen_config']['learning_rate'],
                                'validation_selection_score':payload['selected_model']['chosen_config']['mean_validation_score'],
                                'evaluation_episodes':payload['sample_size'],
                                **{m:sm.get(m) for m in METRICS}})
            if 'full' in raw:
                for algo in ('ippo','qmix'):
                    if algo in raw:
                        for metric in METRICS:
                            if any(r.get(metric) in (None,'') for r in (*raw['full'],*raw[algo])):
                                continue
                            stat=_bootstrap_matched_deltas(raw['full'],raw[algo],metric)
                            comparisons.append({'fleet_uavs':n,'protocol':protocol,
                                                'comparison':'full_minus_'+algo,'metric':metric,**stat})
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=ROOT/'reports'/f'scaling_{stamp}'
    output.mkdir(parents=True,exist_ok=False)
    def write_csv(name, rows, fields):
        with (output/name).open('x', newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f, fieldnames=fields,extrasaction='ignore')
            w.writeheader();w.writerows(rows)
    write_csv('policy_summary.csv', summary,
              ['fleet_uavs','variant','protocol','selected_learning_rate','validation_selection_score',
               'evaluation_episodes',*METRICS])
    write_csv('paired_learning_baselines.csv',comparisons,
              ['fleet_uavs','protocol','comparison','metric','mean_gain_pp','ci95_low_pp','ci95_high_pp',
               'bootstrap_resamples','scenario_count','evaluation_seed_count_per_scenario','training_seed_count'])
    write_csv('pending.csv',pending,['fleet_uavs','algorithm','protocol','missing_path'])
    write_json_new(output/'report_manifest.json',
                  {'status':'COMPLETE' if not pending else 'PARTIAL',
                   'fleet_sizes':list(n_list),'completed':len(summary),'pending':len(pending),
                   'comparisons':len(comparisons),
                   'v8_runner_sha256':sha(FROZEN_V8),'v9_runner_sha256':sha(FROZEN_V9),
                   'environment_sha256':sha(ENV_SOURCE),
                   'limitations':['One native mobility capture per fleet and seed, three-second horizon',
                                  'Analytical uncalibrated UAV-ground wireless; not measured throughput',
                                  'QMIX 900-episode budget != PPO optimizer-update budget',
                                  '95% mean critical SLA is a target, not a physical guarantee',
                                  'Fixed shared 24Mbps link and 2 edge GPU slots induce high-load stress']})
    print(f'V10_SCALING_REPORT: {"COMPLETE" if not pending else "PARTIAL"} completed={len(summary)} pending={len(pending)} comparisons={len(comparisons)}')
    print('REPORT:',output)


def status(n_list):
    for n in n_list:
        print(f'===== {n} UAV =====')
        for seed in SEEDS:
            try:
                capture_preflight(n,seed)
                radio_preflight(n,seed)
                print(f'seed{seed}: VERIFIED_NATIVE_AND_RADIO')
            except (ValueError,FileNotFoundError,RuntimeError) as exc:
                print(f'seed{seed}: PENDING ({str(exc)[:135]})')
        for algo in ALGORITHMS:
            try:verify_selection(n,algo);print(f'{algo}: TRAIN_VERIFIED')
            except (ValueError,FileNotFoundError,KeyError) as exc:print(f'{algo}: TRAIN_PENDING ({str(exc)[:90]})')
            for proto in ('nominal','independent'):
                if (ROOT/f'fleet{n}'/f'{algo}_{proto}_evaluation').exists():
                    try:verify_eval(n,algo,proto);print(f'{algo}/{proto}: VERIFIED')
                    except Exception as exc: print(f'{algo}/{proto}: INVALID ({str(exc)[:90]})')
                else:print(f'{algo}/{proto}: PENDING')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['status','capture','capture-check','diagnose','radio','radio-check',
                                     'config','preflight','benchmark','policy-smoke','train','eval','report'])
    p.add_argument('fleet', nargs='?',type=int,choices=FLEETS)
    p.add_argument('algorithm', nargs='?',choices=ALGORITHMS)
    p.add_argument('--seed',type=int,choices=SEEDS,default=27)
    p.add_argument('--samples',type=int,default=250000)
    p.add_argument('--poll-ms',type=int,default=1)
    p.add_argument('--protocol',choices=('nominal','independent'),default='nominal')
    p.add_argument('--device',choices=('cpu','cuda'),default=os.getenv('NOMS_DEVICE','cpu'))
    args=p.parse_args()
    if args.action in ('status','report'):
        chosen=(args.fleet,) if args.fleet else FLEETS
        (status if args.action=='status' else report)(chosen)
        return
    p.error('Fleet size (128 or 256) is required') if args.fleet is None else None
    n=args.fleet
    if args.action=='capture':attempt_capture(n,args.seed,args.samples,args.poll_ms)
    elif args.action=='capture-check':capture_preflight(n,args.seed)
    elif args.action=='diagnose':diagnose_capture(n,args.seed)
    elif args.action=='radio':prepare_radio(n,args.seed)
    elif args.action=='radio-check':radio_preflight(n,args.seed)
    elif args.action=='config':prepare_config(n)
    elif args.action=='preflight':print(json.dumps(preflight(n,seed41=True),indent=2))
    elif args.action=='benchmark':benchmark(n)
    elif args.action=='policy-smoke':
        if args.algorithm is None:p.error('Algorithm required for optimizer smoke')
        policy_smoke(n,args.algorithm,args.device)
    elif args.action in ('train','eval'):
        if args.algorithm is None:p.error('Algorithm (full|ippo|qmix) required for train/eval')
        if args.action=='train':do_train(n,args.algorithm,args.device)
        else:do_eval(n,args.algorithm,args.protocol,args.device)

if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,RuntimeError,KeyError) as e:
        print('V10_GATE_FAILED:',e,file=sys.stderr)
        sys.exit(1)

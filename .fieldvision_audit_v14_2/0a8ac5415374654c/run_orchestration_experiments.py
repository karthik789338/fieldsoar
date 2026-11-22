#!/usr/bin/env python3
"""FieldVision-NOMS exploratory heuristic baselines with identical profiled tasks.

All external profiles and links retain their original provenance. This module
is the future home of formal MAPPO/baseline experiments, NOT an RL trainer yet.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import yaml

from fieldvision_noms_environment import NOMSSim, execute, load_config

POLICIES = ('local_only','edge_only','cloud_only','edf','network_aware_edf','critical_first')


def compare(config_path: Path, output_root: Path, policies=POLICIES):
    config_path = config_path.resolve(strict=True)
    conf = load_config(config_path)
    if conf.get('workload_profile_source') != 'measured_reference_cpu_proxy_edge_gpu_assumed_cloud':
        raise ValueError('Comparison requires a paired CPU/GPU profile scenario')
    if len(set(policies)) != len(policies) or any(p not in POLICIES for p in policies):
        raise ValueError('Policies must be valid and unique')
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError('Refusing to replace an existing baseline comparison')
    output_root.mkdir(parents=True,exist_ok=True)
    summary_rows=[]
    task_sha=None
    for policy in policies:
        path=output_root / policy
        metrics=execute(config_path,path,policy)
        # The workloads must be exactly identical across every policy.
        manifest=json.loads((path/'run_manifest.json').read_text())
        this_sha=manifest['generated_task_workloads_sha256']
        if task_sha is None: task_sha=this_sha
        if this_sha!=task_sha:
            raise RuntimeError('Unfair comparison: task workload differs across policies')
        summary_rows.append({
            'policy':policy,'seed':conf['seed'], 'uavs':metrics['uavs'],
            'tasks':metrics['generated_tasks'],'completed':metrics['completed_tasks'],
            'on_time':metrics['completed_on_time'],
            'sla_satisfaction':metrics['sla_satisfaction_rate'],
            'mean_completed_latency_s':metrics['mean_completed_latency_s'],
            'p95_completed_latency_s':metrics['p95_completed_latency_s'],
            'jain_on_time_fairness':metrics['jain_on_time_completion_fairness'],
            'edge_slot_utilization':metrics['edge_compute_slot_utilization'],
            'local_slot_utilization':metrics['local_compute_slot_utilization'],
            'network_peak_mbps':metrics['peak_network_use_mbps'],
            'critical_class_sla':metrics['class_sla'].get('critical',{}).get('sla_satisfaction_rate'),
            'standard_class_sla':metrics['class_sla'].get('standard',{}).get('sla_satisfaction_rate'),
            'batch_class_sla':metrics['class_sla'].get('batch',{}).get('sla_satisfaction_rate'),
            'network_trace_provenance':metrics['network_trace_provenance'],
            'profile_provenance':metrics['profile_provenance'],
            'workload_sha256':this_sha,
        })
    results=output_root/'baseline_comparison.csv'
    with results.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary_rows[0]))
        writer.writeheader();writer.writerows(summary_rows)
    note={
      'analysis_scope':'SINGLE_SYNTHETIC_EXPLORATORY_SCENARIO_NO_STATISTICAL_CI',
      'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
      'policies':list(policies), 'workloads_identical_sha256':task_sha,
      'baseline_comparison_sha256':hashlib.sha256(results.read_bytes()).hexdigest(),
      'limitations':['Reference vision checkpoints are not trained for agriculture',
                    'Hellbender CPU is not actual UAV hardware',
                    'Cloud runtime and deadlines are assumed',
                    'Radio rate model is uncalibrated and trajectory-driven',
                    'This test does not evaluate or train MAPPO, and is not scientific evidence of algorithm superiority'],
    }
    (output_root/'comparison_manifest.json').write_text(json.dumps(note,indent=2)+'\n')
    print('IDENTICAL_TASK_COMPARISON: PASS')
    print('BASELINE_TABLE:',results)
    for row in summary_rows:
        print(f"{row['policy']:<22} ontime {row['on_time']}/{row['tasks']}  SLA {row['sla_satisfaction']}")
    return summary_rows


def sensitivity_sweep(config_path: Path, output_root: Path,
                      slowdown_factors: list[float], policies=POLICIES):
    """Vary assumed onboard runtime independently of paired measured inputs.

    NO numerical scale represents measured UAV-class hardware. Run all specified
    factors; never select a factor solely because a favored policy wins.
    """
    conf = load_config(config_path.resolve(strict=True))
    if conf.get('workload_profile_source') != 'measured_reference_cpu_proxy_edge_gpu_assumed_cloud':
        raise ValueError('Sensitivity sweep requires paired CPU/GPU reference inputs')
    if not slowdown_factors or len(set(slowdown_factors)) != len(slowdown_factors):
        raise ValueError('Factors must be a nonempty unique list')
    if any(not math.isfinite(f) or f < 1 for f in slowdown_factors):
        raise ValueError('Slowdown factors must be finite and >= 1')
    if output_root.exists():
        raise FileExistsError(f'Refusing to replace an existing sweep directory: {output_root}')
    output_root.mkdir(parents=True)
    scenarios = output_root / 'scenario_configs'
    scenarios.mkdir()
    all_rows=[]
    underlying_task_hash = None
    for factor in slowdown_factors:
        label = str(int(factor)) if factor.is_integer() else str(factor).replace('.', 'p')
        scenario_name = f'onboard_cpu_proxy_slowdown_{label}x'
        one = json.loads(json.dumps(conf))
        one['profiled_workloads']['assumed_onboard_cpu_slowdown_factor'] = factor
        # Record assumptions in independent, replayable YAML file.
        scenario_yaml = scenarios / f'{scenario_name}.yaml'
        scenario_yaml.write_text(yaml.safe_dump(one, sort_keys=False), encoding='utf-8')
        # Check that source sample identities and exogenous scenarios are the
        # same even though *assumed local runtimes* intentionally differ.
        sim = NOMSSim(one)
        task_inputs = [{
            'job_id':job.job_id, 'uav_id':job.uav_id,
            'image':job.profile_image_path, 'arrival_s':job.arrival_s,
            'deadline_s':job.deadline_s, 'input_mb':job.input_mb,
            'edge_runtime_s':job.edge_runtime_s,
        } for job in sim.jobs.values()]
        digest = hashlib.sha256(json.dumps(task_inputs,sort_keys=True).encode()).hexdigest()
        if underlying_task_hash is None:
            underlying_task_hash=digest
        elif digest != underlying_task_hash:
            raise RuntimeError('Source workload identities differ across sensitivity conditions')
        rows=compare(scenario_yaml, output_root/scenario_name, policies)
        for row in rows:
            all_rows.append({'scenario':scenario_name,'assumed_onboard_slowdown':factor,**row})
    full_csv = output_root/'sensitivity_comparison.csv'
    with full_csv.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    manifest = {
      'scope':'PREDECLARED_CPU_PROXY_SENSITIVITY_NOT_UAV_HARDWARE_VALIDATION',
      'source_config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
      'source_workloads_without_scaled_local_time_sha256':underlying_task_hash,
      'onboard_slowdown_factors':slowdown_factors,
      'policies':list(policies),
      'comparison_sha256':hashlib.sha256(full_csv.read_bytes()).hexdigest(),
      'network_provenance':'trajectory_driven_analytical_uncalibrated',
      'cloud_runtime':'assumed_not_measured',
      'interpretation':'Exploratory robustness/sensitivity only; none of these assumed scale factors is a measured UAV processor.'
    }
    (output_root/'sensitivity_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('ONBOARD_COMPUTE_SENSITIVITY_SWEEP: PASS')
    print('COMPARISON:',full_csv)
    return all_rows




# ---- NOMS fleet scalability: per-fleet native mobility, fixed shared resources ----
# This intentionally does not reuse the 4-UAV trace at larger fleet sizes.
# The existing 'radio-trace' subcommand must generate a distinct trace from
# independent, same-sized UavNetSim mobility records for each fleet.

def _fleet_scaling_config(base, n_uavs, trace_path, slowdown, network_scale):
    import copy
    n=int(n_uavs)
    original=int(base['n_uavs'])
    if n < 2 or original < 2:
        raise ValueError('Fleet sizes must be >=2')
    job_cfg=base.get('profiled_workloads')
    if not job_cfg:
        raise ValueError('Fleet experiment requires paired empirical task profiles')
    base_jobs=int(job_cfg['job_count'])
    base_burst=int(job_cfg['jobs_per_burst'])
    if base_jobs % original or base_burst % original:
        raise ValueError('Original task count and burst size must be multiples of base UAV count')
    if slowdown < 1 or not math.isfinite(slowdown) or network_scale <= 0 or not math.isfinite(network_scale):
        raise ValueError('Nonfinite/invalid resource scaling assumption')
    cfg=copy.deepcopy(base)
    cfg['n_uavs']=n
    cfg['profiled_workloads']['job_count']=base_jobs//original*n
    cfg['profiled_workloads']['jobs_per_burst']=base_burst//original*n
    cfg['profiled_workloads']['assumed_onboard_cpu_slowdown_factor']=float(slowdown)
    cfg['network']['capacity_scale']=float(network_scale)
    cfg['network']['trace_csv']=str(Path(trace_path).resolve())
    return cfg


def _fleet_trace_preflight(path, n_uavs, expected_duration, expected_tick,
                           require_native_mobility=True, expected_shared_bottleneck=None):
    path=Path(path).resolve(strict=True)
    sidecar=path.with_suffix('.manifest.json')
    if not sidecar.is_file():
        raise FileNotFoundError(f'Missing radio generator provenance manifest: {sidecar}')
    meta=json.loads(sidecar.read_text(encoding='utf-8'))
    if meta.get('provenance')!='trajectory_driven_analytical_uncalibrated':
        raise ValueError('Fleet evaluation requires explicitly uncalibrated trajectory-driven provenance')
    if sha256_file_for_runner(path) != meta.get('generated_trace_sha256'):
        raise ValueError('Radio trace checksum does not match its manifest')
    assumptions=meta.get('assumptions',{})
    if int(assumptions.get('n_uavs',-1))!=int(n_uavs):
        raise ValueError('Mobility/radio fleet size differs from requested fleet')
    if (abs(float(assumptions.get('duration_s',-1))-float(expected_duration))>1e-7 or
            abs(float(assumptions.get('tick_s',-1))-float(expected_tick))>1e-7):
        raise ValueError('Radio trace must cover precisely the requested horizon and tick')
    from fieldvision_noms_environment import read_network_trace
    trace_rows=read_network_trace(path,int(n_uavs),float(expected_tick),float(expected_duration))
    if expected_shared_bottleneck is not None and any(
            abs(float(row['bottleneck_mbps'])-float(expected_shared_bottleneck))>1e-7
            for row in trace_rows):
        raise ValueError('Shared bottleneck differs across fleet sizes; fix radio assumptions')
    if require_native_mobility:
        trajectory_file=Path(meta.get('source_trajectory_csv',''))
        capture_file=Path(meta.get('capture_manifest_path',''))
        if not trajectory_file.is_file() or not capture_file.is_file():
            raise FileNotFoundError('Cannot verify separate native UavNetSim mobility/capture manifest')
        capture=json.loads(capture_file.read_text(encoding='utf-8'))
        if (sha256_file_for_runner(trajectory_file)!=meta.get('source_trajectory_sha256') or
                sha256_file_for_runner(capture_file)!=meta.get('capture_manifest_sha256') or
                capture.get('trajectories_sha256')!=meta.get('source_trajectory_sha256') or
                int(capture.get('command_parameters',{}).get('nodes',-1))!=int(n_uavs)):
            raise ValueError('Native mobility/capture checksums or fleet size do not match')
    return {'trace_sha256':sha256_file_for_runner(path),'generator_manifest_sha256':sha256_file_for_runner(sidecar),
            'mobility_sha256':meta.get('source_trajectory_sha256'),
            'uavnetsim_commit':meta.get('uavnetsim_commit'),
            'network_provenance':meta.get('provenance')}


def sha256_file_for_runner(path):
    dig=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            dig.update(block)
    return dig.hexdigest()


def fleet_scaling_baselines(base_path, output_root, fleet_sizes, slowdowns, network_scales,
                            trace_template='data/traces/uavnetsim_trajectory_analytical_uplink_{uavs}uav_3s_seed27.csv',
                            policies=POLICIES):
    """Controlled fleet-size stress screening before per-fleet MAPPO training.

    All network profiles MUST originate in distinct matching-size native UAV
    mobility captures; no resampling/faking per-UAV traces or capacity values.
    Within a fleet size, resources (shared 24 Mbps radio and 2 edge GPU slots)
    are held fixed, NOT multiplied by n_uavs. With scaled task count, each UAV
    has the same average arrivals but more aggregate contention.
    """
    base_path=Path(base_path).resolve(strict=True)
    base=load_config(base_path)
    if base.get('workload_profile_source')!='measured_reference_cpu_proxy_edge_gpu_assumed_cloud':
        raise ValueError('Expected the paired Hellbender CPU/V100 workload scenario')
    if (not fleet_sizes or not slowdowns or not network_scales or
            len(set(fleet_sizes))!=len(fleet_sizes) or
            len(set(slowdowns))!=len(slowdowns) or
            len(set(network_scales))!=len(network_scales)):
        raise ValueError('Nonempty unique fleet sizes and scaling factors required')
    if not policies or any(policy not in POLICIES for policy in policies):
        raise ValueError('Invalid policies')
    root=Path(output_root)
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite prior fleet experiment: {root}')
    # Strict preflight for EVERY requested fleet, BEFORE creating output.
    traces={}
    provenance={}
    for n in fleet_sizes:
        trace_path=Path(trace_template.format(uavs=int(n)))
        info=_fleet_trace_preflight(trace_path,n,base['duration_s'],base['tick_s'],
                                     expected_shared_bottleneck=base['network']['bottleneck_mbps'])
        traces[int(n)]=trace_path.resolve()
        provenance[int(n)]=info
    for slowdown in slowdowns:
        for network_scale in network_scales:
            for n in fleet_sizes:
                _fleet_scaling_config(base,n,traces[int(n)],float(slowdown),float(network_scale))
    root.mkdir(parents=True)
    cfg_dir=root/'fleet_configs';cfg_dir.mkdir()
    results=[]
    for n in fleet_sizes:
        n=int(n)
        neutral=_fleet_scaling_config(base,n,traces[n],1.,1.)
        neutral_path=cfg_dir/f'fleet_{n}uav_base.yaml'
        neutral_path.write_text(yaml.safe_dump(neutral,sort_keys=False),encoding='utf-8')
        for slowdown in slowdowns:
            for network_scale in network_scales:
                slug=f'fleet{n}_slow{float(slowdown):g}_net{float(network_scale):g}'.replace('.','p')
                cfg=_fleet_scaling_config(base,n,traces[n],float(slowdown),float(network_scale))
                cfg_path=cfg_dir/f'{slug}.yaml'
                cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
                rows=compare(cfg_path,root/slug,policies)
                for row in rows:
                    results.append({'fleet_uavs':n,'slowdown_assumed':float(slowdown),
                        'network_scale_assumed':float(network_scale),
                        'scenario':slug,'trace_sha256':provenance[n]['trace_sha256'],**row})
    report=root/'fleet_scaling_baselines.csv'
    _save_csv(report,results)
    details={
      'status':'FLEET_BASELINE_SCALING_SCREENING_ONLY',
      'base_config_sha256':sha256_file_for_runner(base_path),
      'network_provenance':'trajectory_driven_analytical_uncalibrated',
      'fleet_sizes':[int(i) for i in fleet_sizes],
      'slowdown_factors':[float(s) for s in slowdowns],
      'network_scales':[float(s) for s in network_scales],
      'fixed_edge_slots':int(base['resources']['edge_slots']),
      'fixed_cloud_slots':int(base['resources']['cloud_slots']),
      'policies':list(policies),'per_fleet_trace_provenance':{str(k):v for k,v in provenance.items()},
      'comparison_sha256':sha256_file_for_runner(report),
      'interpretation':'Only 3-second native UAV trajectories; analytical uncalibrated link capacities; CPU slowdowns simulated. Not paper-grade fleet scalability or MAPPO evaluation.',
      'training_note':'Reusable base fleet configs have factor=1 and may be used to train independent fleet-specific MAPPO with original tuning scenario multipliers. Never multiply slowdown twice.'
    }
    (root/'fleet_scaling_manifest.json').write_text(json.dumps(details,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('FLEET_SCALING_BASELINES: PASS')
    print('COMPARISON:',report)
    return results


# ---- FieldVision-NOMS parameter-shared MAPPO / validation-only hyperparameter sweep ----
# No new trainer script: the same runner handles training and comparisons.
# The actor uses ONLY local features. The critic uses the centralized state.


def _import_rl():
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    from fieldvision_noms_environment import NOMSOrchestrationEnv
    return np, torch, nn, Categorical, NOMSOrchestrationEnv


VALID_POLICY_VARIANTS = ('full', 'decentralized_critic', 'fixed_edf_task', 'fixed_network_tier', 'fixed_compute_tier', 'placement_only', 'no_class_metadata')


def _pooled_critic_factory(nn, torch, n_uavs, nlocal, hidden, dropout):
    """Shared UAV embedding, permutation-invariant critic context, 7 global scalars."""
    class PooledCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(nlocal, hidden), nn.Tanh())
            self.value_head = nn.Sequential(
                nn.Linear(2*hidden+7, hidden), nn.Tanh(), nn.Dropout(dropout),
                nn.Linear(hidden, 1))

        def forward(self, state):
            if state.ndim != 2 or state.shape[-1] != n_uavs*nlocal+7:
                raise ValueError('Central critic state has unexpected shape')
            x = state[:, :n_uavs*nlocal].reshape(-1, n_uavs, nlocal)
            z = self.embed(x)
            pooled = torch.cat((z.mean(dim=1), z.amax(dim=1), state[:, n_uavs*nlocal:]), dim=-1)
            return self.value_head(pooled)

    return PooledCritic()


class MAPPOPolicy:
    """Parameter-shared, masked autoregressive categorical heads.

    The actor's dropout mask is sampled exactly once per rollout transition,
    stored, and REUSED for every PPO likelihood-ratio evaluation. This keeps
    log pi_new == log pi_old before gradient updates, even at dropout > 0.
    Deterministic validation/test uses all-ones masks (dropout disabled).
    """
    def __init__(self, nlocal, nglobal, k, hidden=128, dropout=0., lr=3e-4, device='cpu', variant='full', critic_arch='flat'):
        np, torch, nn, Categorical, _ = _import_rl()
        if not (0 <= dropout < 1) or hidden < 8 or lr <= 0:
            raise ValueError('Invalid hidden size, dropout, or learning rate')
        if variant not in VALID_POLICY_VARIANTS:
            raise ValueError('Unknown policy variant: '+str(variant))
        if critic_arch not in ('flat', 'pooled'):
            raise ValueError('critic_arch must be flat or pooled')
        self.critic_arch = critic_arch
        self.variant = variant
        self.torch, self.nn, self.Categorical = torch, nn, Categorical
        self.np, self.k, self.hidden, self.dropout = np, k, hidden, dropout
        self.device = torch.device(device)
        self.actor = nn.ModuleDict({
            'fc1': nn.Linear(nlocal, hidden),
            'fc2': nn.Linear(hidden, hidden),
            'task': nn.Linear(hidden, k+1),
            'place': nn.Linear(hidden, 3),
            'network': nn.Linear(hidden, 3),
            'compute': nn.Linear(hidden, 3),
        }).to(self.device)
        if critic_arch == 'pooled' and variant != 'decentralized_critic':
            if (nglobal-7) % nlocal:
                raise ValueError('Global dimensions do not match per-UAV observations')
            n_uavs = (nglobal-7)//nlocal
            self.critic = _pooled_critic_factory(nn, torch, n_uavs, nlocal, hidden, dropout).to(self.device)
        else:
            # Local-critic ablation intentionally remains one critic per UAV
            # even when the centralized model uses fleet-invariant pooling.
            self.critic = nn.Sequential(
                nn.Linear(nlocal if variant == 'decentralized_critic' else nglobal, hidden), nn.Tanh(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1),
            ).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def dropout_mask(self, count, sample=False):
        torch = self.torch
        if sample and self.dropout:
            return (torch.rand((count, self.hidden), device=self.device) >= self.dropout).float() / (1.-self.dropout)
        return torch.ones((count, self.hidden), device=self.device)

    def actor_logits(self, local, fixed_dropout):
        t = self.torch
        h = t.tanh(self.actor['fc1'](local)) * fixed_dropout
        h = t.tanh(self.actor['fc2'](h))
        return tuple(self.actor[key](h) for key in ('task','place','network','compute'))

    def _cat(self, logits, mask=None):
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        return self.Categorical(logits=logits)

    def score(self, local, task_mask, placement_mask, actions=None, fixed_dropout=None, deterministic=False):
        """Policy with conditional, non-trainable ablated heads excluded from PPO loss.

        For fixed_edf_task/placement_only, candidate task 1 is the earliest
        feasible deadline among the environment's sorted top-K options.
        When no task is feasible, choose defer. This is not oracle scheduling.
        """
        torch = self.torch
        batch = len(local)
        if fixed_dropout is None:
            fixed_dropout = self.dropout_mask(batch, sample=False)
        if self.variant == 'no_class_metadata':
            # Remove explicit service-class one-hot data from ACTOR only.
            # Deadline ordering and runtime features still exist: this is NOT
            # a complete removal of all implicit SLA information.
            local = local.clone()
            for rank in range(self.k):
                offset = 10 + rank*10 + 6
                local[:, offset:offset+3] = 0.0
        task_logits, place_logits, network_logits, compute_logits = self.actor_logits(local, fixed_dropout)
        task_dist = self._cat(task_logits, task_mask)
        fixed_task = self.variant in ('fixed_edf_task','placement_only')
        fixed_network = self.variant in ('fixed_network_tier','placement_only')
        fixed_compute = self.variant in ('fixed_compute_tier','placement_only')
        if fixed_task:
            has_job = task_mask[:,1:].bool().any(dim=-1)
            first = task_mask[:,1:].float().argmax(dim=-1) + 1
            task = torch.where(has_job,first,torch.zeros_like(first))
        elif actions is None:
            task = task_dist.probs.argmax(-1) if deterministic else task_dist.sample()
        else:
            task = actions[:,0].long()
        rows = torch.arange(batch, device=self.device)
        place_mask = placement_mask[rows,task]
        place_dist = self._cat(place_logits,place_mask)
        if actions is None:
            placement = place_dist.probs.argmax(-1) if deterministic else place_dist.sample()
        else:
            placement = actions[:,1].long()
        remote = (task != 0) & (placement != 0)
        net_dist = self._cat(network_logits)
        comp_dist = self._cat(compute_logits)
        if actions is None:
            net = (net_dist.probs.argmax(-1) if deterministic else net_dist.sample())
            comp = (comp_dist.probs.argmax(-1) if deterministic else comp_dist.sample())
            if fixed_network:
                net = torch.ones_like(net)
            if fixed_compute:
                comp = torch.ones_like(comp)
            net = torch.where(remote,net,torch.ones_like(net))
            comp = torch.where(remote,comp,torch.ones_like(comp))
            selected = torch.stack((task,placement,net,comp),dim=-1)
        else:
            net, comp = actions[:,2].long(), actions[:,3].long()
            selected = actions
            # Fail immediately if recorded actions violate the fixed-head contract.
            if fixed_task and not torch.equal(task,actions[:,0].long()):
                raise ValueError('Recorded task violates deterministic EDF task ablation')
            if fixed_network and torch.any(net!=1):
                raise ValueError('Recorded network tier violates fixed standard ablation')
            if fixed_compute and torch.any(comp!=1):
                raise ValueError('Recorded compute tier violates fixed standard ablation')
        zeros = torch.zeros(batch,device=self.device)
        logp = (zeros if fixed_task else task_dist.log_prob(task)) + place_dist.log_prob(placement)
        entropy = (zeros if fixed_task else task_dist.entropy()) + place_dist.entropy()
        if not fixed_network:
            logp = logp + torch.where(remote,net_dist.log_prob(net),zeros)
            entropy = entropy + torch.where(remote,net_dist.entropy(),zeros)
        if not fixed_compute:
            logp = logp + torch.where(remote,comp_dist.log_prob(comp),zeros)
            entropy = entropy + torch.where(remote,comp_dist.entropy(),zeros)
        return selected, logp, entropy

    def predict(self, obs, training):
        torch = self.torch
        with torch.no_grad():
            local = torch.as_tensor(obs['local_features'],dtype=torch.float32,device=self.device)
            task_mask = torch.as_tensor(obs['task_mask'],device=self.device)
            placement_mask = torch.as_tensor(obs['placement_mask'],device=self.device)
            fixed = self.dropout_mask(len(local),sample=training)
            a,lp,_ = self.score(local,task_mask,placement_mask,fixed_dropout=fixed,
                                deterministic=not training)
            if self.variant == 'decentralized_critic':
                v = self.critic(local).squeeze(-1).cpu().numpy()
            else:
                v = self.critic(torch.as_tensor(obs['global_state'],dtype=torch.float32,
                                                device=self.device).unsqueeze(0)).item()
        return a.cpu().numpy(),lp.cpu().numpy(),fixed.cpu().numpy(),v


def _validate_experiment_setup(tuning):
    if tuning.get('schema_version') != 1:
        raise ValueError('Unknown MAPPO tuning config schema_version')
    for key in ('training_scenarios','validation_scenarios','heldout_scenarios'):
        if not tuning.get(key):
            raise ValueError(f'Missing {key}')
        for scenario in tuning[key]:
            if not all(k in scenario for k in ('name','onboard_slowdown','network_scale')):
                raise ValueError('Invalid scenario: '+str(scenario))
            if float(scenario['onboard_slowdown']) <= 0 or float(scenario['network_scale']) <= 0:
                raise ValueError('Invalid scenario scaling')
            if 'network_trace_template' in scenario and '{uavs}' not in str(scenario['network_trace_template']):
                raise ValueError('H3 independent network trace template must use {uavs}')
            if 'burst_multiplier' in scenario and not float(scenario['burst_multiplier']).is_integer():
                raise ValueError('burst_multiplier must be integer')
            for key in ('edge_slots','cloud_slots'):
                if key in scenario and int(scenario[key])<1:
                    raise ValueError('Compute slots must remain positive')
            for window in scenario.get('outage_windows_s',[]):
                if len(window)!=2 or not (0<=float(window[0])<float(window[1])):
                    raise ValueError('Invalid outage window')
    names = [str(s['name']) for k in ('training_scenarios','validation_scenarios','heldout_scenarios')
             for s in tuning[k]]
    if len(names) != len(set(names)):
        raise ValueError('Training/validation/heldout scenario names must be disjoint')
    vals = [set(tuning[key]['seeds']) for key in ('training','validation','heldout')]
    if any(not x for x in vals) or any(vals[i] & vals[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Training, validation, and heldout seed sets must be disjoint')
    grid=tuning['grid']
    for field in ('episodes','ppo_epochs','dropout'):
        if not grid.get(field):
            raise ValueError('Hyperparameter grid is missing '+field)
    if any(int(e)<=0 for e in grid['episodes']+grid['ppo_epochs']):
        raise ValueError('Episode/epoch budgets must be positive')
    if any(not 0 <= float(d) < 1 for d in grid['dropout']):
        raise ValueError('Dropout must be in [0,1)')
    rates=grid.get('learning_rates', [tuning.get('learning_rate',3e-4)])
    if not rates or any(not math.isfinite(float(lr)) or float(lr)<=0 or float(lr)>1 for lr in rates):
        raise ValueError('Learning rates must be nonempty, finite and in (0,1]')
    if len({float(lr) for lr in rates})!=len(rates):
        raise ValueError('Learning-rate grid must not contain duplicate rates')


def scenario_config(base, scenario):
    """Apply explicit capacity assumptions, never peek at a future trace tick."""
    import copy
    cfg=copy.deepcopy(base)
    if 'profiled_workloads' in cfg:
        cfg['profiled_workloads']['assumed_onboard_cpu_slowdown_factor']=float(scenario['onboard_slowdown'])
    else:
        # Explicitly synthetic test fixtures: no empirical profile multipliers.
        for job in cfg.get('jobs',[]):
            job['local_runtime_s'] *= float(scenario['onboard_slowdown'])
    cfg['network']['capacity_scale']=float(scenario['network_scale'])
    if 'dataset_cycle' in scenario:
        if 'profiled_workloads' not in cfg:
            raise ValueError('dataset_cycle applies only to profiled workloads')
        names=set(cfg['profiled_workloads']['datasets'])
        cycle=list(scenario['dataset_cycle'])
        if not cycle or any(name not in names for name in cycle):
            raise ValueError('dataset_cycle must reference existing datasets')
        cfg['profiled_workloads']['dataset_cycle']=cycle
    if 'burst_multiplier' in scenario:
        if 'profiled_workloads' not in cfg:
            raise ValueError('burst_multiplier applies only to profiled workloads')
        mult=int(scenario['burst_multiplier'])
        if mult<1 or mult!=float(scenario['burst_multiplier']):
            raise ValueError('burst_multiplier must be a positive integer')
        prof=cfg['profiled_workloads']
        new_burst=int(prof['jobs_per_burst'])*mult
        if int(prof['job_count'])%new_burst:
            raise ValueError('Burst multiplier must divide the job count')
        prof['jobs_per_burst']=new_burst
    if 'network_trace_template' in scenario:
        trace=Path(str(scenario['network_trace_template']).format(uavs=int(cfg['n_uavs'])))
        # The trace MUST come from a distinct native capture. No synthetic
        # resampling of the original 3s link envelopes as a new trace.
        info=_fleet_trace_preflight(trace,cfg['n_uavs'],cfg['duration_s'],cfg['tick_s'],
                                    expected_shared_bottleneck=cfg['network']['bottleneck_mbps'])
        original=Path(base['network']['trace_csv']).resolve(strict=True)
        if trace.resolve()==original:
            raise ValueError('H3 unseen trace must be distinct from training trace')
        original_meta=json.loads(original.with_suffix('.manifest.json').read_text())
        new_meta=json.loads(trace.with_suffix('.manifest.json').read_text())
        if (new_meta.get('source_trajectory_sha256') == original_meta.get('source_trajectory_sha256')):
            raise ValueError('H3 unseen trace reuses original mobility capture')
        cfg['network']['trace_csv']=str(trace.resolve())
        cfg['network']['trace_provenance']=info['network_provenance']
    if 'outage_windows_s' in scenario:
        cfg['network']['outage_windows_s']=scenario['outage_windows_s']
    if 'edge_slots' in scenario:
        cfg['resources']['edge_slots']=int(scenario['edge_slots'])
    if 'cloud_slots' in scenario:
        cfg['resources']['cloud_slots']=int(scenario['cloud_slots'])
    return cfg


def _checkpoint_validation_score(rows, summary, tuning):
    """Validation-only checkpoint rule; never consumes a held-out result.

    Feasible: maximize mean overall SLA among all checkpoints whose LOWEST
    validation-scenario critical SLA meets the declared floor.
    Otherwise: maximize the lowest validation-scenario critical SLA.
    Using a 2+score value keeps feasible checkpoints above infeasible ones.
    """
    from collections import defaultdict
    by_scenario=defaultdict(list)
    for r in rows:
        if r['critical_sla'] is None:
            raise ValueError('Missing validation critical SLA')
        by_scenario[r['scenario']].append(float(r['critical_sla']))
    worst=min(statistics.mean(v) for v in by_scenario.values())
    rule=tuning.get('checkpoint_selection', {'mode':'mean_score'})
    if rule.get('mode','mean_score')=='mean_score':
        return summary['selection_score'],worst
    if rule.get('mode')!='scenario_critical_floor':
        raise ValueError('Unsupported checkpoint selection mode')
    threshold=float(rule.get('critical_floor',.95))
    if not (0 < threshold <= 1):
        raise ValueError('Invalid checkpoint critical floor')
    if worst >= threshold:
        return 2.+summary['overall_sla'],worst
    return worst+1.e-5*summary['overall_sla'],worst


def _average_scores(rows):
    import statistics
    def metric(key):
        valid=[float(x[key]) for x in rows if x[key] is not None]
        return float(statistics.mean(valid)) if valid else None
    return {'critical_sla':metric('critical_sla'),'overall_sla':metric('overall_sla'),
            'standard_sla':metric('standard_sla'),'batch_sla':metric('batch_sla'),
            'fairness':metric('fairness'), 'selection_score':metric('selection_score')}


def evaluate_policy(policy, base, scenarios, seeds, top_k=3, audit_actions=False):
    """Deterministic evaluation; all scenarios/seed pairs, not handpicked outcomes."""
    _,_,_,_, NOMSOrchestrationEnv = _import_rl()
    rows=[]
    for scenario in scenarios:
        cfg=scenario_config(base,scenario)
        for seed in seeds:
            env=NOMSOrchestrationEnv(cfg,top_k=top_k)
            obs,_=env.reset(seed=int(seed))
            terminated=False
            placements={'local':0,'edge':0,'cloud':0,'defer':0}
            while not terminated:
                action,_,_,_=policy.predict(obs,training=False)
                if audit_actions:
                    for task, place,_,_ in action:
                        key='defer' if task==0 else ['local','edge','cloud'][int(place)]
                        placements[key]+=1
                obs,_,terminated,truncated,info=env.step(action)
                if truncated:
                    raise RuntimeError('Unexpected truncated episode')
            result=info['episode_metrics']
            by_class_place={}
            timely_by_class_place={}
            if audit_actions:
                for job in env.sim.jobs.values():
                    if job.placement is None or job.arrival_s >= env.sim.horizon:
                        continue
                    key=f'{job.service_class}:{job.placement}'
                    by_class_place[key]=by_class_place.get(key,0)+1
                    if job.completed_s is not None and job.completed_s <= job.deadline_s+1e-9:
                        timely_by_class_place[key]=timely_by_class_place.get(key,0)+1
            critical=result['class_sla'].get('critical',{}).get('sla_satisfaction_rate')
            overall=result['sla_satisfaction_rate']
            standard=result['class_sla'].get('standard',{}).get('sla_satisfaction_rate')
            batch=result['class_sla'].get('batch',{}).get('sla_satisfaction_rate')
            # Both terms are fixed *before* test evaluation; never tune on heldout.
            score=(0.65*critical+0.35*overall) if critical is not None and overall is not None else None
            rows.append({'scenario':scenario['name'],'seed':int(seed),
                         'overall_sla':overall,'critical_sla':critical,
                         'standard_sla':standard,'batch_sla':batch,
                         'fairness':result['jain_on_time_completion_fairness'],
                         'selection_score':score,'completed':result['completed_tasks'],
                         'on_time':result['completed_on_time'],
                         'task_count':result['generated_tasks'],
                         'mean_latency_s':result['mean_completed_latency_s'],
                         'p95_latency_s':result['p95_completed_latency_s'],
                         'placements':placements if audit_actions else None,
                         'scheduled_jobs_by_class_placement':by_class_place if audit_actions else None,
                         'on_time_jobs_by_class_placement':timely_by_class_place if audit_actions else None})
    return rows, _average_scores(rows)


def _save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k:(json.dumps(v,sort_keys=True) if isinstance(v,dict) else v) for k,v in r.items()})


def _create_policy(base, top_k, config, device, init_seed):
    _,torch,_,_, NOMSOrchestrationEnv = _import_rl()
    env=NOMSOrchestrationEnv(base,top_k=top_k)
    obs,_=env.reset(seed=init_seed)
    torch.manual_seed(int(init_seed))
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no GPU available in allocation')
    return MAPPOPolicy(obs['local_features'].shape[-1], len(obs['global_state']),
                       top_k, hidden=int(config.get('hidden_size',128)),
                       dropout=float(config['dropout']),
                       lr=float(config.get('learning_rate',3e-4)),device=device,
                       variant=config.get('variant','full'),
                       critic_arch=config.get('critic_arch','flat'))


def _rollout(policy, cfg, seed, top_k, multipliers, gamma=.99, gae_lambda=.95):
    np,torch,_,_, NOMSOrchestrationEnv = _import_rl()
    env=NOMSOrchestrationEnv(cfg,top_k=top_k,multipliers=multipliers)
    obs,_=env.reset(seed=seed)
    records=[]; done=False
    policy.critic.eval()
    while not done:
        actions,logps,drop_mask,value=policy.predict(obs,training=True)
        next_obs,reward,done,trunc,info=env.step(actions)
        records.append({'local':obs['local_features'].copy(),
                        'global':obs['global_state'].copy(),
                        'task_mask':obs['task_mask'].copy(),
                        'placement_mask':obs['placement_mask'].copy(),
                        'actions':actions.copy(),'old_logp':logps.copy(),
                        'fixed_dropout':drop_mask.copy(),
                        'value':value,'reward':reward})
        if trunc:
            raise RuntimeError('Unexpected truncated episode')
        obs=next_obs
    adv=np.zeros_like(records[0]['value'],dtype=float) if policy.variant == 'decentralized_critic' else 0.
    next_v=np.zeros_like(adv) if policy.variant == 'decentralized_critic' else 0.
    for r in reversed(records):
        delta=r['reward']+gamma*next_v-r['value']
        adv=delta+gamma*gae_lambda*adv
        r['advantage']=adv.copy() if policy.variant == 'decentralized_critic' else float(adv)
        r['return']=(adv+r['value']).copy() if policy.variant == 'decentralized_critic' else float(adv+r['value'])
        next_v=r['value']
    return records, info['episode_metrics']


def _ppo_update(policy, records, ppo_epochs, minibatch_steps=64,
                clip=.2, entropy_coef=.01, value_coef=.5, max_grad_norm=.5):
    np,torch,_,_,_= _import_rl()
    device=policy.device
    local=torch.as_tensor(np.stack([x['local'] for x in records]),dtype=torch.float32,device=device)
    state=torch.as_tensor(np.stack([x['global'] for x in records]),dtype=torch.float32,device=device)
    task_mask=torch.as_tensor(np.stack([x['task_mask'] for x in records]),device=device)
    place_mask=torch.as_tensor(np.stack([x['placement_mask'] for x in records]),device=device)
    actions=torch.as_tensor(np.stack([x['actions'] for x in records]),dtype=torch.long,device=device)
    oldlp=torch.as_tensor(np.stack([x['old_logp'] for x in records]),dtype=torch.float32,device=device)
    fixed=torch.as_tensor(np.stack([x['fixed_dropout'] for x in records]),dtype=torch.float32,device=device)
    advantages=torch.as_tensor(np.stack([x['advantage'] for x in records]),dtype=torch.float32,device=device)
    returns=torch.as_tensor(np.stack([x['return'] for x in records]),dtype=torch.float32,device=device)
    if len(records)>1:
        advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    logs=[]
    for epoch in range(int(ppo_epochs)):
        order=torch.randperm(len(records),device=device)
        for idx in order.split(minibatch_steps):
            S=len(idx); n=local.shape[1]
            local_flat=local[idx].reshape(S*n,-1)
            mask_task=task_mask[idx].reshape(S*n,-1)
            mask_place=place_mask[idx].reshape(S*n,place_mask.shape[-2],3)
            a_flat=actions[idx].reshape(S*n,4)
            old_flat=oldlp[idx].reshape(S*n)
            dropout_flat=fixed[idx].reshape(S*n,-1)
            _,newlogp,entropy=policy.score(local_flat,mask_task,mask_place,
                                            actions=a_flat,fixed_dropout=dropout_flat)
            ratio=torch.exp(newlogp-old_flat)
            adv=(advantages[idx].reshape(-1) if policy.variant == 'decentralized_critic'
                 else advantages[idx].repeat_interleave(n))
            loss_pi=-torch.minimum(ratio*adv,ratio.clamp(1.-clip,1.+clip)*adv).mean()-entropy_coef*entropy.mean()
            policy.actor_opt.zero_grad(set_to_none=True)
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),max_grad_norm)
            policy.actor_opt.step()

            policy.critic.train()
            if policy.variant == 'decentralized_critic':
                predicted=policy.critic(local[idx].reshape(S*n,-1)).reshape(S,n)
            else:
                predicted=policy.critic(state[idx]).squeeze(-1)
            loss_v=value_coef*(predicted-returns[idx]).square().mean()
            policy.critic_opt.zero_grad(set_to_none=True)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),max_grad_norm)
            policy.critic_opt.step()
            policy.critic.eval()
            logs.append((float(loss_pi.item()),float(loss_v.item()),
                         float((old_flat-newlogp.detach()).mean().item())))
    return {'actor_loss':sum(v[0] for v in logs)/len(logs),
            'critic_loss':sum(v[1] for v in logs)/len(logs),
            'approx_kl':sum(v[2] for v in logs)/len(logs)}


def _save_checkpoint(policy, path, params, episode, score, multipliers):
    torch=policy.torch
    torch.save({'actor':policy.actor.state_dict(),'critic':policy.critic.state_dict(),
                'params':params,'episode':episode,'validation_score':score,
                'multipliers':multipliers},path)


def train_mappo_one(base, tuning, output, params, seed=11, device='cpu'):
    """Single independent run. SELECT checkpoint only on validation seeds."""
    _, torch, _, _, _ = _import_rl()
    if output.exists():
        raise FileExistsError(f'Training directory already exists: {output}')
    output.mkdir(parents=True)
    # Avoid oversubscription for tiny neural networks on shared HPC resources.
    torch.set_num_threads(int(tuning.get('torch_threads',2)))
    policy=_create_policy(base,int(tuning.get('top_k',3)),params,device,seed)
    training=tuning['training_scenarios']; validation=tuning['validation_scenarios']
    vali_seeds=tuning['validation']['seeds']
    gamma=float(tuning.get('gamma',.99)); lam=float(tuning.get('gae_lambda',.95))
    constrained=tuning.get('reward_design',{}).get('mode')=='constrained_balanced'
    multipliers=({'critical':float(tuning.get('critical_dual_initial',2.)),
                  'standard':0.,'batch':0.} if constrained else
                 {'critical':1.,'standard':.5,'batch':.25})
    sla_targets=tuning.get('sla_targets',{'critical':.95,'standard':.8,'batch':.6})
    if constrained and (float(sla_targets.get('standard',0.))!=0. or
                        float(sla_targets.get('batch',0.))!=0.):
        raise ValueError('constrained_balanced uses only critical dual; set other targets to zero')
    dual_lr=float(tuning.get('dual_lr',.05))
    dual_max=float(tuning.get('dual_max',8.))
    if not math.isfinite(dual_lr) or dual_lr<=0 or not math.isfinite(dual_max) or dual_max<=0:
        raise ValueError('Invalid dual learning rate or dual maximum')
    batch=[]; history=[]; best_score=float('-inf'); best_episode=0
    update_every=int(tuning.get('update_every_episodes',4))
    eval_interval=int(tuning.get('eval_interval_episodes',50))
    n_episodes=int(params['episodes'])
    episodes_per_seed=int(tuning.get('train_seed_stride',10000))
    for ep in range(1,n_episodes+1):
        # Deterministic predefined scenario cycling for each training seed.
        which=training[(ep-1+seed)%len(training)]
        cfg=scenario_config(base,which)
        episode_seed=episodes_per_seed*seed+ep
        if 'reward_design' in tuning:
            cfg['reward_design']=tuning['reward_design']
        rollout,metrics=_rollout(policy,cfg,episode_seed,policy.k,multipliers,gamma,lam)
        batch.extend(rollout)
        # Projected dual update. Does NOT certify population-level SLA guarantees.
        for cls,target in sla_targets.items():
            row=metrics['class_sla'].get(cls,{})
            if row.get('sla_satisfaction_rate') is not None:
                violation=float(target)-float(row['sla_satisfaction_rate'])
                multipliers[cls]=max(0.,min(dual_max,multipliers.get(cls,0.)+dual_lr*violation))
        update_info={}
        if ep%update_every==0 or ep==n_episodes:
            update_info=_ppo_update(policy,batch,int(params['ppo_epochs']),
                                    minibatch_steps=int(tuning.get('minibatch_steps',64)))
            batch.clear()
        rec={'episode':ep,'scenario':which['name'], 'train_seed':episode_seed,
             'on_time':metrics['completed_on_time'], 'total':metrics['generated_tasks'],
             'train_sla':metrics['sla_satisfaction_rate'],
             'train_critical_sla':metrics['class_sla']['critical']['sla_satisfaction_rate'],
             'train_standard_sla':metrics['class_sla'].get('standard',{}).get('sla_satisfaction_rate'),
             'train_batch_sla':metrics['class_sla'].get('batch',{}).get('sla_satisfaction_rate'),
             'dual_critical':round(multipliers['critical'],5), **update_info}
        history.append(rec)
        if ep%eval_interval==0 or ep==n_episodes:
            validation_rows, scores=evaluate_policy(policy,base,validation,vali_seeds,top_k=policy.k)
            chosen_score, worst_critical = _checkpoint_validation_score(validation_rows,scores,tuning)
            rec['validation_worst_scenario_critical_sla']=worst_critical
            rec['validation_score']=chosen_score
            rec['validation_sla']=scores['overall_sla']
            rec['validation_critical_sla']=scores['critical_sla']
            rec['validation_standard_sla']=scores['standard_sla']
            rec['validation_batch_sla']=scores['batch_sla']
            score=chosen_score
            if score is not None and score > best_score+1e-9:
                best_score=score; best_episode=ep
                _save_checkpoint(policy,output/'best_validation_checkpoint.pt',params,ep,score,multipliers.copy())
            print(f"MAPPO_TRAIN ep={ep}/{n_episodes} epochs={params['ppo_epochs']} "
                  f"dropout={params['dropout']} val_score={score:.4f} "
                  f"best={best_score:.4f} best_ep={best_episode}",flush=True)
    _save_checkpoint(policy,output/'last_checkpoint.pt',params,n_episodes,best_score,multipliers.copy())
    _save_csv(output/'training_history.csv',history)
    manifest={'scope':'EXPLORATORY_MAPPO_HYPERPARAMETER_TUNING_UNCALIBRATED_RADIO',
              'training_seed':seed,'params':params,'validation_best':best_score,
              'reward_design':tuning.get('reward_design',{'mode':'legacy'}),
              'checkpoint_selection':tuning.get('checkpoint_selection',{'mode':'mean_score'}),
              'dual_max':dual_max,'critical_dual_initial':(float(tuning.get('critical_dual_initial',2.)) if constrained else 1.),
              'critical_dual_final':multipliers.get('critical'),
              'validation_best_episode':best_episode,'episodes_executed':n_episodes,
              'training_scenarios':training,'validation_scenarios':validation,
              'heldout_evaluated':False,
              'actor_local_only':True,'centralized_critic':True,
              'actor_dropout_masks_reused_for_ppo_ratio':True,
              'dual_updates_are_heuristic_not_proven_sla_guarantees':True,
              'network_capacity_scale_is_analytical_sensitivity_not_measurement':True,
              'base_config_sha256':hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest(),
              'torch':torch.__version__}
    (output/'training_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return {'trial':output.name,'episodes':n_episodes,'ppo_epochs':params['ppo_epochs'],
            'dropout':params['dropout'],'learning_rate':params['learning_rate'],
            'train_seed':seed,'best_validation_score':best_score,'best_episode':best_episode,
            'path':str(output)}


def _load_selected_policy(base, tuning, checkpoint, device='cpu'):
    _,torch,_,_,_= _import_rl()
    payload=torch.load(checkpoint,map_location=device,weights_only=True)
    policy=_create_policy(base,int(tuning.get('top_k',3)),payload['params'],device,0)
    policy.actor.load_state_dict(payload['actor'])
    policy.critic.load_state_dict(payload['critic'])
    policy.actor.eval();policy.critic.eval()
    return policy


def run_mappo_sweep(base_path, tuning_path, output_root, device='cpu', max_trials=None, variant='full'):
    import itertools
    if variant not in VALID_POLICY_VARIANTS:
        raise ValueError('Unknown ablation variant: '+str(variant))
    tuning=yaml.safe_load(tuning_path.read_text())
    _validate_experiment_setup(tuning)
    base=load_config(base_path.resolve(strict=True))
    if output_root.exists():
        raise FileExistsError('Refusing to replace existing MAPPO sweep directory')
    output_root.mkdir(parents=True)
    (output_root/'tuning_configuration.yaml').write_text(yaml.safe_dump(tuning,sort_keys=False))
    learning_rates=tuning['grid'].get('learning_rates',[tuning.get('learning_rate',3e-4)])
    params_list=[{'episodes':int(e),'ppo_epochs':int(k),'dropout':float(d),
                  'hidden_size':int(tuning.get('hidden_size',128)),
                  'learning_rate':float(lr),
                  'variant':variant,'critic_arch':tuning.get('critic_arch','flat')}
                 for e,k,d,lr in itertools.product(tuning['grid']['episodes'],
                         tuning['grid']['ppo_epochs'],tuning['grid']['dropout'],learning_rates)]
    if max_trials is not None:
        if max_trials < 1:
            raise ValueError('max_trials must be positive')
        params_list=params_list[:max_trials]
    records=[]
    for ix,params in enumerate(params_list):
        label=f"episodes{params['episodes']}_epochs{params['ppo_epochs']}_dropout{int(100*params['dropout']):02d}"
        # Learning-rate trials must never share a checkpoint or log directory.
        if 'learning_rates' in tuning['grid']:
            label+=f"_lr{format(params['learning_rate'],'.8g').replace('.', 'p').replace('-', 'm').replace('+', 'p')}"
        for seed in tuning['training']['seeds']:
            trial=output_root/f'{label}_seed{seed}'
            row=train_mappo_one(base,tuning,trial,params,int(seed),device)
            records.append(row)
            _save_csv(output_root/'validation_sweep.csv',records)
    # Select configuration on VALIDATION ONLY, aggregate over train seeds.
    from collections import defaultdict
    groups=defaultdict(list)
    for row in records:
        groups[(row['episodes'],row['ppo_epochs'],row['dropout'],row['learning_rate'])].append(row)
    summaries=[]
    for key,rows in groups.items():
        summaries.append({'episodes':key[0],'ppo_epochs':key[1],'dropout':key[2],
                          'learning_rate':key[3],
                          'mean_validation_score':statistics.mean(x['best_validation_score'] for x in rows),
                          'trial_count':len(rows)})
    summaries.sort(key=lambda x:(-x['mean_validation_score'],x['episodes'],x['ppo_epochs'],x['dropout'],x['learning_rate']))
    _save_csv(output_root/'validation_ranking.csv',summaries)
    chosen=summaries[0]
    winners=[r for r in records if (r['episodes'],r['ppo_epochs'],r['dropout'],r['learning_rate']) ==
             (chosen['episodes'],chosen['ppo_epochs'],chosen['dropout'],chosen['learning_rate'])]
    winning_trial=max(winners,key=lambda r:(r['best_validation_score'],-r['train_seed']))
    record={'policy_variant':variant,'chosen_config':chosen,'chosen_training_trial':winning_trial,
            'selected_training_trials': sorted(winners, key=lambda x: x['train_seed']),
            'checkpoint_path':str(Path(winning_trial['path'])/'best_validation_checkpoint.pt'),
            'heldout_evaluates_all_selected_training_seeds': True,
            'selection_based_on':'validation_only','heldout_evaluated':False,
            'grid_tested':params_list,
            'base_config_sha256':hashlib.sha256(base_path.read_bytes()).hexdigest(),
            'tuning_config_sha256':hashlib.sha256(tuning_path.read_bytes()).hexdigest(),
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'environment_sha256':hashlib.sha256(Path(__file__).with_name('fieldvision_noms_environment.py').read_bytes()).hexdigest(),
            'limitations':['UAV local timings are assumed slowdowns of measured server CPU',
                           'Link capacities are analytical and uncalibrated',
                           'Cloud latency/energy not measured',
                           'Only 4 UAVs and 3s network trajectory currently available']}
    (output_root/'selected_model.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print('MAPPO_VALIDATION_SWEEP: PASS')
    print('SELECTED_VALIDATION_ONLY:',chosen)
    print('HELDOUT_NOT_EVALUATED: TRUE')
    return record


def evaluate_mappo_heldout(base_path,tuning_path,sweep_dir,output_dir,device='cpu',evaluation_plan_path=None):
    tuning=yaml.safe_load(tuning_path.read_text());_validate_experiment_setup(tuning)
    base=load_config(base_path.resolve(strict=True))
    selection=json.loads((sweep_dir/'selected_model.json').read_text())
    variant=selection.get('policy_variant','full')
    if variant not in VALID_POLICY_VARIANTS:
        raise ValueError('Unrecognized checkpoint variant')
    if ('base_config_sha256' not in selection or
            selection['base_config_sha256']!=hashlib.sha256(base_path.read_bytes()).hexdigest() or
            selection['tuning_config_sha256']!=hashlib.sha256(tuning_path.read_bytes()).hexdigest() or
            selection['runner_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or
            selection['environment_sha256']!=hashlib.sha256(Path(__file__).with_name('fieldvision_noms_environment.py').read_bytes()).hexdigest()):
        raise ValueError('Code or configuration changed since validation-only selection')
    test_scenarios=tuning['heldout_scenarios']
    test_seeds=tuning['heldout']['seeds']
    extra_plan_hash=None
    if evaluation_plan_path is not None:
        evaluation_plan_path=Path(evaluation_plan_path).resolve(strict=True)
        plan=yaml.safe_load(evaluation_plan_path.read_text())
        if plan.get('schema_version')!=1 or plan.get('protocol')!='independent_mobility_h3':
            raise ValueError('External evaluation plan must declare independent_mobility_h3 protocol')
        if not plan.get('heldout_scenarios') or not plan.get('heldout',{}).get('seeds'):
            raise ValueError('H3 plan requires test scenarios and seeds')
        test_scenarios=plan['heldout_scenarios']
        test_seeds=plan['heldout']['seeds']
        train_names={x['name'] for x in tuning['training_scenarios']+tuning['validation_scenarios']+tuning['heldout_scenarios']}
        if len({x['name'] for x in test_scenarios})!=len(test_scenarios) or any(s['name'] in train_names for s in test_scenarios):
            raise ValueError('H3 test scenario names must be unique and reserved')
        known_seeds=set(tuning['training']['seeds']+tuning['validation']['seeds']+tuning['heldout']['seeds'])
        if any(int(seed) in known_seeds for seed in test_seeds) or len(set(test_seeds))!=len(test_seeds):
            raise ValueError('H3 test seeds overlap or duplicate development seeds')
        for scenario in test_scenarios:
            if 'network_trace_template' not in scenario:
                raise ValueError('H3 test scenarios require independently captured network traces')
        probe={**tuning,'heldout_scenarios':test_scenarios,'heldout':{'seeds':test_seeds}}
        _validate_experiment_setup(probe)
        # All trace/mobility checks are performed before output creation.
        for sc in test_scenarios:
            scenario_config(base,sc)
        extra_plan_hash=hashlib.sha256(evaluation_plan_path.read_bytes()).hexdigest()
    if output_dir.exists():
        raise FileExistsError('Heldout output already exists: '+str(output_dir))
    # The hyperparameter choice is validation-only. Evaluate EVERY training
    # seed corresponding to that choice; do not select the luckiest one on
    # heldout data. Keep the originally marked best checkpoint for provenance.
    winner_trials = selection.get('selected_training_trials')
    if not winner_trials:
        # Compatible with validation selections made using the older runner.
        with (sweep_dir/'validation_sweep.csv').open(newline='',encoding='utf-8') as f:
            trial_rows=list(csv.DictReader(f))
        chosen=selection['chosen_config']
        winner_trials=[r for r in trial_rows
                       if int(r['episodes']) == int(chosen['episodes'])
                       and int(r['ppo_epochs']) == int(chosen['ppo_epochs'])
                       and abs(float(r['dropout'])-float(chosen['dropout'])) < 1e-9
                       and abs(float(r.get('learning_rate',tuning.get('learning_rate',3e-4)))
                               -float(chosen.get('learning_rate',tuning.get('learning_rate',3e-4))))<1e-12]
    # A selected LR is common across all training-seed replicates.
    selected_lr=float(selection['chosen_config'].get('learning_rate',tuning.get('learning_rate',3e-4)))
    for r in winner_trials:
        recorded_lr=float(r.get('learning_rate',selected_lr))
        if abs(recorded_lr-selected_lr)>1e-12:
            raise ValueError('Winner trials mix learning rates')
    expected=set(int(x) for x in tuning['training']['seeds'])
    actual=[int(r['train_seed']) for r in winner_trials]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Selected configuration lacks one checkpoint per declared training seed')
    paths=[(int(r['train_seed']),Path(r['path'])/'best_validation_checkpoint.pt')
           for r in sorted(winner_trials,key=lambda x: int(x['train_seed']))]
    for seed,checkpoint in paths:
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Missing best checkpoint for training seed {seed}: {checkpoint}')
    rows=[]
    for train_seed,checkpoint in paths:
        policy=_load_selected_policy(base,tuning,checkpoint,device)
        one,_=evaluate_policy(policy,base,test_scenarios,
                              test_seeds,top_k=policy.k,audit_actions=True)
        for r in one:
            r['training_seed']=train_seed
        rows.extend(one)
    summary=_average_scores(rows)
    output_dir.mkdir(parents=True)
    _save_csv(output_dir/'heldout_mappo_results.csv',rows)
    # Same jobs, exogenous conditions, and random profile-selection seeds for
    # MAPPO and every fixed/heuristic baseline; never select on held-out runs.
    comparison=[]
    displayed_policy='mappo' if variant=='full' else variant
    for row in rows:
        comparison.append({'policy':displayed_policy,'scenario':row['scenario'],'seed':row['seed'],
                           'training_seed':row['training_seed'],
                           'on_time':row['on_time'],'total':row['task_count'],
                           'overall_sla':row['overall_sla'], 'critical_sla':row['critical_sla'],
                           'standard_sla':row['standard_sla'],'batch_sla':row['batch_sla'],
                           'p95_latency_s':row['p95_latency_s'],'fairness':row['fairness']})
    for scenario in test_scenarios:
        for seed in test_seeds:
            cfg=scenario_config(base,scenario);cfg['seed']=int(seed)
            base_jobs=NOMSSim(cfg).job_definitions
            canonical_hash=hashlib.sha256(json.dumps(base_jobs,sort_keys=True).encode()).hexdigest()
            for named in POLICIES:
                sim=NOMSSim(cfg)
                actual_hash=hashlib.sha256(json.dumps(sim.job_definitions,sort_keys=True).encode()).hexdigest()
                if actual_hash!=canonical_hash:
                    raise RuntimeError('Baseline task identity mismatch on heldout evaluation')
                met=sim.run(named)
                comparison.append({'policy':named,'scenario':scenario['name'],
                    'seed':int(seed),'on_time':met['completed_on_time'],
                    'total':met['generated_tasks'],'overall_sla':met['sla_satisfaction_rate'],
                    'critical_sla':met['class_sla']['critical']['sla_satisfaction_rate'],
                    'standard_sla':met['class_sla'].get('standard',{}).get('sla_satisfaction_rate'),
                    'batch_sla':met['class_sla'].get('batch',{}).get('sla_satisfaction_rate'),
                    'p95_latency_s':met['p95_completed_latency_s'],
                    'fairness':met['jain_on_time_completion_fairness']})
    _save_csv(output_dir/'heldout_policy_comparison.csv',comparison)
    # Pair each heuristic to the MEAN of all selected policy training seeds
    # for the exact same (scenario, evaluation seed) exogenous realization.
    # This avoids counting the three correlated MAPPO policy replicates as
    # 30 independent task-arrival/network realizations (there are only 10).
    grouped_by_pair={}
    for item in comparison:
        key=(item['scenario'],int(item['seed']))
        grouped_by_pair.setdefault(key,{}).setdefault(item['policy'],[]).append(item)
    paired=[]
    for (scenario,eval_seed),groups in sorted(grouped_by_pair.items()):
        selected=groups[displayed_policy]
        for baseline in POLICIES:
            ref=groups[baseline]
            if len(selected)!=len(paths) or len(ref)!=1:
                raise RuntimeError('Paired evaluation seed/training-seed coverage mismatch')
            for metric in ('overall_sla','critical_sla','standard_sla','batch_sla'):
                if selected[0][metric] is None or ref[0][metric] is None:
                    continue
                learned_mean=statistics.mean(x[metric] for x in selected)
                paired.append({'scenario':scenario,'evaluation_seed':eval_seed,
                    'learned_policy':displayed_policy,'baseline':baseline,
                    'metric':metric,'learned_mean_across_training_seeds':learned_mean,
                    'baseline_value':ref[0][metric],
                    'paired_difference':learned_mean-ref[0][metric],
                    'training_seed_count':len(paths)})
    _save_csv(output_dir/'paired_heldout_deltas.csv',paired)
    grouped={}
    for named in [displayed_policy,*POLICIES]:
        sub=[r for r in comparison if r['policy']==named]
        grouped[named]={'mean_overall_sla':statistics.mean(r['overall_sla'] for r in sub),
                       'mean_critical_sla':statistics.mean(r['critical_sla'] for r in sub),
                       'mean_standard_sla':statistics.mean(r['standard_sla'] for r in sub if r['standard_sla'] is not None),
                       'mean_batch_sla':statistics.mean(r['batch_sla'] for r in sub if r['batch_sla'] is not None),
                       'evaluation_episodes':len(sub)}
    (output_dir/'heldout_baseline_summary.json').write_text(json.dumps(grouped,indent=2,sort_keys=True)+'\n')
    per_training_seed = []
    for seed in sorted(expected):
        subset=[r for r in rows if r['training_seed']==seed]
        per_training_seed.append({'training_seed':seed,**_average_scores(subset),
                                  'evaluation_episodes':len(subset)})
    (output_dir/'heldout_summary.json').write_text(json.dumps({
        'scope':('H3_INDEPENDENT_MOBILITY_EVALUATION' if extra_plan_hash else 'H1_H2_HELDOUT_UNCALIBRATED_NETWORK_SENSITIVITY'),
        'external_evaluation_plan_sha256':extra_plan_hash,
        'model_selection':'training/validation only; all selected training seeds evaluated',
        'summary':summary,'sample_size':len(rows),'selected_model':selection,
        'training_seed_count':len(expected),'per_training_seed':per_training_seed,
        'policy_variant':variant,'compared_policies':[displayed_policy,*POLICIES],
        'no_physical_sla_guarantee':True},indent=2)+'\n')
    print('MAPPO_HELDOUT_EVALUATION: PASS')
    print(json.dumps(summary,indent=2))
    return rows,summary



# ---- V6: optimistic necessary-condition feasibility audit (no training) ----
def _individual_deadline_lower_bounds(sim, job):
    """Per-job optimistic completion LOWER bounds, not executable schedules.

    Assumes job has a free local CPU or its own idle edge/cloud slot, no
    shared radio competition, no backlog, and oracle access to the MAXIMUM
    future link capacity. Upload tick/queue/RTT overhead is omitted. Thus a
    failed lower bound is certainly impossible; a passing bound is NOT proof
    of schedulability, even for a single job in the real system.
    """
    dispatch = math.ceil((job.arrival_s-1.e-9)/sim.dt)*sim.dt
    local = dispatch + job.local_runtime_s
    if sim.trace_rows is not None:
        after = [r for r in sim.trace_rows if r['start_time_s'] >= dispatch-1.e-9]
        max_common = max((r['bottleneck_mbps'] for r in after),default=0.)
        max_uav = max((r[f'uav_{job.uav_id}_link_cap_mbps'] for r in after),default=0.)
    else:
        cap_schedule = [sim.base_mbps]
        cap_schedule.extend(cap for t,cap in sim.capacity_schedule if t >= dispatch-1.e-9)
        max_common=max(cap_schedule)
        max_uav=max_common
    # min(max_common, max_uav) is an optimistic capacity upper bound, even
    # if the two maxima occur at different times.
    max_rate = min(max_common,max_uav)*sim.network_capacity_scale
    min_upload = job.input_mb*8./max_rate if max_rate>0 else float('inf')
    edge = dispatch + min_upload + job.edge_runtime_s
    cloud = dispatch + min_upload + job.cloud_runtime_s
    return {'local':local,'edge':edge,'cloud':cloud,
            'best':min(local,edge,cloud)}


def run_feasibility_audit(base_path, tuning_path, output_root, split='validation'):
    """Prospective class-level necessary-feasibility audit on TRAIN/VAL only.

    Does not open held-out seeds or old held-out results. Uses the exact
    measured proxy/assumed runtimes and validated analytical link envelopes.
    """
    if split not in ('training','validation'):
        raise ValueError('Feasibility preflight is restricted to training/validation')
    base_path=Path(base_path).resolve(strict=True)
    tuning_path=Path(tuning_path).resolve(strict=True)
    base=load_config(base_path)
    tuning=yaml.safe_load(tuning_path.read_text())
    _validate_experiment_setup(tuning)
    scenarios=tuning['training_scenarios' if split=='training' else 'validation_scenarios']
    seeds=([int(tuning['training']['seeds'][0])*int(tuning.get('train_seed_stride',10000))+i
            for i in (1,2,3,4,5)] if split=='training' else list(tuning['validation']['seeds']))
    # Train seeds here are diagnostic samples only, NEVER training checkpoint seeds.
    if output_root.exists():
        raise FileExistsError('Refusing to replace existing feasibility audit')
    output_root.mkdir(parents=True)
    targets={k:float(v) for k,v in tuning.get('feasibility_reference_targets',
              {'critical':.95,'standard':.8,'batch':.6}).items()}
    if set(targets)!={'critical','standard','batch'} or any(not 0<=v<=1 for v in targets.values()):
        raise ValueError('Invalid feasibility_reference_targets')
    jobs=[]; overview=[]
    for scenario in scenarios:
        cfg=scenario_config(base,scenario)
        for seed in seeds:
            cfg['seed']=int(seed)
            sim=NOMSSim(cfg)
            # Upper bound on ALL transferred data even with no contention:
            # cannot transmit more than the sum of shared-capacity tick budgets.
            radio_budget_mb=sum(float(r['bottleneck_mbps'])*sim.dt/8.
                                for r in sim.trace_rows)*sim.network_capacity_scale if sim.trace_rows else (
                                    sum(next((cap for t,cap in reversed([(0.,sim.base_mbps),*sim.capacity_schedule])
                                              if t<=k*sim.dt+1.e-9),sim.base_mbps)
                                        * sim.network_capacity_scale*sim.dt/8.
                                        for k in range(round(sim.horizon/sim.dt))))
            for job in sim.jobs.values():
                if job.arrival_s >= sim.horizon:
                    continue
                bound=_individual_deadline_lower_bounds(sim,job)
                # At horizon a job cannot be recorded completed after termination.
                cutoff=min(job.deadline_s,sim.horizon)
                pass_place={k:bool(t<=cutoff+1.e-9) for k,t in bound.items()
                            if k!='best'}
                record={'scenario':scenario['name'],'seed':int(seed),
                        'job_id':job.job_id,'uav_id':job.uav_id,
                        'class':job.service_class,'deadline_s':job.deadline_s,
                        'arrival_s':job.arrival_s,'deadline_cutoff_s':cutoff,
                        'optimistic_best_completion_s':bound['best'],
                        'optimistic_slack_s':cutoff-bound['best'],
                        'local_necessary_feasible':pass_place['local'],
                        'edge_necessary_feasible':pass_place['edge'],
                        'cloud_necessary_feasible':pass_place['cloud'],
                        'any_necessary_feasible':any(pass_place.values()),
                        'radio_oracle_and_no_contention_assumed':True}
                jobs.append(record)
            for cls in ('critical','standard','batch'):
                subset=[r for r in jobs if r['scenario']==scenario['name']
                        and r['seed']==int(seed) and r['class']==cls]
                count=len(subset)
                if not count:
                    continue
                possible=sum(r['any_necessary_feasible'] for r in subset)
                target=targets[cls]
                required=math.ceil(target*count-1.e-9)
                local_eligible=sum(r['local_necessary_feasible'] for r in subset)
                # Lower bound on total bytes necessary to achieve the class
                # target: optimistically schedule every locally possible task
                # on independent idle CPUs, then offload the cheapest remaining
                # remote-capable tasks with no competing flows or queuing.
                remote_needed=max(0,required-local_eligible)
                forced_remote_bytes=sorted(
                    float(sim.jobs[r['job_id']].input_mb) for r in subset
                    if not r['local_necessary_feasible'] and
                       (r['edge_necessary_feasible'] or r['cloud_necessary_feasible']))
                required_mb=(sum(forced_remote_bytes[:remote_needed])
                             if remote_needed<=len(forced_remote_bytes) else float('inf'))
                class_necessary_impossible=(possible<required or required_mb>radio_budget_mb+1.e-9)
                overview.append({'scenario':scenario['name'],'seed':int(seed),
                                 'class':cls,'tasks':count,
                                 'reference_sla_target':target,
                                 'necessarily_impossible':count-possible,
                                 'optimistic_possible_ceiling':possible/count,
                                 'local_possible':local_eligible,
                                 'edge_possible':sum(r['edge_necessary_feasible'] for r in subset),
                                 'cloud_possible':sum(r['cloud_necessary_feasible'] for r in subset),
                                 'optimistic_min_radio_mb_for_target':required_mb,
                                 'optimistic_total_radio_budget_mb':radio_budget_mb,
                                 'target_certainly_infeasible_by_necessary_bounds':class_necessary_impossible})
    _save_csv(output_root/'individual_necessary_bounds.csv',jobs)
    _save_csv(output_root/'class_feasibility_ceiling.csv',overview)
    manifest={'status':'NECESSARY_BOUNDS_AUDIT_ONLY',
              'split':split,'base_config_sha256':sha256_file_for_runner(base_path),
              'tuning_config_sha256':sha256_file_for_runner(tuning_path),
              'runner_sha256':sha256_file_for_runner(Path(__file__)),
              'scenario_names':[x['name'] for x in scenarios],
              'seeds':seeds,'reference_sla_targets':targets,
              'trace_provenance':base['network'].get('trace_provenance'),
              'method':'optimistic per-job completion bounds and aggregate shared-radio byte budget; oracle max link, zero contention, zero queue, isolated compute slots',
              'interpretation':'Necessary conditions only: passing does NOT show joint feasibility. Failure certifies impossible under assumptions. Does NOT calibrate physical hardware.',
              'heldout_used':False}
    (output_root/'audit_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('NECESSARY_FEASIBILITY_AUDIT: PASS', 'fleet=',base['n_uavs'],'split=',split)
    for cls in ('critical','standard','batch'):
        vals=[r['optimistic_possible_ceiling'] for r in overview if r['class']==cls]
        if vals:
            print(f'  {cls:10s} optimistic_ceiling_mean={statistics.mean(vals):.4f} across {len(vals)} diagnostic cases')
    print('CLASS_CEILINGS:',output_root/'class_feasibility_ceiling.csv')
    return overview


# ---- H1/H2/H3: matched, audit-friendly hypothesis report ----
RESEARCH_VARIANTS=('full','full_constrained','placement_only','fixed_edf_task',
                   'fixed_network_tier','fixed_compute_tier','decentralized_critic','no_class_metadata')

def _bootstrap_matched_deltas(learned, reference, metric, n_boot=5000, seed=2027):
    """Paired hierarchical bootstrap over scenario, evaluation and training seeds.

    Report an exploratory conditional CI; mobility captures are NOT bootstrap
    replicates. Every learning policy must use identical scenario and task seeds.
    """
    import numpy as np
    from collections import defaultdict
    lmap={(str(r['scenario']),int(r['seed']),int(r['training_seed'])):float(r[metric])
          for r in learned if r.get(metric) not in ('',None)}
    rmap={(str(r['scenario']),int(r['seed']),int(r['training_seed'])):float(r[metric])
          for r in reference if r.get(metric) not in ('',None)}
    if not lmap or set(lmap)!=set(rmap):
        raise ValueError('Unpaired or missing learned-policy evaluation rows for '+metric)
    vals=defaultdict(dict)
    for key,v in lmap.items():
        scenario,eval_seed,training_seed=key
        vals[scenario][(eval_seed,training_seed)]=v-rmap[key]
    expected_scenarios=len(vals)
    if not expected_scenarios:
        raise ValueError('No paired scenarios')
    eval_groups=[];train_seeds=None
    for scenario, sub in sorted(vals.items()):
        e=sorted({key[0] for key in sub})
        t=sorted({key[1] for key in sub})
        if train_seeds is None:train_seeds=t
        if train_seeds!=t or len(sub)!=len(e)*len(t):
            raise ValueError('Incomplete evaluation/training seed grid')
        eval_groups.append(np.array([[sub[(ei,ti)] for ti in t] for ei in e],dtype=float))
    if len({x.shape for x in eval_groups})!=1:
        raise ValueError('Different evaluation seed counts among scenarios')
    observed=float(np.mean([x.mean() for x in eval_groups]))
    rng=np.random.default_rng(seed)
    # Vectorized hierarchical bootstrap; resample training policies jointly
    # across scenarios to retain correlation of each trained checkpoint.
    tc=rng.integers(0,len(train_seeds),size=(n_boot,len(train_seeds)))
    per_scenario=[]
    for a in eval_groups:
        er=rng.integers(0,a.shape[0],size=(n_boot,a.shape[0]))
        per_scenario.append(a[er[:,:,None],tc[:,None,:]].mean(axis=(1,2)))
    sims=100*np.mean(np.stack(per_scenario,axis=0),axis=0)
    lo,hi=np.percentile(sims,[2.5,97.5])
    return {'mean_gain_pp':100*observed,'ci95_low_pp':float(lo),
            'ci95_high_pp':float(hi),'bootstrap_resamples':n_boot,
            'scenario_count':expected_scenarios,
            'evaluation_seed_count_per_scenario':eval_groups[0].shape[0],
            'training_seed_count':len(train_seeds)}


def generate_research_report(research_root, output_root, fleet_sizes, protocol='nominal'):
    """Create honest H1-H3 reports from ONLY complete matched evaluated runs.

    Missing cells appear in pending.csv; no zero-filling or imputation.
    For H3 'independent', a disjoint independently captured network source
    must have been enforced by scenario_config at evaluation time.
    """
    research_root=Path(research_root);output_root=Path(output_root)
    if output_root.exists():
        raise FileExistsError('Refusing to overwrite previous research report')
    if protocol not in ('nominal','independent'):
        raise ValueError('Unknown research protocol')
    rows=[]; pending=[]; raw={}; baselines={}
    for n in fleet_sizes:
        for variant in RESEARCH_VARIANTS:
            directory=research_root/f'fleet{int(n)}'/f'{variant}_{protocol}_evaluation'
            summary=directory/'heldout_summary.json'
            comparison=directory/'heldout_policy_comparison.csv'
            if not summary.is_file() or not comparison.is_file():
                pending.append({'fleet_uavs':int(n),'variant':variant,'protocol':protocol,
                                'missing_path':str(summary if not summary.is_file() else comparison)})
                continue
            payload=json.loads(summary.read_text())
            expected_scope=('H3_INDEPENDENT_MOBILITY_EVALUATION' if protocol=='independent'
                            else 'H1_H2_HELDOUT_UNCALIBRATED_NETWORK_SENSITIVITY')
            if payload.get('scope')!=expected_scope:
                raise ValueError(f'Evaluation provenance is not correct for {directory}')
            if protocol=='independent' and not payload.get('external_evaluation_plan_sha256'):
                raise ValueError('H3 report requires a reserved independent-mobility test plan')
            selected=payload['selected_model']
            expect_variant='full' if variant=='full_constrained' else variant
            if selected.get('policy_variant','full')!=expect_variant:
                raise ValueError(f'Variant mismatch for {directory}')
            summary_metrics=payload['summary']
            rows.append({'fleet_uavs':int(n),'variant':variant,'protocol':protocol,
                         'evaluation_episodes':payload['sample_size'],
                         'training_seeds':payload['training_seed_count'],
                         'selected_learning_rate':selected.get('chosen_config',{}).get('learning_rate'),
                         'selection_validation_score':selected.get('chosen_config',{}).get('mean_validation_score'),
                         **{key:summary_metrics.get(key) for key in
                            ('overall_sla','critical_sla','standard_sla','batch_sla','fairness')}})
            with comparison.open(newline='',encoding='utf-8') as file:
                records=list(csv.DictReader(file))
            selected_rows=[r for r in records if r['policy']==('mappo' if expect_variant=='full' else expect_variant)]
            if len(selected_rows)!=payload['sample_size']:
                raise ValueError('Missing or duplicated learned rows in '+str(directory))
            raw[(int(n),variant)]=selected_rows
            if variant=='full':
                baselines[int(n)]=[r for r in records if r['policy'] in POLICIES]
    comparisons=[]
    for n in fleet_sizes:
        n=int(n)
        if (n,'full') not in raw:
            continue
        base=raw[(n,'full')]
        for variant in RESEARCH_VARIANTS:
            if variant=='full' or (n,variant) not in raw:
                continue
            alternative=raw[(n,variant)]
            for metric in ('overall_sla','critical_sla','standard_sla','batch_sla','fairness'):
                if any(r.get(metric) in ('',None) for r in [*base,*alternative]):
                    if metric!='fairness':
                        raise ValueError('Required SLA metric missing for '+variant)
                    comparisons.append({'fleet_uavs':n,'comparison':'full_minus_'+variant,
                         'metric':metric,'protocol':protocol,'estimable':False,
                         'missingness_note':'Fairness undefined when no UAV completed any job on time'})
                    continue
                result=_bootstrap_matched_deltas(base,alternative,metric)
                comparisons.append({'fleet_uavs':n,'comparison':'full_minus_'+variant,
                                    'metric':metric,'protocol':protocol,'estimable':True,**result})
        if n in baselines:
            ref=baselines[n]
            for policy in POLICIES:
                ref_rows=[r for r in ref if r['policy']==policy]
                # Three selected MAPPO training-seed replicas share the same
                # exogenous tasks as ONE heuristic run; repeat each heuristic
                # value by training-seed strictly for paired calculation.
                bg={(r['scenario'],int(r['seed'])):r for r in ref_rows}
                expanded=[]
                for learned in base:
                    key=(learned['scenario'],int(learned['seed']))
                    if key not in bg:
                        raise ValueError('Heuristic baseline does not cover learned test scenarios')
                    expanded.append({**bg[key],'training_seed':learned['training_seed']})
                for metric in ('overall_sla','critical_sla','standard_sla','batch_sla','fairness'):
                    if any(r.get(metric) in ('',None) for r in [*base,*expanded]):
                        if metric!='fairness':
                            raise ValueError('Required heuristic SLA metric missing for '+policy)
                        comparisons.append({'fleet_uavs':n,'comparison':'full_minus_heuristic_'+policy,
                             'metric':metric,'protocol':protocol,'estimable':False,
                             'missingness_note':'Fairness undefined when no UAV completed any job on time'})
                        continue
                    r=_bootstrap_matched_deltas(base,expanded,metric)
                    comparisons.append({'fleet_uavs':n,'comparison':'full_minus_heuristic_'+policy,
                                        'metric':metric,'protocol':protocol,'estimable':True,**r})
    output_root.mkdir(parents=True)
    _save_csv(output_root/'policy_summary.csv',rows)
    _save_csv(output_root/'paired_bootstrap.csv',comparisons)
    if pending:_save_csv(output_root/'pending.csv',pending)
    else:(output_root/'pending.csv').write_text('fleet_uavs,variant,protocol,missing_path\n')
    manifest={'status':'COMPLETE' if not pending else 'PARTIAL',
              'protocol':protocol,'fleets':[int(n) for n in fleet_sizes],
              'complete_cells':len(rows),'pending_cells':len(pending),
              'method':'matched hierarchical bootstrap over evaluation/task and training seeds within each scenario',
              'limitations':['single 3-second radio capture per fleet per protocol',
                            'analytic uncalibrated uplink',
                            'CPU proxy is not UAV hardware',
                            'H3 independent protocol uses a different native mobility capture, NOT calibrated real-world wireless',
                            '95-percent mean critical SLA does not constitute a guarantee',
                            'no energy measurement',
                            'bootstrap CIs are conditional on scenarios and mobility captures'],
              'report_sha256':sha256_file_for_runner(output_root/'policy_summary.csv') if rows else None,
              'paired_sha256':sha256_file_for_runner(output_root/'paired_bootstrap.csv') if comparisons else None}
    (output_root/'report_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('H123_RESEARCH_REPORT:',manifest['status'],'complete=',len(rows),'pending=',len(pending))
    print('POLICIES:',output_root/'policy_summary.csv')
    print('PAIRED:',output_root/'paired_bootstrap.csv')
    return rows,comparisons

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/profiled_128paired_cpu_gpu_4uav_network_contract_seed27.yaml'))
    p.add_argument('--output-root',type=Path,default=Path('results/profiled_128paired_baselines_4uav_seed27'))
    p.add_argument('--policies',nargs='+',default=list(POLICIES),choices=POLICIES)
    p.add_argument('--onboard-slowdowns',nargs='+',type=float,default=None)
    p.add_argument('--mode',choices=['baselines','fleet-baselines','mappo-train','mappo-sweep','mappo-heldout','feasibility-audit','research-report'],default='baselines')
    p.add_argument('--variant',choices=VALID_POLICY_VARIANTS,default='full',
                   help='MAPPO/ablation variant for new mappo-sweep runs; never retunes heldout')
    p.add_argument('--fleet-sizes',nargs='+',type=int,default=[4,8,16,32])
    p.add_argument('--network-scales',nargs='+',type=float,default=[1.,2.])
    p.add_argument('--fleet-trace-template',default='data/traces/uavnetsim_trajectory_analytical_uplink_{uavs}uav_3s_seed27.csv')
    p.add_argument('--tuning-config',type=Path,default=Path('configs/mappo_hyperparameter_tuning.yaml'))
    p.add_argument('--audit-split',choices=['training','validation'],default='validation')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--ppo-epochs',type=int,default=4)
    p.add_argument('--dropout',type=float,default=0.)
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--max-trials',type=int,default=None)
    p.add_argument('--sweep-root',type=Path,default=None)
    p.add_argument('--evaluation-plan',type=Path,default=None,help='Prospectively reserved independent-mobility H3 test plan')
    p.add_argument('--research-root',type=Path,default=Path('results/noms_h123_v7'))
    p.add_argument('--protocol',choices=['nominal','independent'],default='nominal')
    args=p.parse_args()
    if args.mode=='baselines':
        if args.onboard_slowdowns is None:
            compare(args.config,args.output_root,args.policies)
        else:
            sensitivity_sweep(args.config,args.output_root,args.onboard_slowdowns,args.policies)
    elif args.mode=='fleet-baselines':
        fleet_scaling_baselines(args.config,args.output_root,args.fleet_sizes,
                               args.onboard_slowdowns or [8.,16.],args.network_scales,
                               args.fleet_trace_template,args.policies)
    elif args.mode=='feasibility-audit':
        run_feasibility_audit(args.config,args.tuning_config,args.output_root,args.audit_split)
    elif args.mode=='research-report':
        generate_research_report(args.research_root,args.output_root,args.fleet_sizes,args.protocol)
    elif args.mode=='mappo-train':
        tuning=yaml.safe_load(args.tuning_config.read_text());_validate_experiment_setup(tuning)
        params={'episodes':args.episodes,'ppo_epochs':args.ppo_epochs,
                'dropout':args.dropout,'hidden_size':tuning.get('hidden_size',128),
                'learning_rate':tuning.get('learning_rate',3e-4),'variant':args.variant,
                'critic_arch':tuning.get('critic_arch','flat')}
        print(train_mappo_one(load_config(args.config),tuning,args.output_root,params,args.seed,args.device))
    elif args.mode=='mappo-sweep':
        run_mappo_sweep(args.config,args.tuning_config,args.output_root,args.device,args.max_trials,variant=args.variant)
    else:
        if args.sweep_root is None:
            p.error('--sweep-root required for mappo-heldout')
        evaluate_mappo_heldout(args.config,args.tuning_config,args.sweep_root,args.output_root,args.device,args.evaluation_plan)

if __name__=='__main__': main()

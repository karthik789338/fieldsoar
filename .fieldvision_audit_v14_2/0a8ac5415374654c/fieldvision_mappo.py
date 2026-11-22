#!/usr/bin/env python3
"""FieldVision-NOMS: reproducible parameter-shared MAPPO research pilot.

Reuses *existing* fieldvision_noms_environment.NOMSSim and
run_orchestration_experiments module without replacing either simulator or data
profiler. Provides CTDE, masked task+placement+network+compute actions, event-
level GAE/PPO, zero-lookahead deterministic evaluation and frozen heuristic
comparisons. All wireless models/local execution assumptions remain synthetic.

CLI: probe | train | evaluate. See Notebook 03 and README.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

K = 3
CATEGORIES = ('CWD30', 'RumexWeeds', 'LeafySpurge')
CLASSES = ('batch', 'standard', 'critical')
TIERS = ('low', 'standard', 'high')
FEATURES_PER_TASK = 11
SCALAR_FEATURES = 9
OBS_DIM = SCALAR_FEATURES + K * FEATURES_PER_TASK
MAX_AGENTS = 12
GLOBAL_DIM = MAX_AGENTS * OBS_DIM + MAX_AGENTS
HEAD_DIMS = (K + 1, 2, 3, 3)    # task+defer, local/edge, network, compute
WEIGHTS = {'critical': 3.0, 'standard': 1.5, 'batch': 1.0}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def exact_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic CPU evaluation, GPU where deterministic kernels permit
    torch.use_deterministic_algorithms(True, warn_only=True)


def imports_contract():
    base = importlib.import_module('run_orchestration_experiments')
    env = importlib.import_module('fieldvision_noms_environment')
    for attr in ('load_profiles', 'create_jobs', 'create_synthetic_network',
                 'make_actions', 'metrics_from_run'):
        if not callable(getattr(base, attr, None)):
            raise RuntimeError(f'Installed Notebook 02 runner missing required function: {attr}')
    for attr in ('NOMSSim', 'Action'):
        if not hasattr(env, attr):
            raise RuntimeError(f'Installed simulator lacks {attr}; upload source for adapter review')
    return base, env


def read_workload_sources(root: Path, base):
    sample = root / 'data/manifests/agricultural_workload_samples_seed2027.json'
    csv_path = root / 'results/reference_pretrained_gpu_profiling_seed2027/agricultural_reference_inference_profiles.csv'
    meta_path = csv_path.parent / 'agricultural_reference_inference_manifest.json'
    for path in (sample, csv_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(f'Notebook 01 required input missing: {path}')
    meta = json.loads(meta_path.read_text())
    if (meta.get('profiles_sha256') != sha256(csv_path) or
            meta.get('sample_manifest_sha256') != sha256(sample) or
            meta.get('weights_mode') != 'pretrained' or meta.get('device') != 'cuda'):
        raise RuntimeError('Frozen reference-GPU profile provenance mismatch')
    return base.load_profiles(csv_path, sample), dict(
        sample_sha256=sha256(sample), profiles_sha256=sha256(csv_path),
        profiles_manifest_sha256=sha256(meta_path))


def make_scenario(base, profiles, seed, n_uavs, bw, local_x, arrival_s, tick_s,
                  drain_s, arrivals_per_uav_s, cloud_x=.8):
    if n_uavs > MAX_AGENTS or n_uavs < 1:
        raise ValueError(f'Fleet must be within 1..{MAX_AGENTS}; increase MAX_AGENTS for larger studies')
    jobs = base.create_jobs(profiles, n_uavs, seed, arrival_s, tick_s,
                            arrivals_per_uav_s, local_x, cloud_x)
    horizon = arrival_s + drain_s
    trace = base.create_synthetic_network(n_uavs, seed, horizon, tick_s, bw)
    if not jobs or any(job['deadline_s'] > horizon + 1e-8 for job in jobs):
        raise ValueError('Need at least one job and mature deadlines within evaluation horizon')
    return {
        'schema_version': 1, 'workload_profile_source': 'synthetic_smoke_only',
        'seed': seed, 'n_uavs': n_uavs, 'duration_s': horizon, 'tick_s': tick_s,
        'network': {'bottleneck_mbps': bw, 'trace_provenance': 'synthetic_contract_only'},
        'resources': {'edge_slots': 2, 'cloud_slots': 3}, 'jobs': jobs,
    }, trace


def materialize_scenario(base, config, trace, directory: Path):
    """Provide the exact on-disk trace required by the existing NOMSSim."""
    import yaml
    directory.mkdir(parents=True, exist_ok=False)
    trace_path = directory / 'synthetic_capacity_envelope.csv'
    base.write_csv(trace_path, trace)
    config = copy.deepcopy(config)
    config['network']['trace_csv'] = str(trace_path.resolve())
    path = directory / 'scenario_configuration.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config, {'scenario_sha256': sha256(path), 'network_sha256': sha256(trace_path)}


def _f(x, cap):
    return max(-3.0, min(3.0, float(x) / cap))


def observe_uav(obs: dict, jobs: dict, now: float):
    """Strictly local/broadcast data, never future link values or other UAV queues."""
    required = ('waiting_tasks', 'local_busy', 'observed_shared_capacity_mbps',
                'observed_uav_link_cap_mbps', 'edge_queue_length')
    for field in required:
        if field not in obs:
            raise RuntimeError(f'Missing local observer field {field}')
    waiting = [jobs[x['task_id']] for x in obs['waiting_tasks']]
    candidates = sorted(waiting, key=lambda j: (j.deadline_s, j.job_id))[:K]
    critical = sum(j.service_class == 'critical' for j in waiting)
    std = sum(j.service_class == 'standard' for j in waiting)
    out = [
        _f(len(waiting), 10), _f(critical, 6), _f(std, 6),
        float(bool(obs['local_busy'])),
        _f(obs['observed_uav_link_cap_mbps'], 25),
        _f(obs['observed_shared_capacity_mbps'], 30),
        _f(obs['edge_queue_length'], 8),
        _f(obs.get('cloud_queue_length', 0), 8),
        _f(now, 120),
    ]
    for k in range(K):
        if k >= len(candidates):
            out.extend([0.0] * FEATURES_PER_TASK)
            continue
        j = candidates[k]
        # Knowledge of estimated job runtimes is allowed: service metadata.
        out.extend([
            1.0, _f(j.input_mb, 10), _f(j.local_runtime_s, 8),
            _f(j.edge_runtime_s, 1), _f(j.deadline_s - now, 30),
            _f(now - j.arrival_s, 15),
            float(j.service_class == 'critical'),
            float(j.service_class == 'standard'),
            float(j.dataset == 'CWD30'),
            float(j.dataset == 'RumexWeeds'),
            float(j.dataset == 'LeafySpurge'),
        ])
    x = np.array(out, dtype=np.float32)
    assert x.shape == (OBS_DIM,), (x.shape, OBS_DIM)
    task_mask = np.array([k < len(candidates) for k in range(K)] + [True], dtype=np.bool_)
    loc_mask = np.array([not obs['local_busy'],
                         obs['observed_shared_capacity_mbps'] > 1e-8 and
                         obs['observed_uav_link_cap_mbps'] > 1e-8], dtype=np.bool_)
    return x, candidates, task_mask, loc_mask


def snapshot(sim, n_uavs: int):
    """One simultaneous snapshot, with actor/critic separated by construction."""
    observations = sim.observations()
    by_agent = [observe_uav(observations[i], sim.jobs, sim.time_s)
                for i in range(n_uavs)]
    global_state = np.zeros((MAX_AGENTS, OBS_DIM), dtype=np.float32)
    global_state[:n_uavs] = np.stack([row[0] for row in by_agent])
    present = np.zeros(MAX_AGENTS, dtype=np.float32)
    present[:n_uavs] = 1
    state = np.concatenate([global_state.reshape(-1), present])
    assert state.size == GLOBAL_DIM
    return by_agent, state


class Actor(nn.Module):
    """Parameter-shared *autoregressive* actor; inputs remain per-UAV only.

    Placement is conditioned on the task choice. Network/compute tiers are
    conditioned on both the task and placement. This avoids the old independent
    heads, which could not explicitly choose different placement distributions
    for different selected queued tasks.
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.task_head = nn.Linear(hidden, K + 1)
        self.placement_head = nn.Sequential(nn.Linear(hidden + K + 1, hidden),
                                            nn.Tanh(), nn.Linear(hidden, 2))
        self.network_head = nn.Sequential(nn.Linear(hidden + K + 1 + 2, hidden),
                                          nn.Tanh(), nn.Linear(hidden, 3))
        self.compute_head = nn.Sequential(nn.Linear(hidden + K + 1 + 2, hidden),
                                          nn.Tanh(), nn.Linear(hidden, 3))

    def forward(self, obs, task=None, placement=None):
        """Accept one observation or a batch; action indices are *past* samples.

        During PPO updates the recorded actions are passed back in to evaluate
        precisely the same conditional probability factorization as rollout.
        """
        from torch.nn import functional as F
        hidden = self.body(obs)
        batched = obs.ndim == 2
        shape = (obs.shape[0],) if batched else ()
        device = obs.device
        chosen_task = torch.full(shape, K, dtype=torch.long, device=device) if task is None else \
            torch.as_tensor(task, dtype=torch.long, device=device)
        chosen_place = torch.zeros(shape, dtype=torch.long, device=device) if placement is None else \
            torch.as_tensor(placement, dtype=torch.long, device=device)
        task_onehot = F.one_hot(chosen_task, num_classes=K + 1).to(hidden.dtype)
        task_context = torch.cat((hidden, task_onehot), dim=-1)
        place_onehot = F.one_hot(chosen_place, num_classes=2).to(hidden.dtype)
        resource_context = torch.cat((task_context, place_onehot), dim=-1)
        return [self.task_head(hidden), self.placement_head(task_context),
                self.network_head(resource_context), self.compute_head(resource_context)]


class Critic(nn.Module):
    """CTDE training only. Global joint observation never passed to actors."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(GLOBAL_DIM, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, state):
        return self.net(state).squeeze(-1)


def safe_categorical(logits, mask):
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if mask.shape != logits.shape:
        raise ValueError(f'Action mask shape {mask.shape} != {logits.shape}')
    if not bool(mask.any(dim=-1).all()):
        raise ValueError('Mask has a row without a valid action')
    return Categorical(logits=logits.masked_fill(~mask, -1.e9))


def choose_action(actor, encoded, deterministic=False):
    x, candidates, task_mask, loc_mask = encoded
    features = torch.from_numpy(x)
    # Only the task head is evaluated before a task has been selected.
    task_logits = actor(features)[0]
    tm = task_mask.copy()
    if not loc_mask.any():
        tm[:K] = False
    dist_task = safe_categorical(task_logits, tm)
    sampled = torch.argmax(dist_task.probs) if deterministic else dist_task.sample()
    selected = int(sampled.item())
    if selected == K:
        place = 0; net = 1; comp = 1
        pm = np.array([True, False]); nm = np.array([False, True, False]); cm = nm.copy()
    else:
        # Conditioning on *this* task lets the actor learn that uploading a
        # large detection image differs from uploading a small classification image.
        conditional = actor(features, task=selected)
        pm = loc_mask.copy()
        dist_place = safe_categorical(conditional[1], pm)
        sampled_place = torch.argmax(dist_place.probs) if deterministic else dist_place.sample()
        place = int(sampled_place.item())
        nm = np.array([True, True, True]) if place == 1 else np.array([False, True, False])
        cm = nm.copy()
        resource_logits = actor(features, task=selected, placement=place)
        dist_net = safe_categorical(resource_logits[2], nm)
        dist_comp = safe_categorical(resource_logits[3], cm)
        net = int((torch.argmax(dist_net.probs) if deterministic else dist_net.sample()).item())
        comp = int((torch.argmax(dist_comp.probs) if deterministic else dist_comp.sample()).item())
    masks = (tm, pm, nm, cm)
    acts = (selected, place, net, comp)
    logits = actor(features, task=selected, placement=place)
    logp, ent = calculate_logp_entropy(logits, masks, acts)
    return acts, masks, float(logp.item()), float(ent.item())


def calculate_logp_entropy(logits, masks, actions):
    """Factorized joint log probability for single or batched conditional heads."""
    logp = torch.zeros(logits[0].shape[:-1], dtype=logits[0].dtype, device=logits[0].device)
    entropy = torch.zeros_like(logp)
    indices = (torch.as_tensor(actions, device=logits[0].device).unbind(-1)
               if logits[0].ndim == 2 else actions)
    for l, m, a in zip(logits, masks, indices):
        dist = safe_categorical(l, m)
        val = torch.as_tensor(a, dtype=torch.long, device=l.device)
        logp = logp + dist.log_prob(val)
        entropy = entropy + dist.entropy()
    return logp, entropy


def to_sim_action(env, decoded, encoded):
    acts, _mask, _lp, _ent = decoded
    index, place, net, comp = acts
    if index == K:
        return env.Action()
    job = encoded[1][index]
    return env.Action(job.job_id, ('local','edge')[place], TIERS[net], TIERS[comp])


class OutcomeReward:
    """Event-driven team reward, using only events that happened by this time."""
    def __init__(self, sim):
        self.cursor = 0
        self.rewarded = set()
        self.completed_ids = set()
        self.weights = {k: WEIGHTS[v.service_class] for k,v in sim.jobs.items()}

    def update(self, sim):
        reward = 0.0
        new_events = sim.events[self.cursor:]
        self.cursor = len(sim.events)
        for event in new_events:
            if event['event'] != 'completed':
                continue
            key = event['task_id']
            if key not in sim.jobs:
                raise ValueError('Completion event refers to non-existent job')
            self.completed_ids.add(key)
            if key in self.rewarded:
                continue
            j = sim.jobs[key]
            t = float(event['time_s'])
            if t <= float(j.deadline_s) + 1e-9:
                self.rewarded.add(key)
                reward += self.weights[key]
        # Mark newly expired deadlines as misses. Only training reward needs this
        # accounting; externally reported SLA comes from the frozen runner.
        for key,j in sim.jobs.items():
            if key not in self.rewarded and j.deadline_s <= sim.time_s + 1e-8:
                self.rewarded.add(key)
                reward -= self.weights[key]
        return reward / max(1, len(sim.jobs))


@dataclass
class Transition:
    state: np.ndarray
    actor_data: list      # per sampled UAV (obs, masks, action, old logp)
    start_t: float
    value: float
    cumulative: float = 0.0
    discount: float = 1.0
    elapsed_s: float = 0.0
    done: bool = False
    next_value: float = 0.0
    advantage: float = 0.0
    ret: float = 0.0


def rollout(config, env, actor, critic=None, deterministic=False,
            training=False, gamma_per_second=.99, decision_interval_s=.1):
    """Use a fixed action cadence; the 10-ms authoritative simulator still runs.

    For fairness, the heuristic comparisons must use the same decision cadence.
    Rewards accrue between decision events, including when no jobs are actionable.
    """
    sim = env.NOMSSim(config)
    decision_stride = tick_stride(sim.dt, decision_interval_s)
    rew = OutcomeReward(sim)
    episode = []
    pending = None
    arrivals = len(sim.jobs)
    n_agents = int(config['n_uavs'])
    total_reward = 0.0
    if training and critic is None:
        raise ValueError('Training requires centralized critic')
    tick = 0
    while sim.time_s < sim.horizon - 1e-9:
        before = sim.time_s
        actions = {}
        if tick % decision_stride == 0:
            snap, global_state = snapshot(sim, n_agents)
            any_jobs = any((enc[1] and enc[3].any()) for enc in snap)
            selected = []
            if any_jobs:
                value = 0.0
                if critic is not None:
                    with torch.no_grad():
                        value = float(critic(torch.from_numpy(global_state)).item())
                if pending is not None:
                    pending.next_value = value
                    episode.append(pending)
                    pending = None
                for i, enc in enumerate(snap):
                    if not enc[1] or not enc[3].any():
                        actions[i] = env.Action()
                        continue
                    with torch.no_grad():
                        dec = choose_action(actor, enc, deterministic=deterministic)
                    actions[i] = to_sim_action(env, dec, enc)
                    if training:
                        selected.append((enc[0].copy(), tuple(m.copy() for m in dec[1]),
                                         tuple(dec[0]), dec[2]))
                if training and selected:
                    pending = Transition(state=global_state.copy(), actor_data=selected,
                                         start_t=before, value=value)
        sim.step(actions=actions)  # empty action dict means defer until next decision tick
        tick += 1
        dt = max(0.0, sim.time_s - before)
        r = rew.update(sim)
        total_reward += r
        if pending is not None:
            pending.cumulative += pending.discount * r
            pending.discount *= gamma_per_second ** dt
            pending.elapsed_s += dt
        if dt <= 0:
            raise RuntimeError('Simulator step failed to advance time')
    if pending is not None:
        pending.done = True
        pending.next_value = 0.0
        episode.append(pending)
    if len(rew.rewarded) != arrivals:
        raise AssertionError(f'Not all deadlines matured: {len(rew.rewarded)}/{arrivals}')
    return sim, episode, total_reward


def tick_stride(tick_s, decision_interval_s):
    if decision_interval_s < tick_s - 1e-9:
        raise ValueError('Decision interval cannot be less than simulation tick')
    steps = round(decision_interval_s / tick_s)
    if steps < 1 or abs(steps * tick_s - decision_interval_s) > 1e-8:
        raise ValueError('Decision interval must be a positive multiple of simulator tick')
    return steps


def gae(episode, lam=.95):
    continuation = 0.0
    for t in reversed(episode):
        delta = t.cumulative + (0 if t.done else t.discount*t.next_value) - t.value
        continuation = delta + (0 if t.done else t.discount*(lam**t.elapsed_s)*continuation)
        t.advantage = continuation
        t.ret = t.value + continuation
    return episode


def update_ppo(actor, critic, actor_optim, critic_optim, episodes,
               ppo_epochs=4, clip=.2, entropy_coef=.01, value_coef=.5,
               minibatch_size=256):
    """Vectorized, minibatched PPO; preserve exact conditional action likelihoods."""
    all_events = [t for ep in episodes for t in ep]
    all_agents = [(t, record) for t in all_events for record in t.actor_data]
    if not all_agents:
        raise RuntimeError('No decision events produced. Check scenario arrival rate.')
    if minibatch_size < 1:
        raise ValueError('minibatch_size must be positive')
    agent_obs = torch.from_numpy(np.stack([r[0] for _, r in all_agents]))
    agent_acts = torch.as_tensor(np.asarray([r[2] for _, r in all_agents]), dtype=torch.long)
    agent_masks = [torch.as_tensor(np.stack([r[1][h] for _, r in all_agents]), dtype=torch.bool)
                   for h in range(4)]
    old_logp = torch.tensor([r[3] for _, r in all_agents], dtype=torch.float32)
    advantage = torch.tensor([t.advantage for t, _ in all_agents], dtype=torch.float32)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
    states = torch.from_numpy(np.stack([t.state for t in all_events]))
    returns = torch.tensor([t.ret for t in all_events], dtype=torch.float32)
    loss_metrics = []
    clip_fracs = []
    for _ in range(ppo_epochs):
        for indices in torch.randperm(len(all_agents)).split(minibatch_size):
            logits = actor(agent_obs[indices], task=agent_acts[indices, 0],
                           placement=agent_acts[indices, 1])
            masks = [m[indices] for m in agent_masks]
            new_logp, entropy = calculate_logp_entropy(logits, masks, agent_acts[indices])
            ratios = torch.exp(new_logp - old_logp[indices])
            obj1 = ratios * advantage[indices]
            obj2 = torch.clamp(ratios, 1-clip, 1+clip) * advantage[indices]
            loss = -torch.minimum(obj1, obj2).mean() - entropy_coef * entropy.mean()
            actor_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), .5)
            actor_optim.step()
            clip_fracs.append(float(((ratios.detach() - 1).abs() > clip).float().mean()))
            loss_metrics.append(('actor', float(loss.detach())))
        for indices in torch.randperm(len(all_events)).split(minibatch_size):
            values = critic(states[indices])
            value_loss = value_coef * ((values - returns[indices])**2).mean()
            critic_optim.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), .5)
            critic_optim.step()
            loss_metrics.append(('critic', float(value_loss.detach())))
    return {
        'actor_loss': statistics.mean(v for kind, v in loss_metrics if kind == 'actor'),
        'critic_loss': statistics.mean(v for kind, v in loss_metrics if kind == 'critic'),
        'ppo_clip_fraction': statistics.mean(clip_fracs),
        'agent_decisions': len(all_agents), 'joint_decisions': len(all_events),
        'optimizer_step': True,
    }


def write_json(path, data):
    Path(path).write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')


def training_manifest(root, provenance, args):
    runner_path=root/'run_orchestration_experiments.py'
    simulator_path=root/'fieldvision_noms_environment.py'
    return {
        'experiment':'FIELDVISION_NOMS_MAPPO_CTDE_DEVELOPMENT_PILOT',
        'claims':'Algorithmic MAPPO pilot; network/local device synthetic and uncalibrated',
        'simulator_sha256':sha256(simulator_path),
        'heuristic_runner_sha256':sha256(runner_path),
        'mappo_source_sha256':sha256(Path(__file__)),
        'profile_provenance':provenance,
        'torch_version':torch.__version__,
        'python_version':os.sys.version.split()[0],
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'parameters':vars(args).copy(),
        'training_seed':args.seed,
        'train_scenarios':[args.seed+i for i in range(args.episodes)],
        'evaluation_seed':args.eval_seed,
        'training_eval_seed_disjoint':args.eval_seed not in [args.seed+i for i in range(args.episodes)],
        'actor_input':'local_and_broadcast_only; no global critic state at execution',
        'centralized_critic':'padded joint vector of contemporaneous local observations',
        'action_heads':'autoregressive task to placement to network/compute tiers',
        'decision_interval_s':args.decision_interval_s,
        'energy_measured':False,'cloud_actions_enabled':False,
    }


def _prepare(root, args, materialized):
    base, env = imports_contract()
    profiles, provenance = read_workload_sources(root, base)
    config, trace = make_scenario(base, profiles, args.seed, args.n_uavs,
                                  args.bw_mbps, args.local_multiplier,
                                  args.arrival_s, args.tick_s, args.drain_s,
                                  args.arrival_rate)
    config, hashes=materialize_scenario(base,config,trace,materialized)
    return base,env,config,provenance,hashes


def probe(args):
    root=Path(args.project_root).resolve()
    out=Path(args.output).resolve()
    if out.exists(): raise FileExistsError(f'Refusing overwrite: {out}')
    out.mkdir(parents=True)
    cfgdir=out/'scenario'
    base,env,config,provenance,hashes=_prepare(root,args,cfgdir)
    exact_seed(args.seed)
    sim=env.NOMSSim(config)
    snap, state=snapshot(sim,int(config['n_uavs']))
    a=Actor();c=Critic()
    with torch.no_grad():
        result=[choose_action(a,enc,deterministic=True) for enc in snap if enc[1]]
    # Step the first tick to verify service and observational contracts.
    sim.step(actions={i:env.Action() for i in range(config['n_uavs'])})
    fields=['observations','step','summary','events','jobs','network_history','compute_history','horizon','time_s']
    for name in fields:
        if not hasattr(sim,name):raise RuntimeError(f'NOMSSim lacks required {name}')
    info={'status':'MAPPO_INTERFACE_PROBE_PASS','scenario_hashes':hashes,
          'simulator_source_sha256':sha256(root/'fieldvision_noms_environment.py'),
          'features':OBS_DIM,'global_features':GLOBAL_DIM,'fleet':config['n_uavs'],
          'sampled_action_count':len(result), 'time_after_step':sim.time_s,
          'decision_interval_s':args.decision_interval_s,
          'network_source':'synthetic_contract_only'}
    write_json(out/'interface_probe.json',info)
    print('MAPPO_INTERFACE_PROBE: PASS')
    print(json.dumps(info,indent=2))


def train(args):
    root=Path(args.project_root).resolve()
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(f'Refusing overwrite: {out}')
    if args.eval_seed in [args.seed+i for i in range(args.episodes)]:
        raise ValueError('Evaluation seed must not occur in training episodes')
    base,env=imports_contract()
    profiles, provenance=read_workload_sources(root,base)
    out.mkdir(parents=True)
    exact_seed(args.seed)
    actor=Actor(hidden=args.hidden)
    critic=Critic(hidden=args.hidden)
    actor_optim=torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim=torch.optim.Adam(critic.parameters(),lr=args.learning_rate)
    logs=[]
    accumulated_episodes=[]
    for ep in range(args.episodes):
        seed=args.seed+ep
        bw=args.bw_mbps if ep%2==0 else args.bw_secondary
        local=args.local_multiplier if ep%2==0 else args.local_secondary
        config,trace=make_scenario(base,profiles,seed,args.n_uavs,bw,local,
                                   args.arrival_s,args.tick_s,args.drain_s,args.arrival_rate)
        scenario_dir=out/'training_scenarios'/f'episode{ep:03d}_seed{seed}'
        config, hashes=materialize_scenario(base,config,trace,scenario_dir)
        sim,events,score=rollout(config,env,actor,critic,training=True,
                                   decision_interval_s=args.decision_interval_s)
        gae(events)
        accumulated_episodes.append(events)
        if (ep + 1) % args.episodes_per_update == 0 or ep == args.episodes - 1:
            update=update_ppo(actor,critic,actor_optim,critic_optim,accumulated_episodes,
                              ppo_epochs=args.ppo_epochs,
                              minibatch_size=args.minibatch_size)
            accumulated_episodes=[]
        else:
            update={'actor_loss':None,'critic_loss':None,'ppo_clip_fraction':None,
                    'agent_decisions':sum(len(t.actor_data) for t in events),
                    'joint_decisions':len(events),'optimizer_step':False}
        metrics=base.metrics_from_run(sim,'mappo_training',sim.summary('mappo_training'),config)[1]
        record={'episode':ep,'seed':seed,'bandwidth_mbps':bw,'local_multiplier':local,
                'team_reward':score,'sla_satisfaction_rate':metrics['sla_satisfaction_rate'],
                'critical_sla':metrics['critical_sla'],**update}
        logs.append(record)
        print(f"MAPPO EP={ep:03d} seed={seed} SLA={record['sla_satisfaction_rate']:.3f} "
              f"critical={record['critical_sla']} decisions={update['agent_decisions']}",flush=True)
    checkpoint={'actor_state_dict':actor.state_dict(),'critic_state_dict':critic.state_dict(),
                'optimizer_actor':actor_optim.state_dict(),'optimizer_critic':critic_optim.state_dict(),
                'seed':args.seed,'training_episode_count':args.episodes,'n_uavs':args.n_uavs,'hidden':args.hidden,'obs_dim':OBS_DIM,
                'max_agents':MAX_AGENTS,'decision_interval_s':args.decision_interval_s,
                'source_sha256':sha256(Path(__file__)),
                'simulator_sha256':sha256(root/'fieldvision_noms_environment.py'),
                'runner_sha256':sha256(root/'run_orchestration_experiments.py')}
    torch.save(checkpoint,out/'mappo_pilot_checkpoint.pt')
    base.write_csv(out/'training_history.csv',logs)
    write_json(out/'training_manifest.json',training_manifest(root,provenance,args))
    print('MAPPO_TRAINING_PILOT: PASS')
    print('RESULTS:',out)


def evaluate(args):
    root=Path(args.project_root).resolve()
    base,env=imports_contract()
    profiles,provenance=read_workload_sources(root,base)
    ckpt_path=Path(args.checkpoint).resolve()
    # Only load your own checkpoint from the training script; torch.save uses pickle.
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=True)
    if ckpt.get('source_sha256')!=sha256(Path(__file__)):
        raise RuntimeError('Checkpoint MAPPO source mismatch')
    if ckpt.get('simulator_sha256') != sha256(root/'fieldvision_noms_environment.py'):
        raise RuntimeError('Simulator source differs from training: incompatible benchmark')
    if ckpt.get('runner_sha256') != sha256(root/'run_orchestration_experiments.py'):
        raise RuntimeError('Heuristic runner differs from training: incompatible benchmark')
    if abs(ckpt.get('decision_interval_s', -1.0)-args.decision_interval_s)>1e-9:
        raise ValueError('Evaluation cadence must match training checkpoint')
    if ckpt.get('seed')<=args.eval_seed<ckpt.get('seed')+ckpt['training_episode_count']:
        raise ValueError('Evaluation seed overlaps training-seed range')
    actor=Actor(hidden=ckpt['hidden']);actor.load_state_dict(ckpt['actor_state_dict']);actor.eval()
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(f'Refusing overwrite: {out}')
    out.mkdir(parents=True)
    results=[]; scenario_hashes=[]
    for n in args.eval_fleets:
        for bw in args.eval_bws:
            for local in args.eval_locals:
                seed=args.eval_seed
                name=f'f{n}_seed{seed}_bw{bw:g}_local{local:g}'
                cfg,tr=make_scenario(base,profiles,seed,n,bw,local,args.arrival_s,
                                     args.tick_s,args.drain_s,args.arrival_rate)
                cfg,hashes=materialize_scenario(base,cfg,tr,out/name)
                scenario_hashes.append({'name':name,**hashes})
                exact_seed(seed)
                with torch.no_grad():
                    learned_sim,_,_ = rollout(cfg,env,actor,critic=None,deterministic=True,
                                              decision_interval_s=args.decision_interval_s)
                details,metrics=base.metrics_from_run(learned_sim,'mappo_deterministic',
                           learned_sim.summary('mappo_deterministic'),cfg)
                (out/name/'mappo_deterministic').mkdir()
                base.write_csv(out/name/'mappo_deterministic'/'task_outcomes.csv',details)
                metrics['edge_task_count']=sum(r['placement']=='edge' for r in details)
                metrics['local_task_count']=sum(r['placement']=='local' for r in details)
                results.append({'scenario':name,'seed':seed,'n_uavs':n,
                                'bottleneck_mbps':bw,'local_multiplier':local,**metrics})
                for policy in base.POLICIES:
                    sim=env.NOMSSim(cfg)
                    heuristic_rng=random.Random(seed)
                    step_count=0
                    stride=tick_stride(sim.dt,args.decision_interval_s)
                    while sim.time_s < sim.horizon - 1.e-9:
                        act=(base.make_actions(sim,policy,cfg,heuristic_rng)
                             if step_count % stride == 0 else {})
                        sim.step(actions=act)
                        step_count+=1
                    detail,met=base.metrics_from_run(sim,policy,sim.summary(policy),cfg)
                    (out/name/policy).mkdir()
                    base.write_csv(out/name/policy/'task_outcomes.csv',detail)
                    met['edge_task_count']=sum(r['placement']=='edge' for r in detail)
                    met['local_task_count']=sum(r['placement']=='local' for r in detail)
                    results.append({'scenario':name,'seed':seed,'n_uavs':n,
                                    'bottleneck_mbps':bw,'local_multiplier':local,**met})
                print('EVALUATED:',name,flush=True)
    base.write_csv(out/'heldout_policy_comparison.csv',results)
    write_json(out/'heldout_manifest.json',{
        'checkpoint_sha256':sha256(ckpt_path),
        'learned_policy':'MAPPO_autoregressive_parameter_shared_actor_CTDE_critic',
        'comparison':'all policies evaluated on exactly identical scenario/task/trace',
        'source_sha256':sha256(Path(__file__)),
        'training_seed':ckpt['seed'],'evaluation_seed':args.eval_seed,
        'scenario_hashes':scenario_hashes, 'results_count':len(results),
        'network_provenance':'synthetic_contract_only',
        'learning_claim':'none_until_replication_and_calibration',
        'decision_interval_s':args.decision_interval_s,
        'baseline_policies':list(base.POLICIES)})
    print('MAPPO_HELDOUT_EVALUATION: PASS')
    print('RESULTS:',out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['probe','train','evaluate'])
    ap.add_argument('--project-root',default='.')
    ap.add_argument('--output',required=True)
    ap.add_argument('--checkpoint',default=None)
    ap.add_argument('--n-uavs',type=int,default=4)
    ap.add_argument('--eval-fleets',type=int,nargs='+',default=[4])
    ap.add_argument('--seed',type=int,default=3101)
    ap.add_argument('--eval-seed',type=int,default=4101)
    ap.add_argument('--episodes',type=int,default=3)
    ap.add_argument('--bw-mbps',type=float,default=18.)
    ap.add_argument('--bw-secondary',type=float,default=12.)
    ap.add_argument('--local-multiplier',type=float,default=45.)
    ap.add_argument('--local-secondary',type=float,default=20.)
    ap.add_argument('--eval-bws',type=float,nargs='+',default=[12.,24.])
    ap.add_argument('--eval-locals',type=float,nargs='+',default=[20.,45.])
    ap.add_argument('--arrival-s',type=float,default=90.)
    ap.add_argument('--drain-s',type=float,default=30.)
    ap.add_argument('--tick-s',type=float,default=.01)
    ap.add_argument('--arrival-rate',type=float,default=.32)
    ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--learning-rate',type=float,default=3.e-4)
    ap.add_argument('--ppo-epochs',type=int,default=4)
    ap.add_argument('--episodes-per-update',type=int,default=4)
    ap.add_argument('--minibatch-size',type=int,default=256)
    ap.add_argument('--decision-interval-s',type=float,default=.1)
    args=ap.parse_args()
    if (args.episodes<1 or args.n_uavs<1 or args.n_uavs>MAX_AGENTS or
        args.drain_s<30 or args.arrival_s<=0 or args.tick_s<=0 or
        args.arrival_rate<=0 or min(args.eval_fleets)<1 or max(args.eval_fleets)>MAX_AGENTS or
        args.episodes_per_update<1 or args.minibatch_size<1):
        ap.error('Invalid mission, fleet, arrival or drain parameters')
    tick_stride(args.tick_s,args.decision_interval_s)
    if args.command=='probe':probe(args)
    elif args.command=='train':train(args)
    else:
        if args.checkpoint is None: ap.error('evaluate requires --checkpoint')
        evaluate(args)

if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""FieldVision-NOMS dataset registration and metadata-only inventory.

Reads the seven-dataset source catalog and registers *existing* agricultural
copies by symlink. Does not download, modify, extract, or run models on data.

Run under the project Python 3.12 environment. Use the `profile` subcommand in
later phases once real models and representative compute hardware are frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import os
import platform
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp', '.dng', '.jp2'}
ARCHIVE_EXT = {'.zip', '.tar', '.tgz', '.7z', '.gz'}
PREVIOUS = ('CWD30', 'LeafySpurge', 'RumexWeeds')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def load_registry(root):
    path = root / 'configs' / 'agricultural_dataset_sources.yaml'
    record = yaml.safe_load(path.read_text(encoding='utf-8'))
    datasets = record['datasets']
    names = [d['id'] for d in datasets]
    if len(names) != len(set(names)) or len(names) != 7:
        raise ValueError('Expected seven uniquely named dataset records')
    return datasets


def original_locations(home):
    # The explicit --agri-root argument overrides these common locations.
    return [
        home / 'data' / 'agri' / 'datasets',
        home / 'agri' / 'datasets',
        Path('/mnt/pixstor/data') / home.name / 'agri' / 'datasets',
    ]


def find_source(srcroot, name):
    if not srcroot.is_dir():
        return None
    exact = srcroot / name
    if exact.exists():
        return exact.resolve()
    matches = [p for p in srcroot.iterdir() if p.name.lower() == name.lower()]
    return matches[0].resolve() if len(matches) == 1 else None


def register(root, agri_root=None):
    targets = [Path(agri_root).expanduser()] if agri_root else original_locations(Path.home())
    chosen = next((p for p in targets if p.is_dir()), None)
    raw = root / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    summary = {'generated_at_utc': timestamp(), 'source_dir': str(chosen) if chosen else None, 'registered': [], 'skipped': []}
    if not chosen:
        print('No existing agri/datasets directory detected. Use --agri-root /absolute/path/to/agri/datasets')
        return summary
    print(f'EXISTING_DATASETS_DIR: {chosen}')
    for name in PREVIOUS:
        source = find_source(chosen, name)
        dest = raw / name
        if source is None:
            summary['skipped'].append({'id': name, 'reason': 'source_not_found'})
            print(f'NOT_FOUND: {name}')
        elif dest.is_symlink() or dest.exists():
            summary['skipped'].append({'id': name, 'reason': 'destination_exists', 'destination': str(dest)})
            print(f'EXISTS_NOT_CHANGED: {dest}')
        else:
            dest.symlink_to(source, target_is_directory=source.is_dir())
            summary['registered'].append({'id': name, 'source': str(source), 'destination': str(dest)})
            print(f'LINKED: {dest} -> {source}')
    manifest = root / 'data' / 'manifests' / 'dataset_registration.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def inspect_zip_metadata(archive):
    # Central-directory reading ONLY: does not extract files or verify content CRC.
    try:
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            return {'metadata_readable': True,
                    'member_count': sum(not i.is_dir() for i in infos),
                    'image_member_count': sum(not i.is_dir() and Path(i.filename).suffix.lower() in IMAGE_EXT for i in infos),
                    'uncompressed_bytes_reported': sum(i.file_size for i in infos if not i.is_dir())}
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        return {'metadata_readable': False, 'error': str(exc)}


def inventory_item(raw, entry, inspect_archives=False):
    name = entry['id']
    target = raw / name
    base = {'dataset_id': name, 'source_url': entry['source_url'], 'role': entry['role'],
            'license': entry['license'], 'registered_path': str(target),
            'exists': target.exists(), 'symlink': target.is_symlink()}
    if not target.exists():
        base['status'] = 'BROKEN_LINK' if target.is_symlink() else 'NOT_REGISTERED'
        return base
    base['resolved_path'] = str(target.resolve())
    counts = Counter()
    total_bytes = 0
    samples = []
    archives = []
    def observe(path, display):
        nonlocal total_bytes
        try:
            st = path.stat()
        except OSError:
            counts['unreadable_files'] += 1
            return
        total_bytes += st.st_size
        counts['file_count'] += 1
        ext = path.suffix.lower()
        if ext in IMAGE_EXT:
            counts['image_files'] += 1
        if ext in ARCHIVE_EXT:
            counts['archive_files'] += 1
            if inspect_archives and ext == '.zip':
                archives.append({'path': str(path), **inspect_zip_metadata(path)})
        if len(samples) < 6:
            samples.append({'path': display, 'bytes': st.st_size})
        counts[f'extension:{ext or "(none)"}'] += 1
    if target.is_file():
        observe(target, target.name)
    elif target.is_dir():
        for folder, dirs, files in os.walk(target, followlinks=False):
            dirs.sort()
            files.sort()
            for filename in files:
                p = Path(folder) / filename
                if p.is_symlink():
                    counts['nested_symlinks_skipped'] += 1
                    continue
                observe(p, str(p.relative_to(target)))
    else:
        base['status'] = 'UNSUPPORTED_PATH_TYPE'
        return base
    base.update(status='REGISTERED_METADATA_ONLY', total_stored_bytes=total_bytes,
                file_count=counts['file_count'], image_file_count=counts['image_files'],
                archive_file_count=counts['archive_files'], other_unreadable_files=counts['unreadable_files'],
                nested_symlinks_skipped=counts['nested_symlinks_skipped'],
                file_extensions={k[10:]: v for k, v in sorted(counts.items()) if k.startswith('extension:')},
                sample_files=samples)
    if inspect_archives:
        base['zip_archive_metadata'] = archives
    return base


def inventory(root, inspect_archives=False):
    records = [inventory_item(root / 'data' / 'raw', row, inspect_archives)
               for row in load_registry(root)]
    output = {'schema_version': 1, 'generated_at_utc': timestamp(), 'python': sys.version,
              'host': platform.node(), 'verification_level': 'metadata_only_no_content_hash_or_label_audit',
              'datasets': records}
    outdir = root / 'data' / 'manifests'
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'agricultural_dataset_inventory.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    with (outdir / 'agricultural_dataset_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset_id', 'status', 'image_file_count', 'archive_file_count', 'file_count', 'total_stored_bytes', 'resolved_path'])
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, '') for k in w.fieldnames})
    for r in records:
        print(f"{r['dataset_id']:18s} {r['status']:26s} images={r.get('image_file_count', 0):>7} "
              f"archives={r.get('archive_file_count', 0):>3} bytes={r.get('total_stored_bytes', 0):>13}")
        for z in r.get('zip_archive_metadata', []):
            print(f"  ZIP_MEMBERS: {z['path']} images={z.get('image_member_count', '?')} readable={z['metadata_readable']}")
    print('INVENTORY_WRITTEN:', outdir / 'agricultural_dataset_inventory.json')
    return output


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()



# Trajectory-driven uplink model, explicitly UNCALIBRATED.
# This uses UavNetSim mobility samples but is NOT native Sionna channel data.
def load_radio_assumptions(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    needed = ('n_uavs', 'duration_s', 'tick_s', 'seed', 'ground_station_m',
              'carrier_frequency_ghz', 'channel_bandwidth_mhz', 'tx_power_dbm',
              'noise_figure_db', 'path_loss_exponent', 'shadow_sigma_db',
              'shadow_ar1_rho', 'rician_k', 'interference_margin_db',
              'snr_disconnect_db', 'protocol_efficiency', 'max_link_mbps',
              'shared_bottleneck_mbps')
    if not isinstance(cfg, dict) or not all(k in cfg for k in needed):
        raise ValueError('Incomplete analytical radio assumptions YAML')
    if not isinstance(cfg['ground_station_m'], list) or len(cfg['ground_station_m']) != 3:
        raise ValueError('ground_station_m must be [x,y,z] in meters')
    n = int(cfg['n_uavs']); dt = float(cfg['tick_s']); duration = float(cfg['duration_s'])
    if n < 1 or dt <= 0 or duration <= 0 or abs(round(duration/dt)*dt-duration) > 1e-7:
        raise ValueError('Invalid n_uavs, tick_s or duration_s')
    positive = ('carrier_frequency_ghz','channel_bandwidth_mhz','path_loss_exponent',
                'max_link_mbps','shared_bottleneck_mbps')
    if any(not math.isfinite(float(cfg[k])) or float(cfg[k]) <= 0 for k in positive):
        raise ValueError('Radio frequency/bandwidth/loss/capacity must be positive')
    if float(cfg['shadow_sigma_db']) < 0 or not 0 <= float(cfg['shadow_ar1_rho']) < 1:
        raise ValueError('Invalid shadowing assumptions')
    if float(cfg['rician_k']) < 0 or not 0 < float(cfg['protocol_efficiency']) <= 1:
        raise ValueError('Invalid fading/protocol efficiency')
    for k in ('tx_power_dbm','noise_figure_db','interference_margin_db','snr_disconnect_db'):
        if not math.isfinite(float(cfg[k])):
            raise ValueError('Nonfinite radio parameter: '+k)
    if not all(math.isfinite(float(x)) for x in cfg['ground_station_m']):
        raise ValueError('Nonfinite base-station coordinate')
    return cfg


def read_uavnetsim_trajectory(path, n_uavs, duration, dt):
    """Require complete (tick,UAV) coverage; never extrapolate a 3s capture to 40s."""
    required = ('sim_time_s','uav_id','x_m','y_m','z_m','sleeping')
    ticks = round(duration/dt)
    values = {}
    with Path(path).open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not all(c in reader.fieldnames for c in required):
            raise ValueError('UavNetSim trajectory CSV lacks required columns')
        for row in reader:
            t = float(row['sim_time_s']); uav = int(row['uav_id'])
            index = round(t/dt)
            if t < -1e-7 or index >= ticks:
                continue  # Extra later trajectory data are not used
            if abs(index*dt-t) > 1e-6 or not 0 <= uav < n_uavs:
                raise ValueError('Misaligned timestamp or out-of-range UAV ID')
            xyz = tuple(float(row[k]) for k in ('x_m','y_m','z_m'))
            if not all(math.isfinite(v) for v in xyz):
                raise ValueError('Nonfinite UAV position')
            sleep = str(row['sleeping']).strip().lower()
            if sleep not in ('true','false','0','1'):
                raise ValueError('Invalid sleeping field')
            key = (index,uav)
            if key in values:
                raise ValueError(f'Duplicate trajectory (tick,UAV): {key}')
            values[key] = (*xyz, sleep in ('true','1'))
    missing = [(k,i) for k in range(ticks) for i in range(n_uavs) if (k,i) not in values]
    if missing:
        raise ValueError(f'Trajectory incomplete: {len(missing)} missing tick/UAV positions, first={missing[0]}')
    return values


def build_analytical_radio_trace(trajectory, cfg):
    """Emit analytical, action-independent UAV -> ground-station capacity envelopes.

    The local per-UAV envelope derives from log-distance loss + AR(1) lognormal
    shadowing + Rician fading + Shannon bound scaled by ASSUMED MAC efficiency.
    The shared bottleneck is an explicit assumed transport cap. This is NOT
    calibrated radio performance or a packet simulator, and is unsuitable for
    a measured-network claim until validated independently.
    """
    rng = random.Random(int(cfg['seed']))
    n = int(cfg['n_uavs']); dt = float(cfg['tick_s'])
    ticks = round(float(cfg['duration_s'])/dt)
    station = tuple(float(x) for x in cfg['ground_station_m'])
    frequency = float(cfg['carrier_frequency_ghz']) * 1e9
    bandwidth_hz = float(cfg['channel_bandwidth_mhz']) * 1e6
    fspl_1m_db = 20 * math.log10(4 * math.pi * frequency / 299792458.0)
    noise_dbm = -174 + 10 * math.log10(bandwidth_hz) + float(cfg['noise_figure_db'])
    sigma = float(cfg['shadow_sigma_db']); rho = float(cfg['shadow_ar1_rho'])
    k_factor = float(cfg['rician_k'])
    shadow = [0.0 for _ in range(n)]
    trace = []; diag = []
    for tick in range(ticks):
        caps = []
        for i in range(n):
            x,y,z,sleeping = trajectory[(tick,i)]
            distance = max(1.0, math.dist((x,y,z),station))
            shadow[i] = rho*shadow[i] + math.sqrt(1-rho*rho)*sigma*rng.gauss(0,1)
            pathloss_db = fspl_1m_db + 10 * float(cfg['path_loss_exponent']) * math.log10(distance) + shadow[i]
            fading_re = math.sqrt(k_factor/(k_factor+1)) + rng.gauss(0, math.sqrt(1/(2*(k_factor+1))))
            fading_im = rng.gauss(0, math.sqrt(1/(2*(k_factor+1))))
            fading_power = max(fading_re*fading_re + fading_im*fading_im, 1e-12)
            snr_db = (float(cfg['tx_power_dbm']) - pathloss_db - noise_dbm
                      - float(cfg['interference_margin_db']) + 10*math.log10(fading_power))
            if sleeping or snr_db < float(cfg['snr_disconnect_db']):
                capacity_mbps = 0.0
            else:
                snr_linear = 10**(snr_db/10)
                capacity_mbps = min(float(cfg['max_link_mbps']),
                    float(cfg['channel_bandwidth_mhz'])*math.log2(1+snr_linear)*float(cfg['protocol_efficiency']))
            caps.append(round(capacity_mbps,8))
            diag.append({'start_time_s':round(tick*dt,9),'uav_id':i,
                         'distance_to_ground_station_m':round(distance,6),
                         'shadowing_db':round(shadow[i],6), 'snr_db':round(snr_db,6),
                         'link_cap_mbps':round(capacity_mbps,8),'sleeping':int(sleeping)})
        trace.append({'start_time_s':round(tick*dt,9),
                      'bottleneck_mbps':float(cfg['shared_bottleneck_mbps']),
                      **{f'uav_{i}_link_cap_mbps':caps[i] for i in range(n)}})
    return trace,diag


def write_dict_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def prepare_radio_trace(trajectory_path, assumption_path, output_path, overwrite=False):
    trajectory_path = Path(trajectory_path).resolve(strict=True)
    assumption_path = Path(assumption_path).resolve(strict=True)
    output_path = Path(output_path).resolve()
    manifest_path = output_path.with_suffix('.manifest.json')
    diagnostics_path = output_path.with_suffix('.diagnostics.csv')
    if not overwrite and any(p.exists() for p in (output_path,manifest_path,diagnostics_path)):
        raise FileExistsError('Refusing to overwrite trace, diagnostics, or manifest; use --overwrite explicitly')
    source_manifest = trajectory_path.with_name('uavnetsim_capture_manifest.json')
    if not source_manifest.is_file():
        raise FileNotFoundError('Missing UavNetSim capture manifest adjacent to trajectory CSV')
    capture = json.loads(source_manifest.read_text(encoding='utf-8'))
    trajectory_sha = hash_file(trajectory_path)
    if capture.get('trajectories_sha256') != trajectory_sha:
        raise ValueError('Trajectory SHA256 disagrees with UavNetSim capture manifest')
    cfg = load_radio_assumptions(assumption_path)
    positions = read_uavnetsim_trajectory(trajectory_path,int(cfg['n_uavs']),
                                          float(cfg['duration_s']),float(cfg['tick_s']))
    trace, diagnostics = build_analytical_radio_trace(positions,cfg)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    write_dict_csv(output_path,trace)
    write_dict_csv(diagnostics_path,diagnostics)
    record = {
      'schema_version':1,
      'provenance':'trajectory_driven_analytical_uncalibrated',
      'measurement_claim':'NONE: trajectories are from UavNetSim, radio rates are analytical assumptions, not measured Sionna/AERPAW capacity',
      'ground_edge_radio_modeled_in_uavnetsim':False,
      'source_trajectory_csv':str(trajectory_path),'source_trajectory_sha256':trajectory_sha,
      'capture_manifest_path':str(source_manifest),'capture_manifest_sha256':hash_file(source_manifest),
      'uavnetsim_commit':capture.get('uavnetsim_git',{}).get('commit'),
      'assumptions_yaml':str(assumption_path),'assumptions_sha256':hash_file(assumption_path),
      'assumptions':cfg,'generated_trace_csv':str(output_path),
      'generated_trace_sha256':hash_file(output_path),
      'diagnostics_csv':str(diagnostics_path),'diagnostics_sha256':hash_file(diagnostics_path),
      'generator_sha256':hash_file(Path(__file__)),
      'note':'Research scaffolding. Do NOT report as calibrated UAV-to-edge radio performance.'
    }
    manifest_path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('TRAJECTORY_DRIVEN_ANALYTICAL_TRACE: PASS')
    print('TIME_STEPS:',len(trace),'UAVS:',cfg['n_uavs'])
    print('TRACE:',output_path)
    print('MANIFEST:',manifest_path)
    return record


# Real-input, frozen-architecture workload profiling (no model training).
# The original imagery is never changed. Reference weights are NOT weed-trained.
WORKLOAD_REGISTRY = {
    'CWD30': ('classification', 'mobilenet_v3_small'),
    'RumexWeeds': ('detection', 'fasterrcnn_mobilenet_v3_large_320_fpn'),
    'LeafySpurge': ('segmentation', 'lraspp_mobilenet_v3_large'),
}


def _image_candidates(dataset_root, dataset_id, max_stored_bytes):
    """Enumerate candidate image locations without decompressing the CWD30 ZIP."""
    if dataset_id == 'CWD30':
        archive = dataset_root / 'data.zip'
        if not archive.is_file():
            raise FileNotFoundError(f'Expected existing CWD30 archive: {archive}')
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                if (member.is_dir() or Path(member.filename).suffix.lower() not in IMAGE_EXT
                        or member.file_size > max_stored_bytes or member.file_size <= 0
                        or member.flag_bits & 1):
                    continue
                yield ('zip', str(archive), member.filename, member.file_size)
    else:
        if not dataset_root.is_dir():
            raise FileNotFoundError(f'Missing dataset: {dataset_root}')
        for folder, dirs, names in os.walk(dataset_root, followlinks=False):
            dirs.sort()
            for filename in sorted(names):
                image_path = Path(folder)/filename
                if image_path.is_symlink() or image_path.suffix.lower() not in IMAGE_EXT:
                    continue
                # Avoid annotation files, masks and full orthomosaics for image inference.
                relative = image_path.relative_to(dataset_root)
                component_names = [s.lower() for s in relative.parts[:-1]]
                if any(('mask' in s or 'label' in s or 'annotation' in s or
                         'ortho' in s or 'mosaic' in s) for s in component_names):
                    continue
                size = image_path.stat().st_size
                if 0 < size <= max_stored_bytes:
                    yield ('file', str(image_path), str(relative), size)


def _read_image_bytes(record):
    kind, path, name, _ = record
    if kind == 'zip':
        with zipfile.ZipFile(path) as stream:
            return stream.read(name)
    return Path(path).read_bytes()


def build_workload_sample_manifest(root, output_path, sample_count=16, seed=2027,
                                   max_stored_mb=64, min_dimension=128):
    """Decode and checksum a reproducible *sample*, without extracting CWD30.

    This verifies only selected items, NOT an entire dataset. All input bytes
    remain untouched. A later checkpointed model must be used for final claims.
    """
    from io import BytesIO
    from PIL import Image, ImageOps, UnidentifiedImageError
    if sample_count < 1 or min_dimension < 1 or max_stored_mb < 1:
        raise ValueError('sample_count, min_dimension and max_stored_mb must be positive')
    out = Path(output_path).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing sample manifest: {out}')
    registry_ids = {x['id'] for x in load_registry(root)}
    selected = []
    summary = {}
    for dataset_id, (family, architecture) in WORKLOAD_REGISTRY.items():
        if dataset_id not in registry_ids:
            raise ValueError(f'Dataset missing from source catalog: {dataset_id}')
        src = (root / 'data' / 'raw' / dataset_id).resolve(strict=True)
        candidates = list(_image_candidates(src, dataset_id, max_stored_mb * 1024**2))
        if not candidates:
            raise ValueError(f'No eligible images: {dataset_id}')
        rng = random.Random(f'{seed}:{dataset_id}')
        rng.shuffle(candidates)
        # Over-sample candidates to skip corrupt/small files, while bounded in
        # memory and keeping deterministic filenames and selections.
        accepted = 0
        rejected = Counter()
        for item in candidates:
            if accepted >= sample_count:
                break
            try:
                raw = _read_image_bytes(item)
                with Image.open(BytesIO(raw)) as image:
                    w, h = image.size
                    if min(w,h) < min_dimension or w*h > 32_000_000:
                        rejected['resolution_limits'] += 1
                        continue
                    image = ImageOps.exif_transpose(image)
                    image.convert('RGB').resize((224, 224)).load()
                    fmt = image.format or Path(item[2]).suffix.lstrip('.')
                selected.append({
                    'dataset_id':dataset_id,
                    'workload_family':family,
                    'reference_architecture':architecture,
                    'storage_kind':item[0],
                    'source_path':item[1],
                    'relative_image_path':item[2],
                    'stored_bytes':len(raw),
                    'content_sha256':hashlib.sha256(raw).hexdigest(),
                    'image_width':w,
                    'image_height':h,
                    'input_format':fmt,
                })
                accepted += 1
            except (OSError, ValueError, UnidentifiedImageError, zipfile.BadZipFile) as exc:
                rejected[type(exc).__name__] += 1
        if accepted != sample_count:
            raise ValueError(f'Only {accepted}/{sample_count} valid samples from {dataset_id}: {dict(rejected)}')
        summary[dataset_id] = {'sampled':accepted, 'eligible_candidate_count':len(candidates),
                               'sample_rejection_counts':dict(rejected)}
    output = {'schema_version':1, 'purpose':'empirical_compute_profiling_input_selection',
              'sampling_seed':seed, 'per_dataset_sample_count':sample_count,
              'data_validation_scope':'sample_only_not_full_dataset_integrity',
              'model_accuracy_assessed':False,
              'source_catalog_sha256':hash_file(root/'configs'/'agricultural_dataset_sources.yaml'),
              'summary':summary,'samples':selected}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('AGRICULTURAL_WORKLOAD_SAMPLE_MANIFEST: PASS')
    for key, val in summary.items():
        print(key, 'validated_samples=',val['sampled'],'candidates=',val['eligible_candidate_count'])
    print('SAMPLE_MANIFEST:',out)
    return output


def _load_reference_model(dataset_id, weights_mode):
    """Use real PyTorch operators with an explicit checkpoint provenance flag."""
    if dataset_id == 'CWD30':
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if weights_mode == 'pretrained' else None
        return mobilenet_v3_small(weights=weights), weights
    if dataset_id == 'RumexWeeds':
        from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_320_fpn,
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights)
        weights = (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                   if weights_mode == 'pretrained' else None)
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights, weights_backbone=None,
            min_size=320, max_size=320)
        return model, weights
    if dataset_id == 'LeafySpurge':
        from torchvision.models.segmentation import (lraspp_mobilenet_v3_large,
            LRASPP_MobileNet_V3_Large_Weights)
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if weights_mode == 'pretrained' else None
        model = lraspp_mobilenet_v3_large(weights=weights, weights_backbone=None)
        return model, weights
    raise ValueError(f'Unsupported reference model dataset: {dataset_id}')


def profile_reference_models(input_manifest, output_dir, device='cpu', weights_mode='pretrained',
                             repeats=3, warmup=3, max_samples=None):
    """Measure real image decode/preprocess and frozen forward pass runtimes.

    ImageNet/COCO/VOC weights are generic references: they do NOT establish
    trained weed/forage prediction quality. Architecture-only weights may be
    used for a compute-stack smoke run but NEVER as agricultural AI models.
    """
    if weights_mode not in ('pretrained','architecture_only'):
        raise ValueError('weights_mode must be pretrained or architecture_only')
    if repeats < 1 or warmup < 0 or (max_samples is not None and max_samples < 1):
        raise ValueError('Invalid repeats/warmup/max_samples')
    import time
    import statistics
    from io import BytesIO
    from PIL import Image, ImageOps
    import torch
    import torchvision
    import numpy as np
    manifest_path=Path(input_manifest).resolve(strict=True)
    source=json.loads(manifest_path.read_text(encoding='utf-8'))
    if source.get('schema_version') != 1 or not source.get('samples'):
        raise ValueError('Invalid sample manifest')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA, but PyTorch cannot access a CUDA device')
    if device not in ('cpu','cuda'):
        raise ValueError('device must be cpu or cuda')
    if device == 'cpu':
        torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
    out=Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Output directory exists, refusing overwrite: {out}')
    out.mkdir(parents=True)
    rows=[]
    model_infos={}
    try:
        for dataset_id in WORKLOAD_REGISTRY:
            samples=[s for s in source['samples'] if s['dataset_id']==dataset_id]
            samples=samples[:max_samples] if max_samples else samples
            if not samples:
                raise ValueError(f'No selected images for {dataset_id}')
            model, weight_enum=_load_reference_model(dataset_id, weights_mode)
            model.to(device).eval()
            model_infos[dataset_id] = {'architecture':WORKLOAD_REGISTRY[dataset_id][1],
                'model_origin':'torchvision',
                'weights_mode':weights_mode,
                'weights_url':getattr(weight_enum,'url',None),
                'agriculture_finetuned':False,
                'accuracy_evaluated':False}
            if weight_enum is not None:
                from urllib.parse import urlparse
                p=Path(torch.hub.get_dir())/'checkpoints'/Path(urlparse(weight_enum.url).path).name
                if p.is_file():
                    model_infos[dataset_id]['local_weights_sha256']=hash_file(p)
            for idx,sample in enumerate(samples):
                identity=(sample['storage_kind'], sample['source_path'], sample['relative_image_path'],sample['stored_bytes'])
                raw=_read_image_bytes(identity)
                if hashlib.sha256(raw).hexdigest()!=sample['content_sha256']:
                    raise ValueError(f'Sampled input checksum changed: {sample["relative_image_path"]}')
                a=time.perf_counter()
                with Image.open(BytesIO(raw)) as image:
                    image=ImageOps.exif_transpose(image).convert('RGB')
                    image=image.resize((320,320) if dataset_id!='CWD30' else (224,224))
                    arr=np.asarray(image, dtype=np.uint8).copy()
                tensor=torch.from_numpy(arr).permute(2,0,1).float().div_(255).to(device)
                # TorchVision classification/segmentation reference checkpoints
                # expect normalized RGB inputs. Detection models normalize internally.
                if dataset_id != 'RumexWeeds':
                    mean=tensor.new_tensor((0.485,0.456,0.406)).view(3,1,1)
                    std=tensor.new_tensor((0.229,0.224,0.225)).view(3,1,1)
                    tensor=tensor.sub(mean).div(std)
                if device=='cuda':
                    torch.cuda.synchronize()
                preprocess_s=time.perf_counter()-a
                def forward():
                    with torch.inference_mode():
                        if dataset_id=='RumexWeeds':
                            return model([tensor])
                        return model(tensor.unsqueeze(0))
                if device=='cuda':
                    torch.cuda.synchronize()
                for _ in range(warmup):
                    forward()
                if device=='cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                durations=[]
                for _ in range(repeats):
                    if device=='cuda':
                        torch.cuda.synchronize()
                    start=time.perf_counter()
                    forward()
                    if device=='cuda':
                        torch.cuda.synchronize()
                    durations.append(time.perf_counter()-start)
                rows.append({'dataset_id':dataset_id, 'image_path':sample['relative_image_path'],
                    'model':WORKLOAD_REGISTRY[dataset_id][1], 'weights_mode':weights_mode,
                    'input_bytes':len(raw),'width_px':sample['image_width'],'height_px':sample['image_height'],
                    'preprocess_ms':round(preprocess_s*1000,5),
                    'inference_mean_ms':round((sum(durations)/len(durations))*1000,5),
                    'inference_min_ms':round(min(durations)*1000,5),
                    'inference_max_ms':round(max(durations)*1000,5),
                    'peak_gpu_allocated_mb':(round(torch.cuda.max_memory_allocated()/1024**2,3)
                        if device=='cuda' else None),
                    'device':device})
            del model
            if device=='cuda':
                torch.cuda.empty_cache()
        csv_path=out/'agricultural_reference_inference_profiles.csv'
        write_dict_csv(csv_path,rows)
        summary={}
        for dataset_id in WORKLOAD_REGISTRY:
            times=[x['inference_mean_ms'] for x in rows if x['dataset_id']==dataset_id]
            summary[dataset_id]={'samples':len(times), 'mean_inference_ms':sum(times)/len(times),
                'median_inference_ms':statistics.median(times),
                'p95_inference_ms':float(np.percentile(times,95)),
                'min_inference_ms':min(times),'max_inference_ms':max(times),
                'mean_input_mb':sum(x['input_bytes'] for x in rows if x['dataset_id']==dataset_id)/len(times)/1024**2}
        provenance={
            'schema_version':1, 'profile_kind':'frozen_reference_architecture_forward_pass',
            'NOT_agricultural_model_validation':True,
            'training_or_finetuning_performed':False,
            'weights_mode':weights_mode,
            'reference_weight_limitation':'Pretrained generic computer vision weights are not agricultural weed/crop checkpoints',
            'device':device, 'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
            'gpu_compute_capability':(torch.cuda.get_device_capability(0) if device=='cuda' else None),
            'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'torch_version':torch.__version__, 'torchvision_version':torchvision.__version__,
            'python':sys.version, 'platform':platform.platform(),
            'sample_manifest_sha256':hash_file(manifest_path),
            'script_sha256':hash_file(Path(__file__)),
            'model_details':model_infos, 'warmup':warmup,'repeats':repeats,
            'profiles_sha256':hash_file(csv_path),'dataset_summary':summary,
            'energy_measured':False, 'gpu_utilization_measured':False,
            'note':'Processing-time measurements; application accuracy, SLA deadlines, Jetson runtime and energy remain unmeasured.'}
        (out/'agricultural_reference_inference_manifest.json').write_text(
            json.dumps(provenance,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
        print('FROZEN_REFERENCE_MODEL_PROFILING: PASS')
        print(json.dumps(summary,indent=2))
        print('PROFILES_WRITTEN:',csv_path)
        return provenance
    except Exception:
        # Preserve partial outputs for debugging rather than silently marking PASS.
        (out/'profiling_failed.json').write_text(json.dumps({'status':'FAILED',
            'note':'See process stderr. No success manifest was written.'},indent=2)+'\n')
        raise


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--project-root', type=Path, default=Path.cwd())
    sub = cli.add_subparsers(dest='command', required=True)
    reg = sub.add_parser('register', help='Create non-destructive links to previously downloaded data')
    reg.add_argument('--agri-root', type=Path, default=None)
    inv = sub.add_parser('inventory', help='Inventory all seven datasets without modifying them')
    inv.add_argument('--inspect-archives', action='store_true', help='Read ZIP central directories, without extraction/CRC checks')
    checksum = sub.add_parser('checksum', help='Compute SHA256 for one explicit file (may be slow)')
    checksum.add_argument('file', type=Path)
    radio = sub.add_parser('radio-trace', help='Generate UNCALIBRATED analytical UAV-edge capacity from native UavNetSim mobility')
    radio.add_argument('--trajectory-csv', type=Path, required=True)
    radio.add_argument('--radio-config', type=Path, required=True)
    radio.add_argument('--output-trace', type=Path, required=True)
    radio.add_argument('--overwrite', action='store_true')
    samples = sub.add_parser('sample-workloads', help='Select and verify deterministic real-image samples')
    samples.add_argument('--output',type=Path,default=Path('data/manifests/agricultural_workload_samples_seed2027.json'))
    samples.add_argument('--per-dataset',type=int,default=16)
    samples.add_argument('--seed',type=int,default=2027)
    samples.add_argument('--max-stored-mb',type=int,default=64)
    samples.add_argument('--min-dimension',type=int,default=128)
    profile = sub.add_parser('profile-workloads', help='Benchmark frozen torchvision reference models on real images')
    profile.add_argument('--samples',type=Path,default=Path('data/manifests/agricultural_workload_samples_seed2027.json'))
    profile.add_argument('--output',type=Path,required=True)
    profile.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    profile.add_argument('--weights',choices=['pretrained','architecture_only'],default='pretrained')
    profile.add_argument('--warmup',type=int,default=3)
    profile.add_argument('--repeats',type=int,default=3)
    profile.add_argument('--max-samples',type=int,default=None)
    args = cli.parse_args()
    root = args.project_root.expanduser().resolve()
    if args.command == 'register':
        register(root, args.agri_root)
    elif args.command == 'inventory':
        inventory(root, args.inspect_archives)
    elif args.command == 'radio-trace':
        prepare_radio_trace(args.trajectory_csv, args.radio_config, args.output_trace,
                            overwrite=args.overwrite)
    elif args.command == 'sample-workloads':
        build_workload_sample_manifest(root,args.output,args.per_dataset,args.seed,
            args.max_stored_mb,args.min_dimension)
    elif args.command == 'profile-workloads':
        profile_reference_models(args.samples,args.output,args.device,args.weights,
            args.repeats,args.warmup,args.max_samples)
    elif args.command == 'checksum':
        p = args.file.expanduser().resolve(strict=True)
        if not p.is_file():
            cli.error('checksum expects a regular file')
        print(f'{hash_file(p)}  {p}')


if __name__ == '__main__':
    main()

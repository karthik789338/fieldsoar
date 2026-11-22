from pathlib import Path
import hashlib,json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from prepare_research_inputs import prepare_radio_trace

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

h = Path('results/noms_horizon_check')
original = h/'smaller_fleets_cuda_20261120T075503Z'
l = h/'mobility_cuda_20261120T050646Z'
inputs = Path('data/traces')
inputs.mkdir(parents=True, exist_ok=True)
report = {}
for f in (4,8,16,32,64,128):
    root=l if f==128 else original/f'fleet{f}'
    for seed,p in ((27,'nominal'),(41,'stress')):
        capture=Path(f'results/uavnetsim_native_telemetry_{f}uav_seed{seed}/uavnetsim_uav_trajectories.csv')
        source=root/f'traces/seed{seed}'
        cfg=yaml.safe_load((source/'radio_config.yaml').read_text())
        cfg['duration_s']=3.0
        config=inputs/f'radio_{f}uav_seed{seed}.yaml'
        config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        out=inputs/f'uavnetsim_trajectory_analytical_uplink_{f}uav_3s_seed{seed}.csv'
        prepare_radio_trace(capture,config,out,overwrite=True)
        sidecar=out.with_suffix('.manifest.json')
        data=json.loads(sidecar.read_text())
        for key in ('assumptions_yaml','capture_manifest_path','diagnostics_csv','generated_trace_csv','source_trajectory_csv'):
            data[key]=Path(data[key]).relative_to(Path.cwd()).as_posix()
        sidecar.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
        prefix=root/'full'/f'{p}_trace_prefixes'/f'radio_30s_seed{seed}_prefix_3s.csv'
        assert out.read_bytes()==prefix.read_bytes(),(f,seed,'prefix')
        capture_meta=json.loads((source/'capture/uavnetsim_capture_manifest.json').read_text())
        assert sha(capture)==capture_meta['original_3s_trajectory_sha256'], (f,seed)
        expected=capture_meta.get('original_3s_radio_sha256')
        if expected is not None: assert sha(out)==expected,(f,seed,'original')
        report[out.as_posix()]={'trace_sha256':sha(out),'published_prefix':prefix.as_posix(),
                                 'matched_original_trace_sha256':expected is not None}
Path('data/traces/reconstruction_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print('All twelve three-second traces byte-match the archived prefixes.')

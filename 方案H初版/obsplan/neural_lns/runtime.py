"""Read-only released scene loading and explicit Linux/CPU/CUDA runtime evidence."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from obsplan.config import Config
from obsplan.data import load
from obsplan.problem import Problem
from obsplan.visibility import Windows


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('dataset_version') != 'synthetic-v0.2':
        raise ValueError('explicit synthetic-v0.2 dataset required')
    return root, manifest


def load_scene(root, catalog_id, cache_dir, *, allowed_splits=('train', 'validation')):
    """Use published complete windows; never regenerate geometry or write source data."""
    started = time.perf_counter()
    root, manifest = read_manifest(root)
    matches = [r for r in manifest['scenes'] if r['catalog_id'] == catalog_id]
    if len(matches) != 1 or matches[0]['split'] not in allowed_splits:
        raise ValueError('scene missing, duplicated, or forbidden split')
    record = matches[0]
    scene_path = (root / record['path']).resolve()
    cache = Path(cache_dir).resolve()
    if not scene_path.is_relative_to(root) or cache.is_relative_to(root):
        raise ValueError('dataset must be read-only and paths must stay inside it')
    cache.mkdir(parents=True, exist_ok=True)
    cfg = Config(**{**record['config'], 'data_dir': str(scene_path / 'input'),
                    'cache_dir': str(cache)})
    ds = load(cfg)
    windows = root / 'catalogs' / catalog_id / 'windows.npz'
    win = Windows.load(windows, ds.n_stations)
    pb = Problem.build(cfg, ds, win)
    metadata = {'catalog_id': catalog_id, 'scene_id': record['scene_id'],
                'split': record['split'], 'stress_type': record.get('stress_type'),
                'windows_mode': 'released_complete_windows_read_only',
                'windows_sha256': file_hash(windows),
                'manifest_sha256': file_hash(root / 'manifest.json')}
    metadata['data_prepare_s'] = time.perf_counter() - started
    return pb, metadata


def _command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None



def _read_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def _cpu_set_count(value):
    if not value:
        return None
    try:
        cpus = set()
        for part in value.split(','):
            ends = [int(x) for x in part.split('-')]
            if len(ends) == 1:
                ends *= 2
            if len(ends) != 2 or ends[0] < 0 or ends[1] < ends[0] or ends[1] > 1000000:
                return None
            cpus.update(range(ends[0], ends[1] + 1))
        return len(cpus) or None
    except ValueError:
        return None


def _linux_resources(proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    """Record host capacity separately from observable container/affinity limits.

    Read both unified and legacy cgroups, including visible parent limits.
    Missing controls mean unknown, never proof of unlimited rented resources.
    Optional roots make v1/v2 parsing verifiable without modifying the host.
    """
    proc_root, cgroup_root = Path(proc_root), Path(cgroup_root)
    cpuinfo = _read_text(proc_root / 'cpuinfo') or ''
    model = next((line.split(':', 1)[1].strip() for line in cpuinfo.splitlines()
                  if ':' in line and line.split(':', 1)[0].strip() in ('model name', 'Hardware')), None)
    meminfo = _read_text(proc_root / 'meminfo') or ''
    physical = next((int(line.split()[1]) * 1024 for line in meminfo.splitlines()
                     if line.startswith('MemTotal:')), None)
    candidates = {cgroup_root, cgroup_root / 'cpu', cgroup_root / 'cpu,cpuacct', cgroup_root / 'memory', cgroup_root / 'cpuset'}
    membership = _read_text(proc_root / 'self' / 'cgroup')
    for line in (membership or '').splitlines():
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, member = parts
        relative = Path(member.lstrip('/'))
        if '..' in relative.parts:
            continue
        bases = [cgroup_root] if not controllers else [cgroup_root / c for c in controllers.split(',')] + [cgroup_root / controllers]
        for base in bases:
            current = base / relative
            while current.is_relative_to(base):
                candidates.add(current)
                if current == base:
                    break
                current = current.parent
    controls, quotas, memory_limits, cpuset_counts = {}, [], [], []
    for folder in sorted(candidates):
        values = {}
        for name in ('cpu.max', 'cpu.cfs_quota_us', 'cpu.cfs_period_us', 'memory.max',
                     'memory.limit_in_bytes', 'cpuset.cpus.effective', 'cpuset.cpus'):
            value = _read_text(folder / name)
            if value is not None:
                values[name] = value
        if not values:
            continue
        controls[str(folder.relative_to(cgroup_root))] = values
        try:
            if 'cpu.max' in values:
                quota, period = values['cpu.max'].split()
                if quota != 'max' and float(quota) > 0 and float(period) > 0:
                    quotas.append(float(quota) / float(period))
            if 'cpu.cfs_quota_us' in values and 'cpu.cfs_period_us' in values:
                quota, period = float(values['cpu.cfs_quota_us']), float(values['cpu.cfs_period_us'])
                if quota > 0 and period > 0:
                    quotas.append(quota / period)
            for name in ('memory.max', 'memory.limit_in_bytes'):
                if name in values and values[name] != 'max':
                    limit = int(values[name])
                    if 0 < limit < 2 ** 60:  # v1 represents unlimited with a huge sentinel
                        memory_limits.append(limit)
        except (ValueError, OverflowError):
            # Raw controls remain available when a kernel/control format is unknown.
            pass
        cpuset = _cpu_set_count(values.get('cpuset.cpus.effective') or values.get('cpuset.cpus'))
        if cpuset is not None:
            cpuset_counts.append(cpuset)
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    quota = min(quotas) if quotas else None
    memory_limit = min(memory_limits) if memory_limits else None
    cpu_limits = [v for v in (os.cpu_count(), affinity, quota, min(cpuset_counts) if cpuset_counts else None) if v is not None]
    memory_bounds = [v for v in (physical, memory_limit) if v is not None]
    return {'cpu_model': model, 'host_memory_bytes': physical, 'affinity_cpu_count': affinity,
            'cgroup_cpu_quota_cores': quota, 'cgroup_memory_limit_bytes': memory_limit,
            'cgroup_cpuset_cpu_count': min(cpuset_counts) if cpuset_counts else None,
            'effective_cpu_limit_cores': min(cpu_limits) if cpu_limits else None,
            'effective_memory_limit_bytes': min(memory_bounds) if memory_bounds else None,
            'cgroup_controls': controls, 'cgroup_controls_observed': bool(controls),
            'interpretation': 'observed host/affinity/cgroup upper limits; not a billing or dedicated-capacity guarantee'}

def environment(instance_id=None):
    versions = {}
    for name in ('numpy', 'numba', 'torch', 'skyfield', 'sgp4', 'ortools'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    system = platform.system()
    mem, resources = None, None
    if system == 'Linux':
        resources = _linux_resources()
        mem = resources['host_memory_bytes']
    elif system == 'Darwin':
        value = _command(['sysctl', '-n', 'hw.memsize'])
        mem = int(value) if value else None
    cuda_runtime, cuda_available, cuda_error = None, None, None
    if versions.get('torch') is not None:
        try:
            import torch
            cuda_runtime = torch.version.cuda
            cuda_available = torch.cuda.is_available()
        except (ImportError, OSError, RuntimeError) as exc:
            cuda_error = type(exc).__name__
    result = {'system': system, 'machine': platform.machine(), 'python': sys.version.split()[0],
              'cpu': (resources['cpu_model'] if resources else None) or platform.processor(),
              'cpu_count': os.cpu_count(), 'ram_bytes': mem,
              'ram_bytes_scope': 'host physical memory; container limits recorded separately',
              'linux_resources': resources, 'cuda_runtime': cuda_runtime,
              'cuda_available': cuda_available, 'cuda_probe_error': cuda_error,
              'instance_id': instance_id, 'versions': versions,
              'threads': {k: os.environ.get(k) for k in
                          ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')},
              'gpu_inventory': _command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                         '--format=csv,noheader']),
              'linux_gpu_telemetry': _command(['nvidia-smi',
                    '--query-gpu=utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu',
                    '--format=csv,noheader']) if system == 'Linux' else None}
    # Telemetry is measured evidence, not a stable hardware grouping key.
    stable = {k: v for k, v in result.items() if k != 'linux_gpu_telemetry'}
    result['environment_hash'] = canonical_hash(stable)
    return result


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def source_hash():
    root = Path(__file__).resolve().parents[2]
    rows = {str(p.relative_to(root)): file_hash(p) for folder in ('obsplan', 'benchmarks')
            for p in sorted((root/folder).rglob('*.py'))}
    return canonical_hash(rows)

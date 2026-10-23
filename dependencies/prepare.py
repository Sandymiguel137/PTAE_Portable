"""Materialize ground truth, fixed observations, grouped splits and AC context."""
from pathlib import Path
from dataclasses import dataclass, asdict, field
import math
import shutil
import numpy as np
import scipy.sparse as sp
from .common import digest, write_json, read_json, stable_seed, Moments


@dataclass
class PreparationConfig:
    voltage_dir: str
    output_dir: str
    node_order_path: str
    system_dir: str
    voltage_layout: str = 'nodes_time'
    voltage_unit: str = 'pu'
    measurement_mode: str = 'fixed'  # fixed Y or per-trajectory CSC data[T,nnz]
    physics_dir: str = ''
    power_base_va: float = 1_000_000.
    sensor_pool: str = 'bundled'  # bundled, all_nodes, custom
    sensor_indices_path: str = ''
    sensor_percentage: float = 100.  # percentage of the selected pool, not all nodes
    seed: int = 42
    group_file: str = ''  # JSON {"V1.npy": "group1", ...}
    split_file: str = ''  # optional train/val/cal/test filename lists
    split_fractions: dict = field(default_factory=lambda: dict(train=.70, val=.10, cal=.10, test=.10))
    noise_std_fraction: tuple = (.01, .01, .01)  # fraction of clean TRAIN standard deviation
    noise_absolute_std: tuple = (0., 0., 0.)  # pu; added in quadrature
    fdi_trajectory_probability: float = 1.
    fdi_sensor_percentage: float = 20.
    fdi_duration_fraction: tuple = (.20, .50)
    fdi_strength_std: tuple = (2., 5.)
    fdi_channels: tuple = ('Vm', 'P', 'Q')
    fdi_kind: str = 'bias'  # bias or ramp; additive measurement tampering
    dropout_probability: float = 0.
    chunk_size: int = 256


def load_y(path):
    # Original bundle uses CSC data/indices/indptr/shape, sometimes without scipy format key.
    with np.load(path, allow_pickle=False) as z:
        if 'format' in z:
            y = sp.load_npz(path)
        else:
            y = sp.csc_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
    y = y.astype(np.complex128).tocsc()
    if y.shape[0] != y.shape[1] or not np.isfinite(y.data).all():
        raise ValueError('Ybus must be a finite square matrix')
    return y


def measure(voltage, sensors, y=None, values=None, rows=None, cols=None):
    """voltage[T,N]; P+jQ = V * conj(YV), positive network injection, in pu."""
    v = np.asarray(voltage, np.complex128)
    if values is None:
        current = (y[sensors, :] @ v.T).T
    else:
        current = np.zeros((len(v), len(sensors)), np.complex128)
        for t in range(len(v)):
            np.add.at(current[t], rows, values[t] * v[t, cols])
    power = v[:, sensors] * current.conj()
    return np.stack((abs(v[:, sensors]), power.real, power.imag), axis=-1)


def make_splits(files, groups, cfg):
    names = [p.name for p in files]
    if cfg.split_file:
        result = read_json(cfg.split_file)
    else:
        fractions = cfg.split_fractions
        parts = ['train', 'val', 'cal', 'test']
        if set(fractions) != set(parts) or any(v <= 0 for v in fractions.values()) or not np.isclose(sum(fractions.values()), 1):
            raise ValueError('Four positive split fractions must sum to 1')
        unique = sorted(set(groups.values()))
        if len(unique) < 4:
            raise ValueError('At least four independent groups/trajectories are needed for train/val/cal/test')
        np.random.default_rng(cfg.seed).shuffle(unique)
        target = np.array([fractions[p] * len(unique) for p in parts])
        counts = np.ones(4, dtype=int)
        for _ in range(len(unique) - 4):
            counts[np.argmax(target - counts)] += 1
        result = {}; offset = 0
        for part, count in zip(parts, counts):
            selected = set(unique[offset:offset + count]); offset += count
            result[part] = [n for n in names if groups[n] in selected]
    if set(result) != {'train', 'val', 'cal', 'test'} or any(not v for v in result.values()):
        raise ValueError('Every split must be nonempty')
    assigned = [n for values in result.values() for n in values]
    if len(set(assigned)) != len(assigned) or set(assigned) != set(names):
        raise ValueError('Split lists must assign every voltage file exactly once')
    owner = {}
    for part, values in result.items():
        for name in values:
            group = groups[name]
            if group in owner and owner[group] != part:
                raise ValueError(f'Related group {group} crosses partitions')
            owner[group] = part
    return result


def prepare_dataset(cfg):
    """New output directory only; publish manifest last so partial runs cannot train."""
    src, out, system = (Path(p).expanduser().resolve() for p in (cfg.voltage_dir, cfg.output_dir, cfg.system_dir))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new empty output directory: {out}')
    if out == src or out in src.parents:
        raise ValueError('Output cannot replace or contain the source directory')
    if cfg.voltage_layout not in ('nodes_time', 'time_nodes') or cfg.voltage_unit not in ('pu', 'volts'):
        raise ValueError('Choose explicit voltage layout and unit')
    if cfg.measurement_mode not in ('fixed', 'native'):
        raise ValueError('measurement_mode must be fixed or native')
    if cfg.measurement_mode == 'native' and not cfg.physics_dir:
        raise ValueError('native mode requires physics_dir with Y<number>.npz')
    if not 0 < cfg.sensor_percentage <= 100 or not 0 <= cfg.fdi_sensor_percentage <= 100:
        raise ValueError('Invalid sensor percentage')
    for v in [cfg.fdi_trajectory_probability, cfg.dropout_probability]:
        if not 0 <= v <= 1:
            raise ValueError('Probabilities must be between zero and one')
    if not 0 < cfg.fdi_duration_fraction[0] <= cfg.fdi_duration_fraction[1] <= 1:
        raise ValueError('Invalid FDI duration range')
    if not 0 < cfg.fdi_strength_std[0] <= cfg.fdi_strength_std[1]:
        raise ValueError('FDI strength must be positive')
    if cfg.fdi_kind not in ('bias', 'ramp') or not cfg.fdi_channels or not set(cfg.fdi_channels) <= {'Vm', 'P', 'Q'}:
        raise ValueError('FDI kind/channels are invalid')
    if cfg.chunk_size < 1 or not np.isfinite(cfg.power_base_va) or cfg.power_base_va <= 0:
        raise ValueError('Positive chunk size and power base required')
    ns = np.array(cfg.noise_std_fraction); na = np.array(cfg.noise_absolute_std)
    if ns.shape != (3,) or na.shape != (3,) or not np.isfinite(ns).all() or not np.isfinite(na).all() or np.any(ns < 0) or np.any(na < 0):
        raise ValueError('Noise standard deviations need three nonnegative finite values')
    nodes = read_json(system / 'ynodeorder.json')
    if len(set(nodes)) != len(nodes) or read_json(cfg.node_order_path) != nodes:
        raise ValueError('Dataset node order must exactly match the supplied system node order')
    n = len(nodes); y = load_y(system / 'ybus_pu.npz')
    if y.shape != (n, n):
        raise ValueError('Ybus shape differs from node order')
    if cfg.sensor_pool == 'bundled':
        pool = np.load(system / 'sensor_indices_0based.npy', allow_pickle=False)
    elif cfg.sensor_pool == 'all_nodes':
        pool = np.arange(n)
    elif cfg.sensor_pool == 'custom':
        pool = np.load(cfg.sensor_indices_path, allow_pickle=False)
    else:
        raise ValueError('sensor_pool must be bundled, all_nodes, or custom')
    if pool.ndim != 1 or not np.issubdtype(pool.dtype, np.integer) or not len(pool) or len(set(pool)) != len(pool) or pool.min() < 0 or pool.max() >= n:
        raise ValueError('Sensor pool must contain unique zero-based node indices')
    count = max(1, math.ceil(len(pool) * cfg.sensor_percentage / 100))
    sensors = np.sort(np.random.default_rng(cfg.seed).choice(pool, count, replace=False))
    files = sorted([p for p in src.glob('V*.npy') if p.stem[1:].isdigit()], key=lambda p: int(p.stem[1:]))
    if not files:
        raise FileNotFoundError('Expected complex V<number>.npy trajectories')
    groups = {}; grouping = 'one_file_per_group'
    if cfg.group_file:
        groups = {k: str(v) for k, v in read_json(cfg.group_file).items()}; grouping = 'explicit_group_file'
    else:
        for p in files:
            meta = read_json(p.with_suffix('.json')) if p.with_suffix('.json').exists() else {}
            groups[p.name] = str(meta.get('group', p.name))
        if any(groups[p.name] != p.name for p in files):
            grouping = 'trajectory_metadata_group'
    if set(groups) != {p.name for p in files}:
        raise ValueError('Group mapping must cover exactly all voltage filenames')
    splits = make_splits(files, groups, cfg)
    owner = {name: part for part, names in splits.items() for name in names}
    base = np.load(system / 'node_voltage_base_volts.npy') if cfg.voltage_unit == 'volts' else None
    if base is not None and (base.shape != (n,) or not np.isfinite(base).all() or np.any(base <= 0)):
        raise ValueError('Converting volts requires positive per-node voltage bases')
    out.mkdir(parents=True, exist_ok=True)
    for name in ['truth', 'clean', 'physics', 'system', 'noisy', 'fdi', 'labels', 'availability']:
        (out / name).mkdir()
    shutil.copy2(system / 'ybus_pu.npz', out / 'system/ybus_pu.npz')
    write_json(out / 'system/ynodeorder.json', nodes)
    np.save(out / 'system/sensor_indices_0based.npy', sensors)
    if (system / 'node_voltage_base_volts.npy').exists():
        shutil.copy2(system / 'node_voltage_base_volts.npy', out / 'system/node_voltage_base_volts.npy')
    moments = {key: Moments() for key in ('vr', 'vi', 'm')}
    records = {}; seen = {}
    for index, path in enumerate(files, 1):
        h = digest(path)
        if h in seen and owner[seen[h]] != owner[path.name]:
            raise ValueError('Identical voltage trajectories cross partitions; group duplicates together')
        seen[h] = path.name
        original = np.load(path, mmap_mode='r', allow_pickle=False)
        v = original.T if cfg.voltage_layout == 'nodes_time' else original
        if v.ndim != 2 or v.shape[1] != n or len(v) == 0 or not np.iscomplexobj(v):
            raise ValueError(f'{path}: expected complex voltage with {n} nodes')
        t_count = len(v)
        truth = np.lib.format.open_memmap(out / 'truth' / path.name, mode='w+', dtype=np.complex128, shape=v.shape)
        clean = np.lib.format.open_memmap(out / 'clean' / path.name, mode='w+', dtype=np.float64, shape=(t_count, count, 3))
        native = None; rows = cols = None
        if cfg.measurement_mode == 'native':
            yfile = Path(cfg.physics_dir).expanduser() / ('Y' + path.stem[1:] + '.npz')
            with np.load(yfile, allow_pickle=False) as z:
                data, ind, ptr, shape = z['data'], z['indices'], z['indptr'], tuple(z['shape'])
            if shape != (n, n) or data.shape != (t_count, len(ind)) or not np.isfinite(data).all():
                raise ValueError(f'Invalid per-snapshot CSC matrix: {yfile}')
            marker = sp.csc_matrix((np.arange(1, len(ind)+1), ind, ptr), shape=shape)[sensors, :].tocoo()
            rows, cols = marker.row, marker.col
            np.savez(out / 'physics' / (path.stem + '_pattern.npz'), rows=rows, cols=cols)
            native = np.ascontiguousarray(data[:, marker.data-1])
            np.save(out / 'physics' / path.name, native)
            del data
        for start in range(0, t_count, cfg.chunk_size):
            stop = min(start + cfg.chunk_size, t_count)
            block = np.array(v[start:stop], np.complex128, copy=True)
            if base is not None:
                block /= base
            if not np.isfinite(block).all():
                raise ValueError(f'Nonfinite ground truth: {path}')
            truth[start:stop] = block
            m = measure(block, sensors, y=y, values=None if native is None else native[start:stop], rows=rows, cols=cols)
            if not np.isfinite(m).all():
                raise ValueError('Nonfinite AC measurements')
            clean[start:stop] = m
            if owner[path.name] == 'train':
                moments['vr'].add(block.real); moments['vi'].add(block.imag); moments['m'].add(m)
        truth.flush(); clean.flush(); del truth, clean
        records[path.name] = dict(samples=t_count, group=groups[path.name], source_sha256=h)
        print(f'AC measurements {index}/{len(files)}: {path.name} ({t_count} samples)', flush=True)
    vr, vrs = moments['vr'].result(1e-3); vi, vis = moments['vi'].result(1e-3); am, ast = moments['m'].result(1e-6)
    np.savez(out / 'normalization.npz', vr_mean=vr, vr_std=vrs, vi_mean=vi, vi_std=vis, ami_mean=am, ami_std=ast)
    noise_sigma = np.hypot(ast * ns, na)
    channels = [['Vm', 'P', 'Q'].index(c) for c in cfg.fdi_channels]
    for path in files:
        clean = np.load(out / 'clean' / path.name, mmap_mode='r'); t_count = len(clean)
        rng = np.random.default_rng(stable_seed(cfg.seed, path.name + ':noise'))
        noisy = np.array(clean) + rng.normal(size=clean.shape) * noise_sigma
        np.save(out / 'noisy' / path.name, noisy)
        attacked = noisy.copy(); labels = np.zeros(clean.shape, dtype=np.uint8)
        rng = np.random.default_rng(stable_seed(cfg.seed, path.name + ':fdi'))
        recipe = {'active': False}
        if cfg.fdi_sensor_percentage > 0 and rng.random() < cfg.fdi_trajectory_probability:
            k = min(count, max(1, math.ceil(count * cfg.fdi_sensor_percentage / 100)))
            selected = np.sort(rng.choice(count, k, replace=False))
            duration = min(t_count, max(1, round(t_count * rng.uniform(*cfg.fdi_duration_fraction))))
            start = int(rng.integers(0, t_count-duration+1)); stop = start+duration
            amplitude = rng.uniform(*cfg.fdi_strength_std, size=(k, len(channels))) * rng.choice([-1., 1.], size=(k, len(channels))) * ast[np.ix_(selected, channels)]
            envelope = np.ones(duration) if cfg.fdi_kind == 'bias' else np.linspace(1/duration, 1, duration)
            ix = np.ix_(np.arange(start, stop), selected, channels)
            attacked[ix] += envelope[:, None, None] * amplitude[None]
            labels[ix] = 1
            recipe = dict(active=True, kind=cfg.fdi_kind, start=start, stop_exclusive=stop,
                          sensor_positions=selected.tolist(), node_indices=sensors[selected].tolist(), channels=list(cfg.fdi_channels), amplitude_pu=amplitude.tolist())
        np.save(out / 'fdi' / path.name, attacked)
        np.save(out / 'labels' / path.name, labels)
        rng = np.random.default_rng(stable_seed(cfg.seed, path.name + ':availability'))
        available = (rng.random(clean.shape[:2]) >= cfg.dropout_probability).astype(np.uint8)
        np.save(out / 'availability' / path.name, available)
        write_json(out / 'labels' / (path.stem + '.json'), recipe)
    write_json(out / 'splits.json', splits)
    write_json(out / 'preparation_config.json', asdict(cfg))
    hashes = {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    manifest = dict(format='ptae-portable-prepared-v1', complete=True, nodes=n, sensors=count,
        measurement_mode=cfg.measurement_mode, power_base_va=cfg.power_base_va,
        voltage_unit='pu', power_unit='pu', channel_order=['Vm', 'P', 'Q'], truth_layout='time_nodes', measurement_layout='time_sensors_channels',
        physics_definition='I=YV; S=V*conj(I). P/Q follow the supplied Y (including any Norton shunts). Not individual PV/load terminal powers.',
        sensor_selection=dict(pool=cfg.sensor_pool, pool_size=len(pool), percentage_of_pool=cfg.sensor_percentage, actual_percentage_of_all_nodes=100*count/n),
        grouping=grouping, splits=splits, records=records, hashes=hashes,
        normalization_partition='train', noise_order='clean -> Gaussian noise -> additive FDI; availability is an independent mask',
        label_meaning='Injected measurement FDI only. Physical imbalance/oscillation already in V remain ground truth.',
        independence_note='Grouped holdout, not proof of independence across dates. Supply groups for shared days/scenarios/profile windows.')
    write_json(out / 'manifest.json', manifest)
    print('Prepared:', out, '\nPartitions:', {k: len(v) for k,v in splits.items()}, flush=True)
    return manifest

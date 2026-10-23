"""Freeze completed native runs, split first, purge shared states, then add noise/FDI.

The accepted subset has its own manifest. The incomplete 400-run source is never
marked complete or modified. No legacy prepared datasets are overwritten.
"""
import argparse
from collections import Counter
import csv
import hashlib
import os
from pathlib import Path
import shutil

import numpy as np

from dependencies.common import Moments, digest, read_json, write_json, stable_seed
from dependencies.corruptions import allocate, assign_families, corrupt
from dependencies.prepare import load_y, measure

PARTS = ('train', 'val', 'cal', 'test')
FRACTIONS = np.array([.70, .10, .10, .10])


def stratified_split(metadata, seed):
    """Exact global quotas, balanced physical categories, one source run per group."""
    names = sorted(metadata, key=lambda name: int(Path(name).stem[1:]))
    categories = sorted({m['attack'] for m in metadata.values()})
    targets = allocate(len(names), FRACTIONS)
    # Give the largest remainder to test when the three holdout remainders tie.
    if len(names) == 393:
        targets = np.array([275, 39, 39, 40])
    sizes = np.array([sum(metadata[n]['attack'] == c for n in names) for c in categories])
    ideal = sizes[:, None]*FRACTIONS
    counts = np.floor(ideal).astype(int)
    rows = sizes-counts.sum(axis=1); columns = targets-counts.sum(axis=0)
    while rows.sum():
        score = ideal-counts
        score[(rows == 0), :] = -np.inf; score[:, columns == 0] = -np.inf
        a, b = np.unravel_index(np.argmax(score), score.shape)
        counts[a, b] += 1; rows[a] -= 1; columns[b] -= 1
    result = {p: [] for p in PARTS}
    for i, category in enumerate(categories):
        group = [n for n in names if metadata[n]['attack'] == category]
        np.random.default_rng(stable_seed(seed, category+':split')).shuffle(group)
        offset = 0
        for part, count in zip(PARTS, counts[i]):
            result[part].extend(group[offset:offset+count]); offset += count
    for part in PARTS:
        result[part].sort(key=lambda n: int(Path(n).stem[1:]))
    return result


def unique_indices(files, splits, decimals=10):
    """Keep each quantized all-node complex state once, train -> val -> cal -> test.

    A 1e-10 pu rounding grid also catches numerical copies. This is an explicit
    duplicate-state exclusion, not a claim of independence of operating profiles.
    """
    seen = set(); result = {}; cross = Counter(); within = Counter()
    owner = {}
    for part in PARTS:
        for name in splits[part]:
            values = np.load(files[name], mmap_mode='r')
            selected = []
            for start in range(0, len(values), 128):
                z = values[start:start+128]
                rounded = np.round(np.stack([z.real, z.imag], -1), decimals=decimals)
                rounded[rounded == 0] = 0.  # Canonicalize signed zero.
                for j, row in enumerate(rounded):
                    key = hashlib.sha256(row.tobytes()).digest()
                    if key in seen:
                        (within if owner[key] == part else cross)[part] += 1
                    else:
                        seen.add(key); owner[key] = part; selected.append(start+j)
            if not selected:
                raise ValueError(f'{name} has no unique samples; group equivalent scenarios before splitting')
            result[name] = np.array(selected, dtype=np.int32)
        print('Shared-state audit:', part, sum(len(result[n]) for n in splits[part]), 'eligible samples', flush=True)
    return result, dict(rounding_decimals=decimals, unique_states=len(seen),
                        removed_across_partitions=dict(cross), removed_within_partitions=dict(within),
                        retained_cross_partition_duplicates=0, ordering=list(PARTS))


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare(source, output, seed=20261020, expected_count=393, *, source_batch, phase_mapping):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {output}')
    if output == source or output in source.parents or source in output.parents:
        raise ValueError('Keep prepared output separate from source')
    raw = read_json(source/'manifest.json')
    if raw.get('format') != 'pymagic-sensor-npy-v2' or raw.get('measurement_source') != 'AC_YV_from_simulation_network':
        raise ValueError('Expected the native AC phasor export')
    metadata = {}; files = {}; source_hashes = {}; accepted = {}
    batch = Path(source_batch).expanduser().resolve()
    for name, record in raw['records'].items():
        done = read_json(batch/record['source_id']/'complete.json')
        if done['status'] != 'complete' or done['convergence_failures'] != 0:
            raise ValueError(f'Native run is not validated: {name}')
        path = source/'voltages'/name
        sha = digest(path)
        if sha != record['voltage_sha256'] or sha != done['dataset_sha256']:
            raise ValueError(f'Voltage hash mismatch: {name}')
        if record['network_y_sha256'] != digest(source/'system/ybus_pu.npz'):
            raise ValueError('Reference physics differs from source run')
        files[name] = path; metadata[name] = read_json(path.with_suffix('.json'))
        source_hashes[name] = sha; accepted[name] = record
    if len(files) != expected_count:
        raise ValueError(f'Expected {expected_count} completed runs, found {len(files)}')
    if len({m['id'] for m in metadata.values()}) != len(files):
        raise ValueError('Repeated source run')
    if len(set(source_hashes.values())) != len(files):
        raise ValueError('Duplicate full trajectories must be grouped before splitting')
    output.mkdir(parents=True)
    for folder in ['system', 'truth', 'clean', 'noisy', 'fdi', 'labels', 'availability', 'physics',
                   'sample_indices', 'metadata']:
        (output/folder).mkdir()
    splits = stratified_split(metadata, seed)
    write_json(output/'splits.json', splits)  # Frozen before any learned statistics or corruptions.
    write_json(output/'accepted_source.json', dict(source_manifest_sha256=digest(source/'manifest.json'),
               expected_count=expected_count, rejected_source_ids=sorted(set(
                   r['id'] for r in read_json(batch/'plan.json')['trajectories'])-
                   {r['source_id'] for r in accepted.values()}), records=accepted))
    indices, leakage = unique_indices(files, splits)
    write_json(output/'leakage_audit.json', leakage)
    for name, ix in indices.items():
        np.save(output/'sample_indices'/name, ix)
    for name in ['ynodeorder.json', 'sensor_indices_0based.npy', 'node_voltage_base_volts.npy',
                 'ybus_pu.npz', 'network_physics.json']:
        shutil.copy2(source/'system'/name, output/'system'/name)
    nodes = read_json(output/'system/ynodeorder.json')
    sensors = np.load(output/'system/sensor_indices_0based.npy')
    sensor_nodes = [nodes[int(i)] for i in sensors]
    y = load_y(output/'system/ybus_pu.npz')
    phase_path = Path(phase_mapping).expanduser().resolve()
    with phase_path.open() as handle:
        phase_by_bus = {r['bus'].lower(): r['primary_phase'].lower() for r in csv.DictReader(handle)
                        if r['primary_phase'].lower() in ('a', 'b', 'c')}
    write_json(output/'system/primary_phase_by_bus.json', phase_by_bus)
    moments = {key: Moments() for key in ('vr', 'vi', 'm')}
    power_ss = np.zeros((len(sensors), 2)); power_count = 0
    for name in splits['train']:
        v = np.load(files[name], mmap_mode='r')[indices[name]]
        clean = np.load(source/'sensor_clean'/name, mmap_mode='r')[indices[name]]
        moments['vr'].add(v.real); moments['vi'].add(v.imag); moments['m'].add(clean)
        power_ss += (clean[:, :, 1:]**2).sum(axis=0); power_count += len(clean)
    vr, vrs = moments['vr'].result(1e-3); vi, vis = moments['vi'].result(1e-3)
    am, ast = moments['m'].result(1e-6)
    np.savez(output/'normalization.npz', vr_mean=vr, vr_std=vrs, vi_mean=vi, vi_std=vis, ami_mean=am, ami_std=ast)
    power_rms = np.sqrt(power_ss/power_count)
    sigma = np.column_stack([np.full(len(sensors), .001), np.maximum(.01*power_rms, 1e-7)])
    np.save(output/'system/measurement_noise_sigma.npy', sigma)
    families = assign_families(splits, seed)
    owner = {n: p for p, names in splits.items() for n in names}
    records = {}; standardized_ss = np.zeros(3); standardized_sum = np.zeros(3); noise_count = 0
    for index, name in enumerate(sorted(files, key=lambda n: int(Path(n).stem[1:]))):
        v = np.load(files[name], mmap_mode='r')
        clean = np.load(source/'sensor_clean'/name, mmap_mode='r')
        if v.shape != (1200, len(nodes)) or clean.shape != (1200, len(sensors), 3):
            raise ValueError(f'Invalid native array shape: {name}')
        if not np.isfinite(v).all() or not np.isfinite(clean).all():
            raise ValueError(f'Nonfinite native data: {name}')
        # Check the full trajectory in chunks, not just its clean prefix.
        max_error = 0.
        for start in range(0, len(v), 128):
            max_error = max(max_error, float(np.max(abs(measure(v[start:start+128], sensors, y=y)-clean[start:start+128]))))
        if max_error > 1e-10:
            raise ValueError(f'Saved observations do not match the training AC decoder: {name}: {max_error}')
        link_or_copy(files[name], output/'truth'/name)
        link_or_copy(source/'sensor_clean'/name, output/'clean'/name)
        rng = np.random.default_rng(stable_seed(seed, name+':noise'))
        residual = rng.normal(size=clean.shape)
        noisy = clean+residual*sigma[None]
        standardized_sum += residual.sum(axis=(0, 1)); standardized_ss += (residual**2).sum(axis=(0, 1))
        noise_count += residual.shape[0]*residual.shape[1]
        attacked, labels, recipe = corrupt(noisy, families[name], sensor_nodes, phase_by_bus,
                                           power_rms, indices[name], seed, name)
        if not np.isfinite(attacked).all() or not np.array_equal(attacked[labels == 0], noisy[labels == 0]):
            raise ValueError('Unlabeled or nonfinite corruption')
        np.save(output/'noisy'/name, noisy); np.save(output/'fdi'/name, attacked)
        np.save(output/'labels'/name, labels)
        np.save(output/'availability'/name, np.ones(clean.shape[:2], np.uint8))
        recipe['node_indices'] = sensors[recipe.get('sensor_positions', [])].tolist()
        write_json(output/'labels'/Path(name).with_suffix('.json'), recipe)
        meta = metadata[name]
        write_json(output/'metadata'/Path(name).with_suffix('.json'), dict(
            voltage_sha256=source_hashes[name], sample_count=len(v), sampling_seconds=1,
            start_timestamp='2018-06-19T15:50:00', category=meta['attack'], events=meta['events']))
        mode = np.zeros(len(v), np.uint8)
        for event in meta['events']:
            mode[event['start']:event['end_exclusive']] |= 1 if event['kind'] == 'oscillation' else 2
        varying = int(np.argmax(np.ptp(clean[:, :, 0], axis=0)))
        strength = np.max(abs(attacked[:, :, 0]-noisy[:, :, 0]), axis=0)
        target = int(np.argmax(strength))
        suggestions = dict(varying_node=sensor_nodes[varying], truth_span_pu=float(np.ptp(clean[:, varying, 0])),
                           fdi_node=sensor_nodes[target] if strength[target] else None,
                           fdi_peak_pu=float(strength[target]), fdi_samples=int(np.count_nonzero(labels[:, target, 0])))
        np.save(output/'metadata'/name, mode)
        records[name] = dict(samples=len(v), eligible_samples=len(indices[name]), group=meta['id'],
            source_sha256=source_hashes[name], physical_category=meta['attack'], fdi_kind=families[name],
            decoder_max_absolute_error_pu=max_error, suggested_nodes=suggestions,
            observed_clean_sha256=digest(source/'sensor_clean'/name))
        if index % 25 == 0:
            print(f'Prepared {index+1}/{len(files)} trajectories', flush=True)
    noise_mean = standardized_sum/noise_count
    noise_std = np.sqrt(standardized_ss/noise_count-noise_mean**2)
    if np.max(abs(noise_mean)) > .01 or np.max(abs(noise_std-1)) > .01:
        raise ValueError('Gaussian residual verification failed')
    report = dict(complete=True, trajectories=len(files), splits={p: len(n) for p, n in splits.items()},
        physical_categories={p: dict(Counter(metadata[n]['attack'] for n in names)) for p, names in splits.items()},
        attack_families={p: dict(Counter(families[n] for n in names)) for p, names in splits.items()},
        eligible_samples={p: sum(len(indices[n]) for n in names) for p, names in splits.items()},
        gaussian_standardized_mean=noise_mean.tolist(), gaussian_standardized_std=noise_std.tolist(),
        decoder_max_absolute_error_pu=max(r['decoder_max_absolute_error_pu'] for r in records.values()),
        leakage_audit=leakage, source_modified=False)
    write_json(output/'validation_report.json', report)
    write_json(output/'preparation_config.json', dict(source_dir=str(source), output_dir=str(output), seed=seed,
        expected_count=expected_count, voltage_dir=str(output/'metadata'), split_first=True,
        noise=dict(voltage_sigma_pu=.001, power_train_rms_fraction=.01, power_floor_pu=1e-7),
        fdi=dict(families='20% none, 30% bias, 30% ramp, 10% delay, 5% fake oscillation, 5% fake imbalance',
                 sensor_percentage=[5, 10], duration_seconds=[120, 240], voltage_peak_pu=[.001, .003],
                 power_train_rms_fraction=[.01, .03], delay_seconds=[1, 5], oscillation_period_seconds=[12, 40])))
    print('Hashing prepared artifacts...', flush=True)
    hashes = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = dict(format='ptae-portable-prepared-v1', complete=True, nodes=len(nodes), sensors=len(sensors),
        measurement_mode='fixed', power_base_va=1_000_000., voltage_unit='pu', power_unit='pu',
        channel_order=['Vm', 'P', 'Q'], truth_layout='time_nodes', measurement_layout='time_sensors_channels',
        measurement_source=raw['measurement_source'], physics_definition=raw['physics_definition'],
        sensor_selection=dict(pool='bundled', pool_size=len(sensors), percentage_of_pool=100,
                              actual_percentage_of_all_nodes=100*len(sensors)/len(nodes)),
        grouping='source trajectory, stratified by physical category', sample_filter='sample_indices/*.npy',
        splits=splits, records=records, hashes=hashes, normalization_partition='train',
        noise_order='Frozen splits -> shared-state exclusion -> train-only statistics -> Gaussian noise -> FDI',
        label_meaning='Measurement corruption only; real physical oscillation/imbalance is retained in voltage truth',
        independence_note='Same feeder and shared load/solar window. No claim of unseen-date generalization.')
    write_json(output/'manifest.json', manifest)
    print(report, flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Native AC phasor export directory')
    parser.add_argument('--source-batch', required=True, help='Native batch with plan.json and run completion records')
    parser.add_argument('--phase-mapping', required=True, help='Verified pv_mapping.csv with bus and primary_phase columns')
    parser.add_argument('--output', required=True, help='Fresh prepared output directory')
    parser.add_argument('--seed', type=int, default=20261020)
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed, source_batch=args.source_batch, phase_mapping=args.phase_mapping)

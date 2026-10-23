"""Reproducible measurement corruptions; voltage truth is never an input to attacks."""
import numpy as np

from .common import stable_seed


FAMILIES = ('none', 'bias', 'ramp', 'delay', 'fake_oscillation', 'fake_imbalance')
WEIGHTS = np.array([.20, .30, .30, .10, .05, .05])


def allocate(total, weights):
    target = total * np.asarray(weights, float)
    counts = np.floor(target).astype(int)
    for i in np.argsort(-(target-counts), kind='stable')[:total-int(counts.sum())]:
        counts[i] += 1
    return counts


def assign_families(splits, seed):
    result = {}
    for partition, names in splits.items():
        kinds = np.repeat(FAMILIES, allocate(len(names), WEIGHTS))
        np.random.default_rng(stable_seed(seed, partition+':families')).shuffle(kinds)
        result.update(zip(names, kinds.tolist()))
    return result


def corrupt(noisy, family, sensor_nodes, phase_by_bus, power_rms, eligible, seed, name):
    """Tamper with 5–10% of sensors for 120–240 s. Delay replays noisy packets.

    Fake imbalance uses verified primary phases from pv_mapping.csv. Secondary
    conductor suffixes are deliberately not interpreted as primary phase labels.
    """
    if family not in FAMILIES:
        raise ValueError('Unknown measurement attack family')
    attacked = np.array(noisy, copy=True)
    labels = np.zeros(noisy.shape, np.uint8)
    recipe = dict(active=False, kind=family, channels=[], events=[])
    if family == 'none':
        return attacked, labels, recipe
    rng = np.random.default_rng(stable_seed(seed, name+':fdi'))
    steps, count, _ = noisy.shape
    duration = min(steps-1, int(rng.integers(120, 241)))
    candidates = np.asarray(eligible)
    candidates = candidates[(candidates >= min(150, steps//4)) & (candidates < steps-duration)]
    if not len(candidates):
        raise ValueError(f'No eligible interval for {name}')
    start = int(rng.choice(candidates)); stop = start+duration
    k = max(2, int(np.ceil(count*rng.uniform(.05, .10))))
    targets = np.sort(rng.choice(count, min(k, count), replace=False))
    channels = [0, 1, 2]
    amplitude = rng.uniform(.01, .03, (len(targets), 3))
    amplitude[:, 0] = rng.uniform(.001, .003, len(targets))
    amplitude[:, 1:] *= np.maximum(power_rms[targets], 1e-5)
    amplitude *= rng.choice([-1., 1.], amplitude.shape)
    extra = {}
    if family == 'fake_imbalance':
        pools = {phase: [i for i, node in enumerate(sensor_nodes)
                         if phase_by_bus.get(node.rpartition('.')[0]) == phase] for phase in ('a', 'b')}
        if any(not pool for pool in pools.values()):
            raise ValueError('Fake imbalance requires verified A and B primary-phase sensors')
        per_phase = max(1, k//2)
        a, b = [rng.choice(pools[p], min(per_phase, len(pools[p])), replace=False) for p in ('a', 'b')]
        targets = np.r_[a, b]
        channels = [0]
        peak = float(rng.uniform(.001, .003))
        amplitude = np.r_[np.full(len(a), peak), np.full(len(b), -peak)][:, None]
        extra = dict(primary_phases=['a']*len(a)+['b']*len(b), phase_mapping='pv_mapping.csv primary_phase')
    ix = np.ix_(np.arange(start, stop), targets, channels)
    if family == 'delay':
        lag = int(rng.integers(1, 6))
        attacked[ix] = noisy[np.ix_(np.arange(start-lag, stop-lag), targets, channels)]
        extra = dict(delay_seconds=lag, packet_source='Same-trajectory earlier noisy measurements')
    else:
        envelope = np.ones(duration)
        if family == 'ramp':
            envelope = np.linspace(1/duration, 1., duration)
        elif family == 'fake_oscillation':
            period = float(rng.uniform(12., 40.)); phase = float(rng.uniform(0., 2*np.pi))
            envelope = np.sin(2*np.pi*np.arange(duration)/period+phase)
            extra = dict(period_seconds=period, phase_radians=phase)
        attacked[ix] += envelope[:, None, None]*amplitude[None]
    labels[ix] = 1
    event = dict(kind=family, start=start, stop_exclusive=stop, sensor_positions=targets.tolist(),
                 channels=[['Vm', 'P', 'Q'][i] for i in channels],
                 amplitude_pu=None if family == 'delay' else amplitude.tolist(), **extra)
    recipe = dict(active=True, kind=family, **{k:v for k,v in event.items() if k != 'kind'}, events=[event])
    return attacked, labels, recipe

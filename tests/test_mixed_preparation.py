"""Shared-state exclusions, packet-delay semantics and mild phase-aware FDI."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dependencies.mixed_preparation import stratified_split, unique_indices
from dependencies.corruptions import FAMILIES, assign_families, corrupt
from dependencies.experiment import Bundle, Samples, TrainingConfig, train, calibrate, evaluate
from dependencies.prepare import prepare_dataset
from dependencies.common import read_json, write_json, digest
from test_portable import fixture


class MixedPreparationTests(unittest.TestCase):
    def test_stratification_and_families_are_complete_reproducible(self):
        categories = ['imbalance']*100+['oscillation']*99+['overlap']*99+['mixed']*95
        metadata = {f'V{i+1}.npy': dict(attack=c) for i, c in enumerate(categories)}
        splits = stratified_split(metadata, 1)
        self.assertEqual([len(v) for v in splits.values()], [275, 39, 39, 40])
        self.assertEqual(len(set(n for v in splits.values() for n in v)), 393)
        self.assertEqual(splits, stratified_split(metadata, 1))
        families = assign_families(splits, 1)
        for names in splits.values():
            self.assertEqual({families[n] for n in names}, set(FAMILIES))
        for part, names in splits.items():
            for category in set(categories):
                expected = categories.count(category)*len(names)/393
                self.assertLess(abs(sum(metadata[n]['attack'] == category for n in names)-expected), 1.1)

    def test_shared_states_excluded_even_with_different_noise_or_roundoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = {}; splits = {}
            for i, part in enumerate(('train', 'val', 'cal', 'test')):
                name = f'V{i+1}.npy'; path = Path(tmp)/name
                values = np.array([[1+0j, .99+0j], [1+.01*i+0j, .98+.02*i+0j]])
                if i:
                    values[0] += 1e-13
                np.save(path, values); files[name] = path; splits[part] = [name]
            indices, audit = unique_indices(files, splits)
            self.assertEqual(indices['V1.npy'].tolist(), [0, 1])
            for name in ('V2.npy', 'V3.npy', 'V4.npy'):
                self.assertEqual(indices[name].tolist(), [1])
            self.assertEqual(audit['removed_across_partitions'], dict(val=1, cal=1, test=1))

    def test_all_attack_families_preserve_unlabeled_values_and_delay_packets(self):
        rng = np.random.default_rng(2)
        original = rng.normal(size=(1200, 40, 3))*.001
        original[:, :, 0] += 1
        before = original.copy()
        nodes = [f'bus{i}.1' for i in range(40)]
        phases = {f'bus{i}': 'a' if i % 2 else 'b' for i in range(40)}
        for family in FAMILIES:
            attacked, labels, recipe = corrupt(original, family, nodes, phases, np.ones((40, 2))*.001,
                                                np.arange(150, 1200), 3, 'V1.npy')
            np.testing.assert_array_equal(attacked[labels == 0], original[labels == 0])
            repeated = corrupt(original, family, nodes, phases, np.ones((40, 2))*.001,
                               np.arange(150, 1200), 3, 'V1.npy')[0]
            np.testing.assert_array_equal(attacked, repeated)
            if family == 'none':
                self.assertFalse(labels.any()); continue
            self.assertTrue(labels.any())
            a, b = recipe['start'], recipe['stop_exclusive']; targets = recipe['sensor_positions']
            if family == 'delay':
                lag = recipe['delay_seconds']
                np.testing.assert_array_equal(attacked[a:b, targets], original[a-lag:b-lag, targets])
            if family == 'fake_imbalance':
                self.assertFalse(labels[:, :, 1:].any())
                for target, phase in zip(targets, recipe['primary_phases']):
                    self.assertEqual(phases[f'bus{target}'], phase)
                    self.assertTrue(np.all((attacked-original)[a:b, target, 0]*(1 if phase == 'a' else -1) > 0))
        np.testing.assert_array_equal(before, original)

    def test_training_calibration_and_testing_use_original_selected_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, _ = fixture(root); prepare_dataset(cfg)
            out = Path(cfg.output_dir); manifest = read_json(out/'manifest.json')
            (out/'sample_indices').mkdir()
            for name, record in manifest['records'].items():
                np.save(out/'sample_indices'/name, np.array([2, 5], np.int32))
                record['eligible_samples'] = 2
            manifest['sample_filter'] = 'sample_indices/*.npy'
            manifest['hashes'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*')
                                  if p.is_file() and p.name != 'manifest.json'}
            write_json(out/'manifest.json', manifest)
            bundle = Bundle(out); samples = Samples(bundle, 'train', 'fdi')
            self.assertEqual(len(samples), 2*len(bundle.splits['train']))
            name = bundle.splits['train'][0]
            np.testing.assert_array_equal(samples[0][2], bundle.array('truth', name)[2])
            tc = TrainingConfig(str(out), str(root/'run'), device='cpu', epochs=1, patience=1,
                                batch_size=4, threads=1)
            train(tc); calibration = calibrate(tc); reports = evaluate(tc, save_predictions=False)
            self.assertEqual(calibration['snapshots'], 2*len(bundle.splits['cal']))
            self.assertEqual(reports['fdi']['eligible_snapshots'], 2*len(bundle.splits['test']))
            self.assertEqual(reports['fdi']['overall']['value_count'], 6*len(bundle.splits['test']))


if __name__ == '__main__':
    unittest.main()

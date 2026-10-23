"""Deployment parity, input validation, integrity, relocation, and CLI checks."""
from contextlib import redirect_stdout
import copy
from dataclasses import asdict
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from dependencies.common import read_json, write_json
from dependencies.experiment import Bundle, TrainingConfig, train, load_model, predict, identity, matches_identity
from dependencies.prepare import prepare_dataset
from dependencies.deployment import Deployment, checked_path, export_bundle, external_path
from dependencies.inference import infer, predict_files, test_prepared, validate_arrays
from test_portable import fixture


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cfg, _ = fixture(cls.root)
        prepare_dataset(cfg)
        cls.cfg = TrainingConfig(cfg.output_dir, str(cls.root / 'run'), device='cpu', threads=1,
                                 epochs=1, patience=1, batch_size=3)
        with redirect_stdout(io.StringIO()):
            train(cls.cfg)
        cls.bundle_dir = cls.root / 'model'
        export_bundle(cfg.output_dir, cls.cfg.run_dir, cls.bundle_dir)
        cls.deployment = Deployment(cls.bundle_dir, threads=1, batch_size=3)
        cls.bundle = Bundle(cfg.output_dir)
        cls.name = cls.bundle.splits['test'][0]
        cls.raw = cls.bundle.array('fdi', cls.name)
        cls.av = cls.bundle.array('availability', cls.name)
        cls.truth = cls.bundle.array('truth', cls.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_export_prediction_parity_and_relocation(self):
        rt, model = load_model(self.cfg)
        reference = predict(rt, model, self.raw, self.av)
        with tempfile.TemporaryDirectory() as tmp:
            moved = Path(tmp) / 'moved'
            shutil.copytree(self.bundle_dir, moved)
            deployment = Deployment(moved, threads=1, batch_size=3)
            output = Path(tmp) / 'pred.npy'
            report = infer(deployment, self.raw, self.av, self.truth, prediction_path=output)
            np.testing.assert_array_equal(np.load(output), reference)
            self.assertEqual(report['metrics']['value_count'], self.truth.size)
            np.testing.assert_allclose(report['metrics']['complex_phasor_rmse_pu'],
                                       np.sqrt(np.mean(abs(reference-self.truth)**2)), rtol=1e-14)
            filtered = infer(deployment, self.raw, self.av, self.truth, np.array([1, 6]))
            self.assertEqual(filtered['metrics']['value_count'], 2*self.truth.shape[1])

    def test_no_truth_required_and_order_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'raw.npy'
            np.save(raw, self.raw)
            report = predict_files(self.deployment, raw, self.bundle_dir / 'sensor_order.json', root / 'out')
            self.assertIsNone(report['metrics'])
            self.assertEqual(np.load(root / 'out/predictions.npy').shape, self.truth.shape)
            with self.assertRaises(FileExistsError):
                predict_files(self.deployment, raw, self.bundle_dir / 'sensor_order.json', root / 'out')
            wrong = root / 'wrong.json'
            write_json(wrong, list(reversed(self.deployment.sensor_order)))
            with self.assertRaisesRegex(ValueError, 'sensor order'):
                predict_files(self.deployment, raw, wrong, root / 'wrong-out')
            truth = root / 'truth.npy'
            np.save(truth, self.truth)
            with self.assertRaisesRegex(ValueError, 'truth-node-order'):
                predict_files(self.deployment, raw, self.bundle_dir / 'sensor_order.json', root / 'no-order', truth=truth)
            self.assertFalse((root / 'no-order').exists())

    def test_bad_measurements_availability_truth_and_indices(self):
        for raw in (self.raw.transpose(1, 0, 2), self.raw.astype(complex), self.raw[:0], self.raw.astype(int)):
            with self.subTest(shape=raw.shape, dtype=raw.dtype), self.assertRaises(ValueError):
                validate_arrays(self.deployment, raw)
        raw = self.raw.copy(); raw[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'NaN'):
            validate_arrays(self.deployment, raw)
        for av in (self.av[:, :1], np.full(self.av.shape, 2), np.full(self.av.shape, np.inf)):
            with self.assertRaises(ValueError):
                validate_arrays(self.deployment, self.raw, av)
        for truth in (self.truth.real, self.truth[:, :1], self.truth * np.nan):
            with self.assertRaises(ValueError):
                validate_arrays(self.deployment, self.raw, truth=truth)
        for ix in (np.array([]), np.array([1, 1]), np.array([2, 1]), np.array([-1]), np.array([len(self.raw)]), np.array([1.])):
            with self.subTest(indices=ix), self.assertRaises(ValueError):
                validate_arrays(self.deployment, self.raw, indices=ix)

    def test_missing_channels_cannot_leak_and_all_missing_is_finite(self):
        av = self.av.copy(); av[:, 0] = 0
        changed = self.raw.copy(); changed[:, 0] = 1000
        rt, model = self.deployment.runtime, self.deployment.model
        np.testing.assert_array_equal(predict(rt, model, self.raw, av), predict(rt, model, changed, av))
        self.assertTrue(np.isfinite(predict(rt, model, self.raw, np.zeros_like(av))).all())

    def test_asset_integrity_metadata_and_path_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(self.bundle_dir, root / 'model')
            path = root / 'model/normalization.npz'
            with path.open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                Deployment(root / 'model', threads=1)
            shutil.copy2(self.bundle_dir / 'normalization.npz', path)
            meta = read_json(root / 'model/bundle.json')
            meta['training_config']['voltage_std_floor'] = .5
            write_json(root / 'model/bundle.json', meta)
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                Deployment(root / 'model', threads=1)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                checked_path(root, '../outside')
        with self.assertRaises(ValueError):
            Deployment(self.bundle_dir, batch_size=0)
        with self.assertRaisesRegex(ValueError, 'outside the repository'):
            external_path(Path(__file__).resolve().parents[1]/'models')

    def test_layout_migration_preserves_strict_data_config_and_model_checks(self):
        current = identity(self.cfg, self.bundle)
        original = copy.deepcopy(current)
        original['code_sha256']['experiment.py'] = 'c68e81ad11fbeb8d3cab7d0d17ed055442deb69896f77d84cdd667b953255b22'
        self.assertTrue(matches_identity(original, current))
        for section, key, value in (
                ('config', 'batch_size', 999), ('code_sha256', 'model.py', 'changed'),
                ('code_sha256', 'experiment.py', 'unknown')):
            changed = copy.deepcopy(original)
            changed[section][key] = value
            self.assertFalse(matches_identity(changed, current))
        changed = copy.deepcopy(original); changed['prepared_sha256'] = 'different dataset'
        self.assertFalse(matches_identity(changed, current))

    def test_frozen_test_partition_and_dataset_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with redirect_stdout(io.StringIO()):
                report = test_prepared(self.deployment, self.cfg.prepared_dir, root / 'test', ['fdi'])
            condition = report['conditions']['fdi']
            names = self.bundle.splits['test']
            self.assertEqual([v['file'] for v in condition['per_trajectory']], names)
            self.assertEqual(condition['overall']['value_count'], sum(self.bundle.array('truth', n).size for n in names))
            changed = root / 'changed'
            shutil.copytree(self.cfg.prepared_dir, changed)
            array = np.load(changed / 'fdi' / self.name); array.flat[0] += 1
            np.save(changed / 'fdi' / self.name, array)
            with self.assertRaisesRegex(ValueError, 'test asset changed'):
                test_prepared(self.deployment, changed, root / 'bad', ['fdi'])
            self.assertFalse((root / 'bad').exists())

    def test_cli_with_spaces_and_order_rejection(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='ptae cli ') as tmp:
            root = Path(tmp)
            raw = root / 'input measurements.npy'; np.save(raw, self.raw)
            command = [sys.executable, str(repo/'run.py'), 'predict', '--model-dir', str(self.bundle_dir),
                       '--measurements', str(raw), '--sensor-order', str(self.bundle_dir/'sensor_order.json'),
                       '--output', str(root/'output with spaces'), '--threads', '1']
            result = subprocess.run(command, cwd=repo, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root/'output with spaces/predictions.npy').exists())
            wrong_order = root/'wrong-order.json'
            write_json(wrong_order, list(reversed(self.deployment.sensor_order)))
            command[command.index('--sensor-order')+1] = str(wrong_order)
            rejected = subprocess.run(command, cwd=repo, capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 2)
            self.assertIn('sensor order', rejected.stderr)

    def test_cli_preparation_training_export_and_relocated_resume(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='ptae training ') as tmp:
            root = Path(tmp)
            cfg, _ = fixture(root)
            config_path = root/'prepare.json'
            write_json(config_path, asdict(cfg))

            def cli(*args):
                result = subprocess.run([sys.executable, str(repo/'run.py'), *map(str, args)],
                                        cwd=repo, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout

            cli('prepare', '--config', config_path)
            run = root/'training run'
            cli('train', '--prepared-dir', cfg.output_dir, '--run-dir', run, '--epochs', 1,
                '--patience', 1, '--threads', 1, '--batch-size', 4, '--stage', 'all')
            exported = root/'exported model'
            cli('export', '--prepared-dir', cfg.output_dir, '--run-dir', run, '--output', exported)
            self.assertIn('"verified": true', cli('verify', '--model-dir', exported, '--threads', 1))
            moved_data, moved_run = root/'moved data', root/'moved run'
            shutil.move(cfg.output_dir, moved_data); shutil.move(run, moved_run)
            message = cli('train', '--prepared-dir', moved_data, '--run-dir', moved_run,
                          '--epochs', 3, '--threads', 1, '--stage', 'all')
            self.assertIn('saved numerical settings', message)
            self.assertEqual(read_json(moved_run/'training_summary.json')['epochs_completed'], 1)
            blocked = root/'not a run'; blocked.mkdir(); (blocked/'private.txt').write_text('keep')
            result = subprocess.run([sys.executable, str(repo/'run.py'), 'train', '--prepared-dir', str(moved_data),
                                     '--run-dir', str(blocked)], cwd=repo, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual((blocked/'private.txt').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()

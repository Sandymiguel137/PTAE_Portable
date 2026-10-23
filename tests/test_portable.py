"""Physics, leakage, checkpoint resumption, and relocation checks."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
import torch
from dependencies.prepare import PreparationConfig, prepare_dataset, measure
from dependencies.experiment import Bundle, TrainingConfig, Runtime, train, calibrate, evaluate, load_model, predict, checkpoint_report
from dependencies.common import write_json, read_json, digest


def fixture(root, native=False):
    src=root/'voltage';sys=root/'system';src.mkdir();sys.mkdir()
    nodes=['bus.1','bus.2','bus.3']
    y=sp.csc_matrix(np.array([[2-3j,-1+1.5j,-1+1.5j],[-1+1.5j,2-3j,-1+1.5j],[-1+1.5j,-1+1.5j,2-3j]]))
    sp.save_npz(sys/'ybus_pu.npz',y)
    write_json(sys/'ynodeorder.json',nodes);write_json(src/'ynodeorder.json',nodes)
    np.save(sys/'sensor_indices_0based.npy',np.arange(3));np.save(sys/'node_voltage_base_volts.npy',np.ones(3)*120)
    if native:(src/'physics').mkdir()
    for i in range(1,9):
        t=np.arange(8);mag=1+.001*i+.002*np.sin(t/3+np.arange(3)[:,None])
        voltage=mag*np.exp(1j*(np.array([0,-2*np.pi/3,2*np.pi/3])[:,None]+.001*i))
        np.save(src/f'V{i}.npy',voltage)
        if native:
            values=np.stack([y.data*(1+.01*j) for j in t])
            np.savez(src/'physics'/f'Y{i}.npz',data=values,indices=y.indices,indptr=y.indptr,shape=y.shape)
    cfg=PreparationConfig(str(src),str(root/'prepared'),str(src/'ynodeorder.json'),str(sys),
        measurement_mode='native' if native else 'fixed',physics_dir=str(src/'physics') if native else '',
        noise_std_fraction=(.02,.03,.04),fdi_sensor_percentage=100)
    return cfg,y


class PortableTests(unittest.TestCase):
    def test_unclipped_normalization_preserves_large_inputs_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,_=fixture(root);prepare_dataset(cfg)
            tc=TrainingConfig(cfg.output_dir,str(root/'run'),device='cpu',threads=1)
            b=Bundle(cfg.output_dir);rt=Runtime(tc,b)
            # Deliberately large deviations must survive normalization.
            raw=rt.amean[None]+123. * rt.astd.clamp_min(tc.measurement_std_floor)[None]
            available=torch.ones(raw.shape[:2],device=rt.device)
            x=rt.input(raw,available)
            torch.testing.assert_close(x[:,rt.sensors,:3],torch.full_like(x[:,rt.sensors,:3],123.))
            truth=torch.tensor(np.array(b.array('truth',b.splits['train'][0]),copy=True),device=rt.device)
            target=rt.target(truth)
            torch.testing.assert_close(rt.decode(target),truth,rtol=1e-6,atol=1e-7)
            tc.input_normalization_clip=50.
            clipped=rt.input(raw,available)
            torch.testing.assert_close(clipped[:,rt.sensors,:3],torch.full_like(clipped[:,rt.sensors,:3],50.))

    def test_analytic_ac_measurement(self):
        y=sp.csc_matrix([[2-1j,-2+1j],[-2+1j,2-1j]])
        v=np.array([[1+0j,.97-.04j],[1+.01j,.98-.02j]])
        actual=measure(v,np.array([1]),y=y)
        current=(2-1j)*(v[:,1]-v[:,0]);s=v[:,1]*current.conj()
        np.testing.assert_allclose(actual[:,0],np.stack([abs(v[:,1]),s.real,s.imag],-1),atol=1e-14)

    def test_preparation_determinism_truth_and_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,y=fixture(root);prepare_dataset(cfg);b=Bundle(cfg.output_dir)
            name=b.splits['train'][0];truth=b.array('truth',name)
            np.testing.assert_array_equal(truth,np.load(Path(cfg.voltage_dir)/name).T)
            np.testing.assert_allclose(b.array('clean',name),measure(truth,b.sensors,y=y))
            mask=b.array('labels',name).astype(bool);delta=b.array('fdi',name)-b.array('noisy',name)
            self.assertTrue(mask.any());self.assertTrue(np.all(delta[~mask]==0));self.assertTrue(np.all(delta[mask]!=0))
            source=Path(cfg.voltage_dir)/b.splits['test'][0];np.save(source,np.load(source)*1.2)
            second=copy.deepcopy(cfg);second.output_dir=str(root/'second');prepare_dataset(second);other=Bundle(second.output_dir)
            for key,value in b.stats.items():np.testing.assert_array_equal(value,other.stats[key])
            for folder in ['noisy','fdi','labels','availability']:
                np.testing.assert_array_equal(b.array(folder,name),other.array(folder,name))
            changed=root/'prepared/noisy'/name;a=np.load(changed);a.flat[0]+=1;np.save(changed,a)
            with self.assertRaisesRegex(ValueError,'changed'):Bundle(cfg.output_dir)

    def test_group_split_protection_and_sensor_percentage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,y=fixture(root)
            groups={f'V{i}.npy':str((i-1)//2) for i in range(1,9)}
            write_json(root/'groups.json',groups);cfg.group_file=str(root/'groups.json')
            cfg.sensor_pool='all_nodes';cfg.sensor_percentage=50;m=prepare_dataset(cfg);self.assertEqual(m['sensors'],2)
            ownership={}
            for part,names in m['splits'].items():
                for name in names:
                    self.assertTrue(groups[name] not in ownership or ownership[groups[name]]==part);ownership[groups[name]]=part
            bad={'train':['V1.npy','V3.npy','V4.npy','V5.npy','V6.npy'],'val':['V2.npy'],'cal':['V7.npy'],'test':['V8.npy']}
            write_json(root/'bad.json',bad);cfg.split_file=str(root/'bad.json');cfg.output_dir=str(root/'badout')
            with self.assertRaisesRegex(ValueError,'crosses'):prepare_dataset(cfg)

    def test_native_physics_runtime_matches_saved_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.set_num_threads(1);root=Path(tmp);cfg,y=fixture(root,native=True);prepare_dataset(cfg)
            b=Bundle(cfg.output_dir);tc=TrainingConfig(cfg.output_dir,str(root/'run'),device='cpu',threads=1)
            rt=Runtime(tc,b);name='V1.npy';v=torch.tensor(np.array(b.array('truth',name)),requires_grad=True)
            rt.context=[b.physics(name,t,t+1) for t in range(len(v))];result=rt.measurement(v)
            np.testing.assert_allclose(result.detach().numpy(),b.array('clean',name),rtol=1e-12,atol=1e-12)
            result.square().sum().backward();self.assertTrue(torch.isfinite(v.grad).all())
            tc.epochs=1;tc.patience=1;tc.batch_size=8
            self.assertTrue(train(tc)['complete'])

    def test_training_calibration_test_and_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,y=fixture(root);prepare_dataset(cfg)
            tc=TrainingConfig(cfg.output_dir,str(root/'run'),epochs=1,patience=1,batch_size=8,device='cpu',threads=1)
            report=train(tc);self.assertTrue(report['complete']);calibrate(tc);reports=evaluate(tc)
            self.assertEqual(set(reports),{'clean','noisy','fdi'})
            rt,model=load_model(tc);name=rt.bundle.splits['test'][0]
            old=predict(rt,model,rt.bundle.array('fdi',name),rt.bundle.array('availability',name))
            shutil.move(cfg.output_dir,root/'moved_prepared');shutil.move(tc.run_dir,root/'moved_run')
            tc.prepared_dir=str(root/'moved_prepared');tc.run_dir=str(root/'moved_run')
            self.assertEqual(train(tc)['best_epoch'],report['best_epoch'])
            rt,model=load_model(tc);new=predict(rt,model,rt.bundle.array('fdi',name),rt.bundle.array('availability',name))
            np.testing.assert_array_equal(old,new)
            self.assertGreaterEqual(reports['fdi']['empirical_node_time_coverage'],0);self.assertLessEqual(reports['fdi']['empirical_node_time_coverage'],1)

    def test_interrupted_training_resumes_exact_epoch_state(self):
        import dependencies.experiment as experiment
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,y=fixture(root);prepare_dataset(cfg)
            tc=TrainingConfig(cfg.output_dir,str(root/'continuous'),epochs=2,patience=3,batch_size=8,device='cpu',threads=1)
            train(tc)
            expected=torch.load(root/'continuous/last_model.pth',weights_only=True)['model_state_dict']
            tc.run_dir=str(root/'resumed');original=experiment.torch_save
            def interrupt(path,obj):
                original(path,obj)
                if Path(path).name=='last_model.pth':raise RuntimeError('simulated interruption')
            with patch.object(experiment,'torch_save',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated'):train(tc)
            train(tc)
            actual=torch.load(root/'resumed/last_model.pth',weights_only=True)['model_state_dict']
            for key in expected:
                a,b=expected[key],actual[key]
                if a.is_sparse:a,b=a.to_dense(),b.to_dense()
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_interrupted_checkpoint_can_calibrate_test_and_resume(self):
        import dependencies.experiment as experiment
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,y=fixture(root);prepare_dataset(cfg)
            tc=TrainingConfig(cfg.output_dir,str(root/'run'),epochs=2,patience=3,batch_size=8,device='cpu',threads=1)
            original=experiment.torch_save
            def interrupt(path,obj):
                original(path,obj)
                if Path(path).name=='last_model.pth':raise RuntimeError('simulated interruption')
            with patch.object(experiment,'torch_save',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated'):train(tc)
            summary=root/'run/training_summary.json'
            info=checkpoint_report(tc)
            self.assertFalse(info['training_complete']);self.assertEqual(info['epochs_completed'],1)
            calibration=calibrate(tc);reports=evaluate(tc,save_predictions=False)
            self.assertEqual(set(reports),{'clean','noisy','fdi'})
            self.assertFalse(summary.exists())
            self.assertFalse(read_json(root/'run/status.json')['training_complete'])
            report=train(tc);self.assertEqual(report['epochs_completed'],2)
            refreshed=calibrate(tc);evaluate(tc,save_predictions=False)
            self.assertEqual(refreshed['identity']['checkpoint_sha256'],digest(root/'run/best_model.pth'))
            self.assertTrue(checkpoint_report(tc)['training_complete'])
            # A new best checkpoint must not use radii from the earlier checkpoint.
            if refreshed['identity']!=calibration['identity']:
                self.assertNotEqual(refreshed['identity']['checkpoint_sha256'],calibration['identity']['checkpoint_sha256'])

    def test_legacy_checkpoint_identity_stays_strict_and_resumable(self):
        import dependencies.experiment as experiment
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,y=fixture(root);prepare_dataset(cfg)
            tc=TrainingConfig(cfg.output_dir,str(root/'run'),epochs=2,patience=3,batch_size=8,device='cpu',threads=1,input_normalization_clip=50.)
            original=experiment.torch_save
            def interrupt(path,obj):
                original(path,obj)
                if Path(path).name=='last_model.pth':raise RuntimeError('simulated interruption')
            with patch.object(experiment,'torch_save',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated'):train(tc)
            legacy=read_json(root/'run/identity.json')
            legacy['code_sha256']['experiment.py']=next(iter(experiment._COMPATIBLE_EXPERIMENT_HASHES))
            write_json(root/'run/identity.json',legacy)
            for name in ['best_model.pth','last_model.pth']:
                path=root/'run'/name;c=torch.load(path,weights_only=True);c['identity']=legacy;original(path,c)
            rt,model=load_model(tc);self.assertFalse(model.training)
            changed=copy.deepcopy(tc);changed.batch_size+=1
            with self.assertRaisesRegex(ValueError,'identity'):load_model(changed)
            current=experiment.identity(tc,rt.bundle)
            unclipped=copy.deepcopy(tc);unclipped.input_normalization_clip=None
            self.assertFalse(experiment.matches_identity(legacy,experiment.identity(unclipped,rt.bundle)))
            bad=copy.deepcopy(legacy);bad['code_sha256']['model.py']='changed'
            self.assertFalse(experiment.matches_identity(bad,current))
            bad=copy.deepcopy(legacy);bad['prepared_sha256']='changed'
            self.assertFalse(experiment.matches_identity(bad,current))
            report=train(tc);self.assertEqual(report['epochs_completed'],2)
            self.assertEqual(report['identity'],legacy)
            self.assertEqual(torch.load(root/'run/last_model.pth',weights_only=True)['identity'],legacy)

    def test_recalibration_refreshes_checkpoint_and_rejects_stale_testing(self):
        import dependencies.experiment as experiment
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,y=fixture(root);prepare_dataset(cfg)
            tc=TrainingConfig(cfg.output_dir,str(root/'run'),epochs=1,patience=1,batch_size=8,device='cpu',threads=1)
            train(tc);first=calibrate(tc)
            # Changing checkpoint bytes emulates selection of a new checkpoint.
            path=root/'run/best_model.pth';c=torch.load(path,weights_only=True)
            c['epoch']+=1;c['best_epoch']=c['epoch'];experiment.torch_save(path,c)
            with self.assertRaisesRegex(ValueError,'Calibration does not match'):evaluate(tc,save_predictions=False)
            second=calibrate(tc)
            self.assertNotEqual(first['identity']['checkpoint_sha256'],second['identity']['checkpoint_sha256'])
            self.assertEqual(second,calibrate(tc))
            reports=evaluate(tc,save_predictions=False)
            self.assertTrue(all(r['checkpoint_sha256']==second['identity']['checkpoint_sha256'] for r in reports.values()))
            radius=root/'run/calibration/fdi_radii.npy';a=np.load(radius);a[0]+=1;np.save(radius,a)
            with self.assertRaisesRegex(ValueError,'radii changed'):calibrate(tc)



if __name__=='__main__':unittest.main()

"""Train the unchanged PTAE from saved measurements; calibrate and test separately."""
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import OrderedDict
import bisect
import json
import time
import random
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .common import digest, read_json, write_json
from .prepare import load_y
from .model import PhysicsTransformerAE, SPEC, training_loss
from .metrics import new_accumulator, update_metrics, finalize_metrics

# Keep training identity independent of optional helper modules in this folder.
CORE_FILES = ('__init__.py', 'common.py', 'corruptions.py', 'experiment.py',
              'metrics.py', 'model.py', 'prepare.py')


def core_hashes():
    root = Path(__file__).parent
    return {name: digest(root / name) for name in CORE_FILES}


@dataclass
class TrainingConfig:
    prepared_dir: str
    run_dir: str
    condition: str = 'fdi'
    device: str = 'auto'
    epochs: int = 200
    patience: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.
    threads: int = 4
    seed: int = 42
    voltage_std_floor: float = 1e-3
    measurement_std_floor: float = 1e-6
    input_normalization_clip: float | None = None


class Bundle:
    def __init__(self, root, verify=True):
        self.root = Path(root).expanduser().resolve()
        self.manifest = read_json(self.root / 'manifest.json')
        if self.manifest.get('format') != 'ptae-portable-prepared-v1' or not self.manifest.get('complete'):
            raise ValueError('Prepared dataset is incomplete or has an unknown format')
        if verify:
            for name, expected in self.manifest['hashes'].items():
                if digest(self.root / name) != expected:
                    raise ValueError('Prepared file changed: ' + name)
        self.identity = digest(self.root / 'manifest.json')
        self.sensors = np.load(self.root / 'system/sensor_indices_0based.npy')
        self.nodes = read_json(self.root / 'system/ynodeorder.json')
        self.y = load_y(self.root / 'system/ybus_pu.npz')
        with np.load(self.root / 'normalization.npz') as z:
            self.stats = {k: z[k] for k in z.files}
        self.splits = self.manifest['splits']
        self.native = self.manifest['measurement_mode'] == 'native'
        self.opened = OrderedDict()

    def array(self, folder, name):
        key = (folder, name)
        if key not in self.opened:
            self.opened[key] = np.load(self.root / folder / name, mmap_mode='r', allow_pickle=False)
            if len(self.opened) > 32:
                self.opened.popitem(last=False)
        self.opened.move_to_end(key)
        return self.opened[key]

    def physics(self, name, start, stop):
        if not self.native:
            return None
        with np.load(self.root / 'physics' / (Path(name).stem + '_pattern.npz')) as z:
            rows, cols = z['rows'], z['cols']
        return rows, cols, np.array(self.array('physics', name)[start:stop], copy=True)

    def indices(self, name):
        """Eligible original timestamps; legacy bundles retain every snapshot."""
        count = self.manifest['records'][name]['samples']
        if not self.manifest.get('sample_filter'):
            return np.arange(count)
        selected = self.array('sample_indices', name)
        if (selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer) or not len(selected)
                or selected[0] < 0 or selected[-1] >= count or np.any(np.diff(selected) <= 0)):
            raise ValueError('Invalid eligible sample indices: '+name)
        if len(selected) != self.manifest['records'][name]['eligible_samples']:
            raise ValueError('Eligible sample count differs from manifest: '+name)
        return selected

    def summary(self):
        return dict(nodes=len(self.nodes), sensors=len(self.sensors), mode=self.manifest['measurement_mode'],
                    trajectories={k: len(v) for k, v in self.splits.items()},
                    samples={k: sum(len(self.indices(n)) for n in v) for k,v in self.splits.items()},
                    sensor_selection=self.manifest['sensor_selection'], grouping=self.manifest['grouping'])


class Samples(Dataset):
    def __init__(self, bundle, partition, condition):
        self.bundle, self.files, self.condition = bundle, bundle.splits[partition], condition
        self.indices = [bundle.indices(n) for n in self.files]
        self.ends = np.cumsum([len(ix) for ix in self.indices]).tolist()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        i = bisect.bisect_right(self.ends, index)
        t = int(self.indices[i][index - (self.ends[i-1] if i else 0)])
        name = self.files[i]; b = self.bundle
        raw = np.array(b.array(self.condition, name)[t], copy=True)
        av = np.array(b.array('availability', name)[t], dtype=np.float64, copy=True)
        truth = np.array(b.array('truth', name)[t], copy=True)
        return raw, av, truth, b.physics(name, t, t+1)


def collate(samples):
    arrays = [torch.from_numpy(np.stack([s[k] for s in samples])) for k in range(3)]
    return (*arrays, [s[3] for s in samples])


class Runtime:
    def __init__(self, cfg, bundle):
        self.cfg, self.bundle = cfg, bundle
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device == 'auto' else cfg.device
        self.device = torch.device(device)
        self.sensors = torch.as_tensor(bundle.sensors, dtype=torch.long, device=self.device)
        s = bundle.stats; t = lambda a: torch.as_tensor(a, device=self.device)
        self.mean = t(np.stack((s['vr_mean'], s['vi_mean']), -1))
        self.std = t(np.stack((s['vr_std'], s['vi_std']), -1)).clamp_min(cfg.voltage_std_floor)
        self.amean, self.astd = t(s['ami_mean']), t(s['ami_std'])
        y = bundle.y.tocoo()
        self.ybus = torch.sparse_coo_tensor(t(np.stack((y.row, y.col))), t(y.data), y.shape).coalesce()
        self.context = None

    def model(self):
        return PhysicsTransformerAE(self.bundle.y, self.bundle.sensors,
            width=SPEC['width'], heads=SPEC['heads'], tokens=SPEC['latent_tokens'],
            layers=SPEC['transformer_layers'], dropout=SPEC['dropout']).to(self.device)

    def input(self, raw, available):
        raw = raw.to(dtype=self.amean.dtype)
        available = available.to(dtype=raw.dtype)
        x = raw.new_zeros((len(raw), len(self.bundle.nodes), 5))
        normalized = (raw-self.amean)/self.astd.clamp_min(self.cfg.measurement_std_floor)
        if self.cfg.input_normalization_clip is not None:
            normalized = normalized.clamp(-self.cfg.input_normalization_clip, self.cfg.input_normalization_clip)
        x[:, self.sensors, :3] = normalized*available[...,None]
        x[:, self.sensors, 3] = available
        x[:, self.sensors, 4] = 1.
        return x.float()

    def target(self, truth):
        return ((torch.view_as_real(truth)-self.mean)/self.std).float()

    def decode(self, ri):
        x = ri*self.std+self.mean
        return torch.complex(x[...,0], x[...,1])

    def measurement(self, v):
        v = v.to(torch.complex128)
        if self.bundle.native:
            if self.context is None or len(self.context) != len(v):
                raise ValueError('Matching per-snapshot Y is required for native physics supervision')
            currents = []
            for i, context in enumerate(self.context):
                rows, cols, values = context
                rows = torch.as_tensor(rows, device=v.device, dtype=torch.long)
                cols = torch.as_tensor(cols, device=v.device, dtype=torch.long)
                values = torch.as_tensor(values[0], device=v.device)
                currents.append(v.new_zeros(len(self.sensors)).index_add(0, rows, values*v[i,cols]))
            current = torch.stack(currents)
        else:
            current = torch.sparse.mm(self.ybus, v.T).T[:, self.sensors]
        selected = v[:, self.sensors]; power = selected*current.conj()
        return torch.stack((selected.abs(), power.real, power.imag), -1)


def torch_save(path, obj):
    path = Path(path); tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp); tmp.replace(path)


def identity(cfg, bundle):
    config = asdict(cfg)
    # Location and execution hardware may change when moving a run to another device.
    for key in ['prepared_dir', 'run_dir', 'device', 'threads']:
        config.pop(key)
    # Legacy runs always clipped normalized inputs at +/-50.
    if config['input_normalization_clip'] == 50.:
        config.pop('input_normalization_clip')
    return dict(prepared_sha256=bundle.identity, config=config,
                code_sha256=core_hashes())



# This exact previous orchestration version used the same model, training loop,
# normalization and inference. Accept its checkpoints without rewriting history.
# All other source hashes, dataset hashes and numerical settings stay strict.
_COMPATIBLE_EXPERIMENT_HASHES = {
    '9fee78c48802ac5ee0339c393a4e5d7f41230629f2e9306393155a873553656c',
    'c75e6097ccfa9d122752abbd158c11435dc0684defe4e82efe95f87a11411031',
}
# Mixed-393 runtime before consolidating the folder layout. Its numerical code
# is unchanged; only source identity enumeration/migration changed in this file.
_PRE_LAYOUT_EXPERIMENT_HASH = 'c68e81ad11fbeb8d3cab7d0d17ed055442deb69896f77d84cdd667b953255b22'


def matches_identity(saved, current):
    if saved == current:
        return True
    previous = saved.get('code_sha256', {}).get('experiment.py')
    known_layout_migration = previous == _PRE_LAYOUT_EXPERIMENT_HASH
    known_legacy_migration = (previous in _COMPATIBLE_EXPERIMENT_HASHES
                              and 'input_normalization_clip' not in current.get('config', {}))
    if not (known_layout_migration or known_legacy_migration):
        return False
    migrated = {**saved, 'code_sha256': {
        **saved['code_sha256'], 'experiment.py': current['code_sha256']['experiment.py'],
    }}
    return migrated == current


def checkpoint_report(cfg, bundle=None):
    """Describe a validation-selected checkpoint, including interrupted runs."""
    bundle = bundle or Bundle(cfg.prepared_dir)
    folder = Path(cfg.run_dir).expanduser()
    path = folder / 'best_model.pth'
    if not path.exists():
        raise FileNotFoundError('No best_model.pth found; train at least one complete epoch first')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not matches_identity(checkpoint['identity'], identity(cfg, bundle)):
        raise ValueError("Model identity differs from data/code/settings. Load this run's saved training_config.json.")
    if checkpoint.get('architecture') != SPEC:
        raise ValueError('Checkpoint architecture differs from the current model')
    if (folder / 'identity.json').exists() and read_json(folder / 'identity.json') != checkpoint['identity']:
        raise ValueError('Checkpoint identity differs from the saved run')
    epochs = checkpoint['epoch']
    if (folder / 'last_model.pth').exists():
        last = torch.load(folder / 'last_model.pth', map_location='cpu', weights_only=True)
        if last['identity'] != checkpoint['identity']:
            raise ValueError('Best and last checkpoints belong to different runs')
        epochs = last['epoch']
    return dict(ready_for_evaluation=True,
                training_complete=(folder / 'training_summary.json').exists(),
                epochs_completed=epochs, best_epoch=checkpoint['epoch'],
                best_validation_state_mse=checkpoint['best'],
                selection_partition='val', checkpoint_sha256=digest(path))


def train(cfg, resume=True):
    if cfg.input_normalization_clip is not None and (not np.isfinite(cfg.input_normalization_clip) or cfg.input_normalization_clip <= 0):
        raise ValueError('input_normalization_clip must be None or a positive finite number')
    if cfg.condition not in ('clean','noisy','fdi'):
        raise ValueError('Unknown saved measurement condition')
    if min(cfg.epochs,cfg.patience,cfg.batch_size,cfg.threads) < 1 or cfg.learning_rate <= 0:
        raise ValueError('Invalid training settings')
    torch.set_num_threads(cfg.threads); random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    bundle = Bundle(cfg.prepared_dir); rt = Runtime(cfg, bundle)
    folder = Path(cfg.run_dir).expanduser().resolve(); folder.mkdir(parents=True,exist_ok=True)
    ident = identity(cfg,bundle)
    if (folder/'identity.json').exists():
        saved_identity = read_json(folder/'identity.json')
        if not matches_identity(saved_identity, ident):
            raise ValueError('Data, code or training settings changed. Use a new run directory.')
        ident = saved_identity  # Preserve provenance and exact resume checkpoint identity.
    if (folder/'training_summary.json').exists():
        if not resume: raise FileExistsError('Completed run exists; choose a new run directory')
        return read_json(folder/'training_summary.json')
    write_json(folder/'identity.json',ident); write_json(folder/'training_config.json',asdict(cfg))
    write_json(folder/'status.json',dict(stage='training',complete=False))
    generator = torch.Generator().manual_seed(cfg.seed)
    loaders = {part: DataLoader(Samples(bundle,part,cfg.condition),batch_size=cfg.batch_size,
                shuffle=part=='train',generator=generator if part=='train' else None,
                num_workers=0,collate_fn=collate) for part in ['train','val']}
    model = rt.model(); opt = torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=5)
    print(json.dumps(dict(event='training_start', model='temporal' if model.temporal else 'point',
        device=str(rt.device), gpu=torch.cuda.get_device_name(rt.device) if rt.device.type=='cuda' else None,
        condition=cfg.condition, epochs=cfg.epochs, patience=cfg.patience, batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay,
        eligible_samples={part:len(loader.dataset) for part,loader in loaders.items()},
        parameters=sum(p.numel() for p in model.parameters()))),flush=True)
    start=1; best=float('inf'); best_epoch=0; stale=0
    if (folder/'last_model.pth').exists():
        if not resume: raise FileExistsError('Checkpoint exists; choose a new run directory')
        c = torch.load(folder/'last_model.pth',map_location=rt.device,weights_only=True)
        if c['identity'] != ident: raise ValueError('Checkpoint identity differs')
        model.load_state_dict(c['model_state_dict']);opt.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler'])
        start=c['epoch']+1;best=c['best'];best_epoch=c['best_epoch'];stale=c['stale']
        generator.set_state(c['generator'].cpu());torch.set_rng_state(c['rng'].cpu())
        if rt.device.type == 'cuda' and c['cuda_rng'] and len(c['cuda_rng'])==torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in c['cuda_rng']])
    def epoch_pass(part):
        fitting=part=='train';model.train(fitting);sums=np.zeros(4);count=0
        pass_started=time.perf_counter();batches=len(loaders[part])
        for batch,(raw,av,truth,context) in enumerate(loaders[part],1):
            raw,av,truth=[x.to(rt.device) for x in (raw,av,truth)];rt.context=context
            with torch.set_grad_enabled(fitting):
                parts=training_loss(model,rt,raw,av,truth,fitting);loss=parts[0]
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                if fitting:
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True);opt.step()
            sums+=np.array([float(x.detach()) for x in parts])*len(truth);count+=len(truth)
            if batch==1 or batch%200==0 or batch==batches:
                elapsed=time.perf_counter()-pass_started
                progress=dict(event='batch_progress',epoch=epoch,partition=part,batch=batch,batches=batches,
                    samples=count,total_samples=len(loaders[part].dataset),loss=float(sums[0]/count),
                    state_loss=float(sums[1]/count),seconds=elapsed,
                    estimated_phase_seconds_remaining=elapsed*(len(loaders[part].dataset)-count)/count,
                    learning_rate=opt.param_groups[0]['lr'])
                print(json.dumps(progress),flush=True)
                write_json(folder/'status.json',dict(stage='training',complete=False,**progress))
        return dict(zip(['loss','state_loss','reconstruction_loss','physics_loss'],(sums/count).tolist()))
    completed=start-1
    if stale < cfg.patience:
        for epoch in range(start,cfg.epochs+1):
            began=time.perf_counter();tr=epoch_pass('train');va=epoch_pass('val');scheduler.step(va['state_loss'])
            if va['state_loss'] < best: best=va['state_loss'];best_epoch=epoch;stale=0
            else: stale+=1
            common=dict(identity=ident,model_state_dict=model.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,architecture=SPEC)
            if best_epoch==epoch:torch_save(folder/'best_model.pth',common)
            torch_save(folder/'last_model.pth',{**common,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                'generator':generator.get_state(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if rt.device.type=='cuda' else []})
            row=dict(epoch=epoch,train=tr,validation=va,best_epoch=best_epoch,seconds=time.perf_counter()-began)
            with (folder/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write_json(folder/'status.json',dict(stage='training',complete=False,epoch=epoch,best_epoch=best_epoch))
            print(json.dumps(row),flush=True);completed=epoch
            if stale>=cfg.patience:break
    report=dict(complete=True,epochs_completed=completed,best_epoch=best_epoch,best_validation_state_mse=best,
                parameters=sum(p.numel() for p in model.parameters()),selection_partition='val',identity=ident)
    write_json(folder/'training_summary.json',report)
    write_json(folder/'status.json',dict(stage='trained',complete=False,best_epoch=best_epoch))
    return report


def load_model(cfg, bundle=None):
    bundle=bundle or Bundle(cfg.prepared_dir);rt=Runtime(cfg,bundle)
    folder=Path(cfg.run_dir).expanduser()
    if not (folder/'best_model.pth').exists():
        raise FileNotFoundError('No best_model.pth found; train at least one complete epoch first')
    c=torch.load(folder/'best_model.pth',map_location=rt.device,weights_only=True)
    if not matches_identity(c['identity'], identity(cfg,bundle)):
        raise ValueError("Model identity differs from data/code/settings. Load this run's saved training_config.json.")
    if c.get('architecture') != SPEC:raise ValueError('Checkpoint architecture differs from the current model')
    if (folder/'identity.json').exists() and read_json(folder/'identity.json') != c['identity']:
        raise ValueError('Checkpoint identity differs from the saved run')
    model=rt.model();model.load_state_dict(c['model_state_dict']);model.eval()
    return rt,model


def predict(rt,model,raw,available):
    """Inference uses saved measurements only; Y snapshots and voltage labels are not inputs."""
    raw=np.asarray(raw);available=np.asarray(available)
    if raw.ndim!=3 or raw.shape[1:]!=(len(rt.sensors),3) or available.shape!=raw.shape[:2]:
        raise ValueError('Expected raw[T,M,3] and available[T,M]')
    if not len(raw) or not np.isfinite(raw).all() or not np.isin(available,[0,1]).all():
        raise ValueError('Invalid measurements or availability')
    results=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(raw),rt.cfg.batch_size):
            stop=start+rt.cfg.batch_size
            x=rt.input(torch.tensor(raw[start:stop],device=rt.device),torch.tensor(available[start:stop],device=rt.device))
            results.append(rt.decode(model(x)).cpu().numpy())
    result=np.concatenate(results)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite voltage prediction')
    return result


def calibrate(cfg, conditions=('clean','noisy','fdi'), coverage=.95):
    """Empirical per-node complex-error radii. No IID/time-series coverage guarantee."""
    if not 0<coverage<1:raise ValueError('coverage must be between 0 and 1')
    if not conditions or not set(conditions)<={'clean','noisy','fdi'}:raise ValueError('Invalid calibration conditions')
    torch.set_num_threads(cfg.threads);b=Bundle(cfg.prepared_dir);rt,model=load_model(cfg,b)
    folder=Path(cfg.run_dir).expanduser()/'calibration';folder.mkdir(parents=True,exist_ok=True)
    calibration_identity=dict(prepared_sha256=b.identity,checkpoint_sha256=digest(folder.parent/'best_model.pth'),coverage=coverage,conditions=list(conditions))
    if (folder/'report.json').exists():
        existing=read_json(folder/'report.json')
        if existing['identity']==calibration_identity:
            for condition,h in existing['radii_sha256'].items():
                if digest(folder/(condition+'_radii.npy'))!=h:raise ValueError('Calibration radii changed')
            return existing
        if existing['identity']['prepared_sha256'] != b.identity:
            raise ValueError('Calibration dataset changed; use a new run')
        print('Recalibrating for the current best checkpoint and evaluation settings.',flush=True)
    count=sum(len(b.indices(n)) for n in b.splits['cal'])
    for condition in conditions:
        scratch=folder/(condition+'_errors.tmp.npy')
        errors=np.lib.format.open_memmap(scratch,mode='w+',dtype=np.float64,shape=(count,len(b.nodes)))
        offset=0
        for name in b.splits['cal']:
            ix=b.indices(name)
            truth=b.array('truth',name)[ix]
            pred=predict(rt,model,b.array(condition,name)[ix],b.array('availability',name)[ix])
            errors[offset:offset+len(truth)]=abs(pred-truth);offset+=len(truth)
        errors.flush();radii=np.empty(len(b.nodes))
        for start in range(0,len(b.nodes),64):
            radii[start:start+64]=np.quantile(errors[:,start:start+64],coverage,axis=0,method='higher')
        np.save(folder/(condition+'_radii.npy'),radii);del errors;scratch.unlink()
    report=dict(identity=calibration_identity,partition='cal',snapshots=count,trajectories=len(b.splits['cal']),
        interpretation='Empirical pooled-time per-node complex-voltage error quantiles. Not simultaneous or distribution-free coverage.',
        radii_sha256={c:digest(folder/(c+'_radii.npy')) for c in conditions})
    write_json(folder/'report.json',report);print('Calibrated on held-out calibration trajectories.',flush=True)
    return report


def evaluate(cfg, conditions=('clean','noisy','fdi'), save_predictions=True):
    torch.set_num_threads(cfg.threads);b=Bundle(cfg.prepared_dir);rt,model=load_model(cfg,b)
    folder=Path(cfg.run_dir).expanduser()/'results';folder.mkdir(parents=True,exist_ok=True)
    calibration=read_json(folder.parent/'calibration/report.json')
    if calibration['identity']['prepared_sha256']!=b.identity or calibration['identity']['checkpoint_sha256']!=digest(folder.parent/'best_model.pth'):
        raise ValueError('Calibration does not match dataset and checkpoint')
    reports={}
    for condition in conditions:
        radius_path=folder.parent/'calibration'/(condition+'_radii.npy')
        if digest(radius_path)!=calibration['radii_sha256'][condition]:raise ValueError('Calibration radii changed')
        radii=np.load(radius_path);acc=new_accumulator();active_acc=new_accumulator();covered=0;count=0;active_count=0;individual=[]
        dest=folder/condition;dest.mkdir(exist_ok=True)
        regime_acc={}; family_acc={}
        for name in b.splits['test']:
            truth=b.array('truth',name);ix=b.indices(name)
            if save_predictions:
                pred=predict(rt,model,b.array(condition,name),b.array('availability',name))
                filtered_pred=pred[ix]
            else:
                pred=None
                filtered_pred=predict(rt,model,b.array(condition,name)[ix],b.array('availability',name)[ix])
            filtered_truth=truth[ix]
            update_metrics(acc,filtered_pred,filtered_truth);one=new_accumulator();update_metrics(one,filtered_pred,filtered_truth)
            covered+=int((abs(filtered_pred-filtered_truth)<=radii).sum());count+=filtered_truth.size
            active=b.array('labels',name).any(axis=(1,2)) if condition=='fdi' else np.zeros(len(truth),bool)
            active=active[ix]
            if active.any():
                update_metrics(active_acc,filtered_pred[active],filtered_truth[active]);active_count+=int(active.sum())
                kind=b.manifest['records'][name].get('fdi_kind','unspecified')
                update_metrics(family_acc.setdefault(kind,new_accumulator()),filtered_pred[active],filtered_truth[active])
            if (b.root/'metadata'/name).exists():
                mode=b.array('metadata',name)[ix]
                for value,label in enumerate(('none','oscillation','imbalance','overlap')):
                    mask=mode==value
                    if mask.any():update_metrics(regime_acc.setdefault(label,new_accumulator()),filtered_pred[mask],filtered_truth[mask])
            individual.append(dict(file=name,eligible_snapshots=len(ix),metrics=finalize_metrics(one)))
            if save_predictions:np.save(dest/name,pred)
        report=dict(partition='test',condition=condition,overall=finalize_metrics(acc),
            fdi_active=finalize_metrics(active_acc) if active_count else None,fdi_active_snapshots=active_count,
            physical_regimes={k:finalize_metrics(v) for k,v in regime_acc.items()},
            fdi_families={k:finalize_metrics(v) for k,v in family_acc.items()},
            eligible_snapshots=sum(len(b.indices(n)) for n in b.splits['test']),
            empirical_node_time_coverage=covered/count,nominal_calibration_quantile=calibration['identity']['coverage'],
            per_trajectory=individual,prediction_layout='time_nodes',prepared_sha256=b.identity,
            checkpoint_sha256=calibration['identity']['checkpoint_sha256'],calibration_sha256=digest(folder.parent/'calibration/report.json'))
        write_json(dest/'metrics.json',report);reports[condition]=report
        print(condition,report['overall'],flush=True)
    write_json(folder/'summary.json',reports)
    write_json(folder.parent/'status.json',dict(stage='complete',complete=True,conditions=list(conditions),
        training_complete=(folder.parent/'training_summary.json').exists(),
        evaluated_checkpoint_sha256=calibration['identity']['checkpoint_sha256']))
    return reports

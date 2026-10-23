"""Portable identities, atomic records, and stable training-only moments."""
from pathlib import Path
import hashlib
import json
import numpy as np


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def stable_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little')


class Moments:
    def __init__(self):
        self.count = 0

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        mu = values.mean(axis=0)
        ss = ((values - mu) ** 2).sum(axis=0)
        if not self.count:
            self.mean, self.ss, self.count = mu, ss, n
        else:
            delta = mu - self.mean
            total = self.count + n
            self.ss += ss + delta ** 2 * (self.count * n / total)
            self.mean += delta * (n / total)
            self.count = total

    def result(self, floor):
        return self.mean, np.maximum(np.sqrt(self.ss / max(1, self.count - 1)), floor)

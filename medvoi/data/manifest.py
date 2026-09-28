from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_array(path: str):
    p = Path(path)
    s = ''.join(p.suffixes).lower()
    if s.endswith('.npy'):
        return np.load(p)
    if s.endswith('.npz'):
        z = np.load(p)
        key = 'arr_0' if 'arr_0' in z else list(z.keys())[0]
        return z[key]
    if s.endswith('.pt') or s.endswith('.pth'):
        x = torch.load(p, map_location='cpu')
        if torch.is_tensor(x):
            return x.numpy()
        if isinstance(x, dict):
            for v in x.values():
                if torch.is_tensor(v):
                    return v.numpy()
        raise ValueError(f'Cannot infer tensor from {p}')
    if s.endswith('.nii') or s.endswith('.nii.gz'):
        import nibabel as nib
        return np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)
    raise ValueError(f'Unsupported file type: {p}')


def zscore_nonzero(x):
    x = np.asarray(x, dtype=np.float32)
    mask = x != 0
    if mask.any():
        mu, sd = x[mask].mean(), x[mask].std()
        sd = max(float(sd), 1e-6)
        x = x.copy()
        x[mask] = (x[mask] - mu) / sd
    return x


def _center_crop_or_pad(arr, target_shape, is_label=False):
    arr = np.asarray(arr)
    spatial = arr.shape[-len(target_shape):]
    pads = []
    for a, t in zip(spatial, target_shape):
        d = max(t - a, 0)
        pads.append((d // 2, d - d // 2))
    pad_cfg = [(0,0)] * (arr.ndim-len(target_shape)) + pads
    arr = np.pad(arr, pad_cfg, mode='constant')
    spatial = arr.shape[-len(target_shape):]
    slices = [slice(None)]*(arr.ndim-len(target_shape))
    for a, t in zip(spatial, target_shape):
        st = max((a-t)//2, 0)
        slices.append(slice(st, st+t))
    return arr[tuple(slices)]


class ManifestDataset(Dataset):
    """Manifest-driven loader for MedVOI datasets.

    Each source column points to .npy/.npz/.pt/.nii(.gz). Classification labels
    are scalar values in `label`; segmentation labels point to an array file.
    """
    def __init__(self, manifest: str, split: str, sources: List[str], task: str,
                 target_shape=None, normalize=True, label_col='label'):
        df = pd.read_csv(manifest)
        if 'split' not in df:
            raise ValueError('Manifest must contain a split column.')
        self.df = df[df['split'].astype(str) == str(split)].reset_index(drop=True)
        self.sources = sources
        self.task = task
        self.target_shape = tuple(target_shape) if target_shape else None
        self.normalize = normalize
        self.label_col = label_col
        missing = [c for c in ['patient_id', *sources, label_col] if c not in self.df.columns]
        if missing:
            raise ValueError(f'Missing manifest columns: {missing}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xs: Dict[str, torch.Tensor] = {}
        for s in self.sources:
            x = _load_array(str(row[s]))
            if self.normalize and x.ndim >= 2:
                x = zscore_nonzero(x)
            if self.target_shape and x.ndim >= len(self.target_shape):
                x = _center_crop_or_pad(x, self.target_shape)
            if x.ndim in (2,3):
                x = x[None]
            xs[s] = torch.as_tensor(x, dtype=torch.float32)
        if self.task == 'segmentation':
            y = _load_array(str(row[self.label_col]))
            if self.target_shape:
                y = _center_crop_or_pad(y, self.target_shape, is_label=True)
            # Supports integer labels or already multi-channel labels.
            y = torch.as_tensor(y)
            if y.ndim == len(self.target_shape or y.shape):
                y = y.long()
            else:
                y = y.float()
        else:
            y = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        return {'patient_id': str(row.patient_id), 'sources': xs, 'target': y}


def collate_manifest(batch):
    ids = [b['patient_id'] for b in batch]
    source_names = list(batch[0]['sources'])
    xs = {s: torch.stack([b['sources'][s] for b in batch], 0) for s in source_names}
    ys = torch.stack([b['target'] for b in batch], 0)
    return {'patient_id': ids, 'sources': xs, 'target': ys}

from __future__ import annotations
import json, os, random
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['_config_path'] = str(path)
    return cfg


def save_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: Dict[str, Any]) -> torch.device:
    requested = cfg.get('runtime', {}).get('device', 'cuda')
    if requested.startswith('cuda') and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device('cpu')


def ensure_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def subset_key(mask) -> str:
    arr = np.asarray(mask).astype(int).tolist()
    return ''.join(map(str, arr))


def parse_subset_key(key: str) -> np.ndarray:
    return np.asarray([int(x) for x in key], dtype=np.int64)


def save_checkpoint(path, model, optimizer=None, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    obj = {'model': model.state_dict(), 'extra': extra or {}}
    if optimizer is not None:
        obj['optimizer'] = optimizer.state_dict()
    torch.save(obj, path)


def load_checkpoint(path, model, optimizer=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('extra', {})

from __future__ import annotations
import itertools
import numpy as np


def all_nonempty_masks(n_sources: int):
    out = []
    for k in range(1, n_sources + 1):
        for idx in itertools.combinations(range(n_sources), k):
            m = np.zeros(n_sources, dtype=np.int64)
            m[list(idx)] = 1
            out.append(m)
    return out


def admissible_masks(n_sources: int):
    return [m for m in all_nonempty_masks(n_sources) if m.sum() < n_sources]


def candidate_indices(mask):
    mask = np.asarray(mask)
    return np.where(mask == 0)[0].tolist()


def add_candidate(mask, candidate: int):
    x = np.asarray(mask, dtype=np.int64).copy()
    x[candidate] = 1
    return x


def sample_mask(batch_size, n_sources, schedule, rng=None):
    rng = rng or np.random.default_rng()
    sizes = np.arange(1, n_sources + 1)
    probs = np.array([schedule.get(str(k), schedule.get(k, 0.0)) for k in sizes], dtype=float)
    probs = probs / probs.sum()
    masks = []
    for _ in range(batch_size):
        k = int(rng.choice(sizes, p=probs))
        idx = rng.choice(np.arange(n_sources), size=k, replace=False)
        m = np.zeros(n_sources, dtype=np.float32)
        m[idx] = 1.0
        masks.append(m)
    return np.stack(masks)

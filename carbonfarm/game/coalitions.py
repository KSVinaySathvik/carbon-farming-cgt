"""Coalition rows for the allocation stage (Sec. 4.7).

exact_game enumerates all 2^n - 2 proper coalitions. sampled_game is used for larger n and always
keeps every singleton and co-singleton. deviation_values gives vhat(T) = v(T) - sum_{i in T} g_i (Eq. 27).
"""
from __future__ import annotations

import numpy as np

from carbonfarm.model.solver import CoalitionSolver


def membership_matrix(masks: np.ndarray, n: int) -> np.ndarray:
    return ((masks[:, None] >> np.arange(n)) & 1).astype(float)


def exact_game(solver: CoalitionSolver, V: np.ndarray | None = None):
    n = solver.n
    V = solver.lattice_values() if V is None else V
    masks = np.arange(1, (1 << n) - 1, dtype=np.int64)          # proper, non-empty
    return membership_matrix(masks, n), V[masks]


def sampled_game(solver: CoalitionSolver, n_sets: int, rng: np.random.Generator):
    """Sampled proper coalitions.  Always contains all n singletons and (n > 2) all n co-singletons."""
    n = solver.n
    n_sets = int(min(n_sets, 2 ** min(n, 60) - 2))
    eye = np.eye(n, dtype=bool)
    rows = list(eye) + (list(~eye) if n > 2 else [])
    keys = {r.tobytes() for r in rows}
    while len(rows) < n_sets:
        mask = rng.integers(0, 2, size=n).astype(bool)
        if 0 < mask.sum() < n and mask.tobytes() not in keys:
            keys.add(mask.tobytes()); rows.append(mask)
    Mem = np.array(rows, dtype=bool)
    Mem = Mem[np.lexsort((Mem.argmax(1), Mem.sum(1)))]           # order by size, then first member
    return Mem.astype(float), solver.values(Mem)


def deviation_values(A: np.ndarray, v_total: np.ndarray, g_F: np.ndarray) -> np.ndarray:
    """vhat(T) = v(T) - sum_{i in T} g_i(Pi^F_i)."""
    return v_total - A @ g_F

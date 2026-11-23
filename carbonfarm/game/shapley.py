"""Exact and Monte Carlo Shapley values (Eq. 5/31).

shapley_exact is vectorised over the 2^n value lattice. shapley_mc samples permutations, optionally
antithetic (each permutation paired with its reverse). By Remark 2, phi_i(vhat) = phi_i(v) - g_i.
"""
from __future__ import annotations

import math

import numpy as np


def shapley_exact(V: np.ndarray, n: int) -> np.ndarray:
    masks = np.arange(1 << n, dtype=np.int64)
    pc = np.zeros(1 << n, dtype=np.int64)
    for i in range(n):
        pc += (masks >> i) & 1
    w = np.array([math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)])
    phi = np.zeros(n)
    for i in range(n):
        S = masks[((masks >> i) & 1) == 0]
        phi[i] = np.sum(w[pc[S]] * (V[S | (1 << i)] - V[S]))
    return phi


def shapley_mc(solver, n_perms: int, rng: np.random.Generator, antithetic: bool = True,
               batch: int = 64) -> np.ndarray:
    """Permutation-sampling estimate.  With antithetic=True, `n_perms` base permutations are used
    together with their reverses (2*n_perms marginal paths in total)."""
    n = solver.n
    phi = np.zeros(n)
    paths = 0
    done = 0
    pos = np.arange(n)[None, :, None]
    while done < n_perms:
        b = min(batch, n_perms - done)
        perms = np.array([rng.permutation(n) for _ in range(b)])
        if antithetic:
            perms = np.concatenate([perms, perms[:, ::-1]], axis=0)
        rank = np.argsort(perms, axis=1)                                   # rank[p, i] = position of i
        Mem = (rank[:, None, :] <= pos)                                    # (P, n, n): prefix k contains i
        P = perms.shape[0]
        vals = solver.values(Mem.reshape(P * n, n)).reshape(P, n)
        marg = np.diff(vals, axis=1, prepend=0.0)
        np.add.at(phi, perms.reshape(-1), marg.reshape(-1))
        paths += P
        done += b
    return phi / paths

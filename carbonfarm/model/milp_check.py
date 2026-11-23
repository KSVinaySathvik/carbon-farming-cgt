"""MILP of Sec. 4.6, used as an independent check of the enumeration solver.

The constraints and objective separate by farmer, so a coalition is solved as |S| small MILPs.
"""
from __future__ import annotations

import numpy as np
from carbonfarm.game import backend

from carbonfarm.data.instance import Instance


def _farmer_milp(inst: Instance, i: int, kappa: float):
    """max h_i over admissible portfolios satisfying (20)-(21); returns (value or None)."""
    pr, par = inst.practices, inst.params
    m = pr.m
    compat = [(j, k) for j in range(m) for k in range(j + 1, m) if pr.incompat[j, k] == 0]
    P = len(compat)
    # variable layout: x[m] | z[P] | ind[m+1] | p[P]
    ix, iz, ii, ip = 0, m, m + P, m + P + (m + 1)
    nv = ip + P
    rows, lo, hi = [], [], []

    def add(coefs: dict, l, u):
        r = np.zeros(nv)
        for k, v in coefs.items():
            r[k] += v
        rows.append(r); lo.append(l); hi.append(u)

    add({ii + s: 1.0 for s in range(m + 1)}, 1, 1)                                           # one-hot size
    add({**{ix + j: 1.0 for j in range(m)}, **{ii + s: -float(s) for s in range(m + 1)}}, 0, 0)  # sum x = s
    w = [2.0 / (s * (s - 1)) if s >= 2 else 0.0 for s in range(m + 1)]
    for q, (j, k) in enumerate(compat):
        add({iz + q: 1, ix + j: -1}, -np.inf, 0)
        add({iz + q: 1, ix + k: -1}, -np.inf, 0)
        add({iz + q: 1, ix + j: -1, ix + k: -1}, -1, np.inf)
        add({ip + q: 1, **{ii + s: -w[s] for s in range(m + 1)}}, -np.inf, 0)
        add({ip + q: 1, iz + q: -1}, -np.inf, 0)
        add({ip + q: 1, **{ii + s: -w[s] for s in range(m + 1)}, iz + q: -1}, -1, np.inf)
    for j, k in pr.incompatible_pairs():
        add({ix + j: 1, ix + k: 1}, -np.inf, 1)

    FS, B = inst.farm_size[i], inst.budget[i]
    ypp = par.yield_value_per_pct
    R = np.zeros(nv); G = np.zeros(nv)
    for j in range(m):
        R[ix + j] = FS * inst.cmp[i, j] * par.ccp
        G[ix + j] = FS * (ypp * pr.yield_pct[j] - pr.oc[j])
    for q, (j, k) in enumerate(compat):
        R[ip + q] = FS * pr.alpha[j, k] * par.ccp
        G[ip + q] = FS * (ypp * pr.gamma[j, k] - pr.beta[j, k])
    H = R + G
    add({c: v for c, v in enumerate(G) if v}, kappa - B, np.inf)                              # Eq. (20)
    add({c: v for c, v in enumerate(H) if v}, kappa, np.inf)                                  # Eq. (21)

    integrality = np.zeros(nv)
    integrality[ix:ix + m] = 1; integrality[iz:iz + P] = 1; integrality[ii:ii + m + 1] = 1
    val = backend.milp(-H, np.array(rows), lo, hi, integrality)
    return None if val is None else float(-val)


def milp_value(inst: Instance, members) -> float:
    members = sorted(members)
    area = float(inst.farm_size[members].sum())
    C = float(inst.params.cert_cost(area, len(members)))
    total = 0.0
    for i in members:
        h = _farmer_milp(inst, i, inst.farm_size[i] / area * C)
        if h is None:
            return 0.0
        total += h
    opt = total - C
    return max(0.0, opt) if opt > inst.params.feas_tol else 0.0

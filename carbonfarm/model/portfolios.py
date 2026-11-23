"""Admissible practice portfolios and per-farmer quantities (Eqs. 6-9, 13-15, 20-21, 34-37).

    R_ik  carbon revenue        FS_i * CMP_i(P_k) * CCP
    G_ik  private payoff g_i    FS_i * (dY(P_k) - OC(P_k))
    H_ik  gross benefit h_i     R_ik + G_ik
    M_ik  affordability margin  min{H_ik, B_i + G_ik}

The per-farmer constraints depend only on the farmer's own portfolio and on
kappa_i(S) = FS_i * C(S) / A(S), and the JPO objective is additive, so the JPO decomposes by farmer.
"""
from __future__ import annotations

import numpy as np

from carbonfarm.data.instance import Instance


class PortfolioTable:
    def __init__(self, inst: Instance):
        pr, par = inst.practices, inst.params
        m, n = pr.m, inst.n
        self.inst, self.m, self.n = inst, m, n
        bad = pr.incompatible_pairs()
        masks = [mk for mk in range(1 << m) if not any((mk >> j) & 1 and (mk >> k) & 1 for j, k in bad)]
        self.masks = np.array(masks, dtype=np.int64)
        X = ((self.masks[:, None] >> np.arange(m)) & 1).astype(float)
        self.X = X
        s = X.sum(1)
        w = np.where(s >= 2, 2.0 / np.maximum(s * (s - 1.0), 1.0), 0.0)    # Eq. (6): average pairwise effect
        iu, ju = np.triu_indices(m, 1)
        pair_in = X[:, iu] * X[:, ju]                                      # (K, #pairs)
        a_term = w * (pair_in @ pr.alpha[iu, ju])
        b_term = w * (pair_in @ pr.beta[iu, ju])
        g_term = w * (pair_in @ pr.gamma[iu, ju])
        oc_ha = X @ pr.oc + b_term                                         # Eq. (8)  $/ha
        yr_ha = par.yield_value_per_pct * (X @ pr.yield_pct + g_term)      # Eq. (9)  $/ha
        FS, B = inst.farm_size, inst.budget
        self.FS, self.B = FS, B
        cmp_ = inst.cmp @ X.T + a_term[None, :]                            # (n, K)  Eq. (7)
        self.seq = FS[:, None] * cmp_                                      # tCO2e
        self.R = self.seq * par.ccp                                        # Eq. (13)
        self.OC = FS[:, None] * oc_ha[None, :]
        self.YR = FS[:, None] * yr_ha[None, :]
        self.G = self.YR - self.OC                                         # Eq. (14)
        self.H = self.R + self.G                                           # Eq. (15)
        self.M = np.minimum(self.H, B[:, None] + self.G)                   # Eq. (36)/(37) in absolute $
        order = np.argsort(self.M, axis=1, kind="stable")
        self._m_asc = np.take_along_axis(self.M, order, axis=1)
        H_sorted = np.take_along_axis(self.H, order, axis=1)
        suf = np.maximum.accumulate(H_sorted[:, ::-1], axis=1)[:, ::-1]
        self._h_suffix_max = np.concatenate([suf, np.full((n, 1), -np.inf)], axis=1)

    @property
    def K(self) -> int:
        return len(self.masks)

    def portfolio_names(self, k: int) -> list:
        return [self.inst.practices.names[j] for j in range(self.m) if self.X[k, j] > 0.5]

    def best_h(self, i: int, kappa, tol: float):
        """Vectorised: max{H_ik : M_ik >= kappa - tol} for farmer i and an array of kappa values;
        -inf where no portfolio is admissible."""
        pos = np.searchsorted(self._m_asc[i], np.asarray(kappa) - tol, side="left")
        return self._h_suffix_max[i][pos]

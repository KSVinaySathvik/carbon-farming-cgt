"""Exact coalition values (Sec. 4.5-4.6, Eqs. 23-25).

    OPT(S) = sum_{i in S} max{h_i(P) : B_i + g_i(P) >= kappa_i(S), h_i(P) >= kappa_i(S)} - C(S)
    v(S)   = max{0, OPT(S)}

SPO is the singleton case.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from carbonfarm.data.instance import Instance
from carbonfarm.model.portfolios import PortfolioTable


@dataclass
class CoalitionSolution:
    members: np.ndarray
    value: float
    opt: float
    certifying: bool
    cert_cost: float
    portfolio_idx: np.ndarray = field(default_factory=lambda: np.array([], int))   # row k of PortfolioTable
    portfolios: list = field(default_factory=list)                                  # names, per member
    R: np.ndarray = None               # carbon revenue per member
    g: np.ndarray = None               # private payoff per member
    oc: np.ndarray = None
    yr: np.ndarray = None
    seq: np.ndarray = None
    kappa: np.ndarray = None           # area-proportional certification outlay per member


class CoalitionSolver:
    def __init__(self, inst: Instance, table: PortfolioTable | None = None):
        self.inst = inst
        self.table = table or PortfolioTable(inst)
        self.n = inst.n
        self.tol = inst.params.feas_tol

    def values(self, member_matrix: np.ndarray) -> np.ndarray:
        """v(S) for each row of a (B, n) boolean membership matrix."""
        Mem = np.asarray(member_matrix, dtype=bool)
        B = Mem.shape[0]
        tb, par = self.table, self.inst.params
        FS = tb.FS
        area = Mem @ FS
        cnt = Mem.sum(1)
        C = np.where(cnt > 0, par.cert_cost(np.maximum(area, 1e-300), cnt), 0.0)
        total = np.zeros(B)
        feasible = cnt > 0
        for i in range(self.n):
            rows = np.flatnonzero(Mem[:, i])
            if rows.size == 0:
                continue
            kappa = FS[i] * C[rows] / area[rows]
            best = tb.best_h(i, kappa, self.tol)
            total[rows] += best
            feasible[rows[~np.isfinite(best)]] = False
        opt = total - C
        return np.where(feasible & (opt > self.tol), opt, 0.0)

    def value(self, members) -> float:
        row = np.zeros((1, self.n), dtype=bool)
        row[0, list(members)] = True
        return float(self.values(row)[0])

    def lattice_values(self) -> np.ndarray:
        n = self.n
        if n > 22:
            raise ValueError("lattice enumeration is limited to n <= 22; use sampling for larger n")
        masks = np.arange(1 << n, dtype=np.int64)
        Mem = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)
        return self.values(Mem)

    def solo_values(self) -> np.ndarray:
        return self.values(np.eye(self.n, dtype=bool))

    def solve(self, members) -> CoalitionSolution:
        members = np.array(sorted(members), dtype=int)
        tb, par, tol = self.table, self.inst.params, self.tol
        area = float(tb.FS[members].sum())
        C = float(par.cert_cost(area, len(members)))
        kappa = tb.FS[members] / area * C
        ks, total = [], 0.0
        for i, kap in zip(members, kappa):
            feas = tb.M[i] >= kap - tol
            if not feas.any():
                return CoalitionSolution(members, 0.0, -np.inf, False, C)
            hv = np.where(feas, tb.H[i], -np.inf)
            best = hv.max()
            ks.append(int(np.argmax(hv >= best - tol)))            # smallest bitmask among ties
            total += best
        opt = total - C
        if opt <= tol:
            return CoalitionSolution(members, 0.0, opt, False, C)
        ks = np.array(ks)
        pick = lambda arr: arr[members, ks]
        return CoalitionSolution(
            members=members, value=float(opt), opt=float(opt), certifying=True, cert_cost=C,
            portfolio_idx=ks, portfolios=[tb.portfolio_names(k) for k in ks],
            R=pick(tb.R), g=pick(tb.G), oc=pick(tb.OC), yr=pick(tb.YR), seq=pick(tb.seq), kappa=kappa)

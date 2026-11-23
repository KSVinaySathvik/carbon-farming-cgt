"""Full analysis of one instance: SPO, grand-coalition JPO, coalition values, deviation game,
allocations (Shapley, core, least core, nucleolus, two baselines) and their stability.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from carbonfarm.config import EXACT_MAX_N
from carbonfarm.data.instance import Instance
from carbonfarm.game.baselines import area_proportional, attributable_revenue
from carbonfarm.game.coalitions import deviation_values, exact_game, sampled_game
from carbonfarm.game.core_solvers import (analytic_center, core_centroid, core_intervals, least_core,
                                          nucleolus)
from carbonfarm.game.shapley import shapley_exact, shapley_mc
from carbonfarm.game.stability import nan_metrics, stability_metrics
from carbonfarm.model.solver import CoalitionSolver

MECHANISMS = ("shapley", "core", "least_core", "nucleolus", "area_proportional", "attributable_revenue")
MECH_LABEL = {"shapley": "Shapley", "core": "Core (centroid)", "least_core": "Least core",
              "nucleolus": "Nucleolus", "area_proportional": "Area-proportional",
              "attributable_revenue": "Attributable-revenue"}


@dataclass
class Analysis:
    inst: Instance
    level: str
    exact: bool = True
    v_solo: np.ndarray = None
    solo_portfolios: list = field(default_factory=list)
    grand_feasible: bool = False
    v_grand: float = 0.0
    p_grand: float = 0.0
    g_F: np.ndarray = None
    R_F: np.ndarray = None
    kappa_F: np.ndarray = None
    seq_F: np.ndarray = None
    gc_portfolios: list = field(default_factory=list)
    gc_cert_cost: float = np.nan
    frac_certifying: float = np.nan
    eps_star: float = np.nan
    core_nonempty: float = np.nan
    core_thin: bool = False
    shares: dict = field(default_factory=dict)          # mechanism -> x_i (transferable share)
    stability: dict = field(default_factory=dict)       # mechanism -> metrics dict
    core_range: np.ndarray = None
    core_stderr: np.ndarray = None
    nucleolus_stages: int = 0
    remark3_rate: float = np.nan
    timings: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.inst.n

    @property
    def payoffs(self) -> dict:
        return {k: x + self.g_F for k, x in self.shares.items()}

    def allocation_frame(self) -> pd.DataFrame:
        if not self.grand_feasible:
            return pd.DataFrame()
        d = {"farm_ha": self.inst.farm_size, "avg_cmp": self.inst.cmp.mean(1),
             "v_solo": self.v_solo, "vhat_solo": self.v_solo - self.g_F, "g_F": self.g_F,
             "gc_portfolio": [" + ".join(p) for p in self.gc_portfolios]}
        for k, x in self.shares.items():
            d[f"x_{k}"] = x
        for k, u in self.payoffs.items():
            d[f"u_{k}"] = u
        return pd.DataFrame(d, index=[f"F{i+1}" for i in range(self.n)])

    def summary_row(self) -> dict:
        inst, par = self.inst, self.inst.params
        row = {"n": self.n, "seed": inst.seed, "avg_farm_ha": inst.avg_farm_size,
               "total_ha": float(inst.farm_size.sum()),
               "solo_feasible": int((self.v_solo > 0).sum()), "sum_v_solo": float(self.v_solo.sum()),
               "grand_feasible": int(self.grand_feasible), "v_N": self.v_grand, "p_N": self.p_grand,
               "G_N": float(self.v_grand - self.p_grand), "delta_v": self.v_grand - float(self.v_solo.sum())}
        if not self.grand_feasible:
            return row
        row.update({"cert_cost": self.gc_cert_cost, "cert_cost_per_ha": self.gc_cert_cost / inst.farm_size.sum(),
                    "total_seq_tCO2e": float(self.seq_F.sum()),
                    "avg_portfolio_size": float(np.mean([len(p) for p in self.gc_portfolios])),
                    "frac_coalitions_certifying": self.frac_certifying,
                    "eps_star": self.eps_star, "core_nonempty": self.core_nonempty})
        names = inst.practices.names
        for j, nm in enumerate(names):
            row[f"adopt_{j}"] = float(np.mean([nm in p for p in self.gc_portfolios]))
        for k in MECHANISMS:
            m = self.stability.get(k)
            if m is None:
                continue
            row[f"stable_{k}"] = m["is_stable"]
            row[f"viol_{k}"] = m["violations"]
            row[f"maxexcess_{k}"] = m["max_excess"]
            row[f"irviol_{k}"] = m["ir_violations"]
            if k in self.shares and np.all(np.isfinite(self.shares[k])):
                row[f"mingain_{k}"] = float(np.min(self.shares[k] + self.g_F - self.v_solo))
        return row


def analyze(inst: Instance, level: str = "full", seed: int = 0, exact_max_n: int = EXACT_MAX_N,
            mc_perms: int = 1250, mc_sets: int = 15000, core_steps: int = 30000,
            core_point: str = "centroid", core_min_width: float = 1e-3,
            solver: CoalitionSolver | None = None) -> Analysis:
    assert level in ("light", "stability", "full")
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    par = inst.params
    solver = solver or CoalitionSolver(inst)
    n = inst.n
    res = Analysis(inst=inst, level=level, exact=n <= exact_max_n)

    res.v_solo = solver.solo_values()
    if level == "full":
        res.solo_portfolios = [solver.solve([i]).portfolios[0] if res.v_solo[i] > 0 else [] for i in range(n)]

    gc = solver.solve(range(n))
    res.grand_feasible = bool(gc.certifying)
    if not gc.certifying:
        res.timings["total"] = time.perf_counter() - t0
        return res
    res.v_grand = gc.value
    res.g_F, res.R_F, res.kappa_F, res.seq_F = gc.g, gc.R, gc.kappa, gc.seq
    res.gc_portfolios, res.gc_cert_cost = gc.portfolios, gc.cert_cost
    res.p_grand = gc.value - float(gc.g.sum())
    assert abs(res.p_grand - (gc.R.sum() - gc.cert_cost)) < 1e-6 * max(1.0, abs(res.p_grand)), "p(F) != R - C"
    if level == "light":
        res.timings["total"] = time.perf_counter() - t0
        return res

    t1 = time.perf_counter()
    if res.exact:
        V = solver.lattice_values()
        assert abs(V[-1] - gc.value) < 1e-6 * max(1.0, gc.value), "lattice v(F) != JPO v(F)"
        res.frac_certifying = float((V[1:] > 0).mean())
        A, v_tot = exact_game(solver, V)
        shapley_total = shapley_exact(V, n)
    else:
        A, v_tot = sampled_game(solver, mc_sets, rng)
        shapley_total = shapley_mc(solver, mc_perms, rng, antithetic=True)
    v_hat = deviation_values(A, v_tot, res.g_F)
    single = np.flatnonzero(A.sum(1) == 1)
    for r in single:                                                          # consistency of the IR rows
        i = int(np.argmax(A[r]))
        assert abs(v_hat[r] - (res.v_solo[i] - res.g_F[i])) < 1e-6 * max(1.0, abs(v_hat[r])), "IR row mismatch"
    shapley = shapley_total - res.g_F                                         # Remark 2
    assert abs(shapley.sum() - res.p_grand) < 1e-5 * max(1.0, abs(res.p_grand)), "Shapley not efficient"
    res.timings["game"] = time.perf_counter() - t1

    res.shares["shapley"] = shapley
    res.shares["area_proportional"] = area_proportional(inst.farm_size, res.p_grand)
    res.shares["attributable_revenue"] = attributable_revenue(res.R_F, res.kappa_F)

    t2 = time.perf_counter()
    if level == "full":
        lc = least_core(res.p_grand, A, v_hat, par.stab_tol)
        res.eps_star = lc["epsilon_star"]
        res.shares["least_core"] = lc["allocation"]
    else:
        from carbonfarm.game.core_solvers import least_core_epsilon
        res.eps_star, _ = least_core_epsilon(res.p_grand, A, v_hat)
    res.core_nonempty = float(res.eps_star <= par.stab_tol)

    if level == "full":
        nuc = nucleolus(res.p_grand, A, v_hat)
        res.shares["nucleolus"] = nuc["allocation"]
        res.nucleolus_stages = nuc["n_stages"]
        if res.core_nonempty:
            res.core_range = core_intervals(res.p_grand, A, v_hat)
            if res.eps_star < -core_min_width:
                if core_point == "analytic":
                    res.shares["core"] = analytic_center(res.p_grand, A, v_hat, x0=lc["lp_point"])
                    res.core_stderr = np.zeros(n)
                else:
                    cc = core_centroid(res.p_grand, A, v_hat, x0=lc["lp_point"], rng=rng, n_steps=core_steps)
                    res.shares["core"] = cc["allocation"]
                    res.core_stderr = cc["stderr"]
            else:                                                             # core has (almost) no interior
                res.core_thin = True
                res.shares["core"] = lc["allocation"]
        else:
            res.shares["core"] = np.full(n, np.nan)
    res.timings["core"] = time.perf_counter() - t2

    for k in MECHANISMS:
        if k in res.shares and np.all(np.isfinite(res.shares[k])):
            res.stability[k] = stability_metrics(res.shares[k], res.p_grand, A, v_hat, par.stab_tol)
        elif k in res.shares:
            res.stability[k] = nan_metrics()

    if level == "full" and "least_core" in res.stability:
        ex = res.stability["least_core"]["excess"]
        bind = np.flatnonzero(ex >= ex.max() - 1e-6)[:200]
        changed = 0
        for r in bind:
            S = np.flatnonzero(A[r] > 0)
            sol = solver.solve(S)
            if sol.certifying and any(sol.portfolios[a] != res.gc_portfolios[i] for a, i in enumerate(S)):
                changed += 1
        res.remark3_rate = changed / max(1, len(bind))
    res.timings["total"] = time.perf_counter() - t0
    return res

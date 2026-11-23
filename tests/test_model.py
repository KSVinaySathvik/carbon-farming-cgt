"""Tests for the optimisation model: paper instances, MILP and brute-force agreement, lattice solver,
practice tables and pipeline sanity checks.
"""
import itertools

import numpy as np

from carbonfarm import analyze, make_instance
from carbonfarm.config import DATA_DIR, Params
from carbonfarm.data.instance import Instance
from carbonfarm.data.practices import PracticeData, build_practice_data, load_practice_csvs
from carbonfarm.game.shapley import shapley_exact, shapley_mc
from carbonfarm.model.milp_check import milp_value
from carbonfarm.model.solver import CoalitionSolver


def toy(carbon, oc, farmers, var_mrv=10.0, fix_mrv=0.0, fix_t=0.0, var_t=10.0, delta=0.5):
    """Two mutually exclusive practices, CCP = 1 (so CMP == carbon revenue in $/ha), no yield effect.
    farmers: list of (FS, B_total); carbon: per-practice list or per-farmer list of lists."""
    n, m = len(farmers), len(oc)
    carbon = np.array(carbon, float)
    cmp_ = np.tile(carbon, (n, 1)) if carbon.ndim == 1 else carbon
    pr = PracticeData(names=tuple(f"P{j+1}" for j in range(m)), cmp_base=np.zeros(m), oc=np.array(oc, float),
                      yield_pct=np.zeros(m), alpha=np.zeros((m, m)), beta=np.zeros((m, m)),
                      gamma=np.zeros((m, m)), incompat=np.array([[0, 1], [1, 0]]))
    par = Params(ccp=1.0, fix_mrv=fix_mrv, var_mrv=var_mrv, delta=delta, fix_t=fix_t, var_t=var_t,
                 budget_per_ha=1.0, yield_t_per_ha=0.0, crop_price=0.0)
    fs = np.array([f[0] for f in farmers], float)
    B = np.array([f[1] for f in farmers], float)
    return Instance(params=par, practices=pr, seed=0, farm_size=fs, budget_noise=np.zeros(n), budget_area=B,
                    cmp=cmp_)


def both(inst, S):
    a = CoalitionSolver(inst).value(S)
    b = milp_value(inst, S)
    assert abs(a - b) < 1e-6, (S, a, b)
    return a


def test_theorem2_non_convexity():
    inst = toy([100, 40], [10, 0], [(1, 25), (24, 1920), (24, 480)])
    v = lambda S: both(inst, S)
    for S, val in {(0,): 20, (0, 2): 2180, (0, 1): 2180, (0, 1, 2): 4310}.items():
        assert abs(v(S) - val) < 1e-6, (S, v(S), val)
    d_S, d_T = v((0, 2)) - v((0,)), v((0, 1, 2)) - v((0, 1))
    assert abs(d_S - 2160) < 1e-6 and abs(d_T - 2130) < 1e-6 and d_T < d_S      # convexity fails by 30
    print("Theorem 2 OK: Delta_3(S)=2160 > Delta_3(T)=2130")


def test_remark4_superadditivity_needs_H():
    carbon = [[10, 3]] + [[1000, 1000]] * 10
    inst = toy(carbon, [0.5, 0.0], [(100, 250)] + [(0.1, 20)] * 10)
    vS, vT, vQ = both(inst, (0,)), both(inst, tuple(range(1, 11))), both(inst, tuple(range(11)))
    assert abs(vS - 840) < 1e-6 and abs(vT - 890) < 1e-6 and abs(vQ - 1089.5) < 0.01, (vS, vT, vQ)
    assert vQ < vS + vT
    print(f"Remark 4 OK: v(S)={vS:.2f} v(T)={vT:.2f} v(Q)={vQ:.2f} < {vS + vT:.2f}")


def test_enumeration_vs_milp_vs_bruteforce():
    rng = np.random.default_rng(1)
    m, n = 4, 3
    sym = lambda sc: (lambda M: (M + M.T) / 2 * (1 - np.eye(m)))(rng.normal(0, sc, (m, m)))
    incompat = np.zeros((m, m), int); incompat[0, 3] = incompat[3, 0] = 1
    alpha, beta, gamma = sym(0.2), sym(20), sym(3)
    for M in (alpha, beta, gamma):
        M[0, 3] = M[3, 0] = 0.0
    pr = PracticeData(tuple("ABCD"), np.ones(m), rng.uniform(-10, 60, m), rng.uniform(-10, 15, m),
                      alpha, beta, gamma, incompat)
    par = Params(ccp=60.0, fix_mrv=100.0, fix_t=50.0, budget_per_ha=1.0)
    fs, B = np.array([0.8, 1.4, 0.3]), np.array([300.0, 500.0, 80.0])
    inst = Instance(par, pr, 0, fs, np.zeros(n), B, rng.uniform(0.5, 3, (n, m)))

    def brute(S):
        A = fs[list(S)].sum()
        C = par.fix_mrv + par.var_mrv * A ** par.delta + par.fix_t + par.var_t * len(S)
        tot = 0.0
        for i in S:
            kap, best = fs[i] / A * C, None
            for mask in range(1 << m):
                P = [j for j in range(m) if mask >> j & 1]
                if 0 in P and 3 in P:
                    continue
                s = len(P); w = 2 / (s * (s - 1)) if s >= 2 else 0.0
                pr_ = [(j, k) for a, j in enumerate(P) for k in P[a + 1:]]
                cmp_ = sum(inst.cmp[i, j] for j in P) + w * sum(alpha[j, k] for j, k in pr_)
                oc = sum(pr.oc[j] for j in P) + w * sum(beta[j, k] for j, k in pr_)
                dy = par.yield_value_per_pct * (sum(pr.yield_pct[j] for j in P) + w * sum(gamma[j, k] for j, k in pr_))
                R, g = fs[i] * cmp_ * par.ccp, fs[i] * (dy - oc)
                if B[i] + g >= kap - 1e-9 and R + g >= kap - 1e-9:
                    best = R + g if best is None else max(best, R + g)
            if best is None:
                return 0.0
            tot += best
        return max(0.0, tot - C)

    solver = CoalitionSolver(inst)
    for r in range(1, n + 1):
        for S in itertools.combinations(range(n), r):
            a, b, c = solver.value(S), brute(S), milp_value(inst, S)
            assert abs(a - b) < 1e-6 and abs(a - c) < 1e-5, (S, a, b, c)
    print("enumeration == brute force == scipy MILP on all coalitions (w = 2/(s(s-1)))")


def test_lattice_equals_single_solves():
    for seed in (42, 43):
        inst = make_instance(15, seed)
        sv = CoalitionSolver(inst)
        V = sv.lattice_values()
        rng = np.random.default_rng(seed)
        for mask in rng.integers(1, 1 << 15, 200):
            S = [i for i in range(15) if mask >> i & 1]
            sol = sv.solve(S)
            assert abs(sol.value - V[mask]) < 1e-8, (mask, sol.value, V[mask])
    print("lattice == per-coalition solver for panels 42, 43")


def test_data_determinism_and_csv():
    a, b = build_practice_data(), load_practice_csvs(DATA_DIR / "Dataset_USD")
    for f in ("cmp_base", "oc", "yield_pct", "alpha", "beta", "gamma", "incompat"):
        assert np.allclose(getattr(a, f), getattr(b, f), atol=1e-12), f
    i1, i2 = make_instance(15, 42), make_instance(15, 42)
    assert np.array_equal(i1.farm_size, i2.farm_size) and np.array_equal(i1.cmp, i2.cmp)
    assert i1.farm_size.min() >= 0.10 and i1.farm_size.max() <= 2.0 and np.abs(i1.budget_noise).max() <= i1.params.budget_noise_clip + 1e-12
    print("practice tables == CSV; panels deterministic and within bounds")


def test_pipeline_sanity_and_mc_shapley():
    inst = make_instance(15, 42)
    res = analyze(inst, level="stability")
    assert res.grand_feasible and abs(res.shares["shapley"].sum() - res.p_grand) < 1e-6
    for k in ("area_proportional", "attributable_revenue"):
        assert abs(res.shares[k].sum() - res.p_grand) < 1e-6
    sv = CoalitionSolver(inst)
    phi = shapley_exact(sv.lattice_values(), 15)
    mc = shapley_mc(sv, 1500, np.random.default_rng(0), antithetic=True)
    assert abs(mc.sum() - phi.sum()) < 1e-6                                       # MC Shapley is exactly efficient
    assert np.abs(mc - phi).max() < 0.05 * np.abs(phi).max(), np.abs(mc - phi).max()
    print("pipeline efficiency OK; MC Shapley within 5% of exact")


if __name__ == "__main__":
    test_theorem2_non_convexity(); test_remark4_superadditivity_needs_H(); test_enumeration_vs_milp_vs_bruteforce()
    test_data_determinism_and_csv(); test_lattice_equals_single_solves()
    test_pipeline_sanity_and_mc_shapley()
    print("ALL PASSED")

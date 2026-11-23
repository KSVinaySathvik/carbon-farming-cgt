"""Superadditivity and convexity checks over all 2^n coalitions (Sec. 5, Remarks 5 and 7).

    python -m experiments.check_structure [--n 15 --seeds 5]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from carbonfarm import make_instance
from carbonfarm.config import PANEL_SEEDS
from carbonfarm.model.solver import CoalitionSolver

TOL = 1e-7
ROOT = Path(__file__).resolve().parents[1]


def superadditivity(V, feas):
    """Return (violations, tested pairs, merges that are infeasible) over disjoint feasible pairs."""
    F = np.flatnonzero(feas)
    viol = tested = lost = 0
    for S in F:
        T = F[((F & S) == 0) & (F > S)]                       # each unordered pair once
        if len(T) == 0:
            continue
        U = S | T
        ok = feas[U]
        lost += int((~ok).sum())
        tested += int(ok.sum())
        viol += int((V[U][ok] < V[S] + V[T][ok] - TOL).sum())
    return viol, tested, lost


def convexity(V, n):
    """Return (violations, tested inequalities) of v(S+i+j) - v(S+j) >= v(S+i) - v(S)."""
    masks = np.arange(1 << n)
    cv = ct = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            S = masks[((masks >> i) & 1 == 0) & ((masks >> j) & 1 == 0)]
            lhs = V[S | (1 << i) | (1 << j)] - V[S | (1 << j)]
            rhs = V[S | (1 << i)] - V[S]
            ct += len(S)
            cv += int((lhs < rhs - TOL).sum())
    return cv, ct


def nonpooling_rates():
    f = ROOT / "results" / "ablations" / "a6_allocation_comparison" / "panel_meta.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    return dict(zip(d["seed"], d["remark3_portfolio_change_rate"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=len(PANEL_SEEDS), help="number of panels (of the 10)")
    a = ap.parse_args(argv)
    n, N = a.n, 1 << a.n
    npool = nonpooling_rates()
    rows = []
    for seed in PANEL_SEEDS[:a.seeds]:
        V = CoalitionSolver(make_instance(n, seed)).lattice_values()
        feas = V > 0
        sv, st, sl = superadditivity(V, feas)
        cv, ct = convexity(V, n)
        rows.append(dict(seed=seed, n=n, v_N=V[-1], feasible=int(feas.sum()), coalitions=N - 1,
                         superadd_violations=sv, superadd_tested=st, merges_infeasible=sl,
                         convexity_violations=cv, convexity_tested=ct, convexity_pct=100 * cv / ct,
                         nonpooling_rate=npool.get(seed, np.nan)))
        r = rows[-1]
        print(f"seed {seed}: v(N)={r['v_N']:,.1f} | feasible {r['feasible']}/{N-1} | "
              f"superadd violations {sv}/{st} (+{sl} merges infeasible) | "
              f"convexity violations {cv}/{ct} ({r['convexity_pct']:.2f}%) | non-pooling rate {r['nonpooling_rate']}")
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "structure"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"structure_n{n}.csv", index=False)
    print("\nLaTeX rows (seed & v(N) & feasible & superadd. violations (tested) & convexity violations):")
    for r in rows:
        print(f"{r['seed']} & {r['v_N']:,.0f} & {r['feasible']:,} / {r['coalitions']:,} & "
              f"{r['superadd_violations']} ({r['superadd_tested']:,} pairs) & "
              f"{r['convexity_pct']:.2f}\\% ({r['convexity_violations']:,} of {r['convexity_tested']:,}) \\\\")


if __name__ == "__main__":
    main()

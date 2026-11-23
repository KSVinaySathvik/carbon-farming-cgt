"""Ablation 5: carbon credit price sweep and exact break-even prices.

    python -m experiments.ablations.a5_carbon_price
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from carbonfarm import make_instance
from carbonfarm.config import PANEL_SEEDS, RESULTS_DIR, Params
from carbonfarm.data.practices import build_practice_data
from carbonfarm.model.solver import CoalitionSolver
from experiments.ablations.common import add_common_args, apply_solver, ABL_DIR, default_jobs, load_sweep, md_table, paper_table, run_sweep
from experiments.ablations.plotting import OKABE, sweep_figure

NAME, KEY = "a5_carbon_price", "ccp"
GRID = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500]


def apply(inst, value):
    return inst.with_params(ccp=float(value))


def _ok(inst, members, ccp, criterion):
    i2 = inst.with_params(ccp=float(ccp))
    sv = CoalitionSolver(i2)
    if criterion == "certifies":
        return sv.value(members) > 0
    sol = sv.solve(members)
    return bool(sol.certifying and sol.R.sum() - sol.cert_cost > 0)


def breakeven_ccp(inst, members, criterion="certifies", lo=0.5, hi=1e5, n_scan=120):
    """Smallest CCP satisfying the criterion (np.inf if none <= hi): geometric scan, then bisection."""
    grid = np.geomspace(lo, hi, n_scan)
    ok = [_ok(inst, members, c, criterion) for c in grid]
    if not any(ok):
        return np.inf
    k = int(np.argmax(ok))
    a, b = (grid[k - 1] if k > 0 else 0.0), grid[k]
    for _ in range(40):
        mid = 0.5 * (a + b)
        a, b = (a, mid) if _ok(inst, members, mid, criterion) else (mid, b)
    return float(b)


def _breakeven_job(seed):
    """who: grand_certifies   v(N) > 0                (private gains alone may carry it)
            grand_pool_pos    p(N) > 0                (carbon revenue alone covers C(N))
            solo              v({i}) > 0 at the baseline budget
            solo_no_budget    v({i}) > 0 if the budget constraint (20) were slack (b-bar = 1e6)"""
    inst = make_instance(15, seed)
    rich = inst.with_budget_per_ha(1e6)
    allf = range(inst.n)
    rows = [{"seed": seed, "who": "grand_certifies", "farmer": -1, "farm_ha": float(inst.farm_size.sum()),
             "ccp_breakeven": breakeven_ccp(inst, allf)},
            {"seed": seed, "who": "grand_pool_pos", "farmer": -1, "farm_ha": float(inst.farm_size.sum()),
             "ccp_breakeven": breakeven_ccp(inst, allf, "pool_positive")}]
    for i in allf:
        rows.append({"seed": seed, "who": "solo", "farmer": i, "farm_ha": float(inst.farm_size[i]),
                     "ccp_breakeven": breakeven_ccp(inst, [i])})
        rows.append({"seed": seed, "who": "solo_no_budget", "farmer": i, "farm_ha": float(inst.farm_size[i]),
                     "ccp_breakeven": breakeven_ccp(rich, [i])})
    return rows


def run_breakeven(seeds, n_jobs=None):
    n_jobs = n_jobs or default_jobs()
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        rows = [r for rs in ex.map(_breakeven_job, seeds) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(ABL_DIR / NAME / "breakeven.csv", index=False)
    return df


def plot_all(summ, be):
    out = RESULTS_DIR / "figures" / f"{NAME}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sweep_figure(summ, KEY, "Carbon credit price CCP ($/tCO2e)", out, baseline=Params().ccp, logx=True,
                 practice_names=build_practice_data().short)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    sty = {"solo_no_budget": (OKABE[1], "solo, budget constraint relaxed"), "grand_pool_pos": (OKABE[0], "grand coalition: p(N) > 0")}
    for who, (c, lab) in sty.items():
        d = be[be.who == who]
        fin = d.ccp_breakeven.replace(np.inf, np.nan)
        axs[0].scatter(d.farm_ha, fin, s=16, alpha=0.7, color=c, label=lab)
        v = np.sort(fin.dropna().to_numpy())
        axs[1].step(v, np.arange(1, len(v) + 1) / len(d), where="post", color=c, lw=2, label=lab)
    nsolo = (be[be.who == "solo"].ccp_breakeven == np.inf).mean()
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("area (ha): farm, or total for coalition")
    axs[0].set_ylabel("break-even CCP ($/tCO2e)"); axs[0].axhline(Params().ccp, color="grey", ls=":"); axs[0].legend(frameon=False)
    axs[0].set_title(f"price needed (solo at baseline budget: never viable for {100*nsolo:.0f}% of farmers)")
    axs[1].set_xscale("log"); axs[1].axvline(Params().ccp, color="grey", ls=":"); axs[1].set_xlabel("CCP ($/tCO2e)")
    axs[1].set_ylabel("share of farmers / panels viable"); axs[1].legend(frameon=False); axs[1].set_title("viability vs carbon price")
    out2 = RESULTS_DIR / "figures" / "a5_breakeven.png"
    fig.savefig(out2); plt.close(fig)
    return out, out2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(ap)
    a = ap.parse_args(argv)
    apply_solver(a)
    seeds = PANEL_SEEDS[:a.seeds]
    if a.plot_only:
        _, summ, _ = load_sweep(NAME); be = pd.read_csv(ABL_DIR / NAME / "breakeven.csv")
    else:
        _, summ = run_sweep(NAME, KEY, GRID, apply, level="stability", seeds=seeds, n_jobs=a.jobs,
                            description=__doc__.split("\n\n")[0].strip())
        be = run_breakeven(seeds, a.jobs)
    print(md_table(paper_table(summ, KEY, "CCP ($/tCO2e)")))
    print("\nbreak-even CCP ($/tCO2e), median [min, max] over panels/farmers:")
    for who, d in be.groupby("who"):
        v = d.ccp_breakeven
        print(f"  {who:<16s} {v.median():10.1f}  [{v.min():.1f}, {v.max():.1f}]   never viable: {100*(v == np.inf).mean():.0f}%")
    print("figures:", *plot_all(summ, be))


if __name__ == "__main__":
    main()

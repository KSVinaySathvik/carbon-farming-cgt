"""Ablation 3 (Sec. 7.2, Table 4): the four certification cost components, one at a time.

C(S) = Fix_MRV + Var_MRV * A(S)^delta + Fix_T + Var_T * |S|  (Eqs. 10-12).

    python -m experiments.ablations.a3_cost_components
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from carbonfarm.config import PANEL_SEEDS, RESULTS_DIR, Params
from carbonfarm.data.practices import build_practice_data
from experiments.ablations.common import add_common_args, apply_solver, load_sweep, md_table, paper_table, run_sweep
from experiments.ablations.plotting import OKABE, sweep_figure

COMPONENTS = {   # key: (label, grid, paper grid)
    "fix_mrv": ("Fixed MRV Fix_MRV ($)", [0, 500, 1000, 2000, 3000, 4000, 6000, 8000], [0, 1000, 2000, 4000]),
    "var_mrv": ("Variable MRV Var_MRV ($/ha^delta)", [0, 20, 45, 65, 80, 120, 160, 240], [0, 45, 65, 80]),
    "fix_t": ("Fixed transaction Fix_T ($)", [0, 250, 500, 1000, 2000, 4000, 6000, 8000], []),
    "var_t": ("Variable transaction Var_T ($/farmer)", [0, 5, 10, 20, 40, 80, 160, 320], []),
}


class _Apply:
    """Picklable callable: set one cost parameter."""
    def __init__(self, key):
        self.key = key

    def __call__(self, inst, value):
        return inst.with_params(**{self.key: float(value)})


def plot_one(key, summ):
    label, _, paper = COMPONENTS[key]
    out = RESULTS_DIR / "figures" / f"a3_{key}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sweep_figure(summ, key, label, out, baseline=getattr(Params(), key), paper_values=paper,
                 practice_names=build_practice_data().short)
    return out


def plot_overlay(summaries):
    base = Params()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for (key, summ), c in zip(summaries.items(), OKABE):
        mult = summ[key] / getattr(base, key)
        lab = COMPONENTS[key][0].split(" (")[0]
        axs[0].plot(mult, summ["v_N"], marker="o", ms=3, color=c, label=lab)
        axs[1].plot(mult, summ["solo_feasible"], marker="o", ms=3, color=c, label=lab)
        axs[2].plot(mult, summ["cert_cost_per_ha"] if "cert_cost_per_ha" in summ else summ["v_N"] * np.nan,
                    marker="o", ms=3, color=c, label=lab)
    axs[0].set_title("v(N) vs multiple of baseline value"); axs[0].set_ylabel("$ per season")
    axs[1].set_title("solo-feasible farmers (of 15)")
    axs[2].set_title("certification cost per ha at N ($/ha)")
    for a in axs:
        a.set_xlabel("component value / baseline value"); a.axvline(1, color="grey", ls=":"); a.legend(frameon=False, fontsize=8)
    out = RESULTS_DIR / "figures" / "a3_overlay.png"
    fig.savefig(out); plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(ap)
    a = ap.parse_args(argv)
    apply_solver(a)
    summaries = {}
    for key, (label, grid, _) in COMPONENTS.items():
        name = f"a3_{key}"
        if a.plot_only:
            _, summ, _ = load_sweep(name)
        else:
            _, summ = run_sweep(name, key, grid, _Apply(key), level="stability", seeds=PANEL_SEEDS[:a.seeds],
                                n_jobs=a.jobs, description=__doc__.split("\n\n")[0].strip())
        summaries[key] = summ
        print(f"\n== {label} ==")
        print(md_table(paper_table(summ, key, key)))
        plot_one(key, summ)
    print("overlay figure:", plot_overlay(summaries))


if __name__ == "__main__":
    main()

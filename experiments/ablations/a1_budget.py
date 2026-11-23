"""Ablation 1 (Sec. 7.1, Table 2): mean budget per hectare.

    python -m experiments.ablations.a1_budget [--seeds 3 --jobs 2 --plot-only]
"""
from __future__ import annotations

import argparse

from carbonfarm.config import PANEL_SEEDS, RESULTS_DIR, Params
from carbonfarm.data.practices import build_practice_data
from experiments.ablations.common import add_common_args, apply_solver, load_sweep, md_table, paper_table, run_sweep
from experiments.ablations.plotting import sweep_figure

NAME, KEY = "a1_budget", "budget_per_ha"
GRID = [0, 25, 50, 75, 100, 140, 200, 400, 600, 1000]
PAPER_GRID = [140, 200, 400, 600]


def apply(inst, value):
    return inst.with_budget_per_ha(value)


def plot(summ):
    out = RESULTS_DIR / "figures" / f"{NAME}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sweep_figure(summ, KEY, "Mean budget b-bar ($/ha)", out, baseline=Params().budget_per_ha,
                 paper_values=PAPER_GRID, practice_names=build_practice_data().short)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(ap)
    a = ap.parse_args(argv)
    apply_solver(a)
    if a.plot_only:
        _, summ, _ = load_sweep(NAME)
    else:
        _, summ = run_sweep(NAME, KEY, GRID, apply, level="stability", seeds=PANEL_SEEDS[:a.seeds], n_jobs=a.jobs,
                            description=__doc__.split("\n\n")[0].strip())
    print(md_table(paper_table(summ, KEY, "budget ($/ha)")))
    print("figure:", plot(summ))


if __name__ == "__main__":
    main()

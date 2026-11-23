"""Ablation 2 (Sec. 7.1, Table 3): farm areas scaled by k, budget per hectare fixed.

    python -m experiments.ablations.a2_farm_scale [--seeds 3 --jobs 2 --plot-only]
"""
from __future__ import annotations

import argparse

from carbonfarm.config import PANEL_SEEDS, RESULTS_DIR
from carbonfarm.data.practices import build_practice_data
from experiments.ablations.common import add_common_args, apply_solver, load_sweep, md_table, paper_table, run_sweep
from experiments.ablations.plotting import sweep_figure

NAME, KEY = "a2_farm_scale", "scale"
GRID = [0.5, 1, 2, 3, 5, 7.5, 10, 15, 20]
PAPER_GRID = [1, 3, 5, 10]


def apply(inst, value):
    return inst.with_scaled_farms(value, scale_budget=True)


def plot(summ):
    out = RESULTS_DIR / "figures" / f"{NAME}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sweep_figure(summ, KEY, "Farm-size scale factor k (x baseline area)", out, baseline=1, logx=True,
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
    print(md_table(paper_table(summ, KEY, "scale k")))
    print("figure:", plot(summ))


if __name__ == "__main__":
    main()

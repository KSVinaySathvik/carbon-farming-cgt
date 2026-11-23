"""Ablation 4 (Sec. 7.3, Table 5): MRV scale exponent delta, and population size n (Sec. 7.5).

    python -m experiments.ablations.a4_mrv_scaling
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from carbonfarm.config import PANEL_SEEDS, RESULTS_DIR, Params
from carbonfarm.data.practices import build_practice_data
from experiments.ablations.common import add_common_args, apply_solver, load_sweep, md_table, paper_table, run_sweep
from experiments.ablations.plotting import OKABE, sweep_figure

DELTA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DELTA_PAPER = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
N_GRID = [5, 10, 15, 25, 50, 100, 250, 500, 1000]


def apply_delta(inst, value):
    return inst.with_params(delta=float(value))


def identity(inst, value):
    return inst


def plot_delta(summ):
    out = RESULTS_DIR / "figures" / "a4_delta.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sweep_figure(summ, "delta", "MRV scale exponent delta", out, baseline=Params().delta, paper_values=DELTA_PAPER,
                 practice_names=build_practice_data().short)
    return out


def plot_population(summ):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    x = summ["n"]
    axs[0].plot(x, summ["v_N"] / x, marker="o", color=OKABE[0]); axs[0].set_title("v(N) per farmer ($)")
    axs[1].plot(x, summ["cert_cost_per_ha"], marker="o", color=OKABE[1]); axs[1].set_title("certification cost per ha ($/ha)")
    axs[2].plot(x, summ["solo_feasible"] / x, marker="o", color=OKABE[2]); axs[2].set_title("share of farmers solo-feasible")
    axs[3].plot(x, summ["p_N"] / x, marker="o", color=OKABE[3], label="p(N)/n"); axs[3].plot(x, summ["G_N"] / x, marker="o", color=OKABE[4], label="G(N)/n")
    axs[3].legend(frameon=False); axs[3].set_title("pooled vs private surplus per farmer ($)")
    for a in axs:
        a.set_xscale("log"); a.set_xlabel("number of farmers n")
    out = RESULTS_DIR / "figures" / "a4_population.png"
    fig.savefig(out); plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(ap)
    a = ap.parse_args(argv)
    apply_solver(a)
    seeds = PANEL_SEEDS[:a.seeds]
    desc = __doc__.split("\n\n")[0].strip()
    if a.plot_only:
        _, s1, _ = load_sweep("a4_delta"); _, s2, _ = load_sweep("a4_population")
    else:
        _, s1 = run_sweep("a4_delta", "delta", DELTA_GRID, apply_delta, level="stability", seeds=seeds, n_jobs=a.jobs,
                          description=desc)
        # population scaling: vary n (the panel itself changes), grand coalition only
        rows = []
        for n in N_GRID:
            r, _ = run_sweep(f"population_n{n}", "n_req", [n], identity, level="light", seeds=seeds, n=n,
                             n_jobs=a.jobs, save=False)
            rows.append(r)
        raw = pd.concat(rows, ignore_index=True)
        from experiments.ablations.common import ABL_DIR, summarize
        raw["n"] = raw["n_req"]
        s2 = summarize(raw.drop(columns=["is_baseline"]), "n_req")
        out = ABL_DIR / "a4_population"; out.mkdir(parents=True, exist_ok=True)
        raw.to_csv(out / "raw.csv", index=False); s2.to_csv(out / "summary.csv", index=False)
        import json
        (out / "meta.json").write_text(json.dumps({"name": "a4_population", "values": N_GRID, "seeds": list(seeds),
                                                   "level": "light", "description": desc}, indent=2))
    s2 = s2.assign(n=s2["n_req"])
    print("== delta ==");  print(md_table(paper_table(s1, "delta", "delta")))
    print("\n== population scaling ==")
    print(md_table(s2[["n", "solo_feasible", "v_N", "p_N", "G_N", "cert_cost_per_ha"]]))
    print("figures:", plot_delta(s1), plot_population(s2))


if __name__ == "__main__":
    main()

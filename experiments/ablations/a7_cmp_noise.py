"""Ablation 7: stability of the sharing rules as the farmer-specific sequestration noise grows.

    python -m experiments.ablations.a7_cmp_noise
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm

from carbonfarm import analyze, make_instance
from carbonfarm.config import PANEL_SEEDS, RESULTS_DIR, Params
from carbonfarm.model.solver import CoalitionSolver
from experiments.ablations.common import (ABL_DIR, add_common_args, apply_solver, load_sweep, md_table,
                                          run_sweep)
from experiments.ablations.plotting import OKABE

NAME, KEY = "a7_cmp_noise", "cmp_noise"
GRID = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]       # must stay < 1: CMP_ij = CMP_j (1 + U(-a, a)) >= 0
CCP_LEVELS = [60, 150, 300]
CONTROL_GRID = [0.0, 0.2, 0.5, 0.8, 0.95]
MECHS = [("area_proportional", "Area-proportional", OKABE[1]), ("shapley", "Shapley", OKABE[2]),
         ("attributable_revenue", "Attributable-revenue", OKABE[3])]
OUT = ABL_DIR / NAME


def apply(inst, value):
    """Regenerate the same panel (same seed) with a different CMP noise amplitude."""
    assert 0.0 <= value < 1.0, "cmp_noise must be in [0, 1)"
    return make_instance(inst.n, inst.seed, inst.params.replace(cmp_noise=float(value)), inst.practices)


def apply_budget_noise(inst, value):
    return make_instance(inst.n, inst.seed, inst.params.replace(budget_noise_sd=float(value),
                                                                 budget_noise_clip=float(value)), inst.practices)


def unstable_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, label, _ in MECHS:
        if f"viol_{k}" not in raw:
            continue
        sub = raw[raw[f"stable_{k}"] == 0]
        for r in sub.itertuples():
            rows.append({"ccp": r.ccp, "cmp_noise": r.cmp_noise, "seed": r.seed, "mechanism": label,
                         "violations": int(getattr(r, f"viol_{k}")), "max_excess": getattr(r, f"maxexcess_{k}")})
    cols = ["ccp", "cmp_noise", "seed", "mechanism", "violations", "max_excess"]
    return pd.DataFrame(rows, columns=cols).sort_values(cols[:4]).reset_index(drop=True)


def blocker_table(ccps, noise, seeds, tol: float = 1e-6) -> pd.DataFrame:
    """Worst blocking coalition of each unstable (ccp, panel, mechanism) at the given noise level."""
    rows = []
    for c in ccps:
        for s in seeds:
            inst = make_instance(15, int(s), Params(ccp=float(c), cmp_noise=float(noise)))
            res = analyze(inst, level="stability")
            if not res.grand_feasible:
                continue
            avg_cmp = inst.cmp.mean(1)
            for k, label, _ in MECHS:
                m = res.stability.get(k)
                if m is None or m["max_excess"] <= tol:
                    continue
                mask = int(np.argmax(m["excess"])) + 1
                mem = np.flatnonzero((mask >> np.arange(inst.n)) & 1)
                xT = float(res.shares[k][mem].sum())
                rows.append({"ccp": c, "cmp_noise": noise, "seed": int(s), "mechanism": label,
                             "n_blocking": m["violations"], "coalition": " ".join(f"F{i+1}" for i in mem),
                             "size": len(mem), "area_ha": float(inst.farm_size[mem].sum()),
                             "share_of_area": float(inst.farm_size[mem].sum() / inst.farm_size.sum()),
                             "mean_cmp_members": float(avg_cmp[mem].mean()), "mean_cmp_panel": float(avg_cmp.mean()),
                             "vhat_T": xT + m["max_excess"], "x_T": xT, "excess": m["max_excess"]})
    return pd.DataFrame(rows)


def fig_curves(summs: dict, out):
    ccps = list(summs)
    fig, axs = plt.subplots(2, len(ccps), figsize=(4.6 * len(ccps), 7), constrained_layout=True, squeeze=False)
    for j, c in enumerate(ccps):
        s = summs[c]; x = s[KEY].to_numpy(float)
        ax = axs[0, j]
        for k, lab, col in MECHS:
            if f"stable_{k}" in s:
                ax.plot(x, 1 - s[f"stable_{k}"], marker="o", ms=3.5, lw=1.7, color=col, label=lab)
        ax.plot(x, 1 - s["core_nonempty"], "k--", lw=1.2, label="core empty")
        ax.axvline(Params().cmp_noise, color="grey", ls=":", lw=1)
        ax.set_ylim(-0.03, 1.05); ax.set_title(f"CCP = {c} $/tCO2e"); ax.set_ylabel("fraction of panels unstable")
        if j == 0:
            ax.legend(frameon=False, fontsize=7)
        ax = axs[1, j]
        for k, lab, col in MECHS:
            if f"viol_{k}" in s:
                ax.plot(x, s[f"viol_{k}"], marker="o", ms=3.5, lw=1.7, color=col, label=lab)
        ax.set_yscale("symlog", linthresh=1); ax.set_ylabel("mean # blocking coalitions (symlog)")
        ax.axvline(Params().cmp_noise, color="grey", ls=":", lw=1); ax.set_xlabel("cmp_noise (CMP_ij = CMP_j(1+U(-a,a)))")
    fig.suptitle("Ablation 7 - instability vs farmer-specific CMP noise (dotted = paper baseline 0.20)")
    fig.savefig(out); plt.close(fig)


def fig_panels(raws: dict, out, seeds):
    ccps = list(raws)
    mechs = [m for m in MECHS if m[0] in ("area_proportional", "shapley")]
    fig, axs = plt.subplots(len(mechs), len(ccps), figsize=(4.6 * len(ccps), 3.9 * len(mechs)),
                            constrained_layout=True, squeeze=False)
    norm = SymLogNorm(linthresh=10, vmin=-1000, vmax=1000, base=10)
    for i, (k, lab, _) in enumerate(mechs):
        for j, c in enumerate(ccps):
            r = raws[c]
            ex = r.pivot(index="seed", columns=KEY, values=f"maxexcess_{k}").reindex(index=seeds)
            nv = r.pivot(index="seed", columns=KEY, values=f"viol_{k}").reindex(index=seeds, columns=ex.columns)
            ax = axs[i, j]
            im = ax.imshow(ex.to_numpy(float), cmap="RdBu_r", norm=norm, aspect="auto")
            ax.set_xticks(range(ex.shape[1])); ax.set_xticklabels([f"{v:g}" for v in ex.columns], fontsize=7)
            ax.set_yticks(range(ex.shape[0])); ax.set_yticklabels(ex.index, fontsize=7)
            for (a, b), v in np.ndenumerate(nv.to_numpy(float)):
                if v > 0:
                    ax.text(b, a, f"{int(v)}", ha="center", va="center", fontsize=6.5, color="k", fontweight="bold")
            ax.grid(False); ax.set_title(f"{lab}, CCP={c}", fontsize=9)
            ax.set_xlabel("cmp_noise"); ax.set_ylabel("panel seed")
    fig.colorbar(im, ax=axs, shrink=0.8, label="max excess  max_T [vhat(T) - x(T)]   ($; red > 0 = unstable; numbers = # blocking coalitions)")
    fig.suptitle("Ablation 7 - which panels are unstable")
    fig.savefig(out); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(ap)
    ap.add_argument("--ccp", type=float, nargs="+", default=CCP_LEVELS, help="carbon prices to cross with the noise grid")
    ap.add_argument("--detail-noise", type=float, default=0.95, help="noise level for blockers.csv")
    ap.add_argument("--no-control", action="store_true", help="skip the budget-noise control sweep")
    a = ap.parse_args(argv)
    apply_solver(a)
    seeds = list(PANEL_SEEDS[:a.seeds])
    ccps = [int(c) if float(c).is_integer() else c for c in a.ccp]
    OUT.mkdir(parents=True, exist_ok=True)
    fig_dir = RESULTS_DIR / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    doc = __doc__.split("\n\n")[0].strip()
    raws, summs = {}, {}
    for c in ccps:
        nm = f"{NAME}/ccp{c}"
        if a.plot_only:
            raws[c], summs[c], _ = load_sweep(nm)
        else:
            raws[c], summs[c] = run_sweep(nm, KEY, GRID, apply, level="stability", seeds=seeds, n_jobs=a.jobs,
                                          params=Params(ccp=float(c)), description=doc)
        raws[c]["ccp"] = c
    raw_all = pd.concat(raws.values(), ignore_index=True)
    uns = unstable_table(raw_all)
    if not a.plot_only:
        uns.to_csv(OUT / "unstable.csv", index=False)
        blk = blocker_table(ccps, a.detail_noise, seeds)
        blk.to_csv(OUT / "blockers.csv", index=False)
        if not a.no_control:
            c0 = 150 if 150 in ccps else ccps[0]
            run_sweep(f"{NAME}/control_budget_noise", "budget_noise_sd", CONTROL_GRID, apply_budget_noise,
                      level="stability", seeds=seeds, n_jobs=a.jobs,
                      params=Params(ccp=float(c0), cmp_noise=0.95),
                      description=f"CONTROL: budget noise sd=clip varied at cmp_noise=0.95, ccp={c0}")
    fig_curves(summs, fig_dir / "a7_instability_curves.png")
    fig_panels(raws, fig_dir / "a7_instability_panels.png", seeds)
    show = ["ccp", KEY] + [f"{p}_{k}" for k, _, _ in MECHS for p in ("viol", "stable") if f"{p}_{k}" in raw_all]
    agg = raw_all.groupby(["ccp", KEY])[show[2:]].mean().reset_index()
    print(md_table(agg, "{:,.2f}"))
    print(f"\n{len(uns)} unstable (ccp, noise, panel, mechanism) cells -> {OUT / 'unstable.csv'}")
    print("figures:", fig_dir / "a7_instability_curves.png", fig_dir / "a7_instability_panels.png")


if __name__ == "__main__":
    main()

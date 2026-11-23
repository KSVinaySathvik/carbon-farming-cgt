"""Ablation 6 (Sec. 7.4, Tables 6-7): Shapley, core centroid, least core, nucleolus and the two
baselines on the 10 benchmark panels (n = 15, exact).

    python -m experiments.ablations.a6_allocation_comparison
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from carbonfarm import analyze, make_instance
from carbonfarm.config import PANEL_SEEDS, RESULTS_DIR
from carbonfarm.pipeline import MECH_LABEL, MECHANISMS
from experiments.ablations.common import add_common_args, apply_solver, ABL_DIR, default_jobs, md_table
from experiments.ablations.plotting import OKABE

NAME = "a6_allocation_comparison"
OUT = ABL_DIR / NAME
COLORS = dict(zip(MECHANISMS, [OKABE[0], OKABE[7], OKABE[5], OKABE[2], OKABE[1], OKABE[3]]))


def _gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum())) if x.sum() > 0 else np.nan


def _job(args):
    seed, core_steps, core_point = args
    inst = make_instance(15, seed)
    res = analyze(inst, level="full", seed=seed, core_steps=core_steps, core_point=core_point)
    if not res.grand_feasible:
        return seed, None
    frame = res.allocation_frame().assign(seed=seed)
    rows = []
    for k in MECHANISMS:
        m = res.stability.get(k)
        if m:
            rows.append({"seed": seed, "mechanism": k, "stable": m["is_stable"], "violations": m["violations"],
                         "max_excess": m["max_excess"], "ir_violations": m["ir_violations"],
                         "min_gain_over_solo": float(np.min(res.shares[k] + res.g_F - res.v_solo)) if np.all(np.isfinite(res.shares[k])) else np.nan})
    rng = None if res.core_range is None else pd.DataFrame(res.core_range, columns=["x_min", "x_max"])
    se = None if res.core_stderr is None else res.core_stderr
    return seed, {"frame": frame, "stab": pd.DataFrame(rows), "core_range": rng, "core_se": se,
                  "eps_star": res.eps_star, "p_N": res.p_grand, "v_N": res.v_grand, "core_thin": res.core_thin,
                  "farm": inst.farm_size, "cmp": inst.cmp.mean(1), "shares": res.shares, "g": res.g_F,
                  "remark3": res.remark3_rate}


def aggregate(results):
    OUT.mkdir(parents=True, exist_ok=True)
    stab, dist, eq, meta = [], [], [], []
    for seed, r in results:
        if r is None:
            continue
        r["frame"].to_csv(OUT / f"allocation_seed{seed}.csv")
        if r["core_range"] is not None:
            r["core_range"].assign(stderr_centroid=r["core_se"]).to_csv(OUT / f"core_range_seed{seed}.csv", index=False)
        stab.append(r["stab"])
        sh, p = r["shares"], r["p_N"]
        valid = [k for k in MECHANISMS if k in sh and np.all(np.isfinite(sh[k]))]
        for a in valid:
            for b in valid:
                dist.append({"seed": seed, "a": a, "b": b, "l1_pct_of_pF": 100 * np.abs(sh[a] - sh[b]).sum() / abs(p)})
        half = r["farm"] <= np.median(r["farm"])
        for k in valid:
            u = sh[k] + r["g"]
            eq.append({"seed": seed, "mechanism": k, "gini_u": _gini(u),
                       "corr_x_farm_size": float(np.corrcoef(sh[k], r["farm"])[0, 1]),
                       "corr_x_cmp": float(np.corrcoef(sh[k], r["cmp"])[0, 1]),
                       "share_of_pF_to_smaller_half": float(sh[k][half].sum() / p),
                       "cv_payoff_per_ha": float(np.std(u / r["farm"]) / np.mean(u / r["farm"]))})
        meta.append({"seed": seed, "v_N": r["v_N"], "p_N": p, "eps_star": r["eps_star"], "core_thin": r["core_thin"],
                     "remark3_portfolio_change_rate": r["remark3"]})
    stab = pd.concat(stab, ignore_index=True); stab.to_csv(OUT / "stability.csv", index=False)
    ssum = stab.groupby("mechanism").agg(frac_stable=("stable", "mean"), mean_violations=("violations", "mean"),
                                         mean_max_excess=("max_excess", "mean"), mean_ir_violations=("ir_violations", "mean"),
                                         mean_min_gain_over_solo=("min_gain_over_solo", "mean"),
                                         n_panels=("seed", "count")).reindex(MECHANISMS).reset_index()
    ssum.to_csv(OUT / "stability_summary.csv", index=False)
    dist = pd.DataFrame(dist).groupby(["a", "b"]).l1_pct_of_pF.mean().unstack().reindex(index=MECHANISMS, columns=MECHANISMS)
    dist.to_csv(OUT / "distance.csv")
    eq = pd.DataFrame(eq); eq.to_csv(OUT / "equity_per_seed.csv", index=False)
    eq.groupby("mechanism").mean(numeric_only=True).drop(columns="seed").reindex(MECHANISMS).to_csv(OUT / "equity.csv")
    pd.DataFrame(meta).to_csv(OUT / "panel_meta.csv", index=False)
    return ssum, dist


def table6(seed=42):
    """Paper Table 6 analogue: final net payoffs u_i = x_i + g_i for one panel."""
    f = pd.read_csv(OUT / f"allocation_seed{seed}.csv", index_col=0)
    cols = {"farm_ha": "Size (ha)", "avg_cmp": "Avg CMP"}
    t = f[list(cols)].rename(columns=cols)
    for k in MECHANISMS:
        t[MECH_LABEL[k]] = f[f"u_{k}"]
    return t.sort_values("Size (ha)")


def plot_all(seed=42):
    f = pd.read_csv(OUT / f"allocation_seed{seed}.csv", index_col=0).sort_values("farm_ha")
    cr = OUT / f"core_range_seed{seed}.csv"
    rng = pd.read_csv(cr) if cr.exists() else None
    order = [int(i[1:]) - 1 for i in f.index]
    out = []
    fig, axs = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    w = 0.13
    for ax, pref, lab in ((axs[0], "x", "transferable share x_i ($)"), (axs[1], "u", "final payoff u_i = x_i + g_i ($)")):
        for j, k in enumerate(MECHANISMS):
            ax.bar(np.arange(len(f)) + (j - 2.5) * w, f[f"{pref}_{k}"], w, color=COLORS[k], label=MECH_LABEL[k])
        if rng is not None:
            lo, hi = rng.x_min.to_numpy()[order], rng.x_max.to_numpy()[order]
            if pref == "u":
                lo, hi = lo + f.g_F.to_numpy(), hi + f.g_F.to_numpy()
            ax.vlines(np.arange(len(f)) - 2.5 * w - 0.07, lo, hi, color="grey", lw=1.2, alpha=0.7,
                      label="core range (min-max)")
        ax.set_xticks(np.arange(len(f))); ax.set_xticklabels([f"{i}\n{s:.2f} ha" for i, s in zip(f.index, f.farm_ha)], fontsize=7)
        ax.set_ylabel(lab); ax.axhline(0, color="k", lw=0.6)
    axs[0].legend(ncol=7, frameon=False, fontsize=8, loc="upper left")
    axs[0].set_title(f"Allocation of the pooled carbon surplus, panel seed {seed} (farmers sorted by size)")
    p1 = RESULTS_DIR / "figures" / "a6_allocation_seed42.png"; p1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p1); plt.close(fig); out.append(p1)

    st = pd.read_csv(OUT / "stability.csv"); ss = pd.read_csv(OUT / "stability_summary.csv").set_index("mechanism")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    ks = list(MECHANISMS)
    axs[0].bar(range(6), [ss.loc[k, "frac_stable"] for k in ks], color=[COLORS[k] for k in ks]); axs[0].set_ylim(0, 1.05)
    axs[0].set_title("fraction of panels where allocation is stable (no blocking coalition)")
    axs[1].bar(range(6), [ss.loc[k, "mean_violations"] for k in ks], color=[COLORS[k] for k in ks]); axs[1].set_title("mean # blocking coalitions")
    bp = axs[2].boxplot([st[st.mechanism == k].max_excess.dropna() for k in ks], patch_artist=True, widths=0.6)
    for patch, k in zip(bp["boxes"], ks):
        patch.set_facecolor(COLORS[k]); patch.set_alpha(0.7)
    axs[2].axhline(0, color="k", lw=0.8); axs[2].set_title("max coalition excess ($, <0 = stable)")
    for a in axs:
        a.set_xticks(range(1 if a is axs[2] else 0, 7 if a is axs[2] else 6)); a.set_xticklabels([MECH_LABEL[k] for k in ks], rotation=30, ha="right", fontsize=8)
    p2 = RESULTS_DIR / "figures" / "a6_stability.png"; fig.savefig(p2); plt.close(fig); out.append(p2)

    d = pd.read_csv(OUT / "distance.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    im = ax.imshow(d.to_numpy(float), cmap="viridis_r")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels([MECH_LABEL[k] for k in d.columns], rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels([MECH_LABEL[k] for k in d.index], fontsize=8)
    for i in range(6):
        for j in range(6):
            v = d.to_numpy(float)[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="w" if v > np.nanmax(d.to_numpy(float)) / 2 else "k", fontsize=8)
    ax.grid(False); fig.colorbar(im, label="mean L1 distance, % of p(F)"); ax.set_title("how different are the allocations?")
    p3 = RESULTS_DIR / "figures" / "a6_distance.png"; fig.savefig(p3); plt.close(fig); out.append(p3)

    allf = pd.concat([pd.read_csv(p, index_col=0) for p in sorted(OUT.glob("allocation_seed*.csv"))])
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for k in MECHANISMS:
        col = f"u_{k}"
        if col in allf and allf[col].notna().all():
            ax.scatter(allf.farm_ha, allf[col] / allf.farm_ha, s=9, alpha=0.5, color=COLORS[k], label=MECH_LABEL[k])
    ax.set_xlabel("farm size (ha)"); ax.set_ylabel("final payoff per hectare ($/ha)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("per-hectare payoff vs farm size (all panels)")
    p4 = RESULTS_DIR / "figures" / "a6_payoff_per_ha.png"; fig.savefig(p4); plt.close(fig); out.append(p4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(ap)
    ap.add_argument("--core-steps", type=int, default=30000, help="hit-and-run steps for the core centroid")
    ap.add_argument("--core-point", choices=["centroid", "analytic"], default="centroid")
    a = ap.parse_args(argv)
    apply_solver(a)
    if not a.plot_only:
        seeds = PANEL_SEEDS[:a.seeds]
        with ProcessPoolExecutor(max_workers=a.jobs or default_jobs()) as ex:
            results = list(ex.map(_job, [(s, a.core_steps, a.core_point) for s in seeds]))
        ssum, dist = aggregate(results)
    else:
        ssum = pd.read_csv(OUT / "stability_summary.csv"); dist = pd.read_csv(OUT / "distance.csv", index_col=0)
    print("== Table 6 analogue: final payoffs u_i, panel seed 42 ==")
    print(md_table(table6().round(2)))
    print("\n== Table 7 analogue: stability (mean over panels) ==")
    print(md_table(ssum.assign(mechanism=ssum.mechanism.map(MECH_LABEL))))
    print("\n== mean L1 distance between allocations (% of p(F)) ==")
    print(md_table(dist.round(1).reset_index().rename(columns={"a": ""}), "{:,.1f}"))
    print("figures:", *plot_all())


if __name__ == "__main__":
    main()

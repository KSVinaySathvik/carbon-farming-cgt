"""Six-panel summary figure for a one-parameter sweep (means over panels, +-1 sd)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110, "savefig.dpi": 150})


def _band(ax, x, summ, col, label, color, ls="-"):
    if col not in summ:
        return
    y, sd = summ[col].to_numpy(float), summ.get(col + "_sd", 0 * summ[col]).fillna(0).to_numpy(float)
    ax.plot(x, y, ls, color=color, marker="o", ms=3, lw=1.6, label=label)
    ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.13, lw=0)


def _mark(ax, baseline, paper_values, x):
    if baseline is not None:
        ax.axvline(baseline, color="grey", ls=":", lw=1)
    if paper_values is not None:
        ymin = ax.get_ylim()[0]
        ax.plot([v for v in paper_values if v in set(x)], [ymin] * len([v for v in paper_values if v in set(x)]),
                "^", color="grey", ms=4, clip_on=False)


def sweep_figure(summ, key, xlabel, out, baseline=None, paper_values=None, logx=False, practice_names=None,
                 title=None, n_farmers=15):
    x = summ[key].to_numpy(float)
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    ax = axs[0, 0]
    _band(ax, x, summ, "v_N", "v(N) total surplus", OKABE[0])
    _band(ax, x, summ, "p_N", "p(N) pooled (transferable)", OKABE[2])
    _band(ax, x, summ, "G_N", "G(N) private gains", OKABE[4])
    _band(ax, x, summ, "sum_v_solo", "sum of solo values", OKABE[1], "--")
    ax.set_ylabel("$ per season"); ax.set_title("(a) value of cooperation"); ax.legend(frameon=False, fontsize=7)
    ax = axs[0, 1]
    _band(ax, x, summ, "solo_feasible", "solo-feasible farmers", OKABE[1]); ax.set_ylim(-0.3, n_farmers + 0.5)
    ax.set_ylabel(f"# farmers (of {n_farmers})"); ax.set_title("(b) solo participation feasible")
    ax = axs[0, 2]
    _band(ax, x, summ, "delta_v", "v(N) - sum v({i})", OKABE[3]); ax.set_ylabel("$ per season")
    ax.set_title("(c) cooperative gain")
    ax = axs[1, 0]
    names = practice_names or [f"P{j}" for j in range(8)]
    for j, nm in enumerate(names):
        c = f"adopt_{j}"
        if c in summ:
            ax.plot(x, summ[c], marker="o", ms=2.5, lw=1.4, color=OKABE[j % 8], ls="-" if j < 8 else "--", label=nm)
    ax.set_ylim(-0.03, 1.03); ax.set_ylabel("share of farmers"); ax.set_title("(d) practice adoption in grand coalition")
    ax.legend(frameon=False, fontsize=7, ncol=4)
    ax = axs[1, 1]
    for col, lab, c, ls in [("grand_feasible", "grand coalition certifies", OKABE[7], "-"),
                            ("core_nonempty", "core non-empty (LC/nucleolus stable)", OKABE[0], "-"),
                            ("stable_shapley", "Shapley stable", OKABE[2], "--"),
                            ("stable_area_proportional", "area-proportional stable", OKABE[1], "--"),
                            ("stable_attributable_revenue", "attrib.-revenue stable", OKABE[3], "--")]:
        if col in summ:
            ax.plot(x, summ[col], ls, color=c, marker="o", ms=3, lw=1.5, label=lab)
    ax.set_ylim(-0.03, 1.05); ax.set_ylabel("fraction of panels"); ax.set_title("(e) feasibility and stability")
    ax.legend(frameon=False, fontsize=7)
    ax = axs[1, 2]
    _band(ax, x, summ, "eps_star", "least-core eps*", OKABE[0]); ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("$  (<0: core non-empty)"); ax.set_title("(f) least-core epsilon*")
    for a in axs.ravel():
        a.set_xlabel(xlabel)
        if logx:
            a.set_xscale("symlog" if (x <= 0).any() else "log")
        _mark(a, baseline, None, x)
    fig.suptitle(title or f"Sensitivity to {xlabel} (mean +- sd over {int(summ['n_panels'].iloc[0])} panels; dotted line = baseline)")
    fig.savefig(out); plt.close(fig)
    return out

"""Build results/SUMMARY.md from the saved ablation outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from carbonfarm.config import RESULTS_DIR
from carbonfarm.pipeline import MECH_LABEL
from experiments.ablations.common import ABL_DIR, load_sweep, md_table, paper_table

SWEEPS = [("a1_budget", "budget_per_ha", "Ablation 1 - budget ($/ha)", "a1_budget.png"),
          ("a2_farm_scale", "scale", "Ablation 2 - farm-size scale k", "a2_farm_scale.png"),
          ("a3_fix_mrv", "fix_mrv", "Ablation 3a - fixed MRV ($)", "a3_fix_mrv.png"),
          ("a3_var_mrv", "var_mrv", "Ablation 3b - variable MRV ($/ha^delta)", "a3_var_mrv.png"),
          ("a3_fix_t", "fix_t", "Ablation 3c - fixed transaction ($)", "a3_fix_t.png"),
          ("a3_var_t", "var_t", "Ablation 3d - variable transaction ($/farmer)", "a3_var_t.png"),
          ("a4_delta", "delta", "Ablation 4 - MRV scale exponent delta", "a4_delta.png"),
          ("a5_carbon_price", "ccp", "Ablation 5 - carbon credit price ($/tCO2e)", "a5_carbon_price.png")]


def main(argv=None):
    L = ["# Results summary", "", "All numbers are means over the benchmark panels (n = 15 farmers, exact). "
         "Dollar values are per season. See each figure for +-1 sd bands.", ""]
    for name, key, title, fig in SWEEPS:
        if not (ABL_DIR / name / "summary.csv").exists():
            continue
        _, summ, meta = load_sweep(name)
        t = paper_table(summ, key, key)
        L += [f"## {title}", f"![{name}](figures/{fig})", "", md_table(t), ""]
    pop = ABL_DIR / "a4_population" / "summary.csv"
    if pop.exists():
        p = pd.read_csv(pop)
        L += ["## Ablation 4b - population scaling (number of farmers)", "![pop](figures/a4_population.png)", "",
              md_table(p[["n_req", "solo_feasible", "v_N", "p_N", "G_N", "cert_cost_per_ha"]]), ""]
    be = ABL_DIR / "a5_carbon_price" / "breakeven.csv"
    if be.exists():
        b = pd.read_csv(be)
        g = b.groupby("who").ccp_breakeven.agg(median="median", min="min", max="max",
                                                 never=lambda v: 100 * (v == np.inf).mean()).reset_index()
        L += ["## Ablation 5b - break-even carbon prices ($/tCO2e)", "![be](figures/a5_breakeven.png)", "", md_table(g), ""]
    a6 = ABL_DIR / "a6_allocation_comparison"
    if (a6 / "stability_summary.csv").exists():
        s = pd.read_csv(a6 / "stability_summary.csv"); s["mechanism"] = s.mechanism.map(MECH_LABEL)
        d = pd.read_csv(a6 / "distance.csv", index_col=0).round(1).reset_index().rename(columns={"a": ""})
        e = pd.read_csv(a6 / "equity.csv").round(3); e["mechanism"] = e.mechanism.map(MECH_LABEL)
        t6 = pd.read_csv(a6 / "allocation_seed42.csv", index_col=0).sort_values("farm_ha")
        cols = ["farm_ha", "avg_cmp"] + [f"u_{k}" for k in MECH_LABEL]
        t6 = t6[cols].round(2); t6.columns = ["Size (ha)", "Avg CMP"] + list(MECH_LABEL.values())
        L += ["## Ablation 6 - allocation comparison at the baseline", "![a6](figures/a6_allocation_seed42.png)", "",
              "### Final payoffs u_i = x_i + g_i, panel seed 42 (Table 6 analogue)", md_table(t6), "",
              "### Stability (Table 7 analogue, mean over panels)", "![s](figures/a6_stability.png)", "", md_table(s), "",
              "### Mean L1 distance between allocations, % of p(F)", md_table(d, "{:,.1f}"), "",
              "### Equity indicators", md_table(e, "{:,.3f}"), ""]
    a7 = ABL_DIR / "a7_cmp_noise"
    if (a7 / "unstable.csv").exists():
        parts = [pd.read_csv(d / "raw.csv").assign(ccp=int(d.name[3:])) for d in sorted(a7.glob("ccp*"))]
        raw7 = pd.concat(parts, ignore_index=True)
        cols = [c for c in ("viol_area_proportional", "stable_area_proportional", "viol_shapley", "stable_shapley",
                            "viol_attributable_revenue") if c in raw7]
        t7 = raw7.groupby(["ccp", "cmp_noise"])[cols].mean().reset_index()
        t7.columns = ["CCP", "cmp_noise"] + [c.replace("viol_", "# blocking ").replace("stable_", "frac stable ")
                                              .replace("_", " ") for c in cols]
        L += ["## Ablation 7 - instability vs CMP noise (and carbon price)",
              "![a7c](figures/a7_instability_curves.png)", "![a7p](figures/a7_instability_panels.png)", "",
              "Means over the panels. Core / least core / nucleolus stayed non-empty in every run. "
              "Per-panel unstable cells: `results/ablations/a7_cmp_noise/unstable.csv`; worst blocking coalition "
              "per unstable panel: `blockers.csv`.", "", md_table(t7, "{:,.2f}"), ""]
        ctl = a7 / "control_budget_noise" / "summary.csv"
        if ctl.exists():
            c7 = pd.read_csv(ctl)[["budget_noise_sd", "viol_shapley", "viol_area_proportional", "eps_star", "v_N"]]
            L += ["### Control - budget noise (sd = clip) at cmp_noise 0.95", md_table(c7, "{:,.2f}"), ""]
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(L))
    print("wrote", RESULTS_DIR / "SUMMARY.md")


if __name__ == "__main__":
    main()

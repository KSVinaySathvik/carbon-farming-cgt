"""Run the full pipeline on one farmer panel. n <= 16 is exact; larger n uses sampled coalitions
and antithetic Monte Carlo Shapley.

    python -m experiments.run_baseline
    python -m experiments.run_baseline --n 250 --mc-sets 15000 --mc-perms 1250
    python -m experiments.run_baseline --ccp 100 --fix-mrv 1000 --solver highs
"""
from __future__ import annotations

import argparse
import json
import os

from carbonfarm import analyze, make_instance
from carbonfarm.config import RESULTS_DIR, Params
from carbonfarm.game import backend
from carbonfarm.pipeline import MECH_LABEL
from experiments.ablations.common import md_table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=15); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--level", choices=["light", "stability", "full"], default="full")
    ap.add_argument("--solver", choices=["auto", "gurobi", "highs"], default="auto")
    ap.add_argument("--mc-sets", type=int, default=15000); ap.add_argument("--mc-perms", type=int, default=1250)
    ap.add_argument("--core-steps", type=int, default=30000)
    for k, v in Params().to_dict().items():
        if k.endswith("_tol"):
            continue
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    a = ap.parse_args(argv)
    os.environ["CARBONFARM_SOLVER"] = backend.set_backend(a.solver)
    params = Params(**{k: getattr(a, k) for k in Params().to_dict() if not k.endswith("_tol")})
    inst = make_instance(a.n, a.seed, params)
    res = analyze(inst, level=a.level, seed=a.seed, mc_sets=a.mc_sets, mc_perms=a.mc_perms, core_steps=a.core_steps)
    row = res.summary_row()
    print(f"solver={backend.get_backend()}  n={a.n}  solo-feasible={row['solo_feasible']}  v(N)={row['v_N']:,.2f}  "
          f"p(N)={row['p_N']:,.2f}  G(N)={row['G_N']:,.2f}  eps*={row.get('eps_star', float('nan')):,.3f}")
    if not res.grand_feasible:
        print("grand coalition does not certify under these parameters"); return
    fr = res.allocation_frame()
    if a.n <= 40:
        show = fr[["farm_ha", "v_solo", "g_F"] + [f"u_{k}" for k in res.shares]].round(2)
        show.columns = ["farm_ha", "v_solo", "g_F"] + [MECH_LABEL[k] for k in res.shares]
        print("\nFinal payoffs u_i = x_i + g_i\n" + md_table(show.reset_index()))
    print("\nStability (blocking-coalition test on all rows)")
    print(md_table(__import__("pandas").DataFrame(
        [{"mechanism": MECH_LABEL[k], "stable": m["is_stable"], "violations": m["violations"],
          "max_excess": m["max_excess"], "IR violations": m["ir_violations"]} for k, m in res.stability.items()])))
    out = RESULTS_DIR / "baseline" / f"N{a.n}_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    fr.to_csv(out / "allocation.csv")
    (out / "summary.json").write_text(json.dumps({k: (v if not hasattr(v, "item") else v.item()) for k, v in row.items()}, indent=2, default=str))
    print("saved to", out)


if __name__ == "__main__":
    main()

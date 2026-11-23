"""Parameter sweeps over the benchmark panels and table helpers shared by the ablations."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from carbonfarm import analyze, make_instance
from carbonfarm.config import PANEL_SEEDS, RESULTS_DIR, Params

ABL_DIR = RESULTS_DIR / "ablations"


def add_common_args(ap):
    ap.add_argument("--seeds", type=int, default=len(PANEL_SEEDS), help="number of panels (of the 10)")
    ap.add_argument("--jobs", type=int, default=None, help="parallel worker processes")
    ap.add_argument("--solver", choices=["auto", "gurobi", "highs"], default="auto",
                    help="LP/QP/MILP back-end (default: Gurobi if available, else HiGHS)")
    ap.add_argument("--plot-only", action="store_true", help="re-plot from saved CSVs")
    return ap


def apply_solver(args) -> str:
    from carbonfarm.game import backend
    name = backend.set_backend(args.solver)
    os.environ["CARBONFARM_SOLVER"] = name          # inherited by worker processes
    print(f"[solver back-end: {name}]")
    return name


def default_jobs() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _job(args):
    apply_fn, key, value, seed, n, level, params, kw = args
    inst = make_instance(n, seed, params)
    inst = apply_fn(inst, value)
    row = analyze(inst, level=level, **kw).summary_row()
    return {key: value, **row}


def run_sweep(name: str, key: str, values, apply_fn, level: str = "stability", seeds=PANEL_SEEDS,
              n: int = 15, n_jobs: int | None = None, params: Params | None = None,
              analyze_kwargs: dict | None = None, description: str = "", save: bool = True):
    """Evaluate `apply_fn(instance, value)` for every value and every panel seed."""
    params = params or Params()
    jobs = [(apply_fn, key, v, int(s), n, level, params, analyze_kwargs or {}) for s in seeds for v in values]
    n_jobs = n_jobs or default_jobs()
    t0 = time.perf_counter()
    if n_jobs == 1:
        rows = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_job, jobs, chunksize=max(1, len(jobs) // (4 * n_jobs))))
    raw = pd.DataFrame(rows)
    base_val = getattr(params, key, None)
    raw["is_baseline"] = (raw[key] == base_val) if base_val is not None else False
    summ = summarize(raw, key)
    if save:
        out = ABL_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        raw.to_csv(out / "raw.csv", index=False)
        summ.to_csv(out / "summary.csv", index=False)
        (out / "meta.json").write_text(json.dumps({
            "name": name, "description": description, "key": key, "values": list(map(float, values)),
            "seeds": list(map(int, seeds)), "n_farmers": n, "level": level, "baseline": params.to_dict(),
            "runtime_s": round(time.perf_counter() - t0, 1)}, indent=2))
    print(f"[{name}] {len(jobs)} runs in {time.perf_counter() - t0:.1f}s")
    return raw, summ


def summarize(raw: pd.DataFrame, key: str) -> pd.DataFrame:
    num = raw.select_dtypes(include=[np.number, bool]).drop(columns=["seed", key, "is_baseline"], errors="ignore")
    g = raw.groupby(key)
    mean = g[num.columns].mean()
    sd = g[num.columns].std(ddof=1).add_suffix("_sd")
    out = pd.concat([mean, sd], axis=1).reset_index()
    out["n_panels"] = g.size().values
    return out


def load_sweep(name: str):
    d = ABL_DIR / name
    return pd.read_csv(d / "raw.csv"), pd.read_csv(d / "summary.csv"), json.loads((d / "meta.json").read_text())


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return "" if np.isnan(x) else floatfmt.format(x)
        return str(x)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def paper_table(summ: pd.DataFrame, key: str, key_label: str, n: int = 15) -> pd.DataFrame:
    """Columns of Tables 2-5 of the paper (+ coalition surplus split and core status)."""
    t = pd.DataFrame({
        key_label: summ[key],
        "avg size (ha)": summ["avg_farm_ha"],
        f"solo-feasible (of {n})": summ["solo_feasible"],
        "sum v({i})": summ["sum_v_solo"],
        "v(N)": summ["v_N"],
        "dv = v(N) - sum v({i})": summ["delta_v"],
        "p(N)": summ["p_N"],
        "grand feasible": summ["grand_feasible"],
    })
    if "core_nonempty" in summ:
        t["core non-empty"] = summ["core_nonempty"]
    if "stable_shapley" in summ:
        t["Shapley stable"] = summ["stable_shapley"]
    return t

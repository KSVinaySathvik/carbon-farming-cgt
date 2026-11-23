"""Run the tests, all ablations and write results/SUMMARY.md.

    python -m experiments.run_all [--quick --solver highs --jobs 2]
"""
from __future__ import annotations

import argparse
import importlib

from experiments import make_report

STEPS = ["a1_budget", "a2_farm_scale", "a3_cost_components", "a4_mrv_scaling", "a5_carbon_price",
         "a6_allocation_comparison", "a7_cmp_noise"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--solver", default="auto")
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args(argv)
    common = ["--solver", a.solver] + (["--jobs", str(a.jobs)] if a.jobs else []) + (["--seeds", "3"] if a.quick else [])
    for name in STEPS:
        extra = ["--core-steps", "8000"] if (a.quick and name.startswith("a6")) else []
        print(f"\n######## {name} ########")
        importlib.import_module(f"experiments.ablations.{name}").main(common + extra)
    make_report.main([])


if __name__ == "__main__":
    main()

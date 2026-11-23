"""Error of plain vs antithetic Monte Carlo Shapley against the exact value.

    python -m experiments.diagnostics_mc_shapley --n 12 --trials 30 --paths 200
"""
import argparse

import numpy as np

from carbonfarm import make_instance
from carbonfarm.game.shapley import shapley_exact, shapley_mc
from carbonfarm.model.solver import CoalitionSolver


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=30); ap.add_argument("--paths", type=int, default=200)
    a = ap.parse_args(argv)
    sv = CoalitionSolver(make_instance(a.n, a.seed))
    exact = shapley_exact(sv.lattice_values(), a.n)
    std, ant = [], []
    for t in range(a.trials):
        std.append(np.mean(np.abs(shapley_mc(sv, a.paths, np.random.default_rng(t), antithetic=False) - exact)))
        ant.append(np.mean(np.abs(shapley_mc(sv, a.paths // 2, np.random.default_rng(t), antithetic=True) - exact)))
    print(f"n={a.n}, {a.paths} marginal paths, {a.trials} trials")
    print(f"standard   MAE {np.mean(std):8.3f} $  (sd {np.std(std):.3f}, max {np.max(std):.3f})")
    print(f"antithetic MAE {np.mean(ant):8.3f} $  (sd {np.std(ant):.3f}, max {np.max(ant):.3f})")


if __name__ == "__main__":
    main()

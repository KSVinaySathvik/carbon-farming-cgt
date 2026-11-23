# Cooperative Game Theory for Smallholder Carbon Farming

Code for the paper *A Cooperative Game Theoretic Approach to Sustainable, Smallholder Carbon Farming* (Game Theory Lab, IISc).

Smallholder farms are usually too small to pay for carbon certification on their own. This project models farmers pooling their land into a coalition that is certified together. Each coalition chooses a portfolio of carbon-farming practices for its members (joint portfolio optimization), and the pooled carbon surplus is then shared using cooperative game theory: the Shapley value, the core, the least core and the nucleolus, compared against area-proportional and attributable-revenue baselines. For up to 16 farmers every coalition is solved exactly; larger populations use sampled coalitions and antithetic Monte Carlo Shapley estimates.

## Layout

```
carbonfarm/
  config.py              baseline parameters (Table 1), seeds, paths
  data/                  practice tables (App. B.1-B.3), farmer panels (App. B.5), Instance
  model/                 portfolio enumeration, exact coalition solver (SPO / JPO / v(S)), MILP check
  game/                  Shapley, least core, nucleolus, core geometry, baselines, stability
  pipeline.py            analyze(instance) -> shares, payoffs, stability
experiments/
  run_baseline.py        one farmer panel, any n
  ablations/             a1 budget, a2 farm scale, a3 cost components, a4 MRV scaling,
                         a5 carbon price, a6 allocation comparison, a7 CMP noise
  run_all.py             tests + all ablations + results/SUMMARY.md
  check_structure.py     superadditivity and convexity over all coalitions
  diagnostics_mc_shapley.py
tests/                   paper instances, solver unit tests
data/Dataset_USD/        practice tables as CSV
results/                 outputs of the runs in the paper (CSV, figures, SUMMARY.md)
```

## Setup

```bash
pip install -r requirements.txt
```

Gurobi is the default solver. Without a licence the code falls back to HiGHS (SciPy + CVXPY). The size-limited pip licence of Gurobi cannot hold the n = 15 least-core LPs (32,766 rows); those models switch to HiGHS with a warning.

## Usage

```bash
python -m pytest tests -q                       # a few seconds
python -m experiments.run_baseline              # panel 42, n = 15
python -m experiments.ablations.a1_budget       # a single ablation
python -m experiments.run_all                   # everything, writes results/SUMMARY.md
```

Solver choice: `--solver gurobi|highs|auto` or the `CARBONFARM_SOLVER` environment variable. Ablations also take `--seeds k` (first k of the 10 panels), `--jobs` and `--plot-only`.

## Ablations

| # | Script | Varied | Outputs |
|---|---|---|---|
| 1 | `a1_budget` | budget per ha, 0 to 1000 $/ha | solo feasibility, v(N), p(N), G(N), adoption, stability |
| 2 | `a2_farm_scale` | farm areas ×0.5 to ×20 | same |
| 3 | `a3_cost_components` | fixed/variable MRV and transaction costs | same |
| 4 | `a4_mrv_scaling` | MRV exponent delta; n = 5 to 1000 farmers | same, plus per-farmer value and cost per ha |
| 5 | `a5_carbon_price` | carbon price 10 to 500 $/tCO2e | same, plus break-even prices |
| 6 | `a6_allocation_comparison` | baseline, 10 panels | per-farmer shares and payoffs, stability, L1 distances, equity, core ranges |
| 7 | `a7_cmp_noise` | farmer-specific sequestration noise | when each sharing rule becomes unstable |

The core is a set, so ablation 6 reports its centroid (adaptive hit-and-run) together with each farmer's min-max range over the core, or NaN when the core is empty. Parameter sweeps check stability through the least-core epsilon: the core is non-empty exactly when epsilon* <= 0.

Summary tables and figures for all ablations are in [`results/SUMMARY.md`](results/SUMMARY.md).

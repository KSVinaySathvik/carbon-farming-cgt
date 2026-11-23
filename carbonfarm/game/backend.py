"""LP / QP / MILP back-ends: Gurobi by default, SciPy/HiGHS + CVXPY as a licence-free fallback.

    lp(c, A_ub, b_ub, A_eq, b_eq)      min c.x  s.t.  A_ub x <= b_ub, A_eq x = b_eq
    qp_min_norm(A, rhs, p)             min ||x||^2  s.t.  sum x = p, A x >= rhs
    milp(c, A, lo, hi, integrality)    min c.x  s.t.  lo <= A x <= hi, 0 <= x <= 1

The full n = 15 least-core and nucleolus LPs exceed the size-limited pip licence of Gurobi; in that
case those models fall back to HiGHS.
"""
from __future__ import annotations

import os
import warnings
from types import SimpleNamespace

import numpy as np

_BACKEND = None          # resolved lazily
_WARNED = False


def gurobi_available() -> bool:
    try:
        import gurobipy as gp
        env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
        m = gp.Model(env=env); m.addVar(); m.update(); m.dispose(); env.dispose()
        return True
    except Exception:
        return False


def set_backend(name: str) -> str:
    global _BACKEND, _WARNED
    name = name.lower()
    if name not in ("auto", "gurobi", "highs"):
        raise ValueError("backend must be auto, gurobi or highs")
    if name == "auto":
        name = "gurobi" if gurobi_available() else "highs"
        if name == "highs" and not _WARNED:
            warnings.warn("Gurobi not available/licensed -> using the open-source HiGHS back-end")
            _WARNED = True
    elif name == "gurobi" and not gurobi_available():
        raise RuntimeError("gurobipy is not importable or no valid licence was found")
    _BACKEND = name
    return name


def get_backend() -> str:
    return _BACKEND or set_backend(os.environ.get("CARBONFARM_SOLVER", "auto"))


def _grb_env():
    import gurobipy as gp
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", int(os.environ.get("CARBONFARM_THREADS", "1")))
    env.start()
    return env


def _lp_gurobi(c, A_ub, b_ub, A_eq, b_eq):
    c = np.asarray(c, float)
    if True:
        import gurobipy as gp
        from gurobipy import GRB
        env = _grb_env(); m = gp.Model(env=env)
        x = m.addMVar(len(c), lb=-GRB.INFINITY, ub=GRB.INFINITY)
        cu = m.addConstr(np.asarray(A_ub) @ x <= np.asarray(b_ub)) if A_ub is not None and len(b_ub) else None
        if A_eq is not None and len(b_eq):
            m.addConstr(np.asarray(A_eq) @ x == np.asarray(b_eq))
        m.setObjective(c @ x, GRB.MINIMIZE)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            out = SimpleNamespace(status=1, message=f"gurobi status {m.Status}", x=None, fun=None,
                                  ineqlin=SimpleNamespace(marginals=None))
        else:
            marg = np.asarray(cu.Pi) if cu is not None else np.array([])      # <= 0 for "<=" rows (min problem)
            out = SimpleNamespace(status=0, message="optimal", x=x.X.copy(), fun=float(m.ObjVal),
                                  ineqlin=SimpleNamespace(marginals=marg))
        m.dispose(); env.dispose()
        return out


def _lp_highs(c, A_ub, b_ub, A_eq, b_eq):
    from scipy.optimize import linprog
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(None, None)] * len(c), method="highs")


def _too_large(e) -> bool:
    return "size-limited" in str(e) or "too large" in str(e)


def _fallback_warning():
    global _WARNED
    if not _WARNED:
        warnings.warn("Gurobi licence is size-limited and the model is too large -> falling back to HiGHS for "
                      "this (and later oversized) models. A full Gurobi licence removes the limit.")
        _WARNED = True


def lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
    if get_backend() == "gurobi":
        try:
            return _lp_gurobi(c, A_ub, b_ub, A_eq, b_eq)
        except Exception as e:
            if not _too_large(e):
                raise
            _fallback_warning()
    return _lp_highs(c, A_ub, b_ub, A_eq, b_eq)


def qp_min_norm(A, rhs, p, x_fallback=None):
    """argmin ||x||^2  s.t. sum(x) = p, A x >= rhs.  Returns None on failure."""
    if get_backend() == "gurobi":
        try:
            return _qp_gurobi(A, rhs, p)
        except Exception as e:
            if not _too_large(e):
                raise
            _fallback_warning()
    return _qp_highs(A, rhs, p)


def _qp_gurobi(A, rhs, p):
    n = A.shape[1]
    if True:
        import gurobipy as gp
        from gurobipy import GRB
        env = _grb_env(); m = gp.Model(env=env)
        x = m.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        m.addConstr(x.sum() == p); m.addConstr(A @ x >= rhs)
        m.setObjective(x @ x, GRB.MINIMIZE)
        m.setParam("Method", 2); m.setParam("BarConvTol", 1e-10)
        m.optimize()
        res = x.X.copy() if m.Status == GRB.OPTIMAL else None
        m.dispose(); env.dispose()
        return res


def _qp_highs(A, rhs, p):
    n = A.shape[1]
    import cvxpy as cp
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.sum(x) == p, A @ x >= rhs])
    try:
        prob.solve(solver=cp.CLARABEL)
        if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
            return np.asarray(x.value)
    except Exception as e:                                                       # pragma: no cover
        warnings.warn(f"QP failed: {e}")
    return None


def milp(c, A, lo, hi, integrality, rel_gap=1e-10):
    """min c.x  s.t. lo <= A x <= hi, 0 <= x <= 1, integrality[j]=1 -> binary.  Returns objective or None."""
    c = np.asarray(c, float); A = np.asarray(A, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    if get_backend() == "gurobi":
        import gurobipy as gp
        from gurobipy import GRB
        env = _grb_env(); m = gp.Model(env=env)
        vt = np.where(np.asarray(integrality) > 0, GRB.BINARY, GRB.CONTINUOUS)
        x = m.addMVar(len(c), lb=0.0, ub=1.0, vtype=vt)
        fl, fh = np.isfinite(lo), np.isfinite(hi)
        if fl.any():
            m.addConstr(A[fl] @ x >= lo[fl])
        if fh.any():
            m.addConstr(A[fh] @ x <= hi[fh])
        m.setObjective(c @ x, GRB.MINIMIZE); m.setParam("MIPGap", rel_gap)
        m.optimize()
        val = float(m.ObjVal) if m.Status == GRB.OPTIMAL else None
        m.dispose(); env.dispose()
        return val
    from scipy.optimize import Bounds, LinearConstraint, milp as _milp
    r = _milp(c=c, constraints=LinearConstraint(A, lo, hi), integrality=integrality,
              bounds=Bounds(np.zeros(len(c)), np.ones(len(c))), options={"mip_rel_gap": rel_gap})
    return None if r.status != 0 else float(r.fun)

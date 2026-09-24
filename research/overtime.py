"""
Paying for unpaid service (Round 8, REPORT section 5.12).

Earlier rounds counted only the staffed hours: service after the doors close,
and a closing window finishing its citizen after its hour ends, cost nothing.
Here that work is paid at kappa times the regular rate (kappa = 1: same wage,
1.5: time and a half). Staff are assumed to leave once they have nobody left to
serve, so the unpaid work is exactly the service time delivered outside paid
windows (the simulator's spill_busy and overtime_busy columns).

With kappa = 1 the paid cost of a plan is total service work plus idle window
time during the day. Total work is fixed by demand, so the cheapest plan is
the one that idles least while meeting the target.
"""

import itertools
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from staffing_methods import (  # noqa: E402
    DESIGN_REPS, DESIGN_SEED, EVAL_REPS, EVAL_SEED, SLOT_MINUTES, SLOTS, ratio_ci,
)
from optimizer import run_simulation  # noqa: E402
from fluid import DAY, _drain_steps, _grid  # noqa: E402


# ============================================================================
# Paid cost of a plan, by simulation
# ============================================================================

def paid_evaluation(plan, rates, mean_service, threshold, kappa, reps=EVAL_REPS,
                    seed=EVAL_SEED, **sim_kwargs) -> dict:
    """Per-hour late rates (with day-clustered CIs) and the paid cost of a plan."""
    r = run_simulation(list(plan), rates, replications=reps, seed=seed,
                       mean_service=mean_service, wait_threshold=threshold, **sim_kwargs)
    arr = np.array(r.daily_arrivals, float)
    late = np.array(r.daily_late, float)
    per_hour = [ratio_ci(late[:, i], arr[:, i]) for i in range(SLOTS)]
    spill = float(np.mean(r.daily_spill)) / 60.0
    after = float(np.mean(r.daily_overtime_busy)) / 60.0
    # Alternative accounting: every last-hour window stays until the office empties
    all_stay = plan[-1] * r.avg_overtime / 60.0
    return {"plan": list(plan), "window_hours": sum(plan),
            "spill_hours": spill, "overtime_hours": after,
            "paid_cost": sum(plan) + kappa * (spill + after),
            "paid_cost_all_stay": sum(plan) + kappa * (spill + all_stay),
            "late": [p for p, _ in per_hour], "late_ci": [ci for _, ci in per_hour]}


def _ok(ev, alpha):
    return all(hi <= alpha for _, hi in ev["late_ci"])


def paid_staffing(rates, mean_service, threshold, alpha, kappa, start, reps=DESIGN_REPS,
                  seed=DESIGN_SEED, max_iter=200, **sim_kwargs):
    """
    Cheapest paid cost meeting every hour's upper 95% bound <= alpha on the
    design seeds (common random numbers). Local search from a feasible start
    over single additions, single removals and one-window moves between
    hours; each step takes the feasible neighbour with the lowest paid cost
    and stops when none is cheaper. Returns (plan, evaluations used).
    """
    def ev(plan):
        return paid_evaluation(plan, rates, mean_service, threshold, kappa, reps, seed,
                               **sim_kwargs)

    plan = list(start)
    current = ev(plan)
    calls = 1
    if not _ok(current, alpha):
        raise ValueError("paid_staffing needs a feasible start")
    for _ in range(max_iter):
        moves = []
        for i in range(SLOTS):
            moves.append([c + (k == i) for k, c in enumerate(plan)])
            if plan[i] > 1:
                moves.append([c - (k == i) for k, c in enumerate(plan)])
        for i, j in itertools.permutations(range(SLOTS), 2):
            if plan[i] > 1:
                moves.append([c - (k == i) + (k == j) for k, c in enumerate(plan)])
        with ThreadPoolExecutor() as pool:
            results = list(pool.map(ev, moves))
        calls += len(moves)
        feasible = [e for e in results if _ok(e, alpha)]
        best = min(feasible, key=lambda e: (e["paid_cost"], max(e["late"])), default=None)
        if best is None or best["paid_cost"] >= current["paid_cost"] - 1e-9:
            return plan, calls
        plan, current = best["plan"], best
    raise RuntimeError("paid_staffing did not converge")


# ============================================================================
# The corrected (non-preemptive) fluid with paid overtime
# ============================================================================

def fluid_paid(rates_per_hour: list, mean_service: float, threshold: float, kappa: float,
               dt: float = None, time_limit: float = 600.0) -> dict:
    """
    fluid_staffing_nonpreemptive (alpha = 0) with unpaid work charged at kappa:
    spill = integral of the citizens finishing at closed windows during the
    day, and post-close work = S x (citizens inside at closing) (exponential
    service is memoryless). Returns the plan and the cost split.
    """
    dt = _grid(mean_service, dt)
    mu = 1.0 / mean_service
    n = int(round(DAY / dt))
    m = _drain_steps(threshold, dt)
    N = n + m + 1
    t = np.arange(N + 1) * dt
    hour = np.minimum((t // SLOT_MINUTES).astype(int), SLOTS - 1)
    lam = np.where(t < DAY, np.array(rates_per_hour, float)[hour] / 60.0, 0.0)
    A_cum = np.concatenate([[0.0], np.cumsum(lam[:N] * dt)])
    per_hour = int(round(SLOT_MINUTES / dt))
    boundary = {i * per_hour: i for i in range(1, SLOTS)}

    ia, ie = SLOTS, SLOTS + N + 1
    iS, is_ = ie + N + 1, ie + 2 * (N + 1)
    iy = is_ + N
    idl = iy + SLOTS - 1
    n_var = idl + SLOTS - 1
    big = 10.0 * max(max(rates_per_hour) / 60.0 * mean_service, 1e-6) + 10.0

    A = lil_matrix((6 * (N + 1) + 3 * SLOTS + n, n_var))
    lb, ub = [], []

    def row(lo, hi):
        lb.append(lo)
        ub.append(hi)
        return len(lb) - 1

    for k in range(N):
        q = row(0.0, 0.0)
        A[q, iS + k + 1], A[q, iS + k], A[q, is_ + k] = 1.0, -1.0, -dt
        q = row(0.0, 0.0)
        A[q, ia + k + 1], A[q, ia + k], A[q, is_ + k] = 1.0, -(1 - mu * dt), -dt
        if k + 1 in boundary:
            A[q, iy + boundary[k + 1] - 1] = 1.0
        q = row(0.0, 0.0)
        A[q, ie + k + 1], A[q, ie + k] = 1.0, -(1 - mu * dt)
        if k + 1 in boundary:
            A[q, iy + boundary[k + 1] - 1] = -1.0
    for k in range(N + 1):
        q = row(-np.inf, A_cum[k])
        A[q, iS + k] = 1.0
        q = row(-np.inf, 0.0)
        A[q, ia + k], A[q, hour[k]] = 1.0, -1.0
    for i in range(1, SLOTS):
        q = row(-np.inf, big)
        A[q, iy + i - 1], A[q, idl + i - 1], A[q, i - 1], A[q, i] = 1.0, big, -1.0, 1.0
        q = row(-np.inf, 0.0)
        A[q, iy + i - 1], A[q, idl + i - 1] = 1.0, -big
    for k in range(n):
        q = row(-np.inf, -A_cum[k + 1])
        A[q, iS + k + 1 + m] = -1.0

    lower = np.zeros(n_var)
    upper = np.full(n_var, np.inf)
    upper[[ia, ie, iS]] = 0.0
    upper[idl:] = 1.0
    integrality = np.zeros(n_var)
    integrality[idl:] = 1
    # Objective in window-hours: sum c + kappa (spill + S * inside at closing) / 60
    objective = np.zeros(n_var)
    objective[:SLOTS] = 1.0
    objective[ie:ie + n] += kappa * dt / 60.0                       # spill before closing
    close = kappa * mean_service / 60.0
    objective[ia + n] += close
    objective[ie + n] += close
    objective[iS + n] -= close                                       # queue = A_n - S_n
    constant = close * A_cum[n]
    res = milp(c=objective, integrality=integrality, bounds=Bounds(lower, upper),
               constraints=LinearConstraint(A[:len(lb)].tocsr(), np.array(lb), np.array(ub)),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-6})
    if res.x is None:
        raise RuntimeError(f"fluid program failed: {res.message}")
    x = res.x
    spill = float(np.sum(x[ie:ie + n]) * dt / 60.0)
    inside = float(x[ia + n] + x[ie + n] + A_cum[n] - x[iS + n])
    after = mean_service * inside / 60.0
    return {"plan": list(x[:SLOTS]), "window_hours": float(np.sum(x[:SLOTS])),
            "spill_hours": spill, "overtime_hours": after,
            "paid_cost": float(res.fun + constant), "optimal": res.status == 0}

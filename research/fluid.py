"""
A finite-horizon fluid model of the walk-in day (Round 7, REPORT section 5.11).

Deterministic limit of the office as it grows (exponential service, mean S):
X(t) citizens in the office (in service or waiting), opening empty, with

    dX/dt = lambda(t) - mu min(X, c(t)),        mu = 1 / S,

and service continuing with the last hour's windows after the doors close,
as in the simulator. A citizen arriving at t finds q = (X - c)+ ahead and,
under FIFO with every window busy, waits w(t) with

    integral_t^{t + w} mu c(u) du = q(t),   so   w(t) <= T  <=>  q(t) <= mu integral_t^{t+T} c,

which is linear in the plan c and the state X. The per-hour target becomes:
at most alpha of each hour's arrivals may have w > T. Minimizing window-hours
(or paid shift hours) subject to it is a mixed-integer linear program; with
alpha = 0 it is a linear program. Everything scales with the load, so one
solve at 1 Erlang gives the fluid constant for every office size.

Approximation: when a window closes, fluid capacity drops at once, whereas
in the simulator the closing window first finishes its citizen.
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from staffing_methods import SLOT_MINUTES, SLOTS  # noqa: E402

DAY = SLOTS * SLOT_MINUTES


def _grid(mean_service: float, dt: float = None) -> float:
    return dt if dt else min(1.0, mean_service / 8.0)


def _capacity_overlap(t: float, threshold: float) -> np.ndarray:
    """Minutes of [t, t + T] inside each hour; after closing the last hour's windows serve."""
    out = np.zeros(SLOTS)
    for i in range(SLOTS):
        lo = i * SLOT_MINUTES
        hi = (i + 1) * SLOT_MINUTES if i < SLOTS - 1 else math.inf
        out[i] = max(0.0, min(hi, t + threshold) - max(lo, t))
    return out


def fluid_day(plan: list, rates_per_hour: list, mean_service: float, threshold: float,
              dt: float = None) -> dict:
    """
    Forward simulation of the fluid for a given hourly plan.

    Returns the time grid, X(t), the queue, each arrival's wait w(t) and each
    hour's share of arrivals waiting longer than the threshold.
    """
    dt = _grid(mean_service, dt)
    mu = 1.0 / mean_service
    n = int(round(DAY / dt))
    t = np.arange(n) * dt
    hour = np.minimum((t // SLOT_MINUTES).astype(int), SLOTS - 1)
    lam = np.array(rates_per_hour, float)[hour] / 60.0
    c = np.array(plan, float)[hour]
    X = np.zeros(n + 1)
    for k in range(n):
        X[k + 1] = X[k] + (lam[k] - mu * min(X[k], c[k])) * dt
    queue = np.maximum(X[:-1] - c, 0.0)
    # Cumulative service capacity C(t) = mu * integral_0^t c, piecewise linear;
    # the wait solves C(t + w) = C(t) + q
    knots = np.arange(SLOTS + 1) * SLOT_MINUTES
    cum = np.concatenate([[0.0], np.cumsum(np.array(plan, float) * mu * SLOT_MINUTES)])
    far = DAY + 1e6
    knots = np.append(knots, far)
    cum = np.append(cum, cum[-1] + plan[-1] * mu * (far - DAY))
    now = np.interp(t, knots, cum)
    wait = np.interp(now + queue, cum, knots) - t
    late = [float(np.mean(wait[hour == i] > threshold + 1e-9)) for i in range(SLOTS)]
    return {"t": t, "X": X[:-1], "queue": queue, "wait": wait, "late": late}


def fluid_staffing(rates_per_hour: list, mean_service: float, threshold: float,
                   alpha: float, menu=None, integer: bool = False, dt: float = None,
                   time_limit: float = 600.0) -> dict:
    """
    Cheapest fluid plan meeting the per-hour target.

    menu=None optimizes hourly windows c_i (window-hours); with a Roster menu
    it optimizes shift counts (paid hours). integer=True makes the windows or
    shift counts whole numbers. Returns the plan, its cost and the solver status.
    """
    dt = _grid(mean_service, dt)
    mu = 1.0 / mean_service
    n = int(round(DAY / dt))
    t = np.arange(n) * dt
    hour = np.minimum((t // SLOT_MINUTES).astype(int), SLOTS - 1)
    lam = np.array(rates_per_hour, float)[hour] / 60.0

    # Plan variables: hourly windows, or shift counts with windows = cover @ x
    if menu is None:
        cover = np.eye(SLOTS)
        cost = np.ones(SLOTS)
    else:
        cover = np.array(menu.cover, float)
        cost = np.array(menu.hours, float)
    n_plan = cover.shape[1]
    late_vars = alpha > 0
    # Layout: plan | X_0..X_n | d_0..d_{n-1} | q_0..q_{n-1} | z_0..z_{n-1}
    iX = n_plan
    iD = iX + n + 1
    iQ = iD + n
    iZ = iQ + n
    n_var = iZ + (n if late_vars else 0)
    # q_k never exceeds the arrivals so far (a tight big-M keeps the MILP small)
    arrived = np.concatenate([[0.0], np.cumsum(lam * dt)])[:n] + 1e-9

    rows = 5 * n + (SLOTS if late_vars else 0)
    A = lil_matrix((rows, n_var))
    lb = np.full(rows, -np.inf)
    ub = np.zeros(rows)
    r = 0
    for k in range(n):
        # X_{k+1} - X_k + dt d_k = dt lambda_k
        A[r, iX + k + 1], A[r, iX + k], A[r, iD + k] = 1.0, -1.0, dt
        lb[r] = ub[r] = dt * lam[k]
        r += 1
        # d_k <= mu X_k
        A[r, iD + k], A[r, iX + k] = 1.0, -mu
        r += 1
        # d_k <= mu c_{hour(k)}
        A[r, iD + k] = 1.0
        for j in range(n_plan):
            if cover[hour[k], j]:
                A[r, j] = -mu * cover[hour[k], j]
        r += 1
        # q_k >= X_k - c_{hour(k)}
        A[r, iX + k], A[r, iQ + k] = 1.0, -1.0
        for j in range(n_plan):
            if cover[hour[k], j]:
                A[r, j] = -cover[hour[k], j]
        r += 1
        # q_k <= mu * integral_t^{t+T} c  (+ M z_k if this arrival may be late)
        A[r, iQ + k] = 1.0
        overlap = _capacity_overlap(t[k], threshold) @ cover
        for j in range(n_plan):
            if overlap[j]:
                A[r, j] = -mu * overlap[j]
        if late_vars:
            A[r, iZ + k] = -arrived[k]
        r += 1
    if late_vars:
        per_hour = int(round(SLOT_MINUTES / dt))
        for i in range(SLOTS):
            for k in range(i * per_hour, (i + 1) * per_hour):
                A[r, iZ + k] = 1.0
            ub[r] = alpha * per_hour
            r += 1

    lower = np.zeros(n_var)
    upper = np.full(n_var, np.inf)
    upper[iX] = 0.0                               # The office opens empty
    integrality = np.zeros(n_var)
    if integer:
        integrality[:n_plan] = 1
    if late_vars:
        upper[iZ:] = 1.0
        integrality[iZ:] = 1
    objective = np.zeros(n_var)
    objective[:n_plan] = cost
    res = milp(c=objective, integrality=integrality, bounds=Bounds(lower, upper),
               constraints=LinearConstraint(A.tocsr(), lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-6})
    if res.x is None:
        raise RuntimeError(f"fluid program failed: {res.message}")
    x = res.x[:n_plan]
    return {"plan": list(cover @ x), "units": list(x), "cost": float(cost @ x),
            "status": res.message, "optimal": res.status == 0,
            "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0)}


# ============================================================================
# Non-preemptive closing (post hoc, Round 7): a closing window finishes its
# citizen. Citizens in service split into `active` (on open windows) and
# `finishing` (on windows that just closed), and waits come from FIFO
# cumulative counts: arrivals up to t are all on time when the service starts
# by t + T cover them, S(t + T) >= A(t).
# ============================================================================

def _drain_steps(threshold: float, dt: float) -> int:
    return int(math.ceil(threshold / dt - 1e-9))


def fluid_day_nonpreemptive(plan: list, rates_per_hour: list, mean_service: float,
                            threshold: float, dt: float = None) -> dict:
    """Forward simulation with non-preemptive window closing; see fluid_day."""
    dt = _grid(mean_service, dt)
    mu = 1.0 / mean_service
    n = int(round(DAY / dt))
    horizon = n + int(math.ceil(4 * DAY / dt))           # drain long enough for any wait
    t = np.arange(horizon + 1) * dt
    hour = np.minimum((t // SLOT_MINUTES).astype(int), SLOTS - 1)
    lam = np.where(t < DAY, np.array(rates_per_hour, float)[hour] / 60.0, 0.0)
    c = np.array(plan, float)[hour]
    active = finishing = queue = 0.0
    started = np.zeros(horizon + 1)
    arrived = np.zeros(horizon + 1)
    X = np.zeros(horizon + 1)
    for k in range(horizon):
        if k > 0 and c[k] < c[k - 1] and active > c[k]:
            moved = active - c[k]                         # these windows just closed
            active -= moved
            finishing += moved
        free = max(0.0, c[k] - active) + mu * active * dt
        starts = min(queue + lam[k] * dt, free)
        queue += lam[k] * dt - starts
        active += starts - mu * active * dt
        finishing -= mu * finishing * dt
        started[k + 1] = started[k] + starts
        arrived[k + 1] = arrived[k] + lam[k] * dt
        X[k + 1] = active + finishing + queue
    # Arrivals of step k (up to t_{k+1}) start once cumulative starts first reach
    # A_{k+1} (np.interp is unsafe here: `started` has flat stretches)
    target = arrived[1:n + 1] - 1e-9 * max(arrived[n], 1.0)
    idx = np.clip(np.searchsorted(started, target, side="left"), 1, horizon)
    step = np.maximum(started[idx] - started[idx - 1], 1e-300)
    frac = np.clip((target - started[idx - 1]) / step, 0.0, 1.0)
    start_time = t[idx - 1] + frac * dt
    wait = start_time - t[1:n + 1]
    hr = hour[:n]
    late = [float(np.mean(wait[hr == i] > threshold + 1e-6)) for i in range(SLOTS)]
    return {"t": t[:n], "X": X[:n], "wait": wait, "late": late}


def fluid_staffing_nonpreemptive(rates_per_hour: list, mean_service: float, threshold: float,
                                 alpha: float, menu=None, dt: float = None,
                                 time_limit: float = 600.0) -> dict:
    """fluid_staffing with non-preemptive closing and exact FIFO waits."""
    dt = _grid(mean_service, dt)
    mu = 1.0 / mean_service
    n = int(round(DAY / dt))
    m = _drain_steps(threshold, dt)
    N = n + m + 1                                          # last step index with a state
    t = np.arange(N + 1) * dt
    hour = np.minimum((t // SLOT_MINUTES).astype(int), SLOTS - 1)
    lam = np.where(t < DAY, np.array(rates_per_hour, float)[hour] / 60.0, 0.0)
    A_cum = np.concatenate([[0.0], np.cumsum(lam[:N] * dt)])   # arrivals by t_k
    per_hour = int(round(SLOT_MINUTES / dt))
    boundary = {i * per_hour: i for i in range(1, SLOTS)}

    if menu is None:
        cover, cost = np.eye(SLOTS), np.ones(SLOTS)
    else:
        cover, cost = np.array(menu.cover, float), np.array(menu.hours, float)
    n_plan = cover.shape[1]
    late_vars = alpha > 0
    # Layout: plan | a_0..a_N | e_0..e_N | S_0..S_N | s_0..s_{N-1} | y_1..y_7 | delta_1..7 | z
    ia, ie = n_plan, n_plan + N + 1
    iS, is_ = ie + N + 1, ie + 2 * (N + 1)
    iy = is_ + N
    idl = iy + SLOTS - 1
    iz = idl + SLOTS - 1
    n_var = iz + (n if late_vars else 0)
    big = 10.0 * max(max(rates_per_hour) / 60.0 * mean_service, 1e-6) + 10.0

    rows_list, lb_list, ub_list = [], [], []
    A = lil_matrix((6 * (N + 1) + 3 * SLOTS + n + SLOTS, n_var))
    r = 0

    def row(lo, hi):
        nonlocal r
        lb_list.append(lo)
        ub_list.append(hi)
        r += 1
        return r - 1

    for k in range(N):
        # S_{k+1} = S_k + dt s_k
        q = row(0.0, 0.0)
        A[q, iS + k + 1], A[q, iS + k], A[q, is_ + k] = 1.0, -1.0, -dt
        # a_{k+1} = (1 - mu dt) a_k + dt s_k - y  (y at an hour boundary)
        q = row(0.0, 0.0)
        A[q, ia + k + 1], A[q, ia + k], A[q, is_ + k] = 1.0, -(1 - mu * dt), -dt
        if k + 1 in boundary:
            A[q, iy + boundary[k + 1] - 1] = 1.0
        # e_{k+1} = (1 - mu dt) e_k + y
        q = row(0.0, 0.0)
        A[q, ie + k + 1], A[q, ie + k] = 1.0, -(1 - mu * dt)
        if k + 1 in boundary:
            A[q, iy + boundary[k + 1] - 1] = -1.0
    for k in range(N + 1):
        # Nobody starts before arriving; active citizens fit in the open windows
        q = row(-np.inf, A_cum[k])
        A[q, iS + k] = 1.0
        q = row(-np.inf, 0.0)
        A[q, ia + k] = 1.0
        for j in range(n_plan):
            if cover[hour[k], j]:
                A[q, j] = -cover[hour[k], j]
    for i in range(1, SLOTS):
        # y_i <= max(0, c_{i-1} - c_i): delta_i = 1 when windows close
        q = row(-np.inf, big)
        A[q, iy + i - 1], A[q, idl + i - 1] = 1.0, big
        for j in range(n_plan):
            A[q, j] = -(cover[i - 1, j] - cover[i, j])
        q = row(-np.inf, 0.0)
        A[q, iy + i - 1], A[q, idl + i - 1] = 1.0, -big
    for k in range(n):
        # Arrivals of step k are on time: S_{k+1+m} >= A_{k+1} (unless allowed late)
        q = row(-np.inf, -A_cum[k + 1])
        A[q, iS + k + 1 + m] = -1.0
        if late_vars:
            A[q, iz + k] = -A_cum[k + 1]
    if late_vars:
        for i in range(SLOTS):
            q = row(-np.inf, alpha * per_hour)
            for k in range(i * per_hour, (i + 1) * per_hour):
                A[q, iz + k] = 1.0

    A = A[:r].tocsr()
    lower = np.zeros(n_var)
    upper = np.full(n_var, np.inf)
    upper[[ia, ie, iS]] = 0.0                               # The office opens empty
    upper[idl:idl + SLOTS - 1] = 1.0
    integrality = np.zeros(n_var)
    integrality[idl:idl + SLOTS - 1] = 1
    if late_vars:
        upper[iz:] = 1.0
        integrality[iz:] = 1
    objective = np.zeros(n_var)
    objective[:n_plan] = cost
    res = milp(c=objective, integrality=integrality, bounds=Bounds(lower, upper),
               constraints=LinearConstraint(A, np.array(lb_list), np.array(ub_list)),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-6})
    if res.x is None:
        raise RuntimeError(f"fluid program failed: {res.message}")
    x = res.x[:n_plan]
    return {"plan": list(cover @ x), "units": list(x), "cost": float(cost @ x),
            "status": res.message, "optimal": res.status == 0}

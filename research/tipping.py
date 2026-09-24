"""
Tipping points of mandatory services (Round 9, REPORT section 5.13).

A citizen who leaves a mandatory service returns on a later day with
probability r. With R returners a day and L(R) the day's losses, the
day-to-day steady states solve h(R) = r L(R) - R = 0. A root is stable when
r L'(R) < 1 there; more than one stable root means the office is bistable:
a single bad day can move it from a low state to a high one for good.

Three views of h:
- stationary Erlang-A with returns (one hour, constant rates);
- a fluid of the finite day with exponential reneging, which scales with
  the load, so one solve at 1 Erlang covers every office size;
- the simulator, via return_curve and the day-to-day chain.

In the first two, h is non-increasing (more input never lowers the work
served), so the steady state is unique and there is no tipping point; what
changes near collapse is how slowly the office recovers, 1 / (1 - r L'(R*)).
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from staffing_methods import SLOT_MINUTES, SLOTS  # noqa: E402
from abandonment import renege_abandon, return_rates  # noqa: E402
from fluid import _grid  # noqa: E402

DAY = SLOTS * SLOT_MINUTES


def sign_changes(xs, hs) -> list:
    """Roots of a sampled curve by linear interpolation between sign changes."""
    roots = []
    for i in range(len(xs) - 1):
        a, b = hs[i], hs[i + 1]
        if a == 0.0:
            roots.append(float(xs[i]))
        elif a * b < 0.0:
            roots.append(float(xs[i] + (xs[i + 1] - xs[i]) * a / (a - b)))
    return roots


# ============================================================================
# Stationary Erlang-A with returns
# ============================================================================

def stationary_return_roots(c: int, rate_per_hour: float, mean_service: float,
                            mean_patience: float, return_prob: float,
                            points: int = 400, span: float = 30.0) -> list:
    """
    Roots of g(x) = lambda + r x P_ab(c, x) - x, x the total arrival rate per
    minute, scanned on a log grid over [lambda, span * lambda]. The abandonment
    flow a(x) = x P_ab = x - served(x) is convex in x (served(x) is concave,
    capped at c mu), so g is convex, g(lambda) >= 0, and g has at most one
    root when r < 1, or when lambda < c mu for r = 1. Returns [] if none.
    """
    lam = rate_per_hour / 60.0
    xs = lam * np.geomspace(1.0, span, points)
    gs = [lam + return_prob * x * renege_abandon(c, x, mean_service, mean_patience) - x
          for x in xs]
    return sign_changes(xs, gs)


# ============================================================================
# Fluid of the day with reneging
# ============================================================================

def fluid_day_renege(plan: list, rates_per_hour: list, mean_service: float,
                     mean_patience: float, dt: float = None) -> dict:
    """
    Fluid of the day with exponential reneging (rate theta = 1 / patience per
    waiting citizen) and non-preemptive window closing, as in the simulator.
    Service and reneging continue with the last hour's windows after the doors
    close until the office is empty. The queue is kept by arrival hour and
    served oldest first, so losses are attributed to the hour of arrival.
    Returns per-hour arrivals and losses and the day's total loss.
    """
    dt = _grid(mean_service, dt)
    mu = 1.0 / mean_service
    theta = 1.0 / mean_patience
    n = int(round(DAY / dt))
    rates = np.array(rates_per_hour, float) / 60.0
    active = finishing = 0.0
    queue = np.zeros(SLOTS)            # Waiting citizens by arrival hour, FIFO
    lost = np.zeros(SLOTS)
    served = 0.0
    k = 0
    prev_c = plan[0]
    while True:
        open_ = k < n
        h = min(int(k * dt // SLOT_MINUTES), SLOTS - 1)
        c = plan[h] if open_ else plan[-1]
        if c < prev_c and active > c:
            finishing += active - c
            active = c
        prev_c = c
        if open_:
            queue[h] += rates[h] * dt
        # Starts, oldest cohort first; only those still waiting after them renege
        free = max(0.0, c - active) + mu * active * dt
        starts = 0.0
        for i in range(SLOTS):
            if starts >= free:
                break
            take = min(queue[i], free - starts)
            queue[i] -= take
            starts += take
        gone = queue * (1.0 - math.exp(-theta * dt))
        lost += gone
        queue -= gone
        done = mu * active * dt + mu * finishing * dt
        served += done
        active += starts - mu * active * dt
        finishing -= mu * finishing * dt
        k += 1
        if not open_ and queue.sum() < 1e-9 * max(1.0, rates.sum()) and \
                active + finishing < 1e-6 * max(1.0, rates.sum()):
            break
        if k > n + int(20 * DAY / dt):
            break
    arrivals = rates * SLOT_MINUTES
    return {"arrivals": arrivals, "lost": lost, "losses": float(lost.sum()), "served": served,
            "fail_abandon": (lost / np.maximum(arrivals, 1e-300)).tolist()}


def fluid_losses(plan: list, rates: list, mean_service: float, mean_patience: float,
                 returns_per_day: float, timing: str, dt: float = None) -> float:
    """L(R): the day's losses in the fluid with R returners arriving by `timing`."""
    day = fluid_day_renege(plan, return_rates(rates, returns_per_day, timing),
                           mean_service, mean_patience, dt)
    return day["losses"]


def fluid_return_curve(plan: list, rates: list, mean_service: float, mean_patience: float,
                       return_prob: float, timing: str, R_grid, dt: float = None) -> list:
    """h(R) = r L(R) - R on a grid of daily returners."""
    return [return_prob * fluid_losses(plan, rates, mean_service, mean_patience, R,
                                       timing, dt) - R for R in R_grid]


def classify_roots(R_grid, h) -> dict:
    """
    Fixed points of R -> r L(R) with their stability (stable where h goes
    from + to -). If h > 0 at the grid's end, the backlog grows past it
    (a root 'at infinity' that counts as a stable high state).
    """
    roots = sign_changes(R_grid, h)
    stable = []
    for x in roots:
        i = int(np.searchsorted(R_grid, x))
        i = min(max(i, 1), len(R_grid) - 1)
        stable.append(bool(h[i - 1] > h[i]))
    unbounded = bool(h[-1] > 0)
    n_stable = sum(stable) + (1 if unbounded else 0)
    return {"roots": roots, "stable": stable, "unbounded": unbounded,
            "n_stable": n_stable, "bistable": n_stable >= 2}


def fluid_bifurcation(plan: list, rates: list, mean_service: float, mean_patience: float,
                      return_prob: float, timing: str, scales, R_max_factor: float = 3.0,
                      R_points: int = 241, dt: float = None) -> list:
    """For each staffing scale phi, the fixed points of the fluid with plan phi * plan."""
    fresh = sum(rates)
    R_grid = np.linspace(0.0, R_max_factor * fresh, R_points)
    out = []
    for phi in scales:
        scaled = [phi * c for c in plan]
        h = fluid_return_curve(scaled, rates, mean_service, mean_patience, return_prob,
                               timing, R_grid, dt)
        cls = classify_roots(R_grid, np.array(h))
        cls.update({"phi": float(phi), "window_hours": float(sum(scaled))})
        out.append(cls)
    return out


# ============================================================================
# Simulated return curve and the day-to-day chain
# ============================================================================

def fluid_fixed_point(plan: list, rates: list, mean_service: float, mean_patience: float,
                      return_prob: float, timing: str, max_factor: float = 200.0,
                      dt: float = None) -> dict:
    """
    The fluid's steady state R* = r L(R*) by bisection (h is non-increasing in
    the fluid: more input never lowers the work served), its slope r L'(R*)
    and the relaxation time 1 / (1 - r L'(R*)) in days. R* = inf if no root
    lies below max_factor x fresh demand.
    """
    fresh = sum(rates)
    L = lambda R: fluid_losses(plan, rates, mean_service, mean_patience, R, timing, dt)
    lo, hi = 0.0, fresh
    while return_prob * L(hi) > hi:
        lo, hi = hi, hi * 2
        if hi > max_factor * fresh:
            return {"R": math.inf, "slope": 1.0, "relax_days": math.inf}
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if return_prob * L(mid) > mid else (lo, mid)
    R = (lo + hi) / 2
    d = max(1e-3 * fresh, 1e-6)
    a = max(R - d, 0.0)
    slope = return_prob * (L(R + d) - L(a)) / (R + d - a)
    return {"R": R, "slope": slope,
            "relax_days": 1.0 / (1.0 - slope) if slope < 1 else math.inf}


def fluid_recovery_days(plan: list, rates: list, mean_service: float, mean_patience: float,
                        return_prob: float, timing: str, R_star: float,
                        within: float = 0.10, max_days: int = 2000, dt: float = None) -> int:
    """
    Days for the fluid map R -> r L(R) to come back within `within` x fresh
    demand of R* after one closure day (next day's returners: fresh + R*).
    """
    fresh = sum(rates)
    R, days = fresh + R_star, 0
    while R - R_star > within * fresh and days < max_days:
        R = return_prob * fluid_losses(plan, rates, mean_service, mean_patience, R,
                                       timing, dt)
        days += 1
    return days


def return_curve(staffing: list, rates: list, mean_service: float, return_prob: float,
                 timing: str, R_grid, threshold: float = 15.0, reps: int = 400,
                 seed: int = 100_000, **sim_kwargs) -> list:
    """
    Simulated h(R) = r L(R) - R with a 95% CI, on common random numbers
    across R (same seeds at every grid point). Each point after the first
    also carries a 95% CI on the paired change of h from the previous point.
    """
    from optimizer import run_simulation  # noqa: E402 (lazy: needs the binary)
    out, prev = [], None
    for R in R_grid:
        r = run_simulation(list(staffing), return_rates(rates, R, timing),
                           replications=reps, seed=seed, mean_service=mean_service,
                           wait_threshold=threshold, **sim_kwargs)
        daily = np.array(r.daily_abandoned, float).sum(axis=1)
        m = float(daily.mean())
        half = 1.96 * float(daily.std(ddof=1)) / math.sqrt(len(daily))
        row = {"R": float(R), "L": m, "h": return_prob * m - R,
               "h_low": return_prob * (m - half) - R,
               "h_high": return_prob * (m + half) - R}
        if prev is not None:
            dR = R - prev[0]
            diff = return_prob * (daily - prev[1]) - dR
            dh = float(diff.mean())
            dhalf = 1.96 * float(diff.std(ddof=1)) / math.sqrt(len(diff))
            row.update({"dh": dh, "dh_low": dh - dhalf, "dh_high": dh + dhalf})
        out.append(row)
        prev = (R, daily)
    return out


def simulate_return_chain(staffing: list, rates: list, mean_service: float,
                          return_prob: float, timing: str, days: int,
                          shock_day: int = None, start_returns: float = 0.0,
                          seed: int = 300_000, cap_factor: float = 5.0,
                          threshold: float = 15.0, **sim_kwargs) -> list:
    """
    Day-to-day dynamics: each day is one simulated day with the fresh demand
    plus yesterday's returners (at their expected rate; the simulator draws
    Poisson arrivals). Each loss returns the next day with probability r
    (binomial). On `shock_day` the office is closed: nobody is served and the
    whole day's fresh demand, plus that day's returners, return the next day.
    Stops early once returners exceed cap_factor x fresh demand.
    """
    rng = np.random.default_rng(seed)
    fresh = sum(rates)
    R = start_returns
    path = []
    from optimizer import run_simulation  # noqa: E402
    for day in range(days):
        if shock_day is not None and day == shock_day:
            losses = fresh + R
        else:
            r = run_simulation(list(staffing), return_rates(rates, R, timing),
                               replications=1, seed=seed + day, mean_service=mean_service,
                               wait_threshold=threshold, **sim_kwargs)
            losses = float(sum(r.daily_abandoned[0]))
        path.append({"day": day, "returns": R, "losses": losses})
        R = float(rng.binomial(int(round(losses)), return_prob))
        if R > cap_factor * fresh:
            path.append({"day": day + 1, "returns": R, "losses": math.nan})
            break
    return path

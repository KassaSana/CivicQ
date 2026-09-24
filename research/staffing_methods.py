"""
Staffing methods for time-varying walk-in service systems.

Analytical "effective load" rules (all feed a per-hour Erlang-C calculation):
    SIPP      hourly average arrival rate x mean service time
              (Green, Kolesar & Soares 2001)
    Lag-SIPP  arrival rate averaged over the hour shifted back by the mean
              service time (Green, Kolesar & Soares 2001; Green, Kolesar & Whitt 2007)
    OL-avg /  infinite-server offered load m(t), averaged / maximized over the hour
    OL-max    (Jennings, Mandelbaum, Massey & Whitt 1996)

Simulation-based rule:
    SGS       simulation greedy staffing: start from SIPP, add a window to the
              earliest hour that misses the target, then local search over
              "remove one" and "remove two, add one" moves while every hour
              still meets it (a discrete, hourly analogue of the iterative
              staffing algorithm of Feldman et al. 2008). criterion="ucb"
              requires each hour's upper 95% bound to meet the target.

Evaluation uses a separate seed block from design, and treats days
(replications) as the independent unit when building confidence intervals.
"""

import itertools
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from optimizer import erlang_c, find_simulator, run_simulation  # noqa: E402

SLOTS = 8
SLOT_MINUTES = 60.0
DAY_MINUTES = SLOTS * SLOT_MINUTES

DESIGN_SEED = 1          # Seeds 1..n are used to *choose* plans
EVAL_SEED = 100_000      # Seeds 100000.. are used to *score* plans
DESIGN_REPS = 400
EVAL_REPS = 1000

# Office demand shape (citizens/hour) and its normalized deviations z (max |z| = 1)
OFFICE_RATES = [12.0, 15.0, 10.0, 8.0, 8.0, 12.0, 14.0, 10.0]


def _shape(rates: list) -> list:
    mean = sum(rates) / len(rates)
    dev = [r - mean for r in rates]
    return [d / max(abs(x) for x in dev) for d in dev]


OFFICE_SHAPE = _shape(OFFICE_RATES)
# Robustness shapes (Round 7): one midday peak, and a morning-heavy ramp
SHAPES = {"double": OFFICE_SHAPE,
          "single": _shape([8.0, 10.0, 12.0, 14.0, 14.0, 12.0, 10.0, 8.0]),
          "ramp": _shape([14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0])}


# ============================================================================
# Demand and offered load
# ============================================================================

def arrival_profile(mean_load: float, mean_service: float, amplitude: float,
                    shape: list = OFFICE_SHAPE) -> list:
    """Hourly arrival rates (per hour) with average offered load `mean_load` Erlangs."""
    base_per_hour = mean_load / mean_service * 60.0
    return [base_per_hour * (1.0 + amplitude * z) for z in shape]


def _survival(u: np.ndarray, mean_service: float, dist: str, cv: float) -> np.ndarray:
    """P(S > u) for the service-time distribution."""
    if dist == "exp":
        return np.exp(-u / mean_service)
    if dist == "det":
        return (u < mean_service).astype(float)
    if dist == "lognormal":
        sigma2 = math.log(1.0 + cv * cv)
        mu = math.log(mean_service) - sigma2 / 2
        z = (np.log(np.maximum(u, 1e-12)) - mu) / math.sqrt(2 * sigma2)
        return 0.5 * np.vectorize(math.erfc)(z)
    raise ValueError(dist)


def offered_load(rates_per_hour: list, mean_service: float, dist: str = "exp",
                 cv: float = 1.0, dt: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """
    Infinite-server offered load m(t) = integral_0^t lambda(t-u) P(S > u) du.

    The office opens empty (lambda = 0 before t = 0), which is exactly the
    transient SIPP ignores. Returns (t in minutes, m(t)) on a dt grid over the day.
    """
    t = np.arange(0.0, DAY_MINUTES + dt / 2, dt)
    # Midpoint rule: m(t_k) = sum_i lambda(t_k - (i + 1/2) dt) P(S > (i + 1/2) dt) dt
    mid = t[:-1] + dt / 2
    slot = np.minimum((mid // SLOT_MINUTES).astype(int), SLOTS - 1)
    lam_mid = np.array(rates_per_hour)[slot] / 60.0        # per minute
    kernel = _survival(np.arange(len(mid)) * dt + dt / 2, mean_service, dist, cv)
    m = np.concatenate([[0.0], np.convolve(lam_mid, kernel)[:len(mid)] * dt])
    return t, m


def lagged_rates(rates_per_hour: list, lag: float, dt: float = 0.25) -> list:
    """Hourly averages of lambda(t - lag), with lambda = 0 before opening."""
    out = []
    for i in range(SLOTS):
        grid = np.arange(i * SLOT_MINUTES, (i + 1) * SLOT_MINUTES, dt) + dt / 2 - lag
        idx = np.floor(grid / SLOT_MINUTES).astype(int)
        vals = np.where(idx < 0, 0.0, np.array(rates_per_hour)[np.clip(idx, 0, SLOTS - 1)])
        out.append(float(vals.mean()))
    return out


# ============================================================================
# Erlang-C effective-load staffing
# ============================================================================

def prob_wait_exceeds(c: int, load: float, mean_service: float, threshold: float) -> float:
    """M/M/c P(W > threshold) for an offered load in Erlangs."""
    if load <= 0:
        return 0.0
    if load >= c:
        return 1.0
    return erlang_c(c, load) * math.exp(-(c - load) * threshold / mean_service)


def servers_for_load(load: float, mean_service: float, threshold: float,
                     alpha: float, min_servers: int = 1) -> int:
    c = max(min_servers, 1)
    while prob_wait_exceeds(c, load, mean_service, threshold) > alpha:
        c += 1
    return c


def effective_load_staffing(loads: list, mean_service: float, threshold: float,
                            alpha: float) -> list:
    return [servers_for_load(a, mean_service, threshold, alpha) for a in loads]


def analytic_plans(rates: list, mean_service: float, threshold: float, alpha: float) -> dict:
    """All analytical staffing rules for one scenario."""
    sipp_loads = [r / 60.0 * mean_service for r in rates]
    lag_loads = [r / 60.0 * mean_service for r in lagged_rates(rates, mean_service)]
    t, m = offered_load(rates, mean_service)
    slot = np.minimum((t // SLOT_MINUTES).astype(int), SLOTS - 1)
    ol_avg = [float(m[slot == i].mean()) for i in range(SLOTS)]
    ol_max = [float(m[slot == i].max()) for i in range(SLOTS)]
    return {
        "SIPP": effective_load_staffing(sipp_loads, mean_service, threshold, alpha),
        "Lag-SIPP": effective_load_staffing(lag_loads, mean_service, threshold, alpha),
        "OL-avg": effective_load_staffing(ol_avg, mean_service, threshold, alpha),
        "OL-max": effective_load_staffing(ol_max, mean_service, threshold, alpha),
    }


# ============================================================================
# Evaluation with day-clustered confidence intervals
# ============================================================================

@dataclass
class Evaluation:
    staffing: tuple
    staff_hours: int
    late_prob: list          # Per arrival hour, pooled ratio estimate
    late_ci: list            # (low, high) per hour, days as the independent unit
    overall_late: float      # Pooled over the whole day
    mean_daily_p90: float
    frac_days_p90_ok: float  # Share of days whose own P90 <= threshold
    mean_wait: float

    def worst_hour(self) -> int:
        return int(np.argmax(self.late_prob))

    def hours_missing(self, alpha: float) -> int:
        """Hours whose CI lies entirely above alpha (statistically significant misses)."""
        return sum(1 for lo, _ in self.late_ci if lo > alpha)


def ratio_ci(late: np.ndarray, arrivals: np.ndarray, z: float = 1.96) -> tuple:
    """Ratio estimator sum(L)/sum(A) with a delta-method CI over independent days."""
    total_a = arrivals.sum()
    if total_a == 0:
        return 0.0, (0.0, 0.0)
    p = late.sum() / total_a
    n = len(late)
    resid = late - p * arrivals
    se = math.sqrt((resid ** 2).sum() / (n * (n - 1))) / arrivals.mean() if n > 1 else 0.0
    return p, (max(0.0, p - z * se), min(1.0, p + z * se))


def evaluate(staffing: list, rates: list, mean_service: float, threshold: float = 15.0,
             reps: int = EVAL_REPS, seed: int = EVAL_SEED, metric: str = "late",
             **sim_kwargs) -> Evaluation:
    """
    metric="late": late / served (the only choice without abandonment).
    metric="fail": (late + abandoned) / (served + abandoned), for runs with
                   walk-in abandonment (see abandonment.py).
    """
    r = run_simulation(list(staffing), rates, replications=reps, seed=seed,
                       mean_service=mean_service, wait_threshold=threshold, **sim_kwargs)
    arr = np.array(r.daily_arrivals, dtype=float)
    late = np.array(r.daily_late, dtype=float)
    if metric == "fail":
        aband = np.array(r.daily_abandoned, dtype=float)
        arr, late = arr + aband, late + aband
    elif metric != "late":
        raise ValueError(metric)
    per_hour = [ratio_ci(late[:, i], arr[:, i]) for i in range(SLOTS)]
    overall, _ = ratio_ci(late.sum(axis=1), arr.sum(axis=1))
    return Evaluation(
        staffing=tuple(staffing),
        staff_hours=sum(staffing),
        late_prob=[p for p, _ in per_hour],
        late_ci=[ci for _, ci in per_hour],
        overall_late=overall,
        mean_daily_p90=float(np.mean(r.daily_p90s)),
        frac_days_p90_ok=float(np.mean(np.array(r.daily_p90s) <= threshold)),
        mean_wait=r.mean_wait,
    )


# ============================================================================
# Simulation greedy staffing (SGS)
# ============================================================================

def _feasible(staffing, rates, mean_service, threshold, alpha, reps, seed,
              criterion="point", **kw):
    """
    criterion="point": every hour's estimated late probability <= alpha.
    criterion="ucb":   every hour's upper 95% bound <= alpha (chance-constrained;
                       guards against plans that only pass by sampling luck).
    """
    ev = evaluate(staffing, rates, mean_service, threshold, reps=reps, seed=seed, **kw)
    if criterion == "ucb":
        return all(hi <= alpha for _, hi in ev.late_ci), ev
    return all(p <= alpha for p in ev.late_prob), ev


def simulation_staffing(rates: list, mean_service: float, threshold: float = 15.0,
                        alpha: float = 0.10, reps: int = DESIGN_REPS,
                        seed: int = DESIGN_SEED, start: list = None,
                        max_iter: int = 200, criterion: str = "point",
                        **sim_kwargs) -> tuple[list, int]:
    """
    Smallest-found plan meeting P(W > threshold | arrival hour i) <= alpha in
    every hour, estimated on the design seeds with common random numbers.

    Phase 1 adds one window at a time to the earliest failing hour (later
    hours depend on earlier staffing, never the reverse, except through
    service that spills across a boundary). Phase 2 removes windows while the
    plan stays feasible. Returns (plan, simulator calls used).
    """
    plan = list(start) if start else analytic_plans(rates, mean_service, threshold, alpha)["SIPP"]
    calls = 0
    for _ in range(max_iter):
        ok, ev = _feasible(plan, rates, mean_service, threshold, alpha, reps, seed,
                           criterion, **sim_kwargs)
        calls += 1
        if ok:
            break
        upper = [hi for _, hi in ev.late_ci] if criterion == "ucb" else ev.late_prob
        first_bad = next(i for i, p in enumerate(upper) if p > alpha)
        plan[first_bad] += 1
    else:
        raise RuntimeError("SGS did not converge")

    def check_all(candidates):
        with ThreadPoolExecutor() as pool:
            results = list(pool.map(
                lambda p: _feasible(p, rates, mean_service, threshold, alpha, reps, seed,
                                    criterion, **sim_kwargs),
                candidates))
        return [(max(ev.late_prob), p) for p, (ok, ev) in zip(candidates, results) if ok]

    # Phase 2: local search over cost-reducing moves. First try every
    # single-window removal; if none is feasible, try "remove two, add one"
    # moves, which shift staff between hours while still saving a staff-hour.
    # Among feasible moves keep the one with the lowest worst-hour late
    # probability; stop at a local optimum of this neighborhood.
    while True:
        removals = []
        for i in range(SLOTS):
            if plan[i] > 1:
                trial = list(plan)
                trial[i] -= 1
                removals.append(trial)
        calls += len(removals)
        feasible = check_all(removals)
        if not feasible:
            shifts = []
            for i, k in itertools.combinations(range(SLOTS), 2):
                if plan[i] <= 1 or plan[k] <= 1:
                    continue
                for j in range(SLOTS):
                    if j in (i, k):
                        continue
                    trial = list(plan)
                    trial[i] -= 1
                    trial[k] -= 1
                    trial[j] += 1
                    shifts.append(trial)
            calls += len(shifts)
            feasible = check_all(shifts)
        if not feasible:
            return plan, calls
        plan = min(feasible)[1]

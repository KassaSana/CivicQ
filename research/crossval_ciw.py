"""
Independent re-implementation of the CivicQ office model in Ciw
(Palmer, Knight, Harper & Hawa 2018, Journal of Simulation), used to
cross-validate the C++ simulator.

Model mapping:
    arrivals  ciw.dists.PoissonIntervals: exact piecewise-constant Poisson
              process (Poisson counts per hour, uniform times within the hour)
    service   Exponential / Lognormal (Ciw takes the underlying normal's
              mu, sigma) / Deterministic, all with the requested mean
    windows   ciw.Schedule; after closing the last hour's staffing stays on
              until everyone inside has been served

Schedule semantics differ, and naive use is badly wrong. At every shift end
Ciw brings on a *fresh* set of servers; with preemption=False the busy servers
of the ending shift finish their customers in overtime alongside them. So
hourly shifts with an unchanged count add phantom capacity every hour (the
late rate for two windows at 11.1 arrivals/hour halves, 0.106 vs 0.228).
Consecutive equal counts are therefore merged into one shift, which makes
constant staffing exactly equivalent to CivicQ. When the count really
changes, CivicQ's closing window finishes its customer *as part of* the new
count, which Ciw cannot express; its two nearest options bracket it:
    preemption=False     extra overtime capacity   -> lower bound on lateness
    preemption="resume"  interrupted service       -> upper bound on lateness
"""

import math
import sys
from pathlib import Path

import ciw
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from staffing_methods import SLOT_MINUTES, SLOTS  # noqa: E402


def _service_dist(mean: float, dist: str, cv: float):
    if dist == "exp":
        return ciw.dists.Exponential(rate=1.0 / mean)
    if dist == "det":
        return ciw.dists.Deterministic(value=mean)
    if dist == "lognormal":
        sigma2 = math.log(1.0 + cv * cv)
        return ciw.dists.Lognormal(mean=math.log(mean) - sigma2 / 2, sd=math.sqrt(sigma2))
    raise ValueError(dist)


def ciw_days(staffing: list, rates_per_hour: list, mean_service: float,
             threshold: float = 15.0, reps: int = 1000, seed: int = 500_000,
             dist: str = "exp", cv: float = 1.0, duration: float = SLOTS * SLOT_MINUTES,
             preemption=False):
    """
    Simulate `reps` independent days in Ciw.

    Returns (arrivals[rep][slot], late[rep][slot], mean_wait[rep]) with slots
    indexed by arrival hour, matching CivicQ's per-replication output.
    """
    # The last slot runs until the doors close, as in CivicQ's --duration
    endpoints = [SLOT_MINUTES * (i + 1) for i in range(SLOTS - 1)] + [duration]
    rates = [r / 60.0 for r in rates_per_hour]

    # One Ciw shift per run of equal staffing (see module docstring); the
    # final shift covers the rest of the hour-8 staffing and the drain
    counts, ends = [], []
    for i, c in enumerate(staffing):
        if counts and counts[-1] == c:
            ends[-1] = endpoints[i]
        else:
            counts.append(c)
            ends.append(endpoints[i])
    ends[-1] = 1e12
    arrivals = np.zeros((reps, SLOTS))
    late = np.zeros((reps, SLOTS))
    mean_wait = np.zeros(reps)
    for r in range(reps):
        ciw.seed(seed + r)
        network = ciw.create_network(
            arrival_distributions=[ciw.dists.PoissonIntervals(
                rates=rates, endpoints=endpoints, max_sample_date=duration)],
            service_distributions=[_service_dist(mean_service, dist, cv)],
            number_of_servers=[ciw.Schedule(
                numbers_of_servers=counts, shift_end_dates=ends,
                preemption=preemption)],
        )
        sim = ciw.Simulation(network)
        sim.simulate_until_max_time(duration * 50)
        waits = []
        for rec in sim.get_all_records():
            if rec.record_type != "service":
                continue
            slot = min(int(rec.arrival_date // SLOT_MINUTES), SLOTS - 1)
            arrivals[r, slot] += 1
            late[r, slot] += rec.waiting_time > threshold
            waits.append(rec.waiting_time)
        mean_wait[r] = np.mean(waits) if waits else 0.0
    return arrivals, late, mean_wait


def ratio_and_se(late: np.ndarray, arrivals: np.ndarray) -> tuple:
    """Pooled late rate and its delta-method SE with days as the independent unit."""
    p = late.sum() / arrivals.sum()
    n = len(late)
    resid = late - p * arrivals
    se = math.sqrt((resid ** 2).sum() / (n * (n - 1))) / arrivals.mean()
    return p, se

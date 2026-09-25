"""
Abandonment in walk-in offices: exact per-hour models, staffing rules, scoring
and the return-visit fixed point used by experiment E7.

Two behaviours, one patience distribution (walk-ins only; booked citizens stay):
    renege  hidden queue (a ticket number, no view of the line): the citizen
            joins and leaves once the wait exceeds their patience.
            Stationary model: M/M/c+M, the Erlang-A queue (Garnett, Mandelbaum
            & Reiman 2002), exact for exponential patience.
    balk    visible queue: the citizen sees q people waiting at c open
            windows, expects to wait (q + 1) S / c and leaves at once if that
            exceeds their patience; joiners stay. Stationary model: a
            birth-death chain with state-dependent joining (Naor 1969 style),
            exact for any patience distribution.

Service-level metrics, per arrival hour:
    served-late  late / served. What a ticket log of *served* citizens shows.
    fail         (late + abandoned) / (served + abandoned). The citizen view:
                 anyone who left, or was served after more than T minutes.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from staffing_methods import (  # noqa: E402
    EVAL_REPS, EVAL_SEED, SLOTS, prob_wait_exceeds, ratio_ci,
)
from optimizer import run_simulation  # noqa: E402


# ============================================================================
# Patience distributions (mean in minutes)
# ============================================================================

def patience_survival(x, mean: float, dist: str = "exp", cv: float = 1.0):
    """P(patience > x)."""
    x = np.asarray(x, dtype=float)
    if dist == "exp":
        return np.exp(-x / mean)
    if dist == "det":
        return (x < mean).astype(float)
    if dist == "lognormal":
        sigma2 = math.log(1.0 + cv * cv)
        mu = math.log(mean) - sigma2 / 2
        z = (np.log(np.maximum(x, 1e-300)) - mu) / math.sqrt(2 * sigma2)
        return np.where(x <= 0, 1.0, 0.5 * np.vectorize(math.erfc)(z))
    raise ValueError(dist)


# ============================================================================
# Exact stationary per-hour models
# ============================================================================

@dataclass
class HourMetrics:
    fail: float          # P(abandon, or served after > T)
    abandon: float       # P(leave unserved)
    served_late: float   # P(W > T | served)
    mean_queue: float    # E[number waiting]
    offered_late: float = float("nan")   # P(V > T): the wait a citizen would face if
                                         # they never left (renege model only)


def _birth_death(births: np.ndarray, deaths: np.ndarray) -> np.ndarray:
    """Stationary distribution of a birth-death chain on 0..K (log space)."""
    with np.errstate(divide="ignore"):
        steps = np.log(births[:-1]) - np.log(deaths[1:])
    logp = np.concatenate([[0.0], np.cumsum(steps)])
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


def _state_cap(c: int, lam: float, theta: float) -> int:
    # Waiting-line length is at most Poisson-like with mean ~ lam / theta
    return c + int(lam / max(theta, 1e-9) * 3 + 60)


def renege_metrics(c: int, lam: float, mean_service: float, mean_patience: float,
                   threshold: float, offered: bool = True) -> HourMetrics:
    """
    Erlang-A (M/M/c+M) at arrival rate `lam` per minute.

    A tagged arrival that finds j people waiting (all windows busy) advances
    at rate c*mu + j*theta (a service completion or someone ahead leaving) and
    abandons at rate theta. The absorbing chain gives P(served within T | j)
    for every j with one matrix exponential.
    """
    mu, theta = 1.0 / mean_service, 1.0 / mean_patience
    K = _state_cap(c, lam, theta)
    n = np.arange(K + 1)
    pi = _birth_death(np.full(K + 1, lam),
                      np.minimum(n, c) * mu + np.maximum(n - c, 0) * theta)
    # Only queue lengths with non-negligible probability enter the matrix
    # exponential, which keeps large offices cheap
    tail = np.cumsum(pi[::-1])[::-1]
    J = max(1, int(np.searchsorted(-tail[c:], -1e-13)))   # j = 0..J-1 waiting ahead
    j = np.arange(J)
    advance = c * mu + j * theta
    served_state = np.zeros(J + 1)
    served_state[J] = 1.0

    def absorbed_by_T(leave_rate):
        # Tagged citizen who may leave at `leave_rate`; last state: served.
        # Only the column into "served" is needed, so apply the (sparse,
        # bidiagonal) generator's exponential to a vector; large offices
        # have thousands of queue states
        diag = np.concatenate([-(advance + leave_rate), [0.0]])
        below = np.concatenate([advance[1:], [0.0]])   # j -> j - 1; "served" absorbs
        Q = diags([diag, below], [0, -1], shape=(J + 1, J + 1), format="lil")
        Q[0, J] = advance[0]
        return expm_multiply(Q.tocsr() * threshold, served_state)[:J]

    within_T = absorbed_by_T(theta)                 # P(served by T | j ahead)
    # Same chain for a citizen who never leaves: P(offered wait V <= T | j ahead)
    # (skipped with offered=False, which halves the cost for large offices)
    offered_within_T = absorbed_by_T(0.0) if offered else None
    eventually = np.cumprod(advance / (advance + theta))  # P(served eventually | j ahead)
    immediate = pi[:c].sum()
    wait_pi = pi[c:c + J]
    ok = immediate + (wait_pi * within_T).sum()
    served = immediate + (wait_pi * eventually).sum()
    return HourMetrics(fail=1.0 - ok, abandon=1.0 - served,
                       served_late=(served - ok) / served,
                       mean_queue=float((np.maximum(n - c, 0) * pi).sum()),
                       offered_late=(float((wait_pi * (1.0 - offered_within_T)).sum())
                                     if offered else float("nan")))


def balk_metrics(c: int, lam: float, mean_service: float, mean_patience: float,
                 threshold: float, patience_dist: str = "exp",
                 patience_cv: float = 1.0) -> HourMetrics:
    """
    Visible queue: with k waiting and all c windows busy an arrival joins with
    probability P(patience > (k + 1) S / c). Joiners never leave, so a joiner
    with k ahead waits Erlang(k + 1, c*mu): P(W > T) = P(Poisson(c*mu*T) <= k).
    """
    mu = 1.0 / mean_service
    # Joining probability decays with k; cap where it is negligible
    x = mean_patience
    while patience_survival(x, mean_patience, patience_dist, patience_cv) > 1e-12 and x < 1e6:
        x *= 2
    K = c + int(x * c / mean_service) + 5
    n = np.arange(K + 1)
    k = np.maximum(n - c, 0)
    join = np.where(n < c, 1.0, patience_survival((k + 1) * mean_service / c,
                                                  mean_patience, patience_dist, patience_cv))
    pi = _birth_death(lam * join, np.minimum(n, c) * mu)
    late_if_join = np.where(n < c, 0.0, poisson.cdf(k, c * mu * threshold))
    balk = float((pi * (1.0 - join)).sum())
    joined_late = float((pi * join * late_if_join).sum())
    return HourMetrics(fail=balk + joined_late, abandon=balk,
                       served_late=joined_late / (1.0 - balk),
                       mean_queue=float((k * pi).sum()))


def hour_metrics(mode: str, c: int, rate_per_hour: float, mean_service: float,
                 mean_patience: float, threshold: float, patience_dist: str = "exp",
                 patience_cv: float = 1.0) -> HourMetrics:
    lam = rate_per_hour / 60.0
    if mode == "renege":
        # Exact only for exponential patience; other families use an exponential
        # with the same mean (the usual Erlang-A practice)
        return renege_metrics(c, lam, mean_service, mean_patience, threshold, offered=False)
    if mode == "balk":
        return balk_metrics(c, lam, mean_service, mean_patience, threshold,
                            patience_dist, patience_cv)
    raise ValueError(mode)


def required_windows(rate_per_hour: float, mean_service: float, threshold: float,
                     alpha: float, mode: str = "none", mean_patience: float = 30.0,
                     patience_dist: str = "exp", patience_cv: float = 1.0) -> int:
    """
    Fewest windows meeting P(late or left) <= alpha in the stationary per-hour
    model; mode="none" is Erlang-C. The failure rate falls with every added
    window, so bisection applies.
    """
    load = rate_per_hour / 60.0 * mean_service

    def ok(c):
        if mode == "none":
            return prob_wait_exceeds(c, load, mean_service, threshold) <= alpha
        return hour_metrics(mode, c, rate_per_hour, mean_service, mean_patience, threshold,
                            patience_dist, patience_cv).fail <= alpha

    hi = max(2, int(load + 6 * math.sqrt(load) + 6))
    while not ok(hi):
        hi *= 2
    # ok(lo) is False: throughput lambda (1 - P(leave)) = mu E[busy] < c mu
    # (some window is idle with positive probability), so with
    # c <= (1 - alpha) R more than alpha of arrivals leave (Erlang-C: c <= R).
    # This also settles ties the models can only resolve to ~1e-13 (E9b)
    lo = max(0, math.floor((1.0 - alpha) * load + 1e-9))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def renege_abandon(c: int, lam: float, mean_service: float, mean_patience: float) -> float:
    """Erlang-A P(abandon) = theta E[Q] / lambda; needs no matrix exponential."""
    mu, theta = 1.0 / mean_service, 1.0 / mean_patience
    K = c + int(max(lam - c * mu, 0.0) / theta * 1.5 + 12 * math.sqrt(lam / theta + 1) + 60)
    n = np.arange(K + 1)
    pi = _birth_death(np.full(K + 1, lam),
                      np.minimum(n, c) * mu + np.maximum(n - c, 0) * theta)
    return float(theta * (np.maximum(n - c, 0) * pi).sum() / lam)


def required_windows_with_returns(rate_per_hour: float, mean_service: float,
                                  threshold: float, alpha: float, mean_patience: float,
                                  return_prob: float):
    """
    Stationary hidden queue (Erlang-A) where each citizen who leaves comes back
    with probability r, at the same rate profile. With c windows the total rate
    x solves x = lambda + r x P_ab(c, x); it exists only if the windows can
    serve the fresh demand in the long run. Returns (fewest windows with
    P(late or left) <= alpha at the fixed point, that fixed-point rate).
    """
    lam = rate_per_hour / 60.0

    def fixed_point(c):
        g = lambda x: lam + return_prob * x * renege_abandon(c, x, mean_service,
                                                             mean_patience) - x
        lo, hi = lam, lam * 1.5
        while g(hi) > 0:
            lo, hi = hi, hi * 2
            if hi > 50 * lam:
                return None
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if g(mid) > 0 else (lo, mid)
        return (lo + hi) / 2

    c = max(1, int(lam * mean_service * 0.8))
    while True:
        x = fixed_point(c)
        if x is not None and renege_metrics(c, x, mean_service, mean_patience,
                                            threshold).fail <= alpha:
            return c, x * 60.0
        c += 1


def fluid_discount(threshold: float, alpha: float, mean_patience: float,
                   patience_dist: str = "exp", patience_cv: float = 1.0) -> float:
    """
    Large-office (fluid) staffing discount from abandonment, as a share of the
    offered load. With c < R windows a fraction 1 - c/R of arrivals must leave,
    and those who stay wait w with G(w) = 1 - c/R, G the patience CDF. The
    failure rate is 1 - c/R while w <= T and jumps to 1 beyond it, so the
    cheapest feasible c is R (1 - min(alpha, G(T))). Erlang-C needs c > R, so
    this is also the limiting relative saving.
    """
    return min(alpha, 1.0 - float(patience_survival(threshold, mean_patience,
                                                    patience_dist, patience_cv)))


def sipp_abandonment(rates: list, mean_service: float, threshold: float, alpha: float,
                     mode: str, mean_patience: float, metric: str = "fail",
                     patience_dist: str = "exp", patience_cv: float = 1.0) -> list:
    """Per-hour SIPP with the abandonment-aware stationary model."""
    plan = []
    for r in rates:
        c = 1
        while True:
            m = hour_metrics(mode, c, r, mean_service, mean_patience, threshold,
                             patience_dist, patience_cv)
            value = m.fail if metric == "fail" else m.served_late
            if value <= alpha:
                break
            c += 1
        plan.append(c)
    return plan


# ============================================================================
# Scoring a plan by simulation
# ============================================================================

@dataclass
class AbandonEval:
    staffing: tuple
    staff_hours: int
    served_late: list        # Per hour, late / served
    served_late_ci: list
    fail: list               # Per hour, (late + abandoned) / (served + abandoned)
    fail_ci: list
    abandon: list            # Per hour, abandoned / (served + abandoned)
    overall_fail: float
    overall_abandon: float
    abandoned_per_day: float
    arrivals_per_day: float
    wasted_minutes_per_day: float   # Time spent inside by citizens who then left
    mean_wait_served: float
    balked_per_day: float = 0.0     # Left on arrival (visible line or a wait display)

    def worst(self, metric: str) -> float:
        return max(self.fail if metric == "fail" else self.served_late)

    def misses(self, metric: str, alpha: float) -> int:
        ci = self.fail_ci if metric == "fail" else self.served_late_ci
        return sum(1 for lo, _ in ci if lo > alpha)


def score(staffing: list, rates: list, mean_service: float, threshold: float = 15.0,
          reps: int = EVAL_REPS, seed: int = EVAL_SEED, **sim_kwargs) -> AbandonEval:
    r = run_simulation(list(staffing), rates, replications=reps, seed=seed,
                       mean_service=mean_service, wait_threshold=threshold, **sim_kwargs)
    served = np.array(r.daily_arrivals, dtype=float)
    late = np.array(r.daily_late, dtype=float)
    aband = (np.array(r.daily_abandoned, dtype=float)
             if r.daily_abandoned else np.zeros_like(served))
    everyone = served + aband
    sl = [ratio_ci(late[:, i], served[:, i]) for i in range(SLOTS)]
    fl = [ratio_ci(late[:, i] + aband[:, i], everyone[:, i]) for i in range(SLOTS)]
    ab = [ratio_ci(aband[:, i], everyone[:, i])[0] for i in range(SLOTS)]
    return AbandonEval(
        staffing=tuple(staffing),
        staff_hours=sum(staffing),
        served_late=[p for p, _ in sl], served_late_ci=[ci for _, ci in sl],
        fail=[p for p, _ in fl], fail_ci=[ci for _, ci in fl],
        abandon=ab,
        overall_fail=float((late.sum() + aband.sum()) / everyone.sum()),
        overall_abandon=float(aband.sum() / everyone.sum()),
        abandoned_per_day=float(aband.sum(axis=1).mean()),
        arrivals_per_day=float(everyone.sum(axis=1).mean()),
        wasted_minutes_per_day=float(np.mean(r.daily_abandoned_wait or [0.0])),
        mean_wait_served=r.mean_wait,
        balked_per_day=float(np.mean(r.daily_balked)) if r.daily_balked else 0.0,
    )


# ============================================================================
# Mandatory services: citizens who leave come back on a later day
# ============================================================================

def return_rates(rates: list, returns_per_day: float, timing: str) -> list:
    """Fresh demand plus returning citizens, in citizens per hour."""
    if timing == "profile":       # Returners arrive like everyone else
        total = sum(rates)
        return [r + returns_per_day * r / total for r in rates]
    if timing == "opening":       # Returners come first thing in the morning
        return [rates[0] + returns_per_day] + list(rates[1:])
    raise ValueError(timing)


@dataclass
class ReturnSteadyState:
    stable: bool
    returns_per_day: float        # R at the fixed point R = r * L(R)
    fresh_per_day: float
    evaluation: AbandonEval = None

    @property
    def repeat_visits_per_100(self) -> float:
        """Repeat visits per 100 fresh citizens (per 100 completed transactions when r = 1)."""
        return 100.0 * self.returns_per_day / self.fresh_per_day


def return_fixed_point(staffing: list, rates: list, mean_service: float,
                       return_prob: float, timing: str, threshold: float = 15.0,
                       reps: int = 400, seed: int = EVAL_SEED, tol: float = 0.05,
                       max_factor: float = 3.0, **sim_kwargs) -> ReturnSteadyState:
    """
    Steady state of a mandatory service: each citizen who leaves returns on a
    later day with probability `return_prob`, so daily returns R solve
    R = r * L(R), where L(R) is the expected number who leave on a day with R
    returners. L is increasing in R, so h(R) = r L(R) - R has at most one
    stable root (checked in Round 9, section 5.13), found by bisection on
    common random numbers. If no root lies below `max_factor` x fresh demand,
    the backlog grows without bound.
    """
    fresh = sum(rates)

    def losses(R):
        ev = score(staffing, return_rates(rates, R, timing), mean_service, threshold,
                   reps=reps, seed=seed, **sim_kwargs)
        return ev.abandoned_per_day, ev

    lo, hi = 0.0, max(1.0, fresh * 0.1)
    loss_hi, _ = losses(hi)
    while return_prob * loss_hi > hi:
        lo, hi = hi, hi * 2
        if hi > max_factor * fresh:
            return ReturnSteadyState(False, math.inf, fresh)
        loss_hi, _ = losses(hi)
    while hi - lo > tol * max(1.0, lo):
        mid = (lo + hi) / 2
        loss_mid, _ = losses(mid)
        if return_prob * loss_mid > mid:
            lo = mid
        else:
            hi = mid
    R = (lo + hi) / 2
    _, ev = losses(R)
    return ReturnSteadyState(True, R, fresh, ev)

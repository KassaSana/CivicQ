"""
Wait displays for a mandatory service, where everyone sent home comes back
(Round 14, REPORT section 5.18).

Round 13 (displays.py) showed that the failure-minimizing display says "over
T" once the offered wait V reaches the target. It counted each citizen it
turns away as a failure and stopped there. For a mandatory service they
return, so the day's input is fresh demand plus returners, and the display
changes that input. This module is the stationary theory.

Setting. Fresh walk-ins arrive at rate lam; every loss (balker or reneger)
comes back later with probability r, with the same patience. In steady state
the total arrival rate L solves

    L = lam + r (L - theta(L)),      theta(L) = L P(served at L),

theta being the throughput of the Round 13 exact law at arrival rate L.

T1 (throughput rises with load, for any display of V). By displays.py,
P(V = 0) = p0 = E / (E + L J(L)) with J(L) = int exp(L U(x) - c mu x) dx,
U = int u >= 0 not depending on L, and E = sum_{j<c} pi_j / pi_{c-1}
decreasing in a = L / mu. So p0 falls strictly in L. Busy windows are
E[busy] = c (1 - p0) + p0 m(a), with m(a) < c the mean of a Poisson(a)
truncated to {0, ..., c-1}, increasing in a. Both terms push E[busy] up, and
theta = mu E[busy]. Hence the day-to-day map L -> lam + r (L - theta(L)) has
slope r (1 - theta') in [0, r]: one steady state, always stable, for every
display that is a function of V. Round 9's "a mandatory service cannot tip"
(section 5.13) survives any such display.

T2 (a display that overstates more serves fewer). If phi1 >= phi2 pointwise
then u1 <= u2, U1 <= U2, J1 <= J2 and p0_1 >= p0_2 at every L, so
theta1 <= theta2 at every L and the steady state has L1* >= L2*. The cutoff
display costs repeat visits; the hidden queue (phi = 0) needs the fewest.

When r = 1 the steady state serves exactly the fresh demand (theta(L*) = lam),
so per citizen: visits = L* / lam, the share whose (eventual) service is late
is L* P(served late) / lam, and the oracle cutoff makes it zero. The price of
a display is its exchange rate

    eps = (extra visits per citizen) / (late services avoided per citizen),

which linearizes as eps ~ kappa / theta', kappa being the throughput the
display gives up per late service it prevents (at the hidden steady state)
and 1 / theta' the relaxation time of the day-to-day chain at r = 1
(Round 9's critical slowing down).
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from displays import HIDDEN, Display, DisplayHour, display_hour, served_rate  # noqa: E402
from learning import Patience  # noqa: E402
from abandonment import return_rates  # noqa: E402
from staffing_methods import SLOT_MINUTES  # noqa: E402


def throughput(c: int, total_rate: float, mean_service: float, patience: Patience,
               threshold: float, show: Display = HIDDEN) -> float:
    """theta(L): citizens served per minute at total arrival rate L per minute."""
    hour = display_hour(c, total_rate, mean_service, patience, threshold, show)
    return served_rate(hour, total_rate)


def slope(c, total_rate, mean_service, patience, threshold, show=HIDDEN, rel=1e-3) -> float:
    """theta'(L) by a central difference."""
    d = rel * total_rate
    up = throughput(c, total_rate + d, mean_service, patience, threshold, show)
    down = throughput(c, total_rate - d, mean_service, patience, threshold, show)
    return (up - down) / (2 * d)


# ============================================================================
# Stationary steady state with returns
# ============================================================================

@dataclass
class ReturnState:
    fresh: float           # Fresh arrivals per minute
    total: float           # All arrivals per minute at the steady state (L*)
    hour: DisplayHour      # The exact law at L*
    return_prob: float

    @property
    def visits(self) -> float:
        """Visits per fresh citizen (arrivals / fresh)."""
        return self.total / self.fresh

    @property
    def late_share(self) -> float:
        """Late services per fresh citizen (with r = 1: share served late in the end)."""
        return self.total * self.hour.served_late / self.fresh

    @property
    def lost_min(self) -> float:
        """Minutes lost inside per fresh citizen over all visits (waits and leavers' time)."""
        return self.total * self.hour.lost_min / self.fresh

    @property
    def kpi(self) -> float:
        """The office's own figure: late or left per visit."""
        return self.hour.fail

    @property
    def served(self) -> float:
        return served_rate(self.hour, self.total)


def fixed_point(c: int, fresh_rate: float, mean_service: float, patience: Patience,
                threshold: float, show: Display = HIDDEN, return_prob: float = 1.0,
                max_factor: float = 200.0, iters: int = 60):
    """
    The steady state L* = lam + r (L* - theta(L*)), by bisection (the gap
    g(L) = lam + r (L - theta(L)) - L falls in L by T1). Rates per minute.
    Returns None when no steady state lies below max_factor x lam (for r = 1
    that happens when lam reaches capacity c / S).
    """
    g = lambda L: fresh_rate + return_prob * (
        L - throughput(c, L, mean_service, patience, threshold, show)) - L
    lo, hi = fresh_rate, fresh_rate * 1.05
    while g(hi) > 0:
        lo, hi = hi, hi * 1.5
        if hi > max_factor * fresh_rate:
            return None
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if g(mid) > 0 else (lo, mid)
    total = (lo + hi) / 2
    return ReturnState(fresh_rate, total,
                       display_hour(c, total, mean_service, patience, threshold, show),
                       return_prob)


def relaxation(c, state: ReturnState, mean_service, patience, threshold, show=HIDDEN) -> float:
    """Days for a disturbance to decay by e: 1 / (1 - r (1 - theta'(L*)))."""
    th = slope(c, state.total, mean_service, patience, threshold, show)
    return 1.0 / (1.0 - state.return_prob * (1.0 - th))


@dataclass
class Exchange:
    base: ReturnState
    display: ReturnState
    eps: float             # Extra visits per late service avoided
    kappa: float           # Throughput given up per late service prevented, at the base state
    theta_prime: float     # theta'(L*) under the display
    relax_base: float      # Relaxation times (days) of the day-to-day chain
    relax_display: float

    @property
    def eps_linear(self) -> float:
        return self.kappa / self.theta_prime


def exchange_rate(c: int, fresh_rate: float, mean_service: float, patience: Patience,
                  threshold: float, show: Display, base: Display = HIDDEN,
                  return_prob: float = 1.0):
    """The display's price in repeat visits, against `base`, with its decomposition."""
    b = fixed_point(c, fresh_rate, mean_service, patience, threshold, base, return_prob)
    d = fixed_point(c, fresh_rate, mean_service, patience, threshold, show, return_prob)
    if b is None or d is None:
        return None
    avoided = b.late_share - d.late_share
    at_base = display_hour(c, b.total, mean_service, patience, threshold, show)
    lost = b.served - served_rate(at_base, b.total)
    late_rate_cut = b.total * (b.hour.served_late - at_base.served_late)
    th = slope(c, d.total, mean_service, patience, threshold, show)
    return Exchange(
        base=b, display=d,
        eps=(d.visits - b.visits) / avoided if avoided > 0 else math.nan,
        kappa=lost / late_rate_cut if late_rate_cut > 0 else math.nan,
        theta_prime=th,
        relax_base=relaxation(c, b, mean_service, patience, threshold, base),
        relax_display=1.0 / (1.0 - return_prob * (1.0 - th)))


# ============================================================================
# Time-varying offices, hour by hour (stationary per hour, like SIPP)
# ============================================================================

@dataclass
class DayState:
    returns: float          # R*: returners per day
    fresh: float            # Fresh citizens per day
    late: float             # Late services per day
    losses: float           # Balkers and renegers per day
    lost_min: float         # Minutes lost inside per day
    kpi: float              # Late or left per visit
    slope: float            # r L'(R*) of the day-to-day map

    @property
    def visits(self) -> float:
        return 1.0 + self.returns / self.fresh

    @property
    def late_share(self) -> float:
        return self.late / self.fresh

    @property
    def lost_min_per_citizen(self) -> float:
        return self.lost_min / self.fresh

    @property
    def relax_days(self) -> float:
        return 1.0 / (1.0 - self.slope) if self.slope < 1 else math.inf


def _day(plan, rates_per_hour, mean_service, patience, threshold, show):
    hours = [display_hour(c, r / 60.0, mean_service, patience, threshold, show)
             for c, r in zip(plan, rates_per_hour)]
    n = np.asarray(rates_per_hour, float) * SLOT_MINUTES / 60.0     # Arrivals per hour
    late = float(sum(k * h.served_late for k, h in zip(n, hours)))
    losses = float(sum(k * (h.balk + h.renege) for k, h in zip(n, hours)))
    lost = float(sum(k * h.lost_min for k, h in zip(n, hours)))
    fail = float(sum(k * h.fail for k, h in zip(n, hours)) / n.sum())
    return late, losses, lost, fail


def day_fixed_point(plan, rates, mean_service, patience: Patience, threshold,
                    show: Display = HIDDEN, return_prob: float = 1.0,
                    timing: str = "profile", max_factor: float = 20.0,
                    iters: int = 40):
    """
    Per-hour stationary steady state of a day with returners: R = r L(R),
    L(R) the day's losses with R returners arriving by `timing`
    (abandonment.return_rates). L' <= 1 hour by hour (T1), so the root is
    unique. Returns None when none lies below max_factor x fresh demand.
    """
    fresh = float(sum(rates))
    L = lambda R: _day(plan, return_rates(rates, R, timing), mean_service, patience,
                       threshold, show)
    gap = lambda R: return_prob * L(R)[1] - R
    lo, hi = 0.0, max(1.0, 0.1 * fresh)
    while gap(hi) > 0:
        lo, hi = hi, hi * 2
        if hi > max_factor * fresh:
            return None
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    R = (lo + hi) / 2
    late, losses, lost, fail = L(R)
    d = max(1e-3 * fresh, 1e-6)
    sl = return_prob * (L(R + d)[1] - L(max(R - d, 0.0))[1]) / (R + d - max(R - d, 0.0))
    return DayState(returns=R, fresh=fresh, late=late, losses=losses, lost_min=lost,
                    kpi=fail, slope=sl)


def day_exchange_rate(base: DayState, display: DayState) -> float:
    """Extra visits per late service avoided, per citizen, for two day states."""
    avoided = base.late_share - display.late_share
    return (display.visits - base.visits) / avoided if avoided > 0 else math.nan


def break_even_trip(base, display) -> float:
    """
    K*: minutes lost inside saved per extra visit. A display saves citizens
    time overall unless a wasted trip costs more than K* minutes.
    Works for ReturnState (per fresh citizen) and DayState.
    """
    lost = lambda s: s.lost_min if isinstance(s, ReturnState) else s.lost_min_per_citizen
    extra = display.visits - base.visits
    return (lost(base) - lost(display)) / extra if extra > 0 else math.inf

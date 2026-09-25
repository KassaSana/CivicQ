"""
A theory of wait displays (Round 13, REPORT section 5.17).

Round 12 found, after the fact, that a wait display can lower failures when
the citizens it sends home would have been served late anyway. This module
makes that exact for displays that are a function of the offered wait V (the
wait a citizen would have if they stayed).

Exact law. In the stationary M/M/c+G ticket queue, a walk-in who finds offered
wait V = x is shown phi(x), leaves at once if phi(x) exceeds their patience
tau, and otherwise joins and leaves at tau if not called by then. They are
served iff tau >= max(x, phi(x)), and whether they are served depends only on
x and their own tau (FIFO: nobody behind them matters). So V is still a Markov
process: it falls at rate 1 and jumps by Exp(c mu) at the arrival of a citizen
who will be served. Level crossing (Baccelli & Hebuterne 1981) gives

    f(x) = lam pi_{c-1} exp(lam U(x) - c mu x),   U(x) = int_0^x u,
    u(x) = 1 - G(max(x, phi(x)))            (the share of arrivals served),

with the atom P(V = 0) from the birth-death chain below c, as in
learning.mmcg_hour (which is the case phi = 0).

Optimal display (Theorem). Write A = int_0^T e^{lam U - c mu x} dx,
B = e^{lam U(T) - c mu T} and Q = int_T^inf e^{lam (U(x) - U(T)) - c mu (x - T)} dx,
which depends only on u above T. With a = lam / mu, rho = a / c and
E = sum_{j<c} pi_j / pi_{c-1}, integration by parts gives

    P(served within T) = N / D,   N = E - 1 + B + c mu A,   D = E + lam A + lam B Q.

(i) Overstating above T lowers u there, hence Q and D, and leaves N alone:
    it always lowers failures.
(ii) With nobody admitted above T (Q = 1 / (c mu)), N / D rises with A and B,
    because E (1 - rho) + rho > 0 (E < rho / (rho - 1) when rho > 1), and A
    and B rise with u below T. So, given the cutoff, overstating below T
    always raises failures.
Together: the failure-minimizing display is exact (or understating) below T
and "over T" above it, the cutoff display. Without the cutoff (Q > 1/(c mu)),
overstating below T can lower failures: it holds back work that would
otherwise delay citizens who are admitted and served late.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
from learning import Patience  # noqa: E402

Display = Callable[[np.ndarray], np.ndarray]


# ============================================================================
# Displays as functions of the offered wait
# ============================================================================

def truthful() -> Display:
    return lambda x: np.asarray(x, dtype=float)


def scaled(kappa: float) -> Display:
    return lambda x: kappa * np.asarray(x, dtype=float)


def cutoff(m: float, kappa: float = 1.0) -> Display:
    """Show kappa * V below m minutes, and "too long" (infinity) from m on."""
    def show(x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= m, np.inf, kappa * x)
    return show


HIDDEN = scaled(0.0)


# ============================================================================
# The exact stationary law
# ============================================================================

@dataclass
class DisplayHour:
    fail: float            # P(late or left)
    on_time: float         # P(served within T)
    balk: float            # P(left at once on seeing the display)
    renege: float          # P(joined, then left)
    served_late: float     # P(served after T), of all arrivals
    p_wait: float          # P(V > 0)
    wait_min: float        # Minutes waited by the served, per arrival
    wasted_min: float      # Minutes spent inside by those who left, per arrival
    x: np.ndarray
    f: np.ndarray          # Density of V on the grid (integrates to p_wait)

    @property
    def lost_min(self) -> float:
        return self.wait_min + self.wasted_min


def _partial_mean(patience: Patience, grid: np.ndarray):
    """K(t) = E[tau; tau <= t] = t G(t) - int_0^t G on a grid."""
    G = patience.cdf(grid)
    intG = np.concatenate([[0.0], np.cumsum((G[1:] + G[:-1]) * np.diff(grid) / 2)])
    return grid * G - intG


def display_hour(c: int, lam: float, mean_service: float, patience: Patience,
                 threshold: float, show: Display = HIDDEN, h: float = 0.01) -> DisplayHour:
    """Exact stationary M/M/c+G ticket queue with a display of the offered wait."""
    mu = 1.0 / mean_service
    a = lam / mu
    X = max(4.0 * threshold, 60.0, 20.0 / (c * mu))
    while True:
        x = np.arange(0.0, X + h / 2, h)
        shown = np.minimum(show(x), 1e12)          # "Too long" = beyond any patience
        u = 1.0 - patience.cdf(np.maximum(x, shown))
        # A cutoff makes u jump at the first "too long" point: integrate that
        # cell as a step (u holds its left value up to the jump), not a ramp
        jump = (shown[1:] >= 1e12) & (shown[:-1] < 1e12)
        cell = np.where(jump, u[:-1], (u[1:] + u[:-1]) / 2)
        U = np.concatenate([[0.0], np.cumsum(cell * h)])
        phi = lam * U - c * mu * x
        if phi[-1] < phi.max() - 40.0:
            break
        X *= 2
    j = np.arange(c)
    log_E = np.logaddexp.reduce((j - (c - 1)) * math.log(a) + gammaln(c) - gammaln(j + 1))
    m = phi.max()
    log_J = m + math.log(np.trapezoid(np.exp(phi - m), x))
    log_denom = np.logaddexp(log_E, math.log(lam) + log_J)
    p0 = math.exp(log_E - log_denom)                 # P(V = 0)
    f = np.exp(math.log(lam) + phi - log_denom)

    below = x <= threshold + 1e-9
    served_density = u * f
    on_time = p0 + float(np.trapezoid(served_density[below], x[below]))
    served = p0 + float(np.trapezoid(served_density, x))
    balk = float(np.trapezoid(np.where(x > 0, patience.cdf(shown), 0.0) * f, x))
    wait_min = float(np.trapezoid(x * served_density, x))
    # Renegers: shown(x) <= tau < x, and they stay tau minutes
    K = _partial_mean(patience, x)
    Kshown = np.interp(np.minimum(shown, x), x, K)
    wasted = float(np.trapezoid(np.clip(K - Kshown, 0.0, None) * f, x))
    return DisplayHour(fail=1.0 - on_time, on_time=on_time, balk=balk,
                       renege=1.0 - served - balk, served_late=served - on_time,
                       p_wait=1.0 - p0, wait_min=wait_min, wasted_min=wasted, x=x, f=f)


def erlang_b(c: int, a: float) -> float:
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


# ============================================================================
# Time-varying offices, hour by hour (stationary per hour, like SIPP)
# ============================================================================

def plan_prediction(plan, rates, mean_service, patience: Patience, threshold,
                    show: Display = HIDDEN) -> dict:
    """Arrival-weighted stationary prediction for a whole day, hour by hour."""
    hours = [display_hour(c, r / 60.0, mean_service, patience, threshold, show)
             for c, r in zip(plan, rates)]
    w = np.asarray(rates, dtype=float) / sum(rates)
    avg = lambda k: float(sum(wi * getattr(hh, k) for wi, hh in zip(w, hours)))
    return {"fail": avg("fail"), "balk": avg("balk"), "renege": avg("renege"),
            "served_late": avg("served_late"), "lost_min": avg("lost_min"),
            "hourly_fail": [hh.fail for hh in hours]}


def sipp_display(rates, mean_service, threshold, alpha, patience: Patience,
                 show: Display = HIDDEN) -> list:
    """Per-hour SIPP for P(late or left) <= alpha under a display."""
    plan = []
    for r in rates:
        lam = r / 60.0
        c = max(1, math.floor((1.0 - alpha) * lam * mean_service))
        while display_hour(c, lam, mean_service, patience, threshold, show).fail > alpha:
            c += 1
        while c > 1 and display_hour(c - 1, lam, mean_service, patience, threshold,
                                     show).fail <= alpha:
            c -= 1
        plan.append(c)
    return plan

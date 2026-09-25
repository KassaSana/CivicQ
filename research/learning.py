"""
Learning by staffing (Round 11, REPORT section 5.15).

An office learns its citizens' patience only from its own ticket log
(Round 10), and what the log shows depends on how the office staffs: a
citizen reveals patience only by waiting. Refitting and restaffing is a
feedback loop

    plan -> waits in the log -> fitted patience -> new plan

whose fixed points are self-confirming: the plan produces exactly the data
that justify it.

Tools:
- mmcg_hour: exact stationary M/M/c+G (hidden queue, any patience CDF G),
  from the virtual-wait density of Baccelli & Hebuterne (1981):
      f(x) = lam pi_{c-1} exp(lam H(x) - c mu x),  H(x) = int_0^x (1 - G),
  with the atom P(V = 0) from the birth-death chain below c. An arrival
  fails (late or left) if V > T, or V <= T and patience < V.
- sipp_g: per-hour SIPP for the failure target with that model.
- limit_fit: the large-sample limit of a current-status MLE fitted to the
  tickets a plan produces (stationary, hour by hour), so the loop can be
  iterated without simulation.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abandonment import patience_survival  # noqa: E402


# ============================================================================
# Patience curves
# ============================================================================

@dataclass(frozen=True)
class Patience:
    """A patience distribution: family 'exp' (mean) or 'lognormal' (mean, cv)."""
    family: str
    mean: float
    cv: float = 1.0

    def cdf(self, x):
        dist = "exp" if self.family == "exp" else "lognormal"
        return 1.0 - patience_survival(x, self.mean, dist, self.cv)

    @staticmethod
    def from_fit(fit) -> "Patience":
        """From a patience_logs.ParamFit."""
        if fit.family == "exp":
            return Patience("exp", fit.mean)
        sigma = fit.params[1]
        return Patience("lognormal", fit.mean, math.sqrt(math.expm1(sigma * sigma)))


# ============================================================================
# Exact stationary M/M/c+G
# ============================================================================

@dataclass
class GHour:
    fail: float          # P(late or left)
    abandon: float       # P(left)
    served_late: float   # P(wait > T | served)
    p_wait: float        # P(V > 0)
    x: np.ndarray        # Grid of offered waits V > 0
    f: np.ndarray        # Density of V on the grid (integrates to p_wait)


def _grid_exponent(c, lam, mu, patience, h):
    """lam H(x) - c mu x on a grid long enough for its tail to be negligible."""
    X = max(60.0, 20.0 / (c * mu))
    while True:
        x = np.arange(0.0, X + h / 2, h)
        surv = 1.0 - patience.cdf(x)
        H = np.concatenate([[0.0], np.cumsum((surv[1:] + surv[:-1]) * h / 2)])
        phi = lam * H - c * mu * x
        if phi[-1] < phi.max() - 40.0:
            return x, phi
        X *= 2


def mmcg_hour(c: int, lam: float, mean_service: float, patience: Patience,
              threshold: float, h: float = 0.02) -> GHour:
    """Exact M/M/c+G at arrival rate `lam` per minute (hidden queue, reneging)."""
    mu = 1.0 / mean_service
    a = lam / mu
    x, phi = _grid_exponent(c, lam, mu, patience, h)
    # pi_{c-1} (E + lam J) = 1, with E = sum_{j<c} pi_j / pi_{c-1}
    j = np.arange(c)
    log_E = np.logaddexp.reduce((j - (c - 1)) * math.log(a) + gammaln(c) - gammaln(j + 1))
    m = phi.max()
    w = np.exp(phi - m)
    log_J = m + math.log(np.trapezoid(w, x))
    log_denom = np.logaddexp(log_E, math.log(lam) + log_J)
    f = np.exp(math.log(lam) + phi - log_denom)
    G = patience.cdf(x)
    below = x <= threshold
    p_v_late = float(np.trapezoid(f[~below], x[~below])) if (~below).sum() > 1 else 0.0
    # Carry the piece of the cell that straddles T
    k = int(below.sum())
    if 0 < k < len(x):
        p_v_late += 0.5 * (f[k - 1] + f[k]) * (x[k] - threshold)
    left_before_T = float(np.trapezoid((G * f)[below], x[below]))
    abandon = float(np.trapezoid(G * f, x))
    p_wait = float(np.trapezoid(f, x))
    fail = p_v_late + left_before_T
    served_late = (p_v_late - (abandon - left_before_T)) / max(1.0 - abandon, 1e-300)
    return GHour(fail=fail, abandon=abandon, served_late=served_late, p_wait=p_wait, x=x, f=f)


def sipp_g(rates: list, mean_service: float, threshold: float, alpha: float,
           patience: Patience) -> list:
    """Per-hour SIPP for P(late or left) <= alpha with the exact M/M/c+G model."""
    plan = []
    for r in rates:
        lam = r / 60.0
        c = max(1, math.floor((1.0 - alpha) * lam * mean_service))
        while mmcg_hour(c, lam, mean_service, patience, threshold).fail > alpha:
            c += 1
        # The failure rate falls with c; step back while still feasible
        while c > 1 and mmcg_hour(c - 1, lam, mean_service, patience, threshold).fail <= alpha:
            c -= 1
        plan.append(c)
    return plan


# ============================================================================
# The learning loop in the large-sample, stationary limit
# ============================================================================

def ticket_distribution(plan, rates, mean_service, truth: Patience, threshold=15.0):
    """
    Per hour: (arrivals per hour, grid, density of V > 0) under the stationary
    model. Tickets with V = 0 carry no information and are left out.
    """
    return [(r, g.x, g.f) for r, c in zip(rates, plan)
            for g in [mmcg_hour(c, r / 60.0, mean_service, truth, threshold)]]


def limit_fit(plan, rates, mean_service, truth: Patience, family: str,
              threshold=15.0) -> Patience:
    """
    Where a current-status MLE of `family` converges on the tickets `plan`
    produces: maximize the expected log-likelihood
        sum_h r_h int f_h(v) [G(v) log F(v) + (1 - G(v)) log(1 - F(v))] dv.
    """
    parts = ticket_distribution(plan, rates, mean_service, truth, threshold)
    xs = [x[1:] for _, x, _ in parts]          # V > 0 only
    ws = [r * f[1:] * (x[1] - x[0]) for r, x, f in parts]
    gs = [truth.cdf(x) for x in xs]
    v = np.concatenate(xs)
    wt = np.concatenate(ws)
    g = np.concatenate(gs)
    keep = wt > 1e-14
    v, wt, g = v[keep], wt[keep], g[keep]

    if family == "exp":
        def nll(lt):
            th = math.exp(lt)
            return -float(np.sum(wt * (g * np.log(-np.expm1(-th * v)) - (1 - g) * th * v)))
        r = minimize_scalar(nll, bounds=(math.log(1e-6), math.log(10.0)), method="bounded",
                            options={"xatol": 1e-9})
        return Patience("exp", 1.0 / math.exp(r.x))
    if family == "lognormal":
        lv = np.log(v)

        def nll(p):
            a, b = p
            e = a + b * lv
            return -float(np.sum(wt * (g * norm.logcdf(e) + (1 - g) * norm.logcdf(-e))))
        r = minimize(nll, np.array([-3.3 / 0.5, 1 / 0.5]), method="Nelder-Mead",
                     options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 5000})
        a, b = r.x
        mu, sigma = -a / b, 1.0 / b
        return Patience("lognormal", math.exp(mu + sigma * sigma / 2),
                        math.sqrt(math.expm1(sigma * sigma)))
    raise ValueError(family)


def learning_map(plan, rates, mean_service, truth: Patience, rule: str,
                 threshold=15.0, alpha=0.10):
    """
    One refit-and-restaff step in the limit. Rules:
      A  fit an exponential, staff by Erlang-A SIPP (Round 10's H41 route)
      B  fit the true family, staff by exact M/M/c+G SIPP
    Returns (new plan, fitted patience).
    """
    family = "exp" if rule == "A" else truth.family
    fitted = limit_fit(plan, rates, mean_service, truth, family, threshold)
    return sipp_g(rates, mean_service, threshold, alpha, fitted), fitted


# ============================================================================
# Finite samples: information per day, and what it does to the plan
# ============================================================================

def _tickets(plans_and_weights, rates, mean_service, truth, threshold):
    """Pooled ticket density: V > 0 grid, expected tickets per day at each point, G."""
    v, wt = [], []
    for plan, p in plans_and_weights:
        for r, x, f in ticket_distribution(plan, rates, mean_service, truth, threshold):
            v.append(x[1:])
            wt.append(p * r * f[1:] * (x[1] - x[0]))
    v, wt = np.concatenate(v), np.concatenate(wt)
    keep = wt > 1e-14
    return v[keep], wt[keep], truth.cdf(v[keep])


def _family_cdf(family, params, v):
    """F(v) under natural parameters: exp (log theta,), lognormal probit (a, b)."""
    if family == "exp":
        return -np.expm1(-math.exp(params[0]) * v)
    return norm.cdf(params[0] + params[1] * np.log(v))


def _natural(p: Patience):
    if p.family == "exp":
        return np.array([math.log(1.0 / p.mean)])
    sigma = math.sqrt(math.log1p(p.cv * p.cv))
    mu = math.log(p.mean) - sigma * sigma / 2
    return np.array([-mu / sigma, 1.0 / sigma])


def _from_natural(family, params) -> Patience:
    if family == "exp":
        return Patience("exp", math.exp(-params[0]))
    a, b = params
    b = max(b, 1e-6)
    mu, sigma = -a / b, 1.0 / b
    return Patience("lognormal", math.exp(mu + sigma * sigma / 2),
                    math.sqrt(math.expm1(sigma * sigma)))


def sampling_cov_per_day(designs, rates, mean_service, truth, fitted: Patience,
                         threshold=15.0):
    """
    Sandwich covariance A^-1 B A^-1 of the current-status MLE (natural
    parameters of fitted.family) for ONE day of log, where `designs` is a
    list of (plan, share of days). Divide by n for n days. With the right
    family A = B (Fisher information).
    """
    v, wt, g = _tickets(designs, rates, mean_service, truth, threshold)
    th = _natural(fitted)
    k = len(th)
    eps = 1e-5

    def F(p):
        return np.clip(_family_cdf(fitted.family, p, v), 1e-300, 1 - 1e-16)

    def ell(p):
        f = F(p)
        return float(np.sum(wt * (g * np.log(f) + (1 - g) * np.log1p(-f))))

    dF = np.empty((k, len(v)))
    for i in range(k):
        e = np.zeros(k)
        e[i] = eps
        dF[i] = (F(th + e) - F(th - e)) / (2 * eps)
    A = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = eps * 10, eps * 10
            A[i, j] = -(ell(th + ei + ej) - ell(th + ei - ej) - ell(th - ei + ej)
                        + ell(th - ei - ej)) / (4 * (eps * 10) ** 2)
    f = F(th)
    s2 = (g * (1 - f) ** 2 + (1 - g) * f ** 2) / (f * (1 - f)) ** 2
    B = (dF * wt * s2) @ dF.T
    Ai = np.linalg.inv(A)
    return Ai @ B @ Ai


def plan_distribution(designs, rates, mean_service, truth, rule, days, draws=400,
                      seed=0, threshold=15.0, alpha=0.10):
    """
    Staff-hours of the refitted plan after `days` days of log (normal
    approximation to the MLE, pushed through SIPP). Returns an array of hours.
    """
    family = "exp" if rule == "A" else truth.family
    # Limit of the fit on the pooled design
    v, wt, g = _tickets(designs, rates, mean_service, truth, threshold)
    fitted = _limit_on(v, wt, g, family)
    cov = sampling_cov_per_day(designs, rates, mean_service, truth, fitted, threshold) / days
    rng = np.random.default_rng(seed)
    th = _natural(fitted)
    hours, cache = [], {}
    for z in rng.multivariate_normal(th, cov, size=draws):
        key = tuple(np.round(z, 3))
        if key not in cache:
            cache[key] = sum(sipp_g(rates, mean_service, threshold, alpha,
                                    _from_natural(family, z)))
        hours.append(cache[key])
    return np.array(hours), fitted


def _limit_on(v, wt, g, family) -> Patience:
    if family == "exp":
        def nll(lt):
            th = math.exp(lt)
            return -float(np.sum(wt * (g * np.log(-np.expm1(-th * v)) - (1 - g) * th * v)))
        r = minimize_scalar(nll, bounds=(math.log(1e-6), math.log(10.0)), method="bounded",
                            options={"xatol": 1e-9})
        return Patience("exp", 1.0 / math.exp(r.x))
    lv = np.log(v)

    def nll(p):
        e = p[0] + p[1] * lv
        return -float(np.sum(wt * (g * norm.logcdf(e) + (1 - g) * norm.logcdf(-e))))
    r = minimize(nll, np.array([-6.6, 2.0]), method="Nelder-Mead",
                 options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 5000})
    return _from_natural("lognormal", r.x)


def failures_per_day(plan, rates, mean_service, truth, threshold=15.0) -> float:
    """Expected citizens late or left per day (stationary, hour by hour)."""
    return float(sum(r * mmcg_hour(c, r / 60.0, mean_service, truth, threshold).fail
                     for r, c in zip(rates, plan)))


def explore_plan(plan, phi: float) -> list:
    """An exploration day: each hour's windows scaled by phi (at least one fewer, at least 1)."""
    return [max(1, min(c - 1, round(phi * c))) for c in plan]


def explore_map(plan, rates, mean_service, truth, rule, p: float, phi: float,
                threshold=15.0, alpha=0.10):
    """
    One refit with exploration: a share p of days runs explore_plan(plan, phi),
    the rest run `plan`; the fit uses the pooled tickets. Returns (plan, fitted).
    """
    family = "exp" if rule == "A" else truth.family
    designs = [(plan, 1.0 - p)] + ([(explore_plan(plan, phi), p)] if p > 0 else [])
    v, wt, g = _tickets(designs, rates, mean_service, truth, threshold)
    fitted = _limit_on(v, wt, g, family)
    return sipp_g(rates, mean_service, threshold, alpha, fitted), fitted


def iterate_explore(start, rates, mean_service, truth, rule, p, phi, max_steps=20, **kw):
    """Fixed point of explore_map (or the last plan of a cycle). Returns the path."""
    path, seen, plan = [(list(start), None)], {tuple(start)}, list(start)
    for _ in range(max_steps):
        plan, fitted = explore_map(plan, rates, mean_service, truth, rule, p, phi, **kw)
        path.append((plan, fitted))
        if tuple(plan) in seen:
            break
        seen.add(tuple(plan))
    return path


def iterate_map(start, rates, mean_service, truth, rule, max_steps=20, **kw):
    """Iterate the limit map to a fixed point (or a cycle). Returns the path."""
    path, seen = [(list(start), None)], {tuple(start)}
    plan = list(start)
    for _ in range(max_steps):
        plan, fitted = learning_map(plan, rates, mean_service, truth, rule, **kw)
        path.append((plan, fitted))
        if tuple(plan) in seen:
            break
        seen.add(tuple(plan))
    return path

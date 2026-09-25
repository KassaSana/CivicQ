"""
Learning patience from an office's own ticket log (Round 10, REPORT section 5.14).

A ticket system never sees a citizen leave. It sees a ticket called, and
whether anyone came. For each walk-in ticket the log gives the virtual wait
V (issue to call) and a flag:
    present  patience >= V   (served, wait V)
    absent   patience <  V   (reneged some time before the call)
This is current-status data (interval censoring, case 1). Under FIFO a
citizen's V is set by the people ahead and by staffing, not by their own
patience, so V and patience are independent and G(t) = P(patience < t) is
identified wherever V has support.

Estimators of G:
    naive_km   Kaplan-Meier with an absent ticket's call time as its
               departure (what a call center, which records hang-ups, does)
    cs_npmle   nonparametric MLE: isotonic regression of "absent" on V
               (Groeneboom & Wellner 1992; converges at n^(-1/3))
    cs_mle     parametric MLE (exponential or lognormal), family by AIC

The visible line (balking) leaves no trace in the ticket log. With a
timestamped door counter each arrival is current-status data at its
expected wait (q + 1) S / c, rebuilt from the log (door_counter_data).
"""

import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import isotonic_regression, minimize, minimize_scalar
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from abandonment import patience_survival  # noqa: E402
from optimizer import find_simulator  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
LOG_SEED = 500_000          # Seeds 500000.. generate ticket logs (disjoint from design/eval seeds)
CHUNK_DAYS = 2000           # Days per simulator call (keeps the CSV small)


# ============================================================================
# Ticket logs
# ============================================================================

@dataclass
class TicketLog:
    """Informative walk-in tickets (V > 0) from many days, sorted by day."""
    day: np.ndarray          # Day index of each ticket
    v: np.ndarray            # Virtual wait: issue to call (minutes)
    absent: np.ndarray       # True if nobody came when the ticket was called
    patience: np.ndarray     # True patience (validation only; never seen by an office)
    days: int
    walkins_per_day: float   # All walk-ins, including those served at once

    def day_starts(self) -> np.ndarray:
        """Row offset of each day's first ticket (length days + 1)."""
        return np.searchsorted(self.day, np.arange(self.days + 1))

    def sample(self, n: int, rng: np.random.Generator) -> tuple:
        """v and absent for n distinct days drawn at random."""
        starts = self.day_starts()
        picked = rng.choice(self.days, size=n, replace=False)
        idx = np.concatenate([np.arange(starts[d], starts[d + 1]) for d in picked])
        return self.v[idx], self.absent[idx]

    def first(self, n: int, offset: int = 0) -> tuple:
        """v and absent for days offset .. offset + n - 1 (disjoint blocks)."""
        starts = self.day_starts()
        sl = slice(starts[offset], starts[offset + n])
        return self.v[sl], self.absent[sl]


def _run_chunk(plan, rates, mean_service, mode, mean, dist, cv, days, seed, path):
    args = [str(find_simulator()),
            "--staffing", ",".join(str(int(s)) for s in plan),
            "--arrivals", ",".join(repr(float(r)) for r in rates),
            "--service-time", repr(float(mean_service)),
            "--abandonment", mode, "--patience", repr(float(mean)),
            "--patience-dist", dist, "--patience-cv", repr(float(cv)),
            "--replications", str(days), "--seed", str(seed),
            "--per-replication", "--citizen-log", str(path)]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def _chunks(plan, rates, mean_service, mode, mean, dist, cv, days, seed):
    """Citizen-log rows in chunks of CHUNK_DAYS days; column 0 is the global day index."""
    with tempfile.TemporaryDirectory() as tmp:
        for start in range(0, days, CHUNK_DAYS):
            n = min(CHUNK_DAYS, days - start)
            rows = _run_chunk(plan, rates, mean_service, mode, mean, dist, cv, n,
                              seed + start, Path(tmp) / "log.csv")
            rows[:, 0] += start
            yield rows


def raw_log(plan, rates, mean_service, mode="renege", mean=30.0, dist="exp", cv=1.0,
            days=50, seed=LOG_SEED) -> np.ndarray:
    """Every citizen's row (columns as in the simulator's --citizen-log); small runs only."""
    return np.concatenate(list(_chunks(plan, rates, mean_service, mode, mean, dist, cv,
                                       days, seed)))


def _key(kind, plan, rates, mean_service, mode, mean, dist, cv, days, seed):
    r = "-".join(f"{x:.4f}" for x in rates)
    return (f"{kind}_{mode}_{dist}{mean:g}_cv{cv:g}_S{mean_service:g}_"
            f"{'-'.join(str(int(s)) for s in plan)}_{abs(hash(r)) % 10**8}_d{days}_s{seed}")


# Column indices of the simulator's citizen log
REP, ARRIVAL, BOOKED, PATIENCE, OUTCOME, CALL, LEAVE, QAHEAD, OPEN = range(9)
EST_TICKETS, EST_COUNT, EST_LES = 9, 10, 11   # What each wait display would show


def pooled_ticket_log(plan, rates, mean_service, mean=30.0, dist="exp", cv=1.0,
                      days=20_000, seed=LOG_SEED) -> TicketLog:
    """Hidden-queue ticket log over `days` days (seeds seed, seed + 1, ...), cached."""
    path = CACHE / (_key("tickets", plan, rates, mean_service, "renege", mean, dist, cv,
                         days, seed) + ".npz")
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        parts, walkins = [], 0
        for rows in _chunks(plan, rates, mean_service, "renege", mean, dist, cv, days, seed):
            walkins += int((rows[:, BOOKED] == 0).sum())
            t = ticket_log(rows, days)
            parts.append((t.day, t.v, t.absent, t.patience))
        np.savez(path, day=np.concatenate([p[0] for p in parts]).astype(np.int32),
                 v=np.concatenate([p[1] for p in parts]),
                 absent=np.concatenate([p[2] for p in parts]),
                 patience=np.concatenate([p[3] for p in parts]).astype(np.float32),
                 walkins=walkins)
    z = np.load(path)
    return TicketLog(day=z["day"].astype(np.int64), v=z["v"], absent=z["absent"],
                     patience=z["patience"].astype(float), days=days,
                     walkins_per_day=float(z["walkins"]) / days)


def pooled_door_counts(plan, rates, mean_service, mean=30.0, dist="exp", cv=1.0,
                       days=20_000, seed=LOG_SEED) -> TicketLog:
    """
    Visible line: door-counter data as a TicketLog whose `v` is each arrival's
    expected wait and `absent` whether they balked. Cached.
    """
    path = CACHE / (_key("door", plan, rates, mean_service, "balk", mean, dist, cv,
                         days, seed) + ".npz")
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        parts, walkins = [], 0
        for rows in _chunks(plan, rates, mean_service, "balk", mean, dist, cv, days, seed):
            walkins += int((rows[:, BOOKED] == 0).sum())
            parts.append(door_counter_data(rows, mean_service))
        np.savez(path, day=np.concatenate([p[0] for p in parts]).astype(np.int32),
                 v=np.concatenate([p[1] for p in parts]),
                 absent=np.concatenate([p[2] for p in parts]),
                 patience=np.concatenate([p[3] for p in parts]).astype(np.float32),
                 walkins=walkins)
    z = np.load(path)
    return TicketLog(day=z["day"].astype(np.int64), v=z["v"], absent=z["absent"],
                     patience=z["patience"].astype(float), days=days,
                     walkins_per_day=float(z["walkins"]) / days)


def ticket_log(rows: np.ndarray, days: int) -> TicketLog:
    """The office's view of a hidden queue: walk-in tickets with V > 0."""
    walk = rows[rows[:, BOOKED] == 0]
    v = walk[:, CALL] - walk[:, ARRIVAL]
    keep = v > 1e-9
    w = walk[keep]
    return TicketLog(day=w[:, REP].astype(np.int64), v=v[keep],
                     absent=w[:, OUTCOME] == 1, patience=w[:, PATIENCE],
                     days=days, walkins_per_day=len(walk) / days)


def door_counter_data(rows: np.ndarray, mean_service: float) -> tuple:
    """
    Visible line: each walk-in's expected wait on arrival, as the balking
    rule computes it, and whether they left. An office rebuilds the queue
    length from its log and counts balkers with a timestamped door counter.
    Served at once (a window was free) means an expected wait of 0.
    """
    walk = rows[rows[:, BOOKED] == 0]
    joined_free = (walk[:, OUTCOME] == 0) & (walk[:, CALL] - walk[:, ARRIVAL] < 1e-9)
    est = np.where(joined_free, 0.0,
                   (walk[:, QAHEAD] + 1) * mean_service / walk[:, OPEN])
    keep = est > 0
    return (walk[keep, REP].astype(np.int64), est[keep], walk[keep, OUTCOME] == 2,
            walk[keep, PATIENCE])


# ============================================================================
# Estimators
# ============================================================================

def true_cdf(t, mean: float, dist: str, cv: float = 1.0):
    return 1.0 - patience_survival(t, mean, dist, cv)


def naive_km(v: np.ndarray, absent: np.ndarray):
    """
    Kaplan-Meier treating an absent ticket's call time as the moment the
    citizen left, and a served ticket as censored at its wait. Returns
    (sorted times, G at those times).
    """
    order = np.argsort(v, kind="stable")
    t, d = v[order], absent[order].astype(float)
    at_risk = len(t) - np.arange(len(t))
    surv = np.cumprod(1.0 - d / at_risk)
    return t, 1.0 - surv


def naive_km_limit(v: np.ndarray, g_true, x: float) -> float:
    """
    What naive KM converges to (post hoc, Round 10). Absent tickets are events
    at V, so its hazard at t is P(absent, V in dt) / P(V >= t) = G(t) h_V(t):
    the true CDF times the hazard of the office's own waits. Returns
    1 - exp(-sum over V_i <= x of G(V_i) / #(V >= V_i)), the plug-in with the
    Nelson-Aalen increments of V.
    """
    t = np.sort(v)
    t = t[t <= x]
    at_risk = len(v) - np.arange(len(t))
    return float(1.0 - np.exp(-np.sum(g_true(t) / at_risk)))


def cs_npmle(v: np.ndarray, absent: np.ndarray):
    """Current-status NPMLE: nondecreasing fit of P(absent | V). Returns (sorted V, G)."""
    order = np.argsort(v, kind="stable")
    t = v[order]
    g = isotonic_regression(absent[order].astype(float), increasing=True).x
    return t, g


def step_at(t: np.ndarray, g: np.ndarray, x: float) -> float:
    """Right-continuous step function through (t, g) evaluated at x (0 before t[0])."""
    k = np.searchsorted(t, x, side="right") - 1
    return float(g[k]) if k >= 0 else 0.0


@dataclass
class ParamFit:
    family: str
    params: tuple        # exp: (theta,); lognormal: (mu, sigma)
    loglik: float

    @property
    def aic(self) -> float:
        return 2 * len(self.params) - 2 * self.loglik

    def cdf(self, t):
        t = np.asarray(t, dtype=float)
        if self.family == "exp":
            return 1.0 - np.exp(-self.params[0] * t)
        mu, sigma = self.params
        return norm.cdf((np.log(np.maximum(t, 1e-300)) - mu) / sigma)

    @property
    def mean(self) -> float:
        if self.family == "exp":
            return 1.0 / self.params[0]
        mu, sigma = self.params
        return math.exp(mu + sigma * sigma / 2)


def _exp_loglik(theta, v_abs, v_pres_sum):
    return float(np.sum(np.log(-np.expm1(-theta * v_abs))) - theta * v_pres_sum)


def cs_mle(v: np.ndarray, absent: np.ndarray, family: str) -> ParamFit:
    """Parametric current-status MLE: sum log G(V_absent) + sum log(1 - G(V_present))."""
    v_abs, v_pres = v[absent], v[~absent]
    if family == "exp":
        if len(v_abs) == 0:
            return ParamFit("exp", (0.0,), 0.0)
        s = float(v_pres.sum())
        res = minimize_scalar(lambda lt: -_exp_loglik(math.exp(lt), v_abs, s),
                              bounds=(math.log(1e-5), math.log(10.0)), method="bounded",
                              options={"xatol": 1e-7})
        theta = math.exp(res.x)
        return ParamFit("exp", (theta,), _exp_loglik(theta, v_abs, s))
    if family == "lognormal":
        # P(absent | V) = Phi(a + b ln V) with b = 1 / sigma, a = -mu / sigma:
        # a probit regression of "absent" on ln V, whose log-likelihood is
        # concave in (a, b), so one smooth local search finds the optimum
        la, lp = np.log(v_abs), np.log(v_pres)

        def mills(z):
            return np.exp(norm.logpdf(z) - norm.logcdf(z))

        def nll(p):
            a, b = p
            ea, ep = a + b * la, a + b * lp
            ll = norm.logcdf(ea).sum() + norm.logcdf(-ep).sum()
            ra, rp = mills(ea), mills(-ep)
            grad = np.array([ra.sum() - rp.sum(), (ra * la).sum() - (rp * lp).sum()])
            return -ll, -grad

        mean_log = float(np.log(v).mean())
        r = minimize(nll, np.array([-mean_log / 0.6, 1 / 0.6]), jac=True, method="L-BFGS-B",
                     bounds=[(None, None), (1e-6, None)],
                     options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000})
        a, b = float(r.x[0]), float(r.x[1])
        return ParamFit("lognormal", (-a / b, 1.0 / b), -float(r.fun))
    raise ValueError(family)


def pick_family(v: np.ndarray, absent: np.ndarray) -> tuple:
    """AIC choice between exponential and lognormal. Returns (family, fits)."""
    fits = {f: cs_mle(v, absent, f) for f in ("exp", "lognormal")}
    return min(fits, key=lambda f: fits[f].aic), fits

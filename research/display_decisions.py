"""
Wait displays as decisions under partial information (Round 15, REPORT
section 5.19).

Rounds 12-14 treat a display as a function of the offered wait V. A real
office does not see V. It sees a state X (ticket ages, elapsed services, the
hour), and the twin display (Round 13b) already samples V from its exact
posterior given X. What remains is the rule that turns that posterior into
"too long" or nothing. This module derives the rule from decision theory.

Mean-field model. A display pi(X) in {0, 1} that does not see V still
admits a share a(x) = 1 - E[pi(X) | V = x] of the arrivals at offered wait x.
In the Round 13 exact law that is u(x) = a(x) (1 - G(x)) (displays.display_hour
with `admit`), so every outcome Q (failures, minutes, losses) is a functional
Q(a).

Influence. psi(x) is the change in Q per extra arrival told "too long" at
offered wait x: psi(x) = -(dQ / da(x)) / f(x), f the density of V met by
arrivals. Patience is independent of X and V, so an arrival with state X
moves Q by E[psi(V) | X] when flagged, to first order.

First-order condition. dQ = E[ E[psi_a(V) | X] dpi(X) ], so a locally optimal
display flags exactly the arrivals with E[psi_a(V) | X] < 0 (a fixed point,
since psi depends on a). With full information (X contains V) and a the
cutoff at T, Round 13's theorem gives psi > 0 below T and psi < 0 above: the
cutoff display is the special case. The twin display's quantile rule is the
special case of a psi that is a step at T.

Returns (r = 1, Round 14). Flagging changes the day's losses, and every loss
comes back. Linearized around a steady state with day-map slope s and m_R
extra minutes lost per extra returner, the citizen-time objective
M_K = minutes lost inside + K x visits has, per flagged arrival,

    psi_K(x) = d_lost(x) + (K + m_R) d_losses(x) / (1 - s).

Precision. A display that turns away a would-be-served citizen who would have
been on time buys a repeat visit and no late service. Round 14's exchange
rate eps should therefore scale with 1 / precision, the share of would-be-
served citizens it turns away who would have been served late.
"""

import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from displays import HIDDEN, display_hour  # noqa: E402
from learning import Patience  # noqa: E402

PSI_BIN = 0.5          # Minutes per influence bin (a multiple of display_hour's grid step)
PSI_MAX = 60.0         # psi is tabulated on [0, PSI_MAX]; the simulator holds it flat beyond
ETA = 0.1              # Share of a bin's arrivals flagged (or admitted) per finite difference
QUANTITIES = ("fail", "lost_min", "losses", "served_late")


def cutoff_admit(threshold: float):
    """a(x) of the oracle cutoff display: everyone below T admitted, nobody from T on."""
    return lambda x: (np.asarray(x, dtype=float) < threshold).astype(float)


def all_admit(x):
    return np.ones_like(np.asarray(x, dtype=float))


def _outcomes(hour) -> dict:
    return {"fail": hour.fail, "lost_min": hour.lost_min,
            "losses": hour.balk + hour.renege, "served_late": hour.served_late}


@dataclass
class Influence:
    x: np.ndarray              # Bin centres (minutes)
    mass: np.ndarray           # P(V in bin) met by an arrival, under the base
    psi: dict                  # Quantity -> change per flagged arrival, one per bin
    base: dict                 # Quantity -> value under the base


def influence(c: int, lam: float, mean_service: float, patience: Patience, threshold: float,
              base=None, bin_width: float = PSI_BIN, x_max: float = PSI_MAX,
              eta: float = ETA) -> Influence:
    """
    psi(x) for every quantity by finite differences on the exact law: flag
    (or admit, where the base admits nobody) a share eta of the arrivals whose
    offered wait falls in each bin, and divide the change by the flagged mass.
    Central where the base admits a share in [eta, 1 - eta].
    """
    base = cutoff_admit(threshold) if base is None else base
    h0 = display_hour(c, lam, mean_service, patience, threshold, HIDDEN, admit=base)
    q0 = _outcomes(h0)
    edges = np.arange(0.0, x_max + bin_width / 2, bin_width)
    centres = (edges[:-1] + edges[1:]) / 2
    step = h0.x[1] - h0.x[0]
    psi = {k: np.zeros(len(centres)) for k in QUANTITIES}
    mass = np.zeros(len(centres))
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        inside = (h0.x >= lo - 1e-9) & (h0.x < hi - 1e-9)
        mass[j] = float(h0.f[inside].sum() * step)
        if mass[j] <= 1e-14:
            continue
        a_mid = float(np.mean(base(centres[j:j + 1])))

        def bumped(delta, lo=lo, hi=hi):
            def adm(x):
                x = np.asarray(x, dtype=float)
                return base(x) + delta * ((x >= lo - 1e-9) & (x < hi - 1e-9))
            return _outcomes(display_hour(c, lam, mean_service, patience, threshold,
                                          HIDDEN, admit=adm))
        if eta <= a_mid <= 1.0 - eta:
            up, down = bumped(+eta), bumped(-eta)
            for k in QUANTITIES:
                psi[k][j] = (down[k] - up[k]) / (2 * eta * mass[j])
        else:
            delta = -eta if a_mid > 0.5 else +eta
            q = bumped(delta)
            for k in QUANTITIES:
                psi[k][j] = (q[k] - q0[k]) / (-delta * mass[j])
    # Bins V never reaches: carry the nearest informative value
    ok = mass > 1e-14
    if ok.any():
        for k in QUANTITIES:
            psi[k] = np.interp(centres, centres[ok], psi[k][ok])
    return Influence(x=centres, mass=mass, psi=psi, base=q0)


def implied_threshold(inf: Influence, threshold: float, key: str = "fail",
                      window: float = 5.0) -> float:
    """
    The posterior P(V > T | X) above which a two-point psi says "flag":
    psi_on / (psi_on + |psi_late|), psi averaged (by mass) within `window`
    minutes below and above T. A summary for reporting, not the rule itself.
    """
    psi, m = inf.psi[key], inf.mass
    on = (inf.x < threshold) & (inf.x >= threshold - window)
    late = (inf.x > threshold) & (inf.x <= threshold + window)
    avg = lambda sel: float((psi[sel] * np.maximum(m[sel], 1e-300)).sum()
                            / np.maximum(m[sel], 1e-300).sum())
    p_on, p_late = avg(on), avg(late)
    if p_on <= 0:
        return 0.0          # Flag even when lateness is unlikely
    if p_late >= 0:
        return 1.0          # Never flag
    return p_on / (p_on - p_late)


# ============================================================================
# Per-hour tables for the simulator's Bayes display
# ============================================================================

def _hour_influence(args):
    c, lam, s, patience, threshold, base = args
    return influence(c, lam, s, patience, threshold, base)


def psi_table_failures(plan, rates, mean_service, patience: Patience, threshold: float,
                       bases=None, pmap=map) -> list:
    """(hour, x, psi) rows for failures at the fresh hourly rates, base per hour
    (default the oracle cutoff at T)."""
    bases = bases or [None] * len(plan)
    infs = list(pmap(_hour_influence, [(c, r / 60.0, mean_service, patience, threshold, b)
                                       for c, r, b in zip(plan, rates, bases)]))
    return [(h, float(x), float(p)) for h, inf in enumerate(infs)
            for x, p in zip(inf.x, inf.psi["fail"])], infs


def return_linearization(plan, rates, mean_service, patience: Patience, threshold: float,
                         show, timing: str = "profile"):
    """
    Steady state of the per-hour day with returns under `show` (a display of
    V, e.g. the oracle cutoff), with its hourly total rates, the day map's
    slope s and m_R, the extra minutes lost per extra returner.
    """
    from abandonment import return_rates
    from display_returns import _day, day_fixed_point
    st = day_fixed_point(plan, rates, mean_service, patience, threshold, show, timing=timing)
    if st is None:
        return None
    d = max(1e-3 * st.fresh, 1e-6)
    lost = lambda R: _day(plan, return_rates(rates, R, timing), mean_service, patience,
                          threshold, show)[2]
    lo = max(st.returns - d, 0.0)
    m_R = (lost(st.returns + d) - lost(lo)) / (st.returns + d - lo)
    return {"state": st, "rates": list(return_rates(rates, st.returns, timing)),
            "slope": st.slope, "m_R": m_R}


def psi_table_returns(plan, rates, mean_service, patience: Patience, threshold: float,
                      K: float, lin: dict, pmap=map, bases=None) -> tuple:
    """(hour, x, psi_K) rows around the linearization `lin` (hourly total rates,
    day-map slope, m_R), base per hour (default the oracle cutoff at T)."""
    bases = bases or [None] * len(plan)
    infs = list(pmap(_hour_influence, [(c, r / 60.0, mean_service, patience, threshold, b)
                                       for c, r, b in zip(plan, lin["rates"], bases)]))
    gain = (K + lin["m_R"]) / (1.0 - lin["slope"])
    rows = []
    for h, inf in enumerate(infs):
        psi = inf.psi["lost_min"] + gain * inf.psi["losses"]
        rows += [(h, float(x), float(p)) for x, p in zip(inf.x, psi)]
    return rows, infs


def write_psi(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("hour,x,psi\n")
        for h, x, p in rows:
            f.write(f"{h},{x!r},{p!r}\n")


# ============================================================================
# Citizen logs with the true V and the twin's posterior
# ============================================================================

# Columns of --citizen-log with --log-offered
REP, ARRIVAL, BOOKED, PATIENCE, OUTCOME, CALL, LEAVE, QAHEAD, OPEN = range(9)
EST_TICKETS, EST_COUNT, EST_LES, OFFERED, TWIN_P_LATE, BAYES_SCORE = range(9, 15)


def simulate_log(plan, rates, mean_service, patience: Patience, days: int, seed: int,
                 threshold: float = 15.0, announce: str = "none", display_scale=None,
                 display_cutoff=None, twin_quantile=None, twin_samples: int = 64,
                 display_psi=None, display_psi_weights=None) -> np.ndarray:
    """Every citizen's row, with the true V, twin P(late) and Bayes score appended."""
    from optimizer import find_simulator
    dist = "exp" if patience.family == "exp" else "lognormal"
    args = [str(find_simulator()),
            "--staffing", ",".join(str(int(s)) for s in plan),
            "--arrivals", ",".join(repr(float(r)) for r in rates),
            "--service-time", repr(float(mean_service)),
            "--abandonment", "renege", "--patience", repr(float(patience.mean)),
            "--patience-dist", dist, "--patience-cv", repr(float(patience.cv)),
            "--replications", str(days), "--seed", str(seed), "--per-replication",
            "--twin-samples", str(int(twin_samples)), "--log-offered"]
    if threshold != 15.0:
        args += ["--wait-threshold", repr(float(threshold))]
    if announce != "none":
        args += ["--announce", announce]
    if display_scale is not None:
        args += ["--display-scale", repr(float(display_scale))]
    if display_cutoff is not None:
        args += ["--display-cutoff", ",".join(repr(float(m)) for m in display_cutoff)]
    if twin_quantile is not None:
        args += ["--twin-quantile", repr(float(twin_quantile))]
    if display_psi is not None:
        paths = [display_psi] if isinstance(display_psi, (str, Path)) else list(display_psi)
        for p in paths:
            args += ["--display-psi", str(p)]
    if display_psi_weights is not None:
        args += ["--display-psi-weights", ",".join(repr(float(w)) for w in display_psi_weights)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.csv"
        subprocess.run(args + ["--citizen-log", str(path)], check=True,
                       stdout=subprocess.DEVNULL)
        return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def waiting_walkins(rows: np.ndarray) -> np.ndarray:
    """Walk-ins who met a wait (no free window on arrival): the ones a display acts on."""
    return rows[(rows[:, BOOKED] == 0) & (rows[:, OFFERED] > 0)]


def admit_from_log(rows: np.ndarray, hour: int, bin_width: float = 1.0,
                   x_max: float = PSI_MAX, min_count: int = 20):
    """
    a_hour(x): share of waiting walk-ins at offered wait x who were not turned
    away on arrival, by bins; bins with fewer than `min_count` citizens take
    the nearest estimated value. Returns a function of x.
    """
    w = waiting_walkins(rows)
    w = w[(w[:, ARRIVAL] >= 60.0 * hour) & (w[:, ARRIVAL] < 60.0 * (hour + 1))]
    edges = np.arange(0.0, x_max + bin_width / 2, bin_width)
    centres = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.digitize(w[:, OFFERED], edges) - 1, 0, len(centres) - 1)
    told = w[:, OUTCOME] == 2
    n = np.bincount(idx, minlength=len(centres))
    k = np.bincount(idx, weights=(~told).astype(float), minlength=len(centres))
    ok = n >= min_count
    if not ok.any():
        return all_admit
    a = np.interp(centres, centres[ok], k[ok] / n[ok])
    return lambda x: np.interp(np.asarray(x, dtype=float), centres, a)


# ============================================================================
# Classifier summaries
# ============================================================================

def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties counted half)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return math.nan
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):                     # Average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list:
    """Deciles of the predicted probability: (mean predicted, observed rate, count)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p, kind="mergesort")
    out = []
    for part in np.array_split(order, bins):
        if len(part):
            out.append((float(p[part].mean()), float(y[part].mean()), int(len(part))))
    return out


def turned_away_precision(rows: np.ndarray, threshold: float = 15.0) -> dict:
    """
    Of the waiting walk-ins a display told "too long" who would have been
    served (patience >= V), the share who would have been served late
    (V > T): the display's precision as a classifier of late service.
    """
    w = waiting_walkins(rows)
    told = w[:, OUTCOME] == 2
    would_serve = w[:, PATIENCE] >= w[:, OFFERED]
    sel = told & would_serve
    late = w[:, OFFERED] > threshold
    n = int(sel.sum())
    return {"told": int(told.sum()), "told_would_serve": n,
            "precision": float(late[sel].mean()) if n else math.nan,
            "recall": float((sel & late).sum() / max((would_serve & late).sum(), 1))}

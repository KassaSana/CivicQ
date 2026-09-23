"""
Shift-feasible staffing: staff work shifts, so an hourly staffing profile must
be covered by a set of shifts.

Two-step (the textbook approach; Ingolfsson, Haque & Umnikov 2002):
    1. an hourly requirement r from some staffing rule (SIPP, OL-avg, SGS-UCB)
    2. the cheapest shift set covering r, from a covering integer program

Integrated simulation search (ISS), in the spirit of Ingolfsson et al. 2002 and
Atlason, Epelman & Henderson 2004: local search directly over shift counts,
judging each candidate schedule by simulation rather than by a fixed hourly
requirement, so surplus coverage in one hour can offset need in the next.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from staffing_methods import (  # noqa: E402
    DESIGN_REPS, DESIGN_SEED, SLOTS, _feasible,
)


def _label(start, length):
    def clock(h):
        h = 8 + h
        return str(h if h <= 12 else h - 12)
    return f"{clock(start)}-{clock(start + length)}"


def make_menu(lengths_and_starts):
    """[(start hour index, length in hours)] -> list of (name, start, length)."""
    return [(_label(s, l), s, l) for s, l in lengths_and_starts]


# The office is open hours 0..7 (8AM-4PM). Shifts must fit inside the day.
# STANDARD: one full day plus half days at every start
STANDARD = make_menu([(0, 8)] + [(s, 4) for s in range(5)])
# FLEXIBLE: also 6-hour shifts, i.e. every 4/6/8-hour window inside the day
FLEXIBLE = make_menu([(0, 8)] + [(s, 6) for s in range(3)] + [(s, 4) for s in range(5)])
SHIFTS = STANDARD


class Menu:
    def __init__(self, shifts=STANDARD):
        self.shifts = shifts
        self.cover = np.array([[1 if start <= i < start + length else 0
                                for _, start, length in shifts] for i in range(SLOTS)])
        self.hours = np.array([length for _, _, length in shifts])

    def profile(self, x) -> list:
        """Hourly open windows produced by shift counts x."""
        return [int(v) for v in self.cover @ np.asarray(x)]

    def paid_hours(self, x) -> int:
        return int(self.hours @ np.asarray(x))

    def describe(self, x) -> dict:
        return {self.shifts[j][0]: int(v) for j, v in enumerate(x) if v}

    def cover_ip(self, requirement: list) -> list:
        """Cheapest shift counts (paid hours) covering an hourly requirement."""
        res = milp(c=self.hours, integrality=np.ones(len(self.shifts)),
                   bounds=Bounds(0, np.inf),
                   constraints=LinearConstraint(self.cover, lb=np.array(requirement, float),
                                                ub=np.inf))
        if not res.success:
            raise RuntimeError(res.message)
        return [int(round(v)) for v in res.x]

    def neighbors(self, x):
        """Moves over shift counts that never raise paid hours by more than zero
        after the acceptance test: drop a shift, swap a shift for a shorter or
        equal one elsewhere, or replace two shifts with one."""
        k = len(self.shifts)
        out = []
        for j in range(k):
            if x[j] == 0:
                continue
            y = list(x); y[j] -= 1; out.append(y)                          # drop
            for jj in range(k):
                if jj != j and self.hours[jj] <= self.hours[j]:
                    y = list(x); y[j] -= 1; y[jj] += 1; out.append(y)      # swap
        for a in range(k):
            for b in range(a, k):
                if x[a] - (a == b) <= 0 or x[b] <= 0:
                    continue
                for c in range(k):
                    if self.hours[c] < self.hours[a] + self.hours[b]:
                        y = list(x); y[a] -= 1; y[b] -= 1; y[c] += 1
                        out.append(y)                                      # two -> one
        uniq = {tuple(y) for y in out if tuple(y) != tuple(x) and min(y) >= 0}
        return [list(y) for y in uniq]

    def integrated_search(self, rates, mean_service, threshold, alpha, start,
                          reps=DESIGN_REPS, seed=DESIGN_SEED, **sim_kwargs):
        """
        Local search over shift counts from a feasible `start`. A move is taken
        if the new schedule is feasible (every hour's upper 95% bound <= alpha
        on the design seeds) and cheaper, or equally cheap with a lower worst
        hour. Returns (shift counts, simulator calls).
        """
        def check(x):
            ok, ev = _feasible(self.profile(x), rates, mean_service, threshold, alpha,
                               reps, seed, "ucb", **sim_kwargs)
            return ok, max(ev.late_prob)

        ok, worst = check(start)
        if not ok:
            raise ValueError("integrated search needs a feasible starting schedule")
        x, calls = list(start), 1
        while True:
            cands = [y for y in self.neighbors(x)
                     if min(self.profile(y)) >= 1 and self.paid_hours(y) <= self.paid_hours(x)]
            with ThreadPoolExecutor() as pool:
                results = list(pool.map(check, cands))
            calls += len(cands)
            better = [(self.paid_hours(y), w, y) for y, (good, w) in zip(cands, results)
                      if good and (self.paid_hours(y), w) < (self.paid_hours(x), worst)]
            if not better:
                return x, calls
            _, worst, x = min(better)


# Convenience wrappers for the standard menu
_STD = Menu(STANDARD)
cover_ip = _STD.cover_ip
shifts_to_profile = _STD.profile
paid_hours = _STD.paid_hours
integrated_search = _STD.integrated_search

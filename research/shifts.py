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

Shift menus, roster arithmetic and the local search live in python/roster.py
(standard library only, shared with optimizer.py --shifts); this module adds
the covering IP and the research feasibility criterion.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from staffing_methods import DESIGN_REPS, DESIGN_SEED, _feasible  # noqa: E402
from roster import FLEXIBLE, STANDARD, Roster  # noqa: E402,F401  (python/ is on sys.path)

SHIFTS = STANDARD


class Menu(Roster):
    def cover_ip(self, requirement: list) -> list:
        """Cheapest shift counts (paid hours) covering an hourly requirement."""
        res = milp(c=np.array(self.hours), integrality=np.ones(len(self.shifts)),
                   bounds=Bounds(0, np.inf),
                   constraints=LinearConstraint(np.array(self.cover),
                                                lb=np.array(requirement, float), ub=np.inf))
        if not res.success:
            raise RuntimeError(res.message)
        return [int(round(v)) for v in res.x]

    def integrated_search(self, rates, mean_service, threshold, alpha, start,
                          reps=DESIGN_REPS, seed=DESIGN_SEED, **sim_kwargs):
        """
        Local search over shift counts from a feasible `start`: a move is taken
        if the new schedule is feasible (every hour's upper 95% bound <= alpha
        on the design seeds) and cheaper, or equally cheap with a lower worst
        hour. Returns (shift counts, simulator calls).
        """
        def check(x):
            ok, ev = _feasible(self.profile(x), rates, mean_service, threshold, alpha,
                               reps, seed, "ucb", **sim_kwargs)
            return ok, max(ev.late_prob)

        x, _, calls = self.local_search(start, check)
        return x, calls


# Convenience wrappers for the standard menu
_STD = Menu(STANDARD)
cover_ip = _STD.cover_ip
shifts_to_profile = _STD.profile
paid_hours = _STD.paid_hours
integrated_search = _STD.integrated_search

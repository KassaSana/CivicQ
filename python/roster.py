"""
Shift rosters: staff work shifts, so an hourly staffing profile has to be
covered by whole shifts. Standard library only, so the core tool stays
dependency-free; research/shifts.py builds on these pieces.

A roster is a list of counts, one per shift in a menu. Searching over rosters
directly with simulation, instead of first fixing an hourly requirement and
then covering it, is up to 15% cheaper (research/REPORT.md, section 5.5).
"""

from concurrent.futures import ThreadPoolExecutor

SLOTS = 8   # Hourly slots, 8AM-4PM


def _label(start: int, length: int) -> str:
    def clock(h):
        h = 8 + h
        return str(h if h <= 12 else h - 12)
    return f"{clock(start)}-{clock(start + length)}"


def make_menu(starts_and_lengths) -> list:
    """[(start hour index, length in hours)] -> [(name, start, length)]."""
    return [(_label(s, l), s, l) for s, l in starts_and_lengths]


# STANDARD: one full day plus 4-hour half days at every start
STANDARD = make_menu([(0, 8)] + [(s, 4) for s in range(5)])
# FLEXIBLE: also 6-hour shifts, i.e. every 4/6/8-hour window inside the day
FLEXIBLE = make_menu([(0, 8)] + [(s, 6) for s in range(3)] + [(s, 4) for s in range(5)])
MENUS = {"standard": STANDARD, "flexible": FLEXIBLE}


class Roster:
    """Shift-count arithmetic and local search for one shift menu."""

    def __init__(self, shifts=STANDARD):
        self.shifts = shifts
        self.hours = [length for _, _, length in shifts]
        self.cover = [[1 if start <= i < start + length else 0
                       for _, start, length in shifts] for i in range(SLOTS)]

    def profile(self, x) -> list:
        """Hourly open windows produced by shift counts x."""
        return [sum(c * n for c, n in zip(row, x)) for row in self.cover]

    def paid_hours(self, x) -> int:
        return sum(h * n for h, n in zip(self.hours, x))

    def describe(self, x) -> dict:
        return {self.shifts[j][0]: int(v) for j, v in enumerate(x) if v}

    def format(self, x) -> str:
        return ", ".join(f"{n} x {name}" for name, n in self.describe(x).items())

    def full_days(self, k: int) -> list:
        """Roster of k full-day shifts (the menu's first entry)."""
        return [k] + [0] * (len(self.shifts) - 1)

    def neighbors(self, x) -> list:
        """Drop a shift, swap one for a shorter or equal shift, or replace two
        shifts with one shorter than their total."""
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

    def local_search(self, start, check):
        """
        Improve a feasible roster. `check(x)` returns (feasible, worst_score);
        a move is taken when it is feasible and cheaper, or equally cheap with a
        lower worst score. Returns (roster, path of accepted rosters, calls).
        """
        ok, worst = check(start)
        if not ok:
            raise ValueError("roster search needs a feasible starting roster")
        x, path, calls = list(start), [list(start)], 1
        while True:
            cands = [y for y in self.neighbors(x)
                     if min(self.profile(y)) >= 1 and self.paid_hours(y) <= self.paid_hours(x)]
            with ThreadPoolExecutor() as pool:
                results = list(pool.map(check, cands))
            calls += len(cands)
            better = [(self.paid_hours(y), w, y) for y, (good, w) in zip(cands, results)
                      if good and (self.paid_hours(y), w) < (self.paid_hours(x), worst)]
            if not better:
                return x, path, calls
            _, worst, x = min(better)
            path.append(list(x))

"""
Tests for the research staffing methods and the simulator extensions they use.

    python research/test_research.py
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from staffing_methods import (  # noqa: E402
    OFFICE_RATES, analytic_plans, evaluate, lagged_rates, offered_load,
    prob_wait_exceeds, ratio_ci,
)
from optimizer import find_simulator, run_simulation, sipp_staffing  # noqa: E402

SIMULATOR = find_simulator()


class TestOfferedLoad(unittest.TestCase):
    def test_constant_rate_converges_to_lambda_times_s(self):
        for dist, cv in [("exp", 1.0), ("lognormal", 0.5), ("lognormal", 1.5), ("det", 1.0)]:
            _, m = offered_load([12.0] * 8, 8.0, dist, cv)
            self.assertAlmostEqual(m[-1], 1.6, delta=0.005, msg=dist)

    def test_exponential_step_response(self):
        # Office opens empty: m(t) = lambda * S * (1 - exp(-t / S))
        t, m = offered_load([12.0] * 8, 8.0)
        for minutes in (5, 30, 90):
            k = int(minutes / (t[1] - t[0]))
            expected = 1.6 * (1 - math.exp(-minutes / 8.0))
            self.assertAlmostEqual(m[k], expected, delta=1e-3)

    def test_lagged_rates_shift_demand(self):
        rates = [0, 60, 0, 0, 0, 0, 0, 0]
        lagged = lagged_rates(rates, 30.0)
        # Half of hour 1 and half of hour 2 now see the 60/hour burst
        self.assertAlmostEqual(lagged[1], 30.0, places=6)
        self.assertAlmostEqual(lagged[2], 30.0, places=6)
        self.assertAlmostEqual(lagged[0], 0.0, places=6)


class TestAnalyticRules(unittest.TestCase):
    def test_sipp_matches_optimizer_sipp(self):
        plans = analytic_plans(OFFICE_RATES, 8.0, 15.0, 0.10)
        self.assertEqual(plans["SIPP"], sipp_staffing())

    def test_prob_wait_exceeds_matches_mmc(self):
        # lambda = 15/h, S = 8, c = 3: C = 4/9, P(W > 15) = C * exp(-0.125 * 15)
        self.assertAlmostEqual(prob_wait_exceeds(3, 2.0, 8.0, 15.0),
                               4 / 9 * math.exp(-0.125 * 15), places=10)


class TestRatioCI(unittest.TestCase):
    def test_ratio_and_interval(self):
        late = np.array([1.0, 2.0, 3.0, 2.0])
        arr = np.array([10.0, 20.0, 30.0, 20.0])
        p, (lo, hi) = ratio_ci(late, arr)
        self.assertAlmostEqual(p, 0.1)
        self.assertAlmostEqual(lo, 0.1)   # Every day has exactly 10% late
        self.assertAlmostEqual(hi, 0.1)


class TestShifts(unittest.TestCase):
    def setUp(self):
        from shifts import FLEXIBLE, STANDARD, Menu
        self.std, self.flex = Menu(STANDARD), Menu(FLEXIBLE)

    def test_cover_ip_hand_checked(self):
        self.assertEqual(self.std.describe(self.std.cover_ip([1] * 8)), {"8-4": 1})
        self.assertEqual(self.std.describe(self.std.cover_ip([1, 1, 1, 1, 0, 0, 0, 0])),
                         {"8-12": 1})
        # Six hours of need: a 6h shift only exists on the flexible menu
        self.assertEqual(self.flex.paid_hours(self.flex.cover_ip([1] * 6 + [0, 0])), 6)
        self.assertEqual(self.std.paid_hours(self.std.cover_ip([1] * 6 + [0, 0])), 8)

    def test_profile_and_hours(self):
        x = [1, 0, 1, 0, 0, 1]      # 8-4, 9-1, 12-4
        self.assertEqual(self.std.profile(x), [1, 2, 2, 2, 3, 2, 2, 2])
        self.assertEqual(self.std.paid_hours(x), 16)

    def test_flexible_menu_contains_standard(self):
        std_names = {n for n, _, _ in self.std.shifts}
        self.assertTrue(std_names <= {n for n, _, _ in self.flex.shifts})

    @unittest.skipUnless(SIMULATOR.exists(), "simulator not built")
    def test_integrated_search_never_worse_and_feasible(self):
        from staffing_methods import _feasible
        start = self.std.cover_ip([3] * 8)
        x, _ = self.std.integrated_search(OFFICE_RATES, 8.0, 15.0, 0.10, start)
        self.assertLessEqual(self.std.paid_hours(x), self.std.paid_hours(start))
        ok, _ = _feasible(self.std.profile(x), OFFICE_RATES, 8.0, 15.0, 0.10, 400, 1, "ucb")
        self.assertTrue(ok)


class TestCiwCrossValidation(unittest.TestCase):
    def test_ciw_matches_erlang_c(self):
        try:
            from crossval_ciw import ciw_days, ratio_and_se
        except ImportError:
            self.skipTest("ciw not installed")
        a, l, _ = ciw_days([3] * 8, [15.0] * 8, 8.0, reps=40, duration=20000)
        p, se = ratio_and_se(l[:, 7], a[:, 7])
        expected = prob_wait_exceeds(3, 2.0, 8.0, 15.0)
        self.assertLess(abs(p - expected), 1.96 * se + 0.002)

    def test_equal_shifts_are_merged(self):
        # Hourly shifts with an unchanged count must not add Ciw overtime capacity
        try:
            from crossval_ciw import ciw_days, ratio_and_se
        except ImportError:
            self.skipTest("ciw not installed")
        a, l, _ = ciw_days([2] * 8, [11.125] * 8, 8.0, reps=400)
        p, _ = ratio_and_se(l[:, 7], a[:, 7])
        self.assertGreater(p, 0.18)     # 0.23 when merged; about 0.11 if not


@unittest.skipUnless(SIMULATOR.exists(), f"simulator not built at {SIMULATOR}")
class TestSimulatorExtensions(unittest.TestCase):
    def test_lognormal_and_det_preserve_mean(self):
        for dist, cv in [("lognormal", 0.5), ("lognormal", 1.5), ("det", 1.0)]:
            r = run_simulation([3] * 8, replications=300, service_dist=dist, service_cv=cv)
            self.assertAlmostEqual(r.mean_service, 8.0, delta=0.25, msg=dist)

    def test_less_service_variability_means_shorter_waits(self):
        waits = [run_simulation([2] * 8, replications=300, service_dist=d, service_cv=cv).mean_wait
                 for d, cv in [("det", 1.0), ("lognormal", 0.5), ("exp", 1.0), ("lognormal", 1.5)]]
        self.assertEqual(waits, sorted(waits))

    def test_rate_cv_keeps_mean_demand(self):
        base = run_simulation([3] * 8, replications=1000)
        mixed = run_simulation([3] * 8, replications=1000, rate_cv=0.2)
        self.assertAlmostEqual(mixed.avg_arrived / base.avg_arrived, 1.0, delta=0.03)

    def test_per_hour_late_matches_erlang_c(self):
        # Constant demand over a long day: the pooled late fraction is the
        # steady-state M/M/3 P(W > 15) with offered load 2
        ev = evaluate([3] * 8, [15.0] * 8, 8.0, reps=40, duration=20000)
        expected = prob_wait_exceeds(3, 2.0, 8.0, 15.0)
        low, high = ev.late_ci[7]           # Slot 7 covers minutes 420..20000
        self.assertLessEqual(low, expected)
        self.assertGreaterEqual(high, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)

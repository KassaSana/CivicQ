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
    prob_wait_exceeds, ratio_ci, servers_for_load,
)
from optimizer import find_simulator, run_simulation, sipp_staffing  # noqa: E402
from abandonment import (  # noqa: E402
    balk_metrics, fluid_discount, hour_metrics, renege_abandon, renege_metrics,
    required_windows, required_windows_with_returns, return_rates, score,
)

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
class TestAppointments(unittest.TestCase):
    def test_perfectly_spaced_appointments_never_wait(self):
        # One window, 8-minute fixed service, a booking every 10 minutes
        r = run_simulation([1] * 8, [0.0] * 8, replications=5, service_dist="det",
                           appointments=[10.0 * i for i in range(48)])
        self.assertEqual(r.daily_appt_arrived, [48] * 5)
        self.assertEqual(r.mean_wait, 0.0)

    def test_no_show_rate_and_arrival_counts(self):
        booked = [10.0 * i + 5 for i in range(40)]
        r = run_simulation([3] * 8, [6.0] * 8, replications=2000, appointments=booked,
                           no_show=0.15, punctuality_sd=5.0)
        self.assertAlmostEqual(np.mean(r.daily_appt_arrived) / 40, 0.85, delta=0.01)
        self.assertAlmostEqual(r.avg_arrived, 48 + 34, delta=0.6)

    def test_no_appointments_leaves_walk_in_results_unchanged(self):
        a = run_simulation([2, 3, 3, 2, 2, 3, 3, 2], replications=50)
        b = run_simulation([2, 3, 3, 2, 2, 3, 3, 2], replications=50, appointments=[],
                           no_show=0.3, punctuality_sd=9.0)
        self.assertEqual(a.daily_mean_waits, b.daily_mean_waits)
        self.assertEqual(a.daily_appt_arrived, [0] * 50)


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


class TestAbandonmentModels(unittest.TestCase):
    def test_erlang_a_abandonment_identity(self):
        # P(abandon) = theta * E[queue] / lambda for M/M/c+M
        for c, lam, patience in [(2, 15, 30), (3, 15, 30), (4, 40, 20)]:
            m = renege_metrics(c, lam / 60, 8.0, patience, 15.0)
            self.assertAlmostEqual(m.abandon, m.mean_queue / patience / (lam / 60), places=8)

    def test_long_patience_reduces_to_erlang_c(self):
        expected = prob_wait_exceeds(3, 2.0, 8.0, 15.0)
        self.assertAlmostEqual(renege_metrics(3, 0.25, 8.0, 3000, 15.0).fail, expected, delta=0.002)
        self.assertAlmostEqual(balk_metrics(3, 0.25, 8.0, 1e6, 15.0).fail, expected, places=5)

    def test_offered_wait_splits_the_failure_rate(self):
        # fail = P(V > T) + P(left early with V <= T), both parts non-negative
        m = renege_metrics(3, 0.25, 8.0, 30, 15.0)
        self.assertGreater(m.offered_late, 0.0)
        self.assertGreater(m.fail - m.offered_late, 0.0)
        self.assertAlmostEqual(renege_abandon(3, 0.25, 8.0, 30), m.abandon, places=9)

    def test_required_windows_matches_sipp_without_abandonment(self):
        for rate in (6.0, 12.0, 15.0, 40.0):
            self.assertEqual(required_windows(rate, 8.0, 15.0, 0.10),
                             servers_for_load(rate / 60 * 8, 8.0, 15.0, 0.10))

    def test_required_windows_matches_linear_search(self):
        # The bisection starts at c < (1 - alpha) R, which the throughput bound
        # makes infeasible; the answer must equal the first feasible c from 1
        for rate, mode, patience in [(40.0, "renege", 30.0), (150.0, "renege", 300.0),
                                     (150.0, "balk", 30.0), (90.0, "renege", 120.0)]:
            c = 1
            while hour_metrics(mode, c, rate, 8.0, patience, 15.0).fail > 0.10:
                c += 1
            self.assertEqual(required_windows(rate, 8.0, 15.0, 0.10, mode, patience), c)

    def test_fluid_discount(self):
        self.assertAlmostEqual(fluid_discount(15.0, 0.10, 30.0), 0.10)
        self.assertAlmostEqual(fluid_discount(15.0, 0.10, 60.0, "lognormal", 0.5), 0.0035,
                               delta=0.0002)
        self.assertEqual(fluid_discount(15.0, 0.10, 30.0, "det"), 0.0)

    def test_returns_fixed_point(self):
        # No returns reproduces the plain requirement; returns never need fewer windows
        for load in (4.0, 25.0):
            rate = load / 8 * 60
            plain = required_windows(rate, 8.0, 15.0, 0.10, "renege", 30.0)
            c0, x0 = required_windows_with_returns(rate, 8.0, 15.0, 0.10, 30.0, 0.0)
            c1, x1 = required_windows_with_returns(rate, 8.0, 15.0, 0.10, 30.0, 1.0)
            self.assertEqual(c0, plain)
            self.assertAlmostEqual(x0, rate, places=6)
            self.assertGreaterEqual(c1, c0)
            self.assertGreater(x1, rate)

    def test_return_rates_add_the_returners(self):
        self.assertAlmostEqual(sum(return_rates(OFFICE_RATES, 9.0, "profile")),
                               sum(OFFICE_RATES) + 9.0, places=9)
        self.assertEqual(return_rates(OFFICE_RATES, 9.0, "opening")[0], OFFICE_RATES[0] + 9.0)


@unittest.skipUnless(SIMULATOR.exists(), f"simulator not built at {SIMULATOR}")
class TestAbandonmentSimulator(unittest.TestCase):
    """Constant demand and staffing over a long day against the exact chains."""

    def _check(self, mode, c, lam, patience, dist="exp", cv=1.0):
        ev = score([c] * 8, [lam] * 8, 8.0, reps=40, seed=7, duration=20000,
                   abandonment=mode, patience=patience, patience_dist=dist, patience_cv=cv)
        exact = hour_metrics(mode, c, lam, 8.0, patience, 15.0, dist, cv)
        low, high = ev.fail_ci[7]             # Slot 7 covers minutes 420..20000
        self.assertLessEqual(low, exact.fail, msg=(mode, c, lam, dist))
        self.assertGreaterEqual(high, exact.fail, msg=(mode, c, lam, dist))
        self.assertAlmostEqual(ev.abandon[7], exact.abandon, delta=0.1 * exact.abandon)

    def test_renege_matches_erlang_a(self):
        self._check("renege", 3, 15.0, 30.0)
        self._check("renege", 2, 15.0, 30.0)   # Overloaded without abandonment

    def test_balk_matches_birth_death_chain(self):
        self._check("balk", 3, 15.0, 30.0)
        self._check("balk", 2, 15.0, 30.0, "lognormal", 0.5)

    def test_infinite_patience_changes_nothing(self):
        base = run_simulation([2, 3, 3, 2, 2, 3, 3, 2], replications=100)
        for mode in ("renege", "balk"):
            r = run_simulation([2, 3, 3, 2, 2, 3, 3, 2], replications=100, abandonment=mode,
                               patience=1e9, patience_dist="det")
            self.assertEqual(r.daily_mean_waits, base.daily_mean_waits, msg=mode)
            self.assertEqual(r.daily_abandoned, [[0] * 8] * 100, msg=mode)

    def test_appointment_holders_never_leave(self):
        # No walk-ins, overloaded single window, zero patience: nobody leaves
        r = run_simulation([1] * 8, [0.0] * 8, replications=3, abandonment="renege",
                           patience=0.01, appointments=[5.0 * i for i in range(90)])
        self.assertEqual(sum(map(sum, r.daily_abandoned)), 0)
        self.assertEqual(r.daily_appt_arrived, [90] * 3)

    def test_abandonment_is_aligned_across_plans(self):
        # Common random numbers: the same citizens arrive under every plan
        a = run_simulation([2] * 8, replications=20, abandonment="renege", patience=20)
        b = run_simulation([4] * 8, replications=20, abandonment="renege", patience=20)
        arrivals = lambda r: [sum(x) + sum(y) for x, y in zip(r.daily_arrivals, r.daily_abandoned)]
        self.assertEqual(arrivals(a), arrivals(b))
        self.assertGreater(sum(map(sum, a.daily_abandoned)), sum(map(sum, b.daily_abandoned)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

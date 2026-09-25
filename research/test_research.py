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
from fluid import fluid_day, fluid_staffing  # noqa: E402
from tipping import (  # noqa: E402
    classify_roots, fluid_day_renege, fluid_fixed_point, fluid_return_curve,
    simulate_return_chain, stationary_return_roots,
)
from learning import (  # noqa: E402
    Patience, explore_plan, limit_fit, mmcg_hour, run_history, sipp_g,
)
from patience_logs import (  # noqa: E402
    ARRIVAL, CALL, EST_COUNT, EST_TICKETS, LEAVE, OUTCOME, PATIENCE, cs_mle, cs_npmle, naive_km,
    pick_family, raw_log,
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

    def test_unpaid_service_accounting(self):
        # Every service minute is in a paid slot (utilization) or after closing
        plan = [3, 5, 2, 2, 4, 1, 3, 2]
        r = run_simulation(plan, [20.0, 30, 12, 12, 25, 8, 18, 12], replications=50)
        util = np.array(r.utilization)          # averaged over days
        in_day = float(np.sum(util * np.array(plan) * 60.0))
        total = r.mean_service * r.avg_served
        after = float(np.mean(r.daily_overtime_busy))
        self.assertAlmostEqual(in_day + after, total, delta=0.02 * total)
        self.assertGreater(np.mean(r.daily_spill), 0.0)     # Staffing drops at 9-10, 12-1

    def test_no_spill_without_staffing_cuts_and_overtime_is_r_times_s(self):
        # With spare windows, the office at closing holds Poisson(R) citizens,
        # each needing S more minutes on average: overtime service = R * S
        r = run_simulation([12] * 8, [30.0] * 8, replications=4000, seed=11)
        self.assertEqual(max(r.daily_spill), 0.0)
        self.assertAlmostEqual(np.mean(r.daily_overtime_busy), 4.0 * 8.0, delta=1.5)

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


class TestFluid(unittest.TestCase):
    RATES = [9.0, 12.0, 7.5, 6.0, 6.0, 9.0, 10.5, 7.5]     # about 1.1 Erlangs at S = 8

    def test_empty_start_follows_offered_load(self):
        # With windows to spare nobody queues and X(t) is the offered load m(t)
        day = fluid_day([10] * 8, self.RATES, 8.0, 15.0, dt=0.25)
        t, m = offered_load(self.RATES, 8.0)
        self.assertLess(np.max(np.abs(day["X"] - np.interp(day["t"], t, m))), 0.01)
        self.assertEqual(max(day["late"]), 0.0)

    def test_overload_grows_the_queue_linearly(self):
        # 2 windows, 30 per hour at S = 8: the queue grows at 0.5 - 2/8 = 0.25 per minute
        day = fluid_day([2] * 8, [30.0] * 8, 8.0, 15.0, dt=0.1)
        q = day["queue"]
        k = np.searchsorted(day["t"], [120.0, 180.0])
        self.assertAlmostEqual((q[k[1]] - q[k[0]]) / 60.0, 0.25, places=3)
        # 15-minute wait reached when the queue holds 15 * 2/8 = 3.75 citizens
        self.assertGreater(day["late"][1], 0.9)

    def test_lp_plan_is_feasible_and_below_workload(self):
        sol = fluid_staffing(self.RATES, 8.0, 15.0, 0.0, dt=1.0)
        day = fluid_day(sol["plan"], self.RATES, 8.0, 15.0, dt=0.25)
        self.assertLess(max(day["late"]), 0.02)       # grid error only
        workload = sum(r / 60 * 8.0 for r in self.RATES)
        self.assertLess(sol["cost"], workload)       # opens empty, drains free

    def test_work_conservation(self):
        # 60 * window-hours = S * (arrivals - backlog at closing) + idle window-minutes
        sol = fluid_staffing(self.RATES, 8.0, 15.0, 0.0, dt=1.0)
        day = fluid_day(sol["plan"], self.RATES, 8.0, 15.0, dt=0.05)
        c = np.array(sol["plan"])[np.minimum((day["t"] // 60).astype(int), 7)]
        idle = np.sum(np.maximum(c - day["X"], 0.0)) * 0.05
        arrivals = sum(self.RATES)
        self.assertAlmostEqual(60 * sol["cost"], 8.0 * (arrivals - day["X"][-1]) + idle,
                               delta=0.02 * 60 * sol["cost"])

    def test_nonpreemptive_closing_is_cheaper_and_converges(self):
        from fluid import fluid_day_nonpreemptive, fluid_staffing_nonpreemptive
        # Closing windows that finish their citizen can only add capacity
        drop = fluid_staffing(self.RATES, 8.0, 15.0, 0.0, dt=1.0)
        finish = fluid_staffing_nonpreemptive(self.RATES, 8.0, 15.0, 0.0, dt=1.0)
        self.assertLessEqual(finish["cost"], drop["cost"] + 1e-6)
        # Waits on the solver's own grid overshoot T by a few steps at most, and
        # the overshoot shrinks with the grid
        over = []
        for dt in (1.0, 0.5):
            sol = fluid_staffing_nonpreemptive(self.RATES, 8.0, 15.0, 0.0, dt=dt)
            day = fluid_day_nonpreemptive(sol["plan"], self.RATES, 8.0, 15.0, dt=dt)
            over.append(float(np.max(day["wait"])) - 15.0)
        self.assertLess(over[0], 4 * 1.0)
        self.assertLess(over[1], over[0])

    def test_paid_fluid(self):
        from fluid import fluid_staffing_nonpreemptive
        from overtime import fluid_paid
        free = fluid_staffing_nonpreemptive(self.RATES, 8.0, 15.0, 0.0, dt=1.0)
        self.assertAlmostEqual(fluid_paid(self.RATES, 8.0, 15.0, 0.0, dt=1.0)["paid_cost"],
                               free["cost"], places=4)
        # Paying for all work: cost = work + idle >= work
        workload = sum(r / 60 * 8.0 for r in self.RATES)
        one = fluid_paid(self.RATES, 8.0, 15.0, 1.0, dt=1.0)
        self.assertGreaterEqual(one["paid_cost"], workload - 1e-6)
        self.assertAlmostEqual(one["paid_cost"], one["window_hours"] + one["spill_hours"]
                               + one["overtime_hours"], places=6)
        self.assertGreaterEqual(fluid_paid(self.RATES, 8.0, 15.0, 1.5, dt=1.0)["paid_cost"],
                                one["paid_cost"] - 1e-9)

    def test_late_allowance_never_costs_more(self):
        lp = fluid_staffing(self.RATES, 8.0, 15.0, 0.0, dt=2.0)
        mip = fluid_staffing(self.RATES, 8.0, 15.0, 0.10, dt=2.0)
        self.assertLessEqual(mip["cost"], lp["cost"] + 1e-6)
        day = fluid_day(mip["plan"], self.RATES, 8.0, 15.0, dt=0.25)
        self.assertLessEqual(max(day["late"]), 0.10 + 0.05)


class TestTipping(unittest.TestCase):
    """Round 9: steady states of mandatory services."""

    RATES = [30.0, 45.0, 30.0, 20.0, 20.0, 30.0, 40.0, 25.0]

    def test_stationary_model_has_at_most_one_root(self):
        rng = np.random.default_rng(7)
        for _ in range(25):
            c = int(rng.integers(1, 12))
            rho = float(rng.uniform(0.3, 1.8))
            r = float(rng.choice([0.3, 0.8, 1.0]))
            roots = stationary_return_roots(c, rho * c * 60 / 8.0, 8.0, 30.0, r)
            self.assertLessEqual(len(roots), 1)
            if r < 1.0 or rho < 0.95:
                self.assertEqual(len(roots), 1)

    def test_fluid_conserves_citizens(self):
        plan = [3, 4, 3, 2, 2, 3, 4, 3]
        day = fluid_day_renege(plan, self.RATES, 8.0, 30.0)
        self.assertAlmostEqual(day["served"] + day["losses"], sum(self.RATES), delta=0.05)

    def test_fluid_loses_nobody_with_ample_windows(self):
        day = fluid_day_renege([20] * 8, self.RATES, 8.0, 30.0)
        self.assertLess(day["losses"], 1e-9)

    def test_fluid_return_curve_never_rises(self):
        plan = [5, 7, 5, 4, 4, 5, 6, 4]                       # 40 h for 32 h of work
        grid = np.linspace(0.0, 3 * sum(self.RATES), 31)
        for timing in ("profile", "opening"):
            h = fluid_return_curve(plan, self.RATES, 8.0, 30.0, 1.0, timing, grid)
            self.assertLess(max(np.diff(h)), 0.0)
            fp = fluid_fixed_point(plan, self.RATES, 8.0, 30.0, 1.0, timing)
            self.assertLess(fp["slope"], 1.0)

    def test_classify_roots_finds_bistability(self):
        grid = np.linspace(0.0, 10.0, 101)
        h = (2.0 - grid) * (5.0 - grid) * (8.0 - grid)      # +, -, +, -
        cls = classify_roots(grid, h)
        self.assertEqual(len(cls["roots"]), 3)
        self.assertEqual(cls["stable"], [True, False, True])
        self.assertTrue(cls["bistable"])

    def test_chain_without_returns_stays_empty(self):
        path = simulate_return_chain([3, 4, 3, 2, 2, 3, 4, 3], self.RATES, 8.0, 0.0,
                                     "profile", 5, abandonment="renege", patience=30.0)
        self.assertTrue(all(p["returns"] == 0 for p in path))


class TestPatienceLogs(unittest.TestCase):
    def test_npmle_matches_brute_force_isotonic_fit(self):
        # Brute force: the least-squares nondecreasing fit on 0/1 data is the
        # max-min formula g_i = max_{j<=i} min_{k>=i} mean(y_j..y_k)
        rng = np.random.default_rng(3)
        for _ in range(20):
            v = rng.uniform(0, 30, 12)
            y = rng.uniform(size=12) < v / 30
            t, g = cs_npmle(v, y)
            ys = y[np.argsort(v)].astype(float)
            n = len(ys)
            brute = [max(min(ys[j:k + 1].mean() for k in range(i, n)) for j in range(i + 1))
                     for i in range(n)]
            np.testing.assert_allclose(g, brute, atol=1e-12)
            self.assertTrue(np.all(np.diff(t) >= 0))

    def test_parametric_mle_recovers_iid_current_status(self):
        rng = np.random.default_rng(11)
        v = rng.exponential(20.0, 40_000)
        tau = rng.exponential(30.0, 40_000)
        fit = cs_mle(v, tau < v, "exp")
        self.assertAlmostEqual(fit.mean, 30.0, delta=1.0)
        sigma = math.sqrt(math.log(1.25))
        tau = np.exp(math.log(30.0) - sigma ** 2 / 2 + sigma * rng.standard_normal(40_000))
        fam, fits = pick_family(v, tau < v)
        self.assertEqual(fam, "lognormal")
        self.assertAlmostEqual(fits["lognormal"].mean, 30.0, delta=1.5)
        self.assertAlmostEqual(fits["lognormal"].params[1], sigma, delta=0.05)

    def test_naive_km_without_leavers_is_zero(self):
        t, g = naive_km(np.array([3.0, 1.0, 2.0]), np.array([False, False, False]))
        np.testing.assert_array_equal(g, [0.0, 0.0, 0.0])

    def test_citizen_log_invariants(self):
        for mode in ("renege", "balk"):
            rows = raw_log([2] * 8, OFFICE_RATES, 8.0, mode=mode, days=50, seed=7)
            served = rows[rows[:, OUTCOME] == 0]
            left = rows[rows[:, OUTCOME] != 0]
            self.assertTrue(len(left) > 0)
            self.assertTrue(np.all(served[:, CALL] >= served[:, ARRIVAL]))
            self.assertTrue(np.all(served[:, LEAVE] >= served[:, CALL]))
            if mode == "renege":
                self.assertTrue(np.all(left[:, OUTCOME] == 1))
                np.testing.assert_allclose(left[:, LEAVE], left[:, ARRIVAL] + left[:, PATIENCE])
                # Called only after they left, so an office sees absent <=> patience < V
                self.assertTrue(np.all(left[:, CALL] >= left[:, LEAVE] - 1e-9))
                v = rows[:, CALL] - rows[:, ARRIVAL]
                np.testing.assert_array_equal(rows[:, OUTCOME] == 1, rows[:, PATIENCE] < v)
            else:
                self.assertTrue(np.all(left[:, OUTCOME] == 2))
                np.testing.assert_array_equal(left[:, LEAVE], left[:, ARRIVAL])
                self.assertTrue(np.all(left[:, CALL] == -1))
            # Per-day aggregates are unchanged by logging
            args = [str(SIMULATOR), "--staffing", "2,2,2,2,2,2,2,2", "--abandonment", mode,
                    "--replications", "20", "--seed", "7", "--per-replication"]
            import subprocess
            import tempfile
            plain = subprocess.run(args, capture_output=True, text=True, check=True).stdout
            with tempfile.TemporaryDirectory() as tmp:
                logged = subprocess.run(args + ["--citizen-log", str(Path(tmp) / "c.csv")],
                                        capture_output=True, text=True, check=True).stdout
            self.assertEqual(plain, logged)



class TestLearning(unittest.TestCase):
    def test_mmcg_with_exponential_patience_is_erlang_a(self):
        for c, lam, s, m in [(2, 12, 8, 30), (8, 40, 16, 30), (2, 20, 8, 10), (5, 40, 8, 200)]:
            e = renege_metrics(c, lam / 60, s, m, 15.0)
            g = mmcg_hour(c, lam / 60, s, Patience("exp", m), 15.0)
            self.assertAlmostEqual(g.fail, e.fail, places=5)
            self.assertAlmostEqual(g.abandon, e.abandon, places=5)
            self.assertAlmostEqual(g.served_late, e.served_late, places=5)

    def test_sipp_g_matches_erlang_a_sipp(self):
        from abandonment import sipp_abandonment
        self.assertEqual(sipp_g(OFFICE_RATES, 8.0, 15.0, 0.1, Patience("exp", 30.0)),
                         sipp_abandonment(OFFICE_RATES, 8.0, 15.0, 0.1, "renege", 30.0, "fail"))

    def test_limit_fit_recovers_the_right_family(self):
        plan = [3, 3, 3, 2, 2, 3, 3, 3]
        for truth in (Patience("exp", 30.0), Patience("lognormal", 30.0, 0.5)):
            fit = limit_fit(plan, OFFICE_RATES, 8.0, truth, truth.family)
            self.assertAlmostEqual(fit.mean, 30.0, delta=0.3)
            self.assertAlmostEqual(fit.cv, truth.cv, delta=0.01)

    def test_explore_plan_removes_at_least_one_window(self):
        self.assertEqual(explore_plan([2, 10, 1], 0.9), [1, 9, 1])
        self.assertEqual(explore_plan([30, 40], 0.9), [27, 36])

    def test_history_records_every_period(self):
        out = run_history((OFFICE_RATES, 8.0, 0.1, Patience("exp", 30.0), "A", True,
                           [3, 3, 3, 2, 2, 3, 3, 3], 998, 2, 10, 15.0))
        self.assertEqual([r["period"] for r in out], [1, 2])
        self.assertTrue(all(len(r["new_plan"]) == 8 for r in out))
        # Exploration days run fewer windows, so paid hours fall below the plan
        self.assertLess(out[0]["paid_hours_per_day"], out[0]["hours_run"])


class TestWaitDisplays(unittest.TestCase):
    PLAN = [2, 2, 2, 2, 2, 2, 2, 2]

    def _run(self, *extra):
        import subprocess
        args = [str(SIMULATOR), "--staffing", ",".join(map(str, self.PLAN)), "--replications",
                "60", "--seed", "3", "--per-replication", "--patience-dist", "lognormal",
                "--patience-cv", "0.5", *extra]
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout

    def test_visible_line_is_count_display_plus_commitment(self):
        self.assertEqual(self._run("--abandonment", "balk"),
                         self._run("--abandonment", "renege", "--announce", "count", "--commit"))

    def test_no_display_is_the_hidden_queue(self):
        self.assertEqual(self._run("--abandonment", "renege"),
                         self._run("--abandonment", "renege", "--announce", "none"))

    def test_displays_and_balkers_in_the_log(self):
        rows = raw_log(self.PLAN, OFFICE_RATES, 8.0, mode="renege", days=40, seed=11)
        v = rows[:, CALL] - rows[:, ARRIVAL]
        waited = rows[:, OUTCOME] == 0
        # All displays are 0 exactly when a window was free on arrival
        free = rows[:, EST_COUNT] == 0
        np.testing.assert_array_equal(free, rows[:, EST_TICKETS] == 0)
        self.assertTrue(np.all(v[free & waited] < 1e-9))
        # Uncalled tickets include holders who left, so tickets >= count
        self.assertTrue(np.all(rows[:, EST_TICKETS] >= rows[:, EST_COUNT] - 1e-9))
        # With a display, balkers leave on arrival and are counted per day
        r = run_simulation(self.PLAN, OFFICE_RATES, replications=40, seed=11,
                           abandonment="renege", patience=30.0, announce="tickets")
        self.assertGreater(sum(r.daily_balked), 0)


class TestDisplayTheory(unittest.TestCase):
    """Round 13: exact stationary law of a display of the offered wait."""

    def setUp(self):
        from learning import Patience
        self.pats = [Patience("exp", 30.0), Patience("lognormal", 30.0, 0.5),
                     Patience("lognormal", 30.0, 1.5)]
        self.cases = [(2, 15 / 60, 8.0), (8, 34 / 60, 16.0), (8, 25 / 60, 16.0)]

    def test_no_display_is_mmcg(self):
        from displays import display_hour
        from learning import mmcg_hour
        for p in self.pats:
            for c, lam, s in self.cases:
                self.assertAlmostEqual(display_hour(c, lam, s, p, 15.0).fail,
                                       mmcg_hour(c, lam, s, p, 15.0).fail, places=3)

    def test_always_too_long_is_erlang_b(self):
        from displays import display_hour, erlang_b
        for c, lam, s in self.cases:
            d = display_hour(c, lam, s, self.pats[0], 15.0,
                             lambda x: np.where(x > 0, np.inf, 0.0))
            self.assertAlmostEqual(d.fail, erlang_b(c, lam * s), delta=2e-3)

    def test_probabilities_add_up(self):
        from displays import cutoff, display_hour, scaled
        for show in (scaled(0.0), scaled(2.0), cutoff(15.0)):
            d = display_hour(8, 34 / 60, 16.0, self.pats[1], 15.0, show)
            self.assertAlmostEqual(d.on_time + d.served_late + d.balk + d.renege, 1.0, places=9)
            self.assertGreaterEqual(min(d.balk, d.renege, d.served_late), -1e-9)

    def test_overstating_helps_above_the_threshold(self):
        from displays import display_hour
        rng = np.random.default_rng(5)
        for p in self.pats:
            for c, lam, s in self.cases:
                base = display_hour(c, lam, s, p, 15.0).fail
                for _ in range(3):
                    a, b = sorted(rng.uniform(15.5, 60.0, 2))
                    k = rng.uniform(1.2, 4.0)
                    above = lambda x, a=a, b=b, k=k: np.where((x > a) & (x < b), k * x, x)
                    self.assertLess(display_hour(c, lam, s, p, 15.0, above).fail, base)

    def test_given_the_cutoff_overstating_below_hurts(self):
        from displays import cutoff, display_hour
        rng = np.random.default_rng(6)
        for p in self.pats:
            for c, lam, s in self.cases:
                base = display_hour(c, lam, s, p, 15.0, cutoff(15.0)).fail
                for _ in range(3):
                    a, b = sorted(rng.uniform(0.0, 15.0, 2))
                    k = rng.uniform(1.2, 4.0)
                    show = lambda x, a=a, b=b, k=k: np.where(
                        x >= 15.0, np.inf, np.where((x > a) & (x < b), k * x, x))
                    self.assertGreater(display_hour(c, lam, s, p, 15.0, show).fail, base)

    def test_throughput_balances_busy_windows_at_any_load(self):
        # Flow balance mu E[busy] = served rate; Round 13's trapezoid rule broke
        # it by up to 5% at 20x capacity below a cutoff (section 5.17, correction)
        from displays import cutoff, display_hour, mean_busy, scaled, served_rate
        for p in self.pats:
            for c, s in [(1, 8.0), (4, 8.0), (16, 16.0)]:
                for show in (scaled(0.0), scaled(2.0), cutoff(5.0), cutoff(15.0)):
                    for load in (0.5, 3.0, 20.0):
                        lam = load * c / s
                        d = display_hour(c, lam, s, p, 15.0, show)
                        self.assertAlmostEqual(served_rate(d, lam) * s / c,
                                               mean_busy(c, lam, s, d) / c, delta=1e-4)

    def test_grid_converges(self):
        from displays import cutoff, display_hour
        for load in (1.0, 20.0):
            lam = load * 16 / 16.0
            coarse = display_hour(16, lam, 16.0, self.pats[1], 15.0, cutoff(15.0))
            fine = display_hour(16, lam, 16.0, self.pats[1], 15.0, cutoff(15.0), h=0.001)
            self.assertAlmostEqual(coarse.fail, fine.fail, delta=2e-3)
            self.assertAlmostEqual(coarse.balk, fine.balk, delta=2e-3)

    def test_the_cutoff_at_the_threshold_is_best(self):
        from displays import cutoff, display_hour, scaled
        for p in self.pats:
            for c, lam, s in self.cases:
                best = display_hour(c, lam, s, p, 15.0, cutoff(15.0)).fail
                for show in [scaled(k) for k in (0.0, 1.5, 2.0, 4.0)] +                         [cutoff(m) for m in (5.0, 12.0, 14.0, 16.0, 25.0)]:
                    self.assertLessEqual(best, display_hour(c, lam, s, p, 15.0, show).fail + 1e-9)


class TestDisplayReturns(unittest.TestCase):
    """Round 14: wait displays when everyone sent home comes back."""

    def setUp(self):
        from learning import Patience
        self.pats = [Patience("exp", 30.0), Patience("lognormal", 30.0, 0.5),
                     Patience("lognormal", 30.0, 1.5)]
        self.rng = np.random.default_rng(14)

    def _random_display(self):
        from displays import cutoff, scaled
        kind = self.rng.integers(3)
        if kind == 0:
            return scaled(self.rng.uniform(0.0, 4.0))
        if kind == 1:
            return cutoff(self.rng.uniform(2.0, 40.0), self.rng.uniform(0.0, 2.0))
        a, b = sorted(self.rng.uniform(0.0, 40.0, 2))
        k = self.rng.uniform(1.0, 5.0)
        return lambda x, a=a, b=b, k=k: np.where((x > a) & (x < b), k * x, 0.5 * x)

    def test_throughput_rises_with_load_for_any_display(self):
        # T1: p0 falls and E[busy] rises in the arrival rate
        from display_returns import throughput
        for _ in range(12):
            c = int(self.rng.choice([1, 2, 4, 8, 16]))
            s = float(self.rng.choice([4.0, 8.0, 16.0]))
            p = self.pats[self.rng.integers(3)]
            show = self._random_display()
            loads = np.geomspace(0.2, 20.0, 15) * c / s
            th = [throughput(c, L, s, p, 15.0, show) for L in loads]
            self.assertTrue(np.all(np.diff(th) > -1e-9), (c, s, th))

    def test_overstating_more_serves_fewer(self):
        # T2: phi1 >= phi2 pointwise => theta1 <= theta2 at every load
        from display_returns import throughput
        from displays import cutoff, scaled
        for _ in range(12):
            c = int(self.rng.choice([1, 4, 16]))
            s = float(self.rng.choice([8.0, 16.0]))
            p = self.pats[self.rng.integers(3)]
            k1, k2 = sorted(self.rng.uniform(0.0, 3.0, 2))
            m1, m2 = sorted(self.rng.uniform(3.0, 40.0, 2))
            pairs = [(scaled(k2), scaled(k1)), (cutoff(m1), cutoff(m2)),
                     (cutoff(m1, k2), scaled(k1))]
            for L in np.array([0.5, 1.0, 3.0]) * c / s:
                for more, less in pairs:
                    self.assertLessEqual(throughput(c, L, s, p, 15.0, more),
                                         throughput(c, L, s, p, 15.0, less) + 1e-6)

    def test_mandatory_steady_state_serves_the_fresh_demand(self):
        from display_returns import fixed_point
        from displays import cutoff
        for p in self.pats:
            for show in (None, cutoff(15.0)):
                kw = {} if show is None else {"show": show}
                st = fixed_point(8, 0.95 * 8 / 16.0, 16.0, p, 15.0, **kw)
                self.assertAlmostEqual(st.served, st.fresh, delta=1e-6 * st.fresh)
                self.assertGreaterEqual(st.visits, 1.0)

    def test_no_steady_state_at_capacity(self):
        from display_returns import fixed_point
        self.assertIsNone(fixed_point(4, 1.02 * 4 / 8.0, 8.0, self.pats[0], 15.0))

    def test_oracle_cutoff_serves_nobody_late_and_costs_visits(self):
        from display_returns import exchange_rate
        from displays import cutoff
        for p in self.pats:
            for c, s in [(4, 8.0), (16, 16.0)]:
                ex = exchange_rate(c, 0.9 * c / s, s, p, 15.0, cutoff(15.0))
                self.assertLess(ex.display.late_share, 1e-9)
                self.assertGreater(ex.display.visits, ex.base.visits)
                self.assertGreater(ex.eps, 0.0)
                self.assertGreater(ex.kappa, 0.0)

    def test_without_returns_the_day_is_round_13(self):
        from display_returns import day_fixed_point
        from displays import cutoff, plan_prediction
        plan, rates = [5, 6, 4, 3, 3, 5, 6, 4], [25, 40, 30, 18, 15, 28, 38, 26]
        for show in (cutoff(15.0), cutoff(10.0)):
            st = day_fixed_point(plan, rates, 8.0, self.pats[1], 15.0, show, return_prob=0.0)
            self.assertAlmostEqual(st.returns, 0.0, places=6)
            self.assertAlmostEqual(st.kpi, plan_prediction(plan, rates, 8.0, self.pats[1],
                                                           15.0, show)["fail"], places=9)

    def test_day_steady_state_is_a_fixed_point(self):
        from display_returns import day_fixed_point
        from displays import cutoff
        plan, rates = [5, 6, 4, 3, 3, 5, 6, 4], [25, 40, 30, 18, 15, 28, 38, 26]
        for p in self.pats:
            h = day_fixed_point(plan, rates, 8.0, p, 15.0)
            d = day_fixed_point(plan, rates, 8.0, p, 15.0, cutoff(15.0))
            for st in (h, d):
                self.assertAlmostEqual(st.losses, st.returns, delta=1e-6 * st.fresh)
                self.assertLess(st.slope, 1.0)
            self.assertGreaterEqual(d.returns, h.returns)       # T2, hour by hour


@unittest.skipUnless(SIMULATOR.exists(), f"simulator not built at {SIMULATOR}")
class TestOracleDisplay(unittest.TestCase):
    PLAN = [7, 13, 9, 6, 5, 8, 12, 9]
    RATES = [40, 75, 55, 35, 30, 50, 70, 50]

    def _rows(self, *extra):
        import csv
        import io
        import subprocess
        args = [str(SIMULATOR), "--staffing", ",".join(map(str, self.PLAN)),
                "--arrivals", ",".join(map(str, self.RATES)), "--service-time", "16",
                "--replications", "60", "--seed", "3", "--per-replication",
                "--abandonment", "renege", "--patience-dist", "lognormal",
                "--patience-cv", "0.5", *extra]
        out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
        return list(csv.DictReader(io.StringIO(out)))

    def test_the_exact_wait_changes_no_outcome(self):
        # Proposition (Round 12): a display never above V is outcome-neutral, and
        # the exact V sends every leaver home at once
        hidden, oracle = self._rows(), self._rows("--announce", "oracle")
        keys = [k for k in hidden[0] if k.startswith(("late_", "aband_", "arr_"))
                and k != "aband_wait_sum"] + ["served", "mean_wait"]
        for h, o in zip(hidden, oracle):
            self.assertEqual([h[k] for k in keys], [o[k] for k in keys])
            self.assertEqual(float(o["aband_wait_sum"]), 0.0)
            self.assertEqual(int(o["balked"]), sum(int(o[f"aband_{i}"]) for i in range(8)))

    def test_scale_one_without_cutoff_is_the_plain_display(self):
        for name in ("count", "tickets"):
            self.assertEqual(self._rows("--announce", name),
                             self._rows("--announce", name, "--display-scale", "1"))

    def test_twin_display_that_never_says_too_long_is_the_hidden_queue(self):
        # Its sampling uses its own stream: a display nobody acts on changes nothing
        hidden = self._rows()
        twin = self._rows("--announce", "twin", "--display-scale", "0",
                          "--twin-samples", "8")
        self.assertEqual(hidden, twin)

    def test_cutoff_zero_turns_away_everyone_who_would_wait(self):
        a = self._rows("--announce", "count", "--display-cutoff", "0")
        b = self._rows("--announce", "oracle", "--display-scale", "1e9")
        self.assertEqual([r["served"] for r in a], [r["served"] for r in b])


class TestDisplayDecisions(unittest.TestCase):
    """Round 15: displays as decisions under partial information."""

    def test_full_admission_is_the_plain_law(self):
        from displays import display_hour
        p = Patience("lognormal", 30.0, 0.5)
        a = display_hour(4, 0.5, 8.0, p, 15.0)
        b = display_hour(4, 0.5, 8.0, p, 15.0, admit=lambda x: np.ones_like(x))
        for k in ("fail", "balk", "renege", "served_late", "wait_min", "wasted_min"):
            self.assertAlmostEqual(getattr(a, k), getattr(b, k), places=12)

    def test_admitting_nobody_above_t_is_the_cutoff_display(self):
        from display_decisions import cutoff_admit
        from displays import HIDDEN, cutoff, display_hour
        p = Patience("exp", 30.0)
        a = display_hour(4, 0.55, 8.0, p, 15.0, cutoff(15.0))
        b = display_hour(4, 0.55, 8.0, p, 15.0, HIDDEN, admit=cutoff_admit(15.0))
        self.assertAlmostEqual(a.fail, b.fail, places=3)
        # cutoff() shows V below T, so leavers go at once instead of reneging
        self.assertAlmostEqual(a.balk + a.renege, b.balk + b.renege, places=3)

    def test_influence_has_the_theorem_signs_and_is_first_order(self):
        # Round 13's theorem: under the cutoff, flagging below T costs failures,
        # flagging above T saves them
        from display_decisions import influence
        from displays import HIDDEN, display_hour
        p = Patience("lognormal", 30.0, 0.5)
        inf = influence(4, 0.5, 8.0, p, 15.0)
        live = inf.mass > 1e-6
        self.assertTrue((inf.psi["fail"][live & (inf.x < 15)] > 0).all())
        self.assertTrue((inf.psi["fail"][live & (inf.x > 15)] < 0).all())
        pert = lambda x: 0.03 * (np.asarray(x) < 40)
        adm = lambda x: np.where(np.asarray(x) < 15, 1 - pert(x), pert(x))
        exact = display_hour(4, 0.5, 8.0, p, 15.0, HIDDEN, admit=adm).fail - inf.base["fail"]
        flagged = np.where(inf.x < 15, pert(inf.x), -pert(inf.x)) * inf.mass
        self.assertAlmostEqual(float((inf.psi["fail"] * flagged).sum()) / exact, 1.0, delta=0.05)

    def test_auc_and_reliability(self):
        from display_decisions import auc, reliability
        rng = np.random.default_rng(0)
        s, y = rng.normal(size=300), rng.random(300) < 0.4
        s[::7] = 0.5                          # Ties
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]])
        self.assertAlmostEqual(auc(s, y), brute, places=12)
        rel = reliability(np.linspace(0, 1, 100), np.ones(100), bins=4)
        self.assertEqual([n for _, _, n in rel], [25] * 4)


@unittest.skipUnless(SIMULATOR.exists(), f"simulator not built at {SIMULATOR}")
class TestBayesDisplay(unittest.TestCase):
    PLAN, RATES, _rows = TestOracleDisplay.PLAN, TestOracleDisplay.RATES, TestOracleDisplay._rows

    def test_step_psi_is_the_twin_quantile_rule(self):
        # The twin's quantile rule is the Bayes display with a step psi at T:
        # flag iff more than half of 64 draws reach 15 <=> the 32nd smallest does
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.csv"
            path.write_text("hour,x,psi\n" + "".join(
                f"{h},{x!r},{v}\n" for h in range(8)
                for x, v in ((0.0, 1), (15 - 1e-9, 1), (15.0, -1), (1000.0, -1))))
            bayes = self._rows("--announce", "bayes", "--display-psi", str(path))
        twin = self._rows("--announce", "twin", "--display-scale", "0",
                          "--display-cutoff", "15", "--twin-quantile", "0.484375")
        self.assertEqual(bayes, twin)
        self.assertGreater(sum(int(r["balked"]) for r in bayes), 0)

    def test_logging_the_offered_wait_changes_nothing_and_is_exact(self):
        # --log-offered draws only on the twin stream; the logged V is the wait a
        # ticket actually had (served or called absent)
        import tempfile
        from display_decisions import OFFERED, waiting_walkins
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "c.csv"
            logged = self._rows("--log-offered", "--citizen-log", str(log))
            rows = np.loadtxt(log, delimiter=",", skiprows=1, ndmin=2)
        self.assertEqual(logged, self._rows())
        w = waiting_walkins(rows)
        called = w[w[:, CALL] >= 0]
        np.testing.assert_allclose(called[:, OFFERED], called[:, CALL] - called[:, ARRIVAL],
                                   atol=1e-6)
        self.assertGreater(len(called), 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)

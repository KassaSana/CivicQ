"""
Validation tests for the queue simulator and optimizer helpers.

The key test runs the simulator with constant demand and staffing over a long
horizon, where the time-varying model reduces to a steady-state M/M/c queue,
and checks the simulated mean wait against the exact Erlang-C value.

Run from the repository root or the python/ directory:
    python python/test_validation.py
"""

import math
import unittest

import optimizer
from optimizer import (
    SimulationResult,
    compute_confidence_interval,
    erlang_c,
    find_simulator,
    mmc_mean_wait,
    pareto_frontier,
    run_simulation,
    sipp_staffing,
    stress_test,
    t_critical_95,
)

SIMULATOR = find_simulator()


class TestErlangC(unittest.TestCase):
    def test_single_server_equals_utilization(self):
        # M/M/1: P(wait) = rho
        self.assertAlmostEqual(erlang_c(1, 0.5), 0.5)

    def test_known_values(self):
        # M/M/2 with rho = 0.5: 2*rho^2 / (1 + rho) = 1/3
        self.assertAlmostEqual(erlang_c(2, 1.0), 1 / 3)
        # M/M/3 with offered load 2: 4/9
        self.assertAlmostEqual(erlang_c(3, 2.0), 4 / 9)

    def test_unstable_system_always_waits(self):
        self.assertEqual(erlang_c(2, 2.0), 1.0)
        self.assertTrue(math.isinf(mmc_mean_wait(2, 15.0)))

    def test_sipp_staffing_within_bounds(self):
        staffing = sipp_staffing()
        self.assertEqual(len(staffing), 8)
        for s in staffing:
            self.assertGreaterEqual(s, optimizer.MIN_WINDOWS_PER_SLOT)
            self.assertLessEqual(s, optimizer.MAX_WINDOWS_PER_SLOT)


class TestRoster(unittest.TestCase):
    def setUp(self):
        from roster import FLEXIBLE, STANDARD, Roster
        self.std, self.flex = Roster(STANDARD), Roster(FLEXIBLE)

    def test_profile_and_hours(self):
        x = [1, 0, 1, 0, 0, 1]      # 8-4, 9-1, 12-4
        self.assertEqual(self.std.profile(x), [1, 2, 2, 2, 3, 2, 2, 2])
        self.assertEqual(self.std.paid_hours(x), 16)
        self.assertEqual(self.std.format(x), "1 x 8-4, 1 x 9-1, 1 x 12-4")

    def test_neighbors_never_add_hours_or_go_negative(self):
        x = [2, 1, 0, 1, 0, 0]
        for y in self.std.neighbors(x):
            self.assertGreaterEqual(min(y), 0)
            self.assertLess(self.std.paid_hours(y), self.std.paid_hours(x) + 1)

    def test_flexible_menu_contains_standard(self):
        self.assertTrue({n for n, _, _ in self.std.shifts} <= {n for n, _, _ in self.flex.shifts})

    @unittest.skipUnless(find_simulator().exists(), "simulator not built")
    def test_roster_search_is_validated_and_no_worse_than_full_days(self):
        report = optimizer.roster_search("flexible", "p90")
        ok, _ = optimizer._meets_target(report["validation"], "p90")
        self.assertTrue(ok)
        self.assertLessEqual(report["paid_hours"], 24)   # 3 full days also meet the target
        self.assertEqual(sum(report["profile"]),
                         sum(report["roster"].profile(report["shifts"])))


class TestOptimizerSampleSizes(unittest.TestCase):
    def test_confirmation_is_large_enough(self):
        # 30 days gave P90 CIs of about +/-4 min and misreported plans near the
        # 15-min target; keep confirmation in the hundreds (research/REPORT.md)
        self.assertGreaterEqual(optimizer.CONFIRM_REPLICATIONS, 200)
        self.assertGreaterEqual(optimizer.CONFIRM_TOP_K, 30)


class TestConfidenceInterval(unittest.TestCase):
    def test_t_values(self):
        self.assertAlmostEqual(t_critical_95(2), 4.303)
        self.assertAlmostEqual(t_critical_95(9), 2.262)
        self.assertAlmostEqual(t_critical_95(29), 2.045)
        self.assertAlmostEqual(t_critical_95(1000), 1.96)

    def test_interval_uses_small_sample_t(self):
        low, high = compute_confidence_interval([1.0, 2.0, 3.0])
        # mean 2, s = 1, n = 3 -> margin = 4.303 / sqrt(3)
        self.assertAlmostEqual(high - 2.0, 4.303 / math.sqrt(3))
        self.assertAlmostEqual(2.0 - low, 4.303 / math.sqrt(3))


def _fake_result(staffing, p90):
    return SimulationResult(staffing=tuple(staffing), mean_wait=0.0, p90_wait=p90,
                            avg_served=0.0, avg_arrived=0.0, utilization=[0.0] * 8,
                            total_staff_hours=sum(staffing))


class TestParetoFrontier(unittest.TestCase):
    def test_keeps_lowest_p90_per_staff_hour_total(self):
        results = [
            _fake_result([2] * 8, 20.0),
            _fake_result([3] + [2] * 7, 12.0),
            _fake_result([2] * 7 + [3], 9.0),
        ]
        frontier = pareto_frontier(results, confirm_replications=0)
        self.assertEqual([r.total_staff_hours for r in frontier], [16, 17])
        self.assertEqual(frontier[1].p90_wait, 9.0)

    def test_known_plans_compete(self):
        results = [_fake_result([3] + [2] * 7, 12.0)]
        known = [_fake_result([2] * 7 + [3], 8.0)]
        frontier = pareto_frontier(results, confirm_replications=0, known=known)
        self.assertEqual(frontier[0].staffing, tuple([2] * 7 + [3]))


@unittest.skipUnless(SIMULATOR.exists(), f"simulator not built at {SIMULATOR}")
class TestStressTest(unittest.TestCase):
    def test_base_factor_matches_plain_run_and_demand_raises_waits(self):
        staffing = [2, 3, 3, 2, 2, 3, 3, 2]
        stress = stress_test(staffing, factors=(1.0, 1.2), simulator_path=SIMULATOR)
        base = run_simulation(staffing, replications=optimizer.CONFIRM_REPLICATIONS,
                              simulator_path=SIMULATOR)
        self.assertAlmostEqual(stress[1.0].p90_wait, base.p90_wait)
        self.assertGreater(stress[1.2].p90_wait, stress[1.0].p90_wait)


@unittest.skipUnless(SIMULATOR.exists(), f"simulator not built at {SIMULATOR}")
class TestSimulatorAgainstErlangC(unittest.TestCase):
    """Constant-rate runs must match steady-state M/M/c theory."""

    DURATION = 20000.0   # minutes; long enough that the empty start is negligible
    REPLICATIONS = 30

    def check_mmc(self, arrivals_per_hour: float, windows: int):
        result = run_simulation(
            [windows] * 8,
            arrival_rates=[arrivals_per_hour] * 8,
            replications=self.REPLICATIONS,
            simulator_path=SIMULATOR,
            duration=self.DURATION,
        )
        expected = mmc_mean_wait(windows, arrivals_per_hour)
        low, high = result.mean_wait_ci
        self.assertLessEqual(
            low, expected,
            f"simulated mean wait {result.mean_wait:.3f} (CI {low:.3f}-{high:.3f}) "
            f"vs Erlang-C {expected:.3f}")
        self.assertGreaterEqual(
            high, expected,
            f"simulated mean wait {result.mean_wait:.3f} (CI {low:.3f}-{high:.3f}) "
            f"vs Erlang-C {expected:.3f}")

    def test_moderate_load(self):
        # Offered load 2, 3 windows (rho = 0.67): Wq = 3.56 min
        self.check_mmc(15.0, 3)

    def test_heavy_load(self):
        # Offered load 1.6, 2 windows (rho = 0.8): Wq = 14.2 min
        self.check_mmc(12.0, 2)


@unittest.skipUnless(SIMULATOR.exists(), f"simulator not built at {SIMULATOR}")
class TestSimulatorInvariants(unittest.TestCase):
    def test_everyone_inside_at_closing_is_served(self):
        result = run_simulation([2, 3, 2, 2, 2, 3, 3, 2], replications=20,
                                simulator_path=SIMULATOR)
        self.assertEqual(result.avg_served, result.avg_arrived)
        self.assertGreaterEqual(result.avg_overtime, 0.0)

    def test_opening_windows_serves_the_queue(self):
        # A morning surge with 1 window builds a backlog and nobody arrives after
        # 9AM. The 3 windows opening at 9AM must pull from that backlog; before
        # the fix only window 0 drained it (slot-1 utilization ~0.25).
        arrivals = [30, 0, 0, 0, 0, 0, 0, 0]
        one_then_four = run_simulation([1, 4, 4, 4, 4, 4, 4, 4], arrival_rates=arrivals,
                                       replications=30, simulator_path=SIMULATOR)
        self.assertGreater(one_then_four.utilization[1], 0.5)

    def test_common_random_numbers(self):
        # Same seeds -> same citizens regardless of staffing
        a = run_simulation([2] * 8, replications=10, simulator_path=SIMULATOR)
        b = run_simulation([4] * 8, replications=10, simulator_path=SIMULATOR)
        self.assertEqual(a.avg_arrived, b.avg_arrived)
        self.assertLess(b.mean_wait, a.mean_wait)


if __name__ == "__main__":
    unittest.main(verbosity=2)

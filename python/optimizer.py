"""
Public-Sector Queue Resource Allocation Optimizer

Grid-search optimization and scenario analysis for government service center
staffing decisions. Integrates with C++ discrete-event simulation engine.

Author: Government Operations Research Team
"""

import subprocess
import csv
import io
import itertools
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

# Optional plotting (graceful fallback if matplotlib is unavailable)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# ============================================================================
# Configuration
# ============================================================================

# Locations the C++ simulator may be built to (MinGW, MSVC multi-config, Unix)
_BUILD_DIR = Path(__file__).parent.parent / "cpp" / "build"
SIMULATOR_CANDIDATES = [
    _BUILD_DIR / "queue_sim.exe",
    _BUILD_DIR / "Release" / "queue_sim.exe",
    _BUILD_DIR / "queue_sim",
]

# Default arrival rates: morning peak, midday lull, afternoon peak (per hour)
DEFAULT_ARRIVAL_RATES = [12.0, 15.0, 10.0, 8.0, 8.0, 12.0, 14.0, 10.0]

# Constraints from project plan. The offered load per slot is 1.07-2.0
# (lambda * 8 min / 60), so a single window is unstable in every slot.
MIN_WINDOWS_PER_SLOT = 2
MAX_WINDOWS_PER_SLOT = 4
MAX_TOTAL_STAFF_HOURS = 28
NUM_REPLICATIONS = 10       # Screening replications per configuration
# Replications for finalists and every reported number. With 30 days the P90
# CI was about +/-4 min, enough to misreport a plan at 13.4 min as missing a
# 15-min target; 300 days narrows it to about +/-1 min (see research/REPORT.md).
CONFIRM_REPLICATIONS = 300
CONFIRM_TOP_K = 30          # Finalists re-evaluated in the second stage (the cost
                            # surface is flat near the optimum, so shortlist widely)
MEAN_SERVICE_TIME = 8.0     # minutes
P90_TARGET = 15.0           # minutes
STRESS_FACTORS = (0.9, 1.0, 1.1, 1.2)   # Demand multipliers for robustness checks
SLOT_NAMES = ["8-9AM", "9-10AM", "10-11AM", "11AM-12PM",
              "12-1PM", "1-2PM", "2-3PM", "3-4PM"]

# Two-sided 95% Student-t critical values, indexed by degrees of freedom
T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def find_simulator() -> Path:
    """Return the first built simulator found, or the default MinGW path."""
    for candidate in SIMULATOR_CANDIDATES:
        if candidate.exists():
            return candidate
    return SIMULATOR_CANDIDATES[0]


def t_critical_95(df: int) -> float:
    """Two-sided 95% t critical value (normal approximation beyond df=30)."""
    return T_TABLE_95.get(df, 1.96)


def compute_confidence_interval(data: list) -> tuple:
    """Compute 95% confidence interval using the t-distribution."""
    n = len(data)
    if n < 2:
        mean = data[0] if data else 0.0
        return (mean, mean)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    std_err = math.sqrt(variance / n)
    margin = t_critical_95(n - 1) * std_err
    return (mean - margin, mean + margin)


@dataclass
class SimulationResult:
    """Results from a single simulation configuration."""
    staffing: tuple
    mean_wait: float
    p90_wait: float
    avg_served: float
    avg_arrived: float
    utilization: list
    total_staff_hours: int
    avg_overtime: float = 0.0          # Minutes past closing to clear the queue
    cost_score: float = 0.0
    mean_wait_ci: tuple = (0.0, 0.0)   # 95% confidence interval
    p90_wait_ci: tuple = (0.0, 0.0)    # 95% confidence interval
    n_replications: int = 1
    # Per arrival hour: fraction of citizens waiting > wait_threshold, pooled
    # over all replications (ratio of sums)
    late_prob_per_slot: list = field(default_factory=list)
    arrivals_per_slot: list = field(default_factory=list)   # Total over replications
    daily_p90s: list = field(default_factory=list)          # One P90 per replication (day)
    daily_mean_waits: list = field(default_factory=list)    # One mean wait per replication
    daily_arrivals: list = field(default_factory=list)      # [rep][slot] arrival counts
    daily_late: list = field(default_factory=list)          # [rep][slot] late counts
    mean_service: float = 0.0
    # (result, mean cost difference, 95% CI) for finalists statistically tied with this one
    tied_alternatives: list = field(default_factory=list)


# ============================================================================
# Simulation Interface
# ============================================================================

def run_simulation(
    staffing: list[int],
    arrival_rates: list[float] = None,
    replications: int = NUM_REPLICATIONS,
    seed: int = 42,
    simulator_path: Path = None,
    duration: Optional[float] = None,
    mean_service: float = MEAN_SERVICE_TIME,
    service_dist: str = "exp",
    service_cv: float = 1.0,
    rate_cv: float = 0.0,
    wait_threshold: float = 15.0
) -> SimulationResult:
    """
    Execute C++ simulator with given staffing configuration.
    Runs all replications in one process and computes confidence intervals.

    Replication i always uses seed + i, and the simulator keeps arrivals and
    service times on separate random streams, so different staffing plans are
    compared on the same citizens (common random numbers).

    Args:
        staffing: List of 8 integers (windows per hourly slot)
        arrival_rates: List of 8 floats (arrivals per hour)
        replications: Number of independent simulation runs
        seed: Base random seed
        simulator_path: Path to queue_sim executable
        duration: Closing time in minutes (simulator default: 480)
        mean_service: Mean service time in minutes
        service_dist: "exp", "lognormal" or "det"
        service_cv: Service-time CV (lognormal only)
        rate_cv: CV of a random day-level demand multiplier (0 = Poisson)
        wait_threshold: Minutes; per-hour late probability counts waits above it

    Returns:
        SimulationResult with aggregated metrics and 95% CIs
    """
    if arrival_rates is None:
        arrival_rates = DEFAULT_ARRIVAL_RATES

    if simulator_path is None:
        simulator_path = find_simulator()

    cmd = [
        str(simulator_path),
        "--staffing", ",".join(str(s) for s in staffing),
        "--arrivals", ",".join(str(a) for a in arrival_rates),
        "--service-time", str(mean_service),
        "--seed", str(seed),
        "--replications", str(replications),
        "--per-replication",
    ]
    if duration is not None:
        cmd += ["--duration", str(duration)]
    if service_dist != "exp":
        cmd += ["--service-dist", service_dist, "--service-cv", str(service_cv)]
    if rate_cv:
        cmd += ["--rate-cv", str(rate_cv)]
    if wait_threshold != 15.0:
        cmd += ["--wait-threshold", str(wait_threshold)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=300
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Simulator not found at {simulator_path}. "
            "Please build the C++ project first."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Simulation failed: {e.stderr}")

    # Parse one CSV row per replication
    rows = [
        {k: float(v) for k, v in row.items()}
        for row in csv.DictReader(io.StringIO(result.stdout))
    ]
    if not rows:
        raise RuntimeError(f"Simulator produced no output: {result.stderr}")

    def column(name: str) -> list:
        return [row[name] for row in rows]

    def mean(values: list) -> float:
        return sum(values) / len(values)

    mean_waits = column('mean_wait')
    p90_waits = column('p90_wait')
    arrivals = [sum(column(f'arr_{i}')) for i in range(8)]
    late = [sum(column(f'late_{i}')) for i in range(8)]

    return SimulationResult(
        staffing=tuple(staffing),
        mean_wait=mean(mean_waits),
        p90_wait=mean(p90_waits),
        avg_served=mean(column('served')),
        avg_arrived=mean(column('arrived')),
        utilization=[mean(column(f'util_{i}')) for i in range(8)],
        total_staff_hours=sum(staffing),
        avg_overtime=mean(column('overtime')),
        mean_wait_ci=compute_confidence_interval(mean_waits),
        p90_wait_ci=compute_confidence_interval(p90_waits),
        n_replications=len(rows),
        late_prob_per_slot=[l / a if a else 0.0 for l, a in zip(late, arrivals)],
        arrivals_per_slot=arrivals,
        daily_p90s=p90_waits,
        daily_mean_waits=mean_waits,
        daily_arrivals=[[int(row[f'arr_{i}']) for i in range(8)] for row in rows],
        daily_late=[[int(row[f'late_{i}']) for i in range(8)] for row in rows],
        mean_service=mean(column('mean_service'))
    )


# ============================================================================
# Analytical Baseline: Erlang-C (M/M/c) and SIPP Staffing
# ============================================================================

def erlang_c(c: int, offered_load: float) -> float:
    """
    Probability an arriving customer must wait in an M/M/c queue.

    Uses the numerically stable Erlang-B recursion. Returns 1.0 when the
    system is unstable (offered_load >= c).
    """
    if offered_load >= c:
        return 1.0
    erlang_b = 1.0
    for k in range(1, c + 1):
        erlang_b = offered_load * erlang_b / (k + offered_load * erlang_b)
    rho = offered_load / c
    return erlang_b / (1 - rho * (1 - erlang_b))


def mmc_mean_wait(c: int, arrivals_per_hour: float,
                  mean_service: float = MEAN_SERVICE_TIME) -> float:
    """Steady-state mean wait in queue (minutes) for M/M/c."""
    lam = arrivals_per_hour / 60.0
    mu = 1.0 / mean_service
    if lam >= c * mu:
        return math.inf
    return erlang_c(c, lam / mu) / (c * mu - lam)


def mmc_prob_wait_exceeds(c: int, arrivals_per_hour: float, threshold: float,
                          mean_service: float = MEAN_SERVICE_TIME) -> float:
    """Steady-state P(wait > threshold minutes) for M/M/c."""
    lam = arrivals_per_hour / 60.0
    mu = 1.0 / mean_service
    if lam >= c * mu:
        return 1.0
    return erlang_c(c, lam / mu) * math.exp(-(c * mu - lam) * threshold)


def sipp_staffing(
    arrival_rates: list[float] = None,
    wait_threshold: float = P90_TARGET,
    max_exceed_prob: float = 0.10
) -> list[int]:
    """
    Stationary Independent Period-by-Period (SIPP) staffing.

    Treats each hour as its own steady-state M/M/c queue and picks the fewest
    windows with P(wait > wait_threshold) <= max_exceed_prob, i.e. a steady-state
    P90 wait within the threshold. This is the textbook Erlang-C method; Green,
    Kolesar & Soares (2001) show it can understaff when demand changes between
    periods, which the simulation measures directly.
    """
    if arrival_rates is None:
        arrival_rates = DEFAULT_ARRIVAL_RATES

    staffing = []
    for lam in arrival_rates:
        c = 1
        while mmc_prob_wait_exceeds(c, lam, wait_threshold) > max_exceed_prob:
            c += 1
        staffing.append(max(MIN_WINDOWS_PER_SLOT, min(c, MAX_WINDOWS_PER_SLOT)))

    return staffing


# ============================================================================
# Optimization: Grid Search
# ============================================================================

def generate_feasible_staffing(
    max_windows: int = MAX_WINDOWS_PER_SLOT,
    max_total: int = MAX_TOTAL_STAFF_HOURS,
    min_windows: int = MIN_WINDOWS_PER_SLOT,
    sample_size: Optional[int] = None
) -> list[tuple]:
    """
    Generate feasible staffing configurations.

    Constraints:
        - Each slot: min_windows <= s_i <= max_windows
        - Total: sum(s_i) <= max_total

    Args:
        sample_size: If set, randomly sample this many configurations
                     (default: evaluate all of them)
    """
    import random

    feasible = []
    ranges = [range(min_windows, max_windows + 1) for _ in range(8)]

    for config in itertools.product(*ranges):
        if sum(config) <= max_total:
            feasible.append(config)

    if sample_size and len(feasible) > sample_size:
        random.seed(42)
        feasible = random.sample(feasible, sample_size)

    return feasible


def compute_cost(
    result: SimulationResult,
    wait_weight: float = 1.0,
    staff_weight: float = 0.5
) -> float:
    """
    Compute weighted cost: w1 * mean_wait + w2 * total_staff_hours

    Per project plan objective function.
    """
    return wait_weight * result.mean_wait + staff_weight * result.total_staff_hours


def _evaluate_all(
    configurations: list[tuple],
    replications: int,
    simulator_path: Path,
    workers: int,
    verbose: bool
) -> list[SimulationResult]:
    """Simulate configurations in parallel (each one is a separate process)."""
    def evaluate(config):
        try:
            return run_simulation(
                list(config),
                replications=replications,
                simulator_path=simulator_path
            )
        except Exception as e:
            if verbose:
                print(f"  Warning: Config {config} failed: {e}")
            return None

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(evaluate, configurations)):
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1}/{len(configurations)}")
            if result is not None:
                results.append(result)
    return results


def grid_search_optimize(
    wait_weight: float = 1.0,
    staff_weight: float = 0.5,
    p90_target: Optional[float] = None,
    simulator_path: Path = None,
    verbose: bool = True,
    sample_size: Optional[int] = None,
    replications: int = NUM_REPLICATIONS,
    confirm_top_k: int = CONFIRM_TOP_K,
    confirm_replications: int = CONFIRM_REPLICATIONS,
    workers: Optional[int] = None
) -> tuple[SimulationResult, list[SimulationResult]]:
    """
    Two-stage grid search over feasible staffing configurations.

    Stage 1 screens every configuration with `replications` runs. Stage 2
    re-evaluates the best `confirm_top_k` candidates whose P90 CI could meet
    the target with `confirm_replications` runs and picks the winner from
    those more precise estimates, so a lucky screening run cannot win.

    Args:
        wait_weight: Weight for mean wait time in cost function
        staff_weight: Weight for staff-hours in cost function
        p90_target: If set, only accept configs whose mean P90 meets it
        simulator_path: Path to simulator executable
        verbose: Print progress updates
        sample_size: Evaluate a random sample instead of every configuration
        replications: Screening replications per configuration
        confirm_top_k: Finalists re-evaluated in stage 2
        confirm_replications: Replications per finalist
        workers: Parallel simulator processes (default: CPU count)

    Returns:
        (best_result, all_screening_results)
    """
    if simulator_path is None:
        simulator_path = find_simulator()
    workers = workers or os.cpu_count() or 4
    configurations = generate_feasible_staffing(sample_size=sample_size)

    if verbose:
        print(f"Grid Search: Evaluating {len(configurations)} configurations "
              f"x {replications} replications")
        print(f"Cost function: {wait_weight}*wait + {staff_weight}*staff_hours")
        if p90_target:
            print(f"P90 target: <= {p90_target} minutes")

    # Stage 1: screen everything
    results = _evaluate_all(configurations, replications, simulator_path, workers, verbose)
    for result in results:
        result.cost_score = compute_cost(result, wait_weight, staff_weight)

    # Keep anything that could plausibly meet the target given screening noise
    candidates = [
        r for r in results
        if p90_target is None or r.p90_wait_ci[0] <= p90_target
    ]
    candidates.sort(key=lambda r: r.cost_score)
    finalists = candidates[:confirm_top_k]

    if verbose:
        print(f"Stage 2: Re-evaluating {len(finalists)} finalists "
              f"x {confirm_replications} replications")

    # Stage 2: confirm finalists with more replications
    confirmed = _evaluate_all(
        [r.staffing for r in finalists], confirm_replications,
        simulator_path, workers, verbose=False
    )
    best_result = None
    best_cost = float('inf')
    for result in confirmed:
        result.cost_score = compute_cost(result, wait_weight, staff_weight)
        meets_constraint = (p90_target is None or result.p90_wait <= p90_target)
        if meets_constraint and result.cost_score < best_cost:
            best_cost = result.cost_score
            best_result = result

    # Finalists share seeds (common random numbers), so compare daily costs
    # pairwise: an alternative whose 95% CI on the cost difference includes 0
    # cannot be told apart from the winner with this many replications
    if best_result is not None:
        def daily_cost(r):
            return [wait_weight * w + staff_weight * r.total_staff_hours
                    for w in r.daily_mean_waits]
        best_daily = daily_cost(best_result)
        for result in confirmed:
            if result is best_result or not (p90_target is None or result.p90_wait <= p90_target):
                continue
            diffs = [a - b for a, b in zip(daily_cost(result), best_daily)]
            low, high = compute_confidence_interval(diffs)
            if low <= 0:
                best_result.tied_alternatives.append((result, sum(diffs) / len(diffs), (low, high)))

    if verbose:
        print(f"Optimization complete. Best cost: {best_cost:.2f}")

    return best_result, results


# ============================================================================
# Scenario Analysis
# ============================================================================

def flat_staffing(windows_per_slot: int) -> list[int]:
    """Uniform staffing across all slots."""
    return [windows_per_slot] * 8


def _print_scenario(s: SimulationResult):
    print(f"    Staffing: {list(s.staffing)}")
    print(f"    Mean wait: {s.mean_wait:.2f} min  (95% CI: {s.mean_wait_ci[0]:.2f}-{s.mean_wait_ci[1]:.2f})")
    print(f"    P90 wait:  {s.p90_wait:.2f} min  (95% CI: {s.p90_wait_ci[0]:.2f}-{s.p90_wait_ci[1]:.2f})")
    print(f"    Staff-hours: {s.total_staff_hours}  |  Overtime: {s.avg_overtime:.1f} min"
          f"  |  n={s.n_replications} replications")


def run_scenario_analysis(simulator_path: Path = None,
                          sample_size: Optional[int] = None
                          ) -> tuple[dict, list[SimulationResult]]:
    """
    Compare three staffing policies per project plan:
    A) Flat staffing
    B) SIPP (per-hour Erlang-C) staffing
    C) Cost-minimized with 15-minute P90 target

    All scenarios are reported with CONFIRM_REPLICATIONS runs on the same
    seeds, so their differences are not driven by random-number noise.

    Returns:
        (scenarios, all_screening_results); the latter feeds pareto_frontier
    """
    scenarios = {}

    print("=" * 60)
    print("SCENARIO ANALYSIS: Government Service Center Staffing")
    print("=" * 60)

    # Scenario A: Flat staffing (3 windows all day)
    print("\n[A] Flat Staffing (3 windows/slot)")
    scenarios['flat'] = run_simulation(
        flat_staffing(3), replications=CONFIRM_REPLICATIONS, simulator_path=simulator_path
    )
    _print_scenario(scenarios['flat'])

    # Scenario B: SIPP / Erlang-C
    print(f"\n[B] SIPP / Erlang-C (steady-state P90 <= {P90_TARGET:.0f} min each hour)")
    scenarios['sipp'] = run_simulation(
        sipp_staffing(), replications=CONFIRM_REPLICATIONS, simulator_path=simulator_path
    )
    _print_scenario(scenarios['sipp'])

    # Scenario C: Optimized with P90 <= 15 minutes
    print(f"\n[C] Optimized (P90 <= {P90_TARGET:.0f} min target)")
    best, all_results = grid_search_optimize(
        wait_weight=1.0,
        staff_weight=0.5,
        p90_target=P90_TARGET,
        simulator_path=simulator_path,
        verbose=False,
        sample_size=sample_size
    )
    if best:
        scenarios['optimized'] = best
        _print_scenario(best)
    else:
        print("    No feasible solution found")

    return scenarios, all_results


# ============================================================================
# Robustness and Trade-offs
# ============================================================================

def pareto_frontier(
    results: list[SimulationResult],
    simulator_path: Path = None,
    confirm_replications: int = CONFIRM_REPLICATIONS,
    top_k: int = 10,
    known: Optional[list[SimulationResult]] = None
) -> list[SimulationResult]:
    """
    Best achievable P90 wait for each total staff-hour budget.

    Takes the `top_k` lowest-P90 plans per staff-hour total from screening
    results, re-simulates them with more replications (unless
    confirm_replications is 0), and keeps the best. The minimum of many noisy
    screening estimates is optimistic, so picking a single plan from screening
    alone tends to select a lucky one rather than the best one.

    `known` are plans already evaluated with confirm_replications on the same
    seeds (e.g. the scenarios); they compete directly with the re-run finalists.
    """
    by_total = {}
    for r in results:
        by_total.setdefault(r.total_staff_hours, []).append(r)

    frontier = []
    for total in sorted(by_total):
        candidates = sorted(by_total[total], key=lambda r: r.p90_wait)[:top_k]
        if confirm_replications:
            candidates = _evaluate_all(
                [r.staffing for r in candidates], confirm_replications,
                simulator_path or find_simulator(), os.cpu_count() or 4, verbose=False
            )
        candidates += [r for r in (known or []) if r.total_staff_hours == total]
        frontier.append(min(candidates, key=lambda r: r.p90_wait))
    return frontier


def stress_test(
    staffing: list[int],
    factors: tuple = STRESS_FACTORS,
    arrival_rates: list[float] = None,
    simulator_path: Path = None,
    replications: int = CONFIRM_REPLICATIONS
) -> dict:
    """
    Re-run a plan with every hourly arrival rate scaled by each factor.

    Uses the same seeds for every factor and plan, so results are comparable.
    Returns {factor: SimulationResult}.
    """
    if arrival_rates is None:
        arrival_rates = DEFAULT_ARRIVAL_RATES
    return {
        f: run_simulation(list(staffing), [a * f for a in arrival_rates],
                          replications=replications, simulator_path=simulator_path)
        for f in factors
    }


def contingency_window(
    staffing: list[int],
    factor: float,
    arrival_rates: list[float] = None,
    simulator_path: Path = None,
    p90_target: float = P90_TARGET
) -> Optional[tuple[int, SimulationResult]]:
    """
    Single extra window that best restores the P90 target under higher demand.

    Returns (slot, result) for the slot whose +1 window gives the lowest P90
    among those meeting the target, or None if no single window is enough.
    """
    if arrival_rates is None:
        arrival_rates = DEFAULT_ARRIVAL_RATES
    scaled = [a * factor for a in arrival_rates]
    best = None
    for slot, windows in enumerate(staffing):
        if windows >= MAX_WINDOWS_PER_SLOT:
            continue
        trial = list(staffing)
        trial[slot] += 1
        result = run_simulation(trial, scaled, replications=CONFIRM_REPLICATIONS,
                                simulator_path=simulator_path)
        if result.p90_wait <= p90_target and (best is None or result.p90_wait < best[1].p90_wait):
            best = (slot, result)
    return best


def print_frontier(frontier: list[SimulationResult]):
    print("\n" + "=" * 60)
    print("TRADE-OFF: best P90 wait per staff-hour budget")
    print("=" * 60)
    print(f"  {'Staff-hrs':>9}  {'P90 wait (95% CI)':<22} {'Mean wait':>9}  Staffing")
    for r in frontier:
        ci = f"{r.p90_wait:.1f} ({r.p90_wait_ci[0]:.1f}-{r.p90_wait_ci[1]:.1f})"
        flag = "" if r.p90_wait <= P90_TARGET else "  > target"
        print(f"  {r.total_staff_hours:>9}  {ci:<22} {r.mean_wait:>9.1f}  "
              f"{list(r.staffing)}{flag}")


def print_stress(stress: dict):
    """stress: {scenario_name: {factor: SimulationResult}}"""
    factors = sorted(next(iter(stress.values())))
    print("\n" + "=" * 60)
    print("ROBUSTNESS: P90 wait (min) if demand differs from forecast")
    print("=" * 60)
    header = "".join(f"{f'{f:.0%} demand':>14}" for f in factors)
    print(f"  {'Scenario':<10}{header}")
    for name, by_factor in stress.items():
        cells = ""
        for f in factors:
            r = by_factor[f]
            mark = "ok" if r.p90_wait <= P90_TARGET else "MISS"
            cells += f"{f'{r.p90_wait:.1f} {mark}':>14}"
        print(f"  {name.capitalize():<10}{cells}")


# ============================================================================
# Decision Support Outputs
# ============================================================================

def generate_recommendation(scenarios: dict, stress: Optional[dict] = None,
                            contingency: Optional[tuple] = None) -> str:
    """
    Generate decision recommendation comparing scenarios.

    Args:
        stress: {scenario_name: {factor: SimulationResult}} from stress_test
        contingency: (factor, contingency_window result) for the optimized plan
    """
    report = []
    report.append("\n" + "=" * 60)
    report.append("DECISION SUPPORT RECOMMENDATION")
    report.append("=" * 60)

    flat = scenarios.get('flat')
    sipp = scenarios.get('sipp')
    optimized = scenarios.get('optimized')

    if flat and sipp:
        wait_change = sipp.p90_wait - flat.p90_wait
        staff_diff = sipp.total_staff_hours - flat.total_staff_hours

        report.append(f"\nSIPP vs Flat staffing:")
        report.append(f"  - P90 wait change: {wait_change:+.1f} minutes")
        report.append(f"  - Staff-hour change: {staff_diff:+d} hours")
        verdict = "meets" if sipp.p90_wait <= P90_TARGET else "MISSES"
        report.append(f"  - SIPP is designed for P90 <= {P90_TARGET:.0f} min; simulated P90 "
                      f"{sipp.p90_wait:.1f} min {verdict} the target")

    if optimized and flat:
        wait_change = optimized.p90_wait - flat.p90_wait
        staff_diff = optimized.total_staff_hours - flat.total_staff_hours

        report.append(f"\nOptimized vs Flat staffing:")
        report.append(f"  - P90 wait change: {wait_change:+.1f} minutes")
        report.append(f"  - Staff-hour change: {staff_diff:+d} hours")

    if optimized:
        staffing = list(optimized.staffing)
        peak_slots = [i for i, s in enumerate(staffing) if s == max(staffing)]
        peaks = [SLOT_NAMES[i] for i in peak_slots]

        report.append(f"\n>>> RECOMMENDATION:")
        report.append(f"    Adopt optimized staffing schedule: {staffing}")
        report.append(f"    Peak staffing periods: {', '.join(peaks)}")
        report.append(f"    Expected P90 wait: {optimized.p90_wait:.1f} minutes "
                      f"(95% CI {optimized.p90_wait_ci[0]:.1f}-{optimized.p90_wait_ci[1]:.1f})")
        report.append(f"    Total daily staff-hours: {optimized.total_staff_hours}")
        for alt, diff, (low, high) in optimized.tied_alternatives:
            report.append(f"    Statistically tied: {list(alt.staffing)} "
                          f"({alt.total_staff_hours} h, cost {diff:+.2f}, "
                          f"95% CI {low:+.2f} to {high:+.2f})")

        if stress and 'optimized' in stress:
            holds = [f for f, r in sorted(stress['optimized'].items())
                     if r.p90_wait <= P90_TARGET]
            report.append(f"    Meets the P90 target up to {max(holds):.0%} of forecast demand"
                          if holds else "    Misses the P90 target at every tested demand level")
        if contingency:
            factor, window = contingency
            if window:
                slot, result = window
                report.append(f"    If demand runs {factor - 1:+.0%}: add one window at "
                              f"{SLOT_NAMES[slot]} (P90 {result.p90_wait:.1f} min)")
            else:
                report.append(f"    If demand runs {factor - 1:+.0%}: one extra window is not "
                              f"enough; re-run the optimizer with the new forecast")

    return "\n".join(report)


def export_results_csv(results: list[SimulationResult], filename: str):
    """Export all results to CSV for further analysis."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'slot_0', 'slot_1', 'slot_2', 'slot_3',
            'slot_4', 'slot_5', 'slot_6', 'slot_7',
            'total_staff_hours', 'mean_wait', 'p90_wait',
            'avg_served', 'avg_overtime', 'cost_score'
        ])
        for r in results:
            writer.writerow([
                *r.staffing,
                r.total_staff_hours,
                f"{r.mean_wait:.2f}",
                f"{r.p90_wait:.2f}",
                f"{r.avg_served:.1f}",
                f"{r.avg_overtime:.1f}",
                f"{r.cost_score:.2f}"
            ])
    print(f"Results exported to {filename}")


def plot_scenarios(scenarios: dict, save_path: Optional[Path] = None, show: bool = True,
                   frontier: Optional[list[SimulationResult]] = None):
    """Visualize scenario comparisons with error bars and staffing heatmap.

    Args:
        frontier: If provided, add a staff-hours vs P90 trade-off panel.
        save_path: If provided, save the figure to this path.
        show: If True, open an interactive window.
    """
    if not HAS_MATPLOTLIB:
        print("Plotting skipped: matplotlib not installed.")
        print("Install with: pip install matplotlib")
        return

    labels = []
    p90 = []
    p90_err = []
    mean_wait = []
    mean_err = []
    staff_hours = []
    staffing_data = []

    display_names = {"flat": "Flat", "sipp": "SIPP", "optimized": "Optimized"}
    for name in ["flat", "sipp", "optimized"]:
        if name in scenarios:
            s = scenarios[name]
            labels.append(display_names[name])
            p90.append(s.p90_wait)
            p90_err.append((s.p90_wait_ci[1] - s.p90_wait_ci[0]) / 2)
            mean_wait.append(s.mean_wait)
            mean_err.append((s.mean_wait_ci[1] - s.mean_wait_ci[0]) / 2)
            staff_hours.append(s.total_staff_hours)
            staffing_data.append(list(s.staffing))

    # Create figure with 2 subplots (3 with the trade-off curve)
    if frontier:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ===== Left plot: Bar chart with error bars =====
    x = range(len(labels))
    width = 0.35

    bars1 = ax1.bar([i - width/2 for i in x], p90, width,
                    yerr=p90_err, capsize=5,
                    label="P90 wait", color="#2ecc71", edgecolor="black")
    bars2 = ax1.bar([i + width/2 for i in x], mean_wait, width,
                    yerr=mean_err, capsize=5,
                    label="Mean wait", color="#3498db", edgecolor="black")
    ax1.axhline(P90_TARGET, color="#e74c3c", linestyle="--", linewidth=1,
                label=f"P90 target ({P90_TARGET:.0f} min)")

    # Add value labels above the error bars
    for bars, vals, errs in ((bars1, p90, p90_err), (bars2, mean_wait, mean_err)):
        for bar, val, err in zip(bars, vals, errs):
            ax1.text(bar.get_x() + bar.get_width()/2, val + err + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Staff-hours go in the tick labels so cost sits next to each scenario
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"{label}\n{sh} staff-hrs" for label, sh in zip(labels, staff_hours)],
                        fontsize=11)
    ax1.set_ylabel("Wait Time (minutes)", fontsize=11)
    ax1.set_title("Service Level Comparison (95% CI)", fontsize=12, fontweight='bold')
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, max(max(p90), P90_TARGET) * 1.4)
    ax1.grid(axis='y', alpha=0.3)

    # ===== Right plot: Staffing heatmap =====
    hours = ['8-9', '9-10', '10-11', '11-12', '12-1', '1-2', '2-3', '3-4']
    staffing_array = list(zip(*staffing_data))  # Transpose

    im = ax2.imshow(staffing_array, cmap='YlOrRd', aspect='auto',
                    vmin=MIN_WINDOWS_PER_SLOT, vmax=MAX_WINDOWS_PER_SLOT)

    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_yticks(range(8))
    ax2.set_yticklabels(hours, fontsize=10)
    ax2.set_ylabel("Hour of Day", fontsize=11)
    ax2.set_title("Staffing Schedule Heatmap", fontsize=12, fontweight='bold')

    # Add text annotations in cells
    for i in range(8):
        for j in range(len(labels)):
            val = staffing_data[j][i]
            color = 'white' if val >= 3 else 'black'
            ax2.text(j, i, str(val), ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax2, shrink=0.8,
                        ticks=range(MIN_WINDOWS_PER_SLOT, MAX_WINDOWS_PER_SLOT + 1))
    cbar.set_label('Windows Open', fontsize=10)

    # ===== Optional third plot: staff-hours vs P90 trade-off =====
    if frontier:
        ax3.errorbar(
            [r.total_staff_hours for r in frontier],
            [r.p90_wait for r in frontier],
            yerr=[[r.p90_wait - r.p90_wait_ci[0] for r in frontier],
                  [r.p90_wait_ci[1] - r.p90_wait for r in frontier]],
            fmt='-o', color='#7f8c8d', capsize=3, label='Best plan per budget')
        markers = {"Flat": "s", "SIPP": "^", "Optimized": "*"}
        for label, sh, val in zip(labels, staff_hours, p90):
            ax3.scatter(sh, val, s=180 if label == "Optimized" else 90,
                        marker=markers[label], zorder=3, label=label)
        ax3.axhline(P90_TARGET, color="#e74c3c", linestyle="--", linewidth=1,
                    label=f"P90 target ({P90_TARGET:.0f} min)")
        ax3.set_xlabel("Daily staff-hours", fontsize=11)
        ax3.set_ylabel("P90 wait (minutes)", fontsize=11)
        ax3.set_title("Cost vs Service Trade-off (95% CI)", fontsize=12, fontweight='bold')
        ax3.legend(loc="upper right", fontsize=9)
        ax3.grid(alpha=0.3)

    plt.suptitle("Government Service Center: Staffing Analysis",
                fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================================
# Shift Rosters
# ============================================================================

ROSTER_VALIDATION_SEED = 50042   # Independent block that re-checks the chosen roster
HOURLY_LATE_TARGET = 0.10        # "hourly" target: <= 10% of each hour's arrivals late


def hourly_late_upper(result: SimulationResult) -> list:
    """
    Upper 95% bound of each arrival hour's late rate (waits > 15 min).

    Ratio estimator over days with a delta-method standard error, so days,
    not citizens, are the independent unit.
    """
    upper = []
    n = len(result.daily_arrivals)
    for i in range(8):
        arrivals = [day[i] for day in result.daily_arrivals]
        late = [day[i] for day in result.daily_late]
        total = sum(arrivals)
        if total == 0 or n < 2:
            upper.append(0.0)
            continue
        p = sum(late) / total
        resid = [l - p * a for l, a in zip(late, arrivals)]
        se = math.sqrt(sum(r * r for r in resid) / (n * (n - 1))) / (total / n)
        upper.append(min(1.0, p + 1.96 * se))
    return upper


def _meets_target(result: SimulationResult, target: str) -> tuple:
    """(feasible, score) for a simulated profile; both targets use upper bounds."""
    if target == "hourly":
        return (max(hourly_late_upper(result)) <= HOURLY_LATE_TARGET,
                max(result.late_prob_per_slot))
    return result.p90_wait_ci[1] <= P90_TARGET, result.p90_wait


def roster_search(menu: str = "standard", target: str = "p90",
                  simulator_path: Path = None,
                  replications: int = CONFIRM_REPLICATIONS) -> dict:
    """
    Cheapest shift roster meeting the service target, found by local search
    over shift counts with every candidate judged by simulation.

    Rosters are compared on common seeds (42..), then the winner is re-checked
    on an independent block; if it fails there, the search steps back along
    its accepted path to the cheapest roster that passes.
    """
    from roster import MENUS, Roster
    roster = Roster(MENUS[menu])
    if simulator_path is None:
        simulator_path = find_simulator()

    def simulate(profile, seed=42):
        return run_simulation(profile, replications=replications, seed=seed,
                              simulator_path=simulator_path)

    def check(x):
        return _meets_target(simulate(roster.profile(x)), target)

    # Start from the fewest full-day shifts that meet the target
    k = 1
    while not check(roster.full_days(k))[0]:
        k += 1
        if k > 50:
            raise RuntimeError("no feasible roster with up to 50 full-day shifts")

    best, path, calls = roster.local_search(roster.full_days(k), check)

    stepped_back = 0
    for x in reversed(path):
        validation = simulate(roster.profile(x), seed=ROSTER_VALIDATION_SEED)
        if _meets_target(validation, target)[0]:
            break
        stepped_back += 1
    else:
        raise RuntimeError("no roster on the search path passed validation")

    # For comparison: trim windows from the roster's own hourly profile while
    # the target still holds, giving an hourly plan with the same service
    hourly = roster.profile(x)
    while True:
        trials = [hourly[:i] + [hourly[i] - 1] + hourly[i + 1:]
                  for i in range(8) if hourly[i] > 1]
        with ThreadPoolExecutor() as pool:
            checks = list(pool.map(lambda p: _meets_target(simulate(p), target), trials))
        ok = [(score, p) for p, (good, score) in zip(trials, checks) if good]
        if not ok:
            break
        hourly = min(ok)[1]

    return {"roster": roster, "shifts": x, "profile": roster.profile(x),
            "paid_hours": roster.paid_hours(x), "validation": validation,
            "stepped_back": stepped_back, "calls": calls, "menu": menu,
            "target": target, "hourly_plan": hourly}


def format_roster_report(report: dict) -> str:
    r = report["validation"]
    roster, x = report["roster"], report["shifts"]
    target = ("mean daily P90 <= 15 min (upper 95% bound)" if report["target"] == "p90"
              else "every hour <= 10% of arrivals waiting > 15 min (upper 95% bound)")
    hourly_hours = sum(report["hourly_plan"])
    lines = [
        "\n" + "=" * 60,
        f"SHIFT ROSTER ({report['menu']} menu)",
        "=" * 60,
        f"    Target: {target}",
        f"    Roster: {roster.format(x)}",
        f"    Windows open by hour: {report['profile']}",
        f"    Paid staff-hours: {report['paid_hours']}",
        f"    P90 wait: {r.p90_wait:.1f} min (95% CI {r.p90_wait_ci[0]:.1f}-{r.p90_wait_ci[1]:.1f})",
        "    Late (> 15 min) by hour: "
        + "  ".join(f"{h} {p * 100:.0f}%" for h, p in
                    zip(["8", "9", "10", "11", "12", "1", "2", "3"], r.late_prob_per_slot)),
        f"    Validated on {r.n_replications} independent days"
        + ("" if report["stepped_back"] == 0
           else f" (the cheapest roster failed; stepped back {report['stepped_back']})"),
        f"    Price of shifts: an hourly plan {report['hourly_plan']} meets the same "
        f"target with {hourly_hours} window-hours; the roster pays "
        f"{report['paid_hours'] - hourly_hours:+d} h "
        f"({100 * (report['paid_hours'] / hourly_hours - 1):+.0f}%)",
    ]
    return "\n".join(lines)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main analysis workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Government Service Center Queue Optimizer"
    )
    parser.add_argument(
        "--simulator",
        type=Path,
        default=None,
        help="Path to queue_sim executable (default: search cpp/build)"
    )
    parser.add_argument(
        "--scenario-analysis",
        action="store_true",
        help="Run full scenario comparison"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run grid search optimization"
    )
    parser.add_argument(
        "--p90-target",
        type=float,
        default=None,
        help="P90 wait limit in minutes for --optimize (default: none)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Evaluate a random sample of configurations (default: all)"
    )
    parser.add_argument(
        "--frontier",
        action="store_true",
        help="Show best P90 wait for each staff-hour budget"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export results to CSV file"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show scenario comparison chart (requires matplotlib)"
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Save scenario comparison figure to a file (e.g., outputs/fig.png)"
    )

    parser.add_argument(
        "--shifts",
        choices=["standard", "flexible"],
        default=None,
        help="Recommend a shift roster: standard (8h + 4h shifts) or flexible (+ 6h)"
    )
    parser.add_argument(
        "--target",
        choices=["p90", "hourly"],
        default="p90",
        help="Roster target: mean daily P90 <= 15 min, or every hour <= 10%% late"
    )

    args = parser.parse_args()
    sim_path = args.simulator or find_simulator()

    if args.scenario_analysis:
        scenarios, all_results = run_scenario_analysis(simulator_path=sim_path,
                                                       sample_size=args.sample_size)

        # Robustness to forecast error (same seeds for every plan and factor)
        stress = {
            name: stress_test(list(s.staffing), simulator_path=sim_path)
            for name, s in scenarios.items()
        }
        print_stress(stress)
        contingency = None
        optimized_stress = stress.get('optimized', {})
        misses = [f for f, r in sorted(optimized_stress.items())
                  if f > 1.0 and r.p90_wait > P90_TARGET]
        if misses:
            contingency = (misses[0], contingency_window(
                list(scenarios['optimized'].staffing), misses[0], simulator_path=sim_path))

        frontier = None
        if args.frontier:
            frontier = pareto_frontier(all_results, simulator_path=sim_path,
                                       known=list(scenarios.values()))
            print_frontier(frontier)

        print(generate_recommendation(scenarios, stress, contingency))
        if args.shifts:
            print(format_roster_report(roster_search(args.shifts, args.target, sim_path)))
        if args.plot or args.save_plot is not None:
            plot_scenarios(scenarios, save_path=args.save_plot, show=args.plot,
                           frontier=frontier)

    elif args.optimize:
        print("Running grid search optimization...")
        best, all_results = grid_search_optimize(
            p90_target=args.p90_target,
            simulator_path=sim_path,
            sample_size=args.sample_size,
            verbose=True
        )

        if best:
            print(f"\nBest configuration found:")
            print(f"  Staffing: {list(best.staffing)}")
            print(f"  Mean wait: {best.mean_wait:.2f} minutes")
            print(f"  P90 wait: {best.p90_wait:.2f} minutes")
            print(f"  Staff-hours: {best.total_staff_hours}")
            print(f"  Cost score: {best.cost_score:.2f}")

        if args.frontier:
            print_frontier(pareto_frontier(all_results, simulator_path=sim_path))

        if args.export:
            export_results_csv(all_results, args.export)

    elif args.shifts:
        print(f"Searching {args.shifts} shift rosters...")
        print(format_roster_report(roster_search(args.shifts, args.target, sim_path)))

    else:
        # Default: single simulation demo
        print("Running single simulation demo...")
        staffing = [2, 3, 2, 2, 2, 3, 3, 2]
        try:
            result = run_simulation(staffing, simulator_path=sim_path)
            print(f"\nStaffing: {staffing}")
            print(f"Mean wait time: {result.mean_wait:.2f} minutes")
            print(f"P90 wait time: {result.p90_wait:.2f} minutes")
            print(f"Citizens served: {result.avg_served:.0f}")
            print(f"Total staff-hours: {result.total_staff_hours}")
        except RuntimeError as e:
            print(f"Error: {e}")
            print("\nTo build the simulator:")
            print("  cd cpp && mkdir build && cd build")
            print("  cmake .. && cmake --build . --config Release")


if __name__ == "__main__":
    main()

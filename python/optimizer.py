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
import statistics
from dataclasses import dataclass
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

# Path to compiled C++ simulator (adjust after build)
SIMULATOR_PATH = Path(__file__).parent.parent / "cpp" / "build" / "queue_sim.exe"

# Default arrival rates: morning peak, midday lull, afternoon peak (per hour)
DEFAULT_ARRIVAL_RATES = [12.0, 15.0, 10.0, 8.0, 8.0, 12.0, 14.0, 10.0]

# Constraints from project plan
MAX_WINDOWS_PER_SLOT = 4
MAX_TOTAL_STAFF_HOURS = 28
NUM_REPLICATIONS = 10  # Default replications per configuration
MEAN_SERVICE_TIME = 8.0  # minutes


def compute_confidence_interval(data: list, confidence: float = 0.95) -> tuple:
    """Compute 95% confidence interval using t-distribution."""
    import math
    n = len(data)
    if n < 2:
        mean = data[0] if data else 0.0
        return (mean, mean)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    std_err = math.sqrt(variance / n)
    # t-value approximation for 95% CI
    t_val = 2.262 if n <= 10 else 1.96
    margin = t_val * std_err
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
    cost_score: float = 0.0
    mean_wait_ci: tuple = (0.0, 0.0)   # 95% confidence interval
    p90_wait_ci: tuple = (0.0, 0.0)    # 95% confidence interval
    n_replications: int = 1


# ============================================================================
# Simulation Interface
# ============================================================================

def run_simulation(
    staffing: list[int],
    arrival_rates: list[float] = None,
    replications: int = NUM_REPLICATIONS,
    seed: int = 42,
    simulator_path: Path = None
) -> SimulationResult:
    """
    Execute C++ simulator with given staffing configuration.
    Runs multiple replications and computes confidence intervals.
    
    Args:
        staffing: List of 8 integers (windows per hourly slot)
        arrival_rates: List of 8 floats (arrivals per hour)
        replications: Number of independent simulation runs
        seed: Base random seed
        simulator_path: Path to queue_sim executable
    
    Returns:
        SimulationResult with aggregated metrics and 95% CIs
    """
    if arrival_rates is None:
        arrival_rates = DEFAULT_ARRIVAL_RATES
    
    if simulator_path is None:
        simulator_path = SIMULATOR_PATH
    
    staffing_str = ",".join(str(s) for s in staffing)
    arrivals_str = ",".join(str(a) for a in arrival_rates)
    
    # Run each replication separately to collect individual results for CI
    mean_waits = []
    p90_waits = []
    served_counts = []
    arrived_counts = []
    
    for rep in range(replications):
        cmd = [
            str(simulator_path),
            "--staffing", staffing_str,
            "--arrivals", arrivals_str,
            "--service-time", str(MEAN_SERVICE_TIME),
            "--seed", str(seed + rep),
            "--replications", "1"
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=60
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Simulator not found at {simulator_path}. "
                "Please build the C++ project first."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Simulation failed: {e.stderr}")
        
        # Parse CSV output
        metrics = {}
        reader = csv.DictReader(io.StringIO(result.stdout))
        for row in reader:
            metrics[row['metric']] = float(row['value'])
        
        mean_waits.append(metrics.get('mean_wait_time', 0.0))
        p90_waits.append(metrics.get('p90_wait_time', 0.0))
        served_counts.append(metrics.get('avg_served', 0.0))
        arrived_counts.append(metrics.get('avg_arrived', 0.0))
    
    # Compute point estimates and confidence intervals
    mean_wait = sum(mean_waits) / len(mean_waits)
    p90_wait = sum(p90_waits) / len(p90_waits)
    avg_served = sum(served_counts) / len(served_counts)
    avg_arrived = sum(arrived_counts) / len(arrived_counts)
    
    mean_wait_ci = compute_confidence_interval(mean_waits)
    p90_wait_ci = compute_confidence_interval(p90_waits)
    
    # Get utilization from last run (stable across replications)
    utilization = [
        metrics.get(f'utilization_slot_{i}', 0.0) 
        for i in range(8)
    ]
    
    return SimulationResult(
        staffing=tuple(staffing),
        mean_wait=mean_wait,
        p90_wait=p90_wait,
        avg_served=avg_served,
        avg_arrived=avg_arrived,
        utilization=utilization,
        total_staff_hours=sum(staffing),
        mean_wait_ci=mean_wait_ci,
        p90_wait_ci=p90_wait_ci,
        n_replications=replications
    )


# ============================================================================
# Optimization: Grid Search
# ============================================================================

def generate_feasible_staffing(
    max_windows: int = MAX_WINDOWS_PER_SLOT,
    max_total: int = MAX_TOTAL_STAFF_HOURS,
    min_windows: int = 2,
    sample_size: Optional[int] = None
) -> list[tuple]:
    """
    Generate feasible staffing configurations.
    
    Constraints:
        - Each slot: min_windows <= s_i <= max_windows
        - Total: sum(s_i) <= max_total
        
    Args:
        sample_size: If set, randomly sample this many configurations
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


def grid_search_optimize(
    wait_weight: float = 1.0,
    staff_weight: float = 0.5,
    p90_target: Optional[float] = None,
    simulator_path: Path = None,
    verbose: bool = True,
    quick_mode: bool = False,
    sample_size: int = 100
) -> tuple[SimulationResult, list[SimulationResult]]:
    """
    Grid search over feasible staffing configurations.
    
    Args:
        wait_weight: Weight for mean wait time in cost function
        staff_weight: Weight for staff-hours in cost function
        p90_target: If set, filter to configs meeting this p90 threshold
        simulator_path: Path to simulator executable
        verbose: Print progress updates
        quick_mode: Use fewer replications for faster search
        sample_size: Max configurations to evaluate (random sampling)
    
    Returns:
        (best_result, all_results)
    """
    configurations = generate_feasible_staffing(sample_size=sample_size)
    replications = 3 if quick_mode else NUM_REPLICATIONS
    
    if verbose:
        print(f"Grid Search: Evaluating {len(configurations)} configurations")
        print(f"Cost function: {wait_weight}*wait + {staff_weight}*staff_hours")
        if p90_target:
            print(f"P90 target: <= {p90_target} minutes")
    
    results = []
    best_result = None
    best_cost = float('inf')
    
    for i, config in enumerate(configurations):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(configurations)}")
        
        try:
            result = run_simulation(
                list(config),
                replications=replications,
                simulator_path=simulator_path
            )
            result.cost_score = compute_cost(result, wait_weight, staff_weight)
            results.append(result)
            
            # Check if this is the best (considering p90 constraint if set)
            meets_constraint = (p90_target is None or result.p90_wait <= p90_target)
            if meets_constraint and result.cost_score < best_cost:
                best_cost = result.cost_score
                best_result = result
                
        except Exception as e:
            if verbose:
                print(f"  Warning: Config {config} failed: {e}")
    
    if verbose:
        print(f"Optimization complete. Best cost: {best_cost:.2f}")
    
    return best_result, results


# ============================================================================
# Scenario Analysis
# ============================================================================

def flat_staffing(windows_per_slot: int) -> list[int]:
    """Uniform staffing across all slots."""
    return [windows_per_slot] * 8


def demand_matched_staffing(
    arrival_rates: list[float] = None,
    service_rate: float = 1 / MEAN_SERVICE_TIME,
    buffer: float = 1.2
) -> list[int]:
    """
    Staff proportionally to demand using M/M/c approximation.
    
    s_i >= buffer * (lambda_i / mu)
    """
    if arrival_rates is None:
        arrival_rates = DEFAULT_ARRIVAL_RATES
    
    staffing = []
    for lam in arrival_rates:
        min_servers = int(buffer * lam / (60 * service_rate)) + 1
        staffing.append(max(1, min(min_servers, MAX_WINDOWS_PER_SLOT)))
    
    return staffing


def run_scenario_analysis(simulator_path: Path = None) -> dict:
    """
    Compare three staffing policies per project plan:
    A) Flat staffing
    B) Demand-matched staffing  
    C) Cost-minimized with 15-minute P90 target
    """
    scenarios = {}
    
    print("=" * 60)
    print("SCENARIO ANALYSIS: Government Service Center Staffing")
    print("=" * 60)
    
    # Scenario A: Flat staffing (3 windows all day)
    print("\n[A] Flat Staffing (3 windows/slot)")
    flat = flat_staffing(3)
    scenarios['flat'] = run_simulation(flat, simulator_path=simulator_path)
    s = scenarios['flat']
    print(f"    Staffing: {flat}")
    print(f"    Mean wait: {s.mean_wait:.2f} min  (95% CI: {s.mean_wait_ci[0]:.2f}–{s.mean_wait_ci[1]:.2f})")
    print(f"    P90 wait:  {s.p90_wait:.2f} min  (95% CI: {s.p90_wait_ci[0]:.2f}–{s.p90_wait_ci[1]:.2f})")
    print(f"    Staff-hours: {s.total_staff_hours}  |  n={s.n_replications} replications")
    
    # Scenario B: Demand-matched
    print("\n[B] Demand-Matched Staffing")
    matched = demand_matched_staffing()
    scenarios['matched'] = run_simulation(matched, simulator_path=simulator_path)
    s = scenarios['matched']
    print(f"    Staffing: {matched}")
    print(f"    Mean wait: {s.mean_wait:.2f} min  (95% CI: {s.mean_wait_ci[0]:.2f}–{s.mean_wait_ci[1]:.2f})")
    print(f"    P90 wait:  {s.p90_wait:.2f} min  (95% CI: {s.p90_wait_ci[0]:.2f}–{s.p90_wait_ci[1]:.2f})")
    print(f"    Staff-hours: {s.total_staff_hours}  |  n={s.n_replications} replications")
    
    # Scenario C: Optimized with P90 <= 15 minutes
    print("\n[C] Optimized (P90 <= 15 min target)")
    best, _ = grid_search_optimize(
        wait_weight=1.0,
        staff_weight=0.5,
        p90_target=15.0,
        simulator_path=simulator_path,
        verbose=False,
        quick_mode=True  # Use fewer replications for grid search
    )
    if best:
        scenarios['optimized'] = best
        s = best
        print(f"    Staffing: {list(s.staffing)}")
        print(f"    Mean wait: {s.mean_wait:.2f} min  (95% CI: {s.mean_wait_ci[0]:.2f}–{s.mean_wait_ci[1]:.2f})")
        print(f"    P90 wait:  {s.p90_wait:.2f} min  (95% CI: {s.p90_wait_ci[0]:.2f}–{s.p90_wait_ci[1]:.2f})")
        print(f"    Staff-hours: {s.total_staff_hours}  |  n={s.n_replications} replications")
    else:
        print("    No feasible solution found")
    
    return scenarios


# ============================================================================
# Decision Support Outputs
# ============================================================================

def generate_recommendation(scenarios: dict) -> str:
    """
    Generate decision recommendation comparing scenarios.
    """
    report = []
    report.append("\n" + "=" * 60)
    report.append("DECISION SUPPORT RECOMMENDATION")
    report.append("=" * 60)
    
    flat = scenarios.get('flat')
    matched = scenarios.get('matched')
    optimized = scenarios.get('optimized')
    
    if flat and matched:
        wait_change = matched.p90_wait - flat.p90_wait
        staff_diff = matched.total_staff_hours - flat.total_staff_hours
        
        report.append(f"\nDemand-matched vs Flat staffing:")
        report.append(f"  - P90 wait change: {wait_change:+.1f} minutes")
        report.append(f"  - Staff-hour change: {staff_diff:+d} hours")
    
    if optimized and flat:
        wait_change = optimized.p90_wait - flat.p90_wait
        staff_diff = optimized.total_staff_hours - flat.total_staff_hours
        
        report.append(f"\nOptimized vs Flat staffing:")
        report.append(f"  - P90 wait change: {wait_change:+.1f} minutes")
        report.append(f"  - Staff-hour change: {staff_diff:+d} hours")
    
    if optimized:
        staffing = list(optimized.staffing)
        peak_slots = [i for i, s in enumerate(staffing) if s == max(staffing)]
        slot_names = ["8-9AM", "9-10AM", "10-11AM", "11AM-12PM", 
                      "12-1PM", "1-2PM", "2-3PM", "3-4PM"]
        peaks = [slot_names[i] for i in peak_slots]
        
        report.append(f"\n>>> RECOMMENDATION:")
        report.append(f"    Adopt optimized staffing schedule: {staffing}")
        report.append(f"    Peak staffing periods: {', '.join(peaks)}")
        report.append(f"    Expected P90 wait: {optimized.p90_wait:.1f} minutes")
        report.append(f"    Total daily staff-hours: {optimized.total_staff_hours}")
    
    return "\n".join(report)


def export_results_csv(results: list[SimulationResult], filename: str):
    """Export all results to CSV for further analysis."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'slot_0', 'slot_1', 'slot_2', 'slot_3',
            'slot_4', 'slot_5', 'slot_6', 'slot_7',
            'total_staff_hours', 'mean_wait', 'p90_wait',
            'avg_served', 'cost_score'
        ])
        for r in results:
            writer.writerow([
                *r.staffing,
                r.total_staff_hours,
                f"{r.mean_wait:.2f}",
                f"{r.p90_wait:.2f}",
                f"{r.avg_served:.1f}",
                f"{r.cost_score:.2f}"
            ])
    print(f"Results exported to {filename}")


def plot_scenarios(scenarios: dict, save_path: Optional[Path] = None, show: bool = True):
    """Visualize scenario comparisons with error bars and staffing heatmap.

    Args:
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

    for name in ["flat", "matched", "optimized"]:
        if name in scenarios:
            s = scenarios[name]
            labels.append(name.capitalize())
            p90.append(s.p90_wait)
            p90_err.append((s.p90_wait_ci[1] - s.p90_wait_ci[0]) / 2)
            mean_wait.append(s.mean_wait)
            mean_err.append((s.mean_wait_ci[1] - s.mean_wait_ci[0]) / 2)
            staff_hours.append(s.total_staff_hours)
            staffing_data.append(list(s.staffing))

    # Create figure with 2 subplots
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
    
    # Add value labels on bars
    for bar, val in zip(bars1, p90):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, mean_wait):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Wait Time (minutes)", fontsize=11)
    ax1.set_title("Service Level Comparison (95% CI)", fontsize=12, fontweight='bold')
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, max(p90) * 1.4)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add staff-hours as text annotation
    for i, sh in enumerate(staff_hours):
        ax1.text(i, -max(p90)*0.12, f'{sh} hrs', ha='center', fontsize=10, 
                color='#e74c3c', fontweight='bold')
    ax1.text(len(labels)/2 - 0.5, -max(p90)*0.2, 'Staff-Hours →', 
            ha='center', fontsize=9, color='#e74c3c')
    
    # ===== Right plot: Staffing heatmap =====
    hours = ['8-9', '9-10', '10-11', '11-12', '12-1', '1-2', '2-3', '3-4']
    staffing_array = list(zip(*staffing_data))  # Transpose
    
    im = ax2.imshow(staffing_array, cmap='YlOrRd', aspect='auto', vmin=1, vmax=5)
    
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
    
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Windows Open', fontsize=10)
    
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
        default=SIMULATOR_PATH,
        help="Path to queue_sim executable"
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
    
    args = parser.parse_args()
    
    # Update simulator path
    sim_path = args.simulator
    if not sim_path.exists():
        # Try alternative paths
        alternatives = [
            Path("cpp/build/queue_sim.exe"),
            Path("cpp/build/Release/queue_sim.exe"),
            Path("cpp/build/queue_sim"),
            Path("build/queue_sim.exe"),
        ]
        for alt in alternatives:
            if alt.exists():
                sim_path = alt
                break
    
    if args.scenario_analysis:
        scenarios = run_scenario_analysis(simulator_path=sim_path)
        print(generate_recommendation(scenarios))
        if args.plot or args.save_plot is not None:
            plot_scenarios(scenarios, save_path=args.save_plot, show=args.plot)
        
    elif args.optimize:
        print("Running grid search optimization...")
        best, all_results = grid_search_optimize(
            simulator_path=sim_path,
            verbose=True
        )
        
        if best:
            print(f"\nBest configuration found:")
            print(f"  Staffing: {list(best.staffing)}")
            print(f"  Mean wait: {best.mean_wait:.2f} minutes")
            print(f"  P90 wait: {best.p90_wait:.2f} minutes")
            print(f"  Staff-hours: {best.total_staff_hours}")
            print(f"  Cost score: {best.cost_score:.2f}")
        
        if args.export:
            export_results_csv(all_results, args.export)
    
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

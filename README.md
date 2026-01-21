# Public-Sector Queue Resource Allocation Simulator

A discrete-event simulation (C++) coupled with parameter optimization (Python) to support staffing decisions at a government service center.

## Overview

This project models a municipal permit and licensing office with multiple service windows, heterogeneous service types, and time-varying citizen demand. It enables decision-makers to balance wait-time targets against labor costs through quantitative analysis.

### Why This Matters
- Citizens experience long, unpredictable wait times during peak periods
- Managers lack quantitative tools to justify staffing requests
- Over-staffing wastes taxpayer funds; under-staffing degrades public trust
- Decision support enables data-driven workforce planning and service-level agreements

## System Model

| Component | Specification |
|-----------|---------------|
| **Entities** | Citizens (arrivals), Service Windows (servers) |
| **Resources** | N service windows, each staffed or unstaffed per time slot |
| **Arrival Process** | Non-homogeneous Poisson process; λ(t) varies by hour |
| **Service Process** | Exponential service times; mean μ = 8 minutes |
| **Queue Discipline** | Single FIFO queue feeding all open windows |
| **Time Horizon** | One 8-hour operating day (480 minutes) |

### Key Assumptions
- No appointments
- No balking/reneging (citizens wait indefinitely)
- All service windows are identical
- Citizens are served to completion

## Mathematical Model

**Decision Variables:** Number of open windows per hourly slot: **s** = (s₁, s₂, ..., s₈) where sᵢ ∈ {1, ..., 5}

**Objective Function:** Minimize weighted cost:
```
Cost = w₁ · W̄ + w₂ · Σsᵢ
```
where W̄ = mean wait time, w₁, w₂ are policy weights.

**Performance Metrics:**
- Mean wait time
- 90th-percentile wait time
- Throughput (citizens served)
- Window utilization per slot

## Project Structure

```
queue-simulator/
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── simulation.hpp
│   └── src/
│       ├── simulation.cpp
│       └── main.cpp
├── python/
│   └── optimizer.py
└── README.md
```

## Building the C++ Simulator

### Requirements
- CMake 3.16+
- C++17 compatible compiler (MSVC, GCC, or Clang)

### Windows (MinGW/g++)
```powershell
cd cpp
mkdir build
cd build
g++ -std=c++17 -O2 -Wall -I../include -o queue_sim.exe ../src/simulation.cpp ../src/main.cpp
```

### Windows (MSVC with CMake)
```powershell
cd cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Linux/macOS
```bash
cd cpp
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

The executable `queue_sim` (or `queue_sim.exe`) will be in the build directory.

## Using the C++ Simulator

```bash
# Basic usage with defaults
./queue_sim

# Custom staffing (8 hourly slots)
./queue_sim --staffing 2,3,3,2,2,3,3,2

# Custom arrival rates (citizens per hour)
./queue_sim --arrivals 6,8,5,4,4,6,7,5

# Multiple replications for statistical validity
./queue_sim --replications 30 --seed 42

# Full options
./queue_sim --help
```

### Output Format
CSV with metrics:
```
metric,value
mean_wait_time,12.34
p90_wait_time,22.50
avg_served,45.0
...
```


## Vis 

<img width="2473" height="1363" alt="image" src="https://github.com/user-attachments/assets/ef4877bd-b010-4188-8908-7c84fa62b085" />


## Python Optimizer

### Requirements
- Python 3.10+
- No external packages required (standard library only)
 - Optional for plotting: matplotlib (`pip install matplotlib`)

### Running Scenario Analysis
```bash
cd python
python optimizer.py --scenario-analysis --simulator ../cpp/build/queue_sim.exe
# With visualization (requires matplotlib)
python optimizer.py --scenario-analysis --plot --simulator ../cpp/build/queue_sim.exe
# Save a portfolio figure without opening a window
python optimizer.py --scenario-analysis --save-plot ../outputs/staffing_analysis.png --simulator ../cpp/build/queue_sim.exe
```
#

### Running Grid Search Optimization
```bash
python optimizer.py --optimize --export results.csv
```

### Programmatic Usage
```python
from optimizer import run_simulation, grid_search_optimize, run_scenario_analysis

# Single simulation
result = run_simulation(staffing=[2, 3, 2, 2, 2, 3, 3, 2])
print(f"Mean wait: {result.mean_wait:.1f} min")

# Full scenario comparison
scenarios = run_scenario_analysis()

# Grid search with constraint
best, all_results = grid_search_optimize(
    wait_weight=1.0,
    staff_weight=0.5,
    p90_target=15.0  # Max 15-minute P90 wait
)
```

## Scenario Analysis

The optimizer compares three staffing policies:

| Scenario | Description |
|----------|-------------|
| **A. Flat** | Uniform staffing (3 windows all slots) |
| **B. Demand-Matched** | Staff proportional to arrival rate |
| **C. Optimized** | Cost-minimized under 15-min P90 constraint |

### Example Output
```
[A] Flat Staffing (3 windows/slot)
    Mean wait: 8.5 min, P90 wait: 18.2 min, Staff-hours: 24

[B] Demand-Matched Staffing
    Mean wait: 5.2 min, P90 wait: 12.1 min, Staff-hours: 26

[C] Optimized (P90 <= 15 min target)
    Mean wait: 6.1 min, P90 wait: 14.8 min, Staff-hours: 22

>>> RECOMMENDATION:
    Shifting two staff-hours from midday to 8-10 AM reduces 
    90th-percentile wait from 22 to 14 minutes with no 
    change in labor cost.
```

## Technical Details

### Simulation Engine (C++)
- **Type:** Discrete-Event Simulation (DES)
- **Event Queue:** Priority queue (min-heap by time)
- **Arrival Generation:** Thinning algorithm for non-homogeneous Poisson
- **Service Times:** Inverse-transform sampling from exponential distribution
- **Reproducibility:** Deterministic given seed

### Optimization (Python)
- **Method:** Exhaustive grid search
- **Constraint Space:** ~2,000 feasible configurations
- **Statistical Handling:** 30 replications per configuration, mean ± 95% CI
- **Variance Reduction:** Common random numbers via sequential seeding

## Scope Boundaries

**Intentionally Excluded:**
- Appointment scheduling
- Multiple service types / skill-based routing
- Balking / reneging behavior
- External datasets
- GUI / web dashboard
- Metaheuristics (GA, SA)

**Constraints:**
- Grid search only (no external solvers)

## License

Public domain - developed for government R&D demonstration purposes.

## References

- Banks, J., Carson, J. S., Nelson, B. L., & Nicol, D. M. (2014). *Discrete-Event System Simulation*. Pearson.
- Law, A. M. (2015). *Simulation Modeling and Analysis*. McGraw-Hill.



/**
 * @file simulation.cpp
 * @brief Implementation of Discrete-Event Simulation Engine
 * 
 * Public-Sector Queue Resource Allocation Simulator
 * Municipal permit/licensing office staffing optimization tool.
 */

#include "simulation.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace govqueue {

// ============================================================================
// QueueSimulator Implementation
// ============================================================================

QueueSimulator::QueueSimulator(const SimulationConfig& config)
    : config_(config)
    , rng_(config.random_seed)
    , service_dist_(1.0 / config.mean_service_time)
    , uniform_dist_(0.0, 1.0)
    , current_time_(0.0)
    , next_citizen_id_(0)
{
    // Validate configuration
    if (config_.staffing_per_slot.size() != 8) {
        config_.staffing_per_slot.resize(8, 2);  // Default: 2 windows per slot
    }
    if (config_.arrival_rates.size() != 8) {
        // Default: morning peak, midday lull, afternoon peak (citizens per hour)
        config_.arrival_rates = {12.0, 15.0, 10.0, 8.0, 8.0, 12.0, 14.0, 10.0};
    }
    
    reset();
}

void QueueSimulator::reset() {
    current_time_ = 0.0;
    next_citizen_id_ = 0;
    
    // Clear event queue
    while (!event_queue_.empty()) {
        event_queue_.pop();
    }
    
    // Clear waiting queue
    while (!waiting_queue_.empty()) {
        waiting_queue_.pop();
    }
    
    // Initialize windows (max possible needed)
    int max_windows = *std::max_element(
        config_.staffing_per_slot.begin(), 
        config_.staffing_per_slot.end()
    );
    windows_.clear();
    windows_.reserve(max_windows);
    for (int i = 0; i < max_windows; ++i) {
        windows_.push_back({i, false, -1});
    }
    
    citizens_.clear();
    slot_busy_time_.assign(8, 0.0);
    
    // Reseed RNG
    rng_.seed(config_.random_seed);
    
    // Schedule first arrival
    double first_arrival = generate_next_arrival_time();
    event_queue_.push({first_arrival, EventType::ARRIVAL, -1, next_citizen_id_++});
}

double QueueSimulator::get_arrival_rate(double time) const {
    int slot = get_current_slot(time);
    // Convert from arrivals per hour to arrivals per minute
    return config_.arrival_rates[slot] / 60.0;
}

int QueueSimulator::get_current_slot(double time) const {
    int slot = static_cast<int>(time / 60.0);
    return std::min(slot, 7);  // Clamp to last slot
}

int QueueSimulator::get_open_windows(double time) const {
    int slot = get_current_slot(time);
    return config_.staffing_per_slot[slot];
}

double QueueSimulator::generate_next_arrival_time() {
    // Thinning algorithm for non-homogeneous Poisson process
    // Arrival rates are specified in citizens/hour; convert to citizens/min
    double lambda_max_per_min = (*std::max_element(
        config_.arrival_rates.begin(),
        config_.arrival_rates.end()
    )) / 60.0;
    
    double t = current_time_;
    while (t < config_.simulation_duration) {
        // Generate candidate inter-arrival time using max rate
        double u1 = uniform_dist_(rng_);
        t += -std::log(u1) / lambda_max_per_min;
        
        if (t >= config_.simulation_duration) {
            return config_.simulation_duration + 1.0;  // No more arrivals
        }
        
        // Accept/reject based on actual rate at time t
        double u2 = uniform_dist_(rng_);
        double lambda_t = get_arrival_rate(t);
        if (u2 <= lambda_t / lambda_max_per_min) {
            return t;
        }
    }
    return config_.simulation_duration + 1.0;
}

double QueueSimulator::generate_service_time() {
    return service_dist_(rng_);
}

int QueueSimulator::find_free_window() {
    int open_windows = get_open_windows(current_time_);
    for (int i = 0; i < open_windows && i < static_cast<int>(windows_.size()); ++i) {
        if (!windows_[i].is_busy) {
            return i;
        }
    }
    return -1;  // No free window
}

void QueueSimulator::start_service(int citizen_id, int window_id) {
    windows_[window_id].is_busy = true;
    windows_[window_id].current_citizen_id = citizen_id;
    
    citizens_[citizen_id].service_start_time = current_time_;
    
    double service_time = generate_service_time();
    double departure_time = current_time_ + service_time;
    
    event_queue_.push({departure_time, EventType::DEPARTURE, window_id, citizen_id});
}

void QueueSimulator::process_arrival(const Event& event) {
    // Create citizen record
    Citizen citizen;
    citizen.id = event.citizen_id;
    citizen.arrival_time = current_time_;
    citizen.service_start_time = -1.0;
    citizen.departure_time = -1.0;
    citizens_.push_back(citizen);
    
    // Try to find a free window
    int free_window = find_free_window();
    if (free_window >= 0) {
        start_service(event.citizen_id, free_window);
    } else {
        waiting_queue_.push(event.citizen_id);
    }
    
    // Schedule next arrival
    double next_arrival = generate_next_arrival_time();
    if (next_arrival <= config_.simulation_duration) {
        event_queue_.push({next_arrival, EventType::ARRIVAL, -1, next_citizen_id_++});
    }
}

void QueueSimulator::process_departure(const Event& event) {
    int window_id = event.window_id;
    int citizen_id = event.citizen_id;
    
    // Record departure
    citizens_[citizen_id].departure_time = current_time_;
    
    // Track utilization (busy time since service started)
    double service_duration = current_time_ - citizens_[citizen_id].service_start_time;
    int slot = get_current_slot(citizens_[citizen_id].service_start_time);
    slot_busy_time_[slot] += service_duration;
    
    // Free the window
    windows_[window_id].is_busy = false;
    windows_[window_id].current_citizen_id = -1;
    
    // Check if window is still open (staffing may have changed)
    int open_windows = get_open_windows(current_time_);
    if (window_id >= open_windows) {
        return;  // Window no longer staffed
    }
    
    // Serve next citizen if queue not empty
    if (!waiting_queue_.empty()) {
        int next_citizen = waiting_queue_.front();
        waiting_queue_.pop();
        start_service(next_citizen, window_id);
    }
}

SimulationResults QueueSimulator::run() {
    reset();
    
    while (!event_queue_.empty()) {
        Event event = event_queue_.top();
        event_queue_.pop();
        
        if (event.time > config_.simulation_duration) {
            break;
        }
        
        current_time_ = event.time;
        
        switch (event.type) {
            case EventType::ARRIVAL:
                process_arrival(event);
                break;
            case EventType::DEPARTURE:
                process_departure(event);
                break;
        }
    }
    
    return compute_results();
}

SimulationResults QueueSimulator::compute_results() const {
    SimulationResults results;
    results.total_arrived = static_cast<int>(citizens_.size());
    results.total_served = 0;
    
    std::vector<double> wait_times;
    std::vector<double> service_times;
    
    for (const auto& citizen : citizens_) {
        if (citizen.departure_time > 0) {
            results.total_served++;
            double wait = citizen.service_start_time - citizen.arrival_time;
            double service = citizen.departure_time - citizen.service_start_time;
            wait_times.push_back(wait);
            service_times.push_back(service);
        }
    }
    
    results.all_wait_times = wait_times;
    
    // Compute mean wait time
    if (!wait_times.empty()) {
        results.mean_wait_time = std::accumulate(
            wait_times.begin(), wait_times.end(), 0.0
        ) / wait_times.size();
        
        // Compute 90th percentile
        std::vector<double> sorted_waits = wait_times;
        std::sort(sorted_waits.begin(), sorted_waits.end());
        size_t p90_idx = static_cast<size_t>(0.9 * sorted_waits.size());
        results.p90_wait_time = sorted_waits[std::min(p90_idx, sorted_waits.size() - 1)];
    } else {
        results.mean_wait_time = 0.0;
        results.p90_wait_time = 0.0;
    }
    
    // Compute mean service time
    if (!service_times.empty()) {
        results.mean_service_time = std::accumulate(
            service_times.begin(), service_times.end(), 0.0
        ) / service_times.size();
    } else {
        results.mean_service_time = 0.0;
    }
    
    // Compute utilization per slot
    results.utilization_per_slot.resize(8);
    for (int slot = 0; slot < 8; ++slot) {
        double available_window_minutes = config_.staffing_per_slot[slot] * 60.0;
        if (available_window_minutes > 0) {
            results.utilization_per_slot[slot] = 
                slot_busy_time_[slot] / available_window_minutes;
        } else {
            results.utilization_per_slot[slot] = 0.0;
        }
    }
    
    return results;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::vector<SimulationResults> run_replications(
    const SimulationConfig& config,
    int num_replications,
    int base_seed
) {
    std::vector<SimulationResults> all_results;
    all_results.reserve(num_replications);
    
    SimulationConfig rep_config = config;
    for (int i = 0; i < num_replications; ++i) {
        rep_config.random_seed = base_seed + i;
        QueueSimulator sim(rep_config);
        all_results.push_back(sim.run());
    }
    
    return all_results;
}

std::string results_to_csv(const SimulationResults& results) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    
    oss << "metric,value\n";
    oss << "mean_wait_time," << results.mean_wait_time << "\n";
    oss << "p90_wait_time," << results.p90_wait_time << "\n";
    oss << "mean_service_time," << results.mean_service_time << "\n";
    oss << "total_served," << results.total_served << "\n";
    oss << "total_arrived," << results.total_arrived << "\n";
    
    for (size_t i = 0; i < results.utilization_per_slot.size(); ++i) {
        oss << "utilization_slot_" << i << "," 
            << results.utilization_per_slot[i] << "\n";
    }
    
    return oss.str();
}

} // namespace govqueue

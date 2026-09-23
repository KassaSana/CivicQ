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
    , service_dist_(1.0 / config.mean_service_time)
    , uniform_dist_(0.0, 1.0)
    , current_time_(0.0)
    , last_departure_time_(0.0)
    , next_citizen_id_(0)
{
    // Validate configuration
    if (config_.staffing_per_slot.size() != NUM_SLOTS) {
        config_.staffing_per_slot.resize(NUM_SLOTS, 2);  // Default: 2 windows per slot
    }
    if (config_.arrival_rates.size() != NUM_SLOTS) {
        // Default: morning peak, midday lull, afternoon peak (citizens per hour)
        config_.arrival_rates = {12.0, 15.0, 10.0, 8.0, 8.0, 12.0, 14.0, 10.0};
    }

    reset();
}

void QueueSimulator::reset() {
    current_time_ = 0.0;
    last_departure_time_ = 0.0;
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
    slot_busy_time_.assign(NUM_SLOTS, 0.0);

    // Reseed independent streams (common random numbers across staffing plans)
    std::seed_seq arrival_seed{static_cast<unsigned>(config_.random_seed), 1u};
    std::seed_seq service_seed{static_cast<unsigned>(config_.random_seed), 2u};
    arrival_rng_.seed(arrival_seed);
    service_rng_.seed(service_seed);
    service_dist_.reset();
    uniform_dist_.reset();

    // Slot boundaries: let newly opened windows serve queued citizens
    for (int slot = 1; slot < NUM_SLOTS; ++slot) {
        double t = slot * SLOT_LENGTH;
        if (t < config_.simulation_duration) {
            event_queue_.push({t, EventType::STAFFING_CHANGE, -1, -1});
        }
    }

    // Schedule first arrival
    double first_arrival = generate_next_arrival_time();
    if (first_arrival <= config_.simulation_duration) {
        event_queue_.push({first_arrival, EventType::ARRIVAL, -1, next_citizen_id_++});
    }
}

double QueueSimulator::get_arrival_rate(double time) const {
    int slot = get_current_slot(time);
    // Convert from arrivals per hour to arrivals per minute
    return config_.arrival_rates[slot] / 60.0;
}

int QueueSimulator::get_current_slot(double time) const {
    int slot = static_cast<int>(time / SLOT_LENGTH);
    return std::min(slot, NUM_SLOTS - 1);  // Clamp to last slot (also covers after-close)
}

int QueueSimulator::get_open_windows(double time) const {
    int slot = get_current_slot(time);
    return config_.staffing_per_slot[slot];
}

double QueueSimulator::slot_length(int slot) const {
    // Last slot runs until the doors close (differs from 60 only for custom durations)
    double start = slot * SLOT_LENGTH;
    double end = (slot == NUM_SLOTS - 1) ? config_.simulation_duration
                                         : std::min(start + SLOT_LENGTH, config_.simulation_duration);
    return std::max(0.0, end - start);
}

double QueueSimulator::generate_next_arrival_time() {
    // Thinning algorithm for non-homogeneous Poisson process
    // Arrival rates are specified in citizens/hour; convert to citizens/min
    double lambda_max_per_min = (*std::max_element(
        config_.arrival_rates.begin(),
        config_.arrival_rates.end()
    )) / 60.0;
    if (lambda_max_per_min <= 0.0) {
        return config_.simulation_duration + 1.0;  // No demand at all
    }

    double t = current_time_;
    while (t < config_.simulation_duration) {
        // Generate candidate inter-arrival time using max rate.
        // 1 - U lies in (0, 1], so the log is always finite.
        double u1 = 1.0 - uniform_dist_(arrival_rng_);
        t += -std::log(u1) / lambda_max_per_min;

        if (t >= config_.simulation_duration) {
            return config_.simulation_duration + 1.0;  // No more arrivals
        }

        // Accept/reject based on actual rate at time t
        double u2 = uniform_dist_(arrival_rng_);
        double lambda_t = get_arrival_rate(t);
        if (u2 <= lambda_t / lambda_max_per_min) {
            return t;
        }
    }
    return config_.simulation_duration + 1.0;
}

double QueueSimulator::generate_service_time() {
    return service_dist_(service_rng_);
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

    double departure_time = current_time_ + citizens_[citizen_id].service_time;

    event_queue_.push({departure_time, EventType::DEPARTURE, window_id, citizen_id});
}

void QueueSimulator::serve_waiting_citizens() {
    // FIFO: fill every free open window from the front of the queue
    while (!waiting_queue_.empty()) {
        int free_window = find_free_window();
        if (free_window < 0) {
            break;
        }
        int next_citizen = waiting_queue_.front();
        waiting_queue_.pop();
        start_service(next_citizen, free_window);
    }
}

void QueueSimulator::add_busy_time(double start, double end) {
    // Split a service interval across the slots it spans.
    // Time after closing counts as overtime, not slot utilization.
    end = std::min(end, config_.simulation_duration);
    for (int slot = get_current_slot(start); slot < NUM_SLOTS && start < end; ++slot) {
        double slot_end = (slot == NUM_SLOTS - 1) ? end
                                                  : std::min(end, (slot + 1) * SLOT_LENGTH);
        if (slot_end > start) {
            slot_busy_time_[slot] += slot_end - start;
            start = slot_end;
        }
    }
}

void QueueSimulator::process_arrival(const Event& event) {
    // Create citizen record; service requirement is drawn now so that
    // citizen k needs the same work under every staffing plan
    Citizen citizen;
    citizen.id = event.citizen_id;
    citizen.arrival_time = current_time_;
    citizen.service_time = generate_service_time();
    citizen.service_start_time = -1.0;
    citizen.departure_time = -1.0;
    citizens_.push_back(citizen);

    // Join the back of the queue, then serve in FIFO order
    waiting_queue_.push(event.citizen_id);
    serve_waiting_citizens();

    // Schedule next arrival (doors close at simulation_duration)
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
    last_departure_time_ = std::max(last_departure_time_, current_time_);

    // Track utilization, split across the slots the service spanned
    add_busy_time(citizens_[citizen_id].service_start_time, current_time_);

    // Free the window
    windows_[window_id].is_busy = false;
    windows_[window_id].current_citizen_id = -1;

    // Serve next citizen if queue not empty. find_free_window only returns
    // open windows, so a window closed at a slot boundary stays idle.
    serve_waiting_citizens();
}

void QueueSimulator::process_staffing_change() {
    // Windows opened at this boundary start serving queued citizens immediately
    serve_waiting_citizens();
}

SimulationResults QueueSimulator::run() {
    reset();

    // Run until every citizen who arrived before closing has been served
    while (!event_queue_.empty()) {
        Event event = event_queue_.top();
        event_queue_.pop();

        current_time_ = event.time;

        switch (event.type) {
            case EventType::ARRIVAL:
                process_arrival(event);
                break;
            case EventType::DEPARTURE:
                process_departure(event);
                break;
            case EventType::STAFFING_CHANGE:
                process_staffing_change();
                break;
        }
    }

    return compute_results();
}

SimulationResults QueueSimulator::compute_results() const {
    SimulationResults results;
    results.total_arrived = static_cast<int>(citizens_.size());
    results.total_served = 0;
    results.overtime_minutes = std::max(0.0, last_departure_time_ - config_.simulation_duration);

    std::vector<double> wait_times;
    std::vector<double> service_times;

    for (const auto& citizen : citizens_) {
        if (citizen.departure_time >= 0) {
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

        // Compute 90th percentile (nearest-rank method)
        std::vector<double> sorted_waits = wait_times;
        std::sort(sorted_waits.begin(), sorted_waits.end());
        size_t rank = static_cast<size_t>(std::ceil(0.9 * sorted_waits.size()));
        results.p90_wait_time = sorted_waits[std::max<size_t>(rank, 1) - 1];
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

    // Compute utilization per slot. A window closing at a slot boundary finishes
    // its current citizen, so a slot after a staffing cut can slightly exceed 1.
    results.utilization_per_slot.resize(NUM_SLOTS);
    for (int slot = 0; slot < NUM_SLOTS; ++slot) {
        double available_window_minutes = config_.staffing_per_slot[slot] * slot_length(slot);
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
    oss << "overtime_minutes," << results.overtime_minutes << "\n";

    for (size_t i = 0; i < results.utilization_per_slot.size(); ++i) {
        oss << "utilization_slot_" << i << ","
            << results.utilization_per_slot[i] << "\n";
    }

    return oss.str();
}

} // namespace govqueue

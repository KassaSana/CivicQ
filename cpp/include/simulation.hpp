/**
 * @file simulation.hpp
 * @brief Discrete-Event Simulation Engine for Government Service Center Queue
 * 
 * Public-Sector Queue Resource Allocation Simulator
 * Supports staffing optimization for municipal permit/licensing offices.
 */

#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <vector>
#include <queue>
#include <random>
#include <string>
#include <functional>

namespace govqueue {

/**
 * @brief Event types in the discrete-event simulation
 */
enum class EventType {
    ARRIVAL,
    DEPARTURE
};

/**
 * @brief Simulation event structure
 */
struct Event {
    double time;           // Event timestamp (minutes from start)
    EventType type;        // ARRIVAL or DEPARTURE
    int window_id;         // Service window ID (-1 for arrivals)
    int citizen_id;        // Unique citizen identifier

    // Min-heap comparison (earliest event first)
    bool operator>(const Event& other) const {
        return time > other.time;
    }
};

/**
 * @brief Citizen entity tracking wait and service times
 */
struct Citizen {
    int id;
    double arrival_time;
    double service_start_time;
    double departure_time;
};

/**
 * @brief Service window resource
 */
struct ServiceWindow {
    int id;
    bool is_busy;
    int current_citizen_id;
};

/**
 * @brief Simulation output statistics
 */
struct SimulationResults {
    double mean_wait_time;
    double p90_wait_time;
    double mean_service_time;
    int total_served;
    int total_arrived;
    std::vector<double> utilization_per_slot;  // 8 hourly slots
    std::vector<double> all_wait_times;        // For distribution analysis
};

/**
 * @brief Configuration for a simulation run
 */
struct SimulationConfig {
    std::vector<int> staffing_per_slot;   // Windows open per hour (8 slots)
    std::vector<double> arrival_rates;     // Lambda(t) per hour (8 slots)
    double mean_service_time;              // 1/mu in minutes
    int random_seed;
    double simulation_duration;            // Total minutes (default 480)

    SimulationConfig() 
        : mean_service_time(8.0)
        , random_seed(42)
        , simulation_duration(480.0) {}
};

/**
 * @brief Discrete-Event Simulation Engine
 * 
 * Implements a single-queue, multi-server model with:
 * - Non-homogeneous Poisson arrivals (time-varying lambda)
 * - Exponential service times
 * - FIFO queue discipline
 */
class QueueSimulator {
public:
    explicit QueueSimulator(const SimulationConfig& config);
    
    /**
     * @brief Execute one complete simulation run
     * @return Aggregated performance metrics
     */
    SimulationResults run();

    /**
     * @brief Reset simulator state for a new run
     */
    void reset();

private:
    SimulationConfig config_;
    
    // Random number generation
    std::mt19937 rng_;
    std::exponential_distribution<double> service_dist_;
    std::uniform_real_distribution<double> uniform_dist_;

    // Simulation state
    double current_time_;
    int next_citizen_id_;
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> event_queue_;
    std::queue<int> waiting_queue_;  // Citizen IDs waiting for service
    std::vector<ServiceWindow> windows_;
    std::vector<Citizen> citizens_;

    // Statistics tracking
    std::vector<double> slot_busy_time_;  // Cumulative busy time per slot

    // Helper methods
    double get_arrival_rate(double time) const;
    int get_current_slot(double time) const;
    int get_open_windows(double time) const;
    double generate_next_arrival_time();
    double generate_service_time();
    
    void process_arrival(const Event& event);
    void process_departure(const Event& event);
    void start_service(int citizen_id, int window_id);
    int find_free_window();
    
    SimulationResults compute_results() const;
};

/**
 * @brief Run multiple replications and aggregate statistics
 * 
 * @param config Base configuration
 * @param num_replications Number of independent runs
 * @param base_seed Starting seed (incremented per replication)
 * @return Vector of results from each replication
 */
std::vector<SimulationResults> run_replications(
    const SimulationConfig& config,
    int num_replications,
    int base_seed = 42
);

/**
 * @brief Export results to CSV format string
 */
std::string results_to_csv(const SimulationResults& results);

} // namespace govqueue

#endif // SIMULATION_HPP

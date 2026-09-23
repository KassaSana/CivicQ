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

constexpr int NUM_SLOTS = 8;             // Hourly staffing slots
constexpr double SLOT_LENGTH = 60.0;     // Minutes per slot

/**
 * @brief Event types in the discrete-event simulation
 */
enum class EventType {
    ARRIVAL,
    DEPARTURE,
    STAFFING_CHANGE,   // Slot boundary: newly opened windows pull from the queue
    APPOINTMENT        // A booked citizen arrives (does not schedule further arrivals)
};

/**
 * @brief Service-time distribution family (all parameterized by their mean)
 */
enum class ServiceDist {
    EXPONENTIAL,     // CV = 1 (M/M/c)
    LOGNORMAL,       // CV set by service_cv; empirically realistic (Brown et al. 2005)
    DETERMINISTIC    // CV = 0
};

/**
 * @brief Simulation event structure
 */
struct Event {
    double time;           // Event timestamp (minutes from start)
    EventType type;        // ARRIVAL, DEPARTURE or STAFFING_CHANGE
    int window_id;         // Service window ID (-1 if not applicable)
    int citizen_id;        // Unique citizen identifier (-1 if not applicable)

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
    double service_time;         // Drawn at arrival (common random numbers)
    bool is_appointment;         // Booked arrival rather than walk-in
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
    double overtime_minutes;                   // Minutes past closing to serve everyone inside
    double rate_multiplier;                    // Day-level demand multiplier (1 unless rate_cv > 0)
    std::vector<int> arrivals_per_slot;        // Citizens arriving in each hourly slot
    std::vector<int> late_per_slot;            // Of those, how many waited > wait_threshold
    int appointments_arrived;                  // Booked citizens who showed up
    int appointments_late;                     // Of those, how many waited > wait_threshold
    double appointment_wait_sum;               // Total wait of booked citizens (minutes)
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
    double simulation_duration;            // Doors close at this time (default 480)
    ServiceDist service_dist;              // Service-time distribution family
    double service_cv;                     // Coefficient of variation (lognormal only)
    double rate_cv;                        // CV of the day-level demand multiplier (0 = pure NHPP)
    double wait_threshold;                 // Minutes; waits above this count as late
    std::vector<double> appointment_times; // Booked arrival times (minutes from opening)
    double no_show;                        // Probability a booked citizen does not come
    double punctuality_sd;                 // SD (minutes) of arrival around the booked time

    SimulationConfig()
        : mean_service_time(8.0)
        , random_seed(42)
        , simulation_duration(480.0)
        , service_dist(ServiceDist::EXPONENTIAL)
        , service_cv(1.0)
        , rate_cv(0.0)
        , wait_threshold(15.0)
        , no_show(0.0)
        , punctuality_sd(0.0) {}
};

/**
 * @brief Discrete-Event Simulation Engine
 *
 * Implements a single-queue, multi-server model with:
 * - Non-homogeneous Poisson arrivals (time-varying lambda)
 * - Exponential service times
 * - FIFO queue discipline
 * - Doors close at simulation_duration; citizens already inside are served
 * - Separate random streams for arrivals and service (common random numbers)
 * - Optional lognormal/deterministic service and a random day-level demand
 *   multiplier (a gamma-mixed Poisson process, which is overdispersed)
 * - Optional appointments: booked times with no-shows and punctuality noise,
 *   served FIFO alongside walk-ins
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

    // Random number generation: independent streams so that every staffing
    // plan sees the same citizens with the same service requirements
    std::mt19937 arrival_rng_;
    std::mt19937 service_rng_;
    std::mt19937 rate_rng_;
    std::mt19937 appointment_rng_;
    std::exponential_distribution<double> service_dist_;
    std::lognormal_distribution<double> lognormal_dist_;
    std::uniform_real_distribution<double> uniform_dist_;
    double rate_multiplier_;   // Drawn once per run from rate_rng_

    // Simulation state
    double current_time_;
    double last_departure_time_;
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
    double slot_length(int slot) const;
    double generate_next_arrival_time();
    double generate_service_time();

    void process_arrival(const Event& event);
    void admit_citizen(bool is_appointment);
    void process_departure(const Event& event);
    void process_staffing_change();
    void start_service(int citizen_id, int window_id);
    void serve_waiting_citizens();
    void add_busy_time(double start, double end);
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

/**
 * @file main.cpp
 * @brief Command-line interface for queue simulation
 *
 * Accepts configuration via command-line arguments and
 * outputs CSV results for Python consumption.
 */

#include "simulation.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

using namespace govqueue;

void print_usage() {
    std::cerr << "Usage: queue_sim [options]\n"
              << "Options:\n"
              << "  --staffing s1,s2,...,s8    Windows per hourly slot (8 values)\n"
              << "  --arrivals a1,a2,...,a8    Arrival rates per hour (8 values)\n"
              << "  --service-time MINUTES     Mean service time (default: 8.0)\n"
              << "  --seed SEED                Random seed (default: 42)\n"
              << "  --replications N           Number of runs (default: 1)\n"
              << "  --duration MINUTES         Doors close at this time (default: 480)\n"
              << "  --service-dist NAME        exp | lognormal | det (default: exp)\n"
              << "  --service-cv CV            Service-time CV for lognormal (default: 1.0)\n"
              << "  --rate-cv CV               CV of a random daily demand multiplier (default: 0)\n"
              << "  --wait-threshold MINUTES   Late-wait threshold for per-hour counts (default: 15)\n"
              << "  --appointments t1,t2,...   Booked arrival times in minutes from opening\n"
              << "  --no-show P                Probability a booked citizen does not come (default: 0)\n"
              << "  --punctuality-sd MINUTES   SD of arrival around the booked time (default: 0)\n"
              << "  --abandonment MODE         none | renege (hidden queue) | balk (visible queue)\n"
              << "  --patience MINUTES         Mean walk-in patience (default: 30)\n"
              << "  --patience-dist NAME       exp | lognormal | det (default: exp)\n"
              << "  --patience-cv CV           Patience CV for lognormal (default: 1.0)\n"
              << "  --per-replication          One CSV row per replication instead of averages\n"
              << "  --citizen-log PATH         Also write one CSV row per citizen to PATH\n"
              << "  --announce NAME            renege mode: none | tickets | count | les (default: none)\n"
              << "  --commit                   renege mode: citizens who join never leave\n"
              << "  --output-waits             Include all wait times in output\n"
              << "  --help                     Show this help\n";
}

bool parse_dist(const std::string& name, ServiceDist& out) {
    if (name == "exp") {
        out = ServiceDist::EXPONENTIAL;
    } else if (name == "lognormal") {
        out = ServiceDist::LOGNORMAL;
    } else if (name == "det") {
        out = ServiceDist::DETERMINISTIC;
    } else {
        std::cerr << "Error: unknown distribution '" << name << "'\n";
        return false;
    }
    return true;
}

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stoi(item));
    }
    return result;
}

std::vector<double> parse_double_list(const std::string& s) {
    std::vector<double> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stod(item));
    }
    return result;
}

int main(int argc, char* argv[]) {
    SimulationConfig config;
    int replications = 1;
    bool output_waits = false;
    bool per_replication = false;
    std::string citizen_log_path;

    // Default arrival rates: morning peak, midday lull, afternoon peak
    config.arrival_rates = {12.0, 15.0, 10.0, 8.0, 8.0, 12.0, 14.0, 10.0};
    config.staffing_per_slot = {2, 3, 2, 2, 2, 3, 3, 2};
    config.mean_service_time = 8.0;
    config.random_seed = 42;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        else if (arg == "--staffing" && i + 1 < argc) {
            config.staffing_per_slot = parse_int_list(argv[++i]);
        }
        else if (arg == "--arrivals" && i + 1 < argc) {
            config.arrival_rates = parse_double_list(argv[++i]);
        }
        else if (arg == "--service-time" && i + 1 < argc) {
            config.mean_service_time = std::stod(argv[++i]);
        }
        else if (arg == "--seed" && i + 1 < argc) {
            config.random_seed = std::stoi(argv[++i]);
        }
        else if (arg == "--replications" && i + 1 < argc) {
            replications = std::stoi(argv[++i]);
        }
        else if (arg == "--duration" && i + 1 < argc) {
            config.simulation_duration = std::stod(argv[++i]);
        }
        else if (arg == "--service-dist" && i + 1 < argc) {
            if (!parse_dist(argv[++i], config.service_dist)) {
                return 1;
            }
        }
        else if (arg == "--service-cv" && i + 1 < argc) {
            config.service_cv = std::stod(argv[++i]);
        }
        else if (arg == "--rate-cv" && i + 1 < argc) {
            config.rate_cv = std::stod(argv[++i]);
        }
        else if (arg == "--wait-threshold" && i + 1 < argc) {
            config.wait_threshold = std::stod(argv[++i]);
        }
        else if (arg == "--appointments" && i + 1 < argc) {
            config.appointment_times = parse_double_list(argv[++i]);
        }
        else if (arg == "--no-show" && i + 1 < argc) {
            config.no_show = std::stod(argv[++i]);
        }
        else if (arg == "--punctuality-sd" && i + 1 < argc) {
            config.punctuality_sd = std::stod(argv[++i]);
        }
        else if (arg == "--abandonment" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "none") {
                config.abandonment = Abandonment::NONE;
            } else if (mode == "renege") {
                config.abandonment = Abandonment::RENEGE;
            } else if (mode == "balk") {
                config.abandonment = Abandonment::BALK;
            } else {
                std::cerr << "Error: unknown abandonment mode '" << mode << "'\n";
                return 1;
            }
        }
        else if (arg == "--patience" && i + 1 < argc) {
            config.mean_patience = std::stod(argv[++i]);
        }
        else if (arg == "--patience-dist" && i + 1 < argc) {
            if (!parse_dist(argv[++i], config.patience_dist)) {
                return 1;
            }
        }
        else if (arg == "--patience-cv" && i + 1 < argc) {
            config.patience_cv = std::stod(argv[++i]);
        }
        else if (arg == "--per-replication") {
            per_replication = true;
        }
        else if (arg == "--announce" && i + 1 < argc) {
            std::string name = argv[++i];
            if (name == "none") config.announce = Announce::NONE;
            else if (name == "tickets") config.announce = Announce::TICKETS;
            else if (name == "count") config.announce = Announce::COUNT;
            else if (name == "les") config.announce = Announce::LES;
            else {
                std::cerr << "Error: unknown announcement '" << name << "'\n";
                return 1;
            }
        }
        else if (arg == "--commit") {
            config.commit = true;
        }
        else if (arg == "--citizen-log" && i + 1 < argc) {
            citizen_log_path = argv[++i];
            config.log_citizens = true;
        }
        else if (arg == "--output-waits") {
            output_waits = true;
        }
    }

    // Validate configuration
    if (config.staffing_per_slot.size() != 8) {
        std::cerr << "Error: staffing must have exactly 8 values\n";
        return 1;
    }
    if (config.arrival_rates.size() != 8) {
        std::cerr << "Error: arrivals must have exactly 8 values\n";
        return 1;
    }
    for (int s : config.staffing_per_slot) {
        if (s < 1) {
            std::cerr << "Error: every slot needs at least 1 open window\n";
            return 1;
        }
    }
    if (config.service_cv <= 0.0 || config.rate_cv < 0.0 || config.mean_service_time <= 0.0) {
        std::cerr << "Error: service time and CVs must be positive\n";
        return 1;
    }
    if (config.no_show < 0.0 || config.no_show >= 1.0 || config.punctuality_sd < 0.0) {
        std::cerr << "Error: no-show must be in [0, 1) and punctuality SD non-negative\n";
        return 1;
    }
    if (config.mean_patience <= 0.0 || config.patience_cv <= 0.0) {
        std::cerr << "Error: patience mean and CV must be positive\n";
        return 1;
    }
    if (replications < 1) {
        std::cerr << "Error: replications must be at least 1\n";
        return 1;
    }

    // Run simulation(s)
    auto results = run_replications(config, replications, config.random_seed);

    // Per-citizen log: what a ticket system records (call time, present or
    // not) plus the true patience, which a real office never sees
    if (!citizen_log_path.empty()) {
        std::ofstream log(citizen_log_path);
        if (!log) {
            std::cerr << "Error: cannot write " << citizen_log_path << "\n";
            return 1;
        }
        log << std::setprecision(10);
        log << "rep,arrival,booked,patience,outcome,call_time,leave_time,queue_ahead,open_windows,"
               "est_tickets,est_count,est_les\n";
        for (size_t r = 0; r < results.size(); ++r) {
            for (const auto& c : results[r].citizen_log) {
                double leave = c.abandoned ? c.abandon_time : c.departure_time;
                log << r << "," << c.arrival_time << "," << (c.is_appointment ? 1 : 0) << ","
                    // outcome: 0 served, 1 reneged, 2 balked (left on arrival)
                    << c.patience << "," << (!c.abandoned ? 0 : c.balked ? 2 : 1) << ","
                    << c.call_time << "," << leave << "," << c.queue_ahead << ","
                    << c.open_at_arrival << "," << c.est_tickets << "," << c.est_count << ","
                    << c.est_les << "\n";
            }
        }
    }

    // One row per replication (lets callers compute confidence intervals
    // without launching a process per replication)
    if (per_replication) {
        std::cout << "rep,mean_wait,p90_wait,served,arrived,overtime";
        for (int j = 0; j < 8; ++j) {
            std::cout << ",util_" << j;
        }
        // Appended columns (existing consumers read columns by name)
        std::cout << ",mean_service,rate_multiplier";
        for (int j = 0; j < 8; ++j) {
            std::cout << ",arr_" << j;
        }
        for (int j = 0; j < 8; ++j) {
            std::cout << ",late_" << j;
        }
        std::cout << ",appt_arrived,appt_late,appt_wait_sum";
        for (int j = 0; j < 8; ++j) {
            std::cout << ",aband_" << j;
        }
        std::cout << ",aband_wait_sum,spill_busy,overtime_busy,balked";
        std::cout << "\n";
        for (size_t r = 0; r < results.size(); ++r) {
            const auto& res = results[r];
            std::cout << r << "," << res.mean_wait_time << "," << res.p90_wait_time << ","
                      << res.total_served << "," << res.total_arrived << ","
                      << res.overtime_minutes;
            for (int j = 0; j < 8; ++j) {
                std::cout << "," << res.utilization_per_slot[j];
            }
            std::cout << "," << res.mean_service_time << "," << res.rate_multiplier;
            for (int j = 0; j < 8; ++j) {
                std::cout << "," << res.arrivals_per_slot[j];
            }
            for (int j = 0; j < 8; ++j) {
                std::cout << "," << res.late_per_slot[j];
            }
            std::cout << "," << res.appointments_arrived << "," << res.appointments_late
                      << "," << res.appointment_wait_sum;
            for (int j = 0; j < 8; ++j) {
                std::cout << "," << res.abandoned_per_slot[j];
            }
            std::cout << "," << res.abandoned_wait_sum << "," << res.spill_minutes << ","
                      << res.overtime_busy_minutes << "," << res.balked;
            std::cout << "\n";
        }
        return 0;
    }

    // Aggregate results across replications
    double sum_mean_wait = 0.0, sum_p90_wait = 0.0;
    double sum_served = 0.0, sum_arrived = 0.0, sum_overtime = 0.0;
    std::vector<double> sum_util(8, 0.0);
    std::vector<double> all_waits;

    for (const auto& r : results) {
        sum_mean_wait += r.mean_wait_time;
        sum_p90_wait += r.p90_wait_time;
        sum_served += r.total_served;
        sum_arrived += r.total_arrived;
        sum_overtime += r.overtime_minutes;
        for (int j = 0; j < 8; ++j) {
            sum_util[j] += r.utilization_per_slot[j];
        }
        if (output_waits) {
            all_waits.insert(all_waits.end(),
                           r.all_wait_times.begin(),
                           r.all_wait_times.end());
        }
    }

    int n = replications;

    // Output aggregated CSV
    std::cout << "metric,value\n";
    std::cout << "mean_wait_time," << (sum_mean_wait / n) << "\n";
    std::cout << "p90_wait_time," << (sum_p90_wait / n) << "\n";
    std::cout << "avg_served," << (sum_served / n) << "\n";
    std::cout << "avg_arrived," << (sum_arrived / n) << "\n";
    std::cout << "avg_overtime," << (sum_overtime / n) << "\n";
    std::cout << "replications," << n << "\n";

    for (int j = 0; j < 8; ++j) {
        std::cout << "utilization_slot_" << j << "," << (sum_util[j] / n) << "\n";
    }

    // Output all wait times if requested (for distribution analysis)
    if (output_waits && !all_waits.empty()) {
        std::cout << "\nwait_times\n";
        for (double w : all_waits) {
            std::cout << w << "\n";
        }
    }

    return 0;
}

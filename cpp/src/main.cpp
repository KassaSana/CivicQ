/**
 * @file main.cpp
 * @brief Command-line interface for queue simulation
 * 
 * Accepts JSON-style configuration via stdin or command-line arguments,
 * outputs CSV results for Python consumption.
 */

#include "simulation.hpp"
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
              << "  --output-waits             Include all wait times in output\n"
              << "  --help                     Show this help\n";
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
    
    // Run simulation(s)
    auto results = run_replications(config, replications, config.random_seed);
    
    // Aggregate results across replications
    double sum_mean_wait = 0.0, sum_p90_wait = 0.0;
    double sum_served = 0.0, sum_arrived = 0.0;
    std::vector<double> sum_util(8, 0.0);
    std::vector<double> all_waits;
    
    for (const auto& r : results) {
        sum_mean_wait += r.mean_wait_time;
        sum_p90_wait += r.p90_wait_time;
        sum_served += r.total_served;
        sum_arrived += r.total_arrived;
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

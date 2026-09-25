/**
 * @file simulation.cpp
 * @brief Implementation of Discrete-Event Simulation Engine
 *
 * Public-Sector Queue Resource Allocation Simulator
 * Municipal permit/licensing office staffing optimization tool.
 */

#include "simulation.hpp"
#include <limits>
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
    , lognormal_dist_(
          // Match the requested mean and CV: sigma^2 = ln(1 + cv^2), mu = ln(mean) - sigma^2 / 2
          std::log(config.mean_service_time)
              - 0.5 * std::log(1.0 + config.service_cv * config.service_cv),
          std::sqrt(std::log(1.0 + config.service_cv * config.service_cv)))
    , uniform_dist_(0.0, 1.0)
    , rate_multiplier_(1.0)
    , current_time_(0.0)
    , last_departure_time_(0.0)
    , next_citizen_id_(0)
    , waiting_count_(0)
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
    waiting_count_ = 0;

    // Clear event queue
    while (!event_queue_.empty()) {
        event_queue_.pop();
    }

    // Clear waiting queue
    waiting_queue_.clear();

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
    spill_minutes_ = 0.0;
    overtime_busy_minutes_ = 0.0;
    last_start_wait_ = 0.0;

    // Reseed independent streams (common random numbers across staffing plans)
    std::seed_seq arrival_seed{static_cast<unsigned>(config_.random_seed), 1u};
    std::seed_seq service_seed{static_cast<unsigned>(config_.random_seed), 2u};
    std::seed_seq rate_seed{static_cast<unsigned>(config_.random_seed), 3u};
    std::seed_seq appointment_seed{static_cast<unsigned>(config_.random_seed), 4u};
    std::seed_seq patience_seed{static_cast<unsigned>(config_.random_seed), 5u};
    arrival_rng_.seed(arrival_seed);
    service_rng_.seed(service_seed);
    rate_rng_.seed(rate_seed);
    appointment_rng_.seed(appointment_seed);
    patience_rng_.seed(patience_seed);
    std::seed_seq twin_seed{static_cast<unsigned>(config_.random_seed), 6u};
    twin_rng_.seed(twin_seed);
    service_dist_.reset();
    lognormal_dist_.reset();
    uniform_dist_.reset();

    // Day-level demand multiplier M ~ Gamma(1/cv^2, cv^2), mean 1, CV = rate_cv.
    // Every staffing plan sees the same M for a given seed.
    rate_multiplier_ = 1.0;
    if (config_.rate_cv > 0.0) {
        double shape = 1.0 / (config_.rate_cv * config_.rate_cv);
        std::gamma_distribution<double> gamma(shape, 1.0 / shape);
        rate_multiplier_ = gamma(rate_rng_);
    }

    // Slot boundaries: let newly opened windows serve queued citizens
    for (int slot = 1; slot < NUM_SLOTS; ++slot) {
        double t = slot * SLOT_LENGTH;
        if (t < config_.simulation_duration) {
            event_queue_.push({t, EventType::STAFFING_CHANGE, -1, -1});
        }
    }

    // Booked citizens: each shows with probability 1 - no_show and arrives at
    // the booked time plus Normal(0, punctuality_sd), clipped to opening hours.
    // Both draws are always taken so the stream stays aligned across settings.
    std::uniform_real_distribution<double> show_draw(0.0, 1.0);
    std::normal_distribution<double> punctuality(0.0, 1.0);
    for (double booked : config_.appointment_times) {
        double u = show_draw(appointment_rng_);
        double z = punctuality(appointment_rng_);
        if (u < config_.no_show) {
            continue;
        }
        double t = std::min(std::max(booked + config_.punctuality_sd * z, 0.0),
                            config_.simulation_duration);
        event_queue_.push({t, EventType::APPOINTMENT, -1, -1});
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
    return rate_multiplier_ * config_.arrival_rates[slot] / 60.0;
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
    double lambda_max_per_min = rate_multiplier_ * (*std::max_element(
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
    switch (config_.service_dist) {
        case ServiceDist::LOGNORMAL:
            return lognormal_dist_(service_rng_);
        case ServiceDist::DETERMINISTIC:
            return config_.mean_service_time;
        case ServiceDist::EXPONENTIAL:
        default:
            return service_dist_(service_rng_);
    }
}

double QueueSimulator::generate_patience() {
    double mean = config_.mean_patience;
    switch (config_.patience_dist) {
        case ServiceDist::LOGNORMAL: {
            double sigma2 = std::log(1.0 + config_.patience_cv * config_.patience_cv);
            std::lognormal_distribution<double> d(std::log(mean) - 0.5 * sigma2, std::sqrt(sigma2));
            return d(patience_rng_);
        }
        case ServiceDist::DETERMINISTIC:
            return mean;
        case ServiceDist::EXPONENTIAL:
        default: {
            std::exponential_distribution<double> d(1.0 / mean);
            return d(patience_rng_);
        }
    }
}

double QueueSimulator::replay_start(std::vector<double> free_at,
                                    const std::vector<Ahead>& ahead) const {
    // When a citizen arriving now would be called if they stayed. Later
    // arrivals queue behind them (FIFO), so only the people ahead matter:
    // replay them in order. A window can start a service only while open
    // (index < staffing of the slot); someone whose patience runs out before
    // a window reaches them leaves without using it
    const int n = static_cast<int>(free_at.size());
    auto next_open = [&](int i, double t) {
        for (int slot = get_current_slot(t); slot < NUM_SLOTS; ++slot) {
            if (i < config_.staffing_per_slot[slot]) {
                return std::max(t, slot * SLOT_LENGTH);
            }
        }
        return std::numeric_limits<double>::infinity();
    };
    auto earliest = [&](int& window) {
        double best = std::numeric_limits<double>::infinity();
        window = -1;
        for (int i = 0; i < n; ++i) {
            double t = next_open(i, free_at[i]);
            if (t < best) {
                best = t;
                window = i;
            }
        }
        return best;
    };
    for (const Ahead& p : ahead) {
        int window;
        double start = earliest(window);
        if (window < 0) {
            break;
        }
        if (p.leave_time < start) {
            continue;
        }
        free_at[window] = start + p.service_time;
    }
    int window;
    return earliest(window);
}

double QueueSimulator::offered_wait() const {
    // The exact wait: the people ahead with their drawn service times and patience
    const bool may_renege = config_.abandonment == Abandonment::RENEGE && !config_.commit;
    std::vector<double> free_at(windows_.size(), current_time_);
    for (std::size_t i = 0; i < windows_.size(); ++i) {
        if (windows_[i].is_busy) {
            const Citizen& c = citizens_[windows_[i].current_citizen_id];
            free_at[i] = c.service_start_time + c.service_time;
        }
    }
    std::vector<Ahead> ahead;
    for (int id : waiting_queue_) {
        const Citizen& p = citizens_[id];
        if (p.abandoned) {
            continue;
        }
        double leave = (may_renege && !p.is_appointment) ? p.arrival_time + p.patience
                                                         : std::numeric_limits<double>::infinity();
        ahead.push_back({leave, p.service_time});
    }
    return replay_start(free_at, ahead) - current_time_;
}

double QueueSimulator::draw_service(std::mt19937& rng) const {
    const double mean = config_.mean_service_time;
    switch (config_.service_dist) {
        case ServiceDist::LOGNORMAL: {
            double s2 = std::log(1.0 + config_.service_cv * config_.service_cv);
            std::lognormal_distribution<double> d(std::log(mean) - 0.5 * s2, std::sqrt(s2));
            return d(rng);
        }
        case ServiceDist::DETERMINISTIC:
            return mean;
        case ServiceDist::EXPONENTIAL:
        default: {
            std::exponential_distribution<double> d(1.0 / mean);
            return d(rng);
        }
    }
}

double QueueSimulator::draw_patience(std::mt19937& rng) const {
    const double mean = config_.mean_patience;
    switch (config_.patience_dist) {
        case ServiceDist::LOGNORMAL: {
            double s2 = std::log(1.0 + config_.patience_cv * config_.patience_cv);
            std::lognormal_distribution<double> d(std::log(mean) - 0.5 * s2, std::sqrt(s2));
            return d(rng);
        }
        case ServiceDist::DETERMINISTIC:
            return mean;
        case ServiceDist::EXPONENTIAL:
        default: {
            std::exponential_distribution<double> d(1.0 / mean);
            return d(rng);
        }
    }
}

void QueueSimulator::twin_draws(std::vector<double>& waits) {
    // What a ticket office can predict: it sees its uncalled tickets and their
    // ages (not whether each holder is still there) and how long each window
    // has been serving, and knows the patience and service distributions.
    // Sample the unknowns (holder still present = patience beyond the ticket's
    // age; remaining and future service times) and replay: each sample is a
    // draw of V from its posterior given what the office sees
    const bool may_renege = config_.abandonment == Abandonment::RENEGE && !config_.commit;
    const int k = std::max(1, config_.twin_samples);
    waits.assign(k, 0.0);
    std::vector<double> free_at(windows_.size());
    std::vector<Ahead> ahead;
    ahead.reserve(waiting_queue_.size());
    for (int s = 0; s < k; ++s) {
        for (std::size_t i = 0; i < windows_.size(); ++i) {
            free_at[i] = current_time_;
            if (!windows_[i].is_busy) {
                continue;
            }
            double elapsed = current_time_
                - citizens_[windows_[i].current_citizen_id].service_start_time;
            double total = draw_service(twin_rng_);
            if (config_.service_dist == ServiceDist::EXPONENTIAL) {
                total = elapsed + total;              // Memoryless
            } else {
                for (int tries = 0; total <= elapsed && tries < 200; ++tries) {
                    total = draw_service(twin_rng_);
                }
                total = std::max(total, elapsed);
            }
            free_at[i] = current_time_ - elapsed + total;
        }
        ahead.clear();
        for (int id : waiting_queue_) {
            const Citizen& p = citizens_[id];
            double leave = std::numeric_limits<double>::infinity();
            if (may_renege && !p.is_appointment) {
                double tau = draw_patience(twin_rng_);
                if (p.arrival_time + tau < current_time_) {
                    continue;                          // Holder has already gone
                }
                leave = p.arrival_time + tau;
            }
            ahead.push_back({leave, draw_service(twin_rng_)});
        }
        waits[s] = replay_start(free_at, ahead) - current_time_;
    }
}

double QueueSimulator::twin_quantile_of(std::vector<double> waits) const {
    // The requested quantile of the sampled waits
    const int k = static_cast<int>(waits.size());
    std::size_t idx = std::min<std::size_t>(
        k - 1, static_cast<std::size_t>(config_.twin_quantile * k));
    std::nth_element(waits.begin(), waits.begin() + idx, waits.end());
    return waits[idx];
}

double QueueSimulator::psi_at(const SimulationConfig::PsiTable& t, int hour, double v) const {
    // psi for this hour, linear between grid points and flat beyond them
    const std::size_t h = std::min<std::size_t>(hour, t.x.size() - 1);
    const std::vector<double>& x = t.x[h];
    const std::vector<double>& y = t.psi[h];
    if (v <= x.front()) {
        return y.front();
    }
    if (v >= x.back()) {
        return y.back();
    }
    std::size_t j = std::upper_bound(x.begin(), x.end(), v) - x.begin();
    double w = (v - x[j - 1]) / (x[j] - x[j - 1]);
    return y[j - 1] + w * (y[j] - y[j - 1]);
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
    citizens_[citizen_id].call_time = current_time_;
    last_start_wait_ = current_time_ - citizens_[citizen_id].arrival_time;
    --waiting_count_;

    double departure_time = current_time_ + citizens_[citizen_id].service_time;

    event_queue_.push({departure_time, EventType::DEPARTURE, window_id, citizen_id});
}

void QueueSimulator::serve_waiting_citizens() {
    // FIFO: fill every free open window from the front of the queue. A ticket
    // is called only when a window is free; a reneger's ticket is called then,
    // nobody comes, and the same window calls the next ticket
    while (!waiting_queue_.empty()) {
        int free_window = find_free_window();
        if (free_window < 0) {
            break;
        }
        int next_citizen = waiting_queue_.front();
        waiting_queue_.pop_front();
        if (citizens_[next_citizen].abandoned) {
            citizens_[next_citizen].call_time = current_time_;   // Reneged while in line
            continue;
        }
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

void QueueSimulator::add_unpaid_time(int window_id, double start, double end) {
    // Service time no staffing slot pays for: after the doors close, and on a
    // window whose slot has closed it (open windows are indices < staffing)
    double close = config_.simulation_duration;
    overtime_busy_minutes_ += std::max(0.0, end - std::max(start, close));
    end = std::min(end, close);
    for (int slot = get_current_slot(start); slot < NUM_SLOTS && start < end; ++slot) {
        double slot_end = (slot == NUM_SLOTS - 1) ? end
                                                  : std::min(end, (slot + 1) * SLOT_LENGTH);
        if (slot_end > start && window_id >= config_.staffing_per_slot[slot]) {
            spill_minutes_ += slot_end - start;
        }
        start = std::max(start, slot_end);
    }
}

void QueueSimulator::admit_citizen(bool is_appointment) {
    // Citizen ids are assigned in arrival order and index citizens_. The
    // service requirement is drawn now so that, for given arrivals, citizen k
    // needs the same work under every staffing plan
    Citizen citizen;
    citizen.id = static_cast<int>(citizens_.size());
    citizen.arrival_time = current_time_;
    citizen.service_time = generate_service_time();
    citizen.is_appointment = is_appointment;
    citizen.patience = 0.0;
    citizen.abandoned = false;
    citizen.abandon_time = -1.0;
    citizen.service_start_time = -1.0;
    citizen.departure_time = -1.0;
    citizen.call_time = -1.0;
    citizen.queue_ahead = waiting_count_;
    citizen.open_at_arrival = get_open_windows(current_time_);
    citizen.balked = false;
    // What each display would show. With a window free and nobody waiting the
    // citizen is served at once; otherwise (n + 1) completions at rate c / S
    {
        int open = citizen.open_at_arrival;
        bool free_now = waiting_count_ == 0 && find_free_window() >= 0;
        double per = config_.mean_service_time / open;
        citizen.est_count = free_now ? 0.0 : (waiting_count_ + 1) * per;
        citizen.est_tickets = free_now ? 0.0
            : (static_cast<double>(waiting_queue_.size()) + 1) * per;
        citizen.est_les = free_now ? 0.0 : last_start_wait_;
        citizen.est_oracle = config_.announce == Announce::ORACLE ? offered_wait() : 0.0;
        // TWIN: a quantile of the posterior draws of V. BAYES: flag iff the
        // posterior mean of psi_hour(V) is negative. With log_offered, also the
        // true V and the twin's P(V > threshold), from the same draws. The twin
        // stream is separate, so logging changes no citizen's outcome
        citizen.offered = -1.0;
        citizen.twin_p_late = 0.0;
        citizen.bayes_score = 0.0;
        const bool twin = config_.announce == Announce::TWIN;
        // A mixture whose tables never go negative can never say "too long":
        // skip its draws (the twin stream feeds nothing else)
        bool can_flag = false;
        for (const auto& t : config_.psi_tables) {
            for (const auto& hour : t.psi) {
                for (double p : hour) {
                    can_flag = can_flag || p < 0.0;
                }
            }
        }
        const bool bayes = config_.announce == Announce::BAYES && can_flag;
        if ((twin || bayes || config_.log_offered) && !free_now) {
            std::vector<double> draws;
            twin_draws(draws);
            if (twin) {
                citizen.est_oracle = twin_quantile_of(draws);
            }
            int hour = get_current_slot(current_time_);
            double late = 0.0;
            for (double v : draws) {
                late += v > config_.wait_threshold ? 1.0 : 0.0;
            }
            citizen.twin_p_late = late / draws.size();
            if (bayes) {
                // Each table votes with its weight; a split vote is settled by a
                // uniform draw on the twin stream (never drawn with one table)
                double say = 0.0, total = 0.0;
                for (std::size_t t = 0; t < config_.psi_tables.size(); ++t) {
                    double score = 0.0;
                    for (double v : draws) {
                        score += psi_at(config_.psi_tables[t], hour, v);
                    }
                    score /= draws.size();
                    if (t == 0) {
                        citizen.bayes_score = score;
                    }
                    total += config_.psi_tables[t].weight;
                    say += score < 0.0 ? config_.psi_tables[t].weight : 0.0;
                }
                bool flag = say >= total;
                if (say > 0.0 && say < total) {
                    flag = std::uniform_real_distribution<double>(0.0, total)(twin_rng_) < say;
                }
                citizen.est_oracle = flag ? std::numeric_limits<double>::infinity() : 0.0;
            }
        }
        if (config_.log_offered) {
            citizen.offered = free_now ? 0.0 : offered_wait();
        }
    }
    // Drawn for every citizen, in arrival order, so patience stays aligned
    // across staffing plans just like service requirements
    bool may_abandon = false;
    if (config_.abandonment != Abandonment::NONE) {
        citizen.patience = generate_patience();
        may_abandon = !is_appointment;
    }
    citizens_.push_back(citizen);

    if (may_abandon && config_.abandonment == Abandonment::BALK) {
        // The citizen sees the line: q people waiting and c open windows. With
        // every window busy they expect (q + 1) service completions at rate
        // c / S before their turn; with a free window they are served at once
        if (citizen.est_count > citizen.patience) {
            citizens_.back().abandoned = true;
            citizens_.back().balked = true;
            citizens_.back().abandon_time = current_time_;
            return;
        }
    }
    if (may_abandon && config_.abandonment == Abandonment::RENEGE
            && config_.announce != Announce::NONE) {
        // Ticket queue with a wait display: leave at once if it exceeds patience
        double est = config_.announce == Announce::TICKETS ? citizen.est_tickets
                   : config_.announce == Announce::COUNT ? citizen.est_count
                   : (config_.announce == Announce::ORACLE
                      || config_.announce == Announce::TWIN
                      || config_.announce == Announce::BAYES) ? citizen.est_oracle
                   : citizen.est_les;
        double shown = config_.display_scale * est;
        if (config_.announce == Announce::BAYES) {
            shown = est;                          // "Too long" (infinity) or nothing
        } else if (!config_.display_cutoff.empty()) {
            std::size_t hour = std::min(static_cast<std::size_t>(current_time_ / 60.0),
                                        config_.display_cutoff.size() - 1);
            if (est > 0.0 && est >= config_.display_cutoff[hour]) {
                shown = std::numeric_limits<double>::infinity();
            }
        }
        if (shown > citizen.patience) {
            citizens_.back().abandoned = true;
            citizens_.back().balked = true;
            citizens_.back().abandon_time = current_time_;
            return;
        }
    }

    // Join the back of the queue, then serve in FIFO order
    waiting_queue_.push_back(citizen.id);
    ++waiting_count_;
    serve_waiting_citizens();

    if (may_abandon && config_.abandonment == Abandonment::RENEGE && !config_.commit
            && citizens_[citizen.id].service_start_time < 0) {
        event_queue_.push({current_time_ + citizen.patience, EventType::RENEGE, -1, citizen.id});
    }
}

void QueueSimulator::process_renege(const Event& event) {
    Citizen& citizen = citizens_[event.citizen_id];
    if (citizen.service_start_time >= 0 || citizen.abandoned) {
        return;   // Already being served
    }
    citizen.abandoned = true;
    citizen.abandon_time = current_time_;
    --waiting_count_;   // Removed lazily from waiting_queue_
}

void QueueSimulator::process_arrival(const Event& /*event*/) {
    admit_citizen(false);

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
    add_unpaid_time(window_id, citizens_[citizen_id].service_start_time, current_time_);

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
            case EventType::APPOINTMENT:
                admit_citizen(true);
                break;
            case EventType::RENEGE:
                process_renege(event);
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
    results.rate_multiplier = rate_multiplier_;
    results.arrivals_per_slot.assign(NUM_SLOTS, 0);
    results.late_per_slot.assign(NUM_SLOTS, 0);
    results.appointments_arrived = 0;
    results.appointments_late = 0;
    results.appointment_wait_sum = 0.0;
    results.abandoned_per_slot.assign(NUM_SLOTS, 0);
    results.abandoned_wait_sum = 0.0;
    results.balked = 0;
    results.spill_minutes = spill_minutes_;
    results.overtime_busy_minutes = overtime_busy_minutes_;

    std::vector<double> wait_times;
    std::vector<double> service_times;

    for (const auto& citizen : citizens_) {
        if (citizen.abandoned) {
            results.abandoned_per_slot[get_current_slot(citizen.arrival_time)]++;
            results.abandoned_wait_sum += citizen.abandon_time - citizen.arrival_time;
            if (citizen.balked) {
                results.balked++;
            }
        }
        if (citizen.departure_time >= 0) {
            results.total_served++;
            double wait = citizen.service_start_time - citizen.arrival_time;
            double service = citizen.departure_time - citizen.service_start_time;
            wait_times.push_back(wait);
            service_times.push_back(service);

            // Per-hour service level, indexed by the hour the citizen arrived
            int slot = get_current_slot(citizen.arrival_time);
            results.arrivals_per_slot[slot]++;
            if (wait > config_.wait_threshold) {
                results.late_per_slot[slot]++;
            }
            if (citizen.is_appointment) {
                results.appointments_arrived++;
                results.appointment_wait_sum += wait;
                if (wait > config_.wait_threshold) {
                    results.appointments_late++;
                }
            }
        }
    }

    results.all_wait_times = wait_times;
    if (config_.log_citizens) {
        results.citizen_log = citizens_;
    }

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

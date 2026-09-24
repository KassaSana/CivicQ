# CivicQ Visualizer

**Live: [kassasana.github.io/CivicQ](https://kassasana.github.io/CivicQ/)**

Interactive website for the CivicQ staffing simulator. The discrete-event simulator is ported from `cpp/src/simulation.cpp` to TypeScript and runs in a Web Worker, so every control updates in the browser without a backend. It is laid out as an interactive article to go with [`research/REPORT.md`](../research/REPORT.md): serif text, numbered figures with captions, and the model settings in an *Assumptions* drawer. The first version's design brief is kept in [DESIGN_PROMPT.md](DESIGN_PROMPT.md).

```bash
cd web
npm install
npm run dev      # http://localhost:5173
npm test         # simulator + Erlang-C tests; cross-checks against cpp/build/queue_sim if it is built
npm run build    # static site in web/dist (relative paths, works on GitHub Pages)
```

## Layout

| Path | What it is |
|---|---|
| `src/sim/simulate.ts` | One simulated day (port of `QueueSimulator`), with common random numbers and optional appointments |
| `src/sim/model.ts` | Config, defaults, and `appointmentBook` (port of `research/experiments.appointment_book`) |
| `src/sim/stats.ts` | Replications, t and ratio confidence intervals, histogram and queue curve |
| `src/sim/analytic.ts` | Erlang-C, SIPP, Lag-SIPP, offered load m(t) (ports of `python/optimizer.py` and `research/staffing_methods.py`) |
| `src/sim/worker.ts` | Background runs: current plan, plan comparison, cost-vs-service frontier search |
| `src/sections/*` | One component per page section |
| `src/state.ts` | App state, mirrored into the URL so a view can be shared |

## Validation (`src/sim/sim.test.ts`)

- Erlang-C matches the known values in `python/test_validation.py`.
- Under constant load, the simulated mean wait falls inside the 95% CI of the exact M/M/c value.
- For the default plan, per-hour late shares agree with `research/staffing_methods.evaluate`.
- Bookings per hour match `appointment_book` for all three placements; perfectly spaced bookings with fixed service give zero waits; the show rate matches 1 − no-show; adding bookings leaves the walk-in stream unchanged.
- Mean and P90 wait agree with the C++ executable for four plans, and for 50% counter-cyclical appointments, when it is built.

## Appointments

The *Appointments* controls move a share of expected demand into booked slots, as in report §5.7. Bookings are overbooked by 1/(1 − no-show) so expected arrivals stay constant, and each booked citizen arrives at the slot time plus Normal(0, punctuality SD). The analytic views (Erlang-C, SIPP, offered load) use walk-ins plus expected shows. The results section splits waits for booked citizens and walk-ins.
